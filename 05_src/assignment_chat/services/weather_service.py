from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import re
from typing import Any

import requests

from config import (
    FREEZE_RISK_C,
    HEAT_RISK_C,
    HEAVY_PRECIP_RISK_MM,
    OPEN_METEO_FORECAST_URL,
    OPEN_METEO_GEOCODE_URL,
    PRECIP_HOURS_RISK,
    SNOW_RISK_CM,
    WEATHER_DEFAULT_LOCATION,
    WIND_GUST_RISK_KMH,
    WIND_SPEED_RISK_KMH,
)


@dataclass
class WeatherRiskResult:
    location: str
    start_date: str
    end_date: str
    risk_summary: str
    action_cues: list[str]
    source_note: str


def _parse_date_range(message: str) -> tuple[date, date, str]:
    lowered = message.lower()
    today = date.today()

    # Optional explicit range: YYYY-MM-DD to YYYY-MM-DD
    range_match = re.search(
        r"(\d{4}-\d{2}-\d{2})\s*(?:to|-)\s*(\d{4}-\d{2}-\d{2})", lowered
    )
    if range_match:
        start = date.fromisoformat(range_match.group(1))
        end = date.fromisoformat(range_match.group(2))
        if end < start:
            start, end = end, start
        return start, end, "custom date range"

    single_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", lowered)
    if single_match:
        day = date.fromisoformat(single_match.group(1))
        return day, day, "specific date"

    if "tomorrow" in lowered:
        start = today + timedelta(days=1)
        return start, start, "tomorrow"

    if "next 3 days" in lowered or "3 days" in lowered:
        return today, today + timedelta(days=2), "next 3 days"

    if "today" in lowered:
        return today, today, "today"

    # Default window if user did not specify.
    return today, today + timedelta(days=2), "next 3 days"


def _extract_location(message: str, default_location: str = WEATHER_DEFAULT_LOCATION) -> str:
    def clean_candidate(raw: str) -> str:
        stop_words = {
            "weather",
            "forecast",
            "risk",
            "construction",
            "site",
            "commissioning",
            "summary",
            "for",
            "in",
            "at",
        }
        raw = re.split(
            r"\b(today|tomorrow|next|day|days|week|weekend)\b",
            raw,
            maxsplit=1,
            flags=re.I,
        )[0]
        tokens = []
        for token in raw.split():
            cleaned = token.strip(" ,.")
            if not cleaned:
                continue
            if cleaned.lower() in stop_words:
                continue
            tokens.append(cleaned)
        return " ".join(tokens).strip()

    explicit_match = re.search(
        r"\b(?:in|for|at)\s+([a-zA-Z][a-zA-Z\s\-']{1,60})",
        message,
        flags=re.I,
    )
    if explicit_match:
        candidate = clean_candidate(explicit_match.group(1))
        if candidate:
            return candidate

    # Basic fallback for patterns like "Toronto weather tomorrow".
    leading_city = re.search(
        r"^([a-zA-Z][a-zA-Z\s\-']{1,40})\s+(?:weather|forecast)\b",
        message.strip(),
        flags=re.I,
    )
    if leading_city:
        candidate = clean_candidate(leading_city.group(1))
        if candidate:
            return candidate

    return default_location


def _geocode_location(location: str) -> tuple[str, float, float]:
    response = requests.get(
        OPEN_METEO_GEOCODE_URL,
        params={
            "name": location,
            "count": 1,
            "language": "en",
            "format": "json",
        },
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    results = payload.get("results") or []
    if not results:
        raise ValueError(f"Could not find location: {location}")

    best = results[0]
    name = best.get("name", location)
    country = best.get("country")
    display_name = f"{name}, {country}" if country else name
    return display_name, float(best["latitude"]), float(best["longitude"])


def _fetch_daily_forecast(lat: float, lon: float, start: date, end: date) -> dict[str, Any]:
    response = requests.get(
        OPEN_METEO_FORECAST_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "timezone": "auto",
            "daily": ",".join(
                [
                    "temperature_2m_min",
                    "temperature_2m_max",
                    "precipitation_sum",
                    "precipitation_hours",
                    "snowfall_sum",
                    "wind_speed_10m_max",
                    "wind_gusts_10m_max",
                ]
            ),
        },
        timeout=20,
    )
    response.raise_for_status()
    return response.json()


def _safe_value(data: list[Any], idx: int) -> float:
    try:
        value = data[idx]
        if value is None:
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def _build_risk_and_actions(forecast_payload: dict[str, Any], range_label: str) -> tuple[str, list[str]]:
    daily = forecast_payload.get("daily", {})
    dates = daily.get("time", [])

    if not dates:
        return "Weather data unavailable for the requested window.", []

    lines: list[str] = []
    action_cues: list[str] = []
    risk_flags = {
        "wind": False,
        "freeze": False,
        "precip": False,
        "heat": False,
        "snow": False,
    }

    for idx, day in enumerate(dates):
        t_min = _safe_value(daily.get("temperature_2m_min", []), idx)
        t_max = _safe_value(daily.get("temperature_2m_max", []), idx)
        precip = _safe_value(daily.get("precipitation_sum", []), idx)
        precip_hours = _safe_value(daily.get("precipitation_hours", []), idx)
        snow = _safe_value(daily.get("snowfall_sum", []), idx)
        wind_speed = _safe_value(daily.get("wind_speed_10m_max", []), idx)
        wind_gust = _safe_value(daily.get("wind_gusts_10m_max", []), idx)

        day_risks: list[str] = []
        if wind_gust >= WIND_GUST_RISK_KMH or wind_speed >= WIND_SPEED_RISK_KMH:
            day_risks.append(
                f"wind risk (gust {wind_gust:.0f} km/h, sustained {wind_speed:.0f} km/h)"
            )
            risk_flags["wind"] = True
        if t_min <= FREEZE_RISK_C:
            day_risks.append(f"freeze risk (min {t_min:.1f}C)")
            risk_flags["freeze"] = True
        if precip >= HEAVY_PRECIP_RISK_MM or precip_hours >= PRECIP_HOURS_RISK:
            day_risks.append(
                f"precipitation risk ({precip:.1f} mm over {precip_hours:.1f} h)"
            )
            risk_flags["precip"] = True
        if snow >= SNOW_RISK_CM:
            day_risks.append(f"snow risk ({snow:.1f} cm)")
            risk_flags["snow"] = True
        if t_max >= HEAT_RISK_C:
            day_risks.append(f"heat stress risk (max {t_max:.1f}C)")
            risk_flags["heat"] = True

        if day_risks:
            lines.append(f"- {day}: " + "; ".join(day_risks))
        else:
            lines.append(
                f"- {day}: low weather risk (min {t_min:.1f}C, max {t_max:.1f}C, "
                f"precip {precip:.1f} mm, gust {wind_gust:.0f} km/h)"
            )

    if risk_flags["wind"]:
        action_cues.append(
            "Review crane/exterior lift plan, secure loose materials, and tighten temporary covers."
        )
    if risk_flags["freeze"]:
        action_cues.append(
            "Activate freeze protection checks: preheat sequence, coil protection, and hydronic readiness."
        )
    if risk_flags["precip"] or risk_flags["snow"]:
        action_cues.append(
            "Protect roof/open-air tasks and deliveries; waterproof exposed equipment and materials."
        )
    if risk_flags["heat"]:
        action_cues.append(
            "Plan heat controls: hydration breaks, work-rest cycles, and enclosure ventilation."
        )
    if not action_cues:
        action_cues.append(
            "No major weather constraints detected; continue standard site controls and daily checks."
        )

    summary_header = (
        f"Construction weather risk summary ({range_label}):\n"
        "Threshold-based interpretation for wind, freezing temperatures, precipitation, and heat."
    )
    return summary_header + "\n" + "\n".join(lines), action_cues


def get_construction_weather_risk(message: str) -> WeatherRiskResult:
    location = _extract_location(message)
    start, end, range_label = _parse_date_range(message)
    display_location, lat, lon = _geocode_location(location)
    forecast_payload = _fetch_daily_forecast(lat=lat, lon=lon, start=start, end=end)

    risk_summary, action_cues = _build_risk_and_actions(
        forecast_payload=forecast_payload,
        range_label=range_label,
    )

    return WeatherRiskResult(
        location=display_location,
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        risk_summary=risk_summary,
        action_cues=action_cues,
        source_note=(
            f"Source: Open-Meteo forecast API ({display_location}, lat={lat:.3f}, lon={lon:.3f})."
        ),
    )
