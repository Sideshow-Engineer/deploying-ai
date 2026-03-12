from __future__ import annotations

from typing import Any


HIGH_RISK_TERMS = (
    "freeze",
    "trip",
    "overflow",
    "flood",
    "safety",
    "shutdown",
    "cannot meet",
    "not maintained",
    "cleanroom",
    "negative pressure",
)

MEDIUM_RISK_TERMS = (
    "unstable",
    "hunting",
    "oscillat",
    "inconsistent",
    "complaint",
    "drift",
    "incorrect",
    "noise",
    "alarm",
)


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def triage_issue(
    issue_text: str,
    system_hint: str | None = None,
    stage_hint: str | None = None,
) -> dict[str, Any]:
    text = (issue_text or "").lower()
    system_text = (system_hint or "").lower()
    stage_text = (stage_hint or "").lower()

    score = 0
    if _contains_any(text, HIGH_RISK_TERMS):
        score += 2
    if _contains_any(text, MEDIUM_RISK_TERMS):
        score += 1
    if stage_text in {"startup", "commissioning", "tab"}:
        score += 1

    if score >= 3:
        risk_level = "High"
    elif score >= 1:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    risk_dimensions: list[str] = []
    if _contains_any(text, ("delay", "rework", "startup", "commissioning", "tab")):
        risk_dimensions.append("schedule")
    if _contains_any(text, ("freeze", "trip", "overflow", "shutdown", "safety", "flood")):
        risk_dimensions.append("safety")
    if _contains_any(text, ("incorrect", "drift", "mapping", "calibration", "quality")):
        risk_dimensions.append("quality")
    if _contains_any(text, ("co2", "iaq", "ventilation", "humidity", "odor", "cleanroom")):
        risk_dimensions.append("IAQ")
    if _contains_any(text, ("economizer", "reheat", "energy", "delta t", "efficiency")):
        risk_dimensions.append("energy")
    if not risk_dimensions:
        risk_dimensions.append("schedule")

    likely_disciplines: list[str] = []
    if _contains_any(text + " " + stage_text, ("control", "sensor", "trend", "vfd", "pid", "bas")):
        likely_disciplines.append("Controls")
    if _contains_any(text + " " + system_text, ("coil", "valve", "damper", "drain", "ahu", "vav", "chw", "pump")):
        likely_disciplines.append("Mechanical")
    if _contains_any(text + " " + stage_text, ("tab", "airflow", "balancing", "cfm")):
        likely_disciplines.append("TAB")
    if not likely_disciplines:
        likely_disciplines = ["Mechanical", "Controls"]

    recommended_next_steps = [
        "Capture 24-48h trends for setpoints, commands, feedback, and alarms.",
        "Perform field verification of sensors, actuators, and physical positions at 0/50/100% commands where applicable.",
        "Compare observed behavior against approved sequence-of-operations and latest addenda/RFI responses.",
        "Run a controlled functional test and document pass/fail evidence before changing production logic.",
    ]
    if risk_level == "High":
        recommended_next_steps.insert(
            0,
            "Escalate to immediate coordination with site lead and commissioning authority due to high operational risk.",
        )

    needs_rfi_terms = (
        "design intent",
        "confirm",
        "sequence",
        "setpoint",
        "requirement",
        "unknown",
        "unclear",
        "specified",
        "rfi",
    )
    needs_rfi = _contains_any(text, needs_rfi_terms) or (
        risk_level == "High" and stage_text in {"startup", "commissioning"}
    )
    rfi_reason = (
        "Potential design-intent or sequence ambiguity detected; written clarification is recommended."
        if needs_rfi
        else "No immediate design ambiguity detected; resolve with field checks and controls/mechanical tuning first."
    )

    return {
        "risk_level": risk_level,
        "risk_dimensions": risk_dimensions,
        "likely_disciplines": likely_disciplines,
        "recommended_next_steps": recommended_next_steps,
        "needs_rfi": needs_rfi,
        "rfi_reason": rfi_reason,
    }

