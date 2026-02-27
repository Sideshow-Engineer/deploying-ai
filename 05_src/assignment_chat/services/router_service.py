from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from config import (
    OPENAI_GATEWAY_BASE_URL,
    ROUTER_MAX_ACTIONS,
    ROUTER_MIN_CONFIDENCE,
    ROUTER_MODEL,
)


class RouteAction(BaseModel):
    service: Literal["service1", "service2", "service3"]
    intent: Literal["weather_risk", "semantic_troubleshoot", "triage", "draft_rfi", "other"]
    request_text: str
    use_memory_issue: bool = False
    confidence: float = Field(ge=0.0, le=1.0)


class RouteDecision(BaseModel):
    actions: list[RouteAction]
    needs_clarification: bool = False
    clarification_question: str = ""
    fallback_reason: str = ""


def _get_client() -> OpenAI:
    if not os.getenv("API_GATEWAY_KEY", "").strip():
        candidate_paths = [
            Path(__file__).resolve().parents[2] / ".secrets",  # 05_src/.secrets
            Path(__file__).resolve().parents[2] / ".env",  # 05_src/.env
            Path(__file__).resolve().parents[1] / ".secrets",  # assignment_chat/.secrets
            Path(".secrets"),  # cwd/.secrets
        ]
        for env_path in candidate_paths:
            if env_path.exists():
                load_dotenv(env_path, override=False)

    gateway_key = os.getenv("API_GATEWAY_KEY", "").strip()
    if not gateway_key:
        raise ValueError(
            "API_GATEWAY_KEY is not set. Checked process env and common .secrets paths (e.g., 05_src/.secrets)."
        )

    return OpenAI(
        base_url=OPENAI_GATEWAY_BASE_URL,
        api_key="placeholder",
        default_headers={"x-api-key": gateway_key},
    )


def _history_to_lines(history: list[dict], max_messages: int = 8) -> list[str]:
    lines: list[str] = []
    for item in history[-max_messages:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        if role not in {"user", "assistant"}:
            continue
        content = item.get("content", "")
        text = content if isinstance(content, str) else str(content)
        text = " ".join(text.strip().split())
        if not text:
            continue
        if len(text) > 300:
            text = text[:300] + "..."
        lines.append(f"{role}: {text}")
    return lines


def _default_question(message: str) -> str:
    lowered = message.lower()
    if "service 2" in lowered or "troubleshoot" in lowered:
        return "I can run Service 2. Which issue text should I use?"
    if "rfi" in lowered:
        return "I can draft the RFI. Which issue should it be based on?"
    return "Do you want weather risk (Service 1), semantic troubleshooting (Service 2), or triage/RFI tooling (Service 3)?"


def _safe_parse_decision(raw_content: str) -> RouteDecision:
    try:
        return RouteDecision.model_validate_json(raw_content)
    except Exception:
        try:
            payload = json.loads(raw_content)
            if isinstance(payload, dict):
                return RouteDecision.model_validate(payload)
        except Exception:
            pass
    raise ValueError("Router response is not valid RouteDecision JSON.")


def _normalize_decision(decision: RouteDecision, message: str) -> RouteDecision:
    actions = decision.actions[:ROUTER_MAX_ACTIONS]

    if actions:
        low_conf = all(action.confidence < ROUTER_MIN_CONFIDENCE for action in actions)
        if low_conf:
            return RouteDecision(
                actions=[],
                needs_clarification=True,
                clarification_question=decision.clarification_question or _default_question(message),
                fallback_reason=decision.fallback_reason,
            )
        return RouteDecision(
            actions=actions,
            needs_clarification=False,
            clarification_question="",
            fallback_reason=decision.fallback_reason,
        )

    if decision.needs_clarification:
        return RouteDecision(
            actions=[],
            needs_clarification=True,
            clarification_question=decision.clarification_question or _default_question(message),
            fallback_reason=decision.fallback_reason,
        )

    return RouteDecision(
        actions=[],
        needs_clarification=True,
        clarification_question=_default_question(message),
        fallback_reason=decision.fallback_reason or "No actionable route from router.",
    )


def decide_route(message: str, history: list[dict], memory_issue: str | None) -> RouteDecision:
    try:
        client = _get_client()
    except Exception as exc:
        return RouteDecision(actions=[], fallback_reason=f"router_client_error: {exc}")

    system_prompt = (
        "You are a strict router for an HVAC/MEP chat assistant. "
        "Return JSON only with this exact schema keys: "
        "actions (array), needs_clarification (bool), clarification_question (string), fallback_reason (string). "
        "Each action must include: service, intent, request_text, use_memory_issue, confidence. "
        "Services: "
        "service1=weather API risk summary, "
        "service2=semantic troubleshooting retrieval (MEP KB), "
        "service3=triage/RFI function-calling. "
        "Rules: "
        "1) If user asks 'service 2 for that/this issue' or equivalent follow-up, pick service2 and set use_memory_issue=true. "
        "2) If user asks to draft/make/create an RFI, pick service3 with intent='draft_rfi'. "
        "3) If user explicitly asks triage, pick service3 with intent='triage'. "
        "4) If message includes two intents, return up to 2 ordered actions reflecting user order. "
        "5) If unclear, set needs_clarification=true and ask one concise question. "
        "6) confidence must be 0.0-1.0."
    )

    user_payload = {
        "message": message,
        "memory_issue": memory_issue or "",
        "recent_history": _history_to_lines(history=history, max_messages=8),
        "max_actions": ROUTER_MAX_ACTIONS,
    }

    try:
        response = client.chat.completions.create(
            model=ROUTER_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload)},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        parsed = _safe_parse_decision(content)
        return _normalize_decision(parsed, message=message)
    except Exception as exc:
        return RouteDecision(actions=[], fallback_reason=f"router_runtime_error: {exc}")

