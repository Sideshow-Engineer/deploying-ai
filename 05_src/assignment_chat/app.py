from __future__ import annotations

import re
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from guardrails import guardrail_precheck


PERSONA = (
    "You are Systems Pilot, a practical AI engineering assistant. "
    "You are concise, technical, and friendly."
)

RAG_HINTS = (
    "ahu",
    "vav",
    "chw",
    "cleanroom",
    "controls",
    "tab",
    "rfi",
    "symptom",
    "commissioning",
    "startup",
)

WEATHER_HINTS = (
    "weather",
    "forecast",
    "temperature",
    "wind",
    "gust",
    "rain",
    "snow",
    "precip",
    "freez",
    "heat",
    "meteo",
)

FUNCTION_HINTS = (
    "triage",
    "rfi",
    "request for information",
    "draft rfi",
)

DISAMBIG_WEATHER_RAG = (
    "Do you want weather risk (Service 1) or MEP troubleshooting (Service 2)?"
)
DISAMBIG_RAG_FUNCTION = (
    "Do you want MEP troubleshooting retrieval (Service 2) or structured triage/RFI tooling (Service 3)?"
)
def _intent_score(message: str, hints: tuple[str, ...]) -> int:
    lowered = message.lower()
    return sum(1 for hint in hints if hint in lowered)


def rag_intent_score(message: str) -> int:
    return _intent_score(message, RAG_HINTS)


def weather_intent_score(message: str) -> int:
    return _intent_score(message, WEATHER_HINTS)


def function_intent_score(message: str) -> int:
    return _intent_score(message, FUNCTION_HINTS)


def _history_text(item: dict) -> str:
    content = item.get("content", "")
    if isinstance(content, str):
        return content
    return str(content)


def _last_message_by_role(history: list[dict], role: str) -> str:
    for item in reversed(history):
        if isinstance(item, dict) and item.get("role") == role:
            return _history_text(item)
    return ""


def _pending_disambiguation(history: list[dict]) -> str | None:
    last_assistant = _last_message_by_role(history, "assistant")
    if DISAMBIG_WEATHER_RAG in last_assistant:
        return "weather_vs_rag"
    if DISAMBIG_RAG_FUNCTION in last_assistant:
        return "rag_vs_function"
    return None


def _parse_service_choice(message: str) -> str | None:
    lowered = " ".join(message.lower().strip().split())
    if not lowered:
        return None

    # Explicit service references
    match = re.search(r"\bservice\s*([123])\b", lowered)
    if match:
        return f"service{match.group(1)}"
    if lowered in {"1", "2", "3", "option 1", "option 2", "option 3"}:
        return f"service{lowered[-1]}"

    # Short-form semantic choices for disambiguation follow-ups.
    if any(token in lowered for token in ("weather", "forecast", "meteo")):
        return "service1"
    if any(token in lowered for token in ("service 2", "rag", "retrieval", "semantic")):
        return "service2"
    if any(token in lowered for token in ("service 3", "triage", "draft rfi", "function")):
        return "service3"

    return None


def _run_service3(message: str, history: list[dict] | None = None) -> str:
    try:
        from services.function_calling_service import run_service3_function_calling

        tool_response = run_service3_function_calling(
            user_message=message,
            history=history,
        )
        return f"{PERSONA}\n\n{tool_response}"
    except Exception as exc:
        return (
            f"{PERSONA}\n\n"
            "Service 3 (function-calling tools) is unavailable right now.\n"
            "Please check API gateway access and try again.\n"
            f"Details: {exc}"
        )


def _run_service1(message: str) -> str:
    try:
        from services.weather_service import get_construction_weather_risk

        result = get_construction_weather_risk(message)
        cues = "\n".join(f"- {cue}" for cue in result.action_cues)
        return (
            f"{PERSONA}\n\n"
            f"Location: {result.location}\n"
            f"Window: {result.start_date} to {result.end_date}\n\n"
            f"{result.risk_summary}\n\n"
            f"Action cues:\n{cues}\n\n"
            f"{result.source_note}"
        )
    except Exception as exc:
        return (
            f"{PERSONA}\n\n"
            "Service 1 (weather API risk summary) is unavailable right now.\n"
            "Please check network access and try again.\n"
            f"Details: {exc}"
        )


def _run_service2(message: str) -> str:
    try:
        from services.rag_service import query_issue_kb

        result = query_issue_kb(message)
        citations = "\n".join(f"- {entry}" for entry in result.citations)
        return f"{PERSONA}\n\n{result.answer}\n\nCitations:\n{citations}"
    except Exception as exc:
        return (
            f"{PERSONA}\n\n"
            "Service 2 (semantic query) is configured but unavailable right now.\n"
            "Run `python 05_src/assignment_chat/scripts/build_chroma.py` first.\n"
            f"Details: {exc}"
        )


def assignment_chat(message: str, history: list[dict]) -> str:
    """
    Initial chat scaffold for Assignment 2.
    This currently provides guardrails + personality while service handlers
    are implemented incrementally.
    """
    guardrail_reply = guardrail_precheck(message)
    if guardrail_reply:
        return guardrail_reply

    pending = _pending_disambiguation(history)
    choice = _parse_service_choice(message)
    if pending and choice:
        original_query = _last_message_by_role(history, "user")
        target_query = original_query or message

        if pending == "weather_vs_rag":
            if choice == "service1":
                return _run_service1(target_query)
            if choice == "service2":
                return _run_service2(target_query)
            return (
                f"{PERSONA}\n\n"
                "Please choose `Service 1` for weather risk or `Service 2` for MEP troubleshooting."
            )

        if pending == "rag_vs_function":
            if choice == "service2":
                return _run_service2(target_query)
            if choice == "service3":
                return _run_service3(target_query, history=history)
            return (
                f"{PERSONA}\n\n"
                "Please choose `Service 2` for retrieval or `Service 3` for triage/RFI tooling."
            )

    weather_score = weather_intent_score(message)
    rag_score = rag_intent_score(message)
    function_score = function_intent_score(message)

    if weather_score > 0 and rag_score > 0 and function_score == 0:
        return (
            f"{PERSONA}\n\n"
            "I detected both weather and MEP troubleshooting intent.\n"
            f"{DISAMBIG_WEATHER_RAG}"
        )

    if function_score > 0 and rag_score > 0:
        return (
            f"{PERSONA}\n\n"
            "I detected both semantic troubleshooting and function-calling intent.\n"
            f"{DISAMBIG_RAG_FUNCTION}"
        )

    if function_score > 0:
        return _run_service3(message, history=history)

    if weather_score > 0:
        return _run_service1(message)

    if rag_score > 0:
        return _run_service2(message)

    # Placeholder response for Services 1 and 3 while they are implemented.
    return (
        f"{PERSONA}\n\n"
        "Scaffold is running. Services currently available:\n"
        "1) Service 1 weather risk summary (Open-Meteo)\n"
        "2) Service 2 semantic query (MEP issue KB)\n\n"
        "3) Service 3 function-calling tools (triage_issue + draft_rfi)\n\n"
        f"Your message: {message}"
    )


def create_app() -> gr.ChatInterface:
    return gr.ChatInterface(fn=assignment_chat, type="messages")


def _load_env_files() -> None:
    # Match course layout first, then fall back to repo-root/local variants.
    candidate_paths = [
        Path(__file__).resolve().parents[1] / ".secrets",  # 05_src/.secrets
        Path(__file__).resolve().parents[1] / ".env",      # 05_src/.env
        Path(__file__).resolve().parent / ".secrets",      # assignment_chat/.secrets
        Path(".secrets"),                                  # cwd/.secrets
    ]
    for env_path in candidate_paths:
        if env_path.exists():
            load_dotenv(env_path, override=False)


if __name__ == "__main__":
    _load_env_files()
    app = create_app()
    app.launch()
