from __future__ import annotations

import logging
import re
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from guardrails import guardrail_precheck

from config import ROUTER_MAX_ACTIONS, USE_LLM_ROUTER


logger = logging.getLogger(__name__)

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
    "hot",
    "heatwave",
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
RFI_FOLLOWUP_PROMPT = (
    "RFI follow-up: Reply `yes` to draft an RFI for this issue, or `no` to skip."
)


def _intent_score(message: str, hints: tuple[str, ...]) -> int:
    lowered = message.lower()
    return sum(1 for hint in hints if hint in lowered)


def rag_intent_score(message: str) -> int:
    return _intent_score(message, RAG_HINTS)


def weather_intent_score(message: str) -> int:
    lowered = message.lower()
    score = _intent_score(message, WEATHER_HINTS)
    # Count explicit "heat" words but avoid false positives like "reheat".
    if re.search(r"\bheat\b", lowered):
        score += 1
    return score


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

    # Tolerate common misspellings like "servie 2"/"servce 2".
    fuzzy_service_num = re.search(r"\bserv[a-z]*\s*([123])\b", lowered)
    if fuzzy_service_num:
        return f"service{fuzzy_service_num.group(1)}"

    match = re.search(r"\bservice\s*([123])\b", lowered)
    if match:
        return f"service{match.group(1)}"
    if lowered in {"1", "2", "3", "option 1", "option 2", "option 3"}:
        return f"service{lowered[-1]}"
    if lowered in {"service one", "service two", "service three"}:
        return {
            "service one": "service1",
            "service two": "service2",
            "service three": "service3",
        }[lowered]

    if any(token in lowered for token in ("weather", "forecast", "meteo")):
        return "service1"
    if any(
        token in lowered
        for token in (
            "service 2",
            "rag",
            "retrieval",
            "semantic",
            "mep troubleshooting",
            "troubleshooting",
            "service two",
        )
    ):
        return "service2"
    if any(token in lowered for token in ("service 3", "triage", "draft rfi", "function")):
        return "service3"

    return None


def _is_affirmative(message: str) -> bool:
    lowered = " ".join(message.lower().strip().split())
    return lowered in {
        "y",
        "yes",
        "yep",
        "sure",
        "ok",
        "okay",
        "need",
        "need rfi",
        "draft rfi",
        "please do",
    }


def _is_negative(message: str) -> bool:
    lowered = " ".join(message.lower().strip().split())
    return lowered in {"n", "no", "nope", "skip", "not now"}


def _is_rfi_followup_request(message: str) -> bool:
    lowered = " ".join(message.lower().strip().split())
    return any(
        phrase in lowered
        for phrase in (
            "draft it",
            "create it",
            "make it",
            "write it",
            "turn this into rfi",
            "convert to rfi",
        )
    )


def _is_explicit_rfi_request(message: str) -> bool:
    lowered = " ".join(message.lower().strip().split())
    patterns = (
        r"\b(draft|create|make|write)\b.*\brfi\b",
        r"\brfi\b.*\b(issue|that|this|it)\b",
        r"\brequest for information\b",
    )
    return any(re.search(pattern, lowered) for pattern in patterns)


def _is_reference_only_request(message: str) -> bool:
    lowered = " ".join(message.lower().strip().split())
    if any(
        phrase in lowered
        for phrase in (
            "that issue",
            "this issue",
            "same issue",
            "for that issue",
            "on that issue",
        )
    ):
        return True
    return lowered in {
        "rfi",
        "draft rfi",
        "make rfi",
        "create rfi",
        "write rfi",
        "also draft an rfi",
        "also make an rfi",
    }


def _references_prior_issue(message: str) -> bool:
    lowered = " ".join(message.lower().strip().split())
    return any(
        phrase in lowered
        for phrase in (
            "that issue",
            "this issue",
            "last issue",
            "same issue",
            "for that issue",
            "on that issue",
            "for the last issue",
        )
    )


def _has_explicit_rfi_fields(message: str) -> bool:
    lowered = " ".join(message.lower().strip().split())
    markers = (
        "project ",
        "system ",
        "location ",
        "issue ",
        "observations:",
        "question:",
        "needed by",
        "needed_by_date",
    )
    return sum(1 for marker in markers if marker in lowered) >= 3


def _extract_triage_issue_text(message: str) -> str:
    text = message.strip()
    patterns = (
        r"(?is)^\s*triage\s+this\s+issue\s*:\s*(.+)$",
        r"(?is)^\s*triage\s*:\s*(.+)$",
        r"(?is)^\s*triage\s+(.+)$",
    )
    for pattern in patterns:
        match = re.match(pattern, text)
        if match:
            return match.group(1).strip()
    return text


def _is_explicit_triage_request(message: str) -> bool:
    lowered = message.lower().strip()
    if lowered in {"triage", "service 3", "structured triage/rfi tooling"}:
        return False
    return bool(re.match(r"^\s*triage\b", lowered))


def _pending_rfi_issue_from_history(history: list[dict]) -> str | None:
    if not history:
        return None

    for idx in range(len(history) - 1, -1, -1):
        item = history[idx]
        if not isinstance(item, dict):
            continue
        if item.get("role") != "assistant":
            continue
        assistant_text = _history_text(item)
        if RFI_FOLLOWUP_PROMPT not in assistant_text:
            continue
        for jdx in range(idx - 1, -1, -1):
            prev = history[jdx]
            if not isinstance(prev, dict) or prev.get("role") != "user":
                continue
            issue_text = _extract_triage_issue_text(_history_text(prev))
            return issue_text if issue_text else None
        return None
    return None


def _is_service2_answer(text: str) -> bool:
    return "Best match:" in text and "Citations:" in text


def _last_rag_issue_from_history(history: list[dict]) -> str | None:
    if not history:
        return None

    for idx in range(len(history) - 1, -1, -1):
        item = history[idx]
        if not isinstance(item, dict) or item.get("role") != "assistant":
            continue
        if not _is_service2_answer(_history_text(item)):
            continue

        for jdx in range(idx - 1, -1, -1):
            prev = history[jdx]
            if not isinstance(prev, dict) or prev.get("role") != "user":
                continue
            candidate = _history_text(prev).strip()
            if not candidate:
                continue
            if _parse_service_choice(candidate):
                continue
            return candidate
        return None
    return None


def _is_service3_rfi_answer(text: str) -> bool:
    lowered = text.lower()
    return (
        ("rfi draft:" in lowered)
        or ("subject: rfi:" in lowered)
        or ("drafted rfi" in lowered)
        or ("request for information" in lowered)
    )


def _extract_issue_from_rfi_answer(text: str) -> str:
    match = re.search(
        r"(?is)issue summary:\s*(.+?)(?:\n\s*observed conditions:|\n\s*question:|\n\n|$)",
        text,
    )
    if not match:
        return ""
    issue = " ".join(match.group(1).split())
    issue = re.sub(r"\s*\(based on kb:.*?\)\s*$", "", issue, flags=re.IGNORECASE)
    return issue.strip(" .")


def _last_service3_issue_from_history(history: list[dict]) -> str | None:
    if not history:
        return None

    for item in reversed(history):
        if not isinstance(item, dict) or item.get("role") != "assistant":
            continue
        text = _history_text(item)
        if not _is_service3_rfi_answer(text):
            continue
        issue = _extract_issue_from_rfi_answer(text)
        if issue:
            return issue
    return None


def resolve_last_issue(history: list[dict]) -> str | None:
    if not history:
        return None

    # Use recency across sources instead of fixed source-priority, so follow-ups
    # refer to the latest issue context in the active chat flow.
    for idx in range(len(history) - 1, -1, -1):
        item = history[idx]
        if not isinstance(item, dict):
            continue

        role = item.get("role")
        text = _history_text(item).strip()
        if not text:
            continue

        if role == "user" and _is_explicit_triage_request(text):
            issue = _extract_triage_issue_text(text)
            if issue:
                return issue

        if role == "assistant" and _is_service2_answer(text):
            for jdx in range(idx - 1, -1, -1):
                prev = history[jdx]
                if not isinstance(prev, dict) or prev.get("role") != "user":
                    continue
                candidate = _history_text(prev).strip()
                if not candidate:
                    continue
                if _parse_service_choice(candidate):
                    continue
                return candidate

        if role == "assistant" and _is_service3_rfi_answer(text):
            issue = _extract_issue_from_rfi_answer(text)
            if issue:
                return issue

    return None


def _run_explicit_triage(message: str) -> str:
    issue_text = _extract_triage_issue_text(message)
    try:
        from services.function_calling_service import (
            format_triage_with_lookup_output,
            triage_with_lookup,
        )

        result = triage_with_lookup(issue_text=issue_text)
        triage_info = result.get("triage", {})
        needs_rfi = bool(triage_info.get("needs_rfi"))
        recommendation = (
            "RFI recommendation: triage indicates design clarification is likely needed."
            if needs_rfi
            else "RFI recommendation: triage does not require RFI by default, but you can still draft one."
        )
        response_text = (
            f"{format_triage_with_lookup_output(result)}\n\n"
            f"{recommendation}\n"
            f"{RFI_FOLLOWUP_PROMPT}"
        )
        return f"{PERSONA}\n\n{response_text}"
    except Exception as exc:
        return (
            f"{PERSONA}\n\n"
            "Service 3 triage is unavailable right now.\n"
            "Please check API gateway access and try again.\n"
            f"Details: {exc}"
        )


def _run_rfi_from_issue(issue_text: str) -> str:
    try:
        from services.function_calling_service import (
            draft_rfi_from_issue,
            format_draft_rfi_from_issue_output,
        )

        rfi_result = draft_rfi_from_issue(issue_text=issue_text)
        return f"{PERSONA}\n\n{format_draft_rfi_from_issue_output(rfi_result)}"
    except Exception as exc:
        return (
            f"{PERSONA}\n\n"
            "Service 3 RFI drafting is unavailable right now.\n"
            "Please check API gateway access and try again.\n"
            f"Details: {exc}"
        )


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


def _strip_persona_prefix(text: str) -> str:
    prefix = f"{PERSONA}\n\n"
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def _dispatch_router_action(
    action: object,
    message: str,
    history: list[dict],
    memory_issue: str | None,
) -> tuple[str, bool]:
    service = getattr(action, "service")
    intent = getattr(action, "intent")
    request_text = (getattr(action, "request_text") or "").strip() or message
    use_memory_issue = bool(getattr(action, "use_memory_issue", False))

    if service == "service1":
        return _strip_persona_prefix(_run_service1(request_text)), False

    if service == "service2":
        if use_memory_issue or _references_prior_issue(request_text):
            if not memory_issue:
                return "I can run Service 2, but I need the issue text to use.", True
            return _strip_persona_prefix(_run_service2(memory_issue)), False
        return _strip_persona_prefix(_run_service2(request_text)), False

    if service == "service3":
        if intent == "draft_rfi":
            should_use_memory = (
                (use_memory_issue or _is_reference_only_request(request_text))
                and not _has_explicit_rfi_fields(request_text)
            )
            if should_use_memory:
                if not memory_issue:
                    return "I can draft the RFI, but I need the issue text to use.", True
                return _strip_persona_prefix(_run_rfi_from_issue(memory_issue)), False
            return _strip_persona_prefix(_run_service3(request_text, history=history)), False
        return _strip_persona_prefix(_run_service3(request_text, history=history)), False

    return "I need a clearer request to choose a service.", True


def _execute_router_plan(
    message: str,
    history: list[dict],
    memory_issue: str | None,
) -> str | None:
    try:
        from services.router_service import decide_route
    except Exception as exc:
        logger.warning("router import failed: %s", exc)
        return None

    decision = decide_route(
        message=message,
        history=history,
        memory_issue=memory_issue,
    )

    if decision.needs_clarification:
        question = decision.clarification_question or (
            "Do you want weather risk (Service 1), semantic troubleshooting (Service 2), "
            "or triage/RFI tooling (Service 3)?"
        )
        return f"{PERSONA}\n\n{question}"

    if not decision.actions:
        if decision.fallback_reason:
            logger.warning("router fallback: %s", decision.fallback_reason)
        return None

    sections: list[tuple[str, str, str]] = []
    for action in decision.actions[:ROUTER_MAX_ACTIONS]:
        content, is_clarification = _dispatch_router_action(
            action=action,
            message=message,
            history=history,
            memory_issue=memory_issue,
        )
        if is_clarification:
            return f"{PERSONA}\n\n{content}"
        sections.append((getattr(action, "service"), getattr(action, "intent"), content))

    if len(sections) == 1:
        body = sections[0][2]
    else:
        body = "\n\n".join(
            f"Service {service[-1]} ({intent}) result:\n{content}"
            for service, intent, content in sections
        )
    return f"{PERSONA}\n\n{body}"


def _deterministic_route(
    message: str,
    history: list[dict],
    memory_issue: str | None,
) -> str:
    if _is_explicit_rfi_request(message):
        if memory_issue and _is_reference_only_request(message):
            return _run_rfi_from_issue(memory_issue)
        return _run_service3(message, history=history)

    if memory_issue and (_is_affirmative(message) or _is_rfi_followup_request(message)):
        return _run_rfi_from_issue(memory_issue)

    if _is_explicit_triage_request(message):
        return _run_explicit_triage(message)

    pending = _pending_disambiguation(history)
    choice = _parse_service_choice(message)
    if not pending and choice:
        if choice == "service1":
            return _run_service1(message)
        if choice == "service2":
            if _references_prior_issue(message):
                if memory_issue:
                    return _run_service2(memory_issue)
                return (
                    f"{PERSONA}\n\n"
                    "I can run Service 2, but I need the issue text to use."
                )
            return _run_service2(message)
        if choice == "service3":
            if _references_prior_issue(message):
                if memory_issue:
                    return _run_rfi_from_issue(memory_issue)
                return (
                    f"{PERSONA}\n\n"
                    "I can run Service 3, but I need the issue text to use."
                )
            return _run_service3(message, history=history)

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

    return (
        f"{PERSONA}\n\n"
        "Scaffold is running. Services currently available:\n"
        "1) Service 1 weather risk summary (Open-Meteo)\n"
        "2) Service 2 semantic query (MEP issue KB)\n\n"
        "3) Service 3 function-calling tools (triage_issue + draft_rfi)\n\n"
        f"Your message: {message}"
    )


def assignment_chat(message: str, history: list[dict]) -> str:
    guardrail_reply = guardrail_precheck(message)
    if guardrail_reply:
        return guardrail_reply

    # Keep explicit triage yes/no follow-up shortcut.
    pending_issue = _pending_rfi_issue_from_history(history)
    if pending_issue:
        if _is_affirmative(message):
            return _run_rfi_from_issue(pending_issue)
        if _is_negative(message):
            return (
                f"{PERSONA}\n\n"
                "Understood. I will skip RFI drafting for this issue. "
                "Send a new issue anytime."
            )

    memory_issue = resolve_last_issue(history)

    # Fast path for common follow-up: draft RFI using latest issue context.
    if (
        memory_issue
        and _references_prior_issue(message)
        and (_is_rfi_followup_request(message) or _is_explicit_rfi_request(message))
        and not _has_explicit_rfi_fields(message)
    ):
        return _run_rfi_from_issue(memory_issue)

    # Fast path: explicit service selection should execute directly, not wait for LLM routing.
    if _parse_service_choice(message):
        return _deterministic_route(
            message=message,
            history=history,
            memory_issue=memory_issue,
        )

    if USE_LLM_ROUTER:
        router_response = _execute_router_plan(
            message=message,
            history=history,
            memory_issue=memory_issue,
        )
        if router_response is not None:
            return router_response

    return _deterministic_route(
        message=message,
        history=history,
        memory_issue=memory_issue,
    )


def create_app() -> gr.ChatInterface:
    return gr.ChatInterface(fn=assignment_chat, type="messages")


def _load_env_files() -> None:
    candidate_paths = [
        Path(__file__).resolve().parents[1] / ".secrets",  # 05_src/.secrets
        Path(__file__).resolve().parents[1] / ".env",  # 05_src/.env
        Path(__file__).resolve().parent / ".secrets",  # assignment_chat/.secrets
        Path(".secrets"),  # cwd/.secrets
    ]
    for env_path in candidate_paths:
        if env_path.exists():
            load_dotenv(env_path, override=False)


if __name__ == "__main__":
    _load_env_files()
    app = create_app()
    app.launch()
