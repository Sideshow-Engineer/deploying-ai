from __future__ import annotations

import re

import gradio as gr
from dotenv import load_dotenv


PERSONA = (
    "You are Systems Pilot, a practical AI engineering assistant. "
    "You are concise, technical, and friendly."
)

RESTRICTED_TOPICS_PATTERN = re.compile(
    r"\b(cat|cats|dog|dogs|horoscope|horoscopes|zodiac|taylor swift)\b",
    flags=re.IGNORECASE,
)
PROMPT_ATTACK_PATTERN = re.compile(
    r"(system prompt|developer prompt|ignore previous|reveal.*prompt|jailbreak)",
    flags=re.IGNORECASE,
)


def blocked_message() -> str:
    return (
        "I cannot help with that request. Please ask about supported assignment "
        "services instead."
    )


def should_use_rag(message: str) -> bool:
    rag_hints = (
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
    lowered = message.lower()
    return any(hint in lowered for hint in rag_hints)


def assignment_chat(message: str, history: list[dict]) -> str:
    """
    Initial chat scaffold for Assignment 2.
    This currently provides guardrails + personality while service handlers
    are implemented incrementally.
    """
    if PROMPT_ATTACK_PATTERN.search(message):
        return blocked_message()

    if RESTRICTED_TOPICS_PATTERN.search(message):
        return blocked_message()

    if should_use_rag(message):
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

    # Placeholder response for Services 1 and 3 while they are implemented.
    return (
        f"{PERSONA}\n\n"
        "Scaffold is running. Service 2 (semantic query) is available for MEP issues.\n"
        "Next step is connecting Services 1 and 3.\n\n"
        f"Your message: {message}"
    )


def create_app() -> gr.ChatInterface:
    return gr.ChatInterface(fn=assignment_chat, type="messages")


if __name__ == "__main__":
    load_dotenv(".secrets")
    app = create_app()
    app.launch()
