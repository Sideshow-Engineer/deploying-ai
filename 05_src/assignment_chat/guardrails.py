from __future__ import annotations

import re

# Assignment-restricted topics
RESTRICTED_TOPICS_PATTERN = re.compile(
    r"\b(cat|cats|dog|dogs|horoscope|horoscopes|zodiac|zodiac signs|taylor swift)\b",
    flags=re.IGNORECASE,
)

# Prompt-protection checks
PROMPT_ATTACK_PATTERN = re.compile(
    r"(system prompt|developer prompt|ignore previous|ignore instructions|reveal.*prompt|jailbreak)",
    flags=re.IGNORECASE,
)


def guardrail_precheck(message: str) -> str | None:
    """Return a brief refusal message when a guardrail is triggered."""
    if PROMPT_ATTACK_PATTERN.search(message):
        return "I cannot help with requests to reveal or modify system instructions."

    if RESTRICTED_TOPICS_PATTERN.search(message):
        return "I cannot help with that topic. Please ask about assignment services."

    return None

