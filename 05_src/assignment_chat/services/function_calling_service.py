from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from config import FUNCTION_CALLING_MODEL, OPENAI_GATEWAY_BASE_URL
from services.rag_service import lookup_issue_kb
from tools.draft_rfi import draft_rfi
from tools.triage_issue import triage_issue

MAX_TOOL_STEPS = 3

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "lookup_issue_kb",
            "description": (
                "Lookup local MEP issue KB and return best match with root causes, "
                "field checks, recommended actions, and RFI prompt."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query_text": {
                        "type": "string",
                        "description": "Issue/query text for knowledge-base lookup.",
                    },
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 8,
                        "description": "Number of matches to retrieve.",
                    },
                },
                "required": ["query_text"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "triage_issue",
            "description": (
                "Classify issue risk level and provide practical next steps for "
                "commissioning/startup troubleshooting."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "issue_text": {
                        "type": "string",
                        "description": "Free-form issue description from user.",
                    },
                    "system_hint": {
                        "type": "string",
                        "description": "Optional system hint (AHU/VAV/CHW/etc.).",
                    },
                    "stage_hint": {
                        "type": "string",
                        "description": "Optional stage hint (Startup/Commissioning/TAB).",
                    },
                },
                "required": ["issue_text"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "draft_rfi",
            "description": "Draft a professional RFI from structured fields.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project": {"type": "string"},
                    "system": {"type": "string"},
                    "location_scope": {"type": "string"},
                    "issue_summary": {"type": "string"},
                    "observations": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "question_to_designer": {"type": "string"},
                    "needed_by_date": {"type": "string"},
                    "attachments_suggested": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "project",
                    "system",
                    "location_scope",
                    "issue_summary",
                    "observations",
                    "question_to_designer",
                    "needed_by_date",
                ],
                "additionalProperties": False,
            },
        },
    },
]


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


def _safe_json_loads(raw: str) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        loaded = json.loads(raw)
        if isinstance(loaded, dict):
            return loaded
    except Exception:
        pass
    return {}


def _execute_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name == "lookup_issue_kb":
        try:
            top_k = int(arguments.get("top_k", 5))
        except Exception:
            top_k = 5
        top_k = min(max(top_k, 1), 8)
        return lookup_issue_kb(
            query_text=arguments.get("query_text", ""),
            top_k=top_k,
        )
    if name == "triage_issue":
        return triage_issue(
            issue_text=arguments.get("issue_text", ""),
            system_hint=arguments.get("system_hint"),
            stage_hint=arguments.get("stage_hint"),
        )
    if name == "draft_rfi":
        return draft_rfi(
            project=arguments.get("project", ""),
            system=arguments.get("system", ""),
            location_scope=arguments.get("location_scope", ""),
            issue_summary=arguments.get("issue_summary", ""),
            observations=arguments.get("observations", []),
            question_to_designer=arguments.get("question_to_designer", ""),
            needed_by_date=arguments.get("needed_by_date", ""),
            attachments_suggested=arguments.get("attachments_suggested"),
        )
    raise ValueError(f"Unknown tool: {name}")


def _format_lookup_output(result: dict[str, Any]) -> str:
    best = result.get("best_match")
    if not best:
        return "KB lookup result: no confident match found."

    related = result.get("related_matches", [])
    related_text = "\n".join(
        f"- {item.get('id', '')}: {item.get('title', '')}" for item in related
    ) or "- None"

    return (
        "KB lookup result:\n"
        f"- citation: {result.get('citation', '')}\n"
        f"- system: {best.get('system', '')}\n"
        f"- stage: {best.get('stage', '')}\n"
        f"- location_scope: {best.get('location_scope', '')}\n"
        f"- root_causes: {best.get('root_causes', '')}\n"
        f"- field_checks: {best.get('field_checks', '')}\n"
        f"- recommended_actions: {best.get('recommended_actions', '')}\n"
        f"- rfi_template_question: {best.get('rfi_template_question', '')}\n"
        f"- risk_tags: {best.get('risk_tags', '')}\n"
        f"- related_matches:\n{related_text}"
    )


def _format_triage_output(result: dict[str, Any]) -> str:
    risk_dimensions = ", ".join(result.get("risk_dimensions", [])) or "None"
    likely_disciplines = ", ".join(result.get("likely_disciplines", [])) or "None"
    recommended_steps = result.get("recommended_next_steps", [])
    steps_text = "\n".join(
        f"{idx}. {step}" for idx, step in enumerate(recommended_steps, start=1)
    ) or "1. No steps provided."

    return (
        "Structured triage result:\n"
        f"- risk_level: {result.get('risk_level', 'Unknown')}\n"
        f"- risk_dimensions: {risk_dimensions}\n"
        f"- likely_disciplines: {likely_disciplines}\n"
        f"- needs_rfi: {result.get('needs_rfi', False)}\n"
        f"- rfi_reason: {result.get('rfi_reason', '')}\n\n"
        "recommended_next_steps:\n"
        f"{steps_text}"
    )


def _format_draft_rfi_output(result: dict[str, Any]) -> str:
    attachments = result.get("attachments_suggested", [])
    attachments_text = "\n".join(f"- {item}" for item in attachments) or "- None"

    return (
        "RFI draft:\n"
        f"subject: {result.get('subject', '')}\n\n"
        f"background:\n{result.get('background', '')}\n\n"
        f"question:\n{result.get('question', '')}\n\n"
        f"impact_if_unresolved:\n{result.get('impact_if_unresolved', '')}\n\n"
        f"proposed_direction_optional:\n{result.get('proposed_direction_optional', '')}\n\n"
        f"attachments_suggested:\n{attachments_text}\n\n"
        f"needed_by_date: {result.get('needed_by_date', '')}"
    )


def _format_tool_output(name: str, result: dict[str, Any]) -> str:
    if name == "lookup_issue_kb":
        return _format_lookup_output(result)
    if name == "triage_issue":
        return _format_triage_output(result)
    if name == "draft_rfi":
        return _format_draft_rfi_output(result)
    return json.dumps(result, indent=2)


def _history_to_messages(
    history: list[dict] | None,
    max_turn_messages: int = 8,
) -> list[dict[str, str]]:
    if not history:
        return []

    normalized: list[dict[str, str]] = []
    for item in history[-max_turn_messages:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        if role not in {"user", "assistant"}:
            continue
        content = item.get("content", "")
        text = content if isinstance(content, str) else str(content)
        text = text.strip()
        if not text:
            continue
        normalized.append({"role": role, "content": text})
    return normalized


def run_service3_function_calling(
    user_message: str,
    history: list[dict] | None = None,
    max_steps: int = MAX_TOOL_STEPS,
) -> str:
    client = _get_client()

    system_prompt = (
        "You are Systems Pilot Service 3. "
        "You may call multiple tools in sequence. "
        "For RFI drafting, prefer workflow: lookup_issue_kb -> triage_issue -> draft_rfi. "
        "If the user references prior context (e.g., 'that', 'same location'), use conversation history. "
        "When KB lookup is used, include citation in your final response."
    )
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    messages.extend(_history_to_messages(history=history, max_turn_messages=8))
    messages.append({"role": "user", "content": user_message})

    executed_tool_outputs: list[tuple[str, dict[str, Any]]] = []

    for _ in range(max_steps):
        response = client.chat.completions.create(
            model=FUNCTION_CALLING_MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0,
        )
        assistant_message = response.choices[0].message
        tool_calls = assistant_message.tool_calls or []

        if tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_message.content or "",
                    "tool_calls": [call.model_dump() for call in tool_calls],
                }
            )
            for call in tool_calls:
                arguments = _safe_json_loads(call.function.arguments or "{}")
                tool_result = _execute_tool(call.function.name, arguments)
                executed_tool_outputs.append((call.function.name, tool_result))
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(tool_result),
                    }
                )
            continue

        if assistant_message.content and assistant_message.content.strip():
            return assistant_message.content
        break

    if executed_tool_outputs:
        return "\n\n".join(
            _format_tool_output(name, result) for name, result in executed_tool_outputs
        )

    return "No tool call was selected. Ask me to triage an issue or draft an RFI."
