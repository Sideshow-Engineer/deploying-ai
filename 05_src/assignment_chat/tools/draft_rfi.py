from __future__ import annotations

from typing import Any


def draft_rfi(
    project: str,
    system: str,
    location_scope: str,
    issue_summary: str,
    observations: list[str],
    question_to_designer: str,
    needed_by_date: str,
    attachments_suggested: list[str] | None = None,
) -> dict[str, Any]:
    obs_lines = [f"- {item}" for item in observations if item and item.strip()]
    observations_block = "\n".join(obs_lines) if obs_lines else "- No field observations provided."

    default_attachments = [
        "Trend logs (commands, feedback, and alarms)",
        "Site photos or screenshots",
        "Relevant sequence-of-operations excerpt",
    ]
    attachment_list = attachments_suggested or default_attachments

    subject = f"RFI: {project} | {system} | {location_scope}"
    background = (
        f"Issue summary:\n{issue_summary}\n\n"
        f"Observed conditions:\n{observations_block}"
    )
    question = question_to_designer
    impact_if_unresolved = (
        "Potential impacts include schedule delay, repeat commissioning effort, "
        "and unresolved performance/risk exposure in operations."
    )
    proposed_direction_optional = (
        "Pending designer response, maintain conservative operating setpoints and "
        "avoid permanent logic changes that could conflict with design intent."
    )

    return {
        "subject": subject,
        "background": background,
        "question": question,
        "impact_if_unresolved": impact_if_unresolved,
        "proposed_direction_optional": proposed_direction_optional,
        "attachments_suggested": attachment_list,
        "needed_by_date": needed_by_date,
    }

