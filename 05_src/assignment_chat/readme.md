# Assignment 2 Chat Client

## Overview
This project implements a Gradio chat assistant named **Systems Pilot**.  
The client provides three services through one interface:

1. API service (Open-Meteo weather -> construction risk summary)
2. Semantic query service (local MEP issue knowledge base with ChromaDB)
3. Function-calling service (`triage_issue` and `draft_rfi`)

The chat keeps conversation history and routes user requests by intent.

## Service 1: API Calls (Open-Meteo)
- Runtime module: `services/weather_service.py`
- Backend APIs:
  - Open-Meteo geocoding API
  - Open-Meteo forecast API
- Input handling:
  - location (default: Toronto)
  - date intent (`today`, `tomorrow`, `next 3 days`, or explicit date/range)
- Output:
  - transformed weather risk summary for commissioning/site planning
  - action cues (not raw JSON)

### Service 1 Heuristics
Threshold-based and explainable rules are used:
- wind risk: gust >= 45 km/h or sustained wind >= 30 km/h
- freeze risk: min temperature <= 1C
- precipitation risk: precipitation >= 10 mm/day or precipitation hours >= 6 h/day
- snow risk: snowfall >= 1 cm/day
- heat risk: max temperature >= 30C

## Service 2: Semantic Query (RAG-style retrieval)
- Runtime module: `services/rag_service.py`
- Dataset: `data/mep_issue_kb.csv`
- Vector store: persistent ChromaDB in `chroma_db/`
- Build script: `scripts/build_chroma.py`
- Retrieval metadata includes:
  - `root_causes`
  - `field_checks`
  - `recommended_actions`
  - `rfi_template_question`
  - `risk_tags`
- Output:
  - grounded troubleshooting response
  - citations (`MEP-xxx: title`)

### Build/Refresh the Vector Index
```bash
python 05_src/assignment_chat/scripts/build_chroma.py
```
Run this after any KB CSV/schema updates so metadata fields stay in sync.

### Evaluate Retrieval/Groundedness
```bash
python 05_src/assignment_chat/scripts/eval_rag.py
```

Optional JSON report:
```bash
python 05_src/assignment_chat/scripts/eval_rag.py --json-out 05_src/assignment_chat/data/rag_eval_report.json
```

## Service 3: Function Calling
- Runtime module: `services/function_calling_service.py`
- Tool modules:
  - `lookup_issue_kb` (wrapper around Service 2 KB lookup)
  - `tools/triage_issue.py`
  - `tools/draft_rfi.py`
- Assignment tool option used: **Function Calling**
- Multi-step tool loop supported (up to 3 steps):
  - typical RFI flow: `lookup_issue_kb -> triage_issue -> draft_rfi`
- Recent conversation turns are passed to Service 3 for follow-up continuity.

### Tool: `lookup_issue_kb`
Input:
- `query_text` (required)
- `top_k` (optional)

Output:
- `best_match` (id/title/system/stage/location plus key troubleshooting fields)
- `related_matches`
- `citation`

### Tool: `triage_issue`
Input:
- `issue_text` (required)
- `system_hint` (optional)
- `stage_hint` (optional)

Output:
- `risk_level`
- `risk_dimensions`
- `likely_disciplines`
- `recommended_next_steps`
- `needs_rfi`
- `rfi_reason`

### Tool: `draft_rfi`
Input:
- `project`
- `system`
- `location_scope`
- `issue_summary`
- `observations` (list)
- `question_to_designer`
- `needed_by_date`
- `attachments_suggested` (optional)

Output:
- `subject`
- `background`
- `question`
- `impact_if_unresolved`
- `proposed_direction_optional`
- `attachments_suggested`
- `needed_by_date`

## Guardrails
Guardrails are implemented in `guardrails.py` and run as a pre-check before service routing.

Blocked categories:
- Attempts to reveal/modify instructions (e.g., prompt injection patterns)
- Restricted topics from assignment requirements:
  - cats/dogs
  - horoscopes/zodiac
  - Taylor Swift

Behavior:
- Returns a brief refusal message when triggered.

## Run the App
```bash
python 05_src/assignment_chat/app.py
```

## Example Prompts

### Service 1
- `Weather risk for Toronto tomorrow`
- `Construction forecast in Mississauga next 3 days`
- `Forecast in Hamilton 2026-02-20 to 2026-02-22`

### Service 2
- `AHU commissioning issue: OA damper command changes but OA flow stays low`
- `VAV zone overheats while reheat command is zero`
- `CHW pump VFD hunting and unstable differential pressure`

### Service 3
- `Triage this issue: AHU mixed air temperature oscillates during economizer changeover.`
- `Draft RFI for project North Tower, system AHU-3, location mechanical room, issue economizer hunting, observations: MAT swings; dampers move continuously, question: confirm deadband and changeover strategy, needed by 2026-02-28.`
