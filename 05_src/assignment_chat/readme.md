# Assignment 2 Chat Client (Starter)

## Nature of the Chat Client
This project implements a conversational AI assistant called **Systems Pilot**.
The client is designed to be practical, structured, and implementation-focused.
It responds in a concise technical style and is intended to orchestrate multiple
services behind a single chat interface.

The user interface is chat-based and built with Gradio. Conversation state
(`history`) is preserved across turns through the interface so the assistant
maintains short-term memory.

## Services Provided
The system is being built in three services:

1. API Calls Service
- A tool backed by a public API.
- The API result will be transformed before returning (not verbatim output).

2. Semantic Query Service (Implemented MVP)
- Question answering over a local MEP dataset using semantic retrieval.
- Data source: `data/mep_issue_kb.csv`.
- Storage: ChromaDB persistent directory `chroma_db/`.
- Runtime service: `services/rag_service.py`.
- Index build script: `scripts/build_chroma.py`.
- Output style: short synthesized triage response + issue citations (`MEP-xxx: title`).

3. Custom Service
- One additional service using one of:
  - Function calling
  - Web search
  - MCP server connection

## Guardrails
The chat client applies guardrails to:
- Block attempts to reveal or modify system/developer prompts.
- Refuse restricted topics:
  - Cats or dogs
  - Horoscopes or Zodiac Signs
  - Taylor Swift

## Implementation Decisions
1. Incremental build strategy
- Start with a stable chat scaffold and guardrails, then add one service at a
  time to reduce integration risk.

2. UI first, services second
- Keep a working Gradio interface available from the beginning to test memory,
  tone, and response behavior continuously.

3. Simple repository footprint
- Keep all assignment code under `05_src/assignment_chat`.
- Use only course-provided dependencies.

4. Observability and testing approach
- Validate each service independently before wiring into routing logic.
- Use short manual test scripts/prompts during development and document
  assumptions in this file.

## How To Build Service 2 Index
Run once after data updates:

```bash
python 05_src/assignment_chat/scripts/build_chroma.py
```

This builds/refreshes the persistent Chroma collection using:
- collection name: `mep_issue_kb`
- embedding model: `all-MiniLM-L6-v2`
- persistence path: `05_src/assignment_chat/chroma_db`

After building locally, keep the generated `chroma_db/` files in your branch so reviewers can run the app without regenerating embeddings.

## Current Status
- `app.py` includes:
  - Chat interface
  - Personality baseline
  - Guardrail checks
  - Routing to Service 2 (RAG) when issue-related queries are detected
- Service 2 implementation completed at MVP level.
- Services 1 and 3 are next.
