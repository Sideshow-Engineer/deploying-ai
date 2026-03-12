from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import chromadb
from chromadb.utils import embedding_functions

from config import CHROMA_DIR, COLLECTION_NAME, DEFAULT_TOP_K, EMBEDDING_MODEL_NAME


@dataclass
class RagResult:
    answer: str
    citations: list[str]


@dataclass
class RagMatch:
    id: str
    title: str
    system: str
    discipline: str
    stage: str
    location_scope: str
    root_causes: str
    field_checks: str
    recommended_actions: str
    rfi_template_question: str
    risk_tags: str


_SYSTEM_MAP = {
    "ahu": "AHU",
    "vav": "VAV",
    "chw": "CHW",
    "controls": "Controls",
    "tab": "TAB",
    "cleanroom": "Cleanroom Pressurization",
}

_STAGE_MAP = {
    "startup": "Startup",
    "commissioning": "Commissioning",
    "tab": "TAB",
}


def _collection() -> chromadb.api.models.Collection.Collection:
    if not CHROMA_DIR.exists():
        raise RuntimeError(
            f"Chroma DB not found at {CHROMA_DIR}. "
            "Run scripts/build_chroma.py first."
        )

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )
    return client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )


def _first_match_map(query: str, mapping: dict[str, str]) -> str | None:
    lowered = query.lower()
    for key, value in mapping.items():
        if re.search(rf"\b{re.escape(key)}\b", lowered):
            return value
    return None


def _where_filter(query: str) -> dict[str, Any] | None:
    system_value = _first_match_map(query, _SYSTEM_MAP)
    stage_value = _first_match_map(query, _STAGE_MAP)

    clauses: list[dict[str, str]] = []
    if system_value:
        clauses.append({"system": system_value})
    if stage_value:
        clauses.append({"stage": stage_value})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _extract_field(document: str, field_name: str) -> str:
    if not document:
        return ""
    prefix = f"{field_name}:"
    for line in document.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return ""


def _to_match(
    metadata: dict[str, Any] | None,
    fallback_id: str = "",
    fallback_title: str = "",
    document: str = "",
) -> RagMatch:
    meta = metadata or {}
    return RagMatch(
        id=str(meta.get("id", fallback_id)),
        title=str(meta.get("title", fallback_title)),
        system=str(meta.get("system", "")),
        discipline=str(meta.get("discipline", "")),
        stage=str(meta.get("stage", "")),
        location_scope=str(meta.get("location_scope", "")),
        root_causes=str(meta.get("root_causes", _extract_field(document, "Root causes"))),
        field_checks=str(meta.get("field_checks", _extract_field(document, "Field checks"))),
        recommended_actions=str(
            meta.get("recommended_actions", _extract_field(document, "Recommended actions"))
        ),
        rfi_template_question=str(meta.get("rfi_template_question", _extract_field(document, "RFI prompt"))),
        risk_tags=str(meta.get("risk_tags", "")),
    )


def _build_answer(matches: list[RagMatch]) -> str:
    if not matches:
        return (
            "I could not find a confident match in the local MEP issue knowledge base. "
            "Try adding system details (AHU/VAV/CHW), stage, symptoms, and location."
        )

    best = matches[0]

    related = ", ".join(
        f"{m.id} ({m.title})" for m in matches[1:4]
    )
    related_text = f"\nRelated cases: {related}." if related else ""

    return (
        f"Best match: {best.id} - {best.title}.\n"
        f"Likely root causes: {best.root_causes}\n"
        f"Field checks: {best.field_checks}\n"
        f"Recommended actions: {best.recommended_actions}{related_text}"
    )


def _query_matches(query: str, top_k: int = DEFAULT_TOP_K) -> list[RagMatch]:
    collection = _collection()
    where = _where_filter(query)
    response = collection.query(
        query_texts=[query],
        n_results=top_k,
        where=where,
    )

    ids = response.get("ids", [[]])[0]
    documents = response.get("documents", [[]])[0]
    metadatas = response.get("metadatas", [[]])[0]
    titles = [metadata.get("title", "") if metadata else "" for metadata in metadatas]
    return [
        _to_match(
            metadata=md,
            fallback_id=issue_id,
            fallback_title=title,
            document=document,
        )
        for issue_id, title, md, document in zip(ids, titles, metadatas, documents)
    ]


def query_issue_kb(query: str, top_k: int = DEFAULT_TOP_K) -> RagResult:
    matches = _query_matches(query=query, top_k=top_k)
    answer = _build_answer(matches=matches)
    citations = [f"{m.id}: {m.title}" for m in matches]
    return RagResult(answer=answer, citations=citations)


def lookup_issue_kb(query_text: str, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
    matches = _query_matches(query=query_text, top_k=top_k)
    if not matches:
        return {
            "query": query_text,
            "best_match": None,
            "related_matches": [],
            "citation": "",
            "note": "No confident KB match found.",
        }

    best = matches[0]
    related = matches[1:4]
    return {
        "query": query_text,
        "best_match": {
            "id": best.id,
            "title": best.title,
            "system": best.system,
            "discipline": best.discipline,
            "stage": best.stage,
            "location_scope": best.location_scope,
            "root_causes": best.root_causes,
            "field_checks": best.field_checks,
            "recommended_actions": best.recommended_actions,
            "rfi_template_question": best.rfi_template_question,
            "risk_tags": best.risk_tags,
        },
        "related_matches": [
            {"id": item.id, "title": item.title, "system": item.system}
            for item in related
        ],
        "citation": f"{best.id}: {best.title}",
    }
