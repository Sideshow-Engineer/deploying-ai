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
    prefix = f"{field_name}:"
    for line in document.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return ""


def _build_answer(documents: list[str], ids: list[str], titles: list[str]) -> str:
    if not documents:
        return (
            "I could not find a confident match in the local MEP issue knowledge base. "
            "Try adding system details (AHU/VAV/CHW), stage, symptoms, and location."
        )

    best_doc = documents[0]
    best_id = ids[0]
    best_title = titles[0]
    best_root_causes = _extract_field(best_doc, "Root causes")
    best_field_checks = _extract_field(best_doc, "Field checks")
    best_actions = _extract_field(best_doc, "Recommended actions")

    related = ", ".join(
        f"{issue_id} ({title})" for issue_id, title in list(zip(ids, titles))[1:4]
    )
    related_text = f"\nRelated cases: {related}." if related else ""

    return (
        f"Best match: {best_id} - {best_title}.\n"
        f"Likely root causes: {best_root_causes}\n"
        f"Field checks: {best_field_checks}\n"
        f"Recommended actions: {best_actions}{related_text}"
    )


def query_issue_kb(query: str, top_k: int = DEFAULT_TOP_K) -> RagResult:
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

    answer = _build_answer(documents=documents, ids=ids, titles=titles)
    citations = [f"{issue_id}: {title}" for issue_id, title in zip(ids, titles)]

    return RagResult(answer=answer, citations=citations)

