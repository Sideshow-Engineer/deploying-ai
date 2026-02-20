from __future__ import annotations

import argparse
from pathlib import Path
import sys

import chromadb
from chromadb.utils import embedding_functions
import pandas as pd

# Allow "python scripts/build_chroma.py" to import config.py from parent folder.
PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from config import (  # noqa: E402
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    KB_CSV_PATH,
)


REQUIRED_COLUMNS = [
    "id",
    "title",
    "system",
    "discipline",
    "stage",
    "location_scope",
    "symptoms",
    "likely_root_causes",
    "field_checks",
    "recommended_actions",
    "rfi_template_question",
    "risk_tags",
]


def _normalize_dataframe(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8")
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    return df.fillna("")


def _row_to_document(row: pd.Series) -> str:
    return "\n".join(
        [
            f"ID: {row['id']}",
            f"Title: {row['title']}",
            f"System: {row['system']}",
            f"Discipline: {row['discipline']}",
            f"Stage: {row['stage']}",
            f"Location: {row['location_scope']}",
            f"Symptoms: {row['symptoms']}",
            f"Root causes: {row['likely_root_causes']}",
            f"Field checks: {row['field_checks']}",
            f"Recommended actions: {row['recommended_actions']}",
            f"RFI prompt: {row['rfi_template_question']}",
            f"Risk tags: {row['risk_tags']}",
        ]
    )


def _row_to_metadata(row: pd.Series) -> dict[str, str]:
    return {
        "id": str(row["id"]),
        "title": str(row["title"]),
        "system": str(row["system"]),
        "discipline": str(row["discipline"]),
        "stage": str(row["stage"]),
        "location_scope": str(row["location_scope"]),
        "root_causes": str(row["likely_root_causes"]),
        "field_checks": str(row["field_checks"]),
        "recommended_actions": str(row["recommended_actions"]),
        "rfi_template_question": str(row["rfi_template_question"]),
        "risk_tags": str(row["risk_tags"]),
    }


def _build_collection(
    csv_path: Path,
    chroma_dir: Path,
    collection_name: str,
    embedding_model: str,
    reset: bool,
) -> None:
    df = _normalize_dataframe(csv_path)
    ids = df["id"].astype(str).tolist()
    documents = [_row_to_document(row) for _, row in df.iterrows()]
    metadatas = [_row_to_metadata(row) for _, row in df.iterrows()]

    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=embedding_model
    )

    if reset:
        try:
            client.delete_collection(name=collection_name)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    total = collection.count()
    print(f"Indexed records: {len(ids)}")
    print(f"Collection count: {total}")
    print(f"Persisted directory: {chroma_dir}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a persistent Chroma index from mep_issue_kb.csv."
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=KB_CSV_PATH,
        help=f"Path to source CSV (default: {KB_CSV_PATH})",
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=CHROMA_DIR,
        help=f"Path to persistent Chroma directory (default: {CHROMA_DIR})",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=COLLECTION_NAME,
        help=f"Collection name (default: {COLLECTION_NAME})",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=EMBEDDING_MODEL_NAME,
        help=f"Sentence-Transformers model name (default: {EMBEDDING_MODEL_NAME})",
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Do not delete existing collection before indexing.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    _build_collection(
        csv_path=args.csv_path,
        chroma_dir=args.persist_dir,
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        reset=not args.no_reset,
    )


if __name__ == "__main__":
    main()
