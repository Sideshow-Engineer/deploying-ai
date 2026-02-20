from __future__ import annotations

from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
KB_CSV_PATH = DATA_DIR / "mep_issue_kb.csv"

# Chroma settings
COLLECTION_NAME = "mep_issue_kb"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 5

