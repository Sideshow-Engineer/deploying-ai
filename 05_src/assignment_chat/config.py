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

# LLM gateway settings (used by Service 3 function-calling orchestration)
OPENAI_GATEWAY_BASE_URL = (
    "https://k7uffyg03f.execute-api.us-east-1.amazonaws.com/prod/openai/v1"
)
FUNCTION_CALLING_MODEL = "gpt-4o-mini"

# Router settings (hybrid LLM intent router)
USE_LLM_ROUTER = True
ROUTER_MODEL = "gpt-4o-mini"
ROUTER_MAX_ACTIONS = 2
ROUTER_MIN_CONFIDENCE = 0.55

# Weather service settings (Open-Meteo)
WEATHER_DEFAULT_LOCATION = "Toronto"
OPEN_METEO_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Risk thresholds (construction/commissioning-oriented heuristics)
WIND_GUST_RISK_KMH = 45.0
WIND_SPEED_RISK_KMH = 30.0
HEAVY_PRECIP_RISK_MM = 10.0
PRECIP_HOURS_RISK = 6.0
FREEZE_RISK_C = 1.0
HEAT_RISK_C = 30.0
SNOW_RISK_CM = 1.0
