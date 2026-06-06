"""Configuration centrale de l'application F1 Prévision v0.5."""

import os

# ─── API Keys ────────────────────────────────────────────────────────────────
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")

# ─── API Endpoints ───────────────────────────────────────────────────────────
JOLPICA_BASE_URL = "https://api.jolpi.ca/ergast/f1"
OPENF1_BASE_URL = "https://api.openf1.org/v1"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_HISTORY_URL = "https://archive-api.open-meteo.com/v1/archive"

# ─── FastF1 Cache ─────────────────────────────────────────────────────────────
FASTF1_CACHE_DIR = "/tmp/fastf1_cache"

# ─── Model Paths ─────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "artifacts")
QUALI_MODEL_PATH = os.path.join(MODEL_DIR, "quali_model.pkl")
RACE_MODEL_PATH = os.path.join(MODEL_DIR, "race_model.pkl")
ENCODERS_PATH = os.path.join(MODEL_DIR, "encoders.pkl")

# ─── ML Parameters ───────────────────────────────────────────────────────────
XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "random_state": 42,
    "n_jobs": -1,
}

TRAINING_YEARS = list(range(2018, 2026))  # Données d'entraînement
CURRENT_YEAR = 2026

# ─── Calendrier F1 2026 — Coordonnées GPS pour météo ─────────────────────────
CIRCUIT_COORDS = {
    "bahrain": {"lat": 26.0325, "lon": 50.5106, "timezone": "Asia/Bahrain"},
    "jeddah": {"lat": 21.6319, "lon": 39.1044, "timezone": "Asia/Riyadh"},
    "albert_park": {"lat": -37.8497, "lon": 144.9680, "timezone": "Australia/Melbourne"},
    "suzuka": {"lat": 34.8431, "lon": 136.5406, "timezone": "Asia/Tokyo"},
    "shanghai": {"lat": 31.3389, "lon": 121.2198, "timezone": "Asia/Shanghai"},
    "miami": {"lat": 25.9581, "lon": -80.2389, "timezone": "America/New_York"},
    "imola": {"lat": 44.3439, "lon": 11.7167, "timezone": "Europe/Rome"},
    "monaco": {"lat": 43.7347, "lon": 7.4206, "timezone": "Europe/Monaco"},
    "villeneuve": {"lat": 45.5000, "lon": -73.5228, "timezone": "America/Montreal"},
    "catalunya": {"lat": 41.5700, "lon": 2.2611, "timezone": "Europe/Madrid"},
    "red_bull_ring": {"lat": 47.2197, "lon": 14.7647, "timezone": "Europe/Vienna"},
    "silverstone": {"lat": 52.0786, "lon": -1.0169, "timezone": "Europe/London"},
    "hungaroring": {"lat": 47.5789, "lon": 19.2486, "timezone": "Europe/Budapest"},
    "spa": {"lat": 50.4372, "lon": 5.9714, "timezone": "Europe/Brussels"},
    "zandvoort": {"lat": 52.3888, "lon": 4.5409, "timezone": "Europe/Amsterdam"},
    "monza": {"lat": 45.6156, "lon": 9.2811, "timezone": "Europe/Rome"},
    "baku": {"lat": 40.3725, "lon": 49.8533, "timezone": "Asia/Baku"},
    "marina_bay": {"lat": 1.2914, "lon": 103.8640, "timezone": "Asia/Singapore"},
    "americas": {"lat": 30.1328, "lon": -97.6411, "timezone": "America/Chicago"},
    "rodriguez": {"lat": 19.4042, "lon": -99.0907, "timezone": "America/Mexico_City"},
    "interlagos": {"lat": -23.7036, "lon": -46.6997, "timezone": "America/Sao_Paulo"},
    "vegas": {"lat": 36.1147, "lon": -115.1728, "timezone": "America/Los_Angeles"},
    "losail": {"lat": 25.4900, "lon": 51.4536, "timezone": "Asia/Qatar"},
    "yas_marina": {"lat": 24.4672, "lon": 54.6031, "timezone": "Asia/Dubai"},
}

# Circuit type (impact sur le modèle)
CIRCUIT_TYPES = {
    "monaco": "street", "baku": "street", "marina_bay": "street",
    "vegas": "street", "jeddah": "street",
    "monza": "power", "spa": "power", "silverstone": "balanced",
    "suzuka": "balanced", "red_bull_ring": "power",
    "hungaroring": "downforce", "barcelona": "downforce",
    "americas": "balanced", "interlagos": "balanced",
}

# ─── Mistral Models ───────────────────────────────────────────────────────────
MISTRAL_MODELS = {
    "fast": "mistral-small-latest",
    "balanced": "mistral-medium-2505",
    "best": "mistral-large-latest",
}

# ─── Rate Limiting ────────────────────────────────────────────────────────────
MISTRAL_RPM_LIMIT = 2
MISTRAL_RETRY_DELAY = 35  # secondes

# ─── HTTP Timeouts ────────────────────────────────────────────────────────────
API_TIMEOUT = 30
FASTF1_TIMEOUT = 60
