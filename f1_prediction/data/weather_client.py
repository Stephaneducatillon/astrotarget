"""Client Open-Meteo — météo historique + prévisions, sans clé API."""

import logging
from datetime import datetime, timedelta
from functools import lru_cache

import requests
import requests_cache
from retry_requests import retry

from config import OPEN_METEO_FORECAST_URL, OPEN_METEO_HISTORY_URL, API_TIMEOUT

logger = logging.getLogger(__name__)

# Cache HTTP pour éviter les appels répétés
_cache_session = requests_cache.CachedSession("/tmp/open_meteo_cache", expire_after=3600)
_retry_session = retry(_cache_session, retries=3, backoff_factor=0.5)


def _parse_hourly(hourly: dict, target_hour: int = 14) -> dict:
    """Extrait les données à une heure donnée (défaut 14h = heure typique de session)."""
    times = hourly.get("time", [])
    if not times:
        return {}
    # Trouver l'index le plus proche de target_hour
    idx = 0
    for i, t in enumerate(times):
        try:
            h = int(t.split("T")[1].split(":")[0]) if "T" in t else 0
            if h >= target_hour:
                idx = i
                break
        except (ValueError, IndexError):
            pass
    result = {}
    for key, values in hourly.items():
        if key == "time":
            continue
        if values and idx < len(values):
            result[key] = values[idx]
        else:
            result[key] = None
    return result


def get_weather_forecast(lat: float, lon: float, date: str,
                         session_hour: int = 14) -> dict:
    """
    Prévision météo pour une date/lieu donné.
    date: format 'YYYY-MM-DD'
    Retourne: temperature_2m, precipitation_probability, wind_speed_10m, etc.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join([
            "temperature_2m",
            "precipitation_probability",
            "precipitation",
            "wind_speed_10m",
            "cloud_cover",
            "relative_humidity_2m",
        ]),
        "start_date": date,
        "end_date": date,
        "timezone": "auto",
        "wind_speed_unit": "kmh",
    }
    try:
        r = _retry_session.get(OPEN_METEO_FORECAST_URL, params=params, timeout=API_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        hourly = data.get("hourly", {})
        extracted = _parse_hourly(hourly, session_hour)
        return {
            "temperature": extracted.get("temperature_2m"),
            "rain_probability": extracted.get("precipitation_probability"),
            "rain_mm": extracted.get("precipitation"),
            "wind_speed": extracted.get("wind_speed_10m"),
            "cloud_cover": extracted.get("cloud_cover"),
            "humidity": extracted.get("relative_humidity_2m"),
            "is_wet": (extracted.get("precipitation", 0) or 0) > 0.5,
            "source": "forecast",
        }
    except Exception as e:
        logger.warning(f"Open-Meteo forecast error: {e}")
        return _default_weather()


def get_weather_historical(lat: float, lon: float, date: str,
                           session_hour: int = 14) -> dict:
    """
    Données météo historiques pour entraînement du modèle.
    date: format 'YYYY-MM-DD'
    """
    # L'API archive n'accepte pas les dates futures
    today = datetime.now().strftime("%Y-%m-%d")
    if date > today:
        return get_weather_forecast(lat, lon, date, session_hour)

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join([
            "temperature_2m",
            "precipitation",
            "wind_speed_10m",
            "cloud_cover",
            "relative_humidity_2m",
        ]),
        "start_date": date,
        "end_date": date,
        "timezone": "auto",
    }
    try:
        r = _retry_session.get(OPEN_METEO_HISTORY_URL, params=params, timeout=API_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        hourly = data.get("hourly", {})
        extracted = _parse_hourly(hourly, session_hour)
        return {
            "temperature": extracted.get("temperature_2m"),
            "rain_probability": None,  # Non disponible en historique
            "rain_mm": extracted.get("precipitation"),
            "wind_speed": extracted.get("wind_speed_10m"),
            "cloud_cover": extracted.get("cloud_cover"),
            "humidity": extracted.get("relative_humidity_2m"),
            "is_wet": (extracted.get("precipitation", 0) or 0) > 0.5,
            "source": "historical",
        }
    except Exception as e:
        logger.warning(f"Open-Meteo historical error: {e}")
        return _default_weather()


def get_weather_for_race(lat: float, lon: float, race_date: str) -> dict:
    """
    Météo complète pour un GP : qualifs (samedi) + course (dimanche).
    race_date: date de la course (dimanche), format 'YYYY-MM-DD'
    """
    try:
        race_dt = datetime.strptime(race_date, "%Y-%m-%d")
        quali_date = (race_dt - timedelta(days=1)).strftime("%Y-%m-%d")
    except ValueError:
        quali_date = race_date

    today = datetime.now().strftime("%Y-%m-%d")
    weather_fn = get_weather_forecast if race_date >= today else get_weather_historical

    return {
        "qualifying": weather_fn(lat, lon, quali_date, session_hour=14),
        "race": weather_fn(lat, lon, race_date, session_hour=15),
    }


def _default_weather() -> dict:
    """Météo par défaut si API indisponible."""
    return {
        "temperature": 25.0,
        "rain_probability": 10.0,
        "rain_mm": 0.0,
        "wind_speed": 15.0,
        "cloud_cover": 30.0,
        "humidity": 50.0,
        "is_wet": False,
        "source": "default",
    }


def weather_summary(weather: dict) -> str:
    """Description textuelle des conditions météo."""
    if not weather:
        return "Données météo indisponibles"
    temp = weather.get("temperature")
    rain = weather.get("rain_probability")
    wind = weather.get("wind_speed")
    is_wet = weather.get("is_wet", False)

    parts = []
    if temp is not None:
        parts.append(f"{temp:.0f}°C")
    if is_wet:
        parts.append("Pluie ☔")
    elif rain is not None:
        parts.append(f"Pluie {rain:.0f}%")
    if wind is not None:
        parts.append(f"Vent {wind:.0f} km/h")
    return " | ".join(parts) if parts else "Données météo partielles"
