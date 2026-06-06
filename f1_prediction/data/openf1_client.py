"""Client OpenF1 API — données live 2023+."""

import logging
import time
from functools import lru_cache

import requests

from config import OPENF1_BASE_URL, API_TIMEOUT

logger = logging.getLogger(__name__)


def _get(endpoint: str, params: dict = None, retries: int = 3) -> list | dict:
    """GET avec retry."""
    url = f"{OPENF1_BASE_URL}/{endpoint}"
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=API_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if attempt == retries - 1:
                logger.error(f"OpenF1 error {url}: {e}")
                return []
            time.sleep(2 ** attempt)
    return []


# ─── Sessions ─────────────────────────────────────────────────────────────────

@lru_cache(maxsize=20)
def get_sessions(year: int) -> list[dict]:
    """Toutes les sessions d'une saison."""
    return _get("sessions", {"year": year})


def get_latest_session() -> dict | None:
    """Session la plus récente (live ou terminée)."""
    sessions = _get("sessions", {"session_key": "latest"})
    if isinstance(sessions, list) and sessions:
        return sessions[0]
    elif isinstance(sessions, dict):
        return sessions
    return None


def get_session_by_gp(year: int, gp_name: str,
                      session_type: str = "Race") -> dict | None:
    """Session d'un GP spécifique."""
    sessions = get_sessions(year)
    session_type_map = {
        "Race": "Race", "R": "Race",
        "Qualifying": "Qualifying", "Q": "Qualifying",
        "Practice 1": "Practice 1", "FP1": "Practice 1",
        "Practice 2": "Practice 2", "FP2": "Practice 2",
        "Practice 3": "Practice 3", "FP3": "Practice 3",
    }
    target = session_type_map.get(session_type, session_type)
    for s in sessions:
        if (gp_name.lower() in s.get("location", "").lower() or
                gp_name.lower() in s.get("country_name", "").lower()):
            if s.get("session_name", "") == target:
                return s
    return None


# ─── Pilotes ──────────────────────────────────────────────────────────────────

@lru_cache(maxsize=10)
def get_drivers_in_session(session_key: int) -> list[dict]:
    """Pilotes d'une session."""
    return _get("drivers", {"session_key": session_key})


def get_current_drivers() -> list[dict]:
    """Pilotes de la session la plus récente."""
    session = get_latest_session()
    if not session:
        return []
    return get_drivers_in_session(session.get("session_key", 0))


# ─── Timing live ──────────────────────────────────────────────────────────────

def get_live_position() -> list[dict]:
    """Position en temps réel de tous les pilotes."""
    return _get("position", {"session_key": "latest"})


def get_live_intervals() -> list[dict]:
    """Intervalles entre pilotes en temps réel."""
    return _get("intervals", {"session_key": "latest"})


def get_driver_laps(session_key: int, driver_number: int) -> list[dict]:
    """Tous les tours d'un pilote dans une session."""
    return _get("laps", {"session_key": session_key, "driver_number": driver_number})


def get_stints(session_key: int) -> list[dict]:
    """Stints pneus de tous les pilotes dans une session."""
    return _get("stints", {"session_key": session_key})


def get_pit_stops(session_key: int) -> list[dict]:
    """Arrêts aux stands d'une session."""
    return _get("pit", {"session_key": session_key})


# ─── Weather live ─────────────────────────────────────────────────────────────

def get_live_weather() -> list[dict]:
    """Données météo live de la session en cours."""
    data = _get("weather", {"session_key": "latest"})
    if data and isinstance(data, list):
        # Retourner la plus récente
        return sorted(data, key=lambda x: x.get("date", ""), reverse=True)
    return data


def get_session_weather(session_key: int) -> list[dict]:
    """Données météo d'une session."""
    return _get("weather", {"session_key": session_key})


# ─── Race control ─────────────────────────────────────────────────────────────

def get_race_control_messages(session_key: int) -> list[dict]:
    """Messages race control (safety car, flags, incidents)."""
    return _get("race_control", {"session_key": session_key})


def get_safety_car_count(session_key: int) -> dict:
    """Compte safety car / VSC depuis race control."""
    messages = get_race_control_messages(session_key)
    sc_count = 0
    vsc_count = 0
    for msg in messages:
        cat = msg.get("category", "")
        flag = msg.get("flag", "")
        if cat == "SafetyCar" and flag == "SAFETY CAR":
            sc_count += 1
        elif cat == "SafetyCar" and flag == "VIRTUAL SAFETY CAR":
            vsc_count += 1
    return {"sc_deployments": sc_count, "vsc_deployments": vsc_count}


# ─── Utils ────────────────────────────────────────────────────────────────────

def format_live_standings(positions: list[dict], drivers: list[dict]) -> list[dict]:
    """Combine positions + info pilotes pour affichage."""
    driver_map = {d["driver_number"]: d for d in drivers}
    # Grouper par pilote, garder la dernière position
    latest = {}
    for p in positions:
        num = p.get("driver_number")
        if num not in latest or p.get("date", "") > latest[num].get("date", ""):
            latest[num] = p
    result = []
    for num, pos in sorted(latest.items(), key=lambda x: x[1].get("position", 99)):
        d = driver_map.get(num, {})
        result.append({
            "position": pos.get("position"),
            "driver_number": num,
            "driver_code": d.get("name_acronym", ""),
            "driver_name": f"{d.get('first_name','')} {d.get('last_name','')}".strip(),
            "team": d.get("team_name", ""),
        })
    return result
