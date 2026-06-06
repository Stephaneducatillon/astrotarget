"""Client Jolpica/Ergast API — données historiques F1 depuis 1950."""

import time
import logging
from functools import lru_cache
from typing import Optional

import requests

from config import JOLPICA_BASE_URL, API_TIMEOUT

logger = logging.getLogger(__name__)


def _get(url: str, params: dict = None, retries: int = 3) -> dict:
    """GET avec retry exponentiel."""
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=API_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if attempt == retries - 1:
                logger.error(f"Jolpica API error {url}: {e}")
                return {}
            time.sleep(2 ** attempt)
    return {}


# ─── Calendrier saison ────────────────────────────────────────────────────────

@lru_cache(maxsize=5)
def get_season_schedule(year: int) -> list[dict]:
    """Retourne le calendrier F1 d'une saison."""
    data = _get(f"{JOLPICA_BASE_URL}/{year}.json")
    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    result = []
    for race in races:
        result.append({
            "round": int(race["round"]),
            "name": race["raceName"],
            "circuit_id": race["Circuit"]["circuitId"],
            "circuit_name": race["Circuit"]["circuitName"],
            "country": race["Circuit"]["Location"]["country"],
            "lat": float(race["Circuit"]["Location"]["lat"]),
            "lon": float(race["Circuit"]["Location"]["long"]),
            "date": race["date"],
            "time": race.get("time", ""),
        })
    return result


@lru_cache(maxsize=5)
def get_current_season() -> list[dict]:
    """Calendrier de la saison en cours."""
    from config import CURRENT_YEAR
    return get_season_schedule(CURRENT_YEAR)


# ─── Résultats qualifications ─────────────────────────────────────────────────

@lru_cache(maxsize=50)
def get_qualifying_results(year: int, round_num: int) -> list[dict]:
    """Résultats qualifications pour un GP."""
    data = _get(f"{JOLPICA_BASE_URL}/{year}/{round_num}/qualifying.json")
    quali = (data.get("MRData", {})
             .get("RaceTable", {})
             .get("Races", [{}])[0]
             .get("QualifyingResults", []))
    results = []
    for q in quali:
        driver = q.get("Driver", {})
        constructor = q.get("Constructor", {})
        results.append({
            "position": int(q["position"]),
            "driver_id": driver.get("driverId", ""),
            "driver_code": driver.get("code", ""),
            "driver_name": f"{driver.get('givenName','')} {driver.get('familyName','')}".strip(),
            "team": constructor.get("name", ""),
            "team_id": constructor.get("constructorId", ""),
            "q1": q.get("Q1", ""),
            "q2": q.get("Q2", ""),
            "q3": q.get("Q3", ""),
        })
    return results


# ─── Résultats course ─────────────────────────────────────────────────────────

@lru_cache(maxsize=50)
def get_race_results(year: int, round_num: int) -> list[dict]:
    """Résultats course pour un GP."""
    data = _get(f"{JOLPICA_BASE_URL}/{year}/{round_num}/results.json")
    race_data = (data.get("MRData", {})
                 .get("RaceTable", {})
                 .get("Races", [{}]))
    if not race_data:
        return []
    race = race_data[0]
    results = []
    for r in race.get("Results", []):
        driver = r.get("Driver", {})
        constructor = r.get("Constructor", {})
        results.append({
            "position": int(r["position"]) if r["position"].isdigit() else 99,
            "grid": int(r.get("grid", 0)),
            "driver_id": driver.get("driverId", ""),
            "driver_code": driver.get("code", ""),
            "driver_name": f"{driver.get('givenName','')} {driver.get('familyName','')}".strip(),
            "team": constructor.get("name", ""),
            "team_id": constructor.get("constructorId", ""),
            "points": float(r.get("points", 0)),
            "status": r.get("status", ""),
            "laps": int(r.get("laps", 0)),
            "time": r.get("Time", {}).get("time", ""),
            "fastest_lap_rank": int(r.get("FastestLap", {}).get("rank", 99)),
        })
    return results


# ─── Classements ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=10)
def get_driver_standings(year: int, after_round: Optional[int] = None) -> list[dict]:
    """Classement pilotes après un round donné (ou fin de saison)."""
    if after_round:
        url = f"{JOLPICA_BASE_URL}/{year}/{after_round}/driverStandings.json"
    else:
        url = f"{JOLPICA_BASE_URL}/{year}/driverStandings.json"
    data = _get(url)
    standings_list = (data.get("MRData", {})
                      .get("StandingsTable", {})
                      .get("StandingsLists", [{}]))
    if not standings_list:
        return []
    result = []
    for s in standings_list[0].get("DriverStandings", []):
        driver = s.get("Driver", {})
        constructors = s.get("Constructors", [{}])
        result.append({
            "position": int(s["position"]),
            "driver_id": driver.get("driverId", ""),
            "driver_code": driver.get("code", ""),
            "driver_name": f"{driver.get('givenName','')} {driver.get('familyName','')}".strip(),
            "team": constructors[0].get("name", "") if constructors else "",
            "team_id": constructors[0].get("constructorId", "") if constructors else "",
            "points": float(s.get("points", 0)),
            "wins": int(s.get("wins", 0)),
        })
    return result


@lru_cache(maxsize=10)
def get_constructor_standings(year: int, after_round: Optional[int] = None) -> list[dict]:
    """Classement constructeurs."""
    if after_round:
        url = f"{JOLPICA_BASE_URL}/{year}/{after_round}/constructorStandings.json"
    else:
        url = f"{JOLPICA_BASE_URL}/{year}/constructorStandings.json"
    data = _get(url)
    standings_list = (data.get("MRData", {})
                      .get("StandingsTable", {})
                      .get("StandingsLists", [{}]))
    if not standings_list:
        return []
    result = []
    for s in standings_list[0].get("ConstructorStandings", []):
        team = s.get("Constructor", {})
        result.append({
            "position": int(s["position"]),
            "team_id": team.get("constructorId", ""),
            "team": team.get("name", ""),
            "points": float(s.get("points", 0)),
            "wins": int(s.get("wins", 0)),
        })
    return result


# ─── Historique circuit ────────────────────────────────────────────────────────

def get_circuit_history(circuit_id: str, years: int = 7) -> list[dict]:
    """
    Résultats historiques sur un circuit donné.
    Retourne une liste de dicts avec année, winner, pole, etc.
    """
    from config import CURRENT_YEAR
    history = []
    for year in range(CURRENT_YEAR - years, CURRENT_YEAR):
        schedule = get_season_schedule(year)
        for race in schedule:
            if race["circuit_id"] == circuit_id:
                results = get_race_results(year, race["round"])
                quali = get_qualifying_results(year, race["round"])
                if results:
                    winner = next((r for r in results if r["position"] == 1), None)
                    pole = quali[0] if quali else None
                    history.append({
                        "year": year,
                        "round": race["round"],
                        "winner_id": winner["driver_id"] if winner else "",
                        "winner_team": winner["team_id"] if winner else "",
                        "pole_id": pole["driver_id"] if pole else "",
                        "results": results,
                        "quali": quali,
                    })
                break
    return history


def get_driver_circuit_stats(driver_id: str, circuit_id: str, years: int = 7) -> dict:
    """Statistiques d'un pilote sur un circuit."""
    history = get_circuit_history(circuit_id, years)
    positions = []
    quali_positions = []
    poles = 0
    wins = 0
    podiums = 0
    dnfs = 0
    for h in history:
        for r in h.get("results", []):
            if r["driver_id"] == driver_id:
                pos = r["position"]
                positions.append(pos)
                if pos == 1:
                    wins += 1
                if pos <= 3:
                    podiums += 1
                if "Ret" in r.get("status", "") or pos == 99:
                    dnfs += 1
        for q in h.get("quali", []):
            if q["driver_id"] == driver_id:
                quali_positions.append(q["position"])
                if q["position"] == 1:
                    poles += 1
    n = len(positions)
    return {
        "driver_id": driver_id,
        "circuit_id": circuit_id,
        "races": n,
        "avg_finish": sum(positions) / n if n else 10.0,
        "best_finish": min(positions) if positions else 20,
        "wins": wins,
        "podiums": podiums,
        "dnfs": dnfs,
        "dnf_rate": dnfs / n if n else 0.0,
        "poles": poles,
        "avg_quali": sum(quali_positions) / len(quali_positions) if quali_positions else 10.0,
        "best_quali": min(quali_positions) if quali_positions else 20,
    }


def get_team_circuit_stats(team_id: str, circuit_id: str, years: int = 7) -> dict:
    """Statistiques d'une équipe sur un circuit."""
    history = get_circuit_history(circuit_id, years)
    positions = []
    wins = 0
    podiums = 0
    for h in history:
        for r in h.get("results", []):
            if r["team_id"] == team_id:
                pos = r["position"]
                positions.append(pos)
                if pos == 1:
                    wins += 1
                if pos <= 3:
                    podiums += 1
    n = len(positions)
    return {
        "team_id": team_id,
        "circuit_id": circuit_id,
        "avg_finish": sum(positions) / n if n else 10.0,
        "best_finish": min(positions) if positions else 20,
        "wins": wins,
        "podiums": podiums,
    }


def get_recent_form(driver_id: str, year: int, before_round: int, n_races: int = 4) -> dict:
    """Forme récente d'un pilote (n dernières courses)."""
    results = []
    schedule = get_season_schedule(year)
    for race in sorted(schedule, key=lambda x: x["round"], reverse=True):
        if race["round"] >= before_round:
            continue
        race_results = get_race_results(year, race["round"])
        for r in race_results:
            if r["driver_id"] == driver_id:
                results.append(r["position"])
                break
        if len(results) >= n_races:
            break
    n = len(results)
    return {
        "recent_avg": sum(results) / n if n else 10.0,
        "recent_best": min(results) if results else 20,
        "recent_races": n,
    }
