"""Feature engineering pour les modèles XGBoost F1."""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─── Features qualifications ──────────────────────────────────────────────────

def build_qualifying_features(
    year: int,
    round_num: int,
    circuit_id: str,
    circuit_lat: float,
    circuit_lon: float,
    race_date: str,
    drivers: list[dict],
    driver_standings: list[dict],
    team_standings: list[dict],
    circuit_history: list[dict],
    weather: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Construit la matrice de features pour la prédiction qualifications.

    Retourne un DataFrame avec une ligne par pilote, colonnes = features.
    """
    from data.ergast_client import (
        get_driver_circuit_stats, get_team_circuit_stats, get_recent_form
    )

    # Index standings par driver_id et team_id
    driver_pts_map = {d["driver_id"]: d for d in driver_standings}
    team_pts_map = {t["team_id"]: t for t in team_standings}

    rows = []
    for driver in drivers:
        driver_id = driver["driver_id"]
        team_id = driver["team_id"]

        # 1. Forme championnat (points normalisés)
        drv_standing = driver_pts_map.get(driver_id, {})
        champ_points = drv_standing.get("points", 0)
        champ_position = drv_standing.get("position", 20)
        champ_wins = drv_standing.get("wins", 0)

        # 2. Standings équipe
        team_standing = team_pts_map.get(team_id, {})
        team_points = team_standing.get("points", 0)
        team_position = team_standing.get("position", 10)

        # 3. Historique circuit (pilote)
        circ_stats = get_driver_circuit_stats(driver_id, circuit_id, years=6)
        avg_finish = circ_stats.get("avg_finish", 10.0)
        best_finish = circ_stats.get("best_finish", 20)
        avg_quali = circ_stats.get("avg_quali", 10.0)
        best_quali = circ_stats.get("best_quali", 20)
        pole_count = circ_stats.get("poles", 0)
        circuit_races = circ_stats.get("races", 0)

        # 4. Historique circuit (équipe)
        team_circ = get_team_circuit_stats(team_id, circuit_id, years=6)
        team_avg_finish = team_circ.get("avg_finish", 10.0)
        team_best_finish = team_circ.get("best_finish", 20)

        # 5. Forme récente (3 dernières courses)
        recent = get_recent_form(driver_id, year, before_round=round_num, n_races=3)
        recent_avg = recent.get("recent_avg", 10.0)
        recent_best = recent.get("recent_best", 20)

        # 6. Météo
        temp = 25.0
        is_wet = False
        rain_prob = 10.0
        wind = 15.0
        if weather:
            temp = weather.get("temperature") or 25.0
            is_wet = weather.get("is_wet", False)
            rain_prob = weather.get("rain_probability") or 10.0
            wind = weather.get("wind_speed") or 15.0

        # 7. DNF rate (fiabilité)
        dnf_rate = circ_stats.get("dnf_rate", 0.1)

        row = {
            "driver_id": driver_id,
            "team_id": team_id,
            # Championnat
            "champ_points": champ_points,
            "champ_position": champ_position,
            "champ_wins": champ_wins,
            # Équipe
            "team_points": team_points,
            "team_position": team_position,
            # Circuit pilote
            "circuit_avg_finish": avg_finish,
            "circuit_best_finish": best_finish,
            "circuit_avg_quali": avg_quali,
            "circuit_best_quali": best_quali,
            "circuit_poles": pole_count,
            "circuit_races": circuit_races,
            # Circuit équipe
            "team_circuit_avg": team_avg_finish,
            "team_circuit_best": team_best_finish,
            # Forme récente
            "recent_avg_finish": recent_avg,
            "recent_best_finish": recent_best,
            # Météo
            "temperature": temp,
            "is_wet": int(is_wet),
            "rain_probability": rain_prob,
            "wind_speed": wind,
            # Fiabilité
            "dnf_rate": dnf_rate,
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = _normalize_features(df, exclude=["driver_id", "team_id"])
    return df


def build_race_features(
    year: int,
    round_num: int,
    circuit_id: str,
    drivers: list[dict],
    quali_results: list[dict],
    driver_standings: list[dict],
    team_standings: list[dict],
    circuit_history: list[dict],
    pit_stop_data: Optional[pd.DataFrame] = None,
    weather: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Construit la matrice de features pour la prédiction course.
    Inclut les résultats qualifications comme feature clé.
    """
    from data.ergast_client import (
        get_driver_circuit_stats, get_team_circuit_stats, get_recent_form
    )

    driver_pts_map = {d["driver_id"]: d for d in driver_standings}
    team_pts_map = {t["team_id"]: t for t in team_standings}

    # Map quali position par driver_id
    quali_pos_map = {}
    for q in quali_results:
        quali_pos_map[q["driver_id"]] = q["position"]

    # Map pit stop durée par team
    pit_map = {}
    if pit_stop_data is not None and not pit_stop_data.empty:
        pit_map = dict(zip(pit_stop_data["Team"], pit_stop_data["AvgPitStop_s"]))

    # Ratio pole→victoire sur ce circuit (historique)
    pole_win_rate = _compute_pole_win_rate(circuit_history)

    rows = []
    for driver in drivers:
        driver_id = driver["driver_id"]
        team_id = driver["team_id"]

        drv_standing = driver_pts_map.get(driver_id, {})
        team_standing = team_pts_map.get(team_id, {})
        circ_stats = get_driver_circuit_stats(driver_id, circuit_id, years=6)
        team_circ = get_team_circuit_stats(team_id, circuit_id, years=6)
        recent = get_recent_form(driver_id, year, before_round=round_num, n_races=4)

        # Position de départ (critique)
        grid_pos = quali_pos_map.get(driver_id, 15)

        # Pit stop vitesse (équipe)
        avg_pit = pit_map.get(driver["team"], 24.0)  # 24s = durée typique

        temp = 25.0
        is_wet = False
        rain_prob = 10.0
        wind = 15.0
        if weather:
            temp = weather.get("temperature") or 25.0
            is_wet = weather.get("is_wet", False)
            rain_prob = weather.get("rain_probability") or 10.0
            wind = weather.get("wind_speed") or 15.0

        # Conversion pole→victoire sur ce circuit
        win_from_pole = 1.0 if grid_pos == 1 else 0.0

        # Stats sur ce circuit
        dnf_rate = circ_stats.get("dnf_rate", 0.1)
        circuit_avg_race = circ_stats.get("avg_finish", 10.0)
        circuit_wins = circ_stats.get("wins", 0)

        row = {
            "driver_id": driver_id,
            "team_id": team_id,
            # Grille de départ — feature la plus critique
            "grid_position": grid_pos,
            # Championnat
            "champ_points": drv_standing.get("points", 0),
            "champ_position": drv_standing.get("position", 20),
            "champ_wins": drv_standing.get("wins", 0),
            # Équipe
            "team_points": team_standing.get("points", 0),
            "team_position": team_standing.get("position", 10),
            # Circuit pilote
            "circuit_avg_race": circuit_avg_race,
            "circuit_best_race": circ_stats.get("best_finish", 20),
            "circuit_wins": circuit_wins,
            "circuit_podiums": circ_stats.get("podiums", 0),
            "circuit_races": circ_stats.get("races", 0),
            # Circuit équipe
            "team_circuit_avg": team_circ.get("avg_finish", 10.0),
            "team_circuit_wins": team_circ.get("wins", 0),
            # Pole→win sur ce circuit
            "pole_win_rate_circuit": pole_win_rate,
            "is_on_pole": int(grid_pos == 1),
            # Forme récente
            "recent_avg_finish": recent.get("recent_avg", 10.0),
            "recent_best_finish": recent.get("recent_best", 20),
            # Pit stop
            "avg_pit_stop": avg_pit,
            # Météo
            "temperature": temp,
            "is_wet": int(is_wet),
            "rain_probability": rain_prob,
            "wind_speed": wind,
            # Fiabilité
            "dnf_rate": dnf_rate,
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = _normalize_features(df, exclude=["driver_id", "team_id"])
    return df


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _normalize_features(df: pd.DataFrame, exclude: list[str] = None) -> pd.DataFrame:
    """Min-max normalization des features numériques (optionnel, XGBoost n'en a pas besoin)."""
    return df  # XGBoost n'est pas sensible à l'échelle


def _compute_pole_win_rate(circuit_history: list[dict]) -> float:
    """Taux pole → victoire sur ce circuit historiquement."""
    if not circuit_history:
        return 0.35  # Valeur typique F1
    wins_from_pole = sum(
        1 for h in circuit_history
        if h.get("winner_id") and h.get("pole_id") and
        h["winner_id"] == h["pole_id"]
    )
    return wins_from_pole / len(circuit_history)


def get_feature_names(mode: str = "qualifying") -> list[str]:
    """Noms des features pour SHAP."""
    base = [
        "champ_points", "champ_position", "champ_wins",
        "team_points", "team_position",
        "circuit_avg_finish", "circuit_best_finish",
        "circuit_avg_quali", "circuit_best_quali",
        "circuit_poles", "circuit_races",
        "team_circuit_avg", "team_circuit_best",
        "recent_avg_finish", "recent_best_finish",
        "temperature", "is_wet", "rain_probability", "wind_speed",
        "dnf_rate",
    ]
    if mode == "race":
        base = [
            "grid_position",
            "champ_points", "champ_position", "champ_wins",
            "team_points", "team_position",
            "circuit_avg_race", "circuit_best_race",
            "circuit_wins", "circuit_podiums", "circuit_races",
            "team_circuit_avg", "team_circuit_wins",
            "pole_win_rate_circuit", "is_on_pole",
            "recent_avg_finish", "recent_best_finish",
            "avg_pit_stop",
            "temperature", "is_wet", "rain_probability", "wind_speed",
            "dnf_rate",
        ]
    return base


def compute_shap_top3(shap_values: np.ndarray, feature_names: list[str],
                      n_top: int = 3) -> list[tuple[str, float]]:
    """
    Retourne les 3 features les plus importantes (SHAP) pour un pilote donné.
    shap_values: 1D array (une ligne du shap_values matrix)
    """
    pairs = list(zip(feature_names, shap_values))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    return pairs[:n_top]


def shap_top3_to_text(top3: list[tuple[str, float]]) -> str:
    """Formate les top SHAP features pour le prompt Mistral."""
    labels = {
        "grid_position": "Position de départ",
        "champ_points": "Points au championnat",
        "circuit_avg_race": "Moyenne course sur ce circuit",
        "circuit_avg_quali": "Moyenne qualifs sur ce circuit",
        "is_wet": "Pluie",
        "recent_avg_finish": "Forme récente",
        "team_circuit_avg": "Rythme équipe sur ce circuit",
        "circuit_wins": "Victoires sur ce circuit",
        "circuit_poles": "Poles sur ce circuit",
        "dnf_rate": "Taux abandon",
        "avg_pit_stop": "Vitesse pit stop équipe",
        "pole_win_rate_circuit": "Taux pole→victoire circuit",
        "temperature": "Température",
        "wind_speed": "Vent",
        "rain_probability": "Probabilité de pluie",
    }
    parts = []
    for feat, val in top3:
        label = labels.get(feat, feat)
        direction = "↑" if val > 0 else "↓"
        parts.append(f"{label} {direction} ({val:+.3f})")
    return ", ".join(parts)
