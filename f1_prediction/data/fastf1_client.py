"""Client FastF1 — télémétrie, secteurs, pneus, pit stops."""

import os
import logging
from functools import lru_cache
from typing import Optional

import pandas as pd

from config import FASTF1_CACHE_DIR

logger = logging.getLogger(__name__)

# Initialisation du cache FastF1
os.makedirs(FASTF1_CACHE_DIR, exist_ok=True)

try:
    import fastf1
    fastf1.Cache.enable_cache(FASTF1_CACHE_DIR)
    FASTF1_AVAILABLE = True
    logger.info("FastF1 initialisé avec cache %s", FASTF1_CACHE_DIR)
except ImportError:
    FASTF1_AVAILABLE = False
    logger.warning("FastF1 non disponible — fonctionnalités réduites")


def _load_session(year: int, gp: str | int, session_type: str):
    """Charge une session FastF1."""
    if not FASTF1_AVAILABLE:
        return None
    try:
        import fastf1
        session = fastf1.get_session(year, gp, session_type)
        session.load(telemetry=False, weather=True, messages=False)
        return session
    except Exception as e:
        logger.warning(f"FastF1 load error ({year} {gp} {session_type}): {e}")
        return None


@lru_cache(maxsize=30)
def get_quali_lap_times(year: int, gp: str | int) -> pd.DataFrame:
    """
    Temps au tour des qualifications par pilote.
    Retourne DataFrame: driver_number, driver, team, lap_time_s, compound, sector1, sector2, sector3
    """
    session = _load_session(year, gp, "Q")
    if session is None:
        return pd.DataFrame()
    try:
        laps = session.laps.pick_quicklaps()
        fastest = laps.groupby("Driver").apply(
            lambda x: x.loc[x["LapTime"].idxmin()]
        ).reset_index(drop=True)
        result = fastest[["Driver", "Team", "LapTime", "Compound",
                           "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
        result["LapTime_s"] = result["LapTime"].dt.total_seconds()
        result["Sector1_s"] = result["Sector1Time"].dt.total_seconds()
        result["Sector2_s"] = result["Sector2Time"].dt.total_seconds()
        result["Sector3_s"] = result["Sector3Time"].dt.total_seconds()
        result["LapTime_rank"] = result["LapTime_s"].rank()
        return result
    except Exception as e:
        logger.warning(f"FastF1 quali laps error: {e}")
        return pd.DataFrame()


@lru_cache(maxsize=30)
def get_race_pit_stops(year: int, gp: str | int) -> pd.DataFrame:
    """
    Données pit stops : durée moyenne par équipe et par pilote.
    """
    session = _load_session(year, gp, "R")
    if session is None:
        return pd.DataFrame()
    try:
        laps = session.laps
        pit_laps = laps[laps["PitOutTime"].notna() | laps["PitInTime"].notna()].copy()
        if pit_laps.empty:
            return pd.DataFrame()
        # Durée du pit stop = temps de sortie - temps d'entrée
        pit_stops = []
        for driver in pit_laps["Driver"].unique():
            driver_laps = laps[laps["Driver"] == driver].sort_values("LapNumber")
            for _, lap in driver_laps.iterrows():
                if pd.notna(lap.get("PitInTime")) and pd.notna(lap.get("PitOutTime")):
                    duration = (lap["PitOutTime"] - lap["PitInTime"]).total_seconds()
                    if 1 < duration < 120:  # Filtrer valeurs aberrantes
                        pit_stops.append({
                            "Driver": driver,
                            "Team": lap.get("Team", ""),
                            "Lap": lap["LapNumber"],
                            "Duration_s": duration,
                        })
        df = pd.DataFrame(pit_stops)
        if df.empty:
            return df
        # Durée moyenne par équipe
        team_avg = df.groupby("Team")["Duration_s"].mean().reset_index()
        team_avg.columns = ["Team", "AvgPitStop_s"]
        return team_avg
    except Exception as e:
        logger.warning(f"FastF1 pit stops error: {e}")
        return pd.DataFrame()


@lru_cache(maxsize=30)
def get_tyre_degradation(year: int, gp: str | int) -> dict:
    """
    Modélisation de la dégradation pneus par compound.
    Retourne delta de temps par tour pour chaque compound.
    """
    session = _load_session(year, gp, "R")
    if session is None:
        return {}
    try:
        laps = session.laps
        laps = laps[laps["LapTime"].notna() & laps["Compound"].notna()]
        laps = laps.copy()
        laps["LapTime_s"] = laps["LapTime"].dt.total_seconds()
        laps = laps[laps["LapTime_s"] < laps["LapTime_s"].quantile(0.95)]

        degradation = {}
        for compound in laps["Compound"].unique():
            c_laps = laps[laps["Compound"] == compound].sort_values("TyreLife")
            if len(c_laps) < 5:
                continue
            # Régression linéaire simple : delta_time = f(tyre_life)
            from numpy.polynomial import polynomial as P
            x = c_laps["TyreLife"].values
            y = c_laps["LapTime_s"].values
            try:
                coeffs = P.polyfit(x, y, 1)
                degradation[compound] = float(coeffs[1])  # pente (s/tour)
            except Exception:
                pass
        return degradation
    except Exception as e:
        logger.warning(f"FastF1 tyre degradation error: {e}")
        return {}


@lru_cache(maxsize=20)
def get_team_pace_at_circuit(year: int, gp: str | int) -> pd.DataFrame:
    """
    Rythme de chaque équipe pendant la course (laps représentatifs).
    """
    session = _load_session(year, gp, "R")
    if session is None:
        return pd.DataFrame()
    try:
        laps = session.laps.pick_quicklaps()
        laps = laps.copy()
        laps["LapTime_s"] = laps["LapTime"].dt.total_seconds()
        team_pace = (laps.groupby("Team")["LapTime_s"]
                     .agg(["mean", "median", "min"])
                     .reset_index())
        team_pace.columns = ["Team", "AvgPace_s", "MedianPace_s", "BestPace_s"]
        team_pace["PaceRank"] = team_pace["MedianPace_s"].rank()
        return team_pace
    except Exception as e:
        logger.warning(f"FastF1 team pace error: {e}")
        return pd.DataFrame()


def get_weather_from_session(year: int, gp: str | int,
                             session_type: str = "Q") -> dict:
    """
    Conditions météo enregistrées pendant la session (FastF1).
    Plus précis que Open-Meteo pour les données historiques.
    """
    session = _load_session(year, gp, session_type)
    if session is None:
        return {}
    try:
        weather = session.weather_data
        if weather is None or weather.empty:
            return {}
        return {
            "temperature": float(weather["AirTemp"].mean()),
            "track_temp": float(weather["TrackTemp"].mean()),
            "humidity": float(weather["Humidity"].mean()),
            "wind_speed": float(weather["WindSpeed"].mean()),
            "rain_observed": bool(weather["Rainfall"].any()),
            "source": "fastf1",
        }
    except Exception as e:
        logger.warning(f"FastF1 weather error: {e}")
        return {}


def get_safety_car_history(year: int, gp: str | int) -> dict:
    """
    Compte les périodes de Safety Car / VSC en course.
    """
    session = _load_session(year, gp, "R")
    if session is None:
        return {"sc_laps": 0, "vsc_laps": 0, "sc_deployments": 0}
    try:
        track_status = session.track_status
        if track_status is None or track_status.empty:
            return {"sc_laps": 0, "vsc_laps": 0, "sc_deployments": 0}
        sc_laps = (track_status["Status"] == "4").sum()  # SC
        vsc_laps = (track_status["Status"] == "6").sum()  # VSC
        # Compte les déploiements (transitions 1→4)
        deployments = (
            (track_status["Status"] == "4") &
            (track_status["Status"].shift(1) != "4")
        ).sum()
        return {
            "sc_laps": int(sc_laps),
            "vsc_laps": int(vsc_laps),
            "sc_deployments": int(deployments),
        }
    except Exception as e:
        logger.warning(f"FastF1 safety car error: {e}")
        return {"sc_laps": 0, "vsc_laps": 0, "sc_deployments": 0}
