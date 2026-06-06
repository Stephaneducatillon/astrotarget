"""
Script d'entraînement XGBoost — à exécuter en local.

Usage :
    cd f1_prediction
    python -m models.train --years 2018 2019 2020 2021 2022 2023 2024 2025
    python -m models.train --mode qualifying --years 2020 2021 2022 2023 2024

Les modèles .pkl sont sauvegardés dans models/artifacts/
et doivent être committés dans le repo Git HF Spaces.
"""

import argparse
import logging
import os
import pickle
import sys

import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    MODEL_DIR, QUALI_MODEL_PATH, RACE_MODEL_PATH,
    TRAINING_YEARS, XGBOOST_PARAMS
)
from data.ergast_client import (
    get_season_schedule, get_qualifying_results, get_race_results,
    get_driver_standings, get_constructor_standings, get_circuit_history
)
from data.weather_client import get_weather_historical
from features.engineering import (
    build_qualifying_features, build_race_features, get_feature_names
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

os.makedirs(MODEL_DIR, exist_ok=True)


# ─── Collecte des données ─────────────────────────────────────────────────────

def collect_training_data(years: list[int], mode: str = "qualifying") -> pd.DataFrame:
    """
    Collecte les données d'entraînement depuis Jolpica + Open-Meteo.
    mode: 'qualifying' ou 'race'
    """
    all_rows = []
    for year in years:
        logger.info(f"Collecte année {year}...")
        schedule = get_season_schedule(year)
        driver_standings = get_driver_standings(year)
        team_standings = get_constructor_standings(year)

        for race in schedule:
            round_num = race["round"]
            circuit_id = race["circuit_id"]
            lat = race["lat"]
            lon = race["lon"]
            race_date = race["date"]
            logger.info(f"  {race['name']} (R{round_num})")

            # Résultats réels (target)
            if mode == "qualifying":
                results = get_qualifying_results(year, round_num)
                if not results:
                    continue
            else:
                results = get_race_results(year, round_num)
                if not results:
                    continue

            # Météo historique
            try:
                weather = get_weather_historical(lat, lon, race_date)
            except Exception:
                weather = None

            # Standings au moment du GP
            try:
                drv_standings_at = get_driver_standings(year, round_num - 1) if round_num > 1 else []
                team_standings_at = get_constructor_standings(year, round_num - 1) if round_num > 1 else []
            except Exception:
                drv_standings_at = driver_standings
                team_standings_at = team_standings

            # Historique circuit
            try:
                circ_history = get_circuit_history(circuit_id, years=min(year - 2014, 7))
            except Exception:
                circ_history = []

            # Construire la liste de pilotes à partir des résultats
            if mode == "qualifying":
                drivers = [
                    {"driver_id": r["driver_id"], "team_id": r["team_id"],
                     "team": r["team"], "driver_code": r["driver_code"]}
                    for r in results
                ]
                target_map = {r["driver_id"]: r["position"] for r in results}
                features_df = build_qualifying_features(
                    year=year, round_num=round_num, circuit_id=circuit_id,
                    circuit_lat=lat, circuit_lon=lon, race_date=race_date,
                    drivers=drivers, driver_standings=drv_standings_at,
                    team_standings=team_standings_at, circuit_history=circ_history,
                    weather=weather,
                )
            else:
                quali = get_qualifying_results(year, round_num)
                drivers = [
                    {"driver_id": r["driver_id"], "team_id": r["team_id"],
                     "team": r["team"], "driver_code": r["driver_code"]}
                    for r in results
                ]
                target_map = {r["driver_id"]: r["position"] for r in results}
                features_df = build_race_features(
                    year=year, round_num=round_num, circuit_id=circuit_id,
                    drivers=drivers, quali_results=quali,
                    driver_standings=drv_standings_at,
                    team_standings=team_standings_at,
                    circuit_history=circ_history, weather=weather,
                )

            if features_df.empty:
                continue

            features_df["target_position"] = features_df["driver_id"].map(target_map)
            features_df["year"] = year
            features_df["round"] = round_num
            features_df["circuit_id"] = circuit_id
            all_rows.append(features_df)

        # Purge cache LRU pour économiser la mémoire entre saisons
        get_season_schedule.cache_clear()
        get_qualifying_results.cache_clear()
        get_race_results.cache_clear()
        get_driver_standings.cache_clear()

    if not all_rows:
        logger.warning("Aucune donnée collectée !")
        return pd.DataFrame()

    df = pd.concat(all_rows, ignore_index=True)
    df = df.dropna(subset=["target_position"])
    logger.info(f"Dataset final : {len(df)} lignes, {df.shape[1]} colonnes")
    return df


# ─── Entraînement ─────────────────────────────────────────────────────────────

def train_model(df: pd.DataFrame, mode: str = "qualifying") -> XGBRegressor:
    """
    Entraîne un modèle XGBoost sur le dataset.
    mode: 'qualifying' ou 'race'
    """
    feature_cols = get_feature_names(mode)
    available_cols = [c for c in feature_cols if c in df.columns]

    logger.info(f"Features utilisées ({len(available_cols)}) : {available_cols}")

    X = df[available_cols].fillna(0)
    y = df["target_position"]

    # Encodage des variables catégorielles
    le_driver = LabelEncoder()
    le_team = LabelEncoder()
    if "driver_id" in X.columns:
        X = X.copy()
        X["driver_encoded"] = le_driver.fit_transform(df["driver_id"].fillna("unknown"))
    if "team_id" in X.columns:
        if "driver_encoded" not in X.columns:
            X = X.copy()
        X["team_encoded"] = le_team.fit_transform(df["team_id"].fillna("unknown"))

    # Supprimer les colonnes non numériques
    X = X.select_dtypes(include=[np.number])

    logger.info(f"Shape X: {X.shape}, y: {y.shape}")
    logger.info(f"Distribution y: mean={y.mean():.2f}, std={y.std():.2f}")

    # Cross-validation temporelle (TimeSeriesSplit sur les années)
    tscv = TimeSeriesSplit(n_splits=3)
    model = XGBRegressor(**XGBOOST_PARAMS)

    cv_scores = cross_val_score(
        model, X, y, cv=tscv, scoring="neg_mean_absolute_error"
    )
    logger.info(f"CV MAE: {-cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Entraînement final sur tout le dataset
    model.fit(X, y)

    return model, available_cols, le_driver, le_team


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Métriques d'évaluation selon le CDC."""
    from scipy.stats import spearmanr

    preds = model.predict(X_test.fillna(0))

    # MAE Position
    mae = np.mean(np.abs(preds - y_test))

    # Top-1 Accuracy
    pred_winner = np.argmin(preds)
    true_winner = np.argmin(y_test.values)
    top1_acc = int(pred_winner == true_winner)

    # Top-3 Accuracy (basé sur la corrélation de ranking)
    pred_ranking = np.argsort(preds)[:3]
    true_ranking = np.argsort(y_test.values)[:3]
    top3_acc = len(set(pred_ranking) & set(true_ranking)) / 3

    # Spearman Rank Correlation
    spearman, _ = spearmanr(preds, y_test.values)

    return {
        "mae": float(mae),
        "top1_accuracy": float(top1_acc),
        "top3_accuracy": float(top3_acc),
        "spearman": float(spearman),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Entraînement modèle F1")
    parser.add_argument("--years", nargs="+", type=int, default=TRAINING_YEARS)
    parser.add_argument("--mode", choices=["qualifying", "race", "both"], default="both")
    parser.add_argument("--mlflow", action="store_true", help="Tracker avec MLflow")
    args = parser.parse_args()

    modes = ["qualifying", "race"] if args.mode == "both" else [args.mode]

    for mode in modes:
        logger.info(f"\n{'='*50}")
        logger.info(f"Mode : {mode.upper()}")
        logger.info(f"Années : {args.years}")
        logger.info("="*50)

        # Collecte
        df = collect_training_data(args.years, mode)
        if df.empty:
            logger.error("Aucune donnée — arrêt")
            continue

        # Sauvegarde du dataset
        dataset_path = os.path.join(MODEL_DIR, f"{mode}_dataset.parquet")
        df.to_parquet(dataset_path, index=False)
        logger.info(f"Dataset sauvegardé : {dataset_path}")

        # Train/test split temporel : dernière saison = test
        max_year = df["year"].max()
        train_df = df[df["year"] < max_year]
        test_df = df[df["year"] == max_year]

        logger.info(f"Train: {len(train_df)} lignes | Test: {len(test_df)} lignes")

        # Entraînement
        if args.mlflow:
            mlflow.start_run(run_name=f"f1_{mode}_{max_year}")

        model, feature_cols, le_driver, le_team = train_model(train_df, mode)

        # Évaluation
        feature_cols_avail = [c for c in feature_cols if c in test_df.columns]
        X_test = test_df[feature_cols_avail].select_dtypes(include=[np.number]).fillna(0)
        y_test = test_df["target_position"]
        metrics = evaluate_model(model, X_test, y_test)

        logger.info(f"Métriques test ({max_year}) :")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")

        if args.mlflow:
            mlflow.log_metrics(metrics)
            mlflow.xgboost.log_model(model, f"f1_{mode}_model")
            mlflow.end_run()

        # Sauvegarde
        model_path = QUALI_MODEL_PATH if mode == "qualifying" else RACE_MODEL_PATH
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"Modèle sauvegardé : {model_path}")

        # Sauvegarde feature importance
        fi = pd.DataFrame({
            "feature": feature_cols_avail,
            "importance": model.feature_importances_[:len(feature_cols_avail)],
        }).sort_values("importance", ascending=False)
        fi_path = os.path.join(MODEL_DIR, f"{mode}_feature_importance.csv")
        fi.to_csv(fi_path, index=False)
        logger.info(f"Feature importance : {fi_path}")
        logger.info("\nTop 10 features :")
        logger.info(fi.head(10).to_string(index=False))

    logger.info("\nEntraînement terminé. Commitez les .pkl dans Git HF Spaces.")


if __name__ == "__main__":
    main()
