"""
Prédicteur F1 — XGBoost si modèle disponible, baseline statistique sinon.

Deux modes :
  1. XGBoost : charges les pkl entraînés, produit des prédictions + SHAP
  2. Baseline  : scoring statistique basé sur les features (sans entraînement)
"""

import logging
import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd

from config import QUALI_MODEL_PATH, RACE_MODEL_PATH
from features.engineering import (
    get_feature_names, compute_shap_top3, shap_top3_to_text
)

logger = logging.getLogger(__name__)


class F1Predictor:
    """
    Prédicteur unifié qualifications + course.
    Charge automatiquement XGBoost si les pkl existent,
    sinon utilise le baseline statistique.
    """

    def __init__(self):
        self.quali_model = None
        self.race_model = None
        self.shap_available = False
        self._load_models()

    def _load_models(self):
        """Charge les modèles XGBoost si disponibles."""
        for path, attr in [(QUALI_MODEL_PATH, "quali_model"),
                           (RACE_MODEL_PATH, "race_model")]:
            if os.path.exists(path):
                try:
                    with open(path, "rb") as f:
                        setattr(self, attr, pickle.load(f))
                    logger.info(f"Modèle chargé : {path}")
                except Exception as e:
                    logger.warning(f"Impossible de charger {path}: {e}")

        # Vérifier SHAP
        if self.quali_model is not None or self.race_model is not None:
            try:
                import shap  # noqa: F401
                self.shap_available = True
            except ImportError:
                pass

    @property
    def using_ml(self) -> bool:
        return self.quali_model is not None or self.race_model is not None

    def predict_qualifying(self, features_df: pd.DataFrame) -> list[dict]:
        """
        Prédiction qualifications.

        Retourne une liste triée de :
          {driver_id, predicted_position, win_prob, top3_prob, shap_top3}
        """
        if features_df.empty:
            return []

        feature_cols = get_feature_names("qualifying")
        available_cols = [c for c in feature_cols if c in features_df.columns]
        X = features_df[available_cols].fillna(0).values

        if self.quali_model is not None:
            return self._predict_xgboost(
                self.quali_model, features_df, X, available_cols, mode="qualifying"
            )
        return self._predict_baseline(features_df, mode="qualifying")

    def predict_race(self, features_df: pd.DataFrame) -> list[dict]:
        """
        Prédiction course.

        Retourne une liste triée de :
          {driver_id, predicted_position, win_prob, top3_prob, shap_top3}
        """
        if features_df.empty:
            return []

        feature_cols = get_feature_names("race")
        available_cols = [c for c in feature_cols if c in features_df.columns]
        X = features_df[available_cols].fillna(0).values

        if self.race_model is not None:
            return self._predict_xgboost(
                self.race_model, features_df, X, available_cols, mode="race"
            )
        return self._predict_baseline(features_df, mode="race")

    def _predict_xgboost(
        self,
        model,
        features_df: pd.DataFrame,
        X: np.ndarray,
        feature_cols: list[str],
        mode: str,
    ) -> list[dict]:
        """Prédiction via XGBoost avec SHAP."""
        try:
            # Prédiction de position (regression) ou ranking
            preds = model.predict(X)
        except Exception as e:
            logger.error(f"XGBoost predict error: {e}")
            return self._predict_baseline(features_df, mode)

        # SHAP
        shap_values_all = None
        if self.shap_available:
            try:
                import shap
                explainer = shap.TreeExplainer(model)
                shap_values_all = explainer.shap_values(X)
            except Exception as e:
                logger.warning(f"SHAP error: {e}")

        # Softmax pour convertir scores → probabilités
        exp_neg = np.exp(-preds)  # Score inversé (moins = meilleur)
        probs = exp_neg / exp_neg.sum()
        top3_probs = self._top3_probs(preds)

        results = []
        for i, row in features_df.iterrows():
            idx = features_df.index.get_loc(i)
            shap_top3_text = ""
            if shap_values_all is not None and idx < len(shap_values_all):
                top3 = compute_shap_top3(shap_values_all[idx], feature_cols)
                shap_top3_text = shap_top3_to_text(top3)
            results.append({
                "driver_id": row.get("driver_id", ""),
                "team_id": row.get("team_id", ""),
                "score": float(preds[idx]),
                "win_prob": float(probs[idx]),
                "top3_prob": float(top3_probs[idx]),
                "shap_top3": shap_top3_text,
                "model": "xgboost",
            })

        results.sort(key=lambda x: x["score"])
        for rank, r in enumerate(results):
            r["predicted_position"] = rank + 1
        return results

    def _predict_baseline(
        self, features_df: pd.DataFrame, mode: str
    ) -> list[dict]:
        """
        Baseline statistique : scoring pondéré des features.
        Utilisé quand aucun modèle XGBoost n'est disponible.
        """
        scores = np.zeros(len(features_df))
        df = features_df.reset_index(drop=True)

        if mode == "qualifying":
            # Poids des features qualifications
            weights = {
                "champ_points": -0.15,     # Plus de points = meilleur
                "champ_position": 0.25,     # Moins bon classement = pénalité
                "circuit_avg_quali": 0.30,  # Historique qualif sur ce circuit
                "circuit_best_quali": 0.15,
                "team_points": -0.10,
                "recent_avg_finish": 0.20,
            }
        else:
            weights = {
                "grid_position": 0.40,       # Position de départ — critique
                "champ_points": -0.10,
                "champ_position": 0.15,
                "circuit_avg_race": 0.20,
                "team_circuit_avg": 0.10,
                "recent_avg_finish": 0.15,
                "dnf_rate": 0.05,
            }

        for feat, w in weights.items():
            if feat in df.columns:
                col = df[feat].fillna(df[feat].median() if feat in df else 10)
                # Normaliser
                col_min, col_max = col.min(), col.max()
                if col_max > col_min:
                    col_norm = (col - col_min) / (col_max - col_min)
                else:
                    col_norm = col * 0
                scores += w * col_norm

        # Ajustement pluie : pénalise/aide selon style connu
        if "is_wet" in df.columns and df["is_wet"].any():
            # Boost aléatoire simulant la variabilité par temps de pluie
            np.random.seed(42)
            rain_boost = np.random.normal(0, 0.05, len(scores))
            scores += rain_boost

        # Légère randomisation pour éviter les ex-aequo
        np.random.seed(42)
        noise = np.random.normal(0, 0.02, len(scores))
        scores += noise

        # Softmax pour probabilités
        min_s = scores.min()
        exp_s = np.exp(-(scores - min_s) * 3)  # Scores négatifs = meilleur
        probs = exp_s / exp_s.sum()
        top3_probs = self._top3_probs(scores)

        results = []
        for i in range(len(df)):
            results.append({
                "driver_id": df.iloc[i].get("driver_id", ""),
                "team_id": df.iloc[i].get("team_id", ""),
                "score": float(scores[i]),
                "win_prob": float(probs[i]),
                "top3_prob": float(top3_probs[i]),
                "shap_top3": _baseline_explanation(df.iloc[i], mode),
                "model": "baseline",
            })

        results.sort(key=lambda x: x["score"])
        for rank, r in enumerate(results):
            r["predicted_position"] = rank + 1
        return results

    @staticmethod
    def _top3_probs(scores: np.ndarray) -> np.ndarray:
        """
        Probabilité de finir dans le top 3 par simulation Monte Carlo.
        """
        n = len(scores)
        top3 = np.zeros(n)
        n_sim = 500
        rng = np.random.default_rng(42)
        for _ in range(n_sim):
            noise = rng.normal(0, 0.5, n)
            perturbed = scores + noise
            ranking = np.argsort(perturbed)
            top3[ranking[:3]] += 1
        return top3 / n_sim


def _baseline_explanation(row: pd.Series, mode: str) -> str:
    """Explication textuelle pour le baseline (remplace SHAP)."""
    parts = []
    if mode == "race":
        gp = row.get("grid_position", 10)
        if gp <= 3:
            parts.append(f"Position de départ favorable (P{int(gp)})")
        elif gp > 12:
            parts.append(f"Départ difficile (P{int(gp)})")
    pts = row.get("champ_points", 0)
    if pts > 150:
        parts.append("Forte saison")
    circuit_avg = row.get("circuit_avg_race") or row.get("circuit_avg_quali", 10)
    if circuit_avg and circuit_avg < 5:
        parts.append("Excellent historique sur ce circuit")
    is_wet = row.get("is_wet", 0)
    if is_wet:
        parts.append("Conditions humides (impact variable)")
    return " | ".join(parts) if parts else "Scoring statistique"


# Singleton global
_predictor: Optional[F1Predictor] = None


def get_predictor() -> F1Predictor:
    """Retourne le prédicteur (singleton)."""
    global _predictor
    if _predictor is None:
        _predictor = F1Predictor()
    return _predictor
