"""
Client Mistral API (tier gratuit) — explications narratives F1.

Rate limit : 2 RPM → retry automatique avec backoff 35s.
"""

import logging
import time
from typing import Optional

from config import (
    MISTRAL_API_KEY, MISTRAL_MODELS, MISTRAL_RETRY_DELAY
)

logger = logging.getLogger(__name__)

# Initialisation du client Mistral
_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client
    if not MISTRAL_API_KEY:
        raise ValueError("MISTRAL_API_KEY non défini — analyses IA désactivées")
    try:
        from mistralai import Mistral
        _client = Mistral(api_key=MISTRAL_API_KEY)
        return _client
    except ImportError:
        raise ImportError("pip install mistralai")


def _call_mistral(
    messages: list[dict],
    model: str = "mistral-small-latest",
    max_tokens: int = 300,
    retries: int = 3,
) -> str:
    """
    Appel API Mistral avec retry sur rate limit.
    """
    client = _get_client()
    last_error = None
    for attempt in range(retries):
        try:
            resp = client.chat.complete(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as e:
            err_str = str(e).lower()
            last_error = e
            if "rate" in err_str or "429" in err_str or "too many" in err_str:
                wait = MISTRAL_RETRY_DELAY * (attempt + 1)
                logger.warning(f"Rate limit Mistral — attente {wait}s (tentative {attempt+1}/{retries})")
                time.sleep(wait)
            else:
                logger.error(f"Mistral API error: {e}")
                raise
    raise RuntimeError(f"Mistral API indisponible après {retries} tentatives: {last_error}")


# ─── Fonctions publiques ──────────────────────────────────────────────────────

def explain_prediction(
    prediction: dict,
    circuit: str,
    conditions: dict,
    session_type: str = "course",
) -> str:
    """
    Explication narrative d'une prédiction XGBoost.

    prediction: {winner, win_prob, top3, shap_top3}
    circuit: nom du circuit
    conditions: {temperature, is_wet, rain_probability, wind_speed}
    """
    cond_str = (
        f"Température {conditions.get('temperature', 25):.0f}°C, "
        f"{'Pluie ☔' if conditions.get('is_wet') else 'Sec'}, "
        f"Vent {conditions.get('wind_speed', 15):.0f} km/h"
    )
    top3_str = ", ".join([
        f"{p['driver_name']} ({p['win_prob']:.0%})"
        for p in prediction.get("top3", [])[:3]
    ])
    shap_str = prediction.get("shap_top3", "Historique circuit, Forme saison, Conditions")

    prompt = f"""Tu es un expert F1. Explique cette prédiction en 3-4 phrases concises :

Circuit : {circuit}
Type de session : {session_type}
Conditions : {cond_str}
Vainqueur prédit : {prediction.get('winner', 'N/A')} ({prediction.get('win_prob', 0):.0%} probabilité)
Top 3 prédit : {top3_str}
Facteurs clés du modèle (SHAP) : {shap_str}

Contextualise pourquoi ces facteurs sont déterminants pour CE circuit dans CES conditions. \
Ne répète pas les chiffres, analyse le pourquoi sportif. Sois précis et factuel."""

    return _call_mistral(
        messages=[{"role": "user", "content": prompt}],
        model=MISTRAL_MODELS["fast"],
        max_tokens=350,
    )


def post_race_analysis(
    predicted: dict,
    actual: dict,
    circuit: str,
) -> str:
    """
    Résumé post-GP : prédit vs réel.

    predicted: {winner, top3: list[{driver_name, position}]}
    actual: {winner, top3: list[{driver_name, position}]}
    """
    pred_top3 = ", ".join([
        f"{p['driver_name']} (P{p['predicted_position']})"
        for p in predicted.get("top3", [])[:3]
    ])
    actual_top3 = ", ".join([
        f"{p['driver_name']} (P{p['position']})"
        for p in actual.get("top3", [])[:3]
    ])

    prompt = f"""Compare cette prédiction F1 avec le résultat réel (4-5 phrases) :

Circuit : {circuit}
Vainqueur prédit : {predicted.get('winner', 'N/A')}
Top 3 prédit : {pred_top3}

Vainqueur réel : {actual.get('winner', 'N/A')}
Top 3 réel : {actual_top3}

Identifie :
1) Ce que le modèle a correctement anticipé
2) Les surprises et leurs causes probables (incidents, météo, stratégie)
3) Ce que le modèle aurait dû mieux prendre en compte pour ce GP

Sois précis, factuel, et concis."""

    return _call_mistral(
        messages=[{"role": "user", "content": prompt}],
        model=MISTRAL_MODELS["best"],   # Large pour analyse approfondie
        max_tokens=550,
    )


def chat_about_prediction(
    question: str,
    context: str,
    history: Optional[list[dict]] = None,
) -> str:
    """
    Interface conversationnelle — questions libres sur les prédictions F1.

    question: Question de l'utilisateur
    context: Données de contexte (prédictions actuelles, classements, météo)
    history: Historique de conversation [{role, content}, ...]
    """
    system_msg = {
        "role": "system",
        "content": (
            "Tu es un expert F1 analysant des prédictions ML. "
            "Réponds uniquement sur la base des données fournies dans le contexte. "
            "Si l'information n'est pas disponible, dis-le clairement. "
            "Sois précis, factuel, et concis (< 200 mots)."
        ),
    }
    messages = [system_msg]

    # Ajouter l'historique de conversation
    if history:
        for msg in history[-6:]:  # Max 6 messages d'historique (3 échanges)
            messages.append(msg)

    messages.append({
        "role": "user",
        "content": f"Contexte des prédictions actuelles :\n{context}\n\nQuestion : {question}",
    })

    return _call_mistral(
        messages=messages,
        model=MISTRAL_MODELS["fast"],
        max_tokens=400,
    )


def is_available() -> bool:
    """Vérifie si la clé API Mistral est configurée."""
    return bool(MISTRAL_API_KEY)


def get_status() -> str:
    """Retourne le statut de l'intégration Mistral."""
    if not MISTRAL_API_KEY:
        return "⚠️ MISTRAL_API_KEY non défini — analyses IA désactivées"
    return "✅ Mistral API configurée (tier gratuit, 1B tokens/mois)"
