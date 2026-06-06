"""
Application F1 Prévision v0.5 — Interface Gradio pour HF Spaces.

Stack : FastF1 + Jolpica + OpenF1 + Open-Meteo + XGBoost + Mistral API
Hébergement : Hugging Face Spaces (gratuit)
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

# ─── Path setup ───────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Imports applicatifs ──────────────────────────────────────────────────────
from config import CURRENT_YEAR
from data.ergast_client import (
    get_season_schedule, get_qualifying_results, get_race_results,
    get_driver_standings, get_constructor_standings, get_circuit_history
)
from data.weather_client import get_weather_for_race, weather_summary
from data.openf1_client import get_latest_session, get_current_drivers, get_live_weather
from models.predictor import get_predictor
from features.engineering import build_qualifying_features, build_race_features
import llm.mistral_client as mistral

# ─── CSS personnalisé ─────────────────────────────────────────────────────────
CUSTOM_CSS = """
.f1-header { background: linear-gradient(135deg, #e10600 0%, #1a1a1a 100%);
             color: white; padding: 20px; border-radius: 10px; margin-bottom: 15px; }
.metric-card { background: #1a1a1a; color: white; padding: 15px;
               border-radius: 8px; border-left: 4px solid #e10600; }
.driver-p1 { color: #FFD700; font-weight: bold; }
.driver-p2 { color: #C0C0C0; font-weight: bold; }
.driver-p3 { color: #CD7F32; font-weight: bold; }
.wet-badge { background: #0066cc; color: white; padding: 2px 8px;
             border-radius: 4px; font-size: 0.85em; }
.dry-badge  { background: #e10600; color: white; padding: 2px 8px;
              border-radius: 4px; font-size: 0.85em; }
"""

# ─── Cache mémoire session ────────────────────────────────────────────────────
_cache: dict = {}

TEAM_COLORS = {
    "Red Bull": "#3671C6", "Ferrari": "#E8002D", "Mercedes": "#27F4D2",
    "McLaren": "#FF8000", "Aston Martin": "#229971", "Alpine": "#FF87BC",
    "Williams": "#64C4FF", "Kick Sauber": "#52E252", "Racing Bulls": "#6692FF",
    "Haas": "#B6BABD",
}
DEFAULT_COLOR = "#888888"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _team_color(team: str) -> str:
    for k, v in TEAM_COLORS.items():
        if k.lower() in team.lower():
            return v
    return DEFAULT_COLOR


def _medal(pos: int) -> str:
    return {1: "🥇", 2: "🥈", 3: "🥉"}.get(pos, f"P{pos}")


def _build_prediction_context(
    predictions: list[dict],
    enriched: list[dict],
    circuit_name: str,
    weather: dict,
    mode: str,
) -> str:
    """Construit le texte de contexte pour le chat Mistral."""
    lines = [f"Circuit : {circuit_name}", f"Session : {mode}"]
    lines.append(f"Météo : {weather_summary(weather)}")
    lines.append("\nPrédictions :")
    for p in predictions[:5]:
        drv = next((d for d in enriched if d.get("driver_id") == p["driver_id"]), {})
        name = drv.get("driver_name", p["driver_id"])
        lines.append(
            f"  P{p['predicted_position']}: {name} — "
            f"Victoire {p['win_prob']:.0%} | Top3 {p['top3_prob']:.0%}"
        )
    return "\n".join(lines)


def _get_or_fetch_schedule(year: int) -> list[dict]:
    key = f"schedule_{year}"
    if key not in _cache:
        _cache[key] = get_season_schedule(year)
    return _cache[key]


def _enriched_predictions(predictions: list[dict], drivers: list[dict]) -> list[dict]:
    """Merge prédictions + infos pilotes."""
    driver_map = {d["driver_id"]: d for d in drivers}
    result = []
    for p in predictions:
        d = driver_map.get(p["driver_id"], {})
        result.append({**p, **d})
    return result


# ─── Graphiques Plotly ────────────────────────────────────────────────────────

def make_probability_chart(predictions: list[dict], enriched: list[dict],
                           title: str = "Probabilités de victoire") -> go.Figure:
    """Bar chart horizontal des probabilités."""
    if not predictions:
        return go.Figure()

    driver_map = {d.get("driver_id", ""): d for d in enriched}
    top_n = predictions[:10]

    names = []
    win_probs = []
    top3_probs = []
    colors = []

    for p in reversed(top_n):  # Inversé pour que P1 soit en haut
        d = driver_map.get(p["driver_id"], {})
        name = d.get("driver_name", p["driver_id"])
        team = d.get("team", "")
        names.append(f"{_medal(p['predicted_position'])} {name}")
        win_probs.append(p["win_prob"] * 100)
        top3_probs.append(p["top3_prob"] * 100)
        colors.append(_team_color(team))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Top 3 (%)", y=names, x=top3_probs,
        orientation="h", marker_color=colors, opacity=0.4,
        text=[f"{v:.0f}%" for v in top3_probs], textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="Victoire (%)", y=names, x=win_probs,
        orientation="h", marker_color=colors,
        text=[f"{v:.0f}%" for v in win_probs], textposition="outside",
    ))
    fig.update_layout(
        title=title,
        barmode="overlay",
        plot_bgcolor="#1a1a1a",
        paper_bgcolor="#111111",
        font_color="white",
        height=400,
        margin=dict(l=160, r=80, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis=dict(range=[0, 100], title="%"),
    )
    return fig


def make_ranking_table(predictions: list[dict], enriched: list[dict]) -> pd.DataFrame:
    """Tableau de classement pour affichage Gradio."""
    driver_map = {d.get("driver_id", ""): d for d in enriched}
    rows = []
    for p in predictions[:20]:
        d = driver_map.get(p["driver_id"], {})
        pos = p["predicted_position"]
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(pos, str(pos))
        rows.append({
            "Pos": medal,
            "Pilote": d.get("driver_name", p["driver_id"]),
            "Équipe": d.get("team", p.get("team_id", "")),
            "Victoire": f"{p['win_prob']*100:.1f}%",
            "Top 3": f"{p['top3_prob']*100:.1f}%",
            "Facteurs clés": p.get("shap_top3", "") or "—",
        })
    return pd.DataFrame(rows)


# ─── Logique principale ───────────────────────────────────────────────────────

def load_race_data(year: int, round_num: int) -> dict | None:
    """
    Charge toutes les données nécessaires pour un GP :
    schedule info, standings, historique circuit, météo.
    """
    key = f"data_{year}_{round_num}"
    if key in _cache:
        return _cache[key]

    schedule = _get_or_fetch_schedule(year)
    race_info = next((r for r in schedule if r["round"] == round_num), None)
    if not race_info:
        return None

    circuit_id = race_info["circuit_id"]

    try:
        drv_standings = get_driver_standings(year, round_num - 1 if round_num > 1 else None)
    except Exception:
        drv_standings = []
    try:
        team_standings = get_constructor_standings(year, round_num - 1 if round_num > 1 else None)
    except Exception:
        team_standings = []
    try:
        circ_history = get_circuit_history(circuit_id, years=6)
    except Exception:
        circ_history = []

    # Météo
    weather_data = get_weather_for_race(
        race_info["lat"], race_info["lon"], race_info["date"]
    )

    data = {
        "race_info": race_info,
        "circuit_id": circuit_id,
        "driver_standings": drv_standings,
        "team_standings": team_standings,
        "circuit_history": circ_history,
        "weather": weather_data,
    }
    _cache[key] = data
    return data


def run_qualifying_prediction(year: int, round_num: int) -> tuple:
    """
    Lance la prédiction qualifications.
    Retourne (table_df, figure, narrative_text, context_str)
    """
    predictor = get_predictor()
    data = load_race_data(year, round_num)
    if data is None:
        return pd.DataFrame(), go.Figure(), "❌ Données introuvables pour ce GP.", ""

    race_info = data["race_info"]
    circuit_id = data["circuit_id"]
    weather = data["weather"]["qualifying"]

    # Pilotes = depuis les standings ou résultats récents
    drivers = [
        {
            "driver_id": d["driver_id"],
            "team_id": d["team_id"],
            "team": d["team"],
            "driver_code": d.get("driver_code", ""),
            "driver_name": d.get("driver_name", ""),
        }
        for d in data["driver_standings"]
    ]
    if not drivers:
        return pd.DataFrame(), go.Figure(), "❌ Pilotes introuvables.", ""

    # Feature engineering
    features_df = build_qualifying_features(
        year=year, round_num=round_num,
        circuit_id=circuit_id,
        circuit_lat=race_info["lat"],
        circuit_lon=race_info["lon"],
        race_date=race_info["date"],
        drivers=drivers,
        driver_standings=data["driver_standings"],
        team_standings=data["team_standings"],
        circuit_history=data["circuit_history"],
        weather=weather,
    )

    # Prédiction
    predictions = predictor.predict_qualifying(features_df)
    enriched = _enriched_predictions(predictions, drivers)

    # Graphique + table
    fig = make_probability_chart(predictions, enriched,
                                 title=f"Qualifications — {race_info['name']} {year}")
    table = make_ranking_table(predictions, enriched)

    # Contexte pour le chat
    context = _build_prediction_context(
        predictions, enriched, race_info["name"], weather, "qualifications"
    )

    # Explication Mistral (si disponible)
    narrative = ""
    if predictions and mistral.is_available():
        winner_drv = next(
            (d for d in enriched if d.get("predicted_position") == 1), {}
        )
        try:
            narrative = mistral.explain_prediction(
                prediction={
                    "winner": winner_drv.get("driver_name", "N/A"),
                    "win_prob": predictions[0]["win_prob"],
                    "top3": enriched[:3],
                    "shap_top3": predictions[0].get("shap_top3", ""),
                },
                circuit=race_info["name"],
                conditions=weather,
                session_type="qualifications",
            )
        except Exception as e:
            narrative = f"⚠️ Analyse IA indisponible : {e}"
    elif not mistral.is_available():
        narrative = mistral.get_status()

    return table, fig, narrative, context


def run_race_prediction(year: int, round_num: int,
                        use_actual_quali: bool = True) -> tuple:
    """
    Lance la prédiction course.
    Retourne (table_df, figure, narrative_text, context_str)
    """
    predictor = get_predictor()
    data = load_race_data(year, round_num)
    if data is None:
        return pd.DataFrame(), go.Figure(), "❌ Données introuvables.", ""

    race_info = data["race_info"]
    circuit_id = data["circuit_id"]
    weather = data["weather"]["race"]

    drivers = [
        {
            "driver_id": d["driver_id"],
            "team_id": d["team_id"],
            "team": d["team"],
            "driver_code": d.get("driver_code", ""),
            "driver_name": d.get("driver_name", ""),
        }
        for d in data["driver_standings"]
    ]
    if not drivers:
        return pd.DataFrame(), go.Figure(), "❌ Pilotes introuvables.", ""

    # Résultats qualifications réels (si disponibles)
    quali_results = []
    if use_actual_quali:
        try:
            quali_results = get_qualifying_results(year, round_num)
        except Exception:
            pass

    # Feature engineering
    features_df = build_race_features(
        year=year, round_num=round_num,
        circuit_id=circuit_id,
        drivers=drivers,
        quali_results=quali_results,
        driver_standings=data["driver_standings"],
        team_standings=data["team_standings"],
        circuit_history=data["circuit_history"],
        weather=weather,
    )

    predictions = predictor.predict_race(features_df)
    enriched = _enriched_predictions(predictions, drivers)

    fig = make_probability_chart(predictions, enriched,
                                 title=f"Course — {race_info['name']} {year}")
    table = make_ranking_table(predictions, enriched)

    context = _build_prediction_context(
        predictions, enriched, race_info["name"], weather, "course"
    )

    narrative = ""
    if predictions and mistral.is_available():
        winner_drv = next(
            (d for d in enriched if d.get("predicted_position") == 1), {}
        )
        try:
            narrative = mistral.explain_prediction(
                prediction={
                    "winner": winner_drv.get("driver_name", "N/A"),
                    "win_prob": predictions[0]["win_prob"],
                    "top3": enriched[:3],
                    "shap_top3": predictions[0].get("shap_top3", ""),
                },
                circuit=race_info["name"],
                conditions=weather,
                session_type="course",
            )
        except Exception as e:
            narrative = f"⚠️ Analyse IA indisponible : {e}"
    elif not mistral.is_available():
        narrative = mistral.get_status()

    return table, fig, narrative, context


def run_post_race_analysis(year: int, round_num: int) -> tuple:
    """
    Analyse post-GP : prédictions vs résultats réels.
    Retourne (comparison_df, narrative_text)
    """
    data = load_race_data(year, round_num)
    if data is None:
        return pd.DataFrame(), "❌ Données introuvables."

    # Résultats réels
    actual_results = get_race_results(year, round_num)
    if not actual_results:
        return pd.DataFrame(), "❌ Résultats de course non disponibles pour ce GP."

    # Prédictions
    _, _, _, _ = run_race_prediction(year, round_num)

    # Tableau comparatif
    actual_map = {r["driver_id"]: r for r in actual_results}
    rows = []
    for i, r in enumerate(actual_results[:10]):
        rows.append({
            "Pos Réelle": _medal(r["position"]),
            "Pilote": r["driver_name"],
            "Équipe": r["team"],
            "Status": r.get("status", "Finished"),
        })
    comparison_df = pd.DataFrame(rows)

    # Analyse Mistral
    narrative = ""
    race_info = data["race_info"]
    if mistral.is_available() and actual_results:
        pred_table, _, _, _ = run_race_prediction(year, round_num)
        # Reconstruction simple des prédictions pour Mistral
        predicted = {
            "winner": pred_table.iloc[0]["Pilote"] if not pred_table.empty else "N/A",
            "top3": [
                {"driver_name": pred_table.iloc[i]["Pilote"],
                 "predicted_position": i + 1}
                for i in range(min(3, len(pred_table)))
            ],
        }
        actual = {
            "winner": actual_results[0]["driver_name"] if actual_results else "N/A",
            "top3": [
                {"driver_name": r["driver_name"], "position": r["position"]}
                for r in actual_results[:3]
            ],
        }
        try:
            narrative = mistral.post_race_analysis(predicted, actual, race_info["name"])
        except Exception as e:
            narrative = f"⚠️ Analyse IA indisponible : {e}"

    return comparison_df, narrative


# ─── Live Status ──────────────────────────────────────────────────────────────

def get_live_status() -> str:
    """Affiche l'état de la session live via OpenF1."""
    try:
        session = get_latest_session()
        if not session:
            return "🔴 Aucune session F1 en cours"
        name = session.get("session_name", "")
        gp = session.get("location", "")
        date = session.get("date_start", "")[:10]
        status = "🟢 En cours" if session.get("date_end", "") == "" else "✅ Terminée"
        return f"{status} — {gp} | {name} | {date}"
    except Exception as e:
        return f"⚠️ OpenF1 indisponible : {e}"


def get_live_weather_display() -> str:
    """Météo live depuis OpenF1."""
    try:
        weather_data = get_live_weather()
        if not weather_data:
            return "Données météo live indisponibles"
        w = weather_data[0] if isinstance(weather_data, list) else weather_data
        return (
            f"🌡️ Air {w.get('air_temperature', '?')}°C | "
            f"Piste {w.get('track_temperature', '?')}°C | "
            f"Humidité {w.get('humidity', '?')}% | "
            f"Vent {w.get('wind_speed', '?')} m/s | "
            f"{'🌧️ Pluie' if w.get('rainfall', False) else '☀️ Sec'}"
        )
    except Exception as e:
        return f"⚠️ Météo live : {e}"


# ─── Construction de l'interface Gradio ──────────────────────────────────────

def build_interface() -> gr.Blocks:
    # Calendrier pour le sélecteur
    try:
        schedule = _get_or_fetch_schedule(CURRENT_YEAR)
    except Exception:
        schedule = []

    race_choices = [
        (f"R{r['round']} — {r['name']} ({r['date']})", r["round"])
        for r in schedule
    ]
    if not race_choices:
        race_choices = [(f"R{i} — GP {i}", i) for i in range(1, 25)]

    # Déterminer le GP actuel (prochain non passé)
    today = datetime.now().strftime("%Y-%m-%d")
    default_round = 1
    for r in schedule:
        if r["date"] >= today:
            default_round = r["round"]
            break
        default_round = r["round"]  # Dernier GP si tous passés

    default_idx = next(
        (i for i, (_, rn) in enumerate(race_choices) if rn == default_round), 0
    )

    with gr.Blocks(
        css=CUSTOM_CSS,
        title="🏎️ F1 Prévision v0.5",
        theme=gr.themes.Base(
            primary_hue="red",
            secondary_hue="slate",
            neutral_hue="slate",
        ),
    ) as demo:

        # ── En-tête ──────────────────────────────────────────────────────────
        gr.HTML("""
        <div class="f1-header">
            <h1>🏎️ F1 Prévision — IA + ML</h1>
            <p>Prédictions qualifications & course | XGBoost + SHAP + Mistral API</p>
            <p style="font-size:0.85em; opacity:0.7;">
                Données : FastF1 · Jolpica/Ergast · OpenF1 · Open-Meteo
                | LLM : Mistral (gratuit, 🇫🇷 RGPD)
            </p>
        </div>
        """)

        # ── Sélecteur global ─────────────────────────────────────────────────
        with gr.Row():
            year_input = gr.Slider(
                minimum=2020, maximum=CURRENT_YEAR,
                value=CURRENT_YEAR, step=1, label="Saison",
            )
            round_input = gr.Dropdown(
                choices=race_choices,
                value=race_choices[default_idx][1] if race_choices else 1,
                label="Grand Prix",
                interactive=True,
            )

        # ── Status live ──────────────────────────────────────────────────────
        with gr.Row():
            live_status = gr.Textbox(
                value=get_live_status(),
                label="Session live (OpenF1)",
                interactive=False, max_lines=1,
            )
            live_weather_txt = gr.Textbox(
                value=get_live_weather_display(),
                label="Météo live",
                interactive=False, max_lines=1,
            )
        refresh_live_btn = gr.Button("🔄 Rafraîchir données live", size="sm")

        # ── Onglets ───────────────────────────────────────────────────────────
        with gr.Tabs():

            # ── Tab 1 : Qualifications ────────────────────────────────────────
            with gr.Tab("⏱️ Qualifications"):
                gr.Markdown("### Prédiction de la grille de départ")
                quali_predict_btn = gr.Button(
                    "🔮 Lancer la prédiction qualifications",
                    variant="primary",
                )
                with gr.Row():
                    with gr.Column(scale=2):
                        quali_table = gr.DataFrame(
                            label="Ordre de qualification prédit",
                            interactive=False,
                        )
                    with gr.Column(scale=3):
                        quali_chart = gr.Plot(label="Probabilités")
                quali_narrative = gr.Textbox(
                    label="🤖 Analyse Mistral AI",
                    lines=5, interactive=False,
                    placeholder="Cliquez sur 'Lancer la prédiction' pour obtenir l'analyse IA...",
                )
                quali_context = gr.State("")  # Contexte pour le chat

            # ── Tab 2 : Course ────────────────────────────────────────────────
            with gr.Tab("🏁 Course"):
                gr.Markdown("### Prédiction du résultat de course")
                with gr.Row():
                    use_actual_quali_cb = gr.Checkbox(
                        value=True,
                        label="Utiliser les résultats qualifications réels (si disponibles)",
                    )
                race_predict_btn = gr.Button(
                    "🔮 Lancer la prédiction course",
                    variant="primary",
                )
                with gr.Row():
                    with gr.Column(scale=2):
                        race_table = gr.DataFrame(
                            label="Classement prédit",
                            interactive=False,
                        )
                    with gr.Column(scale=3):
                        race_chart = gr.Plot(label="Probabilités")
                race_narrative = gr.Textbox(
                    label="🤖 Analyse Mistral AI",
                    lines=5, interactive=False,
                    placeholder="Cliquez sur 'Lancer la prédiction' pour l'analyse IA...",
                )
                race_context = gr.State("")

            # ── Tab 3 : Post-GP ───────────────────────────────────────────────
            with gr.Tab("📊 Analyse Post-GP"):
                gr.Markdown(
                    "### Comparaison prédictions vs résultats réels\n"
                    "*(Disponible après la course)*"
                )
                post_race_btn = gr.Button(
                    "📈 Analyser les résultats",
                    variant="primary",
                )
                post_table = gr.DataFrame(
                    label="Résultats réels",
                    interactive=False,
                )
                post_narrative = gr.Textbox(
                    label="🤖 Analyse Mistral (prédit vs réel)",
                    lines=8, interactive=False,
                )

            # ── Tab 4 : Chat ──────────────────────────────────────────────────
            with gr.Tab("💬 Chat F1"):
                gr.Markdown(
                    "### Questions libres sur les prédictions\n"
                    "Posez des questions sur les prédictions en cours. "
                    "L'IA répond sur la base des données chargées."
                )
                mistral_status = gr.Textbox(
                    value=mistral.get_status(),
                    label="Statut Mistral",
                    interactive=False, max_lines=1,
                )
                chatbot = gr.Chatbot(
                    label="Chat F1", height=350,
                    avatar_images=(None, "🤖"),
                )
                with gr.Row():
                    chat_input = gr.Textbox(
                        placeholder="Ex: Pourquoi Verstappen est favori ? "
                                    "Quel impact si il pleut ? ...",
                        label="Question",
                        scale=4,
                    )
                    chat_send_btn = gr.Button("Envoyer", variant="primary", scale=1)
                chat_clear_btn = gr.Button("🗑️ Effacer la conversation", size="sm")
                chat_context = gr.State("")  # Contexte injecté depuis les autres tabs

            # ── Tab 5 : Infos ─────────────────────────────────────────────────
            with gr.Tab("ℹ️ À propos"):
                gr.Markdown(f"""
## F1 Prévision v0.5

Application personnelle de prédiction des résultats F1.

### Sources de données
| Source | Données | Coût |
|--------|---------|------|
| **FastF1** | Télémétrie, pneus, pit stops | Gratuit |
| **Jolpica/Ergast** | Résultats depuis 1950 | Gratuit |
| **OpenF1** | Données live 2023+ | Gratuit |
| **Open-Meteo** | Météo historique + forecast | Gratuit |
| **Mistral API** | Analyses narratives LLM | Gratuit (1B tokens/mois) |

### Modèle ML
- **Algorithme** : XGBoost (Gradient Boosting)
- **Interprétabilité** : SHAP (feature importance)
- **Mode dégradé** : Scoring statistique si modèle non entraîné

### Métriques cibles
| Métrique | Cible |
|----------|-------|
| Top-1 Accuracy | ≥ 30% |
| Top-3 Accuracy | ≥ 60% |
| Spearman Rank Corr. | ≥ 0.60 |
| MAE Position | ≤ 3.5 |

### Hébergement
🤗 **Hugging Face Spaces** — Gratuit | 2 vCPU | 16 GB RAM

### LLM — Mistral API
- Tier gratuit : 1 milliard tokens/mois, 2 req/min
- Serveurs France 🇫🇷 — RGPD natif
- Confidentialité : données non utilisées pour l'entraînement

---
*Statut modèle : {"✅ XGBoost chargé" if get_predictor().using_ml else "⚠️ Mode baseline statistique (entraîner le modèle localement)"}*
                """)

        # ─── Événements ───────────────────────────────────────────────────────

        def update_schedule_on_year(year):
            """Met à jour le sélecteur de GP quand l'année change."""
            try:
                sched = _get_or_fetch_schedule(int(year))
                choices = [
                    (f"R{r['round']} — {r['name']} ({r['date']})", r["round"])
                    for r in sched
                ]
                return gr.Dropdown(choices=choices, value=choices[0][1] if choices else 1)
            except Exception:
                return gr.Dropdown(choices=[], value=1)

        year_input.change(
            fn=update_schedule_on_year,
            inputs=[year_input],
            outputs=[round_input],
        )

        def on_quali_predict(year, round_num):
            table, fig, narrative, context = run_qualifying_prediction(int(year), int(round_num))
            return table, fig, narrative, context

        quali_predict_btn.click(
            fn=on_quali_predict,
            inputs=[year_input, round_input],
            outputs=[quali_table, quali_chart, quali_narrative, quali_context],
            api_name="predict_qualifying",
        )

        def on_race_predict(year, round_num, use_actual):
            table, fig, narrative, context = run_race_prediction(
                int(year), int(round_num), use_actual
            )
            return table, fig, narrative, context

        race_predict_btn.click(
            fn=on_race_predict,
            inputs=[year_input, round_input, use_actual_quali_cb],
            outputs=[race_table, race_chart, race_narrative, race_context],
            api_name="predict_race",
        )

        def on_post_race(year, round_num):
            table, narrative = run_post_race_analysis(int(year), int(round_num))
            return table, narrative

        post_race_btn.click(
            fn=on_post_race,
            inputs=[year_input, round_input],
            outputs=[post_table, post_narrative],
        )

        # ── Chat ──────────────────────────────────────────────────────────────

        def on_chat(question, history, quali_ctx, race_ctx):
            if not question.strip():
                return history, ""
            # Utiliser le contexte disponible (qualifs ou course, selon lequel est rempli)
            context = race_ctx or quali_ctx or "Aucune prédiction chargée."
            if not mistral.is_available():
                reply = "⚠️ MISTRAL_API_KEY non configurée. Définissez ce secret dans HF Spaces."
            else:
                # Convertir l'historique Gradio → format Mistral
                mistral_history = []
                for user_msg, bot_msg in (history or []):
                    mistral_history.append({"role": "user", "content": user_msg})
                    if bot_msg:
                        mistral_history.append({"role": "assistant", "content": bot_msg})
                try:
                    reply = mistral.chat_about_prediction(question, context, mistral_history)
                except Exception as e:
                    reply = f"⚠️ Erreur Mistral : {e}"
            new_history = (history or []) + [(question, reply)]
            return new_history, ""

        chat_send_btn.click(
            fn=on_chat,
            inputs=[chat_input, chatbot, quali_context, race_context],
            outputs=[chatbot, chat_input],
        )
        chat_input.submit(
            fn=on_chat,
            inputs=[chat_input, chatbot, quali_context, race_context],
            outputs=[chatbot, chat_input],
        )
        chat_clear_btn.click(fn=lambda: ([], ""), outputs=[chatbot, chat_input])

        # ── Refresh live ──────────────────────────────────────────────────────

        def on_refresh_live():
            return get_live_status(), get_live_weather_display()

        refresh_live_btn.click(
            fn=on_refresh_live,
            outputs=[live_status, live_weather_txt],
        )

    return demo


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Démarrage F1 Prévision v0.5")
    logger.info(f"Modèle ML : {'XGBoost' if get_predictor().using_ml else 'Baseline statistique'}")
    logger.info(f"Mistral : {mistral.get_status()}")

    demo = build_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # HF Spaces gère l'exposition
        show_error=True,
    )
