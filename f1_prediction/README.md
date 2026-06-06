---
title: F1 Prévision
emoji: 🏎️
colorFrom: red
colorTo: gray
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
tags:
  - f1
  - formula1
  - xgboost
  - prediction
  - sports
  - mistral
---

# 🏎️ F1 Prévision v0.5

Application personnelle de **prévision des résultats F1** (qualifications + Grand Prix) en temps réel.

**Coût infrastructure : 0 €** — 100% gratuit, hébergé sur HF Spaces.

## Stack technique

| Couche | Technologie |
|--------|-------------|
| Données F1 | FastF1, Jolpica/Ergast, OpenF1 API |
| Météo | Open-Meteo (sans clé API) |
| ML | XGBoost + SHAP |
| LLM | Mistral API (tier gratuit, 🇫🇷 RGPD) |
| Interface | Gradio |
| Hébergement | Hugging Face Spaces (gratuit) |

## Fonctionnalités

- **⏱️ Qualifications** — Prédiction de la grille de départ avec probabilités
- **🏁 Course** — Prédiction du classement final
- **📊 Post-GP** — Comparaison prédictions vs résultats réels
- **💬 Chat** — Questions libres sur les prédictions (Mistral AI)
- **🌐 Live** — Données temps réel via OpenF1

## Variables du modèle

### Qualifications
- Historique pilote sur ce circuit (meilleur temps, taux poles)
- Forme récente (3 dernières courses)
- Rythme équipe / type de circuit
- Conditions météo (Open-Meteo)

### Course
- **Position de départ** (feature critique)
- Taux conversion pole → victoire sur ce circuit
- Évolution météo en course
- Vitesse pit stop équipe
- DNF rate pilote/circuit

## Configuration

### Secrets HF Spaces requis

```
MISTRAL_API_KEY = votre_clé_depuis_console.mistral.ai
```

L'application fonctionne sans cette clé (analyses IA désactivées).

## Entraînement du modèle (local)

```bash
cd f1_prediction
pip install -r requirements.txt

# Entraîner sur les saisons 2018-2025
python -m models.train --years 2018 2019 2020 2021 2022 2023 2024 2025 --mode both

# Les .pkl sont générés dans models/artifacts/
# Les committer dans ce repo Git pour les déployer sur HF Spaces
git add models/artifacts/
git commit -m "chore: modèles entraînés 2018-2025"
git push
```

## Métriques cibles

| Métrique | Cible | Baseline |
|----------|-------|---------|
| Top-1 Accuracy (course) | ≥ 30% | ~15% |
| Top-3 Accuracy (course) | ≥ 60% | ~30% |
| Pole Accuracy | ≥ 35% | ~15% |
| Spearman Rank | ≥ 0.60 | ~0.10 |
| MAE Position | ≤ 3.5 | ~6-7 |

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              F1 PRÉVISION v0.5                          │
├───────────────┬──────────────────┬──────────────────────┤
│  DATA LAYER   │   ML LAYER       │   INTERFACE (Gradio) │
│               │                  │                      │
│ FastF1        │ Feature Eng.     │ Onglet Qualifications│
│ Ergast/Jolpi  │ (pandas)         │ Onglet Course        │
│ OpenF1 (live) │                  │ Probabilités         │
│ Open-Meteo    │ XGBoost          │ Analyse narrative    │
│               │ (.pkl dans Git)  │  └─ Mistral (FREE)   │
│               │                  │ Chat prédictions     │
│               │ SHAP             │  └─ Mistral (FREE)   │
│               │ (→ prompts)      │ Post-GP analysis     │
└───────────────┴──────────────────┴──────────────────────┘
  HF Spaces gratuit | Stateless | MISTRAL_API_KEY = Secret
```

## Sources & Licences

- [FastF1](https://docs.fastf1.dev/) — MIT
- [Jolpica-F1/Ergast](https://jolpi.ca/ergast/) — CC BY 4.0
- [OpenF1](https://openf1.org/) — Open source
- [Open-Meteo](https://open-meteo.com/) — CC BY 4.0
- [Mistral AI](https://console.mistral.ai/) — API gratuite, données non réutilisées

---
*Usage personnel, non-commercial | v0.5 | Juin 2026*
