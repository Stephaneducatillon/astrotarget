# astrotarget

Dépôt personnel regroupant plusieurs outils indépendants.

| Dossier | Projet |
|---|---|
| `astrotarget.py` | Planificateur d'observation astronomique (Streamlit) |
| `fits_photometry.py` | Photométrie sur images FITS |
| `f1_prediction/` | Prévision de résultats F1 — XGBoost + Gradio, déployé sur HF Spaces |
| `web/` | **Paddock** — site web F1 : hub week-end, données et pronostics (PWA statique) |

## Paddock (`web/`)

Site web responsive installable, construit à partir des quatre livrables du
projet F1 : dictionnaire de données, règles de pronostic, spécification des
écrans, backlog priorisé.

```bash
cd web && python3 -m http.server 8000
```

Voir [`web/README.md`](web/README.md) pour l'architecture, les sources de
données, les règles métier appliquées et les points d'arbitrage restants.
