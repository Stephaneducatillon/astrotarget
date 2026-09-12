# astrotarget

Planificateur d'observation astronomique.

## SkyScore — application Android

Le dossier [`android/`](android/) contient l'application Android native
**SkyScore**, portage complet de la documentation fonctionnelle et technique
v0.6.4 : les onglets, la carte du ciel interactive, le moteur de score à huit
critères et l'ensemble des formules et règles de gestion. Le scoring suit les
**règles v2.0 du 09/09/2026**, qui réécrivent la magnitude limite, le score de
nuit et le traitement des objets brillants étendus.

L'application s'appelait *CielScore* jusqu'à la version 0.7.0. Depuis la
**1.0.0**, elle porte le nom **SkyScore** et ne dépend plus d'aucun service
d'intelligence artificielle.

- [Documentation de l'application](android/README.md)
- [Conformité au document et écarts assumés](android/CONFORMITE.md)
- [**Formules et règles de scoring**](docs/SkyScore_Formules_et_Regles.pdf) — document de
  référence en PDF, décrivant le moteur de calcul tel qu'il est codé
  (régénérable par `tools/build_reference_pdf.py`)

```bash
cd android && ./gradlew assembleDebug   # APK de débogage
./tools/run_core_tests.sh               # tests du moteur, sans SDK Android
```

## Autres contenus du dépôt

| Chemin | Contenu |
|---|---|
| `NGC.csv` | Catalogue OpenNGC complet, source des catalogues embarqués |
| `astrotarget.py` | Prototype Streamlit initial |
| `f1_prediction/` | Projet distinct : prévision F1 |
| `tools/` | Génération des catalogues, exécution des tests hors SDK |
