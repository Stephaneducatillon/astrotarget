# astrotarget

Planificateur d'observation astronomique.

## CielScore — application Android

Le dossier [`android/`](android/) contient l'application Android native
**CielScore**, portage complet de la documentation fonctionnelle et technique
v0.6.4 : les 8 onglets, la carte du ciel interactive, le moteur de score à huit
critères et l'ensemble des formules et règles de gestion.

- [Documentation de l'application](android/README.md)
- [Conformité au document et écarts assumés](android/CONFORMITE.md)

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
