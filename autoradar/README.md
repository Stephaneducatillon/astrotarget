# AutoRadar — annonces auto d'occasion en Europe (Android)

Moteur de recherche d'annonces de voitures d'occasion, sur le principe d'un agrégateur :
une seule recherche interroge plusieurs sites d'annonces dans **6 pays**, et chaque annonce
s'ouvre sur son site d'origine.

| Pays | Source |
|---|---|
| 🇫🇷 France | AutoScout24.fr |
| 🇧🇪 Belgique | AutoScout24.be |
| 🇳🇱 Pays-Bas | AutoScout24.nl |
| 🇩🇪 Allemagne | AutoScout24.de |
| 🇮🇹 Italie | AutoScout24.it |
| 🇵🇹 Portugal | Standvirtual.com |

## Fonctionnalités
- Recherche par pays, marque, modèle, prix, année, kilométrage, énergie, boîte, mots-clés
- Tri (prix, année, kilométrage) sur l'ensemble des pays, pagination « Afficher plus »
- Nombre de résultats par pays, et indication si un site ne répond pas
- Favoris et recherches enregistrées (stockés sur le téléphone)
- Mode démonstration (annonces fictives marquées « DÉMO ») pour tester sans réseau

## Télécharger l'APK
À chaque modification du dossier `autoradar/`, GitHub Actions compile l'APK
(workflow `.github/workflows/autoradar-apk.yml`) et le publie dans la release
**`autoradar-latest`** du dépôt (onglet *Releases*) — téléchargez `AutoRadar.apk`
depuis le téléphone et installez-le (autoriser les « sources inconnues »).

## Développement
```bash
cd autoradar
npm install
npm test                 # tests des analyseurs de pages
npm run serve            # aperçu navigateur (mode démo automatique, CORS oblige)
npm run android:add && npm run android:sync && npm run android:build   # nécessite le SDK Android
```

## Architecture
- `www/` : application web (HTML/CSS/JS sans framework) embarquée par **Capacitor 8**
- `www/js/sources/*.js` : un module par site (`buildUrl`, `parsePage`, `search`). Pour ajouter
  un site (ex. La Centrale, Marktplaats, mobile.de…), créer un module sur ce modèle et l'ajouter
  à `REAL_SOURCES` dans `www/js/aggregator.js`.
- Dans l'APK, les requêtes passent par le client HTTP natif (pas de restriction CORS).

## Limites
- Les sites sont lus via les données de leurs pages de résultats : si un site change sa
  structure ou bloque les requêtes, son pays s'affiche « indisponible » et le module
  correspondant doit être adapté. Pour une version commerciale, privilégier des flux ou API
  partenaires et vérifier les conditions d'utilisation de chaque site.
- Les modèles sont convertis en « slug » d'URL (ex. `C 220` → `c-220`) : certains noms de
  gamme propres à un site (ex. « Série 3 ») peuvent nécessiter le nom exact du modèle.
