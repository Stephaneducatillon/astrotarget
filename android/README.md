# CielScore — application Android

Portage natif Android de **CielScore**, planificateur d'observation
astronomique, conforme à la documentation fonctionnelle et technique v0.6.4.

L'application répond à la question du document : *que puis-je observer ce soir,
avec mon instrument, depuis mon jardin ?* Elle croise instrument, lieu, météo et
physique de l'objet pour produire une liste classée et actionnable.

> Les écarts par rapport au document sont recensés dans **[CONFORMITE.md](CONFORMITE.md)**.

---

## Construire l'application

Prérequis : JDK 17 et le SDK Android (API 35). Android Studio Ladybug ou plus
récent installe les deux.

```bash
cd android
./gradlew assembleDebug          # APK de débogage
./gradlew assembleRelease        # APK de release signé
./gradlew testDebugUnitTest      # 36 tests de conformité
./gradlew installRelease         # installation sur un appareil connecté
```

Les APK produits se trouvent dans `app/build/outputs/apk/{debug,release}/`.

La CI GitHub Actions (`.github/workflows/android.yml`) exécute les tests, Lint et
les deux constructions à chaque push, vérifie la signature de l'APK de release,
publie les deux comme artefacts et met à jour la Release `apk-latest`.

### Installer l'application sur un téléphone

Le plus simple, sans compte GitHub ni décompression — lien permanent, toujours
à jour :

**https://github.com/Stephaneducatillon/astrotarget/releases/tag/apk-latest**

Ouvrez cette page depuis le téléphone, téléchargez le fichier `.apk`, puis
ouvrez-le. Android demandera d'autoriser l'installation depuis cette source,
l'application ne venant pas du Play Store.

La Release est reconstruite à chaque push : le lien ne change jamais, et comme
toutes les constructions partagent la même signature, les versions suivantes
s'installent par-dessus sans rien perdre.

L'onglet Actions expose aussi les APK comme artefacts (`cielscore-release-apk`
et `cielscore-debug-apk`), mais ils exigent d'être connecté à GitHub et arrivent
sous forme de `.zip` à décompresser.

### Signature

L'APK de release est signé avec la clé de **test** versionnée dans le dépôt,
`keystore/cielscore-test.jks` :

| | |
|---|---|
| Alias | `cielscore-test` |
| Mot de passe du magasin et de la clé | `cielscore` |
| Empreinte SHA-256 | `87:78:87:7B:AC:6F:8A:72:3C:D9:C4:89:2A:6B:D8:EA:68:22:56:DC:AA:91:41:35:41:D6:02:51:7A:A0:83:01` |

Ses identifiants sont publics, exactement comme ceux du `debug.keystore` fourni
avec le SDK Android. Elle existe pour que l'APK soit installable et, surtout,
**mis à jour d'une version à l'autre sans désinstaller** : toutes les
constructions partagent la même signature.

> Cette clé ne doit jamais servir à publier l'application. N'importe qui peut
> signer un APK avec elle, puisqu'elle est dans le dépôt.

Pour une vraie distribution, fournissez votre propre clé par l'environnement —
elle est alors utilisée à la place, sans jamais être versionnée :

```bash
export RELEASE_STORE_FILE=/chemin/vers/ma-cle.jks
export RELEASE_STORE_PASSWORD=…
export RELEASE_KEY_ALIAS=…
export RELEASE_KEY_PASSWORD=…
./gradlew assembleRelease
```

En intégration continue, les mêmes valeurs se placent dans les secrets du dépôt.

### Tester le moteur sans SDK Android

Le cœur de calcul est du Kotlin pur. Il se teste avec le seul compilateur
Kotlin, téléchargé depuis Maven Central :

```bash
./tools/run_core_tests.sh
```

---

## Ce que fait l'application

### Les onglets

| Onglet | Accès | Fonction |
|---|---|---|
| **Informations** | public | Tableau de bord du jour : meilleure cible, image APOD, Soleil et Lune, nuit astronomique, indice Kp, calendrier sur 60 jours, prochains lancements |
| **Dashboard** | connecté | Choix du lieu parmi 34 869 communes, réglage de l'instrument, calcul du Top 20 des cibles, fiche objet détaillée |
| **Explorer** | connecté | Recherche libre dans les catalogues, filtres par catalogue et par type |
| **Sessions** | connecté | Plan de soirée IA, export PDF, enregistrement d'observation, carnet |
| **Statistiques** | connecté | Compteurs, progression Messier et Caldwell, heatmap sur 12 mois, favoris |
| **Assistant IA** | connecté | Chat avec injection du contexte réel de la session |
| **Profil** | public | Connexion, inscription, récupération, clés d'API, mode nuit |

### La carte du ciel

Trois vues, accessibles depuis la fiche de chaque objet :

1. **Carte du ciel** — planétarium hémisphérique dessiné sur le mobile :
   358 étoiles, 31 constellations, planètes, horizon, points cardinaux, cercles
   d'altitude à 30° et 60°, cible mise en évidence par un halo doré et une ligne
   guide. Projection azimutale équidistante, zoom et déplacement au doigt.
2. **Aladin DSS2** — Aladin Lite v3 du CDS, avec le cercle du champ de votre
   oculaire, quatre relevés au choix et un mode nuit.
3. **Stellarium Web** — ouverture externe sur la position, la date et l'objet.

### Le moteur de score

Huit critères pondérés, dont la somme vaut exactement 100 % :

```
Score = 0.25×alt + 0.15×fenêtre + 0.11×seeing + 0.13×transparence
      + 0.08×bortle + 0.06×lune + 0.15×SB + 0.07×nuit
```

Les planètes et les smart télescopes utilisent leurs formules dédiées. Quatre
filtres éliminatoires écartent en amont les objets non observables : altitude
sous 5°, brillance de surface au-delà de la limite du site, magnitude au-delà de
la limite dynamique, couverture nuageuse supérieure à 90 %.

Tout est calculé sur l'appareil : éphémérides, crépuscules, brillance de surface
et score. La recherche de commune et le rattachement d'une position GPS le sont
aussi, le fichier des 34 869 communes étant embarqué. Seuls la météo, l'image du
jour, l'indice Kp, les lancements, l'imagerie et l'IA nécessitent le réseau.

---

## Organisation du code

```
android/app/src/main/
├── assets/                    catalogues embarqués, générés par tools/
│   ├── messier.csv            110 objets
│   ├── caldwell.csv           109 objets
│   ├── ngcic.csv              13 308 objets (OpenNGC)
│   ├── stars.csv              358 étoiles brillantes
│   ├── constellations.csv     31 figures, 239 segments
│   └── communes_bortle.csv    34 869 communes, Bortle estimé
└── java/com/cielscore/app/
    ├── astro/                 éphémérides, crépuscules, projection, calendrier
    ├── scoring/               formules (§5) et moteur de score (§4 et §6)
    ├── catalog/               modèle d'objet et chargement des catalogues
    ├── model/                 paramètres de session, smart télescopes
    ├── data/
    │   ├── net/               Open-Météo, Kp, APOD, lancements, CDS, Mistral
    │   ├── db/                Room : users, observations, statistiques
    │   ├── auth/              PBKDF2-SHA256, codes de récupération
    │   ├── cache/             cache à durée de vie (§4.6)
    │   └── prefs/             DataStore : session, lieu, clés d'API
    ├── ui/                    Compose : 8 écrans, carte du ciel, fiche objet
    ├── export/                export PDF du plan de soirée
    └── util/                  journalisation (§9.4)
```

### Régénérer les catalogues

Les fichiers d'assets sont générés, jamais édités à la main :

```bash
python3 tools/build_catalogs.py   # depuis NGC.csv (OpenNGC), à la racine du dépôt
python3 tools/build_stars.py      # étoiles et figures de constellations
```

`communes_bortle.csv` fait exception : ce n'est pas un fichier généré mais une
donnée source, livrée telle quelle. Ses invariants sont vérifiés par :

```bash
python3 tools/check_communes.py
```

La CI exécute ces trois scripts et vérifie que les assets committés correspondent
bien à leur source.

---

## Clés d'API

Deux fonctions nécessitent une clé, à saisir dans l'onglet **Profil**. Elles
restent sur l'appareil.

| Service | Usage | Obtention |
|---|---|---|
| NASA APOD | Image du jour | https://api.nasa.gov |
| Mistral AI | Guide objet, plan de soirée, assistant | https://console.mistral.ai |

Sans clé, le reste de l'application fonctionne normalement : seules les
fonctions concernées affichent un message explicite.

Les autres services — Open-Météo, GFZ Potsdam, NOAA SWPC, The Space Devs et
CDS Strasbourg — sont gratuits et ne demandent aucune clé.

---

## Vie privée

Aucun compte distant, aucune télémétrie. Le compte, le carnet d'observation et
les clés d'API restent sur l'appareil. Le mot de passe est haché en
PBKDF2-SHA256 sur 260 000 itérations avec un sel de 16 octets par utilisateur ;
le code de récupération est haché de la même manière et n'est affiché qu'une
seule fois. La position n'est envoyée qu'à Open-Météo, pour la météo : le nom de la commune
et son indice de Bortle sont déterminés sur l'appareil.

---

## Sources de données

| Donnée | Fournisseur |
|---|---|
| Catalogues NGC / IC | OpenNGC |
| Météo horaire | Open-Météo |
| Indice Kp | GFZ Potsdam, secours NOAA SWPC |
| Image du jour | NASA APOD |
| Lancements | The Space Devs |
| Imagerie du ciel | CDS Strasbourg (hips2fits, Aladin Lite) |
| Intelligence artificielle | Mistral AI |

Échelle de pollution lumineuse d'après Bortle J. (2001), *Sky & Telescope* ;
magnitude limite visuelle d'après Schaefer B. (1990) ; éphémérides d'après
Meeus, *Astronomical Algorithms*, et les éléments képlériens approchés du JPL.
