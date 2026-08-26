# Conformité à la documentation CielScore

Ce document recense, point par point, ce que l'application Android reprend du
document *CielScore — Documentation fonctionnelle et technique* (version UX
v0.6.4), et **tout ce qui s'en écarte**. Rien n'est laissé implicite : chaque
écart est soit une décision validée, soit une zone que le document ne couvre pas.

---

## 1. Ce qui est repris à l'identique

| Section | Élément | Où c'est implémenté |
|---|---|---|
| 2.1 → 2.8 | Les 8 onglets et leur accès public / connecté | `ui/screens/`, `MainActivity.kt` |
| 3.2 | Les trois vues du panneau latéral | `ui/components/ObjectSheet.kt` |
| 3.3 | Règles d'affichage de la carte (alt > 2°, > 0°, > 5°, Lune > 2°) | `ui/components/SkyMapView.kt` |
| 3.4 | Projection azimutale équidistante | `astro/SkyProjection.kt` |
| 3.5 | Conversion équatorial → horizontal | `astro/AstroMath.kt` |
| 4.1 | Filtres éliminatoires RG-F-01 à RG-F-04 | `scoring/ScoringEngine.kt` |
| 4.2 | Filtrage dynamique et interpolation crépusculaire | `scoring/Formulas.darknessLimits` |
| 4.3 | RG-P-01 à RG-P-05 | `scoring/ScoringEngine.scoreSolarSystem` |
| 4.4 | RG-L-01 à RG-L-04 | `scoring/ScoringEngine.moonScore` |
| 4.5 | RG-I-01 à RG-I-05 | `scoring/Formulas.kt`, `model/SessionParams` |
| 4.6 | Durées de vie et clés de cache | `data/cache/TtlCache.kt` |
| 5.1 → 5.9 | Toutes les formules | `scoring/Formulas.kt` |
| 6.1 / 6.3 / 6.4 | Les trois formules de score et leurs pondérations | `scoring/ScoringEngine.kt` |
| 6.2 | Barème du seeing déduit du vent | `scoring/Formulas.seeingIndex` |
| 6.5 | Lecture du score | `scoring/Formulas.scoreInterpretation` |
| 7.1 → 7.5 | Phases, score de nuit, couleurs, nuits d'été | `astro/Twilight.kt` |
| 8.1 | Messier 110, Caldwell 109, NGC/IC 13 308, 6 corps | `assets/`, `tools/build_catalogs.py` |
| 8.2 | Les 8 interfaces externes | `data/net/` |
| 8.4 | Stratégie de repli service par service | `data/net/`, `model/SkyConditions` |
| 9.1 | PBKDF2-SHA256, 260 000 itérations, sel 16 octets, comparaison à temps constant | `data/auth/PasswordHasher.kt` |
| 9.2 | Tables `users` et `observations`, index | `data/db/Entities.kt` |
| 9.4 | Format et niveaux de journalisation | `util/Log.kt` |
| 10.2 | L'exemple complet M57 est rejoué en test | `DocumentationConformanceTest.kt` |

Les valeurs chiffrées du document (tableaux 5.1, 5.2, 5.5, 5.6, 6.2, 7.2, 10.2)
sont vérifiées automatiquement par **26 tests unitaires**. Le score final de
l'exemple 10.2 est reproduit à 78,4 / 100, pour « environ 78 / 100 » annoncé.

---

## 2. Décisions prises avec vous

| Sujet | Décision retenue |
|---|---|
| Architecture | Application Android native (Kotlin + Jetpack Compose), calculs embarqués. Aucune dépendance au code Python. |
| Périmètre | Les 8 onglets **et** la carte du ciel interactive. |
| Lieu et Bortle | Géolocalisation + recherche de commune via l'API Géo de l'État, l'indice de Bortle étant estimé puis ajustable. |
| Comptes | Compte et carnet 100 % locaux (Room), clés d'API saisies dans le Profil. |
| Smart télescopes | Seuls les **7 modèles détaillés** au tableau 5.9 sont intégrés. |
| Carte du ciel | Figures de constellations **complètes** (358 étoiles, 239 segments) plutôt que les 174 / 113 du §3.3. |
| Objets sans dimensions | Consultables dans l'Explorer, exclus du Top du Dashboard. |
| Formule du §5.9 | La **formule** fait foi, pas le tableau qui l'accompagne (voir 3.1). |
| Seuil des planètes brillantes | **−3°** de hauteur du Soleil, valeur de RG-P-03 (voir 3.1). |
| Critères non définis | Définitions retenues pour « Fenêtre », F/D et champ (voir 3.2). |

---

## 3. Écarts par rapport au document

Les trois points ci-dessous ont été **soumis et arbitrés** : les choix décrits
sont ceux retenus dans l'application, et non des options ouvertes.

### 3.1 Incohérences internes au document

**Le tableau du §5.9 ne découle pas de sa propre formule.**
La formule énoncée est :

```
mag_limite = 2.1 + 5×log10(D_mm) + 2.5×log10(T_sec/60) − (Bortle−1)×0.55
```

Appliquée au Seestar S50 (50 mm, 60 min, Bortle 7), elle donne **11,7** alors que
le tableau annonce **12,6**. L'écart est d'environ **+0,9 magnitude** sur tous les
modèles (+1,3 pour le S30). Le terme constant qui reproduirait le tableau serait
≈ 2,95 au lieu de 2,1.

→ **Arbitrage retenu : l'application applique la formule**, qui est l'élément
normatif. Les magnitudes limites affichées sont donc inférieures d'environ 0,9 à
celles du tableau du §5.9, qui est à considérer comme illustratif.

Pour mémoire, si ce choix devait un jour être revu : porter la constante `2.1` à
≈ 2,95 dans `Formulas.smartTelescopeLimitingMagnitude` reproduirait le tableau.

**Le §2.5 annonce 12 smart télescopes, le §5.9 n'en détaille que 7.**
→ Les 7 documentés sont intégrés (décision validée).

**Le §4.2 et la règle RG-P-03 divergent sur le seuil des planètes brillantes.**
Le tableau du §4.2 place Vénus et Jupiter dès le crépuscule civil (Soleil à 0°),
RG-P-03 précise « (−3°) ».
→ **Arbitrage retenu : −3°**, la règle nommée étant la plus spécifique.

### 3.2 Éléments que le document ne définit pas

**Le critère « Fenêtre » (§6.1, 15 %).**
Le document donne `min(durée_min / 240, 1) × 100` sans définir `durée_min`.
→ **Arbitrage retenu : le temps passé au-dessus de 30° (seuil optimal du §2.2)
pendant que le Soleil est sous l'horizon, évalué sur les 10 heures suivant
l'heure de session** — la même fenêtre que la courbe d'altitude du §2.2.
Voir `ScoringEngine.observationWindowMinutes`.

**Les critères F/D et champ du score smart télescope (§6.4, 5 % chacun).**
Le document énonce les principes (« bonus F/D court », « grand champ favorable
aux nébuleuses étendues ») sans formule.
→ **Arbitrage retenu :**
- F/D : `clip((8 − F/D) / 6, 0, 1) × 100`, soit 100 à F/2 et 0 à partir de F/8 ;
- champ : maximum lorsque l'objet occupe la moitié du champ, nul s'il est
  ponctuel ou plus grand que le champ.
Voir `ScoringEngine.focalRatioScore` et `fieldMatchScore`.

**Les caractéristiques des smart télescopes autres que le diamètre.**
Le §2.5 demande d'afficher « ouverture, focale, capteur, champ » mais le §5.9 ne
fournit que le diamètre.
→ Focale, dimensions de capteur et taille de pixel proviennent des
**spécifications constructeur**, pas du document. Elles sont regroupées dans une
seule table, `SmartTelescope.CATALOG`, pour être corrigées d'un seul geste.

**Le fichier `communes_bortle.csv` (34 870 lignes, §8.1) n'est pas fourni.**
→ Remplacé par l'API Géo de l'État (gratuite, sans clé) pour le nom, le
département et les coordonnées, avec un indice de Bortle **estimé d'après la
population** puis ajustable par l'utilisateur. La valeur par défaut hors commune
reste Bortle 5 (RG-INFO-01). Table d'estimation dans `GeoApi.estimateBortle`.

**Le fond d'étoiles de la carte du ciel.**
Le §3.3 annonce 174 étoiles et 113 segments sans fournir les données.
→ Table J2000 constituée pour l'application : **358 étoiles, 31 figures,
239 segments** (décision validée : figures complètes). Régénérable par
`tools/build_stars.py`.

### 3.3 Adaptations liées au support mobile

| Document | Application Android |
|---|---|
| §1.2 Gradio 6.18 / Python 3.13 | Kotlin + Jetpack Compose, `minSdk` 26 |
| §1.2 Astropy + PyEphem + NumPy | Moteur d'éphémérides Kotlin embarqué (Meeus ch. 25, 45, 47 ; éléments képlériens JPL 1800–2050) |
| §1.2 auth.py + SQLite | Room, base locale `cielscore.db` |
| §9.3 Restauration de la base depuis un dépôt distant | Sauvegarde Android (`backup_rules.xml`) ; aucune base distante |
| §1.2 Hébergement Hugging Face Spaces | APK construit par GitHub Actions |
| §2.5 Diagnostic par focale d'oculaire | Table de 8 focales usuelles, champ apparent 52° |

Le module Caldwell mérite une note : construit depuis OpenNGC par
correspondance C# → NGC/IC comme le prévoit le §8.1, il ne donne que **106**
objets, car C9 (Sh2-155), C41 (Hyades) et C99 (Coalsack) n'ont pas d'entrée
NGC/IC. Les trois sont ajoutés explicitement pour atteindre les **109** annoncés.

---

## 4. Précision des éphémérides

Le moteur remplace Astropy et PyEphem. Précision constatée sur les cas de
référence testés :

| Grandeur | Écart mesuré |
|---|---|
| Déclinaison du Soleil (solstice, équinoxe) | < 0,01° |
| Fraction illuminée de la Lune (syzygies) | < 0,1 % |
| Position de la Lune | de l'ordre de la minute d'arc |
| Positions planétaires | quelques minutes d'arc (éléments moyens JPL, 1800–2050) |
| Durée de la nuit astronomique | < 1 minute |

Largement suffisant pour un calcul d'altitude, de fenêtre d'observation et de
distance angulaire à la Lune.
