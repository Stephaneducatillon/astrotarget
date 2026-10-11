# 🪐 CielScore Satellites — version web (JavaScript)

Site **statique, gratuit, sans serveur et sans compte**, conforme à l'architecture
recommandée par le cahier des charges : les positions sont précalculées une fois
(tables de polynômes de Tchebychev), puis le navigateur fait tout le reste —
calcul, schéma, phénomènes, exports, photo — **y compris hors ligne au télescope**
(application web installable).

```
Moteur CielScore (PyEphem) ──► tools/build_tables.py ──► data/*.bin ──► navigateur
   ou fichiers JPL (.bsp)        1 à 2 fois par an       ~23 Mo        (JS + SVG)
```

## Lancer en local

```bash
cd cielscore_satellites/web
python3 -m http.server 8000        # puis http://localhost:8000
```

Aucune installation ni compilation : HTML, CSS et modules JavaScript natifs.
(Ouvrir `index.html` directement depuis le disque ne fonctionne pas : les navigateurs
bloquent le chargement des modules en `file://`.)

## Publier (Cloudflare Pages, recommandé)

Cloudflare Pages → *Create a project* → dépôt GitHub → **Build command** : *(vide)*,
**Build output directory** : `cielscore_satellites/web`. Le plus gros fichier fait
moins de 500 Ko (limite 25 Mio) et le site compte ~120 fichiers (limite 20 000).
GitHub Pages fonctionne aussi (dossier publié : `cielscore_satellites/web`).

## Fonctions

- **Recherche** : planète, date et heure (UT ou locale, à la seconde), « Maintenant »,
  ±10 min / ±1 h / ±1 j, fuseau horaire, lieu facultatif (Calern C2PU, OHP, Pic du Midi,
  Paris, Cantin, coordonnées libres).
- **Tableau** des 17 grands satellites trié par distance : AD/Déc J2000 et du jour,
  hauteur/azimut, écarts E-O / N-S (″ et rayons), distance, angle de position, état
  (visible, devant le disque, derrière le disque, dans l'ombre), magnitude approchée.
- **Schéma SVG** à l'échelle : disque aplati et orienté, anneaux de Saturne (moitié
  avant devant le disque), ombres sur le disque, nom et coordonnées de chaque
  satellite, 4 orientations, cadre du champ de l'instrument, curseur de temps ±12 h.
- **Infos planète** : hauteur, lever / passage / coucher, diamètre, magnitude,
  inclinaison des anneaux, méridiens centraux I et II de Jupiter.
- **Phénomènes de la nuit** à la seconde : passages, occultations, éclipses, ombres,
  marqués « observable » (critères réglables).
- **Exports** : CSV des coordonnées et des phénomènes, PNG du schéma, PDF par
  l'impression du navigateur (mise en page dédiée).
- **Photo** (FITS, TIFF, PNG, JPEG), traitée dans le navigateur : heure lue dans
  DATE-OBS ou EXIF, calage en deux clics ou par la solution WCS du FITS, annotation,
  écart au point lumineux le plus proche, export PNG et CSV.
- **Mode nuit rouge**, mise en page pour téléphone, fonctionnement hors ligne.

## Organisation

| Fichier | Rôle |
|---|---|
| `index.html`, `style.css`, `app.js` | interface |
| `lib/tables.js` | lecture des tables binaires, évaluation de Tchebychev (Clenshaw) |
| `lib/engine.js` | positions, coordonnées, géométrie du disque, infos planète |
| `lib/phenomena.js` | passages, occultations, éclipses, ombres (balayage + bissection) |
| `lib/render.js` | schéma SVG |
| `lib/photo.js` | FITS, WCS TAN, EXIF, calage, centroïde |
| `lib/exports.js`, `lib/time.js`, `lib/data.js` | exports, fuseaux horaires, préréglages |
| `vendor/astronomy.js` | [Astronomy Engine](https://github.com/cosinekitty/astronomy) 2.1.19 (MIT) : Soleil, horizon, lever/coucher |
| `sw.js`, `manifest.webmanifest` | application installable et hors ligne |
| `data/<planète>/<année>.bin` | tables 2025 → 2060 ; `meta.json` décrit le format |
| `tools/build_tables.py` | fabrication des tables (PyEphem ou fichiers JPL) |

## Tables

Par planète et par année : AD/Déc astrométriques J2000 et apparentes, rayon apparent,
méridiens centraux de Jupiter (float64, tranches de 4 jours) ; pour chaque satellite,
x (est), y (sud), z (vers la Terre) en rayons planétaires (float32, tranches d'un jour,
degré 4 à 11). Erreur d'ajustement : **< 0,0002 rayon** (< 0,004″ pour Jupiter),
vérifiée à chaque génération et par les tests.

```bash
python3 tools/build_tables.py                          # 2025 → 2060 depuis PyEphem (~1 min)
python3 tools/build_tables.py --source jpl --kernels ~/naif   # depuis les fichiers SPICE du JPL
```

### Couverture actuelle (source PyEphem)

Les tables internes de satellites de PyEphem **s'arrêtent début 2040** :

| Période | Jupiter, Saturne | Uranus |
|---|---|---|
| 2025 → 2039 | ✅ | ✅ |
| 2040 | position non disponible | position non disponible |
| 2041 → 2060 | précision réduite (théorie de secours, écarts de 3 à 8″ avec la théorie L1.2 pour les galiléens) | position non disponible |

Pour couvrir 2025–2060 à la précision visée (< 0,1″), il faut fabriquer les tables
depuis les fichiers du JPL (`--source jpl`) : télécharger `de440s.bsp`, `jup365.bsp`,
`sat441.bsp`, `ura184_part-1/2/3.bsp` depuis le service NAIF, les mettre dans un
dossier, et lancer la commande ci-dessus (Skyfield nécessaire : `pip install skyfield`).
Le format des tables ne change pas, le site non plus. Cette source a été vérifiée sur la
géométrie des planètes (de421) ; la partie satellites reste à valider avec les vrais
fichiers et `scripts/validate_horizons.py`.

## Tests

```bash
npm test                         # 16 tests (Node ≥ 20, aucune dépendance)
python3 tools/make_fixtures.py   # régénère les données de référence des tests
```

Les tests vérifient : fidélité des tables à PyEphem, exemples du cahier des charges
(écarts des galiléens, hauteurs, lever de Jupiter, cadrage C2PU, inclinaison des
anneaux), phénomènes d'Io, passages et ombres sur une semaine, couverture 2040,
fuseaux horaires et changement d'heure, orientations, SVG et CSV, lecture FITS et WCS
(comparée à astropy), date EXIF, calage photo sur image synthétique.

## Écarts connus avec la version Python

- **Éclipses** : les deux versions calculent maintenant l'entrée dans l'ombre par la
  géométrie (ellipsoïde de la planète). L'indicateur de PyEphem utilisé auparavant
  entrait dans l'ombre ~8 min trop tard (Io, 11 octobre 2026 : 20:12 au lieu de 20:04 ;
  durée 2 h 10 au lieu des 2 h 17 attendues pour un passage central).
- **Coordonnées du jour** : tables (PyEphem) + parallaxe calculée par Astronomy Engine ;
  PyEphem présente un bruit interne d'environ 0,25″ sur ces coordonnées apparentes.
- **Lever et coucher** : Astronomy Engine (écart de l'ordre de la seconde avec PyEphem).
