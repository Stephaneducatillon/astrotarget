---
title: CielScore Satellites
emoji: 🪐
colorFrom: indigo
colorTo: gray
sdk: streamlit
app_file: app.py
pinned: false
license: mit
---

# 🪐 CielScore Satellites

**Quel satellite se trouve où autour d'une planète**, à la seconde près. Application
gratuite, publique, sans compte, signée CielScore.

On choisit une planète, une date et une heure (UT ou locale) et, en option, un lieu.
L'application renvoie :

1. **le tableau des satellites**, triés par distance à la planète, avec leurs coordonnées
   complètes et leur état (visible, devant le disque, derrière le disque, dans l'ombre) ;
2. **le schéma** : planète au centre (disque aplati, pôle orienté, anneaux de Saturne),
   satellites à l'échelle avec leur nom et leurs coordonnées J2000, cadre du champ de
   l'instrument, quatre orientations (ciel, Newton, renvoi coudé, rotation libre) et un
   curseur qui fait défiler l'heure ;
3. **les infos planète** : hauteur, lever / passage / coucher, diamètre apparent,
   magnitude, inclinaison des anneaux de Saturne, méridien central de Jupiter (I et II) ;
4. **les phénomènes de la nuit**, avec début et fin à la seconde (UT et heure locale) :
   passages devant le disque, occultations, éclipses, ombres sur le disque, marqués
   « observable » si la planète est levée et le Soleil couché ;
5. **les exports** : CSV des coordonnées et des phénomènes, PNG du schéma, PDF
   (schéma + tableaux) ;
6. **l'option photo** : chargement d'une photo FITS, TIFF, PNG ou JPEG, lecture de l'heure
   (DATE-OBS ou EXIF), calage en deux clics (centre de la planète puis un satellite
   reconnu) ou automatique si le FITS contient une solution WCS, annotation de tous les
   satellites et écart avec le point lumineux le plus proche.

Mode nuit en rouge, mise en page lisible sur téléphone.

## Périmètre (V1)

Les 17 grands satellites calculés par PyEphem :

| Planète | Satellites |
|---|---|
| Jupiter | Io, Europe, Ganymède, Callisto |
| Saturne | Mimas, Encelade, Téthys, Dioné, Rhéa, Titan, Hypérion, Japet |
| Uranus | Miranda, Ariel, Umbriel, Titania, Obéron |

Coordonnées fournies pour chaque satellite : AD/Déc J2000, AD/Déc apparentes du jour,
hauteur et azimut (si un lieu est choisi), écarts est-ouest et nord-sud en secondes
d'arc et en rayons planétaires, distance et angle de position, magnitude approchée.

Présélections : Calern (C2PU), OHP, Pic du Midi, Paris, Cantin, ou coordonnées libres ;
instrument C2PU 1,04 m f/12,5 (13 m de focale) avec capteur APS-C ou 1″, ou instrument
personnalisé (focale, capteur, pixel).

## Lancer l'application

```bash
cd cielscore_satellites
pip install -r requirements.txt
streamlit run app.py
```

Lancée en local, une photo chargée ne quitte pas l'ordinateur. Le dossier peut aussi
être publié tel quel comme Space Hugging Face (SDK Streamlit, en-tête ci-dessus).

## Tests et validation

```bash
pytest                                   # 35 tests automatiques
python scripts/validate_horizons.py      # comparaison avec JPL Horizons (accès Internet)
```

Les tests reprennent les exemples calculés du cahier des charges (Cantin, nuit du 11 au
12 octobre 2026 : écarts des galiléens, hauteurs, lever de Jupiter à 00:42 UT, éclipse
d'Io à 20:13, occultation 21:09 → 23:27, Titan et Japet hors du cadre C2PU), vérifient
la cohérence des conventions (satellites alignés sur l'équateur de Jupiter, inclinaison
des anneaux), les passages et ombres sur une semaine, les exports, et le calage photo
sur une image synthétique (rotation, miroir).

`validate_horizons.py` compare, pour les 17 satellites et 50 dates entre 2026 et 2060,
l'écart planète-satellite calculé avec celui de JPL Horizons (cible : < 0,1″).

## Notes techniques

- PyEphem donne pour chaque satellite `x` (est), `y` (sud) et `z` (vers la Terre) en
  rayons équatoriaux, dans le repère du ciel. Les AD/Déc des satellites sont recalculées
  par projection gnomonique depuis ces écarts : PyEphem ajoute l'écart est-ouest à l'AD
  sans le diviser par cos(δ), ce qui décale l'AD de quelques secondes d'arc.
- Occultations et éclipses : indicateurs « visible depuis la Terre » / « éclairé par le
  Soleil » de PyEphem. Passages et ombres : calcul géométrique sur l'ellipsoïde de la
  planète (aplatissement pck00011, pôle IAU), direction du Soleil vue de la planète.
- Heures des phénomènes : balayage à la minute puis bissection à la seconde.
- La précision de PyEphem doit encore être mesurée contre Horizons ; les tables issues
  des fichiers SPICE du JPL (V2) visent < 0,1″.

## Feuille de route

- **V1.1** : Tache Rouge de Jupiter (longitude système II recalée sur JUPOS), import des
  prédictions de phénomènes mutuels de l'IMCCE (campagne PHEMU 2026-2027), coordonnées
  planétocentriques.
- **V2** : fichiers JPL découpés sur 2025-2060 (Skyfield / spiceypy), petits satellites,
  Mars, Neptune, Pluton, filtre par magnitude, application Android.

---
CielScore Satellites — calculs CielScore.
