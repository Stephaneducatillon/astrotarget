# Paddock — hub F1 et pronostics

Site web construit à partir des quatre livrables du projet : dictionnaire de
données, règles de pronostic, spécification des écrans, backlog priorisé.

Application web responsive installable (PWA), **sans build et sans back-end** :
c'est du HTML, du CSS et des modules ES servis tels quels. On l'ouvre avec
n'importe quel serveur statique.

```bash
cd web
python3 -m http.server 8000
# puis http://localhost:8000
```

> Ouvrir `index.html` par `file://` ne marche pas : les modules ES et le
> service worker exigent HTTP.

---

## Ce qui est livré

| Écran | Livrable 3 | État |
|---|---|---|
| Hub week-end | Écran 1 | complet |
| Fiche circuit | Écran 2 | complet, hors tracé géographique |
| Fiche pilote | Écran 3 | complet (Forme · Ce circuit · Saison + duel équipier) |
| Comparateur de tours | Écran 4 | tours et secteurs ; courbes de télémétrie en attente du batch FastF1 |
| Saisie du pronostic | Écran 5 | complet, coefficients en direct |
| Débrief post-course | Écran 6 | complet |
| Ligue | Écran 7 | complet, échange par code de partage |

S'y ajoutent un calendrier de saison, les résultats course et qualifications,
un écran de traçabilité des sources et un écran de réglages.

## Architecture

```
web/
├── index.html              coquille minimale
├── manifest.webmanifest    PWA installable
├── sw.js                   hors ligne : coquille seulement, jamais les API
├── css/app.css             feuille unique, rupture à 768 px
├── data/circuits.json      référentiel MAN des 24 circuits
└── js/
    ├── config.js           sources, TTL, couleurs d'écurie, barème
    ├── contexte.js         chaîne calendrier → GP → circuit → météo
    ├── store.js            profil, pronostics, ligues (localStorage)
    ├── data/               couche d'accès : http, cache, jolpica, openf1,
    │                       openmeteo, reference, demo
    ├── model/              baseline.js (proba_baseline) · scoring.js (barème)
    ├── util/               time (UTC → local), format, dom
    └── views/              un module par écran
```

**Aucune vue n'appelle `fetch`.** Les vues consomment les entités normalisées
du livrable 1 (`gp`, `session`, `entry`, `lap`, `stint`…) exposées par
`js/data/`. C'est la parade au risque « changement de format des flux
OpenF1 » : le jour où une source bouge, un seul module bouge.

Le livrable 4 place cette couche côté serveur. Ce site étant déployable en
statique, elle vit dans le front — mais reste isolée pour pouvoir passer
derrière une API sans toucher aux écrans.

## Sources

| Code | Source | Rôle |
|---|---|---|
| `JOL` | [Jolpica-F1](https://jolpi.ca/ergast/) | calendrier, engagés, résultats, classements |
| `OF1` | [OpenF1](https://openf1.org/) | direction de course, tours, relais, météo piste (2023 →) |
| `OM` | [Open-Meteo](https://open-meteo.com/) | prévision horaire sur lat/lon du circuit |
| `MAN` | référentiel local | fiche circuit : longueur, virages, DRS, appui, abrasivité |
| `CALC` | plateforme | forme, delta qualif↔course, fiabilité, `proba_baseline` |

Préséance : `OF1` pour le live, `JOL` pour l'historique et les classements,
`FF1` pour tout ce qui dérive du timing.

Chaque bloc de l'interface porte le badge de sa source et son horodatage de
rafraîchissement. L'écran **Sources** détaille les licences, les durées de
cache et les limites connues.

## Règles métier appliquées

- **Heures.** Stockées en UTC, converties au fuseau du navigateur, jamais
  affichées sans fuseau explicite. Badge quand le circuit est ailleurs.
- **Fermeture des fenêtres.** Hardcore ferme au début des EL1 (×1,5), standard
  5 min avant l'extinction des feux. La pole ferme au début de Q1 quelle que
  soit la fenêtre. Une seule fenêtre par week-end.
- **Gel du modèle.** `proba_baseline` est figée et horodatée dans le pronostic
  à la soumission. Elle n'est jamais recalculée après la course.
- **Tours valides.** Tout indicateur de rythme n'utilise que les tours propres.
- **Course écourtée.** Moins de 75 % de la distance : bases à 50 %.
- **GP annulé.** Neutralisé, aucun point, jokers rendus.
- **Ligues.** Pronostics des autres membres visibles seulement après fermeture
  de leur fenêtre.
- **Classement.** Sur l'écart au modèle, score brut en colonne secondaire.
  Départage : positions exactes, puis hardcore, puis GP joués, puis antériorité.

## États d'interface

Chaque écran gère les trois états exigés par la Definition of Done :
squelettes de chargement (jamais de spinner plein écran), état vide, et panne
de source avec message explicite (« données OpenF1 indisponibles ») plutôt
qu'une page blanche.

Quand une source tombe, la dernière valeur connue est affichée **avec son
horodatage et un avertissement** — jamais en silence.

## Mode démonstration

Si Jolpica est injoignable (réseau, CORS, limite de débit), le site bascule sur
un jeu local : un week-end fictif calé sur les jours à venir, permettant de
parcourir toute l'interface. Un bandeau permanent le signale sur chaque écran,
et chaque bloc porte le badge `DÉMO`. Rien n'y est officiel. Activable aussi à
la main depuis **Réglages**.

## Accessibilité

- Contraste vérifié en thème clair et sombre, thème forçable dans les réglages.
- **Mode motifs** : les couleurs d'écurie seules ne suffisent pas — plusieurs
  paires sont indistinguables pour les daltoniens. Les motifs ajoutent rayures
  et points aux pastilles.
- Navigation au clavier, cibles tactiles ≥ 42 px, région `aria-live` pour les
  changements d'état, `prefers-reduced-motion` respecté.
- Rendu vérifié à 375 px et en desktop, sans débordement horizontal.

## Ce qui n'est pas là, et pourquoi

- **Réglages de voiture.** Donnée inexistante publiquement : aucune écurie,
  aucune API ne les publie. Ils sont remplacés par des indicateurs dérivés de
  la télémétrie, présentés comme tels. Détail sur l'écran Sources.
- **Courbes de télémétrie** (vitesse, accélérateur, frein, DRS le long du
  tour). Elles viennent de FastF1 en traitement différé (phase 2), pas d'une
  API interrogeable depuis un navigateur.
- **Tracé géographique du circuit.** Il se dessine depuis les positions GPS de
  la voiture, disponibles par FastF1 en batch. En attendant, le tour est
  déroulé en secteurs et zones DRS — exact, plutôt qu'un dessin approximatif
  présenté comme le tracé.
- **Agrégation RSS de presse.** Un flux RSS n'est pas lisible depuis un
  navigateur sans relais serveur. Les messages de direction de course, eux,
  passent bien en direct.
- **Comptes utilisateurs serveur.** Profil, pronostics et ligues vivent dans le
  navigateur ; l'échange entre joueurs passe par un code de partage.

## Point d'arbitrage : le plafond de coefficient

Le livrable 2 §9 demande si le plafond de 4,0 est bien réglé. La réponse est
visible dans l'écran de saisie, avec les chiffres du week-end en cours.

`coef = clamp(1 / proba, 0.5, 4.0)` fonctionne sur les événements à peu
d'issues — safety car ×1,67, pole ×2,29. Sur le **bloc A**, il s'effondre : la
probabilité qu'un pilote donné finisse à une place précise dans un peloton de
vingt ne dépasse guère 25 % même pour le favori, et tout ce qui est sous 25 %
sature. En pratique, quasiment toute la grille touche le plafond, et
pronostiquer le favori en P1 rapporte autant qu'un fond de grille — l'inverse
exact de ce que le barème cherchait.

Trois leviers, tous du ressort du propriétaire du produit :

1. relever le plafond du bloc A ;
2. normaliser le coefficient par la meilleure probabilité de cette place,
   `coef = clamp(p_max_place / p, 0.5, 4.0)` ;
3. réduire la grille à un top 5, qui concentre les probabilités.

Le barème est implémenté **tel qu'écrit** dans le livrable 2 ; `coef_max` et
`taille_grille` sont deux constantes de `js/config.js`.

## Modèle baseline

`js/model/baseline.js` produit `proba_baseline` : notation de force par pilote
(championnat, forme sur 5 GP à décroissance exponentielle, affinité circuit),
puis simulation Plackett–Luce par bruit de Gumbel, 4 000 tirages. La sortie est
**déterministe** pour des entrées données — sinon deux calculs successifs
donneraient des coefficients différents.

C'est volontairement simple et auditable : un modèle mal calibré casse le
scoring. Il a vocation à être remplacé par l'Elo pilote/écurie de la phase 2,
backtesté sur trois saisons avant ouverture des pronostics. L'interface de
sortie (`probas`, `parPilote`, `evenements`, `duels`) ne changera pas.

## Version de démonstration en un fichier

Le site normal a besoin d'un serveur : modules ES, feuille de style et
référentiel JSON sont servis par HTTP. Pour les contextes qui n'en ont pas
— aperçu en bac à sable, envoi par fichier, ouverture en local — un script
produit un HTML autonome contenant tout.

```bash
npm install          # esbuild, uniquement pour ce build
npm run demo         # → web/demo.html
```

Trois adaptations, et seulement trois : le référentiel circuits est intégré au
bundle, le service worker n'est pas enregistré, et le mode démonstration est
forcé — aucun appel vers Jolpica, OpenF1 ou Open-Meteo ne peut aboutir depuis
un fichier local. L'interface le dit au lieu d'échouer en silence.

Les sources du site ne sont jamais modifiées : les substitutions vivent dans
`build-demo.mjs`, et chacune est vérifiée. Si le code source change et qu'un
motif ne correspond plus, le build s'arrête au lieu de produire un fichier
silencieusement cassé.

Le site lui-même n'a aucune dépendance : `esbuild` ne sert qu'à ce build.

## Vérification

```bash
# Syntaxe de tous les modules
for f in $(find js sw.js -name '*.js'); do cp "$f" /tmp/c.mjs && node --check /tmp/c.mjs; done
```

Le moteur de score et le modèle baseline sont vérifiables hors navigateur :
sommes des probabilités marginales, déterminisme, bornes des coefficients,
multiplicateur hardcore, joker, course écourtée, GP neutralisé, écart au
modèle et départage.

## Propriété intellectuelle

Aucun logo, aucune image officielle, aucun live timing redistribué. Les
couleurs d'écurie sont des approximations saisies à la main pour la lisibilité
des graphiques. Usage personnel, non commercial. Ce site n'est affilié ni à la
FIA, ni à Formula One World Championship Limited.
