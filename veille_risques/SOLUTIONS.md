# Veille des risques — propositions techniques

Réponse au document *« Sources de données pour une application de veille des risques »*
(version du 21 septembre 2026, secteur de Douai, Nord).

Ce document ne réécrit pas l'inventaire des sources, qui est solide. Il propose des
**décisions d'architecture**, corrige quelques points sur l'accès aux données, et
signale les **trous fonctionnels** qui feraient échouer l'application le jour où elle
doit servir.

---

## 0. Résumé des propositions

| # | Proposition | Pourquoi |
|---|---|---|
| 1 | Classer les sources par **actionnabilité**, pas par thème | 40 sources traitées à égalité = bruit ; 6 sources déclenchent réellement une action |
| 2 | Cloudflare Workers + **D1** (pas seulement KV) + Pages | La règle « max des 30 derniers jours » exige un historique ; KV seul ne sait pas faire |
| 3 | **3 crons par cadence** (15 min / 1 h / 1 j), pas un cron unique | Le plan gratuit en autorise exactement 3 ; divise la consommation de quotas par ~8 |
| 4 | **Machine à états sur les alertes** (notifier les transitions) | Sans ça : 96 notifications par jour pour une seule vigilance orange |
| 5 | **Surveiller la santé des sources** comme une source à part entière | Le pire scénario n'est pas l'alerte manquée, c'est le connecteur mort en silence |
| 6 | Cache du jeton **OAuth2 RTE**, pas de ré-authentification par run | 96 authentifications/jour pour 2 publications réelles |
| 7 | Reporter GDELT / ACLED / pétrole en **phase 3** | Coût de développement élevé, zéro action déclenchée |
| 8 | Assumer que l'app est **inutile si le réseau tombe** | Le vrai plan de repli est FR-Alert + radio + plan papier |

---

## 1. Le classement à faire d'abord : actionnabilité

Le document mélange deux natures de sources très différentes. C'est la principale
faiblesse structurelle, et elle coûtera cher en développement comme en attention.

**Niveau A — change ce que fait la famille dans les 6 heures** (6 sources)
Vigilance Météo-France · Ecowatt · Hub'Eau hydrométrie (Scarpe) · Vigicrues (tronçon Scarpe) ·
FR-Alert (hors application) · SNCF temps réel sur la ligne du père.

**Niveau B — change une décision dans la semaine** (4 sources)
Hub'Eau piézométrie · prix/ruptures carburants · Vigipirate · qualité de l'air ATMO.

**Niveau C — contexte, consultation à la demande** (tout le reste)
Géorisques, GDELT, ACLED, GDACS, EIA, presse RSS, éCO2mix, AGSI+, épidémies.

**Conséquence concrète :** seul le niveau A justifie un polling à 15 min, une règle
d'alerte et une notification push. Le niveau B se consulte dans le tableau de bord avec
un rafraîchissement quotidien. Le niveau C n'a pas besoin de collecte automatisée du
tout — un lien dans l'application suffit, et Géorisques en particulier est une donnée
**statique** qu'on télécharge une fois et qu'on ne remet à jour que manuellement.

Cela ramène le projet de ~40 connecteurs à **10 connecteurs actifs**, soit la différence
entre un projet qui aboutit et un projet abandonné à 60 %.

---

## 2. Architecture : confirmer Cloudflare Workers, mais avec D1

L'option recommandée par le document est la bonne. Deux corrections importantes.

### 2.1 KV ne suffit pas — il faut D1

La règle *« station de la Scarpe au-dessus de son maximum des 30 derniers jours »* exige
de conserver un historique et de l'interroger. KV est un magasin clé-valeur : pour faire
ce calcul il faudrait relire et réécrire un gros blob à chaque run, ce qui est fragile et
consomme les écritures.

Quotas du plan gratuit (à vérifier au moment de coder, ils bougent) :

| Service | Limite gratuite | Consommation du projet |
|---|---|---|
| Workers | 100 000 requêtes/jour | ~200/jour (crons + usage familial) — très large |
| Workers KV | **1 000 écritures/jour** | 96/jour si **une seule** écriture agrégée par run |
| Workers D1 | 5 Go, **100 000 lignes écrites/jour** | ~2 000/jour — très large |
| Cron Triggers | **3 par Worker**, intervalle minimal 1 min | 3 utilisés exactement (§2.2) |

> ⚠️ **Point de vigilance daté.** Depuis le 1ᵉʳ septembre 2026, Cloudflare **applique
> réellement** les limites quotidiennes D1 sur le plan gratuit : au-delà du quota, les
> requêtes échouent avec une erreur jusqu'à minuit UTC. Ce n'est plus une limite
> théorique. D'où l'importance de la purge (§2.3).

**Répartition proposée :**
- **D1** — séries temporelles (hauteurs Scarpe, piézométrie, prix carburants), historique
  des alertes, table de santé des sources.
- **KV** — un seul objet `snapshot:latest`, l'état courant complet servi à la PWA. Une
  écriture par run, lecture quasi gratuite, très rapide.
- **Pages** — la PWA statique.
- **Secrets Wrangler** — clés Météo-France, RTE, EIA, nom du sujet ntfy.

### 2.2 Trois crons, pas un

Le plan gratuit autorise 3 Cron Triggers par Worker. C'est exactement ce qu'il faut :

```
*/15 * * * *   → niveau A : Vigilance MF, Ecowatt, Hub'Eau Scarpe, Vigicrues, SNCF
17 * * * *     → niveau B : carburants, Vigipirate, ATMO, AROME
30 5 * * *     → niveau C + maintenance : piézométrie, RSS, purge D1, rapport de santé
```

Le document propose « toutes les 15 min » globalement. Interroger Ecowatt (publié 2 fois
par jour) ou la piézométrie (1 fois par jour) toutes les 15 minutes, c'est 96 appels pour
1 à 2 données neuves — du gaspillage de quota côté fournisseur, et un risque de se faire
limiter sur des API publiques gratuites qu'on a intérêt à ménager.

### 2.3 Purge et rétention

Une seule règle a besoin d'historique (30 jours glissants). Prévoir dès le départ, dans
le cron quotidien :

```sql
DELETE FROM observations WHERE ts < unixepoch('now','-40 days');
```

Sans ça, la base grossit indéfiniment et les requêtes de calcul du maximum finissent par
consommer le quota de lignes lues.

### 2.4 Budget CPU : le seul point dur

Le plan gratuit impose un budget CPU serré par invocation. **L'attente réseau ne compte
pas** — on peut donc appeler 10 API en parallèle sans problème. Ce qui compte, c'est le
parsing.

Un seul élément est réellement coûteux : le **GTFS-RT Trip Updates SNCF**, qui est du
protobuf et couvre tout le réseau national (~1 400 entités `trip_update` sur un relevé de
juillet 2026), alors qu'on ne veut que 2 ou 3 trains.

Trois options, par ordre de préférence :
1. **SIRI-Lite / GTFS-RT Service Alerts** — beaucoup plus léger, suffit à détecter
   « perturbation sur la ligne », ce que demande la règle d'alerte du document.
2. Décoder le protobuf, mais **dans un Worker séparé** avec son propre cron, pour
   isoler le risque de dépassement CPU.
3. Ne collecter qu'aux heures de trajet (`*/5 6-9,16-20 * * 1-5`), ce qui divise le coût
   par 4 — mais consomme un cron trigger sur les 3 disponibles.

**Recommandation : commencer par (1).** La règle familiale est « notification simple »,
pas un niveau d'alerte — elle ne justifie pas le coût du protobuf.

### 2.5 Sur GitHub Actions (l'alternative du document)

L'avertissement du document — « un dépôt public expose les données » — est juste mais
sous-estimé. Au-delà de l'adresse : l'**historique git est permanent**. Une donnée
publiée par erreur reste récupérable après suppression du fichier. Combiné aux
déclenchements en retard (les crons GitHub Actions peuvent glisser de 10 à 30 minutes en
période de charge, ce qui est rédhibitoire pour un cron à 15 min), cette option est à
écarter — sauf comme **filet de sécurité** : un job quotidien qui vérifie que le Worker
répond et alerte s'il est muet.

---

## 3. Le trou principal : la répétition des notifications

Le document définit très bien les *conditions* d'alerte (§8) mais jamais le
*déclenchement*. Tel quel, une vigilance orange qui dure 12 heures produit **48
notifications identiques**. Au bout de deux épisodes, la famille coupe les notifications
— et l'application ne sert plus à rien le jour où ça compte.

### 3.1 Notifier les transitions, pas les états

Chaque règle porte un état persisté en D1. On notifie quand l'état change :

| Transition | Action |
|---|---|
| vert → orange | Notification ntfy, priorité `default` |
| orange → rouge | Notification ntfy, priorité `urgent` |
| rouge → orange | Notification de désescalade, priorité `low` |
| orange → vert | Notification de fin, priorité `min` |
| état inchangé | **rien** |

### 3.2 Hystérésis et anti-battement

Deux garde-fous indispensables :

- **Montée immédiate, descente différée.** Passer au niveau supérieur dès la première
  détection. Ne redescendre qu'après **3 relevés consécutifs** sous le seuil (45 min).
  Évite le clignotement sur une station hydrométrique qui oscille autour de son seuil.
- **Rappel plafonné.** Si un état orange dure, un seul rappel toutes les 6 heures,
  jamais plus.

### 3.3 Schéma D1 minimal

```sql
CREATE TABLE alert_state (
  rule_id      TEXT PRIMARY KEY,   -- 'mf_vigilance_59', 'ecowatt', 'scarpe_level'
  level        TEXT NOT NULL,      -- 'vert' | 'orange' | 'rouge'
  since        INTEGER NOT NULL,   -- epoch du passage à ce niveau
  last_notified INTEGER,           -- pour le plafond de rappel
  below_count  INTEGER DEFAULT 0   -- compteur d'hystérésis à la descente
);

CREATE TABLE source_health (
  source_id    TEXT PRIMARY KEY,
  last_ok      INTEGER,            -- dernier succès
  last_error   TEXT,
  fail_streak  INTEGER DEFAULT 0
);

CREATE TABLE observations (
  source_id TEXT, station TEXT, ts INTEGER, metric TEXT, value REAL,
  PRIMARY KEY (source_id, station, ts, metric)
);
```

---

## 4. Le deuxième trou : la santé des sources

Le document ne prévoit rien si une source cesse de répondre. C'est le scénario d'échec le
plus probable et le plus dangereux : **un connecteur Ecowatt cassé pendant un jour rouge
ressemble exactement à un jour vert.** Silence et « tout va bien » sont indiscernables.

Propositions :

1. **Table `source_health`** mise à jour à chaque tentative (succès comme échec).
2. **Le tableau de bord affiche l'âge de chaque donnée**, pas seulement sa valeur.
   Au-delà de 3× la fréquence attendue, badge gris « donnée périmée ». Une valeur sans
   horodatage visible est un mensonge par omission.
3. **La panne est elle-même une alerte.** Si une source de **niveau A** échoue
   3 fois de suite, notification ntfy en priorité basse : « Ecowatt muet depuis 45 min ».
4. **Clé expirée = panne silencieuse classique.** Les jetons Météo-France se créent avec
   une durée ; la créer à 0 (illimitée) comme l'indique la documentation du portail, et
   traiter tout `401`/`403` comme un incident distinct d'un `5xx`, avec un message
   explicite.
5. **Repli sur miroir.** Pour la vigilance météo, garder en second choix un miroir open
   data sans clé (jeux `weatherref-france-vigilance-meteo-departement` sur la plateforme
   Opendatasoft publique). Dégradé, mais mieux que rien si la clé saute un jour de
   tempête.

---

## 5. Corrections et précisions sur les sources

### 5.1 Vigicrues a une API — pas besoin de scraper

Le document indique « Site vigicrues.gouv.fr ; identifiants de stations dans l'onglet
Info ». Il existe en réalité une **API documentée** (documentation sur
`vigicrues.gouv.fr/services/v1.1`), avec notamment :

- `InfoVigiCru.geojson` / `.jsonld` — niveaux de vigilance par tronçon, en GeoJSON ;
- `TronEntVigiCru.json` — référentiel des tronçons ;
- `StaEntVigiCru.json` — stations associées.

C'est plus robuste qu'un scraping, et le GeoJSON s'affiche directement sur la carte
Leaflet sans transformation. Le jeu « tronçons Vigicrues simplifiés » est par ailleurs
publié sur data.gouv.fr, utile comme fond de carte statique.

### 5.2 RTE : mettre en cache le jeton OAuth2

Ecowatt s'authentifie en OAuth2 *client credentials*. Le jeton est valable environ 2
heures. Se ré-authentifier à chaque run, c'est 96 appels d'authentification par jour pour
2 publications réelles — et un bon moyen de se faire limiter.

**À faire :** stocker le jeton en KV avec son expiration, ne le renouveler qu'à
expiration (avec une marge de 5 minutes). Et puisque Ecowatt publie la veille à 17 h et
met à jour à 10 h 30, le cron horaire suffit largement : **inutile de l'interroger toutes
les 15 minutes.**

### 5.3 Hub'Eau : bien passer en v2

L'API hydrométrie v2 (`hubeau.eaufrance.fr/api/v2/hydrometrie/`) est sans clé et sert du
JSON, du GeoJSON et du CSV. Deux points pratiques :

- Repérer les codes de station **une fois pour toutes** via `referentiel/stations`, et
  les figer en configuration. Ne pas les rechercher à chaque run.
- Utiliser `observations_tr` avec `size` et un filtre temporel serré — sans quoi on
  ramène des milliers de points pour n'en garder qu'un.

### 5.4 Règle carburants : le seuil demande un dénominateur stable

La règle « plus de 10 % de stations en rupture dans un rayon de 20 km » est correcte mais
piégeuse : le flux quotidien ne liste pas toujours les mêmes stations. Si 30 stations
disparaissent du fichier un jour, le pourcentage bouge sans qu'aucune rupture réelle
n'ait eu lieu.

**Correction :** figer le dénominateur sur la liste des stations vues au moins une fois
dans les 30 derniers jours, et exiger un **minimum de 15 stations** dans le rayon avant
d'évaluer la règle.

### 5.5 Sources à reporter ou abandonner

- **GDELT, ACLED, EIA (pétrole/gaz)** — aucune de ces sources ne déclenche une action
  familiale à Douai. Le coût de développement est réel (formats particuliers, volumes,
  filtrage géographique pour GDELT). À mettre en phase 3, voire à remplacer par 2 flux
  RSS bien choisis.
- **Enedis** — le document conclut à juste titre « pas d'API publique, lien direct ».
  Garder ce lien, ne pas scraper : la page est protégée et fragile.
- **Ecogaz** — en maison tout-électrique, la tension sur le réseau gaz n'a pas d'effet
  direct. Niveau C.
- **Géorisques** — à traiter comme un **export ponctuel**, pas comme un connecteur.
  Télécharger les couches une fois, les stocker en GeoJSON simplifié dans les assets de
  la PWA, et les rafraîchir à la main une fois par an. Ces données changent de l'ordre de
  quelques fois par décennie.

---

## 6. Confidentialité et accès

Le document pose bien le principe (couches personnelles privées) mais pas le mécanisme.
Quatre propositions :

1. **Sujet ntfy = secret réel.** Un sujet ntfy public est lisible par quiconque devine le
   nom. Générer un nom aléatoire d'au moins 32 caractères, le stocker en secret Wrangler,
   ne jamais l'écrire dans le dépôt ni dans la PWA servie publiquement. Les messages ne
   doivent contenir **aucune adresse** : « Vigilance rouge vent — Nord », jamais le nom
   de l'école.
2. **La PWA doit être protégée**, sinon les couches personnelles sont publiques dès
   qu'on connaît l'URL. Le plus simple et suffisant ici : **Cloudflare Access** en mode
   gratuit (jusqu'à 50 utilisateurs), qui place une authentification devant Pages sans
   code. Une alternative à coût nul : servir les couches personnelles depuis le Worker
   derrière un jeton partagé, plutôt que de les inclure dans le bundle statique.
3. **Aucune donnée personnelle dans le dépôt git.** Les couches maison/écoles/points de
   rendez-vous dans un fichier ignoré, ou dans D1, jamais commité. C'est irréversible
   dans l'historique git.
4. **Attribution OpenStreetMap** — obligatoire (ODbL) dès que la carte est affichée,
   même en usage strictement personnel. Une ligne dans le pied de carte.

---

## 7. Ce que l'application ne fera pas (à assumer)

C'est une remarque de fond, pas technique : **la coupure électrique et la coupure réseau
sont précisément les scénarios où cette application ne fonctionnera pas.** Ecowatt
prévient d'une coupure possible ; pendant la coupure, l'application ne sert plus.

Il faut donc que le dispositif reste honnête sur son périmètre :

- **En amont** (avertissement) : l'application est utile, c'est son vrai métier.
- **Pendant la crise** : le plan repose sur FR-Alert (diffusion cellulaire, indépendante
  du dispositif), la radio FM, OsmAnd hors ligne, et le **plan familial papier**.

Proposition concrète : la PWA affiche, dans chaque notification et sur l'écran d'accueil,
un rappel de la checklist du niveau — ce que le document prévoit déjà — et un onglet
« hors ligne » listant fréquences radio, points de rendez-vous et numéros, **mis en cache
par le service worker** et consultable batterie faible, sans réseau. Prévoir aussi le cas
du téléphone déchargé : une version papier dans le sac de départ.

Vérifier enfin que le service worker met bien en cache le **dernier snapshot connu** et
l'affiche avec son horodatage, plutôt qu'un écran vide, quand le réseau est absent.

---

## 8. Plan par phases

**Phase 1 — le socle qui sert vraiment (≈ 1 à 2 semaines)**
Worker + D1 + KV + Pages. 4 sources : Vigilance Météo-France (59/62), Ecowatt, Hub'Eau
Scarpe, Vigicrues tronçon. Machine à états + ntfy avec anti-répétition. Tableau de bord
avec âge des données. Carte IGN + OSM avec couches personnelles privées.
*À ce stade, 80 % de la valeur est livrée.*

**Phase 2 — confort et robustesse (≈ 1 semaine)**
SNCF Service Alerts sur la ligne du père. Carburants + ruptures. ATMO. Vigipirate.
Surveillance de santé des sources, rapport quotidien. Couches Géorisques statiques.

**Phase 3 — contexte (optionnel)**
Piézométrie, GDACS, RSS CERT-FR et préfectures, séismes EMSC, éCO2mix. À n'engager que
si les phases 1 et 2 tournent sans maintenance depuis un mois.

**À faire avant de coder quoi que ce soit :**
1. Créer les comptes et **générer les clés** (Météo-France durée 0, RTE/Ecowatt) — c'est
   le chemin critique, les validations peuvent prendre quelques jours.
2. **Relever les codes de station** : Scarpe à Douai (Hub'Eau), tronçon Vigicrues
   correspondant, piézomètres du bassin minier. Les figer en configuration.
3. Vérifier chaque quota et chaque format **au moment de coder** : comme le note le
   document, les conditions d'accès évoluent — et les limites Cloudflare ont changé en
   septembre 2026.

---

## Sources vérifiées pour ce document

- [Cloudflare Workers — Pricing](https://developers.cloudflare.com/workers/platform/pricing/) et [Limits](https://developers.cloudflare.com/workers/platform/limits)
- [Cloudflare — D1 enforces free tier daily query limits (1ᵉʳ sept. 2026)](https://developers.cloudflare.com/changelog/post/2026-09-01-d1-free-tier-limit-enforcement/)
- [Hub'Eau — API Hydrométrie](https://hubeau.eaufrance.fr/page/api-hydrometrie) · [documentation v2](https://hubeau.eaufrance.fr/api/v2/hydrometrie/api-docs) · [passage en v2](https://hubeau.eaufrance.fr/news/lapi-hydrometrie-passe-en-version-2)
- [Vigicrues — documentation API v1.1](https://www.vigicrues.gouv.fr/services/v1.1) · [tronçons simplifiés sur data.gouv.fr](https://www.data.gouv.fr/datasets/troncons-de-cours-deau-vigicrues-simplifies-avec-niveau-de-vigilance-crues/)
- [Météo-France — API Données Publiques Vigilance](https://portail-api.meteofrance.fr/web/api/DonneesPubliquesVigilance) · [API Bulletin Vigilance sur data.gouv.fr](https://www.data.gouv.fr/dataservices/api-bulletin-vigilance)
- [Miroir open data — Vigilance météo départementale](https://public.opendatasoft.com/explore/dataset/weatherref-france-vigilance-meteo-departement/api/)
- [transport.data.gouv.fr — GTFS-RT Trip Updates SNCF](https://transport.data.gouv.fr/resources/83197?locale=fr) · [SIRI SX Lite et GTFS-RT Service Alerts](https://ressources.data.sncf.com/explore/dataset/temps-reel-siri-sx-lite/) · [serveur proxy GTFS-RT](https://doc.transport.data.gouv.fr/type-donnees/operateurs-de-transport-regulier-de-personnes/administration-des-donnees-transport-collectif/publier-des-donnees-temps-reel/serveur-proxy-gtfs-rt)
