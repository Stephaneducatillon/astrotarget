# Veille des risques — phase 1

Dispositif de veille familiale pour le secteur de Douai (Nord), suivant
[`SOLUTIONS.md`](SOLUTIONS.md). Cette phase couvre les **4 sources de niveau A**
— celles qui changent ce que fait la famille dans les 6 heures :

| Règle | Source | Déclenche |
|---|---|---|
| `mf_vigilance` | Vigilance Météo-France (59, 62) | orange / rouge sur un phénomène du périmètre |
| `ecowatt` | Ecowatt (RTE), J à J+3 | orange / rouge |
| `scarpe_niveau` | Hub'Eau hydrométrie v2 | dépassement du maximum des 30 derniers jours |
| `vigicrues` | Vigicrues, tronçon surveillé | orange / rouge |

Tout tourne sur le plan **gratuit** Cloudflare : Workers (2 crons sur les 3
disponibles), D1, KV, Pages.

## Architecture

```
  crons */15 et 5h30
        │
        ▼
  ┌───────────────┐   collecte parallèle    ┌──────────────────────────┐
  │    Worker     │◄────────────────────────┤ MF · RTE · Hub'Eau · VC  │
  └───────┬───────┘                          └──────────────────────────┘
          │  règles ► machine à états ► santé des sources
          ├──────────────► D1   (historique 40 j, états, journal)
          ├──────────────► KV   (1 écriture/run : snapshot:latest)
          └──────────────► ntfy (transitions uniquement)
                                   │
                            PWA (Pages) ◄── /api/snapshot
```

## Ce que le code garantit

Les choix listés ci-dessous sont **couverts par les tests** (`npm test`, 53 cas) :

- **Une vigilance orange de 12 h produit 2 notifications, pas 48.** Machine à
  états : on notifie les transitions, montée immédiate, descente confirmée sur
  3 relevés, rappel plafonné à 6 h.
- **Une source en panne n'est jamais « verte ».** Elle est *indéterminée* : le
  dernier état connu est préservé et la panne devient elle-même une alerte au
  3ᵉ échec consécutif (immédiatement si la clé est refusée).
- **La mesure du run n'entre pas dans son propre maximum.** La fenêtre glissante
  est lue avant insertion, sinon le seuil est indépassable par construction.
- **Un run établi = une seule écriture KV** (96/jour contre 1 000 autorisées),
  jeton RTE mis en cache, Ecowatt interrogé au plus toutes les 3 h.
- **Aucune adresse dans les notifications** — elles transitent par un tiers.
- **L'API est fermée par défaut** : sans `ADMIN_TOKEN` configuré, tout est
  refusé plutôt qu'ouvert.

## Mise en service

### 1. Comptes et clés (chemin critique — à lancer en premier)

- **Météo-France** : compte sur `portail-api.meteofrance.fr`, application sur
  « Bulletin Vigilance », clé générée avec une **durée de 0** (illimitée).
  Une clé qui expire est une panne silencieuse.
- **RTE** : compte sur le portail API, application avec la tuile Ecowatt,
  relever `client_id` et `client_secret`.
- **ntfy** : choisir un nom de sujet **aléatoire d'au moins 32 caractères**
  (`openssl rand -hex 24`), s'y abonner depuis l'application Android.
  Un sujet devinable est un sujet public.

### 2. Ressources Cloudflare

```bash
cd veille_risques/worker
npm install -g wrangler && wrangler login

wrangler kv namespace create KV     # reporter l'id dans wrangler.toml
wrangler d1 create veille           # reporter l'id dans wrangler.toml
wrangler d1 execute veille --file=schema.sql --remote

wrangler secret put MF_API_KEY
wrangler secret put RTE_CLIENT_ID
wrangler secret put RTE_CLIENT_SECRET
wrangler secret put NTFY_TOPIC
wrangler secret put ADMIN_TOKEN     # openssl rand -hex 32
wrangler deploy
```

### 3. Relever les codes de station, une fois pour toutes

Ils ne sont pas devinables et ne doivent **jamais** être recherchés à chaque
run. Les endpoints d'administration les listent :

```bash
BASE=https://veille-risques.<sous-domaine>.workers.dev
curl -H "Authorization: Bearer $ADMIN_TOKEN" "$BASE/admin/stations?q=Scarpe"
curl -H "Authorization: Bearer $ADMIN_TOKEN" "$BASE/admin/troncons?q=scarpe"
```

Recopier les entrées utiles dans `STATIONS_HYDRO` et `TRONCONS_VIGICRUES`
(`src/config.js`), puis `wrangler deploy`.

### 4. Vérifier les formats réels

Les parseurs sont tolérants mais la structure exacte des réponses n'a pas pu
être validée en direct à l'écriture (accès réseau restreint). À faire une fois,
avant de faire confiance au dispositif :

```bash
curl -H "Authorization: Bearer $ADMIN_TOKEN" "$BASE/admin/probe?source=vigilance"
curl -H "Authorization: Bearer $ADMIN_TOKEN" "$BASE/admin/probe?source=vigicrues"
curl -H "Authorization: Bearer $ADMIN_TOKEN" "$BASE/admin/notify-test"
curl -X POST -H "Authorization: Bearer $ADMIN_TOKEN" "$BASE/api/run"
```

`/api/run` déclenche un cycle complet et retourne son résultat : c'est le test
de bout en bout réel.

### 5. PWA

```bash
wrangler pages deploy veille_risques/pwa --project-name veille
```

Au premier lancement, depuis la console du navigateur :

```js
localStorage.setItem('api', 'https://veille-risques.<sous-domaine>.workers.dev');
localStorage.setItem('token', '<ADMIN_TOKEN>');
```

Puis « Ajouter à l'écran d'accueil ».

**Renseigner `ALLOWED_ORIGIN`** dans `wrangler.toml` avec l'URL Pages réelle,
puis redéployer le Worker. La PWA et l'API sont sur deux origines distinctes :
sans cet en-tête, le navigateur interdit au script de lire la réponse et le
tableau de bord reste vide. L'origine est explicite, jamais `*`.

**Protéger la page.** Sans cela, les couches personnelles sont publiques dès
qu'on connaît l'URL. Cloudflare Access (gratuit jusqu'à 50 utilisateurs) se
place devant Pages sans écrire une ligne de code.

## Période de chauffe

La règle Scarpe exige `hydro_min_observations` mesures (200, soit ~2 jours à
15 min) avant de s'activer. C'est voulu : sans historique, le « maximum des
30 jours » est calculé sur trois points et déclenche sur n'importe quoi. Le
tableau de bord affiche « historique insuffisant » pendant cette période —
la règle est inactive, et elle le dit.

## Confidentialité

- Aucune adresse, aucun nom d'établissement dans ce dépôt ni dans les
  notifications. L'historique git est permanent : une fuite ne se rattrape pas.
- Les couches personnelles (maison, écoles, points de rendez-vous) vont dans
  `pwa/couches-privees.json`, **ignoré par git**.
- `.dev.vars` est ignoré : ne jamais y substituer un fichier commité.

## Tests

```bash
cd veille_risques/worker && npm test
```

64 cas, sans réseau ni compte : machine à états, parseurs (dont réponses vides
et inattendues), santé des sources, et cycle complet contre de faux D1/KV/fetch.

## Limites connues

- Les formats de réponse des API sont **supposés** d'après leur documentation,
  pas constatés (voir étape 4). Les parseurs dégradent au lieu de lever, mais
  un format inattendu peut produire un « vert » silencieux : lancer `/admin/probe`
  avant de faire confiance au dispositif.
- `Enedis` n'a pas d'API publique : lien direct dans l'onglet Hors ligne.
- **Pendant une coupure d'électricité ou de réseau, ce dispositif ne fonctionne
  pas.** Son métier est l'avertissement en amont. Le reste repose sur FR-Alert,
  la radio, OsmAnd hors ligne et le plan familial papier.
