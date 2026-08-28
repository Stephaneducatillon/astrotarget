# Site du club d'astronomie — MJC de Douai

Site vitrine + espace membres (base documentaire et chat), en HTML, CSS et
JavaScript **sans aucune étape de compilation** : pas de Node, pas de build,
pas de framework. On ouvre un fichier, on modifie, on enregistre.

## Voir le site

Double-cliquer sur `index.html` suffit pour une première visite. Pour que
l'espace membres fonctionne complètement (le navigateur restreint les pages
ouvertes en `file://`), servez plutôt le dossier :

```bash
cd site
python3 -m http.server 8000
# puis http://localhost:8000
```

## Contenu du dossier

| Fichier / dossier    | Rôle |
|----------------------|------|
| `index.html`         | Accueil |
| `club.html`          | Présentation du club, fonctionnement, matériel |
| `activites.html`     | Les activités proposées |
| `agenda.html`        | Séances à venir et passées |
| `galerie.html`       | Images des membres (emplacements à remplir) |
| `adherer.html`       | Adhésion, tarifs, contact |
| `membres.html`       | Espace réservé : base documentaire + chat |
| `css/style.css`      | Toute la mise en forme |
| `js/config.js`       | **Configuration du site** — le fichier à éditer en premier |
| `js/app.js`          | Menu, ciel étoilé, rendu de l'agenda |
| `js/backend.js`      | Accès aux comptes et aux données de l'espace membres |
| `js/membres.js`      | Connexion, base documentaire, chat |
| `data/evenements.js` | **Les dates de l'agenda** |
| `img/`               | Photos de la galerie |
| `SUPABASE.md`        | Activer le vrai espace membres partagé |

## Les trois choses à faire en premier

### 1. Compléter les informations du club

Cherchez `à compléter` dans les pages HTML : adresse, téléphone, tarifs,
année de création, modèles de télescopes. Ces mentions apparaissent
soulignées en jaune sur le site tant qu'elles ne sont pas remplacées.

L'adresse de contact se règle une seule fois, dans `js/config.js`.

### 2. Mettre l'agenda à jour

Tout se passe dans `data/evenements.js`. Copiez un bloc, changez les valeurs :

```js
{
  date: "2026-09-12",              // AAAA-MM-JJ
  heure: "21h00 – 00h00",
  titre: "Soirée d'observation de rentrée",
  type: "observation",             // observation | atelier | conference
                                   // | initiation | sortie
  lieu: "MJC de Douai — parvis",
  resume: "Description en une ou deux phrases.",
  public: "Tout public, à partir de 8 ans",
},
```

Les séances passées basculent automatiquement dans « Séances passées » : rien
à supprimer.

### 3. Remplir la galerie

Déposez les images dans `img/`, puis dans `galerie.html` remplacez chaque
bloc :

```html
<div class="galerie__vignette" aria-hidden="true">🌕</div>
```

par :

```html
<img src="img/lune-premier-quartier.jpg" alt="La Lune au premier quartier">
```

Réduisez les photos à 1600 px de large environ : une page qui met dix
secondes à charger est une page que personne ne regarde.

## L'espace membres

Il fonctionne dans deux modes, réglés par `modeMembre` dans `js/config.js`.

**`"local"` (par défaut) — démonstration.** Comptes, documents et messages
sont enregistrés dans le navigateur du visiteur (`localStorage`). Rien n'est
partagé : deux personnes ne voient pas les mêmes messages. Utile pour
présenter le site au club avant de décider.

**`"supabase"` — le vrai espace du club.** Comptes, base documentaire et chat
hébergés et partagés, avec les nouveaux messages en direct. Un site statique
n'a pas de serveur : ce partage exige forcément un service extérieur. La
marche à suivre complète est dans [`SUPABASE.md`](SUPABASE.md) — offre
gratuite, une vingtaine de minutes, aucune ligne de code à écrire.

Le **code d'accès** (`codeInscription` dans `js/config.js`) est demandé à la
création d'un compte. Il filtre les curieux ; il ne remplace pas les règles de
sécurité décrites dans `SUPABASE.md`. Changez-le à chaque rentrée.

Les **salons du chat** se règlent aussi dans `js/config.js` : ajoutez ou
retirez des entrées de la liste `salons`.

## Montrer le site sans l'héberger

`apercu.html` regroupe tout le site — les sept pages, le style, les scripts —
dans **un seul fichier autonome**. Il s'ouvre par un double-clic, s'envoie par
e-mail et fonctionne sans connexion : de quoi le présenter en réunion de club.

Après une modification du site, régénérez-le :

```bash
python3 site/outils/construire-apercu.py
```

## Mettre le site en ligne

Le dossier ne contient que des fichiers statiques : n'importe quel hébergement
convient (GitHub Pages, Netlify, l'hébergement de la MJC, un simple FTP).

Pour **GitHub Pages**, le dépôt contient déjà le nécessaire
(`.github/workflows/pages.yml`) :

1. fusionnez cette branche dans `main` ;
2. dans le dépôt, *Settings → Pages → Source* : choisissez **GitHub Actions** ;
3. le site est publié sur `https://<compte>.github.io/astrotarget/`, et remis à
   jour à chaque modification de `site/` poussée sur `main`.

Le mode *Deploy from a branch* ne convient pas ici : il n'accepte que la racine
du dépôt ou `docs/`, jamais `site/`.

## Choix techniques

- **Aucune dépendance** en mode local : le site marche hors ligne, y compris
  sur un vieux navigateur. La bibliothèque Supabase n'est chargée que si le
  mode `supabase` est activé.
- **Un seul point d'entrée pour le contenu** : `js/config.js` et
  `data/evenements.js` couvrent l'essentiel des mises à jour courantes, sans
  toucher au HTML.
- **Accessibilité** : navigation au clavier, lien d'évitement, contrastes
  vérifiés, textes alternatifs, `prefers-reduced-motion` respecté.
- **Sécurité de l'espace membre** : en mode local, les mots de passe sont
  dérivés en PBKDF2-SHA256 (150 000 itérations) — mais tout reste dans le
  navigateur, ce n'est pas une protection sérieuse et le bandeau le dit. La
  vraie protection vient du mode Supabase.
