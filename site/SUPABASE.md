# Activer le vrai espace membres (Supabase)

Par défaut, le site fonctionne en **mode démonstration** : les comptes, les
documents et les messages restent dans le navigateur de chaque visiteur. C'est
pratique pour montrer le site, mais rien n'est partagé entre les membres.

Pour que l'espace membres devienne réellement collectif, il faut un service qui
stocke les données. Le site est prévu pour **Supabase**, dont l'offre gratuite
suffit très largement à un club (500 Mo de base, 50 000 utilisateurs actifs par
mois).

Comptez une vingtaine de minutes, sans écrire une ligne de code.

---

## 1. Créer le projet

1. Créez un compte sur <https://supabase.com> puis un nouveau projet
   (région Europe, par exemple `eu-west-3`).
2. Notez le mot de passe de la base : il ne sera plus affiché.
3. Dans **Project Settings → API**, relevez :
   - l'**URL du projet** (`https://xxxxxxxx.supabase.co`) ;
   - la clé **anon public**.

La clé « anon » est faite pour être publiée dans une page web : elle ne donne
accès qu'à ce que les règles de sécurité ci-dessous autorisent. Ne publiez
**jamais** la clé `service_role`.

---

## 2. Créer les tables

Dans **SQL Editor**, collez et exécutez ce script :

```sql
-- Base documentaire ------------------------------------------------------
create table public.documents (
  id          uuid primary key default gen_random_uuid(),
  titre       text not null,
  categorie   text not null default 'Divers',
  description text default '',
  url         text default '',
  auteur      text not null,
  auteur_id   uuid not null references auth.users (id) on delete cascade,
  cree_le     timestamptz not null default now()
);

-- Chat entre membres -----------------------------------------------------
create table public.messages (
  id        uuid primary key default gen_random_uuid(),
  salon     text not null,
  texte     text not null check (char_length(texte) between 1 and 2000),
  auteur    text not null,
  auteur_id uuid not null references auth.users (id) on delete cascade,
  cree_le   timestamptz not null default now()
);

create index messages_salon_date_idx on public.messages (salon, cree_le);

-- Sécurité : rien n'est lisible sans être connecté -----------------------
alter table public.documents enable row level security;
alter table public.messages  enable row level security;

create policy "documents lisibles par les membres connectés"
  on public.documents for select
  to authenticated using (true);

create policy "un membre ajoute un document en son nom"
  on public.documents for insert
  to authenticated with check (auth.uid() = auteur_id);

create policy "un membre ne retire que ses documents"
  on public.documents for delete
  to authenticated using (auth.uid() = auteur_id);

create policy "messages lisibles par les membres connectés"
  on public.messages for select
  to authenticated using (true);

create policy "un membre écrit en son nom"
  on public.messages for insert
  to authenticated with check (auth.uid() = auteur_id);

create policy "un membre supprime ses messages"
  on public.messages for delete
  to authenticated using (auth.uid() = auteur_id);
```

Ces règles font le travail essentiel : **un visiteur non connecté ne voit
rien**, et personne ne peut publier sous le nom d'un autre.

---

## 3. Activer le direct dans le chat

Sans cette étape, les nouveaux messages n'apparaissent qu'au rechargement de la
page.

**Database → Replication → `supabase_realtime`** : ajoutez la table
`public.messages`.

---

## 4. Régler l'inscription

Dans **Authentication → Providers → Email** :

- laissez « Confirm email » activé pour vérifier les adresses (le membre reçoit
  un lien et se connecte ensuite) ;
- ou désactivez-le si vous préférez une inscription immédiate.

Dans **Authentication → URL Configuration**, indiquez l'adresse publique du
site dans *Site URL* (par exemple `https://<compte>.github.io/astrotarget/`),
sans quoi les liens de confirmation renverront vers `localhost`.

---

## 5. Brancher le site

Ouvrez `site/js/config.js` et remplacez :

```js
modeMembre: "supabase",

supabase: {
  url: "https://xxxxxxxx.supabase.co",
  cleAnon: "eyJhbGciOi...",       // clé anon public
},

codeInscription: "VOTRE-CODE",     // code remis aux adhérents
```

Rechargez la page « Espace membres » : le bandeau doit annoncer un **espace
partagé** et non le mode démonstration.

---

## 6. Vérifier

1. Créez un compte depuis le site, avec le code d'accès du club.
2. Ajoutez un document : il doit apparaître dans **Table Editor → documents**.
3. Ouvrez le site sur un autre appareil, connectez-vous avec un second compte :
   les messages doivent arriver sans rechargement.
4. Déconnectez-vous et vérifiez qu'on ne voit plus ni documents ni messages.

---

## Bon à savoir

- **Le code d'accès n'est pas une sécurité.** Il est vérifié dans le
  navigateur : il décourage les curieux, rien de plus. La vraie protection est
  l'obligation d'avoir un compte confirmé (règles RLS ci-dessus).
- **Modération.** Un message publié reste visible par tous les membres. En cas
  de dérapage, supprimez la ligne depuis **Table Editor → messages**.
- **Sauvegarde.** Supabase permet d'exporter les tables en CSV ; pensez-y une
  fois par saison si la base documentaire prend de la valeur.
- **Données personnelles.** Le site stocke un nom d'affichage et une adresse
  e-mail. Prévenez les adhérents et supprimez le compte de qui le demande
  (**Authentication → Users**).
