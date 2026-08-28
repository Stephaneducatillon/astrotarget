/* =========================================================
   Couche d'accès de l'espace membre.

   Une seule interface, deux implémentations :
     • BackendLocal    — localStorage, pour essayer le site sans serveur
     • BackendSupabase — comptes et données réellement partagés

   Le reste du code (membres.js) ne connaît que cette interface :
     connexion(email, mdp)              -> Promise<membre>
     inscription({nom, email, mdp, code})-> Promise<membre>
     deconnexion()                      -> Promise
     membreCourant()                    -> Promise<membre|null>
     listerDocuments()                  -> Promise<doc[]>
     ajouterDocument(doc)               -> Promise<doc>
     supprimerDocument(id)              -> Promise
     listerMessages(salon)              -> Promise<message[]>
     envoyerMessage(salon, texte)       -> Promise<message>
     ecouterMessages(salon, rappel)     -> fonction d'arrêt
   ========================================================= */
(function () {
  "use strict";

  var CFG = window.CONFIG || {};

  /* ==== Utilitaires partagés ============================================ */

  function identifiant() {
    return Date.now().toString(36) + "-" + Math.random().toString(36).slice(2, 9);
  }

  function normaliserEmail(email) {
    return String(email || "").trim().toLowerCase();
  }

  function ErreurMembre(message) {
    var e = new Error(message);
    e.attendue = true; // erreur destinée à être affichée telle quelle
    return e;
  }

  /* ==== Implémentation locale (localStorage) ============================ */

  var BackendLocal = (function () {
    var CLE_MEMBRES = "mjc-astro:membres";
    var CLE_SESSION = "mjc-astro:session";
    var CLE_DOCS = "mjc-astro:documents";
    var CLE_MESSAGES = "mjc-astro:messages";

    function lire(cle, defaut) {
      try {
        var brut = localStorage.getItem(cle);
        return brut ? JSON.parse(brut) : defaut;
      } catch (e) {
        return defaut;
      }
    }

    function ecrire(cle, valeur) {
      try {
        localStorage.setItem(cle, JSON.stringify(valeur));
        return true;
      } catch (e) {
        return false;
      }
    }

    /* Empreinte du mot de passe.
       PBKDF2-SHA256 via WebCrypto quand il est disponible (page servie en
       https ou en localhost). Sur une page ouverte en file://, crypto.subtle
       n'existe pas : on retombe sur une empreinte simple, suffisante pour la
       démo mais qui ne protège rien — d'où l'avertissement affiché. */
    var CRYPTO_DISPO = !!(window.crypto && window.crypto.subtle);

    function octetsVersHex(buffer) {
      return Array.prototype.map
        .call(new Uint8Array(buffer), function (o) {
          return ("00" + o.toString(16)).slice(-2);
        })
        .join("");
    }

    function empreinte(motDePasse, sel) {
      if (!CRYPTO_DISPO) {
        // Repli non cryptographique, uniquement pour le mode démo hors https.
        var h = 5381;
        var chaine = sel + "|" + motDePasse;
        for (var i = 0; i < chaine.length; i++) {
          h = ((h << 5) + h + chaine.charCodeAt(i)) | 0;
        }
        return Promise.resolve("repli:" + (h >>> 0).toString(16));
      }

      var encodeur = new TextEncoder();
      return window.crypto.subtle
        .importKey("raw", encodeur.encode(motDePasse), "PBKDF2", false, ["deriveBits"])
        .then(function (cle) {
          return window.crypto.subtle.deriveBits(
            {
              name: "PBKDF2",
              salt: encodeur.encode(sel),
              iterations: 150000,
              hash: "SHA-256",
            },
            cle,
            256
          );
        })
        .then(octetsVersHex);
    }

    function membrePublic(m) {
      if (!m) return null;
      return { id: m.id, nom: m.nom, email: m.email, inscritLe: m.inscritLe };
    }

    function documentsParDefaut() {
      var maintenant = new Date().toISOString();
      return [
        {
          id: "doc-demo-1",
          titre: "Carte du ciel du mois",
          categorie: "Cartes du ciel",
          description:
            "Carte tournante à imprimer et repères de la voûte céleste pour le mois en cours.",
          url: "https://www.stelvision.com/astro/carte-du-ciel/",
          auteur: "Le club",
          creeLe: maintenant,
          protege: true,
        },
        {
          id: "doc-demo-2",
          titre: "Catalogue Messier commenté",
          categorie: "Catalogues",
          description:
            "Les 110 objets Messier, leur constellation, leur magnitude et la saison où les chercher.",
          url: "https://fr.wikipedia.org/wiki/Objet_du_catalogue_de_Messier",
          auteur: "Le club",
          creeLe: maintenant,
          protege: true,
        },
        {
          id: "doc-demo-3",
          titre: "Stellarium — logiciel de planétarium",
          categorie: "Logiciels",
          description:
            "Planétarium libre et gratuit : préparer une séance, identifier un objet, simuler le ciel d'une date.",
          url: "https://stellarium.org/fr/",
          auteur: "Le club",
          creeLe: maintenant,
          protege: true,
        },
        {
          id: "doc-demo-4",
          titre: "Fiche : préparer sa première soirée",
          categorie: "Fiches pratiques",
          description:
            "Vêtements, lampe rouge, adaptation à l'obscurité, liste d'objets faciles pour débuter. Document interne du club.",
          url: "",
          auteur: "Le club",
          creeLe: maintenant,
          protege: true,
        },
      ];
    }

    function messagesParDefaut() {
      return [];
    }

    return {
      nom: "local",

      init: function () {
        if (lire(CLE_DOCS, null) === null) ecrire(CLE_DOCS, documentsParDefaut());
        if (lire(CLE_MESSAGES, null) === null) ecrire(CLE_MESSAGES, messagesParDefaut());
        return Promise.resolve();
      },

      /* --- Comptes ------------------------------------------------------ */
      inscription: function (donnees) {
        var email = normaliserEmail(donnees.email);
        var nom = String(donnees.nom || "").trim();

        if (nom.length < 2) return Promise.reject(ErreurMembre("Indiquez votre nom (2 caractères minimum)."));
        if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) return Promise.reject(ErreurMembre("Adresse e-mail invalide."));
        if (String(donnees.mdp || "").length < 8) return Promise.reject(ErreurMembre("Le mot de passe doit faire au moins 8 caractères."));
        if (String(donnees.code || "").trim().toUpperCase() !== String(CFG.codeInscription || "").toUpperCase()) {
          return Promise.reject(ErreurMembre("Code d'accès incorrect. Il est remis aux adhérents lors de l'inscription."));
        }

        var membres = lire(CLE_MEMBRES, []);
        if (membres.some(function (m) { return m.email === email; })) {
          return Promise.reject(ErreurMembre("Un compte existe déjà pour cette adresse."));
        }

        var sel = identifiant();
        return empreinte(donnees.mdp, sel).then(function (hash) {
          var membre = {
            id: identifiant(),
            nom: nom,
            email: email,
            sel: sel,
            hash: hash,
            inscritLe: new Date().toISOString(),
          };
          membres.push(membre);
          if (!ecrire(CLE_MEMBRES, membres)) {
            throw ErreurMembre("Le navigateur refuse d'enregistrer les données (mode privé ?).");
          }
          ecrire(CLE_SESSION, membre.id);
          return membrePublic(membre);
        });
      },

      connexion: function (email, mdp) {
        var cible = normaliserEmail(email);
        var membres = lire(CLE_MEMBRES, []);
        var membre = membres.filter(function (m) { return m.email === cible; })[0];

        // Message volontairement identique dans les deux cas.
        if (!membre) return Promise.reject(ErreurMembre("Adresse ou mot de passe incorrect."));

        return empreinte(mdp, membre.sel).then(function (hash) {
          if (hash !== membre.hash) throw ErreurMembre("Adresse ou mot de passe incorrect.");
          ecrire(CLE_SESSION, membre.id);
          return membrePublic(membre);
        });
      },

      deconnexion: function () {
        try { localStorage.removeItem(CLE_SESSION); } catch (e) {}
        return Promise.resolve();
      },

      membreCourant: function () {
        var id = lire(CLE_SESSION, null);
        if (!id) return Promise.resolve(null);
        var membre = lire(CLE_MEMBRES, []).filter(function (m) { return m.id === id; })[0];
        return Promise.resolve(membrePublic(membre));
      },

      /* --- Base documentaire -------------------------------------------- */
      listerDocuments: function () {
        var docs = lire(CLE_DOCS, []).slice();
        docs.sort(function (a, b) { return a.creeLe < b.creeLe ? 1 : -1; });
        return Promise.resolve(docs);
      },

      ajouterDocument: function (doc) {
        return this.membreCourant().then(function (membre) {
          if (!membre) throw ErreurMembre("Connectez-vous pour ajouter un document.");
          if (!String(doc.titre || "").trim()) throw ErreurMembre("Le titre est obligatoire.");

          var docs = lire(CLE_DOCS, []);
          var nouveau = {
            id: identifiant(),
            titre: String(doc.titre).trim(),
            categorie: String(doc.categorie || "Divers").trim(),
            description: String(doc.description || "").trim(),
            url: String(doc.url || "").trim(),
            auteur: membre.nom,
            auteurId: membre.id,
            creeLe: new Date().toISOString(),
          };
          docs.push(nouveau);
          ecrire(CLE_DOCS, docs);
          return nouveau;
        });
      },

      supprimerDocument: function (id) {
        return this.membreCourant().then(function (membre) {
          if (!membre) throw ErreurMembre("Connectez-vous d'abord.");
          var docs = lire(CLE_DOCS, []);
          var cible = docs.filter(function (d) { return d.id === id; })[0];
          if (!cible) return;
          if (cible.protege || (cible.auteurId && cible.auteurId !== membre.id)) {
            throw ErreurMembre("Vous ne pouvez retirer que les documents que vous avez déposés.");
          }
          ecrire(CLE_DOCS, docs.filter(function (d) { return d.id !== id; }));
        });
      },

      /* --- Chat ---------------------------------------------------------- */
      listerMessages: function (salon) {
        var messages = lire(CLE_MESSAGES, []).filter(function (m) { return m.salon === salon; });
        messages.sort(function (a, b) { return a.creeLe < b.creeLe ? -1 : 1; });
        return Promise.resolve(messages);
      },

      envoyerMessage: function (salon, texte) {
        return this.membreCourant().then(function (membre) {
          if (!membre) throw ErreurMembre("Connectez-vous pour écrire.");
          var contenu = String(texte || "").trim();
          if (!contenu) throw ErreurMembre("Message vide.");
          if (contenu.length > 2000) throw ErreurMembre("Message trop long (2000 caractères maximum).");

          var messages = lire(CLE_MESSAGES, []);
          var message = {
            id: identifiant(),
            salon: salon,
            texte: contenu,
            auteur: membre.nom,
            auteurId: membre.id,
            creeLe: new Date().toISOString(),
          };
          messages.push(message);
          // On borne l'historique local pour ne pas saturer le stockage.
          if (messages.length > 500) messages = messages.slice(-500);
          ecrire(CLE_MESSAGES, messages);
          return message;
        });
      },

      /* Synchronisation entre les onglets d'un même navigateur : c'est tout
         ce qu'un site sans serveur peut offrir. */
      ecouterMessages: function (salon, rappel) {
        function surStockage(e) {
          if (e.key === CLE_MESSAGES) rappel();
        }
        window.addEventListener("storage", surStockage);
        return function () { window.removeEventListener("storage", surStockage); };
      },
    };
  })();

  /* ==== Implémentation Supabase ========================================= */

  var BackendSupabase = (function () {
    var client = null;

    function chargerSdk() {
      if (window.supabase && window.supabase.createClient) return Promise.resolve();
      return new Promise(function (resoudre, rejeter) {
        var s = document.createElement("script");
        s.src = "https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2/dist/umd/supabase.js";
        s.onload = resoudre;
        s.onerror = function () { rejeter(ErreurMembre("Impossible de charger la bibliothèque Supabase.")); };
        document.head.appendChild(s);
      });
    }

    function membrePublic(user) {
      if (!user) return null;
      var meta = user.user_metadata || {};
      return {
        id: user.id,
        nom: meta.nom || (user.email || "").split("@")[0],
        email: user.email,
        inscritLe: user.created_at,
      };
    }

    function verifier(reponse) {
      if (reponse.error) {
        var e = ErreurMembre(reponse.error.message);
        throw e;
      }
      return reponse.data;
    }

    return {
      nom: "supabase",

      init: function () {
        return chargerSdk().then(function () {
          client = window.supabase.createClient(CFG.supabase.url, CFG.supabase.cleAnon);
        });
      },

      inscription: function (donnees) {
        if (String(donnees.code || "").trim().toUpperCase() !== String(CFG.codeInscription || "").toUpperCase()) {
          return Promise.reject(ErreurMembre("Code d'accès incorrect."));
        }
        return client.auth
          .signUp({
            email: normaliserEmail(donnees.email),
            password: donnees.mdp,
            options: { data: { nom: String(donnees.nom || "").trim() } },
          })
          .then(verifier)
          .then(function (data) {
            if (!data.session) {
              throw ErreurMembre(
                "Compte créé. Confirmez votre adresse via l'e-mail reçu, puis connectez-vous."
              );
            }
            return membrePublic(data.user);
          });
      },

      connexion: function (email, mdp) {
        return client.auth
          .signInWithPassword({ email: normaliserEmail(email), password: mdp })
          .then(verifier)
          .then(function (data) { return membrePublic(data.user); });
      },

      deconnexion: function () {
        return client.auth.signOut();
      },

      membreCourant: function () {
        return client.auth.getUser().then(function (r) {
          return r.data && r.data.user ? membrePublic(r.data.user) : null;
        });
      },

      listerDocuments: function () {
        return client
          .from("documents")
          .select("*")
          .order("cree_le", { ascending: false })
          .then(verifier)
          .then(function (lignes) { return (lignes || []).map(depuisLigneDoc); });
      },

      ajouterDocument: function (doc) {
        return this.membreCourant().then(function (membre) {
          if (!membre) throw ErreurMembre("Connectez-vous pour ajouter un document.");
          return client
            .from("documents")
            .insert({
              titre: String(doc.titre || "").trim(),
              categorie: String(doc.categorie || "Divers").trim(),
              description: String(doc.description || "").trim(),
              url: String(doc.url || "").trim(),
              auteur: membre.nom,
              auteur_id: membre.id,
            })
            .select()
            .single()
            .then(verifier)
            .then(depuisLigneDoc);
        });
      },

      supprimerDocument: function (id) {
        return client.from("documents").delete().eq("id", id).then(verifier);
      },

      listerMessages: function (salon) {
        return client
          .from("messages")
          .select("*")
          .eq("salon", salon)
          .order("cree_le", { ascending: true })
          .limit(200)
          .then(verifier)
          .then(function (lignes) { return (lignes || []).map(depuisLigneMessage); });
      },

      envoyerMessage: function (salon, texte) {
        return this.membreCourant().then(function (membre) {
          if (!membre) throw ErreurMembre("Connectez-vous pour écrire.");
          var contenu = String(texte || "").trim();
          if (!contenu) throw ErreurMembre("Message vide.");
          return client
            .from("messages")
            .insert({ salon: salon, texte: contenu, auteur: membre.nom, auteur_id: membre.id })
            .select()
            .single()
            .then(verifier)
            .then(depuisLigneMessage);
        });
      },

      ecouterMessages: function (salon, rappel) {
        var canal = client
          .channel("messages-" + salon)
          .on(
            "postgres_changes",
            { event: "INSERT", schema: "public", table: "messages", filter: "salon=eq." + salon },
            function () { rappel(); }
          )
          .subscribe();
        return function () { client.removeChannel(canal); };
      },
    };

    function depuisLigneDoc(l) {
      return {
        id: l.id,
        titre: l.titre,
        categorie: l.categorie,
        description: l.description,
        url: l.url,
        auteur: l.auteur,
        auteurId: l.auteur_id,
        creeLe: l.cree_le,
      };
    }

    function depuisLigneMessage(l) {
      return {
        id: l.id,
        salon: l.salon,
        texte: l.texte,
        auteur: l.auteur,
        auteurId: l.auteur_id,
        creeLe: l.cree_le,
      };
    }
  })();

  /* ==== Sélection du backend ============================================ */

  var supabaseConfigure =
    CFG.modeMembre === "supabase" &&
    CFG.supabase &&
    CFG.supabase.url &&
    CFG.supabase.cleAnon;

  var choisi = supabaseConfigure ? BackendSupabase : BackendLocal;

  window.Backend = choisi;

  // Promesse d'initialisation : membres.js attend `window.BackendPret` avant
  // de toucher à `window.Backend`, qui peut encore changer ci-dessous.
  window.BackendPret = choisi
    .init()
    .catch(function (e) {
      // Si Supabase est injoignable, la page reste utilisable en local
      // plutôt que de rester bloquée sur un écran vide.
      console.error("Espace membre : bascule en mode local.", e);
      window.Backend = BackendLocal;
      return BackendLocal.init();
    })
    .then(function () { return window.Backend; });
})();
