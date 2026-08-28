/* =========================================================
   Espace membre : connexion, base documentaire, chat.
   S'appuie uniquement sur l'interface exposée par backend.js.
   ========================================================= */
(function () {
  "use strict";

  var CFG = window.CONFIG || {};
  var ech = window.echapperHtml;

  var $ = function (sel) { return document.querySelector(sel); };

  var vueConnexion = $("[data-vue-connexion]");
  var vueEspace = $("[data-vue-espace]");
  if (!vueConnexion || !vueEspace) return;

  var membre = null;
  var salonCourant = (CFG.salons && CFG.salons[0] && CFG.salons[0].id) || "general";
  var arreterEcoute = null;

  /* --- Petits helpers d'affichage --------------------------------------- */

  function afficherMessage(el, texte, type) {
    if (!el) return;
    el.textContent = texte;
    el.className = "message message--" + (type || "info");
    el.hidden = !texte;
  }

  function initiales(nom) {
    return String(nom || "?")
      .trim()
      .split(/\s+/)
      .slice(0, 2)
      .map(function (mot) { return mot.charAt(0).toUpperCase(); })
      .join("");
  }

  function heure(iso) {
    var d = new Date(iso);
    if (isNaN(d)) return "";
    var aujourdhui = new Date();
    var memeJour =
      d.getDate() === aujourdhui.getDate() &&
      d.getMonth() === aujourdhui.getMonth() &&
      d.getFullYear() === aujourdhui.getFullYear();
    var h = d.toLocaleTimeString("fr-FR", { hour: "2-digit", minute: "2-digit" });
    return memeJour ? h : d.toLocaleDateString("fr-FR", { day: "numeric", month: "short" }) + " · " + h;
  }

  function dateCourte(iso) {
    var d = new Date(iso);
    return isNaN(d) ? "" : d.toLocaleDateString("fr-FR", { day: "numeric", month: "long", year: "numeric" });
  }

  function erreurLisible(e) {
    if (!e) return "Une erreur est survenue.";
    return e.attendue ? e.message : (e.message || "Une erreur est survenue.");
  }

  /* --- Bandeau expliquant le mode en cours ------------------------------ */

  function majBandeauMode() {
    var partage = window.Backend.nom === "supabase";
    var texte = partage
      ? "<strong>Espace partagé.</strong> Les comptes, les documents et les " +
        "messages sont hébergés sur le serveur du club : ce que vous publiez " +
        "est visible par les autres adhérents."
      : "<strong>Mode démonstration.</strong> Faute de serveur configuré, le " +
        "compte, les documents et les messages restent dans <em>ce " +
        "navigateur</em> : ils ne sont visibles par personne d'autre et " +
        "disparaissent si vous effacez les données du site. Voir " +
        "<code>site/SUPABASE.md</code> pour activer le partage réel.";

    // Le bandeau est présent dans les deux vues (connexion et espace).
    document.querySelectorAll("[data-bandeau-mode]").forEach(function (el) {
      el.innerHTML = texte;
    });
  }

  /* --- Bascule connexion / espace --------------------------------------- */

  function montrerEspace() {
    vueConnexion.hidden = true;
    vueEspace.hidden = false;

    $("[data-membre-nom]").textContent = membre.nom;
    $("[data-membre-email]").textContent = membre.email;
    $("[data-membre-avatar]").textContent = initiales(membre.nom);

    chargerDocuments();
    construireSalons();
    chargerMessages();
    brancherEcoute();
  }

  function montrerConnexion() {
    vueEspace.hidden = true;
    vueConnexion.hidden = false;
    if (arreterEcoute) { arreterEcoute(); arreterEcoute = null; }
  }

  /* --- Onglets connexion / inscription ---------------------------------- */

  document.querySelectorAll("[data-onglet-auth]").forEach(function (bouton) {
    bouton.addEventListener("click", function () {
      var cible = bouton.dataset.ongletAuth;
      document.querySelectorAll("[data-onglet-auth]").forEach(function (b) {
        b.setAttribute("aria-selected", String(b === bouton));
      });
      document.querySelectorAll("[data-panneau-auth]").forEach(function (p) {
        p.hidden = p.dataset.panneauAuth !== cible;
      });
    });
  });

  /* --- Formulaires ------------------------------------------------------ */

  var formConnexion = $("[data-form-connexion]");
  var msgConnexion = $("[data-msg-connexion]");

  formConnexion.addEventListener("submit", function (e) {
    e.preventDefault();
    var bouton = formConnexion.querySelector("button[type=submit]");
    bouton.disabled = true;
    afficherMessage(msgConnexion, "", "info");

    window.Backend
      .connexion(formConnexion.email.value, formConnexion.motdepasse.value)
      .then(function (m) {
        membre = m;
        formConnexion.reset();
        montrerEspace();
      })
      .catch(function (err) { afficherMessage(msgConnexion, erreurLisible(err), "erreur"); })
      .then(function () { bouton.disabled = false; });
  });

  var formInscription = $("[data-form-inscription]");
  var msgInscription = $("[data-msg-inscription]");

  formInscription.addEventListener("submit", function (e) {
    e.preventDefault();
    var bouton = formInscription.querySelector("button[type=submit]");
    bouton.disabled = true;
    afficherMessage(msgInscription, "", "info");

    window.Backend
      .inscription({
        nom: formInscription.nom.value,
        email: formInscription.email.value,
        mdp: formInscription.motdepasse.value,
        code: formInscription.code.value,
      })
      .then(function (m) {
        membre = m;
        formInscription.reset();
        montrerEspace();
      })
      .catch(function (err) { afficherMessage(msgInscription, erreurLisible(err), "erreur"); })
      .then(function () { bouton.disabled = false; });
  });

  $("[data-deconnexion]").addEventListener("click", function () {
    window.Backend.deconnexion().then(function () {
      membre = null;
      montrerConnexion();
    });
  });

  /* --- Base documentaire ------------------------------------------------ */

  var listeDocs = $("[data-liste-docs]");
  var champRecherche = $("[data-recherche-docs]");
  var filtreCategorie = $("[data-filtre-categorie]");
  var tousLesDocs = [];

  function chargerDocuments() {
    listeDocs.innerHTML = '<p class="texte-faible">Chargement…</p>';
    window.Backend
      .listerDocuments()
      .then(function (docs) {
        tousLesDocs = docs;
        remplirCategories(docs);
        rendreDocuments();
      })
      .catch(function (err) {
        listeDocs.innerHTML = '<p class="texte-faible">' + ech(erreurLisible(err)) + "</p>";
      });
  }

  function remplirCategories(docs) {
    var categories = [];
    docs.forEach(function (d) {
      if (d.categorie && categories.indexOf(d.categorie) === -1) categories.push(d.categorie);
    });
    categories.sort();
    var actuelle = filtreCategorie.value;
    filtreCategorie.innerHTML =
      '<option value="">Toutes les catégories</option>' +
      categories.map(function (c) { return '<option value="' + ech(c) + '">' + ech(c) + "</option>"; }).join("");
    if (categories.indexOf(actuelle) !== -1) filtreCategorie.value = actuelle;
  }

  function rendreDocuments() {
    var recherche = champRecherche.value.trim().toLowerCase();
    var categorie = filtreCategorie.value;

    var visibles = tousLesDocs.filter(function (d) {
      if (categorie && d.categorie !== categorie) return false;
      if (!recherche) return true;
      return (d.titre + " " + d.description + " " + d.categorie).toLowerCase().indexOf(recherche) !== -1;
    });

    if (!visibles.length) {
      listeDocs.innerHTML =
        '<p class="texte-faible">Aucun document ne correspond à cette recherche.</p>';
      return;
    }

    listeDocs.innerHTML = visibles.map(carteDoc).join("");

    listeDocs.querySelectorAll("[data-supprimer-doc]").forEach(function (b) {
      b.addEventListener("click", function () {
        if (!window.confirm("Retirer « " + b.dataset.titre + " » de la base documentaire ?")) return;
        window.Backend
          .supprimerDocument(b.dataset.supprimerDoc)
          .then(chargerDocuments)
          .catch(function (err) { window.alert(erreurLisible(err)); });
      });
    });
  }

  function carteDoc(d) {
    var sien = d.auteurId && membre && d.auteurId === membre.id;
    var icone = d.url ? "🔗" : "📄";

    return (
      '<article class="doc">' +
        '<div class="doc__icone" aria-hidden="true">' + icone + "</div>" +
        '<div class="doc__corps">' +
          '<div class="doc__titre">' +
            (d.url
              ? '<a href="' + ech(d.url) + '" target="_blank" rel="noopener noreferrer">' + ech(d.titre) + " ↗</a>"
              : ech(d.titre)) +
          "</div>" +
          (d.description ? '<div class="doc__desc">' + ech(d.description) + "</div>" : "") +
          '<div class="doc__meta">' +
            '<span class="etiquette">' + ech(d.categorie || "Divers") + "</span>" +
            "<span>Déposé par " + ech(d.auteur || "le club") + "</span>" +
            (d.creeLe ? "<span>" + ech(dateCourte(d.creeLe)) + "</span>" : "") +
            (sien
              ? '<button class="btn btn--danger btn--petit" data-supprimer-doc="' + ech(d.id) +
                '" data-titre="' + ech(d.titre) + '">Retirer</button>'
              : "") +
          "</div>" +
        "</div>" +
      "</article>"
    );
  }

  champRecherche.addEventListener("input", rendreDocuments);
  filtreCategorie.addEventListener("change", rendreDocuments);

  var formDoc = $("[data-form-doc]");
  var msgDoc = $("[data-msg-doc]");

  formDoc.addEventListener("submit", function (e) {
    e.preventDefault();
    var bouton = formDoc.querySelector("button[type=submit]");
    bouton.disabled = true;

    window.Backend
      .ajouterDocument({
        titre: formDoc.titre.value,
        categorie: formDoc.categorie.value,
        description: formDoc.description.value,
        url: formDoc.url.value,
      })
      .then(function () {
        formDoc.reset();
        afficherMessage(msgDoc, "Document ajouté à la base.", "succes");
        setTimeout(function () { afficherMessage(msgDoc, "", "info"); }, 4000);
        chargerDocuments();
      })
      .catch(function (err) { afficherMessage(msgDoc, erreurLisible(err), "erreur"); })
      .then(function () { bouton.disabled = false; });
  });

  /* --- Chat -------------------------------------------------------------- */

  var zoneSalons = $("[data-salons]");
  var fil = $("[data-fil]");
  var formChat = $("[data-form-chat]");
  var champMessage = formChat.querySelector("textarea");

  function construireSalons() {
    var salons = CFG.salons || [{ id: "general", nom: "Général" }];
    zoneSalons.innerHTML = salons
      .map(function (s) {
        return (
          '<button type="button" class="chat__salon" data-salon="' + ech(s.id) + '" ' +
          'aria-pressed="' + (s.id === salonCourant) + '">' + ech(s.nom) + "</button>"
        );
      })
      .join("");

    zoneSalons.querySelectorAll("[data-salon]").forEach(function (b) {
      b.addEventListener("click", function () {
        salonCourant = b.dataset.salon;
        zoneSalons.querySelectorAll("[data-salon]").forEach(function (autre) {
          autre.setAttribute("aria-pressed", String(autre === b));
        });
        chargerMessages();
        brancherEcoute();
      });
    });
  }

  function chargerMessages(garderPosition) {
    var enBas = !garderPosition || fil.scrollTop + fil.clientHeight >= fil.scrollHeight - 60;

    window.Backend
      .listerMessages(salonCourant)
      .then(function (messages) {
        if (!messages.length) {
          fil.innerHTML =
            '<p class="chat__vide">Aucun message dans ce salon.<br>' +
            "Lancez la discussion — la prochaine éclaircie ne s'annonce pas toute seule.</p>";
          return;
        }
        fil.innerHTML = messages.map(bulle).join("");
        if (enBas) fil.scrollTop = fil.scrollHeight;
      })
      .catch(function (err) {
        fil.innerHTML = '<p class="chat__vide">' + ech(erreurLisible(err)) + "</p>";
      });
  }

  function bulle(m) {
    var moi = membre && m.auteurId === membre.id;
    return (
      '<div class="msg' + (moi ? " msg--moi" : "") + '">' +
        '<div class="avatar msg__avatar" aria-hidden="true">' + ech(initiales(m.auteur)) + "</div>" +
        '<div class="msg__bulle">' +
          '<div class="msg__auteur">' + ech(moi ? "Vous" : m.auteur) + "</div>" +
          '<div class="msg__texte">' + ech(m.texte) + "</div>" +
          '<div class="msg__heure">' + ech(heure(m.creeLe)) + "</div>" +
        "</div>" +
      "</div>"
    );
  }

  function brancherEcoute() {
    if (arreterEcoute) arreterEcoute();
    arreterEcoute = window.Backend.ecouterMessages(salonCourant, function () {
      chargerMessages(true);
    });
  }

  formChat.addEventListener("submit", function (e) {
    e.preventDefault();
    var texte = champMessage.value;
    if (!texte.trim()) return;

    champMessage.value = "";
    window.Backend
      .envoyerMessage(salonCourant, texte)
      .then(function () { chargerMessages(); })
      .catch(function (err) {
        champMessage.value = texte; // on ne perd pas ce qui a été écrit
        window.alert(erreurLisible(err));
      });
  });

  // Entrée envoie, Maj+Entrée passe à la ligne.
  champMessage.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      formChat.requestSubmit ? formChat.requestSubmit() : formChat.dispatchEvent(new Event("submit", { cancelable: true }));
    }
  });

  /* --- Démarrage --------------------------------------------------------- */

  window.BackendPret
    .then(function () {
      majBandeauMode();
      return window.Backend.membreCourant();
    })
    .then(function (m) {
      if (m) {
        membre = m;
        montrerEspace();
      } else {
        montrerConnexion();
      }
    })
    .catch(function (err) {
      afficherMessage(msgConnexion, erreurLisible(err), "erreur");
      montrerConnexion();
    });
})();
