/* =========================================================
   Comportements communs à toutes les pages du site.
   Aucun framework, aucune étape de compilation.
   ========================================================= */
(function () {
  "use strict";

  /* --- Menu mobile ------------------------------------------------------ */
  var burger = document.querySelector("[data-burger]");
  var nav = document.querySelector("[data-nav]");

  if (burger && nav) {
    var petitEcran = window.matchMedia("(max-width: 880px)");

    function appliquerEtatNav() {
      if (petitEcran.matches) {
        nav.hidden = burger.getAttribute("aria-expanded") !== "true";
      } else {
        nav.hidden = false;
      }
    }

    burger.addEventListener("click", function () {
      var ouvert = burger.getAttribute("aria-expanded") === "true";
      burger.setAttribute("aria-expanded", String(!ouvert));
      appliquerEtatNav();
    });

    petitEcran.addEventListener("change", function () {
      burger.setAttribute("aria-expanded", "false");
      appliquerEtatNav();
    });

    appliquerEtatNav();
  }

  /* --- Année courante dans le pied de page ------------------------------ */
  document.querySelectorAll("[data-annee]").forEach(function (el) {
    el.textContent = String(new Date().getFullYear());
  });

  /* --- Ciel étoilé du héros --------------------------------------------- */
  var canvas = document.querySelector("[data-ciel]");
  if (canvas && canvas.getContext) {
    dessinerCiel(canvas);
    var minuteur;
    window.addEventListener("resize", function () {
      clearTimeout(minuteur);
      minuteur = setTimeout(function () { dessinerCiel(canvas); }, 200);
    });
  }

  function dessinerCiel(cvs) {
    var parent = cvs.parentElement;
    var largeur = parent.offsetWidth;
    var hauteur = parent.offsetHeight;
    if (!largeur || !hauteur) return;

    var ratio = Math.min(window.devicePixelRatio || 1, 2);
    cvs.width = largeur * ratio;
    cvs.height = hauteur * ratio;
    cvs.style.width = largeur + "px";
    cvs.style.height = hauteur + "px";

    var ctx = cvs.getContext("2d");
    ctx.scale(ratio, ratio);
    ctx.clearRect(0, 0, largeur, hauteur);

    // Densité constante quelle que soit la taille de l'écran.
    var nombre = Math.round((largeur * hauteur) / 5200);
    for (var i = 0; i < nombre; i++) {
      var x = Math.random() * largeur;
      var y = Math.random() * hauteur;
      var r = Math.random() * 1.25 + 0.25;
      var a = Math.random() * 0.65 + 0.18;

      // Quelques étoiles légèrement colorées, comme dans un vrai champ.
      var teinte = Math.random();
      var couleur = teinte > 0.93 ? "255, 214, 170"
                  : teinte > 0.86 ? "190, 214, 255"
                  : "255, 255, 255";

      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(" + couleur + "," + a + ")";
      ctx.fill();

      // Halo sur les plus brillantes.
      if (r > 1.2) {
        ctx.beginPath();
        ctx.arc(x, y, r * 3.4, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(" + couleur + ",0.06)";
        ctx.fill();
      }
    }
  }

  /* --- Agenda : rendu partagé accueil / page agenda --------------------- */
  var MOIS = ["janv.", "févr.", "mars", "avril", "mai", "juin",
              "juil.", "août", "sept.", "oct.", "nov.", "déc."];

  var TYPES = {
    observation: { libelle: "Observation", classe: "etiquette--or" },
    atelier:     { libelle: "Atelier",     classe: "etiquette--violet" },
    conference:  { libelle: "Conférence",  classe: "" },
    initiation:  { libelle: "Initiation",  classe: "etiquette--vert" },
    sortie:      { libelle: "Sortie",      classe: "etiquette--vert" },
  };

  window.Agenda = {
    /**
     * Rend les évènements dans un conteneur.
     * @param {HTMLElement} cible
     * @param {{limite?: number, passes?: boolean}} options
     *        passes = true : uniquement les séances déjà écoulées,
     *        de la plus récente à la plus ancienne.
     */
    rendre: function (cible, options) {
      options = options || {};
      var tous = (window.EVENEMENTS || []).slice();

      var maintenant = new Date();

      var liste = tous.filter(function (ev) {
        var termine = new Date(ev.date + "T23:59:59") < maintenant;
        return options.passes ? termine : !termine;
      });

      liste.sort(function (a, b) {
        if (a.date === b.date) return 0;
        // À venir : du plus proche au plus lointain. Passées : l'inverse.
        return (a.date < b.date ? -1 : 1) * (options.passes ? -1 : 1);
      });

      if (options.limite) liste = liste.slice(0, options.limite);

      if (!liste.length) {
        cible.innerHTML = options.passes
          ? '<p class="texte-doux">Aucune séance passée à afficher pour cette saison.</p>'
          : '<p class="texte-doux">Aucune date programmée pour le moment. ' +
            'Le programme de la saison est publié à chaque rentrée — ' +
            '<a href="adherer.html">écrivez-nous</a> pour être prévenu.</p>';
        return;
      }

      cible.innerHTML = liste.map(carteEvenement).join("");
    },
  };

  function carteEvenement(ev) {
    var d = new Date(ev.date + "T12:00:00");
    var type = TYPES[ev.type] || { libelle: ev.type || "Séance", classe: "" };
    var passe = d < new Date();

    return (
      '<article class="evenement' + (passe ? " evenement--passe" : "") + '">' +
        '<div class="evenement__date">' +
          '<div class="evenement__jour">' + d.getDate() + "</div>" +
          '<div class="evenement__mois">' + MOIS[d.getMonth()] + "</div>" +
          '<div class="evenement__annee">' + d.getFullYear() + "</div>" +
        "</div>" +
        "<div>" +
          '<span class="etiquette ' + type.classe + '">' + type.libelle + "</span>" +
          "<h3>" + echapper(ev.titre) + "</h3>" +
          '<div class="evenement__infos">' +
            "<span>🕘 " + echapper(ev.heure || "horaire à préciser") + "</span>" +
            "<span>📍 " + echapper(ev.lieu || "lieu à préciser") + "</span>" +
            (ev.public ? "<span>👥 " + echapper(ev.public) + "</span>" : "") +
          "</div>" +
          '<p class="texte-doux mb-0">' + echapper(ev.resume || "") + "</p>" +
        "</div>" +
      "</article>"
    );
  }

  function echapper(txt) {
    return String(txt).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  window.echapperHtml = echapper;

  /* --- Rendu automatique là où un conteneur est présent ----------------- */
  document.querySelectorAll("[data-agenda]").forEach(function (el) {
    window.Agenda.rendre(el, {
      limite: el.dataset.agendaLimite ? parseInt(el.dataset.agendaLimite, 10) : 0,
      passes: el.dataset.agendaPasses === "true",
    });
  });
})();
