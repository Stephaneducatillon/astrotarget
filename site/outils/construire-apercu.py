#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Assemble tout le site en UN seul fichier HTML autonome.

Utilité : `apercu.html` s'ouvre par un simple double-clic, s'envoie par
e-mail ou se dépose sur une clé USB. Pratique pour montrer le site au club
avant qu'il soit hébergé quelque part.

    python3 site/outils/construire-apercu.py

Le fichier produit contient les sept pages (navigation par ancres), le style
et les scripts intégrés. Il n'a besoin d'aucune connexion.
Le site normal, lui, reste le dossier `site/` : c'est lui qu'on met en ligne.
"""

import os
import re

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SORTIE = os.path.join(RACINE, "apercu.html")

PAGES = [
    ("index", "index.html"),
    ("club", "club.html"),
    ("activites", "activites.html"),
    ("agenda", "agenda.html"),
    ("galerie", "galerie.html"),
    ("adherer", "adherer.html"),
    ("membres", "membres.html"),
]


def lire(chemin):
    with open(os.path.join(RACINE, chemin), encoding="utf-8") as f:
        return f.read()


def corps_principal(html):
    """Contenu du <main> d'une page."""
    m = re.search(r'<main id="contenu">(.*?)</main>', html, re.S)
    if not m:
        raise SystemExit("Pas de <main> trouvé — le gabarit des pages a changé.")
    return m.group(1)


def liens_internes(html):
    """Remplace les liens vers les fichiers .html par des ancres."""
    for cle, fichier in PAGES:
        html = html.replace('href="%s#' % fichier, 'href="#%s-' % cle)
        html = html.replace('href="%s"' % fichier, 'href="#%s"' % cle)
    return html


def construire():
    index = lire("index.html")

    # En-tête et pied de page : repris tels quels depuis l'accueil.
    entete = re.search(r"<header class=\"entete\">.*?</header>", index, re.S).group(0)
    pied = re.search(r"<footer class=\"pied\">.*?</footer>", index, re.S).group(0)

    sections = []
    for cle, fichier in PAGES:
        contenu = corps_principal(lire(fichier))
        sections.append(
            '<div class="page" data-page="%s"%s>\n%s\n</div>'
            % (cle, "" if cle == "index" else " hidden", contenu)
        )

    parties = [
        "<!doctype html>",
        '<html lang="fr">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>Club d'astronomie — MJC de Douai</title>",
        '<link rel="icon" href="data:image/svg+xml,'
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'>"
        "<text y='.9em' font-size='90'>🔭</text></svg>\">",
        # Polices du site ; en cas d'absence de réseau, la pile système prend
        # le relais sans casser la mise en page.
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2'
        '?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600;700'
        '&display=swap">',
        "<style>\n%s\n</style>" % lire("css/style.css"),
        "</head>",
        "<body>",
        '<a class="saut-contenu" href="#contenu">Aller au contenu</a>',
        liens_internes(entete),
        '<main id="contenu">',
        liens_internes("\n".join(sections)),
        "</main>",
        liens_internes(pied),
    ]

    for script in ("js/config.js", "data/evenements.js", "js/app.js",
                   "js/backend.js", "js/membres.js"):
        parties.append("<script>\n%s\n</script>" % lire(script))

    parties.append(ROUTEUR)
    parties.append("</body>\n</html>")

    html = "\n".join(parties)
    with open(SORTIE, "w", encoding="utf-8") as f:
        f.write(html)
    print("écrit %s (%d Ko)" % (SORTIE, len(html.encode("utf-8")) // 1024))


ROUTEUR = """<script>
/* Navigation entre les pages du fichier unique, par ancre. */
(function () {
  "use strict";
  var pages = document.querySelectorAll("[data-page]");
  var liens = document.querySelectorAll(".nav a, .pied a, .marque");

  function afficher(cle) {
    var connue = false;
    pages.forEach(function (p) {
      var visible = p.dataset.page === cle;
      p.hidden = !visible;
      if (visible) connue = true;
    });
    if (!connue) return afficher("index");

    liens.forEach(function (a) {
      var vise = (a.getAttribute("href") || "").replace("#", "");
      if (vise === cle && a.classList.contains("nav__membre") === (cle === "membres")) {
        a.setAttribute("aria-current", "page");
      } else {
        a.removeAttribute("aria-current");
      }
    });

    // Un canvas masqué a une taille nulle : on le redessine une fois visible.
    if (window.redessinerCiels) window.redessinerCiels();
    window.scrollTo(0, 0);
  }

  function depuisAdresse() {
    afficher((location.hash || "#index").slice(1).split("-")[0]);
  }

  window.addEventListener("hashchange", depuisAdresse);
  document.querySelector(".marque").setAttribute("href", "#index");
  depuisAdresse();
})();
</script>"""


if __name__ == "__main__":
    construire()
