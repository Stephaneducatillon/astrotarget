/**
 * Amorçage et routage.
 *
 * Routage par hash : le site se déploie sur n'importe quel hébergement
 * statique, sans règle de réécriture côté serveur.
 */

import { h, vider, $ } from './util/dom.js';
import * as store from './store.js';
import { appliquerTheme, appliquerMotifs } from './views/reglages.js';

import * as hub from './views/hub.js';
import * as circuit from './views/circuit.js';
import * as pilote from './views/pilote.js';
import * as pronostic from './views/pronostic.js';
import * as debrief from './views/debrief.js';
import * as ligue from './views/ligue.js';
import * as comparateur from './views/comparateur.js';
import * as resultats from './views/resultats.js';
import * as sources from './views/sources.js';
import * as reglages from './views/reglages.js';

const ROUTES = [
  { motif: /^\/?$/,                        vue: (r) => hub.rendre(r),                       nav: 'hub',     titre: 'Hub week-end' },
  { motif: /^\/calendrier$/,               vue: (r) => resultats.rendreCalendrier(r),        nav: 'cal',     titre: 'Calendrier' },
  { motif: /^\/circuit(?:\/(.+))?$/,       vue: (r, m) => circuit.rendre(r, { id: m[1] }),   nav: 'cal',     titre: 'Fiche circuit' },
  { motif: /^\/pilotes$/,                  vue: (r) => pilote.rendreListe(r),                nav: 'pilotes', titre: 'Pilotes' },
  { motif: /^\/pilote\/(.+)$/,             vue: (r, m) => pilote.rendre(r, { id: m[1] }),    nav: 'pilotes', titre: 'Fiche pilote' },
  { motif: /^\/pronostic(?:\/(.+))?$/,     vue: (r, m) => pronostic.rendre(r, { id: m[1] }), nav: 'prono',   titre: 'Pronostic' },
  { motif: /^\/debrief(?:\/(.+))?$/,       vue: (r, m) => debrief.rendre(r, { id: m[1] }),   nav: 'prono',   titre: 'Débrief' },
  { motif: /^\/ligue(?:\/(.+))?$/,         vue: (r, m) => ligue.rendre(r, { code: m[1] }),   nav: 'ligue',   titre: 'Ligue' },
  { motif: /^\/comparateur$/,              vue: (r) => comparateur.rendre(r),                nav: 'cal',     titre: 'Comparateur' },
  { motif: /^\/resultats\/(\d+)\/(\d+)$/,  vue: (r, m) => resultats.rendre(r, { saison: m[1], round: m[2] }), nav: 'cal', titre: 'Résultats' },
  { motif: /^\/sources$/,                  vue: (r) => sources.rendre(r),                    nav: 'plus',    titre: 'Sources' },
  { motif: /^\/reglages$/,                 vue: (r) => reglages.rendre(r),                   nav: 'plus',    titre: 'Réglages' },
];

const NAV = [
  { cle: 'hub',     href: '#/',            label: 'Hub',      icone: 'M3 11 12 3l9 8v9a1 1 0 0 1-1 1h-5v-6H9v6H4a1 1 0 0 1-1-1z' },
  { cle: 'cal',     href: '#/calendrier',  label: 'Calendrier', icone: 'M7 3v3M17 3v3M3 9h18M5 5h14a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2z' },
  { cle: 'prono',   href: '#/pronostic',   label: 'Pronostic', icone: 'M12 2 15 9l7 .6-5.3 4.6L18.3 21 12 17.3 5.7 21l1.6-6.8L2 9.6 9 9z' },
  { cle: 'pilotes', href: '#/pilotes',     label: 'Pilotes',  icone: 'M12 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8zM4 21a8 8 0 0 1 16 0' },
  { cle: 'ligue',   href: '#/ligue',       label: 'Ligue',    icone: 'M4 20V10M10 20V4M16 20v-8M22 20h-20' },
];

const LIENS_DESKTOP = [
  ...NAV,
  { cle: 'plus', href: '#/sources', label: 'Sources' },
  { cle: 'reg',  href: '#/reglages', label: 'Réglages' },
];

let navCourante = 'hub';

function construireCoquille() {
  const entete = h('header', { class: 'entete' },
    h('div', { class: 'entete-in' },
      h('a', { class: 'logo', href: '#/' },
        h('span', { class: 'logo-marque', 'aria-hidden': 'true' }),
        'Paddock'),
      h('span', { class: 'entete-espace' }),
      h('nav', { class: 'nav-desktop', id: 'nav-desktop', 'aria-label': 'Navigation principale' },
        LIENS_DESKTOP.map((n) => h('a', { href: n.href, 'data-nav': n.cle }, n.label)))));

  const principal = h('main', { class: 'page', id: 'contenu', tabindex: '-1' });

  const navMobile = h('nav', { class: 'nav-mobile', 'aria-label': 'Navigation' },
    NAV.map((n) => h('a', { href: n.href, 'data-nav': n.cle },
      h('svg', { viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', 'stroke-width': '1.9', 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'aria-hidden': 'true' },
        h('path', { d: n.icone })),
      h('span', {}, n.label))));

  const annonceur = h('div', { id: 'annonceur', class: 'sr', role: 'status', 'aria-live': 'polite' });

  document.body.append(
    h('a', { class: 'saut-contenu', href: '#contenu' }, 'Aller au contenu'),
    entete, principal, navMobile, annonceur);

  return principal;
}

function marquerNav(cle) {
  navCourante = cle;
  document.querySelectorAll('[data-nav]').forEach((a) => {
    if (a.dataset.nav === cle) a.setAttribute('aria-current', 'page');
    else a.removeAttribute('aria-current');
  });
}

async function router(principal) {
  const chemin = location.hash.replace(/^#/, '') || '/';
  const route = ROUTES.find((r) => r.motif.test(chemin));

  hub.nettoyer();

  if (!route) {
    vider(principal).append(h('div', { class: 'etat etat--vide' },
      h('div', { class: 'etat-titre' }, 'Page introuvable'),
      h('p', { class: 'etat-detail' }, `Aucun écran ne correspond à « ${chemin} ».`),
      h('a', { class: 'btn btn--primaire', href: '#/' }, 'Retour au hub')));
    document.title = 'Paddock — page introuvable';
    return;
  }

  marquerNav(route.nav);
  document.title = `${route.titre} · Paddock`;
  const m = route.motif.exec(chemin);

  try {
    await route.vue(principal, m);
  } catch (e) {
    console.error(e);
    vider(principal).append(h('div', { class: 'etat etat--erreur', role: 'alert' },
      h('div', { class: 'etat-titre' }, 'Cet écran n\'a pas pu s\'afficher'),
      h('p', { class: 'etat-detail' }, e?.message || String(e)),
      h('button', { class: 'btn', onclick: () => router(principal) }, 'Réessayer')));
  }
  principal.focus({ preventScroll: true });
  window.scrollTo({ top: 0, behavior: 'instant' in window ? 'instant' : 'auto' });
}

function demarrer() {
  const principal = construireCoquille();
  appliquerTheme();
  appliquerMotifs();

  window.addEventListener('hashchange', () => router(principal));
  store.sAbonner(() => { appliquerTheme(); appliquerMotifs(); });

  router(principal);

  if ('serviceWorker' in navigator && location.protocol.startsWith('http')) {
    navigator.serviceWorker.register(new URL('../sw.js', import.meta.url))
      .catch(() => { /* hors ligne indisponible, le site reste utilisable */ });
  }
}

demarrer();
