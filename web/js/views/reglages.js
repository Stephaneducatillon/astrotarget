/** Réglages : accessibilité, saison, mode démonstration, données locales. */

import { h, vider, annonce } from '../util/dom.js';
import * as store from '../store.js';
import * as cache from '../data/cache.js';
import { invalider } from '../contexte.js';
import { carte } from './composants.js';
import { FUSEAU_LOCAL } from '../util/time.js';

export async function rendre(racine) {
  vider(racine);
  const pile = h('div', { class: 'pile' });
  racine.append(pile);

  const p = store.prefs();
  const profil = store.profil();

  const recharger = () => { invalider(); rendre(racine); };

  // ── Profil ───────────────────────────────────────────────────────────────
  const champPseudo = h('input', { class: 'champ', value: profil?.pseudo || '', maxlength: '24', placeholder: 'Ton pseudo' });
  pile.append(carte('Profil', h('div', {},
    h('label', { class: 'etiquette' }, 'Pseudo'),
    h('div', { class: 'rangee' }, champPseudo,
      h('button', {
        class: 'btn',
        onclick: () => {
          try { store.definirProfil(champPseudo.value); annonce('Pseudo enregistré.'); recharger(); }
          catch (e) { annonce(e.message); }
        },
      }, 'Enregistrer')),
    h('p', { class: 'note', style: { marginTop: 'var(--esp-3)' } },
      'Le profil, les pronostics et les ligues vivent dans ce navigateur. Rien n\'est envoyé à un '
      + 'serveur : ce site est purement statique. Vider les données du site les efface définitivement.'))));

  // ── Affichage ────────────────────────────────────────────────────────────
  pile.append(carte('Affichage', h('div', {},
    reglage('Thème',
      h('div', { class: 'segments' },
        [['auto', 'Système'], ['sombre', 'Sombre'], ['clair', 'Clair']].map(([v, lib]) =>
          h('button', {
            'aria-pressed': String((p.theme || 'auto') === v),
            onclick: () => { store.definirPref('theme', v); appliquerTheme(); recharger(); },
          }, lib))),
      'Le mode sombre est la story 1.12 du backlog.'),

    reglage('Motifs sur les couleurs d\'écurie',
      h('div', { class: 'segments' },
        [[false, 'Désactivé'], [true, 'Activé']].map(([v, lib]) =>
          h('button', {
            'aria-pressed': String(Boolean(p.motifs) === v),
            onclick: () => { store.definirPref('motifs', v); appliquerMotifs(); recharger(); },
          }, lib))),
      'Plusieurs paires d\'écuries sont indistinguables pour les daltoniens. Les motifs ajoutent '
      + 'rayures et points aux pastilles de couleur.'),

    reglage('Fuseau horaire', h('span', { class: 'mono pt-s' }, FUSEAU_LOCAL),
      'Détecté par le navigateur. Toutes les heures sont stockées en UTC et converties à l\'affichage.'))));

  // ── Données ──────────────────────────────────────────────────────────────
  const champSaison = h('input', {
    class: 'champ', type: 'number', min: '1950', max: '2100',
    value: String(p.saison || new Date().getUTCFullYear()), style: { maxWidth: '140px' },
  });

  pile.append(carte('Données', h('div', {},
    reglage('Saison affichée',
      h('div', { class: 'rangee' }, champSaison,
        h('button', {
          class: 'btn btn--fin',
          onclick: () => { store.definirPref('saison', Number(champSaison.value) || null); recharger(); },
        }, 'Appliquer'),
        h('button', {
          class: 'btn btn--fin btn--fantome',
          onclick: () => { store.definirPref('saison', null); recharger(); },
        }, 'Saison en cours')),
      'Permet de consulter une saison écoulée. Jolpica couvre 1950 → aujourd\'hui ; la télémétrie '
      + 'commence en 2018 (FastF1) et le live en 2023 (OpenF1).'),

    reglage('Mode démonstration',
      h('div', { class: 'segments' },
        [[false, 'Données réelles'], [true, 'Démonstration']].map(([v, lib]) =>
          h('button', {
            'aria-pressed': String(Boolean(p.demo) === v),
            onclick: () => { store.definirPref('demo', v); recharger(); },
          }, lib))),
      'Force le jeu de démonstration local : un week-end fictif, calé sur les jours à venir, '
      + 'qui permet de parcourir toute l\'interface sans réseau. Rien n\'y est officiel.'),

    h('div', { class: 'rangee', style: { marginTop: 'var(--esp-4)' } },
      h('button', {
        class: 'btn',
        onclick: () => { cache.vidercache(); invalider(); annonce('Cache vidé.'); recharger(); },
      }, 'Vider le cache des API'),
      h('button', {
        class: 'btn',
        onclick: () => {
          if (confirm('Effacer profil, pronostics et ligues ? Cette action est définitive.')) {
            store.reinitialiser(); cache.vidercache(); invalider(); recharger();
          }
        },
      }, 'Effacer mes données')))));
}

function reglage(libelle, controle, note) {
  return h('div', { class: 'ligne-pro' },
    h('div', { class: 'ligne-pro-lib' }, libelle,
      note ? h('div', { class: 'note' }, note) : null),
    h('div', { class: 'ligne-pro-ctrl' }, controle));
}

export function appliquerTheme() {
  const t = store.prefs().theme || 'auto';
  const racine = document.documentElement;
  // « auto » ne retire le tampon que si c'est nous qui l'avons posé : un hôte
  // peut avoir tamponné data-theme lui-même, et ce n'est pas à nous de l'effacer.
  if (t === 'auto') {
    if (racine.dataset.themeParNous) {
      racine.removeAttribute('data-theme');
      delete racine.dataset.themeParNous;
    }
    return;
  }
  racine.setAttribute('data-theme', t);
  racine.dataset.themeParNous = '1';
}

export function appliquerMotifs() {
  document.body.dataset.motifs = store.prefs().motifs ? '1' : '0';
}
