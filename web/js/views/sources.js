/**
 * Écran de traçabilité.
 *
 * Definition of Done n°1 du livrable 4 : toute donnée affichée est traçable
 * jusqu'à sa source. Cet écran est la contrepartie visible de cette règle,
 * et l'endroit où les limites sont écrites noir sur blanc.
 */

import { h, vider, badgeSource } from '../util/dom.js';
import { SOURCES, TTL, LIMITES, APP } from '../config.js';
import { carte } from './composants.js';

const MS = { 1800000: '30 min', 3600000: '1 h', 21600000: '6 h', 86400000: '24 h', 604800000: '7 j', 7200000: '2 h', 10800000: '3 h' };
const duree = (ms) => (ms === 0 ? 'aucun cache' : MS[ms] || `${Math.round(ms / 60000)} min`);

export async function rendre(racine) {
  vider(racine);
  const pile = h('div', { class: 'pile' });
  racine.append(pile);

  pile.append(h('section', { class: 'heros' },
    h('div', { class: 'heros-sur' }, 'Traçabilité'),
    h('h1', {}, 'D\'où vient chaque donnée'),
    h('p', { class: 'heros-lieu', style: { marginTop: 'var(--esp-3)' } },
      'Aucune donnée n\'est affichée sans que sa source, sa fraîcheur et sa fenêtre de calcul '
      + 'ne soient identifiables. Les badges présents sur chaque bloc renvoient à ce tableau.')));

  pile.append(carte('Sources', h('div', { class: 'tableau-enrobage' },
    h('table', { class: 't' },
      h('thead', {}, h('tr', {},
        h('th', {}, 'Code'), h('th', {}, 'Source'), h('th', {}, 'Couverture'),
        h('th', {}, 'Latence'), h('th', {}, 'Licence'))),
      h('tbody', {},
        Object.values(SOURCES).map((s) => h('tr', {},
          h('td', {}, badgeSource(s.code)),
          h('td', {}, s.url ? h('a', { href: s.url, target: '_blank', rel: 'noopener' }, s.nom) : s.nom),
          h('td', { class: 'pt-s txt-2' }, s.couverture),
          h('td', { class: 'pt-s txt-2' }, s.latence),
          h('td', { class: 'pt-s txt-3' }, s.licence))))))));

  pile.append(carte('Règle de préséance', h('div', {},
    h('p', { class: 'txt-2 pt-s' },
      'Si deux sources donnent la même donnée : ',
      h('b', {}, 'OpenF1'), ' prime pour le live, ',
      h('b', {}, 'Jolpica'), ' prime pour l\'historique et les classements officiels, ',
      h('b', {}, 'FastF1'), ' prime pour tout ce qui est dérivé du timing.'))));

  pile.append(carte('Fraîcheur et cache', h('div', {},
    h('div', { class: 'tableau-enrobage' },
      h('table', { class: 't' },
        h('thead', {}, h('tr', {}, h('th', {}, 'Donnée'), h('th', {}, 'Durée de cache'))),
        h('tbody', {},
          [['Calendrier, pilotes, écuries', TTL.calendrier],
            ['Classements', TTL.classement],
            ['Fiche circuit', TTL.circuit],
            ['Prévision météo (J-2 et moins)', TTL.meteo],
            ['Prévision météo (au-delà)', TTL.meteo_loin],
            ['Session live, direction de course', TTL.live],
            ['Résultats officiels', TTL.resultats],
            ['Indicateurs dérivés', TTL.derives]].map(([lib, ms]) =>
            h('tr', {}, h('td', {}, lib), h('td', { class: 'mono' }, duree(ms))))))),
    h('p', { class: 'note', style: { marginTop: 'var(--esp-3)' } },
      'Le cache sert autant à respecter la limite de débit de Jolpica qu\'à garder le dernier '
      + 'hub consulté lisible hors connexion. Quand une source tombe, la valeur en cache est '
      + 'affichée avec son horodatage et un avertissement, jamais silencieusement.'))));

  pile.append(carte('Limites connues', h('ul', { class: 'liste-nue' },
    LIMITES.map((l) => h('li', { class: 'info-item' },
      h('div', { class: 'info-marque' }), h('div', { class: 'info-txt pt-s' }, l))))));

  pile.append(carte('Les réglages de voiture n\'existent pas publiquement', h('div', {},
    h('p', { class: 'txt-2 pt-s' },
      'Aucune écurie ne publie ses réglages. La FIA non plus. Il n\'existe aucune API, publique '
      + 'ou payante, donnant l\'angle d\'aileron, la hauteur de caisse, la répartition de freinage, '
      + 'les rapports de boîte ou les réglages de différentiel. C\'est de l\'information de '
      + 'compétition, protégée par construction.'),
    h('p', { class: 'txt-2 pt-s' },
      'Ce qui est faisable, et plus utile pour un pronostic : mesurer les ',
      h('b', {}, 'conséquences'), ' des réglages dans la télémétrie. On ne lit pas le choix, '
      + 'on lit son effet.'),
    h('div', { class: 'tableau-enrobage' },
      h('table', { class: 't' },
        h('thead', {}, h('tr', {}, h('th', {}, 'Indicateur dérivé'), h('th', {}, 'Ce qu\'il approche'))),
        h('tbody', {},
          [['Vitesse de pointe', 'traînée aérodynamique'],
            ['Vitesse mini en virage rapide', 'niveau d\'appui'],
            ['Ratio de trim aéro', 'compromis appui / vitesse choisi'],
            ['Gain DRS en km/h', 'efficacité de l\'aileron mobile'],
            ['Durée et point de freinage', 'stabilité au freinage'],
            ['Régime en fin de ligne droite', 'étagement de boîte'],
            ['Écart de trim avec l\'équipier', 'le seul comparatif à matériel égal']].map(([a, b]) =>
            h('tr', {}, h('td', {}, a), h('td', { class: 'txt-2' }, b)))))),
    h('p', { class: 'note note--encadre' },
      'Ces indicateurs seront présentés comme des mesures dérivées, jamais comme « les réglages '
      + 'de l\'écurie ». Ils demandent la télémétrie FastF1 et arrivent avec la phase 2.'))));

  pile.append(carte('Propriété intellectuelle', h('div', {},
    h('p', { class: 'txt-2 pt-s' },
      'Aucun logo, aucune image officielle, aucun live timing redistribué. Les données proviennent '
      + 'd\'API publiques citées ci-dessus, chacune avec sa licence. Les couleurs d\'écurie sont '
      + 'des approximations saisies à la main pour la lisibilité des graphiques.'),
    h('p', { class: 'note' }, `Paddock v${APP.version} — usage personnel, non commercial. `
      + 'Ce site n\'est affilié ni à la FIA, ni à Formula One World Championship Limited.'))));
}
