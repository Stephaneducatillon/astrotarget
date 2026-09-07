/** Briques partagées entre écrans. */

import { h } from '../util/dom.js';
import { ecurie, nomPiloteLong, nombre } from '../util/format.js';
import * as T from '../util/time.js';
import * as om from '../data/openmeteo.js';
import { badgeSource, badgeFraicheur, squelette, erreurSource } from '../util/dom.js';

export function pastilleEcurie(teamId) {
  const e = ecurie(teamId);
  return h('span', {
    class: 'pastille', 'data-motif': e.motif, style: { '--c': e.hex },
    title: e.nom, 'aria-hidden': 'true',
  });
}

export function cellulePilote(p, { equipe = true } = {}) {
  return h('div', { class: 'cel-pilote' },
    pastilleEcurie(p.team_id),
    h('div', { style: { minWidth: 0 } },
      h('b', {}, nomPiloteLong(p)),
      equipe && p.team_nom ? h('div', { class: 'cel-equipe' }, p.team_nom) : null));
}

export function carte(titre, contenu, { source, fraicheur, actions, id } = {}) {
  return h('section', { class: 'carte', id: id || null },
    titre ? h('header', { class: 'carte-tete' },
      h('h2', { class: 'carte-titre' }, titre),
      h('span', { class: 'espace' }),
      fraicheur ? badgeFraicheur(fraicheur) : null,
      source ? badgeSource(source) : null,
      actions || null) : null,
    h('div', { class: 'carte-corps' }, contenu));
}

/**
 * Bloc météo. Règle du livrable 1 : toujours l'heure DE LA PRÉVISION et
 * l'heure de RAFRAÎCHISSEMENT. Une prévision de pluie sans horodatage n'a
 * aucune valeur pour un pronostic.
 */
export function blocMeteo({ heures, ts, cible, libelleCible, tzCircuit, perime, mode_demo }) {
  if (!heures?.length) {
    return erreurSource('OM', 'Aucune prévision disponible pour ces coordonnées.');
  }
  const p = om.pourHeure(heures, cible);
  const r = om.resume(p);

  if (!p) {
    return h('div', {},
      h('p', { class: 'txt-2 pt-s' },
        `La fenêtre de prévision ne couvre pas encore ${libelleCible}. `
        + 'Open-Meteo porte à 16 jours ; la prévision apparaîtra en approchant.'),
      h('p', { class: 'note' }, T.depuis(ts)));
  }

  const cibleLocale = `${T.dateLongue(cible)} ${T.heure(cible)} (${T.nomFuseau(cible)})`;

  return h('div', {},
    h('div', { class: 'rangee', style: { marginBottom: 'var(--esp-3)' } },
      h('span', { class: 'badge' }, libelleCible),
      h('span', { class: 'txt-3 pt-s' }, cibleLocale)),

    h('div', { class: 'meteo-tete' },
      h('div', { class: 'meteo-icone', 'aria-hidden': 'true' }, r.icone),
      h('div', {},
        h('div', { class: 'meteo-temp' }, `${nombre(p.temperature_c, 0)} °C`),
        h('div', { class: 'meteo-resume' }, r.texte))),

    h('div', { class: 'meteo-grille' },
      mini('Pluie', `${nombre(p.proba_pluie_pct, 0)} %`),
      mini('Cumul', `${nombre(p.precipitation_mm, 1)} mm`),
      mini('Piste ≈', p.sol_c === null ? '—' : `${nombre(p.sol_c, 0)} °C`, 'proxy : température du sol, pas une mesure de piste'),
      mini('Vent', `${nombre(p.vent_kmh, 0)} km/h`),
      mini('Humidité', `${nombre(p.humidite_pct, 0)} %`),
      mini('Nuages', `${nombre(p.nuages_pct, 0)} %`)),

    h('details', { style: { marginTop: 'var(--esp-4)' } },
      h('summary', { class: 'pt-s txt-2', style: { cursor: 'pointer' } }, 'Voir heure par heure'),
      h('div', { style: { marginTop: 'var(--esp-3)' } },
        h('div', { class: 'heures' },
          om.fenetre(heures, cible, 4, 5).map((x) => {
            const d = new Date(x.heure_utc);
            const estCible = Math.abs(d - cible) < 45 * 60000;
            return h('div', { class: `heure-col ${estCible ? 'heure-col--cible' : ''}` },
              h('div', { class: 'heure-h' }, T.heure(d)),
              h('div', { class: 'heure-t' }, `${nombre(x.temperature_c, 0)}°`),
              h('div', { class: 'heure-p' }, `${nombre(x.proba_pluie_pct, 0)} %`),
              h('div', { class: 'barre-pluie' },
                h('i', { style: { width: `${Math.max(2, x.proba_pluie_pct || 0)}%` } })));
          })),
        h('p', { class: 'note', style: { marginTop: 'var(--esp-2)' } },
          `Heures affichées dans ton fuseau (${T.nomFuseau(cible)})`
          + (tzCircuit && T.fuseauDifferent(cible, tzCircuit)
            ? ` — le circuit est en ${tzCircuit}.` : '.')))),

    h('div', { class: 'rangee', style: { marginTop: 'var(--esp-3)' } },
      badgeFraicheur(T.depuis(ts)),
      perime ? h('span', { class: 'badge badge--alerte' }, 'source injoignable — dernière valeur connue') : null,
      mode_demo ? h('span', { class: 'badge badge--alerte' }, 'démonstration') : null));
}

function mini(lib, val, titre) {
  return h('div', { class: 'mini', title: titre || null },
    h('div', { class: 'mini-lib' }, lib),
    h('div', { class: 'mini-val' }, val));
}

/** Ligne de statistique avec sa fenêtre de calcul — jamais un chiffre nu. */
export function statBarre(libelle, valeur, ratio, { source = 'CALC', fenetre } = {}) {
  return h('div', { class: 'stat' },
    h('div', { class: 'stat-lib' }, libelle,
      fenetre ? h('div', { class: 'note' }, fenetre) : null),
    ratio !== null && ratio !== undefined
      ? h('div', { class: 'jauge' }, h('i', { style: { width: `${Math.max(0, Math.min(100, ratio * 100))}%` } }))
      : null,
    h('div', { class: 'stat-val' }, valeur),
    badgeSource(source));
}

export function chargement(lignes = 4) { return squelette(lignes); }

/** Bandeau permanent du mode démonstration. Il ne doit jamais être discret. */
export function bandeauDemo() {
  return h('div', { class: 'bandeau bandeau--demo', role: 'status' },
    h('span', { 'aria-hidden': 'true' }, '⚠'),
    h('div', { class: 'bandeau-txt' },
      h('strong', {}, 'Mode démonstration'),
      'Calendrier, engagés et résultats sont un jeu local fictif, généré pour faire fonctionner l\'interface sans réseau. ',
      'Aucune de ces données n\'est officielle.'));
}

export function bandeauPerime(codeSource) {
  return h('div', { class: 'bandeau bandeau--info', role: 'status' },
    h('span', { 'aria-hidden': 'true' }, 'ⓘ'),
    h('div', { class: 'bandeau-txt' },
      h('strong', {}, `Source ${codeSource} injoignable`),
      'Les données affichées viennent du cache local. Leur horodatage est indiqué sur chaque bloc.'));
}
