/** Résultats d'un Grand Prix : course, qualifications, calendrier. */

import { h, vider, etatVide, erreurSource, badgeFraicheur } from '../util/dom.js';
import { nombre, chrono } from '../util/format.js';
import * as T from '../util/time.js';
import * as jolpica from '../data/jolpica.js';
import * as demo from '../data/demo.js';
import { contexte } from '../contexte.js';
import { carte, chargement, cellulePilote } from './composants.js';
import { SESSIONS } from '../config.js';

export async function rendre(racine, params) {
  vider(racine);
  racine.append(chargement(6));

  const ctx = await contexte();
  const saison = Number(params?.saison) || ctx.saison;
  const round = Number(params?.round) || ctx.gp_precedent?.round || ctx.gp?.round;
  const gp = ctx.calendrier?.find((g) => g.saison === saison && g.round === round);

  vider(racine);
  const pile = h('div', { class: 'pile' });
  racine.append(pile);

  pile.append(h('section', { class: 'heros' },
    h('div', { class: 'heros-sur' }, `Saison ${saison} · Manche ${round}`),
    h('h1', {}, gp?.nom || 'Grand Prix'),
    gp ? h('div', { class: 'heros-lieu' }, `${gp.circuit_nom} · ${T.dateLongue(new Date(gp.course_utc))}`) : null));

  if (ctx.mode_demo) {
    const res = demo.resultatsDemo(round);
    const qual = demo.qualifDemo(round);
    pile.append(carte('Course', tableauCourse(res, Date.now()), { source: 'DÉMO' }));
    pile.append(carte('Qualifications', tableauQualif(qual, Date.now()), { source: 'DÉMO' }));
    return;
  }

  const blocC = carte('Course', chargement(6), { source: 'JOL' });
  const blocQ = carte('Qualifications', chargement(6), { source: 'JOL' });
  pile.append(blocC, blocQ);

  jolpica.resultatsCourse(saison, round)
    .then((r) => {
      const corps = blocC.querySelector('.carte-corps');
      vider(corps).append(r.valeur
        ? tableauCourse(r.valeur, r.ts)
        : etatVide('Course non courue', 'Aucun résultat publié pour cette manche.'));
    })
    .catch((e) => vider(blocC.querySelector('.carte-corps')).append(erreurSource('JOL', e.message)));

  jolpica.resultatsQualif(saison, round)
    .then((r) => {
      const corps = blocQ.querySelector('.carte-corps');
      vider(corps).append(r.valeur
        ? tableauQualif(r.valeur, r.ts)
        : etatVide('Qualifications non publiées', 'Rien à afficher pour l\'instant.'));
    })
    .catch((e) => vider(blocQ.querySelector('.carte-corps')).append(erreurSource('JOL', e.message)));
}

function tableauCourse(res, ts) {
  const abandons = res.lignes.filter((l) => l.abandon).length;
  return h('div', {},
    h('div', { class: 'tableau-enrobage' },
      h('table', { class: 't' },
        h('thead', {}, h('tr', {},
          h('th', {}, ''), h('th', {}, 'Pilote'), h('th', { class: 'num' }, 'Grille'),
          h('th', { class: 'num' }, 'Tours'), h('th', {}, 'Statut'), h('th', { class: 'num' }, 'Pts'))),
        h('tbody', {},
          res.lignes.map((l) => h('tr', {},
            h('td', { class: 'pos' }, l.position_texte),
            h('td', {}, cellulePilote(l)),
            h('td', { class: 'num' }, l.grille || '—'),
            h('td', { class: 'num' }, l.tours),
            h('td', { class: 'txt-3 pt-s' },
              l.statut, l.meilleur_tour?.rang === 1 ? h('span', { class: 'badge badge--accent', style: { marginLeft: '6px' } }, 'MT') : null),
            h('td', { class: 'num' }, nombre(l.points))))))),
    h('div', { class: 'rangee', style: { marginTop: 'var(--esp-3)' } },
      h('span', { class: 'note' }, `${abandons} abandon(s) sur ${res.lignes.length} partants`),
      h('span', { class: 'espace' }),
      badgeFraicheur(T.depuis(ts))));
}

function tableauQualif(res, ts) {
  return h('div', {},
    h('div', { class: 'tableau-enrobage' },
      h('table', { class: 't' },
        h('thead', {}, h('tr', {},
          h('th', {}, ''), h('th', {}, 'Pilote'),
          h('th', { class: 'num' }, 'Q1'), h('th', { class: 'num' }, 'Q2'), h('th', { class: 'num' }, 'Q3'))),
        h('tbody', {},
          res.lignes.map((l) => h('tr', {},
            h('td', { class: 'pos' }, l.position),
            h('td', {}, cellulePilote(l)),
            h('td', { class: 'num txt-3' }, l.q1 || '—'),
            h('td', { class: 'num txt-3' }, l.q2 || '—'),
            h('td', { class: 'num' }, l.q3 || '—')))))),
    h('div', { class: 'rangee', style: { marginTop: 'var(--esp-3)' } }, badgeFraicheur(T.depuis(ts))));
}

/** Calendrier complet de la saison. */
export async function rendreCalendrier(racine) {
  vider(racine);
  racine.append(chargement(8));

  const ctx = await contexte();
  vider(racine);
  const pile = h('div', { class: 'pile' });
  racine.append(pile);

  if (!ctx.calendrier?.length) {
    pile.append(etatVide('Calendrier indisponible', `Aucune manche publiée pour ${ctx.saison}.`));
    return;
  }

  const maintenant = Date.now();
  pile.append(carte(`Calendrier ${ctx.saison}`, h('div', {},
    h('div', { class: 'tableau-enrobage' },
      h('table', { class: 't' },
        h('thead', {}, h('tr', {},
          h('th', {}, 'M.'), h('th', {}, 'Grand Prix'), h('th', {}, 'Circuit'),
          h('th', {}, 'Course'), h('th', {}, ''))),
        h('tbody', {},
          ctx.calendrier.map((gp) => {
            const d = new Date(gp.course_utc);
            const passe = d.getTime() < maintenant;
            const courant = gp.id === ctx.gp?.id;
            return h('tr', { style: passe ? { opacity: .6 } : null },
              h('td', { class: 'pos' }, gp.round),
              h('td', {}, h('b', {}, gp.nom),
                gp.format === 'sprint' ? h('span', { class: 'badge', style: { marginLeft: '6px' } }, 'sprint') : null,
                courant ? h('span', { class: 'badge badge--accent', style: { marginLeft: '6px' } }, 'en cours') : null),
              h('td', {}, h('a', { href: `#/circuit/${gp.circuit_id}` }, gp.circuit_nom || gp.circuit_id)),
              h('td', { class: 'pt-s' }, `${T.dateLongue(d)} · ${T.heure(d)}`),
              h('td', {}, passe
                ? h('a', { class: 'btn btn--fin', href: `#/resultats/${gp.saison}/${gp.round}` }, 'Résultats')
                : h('a', { class: 'btn btn--fin', href: `#/pronostic/${gp.id}` }, 'Pronostiquer')));
          })))),
    h('p', { class: 'note', style: { marginTop: 'var(--esp-3)' } },
      `Heures converties dans ton fuseau (${T.FUSEAU_LOCAL}). Stockage en UTC, conversion à l'affichage.`)),
  { source: ctx.mode_demo ? 'DÉMO' : 'JOL', fraicheur: T.depuis(ctx.ts_calendrier) }));
}
