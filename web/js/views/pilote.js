/**
 * Écran 3 — Fiche pilote. Onglets : Forme · Ce circuit · Saison.
 *
 * Le duel d'équipiers est affiché en haut de fiche : c'est le seul comparatif
 * à matériel égal, donc le seul honnête.
 */

import { h, vider, erreurSource, etatVide, badgeSource, badgeFraicheur } from '../util/dom.js';
import { nombre, pourcent, nomPiloteLong, delta, ecurie } from '../util/format.js';
import * as T from '../util/time.js';
import * as jolpica from '../data/jolpica.js';
import { contexte, engagesEtHistorique } from '../contexte.js';
import { formePilote, deltaQualifCourse, fiabilite, affiniteCircuit } from '../model/baseline.js';
import { carte, chargement, cellulePilote, pastilleEcurie } from './composants.js';

const FENETRE = 5;

/** Liste des pilotes = classement du championnat. */
export async function rendreListe(racine) {
  vider(racine);
  racine.append(chargement(6));

  const ctx = await contexte();
  vider(racine);
  const pile = h('div', { class: 'pile' });
  racine.append(pile);

  if (ctx.mode_demo) {
    const { engages } = await engagesEtHistorique(ctx.saison, null);
    pile.append(carte(`Engagés ${ctx.saison}`, tableauPilotes(
      engages.map((e, i) => ({ ...e, position: i + 1 })), Date.now()), { source: 'DÉMO' }));
    return;
  }

  try {
    const r = await jolpica.classementPilotes(ctx.saison);
    if (!r.valeur.lignes.length) throw new Error('Classement vide.');
    pile.append(carte(`Championnat pilotes ${ctx.saison}`,
      tableauPilotes(r.valeur.lignes, r.ts),
      { source: 'JOL', fraicheur: T.depuis(r.ts) }));
  } catch (e) {
    pile.append(erreurSource('JOL', e.message, () => rendreListe(racine)));
  }
}

function tableauPilotes(lignes, ts) {
  return h('div', {},
    h('div', { class: 'tableau-enrobage' },
      h('table', { class: 't' },
        h('thead', {}, h('tr', {},
          h('th', {}, ''), h('th', {}, 'Pilote'), h('th', {}, 'Écurie'),
          h('th', { class: 'num' }, 'Pts'), h('th', { class: 'num' }, 'V'))),
        h('tbody', {},
          lignes.map((l) => h('tr', {},
            h('td', { class: 'pos' }, l.position),
            h('td', {}, h('a', { href: `#/pilote/${l.driver_id}` },
              h('span', { class: 'rangee' }, pastilleEcurie(l.team_id), nomPiloteLong(l)))),
            h('td', { class: 'txt-3 pt-s' }, l.team_nom || '—'),
            h('td', { class: 'num' }, nombre(l.points)),
            h('td', { class: 'num' }, nombre(l.victoires))))))),
    h('div', { class: 'rangee', style: { marginTop: 'var(--esp-3)' } }, badgeFraicheur(T.depuis(ts))));
}

export async function rendre(racine, params) {
  const driverId = params?.id;
  vider(racine);
  racine.append(chargement(6));

  const ctx = await contexte();
  const circuitId = ctx.gp?.circuit_id;

  // En démonstration, l'historique n'existe pas : on montre la fiche
  // d'identité et on dit pourquoi le reste est vide, plutôt que d'aller
  // interroger une source qu'on sait injoignable.
  if (ctx.mode_demo) {
    const { engages } = await engagesEtHistorique(ctx.saison, circuitId);
    const e = engages.find((x) => x.driver_id === driverId);
    vider(racine);
    if (!e) {
      racine.append(etatVide('Pilote inconnu', `Aucun engagé « ${driverId} » dans le jeu de démonstration.`));
      return;
    }
    racine.append(h('div', { class: 'pile' },
      enTetePilote(e, ctx),
      carte('Forme, historique et duel d\'équipiers',
        etatVide('Indisponible en démonstration',
          'Ces onglets lisent l\'historique Jolpica course par course. Le jeu de démonstration '
          + 'ne contient qu\'une liste d\'engagés et un week-end fictif : il n\'y a rien à analyser. '
          + 'Repasse en données réelles depuis les réglages.'),
        { source: 'DÉMO' })));
    return;
  }

  let saison = [], surCircuit = [], entree = null, erreur = null;
  try {
    const [rs, rc, re] = await Promise.allSettled([
      jolpica.resultatsPiloteSaison(ctx.saison, driverId),
      circuitId ? jolpica.resultatsPiloteCircuit(driverId, circuitId) : Promise.resolve({ valeur: [] }),
      jolpica.engages(ctx.saison),
    ]);
    if (rs.status === 'fulfilled') saison = rs.value.valeur;
    if (rc.status === 'fulfilled') surCircuit = rc.value.valeur;
    if (re.status === 'fulfilled') entree = re.value.valeur.find((x) => x.driver_id === driverId) || null;
    if (!entree && rs.status === 'rejected') erreur = rs.reason;
  } catch (e) { erreur = e; }

  vider(racine);
  if (!entree && erreur) {
    racine.append(erreurSource('JOL', erreur.message, () => rendre(racine, params)));
    return;
  }
  if (!entree) {
    racine.append(etatVide('Pilote inconnu', `Aucun engagé « ${driverId} » sur la saison ${ctx.saison}.`));
    return;
  }

  const pile = h('div', { class: 'pile' });
  racine.append(pile);

  pile.append(enTetePilote(entree, ctx));

  // Duel équipier, tout en haut.
  const equipierBloc = carte('Duel d\'équipiers', chargement(2), { source: 'JOL' });
  pile.append(equipierBloc);
  chargerDuel(equipierBloc, ctx, entree);

  const onglets = h('div', { class: 'onglets', role: 'tablist' });
  const panneau = h('div', { style: { marginTop: 'var(--esp-4)' } });
  pile.append(h('section', { class: 'carte' },
    h('div', { class: 'carte-corps' }, onglets, panneau)));

  const vues = {
    forme: () => vueForme(saison, ctx),
    circuit: () => vueCircuit(surCircuit, ctx, circuitId),
    saison: () => vueSaison(saison, entree, ctx),
  };
  const libelles = { forme: 'Forme', circuit: 'Ce circuit', saison: 'Saison' };

  let actif = 'forme';
  const dessiner = () => {
    vider(onglets);
    Object.keys(vues).forEach((cle) => {
      onglets.append(h('button', {
        role: 'tab', 'aria-selected': String(cle === actif),
        onclick: () => { actif = cle; dessiner(); },
      }, libelles[cle]));
    });
    vider(panneau).append(vues[actif]());
  };
  dessiner();
}

function enTetePilote(e, ctx) {
  const eq = ecurie(e.team_id);
  return h('section', { class: 'heros' },
    h('div', { class: 'heros-sur' }, `Saison ${ctx.saison}`),
    h('h1', {}, nomPiloteLong(e)),
    h('div', { class: 'rangee', style: { marginTop: 'var(--esp-3)' } },
      pastilleEcurie(e.team_id),
      h('span', { class: 'heros-lieu' }, e.team_nom || '—'),
      e.numero ? h('span', { class: 'badge' }, `#${e.numero}`) : null,
      e.code ? h('span', { class: 'badge' }, e.code) : null,
      Number.isFinite(e.position_championnat)
        ? h('span', { class: 'badge badge--accent' }, `P${e.position_championnat} au championnat · ${nombre(e.points)} pts`)
        : null));
}

/** Forme : points sur 5 GP, delta qualif↔course, fiabilité. */
function vueForme(saison, ctx) {
  if (!saison.length) {
    return etatVide('Pas encore de course', `Aucun résultat pour ce pilote sur la saison ${ctx.saison}.`);
  }
  const derniers = saison.slice(-FENETRE);
  const forme = formePilote(saison, FENETRE);
  const dqc = deltaQualifCourse(saison, FENETRE);

  return h('div', {},
    h('div', { class: 'meteo-grille' },
      encart('Forme', forme === null ? '—' : nombre(forme, 1),
        `moyenne pondérée des points, ${FENETRE} derniers GP`),
      encart('Delta qualif → course', dqc === null ? '—' : delta(dqc, 1),
        'position course − position grille, négatif = il remonte'),
      encart('Fiabilité', pourcent(1 - fiabilite(saison), 0),
        'courses terminées, replié vers la moyenne')),

    h('h3', { style: { marginTop: 'var(--esp-5)', marginBottom: 'var(--esp-3)' } },
      `Les ${derniers.length} derniers Grands Prix`),
    h('div', { class: 'tableau-enrobage' },
      h('table', { class: 't' },
        h('thead', {}, h('tr', {},
          h('th', {}, 'Grand Prix'), h('th', { class: 'num' }, 'Grille'),
          h('th', { class: 'num' }, 'Arrivée'), h('th', { class: 'num' }, 'Δ'),
          h('th', { class: 'num' }, 'Pts'))),
        h('tbody', {},
          [...derniers].reverse().map((r) => {
            const fini = /^\d+$/.test(String(r.position_texte));
            const d = fini && r.grille ? r.position - r.grille : null;
            return h('tr', {},
              h('td', {}, r.gp_nom),
              h('td', { class: 'num' }, r.grille || '—'),
              h('td', { class: 'num' }, fini ? r.position : h('span', { class: 'txt-3', title: r.statut }, 'ab.')),
              h('td', { class: 'num', style: { color: d < 0 ? 'var(--ok)' : d > 0 ? 'var(--danger)' : 'inherit' } },
                d === null ? '—' : delta(-d, 0)),
              h('td', { class: 'num' }, nombre(r.points)));
          })))),
    h('p', { class: 'note', style: { marginTop: 'var(--esp-3)' } },
      'Δ positif = places gagnées entre la grille et l\'arrivée. Les abandons sont exclus du calcul du delta.'),
    h('div', { class: 'rangee' }, badgeSource('JOL'), badgeSource('CALC')));
}

function vueCircuit(surCircuit, ctx, circuitId) {
  if (!circuitId) return etatVide('Pas de circuit courant', 'Aucun Grand Prix sélectionné.');
  if (!surCircuit.length) {
    return etatVide('Jamais couru ici',
      'Ce pilote n\'a aucun résultat enregistré sur ce circuit dans l\'historique Jolpica.');
  }
  const aff = affiniteCircuit(surCircuit);
  const moy = surCircuit.filter((r) => Number.isFinite(r.position))
    .reduce((s, r, _, a) => s + r.position / a.length, 0);

  return h('div', {},
    h('div', { class: 'meteo-grille' },
      encart('Affinité', aff === null ? '—' : pourcent(aff, 0), 'note 0–100 % dérivée du classement moyen ici'),
      encart('Position moyenne', moy ? nombre(moy, 1) : '—', `sur ${surCircuit.length} édition(s)`),
      encart('Éditions', nombre(surCircuit.length), 'historique Jolpica complet')),
    h('div', { class: 'tableau-enrobage', style: { marginTop: 'var(--esp-4)' } },
      h('table', { class: 't' },
        h('thead', {}, h('tr', {},
          h('th', {}, 'Saison'), h('th', { class: 'num' }, 'Grille'),
          h('th', { class: 'num' }, 'Arrivée'), h('th', {}, 'Statut'))),
        h('tbody', {},
          surCircuit.map((r) => h('tr', {},
            h('td', {}, r.saison),
            h('td', { class: 'num' }, r.grille || '—'),
            h('td', { class: 'num' }, /^\d+$/.test(String(r.position_texte)) ? r.position : '—'),
            h('td', { class: 'txt-3 pt-s' }, r.statut || '—')))))),
    h('p', { class: 'note', style: { marginTop: 'var(--esp-3)' } },
      'Le rythme relatif par type d\'appui demande les temps au tour valides : il arrive avec '
      + 'le batch FastF1 de la phase 2 (story 2.3).'));
}

function vueSaison(saison, entree, ctx) {
  if (!saison.length) return etatVide('Pas encore de course', 'Rien à afficher pour cette saison.');
  const finis = saison.filter((r) => /^\d+$/.test(String(r.position_texte)));
  const abandons = saison.length - finis.length;
  const podiums = finis.filter((r) => r.position <= 3).length;
  const top10 = finis.filter((r) => r.position <= 10).length;

  return h('div', {},
    h('div', { class: 'meteo-grille' },
      encart('Points', nombre(entree.points ?? saison.reduce((s, r) => s + (r.points || 0), 0)), 'championnat'),
      encart('Podiums', nombre(podiums), `sur ${saison.length} GP`),
      encart('Top 10', nombre(top10), `sur ${saison.length} GP`),
      encart('Abandons', `${abandons} / ${saison.length}`, 'toutes causes')),
    h('h3', { style: { marginTop: 'var(--esp-5)', marginBottom: 'var(--esp-3)' } }, 'Course par course'),
    h('div', { class: 'tableau-enrobage' },
      h('table', { class: 't' },
        h('thead', {}, h('tr', {},
          h('th', {}, 'M.'), h('th', {}, 'Grand Prix'),
          h('th', { class: 'num' }, 'Grille'), h('th', { class: 'num' }, 'Arrivée'),
          h('th', { class: 'num' }, 'Pts'))),
        h('tbody', {},
          saison.map((r) => h('tr', {},
            h('td', { class: 'pos' }, r.round),
            h('td', {}, r.gp_nom),
            h('td', { class: 'num' }, r.grille || '—'),
            h('td', { class: 'num' }, /^\d+$/.test(String(r.position_texte)) ? r.position : h('span', { class: 'txt-3' }, 'ab.')),
            h('td', { class: 'num' }, nombre(r.points))))))),
    h('p', { class: 'note', style: { marginTop: 'var(--esp-3)' } },
      'L\'évolution Elo arrive avec le modèle de la phase 2 (story 2.7) : elle demande un '
      + 'calibrage sur deux saisons complètes avant d\'avoir un sens.'));
}

function encart(lib, val, note) {
  return h('div', { class: 'mini' },
    h('div', { class: 'mini-lib' }, lib),
    h('div', { class: 'mini-val' }, val),
    note ? h('div', { class: 'note', style: { marginTop: '2px' } }, note) : null);
}

async function chargerDuel(bloc, ctx, entree) {
  const corps = bloc.querySelector('.carte-corps');
  if (!entree.team_id) {
    vider(corps).append(etatVide('Écurie inconnue', 'Impossible d\'identifier l\'équipier.'));
    return;
  }
  try {
    const r = await jolpica.engages(ctx.saison);
    const equipier = r.valeur.find((x) => x.team_id === entree.team_id && x.driver_id !== entree.driver_id);
    if (!equipier) {
      vider(corps).append(etatVide('Pas d\'équipier identifié',
        'L\'écurie n\'a qu\'un pilote au classement pour l\'instant.'));
      return;
    }

    const [a, b] = await Promise.all([
      jolpica.resultatsPiloteSaison(ctx.saison, entree.driver_id).then((x) => x.valeur).catch(() => []),
      jolpica.resultatsPiloteSaison(ctx.saison, equipier.driver_id).then((x) => x.valeur).catch(() => []),
    ]);

    const parRound = new Map(b.map((r2) => [r2.round, r2]));
    let devantA = 0, devantB = 0, comparables = 0;
    for (const ra of a) {
      const rb = parRound.get(ra.round);
      if (!rb) continue;
      const fa = /^\d+$/.test(String(ra.position_texte));
      const fb = /^\d+$/.test(String(rb.position_texte));
      if (!fa || !fb) continue;
      comparables++;
      if (ra.position < rb.position) devantA++; else devantB++;
    }

    const ratio = comparables ? devantA / comparables : 0.5;
    vider(corps).append(h('div', {},
      h('div', { class: 'rangee', style: { justifyContent: 'space-between' } },
        h('div', { class: 'rangee' }, pastilleEcurie(entree.team_id), h('b', {}, nomPiloteLong(entree))),
        h('span', { class: 'mono' }, `${devantA} — ${devantB}`),
        h('div', { class: 'rangee' }, h('b', {}, nomPiloteLong(equipier)))),
      h('div', { class: 'jauge', style: { width: '100%', marginTop: 'var(--esp-3)' } },
        h('i', { style: { width: `${ratio * 100}%` } })),
      h('p', { class: 'note', style: { marginTop: 'var(--esp-3)' } },
        comparables
          ? `Courses où les deux ont été classés : ${comparables}. C'est le seul comparatif à matériel égal.`
          : 'Aucune course où les deux pilotes ont été classés ensemble.'),
      h('a', { class: 'btn btn--fin', href: `#/pilote/${equipier.driver_id}` },
        `Voir ${nomPiloteLong(equipier)}`)));
  } catch (e) {
    vider(corps).append(erreurSource('JOL', e.message));
  }
}
