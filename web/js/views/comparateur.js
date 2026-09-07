/**
 * Écran 4 — Comparateur de tours (phase 2).
 *
 * Écran dense, desktop prioritaire. Sur mobile, on affiche le delta seul et on
 * invite à passer sur grand écran.
 *
 * Ce qui est réellement faisable ici : comparer les tours valides et les
 * secteurs, depuis OpenF1 (2023 →). Les canaux de télémétrie — vitesse,
 * accélérateur, frein, DRS le long du tour — viennent de FastF1 en traitement
 * différé (livrable 1 §6.1). Ils ne sont pas inventés : la zone est marquée
 * indisponible tant que le batch de la phase 2 n'existe pas.
 */

import { h, vider, etatVide, erreurSource, badgeSource } from '../util/dom.js';
import { nombre, chrono, delta, nomPiloteLong } from '../util/format.js';
import * as T from '../util/time.js';
import * as of1 from '../data/openf1.js';
import { contexte } from '../contexte.js';
import { SESSIONS } from '../config.js';
import { carte, chargement, pastilleEcurie } from './composants.js';

export async function rendre(racine) {
  vider(racine);
  racine.append(chargement(5));

  const ctx = await contexte();
  vider(racine);
  const pile = h('div', { class: 'pile' });
  racine.append(pile);

  if (ctx.mode_demo) {
    pile.append(etatVide('Comparateur indisponible en démonstration',
      'Il lit les tours réels d\'une session OpenF1. Sans réseau, il n\'y a rien à comparer.'));
    return;
  }
  if (ctx.saison < of1.COUVERTURE_DEPUIS) {
    pile.append(etatVide('Hors couverture OpenF1',
      `OpenF1 couvre 2023 → aujourd'hui. Pour ${ctx.saison}, les temps au tour passent par FastF1, en batch.`));
    return;
  }

  let sessions = [];
  try {
    const r = await of1.sessions(ctx.saison);
    const lieu = String(ctx.gp?.ville || '').toLowerCase();
    sessions = r.valeur
      .filter((s) => SESSIONS[s.type])
      .filter((s) => !lieu || String(s.circuit_court || s.lieu || '').toLowerCase().includes(lieu)
        || String(s.lieu || '').toLowerCase().includes(lieu));
    if (!sessions.length) sessions = r.valeur.filter((s) => SESSIONS[s.type]).slice(-8);
  } catch (e) {
    pile.append(erreurSource('OF1', e.message, () => rendre(racine)));
    return;
  }

  if (!sessions.length) {
    pile.append(etatVide('Aucune session', 'OpenF1 n\'expose encore aucune session pour ce Grand Prix.'));
    return;
  }

  const etat = { session: sessions[sessions.length - 1], a: null, b: null };
  const barre = h('div', { class: 'carte-corps' });
  const zone = h('div');
  pile.append(h('section', { class: 'carte' }, barre), zone);

  const dessinerBarre = () => {
    vider(barre).append(h('div', { class: 'rangee' },
      h('select', {
        class: 'champ', style: { maxWidth: '260px' },
        'aria-label': 'Session',
        onchange: (e) => {
          etat.session = sessions.find((s) => String(s.session_key) === e.target.value);
          etat.a = null; etat.b = null;
          charger();
        },
      }, sessions.map((s) => h('option', {
        value: String(s.session_key), selected: s.session_key === etat.session.session_key || null,
      }, `${SESSIONS[s.type]?.long || s.nom} — ${T.dateHeure(new Date(s.debut_utc))}`))),
      badgeSource('OF1')));
  };

  const charger = async () => {
    dessinerBarre();
    vider(zone).append(chargement(5));
    try {
      const [pilotes, tours] = await Promise.all([
        of1.pilotesSession(etat.session.session_key).then((r) => r.valeur),
        of1.tours(etat.session.session_key).then((r) => r.valeur),
      ]);
      if (!tours.length) {
        vider(zone).append(etatVide('Aucun tour',
          'OpenF1 n\'a pas encore publié de tour pour cette session.'));
        return;
      }
      const presents = pilotes.filter((p) => tours.some((l) => l.numero_pilote === p.numero));
      etat.a = etat.a || presents[0]?.numero || null;
      etat.b = etat.b || presents[1]?.numero || null;
      vider(zone).append(vueComparaison(etat, presents, tours, charger));
    } catch (e) {
      vider(zone).append(erreurSource('OF1', e.message, charger));
    }
  };

  charger();
}

function vueComparaison(etat, pilotes, tours, rafraichir) {
  const selecteur = (cle, label) => h('select', {
    class: 'champ', 'aria-label': label,
    onchange: (e) => { etat[cle] = Number(e.target.value); rafraichir(); },
  }, pilotes.map((p) => h('option', {
    value: String(p.numero), selected: p.numero === etat[cle] || null,
  }, `#${p.numero} ${p.nom_complet || p.code}`)));

  const infoA = pilotes.find((p) => p.numero === etat.a);
  const infoB = pilotes.find((p) => p.numero === etat.b);
  const toursA = tours.filter((l) => l.numero_pilote === etat.a);
  const toursB = tours.filter((l) => l.numero_pilote === etat.b);
  const meilleurA = meilleurTourValide(toursA);
  const meilleurB = meilleurTourValide(toursB);

  const pile = h('div', { class: 'pile' });

  pile.append(carte('Pilotes comparés', h('div', { class: 'rangee' },
    selecteur('a', 'Pilote A'), h('span', { class: 'txt-3' }, 'vs'), selecteur('b', 'Pilote B'))));

  if (!meilleurA || !meilleurB) {
    pile.append(etatVide('Pas de tour valide',
      'Au moins un des deux pilotes n\'a aucun tour propre sur cette session : hors stands, '
      + 'hors tour de sortie, hors tour nettement ralenti.'));
    return pile;
  }

  const dt = (meilleurA.duree_ms - meilleurB.duree_ms) / 1000;

  pile.append(carte('Meilleur tour valide', h('div', {},
    h('div', { class: 'tableau-enrobage' },
      h('table', { class: 't' },
        h('thead', {}, h('tr', {},
          h('th', {}, ''), h('th', { class: 'num' }, 'Tour'), h('th', { class: 'num' }, 'S1'),
          h('th', { class: 'num' }, 'S2'), h('th', { class: 'num' }, 'S3'), h('th', { class: 'num' }, 'Chrono'))),
        h('tbody', {},
          ligneTour(infoA, meilleurA), ligneTour(infoB, meilleurB),
          h('tr', {},
            h('td', {}, h('b', {}, 'Δ')),
            h('td', { class: 'num' }, ''),
            ...['s1_ms', 's2_ms', 's3_ms'].map((s) => h('td', {
              class: 'num',
              style: { color: couleurDelta(meilleurA[s] - meilleurB[s]) },
            }, meilleurA[s] && meilleurB[s] ? delta((meilleurA[s] - meilleurB[s]) / 1000) : '—')),
            h('td', { class: 'num', style: { color: couleurDelta(dt), fontWeight: 700 } }, delta(dt)))))),
    h('p', { class: 'note', style: { marginTop: 'var(--esp-3)' } },
      'Δ négatif = le pilote A est plus rapide. Comparaison faite uniquement sur des tours '
      + 'valides : c\'est la première source d\'erreur d\'analyse quand on l\'oublie.'))));

  pile.append(carte('Rythme tour par tour',
    graphique(toursA, toursB, infoA, infoB), { source: 'OF1' }));

  pile.append(carte('Télémétrie', etatVide('Canaux non disponibles côté site',
    'Vitesse, accélérateur, frein et DRS le long du tour viennent de FastF1, en traitement '
    + 'différé après la session (livrable 1 §6.1). Le comparateur de courbes superposées et le '
    + 'curseur synchronisé arrivent avec la story 2.11, une fois ce batch en place.')));

  return pile;
}

function ligneTour(info, tour) {
  return h('tr', {},
    h('td', {}, h('span', { class: 'rangee' },
      h('span', { class: 'pastille', style: { '--c': info?.couleur || '#888' } }),
      h('b', {}, `#${info?.numero} ${info?.code || ''}`))),
    h('td', { class: 'num' }, tour.numero_tour),
    ...['s1_ms', 's2_ms', 's3_ms'].map((s) => h('td', { class: 'num' },
      tour[s] ? nombre(tour[s] / 1000, 3) : '—')),
    h('td', { class: 'num' }, h('b', {}, chrono(tour.duree_ms))));
}

function couleurDelta(v) {
  if (!Number.isFinite(v) || v === 0) return 'inherit';
  return v < 0 ? 'var(--ok)' : 'var(--danger)';
}

function meilleurTourValide(tours) {
  const valides = tours.filter((l) => l.is_valide && l.duree_ms > 0);
  if (!valides.length) return null;
  return valides.reduce((m, l) => (l.duree_ms < m.duree_ms ? l : m), valides[0]);
}

/** Courbe des tours valides. SVG inline : pas de bibliothèque à charger. */
function graphique(toursA, toursB, infoA, infoB) {
  const validesA = toursA.filter((l) => l.is_valide && l.duree_ms);
  const validesB = toursB.filter((l) => l.is_valide && l.duree_ms);
  const tous = [...validesA, ...validesB];
  if (tous.length < 2) return etatVide('Trop peu de tours valides', 'Le graphique demande au moins deux tours propres.');

  const minT = Math.min(...tous.map((l) => l.duree_ms));
  const maxT = Math.max(...tous.map((l) => l.duree_ms));
  const minL = Math.min(...tous.map((l) => l.numero_tour));
  const maxL = Math.max(...tous.map((l) => l.numero_tour));
  const L = 100, H = 42, marge = 3;

  const x = (n) => marge + ((n - minL) / Math.max(1, maxL - minL)) * (L - 2 * marge);
  const y = (ms) => marge + ((ms - minT) / Math.max(1, maxT - minT)) * (H - 2 * marge);
  const chemin = (liste) => liste
    .sort((a, b) => a.numero_tour - b.numero_tour)
    .map((l, i) => `${i ? 'L' : 'M'}${x(l.numero_tour).toFixed(2)},${y(l.duree_ms).toFixed(2)}`)
    .join(' ');

  return h('div', {},
    h('svg', {
      viewBox: `0 0 ${L} ${H}`, class: 'trace', style: { maxHeight: '220px' },
      role: 'img',
      'aria-label': `Temps au tour valides de #${infoA?.numero} et #${infoB?.numero}`,
    },
    h('path', { d: chemin(validesA), fill: 'none', stroke: infoA?.couleur || '#4ea8ff', 'stroke-width': '0.7' }),
    h('path', { d: chemin(validesB), fill: 'none', stroke: infoB?.couleur || '#ff2e3f', 'stroke-width': '0.7', 'stroke-dasharray': '2 1.2' })),
    h('div', { class: 'rangee', style: { marginTop: 'var(--esp-3)' } },
      h('span', { class: 'rangee' },
        h('span', { class: 'pastille', style: { '--c': infoA?.couleur || '#4ea8ff' } }),
        h('span', { class: 'pt-s' }, `#${infoA?.numero} ${infoA?.code || ''} — ${validesA.length} tours valides`)),
      h('span', { class: 'rangee' },
        h('span', { class: 'pastille', style: { '--c': infoB?.couleur || '#ff2e3f' } }),
        h('span', { class: 'pt-s' }, `#${infoB?.numero} ${infoB?.code || ''} — ${validesB.length} tours valides`))),
    h('p', { class: 'note', style: { marginTop: 'var(--esp-2)' } },
      `Axe vertical : ${chrono(minT)} en haut, ${chrono(maxT)} en bas. Seuls les tours propres sont tracés. `
      + 'La validité est ici approchée (hors stands, hors tour de sortie, écart à la médiane) : '
      + 'la règle exacte, avec statut de piste par tour, vient du batch FastF1.'));
}
