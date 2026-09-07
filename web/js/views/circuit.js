/**
 * Écran 2 — Fiche circuit.
 *
 * Règle du livrable 3 : chaque statistique porte sa fenêtre de calcul.
 * « 60 % » sans « sur 5 éditions » n'est pas une information.
 */

import { h, vider, erreurSource, etatVide, badgeSource, badgeFraicheur } from '../util/dom.js';
import { nombre, pourcent, secondes } from '../util/format.js';
import * as T from '../util/time.js';
import * as reference from '../data/reference.js';
import * as jolpica from '../data/jolpica.js';
import { COMPOSES } from '../config.js';
import { contexte } from '../contexte.js';
import { carte, statBarre, chargement, cellulePilote } from './composants.js';

export async function rendre(racine, params) {
  vider(racine);
  racine.append(chargement(5));

  const ctx = await contexte();
  const id = params?.id || ctx.gp?.circuit_id;
  if (!id) {
    vider(racine).append(etatVide('Aucun circuit sélectionné', 'Choisis un Grand Prix depuis le hub.'));
    return;
  }

  let c = null;
  try { c = await reference.circuit(id); } catch { /* référentiel illisible */ }

  vider(racine);
  const pile = h('div', { class: 'pile' });
  racine.append(pile);

  if (!c) {
    // Le circuit existe côté Jolpica mais pas dans le référentiel MAN : on
    // affiche ce qu'on a, sans inventer les champs de saisie manuelle.
    const gp = ctx.calendrier?.find((g) => g.circuit_id === id);
    pile.append(carte(gp?.circuit_nom || id,
      h('div', {},
        h('p', { class: 'txt-2 pt-s' },
          'Ce circuit n\'est pas encore saisi dans le référentiel back-office. '
          + 'Longueur, nombre de virages, zones DRS, appui et abrasivité sont des champs MAN : '
          + 'ils ne sont pas déduits automatiquement, et ne seront pas inventés ici.'),
        gp ? h('p', { class: 'note' }, `Coordonnées connues : ${gp.lat}, ${gp.lon} (source Jolpica).`) : null),
      { source: 'MAN' }));
    return;
  }

  const fenetre = (await reference.circuits())._meta.fenetre_stats;

  pile.append(enTete(c));
  pile.append(rubanTour(c));

  const cols = h('div', { class: 'grille-2' });
  const g = h('div', { class: 'pile' });
  const d = h('div', { class: 'pile' });
  cols.append(g, d);
  pile.append(cols);

  g.append(blocHistoire(c, fenetre));
  g.append(blocDegradation(c, fenetre));
  d.append(blocIdentite(c));
  d.append(blocPalmares(c, ctx));
}

function enTete(c) {
  return h('section', { class: 'heros' },
    h('div', { class: 'heros-sur' }, [c.ville, c.pays].filter(Boolean).join(' · ')),
    h('h1', {}, c.nom),
    h('div', { class: 'rangee', style: { marginTop: 'var(--esp-4)' } },
      h('span', { class: 'badge' }, `${nombre(c.longueur_m / 1000, 3)} km`),
      h('span', { class: 'badge' }, `${c.nb_tours} tours`),
      h('span', { class: 'badge' }, `${c.nb_virages} virages`),
      h('span', { class: 'badge' }, `${c.zones_drs} zone${c.zones_drs > 1 ? 's' : ''} DRS`),
      h('span', { class: 'badge' }, reference.LIBELLES.sens[c.sens] || c.sens)));
}

/**
 * Le tracé réel demande les positions GPS de la voiture, disponibles via
 * FastF1 mais seulement en batch (livrable 1 §6.1). Plutôt qu'un dessin
 * approximatif présenté comme le tracé, on déroule le tour en ligne :
 * secteurs et zones DRS placés proportionnellement. C'est exact et lisible.
 */
function rubanTour(c) {
  const L = c.longueur_m;
  const secteurs = [0.34, 0.35, 0.31];   // découpage indicatif, à remplacer par les marqueurs réels
  let x = 0;
  const blocs = secteurs.map((part, i) => {
    const debut = x; x += part;
    return { i: i + 1, debut, largeur: part };
  });

  return carte('Le tour, déroulé', h('div', {},
    h('svg', { class: 'trace', viewBox: '0 0 100 26', role: 'img', 'aria-label': 'Découpage du tour en secteurs et zones DRS' },
      blocs.map((b) => h('rect', {
        x: (b.debut * 100).toFixed(2), y: 4, width: (b.largeur * 100 - 0.6).toFixed(2), height: 9,
        rx: 2, fill: ['#3a4356', '#48536a', '#5a6780'][b.i - 1],
      })),
      blocs.map((b) => h('text', {
        x: ((b.debut + b.largeur / 2) * 100).toFixed(2), y: 10.4,
        'text-anchor': 'middle', 'font-size': '4.4', fill: '#eef1f6', 'font-weight': '700',
      }, `S${b.i}`)),
      Array.from({ length: c.zones_drs }, (_, i) => {
        const largeur = 9;
        const pos = 6 + i * (86 / Math.max(1, c.zones_drs));
        return h('rect', { x: pos, y: 16, width: largeur, height: 5, rx: 2, fill: '#4ea8ff' });
      }),
      h('text', { x: 0, y: 25, 'font-size': '3.6', fill: '#8a93a5' },
        `${c.zones_drs} zone(s) DRS · ${nombre(L)} m par tour`)),
    h('p', { class: 'note', style: { marginTop: 'var(--esp-3)' } },
      'Schéma de découpage, pas le tracé géographique. Le tracé réel se dessine depuis les '
      + 'positions GPS de la voiture, disponibles par FastF1 en traitement différé (phase 2). '
      + 'Les bornes de secteur affichées ici sont indicatives.')),
  { source: 'MAN' });
}

function blocIdentite(c) {
  const l = (lib, val, src, titre) => h('div', { class: 'stat' },
    h('div', { class: 'stat-lib', title: titre || null }, lib),
    h('div', { class: 'stat-val' }, val),
    badgeSource(src));

  return carte('Identité du tracé', h('div', {},
    l('Longueur', `${nombre(c.longueur_m)} m`, 'MAN'),
    l('Nombre de tours', nombre(c.nb_tours), 'JOL'),
    l('Virages', nombre(c.nb_virages), 'MAN'),
    l('Sens', reference.LIBELLES.sens[c.sens] || '—', 'MAN'),
    l('Appui aérodynamique', reference.LIBELLES.type_appui[c.type_appui] || '—', 'MAN'),
    l('Abrasivité', reference.LIBELLES.abrasivite[c.abrasivite] || '—', 'MAN'),
    l('Altitude', `${nombre(c.altitude_m)} m`, 'MAN', 'Impact sur le moteur : Mexico, Interlagos'),
    l('Perte au stand', secondes(c.pit_loss_s, 1), 'CALC', 'Temps total perdu pour un arrêt, recalculé chaque saison'),
    h('p', { class: 'note note--encadre' },
      'Les champs marqués MAN sont saisis en back-office et restent au statut « à valider » '
      + 'tant que la checklist du livrable 1 n\'est pas arbitrée.')));
}

function blocHistoire(c, fenetre) {
  const f = `sur ${fenetre} éditions`;
  return carte('Ce que dit l\'histoire', h('div', {},
    statBarre('Course avec au moins une safety car', pourcent(c.taux_sc_historique, 0),
      c.taux_sc_historique, { fenetre: 'sur les 10 dernières éditions' }),
    statBarre('Dépassements par course', nombre(c.depassements_moyens),
      Math.min(1, c.depassements_moyens / 50), { fenetre: f }),
    statBarre('Écart pole → P2', secondes(c.ecart_pole_p2_s, 2),
      Math.min(1, c.ecart_pole_p2_s / 0.5), { fenetre: f }),
    statBarre('Arrêts du vainqueur', nombre(c.arrets_vainqueur, 1),
      Math.min(1, c.arrets_vainqueur / 3), { fenetre: f }),
    h('p', { class: 'note note--encadre' },
      'Valeurs de pré-remplissage du référentiel. Le batch de la phase 2 (story 2.5) les '
      + 'recalcule depuis les résultats réels ; tant qu\'il n\'a pas tourné, elles sont indicatives.')));
}

function blocDegradation(c, fenetre) {
  const max = Math.max(...Object.values(c.degradation || {}), 0.01);
  return carte('Dégradation par composé', h('div', {},
    Object.entries(c.degradation || {}).map(([comp, v]) =>
      h('div', { class: 'stat' },
        h('div', { class: 'stat-lib' },
          h('span', { class: 'rangee' },
            h('span', {
              class: 'pastille',
              style: { '--c': COMPOSES[comp]?.hex || '#888' },
              'aria-hidden': 'true',
            }),
            COMPOSES[comp]?.label || comp)),
        h('div', { class: 'jauge jauge--info' }, h('i', { style: { width: `${(v / max) * 100}%` } })),
        h('div', { class: 'stat-val' }, `${nombre(v, 2)} s/tour`),
        badgeSource('CALC'))),
    h('p', { class: 'note', style: { marginTop: 'var(--esp-3)' } },
      `Pente de régression sur les tours valides de chaque relais, ${fenetre} éditions précédentes. `
      + 'Rappel : tout indicateur de rythme se calcule uniquement sur les tours propres — '
      + 'hors stands, hors neutralisation, hors tour de sortie.')));
}

function blocPalmares(c, ctx) {
  const emplacement = carte('Vainqueurs récents', chargement(4), { source: 'JOL' });
  const corps = emplacement.querySelector('.carte-corps');

  if (ctx.mode_demo) {
    vider(corps).append(etatVide('Indisponible en démonstration',
      'Le palmarès demande l\'historique Jolpica.'));
    return emplacement;
  }

  (async () => {
    const saisons = [ctx.saison - 1, ctx.saison - 2, ctx.saison - 3];
    const lignes = [];
    for (const s of saisons) {
      try {
        const cal = await jolpica.calendrier(s);
        const gp = cal.valeur.find((g) => g.circuit_id === c.id);
        if (!gp) continue;
        const r = await jolpica.resultatsCourse(s, gp.round);
        const v = r.valeur?.lignes?.[0];
        if (v) lignes.push({ saison: s, ...v });
      } catch { /* saison indisponible */ }
    }
    if (!lignes.length) {
      vider(corps).append(etatVide('Aucune édition récente trouvée',
        'Ce circuit n\'apparaît pas aux calendriers des trois saisons précédentes.'));
      return;
    }
    vider(corps).append(
      h('div', { class: 'tableau-enrobage' },
        h('table', { class: 't' },
          h('tbody', {},
            lignes.map((l) => h('tr', {},
              h('td', { class: 'pos' }, l.saison),
              h('td', {}, cellulePilote(l)),
              h('td', { class: 'num' }, `P${l.grille} sur la grille`)))))));
  })();

  return emplacement;
}
