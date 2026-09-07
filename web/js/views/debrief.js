/**
 * Écran 6 — Débrief post-course. C'est l'écran de rétention :
 * il doit expliquer POURQUOI, pas seulement COMBIEN.
 */

import { h, vider, etatVide, erreurSource, badgeSource, badgeFraicheur } from '../util/dom.js';
import { nombre, nomPiloteLong, pourcent, delta } from '../util/format.js';
import * as T from '../util/time.js';
import * as store from '../store.js';
import * as demo from '../data/demo.js';
import { contexte } from '../contexte.js';
import { scorer, scoreBaseline, performance } from '../model/scoring.js';
import { construire } from './resultat_reel.js';
import { carte, chargement, cellulePilote, pastilleEcurie } from './composants.js';

export async function rendre(racine, params) {
  vider(racine);
  racine.append(chargement(6));

  const ctx = await contexte();
  const gp = ctx.calendrier?.find((g) => g.id === params?.id) || ctx.gp_precedent || ctx.gp;
  if (!gp) {
    vider(racine).append(etatVide('Aucun Grand Prix', 'Rien à débriefer.'));
    return;
  }

  const pronostic = store.pronostic(gp.id);

  let reel = null, erreur = null;
  try {
    reel = await construire(gp, {
      modeDemo: ctx.mode_demo,
      resultatsDemo: ctx.mode_demo ? demo.resultatsDemo(gp.round) : null,
      qualifDemo: ctx.mode_demo ? demo.qualifDemo(gp.round) : null,
    });
  } catch (e) { erreur = e; }

  vider(racine);
  const pile = h('div', { class: 'pile' });
  racine.append(pile);

  if (!reel) {
    pile.append(etatVide(
      'Résultats pas encore disponibles',
      erreur
        ? `La source n'a pas répondu : ${erreur.message}`
        : 'Le débrief est publié environ 2 h après le drapeau à damier, une fois les résultats officiels ingérés.'));
    if (pronostic?.soumis) pile.append(recapPronostic(pronostic, gp));
    return;
  }

  if (!pronostic?.soumis) {
    pile.append(etatVide('Pas de pronostic sur ce Grand Prix',
      'Le débrief compare ton pronostic au modèle. Sans pronostic soumis, il n\'y a rien à comparer.'));
    pile.append(blocClassement(reel, gp));
    return;
  }

  // Le duel imposé n'est connu que par la baseline figée du joueur.
  const duelImpose = pronostic.baseline_gel?.duels?.[0];
  const reelComplet = {
    ...reel,
    duel_gagnant: duelImpose ? reel.duels[duelImpose.team_id] ?? null : null,
  };

  const resultat = scorer(pronostic, reelComplet);
  const refModele = scoreBaseline(pronostic.baseline_gel, reelComplet);
  const ecart = performance(resultat.total, refModele);

  pile.append(blocScore(gp, resultat, refModele, ecart));

  // En démonstration, le résultat est tiré au sort — y compris pour une course
  // qui n'a pas encore eu lieu. Le dire, sinon l'écran ment.
  if (ctx.mode_demo) {
    const courue = new Date(gp.course_utc).getTime() < Date.now();
    pile.append(h('div', { class: 'bandeau bandeau--demo' },
      h('span', { 'aria-hidden': 'true' }, '⚠'),
      h('div', { class: 'bandeau-txt' },
        h('strong', {}, 'Débrief de démonstration'),
        courue
          ? 'Ce classement est généré localement, il ne correspond à aucune course réelle.'
          : `La course n'a pas encore eu lieu : ce classement est simulé pour montrer l'écran. `
            + `Le vrai débrief est publié environ 2 h après le drapeau à damier.`)));
  }
  if (resultat.ecourtee) {
    pile.append(h('div', { class: 'bandeau bandeau--alerte' },
      h('span', {}, '⚠'),
      h('div', { class: 'bandeau-txt' },
        h('strong', {}, 'Course écourtée'),
        `Moins de 75 % de la distance (${pourcent(reel.distance_pct, 0)}) : tous les points de base sont appliqués à 50 %, comme le règlement sportif le fait pour le championnat.`)));
  }

  const marche = resultat.lignes.filter((l) => l.points > 0).sort((a, b) => b.points - a.points);
  const manque = resultat.lignes.filter((l) => l.points === 0);

  const cols = h('div', { class: 'grille-2' });
  const g = h('div', { class: 'pile' });
  const d = h('div', { class: 'pile' });
  cols.append(g, d);
  pile.append(cols);

  g.append(carte('Ce qui a marché',
    marche.length
      ? h('div', {}, marche.map((l) => ligneScore(l, pronostic, reelComplet, true)))
      : etatVide('Rien n\'est passé', 'Aucune ligne n\'a rapporté de points sur ce Grand Prix.')));

  g.append(carte('Ce qui a manqué',
    manque.length
      ? h('div', {}, manque.map((l) => ligneScore(l, pronostic, reelComplet, false)))
      : etatVide('Sans faute', 'Toutes les lignes ont rapporté.')));

  d.append(blocIndisponibles(reel));
  d.append(blocClassement(reelComplet, gp));

  pile.append(carte('Comment ce score est calculé', h('div', {},
    h('p', { class: 'txt-2 pt-s' },
      'Le score brut sert à l\'affichage. Le classement, lui, se fait sur l\'écart au modèle : ',
      h('code', {}, 'performance = score_joueur − score_baseline'), '. ',
      'Un Grand Prix où tout le monde a bien deviné parce que la course était prévisible ne bouge pas le classement.'),
    h('p', { class: 'note' },
      `Baseline figée le ${T.dateHeure(new Date(pronostic.baseline_ts))}, jamais recalculée depuis. `
      + `Multiplicateur de fenêtre : ×${String(resultat.multiplicateur_fenetre).replace('.', ',')}.`),
    h('div', { class: 'rangee' }, badgeSource('CALC'), badgeSource('JOL'),
      badgeFraicheur(T.depuis(reel.ts))))));
}

function blocScore(gp, resultat, refModele, ecart) {
  const gagne = ecart > 0;
  return h('section', { class: 'heros' },
    h('div', { class: 'heros-sur' }, 'Ton débrief'),
    h('h1', {}, gp.nom),
    h('div', { class: 'score-grand', style: { marginTop: 'var(--esp-4)' } },
      h('span', { class: 'score-val' }, nombre(resultat.total, 0)),
      h('span', { class: 'score-ref' }, `pts · modèle : ${nombre(refModele, 0)}`)),
    h('div', { style: { marginTop: 'var(--esp-3)' } },
      h('span', { class: `verdict ${gagne ? 'verdict--gagne' : 'verdict--perdu'}` },
        gagne ? '✓ Tu as battu le modèle' : '✗ Le modèle t\'a battu',
        h('span', { class: 'mono' }, ` ${delta(ecart, 0)}`))),
    h('p', { class: 'note', style: { marginTop: 'var(--esp-3)' } },
      'C\'est cet écart, pas le score brut, qui fait le classement de ligue.'));
}

function ligneScore(l, pronostic, reel, positif) {
  const nomDe = (id) => {
    const p = pronostic.baseline_gel?.pilotes?.find((x) => x.driver_id === id);
    return p ? nomPiloteLong(p) : (id || '—');
  };
  const estPilote = ['pole', 'meilleur_tour', 'premier_abandon', 'duel'].includes(l.cle)
    || l.cle.startsWith('grille:');

  const valeur = estPilote ? nomDe(l.valeur) : String(l.valeur);
  const reelle = l.valeur_reelle === null || l.valeur_reelle === undefined
    ? null
    : (estPilote ? nomDe(l.valeur_reelle) : String(l.valeur_reelle));

  return h('div', { class: 'ligne-score' },
    h('div', { class: 'ligne-score-txt' },
      h('div', {},
        h('b', {}, l.libelle), ' — ', valeur,
        l.joker > 1 ? h('span', { class: 'badge badge--alerte', style: { marginLeft: '6px' } }, `joker ×${l.joker}`) : null),
      h('div', { class: 'ligne-score-sous' },
        positif
          ? `${l.precision} · base ${nombre(l.base, 0)} × coef ×${nombre(l.coef, 1)} · le modèle donnait ${pourcent(l.proba, 1)}`
          : `${reelle !== null ? `réel : ${reelle}` : 'non réalisé'}${l.detail ? ` · ${l.detail}` : ''} · le modèle donnait ${pourcent(l.proba, 1)} à ton choix`)),
    h('div', { class: `ligne-score-pts ${l.points > 0 ? 'pts--positif' : 'pts--nul'}` },
      `${l.points > 0 ? '+' : ''}${nombre(l.points, 0)}`));
}

/** Ce que la plateforme n'a pas pu vérifier — dit franchement. */
function blocIndisponibles(reel) {
  const manquants = [];
  if (!reel.safety_car_connu) {
    manquants.push('Safety car : la direction de course n\'est lisible que via OpenF1 (2023 →). '
      + 'La ligne n\'a pas été notée plutôt que d\'être comptée comme ratée.');
  }
  if (!reel.pole_connue) manquants.push('Qualifications non publiées : les lignes du bloc B ne sont pas notées.');
  if (!reel.premier_abandon_connu) manquants.push('Aucun abandon : la ligne « premier abandon » est sans objet.');
  if (reel.premier_abandon_connu) {
    manquants.push('Premier abandon déduit du nombre de tours bouclés. L\'ordre exact demande '
      + 'l\'horodatage des messages de direction de course.');
  }
  if (!manquants.length) return null;
  return carte('Ce que la plateforme n\'a pas pu vérifier',
    h('ul', { class: 'liste-nue' },
      manquants.map((m) => h('li', { class: 'note', style: { marginBottom: 'var(--esp-2)' } }, '• ', m))));
}

function blocClassement(reel, gp) {
  return carte('Classement final', h('div', {},
    h('div', { class: 'tableau-enrobage' },
      h('table', { class: 't' },
        h('thead', {}, h('tr', {},
          h('th', {}, ''), h('th', {}, 'Pilote'),
          h('th', { class: 'num' }, 'Grille'), h('th', { class: 'num' }, 'Pts'))),
        h('tbody', {},
          reel.classement.slice(0, 12).map((l) => h('tr', {},
            h('td', { class: 'pos' }, l.position_texte),
            h('td', {}, cellulePilote(l)),
            h('td', { class: 'num' }, l.grille || '—'),
            h('td', { class: 'num' }, nombre(l.points))))))),
    h('div', { class: 'rangee', style: { marginTop: 'var(--esp-3)' } },
      h('a', { class: 'btn btn--fin', href: `#/resultats/${gp.saison}/${gp.round}` }, 'Classement complet'),
      h('span', { class: 'espace' }),
      h('span', { class: 'note' }, `${reel.nb_abandons} abandon(s)`))),
  { source: 'JOL' });
}

function recapPronostic(p, gp) {
  return carte('Ton pronostic', h('div', {},
    h('p', { class: 'txt-2 pt-s' },
      `Soumis le ${T.dateHeure(new Date(p.soumis_ts))} en fenêtre ${p.fenetre}.`),
    h('ol', { style: { paddingLeft: '1.2em', margin: 0 } },
      (p.grille || []).filter(Boolean).map((id) => {
        const pl = p.baseline_gel?.pilotes?.find((x) => x.driver_id === id);
        return h('li', { style: { marginBottom: '4px' } },
          h('span', { class: 'rangee' }, pastilleEcurie(pl?.team_id), pl ? nomPiloteLong(pl) : id));
      }))));
}
