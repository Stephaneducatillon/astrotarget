/**
 * Moteur de score — livrable 2.
 *
 * Principe directeur : on est payé à la difficulté, pas à l'exactitude.
 *   coef   = clamp(1 / proba_baseline, 0.5, 4.0)
 *   points = base × coef
 *
 * Le coefficient est toujours lu dans la baseline FIGÉE à la fermeture
 * (`pronostic.baseline_gel`), jamais dans une baseline recalculée : sinon les
 * coefficients seraient rétroactivement faussés (livrable 2 §7).
 */

import { CONFIG_PRONOSTIC } from '../config.js';
import { probaPosition, pronosticBaseline } from './baseline.js';

const { coef_min, coef_max, mult_hardcore, course_ecourtee_ratio } = CONFIG_PRONOSTIC;

export function coefficient(proba) {
  if (!Number.isFinite(proba) || proba <= 0) return coef_max;
  return Math.max(coef_min, Math.min(coef_max, 1 / proba));
}

/** Barème du bloc A (livrable 2 §2). Pas de malus : aucun point négatif. */
export function baseGrille(positionPronostiquee, positionReelle, dansTop10Reel) {
  if (!Number.isFinite(positionReelle) || !dansTop10Reel) return { base: 0, precision: 'absent' };
  const ecart = Math.abs(positionPronostiquee - positionReelle);
  if (ecart === 0) return { base: 10, precision: 'exact' };
  if (ecart === 1) return { base: 6, precision: '±1' };
  if (ecart === 2) return { base: 3, precision: '±2' };
  return { base: 1, precision: 'top 10' };
}

const BASES = {
  pole: 8, ecart_pole_p2: 4,
  meilleur_tour: 5, nb_abandons: 6, nb_abandons_proche: 3,
  safety_car: 4, premier_abandon: 6, duel: 4,
  nb_arrets: 5, compose_depart: 3, tour_premier_arret: 6,
};

/**
 * @param {object} pronostic     pronostic soumis, baseline figée incluse
 * @param {object} reel          résultat officiel normalisé
 * @returns {object} détail ligne par ligne, jamais un simple total
 */
export function scorer(pronostic, reel) {
  const baseline = pronostic.baseline_gel;
  const lignes = [];

  // Course non courue : GP neutralisé, aucun point, jokers rendus.
  if (reel?.annule) {
    return {
      neutralise: true, motif: 'Grand Prix annulé ou non couru',
      total: 0, brut: 0, lignes: [], joker_rendu: true,
      multiplicateur_fenetre: 1,
    };
  }

  // Course écourtée (< 75 % de la distance) : bases à 50 %, comme le
  // règlement sportif le fait pour les points du championnat.
  const ecourtee = Number.isFinite(reel?.distance_pct) && reel.distance_pct < 0.75;
  const ratioBase = ecourtee ? course_ecourtee_ratio : 1;

  const top10Reel = (reel?.top10 || []).slice(0, 10);
  const positionReelleDe = new Map(top10Reel.map((d, i) => [d, i + 1]));

  // ── Bloc A — grille d'arrivée ────────────────────────────────────────────
  (pronostic.grille || []).forEach((driverId, idx) => {
    if (!driverId) return;
    const position = idx + 1;
    const posReelle = positionReelleDe.get(driverId);
    const { base, precision } = baseGrille(position, posReelle, Boolean(posReelle));
    const proba = probaPosition(baseline, driverId, position);
    lignes.push(ligne({
      cle: `grille:${position}`,
      bloc: 'A', libelle: `P${position}`,
      valeur: driverId, valeur_reelle: top10Reel[idx] || null,
      base: base * ratioBase, proba, precision,
      detail: posReelle ? `arrivé P${posReelle}` : 'hors du top 10',
      reussi: base > 0,
    }));
  });

  // ── Bloc B — qualifications ──────────────────────────────────────────────
  if (pronostic.pole) {
    const ok = pronostic.pole === reel?.pole;
    lignes.push(ligne({
      cle: 'pole', bloc: 'B', libelle: 'Poleman',
      valeur: pronostic.pole, valeur_reelle: reel?.pole,
      base: ok ? BASES.pole * ratioBase : 0,
      proba: baseline?.parPilote?.[pronostic.pole]?.pole ?? 0.05,
      precision: ok ? 'exact' : 'raté', reussi: ok,
    }));
  }
  if (pronostic.ecart_pole_p2) {
    const ok = pronostic.ecart_pole_p2 === reel?.ecart_pole_p2;
    lignes.push(ligne({
      cle: 'ecart_pole_p2', bloc: 'B', libelle: 'Écart pole ↔ P2',
      valeur: pronostic.ecart_pole_p2, valeur_reelle: reel?.ecart_pole_p2,
      base: ok ? BASES.ecart_pole_p2 * ratioBase : 0,
      proba: baseline?.evenements?.ecart_pole_p2?.[pronostic.ecart_pole_p2] ?? 0.33,
      precision: ok ? 'exact' : 'raté', reussi: ok,
    }));
  }

  // ── Bloc C — course ──────────────────────────────────────────────────────
  if (pronostic.meilleur_tour) {
    const ok = pronostic.meilleur_tour === reel?.meilleur_tour;
    lignes.push(ligne({
      cle: 'meilleur_tour', bloc: 'C', libelle: 'Meilleur tour',
      valeur: pronostic.meilleur_tour, valeur_reelle: reel?.meilleur_tour,
      base: ok ? BASES.meilleur_tour * ratioBase : 0,
      proba: baseline?.parPilote?.[pronostic.meilleur_tour]?.meilleur_tour ?? 0.08,
      precision: ok ? 'exact' : 'raté', reussi: ok,
    }));
  }

  if (Number.isFinite(pronostic.nb_abandons)) {
    const reelN = reel?.nb_abandons;
    const ecart = Number.isFinite(reelN) ? Math.abs(pronostic.nb_abandons - reelN) : null;
    const base = ecart === 0 ? BASES.nb_abandons : ecart === 1 ? BASES.nb_abandons_proche : 0;
    lignes.push(ligne({
      cle: 'nb_abandons', bloc: 'C', libelle: 'Nombre d\'abandons',
      valeur: pronostic.nb_abandons, valeur_reelle: reelN,
      base: base * ratioBase,
      proba: baseline?.evenements?.abandons?.[pronostic.nb_abandons] ?? 0.15,
      precision: ecart === 0 ? 'exact' : ecart === 1 ? '±1' : 'raté',
      reussi: base > 0,
    }));
  }

  if (typeof pronostic.safety_car === 'boolean') {
    const ok = pronostic.safety_car === reel?.safety_car;
    const pOui = baseline?.evenements?.safety_car_oui ?? 0.5;
    lignes.push(ligne({
      cle: 'safety_car', bloc: 'C', libelle: 'Safety car',
      valeur: pronostic.safety_car ? 'oui' : 'non',
      valeur_reelle: reel?.safety_car === undefined ? null : (reel.safety_car ? 'oui' : 'non'),
      base: ok ? BASES.safety_car * ratioBase : 0,
      proba: pronostic.safety_car ? pOui : 1 - pOui,
      precision: ok ? 'exact' : 'raté', reussi: ok,
    }));
  }

  if (pronostic.premier_abandon) {
    const ok = pronostic.premier_abandon === reel?.premier_abandon;
    lignes.push(ligne({
      cle: 'premier_abandon', bloc: 'C', libelle: 'Premier abandon',
      valeur: pronostic.premier_abandon, valeur_reelle: reel?.premier_abandon,
      base: ok ? BASES.premier_abandon * ratioBase : 0,
      proba: baseline?.parPilote?.[pronostic.premier_abandon]?.premier_abandon ?? 0.05,
      precision: ok ? 'exact' : 'raté', reussi: ok,
    }));
  }

  if (pronostic.duel) {
    const ok = pronostic.duel === reel?.duel_gagnant;
    const duel = (baseline?.duels || []).find(
      (d) => d.a.driver_id === pronostic.duel || d.b.driver_id === pronostic.duel);
    const p = duel
      ? (duel.a.driver_id === pronostic.duel ? duel.a.proba : duel.b.proba)
      : 0.5;
    lignes.push(ligne({
      cle: 'duel', bloc: 'C', libelle: 'Duel d\'équipiers',
      valeur: pronostic.duel, valeur_reelle: reel?.duel_gagnant,
      base: ok ? BASES.duel * ratioBase : 0, proba: p,
      precision: ok ? 'exact' : 'raté', reussi: ok,
    }));
  }

  // ── Bloc D — stratégie (optionnel, phase 3) ──────────────────────────────
  const s = pronostic.strategie;
  if (s) {
    if (Number.isFinite(s.nb_arrets)) {
      const ok = s.nb_arrets === reel?.strategie?.nb_arrets;
      lignes.push(ligne({
        cle: 'nb_arrets', bloc: 'D', libelle: 'Arrêts du vainqueur',
        valeur: s.nb_arrets, valeur_reelle: reel?.strategie?.nb_arrets,
        base: ok ? BASES.nb_arrets * ratioBase : 0, proba: 0.4,
        precision: ok ? 'exact' : 'raté', reussi: ok,
      }));
    }
    if (s.compose_depart) {
      const ok = s.compose_depart === reel?.strategie?.compose_depart;
      lignes.push(ligne({
        cle: 'compose_depart', bloc: 'D', libelle: 'Composé de départ du poleman',
        valeur: s.compose_depart, valeur_reelle: reel?.strategie?.compose_depart,
        base: ok ? BASES.compose_depart * ratioBase : 0, proba: 0.45,
        precision: ok ? 'exact' : 'raté', reussi: ok,
      }));
    }
    if (Number.isFinite(s.tour_premier_arret)) {
      const r = reel?.strategie?.tour_premier_arret;
      const ok = Number.isFinite(r) && Math.abs(s.tour_premier_arret - r) <= 2;
      lignes.push(ligne({
        cle: 'tour_premier_arret', bloc: 'D', libelle: 'Tour du premier arrêt (±2)',
        valeur: s.tour_premier_arret, valeur_reelle: r,
        base: ok ? BASES.tour_premier_arret * ratioBase : 0, proba: 0.22,
        precision: ok ? '±2' : 'raté', reussi: ok,
      }));
    }
  }

  // ── Joker conviction : une seule ligne doublée (ou triplée). ─────────────
  const joker = pronostic.joker;
  if (joker?.ligne) {
    const cible = lignes.find((l) => l.cle === joker.ligne);
    if (cible) {
      const mult = joker.multiplicateur === 3 ? 3 : 2;
      cible.joker = mult;
      cible.points = arrondi(cible.points * mult);
    }
  }

  // ── Multiplicateur de fenêtre ────────────────────────────────────────────
  const multFenetre = pronostic.fenetre === 'hardcore' ? mult_hardcore : 1;
  const brut = arrondi(lignes.reduce((s2, l) => s2 + l.points, 0));
  const total = arrondi(brut * multFenetre);

  return {
    neutralise: false,
    ecourtee,
    lignes,
    brut,
    total,
    multiplicateur_fenetre: multFenetre,
    joker: joker || null,
  };
}

function ligne({ cle, bloc, libelle, valeur, valeur_reelle, base, proba, precision, detail, reussi }) {
  const coef = coefficient(proba);
  return {
    cle, bloc, libelle, valeur, valeur_reelle,
    base: arrondi(base), proba, coef: Math.round(coef * 100) / 100,
    points: arrondi(base * coef),
    precision, detail: detail || null, reussi: Boolean(reussi), joker: 1,
  };
}

const arrondi = (v) => Math.round(v * 100) / 100;

/**
 * Gain maximum théorique d'un pronostic — l'indicateur affiché en bas de
 * l'écran de saisie. Il rend la stratégie lisible avant de soumettre.
 */
export function gainMaximum(pronostic) {
  const baseline = pronostic.baseline_gel;
  if (!baseline) return 0;
  const parfait = {
    top10: (pronostic.grille || []).slice(0, 10),
    pole: pronostic.pole,
    meilleur_tour: pronostic.meilleur_tour,
    premier_abandon: pronostic.premier_abandon,
    safety_car: pronostic.safety_car,
    nb_abandons: pronostic.nb_abandons,
    ecart_pole_p2: pronostic.ecart_pole_p2,
    duel_gagnant: pronostic.duel,
    distance_pct: 1,
    strategie: pronostic.strategie
      ? { ...pronostic.strategie, tour_premier_arret: pronostic.strategie.tour_premier_arret }
      : undefined,
  };
  return scorer(pronostic, parfait).total;
}

/**
 * Métrique de classement (livrable 2 §5) : le score brut sert à l'affichage,
 * pas au classement. On classe sur l'écart au modèle.
 *
 *   performance = score_joueur − score_baseline
 */
export function scoreBaseline(baselineGel, reel, tailleGrille = 10) {
  if (!baselineGel) return 0;
  const p = pronosticBaseline(baselineGel, tailleGrille);
  const pronosticDuModele = {
    fenetre: 'standard',           // le modèle ne bénéficie d'aucun multiplicateur
    baseline_gel: baselineGel,
    grille: p.grille,
    pole: p.pole,
    meilleur_tour: p.meilleur_tour,
    premier_abandon: p.premier_abandon,
    safety_car: p.safety_car,
    nb_abandons: p.nb_abandons,
    ecart_pole_p2: p.ecart_pole_p2,
    duel: p.duel,
  };
  return scorer(pronosticDuModele, reel).total;
}

export function performance(scoreJoueur, scoreModele) {
  return Math.round((scoreJoueur - scoreModele) * 100) / 100;
}

/**
 * Départage (livrable 2 §6), dans l'ordre :
 * 1. positions exactes cumulées · 2. pronostics en fenêtre hardcore
 * 3. GP joués · 4. antériorité d'inscription
 */
export function comparerJoueurs(a, b) {
  if (b.performance !== a.performance) return b.performance - a.performance;
  if (b.positions_exactes !== a.positions_exactes) return b.positions_exactes - a.positions_exactes;
  if (b.hardcore !== a.hardcore) return b.hardcore - a.hardcore;
  if (b.gp_joues !== a.gp_joues) return b.gp_joues - a.gp_joues;
  return (a.inscrit_ts || 0) - (b.inscrit_ts || 0);
}

/**
 * Fermeture des fenêtres (livrable 2 §3), calculée depuis `session.debut_utc`.
 *
 * Intégrité : côté production, ce calcul appartient au serveur — l'horloge du
 * client ne doit jamais décider. Ici, sans back-end, il est fait localement et
 * l'interface le dit explicitement.
 */
export function fenetres(gp) {
  const s = (t) => gp?.sessions?.find((x) => x.type === t)?.debut_utc || null;
  const el1 = s('FP1') || gp?.debut_utc;
  const course = s('R') || gp?.course_utc;
  const q1 = s('Q') || s('SQ');
  return {
    hardcore: {
      cle: 'hardcore', label: 'Hardcore', multiplicateur: mult_hardcore,
      ferme_utc: el1,
      description: 'Ferme au début des EL1. Tu pronostiques sans rien savoir : ni essais, ni qualifications, ni météo confirmée.',
    },
    standard: {
      cle: 'standard', label: 'Standard', multiplicateur: 1,
      // 5 min avant l'extinction des feux.
      ferme_utc: course ? new Date(new Date(course).getTime() - 5 * 60000).toISOString() : null,
      description: 'Ferme 5 minutes avant l\'extinction des feux. Tu sais tout : grille, météo, pneus.',
    },
    // Le pronostic de pole ferme au début de Q1, quelle que soit la fenêtre.
    pole: { ferme_utc: q1 },
  };
}

export function fenetreOuverte(f, maintenant = new Date()) {
  if (!f?.ferme_utc) return false;
  return new Date(f.ferme_utc).getTime() > maintenant.getTime();
}

/**
 * Diagnostic du plafond de coefficient.
 *
 * Le livrable 2 §9 laisse le plafond de 4,0 à arbitrer. Ce calcul montre le
 * problème avec les chiffres du week-end en cours plutôt qu'en théorie.
 *
 * Le mécanisme « on est payé à la difficulté » suppose que 1/proba discrimine.
 * Pour un événement à peu d'issues (safety car, pole) il discrimine bien. Pour
 * le bloc A, la probabilité qu'un pilote donné finisse à UNE place précise dans
 * un peloton de vingt ne dépasse guère 25 % même pour le favori — or 1/0,25 = 4.
 * Résultat : la quasi-totalité de la grille touche le plafond, et prédire le
 * favori en P1 rapporte autant que prédire un fond de grille. C'est exactement
 * l'effet que le barème cherchait à éviter.
 *
 * Trois leviers possibles, tous du ressort du propriétaire du produit :
 * relever le plafond, normaliser le coefficient du bloc A par la probabilité
 * du meilleur candidat à cette place, ou réduire la grille à un top 5.
 */
export function diagnosticPlafond(baseline, tailleGrille = 10) {
  if (!baseline?.pilotes?.length) return null;
  const seuil = 1 / coef_max;
  let saturees = 0, total = 0;
  const parPosition = [];
  for (let pos = 1; pos <= tailleGrille; pos++) {
    let n = 0;
    for (const p of baseline.pilotes) {
      const proba = probaPosition(baseline, p.driver_id, pos);
      total++;
      if (proba <= seuil) { n++; saturees++; }
    }
    parPosition.push({ position: pos, au_plafond: n, sur: baseline.pilotes.length });
  }
  return {
    seuil,
    au_plafond: saturees,
    total,
    part: total ? saturees / total : 0,
    par_position: parPosition,
    meilleure_proba_p1: Math.max(...baseline.pilotes.map((p) => probaPosition(baseline, p.driver_id, 1))),
  };
}
