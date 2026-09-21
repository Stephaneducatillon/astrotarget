// Machine a etats des alertes.
//
// C'est le coeur du dispositif, et la raison pour laquelle il reste utilisable :
// on notifie les TRANSITIONS, jamais les etats. Une vigilance orange de 12 h
// produit une notification, pas 48.
//
// Trois regles :
//   1. Montee immediate         - une degradation ne se temporise pas.
//   2. Descente differee        - il faut N releves consecutifs sous le seuil,
//                                 sinon une station qui oscille autour de son
//                                 seuil fait clignoter le telephone.
//   3. Rappel plafonne          - un etat non vert qui dure rappelle au plus
//                                 une fois toutes les 6 h.
//
// Fonctions pures : aucune I/O, entierement testables (voir test/state.test.js).

import { NIVEAUX } from './config.js';

export function rang(niveau) {
  const i = NIVEAUX.indexOf(niveau);
  return i === -1 ? 0 : i;
}

const ETAT_INITIAL = { level: 'vert', label: null, since: 0, last_notified: null, below_count: 0 };

/**
 * Calcule l'etat suivant d'une regle et la notification eventuelle.
 *
 * @param {object|null} courant  etat lu en base, ou null au premier passage
 * @param {{level: string, label?: string}} observe  niveau constate ce run
 * @param {number} now  epoch en secondes
 * @param {object} tempo  { releves_avant_descente, rappel_secondes }
 * @returns {{etat: object, notification: object|null}}
 */
export function nextState(courant, observe, now, tempo) {
  const etat = { ...ETAT_INITIAL, ...(courant ?? {}) };
  const avant = etat.level;
  const apres = observe.level;
  const label = observe.label ?? null;

  // --- 1. Montee : immediate.
  if (rang(apres) > rang(avant)) {
    return {
      etat: { level: apres, label, since: now, last_notified: now, below_count: 0 },
      notification: {
        kind: 'montee',
        from: avant,
        to: apres,
        label,
        priority: apres === 'rouge' ? 'urgent' : 'default',
      },
    };
  }

  // --- 2. Descente : differee de N releves consecutifs.
  if (rang(apres) < rang(avant)) {
    const below = etat.below_count + 1;
    if (below < tempo.releves_avant_descente) {
      // On reste au niveau haut, on ne notifie rien, on compte.
      return { etat: { ...etat, below_count: below }, notification: null };
    }
    return {
      etat: { level: apres, label, since: now, last_notified: now, below_count: 0 },
      notification: {
        kind: apres === 'vert' ? 'fin' : 'desescalade',
        from: avant,
        to: apres,
        label,
        priority: apres === 'vert' ? 'min' : 'low',
      },
    };
  }

  // --- 3. Niveau inchange. Le compteur de descente se reinitialise : une
  // oscillation sous le seuil ne doit pas s'accumuler entre deux episodes.
  const stable = { ...etat, label, below_count: 0 };

  if (apres === 'vert') {
    return { etat: stable, notification: null };
  }

  const depuis = stable.last_notified ?? 0;
  if (now - depuis >= tempo.rappel_secondes) {
    return {
      etat: { ...stable, last_notified: now },
      notification: {
        kind: 'rappel',
        from: avant,
        to: apres,
        label,
        priority: 'low',
        depuis: stable.since,
      },
    };
  }

  return { etat: stable, notification: null };
}

/** Niveau familial global : le plus eleve parmi toutes les regles. */
export function niveauGlobal(etats) {
  return etats.reduce((acc, e) => (rang(e.level) > rang(acc) ? e.level : acc), 'vert');
}
