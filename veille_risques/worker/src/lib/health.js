// Sante des sources.
//
// Le scenario d'echec le plus dangereux n'est pas l'alerte manquee, c'est le
// connecteur mort en silence : un Ecowatt casse un jour rouge ressemble
// exactement a un jour vert. On trace donc chaque tentative, et une source de
// niveau A muette devient elle-meme une alerte.

import { TEMPO, SOURCES } from '../config.js';

/**
 * Met a jour l'etat de sante d'une source. Fonction pure.
 * @returns {{sante: object, incident: object|null}}
 */
export function majSante(precedent, resultat, now) {
  const base = precedent ?? { last_ok: null, fail_streak: 0, notified: 0 };

  if (resultat.ok) {
    const retablie = base.notified === 1;
    return {
      sante: { last_ok: now, last_try: now, last_error: null, fail_streak: 0, notified: 0 },
      incident: retablie ? { type: 'retablie' } : null,
    };
  }

  const streak = (base.fail_streak ?? 0) + 1;
  const err = resultat.error;
  const message = err?.message ?? String(err);
  // Une cle refusee est une panne permanente : elle demande une action humaine
  // et ne se resoudra pas toute seule. On ne temporise pas.
  const permanent = err?.permanent === true;
  const seuilAtteint = permanent || streak >= TEMPO.echecs_avant_alerte;

  return {
    sante: {
      last_ok: base.last_ok ?? null,
      last_try: now,
      last_error: message,
      fail_streak: streak,
      notified: seuilAtteint && base.notified !== 1 ? 1 : base.notified,
    },
    incident:
      seuilAtteint && base.notified !== 1
        ? { type: permanent ? 'auth' : 'muette', streak, message }
        : null,
  };
}

/** Age de la donnee et verdict de peremption, pour le tableau de bord. */
export function fraicheur(sourceId, sante, now) {
  const def = SOURCES.find((s) => s.id === sourceId);
  const freq = def?.freq ?? 900;
  const lastOk = sante?.last_ok ?? null;
  if (!lastOk) return { age: null, perimee: true, jamais: true };
  const age = now - lastOk;
  return { age, perimee: age > 3 * freq, jamais: false };
}
