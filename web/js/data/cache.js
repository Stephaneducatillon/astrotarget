/**
 * Cache local avec TTL (livrable 1 §5) et repli sur donnée périmée.
 *
 * Deux raisons d'exister :
 *  - Jolpica applique une limite de débit (risque « bloquant » du livrable 4) ;
 *  - le dernier hub consulté doit rester lisible hors connexion (PWA).
 *
 * Toute valeur rendue porte son horodatage : `ts` remonte jusqu'à l'UI pour
 * afficher « màj il y a X min », obligatoire sur les données live et météo.
 */

const PREFIXE = 'paddock:cache:';

function lire(cle) {
  try {
    const brut = localStorage.getItem(PREFIXE + cle);
    if (!brut) return null;
    const o = JSON.parse(brut);
    return (o && typeof o.ts === 'number') ? o : null;
  } catch { return null; }
}

function ecrire(cle, valeur, ts) {
  try {
    localStorage.setItem(PREFIXE + cle, JSON.stringify({ ts, valeur }));
  } catch (e) {
    // Quota dépassé : on purge les entrées les plus anciennes plutôt que d'échouer.
    purgerAncien(10);
    try { localStorage.setItem(PREFIXE + cle, JSON.stringify({ ts, valeur })); } catch { /* tant pis */ }
  }
}

function purgerAncien(n) {
  const entrees = [];
  for (let i = 0; i < localStorage.length; i++) {
    const k = localStorage.key(i);
    if (!k || !k.startsWith(PREFIXE)) continue;
    try { entrees.push([k, JSON.parse(localStorage.getItem(k)).ts || 0]); } catch { entrees.push([k, 0]); }
  }
  entrees.sort((a, b) => a[1] - b[1]).slice(0, n).forEach(([k]) => localStorage.removeItem(k));
}

export function vidercache() {
  for (let i = localStorage.length - 1; i >= 0; i--) {
    const k = localStorage.key(i);
    if (k && k.startsWith(PREFIXE)) localStorage.removeItem(k);
  }
}

/**
 * @returns {Promise<{valeur:any, ts:number, frais:boolean, perime:boolean, erreur:Error|null}>}
 * `perime: true` signifie « la source a échoué, voici la dernière valeur connue » :
 * l'appelant doit le dire à l'utilisateur, pas le masquer.
 */
export async function avecCache(cle, ttlMs, chargeur) {
  const en_cache = lire(cle);
  const maintenant = Date.now();

  if (en_cache && ttlMs > 0 && maintenant - en_cache.ts < ttlMs) {
    return { valeur: en_cache.valeur, ts: en_cache.ts, frais: false, perime: false, erreur: null };
  }

  try {
    const valeur = await chargeur();
    const ts = Date.now();
    if (ttlMs > 0) ecrire(cle, valeur, ts);
    return { valeur, ts, frais: true, perime: false, erreur: null };
  } catch (erreur) {
    if (en_cache) {
      return { valeur: en_cache.valeur, ts: en_cache.ts, frais: false, perime: true, erreur };
    }
    throw erreur;
  }
}

/** Lecture seule du cache, sans appel réseau (démarrage hors ligne). */
export function depuisCache(cle) {
  return lire(cle);
}
