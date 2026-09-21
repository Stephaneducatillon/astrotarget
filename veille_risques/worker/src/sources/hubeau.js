// Hub'Eau hydrometrie v2 - hauteur d'eau de la Scarpe.
//
// API libre, sans cle. Deux precautions :
//  - les codes de station sont figes en configuration, jamais recherches a
//    chaque run (voir /admin/stations pour les relever une fois) ;
//  - `observations_tr` sert des milliers de points si on ne borne pas : on
//    demande explicitement les plus recents et on garde ce qui est utile.
//
// `resultat_obs` est en millimetres pour la grandeur H.

import { fetchJson } from '../lib/http.js';
import { STATIONS_HYDRO, SEUILS } from '../config.js';

export async function collectHubeau(env, now = Math.floor(Date.now() / 1000), stations_cfg = STATIONS_HYDRO) {
  if (stations_cfg.length === 0) {
    return { stations: [], configure: false };
  }

  const depuis = new Date((now - 3 * 3600) * 1000).toISOString();
  const stations = [];

  for (const st of stations_cfg) {
    const url =
      `${env.HUBEAU_BASE}/observations_tr` +
      `?code_entite=${encodeURIComponent(st.code)}` +
      `&grandeur_hydro=${encodeURIComponent(st.metric ?? 'H')}` +
      `&date_debut_obs=${encodeURIComponent(depuis)}` +
      `&sort=desc&size=20`;

    const raw = await fetchJson(url, { headers: { Accept: 'application/json' } });
    const point = derniereMesure(raw);
    if (point) {
      stations.push({ code: st.code, nom: st.nom, metric: st.metric ?? 'H', ...point });
    }
  }

  return { stations, configure: true };
}

/** Retourne la mesure la plus recente, ou null si le lot est vide. */
export function derniereMesure(raw) {
  const data = Array.isArray(raw?.data) ? raw.data : [];
  let best = null;
  for (const d of data) {
    const ts = Math.floor(new Date(d?.date_obs).getTime() / 1000);
    const value = nombre(d?.resultat_obs);
    if (!Number.isFinite(ts) || value == null) continue;
    if (!best || ts > best.ts) best = { ts, value };
  }
  return best;
}

// Number(null) vaut 0 et Number('') aussi : une mesure absente passerait pour
// une hauteur de zero, donc pour un etiage record. On convertit strictement.
function nombre(v) {
  if (v == null || v === '') return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

/**
 * Regle : depassement du maximum de la fenetre glissante.
 *
 * Le garde-fou `hydro_min_observations` evite le piege de la mise en service :
 * sans historique, le maximum des 30 jours est calcule sur trois points et
 * la moindre variation declenche une alerte.
 *
 * @param {number} valeur  mesure courante
 * @param {{max: number|null, n: number}} fenetre  stats de la fenetre
 * @returns {{level: string, depassement: boolean, max: number|null}}
 */
export function evaluerNiveau(valeur, fenetre) {
  if (!fenetre || fenetre.n < SEUILS.hydro_min_observations || fenetre.max == null) {
    return { level: 'vert', depassement: false, max: fenetre?.max ?? null, insuffisant: true };
  }
  const seuil = fenetre.max * SEUILS.hydro_marge;
  return {
    level: valeur > seuil ? 'orange' : 'vert',
    depassement: valeur > seuil,
    max: fenetre.max,
    insuffisant: false,
  };
}
