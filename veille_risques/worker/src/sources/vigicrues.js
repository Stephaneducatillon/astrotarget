// Vigicrues - vigilance crues par troncon de cours d'eau.
//
// Le document initial indiquait une consultation du site. Il existe en realite
// une API documentee (vigicrues.gouv.fr/services/v1.1) qui sert du JSON-LD et
// du GeoJSON : plus robuste qu'un scraping, et le GeoJSON s'affiche directement
// sur la carte Leaflet.
//
// Niveaux : 1 vert, 2 jaune, 3 orange, 4 rouge.

import { fetchJson } from '../lib/http.js';
import { TRONCONS_VIGICRUES, COULEURS } from '../config.js';

export async function collectVigicrues(env, troncons_cfg = TRONCONS_VIGICRUES) {
  if (troncons_cfg.length === 0) {
    return { troncons: [], pire: 'vert', libelle: null, configure: false };
  }

  const url = `${env.VIGICRUES_BASE}/InfoVigiCru.jsonld/?TypEntVigiCru=8`;
  const raw = await fetchJson(url, { headers: { Accept: 'application/json' } });
  return { ...parseVigicrues(raw, troncons_cfg), configure: true };
}

/** Filtre le flux national sur les seuls troncons surveilles. */
export function parseVigicrues(raw, surveilles) {
  const items = extraireItems(raw);
  const voulus = new Map(surveilles.map((t) => [String(t.code), t.nom]));
  const troncons = [];

  for (const item of items) {
    const code = String(item?.vicCdEntCru ?? item?.CdEntVigiCru ?? item?.code ?? '');
    if (!voulus.has(code)) continue;
    const n = Number(item?.vicNivInfoVigiCru ?? item?.NivInfoVigiCru ?? 1);
    troncons.push({
      code,
      nom: voulus.get(code),
      niveau: COULEURS[n] ?? 'vert',
    });
  }

  const pire = troncons
    .map((t) => (t.niveau === 'jaune' ? 'vert' : t.niveau))
    .reduce((a, b) => (poids(b) > poids(a) ? b : a), 'vert');

  const causes = troncons.filter((t) => poids(t.niveau) > 0).map((t) => t.nom);

  return { troncons, pire, libelle: causes.length ? `Crues : ${causes.join(', ')}` : null };
}

// Le JSON-LD imbrique les entrees sous des cles variables selon le type
// d'entite demande : on cherche le premier tableau d'objets qui porte un code
// de troncon plutot que de figer un chemin.
function extraireItems(raw) {
  if (!raw || typeof raw !== 'object') return [];
  const out = [];
  const visiter = (node, depth = 0) => {
    if (!node || typeof node !== 'object' || depth > 6) return;
    if (Array.isArray(node)) {
      for (const it of node) visiter(it, depth + 1);
      return;
    }
    if ('vicCdEntCru' in node || 'CdEntVigiCru' in node) out.push(node);
    for (const v of Object.values(node)) {
      if (v && typeof v === 'object') visiter(v, depth + 1);
    }
  };
  visiter(raw);
  return out;
}

function poids(c) {
  return { vert: 0, jaune: 0, orange: 1, rouge: 2 }[c] ?? 0;
}
