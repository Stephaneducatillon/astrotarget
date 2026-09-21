// Vigilance Meteo-France (carte en cours).
//
// Acces : cle generee sur portail-api.meteofrance.fr, en-tete `apikey`.
// Creer la cle avec une duree de 0 (illimitee) : une cle expiree est une panne
// silencieuse, exactement ce qu'on cherche a eviter.
//
// Le parsing est volontairement tolerant. La structure exacte n'a pas pu etre
// verifiee en direct (acces reseau restreint pendant l'ecriture) : la fonction
// explore la reponse au lieu de supposer un chemin unique, et `GET /admin/probe`
// permet de constater la forme reelle au premier run.

import { fetchJson } from '../lib/http.js';
import { DEPARTEMENTS, PHENOMENES, COULEURS } from '../config.js';

export async function collectVigilance(env) {
  const url = `${env.MF_VIGILANCE_BASE}/cartevigilance/encours`;
  const raw = await fetchJson(url, {
    headers: { apikey: env.MF_API_KEY, Accept: 'application/json' },
  });
  return parseVigilance(raw);
}

/**
 * Extrait, par departement, la couleur maximale et les phenomenes concernes.
 * @returns {{departements: object, pire: string, libelle: string|null}}
 */
export function parseVigilance(raw) {
  const domaines = collectDomains(raw);
  const departements = {};

  for (const dep of DEPARTEMENTS) {
    const entrees = domaines.filter((d) => String(d.domain_id) === dep);
    let maxColor = 1;
    const phenomenes = [];

    for (const e of entrees) {
      maxColor = Math.max(maxColor, toInt(e.max_color_id));
      for (const p of asArray(e.phenomenon_items)) {
        const id = toInt(p.phenomenon_id);
        const c = toInt(p.phenomenon_max_color_id);
        if (!PHENOMENES[id]) continue;            // phenomene hors perimetre
        maxColor = Math.max(maxColor, c);
        if (c >= 3) phenomenes.push({ nom: PHENOMENES[id], couleur: COULEURS[c] });
      }
    }
    departements[dep] = { couleur: COULEURS[maxColor] ?? 'vert', phenomenes };
  }

  // Le jaune n'est pas un niveau familial : il redescend a vert.
  const pire = Object.values(departements)
    .map((d) => (d.couleur === 'jaune' ? 'vert' : d.couleur))
    .reduce((a, b) => (poids(b) > poids(a) ? b : a), 'vert');

  const causes = [];
  for (const [dep, d] of Object.entries(departements)) {
    for (const p of d.phenomenes) causes.push(`${p.nom} (${dep})`);
  }

  return { departements, pire, libelle: causes.length ? causes.join(', ') : null };
}

function poids(c) {
  return { vert: 0, jaune: 0, orange: 1, rouge: 2 }[c] ?? 0;
}

// La reponse imbrique les domaines sous product.periods[].timelaps.domain_ids[].
// On descend l'arbre a la recherche de tout objet portant un `domain_id` plutot
// que de coder en dur un chemin qui changera.
function collectDomains(node, out = [], depth = 0) {
  if (!node || typeof node !== 'object' || depth > 8) return out;
  if (Array.isArray(node)) {
    for (const item of node) collectDomains(item, out, depth + 1);
    return out;
  }
  if ('domain_id' in node) out.push(node);
  for (const v of Object.values(node)) {
    if (v && typeof v === 'object') collectDomains(v, out, depth + 1);
  }
  return out;
}

function asArray(v) {
  return Array.isArray(v) ? v : [];
}

function toInt(v) {
  const n = parseInt(v, 10);
  return Number.isFinite(n) ? n : 1;
}
