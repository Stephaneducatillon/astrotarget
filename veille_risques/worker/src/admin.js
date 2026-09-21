// Endpoints d'administration - a utiliser une fois, a la mise en service.
//
// Ils repondent a deux besoins concrets identifies avant de coder :
//   1. Les codes de station Hub'Eau et de troncon Vigicrues ne sont pas
//      devinables. On les releve ici, puis on les FIGE dans config.js.
//      Ne jamais les rechercher a chaque run.
//   2. La forme exacte des reponses d'API doit etre constatee, pas supposee.
//      /admin/probe sert la reponse brute tronquee de chaque source.
//
// Toutes ces routes passent par le controle d'acces de index.js.

import { fetchJson } from './lib/http.js';
import { json } from './lib/respond.js';

export async function routeAdmin(url, env) {
  const chemin = url.pathname.replace('/admin/', '');

  if (chemin === 'stations') {
    const q = url.searchParams.get('q') ?? 'Scarpe';
    const u =
      `${env.HUBEAU_BASE}/referentiel/stations` +
      `?libelle_cours_eau=${encodeURIComponent(q)}&format=json&size=50`;
    const raw = await fetchJson(u, { headers: { Accept: 'application/json' } });
    const stations = (raw?.data ?? []).map((s) => ({
      code: s.code_station,
      nom: s.libelle_station,
      cours_eau: s.libelle_cours_eau,
      commune: s.libelle_commune,
      en_service: s.en_service,
      lat: s.latitude_station,
      lon: s.longitude_station,
    }));
    return json({
      aide: 'Recopier les entrees utiles dans STATIONS_HYDRO (config.js), puis redeployer.',
      trouvees: stations.length,
      stations,
    });
  }

  if (chemin === 'troncons') {
    const q = (url.searchParams.get('q') ?? 'scarpe').toLowerCase();
    const raw = await fetchJson(`${env.VIGICRUES_BASE}/TronEntVigiCru.jsonld/`, {
      headers: { Accept: 'application/json' },
    });
    const tous = aplatir(raw);
    const troncons = tous
      .filter((t) => JSON.stringify(t).toLowerCase().includes(q))
      .map((t) => ({
        code: t.vicCdEntCru ?? t.CdEntVigiCru ?? null,
        nom: t.vicLbEntCru ?? t.LbEntVigiCru ?? null,
      }))
      .filter((t) => t.code);
    return json({
      aide: 'Recopier les entrees utiles dans TRONCONS_VIGICRUES (config.js), puis redeployer.',
      trouvees: troncons.length,
      troncons,
    });
  }

  // Constate la forme reelle des reponses. A lancer une fois apres avoir
  // defini les cles, pour valider les parseurs sans deviner.
  if (chemin === 'probe') {
    const cible = url.searchParams.get('source');
    const cibles = {
      vigilance: [
        `${env.MF_VIGILANCE_BASE}/cartevigilance/encours`,
        { apikey: env.MF_API_KEY, Accept: 'application/json' },
      ],
      hubeau: [
        `${env.HUBEAU_BASE}/referentiel/stations?format=json&size=1`,
        { Accept: 'application/json' },
      ],
      vigicrues: [`${env.VIGICRUES_BASE}/InfoVigiCru.jsonld/?TypEntVigiCru=8`, { Accept: 'application/json' }],
    };
    if (!cibles[cible]) {
      return json({ error: 'source inconnue', disponibles: Object.keys(cibles) }, 400);
    }
    const [u, headers] = cibles[cible];
    try {
      const raw = await fetchJson(u, { headers });
      return json({ source: cible, url: u, apercu: tronquer(raw) });
    } catch (err) {
      return json({ source: cible, url: u, erreur: err.message, kind: err.kind }, 502);
    }
  }

  if (chemin === 'notify-test') {
    const { envoyer } = await import('./notify.js');
    const r = await envoyer(env, {
      title: 'Test de notification',
      body: 'Si vous lisez ceci, ntfy est correctement configure.',
      tags: ['white_check_mark'],
      priority: 'low',
    });
    return json(r);
  }

  return json({ error: 'route admin inconnue' }, 404);
}

function aplatir(node, out = [], depth = 0) {
  if (!node || typeof node !== 'object' || depth > 6) return out;
  if (Array.isArray(node)) {
    for (const it of node) aplatir(it, out, depth + 1);
    return out;
  }
  if ('vicCdEntCru' in node || 'CdEntVigiCru' in node) out.push(node);
  for (const v of Object.values(node)) {
    if (v && typeof v === 'object') aplatir(v, out, depth + 1);
  }
  return out;
}

// Limite la profondeur et la taille : la reponse vigilance complete est
// volumineuse et sans interet au-dela de sa structure.
function tronquer(node, depth = 0) {
  if (depth > 4) return '…';
  if (Array.isArray(node)) {
    return node.slice(0, 2).map((n) => tronquer(n, depth + 1));
  }
  if (node && typeof node === 'object') {
    const out = {};
    for (const [k, v] of Object.entries(node).slice(0, 20)) out[k] = tronquer(v, depth + 1);
    return out;
  }
  return node;
}
