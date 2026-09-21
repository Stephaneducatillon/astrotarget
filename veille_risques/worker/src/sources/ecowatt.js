// Ecowatt (RTE) - signal de tension sur le reseau electrique, J a J+3.
//
// Deux points qui font la difference entre un connecteur correct et un
// connecteur qui se fait limiter :
//
//  1. Le jeton OAuth2 vaut ~2 h. On le met en cache dans KV. Se
//     reauthentifier a chaque run, c'est 96 authentifications par jour pour
//     2 publications reelles.
//  2. Ecowatt publie la veille a 17 h et met a jour a 10 h 30. L'interroger
//     toutes les 15 min n'apporte rien : le collecteur s'auto-limite a un
//     appel toutes les 3 h, et sert la derniere valeur connue entre-temps.

import { fetchJson, SourceError } from '../lib/http.js';

const TOKEN_KEY = 'rte:token';
const CACHE_KEY = 'rte:ecowatt';
const CACHE_TTL = 3 * 3600;
const MARGE_EXPIRATION = 300; // on renouvelle 5 min avant l'echeance reelle

export async function collectEcowatt(env, now = Math.floor(Date.now() / 1000)) {
  const cached = await env.KV.get(CACHE_KEY, 'json');
  if (cached && now - cached.fetched_at < CACHE_TTL) {
    return { ...cached.value, depuis_cache: true };
  }

  const token = await getToken(env, now);
  const raw = await fetchJson(env.RTE_ECOWATT_URL, {
    headers: { Authorization: `Bearer ${token}`, Accept: 'application/json' },
  });

  const value = parseEcowatt(raw);
  await env.KV.put(CACHE_KEY, JSON.stringify({ fetched_at: now, value }));
  return { ...value, depuis_cache: false };
}

async function getToken(env, now) {
  const cached = await env.KV.get(TOKEN_KEY, 'json');
  if (cached && cached.expires_at > now + MARGE_EXPIRATION) return cached.token;

  if (!env.RTE_CLIENT_ID || !env.RTE_CLIENT_SECRET) {
    throw new SourceError('auth', 'identifiants RTE absents (secrets non definis)');
  }

  const basic = btoa(`${env.RTE_CLIENT_ID}:${env.RTE_CLIENT_SECRET}`);
  const body = await fetchJson(env.RTE_TOKEN_URL, {
    method: 'POST',
    headers: {
      Authorization: `Basic ${basic}`,
      'Content-Type': 'application/x-www-form-urlencoded',
    },
    body: 'grant_type=client_credentials',
  });

  const token = body?.access_token;
  if (!token) throw new SourceError('auth', 'jeton RTE absent de la reponse');

  const ttl = Number(body.expires_in) || 7200;
  await env.KV.put(
    TOKEN_KEY,
    JSON.stringify({ token, expires_at: now + ttl }),
    { expirationTtl: Math.max(ttl, 60) },
  );
  return token;
}

/**
 * dvalue : 1 vert, 2 orange, 3 rouge.
 * @returns {{jours: Array, pire: string, libelle: string|null}}
 */
export function parseEcowatt(raw) {
  const signaux = Array.isArray(raw?.signals) ? raw.signals : [];
  const jours = signaux
    .map((s) => ({
      jour: s?.jour ?? null,
      niveau: { 1: 'vert', 2: 'orange', 3: 'rouge' }[Number(s?.dvalue)] ?? 'vert',
      message: s?.message ?? null,
      // Heures de tension, utiles pour savoir quand decaler les usages.
      heures: (Array.isArray(s?.values) ? s.values : [])
        .filter((v) => Number(v?.hvalue) >= 2)
        .map((v) => Number(v.pas)),
    }))
    .slice(0, 4); // J a J+3

  const pire = jours.reduce(
    (a, j) => (poids(j.niveau) > poids(a) ? j.niveau : a),
    'vert',
  );

  // La regle familiale declenche sur "aujourd'hui OU dans les 3 jours" : on
  // precise lequel, sinon la notification est inexploitable.
  const premier = jours.find((j) => j.niveau !== 'vert');
  const libelle = premier
    ? `Ecowatt ${premier.niveau} le ${premier.jour ?? '?'}` +
      (premier.heures.length ? ` (${formatHeures(premier.heures)})` : '')
    : null;

  return { jours, pire, libelle };
}

function formatHeures(heures) {
  const tri = [...heures].sort((a, b) => a - b);
  return `${String(tri[0]).padStart(2, '0')}h-${String(tri[tri.length - 1] + 1).padStart(2, '0')}h`;
}

function poids(n) {
  return { vert: 0, orange: 1, rouge: 2 }[n] ?? 0;
}
