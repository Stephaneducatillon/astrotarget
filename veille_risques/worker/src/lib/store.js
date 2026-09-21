// Acces D1 et KV.
//
// Discipline de quota (plan gratuit, limites appliquees reellement depuis le
// 1er septembre 2026) :
//   - KV : UNE seule ecriture par run (l'instantane agrege). 96/jour contre
//     1000 autorisees.
//   - D1 : ecritures groupees en batch, purge quotidienne.

import { TEMPO } from '../config.js';

export const SNAPSHOT_KEY = 'snapshot:latest';

// --- Instantane servi a la PWA ---------------------------------------------

export async function lireSnapshot(env) {
  return (await env.KV.get(SNAPSHOT_KEY, 'json')) ?? null;
}

export async function ecrireSnapshot(env, snapshot) {
  await env.KV.put(SNAPSHOT_KEY, JSON.stringify(snapshot));
}

// --- Etats d'alerte ---------------------------------------------------------

export async function lireEtats(env) {
  const { results } = await env.DB.prepare(
    'SELECT rule_id, level, label, since, last_notified, below_count FROM alert_state',
  ).all();
  const map = new Map();
  for (const r of results ?? []) map.set(r.rule_id, r);
  return map;
}

export function upsertEtat(env, ruleId, etat) {
  return env.DB.prepare(
    `INSERT INTO alert_state (rule_id, level, label, since, last_notified, below_count)
     VALUES (?1, ?2, ?3, ?4, ?5, ?6)
     ON CONFLICT(rule_id) DO UPDATE SET
       level = ?2, label = ?3, since = ?4, last_notified = ?5, below_count = ?6`,
  ).bind(ruleId, etat.level, etat.label ?? null, etat.since, etat.last_notified ?? null, etat.below_count);
}

export function journaliserAlerte(env, ts, ruleId, notif, title, body) {
  return env.DB.prepare(
    `INSERT INTO alert_log (ts, rule_id, from_level, to_level, title, body, priority)
     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)`,
  ).bind(ts, ruleId, notif.from ?? null, notif.to ?? null, title, body, notif.priority ?? null);
}

// --- Observations -----------------------------------------------------------

export function insererObservation(env, sourceId, station, metric, ts, value) {
  return env.DB.prepare(
    `INSERT OR IGNORE INTO observations (source_id, station, metric, ts, value)
     VALUES (?1, ?2, ?3, ?4, ?5)`,
  ).bind(sourceId, station, metric, ts, value);
}

/** Statistiques de la fenetre glissante, pour la regle du maximum sur 30 jours. */
export async function fenetreGlissante(env, station, metric, now, jours) {
  const depuis = now - jours * 86400;
  const row = await env.DB.prepare(
    `SELECT MAX(value) AS max, COUNT(*) AS n FROM observations
     WHERE station = ?1 AND metric = ?2 AND ts >= ?3`,
  ).bind(station, metric, depuis).first();
  return { max: row?.max ?? null, n: Number(row?.n ?? 0) };
}

/** Serie destinee au graphe du tableau de bord. */
export async function serie(env, station, metric, now, jours = 7, pas = 12) {
  const depuis = now - jours * 86400;
  const { results } = await env.DB.prepare(
    `SELECT ts, value FROM observations
     WHERE station = ?1 AND metric = ?2 AND ts >= ?3
     ORDER BY ts ASC`,
  ).bind(station, metric, depuis).all();
  // Decimation cote base evitee volontairement : la fenetre est petite, et
  // un ORDER BY simple coute moins cher en lignes lues qu'une agregation.
  return (results ?? []).filter((_, i) => i % pas === 0);
}

// --- Sante des sources ------------------------------------------------------

export async function lireSante(env) {
  const { results } = await env.DB.prepare(
    'SELECT source_id, last_ok, last_try, last_error, fail_streak, notified FROM source_health',
  ).all();
  const map = new Map();
  for (const r of results ?? []) map.set(r.source_id, r);
  return map;
}

export function upsertSante(env, sourceId, s) {
  return env.DB.prepare(
    `INSERT INTO source_health (source_id, last_ok, last_try, last_error, fail_streak, notified)
     VALUES (?1, ?2, ?3, ?4, ?5, ?6)
     ON CONFLICT(source_id) DO UPDATE SET
       last_ok = ?2, last_try = ?3, last_error = ?4, fail_streak = ?5, notified = ?6`,
  ).bind(sourceId, s.last_ok ?? null, s.last_try, s.last_error ?? null, s.fail_streak, s.notified ? 1 : 0);
}

// --- Maintenance ------------------------------------------------------------

export async function purger(env, now = Math.floor(Date.now() / 1000)) {
  const limiteObs = now - TEMPO.retention_jours * 86400;
  const limiteLog = now - 365 * 86400;
  const r = await env.DB.batch([
    env.DB.prepare('DELETE FROM observations WHERE ts < ?1').bind(limiteObs),
    env.DB.prepare('DELETE FROM alert_log WHERE ts < ?1').bind(limiteLog),
  ]);
  return r.map((x) => x.meta?.changes ?? 0);
}
