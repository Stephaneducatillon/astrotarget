// Faux D1 / KV / fetch, juste assez fidele pour exercer le cycle complet.
// Ne reimplemente pas SQLite : reconnait les requetes reellement utilisees.

export function fakeDB() {
  const tables = {
    alert_state: new Map(),
    source_health: new Map(),
    observations: [],
    alert_log: [],
  };
  const journal = []; // ordre des operations, pour verifier les invariants

  const exec = (sql, args) => {
    if (sql.includes('FROM alert_state')) {
      journal.push('read:alert_state');
      return { results: [...tables.alert_state.values()] };
    }
    if (sql.includes('FROM source_health')) {
      journal.push('read:source_health');
      return { results: [...tables.source_health.values()] };
    }
    if (sql.includes('INSERT INTO alert_state')) {
      journal.push('write:alert_state');
      const [rule_id, level, label, since, last_notified, below_count] = args;
      tables.alert_state.set(rule_id, { rule_id, level, label, since, last_notified, below_count });
      return { meta: { changes: 1 } };
    }
    if (sql.includes('INSERT INTO source_health')) {
      journal.push('write:source_health');
      const [source_id, last_ok, last_try, last_error, fail_streak, notified] = args;
      tables.source_health.set(source_id, { source_id, last_ok, last_try, last_error, fail_streak, notified });
      return { meta: { changes: 1 } };
    }
    if (sql.includes('INSERT OR IGNORE INTO observations')) {
      journal.push('write:observations');
      const [source_id, station, metric, ts, value] = args;
      const cle = `${source_id}|${station}|${metric}|${ts}`;
      if (!tables.observations.some((o) => o._k === cle)) {
        tables.observations.push({ _k: cle, source_id, station, metric, ts, value });
      }
      return { meta: { changes: 1 } };
    }
    if (sql.includes('INSERT INTO alert_log')) {
      journal.push('write:alert_log');
      const [ts, rule_id, from_level, to_level, title, body, priority] = args;
      tables.alert_log.push({ ts, rule_id, from_level, to_level, title, body, priority });
      return { meta: { changes: 1 } };
    }
    if (sql.includes('MAX(value)')) {
      journal.push('read:fenetre');
      const [station, metric, depuis] = args;
      const vals = tables.observations
        .filter((o) => o.station === station && o.metric === metric && o.ts >= depuis)
        .map((o) => o.value);
      return { max: vals.length ? Math.max(...vals) : null, n: vals.length };
    }
    if (sql.includes('SELECT ts, value FROM observations')) {
      const [station, metric, depuis] = args;
      return { results: tables.observations
        .filter((o) => o.station === station && o.metric === metric && o.ts >= depuis)
        .sort((a, b) => a.ts - b.ts)
        .map((o) => ({ ts: o.ts, value: o.value })) };
    }
    if (sql.includes('DELETE FROM observations')) {
      const avant = tables.observations.length;
      tables.observations = tables.observations.filter((o) => o.ts >= args[0]);
      return { meta: { changes: avant - tables.observations.length } };
    }
    if (sql.includes('DELETE FROM alert_log')) {
      const avant = tables.alert_log.length;
      tables.alert_log = tables.alert_log.filter((o) => o.ts >= args[0]);
      return { meta: { changes: avant - tables.alert_log.length } };
    }
    throw new Error(`requete non geree par le faux D1 : ${sql.slice(0, 70)}`);
  };

  const prepare = (sql) => ({
    bind: (...args) => ({
      _sql: sql, _args: args,
      all: async () => exec(sql, args),
      first: async () => exec(sql, args),
      run: async () => exec(sql, args),
    }),
    all: async () => exec(sql, []),
    first: async () => exec(sql, []),
    run: async () => exec(sql, []),
  });

  return {
    prepare,
    batch: async (stmts) => {
      journal.push('batch');
      return stmts.map((s) => exec(s._sql, s._args));
    },
    _tables: tables,
    _journal: journal,
  };
}

export function fakeKV() {
  const m = new Map();
  let writes = 0;
  return {
    get: async (k, type) => {
      const v = m.get(k);
      if (v === undefined) return null;
      return type === 'json' ? JSON.parse(v) : v;
    },
    put: async (k, v) => { writes++; m.set(k, v); },
    _writes: () => writes,
    _raw: m,
  };
}

/** Remplace globalThis.fetch par un routeur pilote par des motifs d URL. */
export function mockFetch(routes) {
  const appels = [];
  globalThis.fetch = async (url, init = {}) => {
    const u = String(url);
    appels.push({ url: u, method: init.method ?? 'GET' });
    for (const [motif, rep] of Object.entries(routes)) {
      if (u.includes(motif)) {
        const r = typeof rep === 'function' ? rep(u, init) : rep;
        if (r instanceof Error) throw r;
        return new Response(JSON.stringify(r.body ?? r), {
          status: r.status ?? 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
    }
    return new Response('not found', { status: 404 });
  };
  return appels;
}

export function fakeEnv(overrides = {}) {
  return {
    DB: fakeDB(),
    KV: fakeKV(),
    MF_VIGILANCE_BASE: 'https://mf.test/v1',
    RTE_TOKEN_URL: 'https://rte.test/token',
    RTE_ECOWATT_URL: 'https://rte.test/ecowatt',
    HUBEAU_BASE: 'https://hubeau.test/v2',
    VIGICRUES_BASE: 'https://vigicrues.test/1',
    NTFY_BASE: 'https://ntfy.test',
    MF_API_KEY: 'cle-mf',
    RTE_CLIENT_ID: 'id',
    RTE_CLIENT_SECRET: 'secret',
    NTFY_TOPIC: 'sujet-tres-long-et-aleatoire-0123456789',
    ADMIN_TOKEN: 'jeton',
    ...overrides,
  };
}
