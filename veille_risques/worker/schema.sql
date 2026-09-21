-- Schema D1 - phase 1
-- wrangler d1 execute veille --file=schema.sql --remote

-- Etat courant de chaque regle d'alerte. C'est ce qui empeche de renotifier
-- 96 fois par jour le meme orange : on ne notifie que les transitions.
CREATE TABLE IF NOT EXISTS alert_state (
  rule_id       TEXT PRIMARY KEY,
  level         TEXT NOT NULL DEFAULT 'vert',  -- vert | orange | rouge
  label         TEXT,                          -- libelle affiche de la cause
  since         INTEGER NOT NULL,              -- epoch s du passage a ce niveau
  last_notified INTEGER,                       -- epoch s de la derniere notif
  below_count   INTEGER NOT NULL DEFAULT 0     -- hysteresis a la descente
);

-- Sante des sources. Une panne silencieuse est indiscernable d'une absence
-- d'alerte : on la trace et on la notifie.
CREATE TABLE IF NOT EXISTS source_health (
  source_id   TEXT PRIMARY KEY,
  last_ok     INTEGER,
  last_try    INTEGER,
  last_error  TEXT,
  fail_streak INTEGER NOT NULL DEFAULT 0,
  notified    INTEGER NOT NULL DEFAULT 0       -- 0/1, evite de repeter la panne
);

-- Series temporelles. Seule la regle "max des 30 derniers jours" en a besoin,
-- mais la table sert aussi a tracer les graphes du tableau de bord.
CREATE TABLE IF NOT EXISTS observations (
  source_id TEXT    NOT NULL,
  station   TEXT    NOT NULL,
  metric    TEXT    NOT NULL,   -- 'H' (mm) | 'Q' (l/s)
  ts        INTEGER NOT NULL,   -- epoch s de la mesure
  value     REAL    NOT NULL,
  PRIMARY KEY (source_id, station, metric, ts)
);
CREATE INDEX IF NOT EXISTS idx_obs_lookup ON observations (station, metric, ts);

-- Journal des notifications envoyees, pour verifier apres coup ce qui est parti.
CREATE TABLE IF NOT EXISTS alert_log (
  id        INTEGER PRIMARY KEY AUTOINCREMENT,
  ts        INTEGER NOT NULL,
  rule_id   TEXT    NOT NULL,
  from_level TEXT,
  to_level  TEXT,
  title     TEXT,
  body      TEXT,
  priority  TEXT
);
CREATE INDEX IF NOT EXISTS idx_alert_log_ts ON alert_log (ts);
