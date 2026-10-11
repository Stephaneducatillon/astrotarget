// Fuseaux horaires (Intl) et formats de date.

const fmtCache = new Map();

function formatter(tz) {
  if (!fmtCache.has(tz)) {
    fmtCache.set(tz, new Intl.DateTimeFormat('en-CA', {
      timeZone: tz, hourCycle: 'h23', year: 'numeric', month: '2-digit', day: '2-digit',
      hour: '2-digit', minute: '2-digit', second: '2-digit',
    }));
  }
  return fmtCache.get(tz);
}

/** Composantes de la date dans le fuseau ``tz``. */
export function zonedParts(date, tz) {
  const p = {};
  for (const { type, value } of formatter(tz).formatToParts(date)) p[type] = value;
  return {
    y: +p.year, m: +p.month, d: +p.day, h: +p.hour % 24, mi: +p.minute, s: +p.second,
  };
}

/** Décalage du fuseau (minutes, positif à l'est) à l'instant ``date``. */
export function tzOffsetMinutes(date, tz) {
  const p = zonedParts(date, tz);
  const asUtc = Date.UTC(p.y, p.m - 1, p.d, p.h, p.mi, p.s);
  return Math.round((asUtc - Math.floor(date.getTime() / 1000) * 1000) / 60000);
}

/** Heure locale (fuseau ``tz``) → Date (instant UTC). */
export function zonedToDate(y, m, d, h, mi, s, tz) {
  const guess = Date.UTC(y, m - 1, d, h, mi, s);
  let t = guess - tzOffsetMinutes(new Date(guess), tz) * 60000;
  t = guess - tzOffsetMinutes(new Date(t), tz) * 60000;
  return new Date(t);
}

export function tzAbbrev(date, tz) {
  const p = new Intl.DateTimeFormat('fr-FR', { timeZone: tz, timeZoneName: 'short' })
    .formatToParts(date).find((x) => x.type === 'timeZoneName');
  return p ? p.value : tz;
}

const pad = (n, w = 2) => String(n).padStart(w, '0');

/** « JJ/MM/AAAA HH:MM:SS » en UT ou dans le fuseau donné. */
export function fmtDateTime(date, tz = null, { seconds = true, year = true } = {}) {
  const p = tz ? zonedParts(date, tz) : {
    y: date.getUTCFullYear(), m: date.getUTCMonth() + 1, d: date.getUTCDate(),
    h: date.getUTCHours(), mi: date.getUTCMinutes(), s: date.getUTCSeconds(),
  };
  const day = year ? `${pad(p.d)}/${pad(p.m)}/${p.y}` : `${pad(p.d)}/${pad(p.m)}`;
  return `${day} ${pad(p.h)}:${pad(p.mi)}${seconds ? `:${pad(p.s)}` : ''}`;
}

export function fmtTime(date, tz = null, seconds = false) {
  return fmtDateTime(date, tz, { seconds }).split(' ')[1];
}

/** Nuit contenant ``date`` : de midi local à midi local le lendemain. */
export function nightWindow(date, tz) {
  const p = zonedParts(date, tz);
  let day = new Date(Date.UTC(p.y, p.m - 1, p.d));
  if (p.h < 12) day = new Date(day.getTime() - 86400000);
  const start = zonedToDate(day.getUTCFullYear(), day.getUTCMonth() + 1, day.getUTCDate(), 12, 0, 0, tz);
  const next = new Date(day.getTime() + 86400000);
  const end = zonedToDate(next.getUTCFullYear(), next.getUTCMonth() + 1, next.getUTCDate(), 12, 0, 0, tz);
  return [start, end];
}
