// Phénomènes des satellites : passages, occultations, éclipses, ombres sur le disque.

import { PLANETS } from './data.js';
import { altitudes, flagsAt } from './engine.js';
import { preload } from './tables.js';

export const KINDS = [
  ['transit', 'Passage devant le disque'],
  ['occultation', 'Occultation (derrière le disque)'],
  ['eclipse', "Éclipse (dans l'ombre de la planète)"],
  ['shadow', 'Ombre sur le disque'],
];
const LABEL = Object.fromEntries(KINDS);

function boolFlags(planet, date) {
  const raw = flagsAt(planet, date);
  if (!raw) return null;
  const out = {};
  for (const k in raw) {
    const f = raw[k];
    out[k] = f ? {
      transit: f.transit, occultation: f.occulted, eclipse: f.eclipsed, shadow: f.shadow !== null,
    } : null;
  }
  return out;
}

function refine(planet, moon, kind, t0, t1, v0, tolMs = 1000) {
  while (t1 - t0 > tolMs) {
    const tm = new Date((t0.getTime() + t1.getTime()) / 2);
    const f = boolFlags(planet, tm)?.[moon];
    if (!f) break;
    if (f[kind] === v0) t0 = tm; else t1 = tm;
  }
  return new Date((t0.getTime() + t1.getTime()) / 2);
}

export function isObservable(planet, site, date, minAlt = 0, maxSun = -0.833) {
  if (!site) return null;
  const [pAlt, sAlt] = altitudes(planet, date, site);
  return pAlt > minAlt && sAlt < maxSun;
}

/** Phénomènes entre ``start`` et ``end`` (Date), heures à la seconde près. */
export async function findPhenomena(planetKey, start, end, site = null,
  { stepS = 60, minAlt = 0, maxSun = -0.833 } = {}) {
  const planet = PLANETS[planetKey];
  await preload(planetKey, [start, end]);
  const step = stepS * 1000;
  const open = new Map();
  const done = [];

  let tPrev = start;
  let fPrev = boolFlags(planet, tPrev);
  if (!fPrev) return [];
  for (const moon in fPrev) {
    if (!fPrev[moon]) continue;
    for (const [kind] of KINDS) {
      if (fPrev[moon][kind]) open.set(`${moon}|${kind}`, { moon, kind, label: LABEL[kind], start: null, end: null });
    }
  }
  let t = start;
  while (t < end) {
    t = new Date(Math.min(t.getTime() + step, end.getTime()));
    const f = boolFlags(planet, t);
    if (!f) break;
    for (const moon in f) {
      if (!f[moon] || !fPrev[moon]) continue;
      for (const [kind] of KINDS) {
        const v0 = fPrev[moon][kind], v1 = f[moon][kind];
        if (v0 === v1) continue;
        const te = refine(planet, moon, kind, tPrev, t, v0);
        const key = `${moon}|${kind}`;
        if (v1) {
          open.set(key, { moon, kind, label: LABEL[kind], start: te, end: null });
        } else {
          const ph = open.get(key) || { moon, kind, label: LABEL[kind], start: null, end: null };
          open.delete(key);
          ph.end = te;
          done.push(ph);
        }
      }
    }
    tPrev = t; fPrev = f;
  }
  done.push(...open.values());
  for (const ph of done) {
    ph.startObservable = ph.start ? isObservable(planet, site, ph.start, minAlt, maxSun) : null;
    ph.endObservable = ph.end ? isObservable(planet, site, ph.end, minAlt, maxSun) : null;
  }
  done.sort((a, b) => (a.start || a.end || start) - (b.start || b.end || start));
  return done;
}
