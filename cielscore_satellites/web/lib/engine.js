// Moteur de calcul : positions des satellites et informations sur la planète.
//
// Les positions viennent des tables précalculées (x vers l'est, y vers le sud,
// z vers la Terre, en rayons équatoriaux, repère du ciel géocentrique). Le
// Soleil, la hauteur, l'azimut, le lever et le coucher sont calculés dans le
// navigateur avec Astronomy Engine (licence MIT).

import * as A from '../vendor/astronomy.js';
import { PLANETS } from './data.js';
import { loadMeta, sampleAsync, sampleSync } from './tables.js';

export const ARCSEC = Math.PI / 180 / 3600;
const DEG = Math.PI / 180;
const TAU = 2 * Math.PI;

export const VISIBLE = 'visible';
export const TRANSIT = 'devant le disque';
export const OCCULTED = 'derrière le disque';
export const ECLIPSED = "dans l'ombre";
export const UNAVAILABLE = 'position non disponible';

// ─── Vecteurs et projections ────────────────────────────────────────────────

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const unit = (ra, dec) => [Math.cos(dec) * Math.cos(ra), Math.cos(dec) * Math.sin(ra), Math.sin(dec)];

/** Vecteurs unitaires (est, nord, vers la Terre) au point (ra, dec). */
export function skyBasis(ra, dec) {
  return [
    [-Math.sin(ra), Math.cos(ra), 0],
    [-Math.sin(dec) * Math.cos(ra), -Math.sin(dec) * Math.sin(ra), Math.cos(dec)],
    unit(ra, dec).map((v) => -v),
  ];
}

/** Projection gnomonique inverse : écarts (radians) → (ra, dec). */
export function offsetToRadec(ra0, dec0, dEast, dNorth) {
  const denom = Math.cos(dec0) - dNorth * Math.sin(dec0);
  const ra = ra0 + Math.atan2(dEast, denom);
  const dec = Math.atan2(Math.sin(dec0) + dNorth * Math.cos(dec0), Math.hypot(dEast, denom));
  return [((ra % TAU) + TAU) % TAU, dec];
}

// ─── Formats ────────────────────────────────────────────────────────────────

function splitSexa(value, decimals) {
  const f = 10 ** decimals;
  const total = Math.round(value * 3600 * f) / f;
  const a = Math.floor(total / 3600);
  const b = Math.floor((total - a * 3600) / 60);
  const c = total - a * 3600 - b * 60;
  return [a, b, c];
}

export function formatRa(ra, decimals = 1) {
  const h = (((ra / TAU) * 24) % 24 + 24) % 24;
  const [hh, mm, ss] = splitSexa(h, decimals);
  return `${hh % 24} h ${String(mm).padStart(2, '0')} min ${ss.toFixed(decimals).padStart(3 + decimals, '0')} s`;
}

export function formatDec(dec, decimals = 0) {
  const sign = dec >= 0 ? '+' : '−';
  const [dd, mm, ss] = splitSexa(Math.abs(dec) / DEG, decimals);
  const s = decimals ? ss.toFixed(decimals).padStart(3 + decimals, '0') : String(Math.round(ss)).padStart(2, '0');
  return `${sign}${String(dd).padStart(2, '0')}° ${String(mm).padStart(2, '0')}′ ${s}″`;
}

export function formatRaShort(ra) {
  const h = (((ra / TAU) * 24) % 24 + 24) % 24;
  const [hh, mm, ss] = splitSexa(h, 1);
  return `${hh % 24}h${String(mm).padStart(2, '0')}m${ss.toFixed(1).padStart(4, '0')}s`;
}

export function formatDecShort(dec) {
  const sign = dec >= 0 ? '+' : '−';
  const [dd, mm, ss] = splitSexa(Math.abs(dec) / DEG, 0);
  return `${sign}${dd}°${String(mm).padStart(2, '0')}′${String(Math.round(ss)).padStart(2, '0')}″`;
}

// ─── Géométrie de la planète ────────────────────────────────────────────────

export function poleRadec(planet, date) {
  const t = (date.getTime() - Date.UTC(2000, 0, 1, 12)) / 86400000 / 36525;
  const [a0, ar, d0, dr] = planet.pole;
  return [(a0 + ar * t) * DEG, (d0 + dr * t) * DEG];
}

/** Angle de position du pôle nord (°) et latitude planétocentrique de la Terre (°). */
export function poleGeometry(planet, date, ra, dec) {
  const [a0, d0] = poleRadec(planet, date);
  const pa = Math.atan2(Math.cos(d0) * Math.sin(a0 - ra),
    Math.sin(d0) * Math.cos(dec) - Math.cos(d0) * Math.sin(dec) * Math.cos(a0 - ra));
  const sinDe = -Math.sin(d0) * Math.sin(dec) - Math.cos(d0) * Math.cos(dec) * Math.cos(a0 - ra);
  return [((pa / DEG) % 360 + 360) % 360, Math.asin(Math.max(-1, Math.min(1, sinDe))) / DEG];
}

export function apparentAxisRatio(f, earthLatDeg) {
  const c = Math.cos(earthLatDeg * DEG);
  return Math.sqrt((1 - f) ** 2 * c * c + (1 - c * c));
}

/** Plus petite racine t > 0 de l'intersection rayon / ellipsoïde (rayon équatorial 1). */
export function rayEllipsoid(o, d, pole, f) {
  const c = 1 / (1 - f) ** 2 - 1;
  const dp = dot(d, pole), op = dot(o, pole);
  const a = dot(d, d) + c * dp * dp;
  const b = 2 * (dot(o, d) + c * op * dp);
  const cc = dot(o, o) + c * op * op - 1;
  const disc = b * b - 4 * a * cc;
  if (disc < 0) return null;
  const sq = Math.sqrt(disc);
  const roots = [(-b - sq) / (2 * a), (-b + sq) / (2 * a)].sort((x, y) => x - y);
  for (const t of roots) if (t > 0) return t;
  return null;
}

// ─── Astronomy Engine ───────────────────────────────────────────────────────

const aeObserver = (site) => new A.Observer(site.lat, site.lon, site.elevation);

function horizon(time, site, raRad, decRad) {
  const h = A.Horizon(time, aeObserver(site), raRad / DEG / 15, decRad / DEG, 'normal');
  return [h.altitude, h.azimuth];
}

/** Hauteurs (°) de la planète et du Soleil. */
export function altitudes(planet, date, site) {
  const time = A.MakeTime(date);
  const obs = aeObserver(site);
  const eq = A.Equator(planet.body, time, obs, true, true);
  const sun = A.Equator('Sun', time, obs, true, true);
  return [
    A.Horizon(time, obs, eq.ra, eq.dec, 'normal').altitude,
    A.Horizon(time, obs, sun.ra, sun.dec, 'normal').altitude,
  ];
}

/** Direction planète → Soleil dans le repère (est, nord, vers la Terre). */
function sunDirection(planet, time, basis) {
  const s = A.GeoVector('Sun', time, true);
  const p = A.GeoVector(planet.body, time, true);
  const v = [s.x - p.x, s.y - p.y, s.z - p.z];
  const n = Math.hypot(...v);
  return basis.map((b) => dot(v, b) / n);
}

function poleInSky(planet, date, basis) {
  const [a0, d0] = poleRadec(planet, date);
  const p = unit(a0, d0);
  return basis.map((b) => dot(p, b));
}

/** Indicateurs de phénomènes pour chaque satellite (tables déjà chargées). */
export function moonFlags(planet, date, tab) {
  const basis = skyBasis(tab.planet.ra, tab.planet.dec);
  const sun = sunDirection(planet, A.MakeTime(date), basis);
  const pole = poleInSky(planet, date, basis);
  const f = planet.flattening;
  const c = 1 / (1 - f) ** 2 - 1;
  const out = {};
  for (const key in tab.moons) {
    const v = tab.moons[key];
    if (!v) { out[key] = null; continue; }
    const pos = [v[0], -v[1], v[2]];
    const occulted = pos[2] < 0 && rayEllipsoid(pos, [0, 0, 1], pole, f) !== null;
    const eclipsed = rayEllipsoid(pos, sun, pole, f) !== null;
    const transit = pos[2] > 0 && rayEllipsoid(pos, [0, 0, -1], pole, f) !== null;
    let shadow = null;
    if (!eclipsed) {
      const back = sun.map((x) => -x);
      const t = rayEllipsoid(pos, back, pole, f);
      if (t !== null) {
        const hit = pos.map((x, i) => x + t * back[i]);
        const k = c * dot(hit, pole);
        if (hit[2] + k * pole[2] > 0) shadow = [hit[0], hit[1]];
      }
    }
    out[key] = { transit, occulted, eclipsed, shadow };
  }
  return out;
}

/** Version synchrone (tables de l'année déjà chargées) utilisée par les phénomènes. */
export function flagsAt(planet, date) {
  const tab = sampleSync(planet.key, date);
  return tab ? moonFlags(planet, date, tab) : null;
}

// ─── Calcul principal ───────────────────────────────────────────────────────

function riseSet(planet, date, site, altNow) {
  const obs = aeObserver(site);
  const time = A.MakeTime(date);
  const res = { rise: null, transit: null, set: null, transitAlt: null };
  try {
    const rise = altNow > 0
      ? A.SearchRiseSet(planet.body, obs, +1, time, -1.2)
      : A.SearchRiseSet(planet.body, obs, +1, time, 1.2);
    if (!rise) return { ...res, circumpolar: altNow > 0, neverUp: altNow <= 0 };
    res.rise = rise.date;
    const tr = A.SearchHourAngle(planet.body, obs, 0, rise, +1);
    res.transit = tr.time.date;
    res.transitAlt = tr.hor.altitude;
    const set = A.SearchRiseSet(planet.body, obs, -1, rise, 1.2);
    res.set = set ? set.date : null;
  } catch (e) {
    // pas de lever ni de coucher sur la période
  }
  return res;
}

/**
 * Positions de tous les satellites de ``planetKey`` à l'instant ``date``.
 * ``site`` : {name, lat, lon, elevation} ou null (géocentrique).
 */
export async function compute(planetKey, date, site = null) {
  const planet = PLANETS[planetKey];
  const meta = await loadMeta(planetKey);
  const tab = await sampleAsync(planetKey, date);
  if (!tab) throw new RangeError(`Pas de tables pour ${date.getUTCFullYear()}`);
  const time = A.MakeTime(date);
  const p = tab.planet;
  const radius = p.radius;
  const raJ = ((p.ra % TAU) + TAU) % TAU, decJ = p.dec;
  let raA = ((p.gra % TAU) + TAU) % TAU, decA = p.gdec;
  const [polePa, earthLat] = poleGeometry(planet, date, raJ, decJ);
  const illum = A.Illumination(planet.body, time);

  const info = {
    name: planet.name, key: planet.key,
    raJ2000: raJ, decJ2000: decJ, raApp: raA, decApp: decA,
    alt: null, az: null, sunAlt: null,
    radius, diameter: 2 * radius, mag: illum.mag,
    earthDistance: illum.geo_dist, sunDistance: illum.helio_dist,
    phaseAngle: illum.phase_angle,
    polePa, earthLat, axisRatio: apparentAxisRatio(planet.flattening, earthLat),
    flattening: planet.flattening, ringOuter: planet.ringOuter,
    // inclinaison des anneaux = latitude de la Terre vue de Saturne (même signe que PyEphem)
    ringTilt: planet.key === 'Saturn' ? earthLat : null,
    cml1: null, cml2: null,
    rise: null, transit: null, set: null, transitAlt: null, circumpolar: false, neverUp: false,
    reducedPrecision: meta.reduced_precision_from && date >= new Date(meta.reduced_precision_from),
    source: meta.source,
  };
  if (planet.key === 'Jupiter') {
    const days = (date.getTime() - Date.UTC(date.getUTCFullYear(), 0, 1)) / 86400000;
    const [r1, r2] = meta.cml_rate_rad_per_day;
    info.cml1 = (((p.cml1 + r1 * days) / DEG) % 360 + 360) % 360;
    info.cml2 = (((p.cml2 + r2 * days) / DEG) % 360 + 360) % 360;
  }

  if (site) {
    // Coordonnées du jour topocentriques : tables + parallaxe (Astronomy Engine)
    const obs = aeObserver(site);
    const topo = A.Equator(planet.body, time, obs, true, true);
    const geoVec = A.RotateVector(A.Rotation_EQJ_EQD(time), A.GeoVector(planet.body, time, true));
    const geo = A.EquatorFromVector(geoVec);
    let dRa = (topo.ra - geo.ra) * 15 * DEG;
    dRa = ((dRa + Math.PI) % TAU + TAU) % TAU - Math.PI;
    raA = ((raA + dRa) % TAU + TAU) % TAU;
    decA += (topo.dec - geo.dec) * DEG;
    info.raApp = raA; info.decApp = decA;
    [info.alt, info.az] = horizon(time, site, raA, decA);
    const sun = A.Equator('Sun', time, obs, true, true);
    info.sunAlt = A.Horizon(time, obs, sun.ra, sun.dec, 'normal').altitude;
    Object.assign(info, riseSet(planet, date, site, info.alt));
  }

  const flags = moonFlags(planet, date, tab);
  const moons = [];
  for (const s of meta.series.slice(1)) {
    const v = tab.moons[s.name];
    const m = { key: s.name, name: s.label, mag: s.h_mag + 5 * Math.log10(info.sunDistance * info.earthDistance) };
    if (!v) {
      Object.assign(m, { state: UNAVAILABLE, available: false });
      moons.push(m);
      continue;
    }
    const east = v[0], north = -v[1];
    const eAs = east * radius, nAs = north * radius;
    const [raMJ, decMJ] = offsetToRadec(raJ, decJ, eAs * ARCSEC, nAs * ARCSEC);
    const [raMA, decMA] = offsetToRadec(raA, decA, eAs * ARCSEC, nAs * ARCSEC);
    const fl = flags[s.name];
    const state = fl.occulted ? OCCULTED : fl.eclipsed ? ECLIPSED : fl.transit ? TRANSIT : VISIBLE;
    Object.assign(m, {
      available: true,
      raJ2000: raMJ, decJ2000: decMJ, raApp: raMA, decApp: decMA,
      alt: null, az: null,
      eastRadii: east, northRadii: north, depthRadii: v[2],
      eastArcsec: eAs, northArcsec: nAs,
      distArcsec: Math.hypot(eAs, nAs), distRadii: Math.hypot(east, north),
      pa: ((Math.atan2(eAs, nAs) / DEG) % 360 + 360) % 360,
      transit: fl.transit, occulted: fl.occulted, eclipsed: fl.eclipsed, shadow: fl.shadow, state,
    });
    if (site) [m.alt, m.az] = horizon(time, site, raMA, decMA);
    moons.push(m);
  }
  moons.sort((a, b) => (a.available ? a.distArcsec : 1e9) - (b.available ? b.distArcsec : 1e9));
  return { planet, when: date, site, info, moons };
}
