// Tests automatiques du site (node --test web/tests).
// Une régénération des tables ne doit rien casser sans qu'on le voie.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { setLoader, sampleAsync } from '../lib/tables.js';
import { compute, formatDec, formatRa, offsetToRadec, ARCSEC, TRANSIT } from '../lib/engine.js';
import { findPhenomena } from '../lib/phenomena.js';
import { drawSvg, View } from '../lib/render.js';
import { nightWindow, zonedToDate, tzOffsetMinutes } from '../lib/time.js';
import { INSTRUMENTS, SITES } from '../lib/data.js';
import { moonsCsv, moonRows } from '../lib/exports.js';
import * as photo from '../lib/photo.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const dataDir = path.join(here, '..', 'data') + path.sep;
const fixture = (f) => path.join(here, 'fixtures', f);
const toAb = (b) => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
setLoader(async (p) => (p.endsWith('.json') ? JSON.parse(fs.readFileSync(p)) : toAb(fs.readFileSync(p))), dataDir);

const CANTIN = SITES.find((s) => s.name === 'Cantin');
const CALERN = SITES.find((s) => s.name.startsWith('Calern'));
const T_EX = new Date(Date.UTC(2026, 9, 12, 4, 0, 0));
const near = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg ?? ''} ${a} ≠ ${b} ± ${tol}`);
const moon = (snap, name) => snap.moons.find((m) => m.name === name);

// ─── Tables : fidélité à PyEphem ────────────────────────────────────────────

test('les tables redonnent PyEphem à moins de 0,0003 rayon', async () => {
  const ref = JSON.parse(fs.readFileSync(fixture('reference.json')));
  let worst = 0, worstAngle = 0;
  for (const r of ref) {
    const t = await sampleAsync(r.planet, new Date(r.iso));
    for (const k in r.moons) {
      const [x, y, z] = t.moons[k];
      const [rx, ry, rz] = r.moons[k];
      worst = Math.max(worst, Math.abs(x - rx), Math.abs(y - ry), Math.abs(z - rz));
    }
    const dra = (((t.planet.ra - r.ra) + 3 * Math.PI) % (2 * Math.PI)) - Math.PI;
    worstAngle = Math.max(worstAngle, Math.abs(dra * Math.cos(r.dec)) / ARCSEC, Math.abs(t.planet.dec - r.dec) / ARCSEC);
    near(t.planet.radius, r.radius, 1e-4, 'rayon apparent');
  }
  assert.ok(worst < 3e-4, `écart max ${worst} rayon`);
  assert.ok(worstAngle < 0.001, `écart planète ${worstAngle}″`);
});

test('positions non disponibles en 2040 et pour Uranus après 2040', async () => {
  const s = await compute('Saturn', new Date(Date.UTC(2040, 5, 1)));
  assert.ok(s.moons.every((m) => !m.available));
  const u = await compute('Uranus', new Date(Date.UTC(2050, 5, 1)));
  assert.ok(u.moons.every((m) => !m.available));
  const j = await compute('Jupiter', new Date(Date.UTC(2050, 5, 1)));
  assert.ok(j.moons.every((m) => m.available));
  assert.ok(j.info.reducedPrecision);
});

test('hors période : erreur explicite', async () => {
  await assert.rejects(() => compute('Jupiter', new Date(Date.UTC(2070, 0, 1))), RangeError);
});

// ─── Exemple du cahier des charges (Cantin, 12 octobre 2026, 04:00 UT) ──────

test('écarts des galiléens', async () => {
  const s = await compute('Jupiter', T_EX, CANTIN);
  for (const [name, ew, ns] of [['Io', 70, -27], ['Europe', 93, -34], ['Ganymède', 237, -91], ['Callisto', -283, 106]]) {
    near(moon(s, name).eastArcsec, ew, 1, name);
    near(moon(s, name).northArcsec, ns, 1, name);
  }
  near(s.info.radius, 17.0, 0.1);
});

test('hauteurs, azimuts et lever de Jupiter', async () => {
  const s = await compute('Jupiter', T_EX, CANTIN);
  for (const [name, alt, az] of [['Io', 30.33, 103.71], ['Europe', 30.33, 103.71], ['Ganymède', 30.29, 103.68], ['Callisto', 30.42, 103.77]]) {
    near(moon(s, name).alt, alt, 0.03, name);
    near(moon(s, name).az, az, 0.03, name);
  }
  near(s.info.rise.getTime(), Date.UTC(2026, 9, 12, 0, 42, 24), 90e3, 'lever');
});

test('coordonnées J2000 cohérentes avec les écarts (projection gnomonique)', async () => {
  const s = await compute('Jupiter', T_EX, CANTIN);
  assert.equal(formatRa(moon(s, 'Io').raJ2000), '9 h 34 min 57.0 s');
  assert.equal(formatDec(moon(s, 'Io').decJ2000), '+15° 02′ 35″');
  const [ra, dec] = offsetToRadec(1, 0.3, 0, 0);
  near(ra, 1, 1e-12); near(dec, 0.3, 1e-12);
});

test('galiléens alignés sur l\'équateur de Jupiter', async () => {
  const s = await compute('Jupiter', T_EX, CANTIN);
  for (const m of s.moons) near((m.pa - s.info.polePa + 360) % 180, 90, 3, m.name);
});

test('Titan et Japet hors du cadre C2PU (exemple de cadrage)', async () => {
  const s = await compute('Saturn', new Date(Date.UTC(2026, 9, 11, 20)), CALERN);
  const half = INSTRUMENTS[0].field[0] / 2;
  near(half, 186, 2);
  near(moon(s, 'Titan').distArcsec, 193, 4);
  near(moon(s, 'Japet').distArcsec, 455, 4);
  near(s.info.ringTilt, -7.19, 0.1, 'inclinaison des anneaux');
});

// ─── Phénomènes ─────────────────────────────────────────────────────────────

test('phénomènes d\'Io, nuit du 11 au 12 octobre 2026', async () => {
  const [a, b] = nightWindow(T_EX, 'Europe/Paris');
  assert.equal(a.toISOString(), '2026-10-11T10:00:00.000Z');
  const ph = await findPhenomena('Jupiter', a, b, CANTIN);
  const occ = ph.find((p) => p.moon === 'Io' && p.kind === 'occultation');
  near(occ.start.getTime(), Date.UTC(2026, 9, 11, 21, 8, 40), 60e3, 'début occultation');
  near(occ.end.getTime(), Date.UTC(2026, 9, 11, 23, 26, 56), 60e3, 'fin occultation');
  const ecl = ph.find((p) => p.moon === 'Io' && p.kind === 'eclipse');
  // Entrée dans l'ombre géométrique : 20:04 (l'indicateur PyEphem donne 20:12, trop tard,
  // voir README) ; durée ≈ 2 h 18 (passage central, proche de l'équinoxe de Jupiter)
  near(ecl.start.getTime(), Date.UTC(2026, 9, 11, 20, 4, 10), 60e3, 'début éclipse');
  near(ecl.end.getTime(), Date.UTC(2026, 9, 11, 22, 22, 20), 60e3, 'fin éclipse');
  assert.equal(ecl.startObservable, false);
});

test('passages et ombres sur une semaine', async () => {
  const ph = await findPhenomena('Jupiter', new Date(Date.UTC(2026, 9, 1)), new Date(Date.UTC(2026, 9, 8)), null, { stepS: 300 });
  for (const m of ['Io', 'Europa']) {
    const kinds = new Set(ph.filter((p) => p.moon === m).map((p) => p.kind));
    for (const k of ['transit', 'shadow', 'occultation', 'eclipse']) assert.ok(kinds.has(k), `${m} ${k}`);
  }
  const tr = ph.find((p) => p.moon === 'Io' && p.kind === 'transit' && p.start && p.end);
  const dur = (tr.end - tr.start) / 3600e3;
  assert.ok(dur > 1.5 && dur < 2.6, `durée ${dur} h`);
  const mid = new Date((tr.start.getTime() + tr.end.getTime()) / 2);
  const s = await compute('Jupiter', mid);
  assert.equal(moon(s, 'Io').state, TRANSIT);
});

// ─── Fuseaux, schéma, exports ───────────────────────────────────────────────

test('fuseaux horaires et changement d\'heure', () => {
  assert.equal(tzOffsetMinutes(new Date(Date.UTC(2026, 6, 1)), 'Europe/Paris'), 120);
  assert.equal(tzOffsetMinutes(new Date(Date.UTC(2026, 11, 1)), 'Europe/Paris'), 60);
  assert.equal(zonedToDate(2026, 10, 25, 12, 0, 0, 'Europe/Paris').toISOString(), '2026-10-25T11:00:00.000Z');
  assert.equal(zonedToDate(2026, 10, 24, 12, 0, 0, 'Europe/Paris').toISOString(), '2026-10-24T10:00:00.000Z');
});

test('orientations du schéma', () => {
  assert.deepEqual(new View(0, false).apply(1, 0).map((v) => Math.round(v)), [-1, 0]);
  assert.deepEqual(new View(180, false).apply(1, 1).map((v) => Math.round(v)), [1, -1]);
  assert.deepEqual(new View(0, true).apply(1, 1).map((v) => Math.round(v)), [1, 1]);
});

test('schéma SVG et exports', async () => {
  const s = await compute('Saturn', new Date(Date.UTC(2026, 9, 11, 20)), CALERN);
  const svg = drawSvg(s, { instrument: INSTRUMENTS[0], night: true, rotation: 180 });
  assert.ok(svg.startsWith('<svg') && svg.endsWith('</svg>'));
  for (const n of ['Titan', 'Japet', 'Saturne', 'CielScore']) assert.ok(svg.includes(n), n);
  const csv = moonsCsv(s);
  assert.ok(csv.split('\n')[0].includes('CielScore'));
  assert.equal(csv.trim().split('\n').length, 2 + 8);
  assert.equal(moonRows(s, 9).length, s.moons.filter((m) => m.mag <= 9).length);
});

// ─── Option photo ───────────────────────────────────────────────────────────

test('lecture FITS, date et WCS (comparé à astropy)', () => {
  const buf = toAb(fs.readFileSync(fixture('wcs.fits')));
  const f = photo.parseFits(buf);
  assert.equal(f.width, 80);
  assert.equal(f.height, 60);
  assert.equal(f.gray[(60 - 1 - 10) * 80 + 20], 1000);   // pixel FITS (20, 10), retourné pour l'affichage
  assert.equal(photo.fitsDate(f.header).date.toISOString(), '2026-10-12T04:00:00.000Z');
  const w = photo.fitsWcs(f.header, f.height);
  near(w.scale, 0.06, 1e-6);
  const exp = JSON.parse(fs.readFileSync(fixture('wcs_expected.json')));
  for (const p of exp.points) {
    const [x, y] = photo.wcsPixel(w, p.ra * Math.PI / 180, p.dec * Math.PI / 180);
    near(x, p.x, 1e-3, 'x');
    near(y, p.y_display, 1e-3, 'y');
  }
});

test('date EXIF d\'un JPEG', () => {
  const d = photo.exifDate(toAb(fs.readFileSync(fixture('exif.jpg'))));
  assert.equal(d.toISOString(), '2026-10-12T06:00:05.000Z');
});

test('calage en deux points sur une image synthétique', async () => {
  const s = await compute('Jupiter', T_EX, CANTIN);
  for (const [rot, mirrored] of [[0, false], [37, false], [200, true]]) {
    const truth = new photo.Calibration([450, 300], 1.2, rot, mirrored);
    const W = 900, H = 600, gray = new Float32Array(W * H);
    let seed = 1;
    const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
    for (let i = 0; i < gray.length; i++) gray[i] = 100 + 6 * (rnd() - 0.5);
    for (let y = 280; y < 321; y++) for (let x = 430; x < 471; x++) {
      if ((x - 450) ** 2 + (y - 300) ** 2 < 400) gray[y * W + x] += 5000;
    }
    for (const m of s.moons) {
      const [mx, my] = truth.skyToPixel(m.eastArcsec, m.northArcsec);
      for (let y = Math.floor(my) - 6; y <= Math.floor(my) + 6; y++) for (let x = Math.floor(mx) - 6; x <= Math.floor(mx) + 6; x++) {
        if (x >= 0 && x < W && y >= 0 && y < H) gray[y * W + x] += 800 * Math.exp(-((x - mx) ** 2 + (y - my) ** 2) / 4.5);
      }
    }
    const img = { gray, width: W, height: H };
    const ref = moon(s, 'Ganymède');
    const refPx = photo.nearestSource(img, ...truth.skyToPixel(ref.eastArcsec, ref.northArcsec));
    const cal = photo.calibrate(photo.detectPlanetCenter(img), refPx, ref.eastArcsec, ref.northArcsec, mirrored);
    near(cal.scale, 1.2, 0.012, 'échelle');
    near(((cal.rotationDeg - rot) % 360 + 540) % 360 - 180, 0, 0.5, 'rotation');
    for (const m of s.moons) {
      const [x, y] = cal.skyToPixel(m.eastArcsec, m.northArcsec);
      if (x < 0 || x >= W || y < 0 || y >= H) continue;
      const fnd = photo.nearestSource(img, x, y);
      assert.ok(fnd && Math.hypot(fnd[0] - x, fnd[1] - y) < 1, m.name);
    }
  }
});
