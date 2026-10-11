// CielScore Satellites — interface du site statique.

import {
  DEFAULT_TZ, INSTRUMENTS, Instrument, ORIENTATIONS, PLANETS, SIGNATURE, SITES, YEAR_MAX, YEAR_MIN,
} from './lib/data.js';
import { compute, formatDec, formatRa, formatDecShort, formatRaShort, VISIBLE } from './lib/engine.js';
import { findPhenomena } from './lib/phenomena.js';
import { drawSvg } from './lib/render.js';
import {
  download, moonRows, moonsCsv, phenomenaRows, rowsCsv, svgToPng,
} from './lib/exports.js';
import { fmtDateTime, nightWindow, tzAbbrev, zonedParts, zonedToDate } from './lib/time.js';
import * as photoLib from './lib/photo.js';

const $ = (id) => document.getElementById(id);
const pad = (n) => String(n).padStart(2, '0');
const esc = (s) => String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

// ─── Préférences (confort, facultatif) ─────────────────────────────────────

const PREF_KEY = 'cielscore-satellites';
function loadPrefs() {
  try { return JSON.parse(localStorage.getItem(PREF_KEY)) || {}; } catch (e) { return {}; }
}
function savePrefs(p) {
  try { localStorage.setItem(PREF_KEY, JSON.stringify(p)); } catch (e) { /* stockage indisponible */ }
}

// ─── Remplissage des listes ─────────────────────────────────────────────────

function fillSelect(sel, items, selected = 0) {
  sel.innerHTML = items.map((t, i) => `<option value="${i}"${i === selected ? ' selected' : ''}>${esc(t)}</option>`).join('');
}

const zones = (Intl.supportedValuesOf ? Intl.supportedValuesOf('timeZone') : [DEFAULT_TZ, 'UTC'])
  .filter((z) => z.includes('/') || z === 'UTC');
$('tz').innerHTML = zones.map((z) => `<option${z === DEFAULT_TZ ? ' selected' : ''}>${z}</option>`).join('');
fillSelect($('site'), ['Aucun (géocentrique)', ...SITES.map((s) => s.name), 'Coordonnées libres'], 1);
fillSelect($('orient'), ORIENTATIONS.map((o) => o.label));
fillSelect($('inst'), ['Aucun', ...INSTRUMENTS.map((i) => i.name), 'Personnalisé'], 1);

const prefs = loadPrefs();
for (const id of ['tz', 'site', 'orient', 'inst', 'lat', 'lon', 'elev', 'focal', 'pix', 'sw', 'sh', 'mag', 'zoom']) {
  if (prefs[id] !== undefined && $(id)) $(id).value = prefs[id];
}
if (prefs.night) $('night').checked = true;
if (prefs.planet) document.querySelector(`input[name=planet][value=${prefs.planet}]`)?.click();

// ─── Instant courant ────────────────────────────────────────────────────────

const tz = () => $('tz').value;

function setInputs(date) {
  const ut = $('input-ut').checked;
  const p = ut ? {
    y: date.getUTCFullYear(), m: date.getUTCMonth() + 1, d: date.getUTCDate(),
    h: date.getUTCHours(), mi: date.getUTCMinutes(), s: date.getUTCSeconds(),
  } : zonedParts(date, tz());
  $('date').value = `${p.y}-${pad(p.m)}-${pad(p.d)}`;
  $('time').value = `${pad(p.h)}:${pad(p.mi)}:${pad(p.s)}`;
  $('offset').value = 0;
}

function readInputs() {
  const [y, m, d] = $('date').value.split('-').map(Number);
  const [h, mi, s = 0] = $('time').value.split(':').map(Number);
  if (!y || Number.isNaN(h)) return null;
  return $('input-ut').checked
    ? new Date(Date.UTC(y, m - 1, d, h, mi, s))
    : zonedToDate(y, m, d, h, mi, s, tz());
}

const currentTime = () => {
  const base = readInputs();
  return base ? new Date(base.getTime() + Number($('offset').value) * 60000) : null;
};

const nowUtc = () => new Date(Math.floor(Date.now() / 1000) * 1000);
setInputs(nowUtc());

// ─── Lecture du formulaire ──────────────────────────────────────────────────

function readForm() {
  const planetKey = document.querySelector('input[name=planet]:checked').value;
  const siteIdx = Number($('site').value);
  let site = null;
  if (siteIdx > 0 && siteIdx <= SITES.length) site = SITES[siteIdx - 1];
  else if (siteIdx === SITES.length + 1) {
    site = { name: 'Coordonnées libres', lat: Number($('lat').value), lon: Number($('lon').value), elevation: Number($('elev').value) };
  }
  const o = ORIENTATIONS[Number($('orient').value)];
  const free = o.rot === null;
  const instIdx = Number($('inst').value);
  let instrument = null;
  if (instIdx > 0 && instIdx <= INSTRUMENTS.length) instrument = INSTRUMENTS[instIdx - 1];
  else if (instIdx === INSTRUMENTS.length + 1) {
    instrument = new Instrument('Instrument personnalisé', Number($('focal').value), Number($('sw').value),
      Number($('sh').value), Number($('pix').value));
  }
  $('custom-site').hidden = siteIdx !== SITES.length + 1;
  $('free-rot').hidden = !free;
  $('custom-inst').hidden = instIdx !== INSTRUMENTS.length + 1;
  $('frame-angle-wrap').hidden = !instrument;
  $('radii-wrap').hidden = $('zoom').value !== 'radii';
  return {
    planetKey, site, instrument,
    rotation: free ? Number($('rot').value) : o.rot,
    mirror: free ? $('free-mirror').checked : o.mirror,
    orientationLabel: o.label.split(' :')[0],
    frameAngle: Number($('frame-angle').value),
    zoom: $('zoom').value, radii: Number($('radii').value),
    magLimit: Number($('mag').value),
    night: $('night').checked,
    minAlt: Number($('minalt').value), maxSun: Number($('maxsun').value),
    detail: $('detail').checked,
  };
}

function updateOutputs() {
  $('rot-out').textContent = `${$('rot').value}°`;
  $('fa-out').textContent = `${$('frame-angle').value}°`;
  $('radii-out').textContent = $('radii').value;
  $('mag-out').textContent = Number($('mag').value).toLocaleString('fr-FR');
  $('minalt-out').textContent = `${$('minalt').value}°`;
  $('maxsun-out').textContent = `${Number($('maxsun').value).toLocaleString('fr-FR')}°`;
  const off = Number($('offset').value);
  const h = Math.trunc(Math.abs(off) / 60), m = Math.abs(off) % 60;
  $('offset-out').textContent = `${off < 0 ? '−' : '+'}${h ? `${h} h ${pad(m)}` : `${m} min`}`;
}

// ─── Affichage ──────────────────────────────────────────────────────────────

function table(el, rows, cols, rowClass = () => '') {
  if (!rows.length) { el.innerHTML = ''; return; }
  el.innerHTML = `<thead><tr>${cols.map((c) => `<th>${esc(c)}</th>`).join('')}</tr></thead><tbody>${
    rows.map((r) => `<tr class="${rowClass(r)}">${cols.map((c) => `<td>${esc(r[c] ?? '—')}</td>`).join('')}</tr>`).join('')
  }</tbody>`;
}

const fmtEvent = (t) => (t ? `${fmtDateTime(t, tz(), { year: false, seconds: false })} (${fmtDateTime(t, null, { year: false, seconds: false }).split(' ')[1]} UT)` : '—');

let state = { snap: null, svg: '', phen: [], phKey: '', form: null };

function alerts(snap, when) {
  const a = [];
  if (when.getUTCFullYear() < YEAR_MIN || when.getUTCFullYear() > YEAR_MAX) {
    a.push(`Date hors de la période couverte (${YEAR_MIN}–${YEAR_MAX}).`);
  }
  if (snap) {
    const missing = snap.moons.filter((m) => !m.available).map((m) => m.name);
    if (missing.length) a.push(`Position non disponible pour : ${missing.join(', ')} (fin des tables de satellites de la source actuelle).`);
    if (snap.info.reducedPrecision) a.push('Après 2040, positions issues d\'une théorie de secours : précision réduite (quelques secondes d\'arc).');
    if (snap.site && snap.info.alt < 0) a.push(`${snap.info.name} est sous l'horizon (${snap.info.alt.toFixed(1)}°) à cet instant.`);
  }
  $('alerts').innerHTML = a.map((t) => `<div class="alert">${esc(t)}</div>`).join('');
}

function renderTable(snap, f) {
  const rows = moonRows(snap, f.magLimit);
  let cols = ['Satellite', 'État', 'Mag ≈', 'AD J2000', 'Déc J2000', 'Hauteur', 'Azimut', 'Écart E-O', 'Écart N-S', 'Distance', 'Angle de position'];
  if (f.detail) {
    cols = ['Satellite', 'État', 'Mag ≈', 'AD J2000', 'Déc J2000', 'AD du jour', 'Déc du jour', 'Hauteur', 'Azimut',
      'Écart E-O', 'Écart N-S', 'E-O (rayons)', 'N-S (rayons)', 'Distance', 'Distance (rayons)', 'Angle de position'];
  }
  if (!snap.site) cols = cols.filter((c) => c !== 'Hauteur' && c !== 'Azimut');
  table($('moons'), rows, cols, (r) => (r['État'].startsWith(VISIBLE) ? '' : 'dim'));
  const hidden = snap.moons.filter((m) => m.mag > f.magLimit).length;
  $('hidden-note').textContent = hidden ? `${hidden} satellite(s) plus faible(s) que la magnitude ${f.magLimit} masqué(s).` : '';
}

function schemaExtent(snap, f) {
  if (f.zoom === 'field' && f.instrument) return Math.max(...f.instrument.field) / 2 * 1.1;
  if (f.zoom === 'radii') return f.radii * snap.info.radius;
  return null;
}

function schemaSize() {
  // écran étroit : format plus haut et moins large, pour garder un texte lisible
  const w = $('schema').clientWidth || 1000;
  if (w >= 760) return { width: 1000, height: 680 };
  const width = Math.max(520, Math.round(w * 1.5));
  return { width, height: Math.round(width * 1.15) };
}

function renderSchema(snap, f) {
  state.svg = drawSvg(snap, {
    ...schemaSize(),
    rotation: f.rotation, mirror: f.mirror, instrument: f.instrument, frameAngle: f.frameAngle,
    extent: schemaExtent(snap, f), magLimit: f.magLimit, night: f.night, tz: tz(),
    orientationLabel: f.orientationLabel,
  });
  $('schema').innerHTML = state.svg;
  if (f.instrument) {
    const [fw, fh] = f.instrument.field;
    const out = snap.moons.filter((m) => m.available && m.mag <= f.magLimit
      && (Math.abs(m.eastArcsec) > fw / 2 || Math.abs(m.northArcsec) > fh / 2)).map((m) => m.name);
    $('frame-note').textContent = `${f.instrument.name} : champ ${(fw / 60).toFixed(1)}′ × ${(fh / 60).toFixed(1)}′ `
      + `(demi-largeur ${Math.round(fw / 2)}″), ${f.instrument.scale.toFixed(3)}″/pixel. `
      + (out.length ? `Hors du cadre centré (non tourné) : ${out.join(', ')}.` : 'Tous les satellites affichés sont dans le cadre centré.');
  } else $('frame-note').textContent = '';
}

function renderInfo(snap) {
  const i = snap.info;
  const metrics = [];
  if (snap.site) metrics.push(['Hauteur', `${i.alt.toFixed(1)}°`, `azimut ${i.az.toFixed(1)}°`]);
  metrics.push(['Diamètre apparent', `${i.diameter.toFixed(2)}″`], ['Magnitude', i.mag.toFixed(2)],
    ['Distance', `${i.earthDistance.toFixed(3)} ua`]);
  if (i.ringTilt !== null) metrics.push(['Anneaux', `${i.ringTilt >= 0 ? '+' : ''}${i.ringTilt.toFixed(2)}°`, 'inclinaison vue de la Terre']);
  if (i.cml2 !== null) metrics.push(['Méridien central', `II ${i.cml2.toFixed(1)}°`, `I ${i.cml1.toFixed(1)}°`]);
  const lines = [
    `AD J2000 ${formatRa(i.raJ2000)}, Déc ${formatDec(i.decJ2000)}`,
    `angle de position du pôle nord ${i.polePa.toFixed(1)}°`,
    `latitude de la Terre vue de ${i.name} ${i.earthLat >= 0 ? '+' : ''}${i.earthLat.toFixed(2)}°`,
  ];
  if (snap.site) {
    if (i.circumpolar) lines.push('planète circumpolaire');
    else if (i.neverUp) lines.push('planète jamais levée');
    else {
      lines.push(`lever ${fmtEvent(i.rise)}, passage au méridien ${fmtEvent(i.transit)}`
        + `${i.transitAlt !== null ? ` (hauteur ${i.transitAlt.toFixed(1)}°)` : ''}, coucher ${fmtEvent(i.set)}`);
    }
    lines.push(`Soleil à ${i.sunAlt >= 0 ? '+' : ''}${i.sunAlt.toFixed(1)}°`);
  }
  state.planetLines = lines;
  $('planet-info').innerHTML = `<div class="info-grid">${metrics.map(([k, v, s]) => `<div class="metric"><div class="k">${esc(k)}</div><div class="v">${esc(v)}</div>${s ? `<div class="k">${esc(s)}</div>` : ''}</div>`).join('')}</div><p>${esc(lines.join(' · '))}</p>`;
  $('source-note').textContent = `Source des tables : ${i.source}.`;
}

async function renderPhenomena(snap, f) {
  const [start, end] = nightWindow(snap.when, tz());
  const key = [f.planetKey, start.toISOString(), snap.site?.name, snap.site?.lat, snap.site?.lon, f.minAlt, f.maxSun, tz()].join('|');
  $('ph-note').textContent = `Du ${fmtDateTime(start, tz(), { seconds: false })} au ${fmtDateTime(end, tz(), { seconds: false })} (${tz()}). `
    + 'Observable : planète levée et Soleil couché (critères réglables).';
  if (key !== state.phKey) {
    state.phKey = key;
    $('phenomena').innerHTML = '<tbody><tr><td>Calcul…</td></tr></tbody>';
    try {
      state.phen = await findPhenomena(f.planetKey, start, end, snap.site, { minAlt: f.minAlt, maxSun: f.maxSun });
    } catch (e) {
      state.phen = [];
    }
    if (key !== state.phKey) return;   // un calcul plus récent a pris le relais
  }
  const labels = Object.fromEntries(snap.moons.map((m) => [m.key, m.name]));
  state.phRows = phenomenaRows(state.phen, labels, tz());
  if (!state.phRows.length) {
    $('phenomena').innerHTML = '<tbody><tr><td>Aucun passage, occultation, éclipse ni ombre sur la période.</td></tr></tbody>';
    return;
  }
  table($('phenomena'), state.phRows, Object.keys(state.phRows[0]).filter((k) => !k.startsWith('_')),
    (r) => (r._observable ? 'obs' : 'dim'));
}

let pending = null;
async function update() {
  updateOutputs();
  const f = readForm();
  state.form = f;
  document.body.classList.toggle('night', f.night);
  savePrefs({
    planet: f.planetKey, night: f.night,
    ...Object.fromEntries(['tz', 'site', 'orient', 'inst', 'lat', 'lon', 'elev', 'focal', 'pix', 'sw', 'sh', 'mag', 'zoom'].map((id) => [id, $(id).value])),
  });
  const when = currentTime();
  if (!when) return;
  const token = Symbol('update');
  pending = token;
  const p = PLANETS[f.planetKey];
  const local = fmtDateTime(when, tz());
  $('title').textContent = `${p.name} et ses satellites`;
  $('subtitle').innerHTML = `<strong>${fmtDateTime(when)} UT</strong> · ${esc(local)} ${esc(tzAbbrev(when, tz()))} (${esc(tz())}) · ${esc(f.site ? f.site.name : 'géocentrique')} · ${esc(SIGNATURE)}`;
  let snap = null;
  try {
    snap = await compute(f.planetKey, when, f.site);
  } catch (e) {
    alerts(null, when);
    $('moons').innerHTML = `<tbody><tr><td>${esc(e.message)}</td></tr></tbody>`;
    $('schema').innerHTML = '';
    return;
  }
  if (pending !== token) return;
  state.snap = snap;
  alerts(snap, when);
  renderTable(snap, f);
  renderSchema(snap, f);
  renderInfo(snap);
  renderPhoto();
  renderPhenomena(snap, f);
}

// Regroupe les mises à jour rapides (curseurs) sur une image
let raf = 0;
const schedule = () => { cancelAnimationFrame(raf); raf = requestAnimationFrame(() => update()); };

// ─── Événements ─────────────────────────────────────────────────────────────

$('form').addEventListener('input', schedule);
$('form').addEventListener('change', schedule);
$('night').addEventListener('change', schedule);
window.addEventListener('resize', () => state.snap && renderSchema(state.snap, state.form));
$('detail').addEventListener('change', schedule);
$('offset').addEventListener('input', schedule);
$('input-ut').addEventListener('change', (e) => {
  // réécrit les champs dans le nouveau mode sans changer l'instant
  e.target.checked = !e.target.checked;
  const t = currentTime();
  e.target.checked = !e.target.checked;
  if (t) setInputs(t);
  schedule();
});
$('tz').addEventListener('change', () => {
  if (!$('input-ut').checked) { /* l'heure saisie reste la même, l'instant change */ }
  schedule();
});
$('now').addEventListener('click', () => { setInputs(nowUtc()); schedule(); });
document.querySelectorAll('[data-step]').forEach((b) => b.addEventListener('click', () => {
  const t = currentTime();
  if (!t) return;
  setInputs(new Date(t.getTime() + Number(b.dataset.step) * 60000));
  schedule();
}));

const stamp = () => {
  const s = state.snap;
  return `${s.planet.key.toLowerCase()}_${s.when.toISOString().slice(0, 19).replace(/[-:]/g, '').replace('T', '_')}UT`;
};
$('dl-csv').addEventListener('click', () => state.snap && download(`cielscore_${stamp()}.csv`,
  `﻿${moonsCsv(state.snap, state.form.magLimit, tz())}`, 'text/csv'));
$('dl-ph').addEventListener('click', () => state.snap && download(`cielscore_phenomenes_${stamp()}.csv`,
  `﻿${rowsCsv(state.phRows || [])}`, 'text/csv'));
$('dl-png').addEventListener('click', async () => {
  if (!state.snap) return;
  const svg = drawSvg(state.snap, {
    ...state.form, extent: schemaExtent(state.snap, state.form), tz: tz(), width: 1400, height: 950,
  });
  download(`cielscore_${stamp()}.png`, await svgToPng(svg, 1.5));
});
$('dl-pdf').addEventListener('click', () => {
  const night = document.body.classList.contains('night');
  if (night) { document.body.classList.remove('night'); renderSchema(state.snap, { ...state.form, night: false }); }
  window.print();
  if (night) { document.body.classList.add('night'); renderSchema(state.snap, state.form); }
});

// ─── Option photo ───────────────────────────────────────────────────────────

const photo = { data: null, center: null, ref: null, marks: [], rows: [], image: null };
const canvas = $('photo-canvas');

$('photo-file').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  $('photo-ui').hidden = false;
  $('photo-date').textContent = 'Lecture…';
  try {
    photo.data = await photoLib.loadPhoto(file);
  } catch (err) {
    $('photo-date').textContent = `Lecture impossible : ${err.message}`;
    return;
  }
  const d = photo.data;
  const rgba = photoLib.displayRgba(d);
  photo.image = new ImageData(rgba, d.width, d.height);
  canvas.width = d.width;
  canvas.height = d.height;
  photo.center = null;
  photo.ref = null;
  $('wcs-wrap').hidden = !d.wcs;
  if (d.date) {
    const exif = d.dateSource === 'EXIF';
    $('photo-date').innerHTML = `Heure lue dans le fichier (${esc(d.dateSource)}) : <strong>${fmtDateTime(d.date)}${exif ? ' (heure de l\'appareil)' : ' UT'}</strong> `
      + `<button type="button" id="use-photo-time">Calculer pour cette heure${exif ? ' (heure locale)' : ''}</button>`;
    $('use-photo-time').addEventListener('click', () => {
      let t = d.date;
      if (exif) {
        t = zonedToDate(t.getUTCFullYear(), t.getUTCMonth() + 1, t.getUTCDate(), t.getUTCHours(), t.getUTCMinutes(), t.getUTCSeconds(), tz());
      }
      setInputs(t);
      schedule();
    });
  } else {
    $('photo-date').textContent = 'Aucune heure trouvée dans le fichier : saisissez-la dans le formulaire.';
  }
  renderPhoto();
});

canvas.addEventListener('click', (e) => {
  if (!photo.data) return;
  const rect = canvas.getBoundingClientRect();
  const x = (e.clientX - rect.left) * (canvas.width / rect.width);
  const y = (e.clientY - rect.top) * (canvas.height / rect.height);
  const mode = document.querySelector('input[name=click]:checked').value;
  if (mode === 'center') {
    photo.center = [x, y];
    document.querySelector('input[name=click][value=ref]').checked = true;
  } else photo.ref = [x, y];
  renderPhoto();
});
$('detect-center').addEventListener('click', () => {
  if (!photo.data) return;
  photo.center = photoLib.detectPlanetCenter(photo.data);
  document.querySelector('input[name=click][value=ref]').checked = true;
  renderPhoto();
});
for (const id of ['cx', 'cy', 'rx', 'ry']) {
  $(id).addEventListener('change', () => {
    photo.center = [Number($('cx').value), Number($('cy').value)];
    if ($('rx').value !== '') photo.ref = [Number($('rx').value), Number($('ry').value)];
    renderPhoto();
  });
}
$('ref-moon').addEventListener('change', renderPhoto);
$('photo-mirror').addEventListener('change', renderPhoto);
$('use-wcs').addEventListener('change', renderPhoto);

function renderPhoto() {
  const d = photo.data, snap = state.snap;
  if (!d || !snap) return;
  const f = state.form;
  const moons = snap.moons.filter((m) => m.available && m.mag <= f.magLimit);
  const visible = moons.filter((m) => m.state === VISIBLE);
  const names = (visible.length ? visible : moons).map((m) => m.name);
  const prev = $('ref-moon').value;
  $('ref-moon').innerHTML = names.map((n) => `<option>${esc(n)}</option>`).join('');
  $('ref-moon').value = names.includes(prev) ? prev : (names.includes('Titan') ? 'Titan' : names[names.length - 1]);
  if (!$('photo-mirror').dataset.touched) $('photo-mirror').checked = f.mirror;

  const useWcs = Boolean(d.wcs) && $('use-wcs').checked;
  $('photo-cal').hidden = useWcs;
  if (photo.center) { $('cx').value = photo.center[0].toFixed(1); $('cy').value = photo.center[1].toFixed(1); }
  if (photo.ref) { $('rx').value = photo.ref[0].toFixed(1); $('ry').value = photo.ref[1].toFixed(1); }

  let toPixel = null, scale = null;
  $('cal-result').textContent = '';
  if (useWcs) {
    toPixel = (m) => photoLib.wcsPixel(d.wcs, m.raJ2000, m.decJ2000);
    scale = d.wcs.scale;
    $('cal-result').textContent = `Calage par la solution WCS du FITS : ${scale.toFixed(3)}″/pixel.`;
  } else if (photo.center && photo.ref) {
    const ref = moons.find((m) => m.name === $('ref-moon').value);
    try {
      const cal = photoLib.calibrate(photo.center, photo.ref, ref.eastArcsec, ref.northArcsec, $('photo-mirror').checked);
      toPixel = (m) => cal.skyToPixel(m.eastArcsec, m.northArcsec);
      scale = cal.scale;
      $('cal-result').innerHTML = `Échelle déduite : <strong>${cal.scale.toFixed(3)}″/pixel</strong>, rotation <strong>${cal.rotationDeg.toFixed(1)}°</strong> `
        + '(par rapport à « nord en haut, est à gauche »)'
        + (f.instrument ? ` — instrument choisi : ${f.instrument.scale.toFixed(3)}″/pixel` : '');
    } catch (err) {
      $('cal-result').textContent = err.message;
    }
  }

  const ctx = canvas.getContext('2d');
  ctx.putImageData(photo.image, 0, 0);
  const u = Math.max(1, d.width / 900);
  const color = f.night ? '#ff3b30' : '#ffd60a';
  const ring = (x, y, rad, c, w = 1.5) => { ctx.beginPath(); ctx.arc(x, y, rad * u, 0, 2 * Math.PI); ctx.strokeStyle = c; ctx.lineWidth = w * u; ctx.stroke(); };
  if (!toPixel) {
    if (photo.center) ring(...photo.center, 8, '#5aa0ff', 2);
    if (photo.ref) ring(...photo.ref, 8, color, 2);
  }
  photo.rows = [];
  if (toPixel) {
    ctx.font = `${12 * u}px Inter, system-ui, sans-serif`;
    for (const m of moons) {
      const [x, y] = toPixel(m);
      const inside = x >= 0 && x < d.width && y >= 0 && y < d.height;
      const found = inside ? photoLib.nearestSource(d, x, y) : null;
      if (inside) {
        ring(x, y, 9, color);
        if (found) {
          ctx.strokeStyle = f.night ? '#ff8a80' : '#30d158';
          ctx.lineWidth = 1.2 * u;
          ctx.beginPath();
          ctx.moveTo(found[0] - 5 * u, found[1]); ctx.lineTo(found[0] + 5 * u, found[1]);
          ctx.moveTo(found[0], found[1] - 5 * u); ctx.lineTo(found[0], found[1] + 5 * u);
          ctx.stroke();
        }
        ctx.fillStyle = color;
        ctx.fillText(m.name, x + 12 * u, y - 10 * u);
        ctx.fillText(`${formatRaShort(m.raJ2000)} ${formatDecShort(m.decJ2000)}`, x + 12 * u, y + 4 * u);
      }
      const dpx = found ? Math.hypot(found[0] - x, found[1] - y) : null;
      photo.rows.push({
        Satellite: m.name, 'État': m.state, 'x calculé': x.toFixed(1), 'y calculé': y.toFixed(1),
        'Point le plus proche': found ? `(${found[0].toFixed(1)}, ${found[1].toFixed(1)})` : '—',
        'Écart (px)': dpx === null ? '—' : dpx.toFixed(2),
        'Écart (″)': dpx === null || !scale ? '—' : (dpx * scale).toFixed(2),
        'AD J2000': formatRa(m.raJ2000), 'Déc J2000': formatDec(m.decJ2000),
      });
    }
    ctx.fillStyle = '#999';
    ctx.font = `${11 * u}px Inter, system-ui, sans-serif`;
    ctx.textAlign = 'right';
    ctx.fillText(`${SIGNATURE} · ${snap.info.name} · ${fmtDateTime(snap.when)} UT`, d.width - 8 * u, d.height - 8 * u);
    ctx.textAlign = 'left';
  }
  table($('photo-table'), photo.rows, photo.rows.length ? Object.keys(photo.rows[0]) : []);
}
$('photo-mirror').addEventListener('change', (e) => { e.target.dataset.touched = '1'; });
$('dl-photo').addEventListener('click', () => canvas.toBlob((b) => download(`cielscore_photo_${stamp()}.png`, b), 'image/png'));
$('dl-photo-csv').addEventListener('click', () => download(`cielscore_photo_${stamp()}.csv`, `﻿${rowsCsv(photo.rows)}`, 'text/csv'));

// ─── Démarrage ──────────────────────────────────────────────────────────────

// Page intégrée dans un cadre (aperçu) : téléchargements et impression y sont bloqués.
let embedded = false;
try { embedded = window.self !== window.top; } catch (e) { embedded = true; }
if (embedded) {
  for (const id of ['dl-csv', 'dl-ph', 'dl-png', 'dl-pdf', 'dl-photo', 'dl-photo-csv']) $(id).hidden = true;
  const note = document.createElement('p');
  note.className = 'muted small';
  note.textContent = 'Version de test : les exports (CSV, PNG, PDF) sont désactivés dans cet aperçu. '
    + 'Ils fonctionnent sur le site publié ou en local.';
  $('dl-csv').parentElement.after(note);
}

update();
if ('serviceWorker' in navigator && location.protocol !== 'file:') {
  navigator.serviceWorker.register('sw.js').catch(() => { /* hors ligne indisponible */ });
}
