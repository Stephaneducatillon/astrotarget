// Tableaux et exports : CSV des coordonnées et des phénomènes, PNG du schéma.

import { SIGNATURE } from './data.js';
import { formatDec, formatRa } from './engine.js';
import { fmtDateTime } from './time.js';

const R2D = 180 / Math.PI;
const ew = (v) => `${Math.abs(v).toFixed(1)}″ ${v >= 0 ? 'E' : 'O'}`;
const ns = (v) => `${Math.abs(v).toFixed(1)}″ ${v >= 0 ? 'N' : 'S'}`;
const r = (v, d) => (v === null || v === undefined ? '' : Number(v.toFixed(d)));

/** Lignes du tableau affiché (une par satellite, triées par distance). */
export function moonRows(snap, magLimit = 99) {
  return snap.moons.filter((m) => m.mag <= magLimit).map((m) => {
    if (!m.available) {
      return { Satellite: m.name, 'État': m.state, 'Mag ≈': m.mag.toFixed(1) };
    }
    return {
      Satellite: m.name,
      'État': m.state + (m.shadow ? ' · ombre sur le disque' : ''),
      'Mag ≈': m.mag.toFixed(1),
      'AD J2000': formatRa(m.raJ2000),
      'Déc J2000': formatDec(m.decJ2000),
      'AD du jour': formatRa(m.raApp),
      'Déc du jour': formatDec(m.decApp),
      Hauteur: m.alt === null ? '' : `${m.alt.toFixed(2)}°`,
      Azimut: m.az === null ? '' : `${m.az.toFixed(2)}°`,
      'Écart E-O': ew(m.eastArcsec),
      'Écart N-S': ns(m.northArcsec),
      'E-O (rayons)': m.eastRadii.toFixed(3),
      'N-S (rayons)': m.northRadii.toFixed(3),
      Distance: `${m.distArcsec.toFixed(1)}″`,
      'Distance (rayons)': m.distRadii.toFixed(2),
      'Angle de position': `${m.pa.toFixed(1)}°`,
    };
  });
}

function csv(rows, header) {
  const cols = header || Object.keys(rows[0] || {});
  const cell = (v) => {
    const s = v === null || v === undefined ? '' : String(v);
    return /[;"\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  return [`# ${SIGNATURE}`, cols.join(';'), ...rows.map((row) => cols.map((c) => cell(row[c])).join(';'))].join('\n');
}

/** CSV numérique (degrés décimaux, secondes d'arc), séparateur « ; ». */
export function moonsCsv(snap, magLimit = 99, tz = 'Europe/Paris') {
  const rows = snap.moons.filter((m) => m.mag <= magLimit).map((m) => ({
    planete: snap.info.name,
    satellite: m.name,
    date_ut: snap.when.toISOString().replace('.000', ''),
    date_locale: `${fmtDateTime(snap.when, tz)} (${tz})`,
    lieu: snap.site ? snap.site.name : 'géocentrique',
    etat: m.state,
    ombre_sur_disque: m.available ? Boolean(m.shadow) : '',
    mag_approx: r(m.mag, 2),
    ad_j2000_deg: m.available ? r(m.raJ2000 * R2D, 7) : '',
    dec_j2000_deg: m.available ? r(m.decJ2000 * R2D, 7) : '',
    ad_j2000: m.available ? formatRa(m.raJ2000, 2) : '',
    dec_j2000: m.available ? formatDec(m.decJ2000, 1) : '',
    ad_jour_deg: m.available ? r(m.raApp * R2D, 7) : '',
    dec_jour_deg: m.available ? r(m.decApp * R2D, 7) : '',
    hauteur_deg: r(m.alt, 3),
    azimut_deg: r(m.az, 3),
    ecart_est_arcsec: m.available ? r(m.eastArcsec, 2) : '',
    ecart_nord_arcsec: m.available ? r(m.northArcsec, 2) : '',
    ecart_est_rayons: m.available ? r(m.eastRadii, 4) : '',
    ecart_nord_rayons: m.available ? r(m.northRadii, 4) : '',
    profondeur_rayons: m.available ? r(m.depthRadii, 4) : '',
    distance_arcsec: m.available ? r(m.distArcsec, 2) : '',
    angle_position_deg: m.available ? r(m.pa, 2) : '',
  }));
  return csv(rows);
}

export function phenomenaRows(list, labels, tz) {
  const f = (t, local) => (t ? fmtDateTime(t, local ? tz : null, { year: false }) : 'en cours');
  const o = (v) => (v === true ? 'oui' : v === false ? 'non' : '—');
  return list.map((p) => ({
    Satellite: labels[p.moon] || p.moon,
    'Phénomène': p.label,
    'Début (UT)': f(p.start, false),
    'Début (local)': f(p.start, true),
    'Fin (UT)': f(p.end, false),
    'Fin (local)': f(p.end, true),
    'Observable début': o(p.startObservable),
    'Observable fin': o(p.endObservable),
    _observable: p.startObservable || p.endObservable,
  }));
}

export function rowsCsv(rows) {
  if (!rows.length) return `# ${SIGNATURE}\n`;
  return csv(rows, Object.keys(rows[0]).filter((k) => !k.startsWith('_')));
}

export function download(name, content, type) {
  const blob = content instanceof Blob ? content : new Blob([content], { type });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
}

/** SVG (texte) → Blob PNG, à l'échelle ``scale``. */
export async function svgToPng(svg, scale = 2) {
  const img = new Image();
  const url = URL.createObjectURL(new Blob([svg], { type: 'image/svg+xml' }));
  await new Promise((resolve, reject) => { img.onload = resolve; img.onerror = reject; img.src = url; });
  const c = document.createElement('canvas');
  c.width = img.width * scale;
  c.height = img.height * scale;
  const ctx = c.getContext('2d');
  ctx.scale(scale, scale);
  ctx.drawImage(img, 0, 0);
  URL.revokeObjectURL(url);
  return new Promise((resolve) => c.toBlob(resolve, 'image/png'));
}
