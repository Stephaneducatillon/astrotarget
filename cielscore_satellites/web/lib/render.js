// Schéma SVG : planète au centre, satellites à l'échelle, nom et coordonnées.

import { ECLIPSED, OCCULTED, TRANSIT, formatDecShort, formatRaShort } from './engine.js';
import { SIGNATURE } from './data.js';
import { fmtDateTime, tzAbbrev, zonedParts } from './time.js';

export const DAY = {
  bg: '#ffffff', fg: '#1a1a1a', muted: '#666666', planet: '#dbe8f7', planetEdge: '#2f6fd0',
  ring: '#c9b98f', moon: '#111111', hidden: '#999999', shadow: '#000000', frame: '#d04a2f',
};
export const NIGHT = {
  bg: '#000000', fg: '#ff3b30', muted: '#b02a20', planet: '#3a0000', planetEdge: '#ff3b30',
  ring: '#7a1a14', moon: '#ff3b30', hidden: '#6b1510', shadow: '#000000', frame: '#ff6a55',
};

const DEG = Math.PI / 180;
const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const n2 = (v) => (Math.round(v * 100) / 100).toString();

/** Passage des coordonnées du ciel (est, nord) aux coordonnées du schéma (x droite, y haut). */
export class View {
  constructor(rotationDeg = 0, mirror = false) {
    this.rot = rotationDeg * DEG;
    this.mirror = mirror;
  }

  apply(east, north) {
    const x = this.mirror ? east : -east;
    const c = Math.cos(this.rot), s = Math.sin(this.rot);
    return [x * c - north * s, x * s + north * c];
  }

  angle(paDeg) {
    const [x, y] = this.apply(Math.sin(paDeg * DEG), Math.cos(paDeg * DEG));
    return Math.atan2(y, x) / DEG;
  }
}

function niceScale(span) {
  for (const v of [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]) if (v >= span) return v;
  return 1000;
}

/**
 * Construit le SVG du schéma.
 * options : rotation, mirror, instrument, frameAngle, extent (″), magLimit, night, tz,
 * orientationLabel, showCoords, width, height.
 */
export function drawSvg(snap, opts = {}) {
  const {
    rotation = 0, mirror = false, instrument = null, frameAngle = 0, extent = null,
    magLimit = 99, night = false, tz = 'Europe/Paris', orientationLabel = 'Vue du ciel',
    showCoords = true, width = 1000, height = 680,
  } = opts;
  const pal = night ? NIGHT : DAY;
  const view = new View(rotation, mirror);
  const info = snap.info;
  const r = info.radius;
  const moons = snap.moons.filter((m) => m.available && m.mag <= magLimit);

  let ext = extent;
  if (!ext) {
    let far = Math.max(r * (info.ringOuter || 1), ...moons.map((m) => m.distArcsec));
    if (instrument) {
      const [fw, fh] = instrument.field;
      far = Math.max(far, Math.hypot(fw, fh) / 2);
    }
    ext = far * 1.18 + r;
  }
  const top = 62, bottom = 26, side = 10;
  const plotW = width - 2 * side, plotH = height - top - bottom;
  const scale = Math.min(plotW / (2 * ext * Math.max(1, plotW / plotH * 0.7)), plotH / (2 * ext * Math.max(0.8, plotH / plotW * 0.7)));
  const cx = width / 2, cy = top + plotH / 2;
  const P = (x, y) => [cx + x * scale, cy - y * scale];
  const out = [];
  const add = (s) => out.push(s);

  add(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" font-family="Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif">`);
  add(`<rect width="100%" height="100%" fill="${pal.bg}"/>`);
  add(`<defs><clipPath id="plot"><rect x="${side}" y="${top}" width="${plotW}" height="${plotH}"/></clipPath></defs>`);
  add('<g clip-path="url(#plot)">');

  // Pôle dans le repère (est, nord, vers la Terre) et anneaux
  const pa = info.polePa * DEG, de = info.earthLat * DEG;
  const pole = [Math.sin(pa) * Math.cos(de), Math.cos(pa) * Math.cos(de), Math.sin(de)];
  let e1 = [pole[1], -pole[0], 0];   // pole × (0, 0, 1)
  const ne1 = Math.hypot(...e1) || 1;
  e1 = e1.map((v) => v / ne1);
  const e2 = [pole[1] * e1[2] - pole[2] * e1[1], pole[2] * e1[0] - pole[0] * e1[2], pole[0] * e1[1] - pole[1] * e1[0]];
  const ring = (rad) => {
    const pts = [];
    for (let i = 0; i <= 360; i++) {
      const ph = i * DEG;
      const v = [0, 1, 2].map((k) => rad * r * (Math.cos(ph) * e1[k] + Math.sin(ph) * e2[k]));
      pts.push({ xy: P(...view.apply(v[0], v[1])), near: v[2] > 0 });
    }
    return pts;
  };
  const path = (pts) => pts.map((p, i) => `${i ? 'L' : 'M'}${n2(p.xy[0])},${n2(p.xy[1])}`).join('');
  let rings = null;
  if (info.ringOuter) {
    const outer = ring(info.ringOuter), inner = ring(1.53);
    rings = { outer, inner };
    add(`<path d="${path(outer)}Z${path([...inner].reverse())}Z" fill="${pal.ring}" fill-opacity="0.35" fill-rule="evenodd" stroke="none"/>`);
    add(`<path d="${path(outer)}" fill="none" stroke="${pal.ring}" stroke-width="1.2"/>`);
    add(`<path d="${path(inner)}" fill="none" stroke="${pal.ring}" stroke-width="0.7"/>`);
  }

  // Disque aplati, axe polaire orienté
  const eqAngle = view.angle(info.polePa) - 90;
  add(`<ellipse cx="${n2(cx)}" cy="${n2(cy)}" rx="${n2(r * scale)}" ry="${n2(r * info.axisRatio * scale)}" transform="rotate(${n2(-eqAngle)} ${n2(cx)} ${n2(cy)})" fill="${pal.planet}" stroke="${pal.planetEdge}" stroke-width="1.5"/>`);

  // Moitié avant des anneaux, par-dessus le disque
  if (rings) {
    const start = rings.outer.findIndex((p, i) => !p.near && rings.outer[(i + 1) % rings.outer.length].near);
    const idx = [];
    for (let k = 1; k <= 361; k++) {
      const i = (start + k) % 361;
      if (!rings.outer[i].near) break;
      idx.push(i);
    }
    if (idx.length) {
      const o = idx.map((i) => rings.outer[i]), inn = idx.map((i) => rings.inner[i]).reverse();
      add(`<path d="${path(o)}${path(inn).replace('M', 'L')}Z" fill="${pal.ring}" fill-opacity="0.55" stroke="none"/>`);
      add(`<path d="${path(o)}" fill="none" stroke="${pal.ring}" stroke-width="1.2"/>`);
      add(`<path d="${path(idx.map((i) => rings.inner[i]))}" fill="none" stroke="${pal.ring}" stroke-width="0.7"/>`);
    }
  }

  // Ombres sur le disque
  for (const m of moons) {
    if (!m.shadow) continue;
    const [x, y] = P(...view.apply(m.shadow[0] * r, m.shadow[1] * r));
    add(`<circle cx="${n2(x)}" cy="${n2(y)}" r="3.2" fill="${pal.shadow}" stroke="${pal.planetEdge}" stroke-width="0.5"/>`);
  }

  // Cadre du champ de l'instrument
  if (instrument) {
    const [fw, fh] = instrument.field;
    const w = fw * scale, h = fh * scale;
    add(`<g transform="rotate(${n2(-frameAngle)} ${n2(cx)} ${n2(cy)})">`);
    add(`<rect x="${n2(cx - w / 2)}" y="${n2(cy - h / 2)}" width="${n2(w)}" height="${n2(h)}" fill="none" stroke="${pal.frame}" stroke-width="1.2" stroke-dasharray="6 4"/>`);
    add(`<text x="${n2(cx + w / 2 + 4)}" y="${n2(cy - h / 2 - 16)}" font-size="10" fill="${pal.frame}">${esc(instrument.name.split(' — ')[0])}</text>`);
    add(`<text x="${n2(cx + w / 2 + 4)}" y="${n2(cy - h / 2 - 4)}" font-size="10" fill="${pal.frame}">${(fw / 60).toFixed(1)}′ × ${(fh / 60).toFixed(1)}′</text>`);
    add('</g>');
  }

  // Étiquettes sans chevauchement (en pixels)
  const ro = r * Math.max(1, info.ringOuter * 0.95) * scale;
  const obstacles = [[cx - ro, cy - r * 1.05 * scale, cx + ro, cy + r * 1.05 * scale]];
  const pts = moons.map((m) => {
    const [x, y] = P(...view.apply(m.eastArcsec, m.northArcsec));
    obstacles.push([x - 5, y - 5, x + 5, y + 5]);
    return { m, x, y };
  });
  const overlaps = (b) => obstacles.some((o) => !(b[2] < o[0] || b[0] > o[2] || b[3] < o[1] || b[1] > o[3]));
  const place = (x, y, lines, fs, gap) => {
    const w = Math.max(...lines.map((t) => t.length)) * fs * 0.56;
    const h = lines.length * fs * 1.25;
    for (let k = 0; k < 10; k++) {
      const far = gap + k * h * 0.9;
      const cands = [[x, y + far + h / 2], [x, y - far - h / 2], [x + far + w / 2, y], [x - far - w / 2, y],
        [x + far * 0.7 + w / 2, y + far * 0.7 + h / 2], [x - far * 0.7 - w / 2, y - far * 0.7 - h / 2],
        [x + far * 0.7 + w / 2, y - far * 0.7 - h / 2], [x - far * 0.7 - w / 2, y + far * 0.7 + h / 2]];
      for (const [qx, qy] of cands) {
        const b = [qx - w / 2, qy - h / 2, qx + w / 2, qy + h / 2];
        if (!overlaps(b)) { obstacles.push(b); return { x: qx, y: qy, h, leader: k > 0 }; }
      }
    }
    const qy = y + gap + h / 2;
    obstacles.push([x - w / 2, qy - h / 2, x + w / 2, qy + h / 2]);
    return { x, y: qy, h, leader: false };
  };
  const textBlock = (pos, lines, fs, color, weight = 'normal') => {
    const y0 = pos.y - pos.h / 2 + fs * 0.95;
    const spans = lines.map((t, i) => `<tspan x="${n2(pos.x)}" y="${n2(y0 + i * fs * 1.25)}">${esc(t)}</tspan>`).join('');
    return `<text text-anchor="middle" font-size="${fs}" font-weight="${weight}" fill="${color}" paint-order="stroke" stroke="${pal.bg}" stroke-width="3" stroke-linejoin="round">${spans}</text>`;
  };

  const labels = [];
  labels.push(textBlock(place(cx, cy, [info.name], 13, r * 1.05 * scale + 4), [info.name], 13, pal.fg, '600'));
  for (const { m, x, y } of [...pts].sort((a, b) => a.m.distArcsec - b.m.distArcsec)) {
    const size = Math.max(2.2, 7 - 0.4 * m.mag);
    if (m.state === OCCULTED) add(`<circle cx="${n2(x)}" cy="${n2(y)}" r="${n2(size)}" fill="none" stroke="${pal.hidden}"/>`);
    else if (m.state === ECLIPSED) add(`<circle cx="${n2(x)}" cy="${n2(y)}" r="${n2(size)}" fill="${pal.bg}" stroke="${pal.hidden}"/>`);
    else if (m.state === TRANSIT) add(`<circle cx="${n2(x)}" cy="${n2(y)}" r="${n2(size)}" fill="${pal.planet}" stroke="${pal.moon}" stroke-width="1.2"/>`);
    else add(`<circle cx="${n2(x)}" cy="${n2(y)}" r="${n2(size)}" fill="${pal.moon}"/>`);
    const lines = [m.name + (m.state === 'visible' ? '' : ` (${m.state})`)];
    if (showCoords) lines.push(`${formatRaShort(m.raJ2000)} ${formatDecShort(m.decJ2000)}`);
    const pos = place(x, y, lines, 10.5, 7);
    if (pos.leader) add(`<line x1="${n2(x)}" y1="${n2(y)}" x2="${n2(pos.x)}" y2="${n2(pos.y)}" stroke="${pal.muted}" stroke-width="0.6"/>`);
    labels.push(textBlock(pos, lines, 10.5, m.state === 'visible' ? pal.fg : pal.muted));
  }
  out.push(...labels);
  add('</g>');

  // Boussole N / E
  const ox = side + 50, oy = top + 55, al = 34;
  for (const [lab, e, nn] of [['N', 0, 1], ['E', 1, 0]]) {
    const [dx, dy] = view.apply(e, nn);
    const x2 = ox + dx * al, y2 = oy - dy * al;
    const ang = Math.atan2(y2 - oy, x2 - ox);
    const hx = (s) => `${n2(x2 - 7 * Math.cos(ang + s))},${n2(y2 - 7 * Math.sin(ang + s))}`;
    add(`<line x1="${ox}" y1="${oy}" x2="${n2(x2)}" y2="${n2(y2)}" stroke="${pal.muted}" stroke-width="1.2"/>`);
    add(`<path d="M${hx(0.45)}L${n2(x2)},${n2(y2)}L${hx(-0.45)}" fill="none" stroke="${pal.muted}" stroke-width="1.2"/>`);
    add(`<text x="${n2(ox + dx * al * 1.35)}" y="${n2(oy - dy * al * 1.35 + 4)}" font-size="11" text-anchor="middle" fill="${pal.muted}">${lab}</text>`);
  }

  // Échelle
  let nr = niceScale((ext * 0.5) / r);
  while (nr * r > ext * 0.9 && nr > 1) nr = nr > 2 ? nr / 2 : 1;
  const bar = nr * r * scale;
  const bx = width - side - 20 - bar, by = height - bottom - 14;
  add(`<path d="M${n2(bx)},${by - 4}V${by + 4}M${n2(bx)},${by}H${n2(bx + bar)}M${n2(bx + bar)},${by - 4}V${by + 4}" stroke="${pal.muted}" fill="none"/>`);
  add(`<text x="${n2(bx + bar / 2)}" y="${by - 8}" font-size="10.5" text-anchor="middle" fill="${pal.muted}">${nr} rayon${nr > 1 ? 's' : ''} de ${esc(info.name)} ≈ ${Math.round(nr * r)}″</text>`);

  // Titres et signature
  const lp = zonedParts(snap.when, tz);
  const loc = `${String(lp.h).padStart(2, '0')}:${String(lp.mi).padStart(2, '0')} ${tzAbbrev(snap.when, tz)}`;
  add(`<text x="${side + 4}" y="24" font-size="16" font-weight="600" fill="${pal.fg}">${esc(info.name)} et ses satellites — ${fmtDateTime(snap.when)} UT (${esc(loc)})</text>`);
  add(`<text x="${side + 4}" y="44" font-size="11" fill="${pal.muted}">${esc(orientationLabel)} · ${esc(snap.site ? snap.site.name : 'géocentrique')} · AD et Déc J2000 sous chaque nom</text>`);
  add(`<text x="${width - side}" y="${height - 6}" font-size="10" text-anchor="end" fill="${pal.muted}">${esc(SIGNATURE)}</text>`);
  add('</svg>');
  return out.join('');
}
