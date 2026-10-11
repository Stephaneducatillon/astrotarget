// Option photo : lecture FITS / TIFF / PNG / JPEG, heure de prise de vue,
// calage (deux clics ou solution WCS du FITS), annotation des satellites.
// Tout se passe dans le navigateur : la photo n'est jamais envoyée sur un serveur.

const DEG = Math.PI / 180;

// ─── FITS ───────────────────────────────────────────────────────────────────

function parseCard(card) {
  const key = card.slice(0, 8).trim();
  if (card.slice(8, 10) !== '= ') return [key, null];
  let v = card.slice(10);
  if (v.trim().startsWith("'")) {
    const m = v.match(/'((?:[^']|'')*)'/);
    return [key, m ? m[1].replace(/''/g, "'").trim() : ''];
  }
  v = v.split('/')[0].trim();
  if (v === 'T') return [key, true];
  if (v === 'F') return [key, false];
  const n = Number(v.replace(/D/i, 'E'));
  return [key, Number.isNaN(n) ? v : n];
}

/** Lit un fichier FITS : première HDU contenant une image 2D (ou un cube moyenné). */
export function parseFits(buffer) {
  const bytes = new Uint8Array(buffer);
  const dv = new DataView(buffer);
  let pos = 0;
  while (pos < bytes.length) {
    const header = {};
    let end = false;
    while (!end && pos < bytes.length) {
      for (let i = 0; i < 36; i++) {
        const card = String.fromCharCode(...bytes.subarray(pos + i * 80, pos + i * 80 + 80));
        if (card.startsWith('END')) { end = true; break; }
        const [k, v] = parseCard(card);
        if (k && v !== null && !(k in header)) header[k] = v;
      }
      pos += 2880;
    }
    const naxis = header.NAXIS || 0;
    const dims = [];
    for (let i = 1; i <= naxis; i++) dims.push(header[`NAXIS${i}`]);
    const bitpix = header.BITPIX;
    const count = dims.length ? dims.reduce((a, b) => a * b, 1) : 0;
    const bytesPer = Math.abs(bitpix) / 8;
    const dataBytes = count * bytesPer;
    if (naxis >= 2 && count > 0) {
      const [nx, ny] = dims;
      const planes = count / (nx * ny);
      const bscale = header.BSCALE ?? 1, bzero = header.BZERO ?? 0;
      const read = {
        8: (o) => dv.getUint8(o), 16: (o) => dv.getInt16(o), 32: (o) => dv.getInt32(o),
        '-32': (o) => dv.getFloat32(o), '-64': (o) => dv.getFloat64(o),
      }[bitpix];
      const gray = new Float32Array(nx * ny);
      for (let p = 0; p < planes; p++) {
        for (let j = 0; j < ny; j++) {
          // ligne FITS j (en bas) → ligne d'affichage ny-1-j (en haut)
          const dst = (ny - 1 - j) * nx;
          for (let i = 0; i < nx; i++) {
            const v = read(pos + ((p * ny + j) * nx + i) * bytesPer) * bscale + bzero;
            gray[dst + i] += (Number.isFinite(v) ? v : 0) / planes;
          }
        }
      }
      return { width: nx, height: ny, gray, header };
    }
    pos += Math.ceil(dataBytes / 2880) * 2880;
  }
  throw new Error('Aucune image trouvée dans le fichier FITS');
}

export function fitsDate(header) {
  for (const k of ['DATE-OBS', 'DATE_OBS', 'DATE-BEG']) {
    let v = header[k];
    if (typeof v !== 'string') continue;
    if (!v.includes('T') && typeof header['TIME-OBS'] === 'string') v = `${v.slice(0, 10)}T${header['TIME-OBS']}`;
    const d = parseIsoUtc(v);
    if (d) return { date: d, source: k };
  }
  return null;
}

function parseIsoUtc(s) {
  const m = String(s).trim().match(/^(\d{4})[-:](\d{2})[-:](\d{2})(?:[T ](\d{2}):(\d{2})(?::(\d{2}(?:\.\d+)?))?)?/);
  if (!m) return null;
  const sec = m[6] ? parseFloat(m[6]) : 0;
  return new Date(Date.UTC(+m[1], +m[2] - 1, +m[3], +(m[4] || 0), +(m[5] || 0), Math.floor(sec), Math.round((sec % 1) * 1000)));
}

/** Solution astrométrique TAN de l'en-tête (sans distorsion SIP), ou null. */
export function fitsWcs(header, height) {
  const ct1 = String(header.CTYPE1 || ''), ct2 = String(header.CTYPE2 || '');
  if (!ct1.startsWith('RA') || !ct2.startsWith('DEC') || !ct1.includes('TAN')) return null;
  let cd;
  if ('CD1_1' in header) {
    cd = [[header.CD1_1, header.CD1_2 || 0], [header.CD2_1 || 0, header.CD2_2]];
  } else if ('CDELT1' in header) {
    const c1 = header.CDELT1, c2 = header.CDELT2;
    if ('PC1_1' in header) {
      cd = [[c1 * header.PC1_1, c1 * (header.PC1_2 || 0)], [c2 * (header.PC2_1 || 0), c2 * header.PC2_2]];
    } else {
      const rho = (header.CROTA2 || 0) * DEG;
      cd = [[c1 * Math.cos(rho), -c2 * Math.sin(rho)], [c1 * Math.sin(rho), c2 * Math.cos(rho)]];
    }
  } else return null;
  const det = cd[0][0] * cd[1][1] - cd[0][1] * cd[1][0];
  if (!det) return null;
  return {
    ra0: header.CRVAL1 * DEG, dec0: header.CRVAL2 * DEG, crpix: [header.CRPIX1, header.CRPIX2], cd, det, height,
    scale: Math.sqrt(Math.abs(det)) * 3600,
  };
}

/** (ra, dec) en radians → pixel d'affichage (x, y vers le bas). */
export function wcsPixel(w, ra, dec) {
  const da = ra - w.ra0;
  const cosc = Math.sin(w.dec0) * Math.sin(dec) + Math.cos(w.dec0) * Math.cos(dec) * Math.cos(da);
  const xi = (Math.cos(dec) * Math.sin(da) / cosc) / DEG;
  const eta = ((Math.sin(dec) * Math.cos(w.dec0) - Math.cos(dec) * Math.sin(w.dec0) * Math.cos(da)) / cosc) / DEG;
  const [[a, b], [c, d]] = w.cd;
  const px = (d * xi - b * eta) / w.det;
  const py = (-c * xi + a * eta) / w.det;
  const x = px + w.crpix[0] - 1, y = py + w.crpix[1] - 1;
  return [x, w.height - 1 - y];
}

// ─── EXIF (JPEG et TIFF) ────────────────────────────────────────────────────

function tiffDate(dv, base) {
  const le = dv.getUint16(base) === 0x4949;
  const u16 = (o) => dv.getUint16(base + o, le), u32 = (o) => dv.getUint32(base + o, le);
  const ascii = (o, n) => {
    let s = '';
    for (let i = 0; i < n; i++) {
      const c = dv.getUint8(base + o + i);
      if (!c) break;
      s += String.fromCharCode(c);
    }
    return s;
  };
  const readIfd = (off) => {
    const tags = {};
    const n = u16(off);
    for (let i = 0; i < n; i++) {
      const e = off + 2 + i * 12;
      const tag = u16(e), type = u16(e + 2), cnt = u32(e + 4);
      if (type === 2) tags[tag] = ascii(cnt > 4 ? u32(e + 8) : e + 8, cnt);
      else if (type === 4 || type === 13) tags[tag] = u32(e + 8);
    }
    return tags;
  };
  const ifd0 = readIfd(u32(4));
  let raw = null;
  if (ifd0[0x8769]) {
    const ex = readIfd(ifd0[0x8769]);
    raw = ex[0x9003] || ex[0x9004];
  }
  raw = raw || ifd0[0x0132];
  return raw ? parseIsoUtc(raw) : null;
}

/** Date EXIF (heure de l'appareil, souvent locale) d'un JPEG ou d'un TIFF. */
export function exifDate(buffer) {
  const dv = new DataView(buffer);
  try {
    if (dv.getUint16(0) === 0xffd8) {
      let o = 2;
      while (o + 4 < dv.byteLength) {
        const marker = dv.getUint16(o), len = dv.getUint16(o + 2);
        if (marker === 0xffe1 && dv.getUint32(o + 4) === 0x45786966) return tiffDate(dv, o + 10);
        if ((marker & 0xff00) !== 0xff00) break;
        o += 2 + len;
      }
      return null;
    }
    const bo = dv.getUint16(0);
    if (bo === 0x4949 || bo === 0x4d4d) return tiffDate(dv, 0);
  } catch (e) {
    return null;
  }
  return null;
}

// ─── Chargement ─────────────────────────────────────────────────────────────

async function loadUtif() {
  if (globalThis.UTIF) return globalThis.UTIF;
  await new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = 'https://cdn.jsdelivr.net/npm/utif@3.1.0/UTIF.js';
    s.onload = resolve;
    s.onerror = () => reject(new Error('Lecteur TIFF indisponible (connexion nécessaire la première fois)'));
    document.head.appendChild(s);
  });
  return globalThis.UTIF;
}

function rgbaToGray(rgba, n) {
  const g = new Float32Array(n);
  for (let i = 0; i < n; i++) g[i] = (rgba[4 * i] + rgba[4 * i + 1] + rgba[4 * i + 2]) / 3;
  return g;
}

/** Charge une photo (objet File du navigateur). */
export async function loadPhoto(file) {
  const buffer = await file.arrayBuffer();
  const name = file.name.toLowerCase();
  if (/\.(fits?|fts)$/.test(name)) {
    const f = parseFits(buffer);
    const d = fitsDate(f.header);
    return {
      width: f.width, height: f.height, gray: f.gray, rgba: null,
      date: d?.date || null, dateSource: d?.source || '', wcs: fitsWcs(f.header, f.height),
    };
  }
  if (/\.tiff?$/.test(name)) {
    const UTIF = await loadUtif();
    const ifds = UTIF.decode(buffer);
    UTIF.decodeImage(buffer, ifds[0]);
    const rgba = UTIF.toRGBA8(ifds[0]);
    const { width, height } = ifds[0];
    const date = exifDate(buffer);
    return { width, height, gray: rgbaToGray(rgba, width * height), rgba: new Uint8ClampedArray(rgba), date, dateSource: date ? 'EXIF' : '', wcs: null };
  }
  const bmp = await createImageBitmap(file);
  const c = new OffscreenCanvas(bmp.width, bmp.height);
  const ctx = c.getContext('2d');
  ctx.drawImage(bmp, 0, 0);
  const rgba = ctx.getImageData(0, 0, bmp.width, bmp.height).data;
  const date = exifDate(buffer);
  return {
    width: bmp.width, height: bmp.height, gray: rgbaToGray(rgba, bmp.width * bmp.height),
    rgba: new Uint8ClampedArray(rgba), date, dateSource: date ? 'EXIF' : '', wcs: null,
  };
}

/** Image affichable (RGBA 8 bits) : couleurs d'origine, ou étirement asinh pour un FITS. */
export function displayRgba(photo) {
  if (photo.rgba) return photo.rgba;
  const g = photo.gray;
  const sample = [];
  const step = Math.max(1, Math.floor(g.length / 200000));
  for (let i = 0; i < g.length; i += step) sample.push(g[i]);
  sample.sort((a, b) => a - b);
  const lo = sample[Math.floor(sample.length * 0.005)];
  let hi = sample[Math.floor(sample.length * 0.998)];
  if (hi <= lo) hi = lo + 1;
  const out = new Uint8ClampedArray(g.length * 4);
  const k = Math.asinh(10);
  for (let i = 0; i < g.length; i++) {
    const x = Math.min(1, Math.max(0, (g[i] - lo) / (hi - lo)));
    const v = Math.round((Math.asinh(10 * x) / k) * 255);
    out[4 * i] = out[4 * i + 1] = out[4 * i + 2] = v;
    out[4 * i + 3] = 255;
  }
  return out;
}

// ─── Mesures ────────────────────────────────────────────────────────────────

function median(arr) {
  const a = Float32Array.from(arr).sort();
  return a.length ? a[a.length >> 1] : 0;
}

/** Barycentre des pixels au-dessus de la moitié du maximum (la planète domine l'image). */
export function detectPlanetCenter(photo) {
  const { gray, width, height } = photo;
  const step = Math.max(1, Math.floor(gray.length / 100000));
  const sub = [];
  for (let i = 0; i < gray.length; i += step) sub.push(gray[i]);
  const bg = median(sub);
  let max = -Infinity;
  for (let i = 0; i < gray.length; i++) if (gray[i] > max) max = gray[i];
  const thr = bg + 0.5 * (max - bg);
  let sx = 0, sy = 0, sw = 0;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const v = gray[y * width + x] - bg;
      if (v + bg > thr) { sx += x * v; sy += y * v; sw += v; }
    }
  }
  return [sx / sw, sy / sw];
}

/** Point lumineux le plus proche de (x, y) : barycentre autour du maximum local. */
export function nearestSource(photo, x, y, radius = 12, nsigma = 5) {
  const { gray, width: w, height: h } = photo;
  const x0 = Math.max(0, Math.floor(x) - radius), x1 = Math.min(w, Math.floor(x) + radius + 1);
  const y0 = Math.max(0, Math.floor(y) - radius), y1 = Math.min(h, Math.floor(y) + radius + 1);
  if (x1 <= x0 || y1 <= y0) return null;
  const win = [];
  let best = -Infinity, bx = 0, by = 0;
  for (let j = y0; j < y1; j++) {
    for (let i = x0; i < x1; i++) {
      const v = gray[j * w + i];
      win.push(v);
      if (v > best) { best = v; bx = i; by = j; }
    }
  }
  const bg = median(win);
  const mad = median(win.map((v) => Math.abs(v - bg))) * 1.4826 + 1e-9;
  if (best - bg < nsigma * mad) return null;
  let sx = 0, sy = 0, sw = 0;
  for (let j = Math.max(y0, by - 2); j < Math.min(y1, by + 3); j++) {
    for (let i = Math.max(x0, bx - 2); i < Math.min(x1, bx + 3); i++) {
      const v = Math.max(0, gray[j * w + i] - bg);
      sx += i * v; sy += j * v; sw += v;
    }
  }
  return sw > 0 ? [sx / sw, sy / sw] : null;
}

// ─── Calage par deux points ─────────────────────────────────────────────────

export class Calibration {
  constructor(center, scale, rotationDeg, mirrored) {
    Object.assign(this, { center, scale, rotationDeg, mirrored });
  }

  skyToPixel(east, north) {
    const u = this.mirrored ? east : -east;
    const t = this.rotationDeg * DEG;
    const u2 = (u * Math.cos(t) - north * Math.sin(t)) / this.scale;
    const v2 = (u * Math.sin(t) + north * Math.cos(t)) / this.scale;
    return [this.center[0] + u2, this.center[1] - v2];
  }
}

/** Échelle et rotation déduites du centre de la planète et d'un satellite identifié. */
export function calibrate(centerPx, refPx, refEast, refNorth, mirrored) {
  const u = mirrored ? refEast : -refEast;
  const du = refPx[0] - centerPx[0], dv = -(refPx[1] - centerPx[1]);
  const dImg = Math.hypot(du, dv);
  if (dImg < 1) throw new Error('Le satellite de référence est trop près du centre de la planète.');
  const scale = Math.hypot(u, refNorth) / dImg;
  const rot = ((Math.atan2(dv, du) - Math.atan2(refNorth, u)) / DEG % 360 + 360) % 360;
  return new Calibration(centerPx, scale, rot, mirrored);
}
