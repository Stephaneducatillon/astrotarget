// Lecture des tables précalculées (polynômes de Tchebychev par tranche).
//
// Format (voir tools/build_tables.py) : un fichier binaire par planète et par
// année, séries mises bout à bout dans l'ordre de meta.json ; pour chaque
// série, nseg × ncomp × (deg + 1) coefficients (float64 pour la planète,
// float32 pour les satellites). Le temps est compté en jours UT depuis le
// 1er janvier 00:00 UT de l'année. Une tranche remplie de NaN signifie
// « position non disponible ».

const DAY_MS = 86400000;

let loader = async (path) => {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path} : ${r.status}`);
  return path.endsWith('.json') ? r.json() : r.arrayBuffer();
};

let baseUrl = new URL('../data/', import.meta.url).href;

/** Remplace le chargeur (tests sous Node : lecture sur disque). */
export function setLoader(fn, base) {
  loader = fn;
  if (base !== undefined) baseUrl = base;
}

const metaCache = new Map();
const yearCache = new Map();

export async function loadMeta(planetKey) {
  const k = planetKey.toLowerCase();
  if (!metaCache.has(k)) metaCache.set(k, loader(`${baseUrl}${k}/meta.json`));
  return metaCache.get(k);
}

function daysInYear(y) {
  return (Date.UTC(y + 1, 0, 1) - Date.UTC(y, 0, 1)) / DAY_MS;
}

async function loadYear(planetKey, year) {
  const k = `${planetKey.toLowerCase()}/${year}`;
  if (!yearCache.has(k)) {
    yearCache.set(k, (async () => {
      const meta = await loadMeta(planetKey);
      if (!meta.years.includes(year)) return null;
      const buf = await loader(`${baseUrl}${k}.bin`);
      const nDays = daysInYear(year);
      const series = {};
      let offset = 0;
      for (const s of meta.series) {
        const nseg = Math.ceil(nDays / s.seg);
        const n = nseg * s.comps.length * (s.deg + 1);
        const Arr = s.type === 'f64' ? Float64Array : Float32Array;
        // copie : l'alignement du tampon d'origine n'est pas garanti
        const bytes = n * Arr.BYTES_PER_ELEMENT;
        const view = new Arr(buf.slice(offset, offset + bytes));
        offset += bytes;
        series[s.name] = { ...s, nseg, data: view };
      }
      if (offset !== buf.byteLength) throw new Error(`Taille inattendue pour ${k}.bin`);
      return series;
    })());
  }
  return yearCache.get(k);
}

function chebEval(data, base, deg, x) {
  // récurrence de Clenshaw
  let b0 = 0, b1 = 0, b2 = 0;
  const x2 = 2 * x;
  for (let i = deg; i >= 1; i--) {
    b2 = b1; b1 = b0;
    b0 = data[base + i] + x2 * b1 - b2;
  }
  return data[base] + x * b0 - b1;
}

function evalSeries(s, dayOffset) {
  let idx = Math.floor(dayOffset / s.seg);
  if (idx >= s.nseg) idx = s.nseg - 1;
  if (idx < 0) idx = 0;
  const x = 2 * (dayOffset - idx * s.seg) / s.seg - 1;
  const ncomp = s.comps.length, n = s.deg + 1;
  const out = new Array(ncomp);
  for (let c = 0; c < ncomp; c++) {
    out[c] = chebEval(s.data, (idx * ncomp + c) * n, s.deg, x);
  }
  return out;
}

// Années déjà chargées (pour les calculs synchrones, ex. phénomènes)
const resolved = new Map();

/**
 * Valeurs des tables à l'instant ``date`` (objet Date) : null si l'année n'est
 * pas couverte, sinon { planet: {ra, dec, gra, gdec, radius, cml1?, cml2?},
 * moons: {Io: [x, y, z] | null, …} } (null : position non disponible).
 */
export async function sampleAsync(planetKey, date) {
  const y = date.getUTCFullYear();
  const series = await loadYear(planetKey, y);
  resolved.set(`${planetKey.toLowerCase()}/${y}`, series);
  return series ? sampleFrom(series, planetKey, date) : null;
}

/** Version synchrone, utilisable après un premier sampleAsync() pour la même année. */
export function sampleSync(planetKey, date) {
  const y = date.getUTCFullYear();
  const series = resolved.get(`${planetKey.toLowerCase()}/${y}`);
  if (series === undefined) throw new Error(`Tables ${planetKey} ${y} non chargées`);
  return series ? sampleFrom(series, planetKey, date) : null;
}

export async function preload(planetKey, dates) {
  for (const d of dates) await sampleAsync(planetKey, d);
}

function sampleFrom(series, planetKey, date) {
  const y = date.getUTCFullYear();
  const off = (date.getTime() - Date.UTC(y, 0, 1)) / DAY_MS;
  const out = { planet: {}, moons: {} };
  for (const name in series) {
    const s = series[name];
    const v = evalSeries(s, off);
    if (name === 'planet') {
      s.comps.forEach((c, i) => { out.planet[c] = v[i]; });
    } else {
      out.moons[name] = v.some((x) => Number.isNaN(x)) ? null : v;
    }
  }
  return out;
}
