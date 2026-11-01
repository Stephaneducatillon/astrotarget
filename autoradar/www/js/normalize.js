// Fonctions utilitaires de normalisation, partagées par toutes les sources.

export const COUNTRIES = {
  FR: { code: 'FR', name: 'France', flag: '🇫🇷' },
  BE: { code: 'BE', name: 'Belgique', flag: '🇧🇪' },
  NL: { code: 'NL', name: 'Pays-Bas', flag: '🇳🇱' },
  DE: { code: 'DE', name: 'Allemagne', flag: '🇩🇪' },
  IT: { code: 'IT', name: 'Italie', flag: '🇮🇹' },
  PT: { code: 'PT', name: 'Portugal', flag: '🇵🇹' },
};

export function toInt(v) {
  if (v === null || v === undefined || v === '') return null;
  if (typeof v === 'number') return Number.isFinite(v) ? Math.round(v) : null;
  const digits = String(v).replace(/[^\d]/g, '');
  return digits ? parseInt(digits, 10) : null;
}

export function yearFrom(v) {
  if (v === null || v === undefined) return null;
  const m = String(v).match(/(19|20)\d{2}/);
  return m ? parseInt(m[0], 10) : null;
}

export function slugify(s) {
  return String(s || '')
    .normalize('NFD').replace(/[̀-ͯ]/g, '')
    .toLowerCase().trim()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
}

const FUEL_RULES = [
  [/plug|rechargeable|phev/i, 'Hybride rechargeable'],
  [/hybri|ibrid|híbrid/i, 'Hybride'],
  [/elec|élec|elett|eléct|elektr/i, 'Électrique'],
  [/diesel|gasóleo|gasoleo/i, 'Diesel'],
  [/lpg|gpl|autogas/i, 'GPL'],
  [/cng|gnv|metano|erdgas|aardgas/i, 'GNV'],
  [/petrol|essence|benzin|benzina|gasolina|super|gaz/i, 'Essence'],
];

export function normalizeFuel(v) {
  if (!v) return null;
  const s = String(v);
  for (const [re, label] of FUEL_RULES) if (re.test(s)) return label;
  return s;
}

export function normalizeGearbox(v) {
  if (!v) return null;
  const s = String(v);
  if (/semi/i.test(s)) return 'Semi-automatique';
  if (/auto/i.test(s)) return 'Automatique';
  if (/man|schalt|handgeschakeld/i.test(s)) return 'Manuelle';
  return s;
}

/** Vérifie qu'une annonce normalisée respecte les critères (filet de sécurité côté app). */
export function matchesCriteria(ad, c) {
  if (c.priceMin && ad.price != null && ad.price < c.priceMin) return false;
  if (c.priceMax && ad.price != null && ad.price > c.priceMax) return false;
  if (c.yearMin && ad.year != null && ad.year < c.yearMin) return false;
  if (c.yearMax && ad.year != null && ad.year > c.yearMax) return false;
  if (c.kmMax && ad.km != null && ad.km > c.kmMax) return false;
  if (c.fuel && ad.fuel && !ad.fuel.startsWith(c.fuel)) return false;
  if (c.gearbox && ad.gearbox && ad.gearbox !== c.gearbox) return false;
  if (c.keywords) {
    const words = c.keywords.toLowerCase().split(/\s+/).filter(Boolean);
    const hay = (ad.title || '').toLowerCase();
    if (!words.every((w) => hay.includes(w))) return false;
  }
  return true;
}

export function sortAds(ads, sort) {
  const nullLast = (v, dir) => (v == null ? Infinity : dir * v);
  const by = {
    price_asc: (a, b) => nullLast(a.price, 1) - nullLast(b.price, 1),
    price_desc: (a, b) => nullLast(a.price, -1) - nullLast(b.price, -1),
    year_desc: (a, b) => nullLast(a.year, -1) - nullLast(b.year, -1),
    km_asc: (a, b) => nullLast(a.km, 1) - nullLast(b.km, 1),
  }[sort];
  return by ? [...ads].sort(by) : ads;
}

/** Retrouve le JSON embarqué par les sites Next.js. */
export function extractNextData(html) {
  const m = html.match(/<script[^>]*id="__NEXT_DATA__"[^>]*>([\s\S]*?)<\/script>/);
  if (!m) return null;
  try { return JSON.parse(m[1]); } catch { return null; }
}

/** Parcours en profondeur : renvoie le premier nœud pour lequel predicate est vrai. */
export function deepFind(obj, predicate, seen = new Set()) {
  if (!obj || typeof obj !== 'object' || seen.has(obj)) return null;
  seen.add(obj);
  if (predicate(obj)) return obj;
  for (const k of Object.keys(obj)) {
    const r = deepFind(obj[k], predicate, seen);
    if (r) return r;
  }
  return null;
}
