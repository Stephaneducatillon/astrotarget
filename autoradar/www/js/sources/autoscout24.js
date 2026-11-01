// Source AutoScout24 : couvre la France, la Belgique, les Pays-Bas, l'Allemagne et l'Italie.
// Les pages de résultats embarquent la liste des annonces dans le JSON __NEXT_DATA__.

import {
  toInt, yearFrom, slugify, normalizeFuel, normalizeGearbox,
  extractNextData, deepFind,
} from '../normalize.js';

const SITES = {
  FR: { origin: 'https://www.autoscout24.fr', prefix: '', cy: 'F' },
  BE: { origin: 'https://www.autoscout24.be', prefix: '/fr', cy: 'B' },
  NL: { origin: 'https://www.autoscout24.nl', prefix: '', cy: 'NL' },
  DE: { origin: 'https://www.autoscout24.de', prefix: '', cy: 'D' },
  IT: { origin: 'https://www.autoscout24.it', prefix: '', cy: 'I' },
};

const FUEL_CODES = {
  Essence: 'B', Diesel: 'D', 'Électrique': 'E', Hybride: '2', 'Hybride rechargeable': '2', GPL: 'L', GNV: 'C',
};
const GEAR_CODES = { Automatique: 'A', Manuelle: 'M', 'Semi-automatique': 'S' };
const SORT = {
  price_asc: ['price', 0], price_desc: ['price', 1], year_desc: ['year', 1], km_asc: ['mileage', 0], relevance: ['standard', 0],
};

export const name = 'AutoScout24';
export const countries = Object.keys(SITES);

export function buildUrl(country, c, page = 1) {
  const site = SITES[country];
  let path = `${site.prefix}/lst`;
  if (c.make) {
    path += `/${slugify(c.make)}`;
    if (c.model) path += `/${slugify(c.model)}`;
  }
  const [sort, desc] = SORT[c.sort] || SORT.relevance;
  const p = new URLSearchParams({
    atype: 'C', cy: site.cy, damaged_listing: 'exclude', ustate: 'N,U', sort, desc: String(desc), page: String(page),
  });
  if (c.priceMin) p.set('pricefrom', c.priceMin);
  if (c.priceMax) p.set('priceto', c.priceMax);
  if (c.yearMin) p.set('fregfrom', c.yearMin);
  if (c.yearMax) p.set('fregto', c.yearMax);
  if (c.kmMax) p.set('kmto', c.kmMax);
  if (c.fuel && FUEL_CODES[c.fuel]) p.set('fuel', FUEL_CODES[c.fuel]);
  if (c.gearbox && GEAR_CODES[c.gearbox]) p.set('gear', GEAR_CODES[c.gearbox]);
  return `${site.origin}${path}?${p.toString()}`;
}

function absoluteUrl(country, url) {
  if (!url) return null;
  if (/^https?:/.test(url)) return url;
  const site = SITES[country];
  const needsPrefix = site.prefix && !url.startsWith(site.prefix + '/');
  return site.origin + (needsPrefix ? site.prefix : '') + url;
}

function pickImage(images) {
  if (!Array.isArray(images) || !images.length) return null;
  const first = images[0];
  const url = typeof first === 'string' ? first : first?.url || first?.src;
  return url ? url.replace(/\/\d+x\d+\.(jpg|webp)$/i, '/480x360.$1') : null;
}

export function parseListing(l, country) {
  const v = l.vehicle || {};
  const t = l.tracking || {};
  const loc = l.location || {};
  const title = [v.make, v.model, v.modelVersionInput || v.variant].filter(Boolean).join(' ').trim();
  const details = Array.isArray(l.vehicleDetails) ? l.vehicleDetails : [];
  const detail = (re) => details.find((d) => re.test(d.ariaLabel || d.iconName || ''))?.data;
  return {
    id: `as24-${country}-${l.id}`,
    source: name,
    country: (loc.countryCode && countryFromCode(loc.countryCode)) || country,
    title: title || l.title || 'Véhicule',
    price: toInt(t.price) ?? toInt(l.price?.priceFormatted),
    year: yearFrom(t.firstRegistration) ?? yearFrom(detail(/regist|immatric|erstzul|bouwjaar|calendar/i)),
    km: toInt(t.mileage) ?? toInt(v.mileageInKm) ?? toInt(detail(/mileage|kilom|road/i)),
    fuel: normalizeFuel(v.fuel || detail(/fuel|carbur|kraftstoff|brandstof|aliment/i)),
    gearbox: normalizeGearbox(v.transmission || detail(/gear|transmi|getriebe|cambio/i)),
    city: [loc.zip, loc.city].filter(Boolean).join(' ') || null,
    image: pickImage(l.images),
    url: absoluteUrl(country, l.url),
  };
}

function countryFromCode(code) {
  const map = { F: 'FR', B: 'BE', NL: 'NL', D: 'DE', I: 'IT', FR: 'FR', BE: 'BE', DE: 'DE', IT: 'IT', P: 'PT', PT: 'PT' };
  return map[String(code).toUpperCase()] || null;
}

export function parsePage(html, country, page = 1) {
  const data = extractNextData(html);
  if (!data) throw new Error('Format de page inattendu (pas de __NEXT_DATA__)');
  const props = data.props?.pageProps || {};
  let listings = props.listings;
  if (!Array.isArray(listings)) {
    const holder = deepFind(data, (o) => Array.isArray(o.listings) && o.listings.some((x) => x && x.vehicle));
    listings = holder?.listings || [];
  }
  const total = toInt(props.numberOfResults) ?? listings.length;
  const ads = listings.filter((l) => l && l.id && l.url).map((l) => parseListing(l, country));
  return { ads, total, hasMore: ads.length > 0 && total > page * 20 };
}

export async function search(http, country, criteria, page = 1) {
  const html = await http(buildUrl(country, criteria, page));
  return parsePage(html, country, page);
}
