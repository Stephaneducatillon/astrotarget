// Source Standvirtual : principal site d'annonces auto au Portugal.
// Les résultats sont dans __NEXT_DATA__ (cache GraphQL « urqlState »).

import {
  toInt, yearFrom, slugify, normalizeFuel, normalizeGearbox, extractNextData, deepFind,
} from '../normalize.js';

const ORIGIN = 'https://www.standvirtual.com';
const SORT = {
  price_asc: 'filter_float_price:asc',
  price_desc: 'filter_float_price:desc',
  year_desc: 'filter_float_first_registration_year:desc',
  km_asc: 'filter_float_mileage:asc',
  relevance: 'relevance_web',
};

export const name = 'Standvirtual';
export const countries = ['PT'];

export function buildUrl(_country, c, page = 1) {
  let path = '/carros';
  if (c.make) {
    path += `/${slugify(c.make)}`;
    if (c.model) path += `/${slugify(c.model)}`;
  }
  const p = new URLSearchParams();
  if (c.priceMin) p.set('search[filter_float_price:from]', c.priceMin);
  if (c.priceMax) p.set('search[filter_float_price:to]', c.priceMax);
  if (c.yearMin) p.set('search[filter_float_first_registration_year:from]', c.yearMin);
  if (c.yearMax) p.set('search[filter_float_first_registration_year:to]', c.yearMax);
  if (c.kmMax) p.set('search[filter_float_mileage:to]', c.kmMax);
  p.set('search[order]', SORT[c.sort] || SORT.relevance);
  if (page > 1) p.set('page', page);
  return `${ORIGIN}${path}?${p.toString()}`;
}

function param(node, key) {
  const list = Array.isArray(node.parameters) ? node.parameters : [];
  const p = list.find((x) => x && x.key === key);
  return p ? (p.displayValue ?? p.value) : null;
}

export function parseNode(n) {
  const loc = n.location || {};
  const city = [loc.city?.name, loc.region?.name].filter(Boolean).join(', ');
  return {
    id: `sv-${n.id}`,
    source: name,
    country: 'PT',
    title: n.title || [param(n, 'make'), param(n, 'model')].filter(Boolean).join(' ') || 'Veículo',
    price: toInt(n.price?.amount?.units ?? n.price?.amount?.value ?? n.price?.value),
    year: yearFrom(param(n, 'first_registration_year')),
    km: toInt(param(n, 'mileage')),
    fuel: normalizeFuel(param(n, 'fuel_type')),
    gearbox: normalizeGearbox(param(n, 'gearbox')),
    city: city || null,
    image: n.thumbnail?.x2 || n.thumbnail?.x1 || null,
    url: n.url && !/^https?:/.test(n.url) ? ORIGIN + n.url : n.url,
  };
}

/** Récupère l'objet { edges, totalCount } quel que soit l'endroit où il est stocké. */
function findSearch(data) {
  const isSearch = (o) => Array.isArray(o.edges) && o.edges.some((e) => e?.node?.url);
  const direct = deepFind(data, isSearch);
  if (direct) return direct;
  const urql = deepFind(data, (o) => o.urqlState && typeof o.urqlState === 'object')?.urqlState;
  for (const entry of Object.values(urql || {})) {
    if (typeof entry?.data !== 'string') continue;
    try {
      const found = deepFind(JSON.parse(entry.data), isSearch);
      if (found) return found;
    } catch { /* entrée non JSON, on ignore */ }
  }
  return null;
}

export function parsePage(html, _country, page = 1) {
  const data = extractNextData(html);
  if (!data) throw new Error('Format de page inattendu (pas de __NEXT_DATA__)');
  const res = findSearch(data);
  if (!res) return { ads: [], total: 0, hasMore: false };
  const ads = res.edges.map((e) => e.node).filter((n) => n?.url).map(parseNode);
  const total = toInt(res.totalCount) ?? ads.length;
  return { ads, total, hasMore: ads.length > 0 && total > page * 32 };
}

export async function search(http, country, criteria, page = 1) {
  const html = await http(buildUrl(country, criteria, page));
  return parsePage(html, country, page);
}
