// Source de démonstration : annonces FICTIVES, clairement marquées « DÉMO » dans l'interface.
// Sert à tester l'application dans un navigateur (où les vrais sites sont bloqués par CORS)
// ou hors connexion. Désactivable dans les Réglages.

import { MAKES } from '../makes.js';

export const name = 'Démo';
export const countries = ['FR', 'BE', 'NL', 'DE', 'IT', 'PT'];

const CITIES = {
  FR: ['75011 Paris', '69003 Lyon', '13008 Marseille', '59000 Lille', '33000 Bordeaux', '31000 Toulouse'],
  BE: ['1000 Bruxelles', '4000 Liège', '2000 Anvers', '9000 Gand', '5000 Namur'],
  NL: ['1012 Amsterdam', '3011 Rotterdam', '3511 Utrecht', '5611 Eindhoven'],
  DE: ['10115 Berlin', '80331 München', '50667 Köln', '20095 Hamburg', '70173 Stuttgart'],
  IT: ['20121 Milano', '00184 Roma', '10121 Torino', '40121 Bologna'],
  PT: ['Lisboa', 'Porto', 'Braga', 'Coimbra', 'Faro'],
};
const FUELS = ['Essence', 'Diesel', 'Hybride', 'Électrique'];
const GEARS = ['Manuelle', 'Automatique'];

function rng(seed) {
  let s = seed >>> 0;
  return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32);
}
function hash(str) {
  let h = 2166136261;
  for (const ch of str) h = Math.imul(h ^ ch.charCodeAt(0), 16777619);
  return h >>> 0;
}

export async function search(_http, country, c, page = 1) {
  const r = rng(hash(`${country}|${c.make}|${c.model}|${page}`));
  const pick = (arr) => arr[Math.floor(r() * arr.length)];
  const makes = c.make ? [c.make] : Object.keys(MAKES);
  const ads = [];
  for (let i = 0; i < 12; i++) {
    const make = pick(makes);
    const model = c.model || pick(MAKES[make] || ['—']);
    const yMin = c.yearMin || 2008;
    const yMax = c.yearMax || 2025;
    const year = yMin + Math.floor(r() * (yMax - yMin + 1));
    const age = 2026 - year;
    const km = Math.min(c.kmMax || Infinity, Math.round((age * 14000 + r() * 20000) / 100) * 100);
    let price = Math.round((32000 * Math.pow(0.86, age) * (0.8 + r() * 0.5)) / 50) * 50;
    if (c.priceMin) price = Math.max(price, c.priceMin);
    if (c.priceMax) price = Math.min(price, c.priceMax);
    const id = `demo-${country}-${page}-${i}`;
    ads.push({
      id, source: name, demo: true, country,
      title: `${make} ${model}`,
      price, year, km,
      fuel: c.fuel || pick(FUELS),
      gearbox: c.gearbox || pick(GEARS),
      city: pick(CITIES[country]),
      image: null,
      url: null,
    });
  }
  return { ads, total: 120, hasMore: page < 10 };
}
