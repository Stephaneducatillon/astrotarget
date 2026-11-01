import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as as24 from '../www/js/sources/autoscout24.js';
import * as sv from '../www/js/sources/standvirtual.js';
import { Search } from '../www/js/aggregator.js';
import { normalizeFuel, normalizeGearbox, sortAds } from '../www/js/normalize.js';

const page = (data) => `<html><body><script id="__NEXT_DATA__" type="application/json">${JSON.stringify(data)}</script></body></html>`;

const AS24_PAGE = page({
  props: {
    pageProps: {
      numberOfResults: 1234,
      listings: [{
        id: 'abc-123',
        url: '/offres/volkswagen-golf-1-5-tsi-essence-abc-123',
        images: ['https://prod.pictures.autoscout24.net/listing-images/xyz.jpg/250x188.webp'],
        price: { priceFormatted: '€ 18 990,-' },
        vehicle: { make: 'Volkswagen', model: 'Golf', modelVersionInput: '1.5 TSI Life', fuel: 'Essence', transmission: 'Boîte manuelle', mileageInKm: '45 000 km' },
        location: { countryCode: 'F', zip: '69003', city: 'Lyon' },
        tracking: { price: '18990', mileage: '45000', firstRegistration: '03-2020' },
      }],
    },
  },
});

test('AutoScout24 : URL de recherche', () => {
  const url = as24.buildUrl('BE', { make: 'Mercedes-Benz', model: 'C 220', priceMax: 20000, fuel: 'Diesel', gearbox: 'Automatique', sort: 'price_asc' }, 2);
  assert.match(url, /^https:\/\/www\.autoscout24\.be\/fr\/lst\/mercedes-benz\/c-220\?/);
  assert.match(url, /cy=B/);
  assert.match(url, /priceto=20000/);
  assert.match(url, /fuel=D/);
  assert.match(url, /gear=A/);
  assert.match(url, /sort=price&desc=0/);
  assert.match(url, /page=2/);
});

test('AutoScout24 : analyse d\'une page', () => {
  const res = as24.parsePage(AS24_PAGE, 'FR');
  assert.equal(res.total, 1234);
  assert.equal(res.hasMore, true);
  const [ad] = res.ads;
  assert.equal(ad.title, 'Volkswagen Golf 1.5 TSI Life');
  assert.equal(ad.price, 18990);
  assert.equal(ad.year, 2020);
  assert.equal(ad.km, 45000);
  assert.equal(ad.fuel, 'Essence');
  assert.equal(ad.gearbox, 'Manuelle');
  assert.equal(ad.country, 'FR');
  assert.equal(ad.city, '69003 Lyon');
  assert.equal(ad.url, 'https://www.autoscout24.fr/offres/volkswagen-golf-1-5-tsi-essence-abc-123');
  assert.match(ad.image, /480x360\.webp$/);
});

test('AutoScout24 : page sans __NEXT_DATA__ → erreur', () => {
  assert.throws(() => as24.parsePage('<html></html>', 'FR'));
});

const SV_PAGE = page({
  props: {
    pageProps: {
      urqlState: {
        '123': {
          data: JSON.stringify({
            advertSearch: {
              totalCount: 50,
              edges: [{
                node: {
                  id: '7890',
                  title: 'Peugeot 208 1.2 PureTech Active',
                  url: 'https://www.standvirtual.com/carros/anuncio/peugeot-208-ID7890.html',
                  location: { city: { name: 'Porto' }, region: { name: 'Porto' } },
                  thumbnail: { x1: 'https://img/x1.jpg', x2: 'https://img/x2.jpg' },
                  price: { amount: { units: 14500, currencyCode: 'EUR' } },
                  parameters: [
                    { key: 'first_registration_year', value: '2021', displayValue: '2021' },
                    { key: 'mileage', value: '38000', displayValue: '38 000 km' },
                    { key: 'fuel_type', value: 'gaz', displayValue: 'Gasolina' },
                    { key: 'gearbox', value: 'manual', displayValue: 'Manual' },
                  ],
                },
              }],
            },
          }),
        },
      },
    },
  },
});

test('Standvirtual : analyse d\'une page', () => {
  const res = sv.parsePage(SV_PAGE, 'PT');
  assert.equal(res.total, 50);
  assert.equal(res.hasMore, true);
  const [ad] = res.ads;
  assert.equal(ad.title, 'Peugeot 208 1.2 PureTech Active');
  assert.equal(ad.price, 14500);
  assert.equal(ad.year, 2021);
  assert.equal(ad.km, 38000);
  assert.equal(ad.fuel, 'Essence');
  assert.equal(ad.gearbox, 'Manuelle');
  assert.equal(ad.city, 'Porto, Porto');
  assert.equal(ad.country, 'PT');
});

test('Standvirtual : URL de recherche', () => {
  const url = sv.buildUrl('PT', { make: 'Peugeot', model: '208', priceMax: 15000, sort: 'price_asc' }, 3);
  assert.ok(url.startsWith('https://www.standvirtual.com/carros/peugeot/208?'));
  assert.ok(decodeURIComponent(url).includes('search[filter_float_price:to]=15000'));
  assert.ok(url.includes('page=3'));
});

test('Normalisation énergie / boîte multilingue', () => {
  assert.equal(normalizeFuel('Benzin'), 'Essence');
  assert.equal(normalizeFuel('Benzina'), 'Essence');
  assert.equal(normalizeFuel('Elektrisch'), 'Électrique');
  assert.equal(normalizeFuel('Elektro/Benzin (Plug-in-Hybrid)'), 'Hybride rechargeable');
  assert.equal(normalizeFuel('Gasóleo'), 'Diesel');
  assert.equal(normalizeGearbox('Automatik'), 'Automatique');
  assert.equal(normalizeGearbox('Schaltgetriebe'), 'Manuelle');
  assert.equal(normalizeGearbox('Handgeschakeld'), 'Manuelle');
});

test('Agrégateur : fusion multi-pays, filtres, tri, erreur isolée', async () => {
  const http = async (url) => {
    if (url.includes('standvirtual')) return SV_PAGE;
    if (url.includes('autoscout24.de')) throw new Error('HTTP 503');
    return AS24_PAGE;
  };
  const s = new Search({ countries: ['FR', 'DE', 'PT'], sort: 'price_asc' }, { http });
  const ads = await s.loadMore();
  assert.equal(ads.length, 2);
  assert.deepEqual(ads.map((a) => a.price), [14500, 18990]);
  assert.equal(s.channels.find((c) => c.country === 'DE').error, 'HTTP 503');

  const filtered = new Search({ countries: ['FR', 'PT'], priceMax: 15000 }, { http });
  assert.equal((await filtered.loadMore()).length, 1);
});

test('Tri : valeurs manquantes en fin de liste', () => {
  const out = sortAds([{ price: null }, { price: 5 }, { price: 2 }], 'price_desc');
  assert.deepEqual(out.map((a) => a.price), [5, 2, null]);
});

test('Mode démo : annonces marquées comme fictives', async () => {
  const s = new Search({ countries: ['IT', 'NL'], make: 'Fiat', model: '500' }, { demoMode: true });
  const ads = await s.loadMore();
  assert.ok(ads.length > 0);
  assert.ok(ads.every((a) => a.demo && a.title === 'Fiat 500'));
});
