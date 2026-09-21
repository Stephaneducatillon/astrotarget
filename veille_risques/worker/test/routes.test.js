// Controle d acces et CORS de l API. Deux defauts classiques :
// une API ouverte qui expose l etat du foyer, et une API correcte que le
// navigateur interdit de lire faute d en-tetes.

import test from 'node:test';
import assert from 'node:assert/strict';
import worker from '../src/index.js';
import { fakeEnv, mockFetch } from './fake-env.js';

const URL_BASE = 'https://w.test';

test('/health repond sans jeton', async () => {
  const r = await worker.fetch(new Request(`${URL_BASE}/health`), fakeEnv());
  assert.equal(r.status, 200);
});

test('/api/snapshot exige un jeton', async () => {
  const r = await worker.fetch(new Request(`${URL_BASE}/api/snapshot`), fakeEnv());
  assert.equal(r.status, 401);
});

test('un mauvais jeton est refuse', async () => {
  const env = fakeEnv();
  const r = await worker.fetch(
    new Request(`${URL_BASE}/api/snapshot`, { headers: { Authorization: 'Bearer faux' } }), env);
  assert.equal(r.status, 401);
});

test('sans ADMIN_TOKEN configure, tout est ferme', async () => {
  // Un deploiement ou le secret a ete oublie ne doit pas s ouvrir a tous.
  const env = fakeEnv({ ADMIN_TOKEN: undefined });
  const r = await worker.fetch(
    new Request(`${URL_BASE}/api/snapshot`, { headers: { Authorization: 'Bearer ' } }), env);
  assert.equal(r.status, 401);
});

test('le jeton en en-tete donne acces', async () => {
  const env = fakeEnv();
  const r = await worker.fetch(
    new Request(`${URL_BASE}/api/snapshot`, { headers: { Authorization: 'Bearer jeton' } }), env);
  assert.equal(r.status, 200);
  assert.equal((await r.json()).vide, true);
});

test('le jeton en parametre aussi (la PWA installee ne pose pas d en-tete)', async () => {
  const env = fakeEnv();
  const r = await worker.fetch(new Request(`${URL_BASE}/api/snapshot?token=jeton`), env);
  assert.equal(r.status, 200);
});

test('les reponses portent l origine autorisee, jamais *', async () => {
  const env = fakeEnv({ ALLOWED_ORIGIN: 'https://veille.pages.dev' });
  const r = await worker.fetch(new Request(`${URL_BASE}/api/snapshot?token=jeton`), env);
  assert.equal(r.headers.get('Access-Control-Allow-Origin'), 'https://veille.pages.dev');
  assert.notEqual(r.headers.get('Access-Control-Allow-Origin'), '*');
});

test('le preflight OPTIONS repond 204 sans exiger de jeton', async () => {
  const env = fakeEnv({ ALLOWED_ORIGIN: 'https://veille.pages.dev' });
  const r = await worker.fetch(new Request(`${URL_BASE}/api/snapshot`, { method: 'OPTIONS' }), env);
  assert.equal(r.status, 204);
  assert.equal(r.headers.get('Access-Control-Allow-Origin'), 'https://veille.pages.dev');
});

test('l instantane n est jamais mis en cache par un intermediaire', async () => {
  const env = fakeEnv();
  const r = await worker.fetch(new Request(`${URL_BASE}/api/snapshot?token=jeton`), env);
  assert.equal(r.headers.get('Cache-Control'), 'no-store');
});

test('/api/run declenche un cycle et retourne son resultat', async () => {
  const env = fakeEnv();
  mockFetch({
    'mf.test': { product: { periods: [{ timelaps: { domain_ids: [{ domain_id: '59', max_color_id: 1, phenomenon_items: [] }] } }] } },
    'rte.test/token': { access_token: 'x', expires_in: 7200 },
    'rte.test/ecowatt': { signals: [] },
    'vigicrues.test': {},
    'ntfy.test': { ok: true },
  });
  const r = await worker.fetch(
    new Request(`${URL_BASE}/api/run`, { method: 'POST', headers: { Authorization: 'Bearer jeton' } }), env);
  assert.equal(r.status, 200);
  assert.equal((await r.json()).niveau, 'vert');
});

test('une route inconnue repond 404, pas 200', async () => {
  const env = fakeEnv();
  const r = await worker.fetch(new Request(`${URL_BASE}/nimporte?token=jeton`), env);
  assert.equal(r.status, 404);
});
