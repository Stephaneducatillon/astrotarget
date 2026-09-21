// Cycle complet : collecte -> regles -> machine a etats -> notifications ->
// instantane. C'est ici que se logent les erreurs de cablage.

import test from 'node:test';
import assert from 'node:assert/strict';
import { cycleNiveauA } from '../src/index.js';
import { fakeEnv, mockFetch } from './fake-env.js';

const T0 = 1_700_000_000;
const STATIONS = [{ code: 'E1', nom: 'Scarpe a Douai', metric: 'H' }];
const TRONCONS = [{ code: 'XX1', nom: 'Scarpe aval' }];
const CFG = { stations: STATIONS, troncons: TRONCONS };

function routesCalmes(over = {}) {
  return {
    'mf.test': { product: { periods: [{ timelaps: { domain_ids: [
      { domain_id: '59', max_color_id: 1, phenomenon_items: [] },
      { domain_id: '62', max_color_id: 1, phenomenon_items: [] },
    ] } }] } },
    'rte.test/token': { access_token: 'jeton-rte', expires_in: 7200 },
    'rte.test/ecowatt': { signals: [{ jour: '2026-01-14', dvalue: 1, values: [] }] },
    'hubeau.test': { data: [{ date_obs: '2026-01-14T10:00:00Z', resultat_obs: 1200 }] },
    'vigicrues.test': { vicItemInfoVigiCru: [{ vicCdEntCru: 'XX1', vicNivInfoVigiCru: 1 }] },
    'ntfy.test': { ok: true },
    ...over,
  };
}

test('journee calme : aucune notification, instantane vert', async () => {
  const env = fakeEnv();
  const appels = mockFetch(routesCalmes());

  const r = await cycleNiveauA(env, T0, CFG);

  assert.equal(r.niveau, 'vert');
  assert.equal(r.notifications, 0);
  assert.equal(r.degrade, false);
  assert.equal(appels.filter((a) => a.url.includes('ntfy')).length, 0, 'aucun ntfy un jour calme');

  const snap = JSON.parse(env.KV._raw.get('snapshot:latest'));
  assert.equal(snap.niveau, 'vert');
  assert.equal(snap.regles.length, 4);
});

test('une seule ecriture KV par cycle (quota du plan gratuit)', async () => {
  const env = fakeEnv();
  mockFetch(routesCalmes());
  await cycleNiveauA(env, T0, CFG);
  // 1 instantane + 1 jeton RTE + 1 cache ecowatt au premier run.
  // Les runs suivants ne reecrivent que l instantane : 96/jour contre 1000.
  const premier = env.KV._writes();
  await cycleNiveauA(env, T0 + 900, CFG);
  assert.equal(env.KV._writes() - premier, 1, 'un run etabli = une seule ecriture KV');
});

test('le jeton RTE est mis en cache, pas redemande a chaque run', async () => {
  const env = fakeEnv();
  const appels = mockFetch(routesCalmes());
  await cycleNiveauA(env, T0, CFG);
  await cycleNiveauA(env, T0 + 900, CFG);
  await cycleNiveauA(env, T0 + 1800, CFG);
  const auth = appels.filter((a) => a.url.includes('/token')).length;
  assert.equal(auth, 1, `${auth} authentifications RTE pour 3 runs`);
});

test('ecowatt n est pas reinterroge toutes les 15 min', async () => {
  const env = fakeEnv();
  const appels = mockFetch(routesCalmes());
  for (let i = 0; i < 8; i++) await cycleNiveauA(env, T0 + i * 900, CFG);
  const n = appels.filter((a) => a.url.includes('ecowatt')).length;
  assert.equal(n, 1, `${n} appels Ecowatt pour 2 h de runs (cache 3 h)`);
});

test('vigilance orange : une notification, puis plus rien', async () => {
  const env = fakeEnv();
  const orange = routesCalmes({
    'mf.test': { product: { periods: [{ timelaps: { domain_ids: [
      { domain_id: '59', max_color_id: 3, phenomenon_items: [{ phenomenon_id: '2', phenomenon_max_color_id: 3 }] },
    ] } }] } },
  });
  const appels = mockFetch(orange);

  const r1 = await cycleNiveauA(env, T0, CFG);
  assert.equal(r1.niveau, 'orange');
  assert.equal(r1.notifications, 1);

  // 3 h d episode = 12 runs supplementaires, aucune notification de plus
  // (le plafond de rappel est a 6 h).
  let suite = 0;
  for (let i = 1; i <= 12; i++) {
    suite += (await cycleNiveauA(env, T0 + i * 900, CFG)).notifications;
  }
  assert.equal(suite, 0);
  assert.equal(appels.filter((a) => a.url.includes('ntfy')).length, 1);

  const log = env.DB._tables.alert_log;
  assert.equal(log.length, 1);
  assert.equal(log[0].rule_id, 'mf_vigilance');
  assert.equal(log[0].to_level, 'orange');
});

test('la notification ne contient aucune adresse', async () => {
  const env = fakeEnv();
  const corps = [];
  globalThis.fetch = (() => {
    const base = mockFetch(routesCalmes({
      'mf.test': { product: { periods: [{ timelaps: { domain_ids: [
        { domain_id: '59', max_color_id: 4, phenomenon_items: [{ phenomenon_id: '1', phenomenon_max_color_id: 4 }] },
      ] } }] } },
    }));
    const inner = globalThis.fetch;
    return async (url, init) => {
      if (String(url).includes('ntfy')) corps.push({ headers: init.headers, body: init.body });
      return inner(url, init);
    };
  })();

  await cycleNiveauA(env, T0, CFG);
  assert.equal(corps.length, 1);
  assert.equal(corps[0].headers.Priority, 'urgent', 'un rouge part en urgent');
  // Le titre transite en en-tete HTTP : ASCII uniquement.
  assert.match(corps[0].headers.Title, /^[\x20-\x7E]*$/);
  assert.match(corps[0].body, /Checklist ROUGE/);
});

test('source en panne : etat preserve, alerte de panne au 3e echec', async () => {
  const env = fakeEnv();

  // D abord un episode orange etabli.
  mockFetch(routesCalmes({
    'mf.test': { product: { periods: [{ timelaps: { domain_ids: [
      { domain_id: '59', max_color_id: 3, phenomenon_items: [{ phenomenon_id: '2', phenomenon_max_color_id: 3 }] },
    ] } }] } },
  }));
  await cycleNiveauA(env, T0, CFG);
  assert.equal(env.DB._tables.alert_state.get('mf_vigilance').level, 'orange');

  // Meteo-France tombe. L etat orange ne doit PAS retomber au vert.
  mockFetch(routesCalmes({ 'mf.test': { status: 500, body: { erreur: 'boom' } } }));
  let notifs = 0;
  for (let i = 1; i <= 3; i++) {
    notifs += (await cycleNiveauA(env, T0 + i * 900, CFG)).notifications;
  }
  assert.equal(env.DB._tables.alert_state.get('mf_vigilance').level, 'orange',
    'une source muette ne doit jamais faire retomber une alerte');
  assert.equal(notifs, 1, 'une seule alerte de panne, au 3e echec');
  assert.equal(env.DB._tables.source_health.get('vigilance').fail_streak, 3);

  const snap = JSON.parse(env.KV._raw.get('snapshot:latest'));
  assert.equal(snap.degrade, true);
  assert.equal(snap.regles.find((r) => r.rule_id === 'mf_vigilance').indetermine, true);
});

test('une cle refusee alerte immediatement', async () => {
  const env = fakeEnv();
  mockFetch(routesCalmes({ 'mf.test': { status: 401, body: {} } }));
  const r = await cycleNiveauA(env, T0, CFG);
  assert.equal(r.notifications, 1);
  assert.match(env.DB._tables.source_health.get('vigilance').last_error, /authentification/);
});

test('la mesure du run n entre pas dans son propre maximum', async () => {
  // Piege classique : inserer avant de calculer la fenetre rend le seuil
  // indepassable par construction, et la regle Scarpe ne se declenche jamais.
  const env = fakeEnv();

  // Historique : 300 mesures a 1200 mm.
  for (let i = 0; i < 300; i++) {
    env.DB._tables.observations.push({
      _k: `h${i}`, source_id: 'hubeau', station: 'E1', metric: 'H',
      ts: T0 - (300 - i) * 900, value: 1200,
    });
  }

  mockFetch(routesCalmes({
    'hubeau.test': { data: [{ date_obs: new Date(T0 * 1000).toISOString(), resultat_obs: 1500 }] },
  }));

  const r = await cycleNiveauA(env, T0, CFG);
  assert.equal(r.niveau, 'orange', '1500 mm depasse le maximum de 1200 mm');

  const etat = env.DB._tables.alert_state.get('scarpe_niveau');
  assert.equal(etat.level, 'orange');
  assert.match(etat.label, /1\.50 m/);
  assert.match(etat.label, /1\.20 m/);

  // Et la mesure a bien ete persistee.
  assert.ok(env.DB._tables.observations.some((o) => o.value === 1500));
});

test('la fenetre est lue avant l ecriture des observations', async () => {
  const env = fakeEnv();
  mockFetch(routesCalmes());
  await cycleNiveauA(env, T0, CFG);
  const j = env.DB._journal;
  const lecture = j.indexOf('read:fenetre');
  const ecriture = j.indexOf('write:observations');
  assert.ok(lecture !== -1 && ecriture !== -1);
  assert.ok(lecture < ecriture, 'la fenetre doit etre calculee avant insertion');
});

test('les regles sont evaluees meme si deux sources sur quatre tombent', async () => {
  const env = fakeEnv();
  mockFetch(routesCalmes({
    'hubeau.test': { status: 503, body: {} },
    'vigicrues.test': { status: 503, body: {} },
    'rte.test/ecowatt': { signals: [{ jour: '2026-01-14', dvalue: 3, values: [{ pas: 19, hvalue: 3 }] }] },
  }));
  const r = await cycleNiveauA(env, T0, CFG);
  assert.equal(r.niveau, 'rouge', 'Ecowatt rouge remonte malgre deux sources mortes');
  assert.equal(r.degrade, true);
});
