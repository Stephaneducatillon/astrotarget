// Sante des sources : le scenario ou une source meurt en silence.

import test from 'node:test';
import assert from 'node:assert/strict';
import { majSante, fraicheur } from '../src/lib/health.js';
import { SourceError } from '../src/lib/http.js';
import { evaluerRegles } from '../src/rules.js';

const T0 = 1_700_000_000;

test('un succes remet le compteur a zero', () => {
  const { sante, incident } = majSante({ last_ok: T0 - 9000, fail_streak: 2, notified: 0 }, { ok: true }, T0);
  assert.equal(sante.fail_streak, 0);
  assert.equal(sante.last_ok, T0);
  assert.equal(incident, null);
});

test('deux echecs ne declenchent rien, le troisieme si', () => {
  let s = null, incident = null;
  const echec = { ok: false, error: new SourceError('http', 'HTTP 503') };
  for (let i = 1; i <= 3; i++) {
    const r = majSante(s, echec, T0 + i * 900);
    s = r.sante;
    incident = r.incident;
    if (i < 3) assert.equal(incident, null, `echec ${i} ne doit rien notifier`);
  }
  assert.equal(s.fail_streak, 3);
  assert.equal(incident.type, 'muette');
});

test('une source deja signalee ne renotifie pas a chaque run', () => {
  let s = { last_ok: T0, fail_streak: 3, notified: 1 };
  for (let i = 1; i <= 20; i++) {
    const r = majSante(s, { ok: false, error: new SourceError('http', 'HTTP 503') }, T0 + i * 900);
    s = r.sante;
    assert.equal(r.incident, null);
  }
  assert.equal(s.fail_streak, 23);
});

test('une cle refusee alerte des le premier echec', () => {
  // 401/403 ne se resoudra pas tout seul : temporiser 45 min n a aucun sens.
  const err = new SourceError('auth', 'authentification refusee (401)', 401);
  const { incident } = majSante(null, { ok: false, error: err }, T0);
  assert.equal(incident.type, 'auth');
});

test('le retablissement est notifie une fois', () => {
  const r1 = majSante({ last_ok: T0 - 9000, fail_streak: 5, notified: 1 }, { ok: true }, T0);
  assert.equal(r1.incident.type, 'retablie');
  const r2 = majSante(r1.sante, { ok: true }, T0 + 900);
  assert.equal(r2.incident, null);
});

test('fraicheur : au-dela de 3x la frequence, la donnee est perimee', () => {
  assert.equal(fraicheur('vigilance', { last_ok: T0 - 900 }, T0).perimee, false);
  assert.equal(fraicheur('vigilance', { last_ok: T0 - 3000 }, T0).perimee, true);
  assert.equal(fraicheur('vigilance', { last_ok: null }, T0).jamais, true);
});

// --- Regles -----------------------------------------------------------------

test('une source en echec est INDETERMINEE, jamais verte', () => {
  // Le point le plus important du moteur : confondre "pas d alerte" et
  // "pas de donnee" rend la veille dangereuse.
  const collectes = new Map([
    ['vigilance', { ok: false, error: new SourceError('http', 'HTTP 500') }],
    ['ecowatt',   { ok: true, data: { pire: 'vert', libelle: null, jours: [] } }],
    ['hubeau',    { ok: true, data: { configure: false, stations: [] } }],
    ['vigicrues', { ok: true, data: { pire: 'vert', libelle: null, troncons: [] } }],
  ]);
  const r = evaluerRegles(collectes, new Map());
  const vig = r.find((x) => x.rule_id === 'mf_vigilance');
  assert.equal(vig.indetermine, true);
  assert.equal(vig.level, null, 'ne doit surtout pas valoir "vert"');

  const eco = r.find((x) => x.rule_id === 'ecowatt');
  assert.equal(eco.indetermine, false);
  assert.equal(eco.level, 'vert');
});

test('regle Scarpe : depassement du maximum sur 30 jours', () => {
  const collectes = new Map([
    ['vigilance', { ok: true, data: { pire: 'vert', libelle: null, departements: {} } }],
    ['ecowatt',   { ok: true, data: { pire: 'vert', libelle: null, jours: [] } }],
    ['hubeau',    { ok: true, data: { configure: true, stations: [
      { code: 'E1', nom: 'Scarpe a Douai', metric: 'H', value: 1800, ts: T0 },
    ] } }],
    ['vigicrues', { ok: true, data: { pire: 'vert', libelle: null, troncons: [] } }],
  ]);
  const fenetres = new Map([['E1:H', { max: 1500, n: 2500 }]]);
  const r = evaluerRegles(collectes, fenetres).find((x) => x.rule_id === 'scarpe_niveau');
  assert.equal(r.level, 'orange');
  assert.match(r.label, /1\.80 m/);
  assert.match(r.label, /1\.50 m/);
});

test('regle Scarpe : historique insuffisant reste vert et le signale', () => {
  const collectes = new Map([
    ['vigilance', { ok: true, data: { pire: 'vert', libelle: null, departements: {} } }],
    ['ecowatt',   { ok: true, data: { pire: 'vert', libelle: null, jours: [] } }],
    ['hubeau',    { ok: true, data: { configure: true, stations: [
      { code: 'E1', nom: 'Scarpe a Douai', metric: 'H', value: 9999, ts: T0 },
    ] } }],
    ['vigicrues', { ok: true, data: { pire: 'vert', libelle: null, troncons: [] } }],
  ]);
  const r = evaluerRegles(collectes, new Map([['E1:H', { max: 100, n: 5 }]]))
    .find((x) => x.rule_id === 'scarpe_niveau');
  assert.equal(r.level, 'vert');
  assert.equal(r.note, 'historique insuffisant');
});
