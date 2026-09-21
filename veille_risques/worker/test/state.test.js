// La machine a etats est la piece dont depend l'utilisabilite du dispositif :
// si elle renotifie, la famille coupe ntfy et la veille ne sert plus a rien.

import test from 'node:test';
import assert from 'node:assert/strict';
import { nextState, niveauGlobal, rang } from '../src/state.js';

const TEMPO = { releves_avant_descente: 3, rappel_secondes: 6 * 3600 };
const T0 = 1_700_000_000;

test('premier passage au vert ne notifie pas', () => {
  const { etat, notification } = nextState(null, { level: 'vert' }, T0, TEMPO);
  assert.equal(etat.level, 'vert');
  assert.equal(notification, null);
});

test('montee vert -> orange notifie immediatement', () => {
  const { etat, notification } = nextState(
    { level: 'vert', since: T0, last_notified: null, below_count: 0 },
    { level: 'orange', label: 'Vent violent (59)' }, T0 + 900, TEMPO,
  );
  assert.equal(etat.level, 'orange');
  assert.equal(etat.since, T0 + 900);
  assert.equal(notification.kind, 'montee');
  assert.equal(notification.priority, 'default');
  assert.equal(notification.label, 'Vent violent (59)');
});

test('montee orange -> rouge notifie en priorite urgente', () => {
  const { notification } = nextState(
    { level: 'orange', since: T0, last_notified: T0, below_count: 0 },
    { level: 'rouge' }, T0 + 900, TEMPO,
  );
  assert.equal(notification.kind, 'montee');
  assert.equal(notification.priority, 'urgent');
});

test('orange qui dure ne renotifie pas avant le plafond de rappel', () => {
  // Le scenario qui tue le dispositif : 12 h d'orange a 15 min = 48 releves.
  let etat = { level: 'orange', label: 'Pluie-inondation (59)', since: T0, last_notified: T0, below_count: 0 };
  let notifications = 0;
  for (let i = 1; i <= 48; i++) {
    const r = nextState(etat, { level: 'orange', label: 'Pluie-inondation (59)' }, T0 + i * 900, TEMPO);
    etat = r.etat;
    if (r.notification) notifications++;
  }
  // 12 h d'episode = exactement 2 rappels (a 6 h et a 12 h), pas 48.
  assert.equal(notifications, 2);
});

test('le rappel repart du dernier envoi, pas du debut de l episode', () => {
  const r1 = nextState(
    { level: 'orange', since: T0, last_notified: T0, below_count: 0 },
    { level: 'orange' }, T0 + 6 * 3600, TEMPO,
  );
  assert.equal(r1.notification.kind, 'rappel');
  assert.equal(r1.etat.last_notified, T0 + 6 * 3600);

  // 15 min plus tard : rien.
  const r2 = nextState(r1.etat, { level: 'orange' }, T0 + 6 * 3600 + 900, TEMPO);
  assert.equal(r2.notification, null);
});

test('descente differee : 2 releves sous le seuil ne suffisent pas', () => {
  let etat = { level: 'orange', since: T0, last_notified: T0, below_count: 0 };
  for (let i = 1; i <= 2; i++) {
    const r = nextState(etat, { level: 'vert' }, T0 + i * 900, TEMPO);
    etat = r.etat;
    assert.equal(r.notification, null, `releve ${i} ne doit rien notifier`);
    assert.equal(etat.level, 'orange', `releve ${i} doit rester orange`);
  }
  assert.equal(etat.below_count, 2);
});

test('descente confirmee au 3e releve consecutif', () => {
  let etat = { level: 'orange', since: T0, last_notified: T0, below_count: 0 };
  let derniere = null;
  for (let i = 1; i <= 3; i++) {
    const r = nextState(etat, { level: 'vert' }, T0 + i * 900, TEMPO);
    etat = r.etat;
    derniere = r.notification;
  }
  assert.equal(etat.level, 'vert');
  assert.equal(etat.below_count, 0);
  assert.equal(derniere.kind, 'fin');
  assert.equal(derniere.priority, 'min');
});

test('une oscillation autour du seuil ne fait pas clignoter le telephone', () => {
  // Station qui repasse au-dessus du seuil avant la confirmation : le
  // compteur doit se remettre a zero, pas s accumuler.
  let etat = { level: 'orange', since: T0, last_notified: T0, below_count: 0 };
  let notifications = 0;
  const suite = ['vert', 'vert', 'orange', 'vert', 'vert', 'orange', 'vert', 'vert'];
  suite.forEach((niveau, i) => {
    const r = nextState(etat, { level: niveau }, T0 + (i + 1) * 900, TEMPO);
    etat = r.etat;
    if (r.notification) notifications++;
  });
  assert.equal(etat.level, 'orange', 'jamais confirme trois fois de suite');
  assert.equal(notifications, 0, 'aucune notification pour une oscillation');
});

test('desescalade rouge -> orange est notifiee en priorite basse', () => {
  let etat = { level: 'rouge', since: T0, last_notified: T0, below_count: 0 };
  let derniere = null;
  for (let i = 1; i <= 3; i++) {
    const r = nextState(etat, { level: 'orange' }, T0 + i * 900, TEMPO);
    etat = r.etat;
    derniere = r.notification;
  }
  assert.equal(etat.level, 'orange');
  assert.equal(derniere.kind, 'desescalade');
  assert.equal(derniere.priority, 'low');
});

test('une remontee pendant la descente est immediate et annule le compteur', () => {
  let etat = { level: 'orange', since: T0, last_notified: T0, below_count: 0 };
  etat = nextState(etat, { level: 'vert' }, T0 + 900, TEMPO).etat;
  etat = nextState(etat, { level: 'vert' }, T0 + 1800, TEMPO).etat;
  assert.equal(etat.below_count, 2);

  const r = nextState(etat, { level: 'rouge' }, T0 + 2700, TEMPO);
  assert.equal(r.etat.level, 'rouge');
  assert.equal(r.etat.below_count, 0);
  assert.equal(r.notification.priority, 'urgent');
});

test('le niveau global est le plus eleve des regles', () => {
  assert.equal(niveauGlobal([{ level: 'vert' }, { level: 'orange' }, { level: 'vert' }]), 'orange');
  assert.equal(niveauGlobal([{ level: 'orange' }, { level: 'rouge' }]), 'rouge');
  assert.equal(niveauGlobal([]), 'vert');
});

test('rang traite un niveau inconnu comme vert plutot que de planter', () => {
  assert.equal(rang('bleu'), 0);
  assert.equal(rang(undefined), 0);
});
