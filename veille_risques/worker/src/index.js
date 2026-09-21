// Point d'entree du Worker.
//
// Deux responsabilites :
//   scheduled() - collecte, evaluation, notifications, ecriture de l'instantane
//   fetch()     - API lue par la PWA + endpoints d'administration
//
// Discipline de quota : une seule ecriture KV par run, ecritures D1 groupees
// en un batch, purge quotidienne.

import { SOURCES, SEUILS, TEMPO } from './config.js';
import { collectAll } from './lib/http.js';
import * as store from './lib/store.js';
import { majSante, fraicheur } from './lib/health.js';
import { nextState, niveauGlobal } from './state.js';
import { evaluerRegles } from './rules.js';
import * as notify from './notify.js';
import { collectVigilance } from './sources/vigilance.js';
import { collectEcowatt } from './sources/ecowatt.js';
import { collectHubeau } from './sources/hubeau.js';
import { collectVigicrues } from './sources/vigicrues.js';
import { routeAdmin } from './admin.js';
import { json, preflight } from './lib/respond.js';

export default {
  async scheduled(event, env, ctx) {
    const now = Math.floor(Date.now() / 1000);
    if (event.cron === '30 5 * * *') {
      ctx.waitUntil(maintenance(env, now));
    } else {
      ctx.waitUntil(cycleNiveauA(env, now));
    }
  },

  async fetch(request, env) {
    const url = new URL(request.url);

    if (request.method === 'OPTIONS') return preflight(env);

    if (url.pathname === '/health') {
      return json({ ok: true, ts: Math.floor(Date.now() / 1000) }, 200, env);
    }

    // Tout le reste expose l'etat du foyer : jamais en acces libre.
    // En production, doubler d'un Cloudflare Access devant Pages.
    if (!autorise(request, url, env)) {
      return json({ error: 'non autorise' }, 401, env);
    }

    if (url.pathname === '/api/snapshot') {
      const snap = await store.lireSnapshot(env);
      return json(snap ?? { vide: true, message: 'aucune collecte effectuee pour le moment' }, 200, env);
    }

    if (url.pathname === '/api/serie') {
      const station = url.searchParams.get('station');
      const metric = url.searchParams.get('metric') ?? 'H';
      if (!station) return json({ error: 'parametre station manquant' }, 400, env);
      const now = Math.floor(Date.now() / 1000);
      const points = await store.serie(env, station, metric, now, 7);
      return json({ station, metric, points }, 200, env);
    }

    // Declenchement manuel, utile pour la mise au point et apres une panne.
    if (url.pathname === '/api/run' && request.method === 'POST') {
      const r = await cycleNiveauA(env, Math.floor(Date.now() / 1000));
      return json(r, 200, env);
    }

    if (url.pathname.startsWith('/admin/')) {
      return routeAdmin(url, env);
    }

    return json({ error: 'route inconnue' }, 404, env);
  },
};

// ---------------------------------------------------------------------------
// Cycle principal
// ---------------------------------------------------------------------------

export async function cycleNiveauA(env, now, cfg = {}) {
  // 1. Collecte. Les quatre sources sont independantes : une panne de l'une
  //    ne doit pas priver des trois autres.
  const resultats = await collectAll([
    { id: 'vigilance', run: () => collectVigilance(env) },
    { id: 'ecowatt',   run: () => collectEcowatt(env, now) },
    { id: 'hubeau',    run: () => collectHubeau(env, now, cfg.stations) },
    { id: 'vigicrues', run: () => collectVigicrues(env, cfg.troncons) },
  ]);
  const collectes = new Map(resultats.map((r) => [r.id, r]));

  // 2. Fenetres glissantes AVANT insertion des mesures du run : sinon la
  //    mesure courante entre dans son propre maximum et le seuil devient
  //    indepassable par construction.
  const fenetres = new Map();
  const hubeau = collectes.get('hubeau');
  if (hubeau?.ok) {
    for (const st of hubeau.data.stations ?? []) {
      const f = await store.fenetreGlissante(env, st.code, st.metric, now, SEUILS.hydro_fenetre_jours);
      fenetres.set(`${st.code}:${st.metric}`, f);
    }
  }

  // 3. Evaluation des regles.
  const evaluations = evaluerRegles(collectes, fenetres);

  // 4. Machine a etats : on ne retient que les transitions.
  const etats = await store.lireEtats(env);
  const ecritures = [];
  const aNotifier = [];
  const etatsFinaux = [];

  for (const ev of evaluations) {
    const precedent = etats.get(ev.rule_id) ?? null;

    if (ev.indetermine) {
      // Source muette : on preserve le dernier etat connu et on le marque.
      // La sante de la source prend le relais pour alerter.
      etatsFinaux.push({
        rule_id: ev.rule_id, nom: ev.nom, indetermine: true,
        level: precedent?.level ?? 'vert', label: precedent?.label ?? null,
        since: precedent?.since ?? now,
      });
      continue;
    }

    const { etat, notification } = nextState(precedent, { level: ev.level, label: ev.label }, now, TEMPO);
    ecritures.push(store.upsertEtat(env, ev.rule_id, etat));
    etatsFinaux.push({ rule_id: ev.rule_id, nom: ev.nom, indetermine: false, note: ev.note ?? null, ...etat });

    if (notification) {
      const msg = notify.composer(ev.nom, notification);
      aNotifier.push({ ...msg, priority: notification.priority });
      ecritures.push(store.journaliserAlerte(env, now, ev.rule_id, notification, msg.title, msg.body));
    }
  }

  // 5. Sante des sources. Une source de niveau A muette est elle-meme une alerte.
  const sante = await store.lireSante(env);
  const santeFinale = {};
  for (const r of resultats) {
    const def = SOURCES.find((s) => s.id === r.id);
    const { sante: s, incident } = majSante(sante.get(r.id), r, now);
    ecritures.push(store.upsertSante(env, r.id, s));
    santeFinale[r.id] = { nom: def?.nom ?? r.id, ...s, ...fraicheur(r.id, s, now) };
    if (incident && def?.niveau === 'A') {
      aNotifier.push({ ...notify.composerIncident(def.nom, incident), priority: 'low' });
    }
  }

  // 6. Mesures du run.
  if (hubeau?.ok) {
    for (const st of hubeau.data.stations ?? []) {
      ecritures.push(store.insererObservation(env, 'hubeau', st.code, st.metric, st.ts, st.value));
    }
  }

  if (ecritures.length) await env.DB.batch(ecritures);

  // 7. Notifications, apres la persistance : mieux vaut une notification
  //    perdue qu'un etat incoherent qui renotifiera en boucle.
  const envois = [];
  for (const m of aNotifier) envois.push(await notify.envoyer(env, m));

  // 8. Instantane : UNE seule ecriture KV.
  const snapshot = {
    ts: now,
    niveau: niveauGlobal(etatsFinaux.filter((e) => !e.indetermine)),
    regles: etatsFinaux,
    sources: santeFinale,
    ecowatt: collectes.get('ecowatt')?.ok ? collectes.get('ecowatt').data.jours : null,
    vigilance: collectes.get('vigilance')?.ok ? collectes.get('vigilance').data.departements : null,
    stations: hubeau?.ok ? hubeau.data.stations : null,
    degrade: resultats.some((r) => !r.ok),
  };
  await store.ecrireSnapshot(env, snapshot);

  return { ts: now, niveau: snapshot.niveau, notifications: envois.length, degrade: snapshot.degrade };
}

async function maintenance(env, now) {
  const [obs, logs] = await store.purger(env, now);
  const sante = await store.lireSante(env);
  const muettes = [];
  for (const [id, s] of sante) {
    const def = SOURCES.find((x) => x.id === id);
    const f = fraicheur(id, s, now);
    if (def?.niveau === 'A' && f.perimee) muettes.push(def.nom);
  }
  if (muettes.length) {
    await notify.envoyer(env, {
      title: 'Rapport quotidien — sources en defaut',
      body: `Sans donnees fraiches : ${muettes.join(', ')}.`,
      tags: ['warning'],
      priority: 'low',
    });
  }
  return { purge: { observations: obs, alertes: logs }, muettes };
}

// ---------------------------------------------------------------------------

function autorise(request, url, env) {
  if (!env.ADMIN_TOKEN) return false; // pas de jeton configure = tout ferme
  const entete = request.headers.get('Authorization');
  if (entete === `Bearer ${env.ADMIN_TOKEN}`) return true;
  // La PWA installee ne peut pas poser d'en-tete sur sa requete de demarrage.
  return url.searchParams.get('token') === env.ADMIN_TOKEN;
}

