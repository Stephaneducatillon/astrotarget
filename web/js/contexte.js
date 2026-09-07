/**
 * Contexte de week-end : ce que toutes les vues partagent.
 *
 * Chaîne de chargement : calendrier → GP courant → fiche circuit → météo.
 * Chaque maillon peut échouer indépendamment ; l'appelant reçoit alors un
 * bloc en erreur explicite plutôt qu'une page vide.
 */

import * as jolpica from './data/jolpica.js';
import * as demo from './data/demo.js';
import * as reference from './data/reference.js';
import { SESSIONS } from './config.js';
import * as store from './store.js';

let promesse = null;

export function saisonCourante() {
  return store.prefs().saison || new Date().getUTCFullYear();
}

export function invalider() { promesse = null; }

export function contexte() {
  if (!promesse) promesse = construire();
  return promesse;
}

async function construire() {
  const saison = saisonCourante();
  const forceDemo = store.prefs().demo;

  let calendrier = null, sourceCal = 'JOL', tsCal = Date.now(), erreurCal = null, perime = false;

  if (forceDemo) {
    calendrier = demo.calendrierDemo(saison);
    sourceCal = 'DEMO';
  } else {
    try {
      const r = await jolpica.calendrier(saison);
      calendrier = r.valeur; tsCal = r.ts; perime = r.perime;
      if (r.perime) erreurCal = r.erreur;
    } catch (e) {
      // Repli : saison précédente, puis jeu de démonstration.
      try {
        const r = await jolpica.calendrier(saison - 1);
        calendrier = r.valeur; tsCal = r.ts; erreurCal = e;
      } catch (e2) {
        calendrier = demo.calendrierDemo(saison);
        sourceCal = 'DEMO'; erreurCal = e2;
      }
    }
  }

  const maintenant = new Date();
  const { courant, precedent, suivant, enWeekEnd } = choisirGP(calendrier, maintenant);
  const fiche = courant ? await ficheCircuit(courant) : null;

  return {
    saison, maintenant,
    calendrier, source_calendrier: sourceCal, ts_calendrier: tsCal,
    calendrier_perime: perime, erreur_calendrier: erreurCal,
    mode_demo: sourceCal === 'DEMO',
    gp: courant, gp_precedent: precedent, gp_suivant: suivant, en_week_end: enWeekEnd,
    circuit: fiche,
  };
}

/** Fiche circuit MAN, complétée par les coordonnées Jolpica si absente. */
async function ficheCircuit(gp) {
  let f = null;
  try { f = await reference.circuit(gp.circuit_id); } catch { f = null; }
  if (f) {
    return { ...f, lat: f.lat ?? gp.lat, lon: f.lon ?? gp.lon, connu: true };
  }
  // Circuit hors référentiel : on n'invente pas les champs MAN.
  return {
    id: gp.circuit_id, nom: gp.circuit_nom, pays: gp.pays, ville: gp.ville,
    lat: gp.lat, lon: gp.lon, tz: null, connu: false,
  };
}

/**
 * Le GP « courant » est celui du week-end en cours, sinon le prochain.
 * Hors week-end de GP, le hub bascule sur « prochain GP dans X jours »
 * + débrief du précédent (livrable 3, règles transverses).
 */
export function choisirGP(calendrier, maintenant = new Date()) {
  const gps = [...(calendrier || [])].sort(
    (a, b) => new Date(a.course_utc || a.debut_utc) - new Date(b.course_utc || b.debut_utc));
  if (!gps.length) return { courant: null, precedent: null, suivant: null, enWeekEnd: false };

  const t = maintenant.getTime();
  const finDe = (gp) => new Date(gp.course_utc || gp.debut_utc).getTime() + 3 * 3600000;
  const debutDe = (gp) => new Date(gp.debut_utc || gp.course_utc).getTime();

  const enCours = gps.find((gp) => t >= debutDe(gp) - 36 * 3600000 && t <= finDe(gp));
  const aVenir = gps.find((gp) => finDe(gp) > t);
  const courant = enCours || aVenir || gps[gps.length - 1];
  const i = gps.indexOf(courant);

  return {
    courant,
    precedent: gps.slice(0, i).reverse().find((gp) => finDe(gp) <= t) || null,
    suivant: gps[i + 1] || null,
    enWeekEnd: Boolean(enCours),
  };
}

/** Prochaine session non commencée, avec repli sur la dernière du week-end. */
export function prochaineSession(gp, maintenant = new Date()) {
  if (!gp?.sessions?.length) return null;
  const t = maintenant.getTime();
  const triees = [...gp.sessions].sort((a, b) => new Date(a.debut_utc) - new Date(b.debut_utc));
  return triees.find((s) => new Date(s.debut_utc).getTime() > t) || null;
}

export function sessionEnCours(gp, maintenant = new Date()) {
  if (!gp?.sessions?.length) return null;
  const t = maintenant.getTime();
  return gp.sessions.find((s) => {
    const d = new Date(s.debut_utc).getTime();
    const duree = s.type === 'R' ? 2.5 * 3600000 : 1.2 * 3600000;
    return t >= d && t <= d + duree;
  }) || null;
}

/** Sessions groupées par jour local, pour l'affichage VEN / SAM / DIM. */
export function sessionsParJour(gp, formateurJour) {
  const groupes = new Map();
  for (const s of [...(gp?.sessions || [])].sort(
    (a, b) => new Date(a.debut_utc) - new Date(b.debut_utc))) {
    const d = new Date(s.debut_utc);
    const cle = formateurJour(d);
    if (!groupes.has(cle)) groupes.set(cle, []);
    groupes.get(cle).push({ ...s, date: d, label: SESSIONS[s.type]?.label || s.type });
  }
  return [...groupes.entries()].map(([jour, sessions]) => ({ jour, sessions }));
}

/** Charge engagés + historique pour le modèle baseline. */
export async function engagesEtHistorique(saison, circuitId, { profond = false } = {}) {
  // Le mode démonstration se lit sur le contexte, pas sur la préférence :
  // le calendrier peut avoir basculé en démonstration tout seul parce que
  // Jolpica était injoignable. Sans ça, on repartirait chercher des engagés
  // sur une source qu'on vient de constater muette.
  const ctx = await contexte();
  if (ctx.mode_demo) {
    return { engages: demo.engagesDemo(), historique: {}, source: 'DEMO', ts: Date.now() };
  }
  const r = await jolpica.engages(saison);
  const engages = r.valeur;
  const historique = {};

  if (profond) {
    // Séquentiel volontairement : Jolpica applique une limite de débit.
    for (const e of engages) {
      historique[e.driver_id] = { saison: [], circuit: [] };
      try {
        const s = await jolpica.resultatsPiloteSaison(saison, e.driver_id);
        historique[e.driver_id].saison = s.valeur;
      } catch { /* pilote sans course cette saison */ }
      if (circuitId) {
        try {
          const c = await jolpica.resultatsPiloteCircuit(e.driver_id, circuitId);
          historique[e.driver_id].circuit = c.valeur;
        } catch { /* jamais couru ici */ }
      }
    }
  }
  return { engages, historique, source: 'JOL', ts: r.ts };
}
