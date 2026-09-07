/**
 * Source OF1 — OpenF1. Préséance sur le live (livrable 1 §1).
 * Couverture 2023 → aujourd'hui uniquement : toute vue qui l'utilise doit
 * gérer le cas « pas de données avant 2023 ».
 */

import { SOURCES, TTL } from '../config.js';
import { getJSON } from './http.js';
import { avecCache } from './cache.js';

const BASE = SOURCES.OF1.base;

export const COUVERTURE_DEPUIS = 2023;

/** Sessions OpenF1 d'une année — porte `session_key`, clé de jointure du live. */
export function sessions(annee) {
  return avecCache(`of1:sessions:${annee}`, TTL.calendrier, async () => {
    const d = await getJSON('OF1', `${BASE}/sessions?year=${annee}`);
    return (d || []).map((s) => ({
      session_key: s.session_key, meeting_key: s.meeting_key,
      type: versType(s.session_name, s.session_type),
      nom: s.session_name, lieu: s.location, pays: s.country_name,
      circuit_court: s.circuit_short_name,
      debut_utc: s.date_start, fin_utc: s.date_end, annee: s.year,
    }));
  });
}

function versType(nom, type) {
  const n = String(nom || '').toLowerCase();
  if (n.includes('practice 1')) return 'FP1';
  if (n.includes('practice 2')) return 'FP2';
  if (n.includes('practice 3')) return 'FP3';
  if (n.includes('sprint qualifying') || n.includes('sprint shootout')) return 'SQ';
  if (n === 'sprint') return 'SPR';
  if (n.includes('qualifying')) return 'Q';
  if (n === 'race' || String(type).toLowerCase() === 'race') return 'R';
  return 'AUTRE';
}

/** Session OpenF1 correspondant à un GP + un type, si elle existe. */
export async function trouverSession(annee, circuitCourt, type) {
  const { valeur } = await sessions(annee);
  const cible = String(circuitCourt || '').toLowerCase();
  return valeur.find((s) => s.type === type
    && (String(s.circuit_court || '').toLowerCase().includes(cible)
      || String(s.lieu || '').toLowerCase().includes(cible))) || null;
}

/** Messages de direction de course — c'est LA source des alertes (livrable 1 §3). */
export function raceControl(sessionKey) {
  return avecCache(`of1:rc:${sessionKey}`, TTL.race_control, async () => {
    const d = await getJSON('OF1', `${BASE}/race_control?session_key=${sessionKey}`);
    return (d || []).map((m) => ({
      date_utc: m.date, categorie: m.category, drapeau: m.flag,
      message: m.message, scope: m.scope, secteur: m.sector,
      numero_pilote: m.driver_number, tour: m.lap_number,
    })).sort((a, b) => new Date(b.date_utc) - new Date(a.date_utc));
  });
}

/** Conditions réelles mesurées (entité `weather_track`). */
export function meteoPiste(sessionKey) {
  return avecCache(`of1:weather:${sessionKey}`, TTL.live || 60000, async () => {
    const d = await getJSON('OF1', `${BASE}/weather?session_key=${sessionKey}`);
    return (d || []).map((w) => ({
      date_utc: w.date, air_c: w.air_temperature, piste_c: w.track_temperature,
      humidite_pct: w.humidity, pression_mbar: w.pressure,
      vent_ms: w.wind_speed, vent_dir_deg: w.wind_direction, pluie: w.rainfall === 1,
    }));
  });
}

export function pilotesSession(sessionKey) {
  return avecCache(`of1:drivers:${sessionKey}`, TTL.pilotes, async () => {
    const d = await getJSON('OF1', `${BASE}/drivers?session_key=${sessionKey}`);
    return (d || []).map((p) => ({
      numero: p.driver_number, code: p.name_acronym, nom_complet: p.full_name,
      prenom: p.first_name, nom: p.last_name, equipe: p.team_name,
      couleur: p.team_colour ? `#${p.team_colour}` : null,
    }));
  });
}

/**
 * Tours d'une session. `is_valide` est calculé ici (source CALC) :
 * règle métier n°1 du livrable 1 — tout indicateur de rythme se calcule
 * uniquement sur les tours valides.
 */
export function tours(sessionKey, numeroPilote = null) {
  const suffixe = numeroPilote ? `&driver_number=${numeroPilote}` : '';
  return avecCache(`of1:laps:${sessionKey}:${numeroPilote || 'tous'}`, TTL.derives, async () => {
    const d = await getJSON('OF1', `${BASE}/laps?session_key=${sessionKey}${suffixe}`);
    const brut = (d || []).map((l) => ({
      numero_pilote: l.driver_number, numero_tour: l.lap_number,
      duree_ms: l.lap_duration != null ? Math.round(l.lap_duration * 1000) : null,
      s1_ms: msSecteur(l.duration_sector_1), s2_ms: msSecteur(l.duration_sector_2), s3_ms: msSecteur(l.duration_sector_3),
      vitesse_i1: l.i1_speed, vitesse_i2: l.i2_speed, vitesse_st: l.st_speed,
      is_pit_out: Boolean(l.is_pit_out_lap), debut_utc: l.date_start,
    }));
    return marquerValides(brut);
  });
}

function msSecteur(v) { return v != null ? Math.round(v * 1000) : null; }

/**
 * Tour propre : hors pit, hors tour de sortie, et pas aberrant.
 * OpenF1 ne donne pas de `track_status` par tour ; on écarte donc aussi les
 * tours nettement plus lents que la médiane (proxy SC/VSC/drapeau).
 * Cette approximation est signalée dans l'UI — elle n'est pas la règle finale,
 * qui sera calculée par le batch FastF1 de la phase 2.
 */
function marquerValides(laps) {
  const parPilote = new Map();
  for (const l of laps) {
    if (!parPilote.has(l.numero_pilote)) parPilote.set(l.numero_pilote, []);
    parPilote.get(l.numero_pilote).push(l);
  }
  for (const liste of parPilote.values()) {
    const durees = liste.map((l) => l.duree_ms).filter((v) => v > 0).sort((a, b) => a - b);
    const mediane = durees.length ? durees[Math.floor(durees.length / 2)] : null;
    const seuil = mediane ? mediane * 1.07 : null;
    liste.forEach((l, i) => {
      const suivantEstPitIn = false; // OpenF1 ne marque pas l'entrée aux stands sur /laps
      l.is_valide = Boolean(
        l.duree_ms && !l.is_pit_out && !suivantEstPitIn
        && (!seuil || l.duree_ms <= seuil)
        && i > 0,
      );
    });
  }
  return laps;
}

/** Relais : composé, âge du pneu (entité `stint`). */
export function relais(sessionKey) {
  return avecCache(`of1:stints:${sessionKey}`, TTL.derives, async () => {
    const d = await getJSON('OF1', `${BASE}/stints?session_key=${sessionKey}`);
    return (d || []).map((s) => ({
      numero_pilote: s.driver_number, compound: s.compound,
      tour_debut: s.lap_start, tour_fin: s.lap_end,
      age_pneu_depart: s.tyre_age_at_start, numero_relais: s.stint_number,
    }));
  });
}

export function arrets(sessionKey) {
  return avecCache(`of1:pit:${sessionKey}`, TTL.derives, async () => {
    const d = await getJSON('OF1', `${BASE}/pit?session_key=${sessionKey}`);
    return (d || []).map((p) => ({
      numero_pilote: p.driver_number, tour: p.lap_number,
      duree_arret_s: p.pit_duration, date_utc: p.date,
    }));
  });
}
