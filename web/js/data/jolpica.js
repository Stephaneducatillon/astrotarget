/**
 * Source JOL — Jolpica-F1 (successeur d'Ergast).
 * Préséance : historique et classements officiels (livrable 1 §1).
 *
 * Rôle du module : traduire la forme Ergast en entités du livrable 1
 * (`gp`, `session`, `entry`, `resultat`). Les vues ne voient jamais `MRData`.
 */

import { SOURCES, TTL } from '../config.js';
import { getJSON } from './http.js';
import { avecCache } from './cache.js';

const BASE = SOURCES.JOL.base;

/** Ergast donne date et heure séparées, l'heure porte déjà le Z. */
function versUTC(date, time) {
  if (!date) return null;
  if (!time) return `${date}T00:00:00Z`;      // heure inconnue : minuit UTC, signalé par heure_connue=false
  return `${date}T${time.endsWith('Z') ? time : `${time}Z`}`;
}

const TYPES_SESSION = [
  ['FirstPractice', 'FP1'], ['SecondPractice', 'FP2'], ['ThirdPractice', 'FP3'],
  ['SprintQualifying', 'SQ'], ['SprintShootout', 'SQ'], ['Sprint', 'SPR'], ['Qualifying', 'Q'],
];

function normaliserGP(r) {
  const sessions = [];
  for (const [cle, type] of TYPES_SESSION) {
    const s = r[cle];
    if (!s || !s.date) continue;
    if (sessions.some((x) => x.type === type)) continue; // SprintQualifying/Shootout : un seul
    sessions.push({ type, debut_utc: versUTC(s.date, s.time), heure_connue: Boolean(s.time) });
  }
  sessions.push({ type: 'R', debut_utc: versUTC(r.date, r.time), heure_connue: Boolean(r.time) });
  sessions.sort((a, b) => new Date(a.debut_utc) - new Date(b.debut_utc));

  const c = r.Circuit || {};
  const loc = c.Location || {};
  return {
    id: `${r.season}-${r.round}`,
    saison: Number(r.season),
    round: Number(r.round),
    nom: r.raceName,
    format: sessions.some((s) => s.type === 'SPR') ? 'sprint' : 'classique',
    circuit_id: c.circuitId,
    circuit_nom: c.circuitName,
    pays: loc.country,
    ville: loc.locality,
    lat: loc.lat !== undefined ? Number(loc.lat) : null,
    lon: loc.long !== undefined ? Number(loc.long) : null,
    sessions,
    debut_utc: sessions[0]?.debut_utc || null,
    course_utc: versUTC(r.date, r.time),
    source: 'JOL',
  };
}

/** Calendrier d'une saison, sessions incluses. */
export function calendrier(saison) {
  return avecCache(`jol:cal:${saison}`, TTL.calendrier, async () => {
    const d = await getJSON('JOL', `${BASE}/${saison}.json?limit=100`);
    const races = d?.MRData?.RaceTable?.Races || [];
    if (!races.length) throw new Error(`Aucun Grand Prix pour la saison ${saison}.`);
    return races.map(normaliserGP);
  });
}

/** `entry` = pilote + écurie + saison + numéro (livrable 1 §2). */
export function engages(saison) {
  return avecCache(`jol:entries:${saison}`, TTL.pilotes, async () => {
    const d = await getJSON('JOL', `${BASE}/${saison}/driverStandings.json?limit=100`);
    const liste = d?.MRData?.StandingsTable?.StandingsLists?.[0]?.DriverStandings || [];
    if (liste.length) return liste.map((s) => versEntry(s.Driver, s.Constructors?.[0], s));
    // Début de saison : le classement n'existe pas encore, on retombe sur les pilotes.
    const dd = await getJSON('JOL', `${BASE}/${saison}/drivers.json?limit=100`);
    const pilotes = dd?.MRData?.DriverTable?.Drivers || [];
    if (!pilotes.length) throw new Error(`Aucun engagé connu pour ${saison}.`);
    return pilotes.map((p) => versEntry(p, null, null));
  });
}

function versEntry(driver, constructor, standing) {
  return {
    driver_id: driver.driverId,
    numero: driver.permanentNumber ? Number(driver.permanentNumber) : null,
    code: driver.code || (driver.familyName || '').slice(0, 3).toUpperCase(),
    prenom: driver.givenName,
    nom: driver.familyName,
    nationalite: driver.nationality,
    date_naissance: driver.dateOfBirth,
    team_id: constructor?.constructorId || null,
    team_nom: constructor?.name || null,
    points: standing ? Number(standing.points) : null,
    position_championnat: standing ? Number(standing.position) : null,
    victoires: standing ? Number(standing.wins) : null,
    source: 'JOL',
  };
}

export function classementPilotes(saison) {
  return avecCache(`jol:wdc:${saison}`, TTL.classement, async () => {
    const d = await getJSON('JOL', `${BASE}/${saison}/driverStandings.json?limit=100`);
    const l = d?.MRData?.StandingsTable?.StandingsLists?.[0];
    return {
      round: l ? Number(l.round) : null,
      lignes: (l?.DriverStandings || []).map((s) => ({
        position: Number(s.position), points: Number(s.points), victoires: Number(s.wins),
        driver_id: s.Driver.driverId, prenom: s.Driver.givenName, nom: s.Driver.familyName,
        code: s.Driver.code, team_id: s.Constructors?.[0]?.constructorId || null,
        team_nom: s.Constructors?.[0]?.name || null,
      })),
    };
  });
}

export function classementEcuries(saison) {
  return avecCache(`jol:wcc:${saison}`, TTL.classement, async () => {
    const d = await getJSON('JOL', `${BASE}/${saison}/constructorStandings.json?limit=100`);
    const l = d?.MRData?.StandingsTable?.StandingsLists?.[0];
    return (l?.ConstructorStandings || []).map((s) => ({
      position: Number(s.position), points: Number(s.points), victoires: Number(s.wins),
      team_id: s.Constructor.constructorId, team_nom: s.Constructor.name,
    }));
  });
}

/** Résultats de course d'un GP. */
export function resultatsCourse(saison, round) {
  return avecCache(`jol:res:${saison}:${round}`, TTL.resultats, async () => {
    const d = await getJSON('JOL', `${BASE}/${saison}/${round}/results.json?limit=100`);
    const race = d?.MRData?.RaceTable?.Races?.[0];
    if (!race) return null;
    return {
      gp_nom: race.raceName,
      round: Number(race.round),
      saison: Number(race.season),
      tours_prevus: null,
      lignes: (race.Results || []).map((r) => ({
        position: Number(r.position),
        position_texte: r.positionText,
        classe: /^\d+$/.test(r.positionText),
        driver_id: r.Driver.driverId,
        code: r.Driver.code,
        prenom: r.Driver.givenName,
        nom: r.Driver.familyName,
        team_id: r.Constructor.constructorId,
        team_nom: r.Constructor.name,
        numero: r.number ? Number(r.number) : null,
        grille: Number(r.grid),
        tours: Number(r.laps),
        points: Number(r.points),
        statut: r.status,
        abandon: !/^\d+$/.test(r.positionText) || /Accident|Collision|Retired|Engine|Gearbox|Hydraulics|Power Unit|Withdrew|Disqualified/i.test(r.status),
        temps: r.Time?.time || null,
        meilleur_tour: r.FastestLap
          ? { rang: Number(r.FastestLap.rank), tour: Number(r.FastestLap.lap), temps: r.FastestLap.Time?.time }
          : null,
      })),
    };
  });
}

/** Résultats de qualification d'un GP. */
export function resultatsQualif(saison, round) {
  return avecCache(`jol:qual:${saison}:${round}`, TTL.resultats, async () => {
    const d = await getJSON('JOL', `${BASE}/${saison}/${round}/qualifying.json?limit=100`);
    const race = d?.MRData?.RaceTable?.Races?.[0];
    if (!race) return null;
    return {
      gp_nom: race.raceName,
      lignes: (race.QualifyingResults || []).map((r) => ({
        position: Number(r.position),
        driver_id: r.Driver.driverId, code: r.Driver.code,
        prenom: r.Driver.givenName, nom: r.Driver.familyName,
        team_id: r.Constructor.constructorId, team_nom: r.Constructor.name,
        q1: r.Q1 || null, q2: r.Q2 || null, q3: r.Q3 || null,
        meilleur: r.Q3 || r.Q2 || r.Q1 || null,
      })),
    };
  });
}

/** Historique d'un pilote sur un circuit — alimente l'onglet « Ce circuit ». */
export function resultatsPiloteCircuit(driverId, circuitId, limite = 12) {
  return avecCache(`jol:pc:${driverId}:${circuitId}`, TTL.derives, async () => {
    const d = await getJSON('JOL', `${BASE}/drivers/${driverId}/circuits/${circuitId}/results.json?limit=${limite}`);
    const races = d?.MRData?.RaceTable?.Races || [];
    return races.map((r) => ({
      saison: Number(r.season), gp_nom: r.raceName,
      position: Number(r.Results?.[0]?.position),
      position_texte: r.Results?.[0]?.positionText,
      grille: Number(r.Results?.[0]?.grid),
      statut: r.Results?.[0]?.status,
      points: Number(r.Results?.[0]?.points),
      team_id: r.Results?.[0]?.Constructor?.constructorId,
    }));
  });
}

/** N derniers résultats d'un pilote sur la saison — alimente `forme_pilote`. */
export function resultatsPiloteSaison(saison, driverId) {
  return avecCache(`jol:ps:${saison}:${driverId}`, TTL.derives, async () => {
    const d = await getJSON('JOL', `${BASE}/${saison}/drivers/${driverId}/results.json?limit=40`);
    const races = d?.MRData?.RaceTable?.Races || [];
    return races.map((r) => ({
      round: Number(r.round), gp_nom: r.raceName, circuit_id: r.Circuit?.circuitId,
      position: Number(r.Results?.[0]?.position),
      position_texte: r.Results?.[0]?.positionText,
      grille: Number(r.Results?.[0]?.grid),
      points: Number(r.Results?.[0]?.points),
      statut: r.Results?.[0]?.status,
      team_id: r.Results?.[0]?.Constructor?.constructorId,
    }));
  });
}

/** Toutes les courses d'une saison avec résultats — socle des indicateurs. */
export function resultatsSaison(saison) {
  return avecCache(`jol:saison:${saison}`, TTL.derives, async () => {
    const d = await getJSON('JOL', `${BASE}/${saison}/results.json?limit=1000`);
    const races = d?.MRData?.RaceTable?.Races || [];
    return races.map((r) => ({
      round: Number(r.round), gp_nom: r.raceName, circuit_id: r.Circuit?.circuitId, date: r.date,
      lignes: (r.Results || []).map((x) => ({
        position: Number(x.position), position_texte: x.positionText,
        driver_id: x.Driver.driverId, team_id: x.Constructor.constructorId,
        grille: Number(x.grid), points: Number(x.points), statut: x.status,
        tours: Number(x.laps),
        abandon: !/^\d+$/.test(x.positionText),
      })),
    }));
  });
}
