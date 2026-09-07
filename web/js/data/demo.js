/**
 * Jeu de démonstration local.
 *
 * Raison d'être : les sources publiques peuvent être injoignables (CORS d'un
 * navigateur, limite de débit, coupure). Plutôt qu'une page blanche — interdite
 * par le livrable 3 — le site bascule sur ce jeu, avec un bandeau permanent.
 *
 * Rien ici n'est une donnée officielle. Le calendrier est calé sur le week-end
 * à venir, les résultats sont générés par un tirage déterministe. C'est un
 * décor fonctionnel, pas un enregistrement.
 */

const ENGAGES = [
  ['verstappen', 1, 'VER', 'Max', 'Verstappen', 'red_bull', 'Red Bull Racing'],
  ['lawson', 30, 'LAW', 'Liam', 'Lawson', 'red_bull', 'Red Bull Racing'],
  ['norris', 4, 'NOR', 'Lando', 'Norris', 'mclaren', 'McLaren'],
  ['piastri', 81, 'PIA', 'Oscar', 'Piastri', 'mclaren', 'McLaren'],
  ['leclerc', 16, 'LEC', 'Charles', 'Leclerc', 'ferrari', 'Ferrari'],
  ['hamilton', 44, 'HAM', 'Lewis', 'Hamilton', 'ferrari', 'Ferrari'],
  ['russell', 63, 'RUS', 'George', 'Russell', 'mercedes', 'Mercedes'],
  ['antonelli', 12, 'ANT', 'Andrea Kimi', 'Antonelli', 'mercedes', 'Mercedes'],
  ['alonso', 14, 'ALO', 'Fernando', 'Alonso', 'aston_martin', 'Aston Martin'],
  ['stroll', 18, 'STR', 'Lance', 'Stroll', 'aston_martin', 'Aston Martin'],
  ['gasly', 10, 'GAS', 'Pierre', 'Gasly', 'alpine', 'Alpine'],
  ['colapinto', 43, 'COL', 'Franco', 'Colapinto', 'alpine', 'Alpine'],
  ['albon', 23, 'ALB', 'Alexander', 'Albon', 'williams', 'Williams'],
  ['sainz', 55, 'SAI', 'Carlos', 'Sainz', 'williams', 'Williams'],
  ['tsunoda', 22, 'TSU', 'Yuki', 'Tsunoda', 'rb', 'Racing Bulls'],
  ['hadjar', 6, 'HAD', 'Isack', 'Hadjar', 'rb', 'Racing Bulls'],
  ['hulkenberg', 27, 'HUL', 'Nico', 'Hülkenberg', 'sauber', 'Sauber'],
  ['bortoleto', 5, 'BOR', 'Gabriel', 'Bortoleto', 'sauber', 'Sauber'],
  ['ocon', 31, 'OCO', 'Esteban', 'Ocon', 'haas', 'Haas'],
  ['bearman', 87, 'BEA', 'Oliver', 'Bearman', 'haas', 'Haas'],
];

/** Générateur pseudo-aléatoire déterministe : le décor ne change pas à chaque F5. */
function rng(graine) {
  let x = graine >>> 0;
  return () => {
    x ^= x << 13; x >>>= 0; x ^= x >> 17; x ^= x << 5; x >>>= 0;
    return x / 4294967296;
  };
}

/** Vendredi de la semaine en cours (ou de la suivante si le week-end est passé). */
function vendrediProchain(maintenant = new Date()) {
  const d = new Date(Date.UTC(
    maintenant.getUTCFullYear(), maintenant.getUTCMonth(), maintenant.getUTCDate(), 0, 0, 0));
  const delta = (5 - d.getUTCDay() + 7) % 7;   // 5 = vendredi
  d.setUTCDate(d.getUTCDate() + delta);
  return d;
}

function aHeure(base, joursApres, hUTC, minUTC = 0) {
  const d = new Date(base);
  d.setUTCDate(d.getUTCDate() + joursApres);
  d.setUTCHours(hUTC, minUTC, 0, 0);
  return d.toISOString();
}

export function calendrierDemo(saison) {
  const v = vendrediProchain();
  const gp = {
    id: `${saison}-demo`, saison, round: 14, nom: 'Grand Prix de démonstration',
    format: 'classique', circuit_id: 'spa', circuit_nom: 'Circuit de Spa-Francorchamps',
    pays: 'Belgique', ville: 'Stavelot', lat: 50.4372, lon: 5.9714,
    sessions: [
      { type: 'FP1', debut_utc: aHeure(v, 0, 11, 30), heure_connue: true },
      { type: 'FP2', debut_utc: aHeure(v, 0, 15, 0), heure_connue: true },
      { type: 'FP3', debut_utc: aHeure(v, 1, 10, 30), heure_connue: true },
      { type: 'Q', debut_utc: aHeure(v, 1, 14, 0), heure_connue: true },
      { type: 'R', debut_utc: aHeure(v, 2, 13, 0), heure_connue: true },
    ],
    debut_utc: aHeure(v, 0, 11, 30),
    course_utc: aHeure(v, 2, 13, 0),
    source: 'DEMO',
  };
  const precedent = {
    ...gp,
    id: `${saison}-demo-prec`, round: 13, nom: 'Grand Prix de démonstration (précédent)',
    circuit_id: 'monza', circuit_nom: 'Autodromo Nazionale di Monza',
    pays: 'Italie', ville: 'Monza', lat: 45.6156, lon: 9.2811,
    sessions: [
      { type: 'FP1', debut_utc: aHeure(v, -7, 11, 30), heure_connue: true },
      { type: 'Q', debut_utc: aHeure(v, -6, 14, 0), heure_connue: true },
      { type: 'R', debut_utc: aHeure(v, -5, 13, 0), heure_connue: true },
    ],
    debut_utc: aHeure(v, -7, 11, 30),
    course_utc: aHeure(v, -5, 13, 0),
  };
  return [precedent, gp];
}

export function engagesDemo() {
  const r = rng(20260401);
  return ENGAGES.map(([id, num, code, prenom, nom, teamId, teamNom], i) => ({
    driver_id: id, numero: num, code, prenom, nom,
    nationalite: '—', date_naissance: null,
    team_id: teamId, team_nom: teamNom,
    points: Math.round((20 - i) * 12 + r() * 25),
    position_championnat: i + 1,
    victoires: i < 3 ? 3 - i : 0,
    source: 'DEMO',
  }));
}

/** Résultat de course tiré au sort autour de l'ordre du championnat. */
export function resultatsDemo(round) {
  const r = rng(1000 + round);
  const engages = engagesDemo();
  const bruite = engages
    .map((e, i) => ({ e, cle: i + (r() - 0.5) * 7 }))
    .sort((a, b) => a.cle - b.cle)
    .map(({ e }) => e);

  const abandons = new Set();
  bruite.forEach((e) => { if (r() < 0.09) abandons.add(e.driver_id); });

  const classes = bruite.filter((e) => !abandons.has(e.driver_id));
  const sortis = bruite.filter((e) => abandons.has(e.driver_id));
  const points = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1];
  const mt = classes[Math.floor(r() * Math.min(6, classes.length))];

  const lignes = [
    ...classes.map((e, i) => ({
      position: i + 1, position_texte: String(i + 1), classe: true,
      driver_id: e.driver_id, code: e.code, prenom: e.prenom, nom: e.nom,
      team_id: e.team_id, team_nom: e.team_nom, numero: e.numero,
      grille: 0, tours: 44, points: points[i] || 0, statut: 'Finished',
      abandon: false, temps: null,
      meilleur_tour: e.driver_id === mt?.driver_id ? { rang: 1, tour: 38, temps: '1:46.286' } : null,
    })),
    ...sortis.map((e, i) => ({
      position: classes.length + i + 1, position_texte: 'R', classe: false,
      driver_id: e.driver_id, code: e.code, prenom: e.prenom, nom: e.nom,
      team_id: e.team_id, team_nom: e.team_nom, numero: e.numero,
      grille: 0, tours: Math.floor(r() * 40), points: 0,
      statut: ['Hydraulics', 'Collision', 'Power Unit', 'Accident'][Math.floor(r() * 4)],
      abandon: true, temps: null, meilleur_tour: null,
    })),
  ];

  const grille = [...engages].map((e, i) => ({ e, cle: i + (r() - 0.5) * 4 }))
    .sort((a, b) => a.cle - b.cle);
  const posGrille = new Map(grille.map(({ e }, i) => [e.driver_id, i + 1]));
  lignes.forEach((l) => { l.grille = posGrille.get(l.driver_id) || 20; });

  return { gp_nom: 'Grand Prix de démonstration', round, saison: null, lignes, safety_car: r() < 0.55 };
}

export function qualifDemo(round) {
  const res = resultatsDemo(round);
  const parGrille = [...res.lignes].sort((a, b) => a.grille - b.grille);
  return {
    gp_nom: res.gp_nom,
    lignes: parGrille.map((l, i) => ({
      position: i + 1, driver_id: l.driver_id, code: l.code, prenom: l.prenom, nom: l.nom,
      team_id: l.team_id, team_nom: l.team_nom,
      q1: null, q2: null, q3: null,
      meilleur: `1:${(41 + i * 0.18).toFixed(3).replace('.', ',')}`,
    })),
  };
}

export function meteoDemo(lat, lon) {
  const r = rng(Math.round((lat + lon) * 1000));
  const base = new Date();
  base.setUTCMinutes(0, 0, 0);
  // 14 jours au pas horaire : la fenêtre doit couvrir le week-end fictif,
  // qui peut être à six jours quand on consulte le site un samedi.
  return Array.from({ length: 336 }, (_, i) => {
    const d = new Date(base.getTime() + (i - 12) * 3600000);
    const heure = d.getUTCHours();
    const jour = Math.sin(((heure - 6) / 24) * Math.PI * 2);
    const p = r();
    return {
      heure_utc: d.toISOString().replace(/\.\d+Z$/, 'Z'),
      temperature_c: Math.round((15 + jour * 5 + p * 3) * 10) / 10,
      humidite_pct: Math.round(60 + (1 - jour) * 12 + p * 10),
      proba_pluie_pct: Math.round(p * 80),
      precipitation_mm: p > 0.7 ? Math.round(p * 30) / 10 : 0,
      vent_kmh: Math.round(8 + p * 14),
      vent_dir_deg: Math.round(p * 360),
      nuages_pct: Math.round(30 + p * 60),
      sol_c: Math.round((18 + jour * 9 + p * 4) * 10) / 10,
      pression_hpa: Math.round(1008 + p * 12),
    };
  });
}

export function raceControlDemo() {
  const maintenant = Date.now();
  return [
    { date_utc: new Date(maintenant - 22 * 60000).toISOString(), categorie: 'Other', drapeau: null,
      message: 'PÉNALITÉ DE 5 PLACES SUR LA GRILLE — VOITURE 16 — CHANGEMENT DE BOÎTE DE VITESSES', numero_pilote: 16 },
    { date_utc: new Date(maintenant - 96 * 60000).toISOString(), categorie: 'Other', drapeau: null,
      message: 'CHANGEMENT D\'ÉLÉMENT DE GROUPE PROPULSEUR — VOITURE 4 — HORS QUOTA', numero_pilote: 4 },
    { date_utc: new Date(maintenant - 5 * 3600000).toISOString(), categorie: 'Flag', drapeau: 'YELLOW',
      message: 'DRAPEAU JAUNE SECTEUR 2 — VOITURE ARRÊTÉE', secteur: 2 },
  ];
}
