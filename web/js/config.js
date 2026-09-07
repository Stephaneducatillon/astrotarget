/**
 * Configuration centrale — traçabilité livrable 1.
 * Toute donnée affichée doit pouvoir remonter à une entrée de SOURCES.
 */

export const APP = {
  nom: 'Paddock',
  version: '1.0.0',
  // Phases livrées par ce front (livrable 4). Sert à afficher honnêtement
  // ce qui n'est pas encore disponible plutôt qu'un écran vide.
  phases: { hub: 1, circuit: 1, pilote: 2, pronostic: 3, debrief: 3, ligue: 3, comparateur: 2 },
};

/** Livrable 1 §1 — sources et préséance. */
export const SOURCES = {
  JOL: {
    code: 'JOL', nom: 'Jolpica-F1 (successeur d\'Ergast)', base: 'https://api.jolpi.ca/ergast/f1',
    couverture: '1950 → aujourd\'hui', latence: 'quelques heures après la session',
    licence: 'CC BY 4.0', url: 'https://jolpi.ca/ergast/',
  },
  OF1: {
    code: 'OF1', nom: 'OpenF1', base: 'https://api.openf1.org/v1',
    couverture: '2023 → aujourd\'hui', latence: '~3 s en live',
    licence: 'open source', url: 'https://openf1.org/',
  },
  OM: {
    code: 'OM', nom: 'Open-Meteo', base: 'https://api.open-meteo.com/v1/forecast',
    couverture: 'prévision 16 jours', latence: 'rafraîchi 4×/jour',
    licence: 'CC BY 4.0', url: 'https://open-meteo.com/',
  },
  FF1: {
    code: 'FF1', nom: 'FastF1 (batch Python)', base: '—',
    couverture: '2018 → aujourd\'hui', latence: 'batch post-session',
    licence: 'MIT', url: 'https://docs.fastf1.dev/',
  },
  CALC: { code: 'CALC', nom: 'Calculé par la plateforme', base: '—', couverture: '—', latence: 'selon job', licence: '—', url: '' },
  MAN: { code: 'MAN', nom: 'Saisie manuelle (back-office)', base: '—', couverture: '—', latence: 'ponctuel', licence: '—', url: '' },
  DEMO: { code: 'DÉMO', nom: 'Jeu de démonstration local', base: 'web/data/demo.json', couverture: 'un week-end fictif', latence: '—', licence: '—', url: '' },
};

/** Livrable 1 §5 — fraîcheur et cache. TTL en millisecondes. */
const MIN = 60 * 1000, H = 60 * MIN, J = 24 * H;
export const TTL = {
  calendrier: 24 * H,
  pilotes: 24 * H,
  classement: 6 * H,
  circuit: 7 * J,
  meteo: 30 * MIN,
  meteo_loin: 3 * H,
  live: 0,
  race_control: 0,
  resultats: 1 * H,
  derives: 2 * H,
};

/** Livrable 1 §1 — limites à documenter dans l'UI. */
export const LIMITES = [
  'Pas de télémétrie avant 2023 via OpenF1 (2018 → via FastF1, en batch).',
  'Jolpica applique une limite de débit : les réponses sont mises en cache localement.',
  'Aucune rediffusion de contenu officiel F1 : ni vidéo, ni live timing officiel, ni image sous licence.',
  'Les réglages de voiture ne sont publiés par personne. Ce qui est affiché ici sont des mesures dérivées de la télémétrie, jamais les réglages d\'une écurie.',
];

/**
 * Couleurs d'écurie (livrable 1 §2 : `team.couleur_hex` est une donnée MAN).
 * `motif` sert au mode accessible : plusieurs paires d'écuries sont
 * indistinguables pour les daltoniens (livrable 3, règles transverses).
 */
export const ECURIES = {
  red_bull:     { nom: 'Red Bull Racing', hex: '#3671C6', motif: 'solide' },
  mclaren:      { nom: 'McLaren',         hex: '#FF8000', motif: 'diagonal' },
  ferrari:      { nom: 'Ferrari',         hex: '#E8002D', motif: 'solide' },
  mercedes:     { nom: 'Mercedes',        hex: '#27F4D2', motif: 'points' },
  aston_martin: { nom: 'Aston Martin',    hex: '#229971', motif: 'diagonal' },
  alpine:       { nom: 'Alpine',          hex: '#FF87BC', motif: 'points' },
  williams:     { nom: 'Williams',        hex: '#64C4FF', motif: 'horizontal' },
  rb:           { nom: 'Racing Bulls',    hex: '#6692FF', motif: 'diagonal' },
  sauber:       { nom: 'Sauber / Audi',   hex: '#52E252', motif: 'horizontal' },
  haas:         { nom: 'Haas',            hex: '#B6BABD', motif: 'points' },
  cadillac:     { nom: 'Cadillac',        hex: '#C9A227', motif: 'horizontal' },
  audi:         { nom: 'Audi',            hex: '#52E252', motif: 'horizontal' },
  racing_bulls: { nom: 'Racing Bulls',    hex: '#6692FF', motif: 'diagonal' },
  alphatauri:   { nom: 'AlphaTauri',      hex: '#6692FF', motif: 'diagonal' },
  alfa:         { nom: 'Alfa Romeo',      hex: '#52E252', motif: 'horizontal' },
  renault:      { nom: 'Renault',         hex: '#FF87BC', motif: 'points' },
  racing_point: { nom: 'Racing Point',    hex: '#229971', motif: 'diagonal' },
  force_india:  { nom: 'Force India',     hex: '#229971', motif: 'diagonal' },
  toro_rosso:   { nom: 'Toro Rosso',      hex: '#6692FF', motif: 'diagonal' },
};

export const COMPOSES = {
  SOFT:   { label: 'Tendre', hex: '#E8002D' },
  MEDIUM: { label: 'Medium', hex: '#F5C518' },
  HARD:   { label: 'Dur',    hex: '#E8E8E8' },
  INTER:  { label: 'Inter',  hex: '#3FBF3F' },
  WET:    { label: 'Pluie',  hex: '#3671C6' },
};

/** Ordre chronologique d'un week-end, sert au tri et aux libellés. */
export const SESSIONS = {
  FP1: { ordre: 1, label: 'EL1',    long: 'Essais libres 1' },
  FP2: { ordre: 2, label: 'EL2',    long: 'Essais libres 2' },
  FP3: { ordre: 3, label: 'EL3',    long: 'Essais libres 3' },
  SQ:  { ordre: 4, label: 'Q. spr', long: 'Qualifications sprint' },
  SPR: { ordre: 5, label: 'Sprint', long: 'Course sprint' },
  Q:   { ordre: 6, label: 'Q',      long: 'Qualifications' },
  R:   { ordre: 7, label: 'Course', long: 'Grand Prix' },
};

export const CONFIG_PRONOSTIC = {
  taille_grille: 10,      // livrable 2 §9 : arbitrage ouvert (10 ou top 5)
  coef_min: 0.5,
  coef_max: 4.0,          // livrable 2 §9 : plafond à arbitrer
  mult_hardcore: 1.5,     // livrable 2 §3
  jokers_triple_saison: 3,
  course_ecourtee_ratio: 0.5,
};
