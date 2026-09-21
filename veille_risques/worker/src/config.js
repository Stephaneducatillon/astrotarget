// Configuration du dispositif. Tout ce qui est propre au foyer est ici, et
// rien d'autre : aucune adresse, aucun nom d'ecole. Les couches personnelles
// vivent hors du depot (voir README, section Confidentialite).

export const DEPARTEMENTS = ['59', '62'];

// Phenomenes de la vigilance Meteo-France retenus pour le passage en niveau
// familial. La cle est l'identifiant de l'API, la valeur le libelle affiche.
// Les avalanches (8) et vagues-submersion (9) ne concernent pas le Douaisis.
export const PHENOMENES = {
  1: 'Vent violent',
  2: 'Pluie-inondation',
  3: 'Orages',
  4: 'Crues',
  5: 'Neige-verglas',
  6: 'Canicule',
  7: 'Grand froid',
};

// Couleurs de l'API vigilance : 1 vert, 2 jaune, 3 orange, 4 rouge.
// Le jaune ne declenche rien cote famille, conformement au plan.
export const COULEURS = { 1: 'vert', 2: 'jaune', 3: 'orange', 4: 'rouge' };

// ---------------------------------------------------------------------------
// Stations a renseigner AVANT le premier deploiement.
//
// Les codes ne sont pas devinables : ils se relevent une fois pour toutes via
//   GET /admin/stations?q=scarpe      (referentiel Hub'Eau)
//   GET /admin/troncons?q=scarpe      (referentiel Vigicrues)
// puis se figent ici. Ne jamais les rechercher a chaque run.
// ---------------------------------------------------------------------------

export const STATIONS_HYDRO = [
  // { code: 'E4035710', nom: 'Scarpe a Douai', metric: 'H' },
];

export const TRONCONS_VIGICRUES = [
  // { code: 'XXX', nom: 'Scarpe aval' },
];

// Seuils et temporisations du moteur d'alerte.
export const SEUILS = {
  // Regle Scarpe : alerte si la hauteur depasse le maximum des N derniers jours.
  hydro_fenetre_jours: 30,
  // Nombre minimal de mesures dans la fenetre avant d'oser evaluer la regle.
  // Sans ce garde-fou, la premiere semaine de service declenche sur un maximum
  // calcule sur trois points.
  hydro_min_observations: 200,
  // Marge relative pour ne pas declencher sur un depassement d'un millimetre.
  hydro_marge: 1.02,
};

export const TEMPO = {
  // Descente : il faut N releves consecutifs sous le seuil pour redescendre.
  // A 15 min, 3 relevés = 45 min. Montee toujours immediate.
  releves_avant_descente: 3,
  // Plafond de rappel d'un etat qui dure, en secondes.
  rappel_secondes: 6 * 3600,
  // Nombre d'echecs consecutifs avant de signaler une source muette.
  echecs_avant_alerte: 3,
  // Retention D1, en jours. La fenetre utile est de 30 j, on garde 40 j.
  retention_jours: 40,
};

// Definition des sources de niveau A. `freq` sert a calculer la peremption
// affichee dans le tableau de bord : au-dela de 3x cette valeur, la donnee est
// marquee perimee.
export const SOURCES = [
  { id: 'vigilance', nom: 'Vigilance Meteo-France', freq: 15 * 60, niveau: 'A' },
  { id: 'ecowatt',   nom: 'Ecowatt (RTE)',          freq: 3 * 3600, niveau: 'A' },
  { id: 'hubeau',    nom: "Hub'Eau - Scarpe",       freq: 15 * 60, niveau: 'A' },
  { id: 'vigicrues', nom: 'Vigicrues - tronçon',    freq: 15 * 60, niveau: 'A' },
];

export const NIVEAUX = ['vert', 'orange', 'rouge'];

// Checklists rappelees dans chaque notification. Volontairement generiques :
// une notification ntfy transite par un serveur tiers, elle ne doit contenir
// ni adresse, ni nom d'etablissement.
export const CHECKLIST = {
  orange: 'Checklist ORANGE du plan familial : verifier telephones charges, eau, lampes.',
  rouge: 'Checklist ROUGE du plan familial : appliquer les consignes, rester joignable.',
};
