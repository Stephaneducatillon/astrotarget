/**
 * Modèle baseline — `proba_baseline` (livrable 1 §4, story 2.7/2.8).
 *
 * C'est le socle du scoring pondéré : le coefficient de difficulté d'un
 * pronostic dérive de la probabilité que ce modèle donnait à l'événement
 * AU MOMENT DE LA FERMETURE.
 *
 * Deux exigences d'intégrité (livrable 2 §7) commandent la conception :
 *  - la sortie doit être DÉTERMINISTE pour des entrées données, sinon deux
 *    calculs successifs donneraient des coefficients différents ;
 *  - elle est figée et horodatée à la fermeture, jamais recalculée après coup.
 *
 * Méthode : notation de force par pilote (championnat + forme + affinité
 * circuit), puis simulation Plackett–Luce par bruit de Gumbel. On lit les
 * probabilités marginales de position sur les tirages. C'est volontairement
 * simple et lisible : un modèle mal calibré casse le scoring (risque du
 * livrable 4), mieux vaut un modèle qu'on peut auditer qu'une boîte noire.
 *
 * Limite assumée : cette version tourne côté client sur les seules données
 * Jolpica. Le modèle Elo pilote/écurie de la phase 2, calibré sur deux
 * saisons de télémétrie, a vocation à le remplacer — l'interface de sortie
 * (`probas`, `evenements`) ne change pas.
 */

const TIRAGES = 4000;
const ETALEMENT = 3.4;        // écart de force entre premier et dernier
const DNF_BASE = 0.075;       // taux d'abandon moyen par pilote et par course
const POIDS_FORME = 0.35;     // part de la forme récente dans la note
const POIDS_CIRCUIT = 0.15;   // part de l'affinité circuit

function rngDeterministe(graine) {
  let x = (graine >>> 0) || 1;
  return () => {
    x ^= x << 13; x >>>= 0; x ^= x >> 17; x ^= x << 5; x >>>= 0;
    return (x >>> 0) / 4294967296;
  };
}

function graineDe(texte) {
  let h = 2166136261;
  for (let i = 0; i < texte.length; i++) { h ^= texte.charCodeAt(i); h = Math.imul(h, 16777619); }
  return h >>> 0;
}

/** Gumbel(0,1) : ajouté à log(force), il produit un tirage Plackett–Luce. */
function gumbel(u) { return -Math.log(-Math.log(Math.max(1e-12, Math.min(1 - 1e-12, u)))); }

function normaliser(valeurs) {
  const max = Math.max(...valeurs), min = Math.min(...valeurs);
  if (!Number.isFinite(max) || max === min) return valeurs.map(() => 0.5);
  return valeurs.map((v) => (v - min) / (max - min));
}

/**
 * `forme_pilote` (livrable 1 §4) : moyenne pondérée des points sur 5 GP,
 * décroissance exponentielle — le dernier GP pèse le plus.
 */
export function formePilote(resultats, fenetre = 5) {
  const derniers = (resultats || []).slice(-fenetre);
  if (!derniers.length) return null;
  let num = 0, den = 0;
  derniers.forEach((r, i) => {
    const poids = Math.exp(-0.35 * (derniers.length - 1 - i));
    num += (Number(r.points) || 0) * poids;
    den += poids;
  });
  return den ? num / den : null;
}

/** `delta_qualif_course` : position course − position grille, moyenne sur 5 GP. */
export function deltaQualifCourse(resultats, fenetre = 5) {
  const valides = (resultats || []).filter((r) => r.grille > 0 && Number.isFinite(r.position)).slice(-fenetre);
  if (!valides.length) return null;
  return valides.reduce((s, r) => s + (r.position - r.grille), 0) / valides.length;
}

/** Taux d'abandon observé, replié vers la moyenne quand l'échantillon est court. */
export function fiabilite(resultats) {
  const n = (resultats || []).length;
  if (!n) return DNF_BASE;
  const abandons = resultats.filter((r) => r.abandon || !/^\d+$/.test(String(r.position_texte ?? r.position))).length;
  const k = 8; // force du repli bayésien
  return (abandons + DNF_BASE * k) / (n + k);
}

/** `coef_circuit` : affinité d'un pilote avec ce tracé, sur son historique. */
export function affiniteCircuit(resultatsCircuit) {
  const valides = (resultatsCircuit || []).filter((r) => Number.isFinite(r.position) && r.position > 0);
  if (!valides.length) return null;
  const moy = valides.reduce((s, r) => s + r.position, 0) / valides.length;
  return Math.max(0, Math.min(1, 1 - (moy - 1) / 19)); // 1 = toujours devant
}

/**
 * @param {object} params
 * @param {Array}  params.engages         entries de la saison
 * @param {object} params.circuit         fiche circuit (MAN/CALC)
 * @param {object} params.historique      { [driver_id]: { saison:[], circuit:[] } }
 * @param {Array}  params.grille          ordre de départ connu, sinon null
 * @param {string} params.cle             identifiant du GP, sert de graine
 */
export function construireBaseline({ engages, circuit, historique = {}, grille = null, cle = 'gp' }) {
  const pilotes = (engages || []).filter((e) => e.driver_id);
  if (pilotes.length < 2) throw new Error('Pas assez d\'engagés pour construire une baseline.');

  const pointsChampionnat = pilotes.map((e) => Number(e.points) || 0);
  const notePoints = normaliser(pointsChampionnat);

  const notesForme = normaliser(pilotes.map((e) => {
    const f = formePilote(historique[e.driver_id]?.saison);
    return f === null ? -1 : f;
  }).map((v, i) => (v < 0 ? notePoints[i] * 25 : v)));

  const notesCircuit = pilotes.map((e, i) => {
    const a = affiniteCircuit(historique[e.driver_id]?.circuit);
    return a === null ? notePoints[i] : a;
  });

  // Note composite → force multiplicative.
  const notes = pilotes.map((_, i) =>
    (1 - POIDS_FORME - POIDS_CIRCUIT) * notePoints[i]
    + POIDS_FORME * notesForme[i]
    + POIDS_CIRCUIT * notesCircuit[i]);

  const forces = notes.map((n) => Math.exp(ETALEMENT * n));

  // Position de départ : sur un circuit où l'on ne dépasse pas, la grille
  // pèse plus lourd que la forme. `depassements_moyens` module ce poids.
  const dep = circuit?.depassements_moyens ?? 25;
  const poidsGrille = grille ? Math.max(0.15, Math.min(0.75, 1 - dep / 55)) : 0;
  const posGrille = new Map((grille || []).map((g, i) => [g.driver_id, i + 1]));

  const forcesFinales = pilotes.map((e, i) => {
    if (!poidsGrille) return forces[i];
    const p = posGrille.get(e.driver_id);
    if (!p) return forces[i];
    const forceGrille = Math.exp(ETALEMENT * (1 - (p - 1) / Math.max(1, pilotes.length - 1)));
    return Math.exp((1 - poidsGrille) * Math.log(forces[i]) + poidsGrille * Math.log(forceGrille));
  });

  // Un circuit à fort taux de neutralisation casse plus de voitures.
  const facteurCasse = 1 + 0.6 * ((circuit?.taux_sc_historique ?? 0.5) - 0.5);
  const pDnf = pilotes.map((e) =>
    Math.max(0.01, Math.min(0.45, fiabilite(historique[e.driver_id]?.saison) * facteurCasse)));

  const rnd = rngDeterministe(graineDe(`${cle}|${pilotes.map((p) => p.driver_id).join(',')}`));
  const n = pilotes.length;
  const logForces = forcesFinales.map((f) => Math.log(f));

  const comptePos = pilotes.map(() => new Array(n).fill(0));
  const comptePremierAbandon = pilotes.map(() => 0);
  const compteAbandons = new Array(n + 1).fill(0);
  const compteMeilleurTour = pilotes.map(() => 0);
  const comptePole = pilotes.map(() => 0);
  let aucunAbandon = 0;

  for (let t = 0; t < TIRAGES; t++) {
    // Qualification : même force, bruit plus faible (moins d'aléa qu'en course).
    const ordreQ = logForces
      .map((lf, i) => ({ i, s: lf * 1.25 + gumbel(rnd()) * 0.75 }))
      .sort((a, b) => b.s - a.s);
    comptePole[ordreQ[0].i]++;

    // Course.
    const abandonne = pDnf.map((p) => rnd() < p);
    const nbAbandons = abandonne.filter(Boolean).length;
    compteAbandons[nbAbandons]++;
    if (nbAbandons === 0) aucunAbandon++;

    if (nbAbandons > 0) {
      // Le premier abandon : tirage pondéré par la fragilité, les faibles
      // tombant plus tôt en moyenne.
      let meilleur = -1, meilleurScore = Infinity;
      for (let i = 0; i < n; i++) {
        if (!abandonne[i]) continue;
        const s = rnd() * (1 + logForces[i]);
        if (s < meilleurScore) { meilleurScore = s; meilleur = i; }
      }
      if (meilleur >= 0) comptePremierAbandon[meilleur]++;
    }

    const classement = logForces
      .map((lf, i) => ({ i, s: lf + gumbel(rnd()), dnf: abandonne[i] }))
      .sort((a, b) => (a.dnf === b.dnf ? b.s - a.s : (a.dnf ? 1 : -1)))
      .map((x) => x.i);

    classement.forEach((idx, pos) => { comptePos[idx][pos]++; });

    // Meilleur tour : très corrélé à la force, souvent chez qui a un arrêt
    // « gratuit » en fin de course. Approché sur les 8 premiers.
    const candidats = classement.slice(0, 8);
    compteMeilleurTour[candidats[Math.floor(rnd() * candidats.length)]]++;
  }

  const probas = {};
  const parPilote = {};
  pilotes.forEach((e, i) => {
    probas[e.driver_id] = comptePos[i].map((c) => c / TIRAGES);
    parPilote[e.driver_id] = {
      pole: comptePole[i] / TIRAGES,
      meilleur_tour: compteMeilleurTour[i] / TIRAGES,
      premier_abandon: comptePremierAbandon[i] / TIRAGES,
      abandon: pDnf[i],
      note: notes[i],
    };
  });

  const distributionAbandons = compteAbandons.map((c) => c / TIRAGES);

  // Duel d'équipiers : un duel imposé par week-end (livrable 2, bloc C).
  const duels = construireDuels(pilotes, probas);

  return {
    ts: Date.now(),
    cle,
    circuit_id: circuit?.id || null,
    pilotes: pilotes.map((e) => ({
      driver_id: e.driver_id, code: e.code, prenom: e.prenom, nom: e.nom,
      team_id: e.team_id, team_nom: e.team_nom, numero: e.numero,
    })),
    probas,
    parPilote,
    evenements: {
      safety_car_oui: Math.max(0.02, Math.min(0.98, circuit?.taux_sc_historique ?? 0.5)),
      abandons: distributionAbandons,
      aucun_abandon: aucunAbandon / TIRAGES,
      ecart_pole_p2: ecartPoleAttendu(circuit),
    },
    duels,
    meta: {
      tirages: TIRAGES,
      grille_connue: Boolean(grille),
      poids_grille: poidsGrille,
      methode: 'Plackett–Luce par bruit de Gumbel, forces issues du championnat, de la forme (5 GP) et de l\'affinité circuit',
      avertissement: 'Modèle client, calibrage provisoire. À remplacer par l\'Elo pilote/écurie de la phase 2, backtesté sur 3 saisons avant ouverture des pronostics.',
    },
  };
}

/** Fourchette d'écart pole ↔ P2 (bloc B) : trois classes, probabilités. */
function ecartPoleAttendu(circuit) {
  const e = circuit?.ecart_pole_p2_s ?? 0.18;
  // Loi log-normale grossière autour de l'écart historique du tracé.
  const p1 = Math.max(0.05, Math.min(0.9, 1 / (1 + Math.exp((e - 0.10) * 22))));
  const p3 = Math.max(0.05, Math.min(0.9, 1 / (1 + Math.exp((0.30 - e) * 22))));
  const p2 = Math.max(0.05, 1 - p1 - p3);
  const s = p1 + p2 + p3;
  return { '<0,1 s': p1 / s, '0,1–0,3 s': p2 / s, '>0,3 s': p3 / s };
}

/** Duel d'équipiers : on retient la paire la plus serrée, c'est la plus jouable. */
function construireDuels(pilotes, probas) {
  const parEquipe = new Map();
  for (const p of pilotes) {
    if (!p.team_id) continue;
    if (!parEquipe.has(p.team_id)) parEquipe.set(p.team_id, []);
    parEquipe.get(p.team_id).push(p);
  }
  const duels = [];
  for (const [teamId, membres] of parEquipe) {
    if (membres.length !== 2) continue;
    const [a, b] = membres;
    const pa = probaDevant(probas[a.driver_id], probas[b.driver_id]);
    duels.push({
      team_id: teamId, team_nom: a.team_nom,
      a: { driver_id: a.driver_id, nom: a.nom, code: a.code, proba: pa },
      b: { driver_id: b.driver_id, nom: b.nom, code: b.code, proba: 1 - pa },
      serrage: Math.abs(pa - 0.5),
    });
  }
  duels.sort((x, y) => x.serrage - y.serrage);
  return duels;
}

/** P(A devant B) à partir des marginales de position. */
function probaDevant(pa, pb) {
  if (!pa || !pb) return 0.5;
  let p = 0;
  for (let i = 0; i < pa.length; i++) {
    let cumB = 0;
    for (let j = i + 1; j < pb.length; j++) cumB += pb[j];
    p += pa[i] * cumB;
  }
  const q = 1 - p;
  return (p + q) > 0 ? p / (p + q) : 0.5;
}

/** Probabilité qu'un pilote termine exactement à une position (1-indexée). */
export function probaPosition(baseline, driverId, position) {
  const v = baseline?.probas?.[driverId]?.[position - 1];
  return Number.isFinite(v) ? v : 0.01;
}

/** Pronostic que soumettrait le modèle lui-même — sert au score de référence. */
export function pronosticBaseline(baseline, tailleGrille = 10) {
  const esperance = baseline.pilotes.map((p) => {
    const probas = baseline.probas[p.driver_id] || [];
    const e = probas.reduce((s, prob, i) => s + prob * (i + 1), 0) || 99;
    return { driver_id: p.driver_id, e };
  }).sort((a, b) => a.e - b.e);

  const meilleur = (cle) => Object.entries(baseline.parPilote)
    .sort((a, b) => b[1][cle] - a[1][cle])[0]?.[0] || null;

  const distrib = baseline.evenements.abandons;
  const nbAbandons = distrib.indexOf(Math.max(...distrib));
  const ecarts = baseline.evenements.ecart_pole_p2;
  const ecart = Object.entries(ecarts).sort((a, b) => b[1] - a[1])[0][0];

  const duel = baseline.duels[0] || null;

  return {
    grille: esperance.slice(0, tailleGrille).map((x) => x.driver_id),
    pole: meilleur('pole'),
    meilleur_tour: meilleur('meilleur_tour'),
    premier_abandon: meilleur('premier_abandon'),
    safety_car: baseline.evenements.safety_car_oui >= 0.5,
    nb_abandons: nbAbandons,
    ecart_pole_p2: ecart,
    duel: duel ? (duel.a.proba >= duel.b.proba ? duel.a.driver_id : duel.b.driver_id) : null,
  };
}
