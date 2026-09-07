/**
 * Normalisation du résultat officiel, tel que le moteur de score l'attend.
 *
 * Certains faits ne sont pas dans Jolpica : la présence d'une safety car vient
 * de la direction de course (OpenF1, 2023 →). Quand un fait manque, il est
 * marqué indisponible et la ligne correspondante n'est pas notée — plutôt que
 * d'être comptée comme ratée, ce qui fausserait le score.
 */

import * as jolpica from '../data/jolpica.js';
import * as of1 from '../data/openf1.js';
import * as reference from '../data/reference.js';

function tempsEnSecondes(t) {
  if (!t) return null;
  const m = /^(?:(\d+):)?(\d+)\.(\d+)$/.exec(String(t).trim());
  if (!m) return null;
  return (Number(m[1] || 0) * 60) + Number(m[2]) + Number(`0.${m[3]}`);
}

export function classeEcart(secondes) {
  if (secondes === null) return null;
  if (secondes < 0.1) return '<0,1 s';
  if (secondes <= 0.3) return '0,1–0,3 s';
  return '>0,3 s';
}

export async function construire(gp, { modeDemo = false, resultatsDemo, qualifDemo } = {}) {
  let course = null, qualif = null, ts = Date.now();
  let scConnu = false, safetyCar = null;

  if (modeDemo) {
    course = resultatsDemo;
    qualif = qualifDemo;
    safetyCar = Boolean(resultatsDemo?.safety_car);
    scConnu = true;
  } else {
    const [rc, rq] = await Promise.allSettled([
      jolpica.resultatsCourse(gp.saison, gp.round),
      jolpica.resultatsQualif(gp.saison, gp.round),
    ]);
    if (rc.status === 'fulfilled') { course = rc.value.valeur; ts = rc.value.ts; }
    if (rq.status === 'fulfilled') qualif = rq.value.valeur;

    // Safety car : uniquement lisible dans les messages de direction de course.
    if (gp.saison >= of1.COUVERTURE_DEPUIS) {
      try {
        const s = await of1.trouverSession(gp.saison, gp.ville || gp.circuit_nom, 'R');
        if (s) {
          const r = await of1.raceControl(s.session_key);
          safetyCar = r.valeur.some((m) =>
            /SAFETY CAR/i.test(m.message || '') && !/VIRTUAL SAFETY CAR ENDING|SAFETY CAR IN THIS LAP/i.test(m.message || ''));
          scConnu = true;
        }
      } catch { /* OpenF1 muet : le fait reste inconnu */ }
    }
  }

  if (!course?.lignes?.length) return null;

  const classes = course.lignes.filter((l) => l.classe);
  const top10 = classes.slice(0, 10).map((l) => l.driver_id);
  const abandons = course.lignes.filter((l) => l.abandon);

  // Premier abandon : celui qui a bouclé le moins de tours. Approximation —
  // l'ordre exact demande l'horodatage des messages de direction de course.
  const premier = abandons.length
    ? [...abandons].sort((a, b) => (a.tours ?? 0) - (b.tours ?? 0))[0].driver_id
    : null;

  const mt = course.lignes.find((l) => l.meilleur_tour?.rang === 1)?.driver_id
    || course.lignes.find((l) => l.meilleur_tour)?.driver_id
    || null;

  // Écart pole ↔ P2, mesuré sur les meilleurs temps de qualification.
  let ecart = null;
  if (qualif?.lignes?.length >= 2) {
    const t1 = tempsEnSecondes(qualif.lignes[0].meilleur);
    const t2 = tempsEnSecondes(qualif.lignes[1].meilleur);
    if (t1 !== null && t2 !== null) ecart = Math.abs(t2 - t1);
  }

  // Distance courue : sert à détecter une course écourtée (< 75 %).
  let distancePct = 1;
  try {
    const c = await reference.circuit(gp.circuit_id);
    const prevus = c?.nb_tours;
    const faits = classes[0]?.tours;
    if (prevus && faits) distancePct = Math.min(1, faits / prevus);
  } catch { /* référentiel absent : on suppose la course complète */ }

  // Duel : gagnant de chaque paire d'équipiers.
  const duels = {};
  const parEquipe = new Map();
  for (const l of course.lignes) {
    if (!l.team_id) continue;
    if (!parEquipe.has(l.team_id)) parEquipe.set(l.team_id, []);
    parEquipe.get(l.team_id).push(l);
  }
  for (const [team, membres] of parEquipe) {
    if (membres.length !== 2) continue;
    const [a, b] = membres;
    const ordre = (x) => (x.classe ? x.position : 100 + (100 - (x.tours || 0)));
    duels[team] = ordre(a) <= ordre(b) ? a.driver_id : b.driver_id;
  }

  return {
    annule: false,
    ts,
    classement: course.lignes,
    top10,
    pole: qualif?.lignes?.[0]?.driver_id || null,
    pole_connue: Boolean(qualif?.lignes?.length),
    meilleur_tour: mt,
    meilleur_tour_connu: Boolean(mt),
    premier_abandon: premier,
    premier_abandon_connu: abandons.length > 0,
    nb_abandons: abandons.length,
    safety_car: safetyCar,
    safety_car_connu: scConnu,
    ecart_pole_p2: classeEcart(ecart),
    ecart_pole_p2_s: ecart,
    duels,
    duel_gagnant: null,        // renseigné par l'appelant selon le duel imposé
    distance_pct: distancePct,
  };
}
