// Moteur de regles - traduction des donnees collectees en niveaux familiaux.
//
// Regle de conception centrale : une source en echec ne vaut PAS "vert".
// Elle est `indeterminee`, et la machine a etats la laisse alors intacte.
// Confondre "pas d'alerte" et "pas de donnee" est precisement le defaut qui
// rend une veille dangereuse.
//
// Les regles reprennent le tableau "Regles de passage aux niveaux familiaux"
// du document source.

import { SEUILS } from './config.js';
import { evaluerNiveau } from './sources/hubeau.js';

export const REGLES = [
  { id: 'mf_vigilance',  source: 'vigilance', nom: 'Vigilance Meteo-France' },
  { id: 'ecowatt',       source: 'ecowatt',   nom: 'Ecowatt' },
  { id: 'scarpe_niveau', source: 'hubeau',    nom: 'Scarpe - hauteur' },
  { id: 'vigicrues',     source: 'vigicrues', nom: 'Vigicrues - Scarpe' },
];

/**
 * @param {Map<string, {ok: boolean, data?: any, error?: any}>} collectes
 * @param {Map<string, {max: number|null, n: number}>} fenetres  par code station
 * @returns {Array<{rule_id, nom, level, label, indetermine}>}
 */
export function evaluerRegles(collectes, fenetres) {
  return REGLES.map((regle) => {
    const c = collectes.get(regle.source);
    if (!c || !c.ok) {
      return { rule_id: regle.id, nom: regle.nom, indetermine: true, level: null, label: null };
    }
    const evalue = EVALUATEURS[regle.id](c.data, fenetres);
    return { rule_id: regle.id, nom: regle.nom, indetermine: false, ...evalue };
  });
}

const EVALUATEURS = {
  // Orange si vigilance orange sur 59 ou 62 pour un phenomene du perimetre,
  // rouge si rouge. Le jaune ne remonte pas (deja neutralise au parsing).
  mf_vigilance: (d) => ({ level: d.pire, label: d.libelle }),

  // Orange ou rouge si le signal du jour ou d'un des 3 jours suivants l'est.
  ecowatt: (d) => ({ level: d.pire, label: d.libelle }),

  // Orange si une station depasse son maximum des 30 derniers jours.
  scarpe_niveau: (d, fenetres) => {
    if (!d.configure || d.stations.length === 0) {
      return { level: 'vert', label: null, indetermine: false, note: 'stations non configurees' };
    }
    const depassements = [];
    let pire = 'vert';
    let insuffisant = true;

    for (const st of d.stations) {
      const f = fenetres.get(`${st.code}:${st.metric}`);
      const r = evaluerNiveau(st.value, f);
      if (!r.insuffisant) insuffisant = false;
      if (r.depassement) {
        pire = 'orange';
        depassements.push(
          `${st.nom} ${(st.value / 1000).toFixed(2)} m (max ${SEUILS.hydro_fenetre_jours} j : ${(r.max / 1000).toFixed(2)} m)`,
        );
      }
    }
    return {
      level: pire,
      label: depassements.length ? depassements.join(' ; ') : null,
      // Historique trop court : on ne pretend pas surveiller ce qu'on ne peut
      // pas encore evaluer. Signale dans le tableau de bord.
      note: insuffisant ? 'historique insuffisant' : null,
    };
  },

  // Rouge si le troncon surveille passe en rouge, orange en orange.
  vigicrues: (d) => ({ level: d.pire, label: d.libelle }),
};
