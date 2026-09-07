/**
 * Données de référence MAN (livrable 1 §2) : fiche circuit statique.
 * Peu nombreuses, changeant rarement — un fichier versionné suffit,
 * pas besoin de scraper. Statut « à valider » tant que la checklist du
 * livrable 1 §7 n'est pas cochée.
 */

let cache = null;

export async function circuits() {
  if (cache) return cache;
  const rep = await fetch(new URL('../../data/circuits.json', import.meta.url));
  if (!rep.ok) throw new Error('Référentiel circuits illisible.');
  cache = await rep.json();
  return cache;
}

export async function circuit(id) {
  const d = await circuits();
  return d.circuits[id] || null;
}

/** Fenêtre de calcul des statistiques historiques, à afficher systématiquement. */
export async function fenetreStats() {
  const d = await circuits();
  return d._meta.fenetre_stats;
}

export const LIBELLES = {
  type_appui: { faible: 'Faible', moyen: 'Moyen', eleve: 'Élevé' },
  abrasivite: { faible: 'Faible', moyenne: 'Moyenne', forte: 'Forte' },
  sens: { horaire: 'Sens horaire', antihoraire: 'Sens antihoraire' },
};
