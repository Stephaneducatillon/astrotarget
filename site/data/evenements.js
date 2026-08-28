/* =========================================================
   Agenda du club — source unique des évènements du site.
   Pour ajouter une séance : copier un bloc, changer les valeurs.
   date : "AAAA-MM-JJ" | heure : texte libre | type : observation
   | atelier | conference | initiation | sortie
   ========================================================= */
window.EVENEMENTS = [
  {
    date: "2026-09-12",
    heure: "21h00 – 00h00",
    titre: "Soirée d'observation de rentrée",
    type: "observation",
    lieu: "Site d'observation — à confirmer",
    resume:
      "Première sortie de la saison : prise en main des instruments du club, " +
      "repérage des constellations d'automne, observation de Saturne et de la " +
      "Voie lactée si le ciel le permet. Ouverte aux non-adhérents.",
    public: "Tout public, à partir de 8 ans",
  },
  {
    date: "2026-09-26",
    heure: "18h30 – 20h30",
    titre: "Atelier : régler et utiliser son télescope",
    type: "atelier",
    lieu: "MJC de Douai — salle du club",
    resume:
      "Collimation, mise en station, choix des oculaires, calcul du " +
      "grossissement utile. Venez avec votre instrument, même celui qui " +
      "dort dans un placard depuis des années.",
    public: "Adhérents et futurs adhérents",
  },
  {
    date: "2026-10-10",
    heure: "20h00 – 22h00",
    titre: "Conférence : la vie des étoiles",
    type: "conference",
    lieu: "MJC de Douai — grande salle",
    resume:
      "De la nébuleuse qui s'effondre à la naine blanche : le cycle complet " +
      "d'une étoile, illustré par les objets que l'on peut pointer soi-même " +
      "depuis le Douaisis.",
    public: "Tout public, entrée libre",
  },
  {
    date: "2026-10-24",
    heure: "20h30 – 23h30",
    titre: "Ciel profond d'automne",
    type: "observation",
    lieu: "Site d'observation — à confirmer",
    resume:
      "Nouvelle Lune : la meilleure fenêtre du mois pour la galaxie " +
      "d'Andromède (M31), le Double Amas de Persée et la nébuleuse " +
      "Dumbbell (M27).",
    public: "Adhérents",
  },
  {
    date: "2026-11-14",
    heure: "14h00 – 17h00",
    titre: "Initiation à l'astrophotographie",
    type: "initiation",
    lieu: "MJC de Douai — salle du club",
    resume:
      "Poses courtes, empilement, traitement : les bases pour obtenir une " +
      "première image du ciel profond avec un appareil photo et un simple " +
      "trépied.",
    public: "Adhérents, 12 places",
  },
  {
    date: "2026-12-05",
    heure: "19h00 – 22h00",
    titre: "Soirée Lune et planètes",
    type: "observation",
    lieu: "MJC de Douai — parvis",
    resume:
      "Observation de la Lune au premier quartier — cratères, mers et " +
      "terminateur — puis Jupiter et ses satellites galiléens. Séance " +
      "courte, idéale pour les familles.",
    public: "Tout public, à partir de 6 ans",
  },
];
