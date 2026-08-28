/* =========================================================
   Configuration du site — le seul fichier à éditer pour
   passer de la démo locale à un vrai espace membre partagé.
   ========================================================= */
window.CONFIG = {
  /* Nom affiché dans les en-têtes et le pied de page. */
  nomClub: "Club d'astronomie",
  nomStructure: "MJC de Douai",

  /* Adresse de contact utilisée sur la page « Adhérer ». */
  email: "contact@example.org", // À COMPLÉTER

  /* ---------------------------------------------------------------
     Mode de l'espace membre.

     "local"    : tout reste dans le navigateur de chaque visiteur
                  (localStorage). Aucun serveur, aucun compte à créer,
                  mais les comptes, documents et messages ne sont PAS
                  partagés entre les personnes. Sert à essayer le site.

     "supabase" : comptes, base documentaire et chat réellement
                  partagés, hébergés sur Supabase (offre gratuite).
                  Renseignez les deux valeurs ci-dessous, puis suivez
                  site/SUPABASE.md pour créer les tables.
     --------------------------------------------------------------- */
  modeMembre: "local",

  supabase: {
    url: "",      // ex. "https://xxxxxxxx.supabase.co"
    cleAnon: "",  // clé « anon public » du projet Supabase
  },

  /* Code remis aux adhérents pour créer leur compte.
     En mode "local" il est vérifié dans le navigateur : il filtre les
     curieux, il ne protège rien. En mode "supabase", la vraie
     protection vient des règles RLS décrites dans SUPABASE.md. */
  codeInscription: "ORION2026",

  /* Salons du chat entre membres. */
  salons: [
    { id: "general",   nom: "Général" },
    { id: "sorties",   nom: "Sorties & météo" },
    { id: "materiel",  nom: "Matériel" },
    { id: "images",    nom: "Nos images" },
  ],
};
