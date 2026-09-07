/**
 * Persistance locale.
 *
 * La phase 3 du livrable 4 prévoit des comptes utilisateurs côté serveur.
 * Ce site est déployable en statique : le profil, les pronostics et les ligues
 * vivent donc dans le navigateur, et l'échange entre joueurs passe par un code
 * de partage. L'interface le dit franchement plutôt que de simuler un serveur.
 *
 * Ce que ce choix respecte quand même :
 *  - un pronostic est modifiable jusqu'à la fermeture, verrouillé après ;
 *  - l'historique des versions est conservé ;
 *  - la baseline est figée et horodatée dans le pronostic au moment de la
 *    soumission, jamais relue depuis un calcul ultérieur.
 */

const CLE = 'paddock:etat:v1';

const DEFAUT = {
  profil: null,
  prefs: { motifs: false, theme: 'auto', saison: null, demo: false },
  pronostics: {},   // gp_id → pronostic (avec `versions`)
  ligues: [],       // { code, nom, cree_ts, membres: [{id, pseudo, inscrit_ts}] }
  imports: {},      // code_ligue → { gp_id → [pronostics d'autres membres] }
  jokers_triples: {}, // saison → nombre utilisé
};

let etat = charger();
const abonnes = new Set();

function charger() {
  try {
    const brut = localStorage.getItem(CLE);
    if (!brut) return structuredClone(DEFAUT);
    return { ...structuredClone(DEFAUT), ...JSON.parse(brut) };
  } catch { return structuredClone(DEFAUT); }
}

function sauver() {
  try { localStorage.setItem(CLE, JSON.stringify(etat)); } catch { /* mode privé */ }
  abonnes.forEach((f) => f(etat));
}

export function sAbonner(f) { abonnes.add(f); return () => abonnes.delete(f); }
export function lire() { return etat; }

// ── Profil ─────────────────────────────────────────────────────────────────

export function profil() { return etat.profil; }

export function definirProfil(pseudo) {
  const p = String(pseudo || '').trim().slice(0, 24);
  if (!p) throw new Error('Un pseudo est nécessaire.');
  etat.profil = etat.profil
    ? { ...etat.profil, pseudo: p }
    : { id: idAleatoire(), pseudo: p, inscrit_ts: Date.now() };
  sauver();
  return etat.profil;
}

function idAleatoire() {
  const t = crypto.getRandomValues(new Uint8Array(8));
  return [...t].map((x) => x.toString(16).padStart(2, '0')).join('');
}

// ── Préférences ────────────────────────────────────────────────────────────

export function prefs() { return etat.prefs; }

export function definirPref(cle, valeur) {
  etat.prefs = { ...etat.prefs, [cle]: valeur };
  sauver();
}

// ── Pronostics ─────────────────────────────────────────────────────────────

export function pronostic(gpId) { return etat.pronostics[gpId] || null; }

export function tousPronostics() { return Object.values(etat.pronostics); }

/**
 * Enregistre un brouillon. Tant que la fenêtre est ouverte, chaque
 * enregistrement empile une version : l'historique n'est jamais écrasé.
 */
export function enregistrerPronostic(gpId, donnees) {
  const existant = etat.pronostics[gpId];
  if (existant?.verrouille) throw new Error('Pronostic verrouillé : la fenêtre est fermée.');

  const versions = existant?.versions || [];
  if (existant) {
    versions.push({ ts: existant.maj_ts || existant.soumis_ts, contenu: sansVersions(existant) });
  }

  etat.pronostics[gpId] = {
    ...existant, ...donnees,
    gp_id: gpId,
    maj_ts: Date.now(),
    versions: versions.slice(-20),
  };
  sauver();
  return etat.pronostics[gpId];
}

/** Soumission : fige la baseline et horodate. Après, plus de recalcul. */
export function soumettrePronostic(gpId, donnees, baselineGel) {
  const p = enregistrerPronostic(gpId, {
    ...donnees,
    soumis: true,
    soumis_ts: Date.now(),
    baseline_gel: baselineGel,
    baseline_ts: baselineGel?.ts || Date.now(),
  });
  if (donnees.joker?.multiplicateur === 3) consommerJokerTriple(donnees.saison);
  return p;
}

export function verrouiller(gpId) {
  const p = etat.pronostics[gpId];
  if (p) { p.verrouille = true; sauver(); }
}

export function supprimerPronostic(gpId) {
  delete etat.pronostics[gpId];
  sauver();
}

function sansVersions(p) {
  const { versions, ...reste } = p;
  return reste;
}

// ── Jokers « double conviction » (×3), 3 par saison, non rechargeables ─────

export function jokersTriplesRestants(saison) {
  const utilises = etat.jokers_triples[saison] || 0;
  return Math.max(0, 3 - utilises);
}

function consommerJokerTriple(saison) {
  etat.jokers_triples[saison] = (etat.jokers_triples[saison] || 0) + 1;
  sauver();
}

/** GP neutralisé : les jokers sont rendus (livrable 2 §7). */
export function rendreJoker(saison) {
  if (etat.jokers_triples[saison]) {
    etat.jokers_triples[saison] -= 1;
    sauver();
  }
}

// ── Ligues ─────────────────────────────────────────────────────────────────

export function ligues() { return etat.ligues; }

export function creerLigue(nom) {
  const p = profil();
  if (!p) throw new Error('Crée d\'abord un profil.');
  const code = codeLigue();
  const l = {
    code, nom: String(nom || 'Ma ligue').slice(0, 40), cree_ts: Date.now(),
    membres: [{ id: p.id, pseudo: p.pseudo, inscrit_ts: p.inscrit_ts }],
  };
  etat.ligues.push(l);
  sauver();
  return l;
}

export function rejoindreLigue(code, nom) {
  const p = profil();
  if (!p) throw new Error('Crée d\'abord un profil.');
  const propre = String(code || '').toUpperCase().replace(/[^A-Z0-9]/g, '').slice(0, 6);
  if (propre.length !== 6) throw new Error('Un code de ligue fait 6 caractères.');
  if (etat.ligues.some((l) => l.code === propre)) throw new Error('Tu es déjà dans cette ligue.');
  const l = {
    code: propre, nom: String(nom || `Ligue ${propre}`).slice(0, 40), cree_ts: Date.now(),
    membres: [{ id: p.id, pseudo: p.pseudo, inscrit_ts: p.inscrit_ts }],
  };
  etat.ligues.push(l);
  sauver();
  return l;
}

export function quitterLigue(code) {
  etat.ligues = etat.ligues.filter((l) => l.code !== code);
  delete etat.imports[code];
  sauver();
}

function codeLigue() {
  const alphabet = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'; // sans I, O, 0, 1
  const t = crypto.getRandomValues(new Uint8Array(6));
  return [...t].map((x) => alphabet[x % alphabet.length]).join('');
}

/**
 * Import du pronostic d'un autre membre, transmis par code de partage.
 * Règle du livrable 2 §8 : les pronostics des autres ne sont visibles
 * qu'après fermeture de leur fenêtre — c'est la vue Ligue qui l'applique.
 */
export function importerPronostic(codeLigue2, charge) {
  const l = etat.ligues.find((x) => x.code === codeLigue2);
  if (!l) throw new Error('Ligue inconnue.');
  if (!charge?.profil?.id || !charge?.pronostic?.gp_id) throw new Error('Code de partage illisible.');

  if (!l.membres.some((m) => m.id === charge.profil.id)) {
    l.membres.push(charge.profil);
  }
  const parLigue = etat.imports[codeLigue2] || (etat.imports[codeLigue2] = {});
  const gp = charge.pronostic.gp_id;
  const liste = parLigue[gp] || (parLigue[gp] = []);
  const i = liste.findIndex((x) => x.profil.id === charge.profil.id);
  if (i >= 0) liste[i] = charge; else liste.push(charge);
  sauver();
  return charge;
}

export function pronosticsLigue(code, gpId) {
  return etat.imports[code]?.[gpId] || [];
}

// ── Partage : encodage transportable ───────────────────────────────────────

export function encoderPartage(objet) {
  const json = JSON.stringify(objet);
  const octets = new TextEncoder().encode(json);
  let bin = '';
  octets.forEach((o) => { bin += String.fromCharCode(o); });
  return btoa(bin).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

export function decoderPartage(texte) {
  const b64 = String(texte).trim().replace(/-/g, '+').replace(/_/g, '/');
  const bin = atob(b64 + '='.repeat((4 - (b64.length % 4)) % 4));
  const octets = Uint8Array.from(bin, (c) => c.charCodeAt(0));
  return JSON.parse(new TextDecoder().decode(octets));
}

export function reinitialiser() {
  etat = structuredClone(DEFAUT);
  sauver();
}
