// PWA de veille - client.
//
// Principe directeur : ne jamais afficher un etat rassurant sans dire d'ou il
// vient ni quand il date. Un ecran vert sur des donnees de la veille est pire
// qu'un ecran d'erreur.

const API = localStorage.getItem('api') ?? '';        // ex. https://veille-risques.xxx.workers.dev
const TOKEN = localStorage.getItem('token') ?? '';
const CACHE_SNAP = 'snapshot';

const $ = (s) => document.querySelector(s);
const NIVEAUX = { vert: 'Vert', orange: 'Orange', rouge: 'Rouge' };

const CHECKLIST = {
  orange: 'Téléphones chargés · eau · lampes · vérifier la ligne du père.',
  rouge: 'Appliquer les consignes du plan familial. Rester joignable. Point de rendez-vous si évacuation.',
};

const LIENS = [
  ['Enedis — coupures en cours', 'Pas d\'API : consultation directe', 'https://www.enedis.fr/panne-et-coupure-delectricite'],
  ['Vigicrues', 'Carte des tronçons', 'https://www.vigicrues.gouv.fr/'],
  ['Météo-France — vigilance', 'Carte officielle', 'https://vigilance.meteofrance.fr/fr'],
  ['Ecowatt', 'Signal électrique', 'https://www.monecowatt.fr/'],
  ['Géorisques — mon adresse', 'Rapport de risques', 'https://www.georisques.gouv.fr/'],
];

// --- Chargement -------------------------------------------------------------

async function charger() {
  try {
    const url = `${API}/api/snapshot${TOKEN ? `?token=${encodeURIComponent(TOKEN)}` : ''}`;
    const res = await fetch(url, { cache: 'no-store' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const snap = await res.json();
    localStorage.setItem(CACHE_SNAP, JSON.stringify({ snap, recu: Date.now() / 1000 }));
    rendre(snap, false);
  } catch (err) {
    // Hors ligne : on sert le dernier instantane connu, clairement date.
    const cache = localStorage.getItem(CACHE_SNAP);
    if (cache) {
      rendre(JSON.parse(cache).snap, true);
    } else {
      $('#niveau').textContent = 'Indisponible';
      $('#causes').textContent = 'Aucune donnée en cache. Vérifier la connexion et la configuration.';
    }
  }
}

// --- Rendu ------------------------------------------------------------------

function rendre(snap, horsLigne) {
  if (snap?.vide) {
    $('#niveau').textContent = 'En attente';
    $('#causes').textContent = 'Aucune collecte effectuée pour le moment.';
    return;
  }

  const now = Date.now() / 1000;
  const niveau = snap.niveau ?? 'vert';

  $('#banniere').className = niveau;
  $('#niveau').className = niveau;
  $('#niveau').textContent = NIVEAUX[niveau] ?? niveau;

  const causes = (snap.regles ?? [])
    .filter((r) => !r.indetermine && r.level !== 'vert' && r.label)
    .map((r) => r.label);
  $('#causes').textContent = causes.length ? causes.join(' · ') : 'Aucune alerte en cours.';

  const cl = $('#checklist');
  if (CHECKLIST[niveau]) { cl.textContent = CHECKLIST[niveau]; cl.hidden = false; }
  else cl.hidden = true;

  $('#horloge').textContent = `maj ${ago(now - snap.ts)}`;

  // Bandeau de confiance.
  const avertissements = [];
  if (horsLigne) avertissements.push(`Hors ligne — affichage du dernier état connu (${ago(now - snap.ts)}).`);
  const muettes = Object.values(snap.sources ?? {}).filter((s) => s.perimee).map((s) => s.nom);
  if (muettes.length) avertissements.push(`<b>Sources sans données fraîches :</b> ${muettes.join(', ')}. Un écran vert ne garantit rien pour ces sources.`);
  const indet = (snap.regles ?? []).filter((r) => r.indetermine).length;
  if (indet && !muettes.length) avertissements.push(`<b>${indet} règle(s) non évaluable(s)</b> ce cycle.`);

  const bandeau = $('#confiance');
  bandeau.innerHTML = avertissements.join('<br>');
  bandeau.classList.toggle('actif', avertissements.length > 0);

  // Onglet Etat : une carte par regle.
  $('#p-etat').innerHTML = (snap.regles ?? []).map((r) => {
    const classe = r.indetermine ? 'inconnu' : r.level;
    const etiq = r.indetermine ? 'inconnu' : NIVEAUX[r.level] ?? r.level;
    const lignes = [];
    if (r.indetermine) lignes.push('<p>Source indisponible — dernier état connu conservé.</p>');
    else if (r.label) lignes.push(`<p>${echapper(r.label)}</p>`);
    if (r.note) lignes.push(`<p class="note">${echapper(r.note)}</p>`);
    if (r.level !== 'vert' && r.since) lignes.push(`<p class="age">depuis ${ago(now - r.since)}</p>`);
    return `<article class="carte ${classe}">
      <h3><span>${echapper(r.nom)}</span><span class="etiquette ${classe}">${etiq}</span></h3>
      ${lignes.join('')}
    </article>`;
  }).join('');

  // Onglet Sources : l'age est la donnee, pas un detail.
  $('#p-src').innerHTML = Object.entries(snap.sources ?? {}).map(([id, s]) => {
    const perimee = s.perimee;
    const age = s.jamais ? 'jamais reçue' : ago(now - s.last_ok);
    return `<article class="carte ${perimee ? 'orange' : 'vert'}">
      <h3><span>${echapper(s.nom)}</span>
        <span class="age ${perimee ? 'perimee' : ''}">${age}</span></h3>
      ${s.last_error ? `<p>${echapper(s.last_error)}</p>` : ''}
      ${s.fail_streak ? `<p>${s.fail_streak} échec(s) consécutif(s)</p>` : ''}
    </article>`;
  }).join('') || '<p>Aucune source enregistrée.</p>';
}

// --- Utilitaires ------------------------------------------------------------

function ago(secondes) {
  const s = Math.max(0, Math.round(secondes));
  if (s < 90) return `il y a ${s} s`;
  const m = Math.round(s / 60);
  if (m < 90) return `il y a ${m} min`;
  const h = Math.round(m / 60);
  if (h < 36) return `il y a ${h} h`;
  return `il y a ${Math.round(h / 24)} j`;
}

function echapper(s) {
  const d = document.createElement('div');
  d.textContent = String(s ?? '');
  return d.innerHTML;
}

// --- Onglets ----------------------------------------------------------------

for (const bouton of document.querySelectorAll('[role="tab"]')) {
  bouton.addEventListener('click', () => {
    for (const b of document.querySelectorAll('[role="tab"]')) {
      const actif = b === bouton;
      b.setAttribute('aria-selected', String(actif));
      document.getElementById(b.getAttribute('aria-controls')).hidden = !actif;
    }
  });
}

$('#liens-hors').innerHTML = LIENS.map(([titre, sous, href]) =>
  `<li><a href="${href}" target="_blank" rel="noopener">${titre}<small>${sous}</small></a></li>`).join('');

// --- Cycle de vie -----------------------------------------------------------

charger();
setInterval(charger, 5 * 60 * 1000);
document.addEventListener('visibilitychange', () => { if (!document.hidden) charger(); });
window.addEventListener('online', charger);

if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('sw.js').catch(() => {});
}
