import { COUNTRIES } from './normalize.js';
import { MAKES } from './makes.js';
import { Search, isNative } from './aggregator.js';

// ---------- Stockage local ----------
const store = {
  get(key, fallback) {
    try { const v = localStorage.getItem('ar.' + key); return v ? JSON.parse(v) : fallback; } catch { return fallback; }
  },
  set(key, value) {
    try { localStorage.setItem('ar.' + key, JSON.stringify(value)); } catch { /* stockage indisponible */ }
  },
};

const $ = (sel) => document.querySelector(sel);
const fmtInt = (n) => (n == null ? '' : n.toLocaleString('fr-FR'));
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

let favorites = store.get('favorites', []);
let savedSearches = store.get('saved', []);
let selectedCountries = new Set(store.get('countries', Object.keys(COUNTRIES)));
let demoMode = store.get('demo', !isNative());
let currentSearch = null;

function toast(msg) {
  const t = $('#toast');
  t.textContent = msg;
  t.classList.add('show');
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => t.classList.remove('show'), 2200);
}

// ---------- Onglets ----------
function showView(name) {
  document.querySelectorAll('.view').forEach((v) => v.classList.toggle('active', v.id === `view-${name}`));
  document.querySelectorAll('.tabbar button').forEach((b) => b.classList.toggle('active', b.dataset.view === name));
  if (name === 'favorites') renderFavorites();
  if (name === 'saved') renderSaved();
  window.scrollTo(0, 0);
}
document.querySelectorAll('.tabbar button').forEach((b) => b.addEventListener('click', () => showView(b.dataset.view)));

// ---------- Formulaire ----------
function renderCountryChips() {
  const box = $('#country-chips');
  box.innerHTML = '';
  for (const c of Object.values(COUNTRIES)) {
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = 'chip' + (selectedCountries.has(c.code) ? ' on' : '');
    chip.textContent = `${c.flag} ${c.name}`;
    chip.addEventListener('click', () => {
      if (selectedCountries.has(c.code)) selectedCountries.delete(c.code); else selectedCountries.add(c.code);
      store.set('countries', [...selectedCountries]);
      renderCountryChips();
    });
    box.appendChild(chip);
  }
}

function fillMakes() {
  $('#make-list').innerHTML = Object.keys(MAKES).map((m) => `<option value="${esc(m)}">`).join('');
}
function fillModels() {
  const key = Object.keys(MAKES).find((m) => m.toLowerCase() === $('#make').value.trim().toLowerCase());
  $('#model-list').innerHTML = (MAKES[key] || []).map((m) => `<option value="${esc(m)}">`).join('');
}
$('#make').addEventListener('input', fillModels);

const FIELDS = ['make', 'model', 'priceMin', 'priceMax', 'yearMin', 'yearMax', 'kmMax', 'fuel', 'gearbox', 'keywords', 'sort'];
const NUMERIC = new Set(['priceMin', 'priceMax', 'yearMin', 'yearMax', 'kmMax']);

function readCriteria() {
  const c = { countries: Object.keys(COUNTRIES).filter((k) => selectedCountries.has(k)) };
  for (const f of FIELDS) {
    const v = $('#' + f).value.trim();
    c[f] = NUMERIC.has(f) ? (v ? parseInt(v, 10) : null) : v;
  }
  return c;
}

function writeCriteria(c) {
  for (const f of FIELDS) $('#' + f).value = c[f] ?? '';
  selectedCountries = new Set(c.countries);
  renderCountryChips();
  fillModels();
}

function describe(c) {
  const parts = [[c.make, c.model].filter(Boolean).join(' ') || 'Toutes marques'];
  if (c.priceMin || c.priceMax) parts.push(`${c.priceMin ? fmtInt(c.priceMin) : '0'} – ${c.priceMax ? fmtInt(c.priceMax) : '∞'} €`);
  if (c.yearMin) parts.push(`≥ ${c.yearMin}`);
  if (c.kmMax) parts.push(`≤ ${fmtInt(c.kmMax)} km`);
  if (c.fuel) parts.push(c.fuel);
  if (c.gearbox) parts.push(c.gearbox);
  return parts.join(' · ');
}

// ---------- Cartes d'annonces ----------
function isFav(id) { return favorites.some((f) => f.id === id); }

function toggleFav(ad, btn) {
  if (isFav(ad.id)) {
    favorites = favorites.filter((f) => f.id !== ad.id);
    toast('Retiré des favoris');
  } else {
    favorites.unshift(ad);
    toast('Ajouté aux favoris');
  }
  store.set('favorites', favorites);
  if (btn) btn.classList.toggle('on', isFav(ad.id));
}

async function openAd(ad) {
  if (!ad.url) { toast('Annonce de démonstration : pas de lien'); return; }
  const browser = globalThis.Capacitor?.Plugins?.Browser;
  if (browser) await browser.open({ url: ad.url });
  else window.open(ad.url, '_blank', 'noopener');
}

function adCard(ad, onFavChange) {
  const el = document.createElement('article');
  el.className = 'ad';
  const country = COUNTRIES[ad.country] || { flag: '🏳', name: ad.country };
  const specs = [ad.year, ad.km != null ? `${fmtInt(ad.km)} km` : null, ad.fuel, ad.gearbox].filter(Boolean).join(' · ');
  el.innerHTML = `
    <div class="photo" ${ad.image ? `style="background-image:url('${esc(ad.image)}')"` : ''}>${ad.image ? '' : '🚗'}</div>
    ${ad.demo ? '<span class="badge-demo">DÉMO</span>' : ''}
    <div class="info">
      <div class="title">${esc(ad.title)}</div>
      <div class="price">${ad.price != null ? fmtInt(ad.price) + ' €' : 'Prix sur demande'}</div>
      <div class="specs">${esc(specs)}</div>
      <div class="where">${country.flag} ${esc(ad.city || country.name)}</div>
      <div class="src">via ${esc(ad.source)}</div>
    </div>
    <button class="fav ${isFav(ad.id) ? 'on' : ''}" aria-label="Favori">♥</button>`;
  el.querySelector('.fav').addEventListener('click', (e) => {
    e.stopPropagation();
    toggleFav(ad, e.currentTarget);
    onFavChange?.();
  });
  el.addEventListener('click', () => openAd(ad));
  return el;
}

// ---------- Recherche ----------
function renderResults() {
  const s = currentSearch;
  const box = $('#results');
  box.innerHTML = '';
  if (!s.ads.length) {
    box.innerHTML = '<div class="empty">Aucune annonce trouvée pour ces critères.</div>';
  } else {
    const frag = document.createDocumentFragment();
    for (const ad of s.ads) frag.appendChild(adCard(ad));
    box.appendChild(frag);
  }
  const total = s.channels.reduce((sum, ch) => sum + (ch.total || 0), 0);
  $('#status-line').textContent = `${fmtInt(total)} annonce${total > 1 ? 's' : ''} · ${s.ads.length} affichée${s.ads.length > 1 ? 's' : ''}`;
  $('#channel-status').innerHTML = s.channels.map((ch) => {
    const c = COUNTRIES[ch.country];
    return ch.error
      ? `<span class="ch err" title="${esc(ch.error)}">${c.flag} indisponible</span>`
      : `<span class="ch ok">${c.flag} ${fmtInt(ch.total ?? 0)}</span>`;
  }).join('');
  $('#results-head').classList.remove('hidden');
  $('#load-more').classList.toggle('hidden', !s.hasMore);

  if (s.channels.length && s.channels.every((ch) => ch.error)) {
    box.innerHTML = `<div class="empty">Impossible de joindre les sites d'annonces.<br>Vérifiez votre connexion internet${isNative() ? '' : ', ou activez le mode démonstration dans Réglages'}.</div>`;
  }
}

async function runSearch(criteria) {
  if (!criteria.countries.length) { toast('Sélectionnez au moins un pays'); return; }
  currentSearch = new Search(criteria, { demoMode });
  $('#results').innerHTML = '';
  $('#results-head').classList.add('hidden');
  $('#load-more').classList.add('hidden');
  $('#loader').classList.remove('hidden');
  try {
    await currentSearch.loadMore();
  } finally {
    $('#loader').classList.add('hidden');
  }
  renderResults();
  $('#results-head').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

$('#search-form').addEventListener('submit', (e) => {
  e.preventDefault();
  document.activeElement?.blur();
  runSearch(readCriteria());
});

$('#load-more').addEventListener('click', async () => {
  const btn = $('#load-more');
  btn.disabled = true;
  btn.textContent = 'Chargement…';
  const y = window.scrollY;
  await currentSearch.loadMore();
  renderResults();
  window.scrollTo(0, y);
  btn.disabled = false;
  btn.textContent = "Afficher plus d'annonces";
});

$('#save-search').addEventListener('click', () => {
  if (!currentSearch) return;
  const c = currentSearch.criteria;
  const key = JSON.stringify(c);
  if (savedSearches.some((s) => JSON.stringify(s.criteria) === key)) { toast('Recherche déjà enregistrée'); return; }
  savedSearches.unshift({ criteria: c, savedAt: Date.now() });
  store.set('saved', savedSearches);
  toast('Recherche enregistrée');
});

// ---------- Favoris ----------
function renderFavorites() {
  const box = $('#favorites');
  box.innerHTML = '';
  if (!favorites.length) {
    box.innerHTML = '<div class="empty">Aucun favori pour l\'instant.<br>Touchez ♥ sur une annonce pour la garder ici.</div>';
    return;
  }
  for (const ad of favorites) box.appendChild(adCard(ad, renderFavorites));
}

// ---------- Recherches enregistrées ----------
function renderSaved() {
  const box = $('#saved');
  box.innerHTML = '';
  if (!savedSearches.length) {
    box.innerHTML = '<div class="empty">Aucune recherche enregistrée.<br>Lancez une recherche puis touchez « Enregistrer cette recherche ».</div>';
    return;
  }
  savedSearches.forEach((s, i) => {
    const el = document.createElement('div');
    el.className = 'card saved-item';
    const flags = s.criteria.countries.map((c) => COUNTRIES[c]?.flag).join(' ');
    el.innerHTML = `
      <div class="desc"><b>${esc(describe(s.criteria))}</b><small>${flags}</small></div>
      <button class="btn primary small">Lancer</button>
      <button class="btn ghost small" aria-label="Supprimer">✕</button>`;
    const [run, del] = el.querySelectorAll('button');
    run.addEventListener('click', () => {
      writeCriteria(s.criteria);
      showView('search');
      runSearch(s.criteria);
    });
    del.addEventListener('click', () => {
      savedSearches.splice(i, 1);
      store.set('saved', savedSearches);
      renderSaved();
    });
    box.appendChild(el);
  });
}

// ---------- Réglages ----------
$('#demo-mode').checked = demoMode;
$('#demo-mode').addEventListener('change', (e) => {
  demoMode = e.target.checked;
  store.set('demo', demoMode);
  toast(demoMode ? 'Mode démonstration activé' : 'Annonces réelles activées');
});
$('#clear-data').addEventListener('click', () => {
  if (!confirm('Effacer tous les favoris et recherches enregistrées ?')) return;
  favorites = [];
  savedSearches = [];
  store.set('favorites', favorites);
  store.set('saved', savedSearches);
  toast('Données effacées');
});

// ---------- Init ----------
renderCountryChips();
fillMakes();
fillModels();
