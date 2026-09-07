/** Micro-helpers de rendu. Pas de framework : le site doit tourner sans build. */

export function h(tag, attrs = {}, ...enfants) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'dataset') Object.assign(el.dataset, v);
    else if (k === 'style' && typeof v === 'object') appliquerStyle(el, v);
    else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? '' : String(v));
  }
  for (const e of enfants.flat(Infinity)) {
    if (e === null || e === undefined || e === false) continue;
    el.append(e instanceof Node ? e : document.createTextNode(String(e)));
  }
  return el;
}

/**
 * Les propriétés personnalisées (`--c`) ne passent pas par Object.assign :
 * el.style n'expose que les propriétés CSS connues, et une clé `--c` y est
 * ignorée en silence. Il faut setProperty. C'est ce qui portait les couleurs
 * d'écurie — sans ça, toutes les pastilles tombaient sur la couleur de repli.
 */
function appliquerStyle(el, styles) {
  for (const [prop, val] of Object.entries(styles)) {
    if (val === null || val === undefined || val === false) continue;
    if (prop.startsWith('--')) el.style.setProperty(prop, String(val));
    else el.style[prop] = val;
  }
}

export const $ = (sel, racine = document) => racine.querySelector(sel);
export const $$ = (sel, racine = document) => [...racine.querySelectorAll(sel)];

export function vider(el) { while (el.firstChild) el.removeChild(el.firstChild); return el; }

/** Squelette de chargement — jamais de spinner plein écran (livrable 3). */
export function squelette(lignes = 3, classe = '') {
  return h('div', { class: `skel ${classe}`, 'aria-busy': 'true', 'aria-label': 'Chargement' },
    Array.from({ length: lignes }, (_, i) =>
      h('div', { class: 'skel-ligne', style: { width: `${100 - i * 12}%` } })));
}

/** État d'erreur de source : message explicite, jamais de page blanche. */
export function erreurSource(codeSource, detail, onRetry) {
  return h('div', { class: 'etat etat--erreur', role: 'alert' },
    h('div', { class: 'etat-titre' }, `Données ${codeSource} indisponibles`),
    h('p', { class: 'etat-detail' }, detail || 'La source n\'a pas répondu.'),
    onRetry ? h('button', { class: 'btn btn--fin', onclick: onRetry }, 'Réessayer') : null);
}

export function etatVide(titre, detail, action) {
  return h('div', { class: 'etat etat--vide' },
    h('div', { class: 'etat-titre' }, titre),
    detail ? h('p', { class: 'etat-detail' }, detail) : null,
    action || null);
}

/** Badge de source : la traçabilité est un critère de Definition of Done. */
export function badgeSource(code, titre) {
  return h('span', { class: `src src--${String(code).toLowerCase()}`, title: titre || `Source : ${code}` }, code);
}

export function badgeFraicheur(texte) {
  return h('span', { class: 'fraicheur' }, texte);
}

export function annonce(message) {
  const live = document.getElementById('annonceur');
  if (live) live.textContent = message;
}
