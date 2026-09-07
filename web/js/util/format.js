import { ECURIES } from '../config.js';

export function nombre(v, dec = 0) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  return new Intl.NumberFormat('fr-FR', { minimumFractionDigits: dec, maximumFractionDigits: dec }).format(v);
}

export function pourcent(v, dec = 0) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  return `${nombre(v * 100, dec)} %`;
}

export function secondes(v, dec = 3) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  return `${nombre(v, dec).replace(',', ',')} s`;
}

/** ms → 1:23.456 */
export function chrono(ms) {
  if (!ms && ms !== 0) return '—';
  const total = ms / 1000;
  const m = Math.floor(total / 60);
  const s = total - m * 60;
  const sStr = s.toFixed(3).padStart(6, '0').replace('.', ',');
  return m > 0 ? `${m}:${sStr}` : sStr;
}

export function delta(v, dec = 3) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  const signe = v > 0 ? '+' : v < 0 ? '−' : '';
  return `${signe}${nombre(Math.abs(v), dec)}`;
}

export function ecurie(id) {
  const cle = String(id || '').toLowerCase().replace(/[^a-z_]/g, '');
  return ECURIES[cle] || { nom: id || 'Écurie inconnue', hex: '#8b8f98', motif: 'solide' };
}

export function couleurEcurie(id) { return ecurie(id).hex; }

/** Nom compact : « M. Verstappen ». */
export function nomPilote(p) {
  if (!p) return '—';
  const prenom = p.prenom || p.givenName || '';
  const nom = p.nom || p.familyName || '';
  return prenom ? `${prenom[0]}. ${nom}` : nom;
}

export function nomPiloteLong(p) {
  if (!p) return '—';
  return `${p.prenom || ''} ${p.nom || ''}`.trim() || p.code || p.id || '—';
}

export function ordinal(n) {
  return n === 1 ? '1er' : `${n}e`;
}

export function pluriel(n, singulier, plur) {
  return `${n} ${n > 1 ? (plur || `${singulier}s`) : singulier}`;
}
