/**
 * Règle transverse (livrable 3) : toute heure est stockée en UTC et
 * convertie à l'affichage. Aucune heure ne s'affiche sans fuseau explicite.
 */

export const FUSEAU_LOCAL = Intl.DateTimeFormat().resolvedOptions().timeZone;

export function parseUTC(iso) {
  if (!iso) return null;
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? null : d;
}

/** Décalage en minutes entre deux fuseaux à une date donnée. */
function offsetMinutes(date, tz) {
  const fmt = new Intl.DateTimeFormat('en-US', {
    timeZone: tz, hour12: false,
    year: 'numeric', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit', second: '2-digit',
  });
  const p = Object.fromEntries(fmt.formatToParts(date).map((x) => [x.type, x.value]));
  const asUTC = Date.UTC(+p.year, +p.month - 1, +p.day, +p.hour % 24, +p.minute, +p.second);
  return (asUTC - date.getTime()) / 60000;
}

/** Vrai si le circuit n'est pas dans le fuseau de l'utilisateur → badge. */
export function fuseauDifferent(date, tzCircuit) {
  if (!date || !tzCircuit || tzCircuit === FUSEAU_LOCAL) return false;
  try {
    return offsetMinutes(date, tzCircuit) !== offsetMinutes(date, FUSEAU_LOCAL);
  } catch { return false; }
}

export function heure(date, tz = FUSEAU_LOCAL) {
  if (!date) return '—';
  return new Intl.DateTimeFormat('fr-FR', { timeZone: tz, hour: '2-digit', minute: '2-digit' }).format(date);
}

export function jourCourt(date, tz = FUSEAU_LOCAL) {
  if (!date) return '—';
  return new Intl.DateTimeFormat('fr-FR', { timeZone: tz, weekday: 'short' })
    .format(date).replace('.', '').toUpperCase();
}

export function dateLongue(date, tz = FUSEAU_LOCAL) {
  if (!date) return '—';
  return new Intl.DateTimeFormat('fr-FR', {
    timeZone: tz, weekday: 'long', day: 'numeric', month: 'long',
  }).format(date);
}

export function dateHeure(date, tz = FUSEAU_LOCAL) {
  if (!date) return '—';
  return new Intl.DateTimeFormat('fr-FR', {
    timeZone: tz, day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit',
  }).format(date);
}

/** Nom court du fuseau, à accoler à toute heure affichée. */
export function nomFuseau(date, tz = FUSEAU_LOCAL) {
  try {
    const parts = new Intl.DateTimeFormat('fr-FR', { timeZone: tz, timeZoneName: 'short' })
      .formatToParts(date || new Date());
    return parts.find((p) => p.type === 'timeZoneName')?.value || tz;
  } catch { return tz; }
}

/** Compte à rebours formaté HH:MM:SS (ou J-n au-delà de 48 h). */
export function rebours(cible, maintenant = new Date()) {
  if (!cible) return { texte: '—', ms: 0, passe: true };
  const ms = cible.getTime() - maintenant.getTime();
  if (ms <= 0) return { texte: 'en cours ou terminée', ms, passe: true };
  const s = Math.floor(ms / 1000);
  const j = Math.floor(s / 86400);
  const h = Math.floor((s % 86400) / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  if (j >= 2) return { texte: `${j} jours ${String(h).padStart(2, '0')} h`, ms, passe: false };
  const hh = String(j * 24 + h).padStart(2, '0');
  return { texte: `${hh}:${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`, ms, passe: false };
}

/** « màj il y a 22 min » — obligatoire sur toute donnée live ou météo. */
export function depuis(ts, maintenant = Date.now()) {
  if (!ts) return 'fraîcheur inconnue';
  const s = Math.max(0, Math.floor((maintenant - ts) / 1000));
  if (s < 60) return 'màj il y a moins d\'une minute';
  const m = Math.floor(s / 60);
  if (m < 60) return `màj il y a ${m} min`;
  const h = Math.floor(m / 60);
  if (h < 24) return `màj il y a ${h} h`;
  return `màj il y a ${Math.floor(h / 24)} j`;
}

export function memeJour(a, b, tz = FUSEAU_LOCAL) {
  if (!a || !b) return false;
  const f = (d) => new Intl.DateTimeFormat('fr-CA', { timeZone: tz, dateStyle: 'short' }).format(d);
  return f(a) === f(b);
}
