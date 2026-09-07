/**
 * Source OM — Open-Meteo. Entité `weather_forecast` (livrable 1 §3).
 * Appel sur `circuit.lat/lon`, granularité horaire.
 *
 * Règle d'affichage : toujours montrer l'heure DE LA PRÉVISION et l'heure de
 * RAFRAÎCHISSEMENT. Une prévision de pluie sans horodatage n'a aucune valeur.
 */

import { SOURCES, TTL } from '../config.js';
import { getJSON } from './http.js';
import { avecCache } from './cache.js';

const CHAMPS = [
  'temperature_2m', 'relative_humidity_2m', 'precipitation_probability',
  'precipitation', 'wind_speed_10m', 'wind_direction_10m',
  'cloud_cover', 'soil_temperature_0cm', 'surface_pressure',
].join(',');

/**
 * @param {number} lat @param {number} lon
 * @param {Date|null} autourDe  centre de la fenêtre (souvent la course)
 */
export function prevision(lat, lon, autourDe = null) {
  if (lat === null || lat === undefined || lon === null || lon === undefined) {
    return Promise.reject(new Error('Coordonnées du circuit inconnues.'));
  }
  const jours = joursNecessaires(autourDe);
  // TTL plus court à l'approche de l'échéance (livrable 1 §5).
  const ttl = jours <= 2 ? TTL.meteo : TTL.meteo_loin;
  const cle = `om:${lat.toFixed(3)}:${lon.toFixed(3)}:${jours}`;

  return avecCache(cle, ttl, async () => {
    const url = `${SOURCES.OM.base}?latitude=${lat}&longitude=${lon}`
      + `&hourly=${CHAMPS}&timezone=UTC&past_days=1&forecast_days=${jours}`;
    const d = await getJSON('OM', url);
    const t = d?.hourly?.time;
    if (!Array.isArray(t) || !t.length) throw new Error('Prévision vide.');
    return t.map((iso, i) => ({
      // Open-Meteo renvoie « 2026-09-07T14:00 » sans Z alors que timezone=UTC.
      heure_utc: `${iso}${iso.endsWith('Z') ? '' : ':00Z'}`.replace(/:00:00Z$/, ':00Z'),
      temperature_c: val(d.hourly.temperature_2m, i),
      humidite_pct: val(d.hourly.relative_humidity_2m, i),
      proba_pluie_pct: val(d.hourly.precipitation_probability, i),
      precipitation_mm: val(d.hourly.precipitation, i),
      vent_kmh: val(d.hourly.wind_speed_10m, i),
      vent_dir_deg: val(d.hourly.wind_direction_10m, i),
      nuages_pct: val(d.hourly.cloud_cover, i),
      // Proxy de température de piste : ce n'est PAS une mesure de piste.
      sol_c: val(d.hourly.soil_temperature_0cm, i),
      pression_hpa: val(d.hourly.surface_pressure, i),
    }));
  });
}

function val(arr, i) {
  const v = arr?.[i];
  return (v === null || v === undefined) ? null : Number(v);
}

function joursNecessaires(autourDe) {
  if (!autourDe) return 3;
  const j = Math.ceil((autourDe.getTime() - Date.now()) / 86400000) + 2;
  return Math.min(16, Math.max(2, j));
}

/** Point de prévision le plus proche d'une heure donnée. */
export function pourHeure(heures, cible) {
  if (!heures?.length || !cible) return null;
  const t = cible.getTime();
  let meilleur = null, ecart = Infinity;
  for (const h of heures) {
    const d = Math.abs(new Date(h.heure_utc).getTime() - t);
    if (d < ecart) { ecart = d; meilleur = h; }
  }
  // Au-delà de 90 min, la prévision ne décrit plus la session demandée.
  return ecart <= 90 * 60 * 1000 ? meilleur : null;
}

/** Fenêtre horaire autour d'une session, pour la vue « heure par heure ». */
export function fenetre(heures, cible, avant = 3, apres = 3) {
  if (!heures?.length || !cible) return [];
  const t = cible.getTime();
  const idx = heures.reduce((best, h, i) => {
    const d = Math.abs(new Date(h.heure_utc).getTime() - t);
    return d < best.d ? { d, i } : best;
  }, { d: Infinity, i: 0 }).i;
  return heures.slice(Math.max(0, idx - avant), idx + apres + 1);
}

/** Libellé synthétique — jamais une icône seule, toujours un texte. */
export function resume(p) {
  if (!p) return { texte: 'Prévision indisponible', icone: '—', pluie: false };
  const pluie = (p.proba_pluie_pct ?? 0) >= 50 || (p.precipitation_mm ?? 0) >= 0.3;
  if (pluie) return { texte: 'Pluie probable', icone: '🌧', pluie: true };
  if ((p.proba_pluie_pct ?? 0) >= 25) return { texte: 'Averses possibles', icone: '🌦', pluie: false };
  if ((p.nuages_pct ?? 0) >= 70) return { texte: 'Couvert', icone: '☁', pluie: false };
  if ((p.nuages_pct ?? 0) >= 30) return { texte: 'Voilé', icone: '⛅', pluie: false };
  return { texte: 'Dégagé', icone: '☀', pluie: false };
}
