// Agrège les résultats des différentes sources, pays par pays, avec pagination indépendante.

import * as autoscout24 from './sources/autoscout24.js';
import * as standvirtual from './sources/standvirtual.js';
import * as demo from './sources/demo.js';
import { matchesCriteria, sortAds } from './normalize.js';

const REAL_SOURCES = [autoscout24, standvirtual];

/** Requête HTTP : natif (pas de CORS) dans l'APK, fetch classique sinon. */
export async function httpGet(url) {
  const cap = globalThis.Capacitor;
  const headers = {
    'Accept': 'text/html,application/xhtml+xml',
    'Accept-Language': 'fr-FR,fr;q=0.9,en;q=0.8',
    'User-Agent': 'Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Mobile Safari/537.36',
  };
  if (cap?.isNativePlatform?.() && cap.Plugins?.CapacitorHttp) {
    const res = await cap.Plugins.CapacitorHttp.get({ url, headers, responseType: 'text' });
    if (res.status >= 400) throw new Error(`HTTP ${res.status}`);
    return typeof res.data === 'string' ? res.data : JSON.stringify(res.data);
  }
  const res = await fetch(url, { headers: { 'Accept': headers.Accept } });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.text();
}

export function isNative() {
  return !!globalThis.Capacitor?.isNativePlatform?.();
}

/** Une recherche = un ensemble de « canaux » (source × pays) paginés séparément. */
export class Search {
  constructor(criteria, { demoMode = false, http = httpGet } = {}) {
    this.criteria = criteria;
    this.http = http;
    const sources = demoMode ? [demo] : REAL_SOURCES;
    this.channels = [];
    for (const country of criteria.countries) {
      const src = sources.find((s) => s.countries.includes(country));
      if (src) this.channels.push({ src, country, page: 0, hasMore: true, total: null, error: null });
    }
    this.ads = [];
    this.seen = new Set();
  }

  get hasMore() {
    return this.channels.some((ch) => ch.hasMore && !ch.error);
  }

  /** Charge la page suivante de chaque canal encore actif. */
  async loadMore() {
    const active = this.channels.filter((ch) => ch.hasMore && !ch.error);
    await Promise.all(active.map(async (ch) => {
      try {
        const res = await ch.src.search(this.http, ch.country, this.criteria, ch.page + 1);
        ch.page += 1;
        ch.total = res.total;
        ch.hasMore = res.hasMore;
        for (const ad of res.ads) {
          if (this.seen.has(ad.id) || !matchesCriteria(ad, this.criteria)) continue;
          this.seen.add(ad.id);
          this.ads.push(ad);
        }
      } catch (e) {
        ch.error = e.message || String(e);
        ch.hasMore = false;
      }
    }));
    this.ads = sortAds(this.ads, this.criteria.sort);
    return this.ads;
  }
}
