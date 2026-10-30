// Calculs astronomiques (basse précision, suffisante pour la planification d'observations).
// Sources : J. Meeus, « Astronomical Algorithms » ; Astronomical Almanac (formules basse précision).
(function () {
  'use strict';
  const D2R = Math.PI / 180, R2D = 180 / Math.PI;
  const sin = (d) => Math.sin(d * D2R), cos = (d) => Math.cos(d * D2R);
  const norm360 = (x) => ((x % 360) + 360) % 360;
  const norm180 = (x) => { const y = norm360(x); return y >= 180 ? y - 360 : y; };

  function jdFromDate(date) { return date.getTime() / 86400000 + 2440587.5; }
  function dateFromJd(jd) { return new Date((jd - 2440587.5) * 86400000); }

  // Temps sidéral moyen de Greenwich (degrés)
  function gmst(jd) {
    const t = (jd - 2451545.0) / 36525;
    return norm360(280.46061837 + 360.98564736629 * (jd - 2451545.0) + 0.000387933 * t * t - t * t * t / 38710000);
  }
  function lst(jd, lonDeg) { return norm360(gmst(jd) + lonDeg); }

  // Précession J2000 -> date (Meeus 21.2/21.4)
  function precess(raDeg, decDeg, jd) {
    const T = (jd - 2451545.0) / 36525;
    const z1 = (2306.2181 * T + 0.30188 * T * T + 0.017998 * T * T * T) / 3600;
    const z = (2306.2181 * T + 1.09468 * T * T + 0.018203 * T * T * T) / 3600;
    const th = (2004.3109 * T - 0.42665 * T * T - 0.041833 * T * T * T) / 3600;
    const A = cos(decDeg) * sin(raDeg + z1);
    const B = cos(th) * cos(decDeg) * cos(raDeg + z1) - sin(th) * sin(decDeg);
    const C = sin(th) * cos(decDeg) * cos(raDeg + z1) + cos(th) * sin(decDeg);
    return { ra: norm360(Math.atan2(A, B) * R2D + z), dec: Math.asin(Math.max(-1, Math.min(1, C))) * R2D };
  }

  // Coordonnées horizontales à partir de l'angle horaire (degrés)
  function altAzFromHa(haDeg, decDeg, latDeg) {
    const sa = sin(decDeg) * sin(latDeg) + cos(decDeg) * cos(latDeg) * cos(haDeg);
    const alt = Math.asin(Math.max(-1, Math.min(1, sa))) * R2D;
    const az = norm360(Math.atan2(-cos(decDeg) * sin(haDeg), sin(decDeg) * cos(latDeg) - cos(decDeg) * sin(latDeg) * cos(haDeg)) * R2D);
    return { alt, az };
  }
  function altAz(raDeg, decDeg, jd, latDeg, lonDeg) {
    return altAzFromHa(lst(jd, lonDeg) - raDeg, decDeg, latDeg);
  }

  function obliquity(jd) { return 23.439291 - 0.0130042 * (jd - 2451545.0) / 36525; }
  function eclToEq(lam, beta, jd) {
    const e = obliquity(jd);
    const ra = norm360(Math.atan2(sin(lam) * cos(e) - Math.tan(beta * D2R) * sin(e), cos(lam)) * R2D);
    const dec = Math.asin(sin(beta) * cos(e) + cos(beta) * sin(e) * sin(lam)) * R2D;
    return { ra, dec };
  }

  // Soleil (précision ~0.01°)
  function sun(jd) {
    const d = jd - 2451545.0;
    const L = norm360(280.460 + 0.9856474 * d);
    const g = norm360(357.528 + 0.9856003 * d);
    const lam = norm360(L + 1.915 * sin(g) + 0.020 * sin(2 * g));
    return Object.assign({ lon: lam }, eclToEq(lam, 0, jd));
  }

  // Lune (précision ~0.3°), parallaxe horizontale incluse
  function moon(jd) {
    const T = (jd - 2451545.0) / 36525;
    const lam = norm360(218.32 + 481267.881 * T
      + 6.29 * sin(135.0 + 477198.87 * T) - 1.27 * sin(259.3 - 413335.36 * T)
      + 0.66 * sin(235.7 + 890534.22 * T) + 0.21 * sin(269.9 + 954397.74 * T)
      - 0.19 * sin(357.5 + 35999.05 * T) - 0.11 * sin(186.5 + 966404.03 * T));
    const beta = 5.13 * sin(93.3 + 483202.02 * T) + 0.28 * sin(228.2 + 960400.89 * T)
      - 0.28 * sin(318.3 + 6003.15 * T) - 0.17 * sin(217.6 - 407332.21 * T);
    const par = 0.9508 + 0.0518 * cos(135.0 + 477198.87 * T) + 0.0095 * cos(259.3 - 413335.36 * T)
      + 0.0078 * cos(235.7 + 890534.22 * T) + 0.0028 * cos(269.9 + 954397.74 * T);
    const eq = eclToEq(lam, beta, jd);
    const s = sun(jd);
    const elong = norm360(lam - s.lon);
    const cosPsi = cos(beta) * cos(lam - s.lon);
    return Object.assign({ lon: lam, lat: beta, parallax: par, illum: (1 - cosPsi) / 2, waxing: elong < 180, elong }, eq);
  }

  function sunAlt(jd, lat, lon) { const s = sun(jd); return altAz(s.ra, s.dec, jd, lat, lon).alt; }
  function moonAltAz(jd, lat, lon) {
    const m = moon(jd);
    const h = altAz(m.ra, m.dec, jd, lat, lon);
    h.alt -= m.parallax * cos(h.alt); // altitude topocentrique
    return Object.assign(h, { illum: m.illum, waxing: m.waxing, ra: m.ra, dec: m.dec });
  }

  // Séparation angulaire (degrés)
  function separation(ra1, dec1, ra2, dec2) {
    const c = sin(dec1) * sin(dec2) + cos(dec1) * cos(dec2) * cos(ra1 - ra2);
    return Math.acos(Math.max(-1, Math.min(1, c))) * R2D;
  }

  // Masse d'air (Hardie 1962), null sous 5° d'altitude
  function airmass(altDeg) {
    if (altDeg < 5) return null;
    const secz = 1 / cos(90 - altDeg), s1 = secz - 1;
    return secz - 0.0018167 * s1 - 0.002875 * s1 * s1 - 0.0008083 * s1 * s1 * s1;
  }

  // Événements solaires de la nuit contenant (ou suivant) l'instant donné.
  // La nuit va de midi local (approx. via la longitude) au midi suivant.
  function nightEvents(jd, lat, lon) {
    // Un JD entier correspond à 12h UT ; le midi solaire moyen local vaut donc N - lon/360.
    // Si l'instant est le matin, la nuit retenue est celle qui a commencé la veille.
    const start = Math.floor(jd + lon / 360) - lon / 360;
    const step = 5 / 1440, levels = { sunset: -0.833, civil: -6, nautical: -12, astro: -18 };
    const ev = { start, end: start + 1 };
    let prevT = start, prevA = sunAlt(start, lat, lon);
    let minAlt = prevA, minT = start;
    for (let t = start + step; t <= start + 1 + 1e-9; t += step) {
      const a = sunAlt(t, lat, lon);
      if (a < minAlt) { minAlt = a; minT = t; }
      for (const [k, lvl] of Object.entries(levels)) {
        if (prevA > lvl && a <= lvl && ev[k + 'Down'] === undefined) ev[k + 'Down'] = prevT + (t - prevT) * (prevA - lvl) / (prevA - a);
        if (prevA < lvl && a >= lvl && ev[k + 'Up'] === undefined) ev[k + 'Up'] = prevT + (t - prevT) * (lvl - prevA) / (a - prevA);
      }
      prevT = t; prevA = a;
    }
    ev.minSunAlt = minAlt; ev.midnight = minT;
    // fenêtre « sombre » : soleil sous -12° (nuit nautique), sinon sous l'horizon
    if (ev.nauticalDown !== undefined && ev.nauticalUp !== undefined) { ev.darkStart = ev.nauticalDown; ev.darkEnd = ev.nauticalUp; }
    else if (minAlt < -12) { ev.darkStart = ev.nauticalDown ?? start; ev.darkEnd = ev.nauticalUp ?? start + 1; }
    else if (ev.sunsetDown !== undefined && ev.sunsetUp !== undefined) { ev.darkStart = ev.sunsetDown; ev.darkEnd = ev.sunsetUp; }
    return ev;
  }

  // Altitude maximale d'un astre fixe sur un intervalle [jd1, jd2] (analytique)
  function maxAltInWindow(ra, dec, jd1, jd2, lat, lon) {
    const ha1 = norm180(lst(jd1, lon) - ra);
    const span = (jd2 - jd1) * 360.98564736629;
    let ha;
    if (ha1 <= 0 && ha1 + span >= 0) ha = 0;
    else if (ha1 + span >= 360) ha = 0;
    else ha = Math.abs(ha1) < Math.abs(norm180(ha1 + span)) ? ha1 : ha1 + span;
    const alt = altAzFromHa(ha, dec, lat).alt;
    const jd = jd1 + norm360(ha - ha1) / 360.98564736629;
    return { alt, jd: Math.min(Math.max(jd, jd1), jd2) };
  }

  // Heure de passage au méridien la plus proche de jd
  function transit(ra, jd, lon) {
    const ha = norm180(lst(jd, lon) - ra);
    return jd - ha / 360.98564736629;
  }

  // Phase d'une variable périodique et prochaine date d'époque (max ou min selon le type)
  function phase(epoch, period, jd) {
    if (!epoch || !period) return null;
    const n = (jd - epoch) / period;
    const ph = n - Math.floor(n);
    return { phase: ph, next: epoch + Math.ceil(n) * period, prev: epoch + Math.floor(n) * period };
  }

  const fmt = {
    hms(deg) {
      let h = norm360(deg) / 15; const hh = Math.floor(h); h = (h - hh) * 60; const mm = Math.floor(h); const ss = (h - mm) * 60;
      return `${String(hh).padStart(2, '0')}h${String(mm).padStart(2, '0')}m${ss.toFixed(0).padStart(2, '0')}s`;
    },
    dms(deg) {
      const s = deg < 0 ? '-' : '+'; let d = Math.abs(deg); const dd = Math.floor(d); d = (d - dd) * 60; const mm = Math.floor(d); const ss = (d - mm) * 60;
      return `${s}${String(dd).padStart(2, '0')}°${String(mm).padStart(2, '0')}'${ss.toFixed(0).padStart(2, '0')}"`;
    },
    cardinal(az) { return ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSO', 'SO', 'OSO', 'O', 'ONO', 'NO', 'NNO'][Math.round(norm360(az) / 22.5) % 16]; },
  };

  window.Astro = { jdFromDate, dateFromJd, gmst, lst, precess, altAz, altAzFromHa, sun, moon, sunAlt, moonAltAz, separation, airmass, nightEvents, maxAltInWindow, transit, phase, fmt, norm360, norm180 };
})();
