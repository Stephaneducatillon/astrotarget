// Onglet « Ciel » : étoiles variables visibles depuis un lieu à un moment donné.
(function () {
  'use strict';
  const { store } = Platform;
  const $ = (id) => document.getElementById(id);
  const PAGE = 300;

  let app, stars, rows = [], shown = PAGE, night = null, ctxJd = 0;
  let program = new Set(store.get('program', []));
  let skyChart = null, dialogStar = null;

  const pad = (n) => String(n).padStart(2, '0');
  const toLocalInput = (d) => `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
  const hhmm = (jd) => { const d = Astro.dateFromJd(jd); return `${pad(d.getHours())}:${pad(d.getMinutes())}`; };
  const dateShort = (jd) => Astro.dateFromJd(jd).toLocaleDateString('fr-FR', { day: '2-digit', month: 'short', year: 'numeric' });
  const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

  function location() {
    return { name: $('loc-name').value.trim(), lat: parseFloat($('loc-lat').value), lon: parseFloat($('loc-lon').value), elev: parseFloat($('loc-elev').value) || 0 };
  }
  function currentJd() {
    const d = new Date($('obs-datetime').value);
    return Astro.jdFromDate(isNaN(d) ? new Date() : d);
  }

  function saveForm() {
    const ids = ['loc-name', 'loc-lat', 'loc-lon', 'loc-elev', 'f-mode', 'f-alt', 'f-mag', 'f-amp', 'f-type', 'f-con', 'f-sort'];
    const v = {}; ids.forEach((id) => { v[id] = $(id).value; });
    v['f-program'] = $('f-program').checked;
    store.set('sky', v);
  }
  function restoreForm() {
    const v = store.get('sky', { 'loc-name': 'Paris', 'loc-lat': '48.8566', 'loc-lon': '2.3522', 'loc-elev': '35' });
    Object.entries(v).forEach(([id, val]) => {
      const el = $(id); if (!el) return;
      if (el.type === 'checkbox') el.checked = !!val; else el.value = val;
    });
  }

  function phaseInfo(s, jd) {
    const p = Astro.phase(s.epoch, s.period, jd);
    if (!p) return { text: '—', title: '' };
    const what = s.epochIsMin ? 'min' : 'max';
    const days = p.next - jd;
    const text = `φ ${p.phase.toFixed(2)} · ${what} ${days < 1 ? 'dans ' + (days * 24).toFixed(1) + ' h' : 'dans ' + days.toFixed(days < 10 ? 1 : 0) + ' j'}`;
    return { text, title: `Prochain ${what === 'max' ? 'maximum' : 'minimum'} prévu : ${Astro.dateFromJd(p.next).toLocaleString('fr-FR')} (éphéméride GCVS, indicative)` };
  }

  function compute() {
    const loc = location();
    if (!isFinite(loc.lat) || !isFinite(loc.lon)) { app.toast('Indiquez une latitude et une longitude valides'); return; }
    saveForm();
    const jd = currentJd(); ctxJd = jd;
    night = Astro.nightEvents(jd, loc.lat, loc.lon);
    const mode = $('f-mode').value;
    const minAlt = parseFloat($('f-alt').value) || 0;
    const magLim = parseFloat($('f-mag').value);
    const ampMin = parseFloat($('f-amp').value) || 0;
    const fam = $('f-type').value, con = $('f-con').value;
    const q = $('f-search').value.trim().toLowerCase();
    const onlyProg = $('f-program').checked;
    const lstNow = Astro.lst(jd, loc.lon);
    const hasDark = night.darkStart !== undefined;

    rows = [];
    for (const s of stars) {
      if (onlyProg && !program.has(s.name)) continue;
      if (isFinite(magLim) && s.max > magLim) continue;
      if (ampMin > 0 && !(s.amp >= ampMin)) continue;
      if (fam && s.fam !== fam) continue;
      if (con && s.con !== con) continue;
      if (q && !(s.name.toLowerCase().includes(q) || (s.type || '').toLowerCase() === q)) continue;
      // test rapide : un astre qui ne monte jamais assez haut est éliminé
      if (90 - Math.abs(loc.lat - s.dec) < minAlt) continue;
      const p = Astro.precess(s.ra, s.dec, jd);
      let alt, az, bestJd = null;
      if (mode === 'night' && hasDark) {
        const best = Astro.maxAltInWindow(p.ra, p.dec, night.darkStart, night.darkEnd, loc.lat, loc.lon);
        if (best.alt < minAlt) continue;
        bestJd = best.jd;
        const h = Astro.altAz(p.ra, p.dec, best.jd, loc.lat, loc.lon);
        alt = h.alt; az = h.az;
      } else {
        const h = Astro.altAzFromHa(lstNow - p.ra, p.dec, loc.lat);
        if (h.alt < minAlt) continue;
        alt = h.alt; az = h.az;
      }
      rows.push({ s, alt, az, bestJd, transit: Astro.transit(p.ra, jd, loc.lon) });
    }
    const sort = $('f-sort').value;
    const cmp = {
      alt: (a, b) => b.alt - a.alt,
      max: (a, b) => a.s.max - b.s.max,
      amp: (a, b) => (b.s.amp ?? -1) - (a.s.amp ?? -1),
      name: (a, b) => a.s.name.localeCompare(b.s.name, 'fr'),
      transit: (a, b) => a.transit - b.transit,
    }[sort];
    rows.sort(cmp);
    shown = PAGE;
    renderSummary(loc, jd);
    renderTable();
    $('sky-form-summary').textContent = `${loc.name || 'Lieu'} (${loc.lat.toFixed(2)}°, ${loc.lon.toFixed(2)}°) · ${Astro.dateFromJd(jd).toLocaleString('fr-FR', { dateStyle: 'medium', timeStyle: 'short' })} · hauteur ≥ ${minAlt}° · mag ≤ ${isFinite(magLim) ? magLim : '∞'}`;
  }

  function renderSummary(loc, jd) {
    const sAlt = Astro.sunAlt(jd, loc.lat, loc.lon);
    const m = Astro.moonAltAz(jd, loc.lat, loc.lon);
    const sky = sAlt > -0.833 ? 'Jour' : sAlt > -6 ? 'Crépuscule civil' : sAlt > -12 ? 'Crépuscule nautique' : sAlt > -18 ? 'Crépuscule astronomique' : 'Nuit noire';
    const ev = night;
    const t = (k) => (ev[k] !== undefined ? hhmm(ev[k]) : '—');
    const moonPhase = m.illum < 0.03 ? 'Nouvelle lune' : m.illum > 0.97 ? 'Pleine lune' : (m.waxing ? 'croissante' : 'décroissante');
    $('sky-summary').innerHTML = `
      <div><span>Ciel :</span> <b>${sky}</b> (Soleil ${sAlt.toFixed(1)}°)</div>
      <div><span>Coucher / lever du Soleil :</span> <b>${t('sunsetDown')} / ${t('sunsetUp')}</b></div>
      <div><span>Nuit astronomique :</span> <b>${t('astroDown')} → ${t('astroUp')}</b></div>
      <div><span>Lune :</span> <b>${Math.round(m.illum * 100)} % ${m.illum >= 0.03 && m.illum <= 0.97 ? moonPhase : ''}</b>${m.illum < 0.03 || m.illum > 0.97 ? ' (' + moonPhase + ')' : ''}, hauteur ${m.alt.toFixed(0)}°</div>
      <div><span>Temps sidéral local :</span> <b>${Astro.fmt.hms(Astro.lst(jd, loc.lon))}</b></div>
      <div><span>JD :</span> <b>${jd.toFixed(4)}</b></div>`;
  }

  function renderTable() {
    const mode = $('f-mode').value;
    $('list-count').textContent = `${rows.length.toLocaleString('fr-FR')} étoile${rows.length > 1 ? 's' : ''} variable${rows.length > 1 ? 's' : ''}${mode === 'night' ? ' observables cette nuit' : ' au-dessus de ' + ($('f-alt').value || 0) + '°'}`;
    const html = [];
    for (const r of rows.slice(0, shown)) {
      const s = r.s, ph = phaseInfo(s, ctxJd), on = program.has(s.name);
      const altCls = r.alt >= 40 ? 'alt-good' : r.alt < 15 ? 'alt-low' : '';
      html.push(`<tr class="clickable" data-i="${s.i}">
        <td><button class="star-btn ${on ? 'on' : ''}" data-prog="${s.i}" title="${on ? 'Retirer de' : 'Ajouter à'} mon programme">${on ? '★' : '☆'}</button></td>
        <td><b>${esc(s.name)}</b></td><td>${esc(s.type)}</td><td>${esc(s.con)}</td>
        <td class="num">${s.max.toFixed(1)}${s.min != null ? '–' + s.min.toFixed(1) : ''} ${esc(s.band === '*' ? '' : s.band || '')}</td>
        <td class="num">${s.period ? (s.period < 10 ? s.period.toFixed(4) : s.period.toFixed(1)) : '—'}</td>
        <td title="${esc(ph.title)}">${ph.text}</td>
        <td class="num ${altCls}">${r.alt.toFixed(1)}°${r.bestJd ? ' <small>(' + hhmm(r.bestJd) + ')</small>' : ''}</td>
        <td>${r.az.toFixed(0)}° ${Astro.fmt.cardinal(r.az)}</td>
        <td>${hhmm(r.transit)}</td>
        <td>${Astro.fmt.hms(s.ra)} ${Astro.fmt.dms(s.dec)}</td></tr>`);
    }
    document.querySelector('#star-table tbody').innerHTML = html.join('');
    $('btn-more').hidden = rows.length <= shown;
  }

  function toggleProgram(name) {
    if (program.has(name)) program.delete(name); else program.add(name);
    store.set('program', [...program]);
  }

  function exportList() {
    const loc = location();
    const head = ['Nom', 'Type', 'Constellation', 'MagMax', 'MagMin', 'Bande', 'Periode_j', 'Epoque_JD', 'Spectre', 'AD_J2000_deg', 'Dec_J2000_deg', 'AD', 'Dec', 'Hauteur_deg', 'Azimut_deg', 'Culmination_locale', 'Phase'];
    const lines = [head.join(';')];
    for (const r of rows) {
      const s = r.s, p = Astro.phase(s.epoch, s.period, ctxJd);
      lines.push([s.name, s.type, s.con, s.max, s.min ?? '', s.band, s.period ?? '', s.epoch ?? '', s.spec, s.ra, s.dec, Astro.fmt.hms(s.ra), Astro.fmt.dms(s.dec),
        r.alt.toFixed(1), r.az.toFixed(0), hhmm(r.transit), p ? p.phase.toFixed(3) : ''].map((v) => String(v).replace(/;/g, ',')).join(';'));
    }
    const d = Astro.dateFromJd(ctxJd);
    const fname = `variables_${(loc.name || 'lieu').replace(/\W+/g, '_')}_${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}_${pad(d.getHours())}${pad(d.getMinutes())}.csv`;
    Platform.saveFile(fname, '﻿' + lines.join('\r\n'), 'text/csv;charset=utf-8').then(() => app.toast('Liste exportée : ' + fname));
  }

  // ── Fiche étoile ──────────────────────────────────────────────────────────
  function openStar(s) {
    dialogStar = s;
    const loc = location();
    const jd = ctxJd || currentJd();
    const p = Astro.precess(s.ra, s.dec, jd);
    const h = Astro.altAz(p.ra, p.dec, jd, loc.lat, loc.lon);
    const m = Astro.moonAltAz(jd, loc.lat, loc.lon);
    const ph = Astro.phase(s.epoch, s.period, jd);
    const am = Astro.airmass(h.alt);
    $('sd-title').textContent = s.name;
    $('sd-body').innerHTML = `<dl class="kv">
      <dt>Type</dt><dd>${esc(s.type)} — ${esc(Catalog.familyLabel(s.fam))}</dd>
      <dt>Constellation</dt><dd>${esc(Catalog.CONSTELLATIONS[s.con] || s.con)}</dd>
      <dt>Magnitudes</dt><dd>${s.max} – ${s.min ?? '?'} (${esc(s.band || '?')})${s.amp != null ? ', amplitude ' + s.amp : ''}</dd>
      <dt>Période</dt><dd>${s.period ? s.period + ' j' : '—'}</dd>
      <dt>Époque</dt><dd>${s.epoch ? s.epoch + ' (' + (s.epochIsMin ? 'minimum' : 'maximum') + ')' : '—'}</dd>
      ${ph ? `<dt>Phase actuelle</dt><dd>${ph.phase.toFixed(3)} — prochain ${s.epochIsMin ? 'minimum' : 'maximum'} : ${Astro.dateFromJd(ph.next).toLocaleString('fr-FR')}</dd>` : ''}
      <dt>Spectre</dt><dd>${esc(s.spec || '—')}</dd>
      <dt>AD / Déc J2000</dt><dd>${Astro.fmt.hms(s.ra)} ${Astro.fmt.dms(s.dec)}</dd>
      <dt>Position</dt><dd>hauteur ${h.alt.toFixed(1)}°, azimut ${h.az.toFixed(0)}° ${Astro.fmt.cardinal(h.az)}${am ? ', masse d\'air ' + am.toFixed(3) : ''}</dd>
      <dt>Culmination</dt><dd>${hhmm(Astro.transit(p.ra, jd, loc.lon))} à ${(90 - Math.abs(loc.lat - p.dec)).toFixed(1)}°</dd>
      <dt>Lune</dt><dd>à ${Astro.separation(p.ra, p.dec, m.ra, m.dec).toFixed(0)}°, éclairée à ${Math.round(m.illum * 100)} %</dd>
    </dl>`;
    $('sd-program').textContent = program.has(s.name) ? '★ Retirer du programme' : '☆ Ajouter à mon programme';
    const nm = encodeURIComponent(s.name);
    $('sd-vsx').href = 'https://www.aavso.org/vsx/index.php?view=results.get&ident=' + nm;
    $('sd-vsp').href = 'https://apps.aavso.org/vsp/?star=' + nm;
    $('sd-lcg').href = 'https://www.aavso.org/LCGv2/index.htm?ident=' + nm;
    $('star-dialog').showModal();
    drawAltitude(s, p, loc);
  }

  function drawAltitude(s, p, loc) {
    const ev = night || Astro.nightEvents(ctxJd, loc.lat, loc.lon);
    const pts = [], moonPts = [];
    for (let t = ev.start; t <= ev.end + 1e-9; t += 10 / 1440) {
      const x = Astro.dateFromJd(t).getTime();
      pts.push({ x, y: Astro.altAz(p.ra, p.dec, t, loc.lat, loc.lon).alt });
      moonPts.push({ x, y: Astro.moonAltAz(t, loc.lat, loc.lon).alt });
    }
    const css = getComputedStyle(document.documentElement);
    const c = (v) => css.getPropertyValue(v).trim();
    const darkBand = {
      id: 'darkBand',
      beforeDatasetsDraw(chart) {
        if (ev.darkStart === undefined) return;
        const { ctx, chartArea: a, scales: { x } } = chart;
        const x1 = x.getPixelForValue(Astro.dateFromJd(ev.darkStart).getTime()), x2 = x.getPixelForValue(Astro.dateFromJd(ev.darkEnd).getTime());
        ctx.save(); ctx.fillStyle = c('--surface-2'); ctx.fillRect(Math.max(a.left, x1), a.top, Math.min(a.right, x2) - Math.max(a.left, x1), a.bottom - a.top); ctx.restore();
      },
    };
    if (skyChart) skyChart.destroy();
    skyChart = new Chart($('sd-chart'), {
      type: 'line',
      data: { datasets: [
        { label: s.name, data: pts, borderColor: c('--series-1'), borderWidth: 2, pointRadius: 0 },
        { label: 'Lune', data: moonPts, borderColor: c('--muted'), borderWidth: 2, borderDash: [5, 4], pointRadius: 0 },
      ] },
      options: {
        animation: false, maintainAspectRatio: false, parsing: false,
        interaction: { mode: 'index', intersect: false },
        scales: {
          x: { type: 'linear', ticks: { color: c('--text-2'), callback: (v) => hhmm(Astro.jdFromDate(new Date(v))), maxTicksLimit: 9 }, grid: { color: c('--grid') } },
          y: { min: 0, max: 90, title: { display: true, text: 'Hauteur (°)', color: c('--text-2') }, ticks: { color: c('--text-2') }, grid: { color: c('--grid') } },
        },
        plugins: {
          legend: { labels: { color: c('--text'), boxWidth: 12 } },
          tooltip: { callbacks: { title: (it) => hhmm(Astro.jdFromDate(new Date(it[0].parsed.x))), label: (it) => `${it.dataset.label} : ${it.parsed.y.toFixed(1)}°` } },
        },
      },
      plugins: [darkBand],
    });
  }

  async function searchPlace() {
    const q = $('loc-name').value.trim();
    if (!q) return app.toast('Saisissez un nom de lieu');
    const box = $('place-results');
    box.textContent = 'Recherche…';
    try {
      const r = await fetch('https://nominatim.openstreetmap.org/search?format=json&limit=5&accept-language=fr&q=' + encodeURIComponent(q));
      const list = await r.json();
      box.innerHTML = '';
      if (!list.length) { box.textContent = 'Aucun résultat'; return; }
      for (const it of list) {
        const b = document.createElement('button');
        b.type = 'button';
        b.textContent = `${it.display_name} (${(+it.lat).toFixed(3)}, ${(+it.lon).toFixed(3)})`;
        b.onclick = () => {
          $('loc-lat').value = (+it.lat).toFixed(4); $('loc-lon').value = (+it.lon).toFixed(4);
          $('loc-name').value = it.display_name.split(',')[0];
          box.innerHTML = ''; compute();
        };
        box.appendChild(b);
      }
    } catch (e) {
      box.textContent = 'Recherche impossible (connexion internet ?). Saisissez les coordonnées à la main.';
    }
  }

  async function geolocate() {
    try {
      const c = await Platform.getPosition();
      $('loc-lat').value = c.latitude.toFixed(4); $('loc-lon').value = c.longitude.toFixed(4);
      if (c.altitude != null) $('loc-elev').value = Math.round(c.altitude);
      $('loc-name').value = 'Ma position';
      compute();
    } catch (e) {
      app.toast('Position indisponible : ' + (e.message || e));
    }
  }

  function init(appRef) {
    app = appRef;
    stars = Catalog.load();
    restoreForm();
    // sur petit écran, le formulaire est replié une fois réglé pour laisser la place à la liste
    if (window.matchMedia('(max-width: 640px)').matches && store.get('sky')) $('sky-form').open = false;
    $('obs-datetime').value = toLocalInput(new Date());
    const fType = $('f-type');
    Catalog.FAMILIES.forEach((f) => fType.add(new Option(f.label, f.id)));
    const fCon = $('f-con');
    Object.entries(Catalog.CONSTELLATIONS).sort((a, b) => a[1].localeCompare(b[1], 'fr')).forEach(([k, v]) => fCon.add(new Option(`${v} (${k})`, k)));
    const saved = store.get('sky', {});
    if (saved['f-type']) fType.value = saved['f-type'];
    if (saved['f-con']) fCon.value = saved['f-con'];

    let timer;
    const debounced = () => { clearTimeout(timer); timer = setTimeout(compute, 250); };
    ['loc-lat', 'loc-lon', 'loc-elev', 'obs-datetime', 'f-mode', 'f-alt', 'f-mag', 'f-amp', 'f-type', 'f-con', 'f-sort', 'f-program', 'f-search']
      .forEach((id) => $(id).addEventListener(id === 'f-search' ? 'input' : 'change', debounced));
    $('loc-name').addEventListener('change', saveForm);
    $('btn-now').onclick = () => { $('obs-datetime').value = toLocalInput(new Date()); compute(); };
    $('btn-search-place').onclick = searchPlace;
    $('btn-geoloc').onclick = geolocate;
    $('btn-more').onclick = () => { shown += PAGE; renderTable(); };
    $('btn-export-list').onclick = exportList;
    document.querySelector('#star-table tbody').addEventListener('click', (e) => {
      const pb = e.target.closest('[data-prog]');
      if (pb) {
        e.stopPropagation();
        toggleProgram(stars[+pb.dataset.prog].name);
        if ($('f-program').checked) compute(); else renderTable();
        return;
      }
      const tr = e.target.closest('tr[data-i]');
      if (tr) openStar(stars[+tr.dataset.i]);
    });
    $('sd-program').onclick = () => { toggleProgram(dialogStar.name); $('sd-program').textContent = program.has(dialogStar.name) ? '★ Retirer du programme' : '☆ Ajouter à mon programme'; renderTable(); };
    compute();
  }

  window.Sky = { init, compute, location, currentJd, program: () => program };
})();
