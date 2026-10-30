// Onglet « Courbes » : import CSV/XLS(X), conversion des dates en JD, courbe de lumière.
(function () {
  'use strict';
  const $ = (id) => document.getElementById(id);
  let app, table = null, workbook = null, chart = null, current = [];

  // Champs reconnus (clé, motif d'en-tête). Les noms AAVSO sont reconnus tels quels.
  const FIELDS = {
    date: /^(date|jd|hjd|bjd|mjd|rjd|julian|jour|obs.?date|date.?obs|time|temps|heure)/i,
    mag: /^(mag|magnitude|vmag|m)$|^mag[ _.-]?(v|calib|std)?$/i,
    err: /(merr|err|sigma|σ|incert|uncert)/i,
    filter: /^(filt|filter|filtre|band|bande)/i,
    name: /^(name|star|target|objet|object|[ée]toile|nom|starid|star.?name)$/i,
    cname: /^(cname|comp|comp1|comp.?name|comparaison)$/i,
    cmag: /^(cmag|comp.?mag|mag.?comp)$/i,
    kname: /^(kname|check|comp2|check.?name)$/i,
    kmag: /^(kmag|check.?mag|mag.?check)$/i,
    amass: /^(amass|airmass|masse.?d.?air|x)$/i,
    chart: /^(chart|carte|sequence)$/i,
    notes: /^(notes?|comment|remarques?)$/i,
    group: /^group$/i,
    trans: /^trans$/i,
    mtype: /^mtype$/i,
  };
  const EXTRA = ['cname', 'cmag', 'kname', 'kmag', 'amass', 'chart', 'notes', 'group', 'trans', 'mtype'];
  const FILTER_COLORS = { V: 1, B: 7, R: 8, I: 2, 'Vis.': 3, TG: 6, CV: 4, U: 5 };
  const FILTER_NORM = { VIS: 'Vis.', 'VIS.': 'Vis.', VISUAL: 'Vis.', JOHNSONV: 'V', JOHNSONB: 'B', COUSINSR: 'R', COUSINSI: 'I', RC: 'R', IC: 'I', CLEAR: 'CV', C: 'CV' };

  const num = (v) => {
    if (typeof v === 'number') return v;
    if (v == null) return NaN;
    const s = String(v).trim().replace(/\s/g, '').replace(',', '.');
    return s === '' ? NaN : Number(s);
  };
  const normFilter = (f) => {
    const s = String(f ?? '').trim();
    if (!s) return '';
    const k = s.toUpperCase().replace(/[\s_-]/g, '');
    return FILTER_NORM[k] || (/^[UBVRI]$/.test(k) ? k : s);
  };

  // ── Lecture de fichiers ─────────────────────────────────────────────────────
  function parseDelimited(text, delim) {
    const rows = []; let row = [], field = '', q = false;
    for (let i = 0; i < text.length; i++) {
      const c = text[i];
      if (q) {
        if (c === '"') { if (text[i + 1] === '"') { field += '"'; i++; } else q = false; } else field += c;
      } else if (c === '"' && field === '') q = true;
      else if (c === delim) { row.push(field); field = ''; }
      else if (c === '\n' || c === '\r') {
        if (c === '\r' && text[i + 1] === '\n') i++;
        row.push(field); field = ''; if (row.some((x) => x.trim() !== '')) rows.push(row); row = [];
      } else field += c;
    }
    row.push(field); if (row.some((x) => x.trim() !== '')) rows.push(row);
    return rows;
  }
  function guessDelim(line) {
    const cands = ['\t', ';', ',', '|'];
    let best = ',', n = 0;
    for (const d of cands) { const k = line.split(d).length - 1; if (k > n) { n = k; best = d; } }
    return best;
  }

  // Fichier AAVSO (format étendu ou visuel) → { header, rows }
  function parseAavsoText(text) {
    const lines = text.replace(/^﻿/, '').split(/\r?\n/);
    const params = {}; let cols = null; const body = [];
    for (const l of lines) {
      if (!l.trim()) continue;
      if (l.startsWith('#')) {
        const m = l.match(/^#\s*([A-Za-z]+)\s*=\s*(.*)$/);
        if (m) params[m[1].toUpperCase()] = m[2].trim();
        else if (/^#\s*NAME\s*[,;\t|]/i.test(l)) cols = l.replace(/^#\s*/, '');
        continue;
      }
      body.push(l);
    }
    const delim = { comma: ',', tab: '\t', semicolon: ';', pipe: '|' }[(params.DELIM || '').toLowerCase()] || params.DELIM || ',';
    const visual = (params.TYPE || '').toUpperCase() === 'VISUAL';
    const header = cols ? cols.split(delim).map((s) => s.trim())
      : visual ? ['NAME', 'DATE', 'MAG', 'COMMENTCODE', 'COMP1', 'COMP2', 'CHART', 'NOTES']
        : ['NAME', 'DATE', 'MAG', 'MERR', 'FILT', 'TRANS', 'MTYPE', 'CNAME', 'CMAG', 'KNAME', 'KMAG', 'AMASS', 'GROUP', 'CHART', 'NOTES'];
    const rows = body.map((l) => {
      const f = l.split(delim);
      // le champ NOTES (dernier) peut contenir le délimiteur
      if (f.length > header.length) f.splice(header.length - 1, f.length, f.slice(header.length - 1).join(delim));
      return f.map((s) => s.trim());
    });
    return { header, rows, params, visual };
  }

  async function readFile(file) {
    const name = file.name.toLowerCase();
    if (/\.(xlsx?|ods)$/.test(name)) {
      if (!window.XLSX) throw new Error('Bibliothèque XLSX non chargée');
      workbook = XLSX.read(await file.arrayBuffer(), { type: 'array' });
      return null;
    }
    workbook = null;
    const text = await file.text();
    if (/^﻿?#\s*TYPE\s*=/im.test(text.slice(0, 2000))) {
      const a = parseAavsoText(text);
      return { header: a.header, rows: a.rows, aavso: a };
    }
    const clean = text.replace(/^﻿/, '');
    const first = clean.split(/\r?\n/).find((l) => l.trim() && !l.startsWith('#')) || '';
    const all = parseDelimited(clean.split(/\r?\n/).filter((l) => !l.startsWith('#')).join('\n'), guessDelim(first));
    return { header: all[0].map((s) => s.trim()), rows: all.slice(1) };
  }

  function sheetTable(sheetName) {
    const aoa = XLSX.utils.sheet_to_json(workbook.Sheets[sheetName], { header: 1, raw: true, defval: '' });
    const start = aoa.findIndex((r) => r.filter((x) => String(x).trim() !== '').length >= 2);
    if (start < 0) return { header: [], rows: [] };
    return { header: aoa[start].map((s) => String(s).trim()), rows: aoa.slice(start + 1).filter((r) => r.some((x) => String(x).trim() !== '')) };
  }

  // ── Dates → JD ─────────────────────────────────────────────────────────────
  function parseCalendar(s) {
    s = String(s).trim();
    let m = s.match(/^(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})(?:[ T](\d{1,2}):(\d{2})(?::(\d{2}(?:[.,]\d+)?))?)?$/);
    if (m) return Date.UTC(+m[3], +m[2] - 1, +m[1], +(m[4] || 0), +(m[5] || 0)) / 1000 + num(m[6] || 0);
    m = s.match(/^(\d{4})[/.-](\d{1,2})[/.-](\d{1,2})(?:[ T](\d{1,2}):(\d{2})(?::(\d{2}(?:[.,]\d+)?))?)?/);
    if (m) return Date.UTC(+m[1], +m[2] - 1, +m[3], +(m[4] || 0), +(m[5] || 0)) / 1000 + num(m[6] || 0);
    const t = Date.parse(s.includes('Z') || /[+-]\d\d:?\d\d$/.test(s) ? s : s + 'Z');
    return isNaN(t) ? NaN : t / 1000;
  }
  function toJd(v, fmt, headerName) {
    const n = num(v);
    if (fmt === 'date' || (fmt === 'auto' && isNaN(n))) {
      const sec = parseCalendar(v);
      return isNaN(sec) ? NaN : sec / 86400 + 2440587.5;
    }
    if (isNaN(n)) return NaN;
    if (fmt === 'jd') return n;
    if (fmt === 'rjd') return n + 2400000;
    if (fmt === 'mjd') return n + 2400000.5;
    if (fmt === 'excel') return n + 2415018.5;
    // automatique
    if (n > 2000000) return n;
    if (/mjd/i.test(headerName)) return n + 2400000.5;
    if (n > 55000 && n < 100000) return n + 2400000;
    if (n > 20000 && n <= 55000) return n + 2415018.5; // numéro de série Excel
    return NaN;
  }

  // ── Mapping & données ─────────────────────────────────────────────────────
  function buildMapping() {
    const box = $('curve-mapping');
    box.hidden = false;
    // colonnes supplémentaires (rapport AAVSO) ajoutées dynamiquement une seule fois
    if (!box.querySelector('[data-map="cname"]')) {
      const row = document.createElement('details');
      row.innerHTML = '<summary>Colonnes supplémentaires pour le rapport AAVSO</summary><div class="row"></div>';
      const r = row.querySelector('.row');
      EXTRA.forEach((k) => { const l = document.createElement('label'); l.innerHTML = `${k.toUpperCase()} <select data-map="${k}"></select>`; r.appendChild(l); });
      box.appendChild(row);
    }
    box.querySelectorAll('select[data-map]').forEach((sel) => {
      const key = sel.dataset.map;
      sel.innerHTML = '';
      sel.add(new Option('—', ''));
      table.header.forEach((h, i) => sel.add(new Option(h || `Colonne ${i + 1}`, String(i))));
      let idx = table.header.findIndex((h) => FIELDS[key].test(h));
      if (key === 'mag' && idx < 0) idx = table.header.findIndex((h) => /mag/i.test(h) && !FIELDS.err.test(h) && !/^(c|k)mag$/i.test(h));
      if (key === 'date' && idx < 0) idx = 0;
      sel.value = idx >= 0 ? String(idx) : '';
    });
    rebuild();
  }

  function mapping() {
    const m = {};
    document.querySelectorAll('#curve-mapping select[data-map]').forEach((s) => { m[s.dataset.map] = s.value === '' ? -1 : +s.value; });
    return m;
  }

  // Toutes les mesures normalisées
  function allPoints() {
    if (!table) return [];
    const m = mapping(), fmt = $('curve-datefmt').value, defName = $('curve-defname').value.trim();
    const pts = [];
    for (const r of table.rows) {
      const jd = toJd(r[m.date], fmt, table.header[m.date] || '');
      const magRaw = m.mag >= 0 ? String(r[m.mag] ?? '').trim() : '';
      const fainter = magRaw.startsWith('<');
      const mag = num(magRaw.replace(/^</, ''));
      if (!isFinite(jd) || !isFinite(mag)) continue;
      const p = {
        jd, mag, fainter,
        err: m.err >= 0 ? num(r[m.err]) : NaN,
        filter: m.filter >= 0 ? normFilter(r[m.filter]) : '',
        name: (m.name >= 0 ? String(r[m.name] ?? '').trim() : '') || defName || 'Étoile',
      };
      EXTRA.forEach((k) => { if (m[k] >= 0) p[k] = String(r[m[k]] ?? '').trim(); });
      pts.push(p);
    }
    return pts.sort((a, b) => a.jd - b.jd);
  }

  function rebuild() {
    const pts = allPoints();
    const names = [...new Set(pts.map((p) => p.name))];
    const sel = $('curve-star'), prev = sel.value;
    sel.innerHTML = '';
    names.forEach((n) => sel.add(new Option(`${n} (${pts.filter((p) => p.name === n).length})`, n)));
    if (names.includes(prev)) sel.value = prev;
    showStar();
  }

  function showStar() {
    const name = $('curve-star').value;
    current = allPoints().filter((p) => p.name === name);
    $('curve-card').hidden = !current.length;
    if (!current.length) { if (table) app.toast('Aucune mesure exploitable : vérifiez les colonnes Date et Magnitude'); return; }
    const s = Catalog.find(name);
    $('curve-title').textContent = `Courbe de lumière — ${name}${s ? ` (${s.type}, ${s.max}–${s.min ?? '?'})` : ''}`;
    if (s && s.period && !$('curve-period').dataset.user) { $('curve-period').value = s.period; $('curve-epoch').value = s.epoch || ''; }
    draw();
    renderTable();
  }

  function colorFor(filter, i) {
    const css = getComputedStyle(document.documentElement);
    const slot = FILTER_COLORS[filter] || [5, 2, 4, 6, 7, 8, 3, 1][i % 8];
    return css.getPropertyValue('--series-' + slot).trim();
  }

  function draw() {
    const xMode = $('curve-x').value;
    $('curve-period-wrap').hidden = $('curve-epoch-wrap').hidden = xMode !== 'phase';
    const P = num($('curve-period').value), E = num($('curve-epoch').value) || (current[0] && current[0].jd);
    if (xMode === 'phase' && !(P > 0)) { app.toast('Indiquez une période pour replier la courbe'); }
    const css = getComputedStyle(document.documentElement), c = (v) => css.getPropertyValue(v).trim();
    const filters = [...new Set(current.map((p) => p.filter || '—'))];
    const xOf = (p) => (xMode === 'date' ? Astro.dateFromJd(p.jd).getTime() : xMode === 'jd' ? p.jd : (((p.jd - E) / P) % 1 + 1) % 1);
    const datasets = filters.map((f, i) => {
      const color = colorFor(f, i);
      let data = current.filter((p) => (p.filter || '—') === f).map((p) => ({ x: xOf(p), y: p.mag, err: p.err, jd: p.jd, fainter: p.fainter }));
      if (xMode === 'phase' && P > 0) data = data.concat(data.map((d) => Object.assign({}, d, { x: d.x + 1 })));
      return {
        label: f, data, backgroundColor: color, borderColor: c('--surface'), borderWidth: 1, pointRadius: 4, pointHoverRadius: 6,
        pointStyle: data.map((d) => (d.fainter ? 'triangle' : 'circle')),
      };
    });
    $('curve-legend').innerHTML = datasets.length > 1 ? datasets.map((d) => `<span><i style="background:${d.backgroundColor}"></i>${d.label} (${d.data.length / (xMode === 'phase' && P > 0 ? 2 : 1)})</span>`).join('') : '';
    const errorBars = {
      id: 'errorBars',
      beforeDatasetsDraw(ch) {
        const { ctx, scales: { y } } = ch;
        ctx.save(); ctx.lineWidth = 1;
        ch.data.datasets.forEach((ds, di) => {
          if (!ch.isDatasetVisible(di)) return;
          ctx.strokeStyle = ds.backgroundColor;
          ch.getDatasetMeta(di).data.forEach((pt, k) => {
            const e = ds.data[k].err;
            if (!(e > 0)) return;
            const y1 = y.getPixelForValue(ds.data[k].y - e), y2 = y.getPixelForValue(ds.data[k].y + e);
            ctx.beginPath(); ctx.moveTo(pt.x, y1); ctx.lineTo(pt.x, y2); ctx.stroke();
          });
        });
        ctx.restore();
      },
    };
    const xFmt = (v) => (xMode === 'date' ? new Date(v).toLocaleDateString('fr-FR', { day: '2-digit', month: '2-digit', year: '2-digit' }) : xMode === 'jd' ? v.toFixed(1) : v.toFixed(2));
    if (chart) chart.destroy();
    chart = new Chart($('curve-chart'), {
      type: 'scatter',
      data: { datasets },
      options: {
        animation: false, maintainAspectRatio: false, parsing: false,
        scales: {
          x: { type: 'linear', title: { display: true, text: xMode === 'date' ? 'Date (UTC)' : xMode === 'jd' ? 'Jour julien' : 'Phase', color: c('--text-2') }, ticks: { color: c('--text-2'), callback: xFmt, maxTicksLimit: 8 }, grid: { color: c('--grid') } },
          y: { reverse: true, title: { display: true, text: 'Magnitude', color: c('--text-2') }, ticks: { color: c('--text-2') }, grid: { color: c('--grid') } },
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              title: (it) => { const d = it[0].raw; return `JD ${d.jd.toFixed(4)} — ${Astro.dateFromJd(d.jd).toISOString().replace('T', ' ').slice(0, 19)} UTC`; },
              label: (it) => { const d = it.raw; return `${it.dataset.label} : ${d.fainter ? '<' : ''}${d.y.toFixed(3)}${d.err > 0 ? ' ± ' + d.err.toFixed(3) : ''}`; },
            },
          },
        },
      },
      plugins: [errorBars],
    });
    const mags = current.map((p) => p.mag);
    const span = current[current.length - 1].jd - current[0].jd;
    $('curve-stats').textContent = `${current.length} mesures · ${filters.join(', ')} · du ${Astro.dateFromJd(current[0].jd).toLocaleDateString('fr-FR')} au ${Astro.dateFromJd(current[current.length - 1].jd).toLocaleDateString('fr-FR')} (${span.toFixed(1)} j) · magnitude ${Math.min(...mags).toFixed(2)} à ${Math.max(...mags).toFixed(2)}`;
  }

  function renderTable() {
    const extras = EXTRA.filter((k) => current.some((p) => p[k]));
    document.querySelector('#curve-table thead').innerHTML = `<tr><th>JD</th><th>Date UTC</th><th>Mag</th><th>Err</th><th>Filtre</th>${extras.map((k) => `<th>${k.toUpperCase()}</th>`).join('')}</tr>`;
    document.querySelector('#curve-table tbody').innerHTML = current.slice(0, 2000).map((p) => `<tr><td>${p.jd.toFixed(5)}</td><td>${Astro.dateFromJd(p.jd).toISOString().replace('T', ' ').slice(0, 19)}</td><td class="num">${p.fainter ? '&lt;' : ''}${p.mag}</td><td class="num">${isFinite(p.err) ? p.err : ''}</td><td>${p.filter}</td>${extras.map((k) => `<td>${String(p[k] || '').replace(/</g, '&lt;')}</td>`).join('')}</tr>`).join('');
  }

  async function onFile(e) {
    const file = e.target.files[0];
    if (!file) return;
    try {
      table = await readFile(file);
      const sheetSel = $('curve-sheet');
      if (workbook) {
        sheetSel.innerHTML = '';
        workbook.SheetNames.forEach((n) => sheetSel.add(new Option(n, n)));
        sheetSel.hidden = workbook.SheetNames.length < 2;
        table = sheetTable(workbook.SheetNames[0]);
      } else sheetSel.hidden = true;
      if (!table.header.length) throw new Error('fichier vide');
      $('curve-period').dataset.user = '';
      if (!$('curve-defname').value) $('curve-defname').value = file.name.replace(/\.[^.]+$/, '').replace(/[_-]+/g, ' ');
      buildMapping();
      app.toast(`${file.name} : ${table.rows.length} lignes lues`);
    } catch (err) {
      app.toast('Lecture impossible : ' + err.message);
    }
    e.target.value = '';
  }

  function init(appRef) {
    app = appRef;
    $('curve-file').addEventListener('change', onFile);
    const drop = $('curve-drop');
    ['dragenter', 'dragover'].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add('over'); }));
    ['dragleave', 'drop'].forEach((t) => drop.addEventListener(t, () => drop.classList.remove('over')));
    drop.addEventListener('drop', (e) => {
      e.preventDefault();
      const f = e.dataTransfer && e.dataTransfer.files[0];
      if (f) onFile({ target: { files: [f] } });
    });
    $('btn-curve-paste').onclick = () => {
      const text = $('curve-paste').value;
      if (!text.trim()) { app.toast('Collez d\'abord le contenu du fichier CSV'); return; }
      onFile({ target: { files: [new File([text], 'texte_colle.csv')] } });
    };
    // Exemple embarqué (page de démonstration) : mesures fictives de R Leo
    if (window.ASTROVAR_SAMPLE_CSV) {
      $('btn-curve-sample').hidden = false;
      $('btn-curve-sample').onclick = () => onFile({ target: { files: [new File([window.ASTROVAR_SAMPLE_CSV], 'exemple_R_Leo.csv')] } });
    }
    $('curve-sheet').addEventListener('change', () => { table = sheetTable($('curve-sheet').value); buildMapping(); });
    $('curve-mapping').addEventListener('change', (e) => { if (e.target.id === 'curve-star') showStar(); else rebuild(); });
    $('curve-defname').addEventListener('input', rebuild);
    $('curve-x').addEventListener('change', () => current.length && draw());
    ['curve-period', 'curve-epoch'].forEach((id) => $(id).addEventListener('change', () => { $('curve-period').dataset.user = '1'; if (current.length) draw(); }));
    $('btn-curve-png').onclick = () => {
      if (!chart) return;
      // fond opaque pour que l'image reste lisible hors de l'application
      const src = chart.canvas, cv = document.createElement('canvas');
      cv.width = src.width; cv.height = src.height;
      const ctx = cv.getContext('2d');
      ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--surface').trim() || '#fff';
      ctx.fillRect(0, 0, cv.width, cv.height); ctx.drawImage(src, 0, 0);
      const name = ($('curve-star').value || 'courbe').replace(/\W+/g, '_');
      Platform.saveFile(`courbe_${name}.png`, { base64: cv.toDataURL('image/png').split(',')[1] }, 'image/png').then((r) => r && app.toast('Image enregistrée'));
    };
    $('btn-curve-to-aavso').onclick = () => {
      if (!current.length) return;
      const n = Aavso.addObservations(current);
      app.toast(`${n} observation(s) ajoutée(s) au rapport AAVSO`);
      app.showTab('aavso');
    };
  }

  window.Curves = { init, parseAavsoText, toJd, parseDelimited, guessDelim };
})();
