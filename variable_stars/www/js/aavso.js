// Onglet « Rapport AAVSO » : compléter / valider / exporter un fichier au format AAVSO
// (Extended File Format ou Visual File Format) puis le déposer sur WebObs.
(function () {
  'use strict';
  const { store } = Platform;
  const $ = (id) => document.getElementById(id);
  const WEBOBS_URL = 'https://www.aavso.org/webobs/file';

  const COLS = {
    EXTENDED: ['NAME', 'DATE', 'MAG', 'MERR', 'FILT', 'TRANS', 'MTYPE', 'CNAME', 'CMAG', 'KNAME', 'KMAG', 'AMASS', 'GROUP', 'CHART', 'NOTES'],
    VISUAL: ['NAME', 'DATE', 'MAG', 'COMMENTCODE', 'COMP1', 'COMP2', 'CHART', 'NOTES'],
  };
  const FILTERS = ['U', 'B', 'V', 'R', 'I', 'J', 'H', 'K', 'TG', 'TB', 'TR', 'CV', 'CR', 'SZ', 'SU', 'SG', 'SR', 'SI', 'STU', 'STV', 'STB', 'STY', 'STHBW', 'STHBN', 'MA', 'MB', 'MI', 'ZS', 'Y', 'HA', 'HAC', 'O', 'Vis.'];
  const NUM_OR_NA = /^(na|-?\d+(\.\d+)?)$/i;

  let app, obs = store.get('aavso.obs', []);
  let saveTimer;

  function settings() {
    return {
      obscode: $('a-obscode').value.trim().toUpperCase(),
      type: $('a-type').value,
      obstype: $('a-obstype').value,
      datetype: $('a-datetype').value,
    };
  }
  function persist() {
    clearTimeout(saveTimer);
    saveTimer = setTimeout(() => {
      store.set('aavso.obs', obs);
      store.set('aavso.settings', settings());
    }, 300);
  }

  const clean = (v) => {
    const s = String(v ?? '').trim();
    return s === '' ? 'na' : s;
  };

  const cleanNum = (v) => clean(v).replace(',', '.');

  function dupKey(o) { return [String(o.NAME).toUpperCase().replace(/\s+/g, ''), Number(o.DATE).toFixed(5), o.FILT || '', o.MAG].join('|'); }

  // points issus de l'onglet Courbes → lignes AAVSO
  function addObservations(points) {
    const seen = new Set(obs.map(dupKey));
    let n = 0;
    for (const p of points) {
      const mag = (p.fainter ? '<' : '') + p.mag;
      const o = {
        NAME: p.name, DATE: p.jd.toFixed(5), MAG: mag, MERR: isFinite(p.err) ? String(p.err) : 'na',
        FILT: p.filter || 'V', TRANS: clean(p.trans || 'NO').toUpperCase(), MTYPE: clean(p.mtype || 'STD').toUpperCase(),
        CNAME: clean(p.cname), CMAG: cleanNum(p.cmag), KNAME: clean(p.kname), KMAG: cleanNum(p.kmag),
        AMASS: cleanNum(p.amass), GROUP: clean(p.group), CHART: clean(p.chart), NOTES: clean(p.notes),
        COMMENTCODE: 'na', COMP1: clean(p.cname), COMP2: clean(p.kname),
      };
      const k = dupKey(o);
      if (seen.has(k)) continue;
      seen.add(k); obs.push(o); n++;
    }
    obs.sort((a, b) => Number(a.DATE) - Number(b.DATE));
    persist(); render();
    return n;
  }

  // ── Validation ────────────────────────────────────────────────────────────
  function validateCell(col, v, st) {
    v = String(v ?? '').trim();
    switch (col) {
      case 'NAME': return v && v !== 'na' ? '' : 'nom manquant';
      case 'DATE': {
        const n = Number(v);
        if (!v || isNaN(n)) return 'date non numérique';
        if (st.datetype !== 'EXCEL' && (n < 2400000 || n > Astro.jdFromDate(new Date()) + 1)) return 'JD hors limites';
        return '';
      }
      case 'MAG': return /^<?-?\d+(\.\d+)?$/.test(v) ? '' : 'magnitude invalide';
      case 'MERR': case 'CMAG': case 'KMAG': case 'AMASS': return NUM_OR_NA.test(v || 'na') ? '' : 'nombre ou na attendu';
      case 'FILT': return FILTERS.includes(v) ? '' : 'filtre inconnu de l\'AAVSO';
      case 'TRANS': return /^(YES|NO)$/i.test(v) ? '' : 'YES ou NO';
      case 'MTYPE': return /^(STD|DIF|ABS)$/i.test(v) ? '' : 'STD, DIF ou ABS';
      case 'COMMENTCODE': return /^(na|[A-Z]+)$/i.test(v || 'na') ? '' : 'codes lettres ou na';
      default: return '';
    }
  }
  function validateAll() {
    const st = settings(), cols = COLS[st.type], errs = [];
    if (!/^[A-Z0-9]{1,6}$/.test(st.obscode)) errs.push('Code observateur AAVSO (OBSCODE) manquant ou invalide.');
    if (!obs.length) errs.push('Aucune observation.');
    obs.forEach((o, i) => cols.forEach((c) => {
      const e = validateCell(c, o[c], st);
      if (e) errs.push(`Ligne ${i + 1}, ${c} : ${e} (« ${o[c] ?? ''} »)`);
    }));
    return errs;
  }

  // ── Rendu ───────────────────────────────────────────────────────────────────
  function render() {
    const st = settings(), cols = COLS[st.type];
    $('a-obstype-wrap').hidden = st.type === 'VISUAL';
    $('a-count').textContent = `${obs.length} observation${obs.length > 1 ? 's' : ''}`;
    document.querySelector('#a-table thead').innerHTML = `<tr><th>#</th>${cols.map((c) => `<th>${c}</th>`).join('')}<th></th></tr>`;
    const esc = (s) => String(s ?? '').replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;');
    document.querySelector('#a-table tbody').innerHTML = obs.map((o, i) => `<tr data-row="${i}"><td>${i + 1}</td>${cols.map((c) => {
      const bad = validateCell(c, o[c], st);
      return `<td><input data-col="${c}" value="${esc(o[c] ?? 'na')}" class="${bad ? 'invalid' : ''}" title="${esc(bad)}"${c === 'FILT' ? ' list="aavso-filters"' : ''}></td>`;
    }).join('')}<td><button class="del" title="Supprimer la ligne">✕</button></td></tr>`).join('');
    $('a-errors').hidden = true;
    $('a-preview').textContent = buildFile().slice(0, 20000);
  }

  function buildFile() {
    const st = settings(), cols = COLS[st.type];
    const head = [
      `#TYPE=${st.type}`,
      `#OBSCODE=${st.obscode}`,
      `#SOFTWARE=AstroVar ${app ? app.version : ''}`.trim(),
      '#DELIM=,',
      `#DATE=${st.datetype}`,
    ];
    if (st.type === 'EXTENDED') head.push(`#OBSTYPE=${st.obstype}`);
    head.push('#' + cols.join(','));
    const body = obs.map((o) => cols.map((c) => {
      let v = clean(o[c]).replace(/,/g, ';').replace(/[\r\n]+/g, ' ');
      if (['TRANS', 'MTYPE'].includes(c)) v = v.toUpperCase();
      return v;
    }).join(','));
    return head.concat(body).join('\r\n') + '\r\n';
  }

  function fileName() {
    const d = new Date(), p = (n) => String(n).padStart(2, '0');
    return `AAVSO_${settings().obscode || 'OBS'}_${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}_${p(d.getHours())}${p(d.getMinutes())}.txt`;
  }

  async function save(strict) {
    const errs = validateAll();
    const box = $('a-errors');
    if (errs.length) {
      box.hidden = false;
      box.innerHTML = `<b>${errs.length} problème(s) à corriger :</b><br>` + errs.slice(0, 50).map((e) => e.replace(/</g, '&lt;')).join('<br>') + (errs.length > 50 ? '<br>…' : '');
      if (strict) return null;
      if (!confirm('Le fichier contient des erreurs qui seront refusées par l\'AAVSO. Enregistrer quand même ?')) return null;
    }
    const name = fileName();
    await Platform.saveFile(name, buildFile(), 'text/plain;charset=utf-8');
    app.toast('Fichier enregistré : ' + name);
    return name;
  }

  async function submit() {
    const name = await save(true);
    if (!name) { app.toast('Corrigez les erreurs avant la transmission'); return; }
    alert(`Le fichier « ${name} » a été enregistré.\n\nLa page WebObs de l'AAVSO va s'ouvrir :\n1. connectez-vous avec votre compte AAVSO ;\n2. choisissez le fichier « ${name} » ;\n3. cliquez sur « Submit ».\n\nL'AAVSO ne propose pas d'API publique de dépôt : l'envoi final se fait donc sur leur site, avec vos identifiants.`);
    Platform.openExternal(WEBOBS_URL);
  }

  function openExisting(e) {
    const file = e.target.files[0];
    if (!file) return;
    file.text().then((text) => {
      const a = Curves.parseAavsoText(text);
      if (!a.params.TYPE) { app.toast('Ce fichier n\'est pas au format AAVSO (#TYPE= absent)'); return; }
      if (obs.length && !confirm(`Ajouter les ${a.rows.length} lignes du fichier aux ${obs.length} observations déjà présentes ?\n(Annuler = remplacer)`)) obs = [];
      const type = a.visual ? 'VISUAL' : 'EXTENDED';
      $('a-type').value = type;
      if (a.params.OBSCODE) $('a-obscode').value = a.params.OBSCODE;
      if (a.params.DATE) $('a-datetype').value = a.params.DATE.toUpperCase();
      if (a.params.OBSTYPE && [...$('a-obstype').options].some((o) => o.value === a.params.OBSTYPE.toUpperCase())) $('a-obstype').value = a.params.OBSTYPE.toUpperCase();
      const seen = new Set(obs.map(dupKey));
      let n = 0;
      for (const r of a.rows) {
        const o = {};
        a.header.forEach((h, i) => { o[h.toUpperCase()] = r[i] ?? 'na'; });
        const k = dupKey(o);
        if (seen.has(k)) continue;
        seen.add(k); obs.push(o); n++;
      }
      persist(); render();
      app.toast(`${n} observation(s) chargée(s) depuis ${file.name}`);
    });
    e.target.value = '';
  }

  function computeAirmass() {
    const loc = Sky.location();
    if (!isFinite(loc.lat) || !isFinite(loc.lon)) { app.toast('Renseignez le lieu dans l\'onglet Ciel'); return; }
    const st = settings();
    let n = 0, unknown = new Set();
    for (const o of obs) {
      const s = Catalog.find(o.NAME);
      if (!s) { unknown.add(o.NAME); continue; }
      let jd = Number(o.DATE);
      if (st.datetype === 'EXCEL') jd += 2415018.5;
      if (!isFinite(jd)) continue;
      const p = Astro.precess(s.ra, s.dec, jd);
      const am = Astro.airmass(Astro.altAz(p.ra, p.dec, jd, loc.lat, loc.lon).alt);
      o.AMASS = am ? am.toFixed(3) : 'na';
      n++;
    }
    persist(); render();
    app.toast(`Masse d'air calculée pour ${n} ligne(s) depuis ${loc.name || 'le lieu choisi'}` + (unknown.size ? ` — inconnues du catalogue : ${[...unknown].slice(0, 3).join(', ')}` : ''));
  }

  function init(appRef) {
    app = appRef;
    const s = store.get('aavso.settings', {});
    if (s.obscode) $('a-obscode').value = s.obscode;
    if (s.type) $('a-type').value = s.type;
    if (s.obstype) $('a-obstype').value = s.obstype;
    if (s.datetype) $('a-datetype').value = s.datetype;
    const dl = document.createElement('datalist');
    dl.id = 'aavso-filters';
    FILTERS.forEach((f) => dl.appendChild(new Option(f, f)));
    document.body.appendChild(dl);

    ['a-obscode', 'a-type', 'a-obstype', 'a-datetype'].forEach((id) => $(id).addEventListener('change', () => { persist(); render(); }));
    $('a-open').addEventListener('change', openExisting);
    $('btn-a-add').onclick = () => {
      const last = obs[obs.length - 1] || {};
      obs.push({ NAME: last.NAME || '', DATE: Astro.jdFromDate(new Date()).toFixed(5), MAG: '', MERR: 'na', FILT: last.FILT || 'V', TRANS: 'NO', MTYPE: 'STD', CNAME: last.CNAME || 'na', CMAG: 'na', KNAME: last.KNAME || 'na', KMAG: 'na', AMASS: 'na', GROUP: 'na', CHART: last.CHART || 'na', NOTES: 'na', COMMENTCODE: 'na', COMP1: last.COMP1 || 'na', COMP2: last.COMP2 || 'na' });
      persist(); render();
    };
    $('btn-a-clear').onclick = () => { if (obs.length && confirm('Supprimer toutes les observations du rapport ?')) { obs = []; persist(); render(); } };
    $('btn-a-airmass').onclick = computeAirmass;
    $('btn-a-save').onclick = () => save(false);
    $('btn-a-submit').onclick = submit;
    const tbody = document.querySelector('#a-table tbody');
    tbody.addEventListener('input', (e) => {
      const inp = e.target.closest('input[data-col]');
      if (!inp) return;
      const i = +inp.closest('tr').dataset.row, c = inp.dataset.col;
      obs[i][c] = inp.value;
      const bad = validateCell(c, inp.value, settings());
      inp.classList.toggle('invalid', !!bad); inp.title = bad;
      persist();
      $('a-preview').textContent = buildFile().slice(0, 20000);
    });
    tbody.addEventListener('click', (e) => {
      if (!e.target.classList.contains('del')) return;
      obs.splice(+e.target.closest('tr').dataset.row, 1);
      persist(); render();
    });
    render();
  }

  window.Aavso = { init, addObservations, buildFile, validateAll };
})();
