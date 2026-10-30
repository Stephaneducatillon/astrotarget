// Point d'entrée : onglets, thème, notifications, initialisation des modules.
(function () {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const THEMES = [['auto', 'Thème auto'], ['light', 'Thème clair'], ['dark', 'Thème sombre'], ['night', 'Mode nuit (rouge)']];

  let toastTimer;
  const app = {
    version: '1.0.0',
    toast(msg) {
      const t = $('toast');
      t.textContent = msg; t.classList.add('show');
      clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.remove('show'), 3500);
    },
    showTab(name) {
      document.querySelectorAll('.tab').forEach((b) => b.classList.toggle('active', b.dataset.tab === name));
      document.querySelectorAll('.tabpanel').forEach((p) => p.classList.toggle('active', p.id === 'tab-' + name));
      Platform.store.set('tab', name);
    },
  };

  function applyTheme(t) {
    if (t === 'auto') document.documentElement.removeAttribute('data-theme');
    else document.documentElement.setAttribute('data-theme', t);
    $('btn-theme').textContent = THEMES.find((x) => x[0] === t)[1];
    Platform.store.set('theme', t);
  }

  document.addEventListener('DOMContentLoaded', () => {
    applyTheme(Platform.store.get('theme', 'auto'));
    $('btn-theme').onclick = () => {
      const cur = Platform.store.get('theme', 'auto');
      const i = THEMES.findIndex((x) => x[0] === cur);
      applyTheme(THEMES[(i + 1) % THEMES.length][0]);
      Sky.compute(); // redessine avec les nouvelles couleurs
    };
    document.querySelectorAll('.tab').forEach((b) => { b.onclick = () => app.showTab(b.dataset.tab); });
    if (!window.GCVS_CATALOG) app.toast('Catalogue GCVS introuvable (data/gcvs.js)');
    Sky.init(app);
    Curves.init(app);
    Aavso.init(app);
    app.showTab(Platform.store.get('tab', 'sky'));
  });

  window.App = app;
})();
