// Produit une page autonome (un seul fichier HTML) pour une démonstration en ligne :
// CSS et scripts intégrés, Chart.js et SheetJS chargés depuis jsDelivr, exemple CSV embarqué.
// Usage : node scripts/build_artifact.js sortie.html
const fs = require('fs');
const path = require('path');
const root = path.join(__dirname, '..');
const www = path.join(root, 'www');
const out = process.argv[2] || path.join(root, 'dist', 'astrovar.html');

const CDN = {
  'lib/chart.umd.min.js': 'https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.js',
  'lib/xlsx.full.min.js': 'https://cdn.jsdelivr.net/npm/xlsx@0.18.5/dist/xlsx.full.min.js',
};
const html = fs.readFileSync(path.join(www, 'index.html'), 'utf8');
const head = html.match(/<title>.*<\/title>/)[0];
let body = html.slice(html.indexOf('<body>') + 6, html.lastIndexOf('</body>'));
const safe = (js) => js.replace(/<\/script/gi, '<\\/script');
body = body.replace(/<script src="([^"]+)"><\/script>/g, (m, src) => {
  if (CDN[src]) return `<script src="${CDN[src]}"></script>`;
  return `<script>\n${safe(fs.readFileSync(path.join(www, src), 'utf8'))}\n</script>`;
});
const sample = fs.readFileSync(path.join(root, 'samples', 'exemple_R_Leo.csv'), 'utf8').replace(/^﻿/, '');
body = body.replace('<script src="https://cdn.jsdelivr', `<script>window.ASTROVAR_SAMPLE_CSV = ${JSON.stringify(sample)};</script>\n  <script src="https://cdn.jsdelivr`);
const css = fs.readFileSync(path.join(www, 'css', 'style.css'), 'utf8');
fs.mkdirSync(path.dirname(out), { recursive: true });
fs.writeFileSync(out, `${head}\n<style>\n${css}\n</style>\n${body.trim()}\n`);
console.log('écrit', out, (fs.statSync(out).size / 1e6).toFixed(1), 'Mo');
