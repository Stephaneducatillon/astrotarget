// Copie les bibliothèques navigateur (Chart.js, SheetJS) dans www/lib pour un fonctionnement hors ligne.
const fs = require('fs');
const path = require('path');
const root = path.join(__dirname, '..');
const lib = path.join(root, 'www', 'lib');
fs.mkdirSync(lib, { recursive: true });
const files = [
  ['node_modules/chart.js/dist/chart.umd.js', 'chart.umd.min.js'],
  ['node_modules/xlsx/dist/xlsx.full.min.js', 'xlsx.full.min.js'],
];
for (const [src, dst] of files) {
  fs.copyFileSync(path.join(root, src), path.join(lib, dst));
  console.log('copié', dst);
}
