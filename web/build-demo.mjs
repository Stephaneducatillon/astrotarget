/**
 * Construit une version de démonstration en UN SEUL fichier HTML.
 *
 * Pourquoi : le site normal est fait de 29 modules ES, d'une feuille de style
 * et d'un référentiel JSON, servis par HTTP. Certains contextes d'affichage
 * (aperçu en bac à sable, envoi par fichier, ouverture en local) n'ont ni
 * serveur ni accès réseau sortant. Ce script produit un fichier autonome,
 * ouvrable directement, qui contient tout.
 *
 * Trois adaptations, et seulement trois :
 *  1. le référentiel des circuits est intégré au bundle plutôt que chargé par
 *     fetch ;
 *  2. le service worker n'est pas enregistré : il n'y a pas de fichier à
 *     mettre en cache, et pas d'origine stable ;
 *  3. le mode démonstration est forcé et verrouillé, parce qu'aucun appel
 *     réseau vers Jolpica, OpenF1 ou Open-Meteo ne peut aboutir depuis ce
 *     contexte. L'interface le dit au lieu d'échouer en silence.
 *
 * Usage : node build-demo.mjs [chemin/de/sortie.html]
 */

import { build } from 'esbuild';
import { readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ICI = dirname(fileURLToPath(import.meta.url));
const SORTIE = resolve(process.argv[2] || resolve(ICI, 'demo.html'));

const css = await readFile(resolve(ICI, 'css/app.css'), 'utf8');
const circuits = await readFile(resolve(ICI, 'data/circuits.json'), 'utf8');

/**
 * Remplace les points d'entrée qui supposent un serveur. Les sources du site
 * ne sont jamais modifiées : la substitution vit ici, dans le build.
 */
/**
 * Remplace les points d'entrée qui supposent un serveur. Les sources du site
 * ne sont jamais modifiées : la substitution vit ici, dans le build.
 *
 * Chaque substitution est vérifiée. Une correspondance ratée arrête le build
 * plutôt que de produire un fichier qui échouerait silencieusement à l'usage.
 */
function remplacer(source, fichier, paires) {
  let out = source;
  for (const [avant, apres] of paires) {
    if (!out.includes(avant)) {
      throw new Error(
        `build-demo : motif introuvable dans ${fichier}.\n`
        + `Le code source a changé depuis l'écriture de ce script.\n`
        + `Attendu :\n${avant}`);
    }
    out = out.replace(avant, apres);
  }
  return out;
}

const plugin = {
  name: 'demo-autonome',
  setup(b) {
    // Le référentiel circuits, servi depuis la mémoire plutôt que par fetch.
    b.onLoad({ filter: /js[\\/]data[\\/]reference\.js$/ }, async (args) => {
      const src = await readFile(args.path, 'utf8');
      return {
        loader: 'js',
        contents: remplacer(src, 'reference.js', [
          ['let cache = null;',
            `const REFERENTIEL_INTEGRE = ${JSON.stringify(circuits)};\nlet cache = null;`],
          [`  const rep = await fetch(new URL('../../data/circuits.json', import.meta.url));
  if (!rep.ok) throw new Error('Référentiel circuits illisible.');
  cache = await rep.json();`,
            '  cache = JSON.parse(REFERENTIEL_INTEGRE);'],
        ]),
      };
    });

    // Pas de service worker, et mode démonstration forcé au démarrage.
    b.onLoad({ filter: /js[\\/]app\.js$/ }, async (args) => {
      const src = await readFile(args.path, 'utf8');
      return {
        loader: 'js',
        contents: remplacer(src, 'app.js', [
          [`  if ('serviceWorker' in navigator && location.protocol.startsWith('http')) {
    navigator.serviceWorker.register(new URL('../sw.js', import.meta.url))
      .catch(() => { /* hors ligne indisponible, le site reste utilisable */ });
  }`,
            '  // Version autonome : aucun service worker à enregistrer.'],
          [`function demarrer() {
  const principal = construireCoquille();`,
            `function demarrer() {
  // Version autonome : le réseau sortant est indisponible, le jeu de
  // démonstration est donc le seul qui puisse fonctionner.
  store.definirPref('demo', true);
  const principal = construireCoquille();`],
        ]),
      };
    });

    // Le réglage « données réelles » n'a pas de sens ici : on l'explique.
    b.onLoad({ filter: /js[\\/]views[\\/]reglages\.js$/ }, async (args) => {
      const src = await readFile(args.path, 'utf8');
      return {
        loader: 'js',
        contents: remplacer(src, 'reglages.js', [
          [`      'Force le jeu de démonstration local : un week-end fictif, calé sur les jours à venir, '
      + 'qui permet de parcourir toute l\\'interface sans réseau. Rien n\\'y est officiel.'`,
            `      'Cette version de démonstration est autonome : elle ne peut joindre ni Jolpica, ni '
      + 'OpenF1, ni Open-Meteo. Le jeu local est donc le seul disponible — un week-end fictif '
      + 'calé sur les jours à venir, qui permet de parcourir toute l\\'interface. Rien n\\'y est '
      + 'officiel. La version complète du site interroge les vraies sources.'`],
        ]),
      };
    });
  },
};

const res = await build({
  entryPoints: [resolve(ICI, 'js/app.js')],
  bundle: true,
  // Module : un script inline de type module est différé par nature, il
  // s'exécute donc après l'analyse du document, que la balise soit placée
  // dans le head ou dans le body.
  format: 'esm',
  target: ['es2022'],
  charset: 'utf8',
  write: false,
  plugins: [plugin],
});

const js = res.outputFiles[0].text;

const html = `<title>Paddock</title>
<meta name="description" content="Hub week-end F1 : horaires, météo et pronostics pondérés à la difficulté.">
<style>
${css}
</style>
<noscript>
  <div style="padding:24px;max-width:60ch;margin:0 auto;font-family:system-ui,sans-serif">
    <h1>Paddock</h1>
    <p>Cette page construit ses écrans dans le navigateur — horaires convertis
    dans ton fuseau, coefficients de pronostic calculés en direct. Elle a besoin
    de JavaScript.</p>
  </div>
</noscript>
<script type="module">
${js}
</script>
`;

await writeFile(SORTIE, html, 'utf8');
const ko = (Buffer.byteLength(html) / 1024).toFixed(0);
console.log(`✅ ${SORTIE} — ${ko} Ko`);
