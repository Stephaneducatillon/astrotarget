// Tests rapides (Node, sans navigateur) des calculs et des formats de fichiers.
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const www = path.join(__dirname, '..', 'www');
const ctx = { window: {}, console, TextEncoder, btoa: (s) => Buffer.from(s, 'binary').toString('base64'), localStorage: null };
vm.createContext(ctx);
for (const f of ['data/gcvs.js', 'js/platform.js', 'js/astro.js', 'js/catalog.js', 'js/curves.js']) {
  vm.runInContext(fs.readFileSync(path.join(www, f), 'utf8'), ctx, { filename: f });
}
const { Astro, Catalog, Curves } = ctx.window;
const near = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg} : ${a} ≠ ${b} (±${tol})`);

// Jour julien
near(Astro.jdFromDate(new Date('2000-01-01T12:00:00Z')), 2451545.0, 1e-9, 'JD J2000');
// Soleil et Lune à Paris le 30/10/2026 21:00 UT (référence : astropy)
const jd = Astro.jdFromDate(new Date('2026-10-30T21:00:00Z'));
near(Astro.sunAlt(jd, 48.8566, 2.3522), -42.953, 0.1, 'hauteur du Soleil');
near(Astro.moonAltAz(jd, 48.8566, 2.3522).alt, 8.93, 0.6, 'hauteur de la Lune');
const ev = Astro.nightEvents(jd, 48.8566, 2.3522);
near((ev.sunsetDown - Astro.jdFromDate(new Date('2026-10-30T16:34:00Z'))) * 1440, 0, 3, 'coucher du Soleil (min)');
near((ev.astroDown - Astro.jdFromDate(new Date('2026-10-30T18:21:00Z'))) * 1440, 0, 3, 'nuit astronomique (min)');
// Masse d'air
near(Astro.airmass(90), 1, 1e-6, 'masse d\'air au zénith');
near(Astro.airmass(30), 1.995, 0.01, 'masse d\'air à 30°');

// Catalogue
const stars = Catalog.load();
assert.ok(stars.length > 45000, 'catalogue GCVS chargé');
const rleo = Catalog.find('R LEO');
assert.ok(rleo && rleo.type === 'M' && rleo.fam === 'mira', 'R Leo est une Mira');
near(rleo.ra, 146.89, 0.05, 'AD de R Leo');
assert.strictEqual(Catalog.family('EA/SD'), 'ecl');
assert.strictEqual(Catalog.family('RRAB'), 'rr');

// Dates
near(Curves.toJd('2460700.5', 'auto', 'JD'), 2460700.5, 1e-9, 'JD brut');
near(Curves.toJd('60700', 'auto', 'MJD'), 2460700.5, 1e-9, 'MJD');
near(Curves.toJd('24/01/2025 20:24', 'auto', 'Date'), 2460700.35, 1e-6, 'date française');
near(Curves.toJd(45681.85, 'auto', 'Date'), 2460700.35, 1e-6, 'série Excel');

// Fichier AAVSO
const a = Curves.parseAavsoText('#TYPE=EXTENDED\n#OBSCODE=XYZ\n#DELIM=,\n#DATE=JD\n#NAME,DATE,MAG,MERR,FILT,TRANS,MTYPE,CNAME,CMAG,KNAME,KMAG,AMASS,GROUP,CHART,NOTES\nR LEO,2460700.35,9.1,0.01,V,NO,STD,ENSEMBLE,na,123,9.8,1.2,na,X123,note, avec virgule\n');
assert.strictEqual(a.params.OBSCODE, 'XYZ');
assert.strictEqual(a.rows.length, 1);
assert.strictEqual(a.rows[0][14], 'note, avec virgule');

console.log('Tous les tests sont passés ✔');
