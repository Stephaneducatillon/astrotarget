/**
 * Service worker.
 *
 * Objectif du livrable 3 : le dernier hub consulté reste lisible hors
 * connexion. On ne met pas les API en cache ici — c'est le rôle du cache
 * applicatif, qui sait porter l'horodatage de chaque valeur et l'afficher.
 * Le service worker ne s'occupe que de la coquille : HTML, CSS, modules,
 * référentiel des circuits.
 */

const VERSION = 'paddock-v1';
const COQUILLE = [
  './', './index.html', './manifest.webmanifest',
  './css/app.css',
  './data/circuits.json',
  './icons/icone.svg',
  './js/app.js', './js/config.js', './js/store.js', './js/contexte.js',
  './js/util/dom.js', './js/util/time.js', './js/util/format.js',
  './js/data/cache.js', './js/data/http.js', './js/data/jolpica.js',
  './js/data/openmeteo.js', './js/data/openf1.js', './js/data/reference.js', './js/data/demo.js',
  './js/model/baseline.js', './js/model/scoring.js',
  './js/views/composants.js', './js/views/hub.js', './js/views/circuit.js',
  './js/views/pilote.js', './js/views/pronostic.js', './js/views/debrief.js',
  './js/views/ligue.js', './js/views/comparateur.js', './js/views/resultats.js',
  './js/views/sources.js', './js/views/reglages.js', './js/views/resultat_reel.js',
];

self.addEventListener('install', (e) => {
  e.waitUntil((async () => {
    const c = await caches.open(VERSION);
    // addAll échoue en bloc si un seul fichier manque : on tolère les absents.
    await Promise.all(COQUILLE.map((u) => c.add(u).catch(() => {})));
    self.skipWaiting();
  })());
});

self.addEventListener('activate', (e) => {
  e.waitUntil((async () => {
    const cles = await caches.keys();
    await Promise.all(cles.filter((k) => k !== VERSION).map((k) => caches.delete(k)));
    await self.clients.claim();
  })());
});

self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method !== 'GET') return;

  const url = new URL(req.url);
  // Les appels d'API passent au réseau : leur fraîcheur est gérée par
  // l'application, qui sait dire « donnée du cache, source injoignable ».
  if (url.origin !== location.origin) return;

  e.respondWith((async () => {
    const cache = await caches.open(VERSION);
    const enCache = await cache.match(req);

    // Cache d'abord, révalidation en arrière-plan : l'affichage est immédiat
    // et la version suivante récupère le fichier à jour. waitUntil garde le
    // worker en vie le temps d'écrire, sinon la révalidation est tuée avec lui
    // et le cache ne se met jamais à jour.
    const reseau = fetch(req)
      .then((rep) => {
        if (rep.ok) e.waitUntil(cache.put(req, rep.clone()));
        return rep;
      })
      .catch(() => null);

    if (enCache) { e.waitUntil(reseau); return enCache; }
    const rep = await reseau;
    if (rep) return rep;

    // Hors ligne et jamais vu : on renvoie la coquille pour que le routeur
    // affiche au moins le dernier état connu.
    return (await cache.match('./index.html')) || Response.error();
  })());
});
