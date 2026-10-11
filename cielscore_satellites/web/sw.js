// Mode hors ligne : l'application est mise en cache à la première visite,
// et chaque année de tables consultée reste disponible sans réseau.
const VERSION = 'cielscore-satellites-v1';
const SHELL = [
  './', 'index.html', 'style.css', 'app.js', 'manifest.webmanifest', 'icon.svg',
  'lib/data.js', 'lib/engine.js', 'lib/exports.js', 'lib/phenomena.js', 'lib/photo.js',
  'lib/render.js', 'lib/tables.js', 'lib/time.js', 'vendor/astronomy.js',
  'data/jupiter/meta.json', 'data/saturn/meta.json', 'data/uranus/meta.json',
];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(VERSION).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (e) => {
  e.waitUntil(caches.keys()
    .then((keys) => Promise.all(keys.filter((k) => k !== VERSION).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});

self.addEventListener('fetch', (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== 'GET' || url.origin !== location.origin) return;
  const isData = url.pathname.endsWith('.bin');
  e.respondWith(caches.open(VERSION).then(async (cache) => {
    const hit = await cache.match(e.request);
    if (hit && isData) return hit;                     // tables : figées pour une version donnée
    try {
      const res = await fetch(e.request);              // reste : réseau d'abord, cache en secours
      if (res.ok) cache.put(e.request, res.clone());
      return res;
    } catch (err) {
      if (hit) return hit;
      throw err;
    }
  }));
});
