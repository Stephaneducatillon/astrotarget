// Service worker : la coque de l'application doit s'ouvrir sans reseau.
//
// Les DONNEES ne sont volontairement pas mises en cache ici : elles passent
// par localStorage cote app.js, avec leur horodatage, pour qu'un etat perime
// soit toujours affiche comme tel. Un cache HTTP transparent servirait un
// vieux "vert" sans le dire - exactement ce qu'il faut eviter.

const CACHE = 'veille-coque-v1';
const COQUE = ['./', 'index.html', 'app.js', 'manifest.webmanifest'];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(COQUE)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys()
      .then((cles) => Promise.all(cles.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  );
});

self.addEventListener('fetch', (e) => {
  const url = new URL(e.request.url);
  // L'API n'est jamais servie depuis le cache : app.js gere lui-meme le repli.
  if (url.pathname.startsWith('/api/')) return;
  if (e.request.method !== 'GET') return;

  e.respondWith(
    caches.match(e.request).then((cache) =>
      cache ?? fetch(e.request).then((res) => {
        if (res.ok && url.origin === self.location.origin) {
          const copie = res.clone();
          caches.open(CACHE).then((c) => c.put(e.request, copie));
        }
        return res;
      }).catch(() => caches.match('index.html')),
    ),
  );
});
