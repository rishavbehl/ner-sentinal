/* NER Sentinel field reporter — offline service worker.
   Cache-first for the app shell so the page opens with no signal at all;
   network-only for API writes, which the page queues in IndexedDB itself. */
const CACHE = 'sentinel-field-v1';
const SHELL = ['/field', '/static/manifest.json'];

self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', e => {
  e.waitUntil(caches.keys().then(ks =>
    Promise.all(ks.filter(k => k !== CACHE).map(k => caches.delete(k)))
  ).then(() => self.clients.claim()));
});

self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);
  if (e.request.method !== 'GET') return;               // POSTs: let the page queue them
  if (url.pathname.startsWith('/api/')) return;         // never cache live API reads

  e.respondWith(
    caches.match(e.request).then(hit => {
      if (hit) {
        // refresh in the background, serve instantly from cache
        fetch(e.request).then(r => {
          if (r.ok) caches.open(CACHE).then(c => c.put(e.request, r.clone()));
        }).catch(() => {});
        return hit;
      }
      return fetch(e.request).then(r => {
        if (r.ok && (url.pathname === '/field' || url.pathname.startsWith('/static/')))
          caches.open(CACHE).then(c => c.put(e.request, r.clone()));
        return r;
      }).catch(() => caches.match('/field'));
    })
  );
});
