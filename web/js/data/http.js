/**
 * Couche d'accès HTTP unique.
 *
 * Parade au risque « changement de format des flux OpenF1 » (livrable 4) :
 * aucune vue n'appelle `fetch` directement. Les vues consomment les entités
 * normalisées exposées par les modules de `js/data/`, qui seuls connaissent
 * la forme des réponses. Le jour où une source change, un seul module bouge.
 *
 * Note d'architecture : le livrable 4 recommande que cette couche vive côté
 * serveur. Ce site est volontairement sans back-end (déployable en statique),
 * elle vit donc dans le front — mais elle reste isolée pour pouvoir être
 * déplacée derrière une API sans toucher aux vues.
 */

const EN_COURS = new Map(); // dédoublonne les appels simultanés identiques

export class ErreurSource extends Error {
  constructor(code, message, cause) {
    super(message);
    this.name = 'ErreurSource';
    this.codeSource = code;
    this.cause = cause;
  }
}

export async function getJSON(codeSource, url, { timeout = 12000, signal } = {}) {
  const cle = `${codeSource}|${url}`;
  if (EN_COURS.has(cle)) return EN_COURS.get(cle);

  const p = (async () => {
    const ctrl = new AbortController();
    const minuteur = setTimeout(() => ctrl.abort(), timeout);
    if (signal) signal.addEventListener('abort', () => ctrl.abort(), { once: true });
    try {
      const rep = await fetch(url, { signal: ctrl.signal, headers: { Accept: 'application/json' } });
      if (rep.status === 429) {
        throw new ErreurSource(codeSource, 'Limite de débit atteinte. Les données affichées viennent du cache.');
      }
      if (!rep.ok) {
        throw new ErreurSource(codeSource, `Réponse ${rep.status} de la source.`);
      }
      return await rep.json();
    } catch (e) {
      if (e instanceof ErreurSource) throw e;
      if (e.name === 'AbortError') throw new ErreurSource(codeSource, 'Délai dépassé.', e);
      throw new ErreurSource(codeSource, 'Source injoignable (réseau ou CORS).', e);
    } finally {
      clearTimeout(minuteur);
      EN_COURS.delete(cle);
    }
  })();

  EN_COURS.set(cle, p);
  return p;
}
