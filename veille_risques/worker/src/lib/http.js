// Acces reseau : timeout, erreurs typees, pas de retry aveugle.
//
// La distinction 401/403 vs 5xx compte : une cle expiree est une panne
// permanente qui demande une action humaine, un 503 est passager. Les
// confondre, c'est decouvrir la cle morte le jour de la tempete.

export class SourceError extends Error {
  constructor(kind, message, status) {
    super(message);
    this.kind = kind;   // 'auth' | 'http' | 'network' | 'timeout' | 'parse'
    this.status = status ?? null;
  }
  get permanent() {
    return this.kind === 'auth';
  }
}

const TIMEOUT_MS = 8000;

export async function fetchJson(url, options = {}) {
  const { timeout = TIMEOUT_MS, ...init } = options;
  let res;
  try {
    res = await fetch(url, { ...init, signal: AbortSignal.timeout(timeout) });
  } catch (err) {
    const kind = err?.name === 'TimeoutError' ? 'timeout' : 'network';
    throw new SourceError(kind, `${kind} sur ${hostOf(url)}: ${err?.message ?? err}`);
  }

  if (res.status === 401 || res.status === 403) {
    throw new SourceError('auth', `authentification refusee (${res.status}) sur ${hostOf(url)}`, res.status);
  }
  if (!res.ok) {
    throw new SourceError('http', `HTTP ${res.status} sur ${hostOf(url)}`, res.status);
  }

  try {
    return await res.json();
  } catch (err) {
    throw new SourceError('parse', `reponse illisible de ${hostOf(url)}: ${err?.message ?? err}`);
  }
}

function hostOf(url) {
  try {
    return new URL(url).host;
  } catch {
    return String(url).slice(0, 60);
  }
}

// Les collectes sont independantes : une source qui tombe ne doit pas empecher
// les autres d'aboutir. On attend tout le monde et on trie ensuite.
export async function collectAll(tasks) {
  const settled = await Promise.allSettled(tasks.map((t) => t.run()));
  return tasks.map((task, i) => {
    const r = settled[i];
    return r.status === 'fulfilled'
      ? { id: task.id, ok: true, data: r.value }
      : { id: task.id, ok: false, error: r.reason };
  });
}
