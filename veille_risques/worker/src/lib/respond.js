// Reponses HTTP. Module a part pour eviter un cycle index.js <-> admin.js.
//
// CORS : la PWA est servie par Pages et l'API par le Worker, donc deux
// origines distinctes. Sans en-tete, le navigateur recupere la reponse mais
// interdit au script de la lire, et le tableau de bord reste vide.
// L'origine autorisee est explicite : pas de '*' sur une API qui expose
// l'etat du foyer.

export function json(body, status = 200, env = null) {
  return new Response(JSON.stringify(body, null, 2), {
    status,
    headers: {
      'Content-Type': 'application/json; charset=utf-8',
      'Cache-Control': 'no-store',
      ...enTetesCors(env),
    },
  });
}

export function enTetesCors(env) {
  const origine = env?.ALLOWED_ORIGIN;
  if (!origine) return {};
  return {
    'Access-Control-Allow-Origin': origine,
    'Access-Control-Allow-Headers': 'Authorization,Content-Type',
    'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
    'Vary': 'Origin',
  };
}

export function preflight(env) {
  return new Response(null, { status: 204, headers: enTetesCors(env) });
}
