// Notifications ntfy.
//
// Contrainte de confidentialite : le message transite par un serveur tiers et
// le sujet est devinable si on le choisit mal. Deux consequences appliquees
// ici : le sujet vient d'un secret (jamais du depot ni du bundle PWA), et le
// corps ne contient JAMAIS d'adresse ni de nom d'etablissement - seulement un
// niveau, une cause generique et le rappel de checklist.

import { CHECKLIST } from './config.js';

const EMOJI = { rouge: 'rotating_light', orange: 'warning', vert: 'white_check_mark' };

export function composer(ruleNom, notif) {
  const { kind, to, label } = notif;

  if (kind === 'montee') {
    return {
      title: `${to.toUpperCase()} — ${ruleNom}`,
      body: [label, CHECKLIST[to]].filter(Boolean).join('\n'),
      tags: [EMOJI[to] ?? 'bell'],
    };
  }
  if (kind === 'rappel') {
    const h = notif.depuis ? Math.round((Date.now() / 1000 - notif.depuis) / 3600) : null;
    return {
      title: `${to.toUpperCase()} maintenu — ${ruleNom}`,
      body: [label, h != null ? `En cours depuis ~${h} h.` : null].filter(Boolean).join('\n'),
      tags: ['hourglass'],
    };
  }
  if (kind === 'desescalade') {
    return {
      title: `Retour en ${to} — ${ruleNom}`,
      body: label ?? 'Niveau abaisse.',
      tags: ['arrow_down'],
    };
  }
  // kind === 'fin'
  return {
    title: `Fin d'alerte — ${ruleNom}`,
    body: 'Retour au vert.',
    tags: [EMOJI.vert],
  };
}

export function composerIncident(sourceNom, incident) {
  if (incident.type === 'retablie') {
    return { title: `Source retablie — ${sourceNom}`, body: 'Les donnees reviennent.', tags: ['arrow_up'] };
  }
  if (incident.type === 'auth') {
    return {
      title: `Cle refusee — ${sourceNom}`,
      body: "Authentification rejetee. Regenerer la cle : sans elle, l'absence d'alerte ne veut plus rien dire.",
      tags: ['key'],
    };
  }
  return {
    title: `Source muette — ${sourceNom}`,
    body: `${incident.streak} echecs consecutifs. Aucune alerte ne peut venir de cette source.`,
    tags: ['mute'],
  };
}

export async function envoyer(env, { title, body, tags, priority = 'default' }) {
  if (!env.NTFY_TOPIC) return { ok: false, raison: 'NTFY_TOPIC non defini' };
  const res = await fetch(`${env.NTFY_BASE}/${env.NTFY_TOPIC}`, {
    method: 'POST',
    headers: {
      Title: asciiHeader(title),
      Priority: priority,
      Tags: (tags ?? []).join(','),
    },
    body,
    signal: AbortSignal.timeout(6000),
  });
  return { ok: res.ok, status: res.status };
}

// Les en-tetes HTTP ne transportent pas d'UTF-8 de maniere fiable : le titre
// part en ASCII, le corps (qui est le payload) garde les accents.
function asciiHeader(s) {
  return String(s)
    .normalize('NFD')
    .replace(/[̀-ͯ]/g, '')
    .replace(/[^\x20-\x7E]/g, '');
}
