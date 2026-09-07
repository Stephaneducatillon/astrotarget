/**
 * Écran 7 — Ligue.
 *
 * Classement sur l'écart au modèle, score brut en colonne secondaire.
 * Les pronostics des autres membres ne sont visibles qu'après fermeture de
 * leur fenêtre (livrable 2 §8) — la règle est appliquée à l'affichage.
 *
 * Sans back-end, l'échange passe par un code de partage que les membres se
 * transmettent. L'interface le dit, plutôt que de faire croire à un serveur.
 */

import { h, vider, etatVide, annonce, badgeSource } from '../util/dom.js';
import { nombre, nomPiloteLong, delta } from '../util/format.js';
import * as T from '../util/time.js';
import * as store from '../store.js';
import * as demo from '../data/demo.js';
import { contexte } from '../contexte.js';
import { scorer, scoreBaseline, performance, comparerJoueurs, fenetres, fenetreOuverte } from '../model/scoring.js';
import { construire } from './resultat_reel.js';
import { carte, chargement, pastilleEcurie } from './composants.js';

export async function rendre(racine, params) {
  vider(racine);
  racine.append(chargement(5));

  const ctx = await contexte();
  const profil = store.profil();

  vider(racine);
  const pile = h('div', { class: 'pile' });
  racine.append(pile);

  if (!profil) {
    pile.append(etatVide('Crée d\'abord un profil',
      'Un pseudo suffit — il reste dans ce navigateur.',
      h('a', { class: 'btn btn--primaire', href: `#/pronostic/${ctx.gp?.id || ''}` }, 'Créer un profil')));
    return;
  }

  const mesLigues = store.ligues();
  pile.append(blocGestion(mesLigues, () => rendre(racine, params)));

  if (!mesLigues.length) {
    pile.append(etatVide('Aucune ligue',
      'Crée une ligue et transmets son code à tes amis, ou rejoins la leur. De 2 à 50 membres.'));
    return;
  }

  const codeActif = params?.code && mesLigues.some((l) => l.code === params.code)
    ? params.code : mesLigues[0].code;
  const ligue = mesLigues.find((l) => l.code === codeActif);

  if (mesLigues.length > 1) {
    pile.append(h('div', { class: 'segments' },
      mesLigues.map((l) => h('button', {
        'aria-pressed': String(l.code === codeActif),
        onclick: () => { location.hash = `#/ligue/${l.code}`; },
      }, l.nom))));
  }

  pile.append(await blocClassement(ligue, ctx));
  pile.append(blocPartage(ligue, ctx, () => rendre(racine, params)));
  pile.append(blocPronosticsMembres(ligue, ctx));
}

function blocGestion(ligues, apres) {
  const champNom = h('input', { class: 'champ', placeholder: 'Nom de la ligue', maxlength: '40' });
  const champCode = h('input', { class: 'champ', placeholder: 'Code à 6 caractères', maxlength: '6', style: { textTransform: 'uppercase' } });

  return carte('Tes ligues', h('div', {},
    ligues.length
      ? h('ul', { class: 'liste-nue' }, ligues.map((l) => h('li', { class: 'ligne-pro' },
        h('div', { class: 'ligne-pro-lib' }, l.nom,
          h('div', { class: 'note mono' }, `code ${l.code} · ${l.membres.length} membre(s)`)),
        h('button', {
          class: 'btn btn--fin',
          onclick: () => { navigator.clipboard?.writeText(l.code); annonce(`Code ${l.code} copié.`); },
        }, 'Copier le code'),
        h('button', {
          class: 'btn btn--fin',
          onclick: () => { store.quitterLigue(l.code); apres(); },
        }, 'Quitter'))))
      : null,

    h('div', { style: { marginTop: 'var(--esp-4)' } },
      h('label', { class: 'etiquette' }, 'Créer une ligue'),
      h('div', { class: 'rangee' }, champNom,
        h('button', {
          class: 'btn btn--primaire',
          onclick: () => {
            try { const l = store.creerLigue(champNom.value); annonce(`Ligue créée, code ${l.code}.`); apres(); }
            catch (e) { annonce(e.message); }
          },
        }, 'Créer'))),

    h('div', { style: { marginTop: 'var(--esp-4)' } },
      h('label', { class: 'etiquette' }, 'Rejoindre avec un code'),
      h('div', { class: 'rangee' }, champCode,
        h('button', {
          class: 'btn',
          onclick: () => {
            try { store.rejoindreLigue(champCode.value); apres(); }
            catch (e) { annonce(e.message); }
          },
        }, 'Rejoindre')))));
}

async function blocClassement(ligue, ctx) {
  const emplacement = carte(`${ligue.nom} — classement`, chargement(4), { source: 'CALC' });
  const corps = emplacement.querySelector('.carte-corps');

  // Grands Prix déjà courus et notables.
  const joues = (ctx.calendrier || []).filter((gp) =>
    new Date(gp.course_utc || gp.debut_utc).getTime() < Date.now() - 2 * 3600000);

  const agrege = new Map(); // id membre → cumul
  const ajouter = (profil, valeurs) => {
    const cle = profil.id;
    const a = agrege.get(cle) || {
      id: cle, pseudo: profil.pseudo, inscrit_ts: profil.inscrit_ts || 0,
      brut: 0, performance: 0, positions_exactes: 0, hardcore: 0, gp_joues: 0,
    };
    a.brut += valeurs.brut; a.performance += valeurs.performance;
    a.positions_exactes += valeurs.exactes; a.hardcore += valeurs.hardcore;
    a.gp_joues += 1;
    agrege.set(cle, a);
  };

  for (const gp of joues.slice(-10)) {
    let reel = null;
    try {
      reel = await construire(gp, {
        modeDemo: ctx.mode_demo,
        resultatsDemo: ctx.mode_demo ? demo.resultatsDemo(gp.round) : null,
        qualifDemo: ctx.mode_demo ? demo.qualifDemo(gp.round) : null,
      });
    } catch { reel = null; }
    if (!reel) continue;

    const candidats = [];
    const mien = store.pronostic(gp.id);
    if (mien?.soumis) candidats.push({ profil: store.profil(), pronostic: mien });
    for (const p of store.pronosticsLigue(ligue.code, gp.id)) candidats.push(p);

    for (const { profil, pronostic } of candidats) {
      const duelImpose = pronostic.baseline_gel?.duels?.[0];
      const reelComplet = { ...reel, duel_gagnant: duelImpose ? reel.duels[duelImpose.team_id] ?? null : null };
      const r = scorer(pronostic, reelComplet);
      if (r.neutralise) continue;
      const ref = scoreBaseline(pronostic.baseline_gel, reelComplet);
      ajouter(profil, {
        brut: r.total,
        performance: performance(r.total, ref),
        exactes: r.lignes.filter((l) => l.cle.startsWith('grille:') && l.precision === 'exact').length,
        hardcore: pronostic.fenetre === 'hardcore' ? 1 : 0,
      });
    }
  }

  const lignes = [...agrege.values()].sort(comparerJoueurs);
  vider(corps);

  if (!lignes.length) {
    corps.append(etatVide('Pas encore de score',
      'Le classement apparaît dès qu\'un Grand Prix pronostiqué a ses résultats publiés.'));
    return emplacement;
  }

  corps.append(
    h('div', { class: 'tableau-enrobage' },
      h('table', { class: 't' },
        h('thead', {}, h('tr', {},
          h('th', {}, ''), h('th', {}, 'Joueur'),
          h('th', { class: 'num' }, 'Écart au modèle'),
          h('th', { class: 'num' }, 'Brut'),
          h('th', { class: 'num' }, 'GP'))),
        h('tbody', {},
          lignes.map((l, i) => h('tr', {},
            h('td', { class: 'pos' }, i + 1),
            h('td', {}, l.pseudo, l.id === store.profil()?.id ? h('span', { class: 'badge', style: { marginLeft: '6px' } }, 'toi') : null),
            h('td', { class: 'num', style: { color: l.performance > 0 ? 'var(--ok)' : l.performance < 0 ? 'var(--danger)' : 'inherit', fontWeight: 700 } },
              delta(l.performance, 0)),
            h('td', { class: 'num txt-2' }, nombre(l.brut, 0)),
            h('td', { class: 'num txt-3' }, l.gp_joues)))))),
    h('p', { class: 'note', style: { marginTop: 'var(--esp-3)' } },
      'Classement sur l\'écart au modèle. Le score brut reste affiché en colonne secondaire : '
      + 'c\'est lui qui parle à un joueur occasionnel, l\'écart qui départage les analystes. '
      + 'Égalité départagée par positions exactes, puis pronostics hardcore, puis GP joués, puis antériorité.'));

  return emplacement;
}

function blocPartage(ligue, ctx, apres) {
  const gp = ctx.gp;
  const mien = gp ? store.pronostic(gp.id) : null;
  const zone = h('div');
  const champ = h('textarea', { class: 'champ', rows: '3', placeholder: 'Colle ici le code reçu d\'un autre membre' });

  return carte('Partager les pronostics', h('div', {},
    h('p', { class: 'note' },
      'Ce site n\'a pas de serveur : chacun exporte son pronostic en un code et le colle dans la '
      + 'ligue des autres. C\'est manuel, mais rien ne quitte ton navigateur sans que tu le décides.'),

    mien?.soumis
      ? h('div', { style: { marginTop: 'var(--esp-3)' } },
        h('button', {
          class: 'btn',
          onclick: () => {
            const code = store.encoderPartage({ profil: store.profil(), pronostic: mien });
            vider(zone).append(
              h('div', { class: 'code-partage' }, code),
              h('button', {
                class: 'btn btn--fin', style: { marginTop: 'var(--esp-2)' },
                onclick: () => { navigator.clipboard?.writeText(code); annonce('Code copié.'); },
              }, 'Copier'));
          },
        }, `Exporter mon pronostic — ${gp.nom}`),
        zone)
      : h('p', { class: 'txt-3 pt-s' }, 'Aucun pronostic soumis sur le Grand Prix en cours.'),

    h('div', { style: { marginTop: 'var(--esp-4)' } },
      h('label', { class: 'etiquette' }, 'Importer le pronostic d\'un membre'),
      champ,
      h('button', {
        class: 'btn', style: { marginTop: 'var(--esp-2)' },
        onclick: () => {
          try {
            const charge = store.decoderPartage(champ.value);
            store.importerPronostic(ligue.code, charge);
            annonce(`Pronostic de ${charge.profil.pseudo} importé.`);
            apres();
          } catch (e) { annonce(`Code illisible : ${e.message}`); }
        },
      }, 'Importer'))));
}

function blocPronosticsMembres(ligue, ctx) {
  const gp = ctx.gp;
  if (!gp) return null;
  const f = fenetres(gp);
  const membres = store.pronosticsLigue(ligue.code, gp.id);
  const mien = store.pronostic(gp.id);

  const visibles = [];
  if (mien?.soumis) visibles.push({ profil: store.profil(), pronostic: mien, moi: true });
  for (const m of membres) {
    const fen = m.pronostic.fenetre === 'hardcore' ? f.hardcore : f.standard;
    // Visible seulement après fermeture de SA fenêtre.
    if (!fenetreOuverte(fen)) visibles.push({ ...m, moi: false });
  }

  const caches = membres.length + (mien?.soumis ? 1 : 0) - visibles.length;

  return carte(`Pronostics — ${gp.nom}`, h('div', {},
    visibles.length
      ? h('div', { class: 'tableau-enrobage' },
        h('table', { class: 't' },
          h('thead', {}, h('tr', {},
            h('th', {}, 'Joueur'), h('th', {}, 'Fenêtre'),
            h('th', {}, 'Top 3 pronostiqué'), h('th', {}, 'Pole'))),
          h('tbody', {},
            visibles.map(({ profil, pronostic, moi }) => {
              const nomDe = (id) => {
                const p = pronostic.baseline_gel?.pilotes?.find((x) => x.driver_id === id);
                return p ? p.nom : id;
              };
              return h('tr', {},
                h('td', {}, profil.pseudo, moi ? h('span', { class: 'badge', style: { marginLeft: '6px' } }, 'toi') : null),
                h('td', {}, h('span', { class: 'badge' }, pronostic.fenetre === 'hardcore' ? 'hardcore ×1,5' : 'standard')),
                h('td', { class: 'pt-s' }, (pronostic.grille || []).slice(0, 3).map(nomDe).join(' · ')),
                h('td', { class: 'pt-s' }, nomDe(pronostic.pole)));
            }))))
      : etatVide('Rien de visible pour l\'instant',
        'Les pronostics des autres membres n\'apparaissent qu\'après fermeture de leur fenêtre.'),
    caches > 0
      ? h('p', { class: 'note', style: { marginTop: 'var(--esp-3)' } },
        `${caches} pronostic(s) masqué(s) : leur fenêtre n'est pas encore fermée.`)
      : null));
}
