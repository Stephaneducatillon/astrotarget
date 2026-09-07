/**
 * Écran 1 — Hub week-end (page d'accueil).
 *
 * Doit répondre à trois questions en moins de cinq secondes :
 * quand ça roule, quel temps il fera, où j'en suis de mon pronostic.
 * Le compte à rebours et la météo de la session la plus proche restent
 * toujours au-dessus de la ligne de flottaison.
 */

import { h, vider, badgeSource, badgeFraicheur, erreurSource, etatVide } from '../util/dom.js';
import * as T from '../util/time.js';
import { SESSIONS } from '../config.js';
import { contexte, prochaineSession, sessionEnCours, sessionsParJour } from '../contexte.js';
import * as om from '../data/openmeteo.js';
import * as of1 from '../data/openf1.js';
import * as demo from '../data/demo.js';
import * as jolpica from '../data/jolpica.js';
import * as store from '../store.js';
import { fenetres, fenetreOuverte } from '../model/scoring.js';
import { carte, blocMeteo, chargement, bandeauDemo, bandeauPerime, cellulePilote } from './composants.js';

let minuteur = null;

export async function rendre(racine) {
  clearInterval(minuteur);
  vider(racine);

  const squelettes = h('div', { class: 'pile' },
    carte('Week-end', chargement(3)),
    carte('Météo', chargement(3)));
  racine.append(squelettes);

  let ctx;
  try {
    ctx = await contexte();
  } catch (e) {
    vider(racine);
    racine.append(erreurSource('JOL', e.message, () => rendre(racine)));
    return;
  }

  vider(racine);
  const pile = h('div', { class: 'pile' });
  racine.append(pile);

  if (ctx.mode_demo) pile.append(bandeauDemo());
  else if (ctx.calendrier_perime) pile.append(bandeauPerime('Jolpica'));

  if (!ctx.gp) {
    pile.append(etatVide('Aucun Grand Prix au calendrier',
      `La saison ${ctx.saison} n'est pas encore publiée par Jolpica.`));
    return;
  }

  const { gp, circuit } = ctx;
  const maintenant = new Date();
  const enCours = sessionEnCours(gp, maintenant);
  const prochaine = prochaineSession(gp, maintenant);
  const cible = enCours || prochaine || gp.sessions[gp.sessions.length - 1];

  // ── Hors week-end : le hub bascule sur « prochain GP dans X jours ». ──────
  const joursAvant = Math.ceil((new Date(gp.debut_utc) - maintenant) / 86400000);
  const horsWeekEnd = !ctx.en_week_end && joursAvant > 2;

  pile.append(blocHeros(gp, circuit, cible, enCours, horsWeekEnd, joursAvant));
  pile.append(blocHoraires(gp, circuit, cible));

  const colonnes = h('div', { class: 'grille-2' });
  const gauche = h('div', { class: 'pile' });
  const droite = h('div', { class: 'pile' });
  colonnes.append(gauche, droite);
  pile.append(colonnes);

  // Météo : bloc prioritaire, chargé indépendamment.
  const emplacementMeteo = carte('Météo', chargement(3),
    { source: ctx.mode_demo ? 'DÉMO' : 'OM' });
  gauche.append(emplacementMeteo);
  chargerMeteo(emplacementMeteo, ctx, cible);

  // Bannière pronostic.
  gauche.append(blocPronostic(gp, ctx.saison, maintenant));

  if (horsWeekEnd && ctx.gp_precedent) {
    gauche.append(blocDebriefPrecedent(ctx));
  }

  // Dernières infos : documents FIA et direction de course AVANT la presse.
  const emplacementInfos = carte('Dernières infos', chargement(3),
    { source: ctx.mode_demo ? 'DÉMO' : 'OF1' });
  droite.append(emplacementInfos);
  chargerInfos(emplacementInfos, ctx);

  droite.append(blocResultats(ctx));

  // Le compte à rebours est la seule chose qui bouge à la seconde.
  minuteur = setInterval(() => {
    const el = racine.querySelector('[data-rebours]');
    if (!el) { clearInterval(minuteur); return; }
    el.textContent = T.rebours(new Date(el.dataset.rebours)).texte;
  }, 1000);
}

function blocHeros(gp, circuit, cible, enCours, horsWeekEnd, joursAvant) {
  const dateCible = cible ? new Date(cible.debut_utc) : null;
  const libelle = cible ? (SESSIONS[cible.type]?.long || cible.type) : '—';

  return h('section', { class: 'heros' },
    h('div', { class: 'heros-sur' }, `Saison ${gp.saison} · Manche ${gp.round}${gp.format === 'sprint' ? ' · format sprint' : ''}`),
    h('h1', {}, gp.nom),
    h('div', { class: 'heros-lieu' },
      [circuit?.nom || gp.circuit_nom, gp.ville, gp.pays].filter(Boolean).join(' · ')),
    h('div', { class: 'rebours' },
      enCours
        ? h('span', { class: 'rebours-val', style: { color: 'var(--accent)' } }, `${libelle} en cours`)
        : h('span', { class: 'rebours-val mono', 'data-rebours': cible?.debut_utc || '' },
          T.rebours(dateCible).texte),
      h('span', { class: 'rebours-lib' },
        enCours ? 'session en piste'
          : horsWeekEnd
            ? `avant les ${SESSIONS[cible?.type]?.long || 'premiers essais'} — prochain GP dans ${joursAvant} jours`
            : `avant ${libelle}`)),
    !cible?.heure_connue
      ? h('p', { class: 'note', style: { marginTop: 'var(--esp-3)' } },
        'Heure de session non publiée par la source : la date seule est connue, le compte à rebours part de minuit UTC.')
      : null);
}

function blocHoraires(gp, circuit, cible) {
  const maintenant = Date.now();
  const jours = sessionsParJour(gp, (d) => T.jourCourt(d));
  const tzCircuit = circuit?.tz;
  const decale = tzCircuit && T.fuseauDifferent(new Date(gp.debut_utc), tzCircuit);

  return carte('Programme', h('div', {},
    jours.map(({ jour, sessions }) =>
      h('div', { class: 'jour' },
        h('div', { class: 'jour-nom' }, jour),
        h('div', { class: 'jour-sessions' },
          sessions.map((s) => {
            const passe = s.date.getTime() + 3600000 < maintenant;
            const estCible = cible && s.type === cible.type;
            return h('div', {
              class: `creneau ${passe ? 'creneau--passe' : ''} ${estCible ? 'creneau--prochain' : ''} ${s.type === 'R' ? 'creneau--course' : ''}`,
              title: `${SESSIONS[s.type]?.long || s.type} — ${T.dateHeure(s.date)} (${T.nomFuseau(s.date)})`,
            },
              h('span', { class: 'creneau-nom' }, s.label),
              h('span', { class: 'creneau-h' }, s.heure_connue ? T.heure(s.date) : '—'));
          })))),
    h('div', { class: 'rangee', style: { marginTop: 'var(--esp-3)' } },
      h('span', { class: 'note' },
        `Heures dans ton fuseau : ${T.FUSEAU_LOCAL} (${T.nomFuseau(new Date(gp.debut_utc))})`),
      decale ? h('span', { class: 'badge badge--fuseau', title: `Le circuit est en ${tzCircuit}` },
        `circuit en ${tzCircuit}`) : null)),
  { source: gp.source === 'DEMO' ? 'DÉMO' : 'JOL' });
}

async function chargerMeteo(emplacement, ctx, cible) {
  const corps = emplacement.querySelector('.carte-corps');
  const lat = ctx.circuit?.lat, lon = ctx.circuit?.lon;
  const dateCible = cible ? new Date(cible.debut_utc) : null;
  const libelle = cible ? `${SESSIONS[cible.type]?.long || cible.type}` : 'prochaine session';

  if (lat === null || lat === undefined) {
    vider(corps).append(erreurSource('OM',
      'Coordonnées du circuit inconnues : la météo est appelée sur lat/lon, sans elles il n\'y a pas d\'appel possible.'));
    return;
  }

  try {
    let heures, ts, perime = false;
    if (ctx.mode_demo) {
      heures = demo.meteoDemo(lat, lon); ts = Date.now();
    } else {
      const r = await om.prevision(lat, lon, dateCible);
      heures = r.valeur; ts = r.ts; perime = r.perime;
    }
    vider(corps).append(blocMeteo({
      heures, ts, cible: dateCible, libelleCible: libelle,
      tzCircuit: ctx.circuit?.tz, perime, mode_demo: ctx.mode_demo,
    }));
  } catch (e) {
    vider(corps).append(erreurSource('OM', e.message, () => chargerMeteo(emplacement, ctx, cible)));
  }
}

function blocPronostic(gp, saison, maintenant) {
  const p = store.pronostic(gp.id);
  const f = fenetres(gp);
  const hardcoreOuverte = fenetreOuverte(f.hardcore, maintenant);
  const standardOuverte = fenetreOuverte(f.standard, maintenant);

  if (p?.soumis) {
    const nb = (p.grille || []).filter(Boolean).length;
    return carte('Ton pronostic', h('div', {},
      h('div', { class: 'rangee' },
        h('span', { class: 'badge badge--ok' }, 'soumis'),
        h('span', { class: 'badge' }, p.fenetre === 'hardcore' ? 'fenêtre hardcore ×1,5' : 'fenêtre standard'),
        p.joker ? h('span', { class: 'badge badge--alerte' }, `joker ×${p.joker.multiplicateur}`) : null),
      h('p', { class: 'txt-2 pt-s', style: { marginTop: 'var(--esp-3)' } },
        `${nb} positions ordonnées · soumis le ${T.dateHeure(new Date(p.soumis_ts))}`),
      h('p', { class: 'note' },
        `Baseline figée le ${T.dateHeure(new Date(p.baseline_ts))} — les coefficients ne bougeront plus.`),
      standardOuverte && p.fenetre === 'standard'
        ? h('a', { class: 'btn btn--fin', href: `#/pronostic/${gp.id}` }, 'Modifier')
        : h('p', { class: 'note' }, 'Fenêtre fermée : le pronostic est verrouillé.')));
  }

  if (!hardcoreOuverte && !standardOuverte) {
    return carte('Ton pronostic', etatVide('Fenêtres fermées',
      'Les deux fenêtres de soumission sont passées pour ce Grand Prix.'));
  }

  const fenetreActive = hardcoreOuverte ? f.hardcore : f.standard;
  const reste = T.rebours(new Date(fenetreActive.ferme_utc), maintenant);

  return h('section', { class: 'carte' },
    h('div', { class: 'carte-corps' },
      h('div', { class: 'bandeau bandeau--alerte', style: { border: 0, padding: 0, background: 'transparent' } },
        h('span', { 'aria-hidden': 'true', style: { fontSize: '1.2rem' } }, '⚠'),
        h('div', { class: 'bandeau-txt' },
          h('strong', {}, 'Ton pronostic n\'est pas soumis'),
          h('div', { class: 'pt-s txt-2' },
            `Fenêtre ${fenetreActive.label.toLowerCase()} `,
            hardcoreOuverte ? h('b', {}, '×1,5 ') : null,
            `ferme dans ${reste.texte}`),
          h('div', { class: 'note', style: { marginTop: 'var(--esp-2)' } },
            `Fermeture : ${T.dateHeure(new Date(fenetreActive.ferme_utc))} (${T.nomFuseau(new Date(fenetreActive.ferme_utc))})`))),
      h('a', { class: 'btn btn--primaire btn--plein', href: `#/pronostic/${gp.id}`, style: { marginTop: 'var(--esp-4)' } },
        'Pronostiquer')));
}

/**
 * Dernières infos. Les documents FIA et la direction de course passent AVANT
 * la presse : c'est l'information actionnable pour un pronostic.
 */
async function chargerInfos(emplacement, ctx) {
  const corps = emplacement.querySelector('.carte-corps');

  if (ctx.mode_demo) {
    vider(corps).append(listeInfos(demo.raceControlDemo(), Date.now()), noteRSS());
    return;
  }
  if (ctx.saison < of1.COUVERTURE_DEPUIS) {
    vider(corps).append(etatVide('Direction de course indisponible',
      `OpenF1 ne couvre que 2023 → aujourd'hui. Pour ${ctx.saison}, il faut passer par le batch FastF1.`));
    return;
  }

  try {
    const session = await of1.trouverSession(
      ctx.saison, ctx.circuit?.ville || ctx.gp.ville, ctx.en_week_end ? 'R' : 'R');
    if (!session) {
      vider(corps).append(etatVide('Rien à signaler',
        'Aucune session OpenF1 n\'est encore ouverte pour ce Grand Prix. Les messages de direction de course apparaîtront dès les EL1.'),
      noteRSS());
      return;
    }
    const r = await of1.raceControl(session.session_key);
    if (!r.valeur.length) {
      vider(corps).append(etatVide('Rien à signaler', 'Aucun message de direction de course sur cette session.'), noteRSS());
      return;
    }
    vider(corps).append(listeInfos(r.valeur.slice(0, 8), r.ts), noteRSS());
  } catch (e) {
    vider(corps).append(erreurSource('OF1', e.message, () => chargerInfos(emplacement, ctx)));
  }
}

function listeInfos(messages, ts) {
  const prioritaire = (m) => /PÉNALITÉ|PENALTY|GRID|GROUPE PROPULSEUR|POWER UNIT|STEWARDS|INVESTIGAT|DISQUALIF/i.test(m.message || '');
  const triees = [...messages].sort((a, b) => (prioritaire(b) ? 1 : 0) - (prioritaire(a) ? 1 : 0));

  return h('div', {},
    triees.map((m) => h('div', { class: 'info-item' },
      h('div', {
        class: `info-marque ${prioritaire(m) ? 'info-marque--fia' : m.drapeau ? 'info-marque--drapeau' : ''}`,
      }),
      h('div', { class: 'info-txt' },
        h('div', {}, m.message || '—'),
        h('div', { class: 'info-h' },
          `${T.dateHeure(new Date(m.date_utc))} · ${T.nomFuseau(new Date(m.date_utc))}`,
          m.numero_pilote ? ` · voiture #${m.numero_pilote}` : '',
          m.tour ? ` · tour ${m.tour}` : '')))),
    h('div', { class: 'rangee', style: { marginTop: 'var(--esp-3)' } }, badgeFraicheur(T.depuis(ts))));
}

function noteRSS() {
  return h('p', { class: 'note note--encadre' },
    'Agrégation de presse (story 1.9) non branchée : un flux RSS ne peut pas être lu '
    + 'depuis un navigateur sans relais côté serveur. Les messages de direction de course, eux, '
    + 'passent bien en direct.');
}

function blocDebriefPrecedent(ctx) {
  const prec = ctx.gp_precedent;
  return carte('Grand Prix précédent', h('div', {},
    h('div', { class: 'rangee' },
      h('b', {}, prec.nom),
      h('span', { class: 'txt-3 pt-s' }, T.dateLongue(new Date(prec.course_utc)))),
    h('div', { class: 'rangee', style: { marginTop: 'var(--esp-3)' } },
      h('a', { class: 'btn btn--fin', href: `#/resultats/${prec.saison}/${prec.round}` }, 'Résultats'),
      store.pronostic(prec.id)?.soumis
        ? h('a', { class: 'btn btn--fin', href: `#/debrief/${prec.id}` }, 'Ton débrief')
        : null)));
}

function blocResultats(ctx) {
  const emplacement = carte('Dernier résultat', chargement(4),
    { source: ctx.mode_demo ? 'DÉMO' : 'JOL' });
  const corps = emplacement.querySelector('.carte-corps');
  const prec = ctx.gp_precedent;

  if (!prec) {
    vider(corps).append(etatVide('Saison pas encore commencée',
      'Le premier Grand Prix n\'a pas encore été couru.'));
    return emplacement;
  }
  if (ctx.mode_demo) {
    vider(corps).append(tableauPodium(demo.resultatsDemo(prec.round), prec, Date.now()));
    return emplacement;
  }

  jolpica.resultatsCourse(prec.saison, prec.round)
    .then((r) => {
      if (!r.valeur) {
        vider(corps).append(etatVide('Résultats non publiés',
          'Jolpica n\'a pas encore publié le classement de cette course.'));
        return;
      }
      vider(corps).append(tableauPodium(r.valeur, prec, r.ts));
    })
    .catch((e) => vider(corps).append(erreurSource('JOL', e.message)));

  return emplacement;
}

function tableauPodium(res, gp, ts) {
  return h('div', {},
    h('div', { class: 'txt-3 pt-s', style: { marginBottom: 'var(--esp-3)' } }, gp.nom),
    h('div', { class: 'tableau-enrobage' },
      h('table', { class: 't' },
        h('tbody', {},
          res.lignes.slice(0, 5).map((l) => h('tr', {},
            h('td', { class: 'pos' }, l.position_texte),
            h('td', {}, cellulePilote(l)),
            h('td', { class: 'num' }, `${l.points} pt${l.points > 1 ? 's' : ''}`)))))),
    h('div', { class: 'rangee', style: { marginTop: 'var(--esp-3)' } },
      h('a', { class: 'btn btn--fin', href: `#/resultats/${gp.saison}/${gp.round}` }, 'Classement complet'),
      h('span', { class: 'espace' }),
      badgeFraicheur(T.depuis(ts))));
}

export function nettoyer() { clearInterval(minuteur); }
