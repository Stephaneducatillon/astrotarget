/**
 * Écran 5 — Saisie du pronostic.
 *
 * Le coefficient affiché en temps réel est le cœur de l'expérience : il
 * transforme la saisie en réflexion stratégique. L'utilisateur voit
 * immédiatement que son pronostic sûr rapporte peu.
 */

import { h, vider, erreurSource, etatVide, annonce, badgeSource } from '../util/dom.js';
import { nombre, nomPiloteLong, pourcent } from '../util/format.js';
import * as T from '../util/time.js';
import { CONFIG_PRONOSTIC, COMPOSES } from '../config.js';
import { contexte, engagesEtHistorique } from '../contexte.js';
import { construireBaseline, probaPosition } from '../model/baseline.js';
import { coefficient, fenetres, fenetreOuverte, gainMaximum, diagnosticPlafond } from '../model/scoring.js';
import * as store from '../store.js';
import { carte, chargement, pastilleEcurie, bandeauDemo } from './composants.js';

const { taille_grille, coef_max, mult_hardcore } = CONFIG_PRONOSTIC;

export async function rendre(racine, params) {
  vider(racine);
  racine.append(chargement(6));

  const ctx = await contexte();
  const gp = ctx.calendrier?.find((g) => g.id === params?.id) || ctx.gp;
  if (!gp) {
    vider(racine).append(etatVide('Grand Prix introuvable', 'Le calendrier ne contient pas ce Grand Prix.'));
    return;
  }

  const profil = store.profil();
  if (!profil) {
    vider(racine).append(blocProfil(() => rendre(racine, params)));
    return;
  }

  let engages, historique, sourceEngages;
  try {
    const r = await engagesEtHistorique(ctx.saison, gp.circuit_id, { profond: false });
    engages = r.engages; historique = r.historique; sourceEngages = r.source;
  } catch (e) {
    vider(racine).append(erreurSource('JOL',
      `Impossible de charger les engagés : ${e.message}. Sans eux, aucun coefficient ne peut être calculé.`,
      () => rendre(racine, params)));
    return;
  }

  let baseline;
  try {
    baseline = construireBaseline({
      engages, circuit: ctx.circuit, historique, grille: null, cle: gp.id,
    });
  } catch (e) {
    vider(racine).append(erreurSource('CALC', e.message));
    return;
  }

  vider(racine);
  racine.append(vueSaisie({ racine, params, ctx, gp, engages, baseline, sourceEngages }));
}

function blocProfil(apres) {
  const champ = h('input', { class: 'champ', type: 'text', maxlength: '24', placeholder: 'Ton pseudo', id: 'pseudo' });
  return carte('Avant de pronostiquer', h('div', {},
    h('p', { class: 'txt-2 pt-s' },
      'Un pseudo suffit. Il reste dans ce navigateur : ce site n\'a pas de serveur de comptes, '
      + 'et n\'envoie rien nulle part.'),
    h('label', { class: 'etiquette', for: 'pseudo' }, 'Pseudo'),
    champ,
    h('button', {
      class: 'btn btn--primaire', style: { marginTop: 'var(--esp-3)' },
      onclick: () => {
        try { store.definirProfil(champ.value); apres(); }
        catch (e) { champ.setAttribute('aria-invalid', 'true'); annonce(e.message); }
      },
    }, 'Continuer')));
}

function vueSaisie({ racine, params, ctx, gp, engages, baseline, sourceEngages }) {
  const maintenant = new Date();
  const f = fenetres(gp);
  const hardcoreOuverte = fenetreOuverte(f.hardcore, maintenant);
  const standardOuverte = fenetreOuverte(f.standard, maintenant);
  const poleOuverte = fenetreOuverte(f.pole, maintenant);

  const existant = store.pronostic(gp.id);
  const dejaSoumis = Boolean(existant?.soumis);

  const pile = h('div', { class: 'pile' });
  if (ctx.mode_demo) pile.append(bandeauDemo());

  if (!hardcoreOuverte && !standardOuverte) {
    pile.append(etatVide('Fenêtres fermées',
      `La fenêtre standard fermait ${T.dateHeure(new Date(f.standard.ferme_utc))}. `
      + 'Un pronostic n\'est plus modifiable après la fermeture de sa fenêtre.'));
    if (dejaSoumis) {
      pile.append(h('a', { class: 'btn btn--primaire', href: `#/debrief/${gp.id}` }, 'Voir le débrief'));
    }
    return pile;
  }

  // Un joueur ne peut soumettre que dans UNE seule fenêtre par week-end.
  const fenetreChoisie = existant?.fenetre
    || (hardcoreOuverte ? 'hardcore' : 'standard');

  const etat = {
    fenetre: hardcoreOuverte ? fenetreChoisie : 'standard',
    grille: [...(existant?.grille || [])],
    pole: existant?.pole || null,
    meilleur_tour: existant?.meilleur_tour || null,
    premier_abandon: existant?.premier_abandon || null,
    safety_car: typeof existant?.safety_car === 'boolean' ? existant.safety_car : null,
    nb_abandons: Number.isFinite(existant?.nb_abandons) ? existant.nb_abandons : 3,
    ecart_pole_p2: existant?.ecart_pole_p2 || null,
    duel: existant?.duel || null,
    joker: existant?.joker || null,
  };
  while (etat.grille.length < taille_grille) etat.grille.push(null);

  const pilotesTries = [...baseline.pilotes].sort((a, b) => {
    const ea = esperance(baseline, a.driver_id), eb = esperance(baseline, b.driver_id);
    return ea - eb;
  });

  const zoneGrille = h('div');
  const zoneEvenements = h('div');
  const zonePied = h('div', { class: 'carte-pied' });

  const rafraichir = () => {
    dessinerGrille(zoneGrille, etat, pilotesTries, baseline, rafraichir);
    dessinerEvenements(zoneEvenements, etat, pilotesTries, baseline, poleOuverte, rafraichir, ctx);
    dessinerPied(zonePied, etat, gp, ctx, baseline, racine, params);
    annonce(`Gain potentiel maximum : ${nombre(gainPotentiel(etat, baseline, gp), 0)} points`);
  };

  pile.append(enTeteFenetre(f, etat, hardcoreOuverte, standardOuverte, maintenant, (cle) => {
    etat.fenetre = cle; rafraichir();
  }));

  pile.append(h('section', { class: 'carte' },
    h('header', { class: 'carte-tete' },
      h('h2', { class: 'carte-titre' }, `Top ${taille_grille} — ordonne les pilotes`),
      h('span', { class: 'espace' }),
      badgeSource(sourceEngages === 'DEMO' ? 'DÉMO' : 'JOL')),
    h('div', { class: 'carte-corps' }, zoneGrille)));

  pile.append(h('section', { class: 'carte' },
    h('header', { class: 'carte-tete' }, h('h2', { class: 'carte-titre' }, 'Qualifications et course')),
    h('div', { class: 'carte-corps' }, zoneEvenements)));

  pile.append(h('section', { class: 'carte' }, zonePied));
  pile.append(notesModele(baseline));

  rafraichir();
  return pile;
}

function esperance(baseline, driverId) {
  const p = baseline.probas[driverId] || [];
  return p.reduce((s, prob, i) => s + prob * (i + 1), 0) || 99;
}

function enTeteFenetre(f, etat, hardcoreOuverte, standardOuverte, maintenant, onChange) {
  const active = etat.fenetre === 'hardcore' ? f.hardcore : f.standard;
  const reste = T.rebours(new Date(active.ferme_utc), maintenant);

  return h('section', { class: 'heros' },
    h('div', { class: 'heros-sur' },
      `Fenêtre ${active.label.toLowerCase()}`,
      etat.fenetre === 'hardcore' ? ` · ×${String(mult_hardcore).replace('.', ',')}` : ''),
    h('div', { class: 'rebours' },
      h('span', { class: 'rebours-val mono' }, reste.texte),
      h('span', { class: 'rebours-lib' }, 'avant fermeture')),
    h('p', { class: 'txt-2 pt-s', style: { marginTop: 'var(--esp-3)' } }, active.description),

    hardcoreOuverte && standardOuverte
      ? h('div', { style: { marginTop: 'var(--esp-3)' } },
        h('div', { class: 'segments' },
          h('button', {
            'aria-pressed': String(etat.fenetre === 'hardcore'),
            onclick: () => onChange('hardcore'),
          }, `Hardcore ×${String(mult_hardcore).replace('.', ',')}`),
          h('button', {
            'aria-pressed': String(etat.fenetre === 'standard'),
            onclick: () => onChange('standard'),
          }, 'Standard ×1')),
        h('p', { class: 'note', style: { marginTop: 'var(--esp-2)' } },
          'Une seule fenêtre par week-end. Viser le ×1,5, c\'est s\'engager avant de savoir.'))
      : null,

    h('p', { class: 'note', style: { marginTop: 'var(--esp-3)' } },
      `Fermeture : ${T.dateHeure(new Date(active.ferme_utc))} (${T.nomFuseau(new Date(active.ferme_utc))}). `
      + 'En production, cette heure est calculée par le serveur depuis `session.debut_utc`, jamais par '
      + 'l\'horloge du navigateur — ici, faute de back-end, elle l\'est localement.'));
}

// ── Grille ──────────────────────────────────────────────────────────────────

function dessinerGrille(zone, etat, pilotes, baseline, rafraichir) {
  vider(zone);

  etat.grille.forEach((driverId, i) => {
    const position = i + 1;
    const p = driverId ? probaPosition(baseline, driverId, position) : null;
    const coef = p === null ? null : coefficient(p);
    const cleJoker = `grille:${position}`;
    const jokerIci = etat.joker?.ligne === cleJoker;

    const select = h('select', {
      class: 'champ', 'aria-label': `Pilote en position ${position}`,
      onchange: (e) => {
        const val = e.target.value || null;
        // Un pilote ne peut occuper deux positions : on échange.
        const autre = etat.grille.indexOf(val);
        if (val && autre >= 0 && autre !== i) etat.grille[autre] = driverId;
        etat.grille[i] = val;
        rafraichir();
      },
    },
    h('option', { value: '' }, '— choisir —'),
    pilotes.map((pl) => h('option', {
      value: pl.driver_id, selected: pl.driver_id === driverId || null,
    }, `${nomPiloteLong(pl)}${pl.team_nom ? ` · ${pl.team_nom}` : ''}`)));

    zone.append(h('div', {
      class: `rang ${jokerIci ? 'joker-actif' : ''}`,
      draggable: 'true',
      ondragstart: (e) => { e.dataTransfer.setData('text/plain', String(i)); e.currentTarget.classList.add('glisse'); },
      ondragend: (e) => e.currentTarget.classList.remove('glisse'),
      ondragover: (e) => { e.preventDefault(); e.currentTarget.classList.add('survol'); },
      ondragleave: (e) => e.currentTarget.classList.remove('survol'),
      ondrop: (e) => {
        e.preventDefault();
        e.currentTarget.classList.remove('survol');
        const de = Number(e.dataTransfer.getData('text/plain'));
        if (Number.isFinite(de) && de !== i) {
          const [x] = etat.grille.splice(de, 1);
          etat.grille.splice(i, 0, x);
          rafraichir();
        }
      },
    },
    h('span', { class: 'poignee', 'aria-hidden': 'true' }, '≡'),
    h('span', { class: 'rang-num' }, position),
    driverId ? pastilleEcurie(pilotes.find((x) => x.driver_id === driverId)?.team_id) : null,
    select,
    coef === null
      ? null
      : h('span', {
        class: `rang-coef ${coef >= 2.5 ? 'rang-coef--tres-fort' : coef >= 1.6 ? 'rang-coef--fort' : ''}`,
        title: `Le modèle donne ${pourcent(p, 1)} à ce pilote pour la position ${position}`,
      }, `×${nombre(coef, 1)}`),
    h('div', { class: 'rang-outils' },
      h('button', {
        title: 'Monter', 'aria-label': `Monter la position ${position}`, disabled: i === 0 || null,
        onclick: () => { echanger(etat.grille, i, i - 1); rafraichir(); },
      }, '↑'),
      h('button', {
        title: 'Descendre', 'aria-label': `Descendre la position ${position}`,
        disabled: i === etat.grille.length - 1 || null,
        onclick: () => { echanger(etat.grille, i, i + 1); rafraichir(); },
      }, '↓'),
      h('button', {
        title: jokerIci ? 'Retirer le joker' : 'Joker conviction sur cette ligne',
        'aria-label': `Joker conviction sur la position ${position}`,
        'aria-pressed': String(jokerIci),
        onclick: () => { basculerJoker(etat, cleJoker); rafraichir(); },
      }, '★'))));
  });

  const remplies = etat.grille.filter(Boolean).length;
  zone.append(h('p', { class: 'note', style: { marginTop: 'var(--esp-3)' } },
    `${remplies} / ${taille_grille} positions renseignées. `
    + 'Le coefficient est calculé position par position, sur la probabilité que ce pilote '
    + 'finisse à cette place précise. Un favori en P1 rapporte peu ; un outsider en P3, beaucoup. '
    + `Le plafond est fixé à ×${String(coef_max).replace('.', ',')}.`));
}

function echanger(t, i, j) { const x = t[i]; t[i] = t[j]; t[j] = x; }

function basculerJoker(etat, cle) {
  if (etat.joker?.ligne === cle) {
    etat.joker = etat.joker.multiplicateur === 2 && store.jokersTriplesRestants(etat.saison) > 0
      ? { ligne: cle, multiplicateur: 3 }
      : null;
  } else {
    etat.joker = { ligne: cle, multiplicateur: 2 };
  }
}

// ── Événements ──────────────────────────────────────────────────────────────

function dessinerEvenements(zone, etat, pilotes, baseline, poleOuverte, rafraichir, ctx) {
  vider(zone);

  const ligne = (cle, libelle, controle, coefValeur, aide) => {
    const jokerIci = etat.joker?.ligne === cle;
    return h('div', { class: `ligne-pro ${jokerIci ? 'joker-actif' : ''}` },
      h('div', { class: 'ligne-pro-lib' }, libelle,
        aide ? h('div', { class: 'note' }, aide) : null),
      h('div', { class: 'ligne-pro-ctrl' }, controle),
      // Pas de coefficient tant que rien n'est choisi : un « — » isolé se lit
      // comme une valeur, alors que c'est juste une case vide.
      coefValeur
        ? h('span', {
          class: `rang-coef ${coefValeur >= 2.5 ? 'rang-coef--tres-fort' : coefValeur >= 1.6 ? 'rang-coef--fort' : ''}`,
          title: 'Coefficient de difficulté appliqué à cette ligne',
        }, `×${nombre(coefValeur, 1)}`)
        : null,
      h('button', {
        class: 'btn-joker', title: 'Joker conviction', 'aria-label': `Joker sur ${libelle}`,
        'aria-pressed': String(jokerIci),
        onclick: () => { basculerJoker(etat, cle); rafraichir(); },
      }, '★'));
  };

  // Pole — ferme au début de Q1, quelle que soit la fenêtre choisie.
  const pPole = etat.pole ? (baseline.parPilote[etat.pole]?.pole ?? 0.05) : null;
  zone.append(ligne('pole', 'Poleman',
    poleOuverte
      ? selectPilote(pilotes, etat.pole, (v) => { etat.pole = v; rafraichir(); })
      : h('span', { class: 'txt-3 pt-s' }, 'fermé — Q1 a commencé'),
    pPole ? coefficient(pPole) : null,
    'Ferme au début de Q1, quelle que soit ta fenêtre'));

  // Écart pole ↔ P2.
  const ecarts = baseline.evenements.ecart_pole_p2;
  zone.append(ligne('ecart_pole_p2', 'Écart pole → P2',
    h('select', {
      class: 'champ', 'aria-label': 'Écart pole P2',
      onchange: (e) => { etat.ecart_pole_p2 = e.target.value || null; rafraichir(); },
    },
    h('option', { value: '' }, '— choisir —'),
    Object.keys(ecarts).map((k) => h('option', { value: k, selected: etat.ecart_pole_p2 === k || null }, k))),
    etat.ecart_pole_p2 ? coefficient(ecarts[etat.ecart_pole_p2]) : null));

  // Meilleur tour.
  const pMt = etat.meilleur_tour ? (baseline.parPilote[etat.meilleur_tour]?.meilleur_tour ?? 0.08) : null;
  zone.append(ligne('meilleur_tour', 'Meilleur tour en course',
    selectPilote(pilotes, etat.meilleur_tour, (v) => { etat.meilleur_tour = v; rafraichir(); }),
    pMt ? coefficient(pMt) : null));

  // Safety car — le coefficient vient du taux historique du circuit.
  const pOui = baseline.evenements.safety_car_oui;
  const pSc = etat.safety_car === null ? null : (etat.safety_car ? pOui : 1 - pOui);
  zone.append(ligne('safety_car', 'Safety car',
    h('div', { class: 'segments' },
      h('button', {
        'aria-pressed': String(etat.safety_car === true),
        onclick: () => { etat.safety_car = etat.safety_car === true ? null : true; rafraichir(); },
      }, 'Oui'),
      h('button', {
        'aria-pressed': String(etat.safety_car === false),
        onclick: () => { etat.safety_car = etat.safety_car === false ? null : false; rafraichir(); },
      }, 'Non')),
    pSc ? coefficient(pSc) : null,
    `${ctx.circuit?.nom || 'Ce circuit'} : ${pourcent(pOui, 0)} des courses avec au moins une SC`));

  // Nombre d'abandons.
  const distrib = baseline.evenements.abandons;
  zone.append(ligne('nb_abandons', 'Nombre d\'abandons',
    h('div', { class: 'compteur' },
      h('button', {
        'aria-label': 'Moins un abandon',
        onclick: () => { etat.nb_abandons = Math.max(0, etat.nb_abandons - 1); rafraichir(); },
      }, '−'),
      h('output', {}, etat.nb_abandons),
      h('button', {
        'aria-label': 'Plus un abandon',
        onclick: () => { etat.nb_abandons = Math.min(20, etat.nb_abandons + 1); rafraichir(); },
      }, '+')),
    coefficient(distrib[etat.nb_abandons] ?? 0.02),
    'Exact : 6 points de base. À ±1 : 3 points.'));

  // Premier abandon.
  const pPa = etat.premier_abandon ? (baseline.parPilote[etat.premier_abandon]?.premier_abandon ?? 0.05) : null;
  zone.append(ligne('premier_abandon', 'Premier abandon',
    selectPilote(pilotes, etat.premier_abandon, (v) => { etat.premier_abandon = v; rafraichir(); }, true),
    pPa ? coefficient(pPa) : null,
    'Très difficile : le coefficient touche souvent le plafond'));

  // Duel d'équipiers imposé.
  const duel = baseline.duels[0];
  if (duel) {
    const pDuel = etat.duel
      ? (duel.a.driver_id === etat.duel ? duel.a.proba : duel.b.proba)
      : null;
    zone.append(ligne('duel', `Duel imposé — ${duel.team_nom}`,
      h('div', { class: 'segments' },
        [duel.a, duel.b].map((c) => h('button', {
          'aria-pressed': String(etat.duel === c.driver_id),
          onclick: () => { etat.duel = etat.duel === c.driver_id ? null : c.driver_id; rafraichir(); },
        }, c.nom))),
      pDuel ? coefficient(pDuel) : null,
      'Un duel imposé par week-end, choisi comme le plus serré du plateau'));
  }

  if (etat.joker) {
    const restants = store.jokersTriplesRestants(baseline.saison || new Date().getUTCFullYear());
    zone.append(h('div', { class: 'bandeau bandeau--alerte', style: { marginTop: 'var(--esp-4)' } },
      h('span', { 'aria-hidden': 'true' }, '★'),
      h('div', { class: 'bandeau-txt' },
        h('strong', {}, `Joker conviction ×${etat.joker.multiplicateur} sur « ${etat.joker.ligne} »`),
        h('div', { class: 'pt-s txt-2' },
          'Doublé en gain comme en absence de gain : zéro reste zéro, il n\'y a pas de perte sèche.'),
        h('div', { class: 'note' },
          etat.joker.multiplicateur === 2
            ? `Reclique sur ★ pour passer en ×3 (${restants} joker${restants > 1 ? 's' : ''} triple restant${restants > 1 ? 's' : ''} cette saison).`
            : 'Joker triple : non rechargeable, 3 par saison.'))));
  }
}

function selectPilote(pilotes, valeur, onChange, inverse = false) {
  const liste = inverse ? [...pilotes].reverse() : pilotes;
  return h('select', {
    class: 'champ', onchange: (e) => onChange(e.target.value || null),
  },
  h('option', { value: '' }, '— choisir —'),
  liste.map((p) => h('option', { value: p.driver_id, selected: p.driver_id === valeur || null },
    `${nomPiloteLong(p)}${p.team_nom ? ` · ${p.team_nom}` : ''}`)));
}

// ── Pied : gain maximum et soumission ───────────────────────────────────────

function gainPotentiel(etat, baseline, gp) {
  return gainMaximum({
    ...etat, gp_id: gp.id, baseline_gel: baseline,
  });
}

function dessinerPied(zone, etat, gp, ctx, baseline, racine, params) {
  vider(zone);
  const gain = gainPotentiel(etat, baseline, gp);
  const complet = etat.grille.filter(Boolean).length === taille_grille;

  zone.append(
    h('div', {},
      h('div', { class: 'carte-titre' }, 'Gain potentiel maximum'),
      h('div', { class: 'score-val', style: { fontSize: '1.9rem' } }, `${nombre(gain, 0)} pts`),
      h('div', { class: 'note' },
        etat.fenetre === 'hardcore'
          ? `multiplicateur de fenêtre ×${String(mult_hardcore).replace('.', ',')} inclus`
          : 'sans multiplicateur de fenêtre')),
    h('span', { class: 'espace' }),
    h('button', {
      class: 'btn', onclick: () => {
        store.enregistrerPronostic(gp.id, { ...etat, saison: ctx.saison });
        annonce('Brouillon enregistré.');
      },
    }, 'Enregistrer le brouillon'),
    h('button', {
      class: 'btn btn--primaire', disabled: !complet || null,
      title: complet ? '' : `Complète les ${taille_grille} positions du bloc A`,
      onclick: () => {
        store.soumettrePronostic(gp.id, { ...etat, saison: ctx.saison, gp_id: gp.id }, baseline);
        location.hash = `#/debrief/${gp.id}`;
      },
    }, 'Soumettre'));

  if (!complet) {
    zone.append(h('p', { class: 'note', style: { flexBasis: '100%' } },
      'Le bloc A (grille d\'arrivée) est obligatoire : les dix positions doivent être renseignées.'));
  }
}

function notesModele(baseline) {
  return carte('D\'où viennent les coefficients', h('div', {},
    h('p', { class: 'txt-2 pt-s' },
      'Chaque coefficient vaut ', h('code', {}, '1 / probabilité'), ', borné entre ×0,5 et ×4,0. '
      + 'La probabilité vient du modèle baseline, figé et horodaté au moment où tu soumets : '
      + 'il n\'est jamais recalculé après la course, sinon les coefficients seraient '
      + 'rétroactivement faussés.'),
    h('p', { class: 'note note--encadre' }, baseline.meta.methode, '.'),
    h('p', { class: 'note note--encadre' }, baseline.meta.avertissement),
    blocPlafond(baseline),
    h('div', { class: 'rangee', style: { marginTop: 'var(--esp-3)' } },
      badgeSource('CALC'),
      h('span', { class: 'note' }, `${nombre(baseline.meta.tirages)} tirages · construit le ${T.dateHeure(new Date(baseline.ts))}`))));
}

/**
 * Le plafond de coefficient, montré avec les chiffres du week-end.
 *
 * Le livrable 2 §9 pose la question « le plafond à 4,0 : trop généreux, ou pas
 * assez ? ». Sur le bloc A, la réponse est visible ici même : la probabilité
 * qu'un pilote finisse à une place précise dans un peloton de vingt dépasse
 * rarement 25 %, et tout ce qui est sous 25 % touche le plafond. L'écart entre
 * un favori et un fond de grille disparaît alors — l'inverse de ce que le
 * barème cherchait.
 */
function blocPlafond(baseline) {
  const d = diagnosticPlafond(baseline, taille_grille);
  if (!d) return null;
  const p1 = d.par_position[0];
  const critique = d.part >= 0.75;

  return h('div', {
    class: `bandeau ${critique ? 'bandeau--alerte' : 'bandeau--info'}`,
    style: { marginTop: 'var(--esp-4)' },
  },
  h('span', { 'aria-hidden': 'true' }, critique ? '⚠' : 'ⓘ'),
  h('div', { class: 'bandeau-txt' },
    h('strong', {}, `Plafond de coefficient : ${p1.au_plafond} pilotes sur ${p1.sur} le touchent en P1`),
    h('div', { class: 'pt-s txt-2' },
      `Sur l'ensemble du bloc A, ${nombre(d.part * 100, 0)} % des couples pilote × position sont `
      + `au plafond ×${String(coef_max).replace('.', ',')}. `
      + `Le meilleur candidat à la victoire n'est lui-même qu'à ${pourcent(d.meilleure_proba_p1, 1)} : `
      + `tout ce qui est sous ${pourcent(d.seuil, 0)} sature.`),
    h('div', { class: 'note', style: { marginTop: 'var(--esp-2)' } },
      'Conséquence : sur la grille, pronostiquer le favori rapporte autant qu\'un fond de grille — '
      + 'exactement ce que le barème voulait éviter. La formule fonctionne bien sur les événements '
      + 'à peu d\'issues (safety car, pole), pas sur une place exacte parmi vingt. '
      + 'Trois leviers, tous à arbitrer : relever le plafond, normaliser le coefficient du bloc A '
      + 'par la meilleure probabilité de la place, ou réduire la grille à un top 5 '
      + '(livrable 2 §9, deux cases déjà ouvertes).')));
}
