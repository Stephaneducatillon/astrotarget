// Parsers : ils doivent extraire ce qui compte et, surtout, ne jamais lever
// sur une reponse inattendue. Une exception dans un parser = une source muette.

import test from 'node:test';
import assert from 'node:assert/strict';
import { parseVigilance } from '../src/sources/vigilance.js';
import { parseEcowatt } from '../src/sources/ecowatt.js';
import { derniereMesure, evaluerNiveau } from '../src/sources/hubeau.js';
import { parseVigicrues } from '../src/sources/vigicrues.js';

// --- Vigilance Meteo-France -------------------------------------------------

const vigilanceOrange = {
  product: {
    periods: [
      {
        echeance: 'J',
        timelaps: {
          domain_ids: [
            {
              domain_id: '59',
              max_color_id: 3,
              phenomenon_items: [
                { phenomenon_id: '1', phenomenon_max_color_id: 3 },
                { phenomenon_id: '3', phenomenon_max_color_id: 1 },
              ],
            },
            { domain_id: '62', max_color_id: 2, phenomenon_items: [] },
            { domain_id: '75', max_color_id: 4, phenomenon_items: [] },
          ],
        },
      },
    ],
  },
};

test('vigilance : orange sur le 59 remonte avec le phenomene', () => {
  const r = parseVigilance(vigilanceOrange);
  assert.equal(r.pire, 'orange');
  assert.equal(r.departements['59'].couleur, 'orange');
  assert.match(r.libelle, /Vent violent \(59\)/);
});

test('vigilance : un departement hors perimetre est ignore', () => {
  // Le 75 est rouge dans la reponse : il ne doit pas remonter.
  const r = parseVigilance(vigilanceOrange);
  assert.equal(r.departements['75'], undefined);
  assert.equal(r.pire, 'orange');
});

test('vigilance : le jaune ne declenche aucun niveau familial', () => {
  const jaune = {
    product: { periods: [{ timelaps: { domain_ids: [
      { domain_id: '59', max_color_id: 2, phenomenon_items: [{ phenomenon_id: '3', phenomenon_max_color_id: 2 }] },
    ] } }] },
  };
  const r = parseVigilance(jaune);
  assert.equal(r.pire, 'vert');
  assert.equal(r.libelle, null);
});

test('vigilance : les avalanches ne concernent pas le Douaisis', () => {
  const avalanche = {
    product: { periods: [{ timelaps: { domain_ids: [
      { domain_id: '59', max_color_id: 1, phenomenon_items: [{ phenomenon_id: '8', phenomenon_max_color_id: 4 }] },
    ] } }] },
  };
  assert.equal(parseVigilance(avalanche).pire, 'vert');
});

test('vigilance : une reponse vide ou inattendue ne leve pas', () => {
  for (const entree of [{}, null, { product: null }, { foo: [1, 2, 3] }, []]) {
    const r = parseVigilance(entree);
    assert.equal(r.pire, 'vert');
  }
});

test('vigilance : la structure est trouvee meme si le chemin change', () => {
  // Meme donnee, imbrication differente : le parseur explore au lieu de
  // supposer product.periods[].timelaps.
  const autre = { data: { autre_niveau: [{ domain_id: '59', max_color_id: 4, phenomenon_items: [
    { phenomenon_id: '2', phenomenon_max_color_id: 4 },
  ] }] } };
  const r = parseVigilance(autre);
  assert.equal(r.pire, 'rouge');
});

// --- Ecowatt ----------------------------------------------------------------

test('ecowatt : signal orange avec creneau horaire', () => {
  const r = parseEcowatt({
    signals: [
      { jour: '2026-01-14', dvalue: 1, values: [] },
      { jour: '2026-01-15', dvalue: 2, values: [
        { pas: 8, hvalue: 1 }, { pas: 18, hvalue: 2 }, { pas: 19, hvalue: 2 },
      ] },
    ],
  });
  assert.equal(r.pire, 'orange');
  assert.match(r.libelle, /2026-01-15/);
  assert.match(r.libelle, /18h-20h/);
});

test('ecowatt : tout vert ne produit pas de libelle', () => {
  const r = parseEcowatt({ signals: [{ jour: '2026-01-14', dvalue: 1, values: [] }] });
  assert.equal(r.pire, 'vert');
  assert.equal(r.libelle, null);
});

test('ecowatt : ne retient que J a J+3', () => {
  const signals = Array.from({ length: 7 }, (_, i) => ({ jour: `j${i}`, dvalue: 1, values: [] }));
  assert.equal(parseEcowatt({ signals }).jours.length, 4);
});

test('ecowatt : reponse vide ne leve pas', () => {
  for (const e of [{}, null, { signals: null }]) {
    assert.equal(parseEcowatt(e).pire, 'vert');
  }
});

// --- Hub'Eau ----------------------------------------------------------------

test('hubeau : retient la mesure la plus recente', () => {
  const r = derniereMesure({ data: [
    { date_obs: '2026-01-14T10:00:00Z', resultat_obs: 1200 },
    { date_obs: '2026-01-14T10:30:00Z', resultat_obs: 1250 },
    { date_obs: '2026-01-14T09:30:00Z', resultat_obs: 1180 },
  ] });
  assert.equal(r.value, 1250);
});

test('hubeau : ignore les points illisibles au lieu de planter', () => {
  const r = derniereMesure({ data: [
    { date_obs: 'pas-une-date', resultat_obs: 9999 },
    { date_obs: '2026-01-14T10:00:00Z', resultat_obs: null },
    { date_obs: '2026-01-14T09:00:00Z', resultat_obs: 1100 },
  ] });
  assert.equal(r.value, 1100);
});

test('hubeau : lot vide retourne null', () => {
  assert.equal(derniereMesure({ data: [] }), null);
  assert.equal(derniereMesure({}), null);
});

test('hubeau : depassement du maximum de la fenetre declenche orange', () => {
  const r = evaluerNiveau(1500, { max: 1400, n: 2000 });
  assert.equal(r.level, 'orange');
  assert.equal(r.depassement, true);
});

test('hubeau : un depassement dans la marge ne declenche pas', () => {
  // 1% au-dessus : bruit de mesure, pas une crue.
  const r = evaluerNiveau(1414, { max: 1400, n: 2000 });
  assert.equal(r.level, 'vert');
});

test('hubeau : historique trop court = pas de regle, pas de fausse alerte', () => {
  // Le piege de la mise en service : 3 points en base, le maximum ne veut
  // rien dire et tout depassement serait un faux positif.
  const r = evaluerNiveau(5000, { max: 100, n: 3 });
  assert.equal(r.level, 'vert');
  assert.equal(r.insuffisant, true);
});

test('hubeau : fenetre absente ne leve pas', () => {
  assert.equal(evaluerNiveau(1500, null).level, 'vert');
  assert.equal(evaluerNiveau(1500, { max: null, n: 9999 }).level, 'vert');
});

// --- Vigicrues --------------------------------------------------------------

const surveilles = [{ code: 'XX1', nom: 'Scarpe aval' }];

test('vigicrues : ne retient que les troncons surveilles', () => {
  const r = parseVigicrues({ vicItemInfoVigiCru: [
    { vicCdEntCru: 'XX1', vicNivInfoVigiCru: 3 },
    { vicCdEntCru: 'ZZ9', vicNivInfoVigiCru: 4 },
  ] }, surveilles);
  assert.equal(r.troncons.length, 1);
  assert.equal(r.pire, 'orange');
  assert.match(r.libelle, /Scarpe aval/);
});

test('vigicrues : rouge sur le troncon surveille remonte en rouge', () => {
  const r = parseVigicrues({ vicItemInfoVigiCru: [{ vicCdEntCru: 'XX1', vicNivInfoVigiCru: 4 }] }, surveilles);
  assert.equal(r.pire, 'rouge');
});

test('vigicrues : le jaune ne declenche pas', () => {
  const r = parseVigicrues({ vicItemInfoVigiCru: [{ vicCdEntCru: 'XX1', vicNivInfoVigiCru: 2 }] }, surveilles);
  assert.equal(r.pire, 'vert');
  assert.equal(r.libelle, null);
});

test('vigicrues : reponse vide ou inattendue ne leve pas', () => {
  for (const e of [{}, null, { autre: 1 }]) {
    assert.equal(parseVigicrues(e, surveilles).pire, 'vert');
  }
});
