// Accès au catalogue GCVS embarqué (data/gcvs.js) et classification des types de variabilité.
(function () {
  'use strict';

  const CONSTELLATIONS = {
    And: 'Andromède', Ant: 'Machine pneumatique', Aps: 'Oiseau de paradis', Aql: 'Aigle', Aqr: 'Verseau', Ara: 'Autel', Ari: 'Bélier',
    Aur: 'Cocher', Boo: 'Bouvier', Cae: 'Burin', Cam: 'Girafe', Cap: 'Capricorne', Car: 'Carène', Cas: 'Cassiopée', Cen: 'Centaure',
    Cep: 'Céphée', Cet: 'Baleine', Cha: 'Caméléon', Cir: 'Compas', CMa: 'Grand Chien', CMi: 'Petit Chien', Cnc: 'Cancer', Col: 'Colombe',
    Com: 'Chevelure de Bérénice', CrA: 'Couronne australe', CrB: 'Couronne boréale', Crt: 'Coupe', Cru: 'Croix du Sud', Crv: 'Corbeau',
    CVn: 'Chiens de chasse', Cyg: 'Cygne', Del: 'Dauphin', Dor: 'Dorade', Dra: 'Dragon', Equ: 'Petit Cheval', Eri: 'Éridan', For: 'Fourneau',
    Gem: 'Gémeaux', Gru: 'Grue', Her: 'Hercule', Hor: 'Horloge', Hya: 'Hydre', Hyi: 'Hydre mâle', Ind: 'Indien', Lac: 'Lézard', Leo: 'Lion',
    Lep: 'Lièvre', Lib: 'Balance', LMi: 'Petit Lion', Lup: 'Loup', Lyn: 'Lynx', Lyr: 'Lyre', Men: 'Table', Mic: 'Microscope', Mon: 'Licorne',
    Mus: 'Mouche', Nor: 'Règle', Oct: 'Octant', Oph: 'Ophiuchus', Ori: 'Orion', Pav: 'Paon', Peg: 'Pégase', Per: 'Persée', Phe: 'Phénix',
    Pic: 'Peintre', PsA: 'Poisson austral', Psc: 'Poissons', Pup: 'Poupe', Pyx: 'Boussole', Ret: 'Réticule', Scl: 'Sculpteur', Sco: 'Scorpion',
    Sct: 'Écu de Sobieski', Ser: 'Serpent', Sex: 'Sextant', Sge: 'Flèche', Sgr: 'Sagittaire', Tau: 'Taureau', Tel: 'Télescope',
    TrA: 'Triangle austral', Tri: 'Triangle', Tuc: 'Toucan', UMa: 'Grande Ourse', UMi: 'Petite Ourse', Vel: 'Voiles', Vir: 'Vierge',
    Vol: 'Poisson volant', Vul: 'Petit Renard',
  };

  // Familles de variabilité (ordre = priorité de reconnaissance)
  const FAMILIES = [
    { id: 'mira', label: 'Mira (M)', re: /^M$/ },
    { id: 'sr', label: 'Semi-régulières (SR)', re: /^SR/ },
    { id: 'irr', label: 'Irrégulières (L, I)', re: /^(L|LB|LC|I|IA|IB|IN|IS)/ },
    { id: 'cep', label: 'Céphéides / RV Tau', re: /^(DCEP|CEP|CW|RV|BLBOO|ACEP)/ },
    { id: 'rr', label: 'RR Lyrae', re: /^RR/ },
    { id: 'dsct', label: 'δ Scuti / SX Phe', re: /^(DSCT|SXPHE|GDOR|ROAP|BCEP|SPB)/ },
    { id: 'ecl', label: 'Binaires à éclipses', re: /^(EA|EB|EW|E$|E\/|EP|EC)/ },
    { id: 'cv', label: 'Cataclysmiques / novae', re: /^(UG|N|NA|NB|NC|NL|NR|ZAND|SN|CV|AM|XN|X)/ },
    { id: 'erupt', label: 'Éruptives (UV, RCB, GCAS…)', re: /^(UV|FU|GCAS|RCB|SDOR|WR|BE|FSCMA|INSB|DYPER|EXOR|UXOR|YSO)/ },
    { id: 'rot', label: 'Rotation (BY, RS, ELL, ACV…)', re: /^(BY|RS|ELL|ACV|SXARI|ROT|FKCOM|PSR)/ },
  ];
  const FAMILY_OTHER = { id: 'other', label: 'Autres / incertaines' };

  // « Eclipse » = l'époque GCVS correspond à un minimum, sinon à un maximum
  const EPOCH_IS_MIN = /^(E|EA|EB|EW|RS|ELL|BY)/;

  function family(type) {
    const t = (type || '').replace(/[:()]/g, '').toUpperCase();
    for (const f of FAMILIES) if (f.re.test(t)) return f.id;
    return FAMILY_OTHER.id;
  }

  let stars = null;
  function load() {
    if (stars) return stars;
    const raw = window.GCVS_CATALOG || [];
    stars = raw.map((r, i) => {
      const name = r[0];
      const con = name.split(' ').pop();
      return {
        i, name, ra: r[1], dec: r[2], type: r[3], max: r[4], min: r[5], period: r[6], epoch: r[7], spec: r[8], band: r[9],
        con: CONSTELLATIONS[con] ? con : '', fam: family(r[3]), epochIsMin: EPOCH_IS_MIN.test(r[3] || ''),
        amp: r[5] != null ? +(r[5] - r[4]).toFixed(2) : null,
        key: name.toLowerCase().replace(/\s+/g, ''),
      };
    });
    return stars;
  }

  function find(name) {
    const k = (name || '').toLowerCase().replace(/\s+/g, '').replace(/^gcvs/, '');
    if (!k) return null;
    return load().find((s) => s.key === k) || null;
  }

  function familyLabel(id) { return (FAMILIES.find((f) => f.id === id) || FAMILY_OTHER).label; }

  window.Catalog = { load, find, family, familyLabel, FAMILIES: FAMILIES.concat([FAMILY_OTHER]), CONSTELLATIONS };
})();
