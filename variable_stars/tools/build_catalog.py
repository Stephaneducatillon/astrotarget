#!/usr/bin/env python3
"""Régénère www/data/gcvs.js à partir du GCVS 5.1 (version Stellarium) et du catalogue HYG.

Sources :
  - GCVS 5.1 (Samus et al., Sternberg Astronomical Institute), extrait diffusé par Stellarium :
    https://github.com/Stellarium/stellarium/blob/master/stars/hip_gaia3/gcvs.cat (gzip)
  - HYG v3.7 (positions Hipparcos) :
    https://raw.githubusercontent.com/astronexus/HYG-Database/main/hyg/v3/hyg_v37.csv.gz
Les étoiles identifiées par un numéro Gaia DR3 sont positionnées grâce à l'index HEALPix
(niveau 12) encodé dans le numéro de source, soit une précision d'environ 1 minute d'arc.

Usage : python3 tools/build_catalog.py gcvs.cat hyg_v37.csv.gz www/data/gcvs.js
"""
import csv
import gzip
import io
import json
import sys

from healpix import gaia_radec


def open_text(path):
    raw = open(path, 'rb').read()
    if raw[:2] == b'\x1f\x8b':
        raw = gzip.decompress(raw)
    return io.StringIO(raw.decode('utf-8', errors='replace'))


def num(s):
    s = s.strip().rstrip(':')
    try:
        return float(s)
    except ValueError:
        return None


def main(gcvs_path, hyg_path, out_path):
    hyg = {}
    for r in csv.DictReader(open_text(hyg_path)):
        if r['hip']:
            hyg[int(r['hip'])] = (float(r['ra']) * 15, float(r['dec']))
    out, missing = [], 0
    for line in open_text(gcvs_path):
        if line.startswith('#') or not line.strip():
            continue
        f = line.rstrip('\n').split('\t')
        sid, name = int(f[0]), ' '.join(f[1].split())
        if sid < 200000:  # numéro Hipparcos
            if sid not in hyg:
                missing += 1
                continue
            ra, dec = hyg[sid]
        else:  # numéro de source Gaia DR3
            ra, dec = gaia_radec(sid)
        mx, flag, mn = num(f[3]), int(f[4] or 0), num(f[5])
        if mx is None:
            continue
        if flag == 1 and mn is not None:  # le minimum est donné sous forme d'amplitude
            mn = round(mx + mn, 2)
        ep, per = num(f[8]), num(f[9])
        if ep is not None:
            ep = round(ep + 2400000, 4)
        out.append([name, round(ra, 4), round(dec, 4), f[2].strip(), mx, mn, per, ep, f[11].strip(), f[7].strip()])
    out.sort(key=lambda r: r[4])
    header = ('// Catalogue généré depuis GCVS 5.1 (Samus et al.) via Stellarium, coordonnées Hipparcos/HYG et Gaia DR3 (HEALPix).\n'
              '// Colonnes : nom, AD (deg), Déc (deg), type, mag max, mag min, période (j), époque (JD), spectre, système photométrique\n'
              'window.GCVS_CATALOG = ')
    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write(header + json.dumps(out, separators=(',', ':'), ensure_ascii=False) + ';\n')
    print(f'{len(out)} étoiles écrites dans {out_path} ({missing} numéros HIP sans position)')


if __name__ == '__main__':
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
