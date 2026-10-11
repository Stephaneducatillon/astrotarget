"""Données de référence des tests JavaScript (web/tests/fixtures).

- reference.json : positions PyEphem (satellites et planète) à des instants
  tirés au hasard, pour vérifier que les tables redonnent PyEphem ;
- wcs.fits + wcs_expected.json : petite image FITS avec solution WCS TAN et
  pixels attendus calculés par astropy ;
- exif.jpg : JPEG avec date EXIF.

    python web/tools/make_fixtures.py
"""

import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import ephem
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from satellites import data  # noqa: E402

OUT = ROOT / "web" / "tests" / "fixtures"


def reference():
    rng = np.random.default_rng(42)
    rows = []
    t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    for planet in data.PLANETS.values():
        for _ in range(60):
            t = t0 + timedelta(seconds=float(rng.uniform(0, 15 * 365.25 * 86400)))   # 2025 → 2039
            t = t.replace(microsecond=0)
            obs = ephem.Observer()
            obs.date = ephem.Date(t.replace(tzinfo=None))
            b = planet.cls()
            b.compute(obs)
            moons = {}
            for m in planet.moons:
                s = m.cls()
                s.compute(obs)
                moons[m.key] = [s.x, s.y, s.z]
            rows.append({"planet": planet.key, "iso": t.isoformat().replace("+00:00", "Z"),
                         "ra": float(b.a_ra), "dec": float(b.a_dec), "radius": b.size / 2, "moons": moons})
    (OUT / "reference.json").write_text(json.dumps(rows))


def wcs_fixture():
    from astropy.io import fits
    from astropy.wcs import WCS

    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [143.7179, 15.0503]
    w.wcs.crpix = [40.5, 30.5]
    s = 0.06 / 3600
    rot = math.radians(23.0)
    w.wcs.cd = [[-s * math.cos(rot), s * math.sin(rot)], [s * math.sin(rot), s * math.cos(rot)]]
    img = np.zeros((60, 80), dtype=np.float32)
    img[10, 20] = 1000.0   # pixel FITS (x=20, y=10)
    hdr = w.to_header()
    hdr["DATE-OBS"] = "2026-10-12T04:00:00.000"
    fits.PrimaryHDU(img, header=hdr).writeto(OUT / "wcs.fits", overwrite=True)
    pts = []
    for dra, ddec in [(0, 0), (1.2, -0.5), (-2.0, 1.1), (0.7, 0.9)]:
        ra = 143.7179 + dra / 3600 / math.cos(math.radians(15.05))
        dec = 15.0503 + ddec / 3600
        x, y = w.all_world2pix([[ra, dec]], 0)[0]
        pts.append({"ra": ra, "dec": dec, "x": float(x), "y_display": float(60 - 1 - y)})
    (OUT / "wcs_expected.json").write_text(json.dumps({"height": 60, "width": 80, "points": pts}))


def exif_fixture():
    from PIL import Image

    img = Image.new("RGB", (16, 12), (10, 10, 10))
    exif = Image.Exif()
    exif[0x0132] = "2026:10:12 06:00:00"
    exif.get_ifd(0x8769)[0x9003] = "2026:10:12 06:00:05"
    img.save(OUT / "exif.jpg", exif=exif)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    reference()
    wcs_fixture()
    exif_fixture()
    print("Données de test écrites dans", OUT)
