"""Comparaison avec JPL Horizons (protocole de validation, point 1).

Pour chaque grand satellite, compare l'écart à la planète (est-ouest,
nord-sud) calculé par l'application avec celui de JPL Horizons, sur des
dates réparties entre 2026 et 2060. Nécessite un accès Internet.

    python scripts/validate_horizons.py              # 50 dates, 17 satellites
    python scripts/validate_horizons.py --dates 10 --planet Jupiter
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from satellites import data, engine  # noqa: E402

API = "https://ssd.jpl.nasa.gov/api/horizons.api"
PLANET_ID = {"Jupiter": 599, "Saturn": 699, "Uranus": 799}
MOON_ID = {
    "Io": 501, "Europa": 502, "Ganymede": 503, "Callisto": 504,
    "Mimas": 601, "Enceladus": 602, "Tethys": 603, "Dione": 604,
    "Rhea": 605, "Titan": 606, "Hyperion": 607, "Iapetus": 608,
    "Ariel": 701, "Umbriel": 702, "Titania": 703, "Oberon": 704, "Miranda": 705,
}


def horizons_radec(target: int, times: list[datetime]):
    """AD/Déc astrométriques géocentriques (degrés) aux instants donnés."""
    tlist = " ".join(f"'{t:%Y-%m-%d %H:%M:%S}'" for t in times)
    params = {
        "format": "json", "COMMAND": f"'{target}'", "CENTER": "'500@399'",
        "MAKE_EPHEM": "YES", "EPHEM_TYPE": "OBSERVER", "TLIST": tlist, "TLIST_TYPE": "CAL",
        "TIME_TYPE": "UT", "QUANTITIES": "'1'", "ANG_FORMAT": "DEG", "EXTRA_PREC": "YES",
        "CSV_FORMAT": "YES",
    }
    url = API + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=120) as resp:
        text = json.load(resp)["result"]
    body = text.split("$$SOE")[1].split("$$EOE")[0]
    out = []
    for line in body.strip().splitlines():
        cols = [c.strip() for c in line.split(",")]
        nums = [c for c in cols[1:] if c and c.replace(".", "", 1).replace("-", "", 1).isdigit()]
        out.append((float(nums[0]), float(nums[1])))
    return out


def offsets(ra, dec, ra0, dec0):
    """Projection gnomonique (secondes d'arc) d'une position autour de (ra0, dec0)."""
    ra, dec, ra0, dec0 = map(math.radians, (ra, dec, ra0, dec0))
    da = ra - ra0
    cosc = math.sin(dec0) * math.sin(dec) + math.cos(dec0) * math.cos(dec) * math.cos(da)
    xi = math.cos(dec) * math.sin(da) / cosc
    eta = (math.sin(dec) * math.cos(dec0) - math.cos(dec) * math.sin(dec0) * math.cos(da)) / cosc
    return math.degrees(xi) * 3600, math.degrees(eta) * 3600


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", type=int, default=50)
    ap.add_argument("--planet", choices=[p.name for p in data.PLANETS.values()])
    ap.add_argument("--tolerance", type=float, default=0.1, help="écart toléré en secondes d'arc")
    args = ap.parse_args()

    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    span = datetime(2060, 12, 31, tzinfo=timezone.utc) - t0
    times = [t0 + span * (i / max(1, args.dates - 1)) + timedelta(hours=7.3 * i) for i in range(args.dates)]
    times = [t.replace(microsecond=0) for t in times]

    worst_all = 0.0
    for planet in data.PLANETS.values():
        if args.planet and planet.name != args.planet:
            continue
        ref = horizons_radec(PLANET_ID[planet.key], times)
        for moon in planet.moons:
            hz = horizons_radec(MOON_ID[moon.key], times)
            errs = []
            for t, (pra, pdec), (mra, mdec) in zip(times, ref, hz):
                e_h, n_h = offsets(mra, mdec, pra, pdec)
                snap = engine.compute(planet, t, None)
                m = next(x for x in snap.moons if x.name == moon.name)
                errs.append(math.hypot(m.east_arcsec - e_h, m.north_arcsec - n_h))
            worst = max(errs)
            worst_all = max(worst_all, worst)
            mean = sum(errs) / len(errs)
            flag = "OK " if worst <= args.tolerance else "ÉCART"
            print(f"{flag} {planet.name:8s} {moon.name:10s} moyenne {mean:6.3f}″  max {worst:6.3f}″")
    print(f"\nÉcart maximal : {worst_all:.3f}″ (cible {args.tolerance}″)")
    return 0 if worst_all <= args.tolerance else 1


if __name__ == "__main__":
    sys.exit(main())
