"""Précalcul des tables de positions pour le site statique CielScore Satellites.

Pour chaque planète et chaque année, ajuste des polynômes de Tchebychev par
tranche (la méthode des éphémérides du JPL) sur :

- la planète : AD et Déc astrométriques J2000, AD et Déc apparentes
  géocentriques du jour, rayon apparent (″), et pour Jupiter les méridiens
  centraux I et II (déroulés) — en float64 ;
- chaque satellite : x (est), y (sud), z (vers la Terre) en rayons
  planétaires, repère du ciel géocentrique — en float32.

Deux sources possibles, même format de sortie :

- ``pyephem`` (par défaut) : moteur CielScore. Ses tables internes de
  satellites s'arrêtent début 2040 : les jours où la position est aberrante
  (ou nulle, Uranus après 2040) sont marqués « non disponibles » ; après 2040,
  Jupiter et Saturne passent sur une théorie de secours moins précise.
- ``jpl`` : fichiers SPICE .bsp du JPL (NAIF) lus avec Skyfield — à télécharger
  au préalable (de440s.bsp, jup365.bsp, sat441.bsp, ura184_part-*.bsp…).

Le script est fait pour être relancé une ou deux fois par an.

    python web/tools/build_tables.py                       # 2025 → 2060, PyEphem
    python web/tools/build_tables.py --years 2026 2027
    python web/tools/build_tables.py --source jpl --kernels ~/naif
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import ephem
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from satellites import data  # noqa: E402
from satellites.engine import pyephem_covers  # noqa: E402

OUT = ROOT / "web" / "data"

# Tranches d'un jour : PyEphem change de table interne à 0 h, tous les 30 à 40 jours,
# et une tranche ne doit pas chevaucher ce raccord. Degré choisi pour une erreur < 0,0002 rayon.
MOON_FIT = {
    "Mimas": (1, 11), "Enceladus": (1, 8), "Tethys": (1, 7), "Dione": (1, 6),
    "Rhea": (1, 5), "Titan": (1, 4), "Hyperion": (1, 4), "Iapetus": (1, 4),
    "Io": (1, 7), "Europa": (1, 6), "Ganymede": (1, 4), "Callisto": (1, 4),
    "Miranda": (1, 8), "Ariel": (1, 6), "Umbriel": (1, 5), "Titania": (1, 4), "Oberon": (1, 4),
}
PLANET_FIT = (4, 12)


def planet_series(planet):
    comps = ["ra", "dec", "gra", "gdec", "radius"]
    if planet.key == "Jupiter":
        comps += ["cml1", "cml2"]
    return comps


# Vitesses nominales des méridiens centraux de Jupiter (°/jour, Meeus) : on tabule
# l'écart à ce mouvement uniforme, qui varie lentement (temps de lumière).
CML_RATE = (877.90 * math.pi / 180, 870.27 * math.pi / 180)


def _cml_cols(planet, dates, start):
    """Méridiens centraux I et II de Jupiter (PyEphem), moins le mouvement uniforme."""
    obs = ephem.Observer()
    out = []
    for d in dates:
        obs.date = d
        b = planet.cls()
        b.compute(obs)
        dt = float(d) - float(start)
        out.append([(float(b.cmlI) - CML_RATE[0] * dt) % (2 * math.pi),
                    (float(b.cmlII) - CML_RATE[1] * dt) % (2 * math.pi)])
    return np.array(out)


class PyEphemSource:
    name = f"PyEphem {ephem.__version__}"

    def planet(self, planet, dates, start):
        obs = ephem.Observer()
        out = []
        for d in dates:
            obs.date = d
            b = planet.cls()
            b.compute(obs)
            out.append([float(b.a_ra), float(b.a_dec), float(b.g_ra), float(b.g_dec), b.size / 2.0])
        arr = np.array(out)
        if planet.key == "Jupiter":
            arr = np.c_[arr, _cml_cols(planet, dates, start)]
        return arr

    def moon(self, planet, moon, dates):
        obs = ephem.Observer()
        out = []
        for d in dates:
            obs.date = d
            m = moon.cls()
            m.compute(obs)
            out.append((m.x, m.y, m.z))
        return np.array(out)


# Identifiants NAIF et rayons équatoriaux (km, pck00011)
NAIF_PLANET = {"Jupiter": (5, 599, 71492.0), "Saturn": (6, 699, 60268.0), "Uranus": (7, 799, 25559.0)}
NAIF_MOON = {
    "Io": 501, "Europa": 502, "Ganymede": 503, "Callisto": 504,
    "Mimas": 601, "Enceladus": 602, "Tethys": 603, "Dione": 604,
    "Rhea": 605, "Titan": 606, "Hyperion": 607, "Iapetus": 608,
    "Ariel": 701, "Umbriel": 702, "Titania": 703, "Oberon": 704, "Miranda": 705,
}


class JplSource:
    """Fichiers SPICE du JPL via Skyfield (positions astrométriques géocentriques)."""

    def __init__(self, kernels: Path):
        from skyfield.api import load, load_file

        self.ts = load.timescale(builtin=True)
        files = sorted(kernels.glob("*.bsp"))
        if not files:
            raise SystemExit(f"Aucun fichier .bsp dans {kernels}")
        self.kernels = [load_file(str(f)) for f in files]
        self.name = "JPL " + ", ".join(f.name for f in files)
        self.earth = self._segment_chain(399)

    def _find(self, center, target):
        for k in self.kernels:
            for seg in k.segments:
                if seg.center == center and seg.target == target:
                    return seg
        return None

    def _segment_chain(self, target):
        """Somme de segments du barycentre du système solaire (0) jusqu'à ``target``."""
        parents = {399: 3, 301: 3, 3: 0}
        if target in parents:
            chain = [target]
            while chain[-1] != 0:
                chain.append(parents[chain[-1]])
        else:
            bary = target // 100 if target > 99 else target
            chain = [target, bary, 0] if target != bary else [bary, 0]
        vec = None
        if len(chain) == 3 and self._find(chain[1], chain[0]) is None and target % 100 == 99:
            chain = chain[1:]   # centre de la planète absent : barycentre du système (écart < 0,01″)
        for tgt, ctr in zip(chain[::-1][1:], chain[::-1][:-1]):
            seg = self._find(ctr, tgt)
            if seg is None:
                raise SystemExit(f"Segment {ctr} → {tgt} introuvable dans les fichiers fournis")
            vec = seg if vec is None else vec + seg
        return vec

    def _times(self, dates):
        return self.ts.ut1_jd(np.asarray(dates, dtype=float) + 2415020.0)

    def _radec(self, body, t):
        astro = self.earth.at(t).observe(body)
        ra, dec, dist = astro.radec()
        g_ra, g_dec, _ = astro.apparent().radec(epoch="date")
        return ra.radians, dec.radians, g_ra.radians, g_dec.radians, dist.km

    def planet(self, planet, dates, start):
        _, naif, radius_km = NAIF_PLANET[planet.key]
        t = self._times(dates)
        ra, dec, gra, gdec, dist = self._radec(self._segment_chain(naif), t)
        arr = np.c_[ra, dec, gra, gdec, np.degrees(np.arcsin(radius_km / dist)) * 3600]
        if planet.key == "Jupiter":
            arr = np.c_[arr, _cml_cols(planet, dates, start)]
        return arr

    def moon(self, planet, moon, dates):
        _, naif, radius_km = NAIF_PLANET[planet.key]
        t = self._times(dates)
        pra, pdec, _, _, pdist = self._radec(self._segment_chain(naif), t)
        mra, mdec, _, _, mdist = self._radec(self._segment_chain(NAIF_MOON[moon.key]), t)
        # projection gnomonique autour de la planète
        da = mra - pra
        cosc = np.sin(pdec) * np.sin(mdec) + np.cos(pdec) * np.cos(mdec) * np.cos(da)
        xi = np.cos(mdec) * np.sin(da) / cosc
        eta = (np.sin(mdec) * np.cos(pdec) - np.cos(mdec) * np.sin(pdec) * np.cos(da)) / cosc
        r_ang = np.arcsin(radius_km / pdist)
        return np.c_[xi / r_ang, -eta / r_ang, (pdist - mdist) / radius_km]


class MoonSampler:
    comps = ("x", "y", "z")

    def __init__(self, src, planet, moon):
        self.src, self.planet, self.moon = src, planet, moon

    def __call__(self, dates):
        return self.src.moon(self.planet, self.moon, dates)


SOURCE = None


def _source():
    return SOURCE


def valid_moon(v):
    """Position plausible : entre 0,5 et 80 rayons du centre de la planète."""
    r = np.sqrt((v ** 2).sum(axis=1))
    return bool(np.all(np.isfinite(r)) and np.all((r > 0.5) & (r < 80)))


def unwrap_cols(arr, cols):
    for c in cols:
        arr[:, c] = np.unwrap(arr[:, c])
    return arr


def fit_series(sampler, start, nseg, seg, deg, unwrap=(), valid=None, covered=None):
    """Coefficients (nseg, ncomp, deg+1), erreur maximale de contrôle, tranches invalides.

    Une tranche dont un échantillon échoue au test ``valid`` est remplie de NaN :
    le site affiche alors « position non disponible ».
    """
    n = deg + 1
    xk = np.cos(np.pi * (np.arange(n) + 0.5) / n)
    coefs, err, bad = [], 0.0, []
    xt = np.array([-0.99, -0.6, -0.2, 0.2, 0.6, 0.99])
    for k in range(nseg):
        a = start + k * seg
        nodes = a + (xk + 1) / 2 * seg
        test = a + (xt + 1) / 2 * seg
        t = np.r_[nodes, test]
        if covered is not None and not covered(a):
            coefs.append(np.full((len(sampler.comps), n), np.nan))
            bad.append(k)
            continue
        v = sampler(t)
        if unwrap:   # angles : déroulés dans l'ordre chronologique
            order = np.argsort(t)
            v[order] = unwrap_cols(v[order], unwrap)
        if valid is not None and not valid(v):
            coefs.append(np.full((v.shape[1], n), np.nan))
            bad.append(k)
            continue
        c = np.polynomial.chebyshev.chebfit(xk, v[:n], deg)          # (n, ncomp)
        err = np.maximum(err, np.abs(np.polynomial.chebyshev.chebval(xt, c).T - v[n:]).max(axis=0))
        coefs.append(c.T)
    return np.array(coefs), err, bad


def _init_worker(source_name, kernels):
    global SOURCE
    SOURCE = JplSource(Path(kernels)) if source_name == "jpl" else PyEphemSource()


def build_year(args):
    planet_name, year = args
    src = _source()
    planet = data.PLANETS[planet_name]
    start_dt = datetime(year, 1, 1, tzinfo=timezone.utc)
    days = (datetime(year + 1, 1, 1, tzinfo=timezone.utc) - start_dt).days
    start = ephem.Date(start_dt.replace(tzinfo=None))
    chunks, report, missing = [], {}, {}

    seg, deg = PLANET_FIT
    nseg = math.ceil(days / seg)
    comps = planet_series(planet)
    unwrap = [0, 2] + ([5, 6] if planet.key == "Jupiter" else [])
    c, err, _ = fit_series(lambda d: src.planet(planet, d, start), start, nseg, seg, deg, unwrap)
    chunks.append(c.astype("<f8").tobytes())
    # erreurs planète en secondes d'arc (angles et rayon), méridiens centraux en degrés
    report["planet"] = float(max(err[:4].max() * 206264.8, err[4]))
    report["cml"] = float(err[5:].max() * 180 / math.pi) if len(err) > 5 else 0.0

    for m in planet.moons:
        seg, deg = MOON_FIT[m.key]
        nseg = math.ceil(days / seg)
        sampler = MoonSampler(src, planet, m)
        covered = None
        if isinstance(src, PyEphemSource):
            covered = lambda d: pyephem_covers(planet.key, ephem.Date(d).datetime())  # noqa: E731
        c, err, bad = fit_series(sampler, start, nseg, seg, deg, valid=valid_moon, covered=covered)
        chunks.append(c.astype("<f4").tobytes())
        report[m.key] = float(np.max(err))
        if bad:
            missing[m.key] = [(start_dt + timedelta(days=k * seg)).strftime("%Y-%m-%d") for k in bad]

    folder = OUT / planet.key.lower()
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{year}.bin").write_bytes(b"".join(chunks))
    return planet_name, year, report, missing


def _ranges(days):
    """Liste de jours « AAAA-MM-JJ » → intervalles [début, fin] contigus."""
    out = []
    for d in sorted(days):
        day = datetime.strptime(d, "%Y-%m-%d")
        if out and (day - datetime.strptime(out[-1][1], "%Y-%m-%d")).days == 1:
            out[-1][1] = d
        else:
            out.append([d, d])
    return out


def write_meta(years, source_name, missing):
    for planet in data.PLANETS.values():
        seg, deg = PLANET_FIT
        meta = {
            "planet": planet.key,
            "name": planet.name,
            "years": years,
            "time": "jours UT depuis le 1er janvier 00:00 UT de l'année",
            "source": source_name,
            "cml_rate_rad_per_day": list(CML_RATE),
            "unavailable": {k: _ranges(v) for k, v in missing.get(planet.name, {}).items()},
            "reduced_precision_from": ("2041-01-01" if source_name.startswith("PyEphem")
                                       and planet.key in ("Jupiter", "Saturn") else None),
            "series": [{"name": "planet", "type": "f64", "comps": planet_series(planet), "seg": seg, "deg": deg}]
            + [{"name": m.key, "label": m.name, "h_mag": m.h_mag, "type": "f32", "comps": ["x", "y", "z"],
                "seg": MOON_FIT[m.key][0], "deg": MOON_FIT[m.key][1]} for m in planet.moons],
        }
        folder = OUT / planet.key.lower()
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs=2, type=int, default=[data.YEAR_MIN, data.YEAR_MAX])
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--source", choices=["pyephem", "jpl"], default="pyephem")
    ap.add_argument("--kernels", default=".", help="dossier des fichiers .bsp (source jpl)")
    a = ap.parse_args()
    years = list(range(a.years[0], a.years[1] + 1))
    tasks = [(p, y) for p in data.PLANETS for y in years]
    worst = 0.0
    missing: dict = {}
    _init_worker(a.source, a.kernels)
    source_name = SOURCE.name
    with ProcessPoolExecutor(a.jobs, initializer=_init_worker, initargs=(a.source, a.kernels)) as pool:
        for name, year, rep, miss in pool.map(build_year, tasks):
            for k, v in miss.items():
                missing.setdefault(name, {}).setdefault(k, []).extend(v)
            moon_err = max(v for k, v in rep.items() if k not in ("planet", "cml"))
            worst = max(worst, moon_err)
            n_miss = sum(len(v) for v in miss.values())
            print(f"{name:8s} {year}  erreur max satellites {moon_err:.2e} rayon"
                  f"  planète {rep['planet']:.1e}″  MC {rep['cml']:.1e}°"
                  + (f"  jours non disponibles {n_miss}" if n_miss else ""), flush=True)
    write_meta(years, source_name, missing)
    print(f"Erreur d'ajustement maximale : {worst:.2e} rayon planétaire")


if __name__ == "__main__":
    main()
