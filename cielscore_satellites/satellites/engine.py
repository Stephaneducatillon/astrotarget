"""Moteur de calcul : positions des satellites et informations sur la planète.

Conventions PyEphem vérifiées (cf. tests) : pour un satellite, ``x`` est
l'écart vers l'est, ``y`` l'écart vers le sud, ``z`` la profondeur (positive
vers la Terre), le tout en rayons équatoriaux de la planète, dans le repère
du ciel (nord céleste en haut).

Les coordonnées célestes des satellites sont recalculées ici à partir des
écarts ``x``/``y`` (projection gnomonique inverse) : PyEphem ajoute l'écart
est-ouest à l'ascension droite sans le diviser par cos(δ), ce qui décale
l'AD de quelques secondes d'arc.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

import ephem
import numpy as np

from .data import Planet, Site

ARCSEC = math.pi / 180.0 / 3600.0
DEG = math.pi / 180.0

# États possibles d'un satellite
VISIBLE = "visible"
TRANSIT = "devant le disque"
OCCULTED = "derrière le disque"
ECLIPSED = "dans l'ombre"


# ─── Outils ──────────────────────────────────────────────────────────────────

def to_ephem_date(dt: datetime) -> ephem.Date:
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return ephem.Date(dt)


def from_ephem_date(d) -> datetime:
    return ephem.Date(d).datetime().replace(tzinfo=timezone.utc)


def make_observer(dt: datetime, site: Site | None) -> ephem.Observer:
    obs = ephem.Observer()
    if site is not None:
        obs.lat = str(site.lat)
        obs.lon = str(site.lon)
        obs.elevation = site.elevation
    obs.date = to_ephem_date(dt)
    obs.epoch = ephem.J2000
    return obs


def unit_vector(ra: float, dec: float) -> np.ndarray:
    cd = math.cos(dec)
    return np.array([cd * math.cos(ra), cd * math.sin(ra), math.sin(dec)])


def sky_basis(ra: float, dec: float):
    """Vecteurs unitaires (est, nord, vers la Terre) au point (ra, dec)."""
    east = np.array([-math.sin(ra), math.cos(ra), 0.0])
    north = np.array([-math.sin(dec) * math.cos(ra),
                      -math.sin(dec) * math.sin(ra),
                      math.cos(dec)])
    toward_earth = -unit_vector(ra, dec)
    return east, north, toward_earth


def offset_to_radec(ra0: float, dec0: float, d_east: float, d_north: float):
    """Projection gnomonique inverse : écarts (radians) → (ra, dec)."""
    denom = math.cos(dec0) - d_north * math.sin(dec0)
    ra = ra0 + math.atan2(d_east, denom)
    dec = math.atan2(math.sin(dec0) + d_north * math.cos(dec0),
                     math.hypot(d_east, denom))
    return ra % (2 * math.pi), dec


def format_ra(ra: float, decimals: int = 1) -> str:
    h = (ra / (2 * math.pi) * 24.0) % 24.0
    total = round(h * 3600.0, decimals)
    hh = int(total // 3600)
    mm = int((total - hh * 3600) // 60)
    ss = total - hh * 3600 - mm * 60
    return f"{hh % 24:d} h {mm:02d} min {ss:0{3 + decimals}.{decimals}f} s"


def format_dec(dec: float, decimals: int = 0) -> str:
    sign = "+" if dec >= 0 else "−"
    total = round(abs(dec) / DEG * 3600.0, decimals)
    dd = int(total // 3600)
    mm = int((total - dd * 3600) // 60)
    ss = total - dd * 3600 - mm * 60
    if decimals:
        s = f"{ss:0{3 + decimals}.{decimals}f}"
    else:
        s = f"{int(round(ss)):02d}"
    return f"{sign}{dd:02d}° {mm:02d}′ {s}″"


def format_ra_short(ra: float) -> str:
    h = (ra / (2 * math.pi) * 24.0) % 24.0
    total = round(h * 3600.0, 1)
    hh = int(total // 3600)
    mm = int((total - hh * 3600) // 60)
    ss = total - hh * 3600 - mm * 60
    return f"{hh % 24}h{mm:02d}m{ss:04.1f}s"


def format_dec_short(dec: float) -> str:
    sign = "+" if dec >= 0 else "−"
    total = round(abs(dec) / DEG * 3600.0)
    dd = int(total // 3600)
    mm = int((total - dd * 3600) // 60)
    ss = int(total - dd * 3600 - mm * 60)
    return f"{sign}{dd}°{mm:02d}′{ss:02d}″"


# ─── Géométrie de la planète ─────────────────────────────────────────────────

def pole_radec(planet: Planet, dt: datetime):
    """Pôle nord IAU (ICRF J2000), en radians."""
    d = float(to_ephem_date(dt)) - float(ephem.J2000)  # jours depuis J2000.0
    t = d / 36525.0
    return ((planet.pole_ra0 + planet.pole_ra_rate * t) * DEG,
            (planet.pole_dec0 + planet.pole_dec_rate * t) * DEG)


def pole_geometry(planet: Planet, dt: datetime, ra: float, dec: float):
    """Angle de position du pôle nord (°) et latitude planétocentrique de la Terre (°)."""
    a0, d0 = pole_radec(planet, dt)
    pa = math.atan2(math.cos(d0) * math.sin(a0 - ra),
                    math.sin(d0) * math.cos(dec)
                    - math.cos(d0) * math.sin(dec) * math.cos(a0 - ra))
    sin_de = (-math.sin(d0) * math.sin(dec)
              - math.cos(d0) * math.cos(dec) * math.cos(a0 - ra))
    return (pa / DEG) % 360.0, math.asin(max(-1.0, min(1.0, sin_de))) / DEG


def apparent_axis_ratio(flattening: float, earth_lat_deg: float) -> float:
    """Rapport petit axe / grand axe du disque vu depuis la Terre."""
    c = math.cos(earth_lat_deg * DEG)
    return math.sqrt((1 - flattening) ** 2 * c * c + (1 - c * c))


def ray_ellipsoid(origin, direction, pole, flattening):
    """Plus petite racine t > 0 de l'intersection rayon / ellipsoïde (rayon équatorial 1)."""
    c = 1.0 / (1.0 - flattening) ** 2 - 1.0
    dp = float(np.dot(direction, pole))
    op = float(np.dot(origin, pole))
    a = float(np.dot(direction, direction)) + c * dp * dp
    b = 2.0 * (float(np.dot(origin, direction)) + c * op * dp)
    cc = float(np.dot(origin, origin)) + c * op * op - 1.0
    disc = b * b - 4 * a * cc
    if disc < 0:
        return None
    sq = math.sqrt(disc)
    for t in sorted(((-b - sq) / (2 * a), (-b + sq) / (2 * a))):
        if t > 0:
            return t
    return None


# ─── Résultats ───────────────────────────────────────────────────────────────

@dataclass
class MoonPosition:
    name: str
    mag: float
    ra_j2000: float
    dec_j2000: float
    ra_app: float
    dec_app: float
    alt: float | None          # degrés
    az: float | None           # degrés
    east_radii: float          # écart est (+) / ouest (−), rayons planétaires
    north_radii: float         # écart nord (+) / sud (−)
    depth_radii: float         # profondeur, positive vers la Terre
    east_arcsec: float
    north_arcsec: float
    dist_arcsec: float
    dist_radii: float
    pa_deg: float              # angle de position (depuis le nord, vers l'est)
    transit: bool
    occulted: bool
    eclipsed: bool
    shadow: tuple | None       # position de l'ombre sur le disque (est, nord) en rayons
    state: str = VISIBLE


@dataclass
class PlanetInfo:
    name: str
    ra_j2000: float
    dec_j2000: float
    ra_app: float
    dec_app: float
    alt: float | None
    az: float | None
    radius_arcsec: float
    diameter_arcsec: float
    mag: float
    earth_distance_au: float
    sun_distance_au: float
    phase: float
    pole_pa: float
    earth_lat: float
    axis_ratio: float
    flattening: float
    ring_outer: float
    ring_tilt: float | None = None   # Saturne : inclinaison des anneaux vue de la Terre
    cml_i: float | None = None       # Jupiter : méridien central système I
    cml_ii: float | None = None      # Jupiter : méridien central système II
    rise: datetime | None = None
    transit_time: datetime | None = None
    set: datetime | None = None
    transit_alt: float | None = None
    circumpolar: bool = False
    never_up: bool = False
    sun_alt: float | None = None


@dataclass
class Snapshot:
    planet: Planet
    when: datetime
    site: Site | None
    info: PlanetInfo
    moons: list = field(default_factory=list)


# ─── Calcul principal ────────────────────────────────────────────────────────

def _moon_mag(h_mag: float, r_au: float, delta_au: float) -> float:
    return h_mag + 5.0 * math.log10(r_au * delta_au)


def _sun_direction(obs: ephem.Observer, body, basis):
    """Direction planète → Soleil dans le repère (est, nord, vers la Terre)."""
    sun = ephem.Sun()
    sun.compute(obs)
    p = unit_vector(float(body.a_ra), float(body.a_dec)) * body.earth_distance
    s = unit_vector(float(sun.a_ra), float(sun.a_dec)) * sun.earth_distance
    v = s - p
    v /= np.linalg.norm(v)
    e, n, w = basis
    return np.array([np.dot(v, e), np.dot(v, n), np.dot(v, w)])


def _pole_in_sky(planet: Planet, dt: datetime, basis):
    a0, d0 = pole_radec(planet, dt)
    p = unit_vector(a0, d0)
    e, n, w = basis
    return np.array([np.dot(p, e), np.dot(p, n), np.dot(p, w)])


def moon_flags(planet: Planet, obs: ephem.Observer, body, dt: datetime):
    """Indicateurs de phénomènes pour chaque satellite, à un instant donné.

    Retourne un dict nom → (transit, occultation, éclipse, ombre, position de l'ombre).
    ``body`` doit déjà être calculé pour ``obs``.
    """
    basis = sky_basis(float(body.a_ra), float(body.a_dec))
    sun_dir = _sun_direction(obs, body, basis)
    pole = _pole_in_sky(planet, dt, basis)
    to_earth = np.array([0.0, 0.0, 1.0])
    out = {}
    for m in planet.moons:
        sat = m.cls()
        sat.compute(obs)
        pos = np.array([sat.x, -sat.y, sat.z])
        occulted = not bool(sat.earth_visible)
        eclipsed = not bool(sat.sun_visible)
        transit = (sat.z > 0 and not occulted
                   and ray_ellipsoid(pos, -to_earth, pole, planet.flattening) is not None)
        shadow = None
        if not eclipsed:
            t = ray_ellipsoid(pos, -sun_dir, pole, planet.flattening)
            if t is not None:
                hit = pos - t * sun_dir
                c = 1.0 / (1.0 - planet.flattening) ** 2 - 1.0
                normal = hit + c * np.dot(hit, pole) * pole
                if normal[2] > 0:   # face tournée vers la Terre
                    shadow = (float(hit[0]), float(hit[1]))
        out[m.name] = (transit, occulted, eclipsed, shadow is not None, shadow)
    return out


def compute(planet: Planet, when: datetime, site: Site | None = None) -> Snapshot:
    """Positions de tous les satellites de ``planet`` à l'instant ``when`` (UTC)."""
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    obs = make_observer(when, site)
    body = planet.cls()
    body.compute(obs)

    radius = body.size / 2.0  # secondes d'arc (rayon équatorial)
    ra_j, dec_j = float(body.a_ra), float(body.a_dec)
    ra_a, dec_a = (float(body.ra), float(body.dec)) if site else (float(body.g_ra), float(body.g_dec))
    pole_pa, earth_lat = pole_geometry(planet, when, ra_j, dec_j)

    sun = ephem.Sun()
    sun.compute(obs)

    info = PlanetInfo(
        name=planet.name,
        ra_j2000=ra_j, dec_j2000=dec_j, ra_app=ra_a, dec_app=dec_a,
        alt=float(body.alt) / DEG if site else None,
        az=float(body.az) / DEG if site else None,
        radius_arcsec=radius, diameter_arcsec=body.size,
        mag=float(body.mag),
        earth_distance_au=float(body.earth_distance),
        sun_distance_au=float(body.sun_distance),
        phase=float(body.phase),
        pole_pa=pole_pa, earth_lat=earth_lat,
        axis_ratio=apparent_axis_ratio(planet.flattening, earth_lat),
        flattening=planet.flattening,
        ring_outer=planet.ring_outer,
        sun_alt=float(sun.alt) / DEG if site else None,
    )
    if planet.key == "Saturn":
        info.ring_tilt = float(body.earth_tilt) / DEG
    if planet.key == "Jupiter":
        info.cml_i = float(body.cmlI) / DEG
        info.cml_ii = float(body.cmlII) / DEG

    if site is not None:
        # Lever / passage / coucher les plus proches autour de l'instant choisi
        o2 = make_observer(when, site)
        b2 = planet.cls()
        try:
            if body.alt > 0:
                info.rise = from_ephem_date(o2.previous_rising(b2))
            else:
                info.rise = from_ephem_date(o2.next_rising(b2))
            o2.date = to_ephem_date(info.rise)
            info.transit_time = from_ephem_date(o2.next_transit(b2))
            info.set = from_ephem_date(o2.next_setting(b2))
        except ephem.AlwaysUpError:
            info.circumpolar = True
        except ephem.NeverUpError:
            info.never_up = True
        o3 = make_observer(info.transit_time or when, site)
        b3 = planet.cls()
        b3.compute(o3)
        info.transit_alt = float(b3.alt) / DEG

    snap = Snapshot(planet=planet, when=when, site=site, info=info)

    flags = moon_flags(planet, obs, body, when)
    for m in planet.moons:
        sat = m.cls()
        sat.compute(obs)
        east_r, north_r = float(sat.x), -float(sat.y)
        east_as, north_as = east_r * radius, north_r * radius
        ra_mj, dec_mj = offset_to_radec(ra_j, dec_j, east_as * ARCSEC, north_as * ARCSEC)
        ra_ma, dec_ma = offset_to_radec(ra_a, dec_a, east_as * ARCSEC, north_as * ARCSEC)
        transit, occulted, eclipsed, _, shadow = flags[m.name]
        if occulted:
            state = OCCULTED
        elif eclipsed:
            state = ECLIPSED
        elif transit:
            state = TRANSIT
        else:
            state = VISIBLE
        snap.moons.append(MoonPosition(
            name=m.name,
            mag=_moon_mag(m.h_mag, info.sun_distance_au, info.earth_distance_au),
            ra_j2000=ra_mj, dec_j2000=dec_mj, ra_app=ra_ma, dec_app=dec_ma,
            alt=float(sat.alt) / DEG if site else None,
            az=float(sat.az) / DEG if site else None,
            east_radii=east_r, north_radii=north_r, depth_radii=float(sat.z),
            east_arcsec=east_as, north_arcsec=north_as,
            dist_arcsec=math.hypot(east_as, north_as),
            dist_radii=math.hypot(east_r, north_r),
            pa_deg=math.degrees(math.atan2(east_as, north_as)) % 360.0,
            transit=transit, occulted=occulted, eclipsed=eclipsed,
            shadow=shadow, state=state,
        ))
    snap.moons.sort(key=lambda m: m.dist_arcsec)
    return snap


def planet_altitudes(planet: Planet, when: datetime, site: Site):
    """Hauteurs (°) de la planète et du Soleil à un instant donné."""
    obs = make_observer(when, site)
    body = planet.cls()
    body.compute(obs)
    sun = ephem.Sun()
    sun.compute(obs)
    return float(body.alt) / DEG, float(sun.alt) / DEG
