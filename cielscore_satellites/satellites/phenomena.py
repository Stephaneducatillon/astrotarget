"""Phénomènes des satellites : passages, occultations, éclipses, ombres sur le disque."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, time as dtime
from zoneinfo import ZoneInfo

from .data import Planet, Site
from .engine import make_observer, moon_flags, planet_altitudes

KINDS = (
    ("transit", "Passage devant le disque"),
    ("occultation", "Occultation (derrière le disque)"),
    ("eclipse", "Éclipse (dans l'ombre de la planète)"),
    ("shadow", "Ombre sur le disque"),
)


@dataclass
class Phenomenon:
    moon: str
    kind: str              # clé de KINDS
    label: str
    start: datetime | None  # None : déjà en cours au début de la fenêtre
    end: datetime | None    # None : toujours en cours à la fin de la fenêtre
    start_observable: bool | None = None
    end_observable: bool | None = None

    @property
    def observable(self) -> bool | None:
        vals = [v for v in (self.start_observable, self.end_observable) if v is not None]
        if not vals:
            return None
        return any(vals)


def night_window(when: datetime, tz_name: str):
    """Nuit contenant ``when`` : de midi local à midi local le lendemain (en UTC)."""
    tz = ZoneInfo(tz_name)
    local = when.astimezone(tz)
    day = local.date() if local.hour >= 12 else local.date() - timedelta(days=1)
    start = datetime.combine(day, dtime(12, 0), tzinfo=tz)
    return start.astimezone(timezone.utc), (start + timedelta(days=1)).astimezone(timezone.utc)


def _flags_at(planet: Planet, site: Site | None, when: datetime):
    obs = make_observer(when, site)
    body = planet.cls()
    body.compute(obs)
    raw = moon_flags(planet, obs, body, when)
    return {name: dict(zip(("transit", "occultation", "eclipse", "shadow"), v[:4]))
            for name, v in raw.items()}


def _refine(planet, site, moon, kind, t0, t1, v0, tol_s=1.0):
    """Bissection de l'instant où l'indicateur ``kind`` change, entre t0 et t1."""
    while (t1 - t0).total_seconds() > tol_s:
        tm = t0 + (t1 - t0) / 2
        vm = _flags_at(planet, site, tm)[moon][kind]
        if vm == v0:
            t0 = tm
        else:
            t1 = tm
    return t0 + (t1 - t0) / 2


def is_observable(planet: Planet, site: Site | None, when: datetime,
                  min_planet_alt: float = 0.0, max_sun_alt: float = -0.833):
    """Observable si la planète est levée et le Soleil couché."""
    if site is None:
        return None
    p_alt, s_alt = planet_altitudes(planet, when, site)
    return p_alt > min_planet_alt and s_alt < max_sun_alt


def find_phenomena(planet: Planet, start: datetime, end: datetime, site: Site | None = None,
                   step_s: int = 60, min_planet_alt: float = 0.0,
                   max_sun_alt: float = -0.833) -> list[Phenomenon]:
    """Liste des phénomènes entre ``start`` et ``end`` (UTC), heures à la seconde près."""
    step = timedelta(seconds=step_s)
    names = [m.name for m in planet.moons]
    labels = dict(KINDS)

    t_prev = start
    f_prev = _flags_at(planet, site, t_prev)
    open_: dict = {}
    for name in names:
        for kind, _ in KINDS:
            if f_prev[name][kind]:
                open_[(name, kind)] = Phenomenon(name, kind, labels[kind], None, None)

    done: list[Phenomenon] = []
    t = start
    while t < end:
        t = min(t + step, end)
        f = _flags_at(planet, site, t)
        for name in names:
            for kind, _ in KINDS:
                v0, v1 = f_prev[name][kind], f[name][kind]
                if v0 == v1:
                    continue
                te = _refine(planet, site, name, kind, t_prev, t, v0)
                if v1:
                    open_[(name, kind)] = Phenomenon(name, kind, labels[kind], te, None)
                else:
                    ph = open_.pop((name, kind), None) or Phenomenon(name, kind, labels[kind], None, None)
                    ph.end = te
                    done.append(ph)
        t_prev, f_prev = t, f
    done.extend(open_.values())

    for ph in done:
        if ph.start is not None:
            ph.start_observable = is_observable(planet, site, ph.start, min_planet_alt, max_sun_alt)
        if ph.end is not None:
            ph.end_observable = is_observable(planet, site, ph.end, min_planet_alt, max_sun_alt)

    done.sort(key=lambda p: p.start or p.end or start)
    return done
