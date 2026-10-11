"""Tests automatiques : une mise à jour ne doit rien casser sans qu'on le voie.

Valeurs de référence : exemples calculés du cahier des charges (Cantin,
nuit du 11 au 12 octobre 2026).
"""

import math
from datetime import datetime, timezone

import pytest

from satellites import data, engine, exports, phenomena, photo, render

CANTIN = data.SITES["Cantin"]
T_EXAMPLE = datetime(2026, 10, 12, 4, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def jupiter():
    return engine.compute(data.JUPITER, T_EXAMPLE, CANTIN)


def _moon(snap, name):
    return next(m for m in snap.moons if m.name == name)


# ─── Positions : exemple du cahier des charges ───────────────────────────────

@pytest.mark.parametrize("name, ew, ns", [
    ("Io", 70, -27), ("Europe", 93, -34), ("Ganymède", 237, -91), ("Callisto", -283, 106),
])
def test_offsets_match_spec_example(jupiter, name, ew, ns):
    m = _moon(jupiter, name)
    assert m.east_arcsec == pytest.approx(ew, abs=1.0)
    assert m.north_arcsec == pytest.approx(ns, abs=1.0)


@pytest.mark.parametrize("name, alt, az", [
    ("Io", 30.33, 103.71), ("Europe", 30.33, 103.71), ("Ganymède", 30.29, 103.68), ("Callisto", 30.42, 103.77),
])
def test_alt_az_match_spec_example(jupiter, name, alt, az):
    m = _moon(jupiter, name)
    assert m.alt == pytest.approx(alt, abs=0.02)
    assert m.az == pytest.approx(az, abs=0.02)


def test_jupiter_radius_and_rise(jupiter):
    assert jupiter.info.radius_arcsec == pytest.approx(17.0, abs=0.1)
    assert jupiter.info.rise.strftime("%H:%M") == "00:42"


def test_radec_consistent_with_offsets(jupiter):
    """Les coordonnées J2000 recalculées redonnent exactement les écarts E-O / N-S."""
    p = jupiter.info
    a0, d0 = p.ra_j2000, p.dec_j2000
    for m in jupiter.moons:
        da, d = m.ra_j2000 - a0, m.dec_j2000
        cosc = math.sin(d0) * math.sin(d) + math.cos(d0) * math.cos(d) * math.cos(da)
        d_east = math.cos(d) * math.sin(da) / cosc / engine.ARCSEC
        d_north = (math.sin(d) * math.cos(d0) - math.cos(d) * math.sin(d0) * math.cos(da)) / cosc / engine.ARCSEC
        assert d_east == pytest.approx(m.east_arcsec, abs=0.001)
        assert d_north == pytest.approx(m.north_arcsec, abs=0.001)


def test_moons_sorted_by_distance(jupiter):
    d = [m.dist_arcsec for m in jupiter.moons]
    assert d == sorted(d)


def test_moons_lie_along_jupiter_equator(jupiter):
    """Les galiléens sont alignés sur l'équateur : angle de position = pôle ± 90°."""
    for m in jupiter.moons:
        diff = (m.pa_deg - jupiter.info.pole_pa) % 180
        assert diff == pytest.approx(90, abs=3)


def test_saturn_ring_tilt_matches_pole_model():
    s = engine.compute(data.SATURN, datetime(2026, 10, 11, 20, tzinfo=timezone.utc), CANTIN)
    assert s.info.earth_lat == pytest.approx(s.info.ring_tilt, abs=0.1)


def test_titan_iapetus_outside_c2pu_frame():
    """Exemple de cadrage : Titan ≈ 193″ et Japet ≈ 455″ hors du cadre de 186″ de demi-largeur."""
    s = engine.compute(data.SATURN, datetime(2026, 10, 11, 20, tzinfo=timezone.utc),
                       data.SITES["Calern — C2PU (OCA)"])
    inst = next(iter(data.INSTRUMENTS.values()))
    half_w = inst.field_arcsec[0] / 2
    assert half_w == pytest.approx(186, abs=2)
    assert _moon(s, "Titan").dist_arcsec == pytest.approx(193, abs=4)
    assert _moon(s, "Japet").dist_arcsec == pytest.approx(455, abs=4)
    assert _moon(s, "Japet").dist_arcsec > half_w


@pytest.mark.parametrize("planet", list(data.PLANETS.values()))
@pytest.mark.parametrize("year", [2025, 2040, 2060])
def test_all_planets_compute_over_period(planet, year):
    s = engine.compute(planet, datetime(year, 6, 1, tzinfo=timezone.utc), None)
    assert len(s.moons) == len(planet.moons)
    assert all(m.alt is None for m in s.moons)
    assert all(math.isfinite(m.east_arcsec) for m in s.moons)


def test_seventeen_moons():
    assert sum(len(p.moons) for p in data.PLANETS.values()) == 17


def test_moon_magnitudes_plausible(jupiter):
    for m in jupiter.moons:
        assert 4.5 < m.mag < 7


# ─── Phénomènes ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def night_events():
    start, end = phenomena.night_window(T_EXAMPLE, "Europe/Paris")
    return phenomena.find_phenomena(data.JUPITER, start, end, CANTIN)


def _event(events, moon, kind):
    return next(e for e in events if e.moon == moon and e.kind == kind)


def test_io_eclipse_start(night_events):
    e = _event(night_events, "Io", "eclipse")
    target = datetime(2026, 10, 11, 20, 13, tzinfo=timezone.utc)
    assert abs((e.start - target).total_seconds()) < 60
    assert e.start_observable is False   # Jupiter se lève à 00:42 UT


def test_io_occultation(night_events):
    e = _event(night_events, "Io", "occultation")
    assert abs((e.start - datetime(2026, 10, 11, 21, 9, tzinfo=timezone.utc)).total_seconds()) < 60
    assert abs((e.end - datetime(2026, 10, 11, 23, 27, tzinfo=timezone.utc)).total_seconds()) < 60


def test_night_window():
    start, end = phenomena.night_window(T_EXAMPLE, "Europe/Paris")
    assert start == datetime(2026, 10, 11, 10, 0, tzinfo=timezone.utc)
    assert (end - start).total_seconds() == 86400


def test_transit_and_shadow_found_over_a_week():
    """Sur une semaine, chaque galiléen intérieur passe devant Jupiter et y projette son ombre."""
    start = datetime(2026, 10, 1, tzinfo=timezone.utc)
    end = datetime(2026, 10, 8, tzinfo=timezone.utc)
    ev = phenomena.find_phenomena(data.JUPITER, start, end, None, step_s=300)
    for moon in ("Io", "Europe"):
        kinds = {e.kind for e in ev if e.moon == moon}
        assert {"transit", "shadow", "occultation"} <= kinds
    # passage : satellite devant le disque au milieu de l'événement
    tr = next(e for e in ev if e.moon == "Io" and e.kind == "transit" and e.start and e.end)
    mid = tr.start + (tr.end - tr.start) / 2
    snap = engine.compute(data.JUPITER, mid, None)
    io = _moon(snap, "Io")
    assert io.state == engine.TRANSIT and io.dist_radii < 1
    # Io traverse le disque en ≈ 2 h 15
    assert 1.5 * 3600 < (tr.end - tr.start).total_seconds() < 2.6 * 3600


# ─── Formats ─────────────────────────────────────────────────────────────────

def test_formatting():
    assert engine.format_ra(math.radians(15 * 9.5)) == "9 h 30 min 00.0 s"
    assert engine.format_dec(math.radians(-(15 + 2 / 60 + 35 / 3600))) == "−15° 02′ 35″"
    assert engine.format_ra_short(math.radians(15 * 9.5)) == "9h30m00.0s"


def test_view_orientations():
    # ciel : est à gauche ; Newton : est à droite, sud en haut ; renvoi coudé : est à droite
    assert render.View(0, False)(1, 0) == pytest.approx((-1, 0))
    x, y = render.View(180, False)(1, 1)
    assert (x, y) == pytest.approx((1, -1))
    assert render.View(0, True)(1, 1) == pytest.approx((1, 1))


def test_exports(jupiter):
    csv = exports.moons_csv(jupiter).decode("utf-8-sig")
    assert "CielScore" in csv.splitlines()[0]
    assert len(csv.strip().splitlines()) == 2 + 4
    fig = render.draw(jupiter, instrument=next(iter(data.INSTRUMENTS.values())))
    png = exports.figure_png(fig, dpi=60)
    assert png[:4] == b"\x89PNG"
    pdf = exports.report_pdf(jupiter, fig, [], planet_lines=["test"])
    assert pdf[:4] == b"%PDF"


# ─── Option photo ────────────────────────────────────────────────────────────

def _synthetic(snap, cal, shape=(600, 900)):
    import numpy as np
    img = np.random.default_rng(0).normal(100, 3, shape)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    cx, cy = cal.center
    img += 5000 * ((xx - cx) ** 2 + (yy - cy) ** 2 < 20 ** 2)
    for m in snap.moons:
        x, y = cal.sky_to_pixel(m.east_arcsec, m.north_arcsec)
        img += 800 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.5 ** 2))
    return img


@pytest.mark.parametrize("rotation, mirrored", [(0, False), (37, False), (200, True)])
def test_photo_calibration_roundtrip(jupiter, rotation, mirrored):
    truth = photo.Calibration((450.0, 300.0), 1.2, rotation, mirrored)
    img = _synthetic(jupiter, truth)
    ref = _moon(jupiter, "Ganymède")
    ref_px = photo.nearest_source(img, *truth.sky_to_pixel(ref.east_arcsec, ref.north_arcsec))
    center = photo.detect_planet_center(img)
    cal = photo.calibrate(center, ref_px, ref.east_arcsec, ref.north_arcsec, mirrored)
    assert cal.scale == pytest.approx(1.2, rel=0.01)
    assert (cal.rotation_deg - rotation) % 360 == pytest.approx(0, abs=0.5) or \
        (cal.rotation_deg - rotation) % 360 == pytest.approx(360, abs=0.5)
    for m in jupiter.moons:
        x, y = cal.sky_to_pixel(m.east_arcsec, m.north_arcsec)
        if 0 <= x < 900 and 0 <= y < 600:
            found = photo.nearest_source(img, x, y)
            assert found is not None
            assert math.hypot(found[0] - x, found[1] - y) < 1.0
