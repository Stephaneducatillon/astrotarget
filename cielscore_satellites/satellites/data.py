"""Données de référence : planètes, satellites, observatoires, instruments."""

from dataclasses import dataclass

import ephem

SIGNATURE = "CielScore Satellites — calculs CielScore (PyEphem)"

# Bornes de calcul annoncées par le cahier des charges.
YEAR_MIN, YEAR_MAX = 2025, 2060


@dataclass(frozen=True)
class Moon:
    key: str          # nom anglais (classe PyEphem)
    name: str         # nom français affiché
    cls: type         # classe PyEphem
    h_mag: float      # magnitude absolue V(1,0) (JPL, Planetary Satellite Physical Parameters)


@dataclass(frozen=True)
class Planet:
    key: str
    name: str
    cls: type
    flattening: float          # aplatissement (pck00011)
    pole_ra0: float            # pôle nord IAU, ICRF J2000 (degrés)
    pole_ra_rate: float        # degrés / siècle julien
    pole_dec0: float
    pole_dec_rate: float
    moons: tuple
    ring_outer: float = 0.0    # rayon extérieur des anneaux (rayons équatoriaux)


JUPITER = Planet(
    "Jupiter", "Jupiter", ephem.Jupiter, 0.06487,
    268.056595, -0.006499, 64.495303, 0.002413,
    (
        Moon("Io", "Io", ephem.Io, -1.68),
        Moon("Europa", "Europe", ephem.Europa, -1.41),
        Moon("Ganymede", "Ganymède", ephem.Ganymede, -2.09),
        Moon("Callisto", "Callisto", ephem.Callisto, -1.05),
    ),
)

SATURN = Planet(
    "Saturn", "Saturne", ephem.Saturn, 0.09796,
    40.589, -0.036, 83.537, -0.004,
    (
        Moon("Mimas", "Mimas", ephem.Mimas, 3.3),
        Moon("Enceladus", "Encelade", ephem.Enceladus, 2.2),
        Moon("Tethys", "Téthys", ephem.Tethys, 0.7),
        Moon("Dione", "Dioné", ephem.Dione, 0.8),
        Moon("Rhea", "Rhéa", ephem.Rhea, 0.1),
        Moon("Titan", "Titan", ephem.Titan, -1.2),
        Moon("Hyperion", "Hypérion", ephem.Hyperion, 4.6),
        Moon("Iapetus", "Japet", ephem.Iapetus, 1.6),
    ),
    ring_outer=136780.0 / 60268.0,
)

URANUS = Planet(
    "Uranus", "Uranus", ephem.Uranus, 0.02293,
    257.311, 0.0, -15.175, 0.0,
    (
        Moon("Miranda", "Miranda", ephem.Miranda, 3.6),
        Moon("Ariel", "Ariel", ephem.Ariel, 1.45),
        Moon("Umbriel", "Umbriel", ephem.Umbriel, 2.10),
        Moon("Titania", "Titania", ephem.Titania, 1.02),
        Moon("Oberon", "Obéron", ephem.Oberon, 1.23),
    ),
)

PLANETS = {p.name: p for p in (SATURN, JUPITER, URANUS)}


@dataclass(frozen=True)
class Site:
    name: str
    lat: float       # degrés, nord positif
    lon: float       # degrés, est positif
    elevation: float  # mètres


SITES = {
    "Calern — C2PU (OCA)": Site("Calern — C2PU (OCA)", 43.7536, 6.9222, 1270),
    "Cantin": Site("Cantin", 50.3089, 3.1269, 40),
    "Observatoire de Haute-Provence": Site("Observatoire de Haute-Provence", 43.9317, 5.7128, 650),
    "Pic du Midi": Site("Pic du Midi", 42.9364, 0.1425, 2877),
    "Paris — Observatoire": Site("Paris — Observatoire", 48.8363, 2.3363, 67),
}


@dataclass(frozen=True)
class Instrument:
    name: str
    focal_mm: float
    sensor_w_mm: float
    sensor_h_mm: float
    pixel_um: float

    @property
    def scale_arcsec_px(self) -> float:
        return 206.264806 * self.pixel_um / self.focal_mm

    @property
    def field_arcsec(self) -> tuple:
        k = 206264.806 / self.focal_mm
        return self.sensor_w_mm * k, self.sensor_h_mm * k


INSTRUMENTS = {
    "C2PU 1,04 m f/12,5 — APS-C 23,5×15,7 mm": Instrument(
        "C2PU 1,04 m f/12,5", 13000, 23.5, 15.7, 3.76),
    "C2PU 1,04 m f/12,5 — capteur 1\" 13,2×8,8 mm": Instrument(
        "C2PU 1,04 m f/12,5 (1\")", 13000, 13.2, 8.8, 2.4),
    "C14 f/11 — APS-C": Instrument("C14 f/11", 3910, 23.5, 15.7, 3.76),
    "Lunette 120 f/7,5 — APS-C": Instrument("Lunette 120 f/7,5", 900, 23.5, 15.7, 3.76),
}

# Orientations du schéma : (rotation en degrés, miroir gauche-droite).
# Référence « ciel » : nord en haut, est à gauche.
ORIENTATIONS = {
    "Ciel : nord en haut, est à gauche": (0.0, False),
    "Newton / sans renvoi : sud en haut, est à droite": (180.0, False),
    "Lunette ou SCT avec renvoi coudé : nord en haut, est à droite": (0.0, True),
    "Rotation libre": (None, False),
}

DEFAULT_TZ = "Europe/Paris"
