import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import requests
import re
import os
import time as tmod
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
import astropy.units as u
import ephem
from datetime import datetime, timezone, time as dtime
import pytz

# ─── 1. CATALOGUE MESSIER COMPLET (110 objets) ───────────────────────────────
CATALOGUE = pd.DataFrame([
    ("M1",   "Nébuleuse",   83.82,   22.01,  8.4),
    ("M2",   "Amas glob.",  323.36,  -0.82,  6.5),
    ("M3",   "Amas glob.",  205.55,  28.38,  6.2),
    ("M4",   "Amas glob.",  245.90, -26.53,  5.9),
    ("M5",   "Amas glob.",  229.64,   2.08,  5.7),
    ("M6",   "Amas ouv.",   265.08, -32.25,  4.2),
    ("M7",   "Amas ouv.",   268.46, -34.84,  3.3),
    ("M8",   "Nébuleuse",   270.92, -24.38,  6.0),
    ("M9",   "Amas glob.",  259.80, -18.52,  7.8),
    ("M10",  "Amas glob.",  254.29,  -4.10,  6.6),
    ("M11",  "Amas ouv.",   282.77,  -6.27,  6.3),
    ("M12",  "Amas glob.",  251.81,  -1.95,  6.7),
    ("M13",  "Amas glob.",  250.42,  36.46,  5.8),
    ("M14",  "Amas glob.",  264.40,  -3.25,  7.6),
    ("M15",  "Amas glob.",  322.49,  12.17,  6.3),
    ("M16",  "Nébuleuse",   274.70, -13.78,  6.4),
    ("M17",  "Nébuleuse",   275.20, -16.18,  6.0),
    ("M18",  "Amas ouv.",   275.08, -17.08,  7.5),
    ("M19",  "Amas glob.",  255.66, -26.27,  6.8),
    ("M20",  "Nébuleuse",   270.63, -23.03,  9.0),
    ("M21",  "Amas ouv.",   271.03, -22.50,  6.5),
    ("M22",  "Amas glob.",  279.10, -23.90,  5.1),
    ("M23",  "Amas ouv.",   269.23, -18.98,  6.9),
    ("M24",  "Amas ouv.",   274.16, -18.55,  4.5),
    ("M25",  "Amas ouv.",   277.94, -19.25,  4.6),
    ("M26",  "Amas ouv.",   281.37,  -9.40,  8.0),
    ("M27",  "Nébuleuse",   299.90,  22.72,  7.4),
    ("M28",  "Amas glob.",  276.14, -24.87,  6.9),
    ("M29",  "Amas ouv.",   305.00,  38.51,  7.1),
    ("M30",  "Amas glob.",  325.09, -23.18,  7.5),
    ("M31",  "Galaxie",      10.68,  41.27,  3.4),
    ("M32",  "Galaxie",      10.67,  40.87,  8.7),
    ("M33",  "Galaxie",      23.46,  30.66,  5.7),
    ("M34",  "Amas ouv.",    40.52,  42.78,  5.5),
    ("M35",  "Amas ouv.",    92.27,  24.33,  5.3),
    ("M36",  "Amas ouv.",    84.08,  34.13,  6.3),
    ("M37",  "Amas ouv.",    88.07,  32.55,  6.2),
    ("M38",  "Amas ouv.",    82.18,  35.83,  7.4),
    ("M39",  "Amas ouv.",   323.92,  48.43,  4.6),
    ("M40",  "Etoile dbl.", 185.56,  58.08,  8.4),
    ("M41",  "Amas ouv.",   101.50, -20.73,  4.5),
    ("M42",  "Nébuleuse",    83.82,  -5.39,  4.0),
    ("M43",  "Nébuleuse",    83.88,  -5.27,  9.0),
    ("M44",  "Amas ouv.",   130.05,  19.98,  3.7),
    ("M45",  "Amas ouv.",    56.75,  24.12,  1.6),
    ("M46",  "Amas ouv.",   115.45, -14.82,  6.0),
    ("M47",  "Amas ouv.",   114.15, -14.48,  4.3),
    ("M48",  "Amas ouv.",   123.42,  -5.80,  5.5),
    ("M49",  "Galaxie",     187.44,   8.00,  8.4),
    ("M50",  "Amas ouv.",   105.71,  -8.37,  6.3),
    ("M51",  "Galaxie",     202.47,  47.20,  8.4),
    ("M52",  "Amas ouv.",   351.19,  61.59,  7.3),
    ("M53",  "Amas glob.",  198.23,  18.17,  7.7),
    ("M54",  "Amas glob.",  283.76, -30.48,  7.7),
    ("M55",  "Amas glob.",  294.99, -30.96,  7.0),
    ("M56",  "Amas glob.",  289.15,  30.18,  8.3),
    ("M57",  "Nébuleuse",   283.40,  33.03,  8.8),
    ("M58",  "Galaxie",     189.43,  11.82,  9.7),
    ("M59",  "Galaxie",     190.51,  11.65, 10.6),
    ("M60",  "Galaxie",     190.92,  11.55,  8.8),
    ("M61",  "Galaxie",     185.48,   4.47,  9.7),
    ("M62",  "Amas glob.",  255.30, -30.11,  6.6),
    ("M63",  "Galaxie",     198.96,  42.03,  8.6),
    ("M64",  "Galaxie",     194.18,  21.68,  8.5),
    ("M65",  "Galaxie",     169.73,  13.09,  9.3),
    ("M66",  "Galaxie",     170.06,  12.99,  8.9),
    ("M67",  "Amas ouv.",   132.83,  11.82,  6.1),
    ("M68",  "Amas glob.",  189.87, -26.74,  8.2),
    ("M69",  "Amas glob.",  277.85, -32.35,  7.7),
    ("M70",  "Amas glob.",  280.80, -32.30,  8.1),
    ("M71",  "Amas glob.",  298.44,  18.78,  8.3),
    ("M72",  "Amas glob.",  313.36, -12.53,  9.4),
    ("M73",  "Amas ouv.",   314.74, -12.63,  9.0),
    ("M74",  "Galaxie",      24.17,  15.78,  9.4),
    ("M75",  "Amas glob.",  301.52, -21.92,  8.6),
    ("M76",  "Nébuleuse",    25.58,  51.57, 10.1),
    ("M77",  "Galaxie",      40.67,  -0.01,  8.9),
    ("M78",  "Nébuleuse",    86.68,   0.07,  8.3),
    ("M79",  "Amas glob.",   81.04, -24.52,  8.0),
    ("M80",  "Amas glob.",  244.26, -22.97,  7.3),
    ("M81",  "Galaxie",     148.89,  69.07,  6.9),
    ("M82",  "Galaxie",     148.97,  69.68,  8.4),
    ("M83",  "Galaxie",     204.25, -29.87,  7.5),
    ("M84",  "Galaxie",     186.27,  12.89,  9.3),
    ("M85",  "Galaxie",     186.35,  18.19,  9.2),
    ("M86",  "Galaxie",     186.55,  12.95,  9.2),
    ("M87",  "Galaxie",     187.71,  12.39,  8.6),
    ("M88",  "Galaxie",     187.99,  14.42,  9.6),
    ("M89",  "Galaxie",     188.92,  12.56, 10.7),
    ("M90",  "Galaxie",     188.87,  13.16,  9.5),
    ("M91",  "Galaxie",     189.21,  14.50, 10.2),
    ("M92",  "Amas glob.",  259.28,  43.13,  6.5),
    ("M93",  "Amas ouv.",   116.10, -23.85,  6.0),
    ("M94",  "Galaxie",     192.72,  41.12,  8.2),
    ("M95",  "Galaxie",     160.99,  11.70,  9.7),
    ("M96",  "Galaxie",     161.69,  11.82,  9.2),
    ("M97",  "Nébuleuse",   168.70,  55.02,  9.9),
    ("M98",  "Galaxie",     183.45,  14.90, 10.1),
    ("M99",  "Galaxie",     184.71,  14.42,  9.9),
    ("M100", "Galaxie",     185.73,  15.82,  9.4),
    ("M101", "Galaxie",     210.80,  54.35,  7.9),
    ("M102", "Galaxie",     226.62,  55.76, 10.0),
    ("M103", "Amas ouv.",    23.34,  60.70,  7.4),
    ("M104", "Galaxie",     190.00, -11.62,  8.0),
    ("M105", "Galaxie",     161.96,  12.58,  9.8),
    ("M106", "Galaxie",     184.74,  47.30,  8.4),
    ("M107", "Amas glob.",  248.13, -13.05,  8.1),
    ("M108", "Galaxie",     167.88,  55.67, 10.0),
    ("M109", "Galaxie",     179.40,  53.37, 10.6),
    ("M110", "Galaxie",      10.09,  41.68,  8.1),
], columns=["nom", "type", "ra", "dec", "magnitude"])

# ─── 1b. CATALOGUE CALDWELL (109 objets) ─────────────────────────────────────
CALDWELL = pd.DataFrame([
    ("C1",  "Amas ouv.",   22.02,  85.25,  7.0),
    ("C2",  "Nébuleuse",  355.40,  81.53, 11.0),
    ("C3",  "Galaxie",    357.72,  80.93,  9.6),
    ("C4",  "Nébuleuse",  104.87,  77.63, 11.4),
    ("C5",  "Galaxie",    150.57,  78.03,  9.2),
    ("C6",  "Amas ouv.",  102.67,  73.37,  7.1),
    ("C7",  "Galaxie",    040.67,  72.18,  8.4),
    ("C8",  "Nébuleuse",  221.46,  71.02,  8.8),
    ("C9",  "Galaxie",    147.43,  69.07,  8.4),
    ("C10", "Galaxie",    148.97,  69.68,  8.4),
    ("C11", "Amas ouv.",  168.63,  69.07,  6.5),
    ("C12", "Galaxie",    192.72,  41.12,  8.2),
    ("C13", "Amas ouv.",  170.90,  53.70,  6.4),
    ("C14", "Amas glob.", 252.85,  36.46,  7.9),
    ("C15", "Nébuleuse",  307.47,  44.53,  6.9),
    ("C16", "Amas ouv.",  357.63,  61.60,  6.4),
    ("C17", "Amas ouv.",   28.40,  60.65,  6.5),
    ("C18", "Nébuleuse",   17.01,  61.87,  8.0),
    ("C19", "Nébuleuse",   75.49,  58.08,  9.6),
    ("C20", "Nébuleuse",   83.82,  61.20,  7.3),
    ("C21", "Amas ouv.",   95.07,  60.65,  7.2),
    ("C22", "Amas glob.", 277.23,  43.13,  9.4),
    ("C23", "Amas ouv.",  113.65,  46.23,  7.1),
    ("C24", "Amas ouv.",  115.52,  49.87,  8.6),
    ("C25", "Amas ouv.",  124.33,  40.45,  8.0),
    ("C26", "Amas ouv.",  126.57,  38.32,  8.0),
    ("C27", "Amas ouv.",  129.07,  34.52,  6.4),
    ("C28", "Amas ouv.",  137.22,  32.53,  7.3),
    ("C29", "Amas ouv.",  138.27,  30.90,  6.6),
    ("C30", "Amas glob.", 244.26, -22.97,  7.5),
    ("C31", "Galaxie",   132.83,  11.82,  5.5),
    ("C32", "Nébuleuse",  224.90,  39.03, 10.0),
    ("C33", "Nébuleuse",  159.47,  55.02,  9.8),
    ("C34", "Amas ouv.",  196.15,  48.93,  5.0),
    ("C35", "Amas ouv.",  170.65,  53.37,  6.7),
    ("C36", "Nébuleuse",   92.05,  22.51,  9.9),
    ("C37", "Nébuleuse",   27.23,  57.87,  9.0),
    ("C38", "Nébuleuse",   97.23,  12.23, 10.8),
    ("C39", "Nébuleuse",  166.13,  55.02,  9.6),
    ("C40", "Nébuleuse",  291.35,  10.73,  8.4),
    ("C41", "Amas glob.", 283.77,  -1.01,  9.0),
    ("C42", "Nébuleuse",   83.82,  -5.39,  9.0),
    ("C43", "Nébuleuse",   86.83,  -5.27,  9.9),
    ("C44", "Amas ouv.",   21.23, -29.88,  4.5),
    ("C45", "Galaxie",    050.67, -25.30,  7.9),
    ("C46", "Amas ouv.",  114.38, -14.48,  4.2),
    ("C47", "Amas ouv.",  116.15, -14.82,  6.0),
    ("C48", "Amas ouv.",  118.40, -26.73,  5.8),
    ("C49", "Galaxie",   204.25, -29.87,  7.6),
    ("C50", "Amas glob.", 201.70, -26.52,  7.2),
    ("C51", "Amas glob.", 196.88, -26.59,  6.0),
    ("C52", "Amas glob.", 190.52, -26.59,  9.9),
    ("C53", "Amas glob.", 245.90, -26.53,  5.9),
    ("C54", "Amas glob.", 254.29,  -4.10,  7.6),
    ("C55", "Amas ouv.",  268.46, -34.84,  3.3),
    ("C56", "Amas ouv.",  265.08, -32.25,  4.2),
    ("C57", "Nébuleuse",  270.92, -24.38,  5.8),
    ("C58", "Amas glob.", 279.10, -23.90,  5.2),
    ("C59", "Amas glob.", 283.76, -30.48,  7.7),
    ("C60", "Galaxie",   190.92,  11.55,  8.8),
    ("C61", "Galaxie",   187.71,  12.39,  8.6),
    ("C62", "Amas glob.", 255.30, -30.11,  6.6),
    ("C63", "Nébuleuse",  300.20,  13.73, 10.8),
    ("C64", "Amas glob.", 312.11,  36.46,  9.2),
    ("C65", "Galaxie",   170.06,  13.09,  9.3),
    ("C66", "Galaxie",   169.73,  13.09,  9.0),
    ("C67", "Amas ouv.",  295.20,  17.53,  8.8),
    ("C68", "Nébuleuse",  299.90,  22.72,  8.3),
    ("C69", "Nébuleuse",  278.97, -26.93,  7.6),
    ("C70", "Galaxie",   040.87, -41.27,  9.1),
    ("C71", "Amas ouv.",  101.50, -20.73,  4.5),
    ("C72", "Amas glob.", 313.36, -12.53,  9.3),
    ("C73", "Amas glob.", 314.74, -12.63,  9.0),
    ("C74", "Amas glob.", 322.49,  12.17,  6.3),
    ("C75", "Amas glob.", 325.09, -23.18,  7.8),
    ("C76", "Amas glob.", 323.36,  -0.82,  6.5),
    ("C77", "Nébuleuse",  334.71, -30.57, 10.5),
    ("C78", "Amas glob.", 330.60, -36.62,  7.2),
    ("C79", "Amas glob.", 335.68, -44.10,  6.7),
    ("C80", "Amas glob.", 201.70, -47.48,  3.6),
    ("C81", "Amas glob.", 252.85, -36.59,  8.1),
    ("C82", "Amas ouv.",  255.66, -41.82,  5.2),
    ("C83", "Galaxie",   204.25, -29.87,  9.5),
    ("C84", "Nébuleuse",  239.18, -51.98,  9.6),
    ("C85", "Nébuleuse",  148.88, -59.52,  6.2),
    ("C86", "Amas ouv.",  161.15, -59.87,  5.1),
    ("C87", "Galaxie",   204.25, -65.35,  8.0),
    ("C88", "Amas ouv.",  195.57, -63.42,  7.9),
    ("C89", "Amas ouv.",  167.42, -60.07,  5.4),
    ("C90", "Nébuleuse",  168.62, -60.33,  9.7),
    ("C91", "Amas ouv.",  189.07, -63.82,  3.0),
    ("C92", "Nébuleuse",  252.38, -59.13,  6.2),
    ("C93", "Amas ouv.",  261.95, -38.53,  4.9),
    ("C94", "Amas glob.", 204.87, -65.45,  4.2),
    ("C95", "Amas glob.", 218.87, -48.07,  5.2),
    ("C96", "Amas ouv.",  223.38, -52.68,  3.8),
    ("C97", "Galaxie",   201.36, -43.02,  8.2),
    ("C98", "Amas ouv.",  219.17, -60.63,  7.3),
    ("C99", "Nébuleuse",  211.57, -63.43,  9.4),
    ("C100","Nébuleuse",  176.82, -65.43,  9.7),
    ("C101","Galaxie",   210.80, -54.35,  9.3),
    ("C102","Amas glob.", 211.48, -61.10,  3.7),
    ("C103","Nébuleuse",  264.32, -37.10,  9.0),
    ("C104","Amas glob.", 261.95, -48.07,  6.3),
    ("C105","Amas glob.", 254.42, -45.37,  8.4),
    ("C106","Amas glob.", 264.32, -48.57,  7.3),
    ("C107","Amas glob.", 248.13, -13.05,  8.1),
    ("C108","Amas glob.", 177.70, -64.50,  9.2),
    ("C109","Nébuleuse",  187.57, -59.12,  9.5),
], columns=["nom", "type", "ra", "dec", "magnitude"])

# ─── 1c. CHARGEMENT NGC/IC (OpenNGC CSV) ─────────────────────────────────────
@st.cache_data
def charger_ngc():
    """Charge le fichier NGC.csv d'OpenNGC si présent."""
    ngc_file = "NGC.csv"
    if not os.path.exists(ngc_file):
        return pd.DataFrame(columns=["nom", "type", "ra", "dec", "magnitude"])
    try:
        df = pd.read_csv(ngc_file, sep=";", low_memory=False)

        # Mapping types OpenNGC → types lisibles
        type_map = {
            "GX":  "Galaxie",
            "OC":  "Amas ouv.",
            "GC":  "Amas glob.",
            "EN":  "Nébuleuse",
            "RN":  "Nébuleuse",
            "PN":  "Nébuleuse",
            "SNR": "Nébuleuse",
            "NF":  "Nébuleuse",
            "BN":  "Nébuleuse",
            "DN":  "Nébuleuse",
            "HII": "Nébuleuse",
            "Cl+N":"Amas ouv.",
            "C+N": "Amas ouv.",
        }

        resultats = []
        for _, row in df.iterrows():
            try:
                nom  = str(row["Name"])
                typ  = type_map.get(str(row.get("Type", "")), "Autre")
                mag  = float(row.get("V-Mag", row.get("B-Mag", 99)))
                if pd.isna(mag):
                    mag = 15.0

                # Conversion RA HH:MM:SS → degrés
                ra_str  = str(row.get("RA", ""))
                dec_str = str(row.get("Dec", ""))
                if not ra_str or ra_str == "nan":
                    continue

                ra_parts  = ra_str.split(":")
                dec_parts = dec_str.split(":")
                if len(ra_parts) < 3 or len(dec_parts) < 3:
                    continue

                ra_deg  = (float(ra_parts[0]) +
                           float(ra_parts[1]) / 60 +
                           float(ra_parts[2]) / 3600) * 15

                signe   = -1 if dec_str.startswith("-") else 1
                dec_deg = signe * (abs(float(dec_parts[0])) +
                                   float(dec_parts[1]) / 60 +
                                   float(dec_parts[2]) / 3600)

                resultats.append((nom, typ, ra_deg, dec_deg, mag))
            except Exception:
                continue

        return pd.DataFrame(resultats,
                            columns=["nom", "type", "ra", "dec", "magnitude"])
    except Exception as e:
        st.warning(f"Erreur chargement NGC.csv : {e}")
        return pd.DataFrame(columns=["nom", "type", "ra", "dec", "magnitude"])


# ─── 2. PHASE DE LUNE ─────────────────────────────────────────────────────────
def get_moon_phase(dt):
    lune = ephem.Moon()
    lune.compute(dt.strftime("%Y/%m/%d %H:%M:%S"))
    return round(lune.phase, 1)

# ─── 3. MÉTÉO OPEN-METEO (gratuit, sans clé API) ──────────────────────────────
def get_meteo(lat, lng, dt):
    try:
        date_str  = dt.strftime("%Y-%m-%d")
        heure_utc = dt.hour
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lng}"
            f"&hourly=cloudcover,relativehumidity_2m,visibility,windspeed_10m"
            f"&start_date={date_str}&end_date={date_str}"
            f"&timezone=UTC"
        )
        response   = requests.get(url, timeout=5)
        data       = response.json()
        nuages     = data["hourly"]["cloudcover"][heure_utc]
        humidite   = data["hourly"]["relativehumidity_2m"][heure_utc]
        visibilite = data["hourly"]["visibility"][heure_utc] / 1000
        vent       = data["hourly"]["windspeed_10m"][heure_utc]
        # Seeing proxy : vent faible = bon seeing (index 1-5)
        # < 5 km/h → 5, 5-15 → 4, 15-25 → 3, 25-40 → 2, > 40 → 1
        if   vent <  5: seeing = 5
        elif vent < 15: seeing = 4
        elif vent < 25: seeing = 3
        elif vent < 40: seeing = 2
        else:           seeing = 1
        return {"nuages": nuages, "humidite": humidite,
                "visibilite": visibilite, "seeing": seeing,
                "vent": vent, "ok": True}
    except Exception:
        return {"nuages": 50, "humidite": 70,
                "visibilite": 20, "seeing": 3,
                "vent": 0, "ok": False}

# ─── 4. ALTITUDE ──────────────────────────────────────────────────────────────
def get_altitude(ra_deg, dec_deg, lat, lng, dt):
    location = EarthLocation(lat=lat * u.deg, lon=lng * u.deg)
    objet    = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
    frame    = AltAz(obstime=Time(dt), location=location)
    return float(objet.transform_to(frame).alt.deg)

def get_fenetre_visibilite(ra_deg, dec_deg, lat, lng, dt):
    """
    Calcule la durée en minutes pendant laquelle l'objet
    reste au-dessus de 30° d'altitude sur les 12 prochaines heures.
    """
    location = EarthLocation(lat=lat * u.deg, lon=lng * u.deg)
    objet    = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
    minutes_visibles = 0
    for delta_min in range(0, 720, 10):  # 12h par pas de 10 min
        t    = Time(dt) + delta_min * u.minute
        frame = AltAz(obstime=t, location=location)
        alt  = float(objet.transform_to(frame).alt.deg)
        if alt >= 30:
            minutes_visibles += 10
    return minutes_visibles


# ─── 5. MAGNITUDE LIMITE ──────────────────────────────────────────────────────
def mag_limite(diametre_mm):
    """Magnitude limite théorique de l'instrument seul."""
    if diametre_mm <= 0:
        return 6.0
    return round(2.1 + 5 * np.log10(diametre_mm), 1)

def mag_limite_reelle(diametre_mm, bortle):
    """
    Magnitude limite réelle = min(instrument, ciel).
    Formule calibrée sur valeurs Bortle reconnues :
    Bortle 3 (Prisches) → ciel 13.5 → instrument 114mm = 12.6 → limite 12.6
    Bortle 5            → ciel 12.1 → instrument 114mm = 12.6 → limite 12.1
    Bortle 7            → ciel 10.7 → instrument 114mm = 12.6 → limite 10.7
    Bortle 8 (Lille)    → ciel 10.0 → instrument 114mm = 12.6 → limite 10.0
    Bortle 9            → ciel  9.3 → instrument 114mm = 12.6 → limite  9.3
    """
    mag_instrument = mag_limite(diametre_mm)
    mag_ciel = round(13.5 - (bortle - 3) * 0.7, 1)
    return min(mag_instrument, mag_ciel)

# ─── 6. SCORE 0-100 ───────────────────────────────────────────────────────────
def calcule_score(altitude, magnitude, moon, bortle,
                  diametre, meteo, fenetre_minutes=120,
                  dist_lune_deg=90):
    """
    Score v2 — formule hybride proto + CDC.
    Intègre fenêtre de visibilité, seeing et distance angulaire lune.
    """
    mag_lim = mag_limite_reelle(diametre, bortle)

    # Éliminatoires
    if altitude < 5:
        return 0
    if magnitude > mag_lim:
        return 0
    if meteo["nuages"] > 90:
        return 0

    # 1. Altitude (30%)
    score_alt = np.clip((altitude - 5) / 25, 0, 1) * 100

    # 2. Fenêtre de visibilité (20%) — max référence = 4h (240 min)
    score_fenetre = min(fenetre_minutes / 240, 1) * 100

    # 3. Seeing (20%) — index 1-5 via proxy vent Open-Meteo
    seeing = meteo.get("seeing", 3)
    score_seeing = (seeing - 1) / 4 * 100

    # 4. Transparence atmosphérique (15%)
    score_transp = 100 - meteo["nuages"]

    # 5. Bortle — pollution lumineuse (10%)
    score_bortle = (9 - bortle) / 8 * 100

    # 6. Lune (5%) — phase + distance angulaire
    # Plus la lune est pleine ET proche, plus elle dégrade
    score_lune = (1 - (moon / 100) * (1 - dist_lune_deg / 180)) * 100

    return round(
        score_alt     * 0.30 +
        score_fenetre * 0.20 +
        score_seeing  * 0.20 +
        score_transp  * 0.15 +
        score_bortle  * 0.10 +
        score_lune    * 0.05, 1
    )


# ─── 7. PLANÈTES (PyEphem temps réel) ────────────────────────────────────────
def get_planetes(lat, lng, dt):
    obs      = ephem.Observer()
    obs.lat  = str(lat)
    obs.lon  = str(lng)
    obs.date = dt.strftime("%Y/%m/%d %H:%M:%S")

    planetes_ephem = {
        "Mercure": ephem.Mercury(),
        "Vénus":   ephem.Venus(),
        "Mars":    ephem.Mars(),
        "Jupiter": ephem.Jupiter(),
        "Saturne": ephem.Saturn(),
        "Uranus":  ephem.Uranus(),
        "Neptune": ephem.Neptune(),
        "Lune":    ephem.Moon(),
    }

    resultats = []
    for nom, corps in planetes_ephem.items():
        corps.compute(obs)
        altitude  = float(corps.alt) * 180 / 3.14159265
        magnitude = float(corps.mag)
        resultats.append({
            "nom":       nom,
            "type":      "Planète",
            "altitude":  round(altitude, 1),
            "magnitude": round(magnitude, 1),
        })
    return pd.DataFrame(resultats)

# ─── 8. CARNET D'OBSERVATIONS (session_state) ────────────────────────────────
CARNET_COLS = ["date", "heure_locale", "site", "objet", "type",
               "instrument", "grossissement", "altitude",
               "conditions", "note", "commentaire"]

def init_carnet():
    """Initialise le carnet en session_state si pas encore fait."""
    if "carnet" not in st.session_state:
        st.session_state.carnet = []

def ajouter_observation(obs):
    """Ajoute une observation dans la session."""
    init_carnet()
    st.session_state.carnet.append(obs)

def get_carnet_df():
    """Retourne le carnet comme DataFrame."""
    init_carnet()
    if st.session_state.carnet:
        return pd.DataFrame(st.session_state.carnet, columns=CARNET_COLS)
    return pd.DataFrame(columns=CARNET_COLS)

# ─── 9. INTERFACE STREAMLIT ───────────────────────────────────────────────────
st.set_page_config(page_title="AstroTarget", page_icon="🔭", layout="wide")
st.title("🔭 AstroTarget")
st.caption("Planificateur d'observation astronomique personnalisé")
st.divider()

# ── Sélection catalogue ──────────────────────────────────────────────────────
st.subheader("📚 Catalogues")
col_cat1, col_cat2, col_cat3, col_cat4 = st.columns(4)
with col_cat1:
    use_messier  = st.checkbox("Messier (110)",  value=True)
with col_cat2:
    use_caldwell = st.checkbox("Caldwell (109)", value=False)
with col_cat3:
    use_ngc      = st.checkbox("NGC / IC",       value=False)
with col_cat4:
    use_planetes = st.checkbox("🪐 Planètes",    value=True)

if use_ngc:
    ngc_data = charger_ngc()
    if ngc_data.empty:
        st.warning("⚠️ NGC.csv introuvable dans le dossier du projet")
    else:
        st.caption(f"✅ {len(ngc_data)} objets NGC/IC chargés")

st.divider()

col1, col2, col3 = st.columns(3)

# ── Lieu ──────────────────────────────────────────────────────────────────────
with col1:
    st.subheader("📍 Lieu d'observation")
    mode_lieu = st.radio("Mode de saisie",
                         ["Commune (liste)", "GPS manuel"],
                         horizontal=True)

    if mode_lieu == "Commune (liste)":
        communes = {
            "⭐ Planétarium Orionis (Douai)":                (50.3321, 3.1156, 6),
            "⭐ Sémaphore de Cantin":                        (50.2999, 3.1166, 4),
            "⭐ Observatoire Charles Fehrenbach (Prisches)": (50.0791, 3.7702, 3),
            "Douai (centre)":                                (50.3700, 3.0800, 7),
            "Lille":                                         (50.6292, 3.0573, 8),
            "Arras":                                         (50.2924, 2.7812, 7),
            "Valenciennes":                                  (50.3573, 3.5235, 7),
            "Cambrai":                                       (50.1764, 3.2364, 6),
            "Lens":                                          (50.4329, 2.8314, 8),
            "Béthune":                                       (50.5300, 2.6400, 7),
            "Dunkerque":                                     (51.0343, 2.3767, 8),
            "Calais":                                        (50.9513, 1.8587, 7),
            "Boulogne-sur-Mer":                              (50.7264, 1.6147, 6),
            "Amiens":                                        (49.8942, 2.2958, 7),
            "Baie de Somme":                                 (50.1800, 1.6000, 3),
            "Forêt de Mormal":                               (50.2200, 3.8300, 4),
        }
        commune = st.selectbox("Commune / Site", list(communes.keys()))
        lat, lng, bortle = communes[commune]
        col_gps, col_bort = st.columns(2)
        col_gps.caption(f"📌 {lat}°N · {lng}°E")
        if bortle <= 3:
            col_bort.success(f"🌑 Bortle {bortle} — Ciel excellent")
        elif bortle <= 5:
            col_bort.info(f"🌒 Bortle {bortle} — Ciel correct")
        elif bortle <= 7:
            col_bort.warning(f"🌔 Bortle {bortle} — Pollution modérée")
        else:
            col_bort.error(f"🌕 Bortle {bortle} — Forte pollution")
    else:
        lat    = st.number_input("Latitude (°N)",  value=50.37,
                                 min_value=-90.0,  max_value=90.0,
                                 step=0.0001, format="%.4f")
        lng    = st.number_input("Longitude (°E)", value=3.08,
                                 min_value=-180.0, max_value=180.0,
                                 step=0.0001, format="%.4f")
        bortle = st.slider("Indice Bortle", min_value=1, max_value=9,
                           value=7, help="1=ciel vierge, 9=centre-ville")
        commune = f"{lat}°N {lng}°E"

# ── Date et heure ─────────────────────────────────────────────────────────────
with col2:
    st.subheader("📅 Date et heure")
    date_obs     = st.date_input("Date d'observation",
                                 value=datetime.now().date())
    heure_saisie = st.text_input(
        "Heure locale FR (HH:MM)",
        value=datetime.now().strftime("%H:%M"),
        max_chars=5,
        placeholder="ex: 21:30"
    )

    heure_valide = bool(re.match(r"^\d{2}:\d{2}$", heure_saisie))
    if heure_valide:
        heure  = int(heure_saisie.split(":")[0])
        minute = int(heure_saisie.split(":")[1])
        if 0 <= heure <= 23 and 0 <= minute <= 59:
            tz_france = pytz.timezone("Europe/Paris")
            dt_local  = tz_france.localize(
                datetime(date_obs.year, date_obs.month, date_obs.day,
                         heure, minute, 0))
            dt_utc    = dt_local.astimezone(pytz.utc)
            heure_obs = dt_utc.time()
            decalage  = int(dt_local.utcoffset().total_seconds() // 3600)
            st.caption(f"🕐 Tu saisis l'heure de Paris/Douai : "
                       f"{heure:02d}h{minute:02d} "
                       f"— UTC : {dt_utc.hour:02d}h{dt_utc.minute:02d} "
                       f"(UTC+{decalage})")
        else:
            st.error("Heure invalide — ex: 21:30")
            st.stop()

    else:
        st.warning("Format attendu : HH:MM — ex: 21:30")
        st.stop()

# ── Instrument ────────────────────────────────────────────────────────────────
with col3:
    st.subheader("🔭 Instrument")

    # Mode visuel ou astrophoto
    col_mode1, col_mode2 = st.columns(2)
    with col_mode1:
        mode_visuel     = st.checkbox("👁️ Visuel",      value=True)
    with col_mode2:
        mode_astrophoto = st.checkbox("📷 Astrophoto",  value=False)

    type_instrument = st.selectbox("Type d'instrument",
                                   ["Télescope", "Jumelles",
                                    "Lunette",   "Oeil nu"])

    if type_instrument in ["Télescope", "Lunette"]:
        diametre = st.number_input("Diamètre (mm)",
                                   min_value=30, max_value=600,
                                   value=114, step=5)
        focale   = st.number_input("Focale (mm)",
                                   min_value=300, max_value=3000,
                                   value=900, step=50)
        mag_lim  = mag_limite(diametre)

        # ── MODE VISUEL ───────────────────────────────────────
        if mode_visuel:
            st.markdown("**👁️ Visuel**")
            barlow_choix_v = st.selectbox("Barlow visuel",
                                          ["Sans Barlow", "Barlow ×2",
                                           "Barlow ×3",   "Barlow ×5"],
                                          key="barlow_visuel")
            barlow_map   = {"Sans Barlow": 1, "Barlow ×2": 2,
                            "Barlow ×3": 3,   "Barlow ×5": 5}
            barlow_val_v = barlow_map[barlow_choix_v]
            focale_eff_v = focale * barlow_val_v
            fd_eff_v     = focale_eff_v / diametre

            oculaire      = st.number_input("Oculaire (mm)",
                                            min_value=4, max_value=40,
                                            value=25, step=1)
            grossissement = round(focale_eff_v / oculaire, 0)
            gros_mini     = round(diametre / 7, 0)
            gros_maxi     = round(diametre * 1.5, 0)
            pouv_sep      = round(116 / diametre, 2)

            col_v1, col_v2 = st.columns(2)
            col_v1.metric("Grossissement",     f"×{int(grossissement)}")
            col_v2.metric("F/D effectif",       f"f/{fd_eff_v:.1f}")
            col_v1.metric("Gros. mini/maxi",   f"×{int(gros_mini)} / ×{int(gros_maxi)}")
            col_v2.metric("Pouvoir séparateur", str(pouv_sep) + '"')
            col_v1.metric("Magnitude limite",   f"{mag_lim}")

            mag_reelle = mag_limite_reelle(diametre, bortle)
            if mag_reelle < mag_lim:
                st.warning(f"⚠️ Ciel limitant (Bortle {bortle}) : "
                           f"mag. accessible réduite à **{mag_reelle}** "
                           f"(au lieu de {mag_lim} en ciel noir)")
            else:
                st.success(f"✅ Ciel non limitant (Bortle {bortle}) : "
                           f"mag. limite = {mag_lim}")
        else:
            grossissement = 0
            focale_eff_v  = focale
            barlow_val_v  = 1

        # ── MODE ASTROPHOTO ───────────────────────────────────
        if mode_astrophoto:
            st.markdown("**📷 Astrophoto**")

            # Barlow astrophoto indépendante
            barlow_choix_a = st.selectbox("Barlow astrophoto",
                                          ["Sans Barlow", "Barlow ×2",
                                           "Barlow ×3",   "Barlow ×5"],
                                          key="barlow_astro")
            barlow_map_a  = {"Sans Barlow": 1, "Barlow ×2": 2,
                             "Barlow ×3": 3,   "Barlow ×5": 5}
            barlow_val_a  = barlow_map_a[barlow_choix_a]
            focale_eff_a  = focale * barlow_val_a
            fd_eff_a      = focale_eff_a / diametre

            taille_pixel = st.number_input("Taille pixel (µm)",
                                           min_value=1.0, max_value=20.0,
                                           value=2.9, step=0.1,
                                           format="%.1f")

            # Planétaire
            st.markdown("*Planétaire (Shannon)*")
            fd_mini  = round(taille_pixel * 3.5, 1)
            fd_ideal = round(taille_pixel * 5.0, 1)
            fd_maxi  = round(taille_pixel * 8.0, 1)

            col_p1, col_p2 = st.columns(2)
            col_p1.metric("F/D réel",    f"f/{fd_eff_a:.1f}")
            col_p2.metric("F/D minimal", f"f/{fd_mini}")
            col_p1.metric("F/D idéal",   f"f/{fd_ideal}")
            col_p2.metric("F/D maximal", f"f/{fd_maxi}")

            if fd_eff_a < fd_mini:
                st.error("⚠️ F/D trop court — sous-échantillonnage.")
            elif fd_eff_a <= fd_ideal:
                st.success("✅ F/D dans la plage optimale")
            elif fd_eff_a <= fd_maxi:
                st.info("🔵 F/D dans la plage acceptable")
            else:
                st.warning("⚠️ F/D trop long — sur-échantillonnage")

            # Ciel profond
            st.markdown("*Ciel profond*")
            col_cp1, col_cp2 = st.columns(2)
            with col_cp1:
                px_largeur = st.number_input("Pixels largeur",
                                             min_value=100, max_value=10000,
                                             value=4000, step=100)
            with col_cp2:
                px_hauteur = st.number_input("Pixels hauteur",
                                             min_value=100, max_value=10000,
                                             value=3000, step=100)

            echantillonnage = round(
                (taille_pixel / focale_eff_a) * 206.265, 2)
            champ_larg = round(
                (px_largeur * taille_pixel / 1000 / focale_eff_a)
                * (180 / 3.14159) * 60, 1)
            champ_haut = round(
                (px_hauteur * taille_pixel / 1000 / focale_eff_a)
                * (180 / 3.14159) * 60, 1)
            capteur_larg = round(px_largeur * taille_pixel / 1000, 1)
            capteur_haut = round(px_hauteur * taille_pixel / 1000, 1)

            col_r1, col_r2 = st.columns(2)
            col_r1.metric("Capteur",
                          f"{capteur_larg}×{capteur_haut} mm")
            col_r2.metric("Échantillonnage",
                          str(echantillonnage) + '"/px')
            col_r1.metric("Champ réel",
                          str(champ_larg) + "'×" + str(champ_haut) + "'")
            col_r2.metric("Focale effective",
                          f"{focale_eff_a} mm")

            if echantillonnage < 0.5:
                st.warning('⚠️ Sur-échantillonnage (< 0.5"/px)')
            elif echantillonnage <= 2.0:
                st.success(f'✅ Échantillonnage optimal ({echantillonnage}"/px)')
            else:
                st.info(f'🔵 Sous-échantillonnage — acceptable pour grands champs ({echantillonnage}"/px)')

            mag_reelle_a = mag_limite_reelle(diametre, bortle)
            if mag_reelle_a < mag_lim:
                st.warning(f"⚠️ Ciel limitant (Bortle {bortle}) : "
                           f"mag. accessible réduite à **{mag_reelle_a}**")

    elif type_instrument == "Jumelles":
        diametre      = st.number_input("Diamètre objectif (mm)",
                                        min_value=30, max_value=100,
                                        value=50, step=5)
        focale        = 0
        grossissement = 0
        barlow_val_v  = 1
        mag_lim       = mag_limite(diametre)

        mag_reelle = mag_limite_reelle(diametre, bortle)
        if mag_reelle < mag_lim:
            st.warning(f"⚠️ Ciel limitant (Bortle {bortle}) : "
                       f"mag. accessible réduite à **{mag_reelle}** "
                       f"(au lieu de {mag_lim} en ciel noir)")
        else:
            st.success(f"✅ Ciel non limitant : mag. limite = {mag_lim}")
        st.metric("Magnitude limite", f"{mag_lim}")

    else:
        # Oeil nu
        diametre      = 7
        focale        = 0
        grossissement = 0
        barlow_val_v  = 1
        mag_lim       = 6.0
        st.metric("Magnitude limite", "6.0")
        mag_reelle = mag_limite_reelle(diametre, bortle)
        if mag_reelle < mag_lim:
            st.warning(f"⚠️ Ciel limitant (Bortle {bortle}) : "
                       f"mag. accessible réduite à **{mag_reelle}**")
        else:
            st.success(f"✅ Ciel non limitant : mag. limite = {mag_lim}")

st.divider()

# ── Bouton de calcul ──────────────────────────────────────────────────────────
if st.button("🚀 Calculer le Top 10 ce soir", type="primary",
             use_container_width=True):
    st.session_state.calcul_fait = False  # Force recalcul

    dt   = datetime.combine(date_obs, heure_obs).replace(tzinfo=timezone.utc)
    moon = get_moon_phase(dt)

    with st.spinner("Récupération météo en cours..."):
        meteo = get_meteo(lat, lng, dt)

    # ── Conditions du soir ────────────────────────────────────
    st.subheader("🌤️ Conditions du soir")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    seeing_labels = {1: "⭐ Très mauvais", 2: "⭐⭐ Mauvais",
                     3: "⭐⭐⭐ Correct", 4: "⭐⭐⭐⭐ Bon",
                     5: "⭐⭐⭐⭐⭐ Excellent"}
    col_m1.metric("🌙 Lune",        f"{moon}%")
    col_m2.metric("☁️ Nuages",      f"{meteo['nuages']}%"
                  if meteo["ok"] else "N/A")
    col_m3.metric("🌬️ Seeing",      seeing_labels.get(meteo.get("seeing", 3), "N/A")
                  if meteo["ok"] else "N/A")
    col_m4.metric("👁️ Visibilité",  f"{meteo['visibilite']:.0f} km"
                  if meteo["ok"] else "N/A")

    if not meteo["ok"]:
        st.warning("⚠️ Météo indisponible — score calculé sans données météo réelles")
    elif meteo["nuages"] > 90:
        st.error("🚫 Ciel bouché — observation impossible ce soir")
    elif meteo["nuages"] > 60:
        st.warning("⚠️ Ciel partiellement nuageux — conditions dégradées")
    else:
        st.success("✅ Bonnes conditions météo pour observer")

    st.divider()

    # ── Calcul scores Messier ─────────────────────────────────
    # Construction du catalogue actif selon sélection
    catalogue_actif = pd.DataFrame(columns=["nom", "type", "ra", "dec", "magnitude"])
    if use_messier:
        catalogue_actif = pd.concat([catalogue_actif, CATALOGUE], ignore_index=True)
    if use_caldwell:
        catalogue_actif = pd.concat([catalogue_actif, CALDWELL], ignore_index=True)
    if use_ngc:
        ngc_data = charger_ngc()
        if not ngc_data.empty:
            ngc_filtre = ngc_data[ngc_data["magnitude"] <= mag_lim + 0.5]
            catalogue_actif = pd.concat([catalogue_actif, ngc_filtre], ignore_index=True)

    # Ajout des planètes si cochées
    if use_planetes:
        df_planetes = get_planetes(lat, lng, dt)
        planetes_df = pd.DataFrame({
            "nom":       df_planetes["nom"],
            "type":      df_planetes["type"],
            "ra":        0.0,
            "dec":       0.0,
            "magnitude": df_planetes["magnitude"],
            "altitude":  df_planetes["altitude"],
            "est_planete": True
        })
    else:
        planetes_df = pd.DataFrame()

    if catalogue_actif.empty and not use_planetes:
        st.warning("⚠️ Sélectionne au moins un catalogue.")
        st.stop()

    nb_objets = len(catalogue_actif)
    with st.spinner(f"Calcul des scores pour {nb_objets} objets en cours..."):
        resultats = []
        for _, row in catalogue_actif.iterrows():
            # Altitude pré-calculée pour les planètes, calculée pour les autres
            if "altitude" in row and row.get("est_planete", False):
                alt = row["altitude"]
            else:
                alt = get_altitude(row["ra"], row["dec"], lat, lng, dt)

            # Score planète simplifié (pas de mag limite instrument)
            if row.get("est_planete", False):
                if alt < 5 or meteo["nuages"] > 90:
                    score = 0
                else:
                    score_alt    = np.clip((alt - 5) / 25, 0, 1) * 100
                    score_nuit   = 100 - meteo["nuages"]
                    score_lune_p = 100 - moon if row["nom"] != "Lune" else 100
                    score = round(score_alt * 0.60 +
                                  score_nuit * 0.25 +
                                  score_lune_p * 0.15, 1)
            else:
                # Calcul fenêtre de visibilité pour score v2
                fenetre = get_fenetre_visibilite(
                    row["ra"], row["dec"], lat, lng, dt)
                # Distance angulaire lune
                lune_ephem = ephem.Moon()
                lune_ephem.compute(dt.strftime("%Y/%m/%d %H:%M:%S"))
                lune_ra  = float(lune_ephem.ra) * 180 / 3.14159
                lune_dec = float(lune_ephem.dec) * 180 / 3.14159
                dist_lune = float(np.sqrt(
                    (row["ra"] - lune_ra)**2 +
                    (row["dec"] - lune_dec)**2))
                dist_lune = min(dist_lune, 180)

                score = calcule_score(
                    alt, row["magnitude"], moon, bortle,
                    diametre, meteo,
                    fenetre_minutes=fenetre,
                    dist_lune_deg=dist_lune)

            resultats.append({
                "Objet":        row["nom"],
                "Type":         row["type"],
                "Score":        score,
                "Altitude (°)": round(alt, 1),
                "Magnitude":    row["magnitude"],
                "Observable":   "✅" if score > 0 else "❌"
            })

        # Ajout planètes si cochées
        if use_planetes and not planetes_df.empty:
            for _, p in planetes_df.iterrows():
                alt = p["altitude"]
                if alt < 5 or meteo["nuages"] > 90:
                    score_p = 0
                else:
                    score_alt    = np.clip((alt - 5) / 25, 0, 1) * 100
                    score_nuit   = 100 - meteo["nuages"]
                    score_lune_p = 100 - moon if p["nom"] != "Lune" else 100
                    score_p      = round(score_alt * 0.60 +
                                         score_nuit * 0.25 +
                                         score_lune_p * 0.15, 1)
                resultats.append({
                    "Objet":        p["nom"],
                    "Type":         "🪐 Planète",
                    "Score":        score_p,
                    "Altitude (°)": round(alt, 1),
                    "Magnitude":    p["magnitude"],
                    "Observable":   "✅" if score_p > 0 else "❌"
                })

    df = pd.DataFrame(resultats).sort_values("Score", ascending=False)
    df_obs = df[df["Observable"] == "✅"].reset_index(drop=True)

    # Sauvegarde résultats pour affichage persistant
    st.session_state.calcul_fait     = True
    st.session_state.df_obs          = df_obs.copy()
    st.session_state.df_all          = df.copy()
    st.session_state.catalogue_actif = catalogue_actif.copy()
    st.session_state.moon_saved      = moon
    st.session_state.meteo_saved     = meteo

    st.rerun()

# ── Affichage résultats (persistant via session_state) ────────────────────────
if st.session_state.get("calcul_fait", False):
    df_obs          = st.session_state.df_obs
    df              = st.session_state.df_all
    catalogue_actif = st.session_state.catalogue_actif
    moon            = st.session_state.get("moon_saved", 0)
    meteo           = st.session_state.get("meteo_saved",
                          {"nuages": 0, "humidite": 0,
                           "visibilite": 40, "seeing": 3, "ok": False})

    st.subheader("🏆 Meilleurs objets ce soir")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Objets observables", len(df_obs))
    col_b.metric("Meilleur score",
                 f"{df_obs['Score'].max():.1f}/100" if len(df_obs) > 0 else "—")
    col_c.metric("Meilleur objet",
                 df_obs.iloc[0]["Objet"] if len(df_obs) > 0 else "—")

    # Sélecteur objet pour la carte
    objets_liste = df_obs["Objet"].tolist() if len(df_obs) > 0 else []
    objet_selec  = st.selectbox(
        "🔭 Sélectionner un objet pour afficher sa carte du ciel :",
        objets_liste,
        index=0 if objets_liste else None
    )

    # Colonnes : tableau à gauche, carte à droite
    col_tab, col_carte = st.columns([1, 1])

    with col_tab:
        st.dataframe(df_obs, use_container_width=True, hide_index=True)
        st.caption(f"{len(df_obs)} objets observables ce soir")

    with col_carte:
        if objet_selec:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from io import BytesIO

            # Infos score depuis df
            row_score = df[df["Objet"] == objet_selec]
            mag_sel   = float(row_score.iloc[0]["Magnitude"]) if not row_score.empty else 0.0
            type_sel  = str(row_score.iloc[0]["Type"])        if not row_score.empty else "—"
            score_sel = float(row_score.iloc[0]["Score"])     if not row_score.empty else 0.0

            # Coordonnées depuis catalogue_actif
            row_cat = catalogue_actif[catalogue_actif["nom"] == objet_selec]
            est_planete = (row_cat.empty or
                           type_sel == "🪐 Planète" or
                           "est_planete" in row_cat.columns and
                           row_cat.iloc[0].get("est_planete", False))

            if not est_planete and not row_cat.empty:
                ra_sel  = float(row_cat.iloc[0]["ra"])
                dec_sel = float(row_cat.iloc[0]["dec"])
            else:
                ra_sel, dec_sel = 0.0, 0.0
                est_planete = True

            # Fiche objet
            st.markdown(f"### 🔭 {objet_selec}")
            st.markdown(
                f"**Type :** {type_sel}  \n"
                f"**Magnitude :** {mag_sel}  \n"
                f"**Score ce soir :** {score_sel}/100  \n"
                f"**RA :** {ra_sel:.2f}°  ·  **Dec :** {dec_sel:+.2f}°"
            )

            if not est_planete:
                with st.spinner("Génération de la carte..."):
                    rayon = 8

                    fig, ax = plt.subplots(1, 1, figsize=(6, 6),
                                           facecolor='#0a0a1a')
                    ax.set_facecolor('#0a0a1a')

                    # Étoiles simulées réalistes autour de l'objet
                    np.random.seed(int(abs(ra_sel * 100 + dec_sel * 10)))
                    n = 80
                    ra_s   = ra_sel  + np.random.normal(0, rayon/2, n)
                    dec_s  = dec_sel + np.random.normal(0, rayon/2, n)
                    mag_s  = np.random.exponential(1.8, n) + 2.5
                    sizes  = np.clip((9 - mag_s) ** 2.3, 1, 180)
                    ax.scatter(ra_s, dec_s, s=sizes, c='white',
                               alpha=0.8, zorder=2)

                    # Couleur selon type objet
                    couleurs = {
                        "Galaxie":    '#FF6B6B',
                        "Nébuleuse":  '#6BFFB8',
                        "Amas ouv.":  '#FFD93D',
                        "Amas glob.": '#FFD93D',
                    }
                    couleur = couleurs.get(type_sel, '#FF8C00')

                    # Marqueur objet
                    ax.scatter([ra_sel], [dec_sel], s=280,
                               c=couleur, marker='*', zorder=6)
                    circle = plt.Circle((ra_sel, dec_sel),
                                        rayon * 0.09, fill=False,
                                        color=couleur, linewidth=2,
                                        zorder=5)
                    ax.add_patch(circle)
                    ax.annotate(objet_selec, (ra_sel, dec_sel),
                                textcoords="offset points",
                                xytext=(10, 8), fontsize=11,
                                color=couleur, fontweight='bold')

                    # Style
                    ax.set_xlim(ra_sel + rayon, ra_sel - rayon)
                    ax.set_ylim(dec_sel - rayon, dec_sel + rayon)
                    ax.set_xlabel("Ascension droite (°)",
                                  color='#aaaaaa', fontsize=9)
                    ax.set_ylabel("Déclinaison (°)",
                                  color='#aaaaaa', fontsize=9)
                    ax.tick_params(colors='#aaaaaa', labelsize=8)
                    for spine in ax.spines.values():
                        spine.set_color('#333333')
                    ax.set_title(f"Carte du ciel — {objet_selec}",
                                 color='white', fontsize=12, pad=8)
                    ax.grid(True, color='#222233',
                            linewidth=0.5, alpha=0.5)

                    # Légende
                    for mag_l, label in [(3, "Étoile brillante"),
                                         (6, "Étoile faible")]:
                        ax.scatter([], [], s=(9-mag_l)**2.3,
                                   c='white', alpha=0.8, label=label)
                    ax.scatter([], [], s=280, c=couleur,
                               marker='*', label=objet_selec)
                    ax.legend(facecolor='#0a0a1a', labelcolor='#aaaaaa',
                              loc='lower right', fontsize=8,
                              framealpha=0.7)

                    plt.tight_layout()
                    buf = BytesIO()
                    plt.savefig(buf, format='png', dpi=110,
                                bbox_inches='tight',
                                facecolor='#0a0a1a')
                    buf.seek(0)
                    plt.close()

                st.image(buf, caption=f"Carte locale — {objet_selec}",
                         use_container_width=True)
                st.caption("⚠️ Étoiles de fond simulées — "
                           "positions relatives correctes")

                # Lien SIMBAD
                url_simbad = (
                    f"https://simbad.cds.unistra.fr/simbad/sim-id"
                    f"?Ident={objet_selec.replace(' ', '+')}"
                )
                st.link_button("🔬 Fiche SIMBAD",
                               url_simbad,
                               use_container_width=True)
            else:
                st.info("🪐 Position en temps réel — "
                        "pas de carte fixe pour les planètes.")

    st.divider()

    # ── Tous les objets du ciel ───────────────────────────────
    st.subheader("📋 Objets du ciel")
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()

    # ── Carnet d'observations ─────────────────────────────────
    st.subheader("📓 Carnet d'observations")

    with st.form("form_observation", clear_on_submit=True):
        st.markdown("**Enregistrer une observation**")
        col_f1, col_f2, col_f3 = st.columns(3)

        with col_f1:
            objet_obs = st.text_input(
                "Objet observé",
                placeholder="ex: M42, Jupiter, M31...")
            type_obs  = st.selectbox(
                "Type",
                ["Nébuleuse", "Galaxie", "Amas ouv.",
                 "Amas glob.", "Planète", "Autre"])

        with col_f2:
            conditions_obs = st.selectbox(
                "Conditions",
                ["⭐⭐⭐⭐⭐ Excellentes",
                 "⭐⭐⭐⭐ Bonnes",
                 "⭐⭐⭐ Correctes",
                 "⭐⭐ Mauvaises",
                 "⭐ Très mauvaises"])
            note_obs = st.slider(
                "Note personnelle", 1, 5, 3,
                help="1=décevant, 5=exceptionnel")

        with col_f3:
            commentaire_obs = st.text_area(
                "Commentaire",
                placeholder="Ce que tu as vu, détails...",
                height=100)

        submitted = st.form_submit_button(
            "💾 Enregistrer l'observation",
            use_container_width=True)

        if submitted and objet_obs:
            gros = f"×{int(grossissement)}" \
                   if type_instrument == "Télescope" else "—"

            alt_obs     = "—"
            objet_upper = objet_obs.upper().strip()
            match       = df[df["Objet"] == objet_upper]
            if not match.empty:
                alt_obs = str(match.iloc[0]["Altitude (°)"])

            site_label = commune if mode_lieu == "Commune (liste)" \
                         else f"{lat}°N {lng}°E"

            obs = {
                "date":          date_obs.strftime("%d/%m/%Y"),
                "heure_locale":  heure_saisie,
                "site":          site_label,
                "objet":         objet_obs,
                "type":          type_obs,
                "instrument":    f"{type_instrument} {diametre}mm",
                "grossissement": gros,
                "altitude":      alt_obs,
                "conditions":    conditions_obs,
                "note":          note_obs,
                "commentaire":   commentaire_obs
            }
            ajouter_observation(obs)
            st.success(f"✅ {objet_obs} enregistré — {len(st.session_state.carnet)} obs. ce soir")

    # ── Historique et export ──────────────────────────────────
    carnet = get_carnet_df()
    if not carnet.empty:
        st.markdown(f"**{len(carnet)} observation(s) cette session**")
        st.dataframe(carnet, use_container_width=True, hide_index=True)
        csv_data = carnet.to_csv(index=False, encoding="utf-8")
        st.download_button(
            label="⬇️ Télécharger mon carnet (CSV)",
            data=csv_data,
            file_name=f"carnet_astrotarget_{date_obs}.csv",
            mime="text/csv",
            use_container_width=True,
            help="Sauvegarde ton carnet sur ton PC avant de fermer la page"
        )
        st.caption("💡 Pense à télécharger ton carnet avant de fermer — "
                   "les observations ne sont pas conservées d'une session à l'autre.")
    else:
        st.info("Aucune observation enregistrée pour cette session.")
