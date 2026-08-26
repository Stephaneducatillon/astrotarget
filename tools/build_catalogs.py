#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CielScore — generation des catalogues embarques dans l'application Android.

Source : NGC.csv (OpenNGC, delimiteur ';') present a la racine du depot.
Sorties : android/app/src/main/assets/{messier,caldwell,ngcic}.csv

Format de sortie (delimiteur ';', 1 ligne d'en-tete) :
    id;name;type;ra_deg;dec_deg;mag;majax;minax;const;common

  id       identifiant court affiche (M31, C14, NGC0891...)
  name     designation OpenNGC (NGC0224, IC4715...)
  type     type CielScore : Galaxie | Nebuleuse | Amas ouvert | Amas globulaire | Autre
  ra_deg   ascension droite J2000 en degres decimaux
  dec_deg  declinaison J2000 en degres decimaux
  mag      magnitude (V-Mag si disponible, sinon B-Mag, sinon vide)
  majax    grand axe en arcmin (vide si inconnu)
  minax    petit axe en arcmin (vide si inconnu ; l'app retombe sur majax)
  const    constellation (abreviation IAU 3 lettres)
  common   nom usuel

Reference : documentation CielScore, section 8.1 « Catalogues d'objets ».
"""
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "NGC.csv")
OUT_DIR = os.path.join(ROOT, "android", "app", "src", "main", "assets")

HEADER = "id;name;type;ra_deg;dec_deg;mag;majax;minax;const;common"

# --- Correspondance des types OpenNGC vers les 5 types CielScore (section 2.3) ---
TYPE_MAP = {
    "G": "Galaxie", "GPair": "Galaxie", "GTrpl": "Galaxie", "GGroup": "Galaxie",
    "GCl": "Amas globulaire",
    "OCl": "Amas ouvert", "Cl+N": "Amas ouvert", "*Ass": "Amas ouvert",
    "PN": "Nebuleuse", "Neb": "Nebuleuse", "HII": "Nebuleuse",
    "RfN": "Nebuleuse", "EmN": "Nebuleuse", "SNR": "Nebuleuse",
    "*": "Autre", "**": "Autre", "Other": "Autre", "Nova": "Autre",
}
# Types exclus : doublons et objets inexistants
EXCLUDED_TYPES = {"Dup", "NonEx"}

# --- Caldwell : correspondance C# -> designation NGC/IC (Patrick Moore, 1995) ---
# Les trois objets sans equivalent NGC/IC (C9 = Sh2-155, C41 = Hyades/Mel 25,
# C99 = Coalsack) ne sont pas derivables d'OpenNGC et sont donc absents,
# conformement au mode de construction decrit en section 8.1.
CALDWELL = {
    1: "NGC0188", 2: "NGC0040", 3: "NGC4236", 4: "NGC7023", 5: "IC0342",
    6: "NGC6543", 7: "NGC2403", 8: "NGC0559", 10: "NGC0663", 11: "NGC7635",
    12: "NGC6946", 13: "NGC0457", 14: "NGC0869", 15: "NGC6826", 16: "NGC7243",
    17: "NGC0147", 18: "NGC0185", 19: "IC5146", 20: "NGC7000", 21: "NGC4449",
    22: "NGC7662", 23: "NGC0891", 24: "NGC1275", 25: "NGC2419", 26: "NGC4244",
    27: "NGC6888", 28: "NGC0752", 29: "NGC5005", 30: "NGC7331", 31: "IC0405",
    32: "NGC4631", 33: "NGC6992", 34: "NGC6960", 35: "NGC4889", 36: "NGC4559",
    37: "NGC6885", 38: "NGC4565", 39: "NGC2392", 40: "NGC3626", 42: "NGC7006",
    43: "NGC7814", 44: "NGC7479", 45: "NGC5248", 46: "NGC2261", 47: "NGC6934",
    48: "NGC2775", 49: "NGC2237", 50: "NGC2244", 51: "IC1613", 52: "NGC4697",
    53: "NGC3115", 54: "NGC2506", 55: "NGC7009", 56: "NGC0246", 57: "NGC6822",
    58: "NGC2360", 59: "NGC3242", 60: "NGC4038", 61: "NGC4039", 62: "NGC0247",
    63: "NGC7293", 64: "NGC2362", 65: "NGC0253", 66: "NGC5694", 67: "NGC1097",
    68: "NGC6729", 69: "NGC6302", 70: "NGC0300", 71: "NGC2477", 72: "NGC0055",
    73: "NGC1851", 74: "NGC3132", 75: "NGC6124", 76: "NGC6231", 77: "NGC5128",
    78: "NGC6541", 79: "NGC3201", 80: "NGC5139", 81: "NGC6352", 82: "NGC6193",
    83: "NGC4945", 84: "NGC5286", 85: "IC2391", 86: "NGC6397", 87: "NGC1261",
    88: "NGC5823", 89: "NGC6087", 90: "NGC2867", 91: "NGC3532", 92: "NGC3372",
    93: "NGC6752", 94: "NGC4755", 95: "NGC6025", 96: "NGC2516", 97: "NGC3766",
    98: "NGC4609", 100: "IC2944", 101: "NGC6744", 102: "IC2602", 103: "NGC2070",
    104: "NGC0362", 105: "NGC4833", 106: "NGC0104", 107: "NGC6101",
    108: "NGC4372", 109: "NGC3195",
}
# Les trois Caldwell sans entree OpenNGC, ajoutes explicitement pour que le
# catalogue compte bien les 109 objets annonces en section 8.1.
# Valeurs : RA/Dec J2000, magnitude, dimensions en arcmin.
CALDWELL_EXTRA = {
    9: ("C9", "Sh2-155", "Nebuleuse", 344.2000, 62.6167, 7.70, 50.0, 30.0, "Cep", "Cave Nebula"),
    41: ("C41", "Melotte 25", "Amas ouvert", 66.7500, 15.8667, 0.50, 330.0, 330.0, "Tau", "Hyades"),
    99: ("C99", "Coalsack", "Nebuleuse", 193.2500, -63.0000, None, 420.0, 300.0, "Cru", "Coalsack"),
}

CALDWELL_NAMES = {
    14: "Double Cluster", 41: "Hyades", 49: "Rosette Nebula",
    63: "Helix Nebula", 77: "Centaurus A", 80: "Omega Centauri",
    92: "Eta Carinae Nebula", 94: "Jewel Box", 99: "Coalsack",
    103: "Tarantula Nebula", 106: "47 Tucanae",
}

# --- Messier absents d'OpenNGC (aucune entree NGC/IC) ---
# M102 est identifie a NGC 5866 (identification la plus couramment retenue).
# Valeurs : RA/Dec J2000, magnitude visuelle, dimensions en arcmin.
MESSIER_EXTRA = {
    40: ("M40", "Winnecke 4", "Autre", 185.5521, 58.0828, 8.40, 0.8, 0.8, "UMa", "Winnecke 4"),
    45: ("M45", "Melotte 22", "Amas ouvert", 56.7500, 24.1167, 1.60, 110.0, 110.0, "Tau", "Pleiades"),
}
MESSIER_ALIAS = {102: "NGC5866"}  # M102 -> NGC 5866


def hms_to_deg(s):
    s = s.strip()
    if not s:
        return None
    h, m, sec = s.split(":")
    return (float(h) + float(m) / 60.0 + float(sec) / 3600.0) * 15.0


def dms_to_deg(s):
    s = s.strip()
    if not s:
        return None
    sign = -1.0 if s[0] == "-" else 1.0
    d, m, sec = s.lstrip("+-").split(":")
    return sign * (float(d) + float(m) / 60.0 + float(sec) / 3600.0)


def num(s):
    s = s.strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def fmt(v, nd=4):
    return "" if v is None else ("%.*f" % (nd, v)).rstrip("0").rstrip(".")


def clean(s):
    return s.replace(";", ",").replace("\n", " ").strip()


def convert(row, ident):
    """Transforme une ligne OpenNGC en tuple de sortie, ou None si inutilisable."""
    ra = hms_to_deg(row["RA"])
    dec = dms_to_deg(row["Dec"])
    if ra is None or dec is None:
        return None
    otype = TYPE_MAP.get(row["Type"].strip(), "Autre")
    mag = num(row["V-Mag"])
    if mag is None:
        mag = num(row["B-Mag"])
    majax = num(row["MajAx"])
    minax = num(row["MinAx"])
    if majax is not None and minax is None:
        minax = majax
    common = clean(row["Common names"].split(",")[0]) if row["Common names"].strip() else ""
    return (ident, row["Name"].strip(), otype, ra, dec, mag, majax, minax,
            row["Const"].strip(), common)


def write(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(HEADER + "\n")
        for r in rows:
            f.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (
                r[0], r[1], r[2], fmt(r[3]), fmt(r[4]),
                fmt(r[5], 2), fmt(r[6], 2), fmt(r[7], 2), r[8], r[9]))
    print("  -> %-14s %5d objets" % (os.path.basename(path), len(rows)))


def main():
    if not os.path.exists(SRC):
        sys.exit("NGC.csv introuvable a la racine du depot : %s" % SRC)
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(SRC, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    by_name = {r["Name"].strip(): r for r in rows}
    print("OpenNGC : %d lignes lues" % len(rows))

    # ---- Messier (110) ----
    messier = {}
    for r in rows:
        m = r["M"].strip()
        if not m:
            continue
        n = int(m)
        out = convert(r, "M%d" % n)
        if out:
            messier[n] = out
    for n, alias in MESSIER_ALIAS.items():
        if n not in messier and alias in by_name:
            out = convert(by_name[alias], "M%d" % n)
            if out:
                messier[n] = out
    for n, t in MESSIER_EXTRA.items():
        messier.setdefault(n, t)
    missing = sorted(set(range(1, 111)) - set(messier))
    if missing:
        print("  ATTENTION Messier manquants : %s" % missing)
    write(os.path.join(OUT_DIR, "messier.csv"), [messier[n] for n in sorted(messier)])

    # ---- Caldwell (derive d'OpenNGC) ----
    caldwell, absent = {}, []
    for n in range(1, 110):
        designation = CALDWELL.get(n)
        out = None
        if designation is not None and designation in by_name:
            out = convert(by_name[designation], "C%d" % n)
        if out is None:
            out = CALDWELL_EXTRA.get(n)
            if out is None:
                absent.append(n)
                continue
        elif n in CALDWELL_NAMES:
            out = out[:9] + (CALDWELL_NAMES[n],)
        caldwell[n] = out
    if absent:
        print("  ATTENTION Caldwell manquants : %s" % absent)
    write(os.path.join(OUT_DIR, "caldwell.csv"), [caldwell[n] for n in sorted(caldwell)])

    # ---- NGC / IC complet ----
    ngcic = []
    for r in rows:
        if r["Type"].strip() in EXCLUDED_TYPES:
            continue
        out = convert(r, r["Name"].strip())
        if out:
            ngcic.append(out)
    write(os.path.join(OUT_DIR, "ngcic.csv"), ngcic)


if __name__ == "__main__":
    main()
