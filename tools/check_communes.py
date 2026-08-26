#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CielScore — validation du fichier des communes embarque.

Contrairement aux catalogues d'objets, communes_bortle.csv n'est pas genere :
c'est une donnee source, livree telle quelle dans les assets (section 8.1).
Ce script en verifie les invariants, pour qu'une modification accidentelle ne
passe pas inapercue.

Usage :  python3 tools/check_communes.py
"""
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSET = os.path.join(ROOT, "android", "app", "src", "main", "assets", "communes_bortle.csv")

EXPECTED_COLUMNS = [
    "code_insee", "commune", "code_departement", "departement",
    "population", "lat", "lng", "bortle_estime", "description_ciel",
]
EXPECTED_COUNT = 34_869

# Bornes geographiques de la France, DOM compris.
LAT_RANGE = (-22.0, 52.0)
LNG_RANGE = (-63.0, 57.0)


def main():
    if not os.path.exists(ASSET):
        sys.exit("Fichier introuvable : %s" % ASSET)

    with open(ASSET, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        if reader.fieldnames != EXPECTED_COLUMNS:
            sys.exit("En-tete inattendu :\n  attendu %s\n  trouve  %s"
                     % (EXPECTED_COLUMNS, reader.fieldnames))
        rows = list(reader)

    errors = []
    seen = set()
    for line, row in enumerate(rows, start=2):
        code = row["code_insee"]
        if code in seen:
            errors.append("ligne %d : code INSEE duplique %s" % (line, code))
        seen.add(code)

        if not row["commune"].strip():
            errors.append("ligne %d : nom de commune vide" % line)

        try:
            bortle = int(row["bortle_estime"])
            if not 1 <= bortle <= 9:
                errors.append("ligne %d : Bortle %d hors de la plage 1-9" % (line, bortle))
        except ValueError:
            errors.append("ligne %d : Bortle illisible (%r)" % (line, row["bortle_estime"]))

        try:
            lat = float(row["lat"])
            lng = float(row["lng"])
            if not LAT_RANGE[0] <= lat <= LAT_RANGE[1]:
                errors.append("ligne %d : latitude %s hors de France" % (line, lat))
            if not LNG_RANGE[0] <= lng <= LNG_RANGE[1]:
                errors.append("ligne %d : longitude %s hors de France" % (line, lng))
        except ValueError:
            errors.append("ligne %d : coordonnees illisibles" % line)

        try:
            if int(row["population"]) < 0:
                errors.append("ligne %d : population negative" % line)
        except ValueError:
            errors.append("ligne %d : population illisible (%r)" % (line, row["population"]))

        if len(errors) > 20:
            errors.append("... (arret apres 20 anomalies)")
            break

    if errors:
        print("communes_bortle.csv : %d anomalie(s)" % len(errors), file=sys.stderr)
        for e in errors:
            print("  " + e, file=sys.stderr)
        sys.exit(1)

    if len(rows) != EXPECTED_COUNT:
        print("ATTENTION : %d communes, %d attendues." % (len(rows), EXPECTED_COUNT))
        print("Si le changement est voulu, mettez a jour EXPECTED_COUNT dans ce script.")
        sys.exit(1)

    departments = {r["code_departement"] for r in rows}
    print("communes_bortle.csv : %d communes, %d departements — aucune anomalie"
          % (len(rows), len(departments)))


if __name__ == "__main__":
    main()
