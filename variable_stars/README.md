# AstroVar — étoiles variables

Application **hors ligne** pour PC (Windows, macOS, Linux) et **Android** (smartphone et tablette) :

1. **Ciel** : liste des étoiles variables visibles depuis le **lieu** indiqué au **moment** choisi
   (≈ 50 000 étoiles du GCVS 5.1 embarquées, pas besoin d'internet sur le terrain).
   - lieu : saisie lat./long., recherche par nom (internet requis) ou position GPS ;
   - visibilité « à l'instant choisi » ou « pendant la nuit » (Soleil < −12°) ;
   - filtres : hauteur minimale, magnitude, amplitude, type (Mira, SR, céphéides, RR Lyr, éclipses, cataclysmiques…), constellation, recherche, « mon programme » (☆) ;
   - hauteur, azimut, heure de culmination, phase et prochain maximum/minimum (éphémérides GCVS) ;
   - résumé de la nuit : coucher/lever du Soleil, nuit astronomique, Lune (phase, hauteur) ;
   - fiche étoile : courbe de hauteur sur la nuit (avec la Lune), liens VSX, cartes AAVSO (VSP) ;
   - export de la liste en CSV (séparateur `;`, s'ouvre directement dans Excel).
2. **Courbes** : import d'un fichier **CSV, TXT, XLS, XLSX** (ou d'un fichier AAVSO) et tracé de la courbe de lumière.
   - colonnes reconnues automatiquement (date/JD, magnitude, erreur, filtre, étoile…) et modifiables ;
   - dates : JD/HJD/BJD, JD réduit, MJD, date calendaire (`24/01/2025 20:24`, `2025-01-24T20:24`), numéro de série Excel ;
   - virgule décimale française acceptée ; plusieurs étoiles et plusieurs filtres par fichier ;
   - axe en date, en JD ou en **phase** (période/époque pré-remplies depuis le GCVS) ; barres d'erreur ; export PNG.
3. **Rapport AAVSO** : compléter un fichier au format AAVSO (**Extended** ou **Visual**) et le transmettre.
   - « Envoyer vers le rapport AAVSO » depuis l'onglet Courbes, ou ouverture d'un fichier AAVSO existant pour le **compléter** (doublons ignorés) ;
   - tableau modifiable, contrôle des champs (filtres AAVSO, JD, `na`…), calcul automatique des **masses d'air** ;
   - « Transmettre à l'AAVSO » : enregistre le fichier puis ouvre la page **WebObs** (<https://www.aavso.org/webobs/file>) où il suffit de se connecter et de choisir le fichier.
     L'AAVSO ne propose pas d'API publique de dépôt : l'envoi final se fait donc sur son site avec vos identifiants.

Thèmes clair / sombre / **mode nuit rouge** (préserve la vision nocturne).

## Installer

Les installateurs sont construits automatiquement par GitHub Actions (workflow `AstroVar`) :
onglet **Actions** du dépôt → dernière exécution → section *Artifacts* :

| Fichier | Plateforme |
|---|---|
| `AstroVar-android-apk` → `AstroVar.apk` | Android 5.1+ (smartphone, tablette). Autoriser « sources inconnues » à l'installation. |
| `AstroVar-Windows` → `AstroVar Setup 1.0.0.exe` (installateur) ou `AstroVar 1.0.0.exe` (portable) | Windows 10/11 |
| `AstroVar-Linux` → `.AppImage` / `.deb` | Linux |
| `AstroVar-macOS` → `.dmg` | macOS (non signé : clic droit → Ouvrir) |

Pousser un tag `v1.0.0` crée en plus une *Release* GitHub avec tous les fichiers.

## Développement

```bash
cd variable_stars
npm install          # dépendances (Electron, Capacitor, Chart.js, SheetJS)
npm start            # lance l'application de bureau
npm run serve        # ou dans un navigateur : http://localhost:8080
npm test             # tests des calculs et des formats
npm run dist         # installateur pour le système courant (dossier dist/)
npm run android:apk  # APK (nécessite Android SDK + JDK 17) → android/app/build/outputs/apk/debug/
```

Structure :

- `www/` — l'application web (HTML/CSS/JS sans framework), partagée par toutes les plateformes
  - `js/astro.js` calculs astronomiques (temps sidéral, précession, Soleil, Lune, masse d'air) ;
  - `js/sky.js` onglet Ciel, `js/curves.js` import et courbes, `js/aavso.js` rapport AAVSO ;
  - `data/gcvs.js` catalogue généré ; `lib/` Chart.js et SheetJS (copiés par `npm run vendor`).
- `electron/` — application de bureau ; `android/` — projet Android (Capacitor).
- `tools/build_catalog.py` — régénère le catalogue (voir l'en-tête du script).

## Sources et précision

- Catalogue : *General Catalogue of Variable Stars* 5.1 (Samus et al., 2017), extrait diffusé par Stellarium ;
  positions Hipparcos (base HYG) et Gaia DR3 (index HEALPix du numéro de source, précision ≈ 1′).
- Positions du Soleil et de la Lune : formules basse précision (≈ 0,01° et ≈ 0,3°), vérifiées contre astropy.
- Les dates de maximum/minimum proviennent des éphémérides GCVS : elles sont indicatives, en particulier pour les Mira.
- Format AAVSO : <https://www.aavso.org/aavso-extended-file-format> et <https://www.aavso.org/aavso-visual-file-format>.
