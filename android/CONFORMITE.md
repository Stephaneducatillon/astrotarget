# Conformité à la documentation

Ce document recense, point par point, ce que l'application Android reprend de
la documentation de référence, et **tout ce qui s'en écarte**. Rien n'est laissé
implicite : chaque écart est soit une décision validée, soit une zone que le
document ne couvre pas.

> **L'application s'appelle SkyScore depuis la version 1.0.0.** Les documents de
> référence cités ci-dessous sont antérieurs et portent le nom d'origine,
> *CielScore* : leurs titres sont reproduits tels quels, une citation ne se
> réécrivant pas. Partout ailleurs, « SkyScore » désigne l'application.

Deux documents font référence :

| Document | Rôle |
|---|---|
| *CielScore — Documentation fonctionnelle et technique* (version UX v0.6.4) | Le socle : onglets, règles de gestion, catalogues, interfaces externes. |
| *CielScore — Moteur de scoring, règles et formules* v2.0 (09/09/2026) | **Remplace** les §5.1, §5.3, §5.9, §7.2 et le critère de brillance de surface du §6.1. Voir la section 4. |

En cas de contradiction entre les deux, **les règles v2.0 l'emportent** : elles
sont postérieures et corrigent explicitement des défauts remontés par les
testeurs.

---

## 1. Ce qui est repris à l'identique

| Section | Élément | Où c'est implémenté |
|---|---|---|
| 2.1 → 2.8 | Les onglets et leur accès public / connecté (voir 3.4 et 3.6 pour les deux retraits) | `ui/screens/`, `MainActivity.kt` |
| 3.2 | Les trois vues du panneau latéral | `ui/components/ObjectSheet.kt` |
| 3.3 | Règles d'affichage de la carte (alt > 2°, > 0°, > 5°, Lune > 2°) | `ui/components/SkyMapView.kt` |
| 3.4 | Projection azimutale équidistante | `astro/SkyProjection.kt` |
| 3.5 | Conversion équatorial → horizontal | `astro/AstroMath.kt` |
| 4.1 | Filtres éliminatoires RG-F-01 à RG-F-04 | `scoring/ScoringEngine.kt` |
| 4.2 | Filtrage dynamique et interpolation crépusculaire | `scoring/Formulas.darknessLimits` |
| 4.3 | RG-P-01 à RG-P-05 | `scoring/ScoringEngine.scoreSolarSystem` |
| 4.4 | RG-L-01 à RG-L-04 | `scoring/ScoringEngine.moonScore` |
| 4.5 | RG-I-01 à RG-I-05 | `scoring/Formulas.kt`, `model/SessionParams` |
| 4.6 | Durées de vie et clés de cache | `data/cache/TtlCache.kt` |
| 5.1 → 5.9 | Toutes les formules, hors celles réécrites par les règles v2.0 (voir 4) | `scoring/Formulas.kt` |
| 6.1 / 6.3 / 6.4 | Les trois formules de score et leurs pondérations | `scoring/ScoringEngine.kt` |
| 6.2 | Barème du seeing déduit du vent | `scoring/Formulas.seeingIndex` |
| 6.5 | Lecture du score | `scoring/Formulas.scoreInterpretation` |
| 7.1 → 7.5 | Phases, couleurs, nuits d'été ; le score de nuit est celui des règles v2.0 | `astro/Twilight.kt` |
| 8.1 | Messier 110, Caldwell 109, NGC/IC 13 308, 6 corps | `assets/`, `tools/build_catalogs.py` |
| 8.1 | 34 869 communes françaises, Bortle automatique France entière | `assets/communes_bortle.csv`, `catalog/CommuneIndex.kt` |
| 8.2 | Les interfaces externes, hors Mistral (voir 3.6) ; la clé NASA est embarquée (voir 3.7) | `data/net/` |
| 8.4 | Stratégie de repli service par service | `data/net/`, `model/SkyConditions` |
| 9.1 | PBKDF2-SHA256, 260 000 itérations, sel 16 octets, comparaison à temps constant | `data/auth/PasswordHasher.kt` |
| 9.2 | Tables `users` et `observations`, index | `data/db/Entities.kt` |
| 9.4 | Format et niveaux de journalisation | `util/Log.kt` |
| 10.2 | L'exemple complet M57 est rejoué en test | `DocumentationConformanceTest.kt` |

Les valeurs chiffrées des deux documents (tableaux 5.2, 5.5, 5.6, 6.2, 10.2 du
premier ; tableaux des pages 6 et 7 des règles v2.0) sont vérifiées
automatiquement par **32 tests de conformité**, sur **65 tests unitaires** au
total. Le tableau de validation de la page 6 des règles v2.0 est reproduit à
**1,4 point près** au pire des cas.

---

## 2. Décisions prises avec vous

| Sujet | Décision retenue |
|---|---|
| Architecture | Application Android native (Kotlin + Jetpack Compose), calculs embarqués. Aucune dépendance au code Python. |
| Périmètre | Les onglets du document **et** la carte du ciel interactive. |
| Lieu et Bortle | Fichier `communes_bortle.csv` embarqué : recherche et rattachement GPS hors ligne, indice de Bortle ajustable. |
| Comptes | Compte et carnet 100 % locaux (Room). Plus aucune clé à saisir : celle de la NASA est embarquée à la construction (voir 3.7). |
| Fonctions IA | **Retirées en version 1.0** : l'application ne dépend plus de Mistral (voir 3.6). |
| Nom | L'application s'appelle **SkyScore** depuis la version 1.0.0. |
| Smart télescopes | Les **7 modèles détaillés** au tableau 5.9, plus les 2 Seestar Pro ajoutés ensuite (voir 3.5). |
| Carte du ciel | Figures de constellations **complètes** (358 étoiles, 239 segments) plutôt que les 174 / 113 du §3.3. |
| Objets sans dimensions | Consultables dans l'Explorer, exclus du Top du Dashboard. |
| Formule du §5.9 | Caduque : les règles v2.0 réécrivent la magnitude limite (voir 4). |
| Tableau de la p. 7 des règles v2.0 | Il **fait foi** face aux exemples chiffrés de la p. 3, qui omettent la correction de pollution. |
| Filtrage crépusculaire du §4.2 | **Conservé** : il reste appliqué en amont du scoring v2.0. |
| Seuil des planètes brillantes | **−3°** de hauteur du Soleil, valeur de RG-P-03 (voir 3.1). |
| Critères non définis | Définitions retenues pour « Fenêtre », F/D et champ (voir 3.2). |

---

## 3. Écarts par rapport au document

Les trois points ci-dessous ont été **soumis et arbitrés** : les choix décrits
sont ceux retenus dans l'application, et non des options ouvertes.

### 3.1 Incohérences internes au document

**Le tableau du §5.9 ne découlait pas de sa propre formule.** Ce point est
**caduc** : les règles v2.0 réécrivent la magnitude limite des smart télescopes,
tableau et formule compris. Voir la section 4.

**Le §2.5 annonce 12 smart télescopes, le §5.9 n'en détaille que 7.**
→ Les 7 documentés ont d'abord été intégrés seuls (décision validée), puis
2 modèles récents s'y sont ajoutés à votre demande (voir 3.5) : le catalogue
en compte **9**.

**Le §4.2 et la règle RG-P-03 divergent sur le seuil des planètes brillantes.**
Le tableau du §4.2 place Vénus et Jupiter dès le crépuscule civil (Soleil à 0°),
RG-P-03 précise « (−3°) ».
→ **Arbitrage retenu : −3°**, la règle nommée étant la plus spécifique.

### 3.2 Éléments que le document ne définit pas

**Le critère « Fenêtre » (§6.1, 15 %).**
Le document donne `min(durée_min / 240, 1) × 100` sans définir `durée_min`.
→ **Arbitrage retenu : le temps passé au-dessus de 30° (seuil optimal du §2.2)
pendant que le Soleil est sous l'horizon, évalué sur les 10 heures suivant
l'heure de session** — la même fenêtre que la courbe d'altitude du §2.2.
Voir `ScoringEngine.observationWindowMinutes`.

**Les critères F/D et champ du score smart télescope (§6.4, 5 % chacun).**
Le document énonce les principes (« bonus F/D court », « grand champ favorable
aux nébuleuses étendues ») sans formule.
→ **Arbitrage retenu :**
- F/D : `clip((8 − F/D) / 6, 0, 1) × 100`, soit 100 à F/2 et 0 à partir de F/8 ;
- champ : maximum lorsque l'objet occupe la moitié du champ, nul s'il est
  ponctuel ou plus grand que le champ.
Voir `ScoringEngine.focalRatioScore` et `fieldMatchScore`.

**Les caractéristiques des smart télescopes autres que le diamètre.**
Le §2.5 demande d'afficher « ouverture, focale, capteur, champ » mais le §5.9 ne
fournit que le diamètre.
→ Focale, dimensions de capteur et taille de pixel proviennent des
**spécifications constructeur**, pas du document. Elles sont regroupées dans une
seule table, `SmartTelescope.CATALOG`, pour être corrigées d'un seul geste.

**Le fond d'étoiles de la carte du ciel.**
Le §3.3 annonce 174 étoiles et 113 segments sans fournir les données.
→ Table J2000 constituée pour l'application : **358 étoiles, 31 figures,
239 segments** (décision validée : figures complètes). Régénérable par
`tools/build_stars.py`.

### 3.3 Adaptations liées au support mobile

| Document | Application Android |
|---|---|
| §1.2 Gradio 6.18 / Python 3.13 | Kotlin + Jetpack Compose, `minSdk` 26 |
| §1.2 Astropy + PyEphem + NumPy | Moteur d'éphémérides Kotlin embarqué (Meeus ch. 25, 45, 47 ; éléments képlériens JPL 1800–2050) |
| §1.2 auth.py + SQLite | Room, base locale `skyscore.db` |
| §9.3 Restauration de la base depuis un dépôt distant | Sauvegarde Android (`backup_rules.xml`) ; aucune base distante |
| §1.2 Hébergement Hugging Face Spaces | APK construit par GitHub Actions |
| §8.2 API Géo en ligne | Inutile : les communes sont embarquées, la recherche est hors ligne |

Le module Caldwell mérite une note : construit depuis OpenNGC par
correspondance C# → NGC/IC comme le prévoit le §8.1, il ne donne que **106**
objets, car C9 (Sh2-155), C41 (Hyades) et C99 (Coalsack) n'ont pas d'entrée
NGC/IC. Les trois sont ajoutés explicitement pour atteindre les **109** annoncés.

---

### 3.4 Onglet Équipement retiré

Le §2.5 décrit un onglet **Équipement** (oculaires, astrophotographie, smart
télescopes). Il a été **retiré à votre demande**, faute d'utilité au quotidien,
avec l'intention de le rétablir plus tard.

Cumulé au retrait de l'Assistant IA (§ 3.6), l'application compte **6 onglets**
au lieu de 8. Rien d'autre n'est
affecté : les formules des §5.7 et §5.8 (grossissement, champ réel, pupille de
sortie, F/D effectif, échantillonnage, bornes de Shannon) restent implémentées
dans `scoring/Formulas.kt` et couvertes par les tests, et les smart télescopes
restent sélectionnables depuis le Dashboard, avec leur formule de score du §6.4.

Pour le rétablir, l'écran est intact dans l'historique Git :

```bash
git log --oneline --diff-filter=D -- '*EquipmentScreen.kt'   # trouver le commit
git show <commit>^:android/app/src/main/java/com/cielscore/app/ui/screens/EquipmentScreen.kt \
  > android/app/src/main/java/com/skyscore/app/ui/screens/EquipmentScreen.kt
```

Le chemin source garde volontairement l'ancien nom de paquet : c'est celui
qu'avait le fichier dans l'historique, avant le renommage en SkyScore. Il reste
ensuite à réintroduire l'entrée `EQUIPMENT` dans l'énumération `AppTab` de
`MainActivity.kt`, sa branche dans le `when`, et à corriger la déclaration
`package` du fichier récupéré.

---

### 3.5 Deux smart télescopes ajoutés hors document

Le document ne connaît ni le **Seestar S30 Pro** ni le **Seestar S50 Pro**,
sortis après sa rédaction. Ils ont été ajoutés à votre demande, avec les
caractéristiques que vous avez relevées sur les fiches constructeur :

| Modèle | Diamètre | Focale | F/D | Capteur | Pixel |
|---|---|---|---|---|---|
| Seestar S30 Pro | 30 mm | 150 mm | f/5,0 | IMX585 | 2,9 µm |
| Seestar S50 Pro | 50 mm | 260 mm | f/5,2 | IMX585 | 2,9 µm |

Les dimensions du capteur ne sont pas relevées mais **déduites** : l'IMX585
compte 3856 × 2180 pixels au pas de 2,9 µm, soit 11,18 × 6,32 mm. La diagonale
obtenue, 12,85 mm, correspond exactement au format 1/1,2 pouce annoncé par le
fabricant — la déduction est donc vérifiable, et un test la verrouille.

Conséquence sur le scoring : **aucune approximation ne pèse sur la magnitude
limite**, qui ne dépend que du diamètre, du Bortle et de la durée de pose
(règles v2.0, § 2). La focale et le capteur ne servent qu'aux deux critères à
5 % du score smart télescope (§6.4) et à l'affichage.

Le catalogue compte donc 9 modèles, dont 7 documentés. Il reste une entrée
incertaine, héritée de la première version : les dimensions du capteur du
**Vespera II**, non vérifiées, et dont la référence est laissée vide dans le
code plutôt que devinée.


---

### 3.6 Fonctions d'intelligence artificielle retirées — version 1.0

Toutes les fonctions reposant sur l'API **Mistral** ont été retirées à votre
demande pour cette première version, jugée trop lourde avec elles. Trois usages
disparaissent :

| Usage | Emplacement | Devenu |
|---|---|---|
| Assistant conversationnel (§2.7) | Onglet **Assistant IA** | Onglet supprimé |
| Plan de soirée (§2.4) | Onglet Sessions | Remplacé par une **feuille de route** : les cibles du dernier calcul, exportables en PDF |
| Guide d'observation (§2.2) | Fiche objet | Carte supprimée |

Ce qui disparaît avec eux : `data/net/MistralApi.kt`, `ui/screens/AssistantScreen.kt`,
la clé d'API Mistral du Profil et son stockage local.

**Ce qui est conservé, et pourquoi.** L'export PDF dépendait du texte généré par
l'IA, mais son tableau des cibles, lui, est entièrement calculé : plutôt que de
le supprimer avec le reste, il devient une feuille de route de la soirée —
conditions du site, puis tableau des cibles. Retirer l'IA ne devait pas coûter
la possibilité d'emporter sa liste sur le terrain.

La clé **NASA APOD reste** : elle sert l'image du jour de l'onglet Informations,
qui n'a rien d'une fonction d'IA. Le Profil ne demande donc plus qu'une seule
clé.

L'application ne dialogue plus avec aucun service d'IA ; la règle
`valider_top_ia` du § 4.5, qui lui était destinée, reste néanmoins implémentée.

---

### 3.7 Clé NASA embarquée dans l'APK

Le §8.2 fait saisir les clés d'API par l'utilisateur, dans le Profil. À votre
demande, **la clé NASA APOD est désormais embarquée dans l'application** : plus
rien à saisir, et la carte « Clé d'API » du Profil disparaît.

Elle n'est pas écrite dans le code source. Ce dépôt étant **public**, une clé
committée y serait indexée par GitHub et moissonnée en quelques heures par les
robots qui scrutent les dépôts publics — un risque bien plus concret que celui
de la décompilation. Elle est donc injectée à la construction, depuis le secret
d'Actions `NASA_API_KEY`, et devient `BuildConfig.NASA_API_KEY`. À défaut de
secret, la construction retombe sur `DEMO_KEY`, la clé publique d'essai de la
NASA.

**Ce que cela ne protège pas.** Une clé embarquée dans un APK reste extractible
par qui le décompile : aucun procédé n'y change rien, un client hors ligne doit
porter le secret qu'il utilise. C'est un compromis assumé, acceptable pour une
clé gratuite, limitée en débit et sans valeur hors de cet usage ; il suffit d'en
régénérer une et de reconstruire si son quota était épuisé par un tiers.

Conséquence de code : `model/ApiKey.kt` et ses tests disparaissent avec la
dernière saisie de clé — plus rien à masquer ni à valider. Le projet compte donc
**65 tests** au lieu de 70.

---

## 4. Règles de scoring v2.0 (09/09/2026)

Le document *Moteur de scoring — règles et formules* v2.0 corrige, à la suite
des retours de testeurs, cinq points du scoring. Tout ce qu'il ne mentionne pas
reste régi par la documentation v0.6.4.

### 4.1 Magnitude limite : le ciel entre dans la formule

```
mag_limite = 2.1 + 5×log10(D_mm) + correction_pollution(Bortle)
```

| Bortle | 1–2 | 3–4 | 5–6 | 7–8 | 9 |
|---|---|---|---|---|---|
| Correction | +1,2 | +0,6 | 0,0 | −0,8 | −1,0 |

La règle **RG-I-02** du premier document — « la limite instrumentale ne dépend
que du diamètre » — est donc **abrogée**. Un 130 mm passe de 12,7 partout à 11,9
depuis un site Bortle 7 et 13,9 depuis un site Bortle 1.

**Écart documenté et arbitré.** Les exemples chiffrés de la page 3 des règles
v2.0 (« D = 60 mm, Bortle 9 → 11,0 ») appliquent la formule *sans* la correction
de pollution, alors même qu'ils annoncent un indice de Bortle ; le tableau de
référence de la page 7, lui, est cohérent d'un diamètre à l'autre.
→ **Arbitrage retenu (validé) : le tableau de la page 7 fait foi.** Il est
rejoué en test, à 0,12 magnitude près — la précision de ses propres arrondis.

Pour les smart télescopes, la durée de pose s'ajoute et s'exprime désormais **en
minutes**, une heure servant de référence :

```
mag_limite_smart = 2.1 + 5×log10(D_mm) + correction_pollution(Bortle)
                 + 1.25×log10(T_min / 60)
```

Voir `Formulas.bortlePollutionCorrection`, `instrumentLimitingMagnitude`,
`smartTelescopeLimitingMagnitude`.

### 4.2 Correctif des objets brillants étendus

C'est la correction principale de la v2.0, et celle qui motivait les retours.

**Le défaut.** M31 (magnitude 3,4, étendue 190′ × 60′) a une brillance de surface
calculée de 22,2 mag/arcsec², au-delà du plafond d'un ciel Bortle 7 (18,5 + 3,5
de tolérance = 22,0). L'ancien facteur `f_sb` tombait à zéro et **M31 obtenait un
score nul**, alors qu'elle se voit à l'œil nu. La cause est structurelle : la
brillance de surface répartit le flux sur toute l'étendue angulaire, quand l'œil,
lui, intègre le flux total.

**La correction, en trois temps :**

1. Les **amas** (ouverts et globulaires — `TYPES_RESOLUS`) sont exemptés du
   critère : résolus en étoiles individuelles, la brillance moyenne n'a pas de
   sens pour eux. `s_sb = 100`, `f_sb = 1`.
2. Un objet dont la magnitude intégrée est **nettement accessible**
   (`magnitude < mag_limite − 2`) n'est plus éliminé, et son sous-score mélange
   les deux lectures :

   ```
   s_sb_diff = clip((sb_lim + 3.5 − sb) / 8.5, 0, 1) × 100
   s_sb_mag  = clip((mag_lim − magnitude) / 6, 0, 1) × 100
   poids     = clip((mag_lim − 2 − magnitude) / (mag_lim − 2), 0, 1)
   s_sb      = s_sb_diff × (1 − poids) + s_sb_mag × poids
   ```

3. Le multiplicateur `f_sb` n'est plus appliqué à ces objets rattrapés
   (`f_sb = 1`), sans quoi le score retomberait à zéro malgré le sous-score.

**Reproduction du tableau de validation** (page 6 : Bortle 7, D = 114 mm,
altitude 55°, Lune au plus mauvais) :

| Objet | Document | Application |
|---|---|---|
| M45 Pléiades | 88 | 88,0 |
| M42 Orion | 83 | 82,7 |
| M31 Andromède | 83 | 82,7 |
| M13 Hercule | 88 | 88,0 |
| M33 Triangle | 80 | 79,0 |
| M101 Pinwheel | 76 | 74,6 |
| NGC 891 | 74 | 73,0 |

Écart maximal **1,4 point**, imputable aux arrondis à l'entier du document et
aux dimensions angulaires, qu'il ne fournit que pour certains objets.

Voir `ScoringEngine.assessSurfaceBrightness` et `Formulas.surfaceBrightnessFactor`.

### 4.3 Filtre d'exclusion RG-F-02 réécrit

L'ancien filtre écartait tout objet dont la brillance dépassait le plafond du
site. Le nouveau n'écarte que la **conjonction** des deux échecs :

```
f_sb <= 0  ET  magnitude non accessible   →   score = 0
```

Les trois autres filtres éliminatoires (altitude < 5°, magnitude > limite,
nuages > 90 %) sont inchangés.

**Arbitrage retenu (validé) : le filtrage crépusculaire du §4.2 est conservé.**
Les règles v2.0 ne le mentionnent pas, mais ne le contredisent pas non plus : il
reste appliqué en amont, et c'est la magnitude limite *instrumentale* — non la
limite crépusculaire — qui sert à juger `magnitude accessible`, faute de quoi un
objet changerait de catégorie au fil du crépuscule.

### 4.4 Score de nuit progressif

```
s_nuit = clip(−altitude_soleil / 18, 0, 1) × 100
```

Les quatre paliers du §7.2 (100 / 70 / 40 / 10 / 0) faisaient sauter le score
global d'une phase crépusculaire à l'autre ; la progression est désormais
continue du coucher du Soleil à la nuit noire. Voir `Twilight.nightScore`.

### 4.5 Validation déterministe (`valider_top_ia`)

Un garde-fou, plus sévère que les filtres éliminatoires, prévu pour filtrer les
cibles **avant de les soumettre à l'assistant IA** — il ne retire rien du Top
affiché :

| Motif de rejet | Seuil |
|---|---|
| Trop bas sur l'horizon | altitude < 20° |
| Hors de portée | magnitude > magnitude limite |
| Observation compromise | nuages > 80 % |
| Éblouissement lunaire | séparation < 30° **et** phase > 60 % |
| Fond de ciel trop lumineux | SB > sb_limite + 3,5 (hors amas) |

La règle **reste implémentée et testée** (`ScoringEngine.aiVetoes` et
`validatedTargets`) bien que son consommateur, l'assistant, ait été retiré en
version 1.0 (§ 3.6) : elle appartient aux règles de scoring, et la retirer
aurait créé un écart avec le document de référence. Elle sera rebranchée le
jour où l'assistant reviendra.

### 4.6 Conséquence corrigée : les amas avec nébulosité (`Cl+N`)

L'exemption des amas (`TYPES_RESOLUS`) rend le **type** de chaque objet
directement déterminant pour son score. Elle a mis au jour un défaut de la
conversion des catalogues, antérieur aux règles v2.0 mais jusque-là sans
conséquence sur le calcul.

`tools/build_catalogs.py` rattachait le type OpenNGC **`Cl+N`** — « amas *avec
nébulosité* » — aux amas ouverts. M42 s'en trouvait classée en amas : absente du
filtre « Nébuleuse » de l'Explorer, et désormais exemptée du critère de
brillance, elle scorait **88** là où les règles v2.0 attendent **83**.

Ce n'était pas un cas isolé : **67 objets** de la source sont codés `Cl+N`, dont
IC 5146 (Cocoon), IC 1396, IC 1805 (le Cœur), IC 1848 (l'Âme), NGC 1333 et
IC 2944. Tous s'observent comme des nébuleuses, et c'est précisément leur
brillance de surface qui décide si on les voit — les exempter du critère les
aurait tous survalorisés.

→ **Correction retenue : `Cl+N` est rattaché à « Nébuleuse »**, dans le
générateur et non dans les fichiers, pour survivre à la prochaine régénération.
Les catalogues sont régénérés : 70 objets changent de type (M42, C19, C100 et
67 entrées NGC/IC), les effectifs restent inchangés (110 / 109 / 13 308), et
M42 retombe à **82,7**.

La classification `*Ass` (associations stellaires) reste rattachée aux amas
ouverts : ces objets sont bien résolus en étoiles individuelles.

---

## 5. Précision des éphémérides

Le moteur remplace Astropy et PyEphem. Précision constatée sur les cas de
référence testés :

| Grandeur | Écart mesuré |
|---|---|
| Déclinaison du Soleil (solstice, équinoxe) | < 0,01° |
| Fraction illuminée de la Lune (syzygies) | < 0,1 % |
| Position de la Lune | de l'ordre de la minute d'arc |
| Positions planétaires | quelques minutes d'arc (éléments moyens JPL, 1800–2050) |
| Durée de la nuit astronomique | < 1 minute |

Largement suffisant pour un calcul d'altitude, de fenêtre d'observation et de
distance angulaire à la Lune.
