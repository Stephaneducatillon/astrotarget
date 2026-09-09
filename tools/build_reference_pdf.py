# -*- coding: utf-8 -*-
"""Genere le document de reference des formules et regles de scoring CielScore.

Le contenu est repris des sources de l'application (Formulas.kt, ScoringEngine.kt,
Twilight.kt, SkyObject.kt) : c'est un miroir du code, pas une reformulation. Toute
evolution du moteur de calcul doit donc etre reportee ici, puis le document
regenere.

Usage :  pip install reportlab && python3 tools/build_reference_pdf.py
Sortie :  docs/CielScore_Formules_et_Regles.pdf
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Table, TableStyle,
    KeepTogether, PageBreak,
)

OUT = "docs/CielScore_Formules_et_Regles.pdf"

ENCRE      = colors.HexColor("#1B2430")
ENCRE_DOUX = colors.HexColor("#55606E")
NUIT       = colors.HexColor("#243B64")
ACCENT     = colors.HexColor("#B4713A")
FILET      = colors.HexColor("#D6D9DE")
FOND       = colors.HexColor("#F4F5F7")
FOND_CODE  = colors.HexColor("#EDEFF2")

ss = getSampleStyleSheet()

def st(name, **kw):
    kw.setdefault("parent", ss["Normal"])
    return ParagraphStyle(name, **kw)

TITRE = st("Titre", fontName="Times-Bold", fontSize=26, leading=30,
           textColor=NUIT, spaceAfter=4)
SOUS_TITRE = st("SousTitre", fontName="Times-Italic", fontSize=12.5, leading=16,
                textColor=ENCRE_DOUX, spaceAfter=2)
H1 = st("H1", fontName="Helvetica-Bold", fontSize=13, leading=16,
        textColor=NUIT, spaceBefore=15, spaceAfter=6, keepWithNext=1)
H2 = st("H2", fontName="Helvetica-Bold", fontSize=10.2, leading=13,
        textColor=ACCENT, spaceBefore=10, spaceAfter=4, keepWithNext=1)
CORPS = st("Corps", fontName="Times-Roman", fontSize=9.8, leading=13.4,
           textColor=ENCRE, alignment=TA_JUSTIFY, spaceAfter=5)
NOTE = st("Note", fontName="Times-Italic", fontSize=8.8, leading=12,
          textColor=ENCRE_DOUX, alignment=TA_JUSTIFY, spaceAfter=4)
CODE = st("Code", fontName="Courier-Bold", fontSize=8.9, leading=12.6,
          textColor=NUIT, leftIndent=7, spaceBefore=1, spaceAfter=1)
CELL = st("Cell", fontName="Times-Roman", fontSize=8.5, leading=11, textColor=ENCRE)
CELL_C = st("CellC", parent=CELL, alignment=1)
CELL_G = st("CellG", parent=CELL, fontName="Times-Bold")
CELL_H = st("CellH", fontName="Helvetica-Bold", fontSize=8.2, leading=10.5,
            textColor=colors.white, alignment=1)
MONO = st("Mono", fontName="Courier", fontSize=8.2, leading=11, textColor=ENCRE)
MONO_C = st("MonoC", parent=MONO, alignment=1)

LARGEUR = A4[0] - 36 * mm


def formule(txt):
    """Un bloc de formule, sur fond gris, sans debordement."""
    t = Table([[Paragraph(txt, CODE)]], colWidths=[LARGEUR])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), FOND_CODE),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LINEBEFORE", (0, 0), (0, -1), 2, ACCENT),
    ]))
    return t


def tableau(entetes, lignes, largeurs, styles_cellules=None, aligns=None):
    data = [[Paragraph(h, CELL_H) for h in entetes]]
    styles_cellules = styles_cellules or [CELL] * len(entetes)
    for ligne in lignes:
        data.append([Paragraph(str(c), styles_cellules[i]) for i, c in enumerate(ligne)])
    t = Table(data, colWidths=largeurs, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), NUIT),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, FILET),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, FOND]),
        ("BOX", (0, 0), (-1, -1), 0.6, FILET),
    ]
    t.setStyle(TableStyle(style))
    return t


def encadre(titre, texte, couleur=ACCENT):
    inner = [Paragraph(titre, st("bt", fontName="Helvetica-Bold", fontSize=8.8,
                                 leading=11, textColor=couleur, spaceAfter=3))]
    for p in texte:
        inner.append(Paragraph(p, st("bc", fontName="Times-Roman", fontSize=8.8,
                                     leading=11.8, textColor=ENCRE,
                                     alignment=TA_JUSTIFY, spaceAfter=3)))
    t = Table([[inner]], colWidths=[LARGEUR])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), FOND),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LINEBEFORE", (0, 0), (0, -1), 2.5, couleur),
    ]))
    return t


# ----------------------------------------------------------------- Contenu
S = []
A = S.append

A(Paragraph("CielScore", TITRE))
A(Paragraph("Formules et regles de scoring &mdash; reference d'implementation", SOUS_TITRE))
hr = Table([[""]], colWidths=[LARGEUR], rowHeights=[2])
hr.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), ACCENT)]))
A(hr)
A(Spacer(1, 8))

A(Paragraph(
    "Ce document decrit le moteur de calcul <b>tel qu'il est reellement code</b> dans "
    "l'application Android CielScore. Il est etabli a partir des sources "
    "(<font face='Courier'>Formulas.kt</font>, <font face='Courier'>ScoringEngine.kt</font>, "
    "<font face='Courier'>Twilight.kt</font>) et non d'une specification : lorsque le code "
    "et un document de reference divergent, c'est le comportement decrit ici qui est celui "
    "de l'application.", CORPS))

A(Paragraph(
    "Il consolide deux sources : la <b>documentation fonctionnelle et technique v0.6.4</b> "
    "pour l'ossature, et les <b>regles de scoring v2.0 du 09/09/2026</b> qui reecrivent la "
    "magnitude limite, le score de nuit et le traitement de la brillance de surface. Les "
    "apports de la v2.0 sont signales par la mention <b>[v2.0]</b>.", CORPS))

A(Spacer(1, 4))
A(tableau(
    ["", ""],
    [["Version de l'application", "0.7.0"],
     ["Regles de scoring", "v2.0 &mdash; 09/09/2026"],
     ["Implementation", "<font face='Courier'>com.cielscore.app.scoring</font>, "
                       "<font face='Courier'>com.cielscore.app.astro</font>"],
     ["Verification", "70 tests unitaires, dont 32 de conformite documentaire"]],
    [46 * mm, LARGEUR - 46 * mm],
    [CELL_G, CELL]))

# ---------------------------------------------------------------- 1
A(Paragraph("1. Constantes fondamentales", H1))

A(Paragraph("Brillance de surface limite du site", H2))
A(Paragraph(
    "L'indice de Bortle caracterise la pollution lumineuse du lieu. Il fixe la brillance du "
    "fond de ciel : un objet moins contraste que ce fond se noie dedans. La tolerance "
    "<font face='Courier'>SB_TOLERANCE</font> = <b>3,5 mag/arcsec<super>2</super></b> definit "
    "une zone de penalite progressive au-dela de cette limite &mdash; un objet y reste "
    "observable, avec une attenuation lineaire.", CORPS))

A(tableau(
    ["Bortle", "SB limite<br/>(mag/arcsec<super>2</super>)", "+ Tolerance",
     "Magnitude a<br/>l'oeil nu (NELM)", "Correction de<br/>pollution [v2.0]", "Description"],
    [["1", "22,0", "25,5", "7,6", "+1,2", "Ciel vierge exceptionnel"],
     ["2", "21,8", "25,3", "7,1", "+1,2", "Ciel vierge"],
     ["3", "21,5", "25,0", "6,6", "+0,6", "Ciel rural"],
     ["4", "21,0", "24,5", "6,1", "+0,6", "Transition rural / peri-urbain"],
     ["5", "20,5", "24,0", "5,6", "0,0", "Ciel peri-urbain"],
     ["6", "19,5", "23,0", "5,1", "0,0", "Banlieue lumineuse"],
     ["7", "18,5", "22,0", "4,6", "&minus;0,8", "Transition banlieue / ville"],
     ["8", "17,5", "21,0", "4,1", "&minus;0,8", "Ciel urbain"],
     ["9", "17,0", "20,5", "3,6", "&minus;1,0", "Centre-ville"]],
    [15 * mm, 24 * mm, 19 * mm, 24 * mm, 25 * mm, LARGEUR - 107 * mm],
    [CELL_C, CELL_C, CELL_C, CELL_C, CELL_C, CELL]))

A(Paragraph("Types resolus", H2))
A(Paragraph(
    "<b>[v2.0]</b> Les <b>amas ouverts</b> et les <b>amas globulaires</b> sont exemptes du "
    "critere de brillance de surface : ils se resolvent en etoiles individuelles, la notion "
    "de brillance moyenne y est sans objet. Pour eux, "
    "<font face='Courier'>s_sb = 100</font> et <font face='Courier'>f_sb = 1</font>.", CORPS))
A(encadre("Consequence sur les catalogues", [
    "Le type d'un objet decide donc de son score. Le type OpenNGC <font face='Courier'>Cl+N</font> "
    "&mdash; amas <i>avec nebulosite</i> &mdash; est pour cette raison rattache aux "
    "<b>nebuleuses</b>, et non aux amas : M42, IC 5146 (Cocoon), IC 1396, IC 1805 (le Coeur), "
    "IC 1848 (l'Ame) et 65 autres objets s'observent comme des nebuleuses, et c'est bien leur "
    "brillance de surface qui decide si on les voit.",
    "Les associations stellaires (<font face='Courier'>*Ass</font>) restent classees en amas "
    "ouverts : elles sont reellement resolues en etoiles."]))

# ---------------------------------------------------------------- 2
A(Paragraph("2. Magnitude limite &mdash; jusqu'ou porte l'instrument", H1))

A(Paragraph("Instrument visuel [v2.0]", H2))
A(formule("mag_lim = 2,1 + 5 &times; log10(D_mm) + correction_pollution(Bortle)"))
A(Paragraph(
    "Changement majeur de la v2.0 : la limite depend desormais du <b>ciel</b>, et plus "
    "seulement du diametre. Un 130 mm annonce 11,9 depuis un site Bortle 7 et 13,9 depuis un "
    "site Bortle 1. La regle RG-I-02 de la documentation v0.6.4, qui la faisait dependre du "
    "seul diametre, est <b>abrogee</b>.", CORPS))
A(Paragraph(
    "A l'oeil nu (D &le; 7 mm), la limite reste la magnitude visuelle du site (NELM), lue dans "
    "le tableau de la section 1.", CORPS))

A(Paragraph("Smart telescope, en astrophotographie [v2.0]", H2))
A(formule("mag_lim = 2,1 + 5 &times; log10(D_mm) + correction_pollution(Bortle)<br/>"
          "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; + 1,25 &times; log10(pose_minutes / 60)"))
A(Paragraph(
    "La duree de pose <b>cumulee</b> remplace la duree de session : c'est l'integration du "
    "capteur qui repousse la limite. Le terme s'annule a une heure de pose, vaut +0,38 a deux "
    "heures et +0,87 a cinq heures. La pose est exprimee <b>en minutes</b>.", CORPS))

A(Paragraph("Tableau de reference &mdash; magnitude limite visuelle", H2))
A(tableau(
    ["Diametre", "Bortle 1&ndash;2", "Bortle 3&ndash;4", "Bortle 5&ndash;6",
     "Bortle 7&ndash;8", "Bortle 9"],
    [["60 mm", "12,2", "11,6", "11,0", "10,2", "10,0"],
     ["114 mm", "13,6", "13,0", "12,4", "11,6", "11,4"],
     ["200 mm", "14,8", "14,2", "13,6", "12,8", "12,6"],
     ["300 mm", "15,6", "15,0", "14,4", "13,6", "13,4"],
     ["400 mm", "16,2", "15,6", "15,1", "14,2", "14,0"]],
    [30 * mm] + [(LARGEUR - 30 * mm) / 5] * 5,
    [CELL_G] + [CELL_C] * 5))
A(Paragraph(
    "Ce tableau <b>fait foi</b> : les exemples chiffres de la page 3 des regles v2.0 "
    "appliquent la formule sans la correction de pollution, alors meme qu'ils annoncent un "
    "indice de Bortle. Arbitrage valide. L'application le reproduit a 0,12 magnitude pres, "
    "soit la precision de ses propres arrondis.", NOTE))

# ---------------------------------------------------------------- 3
A(Paragraph("3. Brillance de surface &mdash; l'objet se detache-t-il du fond ?", H1))

A(Paragraph("Calcul", H2))
A(formule("SB = magnitude + 2,5 &times; log10( &pi; &times; (a/2) &times; (b/2) &times; 3600 )"))
A(Paragraph(
    "<font face='Courier'>a</font> et <font face='Courier'>b</font> sont les axes en "
    "arcminutes ; le facteur 3600 les convertit en arcsec<super>2</super>. Formule standard "
    "IAU. Si les dimensions angulaires sont inconnues, la brillance ne peut pas etre calculee : "
    "l'objet reste consultable dans l'Explorer mais sort du classement du tableau de bord.", CORPS))

A(Paragraph("Facteur f_sb", H2))
A(formule("f_sb = clip( (sb_lim + 3,5 &minus; sb) / 3,5 , 0 , 1 )"))
A(Paragraph(
    "Vaut 1 tant que l'objet est plus contraste que le fond de ciel, decroit lineairement dans "
    "la zone de tolerance, s'annule au-dela. Il <b>multiplie le score visuel final</b> &mdash; "
    "sauf pour les objets rattrapes par le correctif de la section 6, et jamais en mode smart "
    "telescope.", CORPS))

# ---------------------------------------------------------------- 4
A(Paragraph("4. Filtres d'exclusion &mdash; score = 0", H1))
A(Paragraph(
    "Toute condition remplie annule le score. Ces filtres sont evalues avant toute ponderation.", CORPS))
A(tableau(
    ["Code", "Condition", "Motif"],
    [["RG-F-01", "altitude &lt; 5&deg;",
      "Sous l'horizon pratique, ou noye dans la turbulence basse"],
     ["RG-F-03", "magnitude &gt; mag_lim",
      "Hors de portee de l'instrument (limite crepusculaire comprise, cf. section 8)"],
     ["RG-F-04", "nuages &gt; 90 %", "Ciel opaque, observation impossible"],
     ["RG-F-02 <b>[v2.0]</b>", "f_sb &le; 0 <b>ET</b> magnitude non accessible",
      "Objet diffus trop faible pour emerger du fond de ciel"],
     ["&mdash;", "dimensions angulaires inconnues",
      "Ni le filtre ni le critere de brillance ne sont evaluables"]],
    [30 * mm, 52 * mm, LARGEUR - 82 * mm],
    [CELL_C, MONO, CELL]))
A(Paragraph(
    "<b>[v2.0]</b> Le filtre RG-F-02 exigeait auparavant la seule condition "
    "<font face='Courier'>sb &gt; sb_lim</font>. Il faut desormais la <b>conjonction</b> des "
    "deux echecs : c'est ce qui ramene M31, M42 et M45 dans le classement.", NOTE))

# ---------------------------------------------------------------- 5
A(Paragraph("5. Sous-scores intermediaires (0 a 100)", H1))
A(tableau(
    ["Sous-score", "Formule", "Bornes"],
    [["s_alt &mdash; Altitude", "clip((alt &minus; 5) / 25, 0, 1) &times; 100",
      "5&deg; &rarr; 0 &nbsp;|&nbsp; 30&deg; &rarr; 100"],
     ["s_fen &mdash; Fenetre", "min(duree_min / 240, 1) &times; 100",
      "0 h &rarr; 0 &nbsp;|&nbsp; 4 h &rarr; 100"],
     ["s_see &mdash; Seeing", "(indice &minus; 1) / 4 &times; 100",
      "indice 1 &rarr; 0 &nbsp;|&nbsp; 5 &rarr; 100"],
     ["s_tra &mdash; Transparence", "100 &minus; nuages_%",
      "100 % nuages &rarr; 0"],
     ["s_bor &mdash; Bortle", "(9 &minus; bortle) / 8 &times; 100",
      "B9 &rarr; 0 &nbsp;|&nbsp; B1 &rarr; 100"],
     ["s_lune &mdash; Lune", "(1 &minus; phase/100 &times; (1 &minus; dist/180)) &times; 100",
      "pleine et proche &rarr; 0"],
     ["s_sb &mdash; Brillance <b>[v2.0]</b>", "voir section 6", "progressif"],
     ["s_nuit &mdash; Nuit astro. <b>[v2.0]</b>",
      "clip(&minus;alt_soleil / 18, 0, 1) &times; 100",
      "0&deg; &rarr; 0 &nbsp;|&nbsp; &minus;18&deg; &rarr; 100"]],
    [42 * mm, 62 * mm, LARGEUR - 104 * mm],
    [CELL_G, MONO, CELL]))

A(Paragraph("Precisions", H2))
A(Paragraph(
    "<b>Lune.</b> RG-L-01 &mdash; sous l'horizon, la Lune n'a aucun impact, quelle que soit sa "
    "phase : le score vaut 100. RG-L-02 &mdash; l'impact combine la phase et la distance "
    "angulaire a l'objet. RG-L-04 &mdash; nouvelle Lune : aucune penalite.", CORPS))
A(Paragraph(
    "<b>Fenetre.</b> Le document ne definissait pas <font face='Courier'>duree_min</font>. "
    "Retenu : le temps passe <b>au-dessus de 30&deg;</b> pendant que le Soleil est sous "
    "l'horizon, evalue sur les 10 heures suivant l'heure de session &mdash; la meme fenetre que "
    "la courbe d'altitude.", CORPS))
A(Paragraph(
    "<b>Score de nuit [v2.0].</b> Les quatre paliers de la v0.6.4 (100 / 70 / 40 / 10 / 0) "
    "faisaient sauter le score global d'une phase crepusculaire a l'autre. La progression est "
    "desormais continue.", CORPS))
A(Paragraph(
    "<b>Seeing.</b> L'indice, de 1 a 5, est deduit du vent : &lt; 5 km/h &rarr; 5 (excellent) ; "
    "&lt; 15 &rarr; 4 ; &lt; 25 &rarr; 3 ; &le; 40 &rarr; 2 ; au-dela &rarr; 1.", CORPS))

# ---------------------------------------------------------------- 6
A(Paragraph("6. Correctif des objets brillants etendus [v2.0]", H1))

A(encadre("Le defaut corrige", [
    "M31 (magnitude 3,4, etendue 190' &times; 60') a une brillance de surface calculee de "
    "22,2 mag/arcsec<super>2</super>, au-dela du plafond d'un ciel Bortle 7 (18,5 + 3,5 = 22,0). "
    "L'ancien facteur <font face='Courier'>f_sb</font> tombait a zero et <b>M31 obtenait un "
    "score nul</b> &mdash; alors qu'elle se voit a l'oeil nu.",
    "La cause est structurelle : la formule repartit le flux sur <i>toute</i> l'etendue "
    "angulaire, quand l'oeil, lui, integre le flux total. M42 et M45 subissaient le meme sort."],
    couleur=colors.HexColor("#A03D3D")))

A(Paragraph("Algorithme", H2))
A(formule(
    "mag_accessible = magnitude &lt; mag_lim &minus; 2<br/><br/>"
    "<b>si type resolu</b> (amas ouvert ou globulaire) :<br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;s_sb = 100 &nbsp;;&nbsp; f_sb = 1<br/><br/>"
    "<b>sinon</b> :<br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;s_sb_diff = clip((sb_lim + 3,5 &minus; sb) / 8,5, 0, 1) &times; 100<br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;<b>si</b> mag_accessible :<br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;s_sb_mag = clip((mag_lim &minus; magnitude) / 6, 0, 1) &times; 100<br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;poids&nbsp;&nbsp;&nbsp;&nbsp; = clip((mag_lim &minus; 2 &minus; magnitude) / (mag_lim &minus; 2), 0, 1)<br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;s_sb&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; = s_sb_diff &times; (1 &minus; poids) + s_sb_mag &times; poids<br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;f_sb&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; = 1&nbsp;&nbsp;&nbsp;&nbsp;<i>&larr; le multiplicateur est neutralise</i><br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;<b>sinon</b> : s_sb = s_sb_diff&nbsp;&nbsp;;&nbsp;&nbsp;f_sb conserve sa valeur"))

A(Paragraph(
    "Le troisieme temps est essentiel : sans la neutralisation de "
    "<font face='Courier'>f_sb</font>, le score d'un objet rattrape retomberait a zero malgre "
    "un sous-score correct.", CORPS))

A(Paragraph("Validation &mdash; Bortle 7, D = 114 mm, altitude 55&deg;", H2))
A(Paragraph(
    "Tous les criteres hors brillance valent 73 points dans ces conditions (fenetre pleine, "
    "seeing et transparence parfaits, nuit noire, Lune au plus mauvais) : seul "
    "<font face='Courier'>s_sb</font>, qui pese 15 %, distingue les objets.", CORPS))
A(tableau(
    ["Objet", "Magnitude", "Type", "Attendu", "Calcule"],
    [["M45 &mdash; Pleiades", "1,5", "Amas ouvert", "88", "88,0"],
     ["M42 &mdash; Orion", "3,7", "Nebuleuse", "83", "82,7"],
     ["M31 &mdash; Andromede", "3,4", "Galaxie", "83", "82,7"],
     ["M13 &mdash; Hercule", "5,8", "Amas globulaire", "88", "88,0"],
     ["M33 &mdash; Triangle", "5,7", "Galaxie", "80", "79,0"],
     ["M101 &mdash; Pinwheel", "7,9", "Galaxie", "76", "74,6"],
     ["NGC 891", "9,5", "Galaxie", "74", "73,0"]],
    [46 * mm, 22 * mm, 34 * mm, 22 * mm, LARGEUR - 124 * mm],
    [CELL_G, CELL_C, CELL, CELL_C, CELL_C]))
A(Paragraph(
    "Ecart maximal <b>1,4 point</b>, imputable aux arrondis a l'entier du document de reference "
    "et aux dimensions angulaires, qu'il ne fournit que pour certains objets.", NOTE))

# ---------------------------------------------------------------- 7
A(Paragraph("7. Ponderations finales", H1))

A(Paragraph("Score visuel &mdash; observateur a l'oculaire", H2))
A(tableau(
    ["Critere", "Poids", "Critere", "Poids"],
    [["s_alt &mdash; Altitude", "25 %", "s_bor &mdash; Bortle", "8 %"],
     ["s_fen &mdash; Fenetre disponible", "15 %", "s_lune &mdash; Lune", "6 %"],
     ["s_see &mdash; Seeing", "11 %", "s_sb &mdash; Brillance / magnitude", "15 %"],
     ["s_tra &mdash; Transparence", "13 %", "s_nuit &mdash; Nuit astronomique", "7 %"]],
    [(LARGEUR - 44 * mm) / 2, 22 * mm, (LARGEUR - 44 * mm) / 2, 22 * mm],
    [CELL, CELL_C, CELL, CELL_C]))
A(Spacer(1, 4))
A(formule("score_visuel = ( 0,25&middot;s_alt + 0,15&middot;s_fen + 0,11&middot;s_see + 0,13&middot;s_tra<br/>"
          "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; + 0,08&middot;s_bor + 0,06&middot;s_lune + 0,15&middot;s_sb + 0,07&middot;s_nuit ) &times; f_sb"))

A(Paragraph("Score smart telescope &mdash; astrophotographie", H2))
A(tableau(
    ["Critere", "Poids", "Critere", "Poids"],
    [["s_alt &mdash; Altitude", "25 %", "s_lune &mdash; Lune", "15 %"],
     ["s_tra &mdash; Transparence", "20 %", "s_fd &mdash; Vitesse optique F/D", "5 %"],
     ["s_see &mdash; Seeing", "15 %", "s_fov &mdash; Adequation du champ", "5 %"],
     ["s_bor &mdash; Bortle", "15 %", "", ""]],
    [(LARGEUR - 44 * mm) / 2, 22 * mm, (LARGEUR - 44 * mm) / 2, 22 * mm],
    [CELL, CELL_C, CELL, CELL_C]))
A(Spacer(1, 4))
A(formule("score_smart = 0,25&middot;s_alt + 0,20&middot;s_tra + 0,15&middot;s_see + 0,15&middot;s_bor<br/>"
          "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ 0,15&middot;s_lune + 0,05&middot;s_fd + 0,05&middot;s_fov"))
A(Paragraph(
    "<b>f_sb n'intervient pas</b> comme multiplicateur en mode smart telescope : la pose longue "
    "integre le flux. Le filtre d'exclusion, lui, reste applique en amont. Les planetes sont "
    "desactivees dans ce mode (RG-P-05).", CORPS))
A(Paragraph(
    "<b>F/D</b> : <font face='Courier'>clip((8 &minus; F/D) / 6, 0, 1) &times; 100</font> &mdash; "
    "100 a F/2, nul a partir de F/8. <b>Champ</b> : maximum lorsque l'objet occupe la moitie du "
    "champ, nul s'il est ponctuel ou plus grand que le champ. Ces deux formules sont des "
    "definitions retenues : le document enonce les principes sans les chiffrer.", NOTE))

A(Paragraph("Score des planetes et de la Lune", H2))
A(formule("score_planete = 0,40&middot;s_alt + 0,30&middot;s_mag + 0,20&middot;s_tra + 0,10&middot;s_lune"))
A(Paragraph(
    "avec <font face='Courier'>s_mag = clip((8 &minus; mag) / 8 &times; 80 + 20, 0, 100)</font>, "
    "et <font face='Courier'>s_lune = 100 &minus; phase</font> sauf lorsque l'objet <i>est</i> "
    "la Lune, auquel cas il vaut 100.", CORPS))
A(tableau(
    ["Regle", "Enonce"],
    [["RG-P-02", "La Lune est proposee des 2&deg; d'altitude, meme de jour ; les autres corps "
                 "restent soumis au seuil de 5&deg;"],
     ["RG-P-03", "Venus et Jupiter (mag &lt; &minus;1) apparaissent des &minus;3&deg; de hauteur "
                 "du Soleil"],
     ["RG-P-04", "Les autres planetes attendent le crepuscule nautique (&minus;6&deg;)"],
     ["RG-P-05", "Les planetes sont desactivees en mode smart telescope"]],
    [26 * mm, LARGEUR - 26 * mm],
    [CELL_C, CELL]))

# ---------------------------------------------------------------- 8
A(Paragraph("8. Filtrage crepusculaire dynamique", H1))
A(Paragraph(
    "La magnitude limite retenue pour le filtre RG-F-03 depend de l'obscurite reelle du moment. "
    "Ce filtrage de la documentation v0.6.4 est <b>conserve</b> et s'applique en amont du "
    "scoring v2.0 (arbitrage valide).", CORPS))
A(tableau(
    ["Hauteur du Soleil", "Phase", "Ciel profond", "Planetes"],
    [["&gt; 0&deg;", "Jour", "aucun", "Lune seule"],
     ["0&deg; a &minus;6&deg;", "Crepuscule civil", "aucun", "mag &lt; &minus;1 (Lune, Venus, Jupiter)"],
     ["&minus;6&deg; a &minus;12&deg;", "Crepuscule nautique", "mag &le; 2", "mag &le; 3"],
     ["&minus;12&deg; a &minus;18&deg;", "Crepuscule astronomique", "ouverture progressive", "toutes"],
     ["&lt; &minus;18&deg;", "Nuit noire", "limite de l'instrument", "toutes"]],
    [32 * mm, 42 * mm, 42 * mm, LARGEUR - 116 * mm],
    [CELL_C, CELL, CELL, CELL]))
A(Spacer(1, 4))
A(formule("ouverture progressive :&nbsp; fraction = (&minus;alt_soleil &minus; 12) / 6<br/>"
          "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; mag_lim = 3 + fraction &times; (mag_lim_instrument &minus; 3)"))
A(Paragraph(
    "C'est la magnitude limite <b>instrumentale</b>, et non cette limite crepusculaire, qui sert "
    "a juger qu'un objet est <font face='Courier'>mag_accessible</font> au sens de la section 6 : "
    "sans quoi un objet changerait de categorie au fil du crepuscule.", NOTE))

# ---------------------------------------------------------------- 9
A(Paragraph("9. Validation deterministe [v2.0]", H1))
A(Paragraph(
    "Un garde-fou plus severe que les filtres d'exclusion, applique <b>avant de soumettre les "
    "cibles a l'assistant IA</b>. Il ne retire rien du classement affiche : il evite seulement "
    "que l'assistant recommande un objet rasant l'horizon, noye dans les nuages ou colle a une "
    "Lune brillante.", CORPS))
A(tableau(
    ["Motif de rejet", "Seuil"],
    [["Trop bas sur l'horizon", "altitude &lt; 20&deg;"],
     ["Hors de portee instrumentale", "magnitude &gt; mag_lim"],
     ["Observation compromise", "nuages &gt; 80 %"],
     ["Eblouissement lunaire", "separation &lt; 30&deg; <b>ET</b> phase &gt; 60 %"],
     ["Fond de ciel trop lumineux", "SB &gt; sb_lim + 3,5&nbsp;&nbsp;(hors amas resolus)"]],
    [70 * mm, LARGEUR - 70 * mm],
    [CELL, MONO]))
A(Paragraph(
    "Si aucune cible ne passe cette validation, le classement complet est transmis malgre tout, "
    "plutot que de laisser l'assistant sans contexte.", NOTE))

# ---------------------------------------------------------------- 10
A(Paragraph("10. Formules d'instrument", H1))

A(Paragraph("Oculaires", H2))
A(tableau(
    ["Grandeur", "Formule"],
    [["Grossissement", "focale_instrument / focale_oculaire"],
     ["Champ reel (degres)", "champ_apparent / grossissement"],
     ["Pupille de sortie (mm)", "focale_oculaire / (F/D)"],
     ["Grossissement minimal utile", "D / pupille_oeil"],
     ["Grossissement ideal", "D / 3"],
     ["Grossissement maximal", "D &times; 1,5"],
     ["Rapport d'ouverture F/D", "focale / diametre"]],
    [58 * mm, LARGEUR - 58 * mm],
    [CELL_G, MONO]))

A(Paragraph("Astrophotographie", H2))
A(tableau(
    ["Grandeur", "Formule"],
    [["F/D effectif", "(focale &times; barlow) / diametre"],
     ["Echantillonnage (\"/pixel)", "(taille_pixel_um / focale_eff_mm) &times; 206,265"],
     ["Champ du capteur (degres)", "2 &times; atan( taille_capteur_mm / (2 &times; focale_eff_mm) )"],
     ["F/D minimal (Shannon)", "taille_pixel &times; 3,5"],
     ["F/D ideal", "taille_pixel &times; 5,0"],
     ["F/D maximal", "taille_pixel &times; 8,0"]],
    [58 * mm, LARGEUR - 58 * mm],
    [CELL_G, MONO]))
A(Paragraph(
    "Echantillonnage optimal en planetaire : entre 0,5 et 2,0 secondes d'arc par pixel. "
    "En deca, sur-echantillonnage &mdash; image molle, poses inutilement longues. Au-dela, "
    "sous-echantillonnage &mdash; perte de detail.", NOTE))

# ---------------------------------------------------------------- 11
A(Paragraph("11. Lecture du score", H1))
A(tableau(
    ["Score", "Interpretation", "Conduite a tenir"],
    [["85 &ndash; 100", "Conditions excellentes", "Cible prioritaire de la soiree"],
     ["70 &ndash; 84", "Tres favorable", "A programmer sans hesitation"],
     ["50 &ndash; 69", "Observable", "Correct, viser une meilleure altitude si possible"],
     ["25 &ndash; 49", "Difficile", "Reserve aux observateurs experimentes"],
     ["1 &ndash; 24", "Tres difficile", "Conditions marginales"],
     ["0", "Non observable", "Un filtre eliminatoire s'est declenche"]],
    [24 * mm, 46 * mm, LARGEUR - 70 * mm],
    [CELL_C, CELL_G, CELL]))

# ---------------------------------------------------------------- 12
A(Paragraph("12. Ou tout cela est implemente", H1))
A(tableau(
    ["Fichier", "Contenu"],
    [["scoring/Formulas.kt", "Toutes les formules pures : magnitude limite, brillance de "
                             "surface, seeing, oculaires, astrophoto, filtrage crepusculaire"],
     ["scoring/ScoringEngine.kt", "Filtres d'exclusion, sous-scores, ponderations, correctif "
                                  "des objets brillants etendus, validation deterministe"],
     ["astro/Twilight.kt", "Phases crepusculaires, score de nuit, fenetres nocturnes"],
     ["astro/AstroMath.kt", "Jour julien, temps sideral, conversion equatorial &rarr; horizontal, "
                            "separation angulaire"],
     ["astro/SolarSystem.kt", "Ephemerides du Soleil, de la Lune et des planetes "
                              "(Meeus ch. 25, 45, 47 ; elements keplerians JPL)"],
     ["catalog/SkyObject.kt", "Types d'objets et exemption des types resolus"],
     ["tools/build_catalogs.py", "Generation des catalogues depuis OpenNGC, correspondance "
                                 "des types"],
     ["tools/run_core_tests.sh", "Execution des 70 tests sans SDK Android"]],
    [56 * mm, LARGEUR - 56 * mm],
    [MONO, CELL]))

A(Spacer(1, 10))
A(encadre("Precision des ephemerides", [
    "Declinaison du Soleil : &lt; 0,01&deg;. &nbsp; Fraction illuminee de la Lune : &lt; 0,1 %. "
    "&nbsp; Position de la Lune : de l'ordre de la minute d'arc. &nbsp; Positions planetaires : "
    "quelques minutes d'arc (elements moyens JPL, valables de 1800 a 2050). &nbsp; Duree de la "
    "nuit astronomique : &lt; 1 minute.",
    "Largement suffisant pour un calcul d'altitude, de fenetre d'observation et de distance "
    "angulaire a la Lune."], couleur=NUIT))


# --------------------------------------------------------------- Rendu
def decor(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(ENCRE_DOUX)
    canvas.drawString(18 * mm, 12 * mm, "CielScore — Formules et regles de scoring")
    canvas.drawRightString(A4[0] - 18 * mm, 12 * mm, "Page %d" % doc.page)
    canvas.setStrokeColor(FILET)
    canvas.setLineWidth(0.5)
    canvas.line(18 * mm, 15.5 * mm, A4[0] - 18 * mm, 15.5 * mm)
    if doc.page > 1:
        canvas.setFont("Helvetica", 7.5)
        canvas.drawString(18 * mm, A4[1] - 13 * mm, "Version 0.7.0  ·  regles v2.0")
        canvas.line(18 * mm, A4[1] - 15.5 * mm, A4[0] - 18 * mm, A4[1] - 15.5 * mm)
    canvas.restoreState()


doc = BaseDocTemplate(OUT, pagesize=A4,
                      leftMargin=18 * mm, rightMargin=18 * mm,
                      topMargin=20 * mm, bottomMargin=20 * mm,
                      title="CielScore - Formules et regles de scoring",
                      author="CielScore", subject="Moteur de scoring, version 0.7.0")
frame = Frame(18 * mm, 20 * mm, LARGEUR, A4[1] - 40 * mm, id="corps")
doc.addPageTemplates([PageTemplate(id="std", frames=[frame], onPage=decor)])
doc.build(S)
print("ecrit :", OUT)
