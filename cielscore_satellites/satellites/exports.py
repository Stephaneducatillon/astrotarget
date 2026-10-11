"""Tableaux et exports : CSV des coordonnées, PNG du schéma, PDF (tableau + schéma)."""

from __future__ import annotations

import io
from zoneinfo import ZoneInfo

import pandas as pd

from .data import SIGNATURE
from .engine import Snapshot, format_dec, format_ra


def _ew(v: float) -> str:
    return f"{abs(v):.1f}″ {'E' if v >= 0 else 'O'}"


def _ns(v: float) -> str:
    return f"{abs(v):.1f}″ {'N' if v >= 0 else 'S'}"


def moons_table(snap: Snapshot, mag_limit: float = 99.0) -> pd.DataFrame:
    """Tableau affiché : une ligne par satellite, triée par distance à la planète."""
    rows = []
    for m in snap.moons:
        if m.mag > mag_limit:
            continue
        rows.append({
            "Satellite": m.name,
            "État": m.state + (" · ombre sur le disque" if m.shadow else ""),
            "Mag ≈": round(m.mag, 1),
            "AD J2000": format_ra(m.ra_j2000),
            "Déc J2000": format_dec(m.dec_j2000),
            "AD du jour": format_ra(m.ra_app),
            "Déc du jour": format_dec(m.dec_app),
            "Hauteur": None if m.alt is None else f"{m.alt:.2f}°",
            "Azimut": None if m.az is None else f"{m.az:.2f}°",
            "Écart E-O": _ew(m.east_arcsec),
            "Écart N-S": _ns(m.north_arcsec),
            "E-O (rayons)": round(m.east_radii, 3),
            "N-S (rayons)": round(m.north_radii, 3),
            "Distance": f"{m.dist_arcsec:.1f}″",
            "Distance (rayons)": round(m.dist_radii, 2),
            "Angle de position": f"{m.pa_deg:.1f}°",
        })
    return pd.DataFrame(rows)


def moons_csv(snap: Snapshot, mag_limit: float = 99.0, tz_name: str = "Europe/Paris") -> bytes:
    """CSV numérique (degrés décimaux, secondes d'arc), séparateur « ; »."""
    local = snap.when.astimezone(ZoneInfo(tz_name))
    rows = []
    for m in snap.moons:
        if m.mag > mag_limit:
            continue
        rows.append({
            "planete": snap.info.name,
            "satellite": m.name,
            "date_ut": snap.when.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "date_locale": local.isoformat(timespec="seconds"),
            "lieu": snap.site.name if snap.site else "géocentrique",
            "etat": m.state,
            "ombre_sur_disque": bool(m.shadow),
            "mag_approx": round(m.mag, 2),
            "ad_j2000_deg": round(m.ra_j2000 * 180 / 3.141592653589793, 7),
            "dec_j2000_deg": round(m.dec_j2000 * 180 / 3.141592653589793, 7),
            "ad_j2000": format_ra(m.ra_j2000, 2),
            "dec_j2000": format_dec(m.dec_j2000, 1),
            "ad_jour_deg": round(m.ra_app * 180 / 3.141592653589793, 7),
            "dec_jour_deg": round(m.dec_app * 180 / 3.141592653589793, 7),
            "hauteur_deg": None if m.alt is None else round(m.alt, 3),
            "azimut_deg": None if m.az is None else round(m.az, 3),
            "ecart_est_arcsec": round(m.east_arcsec, 2),
            "ecart_nord_arcsec": round(m.north_arcsec, 2),
            "ecart_est_rayons": round(m.east_radii, 4),
            "ecart_nord_rayons": round(m.north_radii, 4),
            "profondeur_rayons": round(m.depth_radii, 4),
            "distance_arcsec": round(m.dist_arcsec, 2),
            "angle_position_deg": round(m.pa_deg, 2),
        })
    buf = io.StringIO()
    buf.write(f"# {SIGNATURE}\n")
    pd.DataFrame(rows).to_csv(buf, sep=";", index=False)
    return buf.getvalue().encode("utf-8-sig")


def phenomena_table(phenomena, tz_name: str = "Europe/Paris") -> pd.DataFrame:
    tz = ZoneInfo(tz_name)

    def fmt(t, local):
        if t is None:
            return "en cours"
        return t.astimezone(tz).strftime("%d/%m %H:%M:%S") if local else t.strftime("%d/%m %H:%M:%S")

    def obs(v):
        return {True: "oui", False: "non", None: "—"}[v]

    rows = []
    for ph in phenomena:
        rows.append({
            "Satellite": ph.moon,
            "Phénomène": ph.label,
            "Début (UT)": fmt(ph.start, False),
            "Début (local)": fmt(ph.start, True),
            "Fin (UT)": fmt(ph.end, False),
            "Fin (local)": fmt(ph.end, True),
            "Observable début": obs(ph.start_observable),
            "Observable fin": obs(ph.end_observable),
        })
    return pd.DataFrame(rows)


def figure_png(fig, dpi: int = 160) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=fig.get_facecolor())
    return buf.getvalue()


def _register_fonts():
    """DejaVu Sans (fournie avec matplotlib) pour les caractères ″ ′ − absents d'Helvetica."""
    import os

    import matplotlib
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    if "DejaVu" not in pdfmetrics.getRegisteredFontNames():
        ttf = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
        pdfmetrics.registerFont(TTFont("DejaVu", os.path.join(ttf, "DejaVuSans.ttf")))
        pdfmetrics.registerFont(TTFont("DejaVu-Bold", os.path.join(ttf, "DejaVuSans-Bold.ttf")))
    return "DejaVu", "DejaVu-Bold"


def report_pdf(snap: Snapshot, fig, phenomena=None, mag_limit: float = 99.0,
               tz_name: str = "Europe/Paris", planet_lines=None) -> bytes:
    """PDF A4 paysage : schéma, tableau des satellites, phénomènes de la nuit."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                    Table, TableStyle)

    font, bold = _register_fonts()
    out = io.BytesIO()
    doc = SimpleDocTemplate(out, pagesize=landscape(A4), leftMargin=1.2 * cm, rightMargin=1.2 * cm,
                            topMargin=1.2 * cm, bottomMargin=1.2 * cm,
                            title=f"CielScore Satellites — {snap.info.name}", author="CielScore")
    styles = getSampleStyleSheet()
    for st in styles.byName.values():
        st.fontName = bold if "Heading" in st.name or st.name == "Title" else font
    small = styles["BodyText"].clone("small", fontSize=8, leading=10)

    def footer(canvas, _doc):
        canvas.saveState()
        canvas.setFont(font, 7)
        canvas.setFillColor(colors.grey)
        canvas.drawRightString(landscape(A4)[0] - 1.2 * cm, 0.6 * cm, SIGNATURE)
        canvas.restoreState()

    local = snap.when.astimezone(ZoneInfo(tz_name))
    site = snap.site.name if snap.site else "géocentrique"
    story = [
        Paragraph(f"CielScore Satellites — {snap.info.name}", styles["Title"]),
        Paragraph(f"{snap.when:%d/%m/%Y %H:%M:%S} UT ({local:%d/%m/%Y %H:%M:%S} {local.tzname()}) · {site}",
                  styles["BodyText"]),
        Spacer(1, 0.3 * cm),
    ]
    png = io.BytesIO(figure_png(fig, dpi=150))
    w_fig, h_fig = fig.get_size_inches()
    width = 25 * cm
    story.append(Image(png, width=width, height=width * h_fig / w_fig))
    if planet_lines:
        story.append(Paragraph(" · ".join(planet_lines), small))
    story.append(PageBreak())

    df = moons_table(snap, mag_limit)
    cols = ["Satellite", "État", "Mag ≈", "AD J2000", "Déc J2000", "AD du jour", "Déc du jour",
            "Hauteur", "Azimut", "Écart E-O", "Écart N-S", "Distance", "Angle de position"]
    data = [cols] + [[("" if v is None else str(v)) for v in row] for row in df[cols].values.tolist()]
    t = Table(data, repeatRows=1)
    t.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), font, 7.5),
        ("FONT", (0, 0), (-1, 0), bold, 7.5),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f2f0ea")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story += [Paragraph("Coordonnées des satellites", styles["Heading2"]), t]

    if phenomena is not None:
        story.append(Spacer(1, 0.5 * cm))
        story.append(Paragraph("Phénomènes de la nuit", styles["Heading2"]))
        pdf_ph = phenomena_table(phenomena, tz_name)
        if pdf_ph.empty:
            story.append(Paragraph("Aucun phénomène sur la période.", styles["BodyText"]))
        else:
            data = [list(pdf_ph.columns)] + pdf_ph.values.tolist()
            t2 = Table(data, repeatRows=1)
            t2.setStyle(TableStyle([
                ("FONT", (0, 0), (-1, -1), font, 7.5),
                ("FONT", (0, 0), (-1, 0), bold, 7.5),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f2f0ea")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
            ]))
            story.append(t2)

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return out.getvalue()
