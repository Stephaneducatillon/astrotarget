"""CielScore Satellites — quel satellite se trouve où autour d'une planète.

Lancement : ``streamlit run app.py``
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo, available_timezones

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

from satellites import data, engine, exports, phenomena, photo, render
from satellites.data import SIGNATURE, Instrument, Site

st.set_page_config(page_title="CielScore Satellites", page_icon="🪐", layout="wide")

NIGHT_CSS = """
<style>
.stApp, [data-testid="stSidebar"], [data-testid="stHeader"] { background: #000 !important; }
.stApp *, [data-testid="stSidebar"] * { color: #ff3b30 !important; border-color: #5a0f0a !important; }
input, textarea, select, [data-baseweb="select"] > div, [data-baseweb="input"] > div,
button, [data-testid="stFileUploaderDropzone"] { background: #140000 !important; }
table, th, td { background: #000 !important; }
img { filter: grayscale(1) sepia(1) saturate(6) hue-rotate(-50deg) brightness(0.75); }
</style>
"""


# ─── État : instant courant ──────────────────────────────────────────────────

def _now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _tz() -> ZoneInfo:
    return ZoneInfo(st.session_state.get("tz", data.DEFAULT_TZ))


def _set_inputs(when_utc: datetime):
    """Recopie un instant UTC dans les champs date / heure (selon le mode de saisie)."""
    shown = when_utc if st.session_state.get("input_ut", True) else when_utc.astimezone(_tz())
    st.session_state["d"] = shown.date()
    st.session_state["t"] = shown.strftime("%H:%M:%S")
    st.session_state["offset"] = 0


def _read_inputs() -> datetime:
    d: date = st.session_state["d"]
    txt = str(st.session_state["t"]).strip()
    try:
        parts = [int(p) for p in txt.split(":")]
        while len(parts) < 3:
            parts.append(0)
        t = time(*parts[:3])
    except (ValueError, TypeError):
        st.sidebar.error("Heure invalide : utilisez HH:MM:SS")
        t = time(0, 0, 0)
    if st.session_state.get("input_ut", True):
        return datetime.combine(d, t, tzinfo=timezone.utc)
    return datetime.combine(d, t, tzinfo=_tz()).astimezone(timezone.utc)


def _shift(delta: timedelta):
    _set_inputs(_read_inputs() + timedelta(minutes=st.session_state.get("offset", 0)) + delta)


def _now():
    _set_inputs(_now_utc())


def _toggle_input_mode():
    # La case vient de changer : on réécrit les champs dans le nouveau mode
    prev_ut = not st.session_state["input_ut"]
    d: date = st.session_state["d"]
    try:
        parts = [int(p) for p in str(st.session_state["t"]).split(":")] + [0, 0]
        t = time(*parts[:3])
    except ValueError:
        t = time(0, 0)
    tz = timezone.utc if prev_ut else _tz()
    when = datetime.combine(d, t, tzinfo=tz).astimezone(timezone.utc)
    shown = when if st.session_state["input_ut"] else when.astimezone(_tz())
    st.session_state["d"] = shown.date()
    st.session_state["t"] = shown.strftime("%H:%M:%S")


def _use_photo_time(dt: datetime):
    _set_inputs(dt)


if "d" not in st.session_state:
    st.session_state["input_ut"] = True
    st.session_state["tz"] = data.DEFAULT_TZ
    _set_inputs(_now_utc())


# ─── Barre latérale : recherche ──────────────────────────────────────────────

sb = st.sidebar
sb.title("🪐 CielScore Satellites")
night = sb.toggle("Mode nuit (rouge)", key="night")
if night:
    st.markdown(NIGHT_CSS, unsafe_allow_html=True)

planet_name = sb.selectbox("Planète", list(data.PLANETS), index=0)
planet = data.PLANETS[planet_name]

sb.subheader("Date et heure")
sb.toggle("Saisie en temps universel (UT)", key="input_ut", on_change=_toggle_input_mode)
c1, c2 = sb.columns(2)
c1.date_input("Date", key="d", min_value=date(data.YEAR_MIN, 1, 1), max_value=date(data.YEAR_MAX, 12, 31),
              format="DD/MM/YYYY")
c2.text_input("Heure (HH:MM:SS)", key="t")
sb.button("Maintenant", on_click=_now, width="stretch")
b = sb.columns([1, 1, 1.5])
b[0].button("−1 j", on_click=_shift, args=(timedelta(days=-1),), width="stretch")
b[1].button("−1 h", on_click=_shift, args=(timedelta(hours=-1),), width="stretch")
b[2].button("−10 min", on_click=_shift, args=(timedelta(minutes=-10),), width="stretch")
b = sb.columns([1.5, 1, 1])
b[0].button("+10 min", on_click=_shift, args=(timedelta(minutes=10),), width="stretch")
b[1].button("+1 h", on_click=_shift, args=(timedelta(hours=1),), width="stretch")
b[2].button("+1 j", on_click=_shift, args=(timedelta(days=1),), width="stretch")

zones = sorted(z for z in available_timezones() if "/" in z and not z.startswith(("Etc", "SystemV")))
sb.selectbox("Fuseau horaire (affichage)", zones, key="tz")
tz_name = st.session_state["tz"]

sb.subheader("Lieu (facultatif)")
site_choice = sb.selectbox("Observatoire", ["Aucun (géocentrique)"] + list(data.SITES) + ["Coordonnées libres"],
                           index=1)
site: Site | None
if site_choice == "Aucun (géocentrique)":
    site = None
elif site_choice == "Coordonnées libres":
    c1, c2, c3 = sb.columns(3)
    lat = c1.number_input("Latitude", -90.0, 90.0, 45.0, format="%.4f")
    lon = c2.number_input("Longitude (E+)", -180.0, 180.0, 5.0, format="%.4f")
    elev = c3.number_input("Altitude (m)", -100.0, 6000.0, 200.0)
    site = Site("Coordonnées libres", lat, lon, elev)
else:
    site = data.SITES[site_choice]

sb.subheader("Vue")
orient_label = sb.selectbox("Orientation de l'image", list(data.ORIENTATIONS))
rot, mirror = data.ORIENTATIONS[orient_label]
if rot is None:
    rot = sb.slider("Rotation (°)", -180, 180, 0)
    mirror = sb.checkbox("Miroir gauche-droite", value=False)

inst_choice = sb.selectbox("Instrument", ["Aucun"] + list(data.INSTRUMENTS) + ["Personnalisé"], index=1)
instrument: Instrument | None
if inst_choice == "Aucun":
    instrument = None
elif inst_choice == "Personnalisé":
    c1, c2 = sb.columns(2)
    focal = c1.number_input("Focale (mm)", 100.0, 50000.0, 13000.0)
    pix = c2.number_input("Pixel (µm)", 0.5, 30.0, 3.76)
    sw = c1.number_input("Capteur L (mm)", 1.0, 60.0, 23.5)
    sh = c2.number_input("Capteur H (mm)", 1.0, 60.0, 15.7)
    instrument = Instrument("Instrument personnalisé", focal, sw, sh, pix)
else:
    instrument = data.INSTRUMENTS[inst_choice]
frame_angle = sb.slider("Angle du cadre (°)", -90, 90, 0) if instrument else 0

zoom = sb.radio("Étendue du schéma", ["Tous les satellites", "Champ de l'instrument", "En rayons planétaires"],
                horizontal=False)
zoom_radii = sb.slider("Rayons planétaires", 2, 80, 12) if zoom == "En rayons planétaires" else None
mag_limit = sb.slider("Magnitude limite", 5.0, 20.0, 18.0, 0.5,
                      help="Masque les satellites plus faibles (≈15 pour un amateur).")

with sb.expander("Critères « observable »"):
    min_alt = st.slider("Hauteur minimale de la planète (°)", -5, 30, 0)
    max_sun = st.slider("Hauteur maximale du Soleil (°)", -18.0, 0.0, -0.8, 0.1,
                        help="−0,8° : Soleil couché ; −6° : fin du crépuscule civil ; −12° : nautique.")


# ─── Calcul ──────────────────────────────────────────────────────────────────

base = _read_inputs()
offset = st.session_state.get("offset", 0)
when = base + timedelta(minutes=offset)
local = when.astimezone(ZoneInfo(tz_name))

st.markdown(f"## {planet.name} et ses satellites")
st.caption(f"**{when:%d/%m/%Y %H:%M:%S} UT** · {local:%d/%m/%Y %H:%M:%S} {local.tzname()} "
           f"({tz_name}) · {site.name if site else 'géocentrique'} · {SIGNATURE}")

if not (data.YEAR_MIN <= when.year <= data.YEAR_MAX):
    st.warning(f"Date hors de la période prévue ({data.YEAR_MIN}–{data.YEAR_MAX}) : "
               "les positions PyEphem restent calculées mais n'ont pas été validées.")

snap = engine.compute(planet, when, site)
info = snap.info
if snap.unavailable:
    st.warning("Position non disponible pour : " + ", ".join(snap.unavailable)
               + " (les tables de satellites de PyEphem s'arrêtent début 2040).")
if info.reduced_precision:
    st.warning("Après 2040, PyEphem utilise une théorie de secours : précision réduite "
               "(quelques secondes d'arc).")

if site is not None and info.alt is not None and info.alt < 0:
    st.info(f"{planet.name} est sous l'horizon ({info.alt:.1f}°) à cet instant.")

# 2. Tableau des satellites
st.markdown("### Tableau des satellites")
df = exports.moons_table(snap, mag_limit)


def show_df(frame: pd.DataFrame):
    if night:
        st.markdown(frame.to_html(index=False, na_rep="—"), unsafe_allow_html=True)
    else:
        st.dataframe(frame, hide_index=True, width="stretch")


main_cols = ["Satellite", "État", "Mag ≈", "AD J2000", "Déc J2000", "Hauteur", "Azimut",
             "Écart E-O", "Écart N-S", "Distance", "Angle de position"]
detail = st.toggle("Toutes les coordonnées (AD/Déc du jour, rayons planétaires)", value=False)
if site is None:
    main_cols = [c for c in main_cols if c not in ("Hauteur", "Azimut")]
show_df(df if detail else df[main_cols])
hidden = sum(1 for m in snap.moons if m.mag > mag_limit)
if hidden:
    st.caption(f"{hidden} satellite(s) plus faible(s) que la magnitude {mag_limit:g} masqué(s).")

# 3. Schéma
st.markdown("### Schéma")
extent = None
if zoom == "Champ de l'instrument" and instrument is not None:
    fw, fh = instrument.field_arcsec
    extent = max(fw, fh) / 2 * 1.1
elif zoom_radii:
    extent = zoom_radii * info.radius_arcsec
fig = render.draw(snap, rotation_deg=rot, mirror=mirror, instrument=instrument,
                  frame_angle_deg=frame_angle, extent_arcsec=extent, mag_limit=mag_limit,
                  night=night, tz_name=tz_name, orientation_label=orient_label.split(":")[0])
st.pyplot(fig, width="stretch")
st.slider("Curseur de temps (minutes autour de l'heure saisie)", -720, 720, key="offset", step=5,
          help="Fait défiler l'heure pour préparer la prise de vue.")

if instrument is not None:
    fw, fh = instrument.field_arcsec
    outside = [m.name for m in snap.moons
               if m.mag <= mag_limit and (abs(m.east_arcsec) > fw / 2 or abs(m.north_arcsec) > fh / 2)]
    st.caption(f"{instrument.name} : champ {fw / 60:.1f}′ × {fh / 60:.1f}′ "
               f"(demi-largeur {fw / 2:.0f}″), {instrument.scale_arcsec_px:.3f}″/pixel. "
               + (f"Hors du cadre centré (cadre non tourné) : {', '.join(outside)}." if outside
                  else "Tous les satellites affichés sont dans le cadre centré."))

# 4. Infos planète
st.markdown("### Infos planète")


def _fmt_t(t: datetime | None) -> str:
    if t is None:
        return "—"
    return f"{t.astimezone(ZoneInfo(tz_name)):%d/%m %H:%M} ({t:%H:%M} UT)"


cols = st.columns(4)
if site is not None:
    cols[0].metric("Hauteur", f"{info.alt:.1f}°", help=f"Azimut {info.az:.1f}°")
cols[1].metric("Diamètre apparent", f"{info.diameter_arcsec:.2f}″")
cols[2].metric("Magnitude", f"{info.mag:.2f}")
cols[3].metric("Distance", f"{info.earth_distance_au:.3f} ua")
lines = [f"AD J2000 {engine.format_ra(info.ra_j2000)}, Déc {engine.format_dec(info.dec_j2000)}",
         f"angle de position du pôle nord {info.pole_pa:.1f}°",
         f"latitude de la Terre vue de {planet.name} {info.earth_lat:+.2f}°"]
if info.ring_tilt is not None:
    lines.append(f"inclinaison des anneaux {info.ring_tilt:+.2f}°")
if info.cml_ii is not None:
    lines.append(f"méridien central : système I {info.cml_i:.1f}°, système II {info.cml_ii:.1f}°")
if site is not None:
    if info.circumpolar:
        lines.append("planète circumpolaire")
    elif info.never_up:
        lines.append("planète jamais levée")
    else:
        lines.append(f"lever {_fmt_t(info.rise)}, passage au méridien {_fmt_t(info.transit_time)} "
                     f"(hauteur {info.transit_alt:.1f}°), coucher {_fmt_t(info.set)}")
    lines.append(f"Soleil à {info.sun_alt:+.1f}°")
st.markdown(" · ".join(lines))

# 5. Phénomènes
st.markdown("### Phénomènes de la nuit")


@st.cache_data(show_spinner="Calcul des phénomènes…", max_entries=64)
def _phenomena(planet_name: str, start: datetime, end: datetime, site: Site | None,
               min_alt: float, max_sun: float):
    return phenomena.find_phenomena(data.PLANETS[planet_name], start, end, site,
                                    min_planet_alt=min_alt, max_sun_alt=max_sun)


n_start, n_end = phenomena.night_window(when, tz_name)
phs = _phenomena(planet.name, n_start, n_end, site, float(min_alt), float(max_sun))
st.caption(f"Du {n_start.astimezone(ZoneInfo(tz_name)):%d/%m %H:%M} au "
           f"{n_end.astimezone(ZoneInfo(tz_name)):%d/%m %H:%M} ({tz_name}). "
           "Observable : planète levée et Soleil couché (critères réglables dans la barre latérale).")
ph_df = exports.phenomena_table(phs, tz_name)
if ph_df.empty:
    st.write("Aucun passage, occultation, éclipse ni ombre sur la période.")
else:
    show_df(ph_df)

# 6. Exports
st.markdown("### Exports")
e1, e2, e3 = st.columns(3)
stamp = f"{planet.key.lower()}_{when:%Y%m%d_%H%M%S}UT"
e1.download_button("CSV des coordonnées", exports.moons_csv(snap, mag_limit, tz_name),
                   file_name=f"cielscore_{stamp}.csv", mime="text/csv", width="stretch")
e2.download_button("Schéma PNG", exports.figure_png(fig), file_name=f"cielscore_{stamp}.png",
                   mime="image/png", width="stretch")
if e3.button("Préparer le PDF", width="stretch"):
    pdf_fig = render.draw(snap, rotation_deg=rot, mirror=mirror, instrument=instrument,
                          frame_angle_deg=frame_angle, extent_arcsec=extent, mag_limit=mag_limit,
                          tz_name=tz_name, orientation_label=orient_label.split(":")[0])
    st.session_state["pdf"] = exports.report_pdf(snap, pdf_fig, phs, mag_limit, tz_name, lines)
    plt.close(pdf_fig)
if st.session_state.get("pdf"):
    e3.download_button("Télécharger le PDF", st.session_state["pdf"], file_name=f"cielscore_{stamp}.pdf",
                       mime="application/pdf", width="stretch")
if not ph_df.empty:
    e1.download_button("CSV des phénomènes", ph_df.to_csv(sep=";", index=False).encode("utf-8-sig"),
                       file_name=f"cielscore_phenomenes_{stamp}.csv", mime="text/csv",
                       width="stretch")
plt.close(fig)

# 7. Option photo
st.markdown("### Identifier les satellites sur une photo (option)")
with st.expander("Charger une photo (FITS, TIFF, PNG, JPEG)", expanded=False):
    st.caption("La photo est traitée par l'application en cours d'exécution ; lancée en local, "
               "elle ne quitte pas votre ordinateur.")
    up = st.file_uploader("Photo", type=["fits", "fit", "fts", "tif", "tiff", "png", "jpg", "jpeg"])
    if up is not None:
        key = f"{up.name}-{up.size}"
        if st.session_state.get("photo_key") != key:
            st.session_state["photo_key"] = key
            st.session_state["photo"] = photo.load(up.getvalue(), up.name)
            st.session_state.pop("p_center", None)
            st.session_state.pop("p_ref", None)
        ph: photo.Photo = st.session_state["photo"]
        h, w = ph.shape

        if ph.date_obs:
            exif_local = False
            if ph.date_source == "EXIF":
                exif_local = st.checkbox("L'heure EXIF est en heure locale (sinon UT)", value=True)
            dt = ph.date_obs
            if exif_local:
                dt = dt.replace(tzinfo=ZoneInfo(tz_name)).astimezone(timezone.utc)
            st.write(f"Heure lue dans le fichier ({ph.date_source}) : **{dt:%d/%m/%Y %H:%M:%S} UT**")
            if abs((dt - when).total_seconds()) > 1:
                st.button("Calculer pour l'heure de la photo", on_click=_use_photo_time, args=(dt,))
        else:
            st.write("Aucune heure trouvée dans le fichier : saisissez-la dans la barre latérale.")

        marks = []
        cal = None
        if ph.wcs is not None:
            st.success("Solution astrométrique (WCS) trouvée dans l'en-tête FITS : calage automatique.")
            use_wcs = st.checkbox("Utiliser le WCS", value=True)
        else:
            use_wcs = False

        if not use_wcs:
            st.markdown("**Calage** : cliquez sur le centre de la planète, puis sur un satellite reconnu.")
            visible = [m for m in snap.moons if m.state == engine.VISIBLE and m.mag <= mag_limit]
            names = [m.name for m in visible] or [m.name for m in snap.moons]
            c1, c2, c3 = st.columns(3)
            if st.session_state.pop("advance_target", False):
                st.session_state["click_target"] = "le satellite de référence"
            target = c1.radio("Le prochain clic place…", ["le centre de la planète", "le satellite de référence"],
                              key="click_target")
            default_ref = "Titan" if "Titan" in names else names[-1]
            ref_name = c2.selectbox("Satellite de référence", names, index=names.index(default_ref))
            mirrored = c3.checkbox("Image inversée (miroir)", value=bool(mirror))
            if c3.button("Détecter le centre automatiquement"):
                st.session_state["p_center"] = photo.detect_planet_center(ph.gray)
                st.session_state["advance_target"] = True
                st.rerun()

            disp = Image.fromarray(photo.display_image(ph)).convert("RGB")
            draw = ImageDraw.Draw(disp)
            rr = max(6, w // 120)
            for k, col in (("p_center", (80, 160, 255)), ("p_ref", (255, 214, 10))):
                if k in st.session_state:
                    x, y = st.session_state[k]
                    draw.ellipse([x - rr, y - rr, x + rr, y + rr], outline=col, width=max(1, rr // 4))
            disp_w = min(w, 900)
            try:
                from streamlit_image_coordinates import streamlit_image_coordinates
                click = streamlit_image_coordinates(disp, width=disp_w, key=f"click-{key}")
                if click and click != st.session_state.get("last_click"):
                    st.session_state["last_click"] = click
                    k = w / click.get("width", disp_w)
                    pt = (click["x"] * k, click["y"] * k)
                    if target.startswith("le centre"):
                        st.session_state["p_center"] = pt
                        st.session_state["advance_target"] = True
                    else:
                        st.session_state["p_ref"] = pt
                    st.rerun()
            except ImportError:
                st.image(disp, width=disp_w)
                st.caption("Installez « streamlit-image-coordinates » pour placer les points d'un clic.")

            c1, c2, c3, c4 = st.columns(4)
            pc = st.session_state.get("p_center", (w / 2, h / 2))
            pr = st.session_state.get("p_ref", (w / 2 + 50, h / 2))
            cx = c1.number_input("Centre x (px)", value=float(pc[0]), format="%.1f")
            cy = c2.number_input("Centre y (px)", value=float(pc[1]), format="%.1f")
            rx = c3.number_input(f"{ref_name} x (px)", value=float(pr[0]), format="%.1f")
            ry = c4.number_input(f"{ref_name} y (px)", value=float(pr[1]), format="%.1f")
            if "p_center" in st.session_state and "p_ref" in st.session_state:
                ref = next(m for m in snap.moons if m.name == ref_name)
                try:
                    cal = photo.calibrate((cx, cy), (rx, ry), ref.east_arcsec, ref.north_arcsec, mirrored)
                    st.write(f"Échelle déduite : **{cal.scale:.3f}″/pixel**, rotation **{cal.rotation_deg:.1f}°** "
                             f"(par rapport à « nord en haut, est à gauche »)"
                             + (f" — instrument choisi : {instrument.scale_arcsec_px:.3f}″/pixel"
                                if instrument else ""))
                except ValueError as exc:
                    st.error(str(exc))

        if use_wcs or cal is not None:
            rows = []
            for m in snap.moons:
                if m.mag > mag_limit:
                    continue
                if use_wcs:
                    x, y = photo.wcs_pixel(ph, m.ra_j2000, m.dec_j2000)
                    scale = None
                else:
                    x, y = cal.sky_to_pixel(m.east_arcsec, m.north_arcsec)
                    scale = cal.scale
                found = photo.nearest_source(ph.gray, x, y) if (0 <= x < w and 0 <= y < h) else None
                coords = f"{engine.format_ra_short(m.ra_j2000)} {engine.format_dec_short(m.dec_j2000)}"
                marks.append(dict(name=m.name, x=x, y=y, found=found, coords=coords))
                dpx = None if found is None else float(np.hypot(found[0] - x, found[1] - y))
                if use_wcs and found is not None:
                    try:
                        scale = float(np.mean(np.abs(ph.wcs.proj_plane_pixel_scales()))) * 3600
                    except Exception:
                        scale = None
                rows.append({
                    "Satellite": m.name, "État": m.state,
                    "x calculé": round(x, 1), "y calculé": round(y, 1),
                    "Point le plus proche": "—" if found is None else f"({found[0]:.1f}, {found[1]:.1f})",
                    "Écart (px)": None if dpx is None else round(dpx, 2),
                    "Écart (″)": None if (dpx is None or scale is None) else round(dpx * scale, 2),
                    "AD J2000": engine.format_ra(m.ra_j2000), "Déc J2000": engine.format_dec(m.dec_j2000),
                })
            afig = photo.annotate(ph, marks, title=f"{planet.name} — {when:%d/%m/%Y %H:%M:%S} UT",
                                  signature=SIGNATURE, night=night)
            st.pyplot(afig, width="stretch")
            pdf_rows = pd.DataFrame(rows)
            show_df(pdf_rows)
            d1, d2 = st.columns(2)
            d1.download_button("Photo annotée (PNG)", exports.figure_png(afig),
                               file_name=f"cielscore_photo_{stamp}.png", mime="image/png")
            d2.download_button("Coordonnées (CSV)", pdf_rows.to_csv(sep=";", index=False).encode("utf-8-sig"),
                               file_name=f"cielscore_photo_{stamp}.csv", mime="text/csv")
            plt.close(afig)

st.divider()
st.caption(f"{SIGNATURE}. Positions des 17 grands satellites (Jupiter, Saturne, Uranus) "
           "calculées avec PyEphem ; positions géocentriques, indépendantes de la météo. "
           "Application gratuite, publique, sans compte.")
