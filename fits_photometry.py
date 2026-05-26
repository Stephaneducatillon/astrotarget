#!/usr/bin/env python3
"""
FITS Photométrie – Application Streamlit de traitement d'images astronomiques.
Fonctionnalités : calibration (bias/dark/flat), fond de ciel 2D,
détection de sources (DAOStarFinder), photométrie d'ouverture, export CSV/FITS.
"""

import io
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from astropy.io import fits
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.table import Table
from astropy.visualization import (ImageNormalize, LinearStretch, LogStretch,
                                   SqrtStretch, ZScaleInterval)
from photutils.aperture import (ApertureStats, CircularAnnulus,
                                 CircularAperture, aperture_photometry)
from photutils.background import Background2D, MedianBackground
from photutils.detection import DAOStarFinder

warnings.filterwarnings("ignore")

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FITS Photométrie",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .metric-label { font-size: 0.8rem !important; }
    .stTabs [data-baseweb="tab"] { font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Session state ─────────────────────────────────────────────────────────────
_STATE_KEYS = [
    "science_data", "science_header",
    "bias_data", "dark_data", "flat_data",
    "calibrated_data",
    "sources_raw", "sources_df",
    "background", "phot_table",
]
for _k in _STATE_KEYS:
    if _k not in st.session_state:
        st.session_state[_k] = None


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_fits(uploaded_file) -> tuple:
    """Return (data ndarray float64, header) from an uploaded FITS file."""
    if uploaded_file is None:
        return None, None
    try:
        raw = uploaded_file.read()
        with fits.open(io.BytesIO(raw)) as hdul:
            for hdu in hdul:
                if hdu.data is not None and hdu.data.ndim >= 2:
                    data = hdu.data.astype(np.float64)
                    if data.ndim == 3:          # cube → first plane
                        data = data[0]
                    return data, hdu.header
    except Exception as exc:
        st.error(f"Impossible de lire le fichier FITS : {exc}")
    return None, None


def calibrate(science, bias=None, dark=None, flat=None,
              exp_sci=1.0, exp_dark=1.0) -> np.ndarray:
    """Bias/dark/flat calibration. Returns calibrated array."""
    result = science.copy()
    if bias is not None:
        result -= bias
    if dark is not None:
        scale = exp_sci / exp_dark if exp_dark > 0 else 1.0
        result -= dark * scale
    if flat is not None:
        flat_norm = flat / np.median(flat)
        flat_norm = np.where(flat_norm > 0.1, flat_norm, 1.0)
        result /= flat_norm
    return result


def estimate_background(data, box_size=64, filter_size=3):
    """2-D background map using photutils. Falls back to global on failure."""
    sigma_clip = SigmaClip(sigma=3.0)
    try:
        bkg = Background2D(
            data,
            box_size=(box_size, box_size),
            filter_size=(filter_size, filter_size),
            sigma_clip=sigma_clip,
            bkg_estimator=MedianBackground(),
        )
        return bkg.background, bkg.background_rms
    except Exception:
        _, med, std = sigma_clipped_stats(data, sigma=3.0)
        return np.full_like(data, med), np.full_like(data, std)


def detect_stars(data, background, background_rms, fwhm=3.0, nsigma=5.0,
                 max_sources=1000):
    """DAOStarFinder source detection on background-subtracted image."""
    sub = data - background
    threshold = nsigma * np.median(background_rms)
    finder = DAOStarFinder(
        fwhm=fwhm,
        threshold=threshold,
        sharplo=0.2, sharphi=1.0,
        roundlo=-1.0, roundhi=1.0,
    )
    _, med, _ = sigma_clipped_stats(sub, sigma=3.0)
    sources = finder(sub - med)
    if sources is not None and len(sources) > max_sources:
        sources.sort("peak")
        sources.reverse()
        sources = sources[:max_sources]
    # Attach the FWHM parameter as a constant column (photutils ≥2 removed per-star FWHM)
    if sources is not None:
        sources["fwhm_det"] = fwhm
    return sources


def run_aperture_photometry(data, background, sources,
                             ap_r=5.0, ann_in=8.0, ann_out=12.0) -> pd.DataFrame:
    """Aperture photometry with local sky annulus subtraction."""
    if sources is None or len(sources) == 0:
        return None

    # photutils ≥2 uses x_centroid / y_centroid
    xcol = "x_centroid" if "x_centroid" in sources.colnames else "xcentroid"
    ycol = "y_centroid" if "y_centroid" in sources.colnames else "ycentroid"

    pos = np.column_stack([sources[xcol], sources[ycol]])
    apertures = CircularAperture(pos, r=ap_r)
    annuli = CircularAnnulus(pos, r_in=ann_in, r_out=ann_out)

    sub = data - background
    phot = aperture_photometry(sub, apertures)

    ann_stats = ApertureStats(sub, annuli, sigma_clip=SigmaClip(sigma=3.0))
    local_bkg = ann_stats.median * apertures.area

    phot["local_bkg"] = local_bkg
    phot["flux_net"] = phot["aperture_sum"] - local_bkg

    valid = phot["flux_net"] > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        phot["mag_inst"] = np.where(
            valid,
            -2.5 * np.log10(np.where(valid, phot["flux_net"], 1.0)),
            np.nan,
        )

    # fwhm_det is the detection FWHM (constant per run); fallback for older photutils
    fwhm_col = "fwhm_det" if "fwhm_det" in sources.colnames else (
        "fwhm" if "fwhm" in sources.colnames else None
    )
    phot["fwhm"]      = sources[fwhm_col] if fwhm_col else np.nan
    phot["sharpness"] = sources["sharpness"]
    phot["roundness"] = sources["roundness2"]
    phot["peak"]      = sources["peak"]
    with np.errstate(invalid="ignore", divide="ignore"):
        phot["snr"] = phot["flux_net"] / np.sqrt(np.abs(phot["aperture_sum"]))

    df = phot.to_pandas()
    # photutils ≥2 names aperture center columns x_center / y_center
    xctr = "x_center" if "x_center" in df.columns else "xcenter"
    yctr = "y_center" if "y_center" in df.columns else "ycenter"
    df = df.rename(columns={xctr: "X", yctr: "Y", "aperture_sum": "flux_brut"})
    for col in ["X", "Y", "flux_brut", "local_bkg", "flux_net",
                "mag_inst", "fwhm", "snr", "peak"]:
        if col in df.columns:
            df[col] = df[col].round(4)
    df = df.sort_values("mag_inst").reset_index(drop=True)
    df.index += 1
    return df


def render_image(data, title="Image", cmap="gray", stretch="zscale",
                 sources=None, ap_r=5.0, figsize=(9, 9)):
    """Return a matplotlib Figure of the FITS image with optional overlays."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(data)

    if stretch == "log":
        vmin = max(vmin, 1.0)
        norm = ImageNormalize(data, vmin=vmin, vmax=vmax, stretch=LogStretch())
    elif stretch == "sqrt":
        norm = ImageNormalize(data, vmin=vmin, vmax=vmax, stretch=SqrtStretch())
    else:
        norm = ImageNormalize(data, vmin=vmin, vmax=vmax, stretch=LinearStretch())

    ax.imshow(data, cmap=cmap, origin="lower", norm=norm, interpolation="nearest")
    ax.set_title(title, color="white", fontsize=13)
    ax.set_xlabel("X (pixels)", color="white")
    ax.set_ylabel("Y (pixels)", color="white")
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")

    if sources is not None and len(sources) > 0:
        for _, row in sources.iterrows():
            circ = plt.Circle((row["X"], row["Y"]), ap_r,
                               color="lime", fill=False, linewidth=0.7, alpha=0.75)
            ax.add_patch(circ)
        ax.set_title(f"{title}  ({len(sources)} sources)", color="white", fontsize=13)

    plt.tight_layout(pad=0.5)
    return fig


def dark_hist(values, xlabel, title, color="#4CAF50"):
    """Return a small dark-themed histogram figure."""
    fig, ax = plt.subplots(figsize=(7, 2.8))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#1a1a2e")
    ax.hist(values.dropna(), bins=40, color=color, edgecolor="none", alpha=0.85)
    ax.set_xlabel(xlabel, color="white", fontsize=10)
    ax.set_ylabel("N", color="white", fontsize=10)
    ax.set_title(title, color="white", fontsize=11)
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")
    plt.tight_layout(pad=0.4)
    return fig


# ─── App ──────────────────────────────────────────────────────────────────────

st.title("🔭 FITS Photométrie")
st.caption("Traitement d'images astronomiques · Calibration · Photométrie d'ouverture")

TABS = st.tabs([
    "📂 Chargement",
    "🔧 Calibration",
    "🖼️ Visualisation",
    "⭐ Détection",
    "📐 Photométrie",
    "📈 Résultats",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Chargement
# ══════════════════════════════════════════════════════════════════════════════
with TABS[0]:
    st.header("Chargement des images FITS")
    st.markdown(
        "Chargez une image scientifique (obligatoire) et des images de "
        "calibration optionnelles (bias, dark, flat)."
    )

    col_sci, col_cal = st.columns(2, gap="large")

    with col_sci:
        st.subheader("Image scientifique")
        sci_file = st.file_uploader(
            "Glissez votre image FITS ici",
            type=["fits", "fit", "fts"],
            key="up_science",
        )
        if sci_file:
            data, hdr = load_fits(sci_file)
            if data is not None:
                st.session_state.science_data = data
                st.session_state.science_header = hdr
                h, w = data.shape
                st.success(f"✅ Image chargée — {w} × {h} pixels")
                c1, c2, c3 = st.columns(3)
                c1.metric("Largeur", f"{w} px")
                c2.metric("Hauteur", f"{h} px")
                exptime = "N/A"
                if hdr:
                    exptime = hdr.get("EXPTIME", hdr.get("EXPOSURE", "N/A"))
                c3.metric("Exposition", f"{exptime} s")

                if hdr:
                    with st.expander("📋 En-tête FITS"):
                        rows = [(str(k), str(v)) for k, v in dict(hdr).items()
                                if str(k).strip() and str(k) not in ("COMMENT", "HISTORY", "")]
                        st.dataframe(
                            pd.DataFrame(rows, columns=["Clé", "Valeur"]),
                            use_container_width=True,
                            height=260,
                        )

    with col_cal:
        st.subheader("Frames de calibration (optionnel)")

        bias_file = st.file_uploader("Master Bias", type=["fits", "fit", "fts"], key="up_bias")
        if bias_file:
            d, _ = load_fits(bias_file)
            if d is not None:
                st.session_state.bias_data = d
                st.success(f"✅ Bias  {d.shape[1]}×{d.shape[0]}")

        dark_file = st.file_uploader("Master Dark", type=["fits", "fit", "fts"], key="up_dark")
        if dark_file:
            d, _ = load_fits(dark_file)
            if d is not None:
                st.session_state.dark_data = d
                st.success(f"✅ Dark  {d.shape[1]}×{d.shape[0]}")

        flat_file = st.file_uploader("Master Flat", type=["fits", "fit", "fts"], key="up_flat")
        if flat_file:
            d, _ = load_fits(flat_file)
            if d is not None:
                st.session_state.flat_data = d
                st.success(f"✅ Flat  {d.shape[1]}×{d.shape[0]}")

    if st.session_state.science_data is None:
        st.info(
            "💡 **Astuce :** Sans image réelle, vous pouvez générer une image synthétique "
            "pour tester l'application."
        )
        if st.button("🎲 Générer une image test (synthétique)"):
            rng = np.random.default_rng(42)
            size = 512
            sky = 500.0
            noise = rng.normal(0, 20, (size, size))
            img = np.full((size, size), sky) + noise
            # Ajouter ~80 étoiles synthétiques
            n_stars = 80
            xs = rng.integers(20, size - 20, n_stars)
            ys = rng.integers(20, size - 20, n_stars)
            fluxes = np.power(10, rng.uniform(2.5, 5.0, n_stars))
            yy, xx = np.mgrid[:size, :size]
            for x, y, flux in zip(xs, ys, fluxes):
                fwhm_px = rng.uniform(2.5, 4.5)
                sigma = fwhm_px / 2.355
                img += flux * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
            img = img.astype(np.float64)
            fake_hdr = fits.Header()
            fake_hdr["EXPTIME"] = 60.0
            fake_hdr["INSTRUME"] = "Synthetic"
            fake_hdr["NAXIS1"] = size
            fake_hdr["NAXIS2"] = size
            st.session_state.science_data = img
            st.session_state.science_header = fake_hdr
            st.success("✅ Image synthétique créée (512×512, ~80 étoiles)")
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Calibration
# ══════════════════════════════════════════════════════════════════════════════
with TABS[1]:
    st.header("Calibration de l'image")

    if st.session_state.science_data is None:
        st.warning("⚠️ Chargez d'abord une image scientifique (onglet **Chargement**).")
    else:
        has_bias = st.session_state.bias_data is not None
        has_dark = st.session_state.dark_data is not None
        has_flat = st.session_state.flat_data is not None
        has_any  = has_bias or has_dark or has_flat

        col_params, col_status = st.columns([2, 1], gap="large")

        with col_params:
            st.subheader("Paramètres")

            apply_bias = st.checkbox(
                "Soustraction du bias", value=has_bias, disabled=not has_bias,
                help="Soustrait le courant de lecture fixe du capteur."
            )
            apply_dark = st.checkbox(
                "Soustraction du dark", value=has_dark, disabled=not has_dark,
                help="Soustrait le courant thermique. Le dark est mis à l'échelle si besoin."
            )

            exp_sci = exp_dark = 1.0
            if apply_dark and has_dark:
                hdr = st.session_state.science_header
                default_exp = float(hdr.get("EXPTIME", hdr.get("EXPOSURE", 60.0))) if hdr else 60.0
                exp_sci = st.number_input(
                    "Temps de pose science (s)", value=default_exp, min_value=0.1, step=1.0
                )
                exp_dark = st.number_input(
                    "Temps de pose dark (s)", value=default_exp, min_value=0.1, step=1.0
                )

            apply_flat = st.checkbox(
                "Correction de flat-field", value=has_flat, disabled=not has_flat,
                help="Corrige les inhomogénéités de réponse du capteur et de l'optique."
            )

            if not has_any:
                st.info(
                    "ℹ️ Aucune image de calibration n'est chargée. "
                    "Vous pouvez tout de même passer à la visualisation."
                )

            apply_btn = st.button(
                "🔧 Appliquer la calibration", type="primary",
                disabled=not has_any,
            )
            no_cal_btn = st.button(
                "➡️ Utiliser l'image brute sans calibration",
                disabled=(st.session_state.calibrated_data is not None),
            )

            if apply_btn:
                with st.spinner("Calibration en cours…"):
                    result = calibrate(
                        st.session_state.science_data,
                        bias=st.session_state.bias_data if apply_bias else None,
                        dark=st.session_state.dark_data if apply_dark else None,
                        flat=st.session_state.flat_data if apply_flat else None,
                        exp_sci=exp_sci,
                        exp_dark=exp_dark,
                    )
                    st.session_state.calibrated_data = result
                    st.success("✅ Calibration appliquée avec succès !")

            if no_cal_btn:
                st.session_state.calibrated_data = st.session_state.science_data.copy()
                st.info("Image brute utilisée (aucune calibration appliquée).")

        with col_status:
            st.subheader("Statut des frames")
            frames = [
                ("Science",    st.session_state.science_data),
                ("Bias",       st.session_state.bias_data),
                ("Dark",       st.session_state.dark_data),
                ("Flat",       st.session_state.flat_data),
                ("Calibrée",   st.session_state.calibrated_data),
            ]
            for name, arr in frames:
                if arr is not None:
                    _, med, std = sigma_clipped_stats(arr, sigma=3.0)
                    st.markdown(f"**{name}** ✅  \nMéd: `{med:.1f}` · σ: `{std:.1f}`")
                else:
                    st.markdown(f"**{name}** — *non chargé*")
                st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Visualisation
# ══════════════════════════════════════════════════════════════════════════════
with TABS[2]:
    st.header("Visualisation de l'image")

    active = (st.session_state.calibrated_data
              if st.session_state.calibrated_data is not None
              else st.session_state.science_data)

    if active is None:
        st.warning("⚠️ Chargez d'abord une image scientifique.")
    else:
        ctrl, viewer = st.columns([1, 3], gap="medium")

        with ctrl:
            st.subheader("Affichage")
            cmap = st.selectbox(
                "Palette colorimétrique",
                ["gray", "inferno", "plasma", "viridis", "hot", "Blues_r", "afmhot"],
            )
            stretch = st.selectbox("Étirement", ["zscale", "sqrt", "log"])

            show_src = st.checkbox(
                "Afficher les sources",
                value=st.session_state.sources_df is not None,
            )
            ap_r_disp = 5.0
            if show_src and st.session_state.sources_df is not None:
                ap_r_disp = st.slider("Rayon d'affichage (px)", 2, 40, 5)

            st.subheader("Statistiques (σ-clip 3σ)")
            mn, med, std = sigma_clipped_stats(active, sigma=3.0)
            st.metric("Moyenne", f"{mn:.1f}")
            st.metric("Médiane", f"{med:.1f}")
            st.metric("Écart-type", f"{std:.1f}")
            st.metric("Min / Max",
                      f"{active.min():.0f} / {active.max():.0f}")

        with viewer:
            label = ("Image calibrée"
                     if st.session_state.calibrated_data is not None
                     else "Image brute")
            src_overlay = st.session_state.sources_df if show_src else None
            fig = render_image(active, title=label, cmap=cmap,
                               stretch=stretch, sources=src_overlay,
                               ap_r=ap_r_disp, figsize=(9, 9))
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Détection des sources
# ══════════════════════════════════════════════════════════════════════════════
with TABS[3]:
    st.header("Détection des sources stellaires")

    active = (st.session_state.calibrated_data
              if st.session_state.calibrated_data is not None
              else st.session_state.science_data)

    if active is None:
        st.warning("⚠️ Chargez d'abord une image scientifique.")
    else:
        col_p, col_r = st.columns([1, 2], gap="medium")

        with col_p:
            st.subheader("Paramètres DAOStarFinder")

            fwhm = st.slider(
                "FWHM estimée (pixels)", 1.0, 20.0, 3.0, 0.5,
                help="Largeur à mi-hauteur typique des étoiles dans l'image.",
            )
            nsigma = st.slider(
                "Seuil de détection (× σ fond)", 2.0, 15.0, 5.0, 0.5,
                help="Multiplier l'écart-type du fond pour définir le seuil.",
            )

            st.subheader("Fond de ciel 2D")
            box_size = st.slider(
                "Taille des boîtes (px)", 20, 256, 64, 4,
                help="Résolution spatiale de l'estimation du fond.",
            )

            max_src = st.number_input(
                "Nombre max de sources", min_value=10, max_value=5000,
                value=500, step=50,
            )

            detect_btn = st.button("⭐ Détecter les sources", type="primary")

        with col_r:
            if detect_btn:
                with st.spinner("Estimation du fond puis détection…"):
                    bg, bg_rms = estimate_background(active, box_size=box_size)
                    st.session_state.background = bg
                    sources = detect_stars(active, bg, bg_rms,
                                           fwhm=fwhm, nsigma=nsigma,
                                           max_sources=max_src)
                if sources is not None and len(sources) > 0:
                    st.session_state.sources_raw = sources
                    xcol = "x_centroid" if "x_centroid" in sources.colnames else "xcentroid"
                    ycol = "y_centroid" if "y_centroid" in sources.colnames else "ycentroid"
                    fwhm_col = "fwhm_det" if "fwhm_det" in sources.colnames else (
                        "fwhm" if "fwhm" in sources.colnames else None
                    )
                    df_s = pd.DataFrame({
                        "X":         np.round(sources[xcol], 2),
                        "Y":         np.round(sources[ycol], 2),
                        "FWHM":      np.round(sources[fwhm_col], 2) if fwhm_col else fwhm,
                        "Sharpness": np.round(sources["sharpness"], 3),
                        "Rondeur":   np.round(sources["roundness2"], 3),
                        "Pic (ADU)": np.round(sources["peak"], 1),
                    })
                    st.session_state.sources_df = df_s
                    st.success(f"✅ {len(sources)} sources détectées")
                else:
                    st.warning("Aucune source détectée — essayez de réduire le seuil.")
                    st.session_state.sources_raw = None
                    st.session_state.sources_df = None

            if st.session_state.sources_df is not None:
                df_s = st.session_state.sources_df
                st.subheader(f"{len(df_s)} sources")

                # FWHM histogram
                fig_fw = dark_hist(
                    df_s["FWHM"], "FWHM (pixels)",
                    "Distribution des FWHM", color="#4CAF50"
                )
                st.pyplot(fig_fw, use_container_width=True)
                plt.close(fig_fw)

                c1, c2, c3 = st.columns(3)
                c1.metric("FWHM médian", f"{df_s['FWHM'].median():.2f} px")
                c2.metric("FWHM min",    f"{df_s['FWHM'].min():.2f} px")
                c3.metric("FWHM max",    f"{df_s['FWHM'].max():.2f} px")

                st.dataframe(df_s, use_container_width=True, height=280)
            else:
                st.info("Lancez la détection pour voir les résultats ici.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Photométrie d'ouverture
# ══════════════════════════════════════════════════════════════════════════════
with TABS[4]:
    st.header("Photométrie d'ouverture")

    active = (st.session_state.calibrated_data
              if st.session_state.calibrated_data is not None
              else st.session_state.science_data)

    if active is None:
        st.warning("⚠️ Chargez d'abord une image scientifique.")
    elif st.session_state.sources_raw is None:
        st.warning("⚠️ Effectuez d'abord la **Détection des sources**.")
    else:
        col_p, col_r = st.columns([1, 2], gap="medium")

        with col_p:
            st.subheader("Géométrie des ouvertures")

            ap_r = st.slider("Rayon d'ouverture (px)", 2.0, 40.0, 5.0, 0.5)
            ann_in = st.slider("Anneau intérieur (px)", 4.0, 60.0, 8.0, 0.5)
            ann_out = st.slider("Anneau extérieur (px)", 6.0, 80.0, 12.0, 0.5)

            if ann_in <= ap_r:
                st.error("⚠️ L'anneau intérieur doit être supérieur au rayon d'ouverture.")

            # Schéma ASCII de l'ouverture
            st.markdown(
                f"""
```
  ┌──────────────────────────────┐
  │   anneau fond :              │
  │   r_in={ann_in:.1f}px r_out={ann_out:.1f}px │
  │     ╭────────────╮           │
  │     │  ouverture │           │
  │     │  r={ap_r:.1f}px   │           │
  │     ╰────────────╯           │
  └──────────────────────────────┘
```"""
            )

            st.subheader("Photons")
            use_gain = st.checkbox("Appliquer le gain CCD")
            gain = 1.0
            if use_gain:
                gain = st.number_input("Gain (e⁻/ADU)", value=1.0,
                                       min_value=0.01, step=0.1)

            phot_btn = st.button(
                "📐 Calculer la photométrie",
                type="primary",
                disabled=(ann_in <= ap_r),
            )

        with col_r:
            if phot_btn:
                with st.spinner("Photométrie en cours…"):
                    bg = (st.session_state.background
                          if st.session_state.background is not None
                          else estimate_background(active)[0])
                    phot_df = run_aperture_photometry(
                        active, bg, st.session_state.sources_raw,
                        ap_r=ap_r, ann_in=ann_in, ann_out=ann_out,
                    )

                if phot_df is not None:
                    if use_gain:
                        phot_df["flux_e"] = (phot_df["flux_net"] * gain).round(2)
                    st.session_state.phot_table = phot_df
                    st.success(f"✅ Photométrie calculée pour **{len(phot_df)}** sources")
                else:
                    st.error("Erreur lors du calcul de la photométrie.")

            if st.session_state.phot_table is not None:
                df = st.session_state.phot_table
                valid = df["mag_inst"].dropna()

                fig_mag = dark_hist(
                    valid, "Magnitude instrumentale",
                    "Distribution des magnitudes instrumentales", color="#FF9800"
                )
                # brighter → left
                fig_mag.axes[0].invert_xaxis()
                st.pyplot(fig_mag, use_container_width=True)
                plt.close(fig_mag)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Sources",      len(df))
                c2.metric("Mag. min",     f"{valid.min():.3f}")
                c3.metric("Mag. max",     f"{valid.max():.3f}")
                c4.metric("SNR médian",   f"{df['snr'].median():.1f}")
            else:
                st.info("Lancez la photométrie pour voir les résultats ici.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Résultats et export
# ══════════════════════════════════════════════════════════════════════════════
with TABS[5]:
    st.header("Résultats et export")

    if st.session_state.phot_table is None:
        st.warning(
            "⚠️ Effectuez d'abord la **Photométrie d'ouverture** "
            "(onglet **Photométrie**)."
        )
    else:
        df = st.session_state.phot_table
        valid = df["mag_inst"].dropna()

        # ── Métriques résumées ──────────────────────────────────────────────
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Étoiles mesurées",  len(df))
        c2.metric("Mag. plus brillante", f"{valid.min():.3f}")
        c3.metric("Mag. plus faible",    f"{valid.max():.3f}")
        c4.metric("SNR médian",          f"{df['snr'].median():.1f}")
        c5.metric("FWHM médian",         f"{df['fwhm'].median():.2f} px")

        st.divider()

        # ── Diagrammes ─────────────────────────────────────────────────────
        plot1, plot2 = st.columns(2, gap="medium")

        with plot1:
            st.subheader("Magnitude vs SNR")
            fig_sc, ax_sc = plt.subplots(figsize=(6, 4.5))
            fig_sc.patch.set_facecolor("#0e1117")
            ax_sc.set_facecolor("#1a1a2e")
            sc = ax_sc.scatter(
                df["mag_inst"], df["snr"],
                c=df["fwhm"], cmap="plasma",
                alpha=0.7, s=18, edgecolors="none",
            )
            cbar = fig_sc.colorbar(sc, ax=ax_sc)
            cbar.set_label("FWHM (px)", color="white")
            cbar.ax.yaxis.set_tick_params(color="white")
            plt.setp(cbar.ax.get_yticklabels(), color="white")
            ax_sc.set_xlabel("Magnitude instrumentale", color="white")
            ax_sc.set_ylabel("SNR", color="white")
            ax_sc.set_yscale("log")
            ax_sc.invert_xaxis()
            ax_sc.tick_params(colors="white")
            for sp in ax_sc.spines.values():
                sp.set_edgecolor("#444")
            plt.tight_layout(pad=0.5)
            st.pyplot(fig_sc, use_container_width=True)
            plt.close(fig_sc)

        with plot2:
            st.subheader("Flux brut vs Flux net")
            fig_fl, ax_fl = plt.subplots(figsize=(6, 4.5))
            fig_fl.patch.set_facecolor("#0e1117")
            ax_fl.set_facecolor("#1a1a2e")
            ax_fl.scatter(df["flux_brut"], df["flux_net"],
                          alpha=0.6, s=18, color="#29B6F6", edgecolors="none")
            lims = (min(df["flux_brut"].min(), df["flux_net"].min()) * 0.9,
                    max(df["flux_brut"].max(), df["flux_net"].max()) * 1.1)
            ax_fl.plot(lims, lims, "r--", linewidth=0.8, label="y = x")
            ax_fl.set_xlabel("Flux brut (ADU)", color="white")
            ax_fl.set_ylabel("Flux net (ADU)", color="white")
            ax_fl.tick_params(colors="white")
            ax_fl.legend(facecolor="#222", labelcolor="white", fontsize=9)
            for sp in ax_fl.spines.values():
                sp.set_edgecolor("#444")
            plt.tight_layout(pad=0.5)
            st.pyplot(fig_fl, use_container_width=True)
            plt.close(fig_fl)

        st.divider()

        # ── Tableau complet ─────────────────────────────────────────────────
        st.subheader("Catalogue photométrique")
        display_cols = ["X", "Y", "flux_net", "mag_inst", "snr", "fwhm", "peak"]
        avail = [c for c in display_cols if c in df.columns]
        st.dataframe(
            df[avail].style.format({
                "X": "{:.2f}", "Y": "{:.2f}",
                "flux_net": "{:.1f}", "mag_inst": "{:.4f}",
                "snr": "{:.1f}", "fwhm": "{:.2f}", "peak": "{:.1f}",
            }),
            use_container_width=True,
            height=360,
        )

        st.divider()

        # ── Export ──────────────────────────────────────────────────────────
        st.subheader("Télécharger les résultats")
        exp1, exp2, exp3 = st.columns(3)

        with exp1:
            buf_csv = io.StringIO()
            df.to_csv(buf_csv, index=True, float_format="%.6f")
            st.download_button(
                "⬇️ CSV",
                data=buf_csv.getvalue(),
                file_name="photometrie.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with exp2:
            buf_tsv = io.StringIO()
            df.to_csv(buf_tsv, index=True, sep="\t", float_format="%.6f")
            st.download_button(
                "⬇️ TSV (Tableur)",
                data=buf_tsv.getvalue(),
                file_name="photometrie.tsv",
                mime="text/tab-separated-values",
                use_container_width=True,
            )

        with exp3:
            buf_fits = io.BytesIO()
            tbl = Table.from_pandas(df.reset_index(drop=True))
            tbl.write(buf_fits, format="fits", overwrite=True)
            buf_fits.seek(0)
            st.download_button(
                "⬇️ FITS (table binaire)",
                data=buf_fits.getvalue(),
                file_name="photometrie.fits",
                mime="application/octet-stream",
                use_container_width=True,
            )
