"""Option photo : chargement, calage (2 clics ou WCS) et annotation des satellites."""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

ARCSEC = math.pi / 180 / 3600


@dataclass
class Photo:
    gray: np.ndarray                 # image en niveaux de gris, orientation d'affichage (y vers le bas)
    rgb: np.ndarray | None           # image couleur 8 bits si disponible
    date_obs: datetime | None        # heure de prise de vue (UTC si connue)
    date_source: str                 # « DATE-OBS », « EXIF », ou ""
    wcs: object | None = None        # astropy.wcs.WCS si l'en-tête FITS contient une solution
    fits_height: int | None = None   # pour convertir y FITS ↔ y affichage

    @property
    def shape(self):
        return self.gray.shape


def _parse_date(value: str) -> datetime | None:
    value = value.strip().replace("Z", "")
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
                "%Y:%m:%d %H:%M:%S", "%Y-%m-%d", "%d/%m/%y"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def load(data: bytes, filename: str) -> Photo:
    name = filename.lower()
    if name.endswith((".fits", ".fit", ".fts", ".fits.gz", ".fit.gz")):
        return _load_fits(data)
    return _load_raster(data)


def _load_fits(data: bytes) -> Photo:
    from astropy.io import fits
    from astropy.wcs import WCS

    with fits.open(io.BytesIO(data)) as hdul:
        hdu = next(h for h in hdul if h.data is not None and h.data.ndim >= 2)
        arr = np.asarray(hdu.data, dtype=float)
        header = hdu.header.copy()
    if arr.ndim == 3:   # cube couleur (3, ny, nx) ou (ny, nx, 3)
        arr = arr.mean(axis=0) if arr.shape[0] in (3, 4) else arr.mean(axis=-1)
    arr = np.nan_to_num(arr)
    date, src = None, ""
    for key in ("DATE-OBS", "DATE_OBS", "DATE-BEG"):
        if key in header:
            date = _parse_date(str(header[key]))
            if date and "T" not in str(header[key]) and "TIME-OBS" in header:
                t = _parse_date(f"{str(header[key])[:10]}T{header['TIME-OBS']}")
                date = t or date
            if date:
                src = key
                break
    wcs = None
    try:
        w = WCS(header, naxis=2)
        if w.has_celestial and w.wcs.ctype[0]:
            wcs = w
    except Exception:  # en-tête WCS absent ou invalide
        wcs = None
    # Convention FITS : première ligne en bas → retournement pour l'affichage
    return Photo(gray=np.flipud(arr), rgb=None, date_obs=date, date_source=src,
                 wcs=wcs, fits_height=arr.shape[0])


def _load_raster(data: bytes) -> Photo:
    from PIL import Image

    img = Image.open(io.BytesIO(data))
    date, src = None, ""
    try:
        exif = img.getexif()
        sub = exif.get_ifd(0x8769) if hasattr(exif, "get_ifd") else {}
        raw = sub.get(36867) or sub.get(36868) or exif.get(306)
        if raw:
            date = _parse_date(str(raw))
            src = "EXIF" if date else ""
    except Exception:
        pass
    if img.mode in ("I;16", "I;16B", "I", "F"):
        gray = np.asarray(img, dtype=float)
        rgb = None
    else:
        rgb = np.asarray(img.convert("RGB"))
        gray = rgb.astype(float).mean(axis=2)
    return Photo(gray=gray, rgb=rgb, date_obs=date, date_source=src)


def stretch(gray: np.ndarray, low: float = 0.5, high: float = 99.8) -> np.ndarray:
    """Étirement asinh entre deux centiles, renvoie une image 8 bits."""
    lo, hi = np.percentile(gray, [low, high])
    if hi <= lo:
        hi = lo + 1
    x = np.clip((gray - lo) / (hi - lo), 0, 1)
    x = np.arcsinh(10 * x) / np.arcsinh(10)
    return (x * 255).astype(np.uint8)


def display_image(photo: Photo) -> np.ndarray:
    return photo.rgb if photo.rgb is not None else stretch(photo.gray)


def detect_planet_center(gray: np.ndarray):
    """Barycentre des pixels au-dessus de la moitié du maximum (la planète domine l'image)."""
    g = gray - np.median(gray)
    mask = g > 0.5 * g.max()
    ys, xs = np.nonzero(mask)
    w = g[mask]
    return float(np.sum(xs * w) / np.sum(w)), float(np.sum(ys * w) / np.sum(w))


# ─── Calage par deux points ──────────────────────────────────────────────────

@dataclass
class Calibration:
    center: tuple          # pixel (x, y) du centre de la planète
    scale: float           # secondes d'arc par pixel
    rotation_deg: float    # rotation de l'image par rapport à « nord en haut, est à gauche »
    mirrored: bool

    def sky_to_pixel(self, east_arcsec: float, north_arcsec: float):
        u = east_arcsec if self.mirrored else -east_arcsec
        v = north_arcsec
        t = math.radians(self.rotation_deg)
        u2 = (u * math.cos(t) - v * math.sin(t)) / self.scale
        v2 = (u * math.sin(t) + v * math.cos(t)) / self.scale
        return self.center[0] + u2, self.center[1] - v2


def calibrate(center_px, ref_px, ref_east: float, ref_north: float, mirrored: bool) -> Calibration:
    """Échelle et rotation déduites du centre de la planète et d'un satellite identifié."""
    u = ref_east if mirrored else -ref_east
    v = ref_north
    du = ref_px[0] - center_px[0]
    dv = -(ref_px[1] - center_px[1])
    d_img = math.hypot(du, dv)
    if d_img < 1:
        raise ValueError("Le satellite de référence est trop près du centre de la planète.")
    scale = math.hypot(u, v) / d_img
    rot = math.degrees(math.atan2(dv, du) - math.atan2(v, u))
    return Calibration(tuple(center_px), scale, rot % 360.0, mirrored)


def wcs_pixel(photo: Photo, ra_rad: float, dec_rad: float):
    """Pixel d'affichage d'une position J2000 grâce à la solution WCS du FITS."""
    x, y = photo.wcs.all_world2pix([[math.degrees(ra_rad), math.degrees(dec_rad)]], 0)[0]
    return float(x), float(photo.fits_height - 1 - y)


def nearest_source(gray: np.ndarray, x: float, y: float, radius: int = 12, nsigma: float = 5.0):
    """Point lumineux le plus proche de (x, y) : barycentre autour du maximum local."""
    h, w = gray.shape
    x0, x1 = max(0, int(x) - radius), min(w, int(x) + radius + 1)
    y0, y1 = max(0, int(y) - radius), min(h, int(y) + radius + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    win = gray[y0:y1, x0:x1]
    bg = np.median(win)
    mad = np.median(np.abs(win - bg)) * 1.4826 + 1e-9
    iy, ix = np.unravel_index(np.argmax(win), win.shape)
    if win[iy, ix] - bg < nsigma * mad:
        return None
    sy0, sy1 = max(0, iy - 2), min(win.shape[0], iy + 3)
    sx0, sx1 = max(0, ix - 2), min(win.shape[1], ix + 3)
    sub = np.clip(win[sy0:sy1, sx0:sx1] - bg, 0, None)
    if sub.sum() <= 0:
        return None
    yy, xx = np.mgrid[sy0:sy1, sx0:sx1]
    return float(x0 + (xx * sub).sum() / sub.sum()), float(y0 + (yy * sub).sum() / sub.sum())


def annotate(photo: Photo, marks, title: str = "", signature: str = "", night: bool = False):
    """Image annotée. ``marks`` : liste de dict (name, x, y, found=(x, y)|None, coords=str)."""
    import matplotlib.pyplot as plt

    img = display_image(photo)
    h, w = photo.shape
    fig_w = 10
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * h / w), dpi=110)
    ax.imshow(img, cmap="gray" if img.ndim == 2 else None, origin="upper")
    color = "#ff3b30" if night else "#ffd60a"
    for mk in marks:
        x, y = mk["x"], mk["y"]
        if not (-0.1 * w < x < 1.1 * w and -0.1 * h < y < 1.1 * h):
            continue
        ax.plot(x, y, marker="o", ms=14, mfc="none", mec=color, mew=1)
        if mk.get("found"):
            fx, fy = mk["found"]
            ax.plot(fx, fy, "+", ms=8, color="#30d158" if not night else "#ff8a80", mew=1)
        ax.text(x + 10, y - 10, f"{mk['name']}\n{mk.get('coords', '')}".strip(), color=color,
                fontsize=8, va="bottom")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9)
    if signature:
        fig.text(0.99, 0.005, signature, ha="right", va="bottom", fontsize=7, color="#888888")
    fig.tight_layout()
    return fig
