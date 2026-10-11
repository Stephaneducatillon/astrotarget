"""Schéma de la vue : planète au centre, satellites à l'échelle, nom et coordonnées."""

from __future__ import annotations

import math

import numpy as np
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Ellipse, Rectangle  # noqa: E402
from matplotlib.transforms import Affine2D  # noqa: E402

from .data import SIGNATURE, Instrument  # noqa: E402
from .engine import (ECLIPSED, OCCULTED, TRANSIT, Snapshot,  # noqa: E402
                     format_dec_short, format_ra_short)

DAY = dict(bg="white", fg="#1a1a1a", muted="#666666", planet="#dbe8f7", planet_edge="#2f6fd0",
           ring="#c9b98f", moon="#111111", hidden="#999999", shadow="#000000",
           frame="#d04a2f", accent="#2f6fd0")
NIGHT = dict(bg="#000000", fg="#ff3b30", muted="#a3261f", planet="#3a0000", planet_edge="#ff3b30",
             ring="#7a1a14", moon="#ff3b30", hidden="#6b1510", shadow="#000000",
             frame="#ff6a55", accent="#ff3b30")


class View:
    """Passage des coordonnées du ciel (est, nord) aux coordonnées du schéma."""

    def __init__(self, rotation_deg: float = 0.0, mirror: bool = False):
        self.rot = math.radians(rotation_deg)
        self.mirror = mirror

    def __call__(self, east: float, north: float):
        x = east if self.mirror else -east   # ciel : est à gauche
        y = north
        c, s = math.cos(self.rot), math.sin(self.rot)
        return x * c - y * s, x * s + y * c

    def angle(self, pa_deg: float) -> float:
        """Angle (degrés, sens trigo sur le schéma) d'une direction d'angle de position ``pa``."""
        x, y = self(math.sin(math.radians(pa_deg)), math.cos(math.radians(pa_deg)))
        return math.degrees(math.atan2(y, x))


def _nice_scale(span: float) -> float:
    for v in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000):
        if v >= span:
            return v
    return 1000


def draw(snap: Snapshot, rotation_deg: float = 0.0, mirror: bool = False,
         instrument: Instrument | None = None, frame_angle_deg: float = 0.0,
         frame_center=(0.0, 0.0), extent_arcsec: float | None = None,
         mag_limit: float = 99.0, night: bool = False, tz_name: str = "Europe/Paris",
         orientation_label: str = "", show_coords: bool = True, figsize=(10, 7)):
    pal = NIGHT if night else DAY
    view = View(rotation_deg, mirror)
    info = snap.info
    r = info.radius_arcsec
    moons = [m for m in snap.moons if m.mag <= mag_limit]

    fig, ax = plt.subplots(figsize=figsize, dpi=110)
    fig.patch.set_facecolor(pal["bg"])
    ax.set_facecolor(pal["bg"])
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Étendue du schéma
    if extent_arcsec is None:
        far = max([m.dist_arcsec for m in moons] + [r * (info.ring_outer or 1.0)])
        if instrument is not None:
            fw, fh = instrument.field_arcsec
            far = max(far, math.hypot(fw, fh) / 2 + math.hypot(*frame_center))
        extent_arcsec = far * 1.18 + r
    ext = extent_arcsec
    ax.set_xlim(-ext * 1.15, ext * 1.15)
    ax.set_ylim(-ext * 0.80, ext * 0.80)

    # Pôle de la planète dans le repère (est, nord, vers la Terre)
    pa, de = math.radians(info.pole_pa), math.radians(info.earth_lat)
    pole = np.array([math.sin(pa) * math.cos(de), math.cos(pa) * math.cos(de), math.sin(de)])
    e1 = np.cross(pole, [0.0, 0.0, 1.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(pole, e1)
    phi = np.linspace(0, 2 * np.pi, 361)

    def ring(radius):
        pts3 = radius * r * (np.outer(np.cos(phi), e1) + np.outer(np.sin(phi), e2))
        xy = np.array([view(p_[0], p_[1]) for p_ in pts3])
        return xy, pts3[:, 2] > 0

    eq_angle = view.angle(info.pole_pa) - 90.0
    rings = None
    if info.ring_outer:
        outer, near = ring(info.ring_outer)
        inner, _ = ring(1.53)
        rings = (outer, inner, near)
        ax.fill(np.r_[outer[:, 0], inner[::-1, 0]], np.r_[outer[:, 1], inner[::-1, 1]],
                fc=pal["ring"], ec="none", alpha=0.35, zorder=1)
        ax.plot(outer[:, 0], outer[:, 1], color=pal["ring"], lw=1.2, zorder=1)
        ax.plot(inner[:, 0], inner[:, 1], color=pal["ring"], lw=0.7, zorder=1)

    # Disque de la planète (aplati, axe polaire orienté)
    ax.add_patch(Ellipse((0, 0), 2 * r, 2 * r * info.axis_ratio, angle=eq_angle,
                         fc=pal["planet"], ec=pal["planet_edge"], lw=1.5, zorder=2))

    # Moitié des anneaux située devant le disque
    if rings is not None:
        outer, inner, near = rings
        # indices de la moitié avant, rendus contigus (passage par φ = 0)
        start = (int(np.flatnonzero(~near)[-1]) + 1) % len(phi)
        idx = np.roll(np.arange(len(phi)), -start)
        idx = idx[near[idx]]
        if idx.size:
            ax.fill(np.r_[outer[idx, 0], inner[idx[::-1], 0]],
                    np.r_[outer[idx, 1], inner[idx[::-1], 1]],
                    fc=pal["ring"], ec="none", alpha=0.55, zorder=3)
            ax.plot(outer[idx, 0], outer[idx, 1], color=pal["ring"], lw=1.2, zorder=3)
            ax.plot(inner[idx, 0], inner[idx, 1], color=pal["ring"], lw=0.7, zorder=3)

    # Ombres sur le disque
    for m in moons:
        if m.shadow is not None:
            sx, sy = view(m.shadow[0] * r, m.shadow[1] * r)
            ax.plot(sx, sy, "o", ms=4, color=pal["shadow"], mec=pal["planet_edge"], mew=0.5, zorder=4)

    # Cadre du champ de l'instrument
    if instrument is not None:
        fw, fh = instrument.field_arcsec
        cx, cy = view(*frame_center)
        rect = Rectangle((cx - fw / 2, cy - fh / 2), fw, fh, fill=False, ec=pal["frame"],
                         lw=1.2, ls="--", zorder=3)
        rect.set_transform(Affine2D().rotate_deg_around(cx, cy, frame_angle_deg) + ax.transData)
        ax.add_patch(rect)
        ax.text(cx + fw / 2, cy + fh / 2, f" {instrument.name}\n {fw / 60:.1f}′ × {fh / 60:.1f}′",
                color=pal["frame"], fontsize=7, va="bottom", ha="left", zorder=6,
                transform=Affine2D().rotate_deg_around(cx, cy, frame_angle_deg) + ax.transData)

    # Titres (avant le placement des étiquettes, qui dépend de la taille des axes)
    when_local = snap.when.astimezone(ZoneInfo(tz_name))
    site = snap.site.name if snap.site else "géocentrique"
    ax.set_title(f"{info.name} et ses satellites — {snap.when:%d/%m/%Y %H:%M:%S} UT "
                 f"({when_local:%H:%M} {when_local.tzname()})",
                 color=pal["fg"], fontsize=11, loc="left", pad=16)
    ax.text(0, 1.01, f"{orientation_label or 'Vue du ciel'} · {site} · AD et Déc J2000 sous chaque nom",
            transform=ax.transAxes, color=pal["muted"], fontsize=8, va="bottom")
    fig.text(0.99, 0.01, SIGNATURE, color=pal["muted"], fontsize=7, ha="right", va="bottom")
    fig.tight_layout()

    # Échelle données / points typographiques
    bbox = ax.get_position()
    fw_in, fh_in = fig.get_size_inches()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    # aspect égal : la plus contraignante des deux dimensions fixe l'échelle
    per_pt = max((x1 - x0) / (bbox.width * fw_in * 72), (y1 - y0) / (bbox.height * fh_in * 72))

    # Satellites
    fs = 7.5
    ro = r * max(1.0, info.ring_outer * 0.95)
    obstacles = [(-ro, -r * 1.05, ro, r * 1.05)]  # disque (et anneaux)
    pts = []
    for m in moons:
        x, y = view(m.east_arcsec, m.north_arcsec)
        pts.append((m, x, y))
        d = 4 * per_pt
        obstacles.append((x - d, y - d, x + d, y + d))

    def overlaps(box):
        return any(not (box[2] < o[0] or box[0] > o[2] or box[3] < o[1] or box[1] > o[3])
                   for o in obstacles)

    def place(x, y, lines, size, gap):
        """Cherche une place libre pour l'étiquette ; trace un trait de rappel si elle est loin."""
        w = max(len(t) for t in lines) * size * 0.56 * per_pt
        h = len(lines) * size * 1.3 * per_pt
        for k in range(10):
            far = gap + k * h * 0.9
            for cx, cy in ((x, y - far - h / 2), (x, y + far + h / 2),
                           (x + far + w / 2, y), (x - far - w / 2, y),
                           (x + far * 0.7 + w / 2, y - far * 0.7 - h / 2),
                           (x - far * 0.7 - w / 2, y + far * 0.7 + h / 2),
                           (x + far * 0.7 + w / 2, y + far * 0.7 + h / 2),
                           (x - far * 0.7 - w / 2, y - far * 0.7 - h / 2)):
                box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
                if not overlaps(box):
                    obstacles.append(box)
                    return cx, cy, k > 0
        cx, cy = x, y - gap - h / 2
        obstacles.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
        return cx, cy, False

    # Nom de la planète
    cx, cy, _ = place(0.0, 0.0, [info.name], 10, r * 1.05 + 3 * per_pt)
    ax.text(cx, cy, info.name, color=pal["fg"], fontsize=10, fontweight="bold",
            ha="center", va="center", zorder=6)

    for m, x, y in sorted(pts, key=lambda p: p[0].dist_arcsec):
        ms = max(2.5, 8.5 - 0.45 * m.mag)
        if m.state == OCCULTED:
            ax.plot(x, y, "o", ms=ms, mfc="none", mec=pal["hidden"], zorder=5)
        elif m.state == ECLIPSED:
            ax.plot(x, y, "o", ms=ms, mfc=pal["bg"], mec=pal["hidden"], zorder=5)
        elif m.state == TRANSIT:
            ax.plot(x, y, "o", ms=ms, mfc=pal["planet"], mec=pal["moon"], mew=1.2, zorder=5)
        else:
            ax.plot(x, y, "o", ms=ms, color=pal["moon"], zorder=5)

        lines = [m.name + ("" if m.state == "visible" else f" ({m.state})")]
        if show_coords:
            lines.append(f"{format_ra_short(m.ra_j2000)} {format_dec_short(m.dec_j2000)}")
        cx, cy, leader = place(x, y, lines, fs, 6 * per_pt)
        if leader:
            ax.plot([x, cx], [y, cy], color=pal["muted"], lw=0.5, zorder=4)
        ax.text(cx, cy, "\n".join(lines), color=pal["fg"] if m.state == "visible" else pal["muted"],
                fontsize=fs, ha="center", va="center", zorder=6, linespacing=1.1,
                bbox=dict(fc=pal["bg"], ec="none", pad=0.5, alpha=0.85))

    # Boussole N/E
    ox, oy = -ext * 0.95, ext * 0.55
    alen = ext * 0.12
    for label, (e, n) in (("N", (0, 1)), ("E", (1, 0))):
        dx, dy = view(e * alen, n * alen)
        ax.annotate("", xy=(ox + dx, oy + dy), xytext=(ox, oy),
                    arrowprops=dict(arrowstyle="->", color=pal["muted"], lw=1))
        ax.text(ox + dx * 1.3, oy + dy * 1.3, label, color=pal["muted"], fontsize=8,
                ha="center", va="center")

    # Échelle
    n_r = _nice_scale(ext * 0.5 / r)
    bar = n_r * r
    while bar > ext * 0.9 and n_r > 1:
        n_r = n_r / 2 if n_r > 2 else 1
        bar = n_r * r
    bx, by = ext * 1.05 - bar, -ext * 0.70
    ax.plot([bx, bx + bar], [by, by], color=pal["muted"], lw=1)
    ax.plot([bx, bx], [by - ext * 0.012, by + ext * 0.012], color=pal["muted"], lw=1)
    ax.plot([bx + bar, bx + bar], [by - ext * 0.012, by + ext * 0.012], color=pal["muted"], lw=1)
    n_txt = f"{n_r:g}"
    ax.text(bx + bar / 2, by + ext * 0.025,
            f"{n_txt} rayon{'s' if n_r > 1 else ''} de {info.name} ≈ {bar:.0f}″",
            color=pal["muted"], fontsize=7.5, ha="center", va="bottom")

    return fig
