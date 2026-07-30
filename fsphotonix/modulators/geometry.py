"""Aperture geometries for illuminated regions and grating sub-regions.

An :class:`Aperture` rasterises a (possibly soft-edged) support mask on a grid,
in coordinates relative to its own centre.  Regions and sub-regions of a
modulator array are described by these objects, so the same code handles a
rectangular tile, a circular illuminated spot, an annulus, or an arbitrary
polygon cut out of a lithographic mask.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .. import engine as eng
from ..grid import Grid
from ..registry import register


class Aperture:
    """Base class.  ``rotation`` is applied about the aperture centre."""

    rotation: float = 0.0
    soft_edge: float = 0.0  # 1/e transition width [m]; 0 = hard edge

    # ---- to implement ------------------------------------------------------ #
    def signed_inside(self, xl, yl):  # pragma: no cover - abstract
        """>= 0 inside, < 0 outside, in units of metres (a signed distance-ish)."""
        raise NotImplementedError

    def bounds(self) -> tuple[float, float, float, float]:  # pragma: no cover
        """``(xmin, xmax, ymin, ymax)`` relative to centre."""
        raise NotImplementedError

    # ---- shared ----------------------------------------------------------- #
    def rotated_coords(self, xl, yl):
        if not self.rotation:
            return xl, yl
        c, s = np.cos(-self.rotation), np.sin(-self.rotation)
        return xl * c - yl * s, xl * s + yl * c

    def mask(self, grid: Grid, cx: float = 0.0, cy: float = 0.0):
        """Rasterise on ``grid``, centred at ``(cx, cy)``.  Values in [0, 1]."""
        X, Y = grid.meshgrid()
        return self.mask_at(X - cx, Y - cy)

    def mask_at(self, xl, yl):
        d = self.signed_inside(*self.rotated_coords(xl, yl))
        if self.soft_edge > 0:
            return (0.5 * (1 + eng.xp.tanh(d / self.soft_edge))).astype(eng.REAL)
        # strictly-inside test, so that abutting sub-apertures never claim the same
        # boundary sample twice (a seam would otherwise be counted by both tiles)
        return (d > 0).astype(eng.REAL)

    @property
    def size(self) -> tuple[float, float]:
        xmin, xmax, ymin, ymax = self.bounds()
        return (xmax - xmin, ymax - ymin)

    def area(self) -> float:
        """Geometric area [m^2] (numeric for exotic shapes)."""
        xmin, xmax, ymin, ymax = self.bounds()
        n = 256
        xl, yl = np.meshgrid(np.linspace(xmin, xmax, n), np.linspace(ymin, ymax, n))
        m = eng.to_numpy(self.mask_at(eng.asarray(xl), eng.asarray(yl)))
        return float(m.mean()) * (xmax - xmin) * (ymax - ymin)

    # ---- set algebra ------------------------------------------------------- #
    def __and__(self, other: "Aperture") -> "Combined":
        return Combined([self, other], op="and")

    def __or__(self, other: "Aperture") -> "Combined":
        return Combined([self, other], op="or")

    def __sub__(self, other: "Aperture") -> "Combined":
        return Combined([self, other], op="sub")


@register("geometry", "rect", "rectangle", "square")
@dataclass
class Rect(Aperture):
    width: float = 1e-3
    height: float | None = None
    rotation: float = 0.0
    soft_edge: float = 0.0

    def __post_init__(self):
        if self.height is None:
            self.height = self.width

    def signed_inside(self, xl, yl):
        return eng.xp.minimum(self.width / 2 - eng.xp.abs(xl), self.height / 2 - eng.xp.abs(yl))

    def bounds(self):
        r = 0.5 * np.hypot(self.width, self.height) if self.rotation else None
        if r is not None:
            return (-r, r, -r, r)
        return (-self.width / 2, self.width / 2, -self.height / 2, self.height / 2)

    def area(self) -> float:
        return self.width * self.height


@register("geometry", "circle", "circular", "disk")
@dataclass
class Circle(Aperture):
    diameter: float = 1e-3
    rotation: float = 0.0
    soft_edge: float = 0.0

    def signed_inside(self, xl, yl):
        return self.diameter / 2 - eng.xp.sqrt(xl**2 + yl**2)

    def bounds(self):
        r = self.diameter / 2
        return (-r, r, -r, r)

    def area(self) -> float:
        return np.pi * (self.diameter / 2) ** 2


@register("geometry", "ellipse", "elliptical")
@dataclass
class Ellipse(Aperture):
    width: float = 1e-3
    height: float = 0.5e-3
    rotation: float = 0.0
    soft_edge: float = 0.0

    def signed_inside(self, xl, yl):
        r = eng.xp.sqrt((2 * xl / self.width) ** 2 + (2 * yl / self.height) ** 2)
        return (1.0 - r) * min(self.width, self.height) / 2

    def bounds(self):
        return (-self.width / 2, self.width / 2, -self.height / 2, self.height / 2)

    def area(self) -> float:
        return np.pi * self.width * self.height / 4


@register("geometry", "annulus", "ring")
@dataclass
class Annulus(Aperture):
    outer_diameter: float = 1e-3
    inner_diameter: float = 0.5e-3
    rotation: float = 0.0
    soft_edge: float = 0.0

    def signed_inside(self, xl, yl):
        r = eng.xp.sqrt(xl**2 + yl**2)
        return eng.xp.minimum(self.outer_diameter / 2 - r, r - self.inner_diameter / 2)

    def bounds(self):
        r = self.outer_diameter / 2
        return (-r, r, -r, r)

    def area(self) -> float:
        return np.pi * ((self.outer_diameter / 2) ** 2 - (self.inner_diameter / 2) ** 2)


@register("geometry", "sector", "wedge", "pie")
@dataclass
class Sector(Aperture):
    """Angular wedge of an annulus -- the natural shape for radially multiplexed
    holograms (many sub-gratings around one illuminated spot)."""

    outer_diameter: float = 1e-3
    inner_diameter: float = 0.0
    start_angle: float = 0.0
    sweep: float = np.pi / 2
    rotation: float = 0.0
    soft_edge: float = 0.0

    def signed_inside(self, xl, yl):
        r = eng.xp.sqrt(xl**2 + yl**2)
        radial = eng.xp.minimum(self.outer_diameter / 2 - r, r - self.inner_diameter / 2)
        a = eng.xp.mod(eng.xp.arctan2(yl, xl) - self.start_angle, 2 * np.pi)
        ang = eng.xp.minimum(a, self.sweep - a) * eng.xp.maximum(r, 1e-12)
        return eng.xp.minimum(radial, ang)

    def bounds(self):
        r = self.outer_diameter / 2
        return (-r, r, -r, r)


@register("geometry", "hexagon", "hex")
@dataclass
class Hexagon(Aperture):
    across_flats: float = 1e-3
    rotation: float = 0.0
    soft_edge: float = 0.0

    def signed_inside(self, xl, yl):
        a = self.across_flats / 2
        d = None
        for k in range(3):
            th = k * np.pi / 3
            proj = a - eng.xp.abs(xl * np.cos(th) + yl * np.sin(th))
            d = proj if d is None else eng.xp.minimum(d, proj)
        return d

    def bounds(self):
        r = self.across_flats / np.sqrt(3)
        return (-r, r, -r, r)

    def area(self) -> float:
        return np.sqrt(3) / 2 * self.across_flats**2


@register("geometry", "polygon")
@dataclass
class Polygon(Aperture):
    """Convex or concave polygon given as ``[(x, y), ...]`` relative to centre."""

    vertices: Sequence[Sequence[float]] = ((-1e-3, -1e-3), (1e-3, -1e-3), (0.0, 1e-3))
    rotation: float = 0.0
    soft_edge: float = 0.0

    def signed_inside(self, xl, yl):
        v = np.asarray(self.vertices, dtype=float)
        xn, yn = eng.to_numpy(xl), eng.to_numpy(yl)
        inside = np.zeros(xn.shape, dtype=bool)
        n = len(v)
        for i in range(n):
            x1, y1 = v[i]
            x2, y2 = v[(i + 1) % n]
            cross = ((y1 > yn) != (y2 > yn)) & (
                xn < (x2 - x1) * (yn - y1) / np.where(y2 - y1 == 0, 1e-30, y2 - y1) + x1
            )
            inside ^= cross
        scale = max(np.ptp(v[:, 0]), np.ptp(v[:, 1]), 1e-12)
        return eng.asarray(np.where(inside, scale, -scale), dtype=eng.REAL)

    def bounds(self):
        v = np.asarray(self.vertices, dtype=float)
        return (float(v[:, 0].min()), float(v[:, 0].max()),
                float(v[:, 1].min()), float(v[:, 1].max()))


@register("geometry", "custom", "mask")
@dataclass
class CustomAperture(Aperture):
    """Wrap ``func(xl, yl) -> mask in [0, 1]`` or a pre-rendered bitmap."""

    func: object = None
    bitmap: object = None
    extent: tuple[float, float] = (1e-3, 1e-3)
    rotation: float = 0.0
    soft_edge: float = 0.0

    def mask_at(self, xl, yl):
        xl, yl = self.rotated_coords(xl, yl)
        if self.func is not None:
            return eng.asarray(self.func(xl, yl), dtype=eng.REAL)
        bm = np.asarray(self.bitmap, dtype=float)
        w, h = self.extent
        ny, nx = bm.shape
        cx = np.clip(((eng.to_numpy(xl) + w / 2) / w * nx).astype(int), 0, nx - 1)
        cy = np.clip(((eng.to_numpy(yl) + h / 2) / h * ny).astype(int), 0, ny - 1)
        out = bm[cy, cx]
        inside = (np.abs(eng.to_numpy(xl)) <= w / 2) & (np.abs(eng.to_numpy(yl)) <= h / 2)
        return eng.asarray(out * inside, dtype=eng.REAL)

    def signed_inside(self, xl, yl):
        return self.mask_at(xl, yl) - 0.5

    def bounds(self):
        w, h = self.extent
        return (-w / 2, w / 2, -h / 2, h / 2)


@dataclass
class Shifted(Aperture):
    """Translate an aperture within its parent's local frame."""

    inner: Aperture = field(default_factory=lambda: Rect())
    dx: float = 0.0
    dy: float = 0.0
    rotation: float = 0.0
    soft_edge: float = 0.0

    def mask_at(self, xl, yl):
        return self.inner.mask_at(xl - self.dx, yl - self.dy)

    def signed_inside(self, xl, yl):
        return self.inner.signed_inside(xl - self.dx, yl - self.dy)

    def bounds(self):
        xmin, xmax, ymin, ymax = self.inner.bounds()
        return (xmin + self.dx, xmax + self.dx, ymin + self.dy, ymax + self.dy)


@dataclass
class Combined(Aperture):
    """Boolean combination of apertures."""

    parts: Sequence[Aperture] = field(default_factory=tuple)
    op: str = "or"
    rotation: float = 0.0
    soft_edge: float = 0.0

    def mask_at(self, xl, yl):
        ms = [p.mask_at(xl, yl) for p in self.parts]
        out = ms[0]
        for m in ms[1:]:
            if self.op == "or":
                out = eng.xp.maximum(out, m)
            elif self.op == "and":
                out = out * m
            elif self.op == "sub":
                out = out * (1 - m)
            else:
                raise ValueError(f"unknown op {self.op!r}")
        return out

    def signed_inside(self, xl, yl):
        return self.mask_at(xl, yl) - 0.5

    def bounds(self):
        bs = np.array([p.bounds() for p in self.parts])
        return (float(bs[:, 0].min()), float(bs[:, 1].max()),
                float(bs[:, 2].min()), float(bs[:, 3].max()))


# --------------------------------------------------------------------------- #
# Sub-region tiling helpers: how several gratings share one illuminated region
# --------------------------------------------------------------------------- #
def tile_region(aperture: Aperture, n: int, mode: str = "stripes",
                gap: float = 0.0) -> list[tuple[Aperture, tuple[float, float]]]:
    """Split one region into ``n`` spatially multiplexed sub-apertures.

    Returns ``[(sub_aperture, (dx, dy)), ...]`` offsets relative to the region
    centre.  The sub-apertures are *not* clipped to the region here -- the region
    mask always multiplies them at render time.  Modes:

    ``stripes``      vertical strips (simple, dispersion-friendly)
    ``grid``         near-square checkerboard of tiles
    ``sectors``      angular wedges (rotationally symmetric fan-out)
    ``rings``        concentric annuli
    ``interleave``   n interleaved fine stripe combs (each sub-grating sees the
                     whole region, at the cost of extra sampling noise)
    """
    w, h = aperture.size
    out: list[tuple[Aperture, tuple[float, float]]] = []
    if n <= 1:
        return [(aperture, (0.0, 0.0))]

    if mode == "stripes":
        sw = w / n
        for i in range(n):
            sub = Rect(width=max(sw - gap, sw * 0.05), height=h)
            out.append((sub, ((i - (n - 1) / 2) * sw, 0.0)))
    elif mode == "grid":
        ncols = int(np.ceil(np.sqrt(n)))
        nrows = int(np.ceil(n / ncols))
        sw, sh = w / ncols, h / nrows
        for i in range(n):
            r, c = divmod(i, ncols)
            sub = Rect(width=max(sw - gap, sw * 0.05), height=max(sh - gap, sh * 0.05))
            out.append((sub,
                        ((c - (ncols - 1) / 2) * sw, (r - (nrows - 1) / 2) * sh)))
    elif mode == "sectors":
        d = max(w, h)
        for i in range(n):
            out.append((Sector(outer_diameter=d, start_angle=2 * np.pi * i / n,
                               sweep=2 * np.pi / n), (0.0, 0.0)))
    elif mode == "rings":
        d = max(w, h)
        for i in range(n):
            out.append((Annulus(outer_diameter=d * (i + 1) / n,
                                inner_diameter=d * i / n), (0.0, 0.0)))
    elif mode == "interleave":
        period = max(w / max(n * 8, 1), 1e-9)
        for i in range(n):
            sub = CustomAperture(
                func=(lambda xl, yl, i=i: (
                    (eng.xp.floor(xl / period).astype(eng.REAL) % n == i).astype(eng.REAL))),
                extent=(w, h),
            )
            out.append((sub, (0.0, 0.0)))
    else:
        raise ValueError(f"unknown tiling mode {mode!r}")
    return out


def as_aperture(spec) -> Aperture:
    from ..registry import build

    if spec is None:
        return Rect()
    if isinstance(spec, Aperture):
        return spec
    return build(spec, "geometry")
