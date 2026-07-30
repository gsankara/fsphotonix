"""Transverse beam profiles.

Each profile returns a complex amplitude map on a :class:`~fsphotonix.grid.Grid`
normalised to **1 W** of optical power; the source scales it by its power.
Position (``x0``, ``y0``) and pointing (``theta_x``, ``theta_y``) are common to
all profiles -- that is how a tiled source array illuminates different regions of
the modulator at different angles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import engine as eng
from ..grid import Grid
from ..registry import register


class BeamProfile:
    """Base class for transverse amplitude distributions."""

    x0: float = 0.0
    y0: float = 0.0
    theta_x: float = 0.0
    theta_y: float = 0.0

    def amplitude(self, grid: Grid, wavelength: float):  # pragma: no cover - abstract
        raise NotImplementedError

    # ---- shared machinery -------------------------------------------------- #
    def local_coords(self, grid: Grid):
        X, Y = grid.meshgrid()
        return X - self.x0, Y - self.y0

    def tilt_phase(self, grid: Grid, wavelength: float):
        if not (self.theta_x or self.theta_y):
            return 1.0
        X, Y = grid.meshgrid()
        k = 2 * np.pi / wavelength
        return eng.xp.exp(
            1j * k * (np.sin(self.theta_x) * X + np.sin(self.theta_y) * Y)
        )

    def field(self, grid: Grid, wavelength: float, power: float = 1.0):
        """Power-normalised complex amplitude including pointing phase."""
        u = eng.asarray(self.amplitude(grid, wavelength), dtype=eng.COMPLEX)
        norm = float(eng.to_numpy(eng.xp.sum(eng.xp.abs(u) ** 2)).item()) * grid.cell_area
        if norm > 0:
            u = u * np.sqrt(power / norm)
        return u * self.tilt_phase(grid, wavelength)


@register("profile", "plane", "plane_wave")
@dataclass
class PlaneWave(BeamProfile):
    """Uniform illumination, optionally truncated by a hard aperture."""

    diameter: float | None = None  # None -> fill the grid
    x0: float = 0.0
    y0: float = 0.0
    theta_x: float = 0.0
    theta_y: float = 0.0

    def amplitude(self, grid: Grid, wavelength: float):
        if self.diameter is None:
            return eng.xp.ones(grid.shape, dtype=eng.COMPLEX)
        X, Y = self.local_coords(grid)
        return (eng.xp.sqrt(X**2 + Y**2) <= self.diameter / 2).astype(eng.COMPLEX)


@register("profile", "gaussian")
@dataclass
class GaussianBeam(BeamProfile):
    """Fundamental Gaussian beam.

    ``waist`` is the 1/e^2 intensity radius w0 at the beam waist; ``z_offset``
    is the (signed) distance from the waist to this plane, which sets the
    curvature and spot size actually rendered here.
    """

    waist: float = 1e-3
    z_offset: float = 0.0
    x0: float = 0.0
    y0: float = 0.0
    theta_x: float = 0.0
    theta_y: float = 0.0
    ellipticity: float = 1.0  # wy / wx

    def q_params(self, wavelength: float) -> tuple[float, float, float]:
        """``(w, R, zR)`` at this plane."""
        zr = np.pi * self.waist**2 / wavelength
        w = self.waist * np.sqrt(1 + (self.z_offset / zr) ** 2)
        r = np.inf if self.z_offset == 0 else self.z_offset * (1 + (zr / self.z_offset) ** 2)
        return w, r, zr

    def amplitude(self, grid: Grid, wavelength: float):
        X, Y = self.local_coords(grid)
        w, r, _ = self.q_params(wavelength)
        wx, wy = w, w * self.ellipticity
        u = eng.xp.exp(-((X / wx) ** 2 + (Y / wy) ** 2)).astype(eng.COMPLEX)
        if np.isfinite(r) and r != 0:
            k = 2 * np.pi / wavelength
            u = u * eng.xp.exp(1j * k * (X**2 + Y**2) / (2 * r))
        return u

    @classmethod
    def from_divergence(cls, half_angle: float, wavelength: float, **kw) -> "GaussianBeam":
        """Construct from far-field half divergence angle [rad]."""
        return cls(waist=wavelength / (np.pi * half_angle), **kw)

    @classmethod
    def from_fiber(cls, mfd: float = 10.4e-6, **kw) -> "GaussianBeam":
        """Launch from a fibre facet of given mode-field diameter."""
        return cls(waist=mfd / 2, **kw)


@register("profile", "tophat", "flattop")
@dataclass
class SuperGaussian(BeamProfile):
    """Flat-top illumination (super-Gaussian of order ``order``).

    A beam-shaped/homogenised illuminator, i.e. what you actually put on an SLM
    so that every pixel sees the same optical power.
    """

    diameter: float = 5e-3
    order: int = 10
    x0: float = 0.0
    y0: float = 0.0
    theta_x: float = 0.0
    theta_y: float = 0.0

    def amplitude(self, grid: Grid, wavelength: float):
        X, Y = self.local_coords(grid)
        r = eng.xp.sqrt(X**2 + Y**2) / (self.diameter / 2)
        return eng.xp.exp(-(r ** (2 * self.order))).astype(eng.COMPLEX)


@register("profile", "rect", "rectangular")
@dataclass
class Rectangular(BeamProfile):
    """Hard rectangular illumination patch (a line-illuminator or a tile)."""

    width: float = 1e-3
    height: float | None = None
    x0: float = 0.0
    y0: float = 0.0
    theta_x: float = 0.0
    theta_y: float = 0.0

    def amplitude(self, grid: Grid, wavelength: float):
        h = self.width if self.height is None else self.height
        X, Y = self.local_coords(grid)
        return (
            (eng.xp.abs(X) <= self.width / 2) & (eng.xp.abs(Y) <= h / 2)
        ).astype(eng.COMPLEX)


@register("profile", "vortex", "lg")
@dataclass
class LaguerreGaussVortex(BeamProfile):
    """Donut / orbital-angular-momentum beam (mode multiplexing studies)."""

    waist: float = 1e-3
    charge: int = 1
    x0: float = 0.0
    y0: float = 0.0
    theta_x: float = 0.0
    theta_y: float = 0.0

    def amplitude(self, grid: Grid, wavelength: float):
        X, Y = self.local_coords(grid)
        r = eng.xp.sqrt(X**2 + Y**2)
        phi = eng.xp.arctan2(Y, X)
        rad = (r / self.waist) ** abs(self.charge) * eng.xp.exp(-((r / self.waist) ** 2))
        return (rad * eng.xp.exp(1j * self.charge * phi)).astype(eng.COMPLEX)


@register("profile", "custom")
@dataclass
class CustomProfile(BeamProfile):
    """Wrap ``func(X, Y, wavelength) -> complex array``."""

    func: object = None
    x0: float = 0.0
    y0: float = 0.0
    theta_x: float = 0.0
    theta_y: float = 0.0

    def amplitude(self, grid: Grid, wavelength: float):
        X, Y = self.local_coords(grid)
        return eng.asarray(self.func(X, Y, wavelength), dtype=eng.COMPLEX)


def as_profile(spec) -> BeamProfile:
    from ..registry import build

    if spec is None:
        return GaussianBeam()
    if isinstance(spec, BeamProfile):
        return spec
    return build(spec, "profile")
