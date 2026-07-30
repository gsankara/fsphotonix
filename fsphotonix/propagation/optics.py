"""Bulk optical elements expressed as zero-distance propagators."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import engine as eng
from ..grid import Grid
from ..registry import register
from .base import Propagator
from .kernels import Fraunhofer


@register("propagator", "thin_lens", "lens_phase")
@dataclass
class ThinLens(Propagator):
    """Ideal thin lens phase (plus an optional clear aperture)."""

    focal_length: float = 0.1
    diameter: float | None = None
    z: float = 0.0
    out_grid: Grid | None = None

    def apply(self, u, in_grid: Grid, out_grid: Grid, wavelength: float, z: float):
        X, Y = in_grid.meshgrid()
        k = 2 * np.pi / wavelength
        t = eng.xp.exp(-1j * k * (X**2 + Y**2) / (2 * self.focal_length))
        if self.diameter is not None:
            t = t * (eng.xp.sqrt(X**2 + Y**2) <= self.diameter / 2)
        return u * t


@register("propagator", "aperture_stop", "stop")
@dataclass
class ApertureStop(Propagator):
    """Hard stop / field mask (vignetting, baffles, spatial filters)."""

    aperture: object = None  # fsphotonix.modulators.geometry.Aperture
    center: tuple[float, float] = (0.0, 0.0)
    z: float = 0.0
    out_grid: Grid | None = None

    def apply(self, u, in_grid: Grid, out_grid: Grid, wavelength: float, z: float):
        from ..modulators.geometry import as_aperture

        ap = as_aperture(self.aperture)
        return u * ap.mask(in_grid, self.center[0], self.center[1])


@register("propagator", "fourier_plane", "f2", "fourier")
@dataclass
class FourierPlane(Propagator):
    """Optical Fourier transform: the back focal plane of a lens.

    The screen coordinate maps linearly to input spatial frequency,
    ``x2 = lambda * f * fx``, so a grating's orders appear at fixed positions and
    all wavelengths share one plane -- the standard way to build a diffractive
    matrix-vector multiplier without a long free-space run.
    """

    focal_length: float = 0.1
    z: float = 0.0
    out_grid: Grid | None = None

    def natural_grid(self, in_grid: Grid, wavelength: float) -> Grid:
        return Grid(nx=in_grid.nx, ny=in_grid.ny,
                    dx=wavelength * self.focal_length / (in_grid.nx * in_grid.dx),
                    dy=wavelength * self.focal_length / (in_grid.ny * in_grid.dy))

    def target_grid(self, in_grid: Grid, wavelength: float | None = None) -> Grid:
        if self.out_grid is not None:
            return self.out_grid
        return self.natural_grid(in_grid, wavelength or 1550e-9)

    def apply(self, u, in_grid: Grid, out_grid: Grid, wavelength: float, z: float):
        return Fraunhofer(z=self.focal_length, quadratic_phase=False).apply(
            u, in_grid, out_grid, wavelength, self.focal_length)


@register("propagator", "relay", "afocal")
@dataclass
class AfocalRelay(Propagator):
    """Ideal magnifying relay (a telescope): scale the field, conserve power."""

    magnification: float = 1.0
    z: float = 0.0
    out_grid: Grid | None = None

    def target_grid(self, in_grid: Grid, wavelength: float | None = None) -> Grid:
        if self.out_grid is not None:
            return self.out_grid
        return in_grid.rescaled(abs(self.magnification))

    def apply(self, u, in_grid: Grid, out_grid: Grid, wavelength: float, z: float):
        from .kernels import _resample

        m = abs(self.magnification)
        scaled = Grid(nx=in_grid.nx, ny=in_grid.ny, dx=in_grid.dx * m, dy=in_grid.dy * m,
                      x0=in_grid.x0 * m, y0=in_grid.y0 * m)
        v = u / m  # amplitude scaling keeps total power constant
        if self.magnification < 0:
            v = v[::-1, ::-1]
        return _resample(v, scaled, out_grid)


@register("propagator", "phase_screen", "aberration")
@dataclass
class PhaseScreen(Propagator):
    """Static aberration / turbulence screen with a power-law spectrum.

    ``strength`` is the rms wavefront error in waves; ``exponent`` shapes the
    spatial spectrum (-11/3 is Kolmogorov).  Useful for asking how much
    misalignment and thermal drift a given fan-out design tolerates.
    """

    strength: float = 0.05
    exponent: float = -11.0 / 3.0
    outer_scale: float = 5e-3
    seed: int = 0
    z: float = 0.0
    out_grid: Grid | None = None

    def _screen(self, grid: Grid):
        cached = getattr(self, "_cache", None)
        key = (grid.shape, grid.dx, grid.dy, self.seed, self.strength, self.exponent)
        if cached is not None and cached[0] == key:
            return cached[1]
        rng = np.random.default_rng(self.seed)
        fx = np.fft.fftfreq(grid.nx, grid.dx)
        fy = np.fft.fftfreq(grid.ny, grid.dy)
        FX, FY = np.meshgrid(fx, fy)
        f = np.sqrt(FX**2 + FY**2)
        f0 = 1.0 / max(self.outer_scale, 1e-9)
        psd = (f**2 + f0**2) ** (self.exponent / 2.0)
        noise = rng.normal(size=grid.shape) + 1j * rng.normal(size=grid.shape)
        ph = np.real(np.fft.ifft2(noise * np.sqrt(psd)))
        rms = ph.std()
        ph = ph / rms * (2 * np.pi * self.strength) if rms > 0 else ph
        out = eng.asarray(np.exp(1j * ph), dtype=eng.COMPLEX)
        self._cache = (key, out)
        return out

    def apply(self, u, in_grid: Grid, out_grid: Grid, wavelength: float, z: float):
        scale = 1550e-9 / wavelength
        s = self._screen(in_grid)
        return u * (s**scale if abs(scale - 1) > 1e-12 else s)


@register("propagator", "misalign", "shift_tilt")
@dataclass
class Misalignment(Propagator):
    """Lateral shift, tilt and defocus of a plane -- the alignment error budget."""

    shift: tuple[float, float] = (0.0, 0.0)
    tilt: tuple[float, float] = (0.0, 0.0)  # rad
    defocus: float = 0.0  # equivalent focal error [m]
    z: float = 0.0
    out_grid: Grid | None = None

    def apply(self, u, in_grid: Grid, out_grid: Grid, wavelength: float, z: float):
        k = 2 * np.pi / wavelength
        X, Y = in_grid.meshgrid()
        t = eng.xp.exp(1j * k * (np.sin(self.tilt[0]) * X + np.sin(self.tilt[1]) * Y))
        if self.defocus:
            t = t * eng.xp.exp(-1j * k * (X**2 + Y**2) / (2 * self.defocus))
        v = u * t
        if any(self.shift):
            sx = int(round(self.shift[0] / in_grid.dx))
            sy = int(round(self.shift[1] / in_grid.dy))
            v = eng.xp.roll(eng.xp.roll(v, sy, axis=0), sx, axis=1)
        return v
