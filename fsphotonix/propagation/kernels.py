"""Free-space propagation kernels.

Which one to use is a sampling question, not a physics question -- all of these
solve the same scalar diffraction problem in different regimes:

``AngularSpectrum``  exact (non-paraxial) same-grid propagation.  Correct for
    short ranges and wide angles; band-limited per Matsushima so the periodic
    wrap-around of the FFT does not alias energy back into the window.
``FresnelTransfer``  paraxial, same grid, one FFT pair.  Well sampled when
    ``z <~ N*dx^2/lambda``.
``FresnelZoom``      paraxial with an **arbitrary output grid** via a chirp-z
    transform.  The general-purpose workhorse of this simulator: the screen is
    usually much larger and coarser than the modulator.
``Fraunhofer``       far field / focal plane; a zoom DFT with no input chirp.
``Auto``             picks between them from the sampling report.

All kernels conserve power to within the sampling error, and all accept negative
``z`` (backward propagation), which the feedback loops use.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import engine as eng
from ..grid import Grid, sampling_report
from ..registry import register
from .base import Propagator


@register("propagator", "angular_spectrum", "asm", "exact")
@dataclass
class AngularSpectrum(Propagator):
    """Band-limited angular spectrum method (same grid in and out)."""

    z: float = 0.0
    out_grid: Grid | None = None
    band_limit: bool = True
    evanescent: bool = False

    def apply(self, u, in_grid: Grid, out_grid: Grid, wavelength: float, z: float):
        if out_grid is not in_grid and (out_grid.shape != in_grid.shape
                                        or abs(out_grid.dx - in_grid.dx) > 1e-18):
            raise ValueError("AngularSpectrum requires identical input and output grids; "
                             "use FresnelZoom for a different output sampling")
        FX, FY = in_grid.freq_meshgrid()
        arg = 1.0 / wavelength**2 - FX**2 - FY**2
        prop = arg > 0
        kz = eng.xp.sqrt(eng.xp.where(prop, arg, 0.0)) * 2 * np.pi
        H = eng.xp.exp(1j * kz * z)
        if not self.evanescent:
            H = H * prop
        else:
            decay = eng.xp.sqrt(eng.xp.where(prop, 0.0, -arg)) * 2 * np.pi
            H = eng.xp.where(prop, H, eng.xp.exp(-decay * abs(z)))
        if self.band_limit and z != 0:
            H = H * _band_limit_mask(in_grid, wavelength, z)
        return eng.ifft2(eng.fft2(u) * H)


def _band_limit_mask(grid: Grid, wavelength: float, z: float):
    """Matsushima band limit: drop frequencies whose rays leave the window."""
    du = 1.0 / (grid.nx * grid.dx)
    dv = 1.0 / (grid.ny * grid.dy)
    u_lim = 1.0 / (wavelength * np.sqrt((2 * du * z) ** 2 + 1))
    v_lim = 1.0 / (wavelength * np.sqrt((2 * dv * z) ** 2 + 1))
    FX, FY = grid.freq_meshgrid()
    return ((eng.xp.abs(FX) < u_lim) & (eng.xp.abs(FY) < v_lim)).astype(eng.REAL)


@register("propagator", "fresnel_tf", "fresnel_transfer")
@dataclass
class FresnelTransfer(Propagator):
    """Paraxial Fresnel propagation by transfer function (same grid)."""

    z: float = 0.0
    out_grid: Grid | None = None

    def apply(self, u, in_grid: Grid, out_grid: Grid, wavelength: float, z: float):
        FX, FY = in_grid.freq_meshgrid()
        H = eng.xp.exp(-1j * np.pi * wavelength * z * (FX**2 + FY**2))
        H = H * np.exp(2j * np.pi * z / wavelength)
        return eng.ifft2(eng.fft2(u) * H)


@register("propagator", "fresnel", "fresnel_zoom", "zoom")
@dataclass
class FresnelZoom(Propagator):
    """Exact discrete Fresnel integral onto an arbitrary output grid.

    Uses a Bluestein chirp-z transform, so the output pitch and extent are free
    parameters and every wavelength lands on the *same* physical grid -- which is
    what makes a WDM system's chromatic walk-off visible instead of hidden in a
    per-wavelength rescaling.
    """

    z: float = 0.0
    out_grid: Grid | None = None

    def apply(self, u, in_grid: Grid, out_grid: Grid, wavelength: float, z: float):
        if z == 0:
            return _resample(u, in_grid, out_grid)
        lam = wavelength
        x1, y1 = in_grid.x(), in_grid.y()
        x2, y2 = out_grid.x(), out_grid.y()
        k = 2 * np.pi / lam
        X1, Y1 = in_grid.meshgrid()
        chirp_in = eng.xp.exp(1j * k * (X1**2 + Y1**2) / (2 * z))
        fx = eng.to_numpy(x2) / (lam * z)
        fy = eng.to_numpy(y2) / (lam * z)
        S = eng.zoom_dft2(u * chirp_in, eng.to_numpy(x1), eng.to_numpy(y1), fx, fy)
        X2, Y2 = out_grid.meshgrid()
        pre = (eng.xp.exp(1j * k * z) / (1j * lam * z)) * eng.xp.exp(
            1j * k * (X2**2 + Y2**2) / (2 * z))
        return pre * S * (in_grid.dx * in_grid.dy)


@register("propagator", "fraunhofer", "far_field")
@dataclass
class Fraunhofer(Propagator):
    """Far-field / focal-plane transform onto an arbitrary output grid."""

    z: float = 0.0
    out_grid: Grid | None = None
    quadratic_phase: bool = True

    def natural_grid(self, in_grid: Grid, wavelength: float) -> Grid:
        """The grid a plain single-FFT Fraunhofer transform would produce."""
        return Grid(nx=in_grid.nx, ny=in_grid.ny,
                    dx=wavelength * abs(self.z) / (in_grid.nx * in_grid.dx),
                    dy=wavelength * abs(self.z) / (in_grid.ny * in_grid.dy))

    def apply(self, u, in_grid: Grid, out_grid: Grid, wavelength: float, z: float):
        lam = wavelength
        k = 2 * np.pi / lam
        fx = eng.to_numpy(out_grid.x()) / (lam * z)
        fy = eng.to_numpy(out_grid.y()) / (lam * z)
        S = eng.zoom_dft2(u, eng.to_numpy(in_grid.x()), eng.to_numpy(in_grid.y()), fx, fy)
        pre = eng.xp.exp(1j * k * z) / (1j * lam * z)
        if self.quadratic_phase:
            X2, Y2 = out_grid.meshgrid()
            pre = pre * eng.xp.exp(1j * k * (X2**2 + Y2**2) / (2 * z))
        return pre * S * (in_grid.dx * in_grid.dy)


@register("propagator", "auto")
@dataclass
class Auto(Propagator):
    """Pick a kernel from the sampling regime.

    Same output grid and a short range -> angular spectrum (exact).  Different
    output grid, or a range past the critical distance -> the zoom Fresnel
    integral.  ``force`` overrides the choice.
    """

    z: float = 0.0
    out_grid: Grid | None = None
    force: str | None = None

    def choose(self, in_grid: Grid, wavelength: float, z: float) -> Propagator:
        if self.force:
            from ..registry import create

            return create("propagator", self.force, z=z, out_grid=self.out_grid)
        same = (self.out_grid is None
                or (self.out_grid.shape == in_grid.shape
                    and abs(self.out_grid.dx - in_grid.dx) < 1e-18
                    and abs(self.out_grid.dy - in_grid.dy) < 1e-18
                    and self.out_grid.x0 == in_grid.x0
                    and self.out_grid.y0 == in_grid.y0))
        rep = sampling_report(in_grid, wavelength, abs(z))
        if same and abs(z) <= rep["z_critical_x"]:
            return AngularSpectrum(z=z, out_grid=self.out_grid)
        return FresnelZoom(z=z, out_grid=self.out_grid)

    def apply(self, u, in_grid: Grid, out_grid: Grid, wavelength: float, z: float):
        return self.choose(in_grid, wavelength, z).apply(u, in_grid, out_grid, wavelength, z)


def _resample(u, in_grid: Grid, out_grid: Grid):
    """Nearest-neighbour resample for the degenerate ``z == 0`` case."""
    if in_grid.shape == out_grid.shape and abs(in_grid.dx - out_grid.dx) < 1e-18:
        return u
    xo = eng.to_numpy(out_grid.x())
    yo = eng.to_numpy(out_grid.y())
    xi = eng.to_numpy(in_grid.x())
    yi = eng.to_numpy(in_grid.y())
    ix = np.clip(np.searchsorted(xi, xo), 0, len(xi) - 1)
    iy = np.clip(np.searchsorted(yi, yo), 0, len(yi) - 1)
    un = eng.to_numpy(u)
    return eng.asarray(un[np.ix_(iy, ix)], dtype=eng.COMPLEX)


def suggest(in_grid: Grid, wavelength: float, z: float, out_grid: Grid | None = None) -> dict:
    """Diagnostics plus the recommended kernel name."""
    rep = sampling_report(in_grid, wavelength, abs(z))
    a = Auto(z=z, out_grid=out_grid)
    rep["recommended"] = type(a.choose(in_grid, wavelength, z)).__name__
    if out_grid is not None:
        rep["out_spots_per_pixel"] = (
            rep["far_field_spot_x"] / out_grid.dx if out_grid.dx else float("inf"))
    return rep
