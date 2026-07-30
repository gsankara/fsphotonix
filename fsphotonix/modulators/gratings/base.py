"""Holographic grating abstraction.

A :class:`Grating` supplies the complex transmittance of one *sub-region* of a
modulator, in coordinates local to that sub-region.  It answers two questions:

``complex_transmittance(xl, yl, wavelength)``
    the full wave-optics map, used by the FFT propagators;

``orders(wavelength)``
    the discrete list of diffraction orders (direction + efficiency), used by the
    fast :class:`~fsphotonix.propagation.spot.SpotPropagator` that makes very
    large systems tractable, and by the design/analysis helpers.

Wavelength scaling.  A grating is designed at ``design_wavelength``.  A phase
structure realised as an optical path difference (SLM voltage, mirror piston,
etched relief) produces ``phi(lambda) = phi_design * lambda_design / lambda``.
That single line is where all the chromatic behaviour of the machine comes from:
run a WDM comb through the same hologram and the orders walk across the screen.
Set ``achromatic=True`` for an idealised phase element that ignores dispersion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ... import engine as eng
from ...registry import build


@dataclass(frozen=True)
class Order:
    """One diffraction order."""

    mx: int
    my: int
    fx: float  # transverse spatial frequency [1/m]
    fy: float
    efficiency: float  # fraction of incident power in this order

    def direction(self, wavelength: float) -> tuple[float, float, float]:
        """Unit direction cosines ``(alpha, beta, gamma)`` of the order."""
        a = wavelength * self.fx
        b = wavelength * self.fy
        g2 = 1.0 - a * a - b * b
        return (a, b, float(np.sqrt(g2)) if g2 > 0 else float("nan"))

    def angles(self, wavelength: float) -> tuple[float, float]:
        """``(theta_x, theta_y)`` deflection angles [rad]."""
        a, b, _ = self.direction(wavelength)
        return (float(np.arcsin(np.clip(a, -1, 1))), float(np.arcsin(np.clip(b, -1, 1))))

    def screen_position(self, wavelength: float, z: float) -> tuple[float, float]:
        """Where this order lands on a screen ``z`` away (exact, non-paraxial)."""
        a, b, g = self.direction(wavelength)
        if not np.isfinite(g) or g <= 0:
            return (float("nan"), float("nan"))
        return (z * a / g, z * b / g)


class Grating:
    """Base class for holographic gratings written onto a modulator sub-region."""

    design_wavelength: float = 1550e-9
    achromatic: bool = False
    efficiency_scale: float = 1.0  # extra loss (fabrication, polarisation, ...)

    # ---- to implement ------------------------------------------------------ #
    def phase(self, xl, yl):
        """Design-wavelength phase [rad].  Return 0 for pure amplitude gratings."""
        return eng.xp.zeros_like(eng.asarray(xl, dtype=eng.REAL))

    def amplitude(self, xl, yl):
        """Design-wavelength amplitude transmittance in [0, 1]."""
        return eng.xp.ones_like(eng.asarray(xl, dtype=eng.REAL))

    def unit_cell(self) -> tuple[tuple[float, float], tuple[float, float] | None] | None:
        """Lattice vectors ``(a1, a2)`` [m]; ``a2=None`` for a 1-D grating.

        Return ``None`` for aperiodic elements (lenses, CGHs), which must then
        override :meth:`orders`.
        """
        return None

    # ---- derived ----------------------------------------------------------- #
    def scale(self, wavelength: float) -> float:
        if self.achromatic or not wavelength:
            return 1.0
        return self.design_wavelength / wavelength

    def complex_transmittance(self, xl, yl, wavelength: float | None = None):
        lam = wavelength or self.design_wavelength
        a = eng.asarray(self.amplitude(xl, yl), dtype=eng.REAL)
        p = eng.asarray(self.phase(xl, yl), dtype=eng.REAL) * self.scale(lam)
        return (a * eng.xp.exp(1j * p)).astype(eng.COMPLEX) * np.sqrt(self.efficiency_scale)

    def orders(self, wavelength: float | None = None, max_order: int = 6,
               samples: int = 512, threshold: float = 1e-5) -> list[Order]:
        """Diffraction orders, computed numerically from one unit cell.

        Memoised per instance: a panel with thousands of sub-gratings queries this
        once per (wavelength, order-count) and reuses it for every subsequent pass,
        which is what keeps the beamlet propagator cheap inside a feedback loop.
        """
        lam = wavelength or self.design_wavelength
        key = (round(lam, 18), max_order, samples, round(threshold, 12))
        cache = getattr(self, "_orders_cache", None)
        if cache is None:
            cache = self._orders_cache = {}
        hit = cache.get(key)
        if hit is not None:
            return hit
        out = self._compute_orders(lam, max_order, samples, threshold)
        cache[key] = out
        return out

    def invalidate_orders(self) -> None:
        """Drop the memoised order table (call after mutating a grating in place)."""
        self._orders_cache = {}

    def _compute_orders(self, lam: float, max_order: int, samples: int,
                        threshold: float) -> list[Order]:
        cell = self.unit_cell()
        if cell is None:
            return [Order(0, 0, 0.0, 0.0, self.efficiency_scale)]
        a1, a2 = cell
        if a2 is None:
            return self._orders_1d(a1, lam, max_order, samples, threshold)
        return self._orders_2d(a1, a2, lam, max_order, samples, threshold)

    def _orders_1d(self, a1, lam, max_order, samples, threshold) -> list[Order]:
        L = float(np.hypot(*a1))
        ux, uy = a1[0] / L, a1[1] / L
        s = np.linspace(0.0, L, samples, endpoint=False)
        t = eng.to_numpy(self.complex_transmittance(eng.asarray(s * ux), eng.asarray(s * uy), lam))
        c = np.fft.fft(np.asarray(t).ravel()) / samples
        out = []
        for m in range(-max_order, max_order + 1):
            eta = float(abs(c[m % samples]) ** 2)
            if eta < threshold:
                continue
            fx, fy = m * ux / L, m * uy / L
            if (lam * fx) ** 2 + (lam * fy) ** 2 >= 1.0:
                continue  # evanescent
            out.append(Order(m, 0, fx, fy, eta))
        return sorted(out, key=lambda o: -o.efficiency)

    def _orders_2d(self, a1, a2, lam, max_order, samples, threshold) -> list[Order]:
        n = min(samples, 256)
        s1, s2 = np.meshgrid(np.linspace(0, 1, n, endpoint=False),
                             np.linspace(0, 1, n, endpoint=False), indexing="ij")
        x = s1 * a1[0] + s2 * a2[0]
        y = s1 * a1[1] + s2 * a2[1]
        t = eng.to_numpy(self.complex_transmittance(eng.asarray(x), eng.asarray(y), lam))
        c = np.fft.fft2(np.asarray(t)) / (n * n)
        # reciprocal lattice
        det = a1[0] * a2[1] - a1[1] * a2[0]
        if abs(det) < 1e-30:
            return self._orders_1d(a1, lam, max_order, samples, threshold)
        b1 = (a2[1] / det, -a2[0] / det)
        b2 = (-a1[1] / det, a1[0] / det)
        out = []
        for m1 in range(-max_order, max_order + 1):
            for m2 in range(-max_order, max_order + 1):
                eta = float(abs(c[m1 % n, m2 % n]) ** 2)
                if eta < threshold:
                    continue
                fx = m1 * b1[0] + m2 * b2[0]
                fy = m1 * b1[1] + m2 * b2[1]
                if (lam * fx) ** 2 + (lam * fy) ** 2 >= 1.0:
                    continue
                out.append(Order(m1, m2, fx, fy, eta))
        return sorted(out, key=lambda o: -o.efficiency)

    # ---- analysis ---------------------------------------------------------- #
    def total_efficiency(self, wavelength: float | None = None, **kw) -> float:
        return float(sum(o.efficiency for o in self.orders(wavelength, **kw)))

    def design_efficiency(self, wavelength: float | None = None, **kw) -> float:
        """Efficiency of the intended (non-zero, highest) order."""
        os_ = [o for o in self.orders(wavelength, **kw) if (o.mx, o.my) != (0, 0)]
        return float(os_[0].efficiency) if os_ else 0.0

    def render(self, aperture_size: tuple[float, float], pitch: float,
               wavelength: float | None = None):
        """Rasterise the grating on a standalone grid (inspection / plotting)."""
        from ...grid import Grid

        g = Grid.from_extent(aperture_size[0], aperture_size[1], pitch)
        X, Y = g.meshgrid()
        return g, self.complex_transmittance(X, Y, wavelength)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(lam0={self.design_wavelength*1e9:.1f}nm)"


# --------------------------------------------------------------------------- #
# Design helpers shared by the concrete gratings
# --------------------------------------------------------------------------- #
def carrier_frequency(theta_x: float, theta_y: float, wavelength: float) -> tuple[float, float]:
    """Spatial frequency [1/m] of the linear phase ramp deflecting to an angle."""
    return (np.sin(theta_x) / wavelength, np.sin(theta_y) / wavelength)


def target_to_angles(dx: float, dy: float, z: float) -> tuple[float, float]:
    """Deflection angles that steer a beam to screen offset ``(dx, dy)`` at range ``z``.

    Exact (non-paraxial): the direction cosines are normalised over the full 3-D
    vector, so wide-angle fan-outs land where they are asked to.
    """
    r = float(np.sqrt(dx * dx + dy * dy + z * z))
    return (float(np.arcsin(dx / r)), float(np.arcsin(dy / r)))


def period_and_angle(fx: float, fy: float) -> tuple[float, float]:
    """Convert a carrier frequency to ``(period [m], orientation [rad])``."""
    f = float(np.hypot(fx, fy))
    if f == 0:
        return (float("inf"), 0.0)
    return (1.0 / f, float(np.arctan2(fy, fx)))


def as_grating(spec) -> Grating | None:
    if spec is None or isinstance(spec, Grating):
        return spec
    return build(spec, "grating")
