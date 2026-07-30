"""Focusing diffractive elements.

Pure deflectors send a *collimated* beam to an angle; the spot on the screen is
then set by far-field diffraction of the sub-region.  A focusing hologram instead
concentrates the sub-region's light into a diffraction-limited spot *at a chosen
range*, which is usually what you want when the screen is a detector or fibre
array a fixed distance away.  :class:`SpotFocuser` combines the two and is the
direct answer to "deflect this region's light to that point on the screen".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ... import engine as eng
from ...registry import register
from .base import Grating, Order, carrier_frequency, target_to_angles


@register("grating", "fresnel_lens", "lens")
@dataclass
class FresnelLens(Grating):
    """Diffractive (wrapped) thin lens of focal length ``focal_length`` [m]."""

    focal_length: float = 0.1
    levels: int | None = None
    design_wavelength: float = 1550e-9
    achromatic: bool = False
    efficiency_scale: float = 1.0

    def phase(self, xl, yl):
        k = 2 * np.pi / self.design_wavelength
        p = -k * (xl**2 + yl**2) / (2 * self.focal_length)
        if self.levels:
            step = 2 * np.pi / self.levels
            p = eng.xp.floor(eng.xp.mod(p, 2 * np.pi) / step) * step
        return p

    def unit_cell(self):
        return None

    def orders(self, wavelength=None, **kw):
        return [Order(0, 0, 0.0, 0.0, self.efficiency_scale)]

    def spot_size(self, aperture: float, wavelength: float | None = None) -> float:
        """Airy-like focal spot diameter for a given aperture width."""
        lam = wavelength or self.design_wavelength
        return 2.44 * lam * self.focal_length / aperture

    def focus_shift(self, wavelength: float) -> float:
        """Chromatic focal shift: a diffractive lens has ``f ~ 1/lambda``."""
        return self.focal_length * self.design_wavelength / wavelength - self.focal_length


@register("grating", "spot_focuser", "off_axis_lens")
@dataclass
class SpotFocuser(Grating):
    """Off-axis focusing hologram: focus at ``(target_x, target_y, range_z)``.

    Equivalent to a diffractive lens of focal length ``range_z`` plus a linear
    carrier.  ``spherical=True`` uses the exact spherical (non-paraxial) phase,
    which matters for the wide deflection angles used in dense fan-outs.
    """

    target_x: float = 1e-3
    target_y: float = 0.0
    range_z: float = 0.1
    spherical: bool = True
    levels: int | None = None
    design_wavelength: float = 1550e-9
    achromatic: bool = False
    efficiency_scale: float = 1.0

    def phase(self, xl, yl):
        k = 2 * np.pi / self.design_wavelength
        if self.spherical:
            r = eng.xp.sqrt((xl - self.target_x) ** 2 + (yl - self.target_y) ** 2
                            + self.range_z**2)
            p = -k * r
        else:
            fx, fy = carrier_frequency(*target_to_angles(self.target_x, self.target_y,
                                                         self.range_z),
                                       self.design_wavelength)
            p = 2 * np.pi * (fx * xl + fy * yl) - k * (xl**2 + yl**2) / (2 * self.range_z)
        if self.levels:
            step = 2 * np.pi / self.levels
            p = eng.xp.floor(eng.xp.mod(p, 2 * np.pi) / step) * step
        return p

    def unit_cell(self):
        return None

    def orders(self, wavelength=None, **kw):
        lam = wavelength or self.design_wavelength
        tx, ty = target_to_angles(self.target_x, self.target_y, self.range_z)
        fx, fy = carrier_frequency(tx, ty, lam)
        s = self.scale(lam)
        eff = self.efficiency_scale
        if self.levels:
            eff *= float(np.sinc(1.0 / self.levels)) ** 2
        return [Order(1, 0, fx * s, fy * s, eff)]

    @classmethod
    def from_target(cls, dx: float, dy: float, z: float, wavelength: float = 1550e-9, **kw):
        return cls(target_x=dx, target_y=dy, range_z=z, design_wavelength=wavelength, **kw)


@register("grating", "axicon")
@dataclass
class Axicon(Grating):
    """Conical phase -> ring on the screen / Bessel beam along the axis."""

    cone_angle: float = 1e-3  # deflection half-angle [rad]
    design_wavelength: float = 1550e-9
    achromatic: bool = False
    efficiency_scale: float = 1.0

    def phase(self, xl, yl):
        k = 2 * np.pi / self.design_wavelength
        return -k * np.sin(self.cone_angle) * eng.xp.sqrt(xl**2 + yl**2)

    def unit_cell(self):
        return None

    def ring_radius(self, z: float) -> float:
        return float(z * np.tan(self.cone_angle))


@register("grating", "zone_plate")
@dataclass
class ZonePlate(Grating):
    """Binary Fresnel zone plate; ``phase_type`` picks amplitude or phase zones.

    A binary phase zone plate reaches ~40% into the design focus, an amplitude
    one ~10% -- the sort of trade a cheap fabricated screen faces.
    """

    focal_length: float = 0.1
    phase_type: bool = True
    design_wavelength: float = 1550e-9
    achromatic: bool = False
    efficiency_scale: float = 1.0

    def _zones(self, xl, yl):
        r2 = xl**2 + yl**2
        n = r2 / (self.design_wavelength * self.focal_length)
        return (eng.xp.mod(n, 2.0) < 1.0).astype(eng.REAL)

    def phase(self, xl, yl):
        if not self.phase_type:
            return eng.xp.zeros_like(eng.asarray(xl, dtype=eng.REAL))
        return np.pi * self._zones(xl, yl)

    def amplitude(self, xl, yl):
        if self.phase_type:
            return eng.xp.ones_like(eng.asarray(xl, dtype=eng.REAL))
        return self._zones(xl, yl)

    def unit_cell(self):
        return None
