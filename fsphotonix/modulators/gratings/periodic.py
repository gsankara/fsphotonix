"""Periodic gratings: blazed, binary phase, amplitude (Ronchi), sinusoidal, crossed."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ... import engine as eng
from ...registry import register
from .base import Grating, Order, carrier_frequency, period_and_angle, target_to_angles


def _project(xl, yl, angle: float):
    """Coordinate along the grating vector."""
    return xl * np.cos(angle) + yl * np.sin(angle)


def _quantize_phase(p, levels: int | None):
    if not levels or levels < 2:
        return p
    step = 2 * np.pi / levels
    return eng.xp.floor(p / step) * step


@register("grating", "blazed", "blaze", "sawtooth")
@dataclass
class BlazedGrating(Grating):
    """Sawtooth (blazed) phase grating -- the workhorse beam deflector.

    ``period`` is the grating spacing Lambda [m]; ``angle`` orients the grating
    vector in the plane (0 = deflect along +x).  ``blaze_depth`` is the sawtooth
    height in units of 2*pi: 1.0 sends everything into +1st order, other values
    model an SLM whose stroke is short of a full wave.  ``levels`` quantises the
    ramp as an etched/staircase DOE would (efficiency ~ sinc^2(1/levels)).
    """

    period: float = 20e-6
    angle: float = 0.0
    blaze_depth: float = 1.0
    levels: int | None = None
    design_wavelength: float = 1550e-9
    achromatic: bool = False
    efficiency_scale: float = 1.0

    def phase(self, xl, yl):
        s = _project(xl, yl, self.angle)
        frac = eng.xp.mod(s / self.period, 1.0)
        return _quantize_phase(2 * np.pi * self.blaze_depth * frac, self.levels)

    def unit_cell(self):
        return ((self.period * np.cos(self.angle), self.period * np.sin(self.angle)), None)

    def _compute_orders(self, lam, max_order, samples, threshold):
        """Closed form for the continuous sawtooth: ``eta_m = sinc^2(depth - m)``.

        Falls back to the numeric unit-cell transform once the ramp is quantised,
        where the staircase creates the extra order comb.
        """
        if self.levels:
            return super()._compute_orders(lam, max_order, samples, threshold)
        depth = self.blaze_depth * self.scale(lam)
        ux, uy = np.cos(self.angle), np.sin(self.angle)
        out = []
        for m in range(-max_order, max_order + 1):
            eta = float(np.sinc(depth - m)) ** 2 * self.efficiency_scale
            if eta < threshold:
                continue
            fx, fy = m * ux / self.period, m * uy / self.period
            if (lam * fx) ** 2 + (lam * fy) ** 2 >= 1.0:
                continue
            out.append(Order(m, 0, fx, fy, eta))
        return sorted(out, key=lambda o: -o.efficiency)

    # ---- design constructors ---------------------------------------------- #
    @classmethod
    def from_deflection(cls, theta_x: float, theta_y: float = 0.0,
                        wavelength: float = 1550e-9, **kw) -> "BlazedGrating":
        """Grating that steers the 1st order to ``(theta_x, theta_y)`` [rad]."""
        fx, fy = carrier_frequency(theta_x, theta_y, wavelength)
        period, angle = period_and_angle(fx, fy)
        return cls(period=period, angle=angle, design_wavelength=wavelength, **kw)

    @classmethod
    def from_target(cls, dx: float, dy: float, z: float,
                    wavelength: float = 1550e-9, **kw) -> "BlazedGrating":
        """Grating that lands the 1st order at screen offset ``(dx, dy)`` at range ``z``."""
        tx, ty = target_to_angles(dx, dy, z)
        return cls.from_deflection(tx, ty, wavelength, **kw)

    @classmethod
    def from_blaze_angle(cls, facet_tilt: float, wavelength: float = 1550e-9,
                         reflective: bool = True, **kw) -> "BlazedGrating":
        """From a physical facet tilt ``facet_tilt`` [rad] (Littrow-like geometry).

        A reflective facet tilted by ``gamma`` deflects by ``2*gamma``.
        """
        return cls.from_deflection(2 * facet_tilt if reflective else facet_tilt,
                                   0.0, wavelength, **kw)

    # ---- analysis --------------------------------------------------------- #
    def ideal_efficiency(self) -> float:
        """Closed-form 1st-order efficiency for the quantised sawtooth."""
        d = self.blaze_depth
        base = float(np.sinc(d - 1.0)) ** 2  # np.sinc(x) = sin(pi x)/(pi x)
        if self.levels and self.levels >= 2:
            base *= float(np.sinc(1.0 / self.levels)) ** 2
        return base * self.efficiency_scale


@register("grating", "phase_ramp", "ideal_deflector")
@dataclass
class PhaseRamp(Grating):
    """Unwrapped linear phase -- an ideal single-order deflector.

    Useful as a reference: no higher orders, no quantisation, 100% efficiency.
    """

    fx: float = 0.0
    fy: float = 0.0
    design_wavelength: float = 1550e-9
    achromatic: bool = False
    efficiency_scale: float = 1.0

    def phase(self, xl, yl):
        return 2 * np.pi * (self.fx * xl + self.fy * yl)

    def unit_cell(self):
        p, a = period_and_angle(self.fx, self.fy)
        if not np.isfinite(p):
            return None
        return ((p * np.cos(a), p * np.sin(a)), None)

    def orders(self, wavelength=None, **kw):
        lam = wavelength or self.design_wavelength
        s = self.scale(lam)
        return [Order(1, 0, self.fx * s, self.fy * s, self.efficiency_scale)]

    @classmethod
    def from_target(cls, dx: float, dy: float, z: float, wavelength: float = 1550e-9, **kw):
        tx, ty = target_to_angles(dx, dy, z)
        fx, fy = carrier_frequency(tx, ty, wavelength)
        return cls(fx=fx, fy=fy, design_wavelength=wavelength, **kw)


@register("grating", "binary_phase", "binary")
@dataclass
class BinaryPhaseGrating(Grating):
    """Two-level phase grating: symmetric +/-1 orders (40.5% each at pi step)."""

    period: float = 20e-6
    angle: float = 0.0
    duty: float = 0.5
    phase_step: float = np.pi
    design_wavelength: float = 1550e-9
    achromatic: bool = False
    efficiency_scale: float = 1.0

    def phase(self, xl, yl):
        s = eng.xp.mod(_project(xl, yl, self.angle) / self.period, 1.0)
        return self.phase_step * (s < self.duty).astype(eng.REAL)

    def unit_cell(self):
        return ((self.period * np.cos(self.angle), self.period * np.sin(self.angle)), None)

    @classmethod
    def from_deflection(cls, theta_x: float, theta_y: float = 0.0,
                        wavelength: float = 1550e-9, **kw):
        fx, fy = carrier_frequency(theta_x, theta_y, wavelength)
        period, angle = period_and_angle(fx, fy)
        return cls(period=period, angle=angle, design_wavelength=wavelength, **kw)


@register("grating", "ronchi", "binary_amplitude")
@dataclass
class RonchiGrating(Grating):
    """Binary *amplitude* grating -- the model for a tilt-free DMD checkerboard
    or a chrome-on-glass mask.  Half the light is simply blocked."""

    period: float = 20e-6
    angle: float = 0.0
    duty: float = 0.5
    off_transmission: float = 0.0
    design_wavelength: float = 1550e-9
    achromatic: bool = True
    efficiency_scale: float = 1.0

    def amplitude(self, xl, yl):
        s = eng.xp.mod(_project(xl, yl, self.angle) / self.period, 1.0)
        on = (s < self.duty).astype(eng.REAL)
        return on + (1 - on) * self.off_transmission

    def unit_cell(self):
        return ((self.period * np.cos(self.angle), self.period * np.sin(self.angle)), None)


@register("grating", "sinusoidal", "sine_phase")
@dataclass
class SinusoidalPhaseGrating(Grating):
    """Sinusoidal phase grating; order efficiencies are Bessel J_n(depth)^2.

    This is what a holographically recorded (interference) grating in a thin
    volume medium looks like.
    """

    period: float = 20e-6
    angle: float = 0.0
    depth: float = 1.8  # peak phase [rad]
    design_wavelength: float = 1550e-9
    achromatic: bool = False
    efficiency_scale: float = 1.0

    def phase(self, xl, yl):
        s = _project(xl, yl, self.angle)
        return self.depth * eng.xp.sin(2 * np.pi * s / self.period)

    def unit_cell(self):
        return ((self.period * np.cos(self.angle), self.period * np.sin(self.angle)), None)


@register("grating", "crossed", "2d_grating")
@dataclass
class CrossedGrating(Grating):
    """Two superposed 1-D gratings -> a 2-D lattice of spots on the screen.

    ``kind`` selects the profile of each axis (``blazed`` or ``binary``).  A
    blazed/blazed cross deflects into a single 2-D order; binary/binary produces
    the classic 2x2 (or larger) symmetric spot lattice.
    """

    period_x: float = 20e-6
    period_y: float = 20e-6
    kind: str = "blazed"
    blaze_depth: float = 1.0
    phase_step: float = np.pi
    levels: int | None = None
    design_wavelength: float = 1550e-9
    achromatic: bool = False
    efficiency_scale: float = 1.0

    def _axis_phase(self, s, period):
        frac = eng.xp.mod(s / period, 1.0)
        if self.kind == "blazed":
            return _quantize_phase(2 * np.pi * self.blaze_depth * frac, self.levels)
        if self.kind == "binary":
            return self.phase_step * (frac < 0.5).astype(eng.REAL)
        raise ValueError(f"unknown crossed grating kind {self.kind!r}")

    def phase(self, xl, yl):
        return self._axis_phase(xl, self.period_x) + self._axis_phase(yl, self.period_y)

    def unit_cell(self):
        return ((self.period_x, 0.0), (0.0, self.period_y))

    @classmethod
    def from_deflection(cls, theta_x: float, theta_y: float,
                        wavelength: float = 1550e-9, **kw):
        fx, fy = carrier_frequency(theta_x, theta_y, wavelength)
        return cls(period_x=1.0 / fx if fx else 1e9, period_y=1.0 / fy if fy else 1e9,
                   design_wavelength=wavelength, **kw)


@register("grating", "flat", "mirror", "none")
@dataclass
class FlatMirror(Grating):
    """No grating: specular reflection / plain transmission with optional loss."""

    reflectance: float = 1.0
    piston: float = 0.0  # constant phase [rad]
    design_wavelength: float = 1550e-9
    achromatic: bool = False
    efficiency_scale: float = 1.0

    def amplitude(self, xl, yl):
        return eng.xp.full(eng.asarray(xl).shape, np.sqrt(self.reflectance), dtype=eng.REAL)

    def phase(self, xl, yl):
        return eng.xp.full(eng.asarray(xl).shape, self.piston, dtype=eng.REAL)

    def orders(self, wavelength=None, **kw):
        return [Order(0, 0, 0.0, 0.0, self.reflectance * self.efficiency_scale)]
