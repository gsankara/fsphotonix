"""Composing gratings: superposition (phase multiplexing) and cascading."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ... import engine as eng
from ...registry import register
from .base import Grating, Order, measure_orders


@register("grating", "superposition", "phase_multiplex")
@dataclass
class Superposition(Grating):
    """Complex sum of several gratings, projected back to phase-only.

    This is *phase multiplexing*: one physical sub-region carries N holograms at
    once and sends light to all their targets simultaneously.  Cheaper in area
    than spatial multiplexing but noisier -- the ``arg()`` projection scatters some
    power into intermodulation orders, which :meth:`orders` reports honestly.

    ``phase_only=False`` keeps the complex sum (an ideal, non-realisable element)
    which is useful as an upper bound when studying a design.
    """

    gratings: Sequence[Grating] = field(default_factory=tuple)
    weights: Sequence[float] | None = None
    phase_only: bool = True
    levels: int | None = None
    #: aperture the hologram is written on, used to *measure* the orders
    aperture_size: tuple[float, float] = (200e-6, 200e-6)
    #: random carrier phases decorrelate the terms before the arg() projection
    randomize_phases: bool = True
    seed: int = 0
    design_wavelength: float = 1550e-9
    achromatic: bool = False
    efficiency_scale: float = 1.0

    def __post_init__(self):
        if self.gratings:
            self.design_wavelength = self.gratings[0].design_wavelength
        n = len(self.gratings)
        w = np.ones(n) if self.weights is None else np.asarray(self.weights, dtype=float)
        self._w = np.sqrt(np.clip(w, 0.0, None))
        # Without this every term is in phase at the origin, so the sum has a
        # huge peak-to-average ratio and the arg() projection -- which keeps only
        # the phase -- throws away wildly different amounts of each term.  Random
        # carrier phases decorrelate them, which is exactly why
        # MultiSpotHologram draws them too.  Seeded, so a design is reproducible.
        rng = np.random.default_rng(self.seed)
        self._carrier = (rng.uniform(0, 2 * np.pi, max(n, 1)) if self.randomize_phases
                         else np.zeros(max(n, 1)))
        self._measured: dict = {}

    def _sum(self, xl, yl, wavelength):
        acc = eng.xp.zeros(eng.asarray(xl).shape, dtype=eng.COMPLEX)
        for g, w, ph in zip(self.gratings, self._w, self._carrier):
            acc = acc + (w * np.exp(1j * ph)) * g.complex_transmittance(xl, yl,
                                                                        wavelength)
        return acc

    def complex_transmittance(self, xl, yl, wavelength=None):
        lam = wavelength or self.design_wavelength
        acc = self._sum(xl, yl, lam)
        if not self.phase_only:
            m = eng.xp.max(eng.xp.abs(acc))
            acc = acc / m if float(eng.to_numpy(m)) > 0 else acc
            return acc * np.sqrt(self.efficiency_scale)
        p = eng.xp.angle(acc)
        if self.levels:
            step = 2 * np.pi / self.levels
            p = eng.xp.floor(eng.xp.mod(p, 2 * np.pi) / step) * step
        return eng.xp.exp(1j * p).astype(eng.COMPLEX) * np.sqrt(self.efficiency_scale)

    def unit_cell(self):
        cells = [g.unit_cell() for g in self.gratings]
        if any(c is None for c in cells):
            return None
        # common cell only exists if all 1-D gratings share a lattice direction
        a1s = [c[0] for c in cells]
        if all(c[1] is None for c in cells) and len(a1s) > 1:
            angs = [np.arctan2(a[1], a[0]) % np.pi for a in a1s]
            if max(angs) - min(angs) < 1e-9:
                lens = [float(np.hypot(*a)) for a in a1s]
                lcm = _period_lcm(lens)
                if lcm is not None:
                    th = angs[0]
                    return ((lcm * np.cos(th), lcm * np.sin(th)), None)
            return None
        return cells[0] if len(cells) == 1 else None

    def orders(self, wavelength=None, samples: int = 256, **kw):
        """Diffraction orders, *measured* rather than apportioned.

        The periodic branch (all constituents sharing a lattice) still computes
        them from the common unit cell.  The aperiodic branch -- which is what
        ``route(mode="superpose")`` always takes, since it builds ``SpotFocuser``
        lenses -- rasterises this hologram over ``aperture_size``, transforms it
        and integrates the power actually arriving at each constituent own
        direction.

        This used to apportion instead: each constituent analytic orders scaled
        by its power share.  Since ``SpotFocuser`` reports unit efficiency, the
        sum came to exactly 1.0 -- a phase-only superposition claiming to lose
        nothing to intermodulation, which is the one thing it certainly does
        lose.  :meth:`stray_fraction` is now that missing power, and the class
        docstring promise that ``orders`` reports it honestly is finally true.
        """
        cell = self.unit_cell()
        if cell is not None:
            return super().orders(wavelength, **kw)
        lam = wavelength or self.design_wavelength
        got = self._measure(lam, samples)
        # Returned in *constituent* order, so entry i belongs to gratings[i] and
        # lines up with weights[i] -- which is what a caller comparing delivered
        # against requested weights needs, and what MultiSpotHologram already
        # does. Sorting by efficiency here (as this branch used to) silently
        # permutes that correspondence; Order.mx carries the index either way.
        return [Order(i, 0, o[0], o[1], eta * self.efficiency_scale)
                for i, (o, eta) in enumerate(zip(got["orders"], got["efficiency"]))]

    def _measure(self, lam: float, samples: int) -> dict:
        key = (round(lam, 18), int(samples))
        hit = self._measured.get(key)
        if hit is not None:
            return hit
        dirs = []
        for g in self.gratings:
            os_ = g.orders(lam)
            pick = next((o for o in os_ if (o.mx, o.my) != (0, 0)),
                        os_[0] if os_ else None)
            dirs.append((pick.fx, pick.fy) if pick is not None else (0.0, 0.0))
        got = measure_orders(self, self.aperture_size, dirs, lam, samples=samples)
        got["orders"] = dirs
        self._measured[key] = got
        return got

    def stray_fraction(self, wavelength=None, samples: int = 256) -> float:
        """Power that reached no designed direction: intermodulation and speckle."""
        lam = wavelength or self.design_wavelength
        return float(self._measure(lam, samples)["stray"])

    def uniformity(self, wavelength=None, samples: int = 256) -> float:
        """Relative rms spread of the delivered weights against those requested.

        The figure of merit for phase multiplexing in a matrix-vector product,
        where every target is an intended output and the error *is* the spread.
        """
        lam = wavelength or self.design_wavelength
        eff = np.asarray(self._measure(lam, samples)["efficiency"], dtype=float)
        if eff.size == 0 or eff.sum() <= 0:
            return 0.0
        want = self._w**2
        want = want / want.sum() if want.sum() > 0 else np.ones_like(eff) / eff.size
        got = eff / eff.sum()
        return float(np.sqrt(np.mean((got - want) ** 2)) / max(want.mean(), 1e-30))

    def resolved(self, wavelength=None, samples: int = 256) -> bool:
        """False when two targets sit closer than this aperture can separate."""
        lam = wavelength or self.design_wavelength
        return bool(self._measure(lam, samples)["resolved"])


def _period_lcm(lengths: Sequence[float], tol: float = 1e-3, max_mult: int = 64):
    """Smallest common period of several 1-D grating periods, if a small one exists."""
    base = min(lengths)
    for k in range(1, max_mult + 1):
        cand = base * k
        if all(abs(cand / L - round(cand / L)) < tol for L in lengths):
            return cand
    return None


@register("grating", "cascade", "product")
@dataclass
class Cascade(Grating):
    """Multiply transmittances: e.g. a deflector *and* a lens *and* an apodiser."""

    gratings: Sequence[Grating] = field(default_factory=tuple)
    design_wavelength: float = 1550e-9
    achromatic: bool = False
    efficiency_scale: float = 1.0

    def __post_init__(self):
        if self.gratings:
            self.design_wavelength = self.gratings[0].design_wavelength

    def complex_transmittance(self, xl, yl, wavelength=None):
        lam = wavelength or self.design_wavelength
        out = eng.xp.ones(eng.asarray(xl).shape, dtype=eng.COMPLEX)
        for g in self.gratings:
            out = out * g.complex_transmittance(xl, yl, lam)
        return out * np.sqrt(self.efficiency_scale)

    def unit_cell(self):
        cells = [g.unit_cell() for g in self.gratings]
        periodic = [c for c in cells if c is not None]
        if len(periodic) != len(cells) or not periodic:
            return None
        return periodic[0] if len(periodic) == 1 else None

    def orders(self, wavelength=None, **kw):
        cell = self.unit_cell()
        if cell is not None:
            return super().orders(wavelength, **kw)
        # convolution of the individual order sets
        sets = [g.orders(wavelength, **kw) for g in self.gratings]
        out = [Order(0, 0, 0.0, 0.0, 1.0)]
        for s in sets:
            nxt = []
            for a in out:
                for b in s:
                    nxt.append(Order(a.mx + b.mx, a.my + b.my, a.fx + b.fx, a.fy + b.fy,
                                     a.efficiency * b.efficiency))
            out = sorted(nxt, key=lambda o: -o.efficiency)[:64]
        return [Order(o.mx, o.my, o.fx, o.fy, o.efficiency * self.efficiency_scale) for o in out]


@register("grating", "apodizer", "amplitude_mask")
@dataclass
class Apodizer(Grating):
    """Pure amplitude shaping ``func(xl, yl) -> [0, 1]``.

    Combine with :class:`Cascade` to weight a sub-region's contribution without
    changing where it points -- a simple way to set matrix coefficients on a
    device that only has binary steering.
    """

    func: object = None
    design_wavelength: float = 1550e-9
    achromatic: bool = True
    efficiency_scale: float = 1.0

    def amplitude(self, xl, yl):
        if self.func is None:
            return eng.xp.ones_like(eng.asarray(xl, dtype=eng.REAL))
        return eng.xp.clip(eng.asarray(self.func(xl, yl), dtype=eng.REAL), 0.0, 1.0)

    def unit_cell(self):
        return None

    def orders(self, wavelength=None, **kw):
        return [Order(0, 0, 0.0, 0.0, self.efficiency_scale)]
