"""Composing gratings: superposition (phase multiplexing) and cascading."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ... import engine as eng
from ...registry import register
from .base import Grating, Order


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
    design_wavelength: float = 1550e-9
    achromatic: bool = False
    efficiency_scale: float = 1.0

    def __post_init__(self):
        if self.gratings:
            self.design_wavelength = self.gratings[0].design_wavelength
        n = len(self.gratings)
        w = np.ones(n) if self.weights is None else np.asarray(self.weights, dtype=float)
        self._w = np.sqrt(np.clip(w, 0.0, None))

    def _sum(self, xl, yl, wavelength):
        acc = eng.xp.zeros(eng.asarray(xl).shape, dtype=eng.COMPLEX)
        for g, w in zip(self.gratings, self._w):
            acc = acc + w * g.complex_transmittance(xl, yl, wavelength)
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

    def orders(self, wavelength=None, **kw):
        cell = self.unit_cell()
        if cell is not None:
            return super().orders(wavelength, **kw)
        # aperiodic: union of the constituents' orders, scaled by measured power
        out: list[Order] = []
        tot = float(sum(w**2 for w in self._w)) or 1.0
        for g, w in zip(self.gratings, self._w):
            share = w**2 / tot
            for o in g.orders(wavelength, **kw):
                out.append(Order(o.mx, o.my, o.fx, o.fy, o.efficiency * share))
        return sorted(out, key=lambda o: -o.efficiency)


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
