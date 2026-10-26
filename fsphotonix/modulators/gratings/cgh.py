"""Computer-generated holograms: arbitrary multi-spot fan-out from one sub-region.

Two synthesis routes are provided, both of which return a *phase-only* hologram
(what an SLM or a piston micro-mirror array can actually display):

``superposition``
    analytic sum of one focusing/deflecting term per target with random or
    optimised carrier phases, then take the argument.  Grid-free, so the same
    hologram object can be rasterised at any sampling; fast enough to re-synthesise
    inside a feedback loop.  Uniformity is decent, efficiency ~70-90%.

``gs``
    Gerchberg-Saxton (with optional weighting, i.e. WGS) on an internal grid,
    then nearest-neighbour lookup when rasterised.  Slower, better uniformity.

Both honour per-target ``weights``, so one region can distribute unequal power to
many screen locations -- the optical fan-out that implements a row of a matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ... import engine as eng
from ...registry import register
from .base import (Grating, Order, carrier_frequency, measure_orders,
                   target_to_angles)


@register("grating", "phase_map", "raw")
@dataclass
class PhaseMap(Grating):
    """A pre-computed phase (and optional amplitude) bitmap.

    The escape hatch: load a hologram from a file, from a fabrication mask, or
    from an external optimiser and use it like any other grating.
    """

    phase_bitmap: object = None  # (ny, nx) array of radians
    amplitude_bitmap: object = None
    extent: tuple[float, float] = (1e-3, 1e-3)
    design_wavelength: float = 1550e-9
    achromatic: bool = False
    efficiency_scale: float = 1.0
    wrap: bool = False  # tile the bitmap instead of clipping to its extent

    def _lookup(self, bm, xl, yl, fill: float):
        bm = np.asarray(bm, dtype=float)
        ny, nx = bm.shape
        w, h = self.extent
        xn, yn = eng.to_numpy(xl), eng.to_numpy(yl)
        cx = ((xn + w / 2) / w * nx)
        cy = ((yn + h / 2) / h * ny)
        if self.wrap:
            ix = np.mod(cx.astype(int), nx)
            iy = np.mod(cy.astype(int), ny)
            return eng.asarray(bm[iy, ix], dtype=eng.REAL)
        ix = np.clip(cx.astype(int), 0, nx - 1)
        iy = np.clip(cy.astype(int), 0, ny - 1)
        inside = (np.abs(xn) <= w / 2) & (np.abs(yn) <= h / 2)
        return eng.asarray(np.where(inside, bm[iy, ix], fill), dtype=eng.REAL)

    def phase(self, xl, yl):
        if self.phase_bitmap is None:
            return eng.xp.zeros_like(eng.asarray(xl, dtype=eng.REAL))
        return self._lookup(self.phase_bitmap, xl, yl, 0.0)

    def amplitude(self, xl, yl):
        if self.amplitude_bitmap is None:
            return eng.xp.ones_like(eng.asarray(xl, dtype=eng.REAL))
        return self._lookup(self.amplitude_bitmap, xl, yl, 0.0)

    def unit_cell(self):
        if not self.wrap:
            return None
        return ((self.extent[0], 0.0), (0.0, self.extent[1]))


@dataclass(frozen=True)
class SpotTarget:
    """One requested screen location."""

    x: float
    y: float
    z: float = 0.1
    weight: float = 1.0
    label: str = ""


@register("grating", "multispot", "cgh", "hologram")
@dataclass
class MultiSpotHologram(Grating):
    """Phase-only hologram that splits one sub-region's light into many spots.

    ``targets`` is a list of ``SpotTarget`` or ``(x, y)`` / ``(x, y, z)`` /
    ``(x, y, z, weight)`` tuples, in screen coordinates at range ``range_z``.
    """

    targets: Sequence = ()
    range_z: float = 0.1
    method: str = "superposition"  # superposition | gs | wgs
    focusing: bool = True  # each term focuses at range_z (else pure deflection)
    levels: int | None = None
    iterations: int = 40
    aperture_size: tuple[float, float] = (200e-6, 200e-6)
    gs_samples: int = 128
    seed: int = 0
    design_wavelength: float = 1550e-9
    achromatic: bool = False
    efficiency_scale: float = 1.0

    def __post_init__(self):
        self.spots = [self._coerce(t) for t in self.targets]
        rng = np.random.default_rng(self.seed)
        self._carrier_phase = rng.uniform(0, 2 * np.pi, max(len(self.spots), 1))
        self._gs_cache = None

    def _coerce(self, t) -> SpotTarget:
        if isinstance(t, SpotTarget):
            return t
        v = list(t)
        return SpotTarget(x=float(v[0]), y=float(v[1]),
                          z=float(v[2]) if len(v) > 2 else self.range_z,
                          weight=float(v[3]) if len(v) > 3 else 1.0)

    # ---- synthesis --------------------------------------------------------- #
    def _analytic_field(self, xl, yl):
        """Complex superposition of one term per target (design wavelength)."""
        k = 2 * np.pi / self.design_wavelength
        acc = eng.xp.zeros(eng.asarray(xl).shape, dtype=eng.COMPLEX)
        for s, ph in zip(self.spots, self._carrier_phase):
            a = np.sqrt(max(s.weight, 0.0))
            if self.focusing:
                r = eng.xp.sqrt((xl - s.x) ** 2 + (yl - s.y) ** 2 + s.z**2)
                term = eng.xp.exp(-1j * (k * r) + 1j * ph)
            else:
                tx, ty = target_to_angles(s.x, s.y, s.z)
                fx, fy = carrier_frequency(tx, ty, self.design_wavelength)
                term = eng.xp.exp(1j * (2 * np.pi * (fx * xl + fy * yl) + ph))
            acc = acc + a * term
        return acc

    def _gs_phase(self):
        """Weighted Gerchberg-Saxton on an internal aperture grid."""
        if self._gs_cache is not None:
            return self._gs_cache
        n = int(self.gs_samples)
        w, h = self.aperture_size
        x = (np.arange(n) - n // 2) * (w / n)
        y = (np.arange(n) - n // 2) * (h / n)
        lam = self.design_wavelength

        # target spatial frequencies -> nearest FFT bins on this aperture grid
        fx_ax = np.fft.fftshift(np.fft.fftfreq(n, w / n))
        fy_ax = np.fft.fftshift(np.fft.fftfreq(n, h / n))
        idx, weights = [], []
        for s in self.spots:
            tx, ty = target_to_angles(s.x, s.y, s.z)
            fx, fy = carrier_frequency(tx, ty, lam)
            ix = int(np.argmin(np.abs(fx_ax - fx)))
            iy = int(np.argmin(np.abs(fy_ax - fy)))
            idx.append((iy, ix))
            weights.append(max(s.weight, 0.0))
        weights = np.asarray(weights, dtype=float)
        if weights.sum() > 0:
            weights = weights / weights.sum()

        rng = np.random.default_rng(self.seed)
        phase = rng.uniform(0, 2 * np.pi, (n, n))
        illum = np.ones((n, n))
        target_amp = np.sqrt(weights)
        gain = np.ones(len(self.spots))

        for _ in range(max(1, self.iterations)):
            far = np.fft.fftshift(np.fft.fft2(illum * np.exp(1j * phase)))
            got = np.array([far[iy, ix] for iy, ix in idx])
            mag = np.abs(got)
            if self.method == "wgs" and mag.max() > 0:
                # damped multiplicative correction toward the requested share
                share = mag / (mag.sum() + 1e-300)
                want = target_amp / (target_amp.sum() + 1e-300)
                gain = np.clip(gain * np.sqrt(want / (share + 1e-12)), 1e-3, 1e3)
            new = np.zeros_like(far)
            for (iy, ix), a, g, ph in zip(idx, target_amp, gain,
                                          np.angle(got) if mag.max() > 0 else np.zeros(len(idx))):
                new[iy, ix] = a * g * np.exp(1j * ph)
            back = np.fft.ifft2(np.fft.ifftshift(new))
            phase = np.angle(back)

        self._gs_cache = (x, y, phase)
        return self._gs_cache

    def phase(self, xl, yl):
        if self.method in ("gs", "wgs"):
            x, y, ph = self._gs_phase()
            xn, yn = eng.to_numpy(xl), eng.to_numpy(yl)
            ix = np.clip(np.round((xn - x[0]) / (x[1] - x[0])).astype(int), 0, len(x) - 1)
            iy = np.clip(np.round((yn - y[0]) / (y[1] - y[0])).astype(int), 0, len(y) - 1)
            p = eng.asarray(ph[iy, ix], dtype=eng.REAL)
        else:
            p = eng.xp.angle(self._analytic_field(xl, yl)).astype(eng.REAL)
        if self.levels:
            step = 2 * np.pi / self.levels
            p = eng.xp.floor(eng.xp.mod(p, 2 * np.pi) / step) * step
        return p

    def unit_cell(self):
        return None

    # ---- analysis ---------------------------------------------------------- #
    def orders(self, wavelength: float | None = None, samples: int = 256, **kw) -> list[Order]:
        """Measured power fraction in each designed spot direction.

        The hologram is rasterised over its aperture, Fourier transformed, and the
        energy in a diffraction-limited window around each target direction is
        integrated -- so the returned efficiencies include the real cross-talk and
        speckle losses of the phase-only approximation.
        """
        lam = wavelength or self.design_wavelength
        if not self.spots:
            return [Order(0, 0, 0.0, 0.0, self.efficiency_scale)]
        cache = getattr(self, "_orders_cache", None)
        if cache is None:
            cache = self._orders_cache = {}
        key = (round(lam, 18), int(samples))
        if key in cache:
            return cache[key]
        dirs = []
        for s in self.spots:
            tx, ty = target_to_angles(s.x, s.y, s.z)
            dirs.append(carrier_frequency(tx, ty, lam))
        got = measure_orders(self, self.aperture_size, dirs, lam, samples=int(samples))
        out = [Order(i, 0, d[0], d[1], eta * self.efficiency_scale)
               for i, (d, eta) in enumerate(zip(dirs, got["efficiency"]))]
        self._last_measure = got
        cache[key] = out
        return out

    def stray_fraction(self, wavelength: float | None = None,
                       samples: int = 256) -> float:
        """Power reaching no designed spot: intermodulation and speckle loss."""
        self.orders(wavelength, samples=samples)
        return float(self._last_measure["stray"])

    def resolved(self, wavelength: float | None = None, samples: int = 256) -> bool:
        """False when two targets sit closer than this aperture can separate."""
        self.orders(wavelength, samples=samples)
        return bool(self._last_measure["resolved"])

    def uniformity(self, wavelength: float | None = None) -> float:
        e = np.array([o.efficiency for o in self.orders(wavelength)])
        if e.size == 0 or e.max() <= 0:
            return 1.0
        return float((e.max() - e.min()) / (e.max() + e.min()))

    @classmethod
    def fanout_grid(cls, nx: int, ny: int, pitch: float, z: float,
                    weights: Sequence[float] | None = None, **kw) -> "MultiSpotHologram":
        """Regular ``nx x ny`` spot lattice of pitch ``pitch`` at range ``z``."""
        xs = (np.arange(nx) - (nx - 1) / 2) * pitch
        ys = (np.arange(ny) - (ny - 1) / 2) * pitch
        tg = []
        k = 0
        for yy in ys:
            for xx in xs:
                wt = 1.0 if weights is None else float(weights[k])
                tg.append(SpotTarget(x=float(xx), y=float(yy), z=z, weight=wt))
                k += 1
        return cls(targets=tg, range_z=z, **kw)
