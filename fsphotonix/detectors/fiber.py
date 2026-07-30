"""Fibre-coupled screens: single-mode (coherent, direction selective) and
multimode (NA-limited, direction tolerant).

These are optical-to-optical elements as much as detectors: the coupled amplitude
can be sent onward through a fibre network, or converted by an attached
photodiode.  Both classes therefore expose :meth:`couple` (complex amplitudes per
fibre per channel) alongside :meth:`detect` (coupled power).

Single-mode coupling is an overlap integral with the LP01 mode, so it is
intrinsically selective in *both* position and incidence angle -- a tilted
wavefront simply does not couple.  Multimode coupling accepts everything inside
the numerical aperture, which is why an MMF is the pragmatic collector for a
wide-angle holographic fan-out.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import engine as eng
from ..fields import Channel, FieldSet
from ..grid import Grid
from ..registry import register
from .base import Detector, DetectorOutput


@dataclass
class _FiberArrayBase(Detector):
    name: str = "fiber_array"
    nx: int = 8
    ny: int = 1
    pitch: float = 250e-6
    positions_in: object = None
    center: tuple[float, float] = (0.0, 0.0)
    tilt: tuple[float, float] = (0.0, 0.0)  # array pointing [rad]
    insertion_loss_db: float = 0.5
    n_time: int = 16

    @property
    def positions(self) -> np.ndarray:  # type: ignore[override]
        if self.positions_in is not None:
            return np.asarray(self.positions_in, dtype=float).reshape(-1, 2)
        xs = (np.arange(self.nx) - (self.nx - 1) / 2) * self.pitch + self.center[0]
        ys = (np.arange(self.ny) - (self.ny - 1) / 2) * self.pitch + self.center[1]
        X, Y = np.meshgrid(xs, ys)
        return np.stack([X.ravel(), Y.ravel()], axis=1)

    @property
    def loss(self) -> float:
        return 10.0 ** (-self.insertion_loss_db / 10.0)

    def couple(self, fields: FieldSet):  # pragma: no cover - abstract
        raise NotImplementedError

    def detect(self, fields: FieldSet, t=None) -> DetectorOutput:
        amps = self.couple(fields)  # {channel_index: complex (N,)}
        pos = self.positions
        n = len(pos)
        ts = np.atleast_1d(np.asarray(t, dtype=float)) if t is not None else np.array([0.0])

        # coherent sum inside each coherence group, incoherent across groups
        total = np.zeros((ts.size, n))
        groups: dict[str, list[int]] = {}
        for i, ch in enumerate(fields.channels):
            groups.setdefault(ch.coherence, []).append(i)
        for idxs in groups.values():
            acc = np.zeros((ts.size, n), dtype=complex)
            for i in idxs:
                a = np.asarray(eng.to_numpy(fields.channels[i].waveform.sample(ts))).reshape(-1, 1)
                acc = acc + a * amps[i].reshape(1, -1)
            total += np.abs(acc) ** 2
        total *= self.loss
        p = total.mean(axis=0)
        return DetectorOutput(values=p, unit="W", optical_power=p, positions=pos,
                              shape=(n,), times=ts,
                              meta={"amplitudes": amps, "time_resolved": total})

    def coupling_efficiency(self, fields: FieldSet) -> np.ndarray:
        """Coupled power / incident power per fibre."""
        out = self.detect(fields)
        tot = fields.total_power()
        return np.asarray(out.values, dtype=float) / max(tot, 1e-300)


@register("detector", "smf", "smf_array", "single_mode_fiber")
@dataclass
class SMFArray(_FiberArrayBase):
    """Single-mode fibre array.

    Coupling is ``|<u, psi>|^2`` with ``psi`` the normalised LP01 mode (Gaussian
    of mode-field diameter ``mfd``), optionally behind a microlens of focal length
    ``lens_f`` -- i.e. the fibre accepts one spatial mode arriving from one
    direction, set by ``tilt``.
    """

    name: str = "smf_array"
    mfd: float = 10.4e-6  # SMF-28 at 1550 nm
    lens_f: float | None = None  # microlens focal length; None -> bare fibre
    lens_diameter: float = 250e-6
    mode_scaling: bool = True  # scale MFD with wavelength

    def _mode(self, grid: Grid, px: float, py: float, wavelength: float):
        w = self.mfd / 2
        if self.mode_scaling:
            w = w * wavelength / 1550e-9
        if self.lens_f is not None:
            # a microlens maps the fibre mode to a larger collimated waist
            w = wavelength * self.lens_f / (np.pi * w)
            w = min(w, self.lens_diameter / 2)
        X, Y = grid.meshgrid()
        psi = eng.xp.exp(-(((X - px) ** 2 + (Y - py) ** 2) / w**2)).astype(eng.COMPLEX)
        if any(self.tilt):
            k = 2 * np.pi / wavelength
            psi = psi * eng.xp.exp(1j * k * (np.sin(self.tilt[0]) * (X - px)
                                             + np.sin(self.tilt[1]) * (Y - py)))
        norm = float(eng.to_numpy(eng.xp.sum(eng.xp.abs(psi) ** 2)).item()) * grid.cell_area
        return psi / np.sqrt(norm) if norm > 0 else psi

    def couple(self, fields: FieldSet):
        pos = self.positions
        grid = fields.grid
        out: dict[int, np.ndarray] = {}
        cache: dict[tuple[int, float], object] = {}
        for i, ch in enumerate(fields.channels):
            a = np.zeros(len(pos), dtype=complex)
            for j, (px, py) in enumerate(pos):
                key = (j, ch.wavelength)
                psi = cache.get(key)
                if psi is None:
                    psi = cache[key] = self._mode(grid, px, py, ch.wavelength)
                ov = eng.xp.sum(ch.u * eng.xp.conj(psi)) * grid.cell_area
                a[j] = complex(eng.to_numpy(ov))
            out[i] = a
        return out

    def info(self) -> dict:
        return {**super().info(), "mfd_um": self.mfd * 1e6, "lens_f": self.lens_f}


@register("detector", "mmf", "mmf_array", "multimode_fiber")
@dataclass
class MMFArray(_FiberArrayBase):
    """Multimode fibre array.

    Coupling is modelled as the power that falls inside the core *and* inside the
    acceptance cone: the field is windowed by the core aperture, transformed to
    angular space, and truncated at ``NA``.  That reproduces the two facts that
    matter for a free-space fan-out -- a big core is forgiving about position, and
    the NA sets a hard limit on how wide an angular spread can be collected.

    ``modal_noise`` adds the speckle-driven power fluctuation a real MMF link
    shows when a coherent source fills only part of its mode set.
    """

    name: str = "mmf_array"
    core_diameter: float = 50e-6
    numerical_aperture: float = 0.2
    modal_noise: float = 0.0  # fractional rms
    seed: int = 0

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    @property
    def n_modes_estimate(self) -> int:
        """Rough mode count: ``0.5 * (pi * a * NA / lambda)^2`` at 1550 nm."""
        v = np.pi * (self.core_diameter / 2) * self.numerical_aperture / 1550e-9
        return max(1, int(0.5 * v**2))

    def couple(self, fields: FieldSet):
        pos = self.positions
        grid = fields.grid
        FX, FY = grid.freq_meshgrid()
        out: dict[int, np.ndarray] = {}
        ang_cache: dict[float, object] = {}
        for i, ch in enumerate(fields.channels):
            lam = ch.wavelength
            am = ang_cache.get(lam)
            if am is None:
                f_max = self.numerical_aperture / lam
                cx, cy = np.sin(self.tilt[0]) / lam, np.sin(self.tilt[1]) / lam
                am = ang_cache[lam] = (
                    ((FX - cx) ** 2 + (FY - cy) ** 2) <= f_max**2).astype(eng.REAL)
            a = np.zeros(len(pos), dtype=complex)
            for j, (px, py) in enumerate(pos):
                X, Y = grid.meshgrid()
                core = (((X - px) ** 2 + (Y - py) ** 2)
                        <= (self.core_diameter / 2) ** 2).astype(eng.REAL)
                windowed = ch.u * core
                filtered = eng.ifft2(eng.fft2(windowed) * am)
                p = float(eng.to_numpy(eng.xp.sum(eng.xp.abs(filtered) ** 2)).item())
                p *= grid.cell_area
                if self.modal_noise > 0:
                    p *= max(0.0, 1 + self.modal_noise * self._rng.standard_normal())
                # MMF scrambles relative phase; keep magnitude, drop phase
                a[j] = np.sqrt(max(p, 0.0))
            out[i] = a
        return out

    def detect(self, fields: FieldSet, t=None) -> DetectorOutput:
        """MMF outputs add incoherently: modal scrambling destroys the phase."""
        amps = self.couple(fields)
        pos = self.positions
        ts = np.atleast_1d(np.asarray(t, dtype=float)) if t is not None else np.array([0.0])
        total = np.zeros((ts.size, len(pos)))
        for i, ch in enumerate(fields.channels):
            w = np.abs(np.asarray(eng.to_numpy(ch.waveform.sample(ts)))).reshape(-1, 1) ** 2
            total += w * (np.abs(amps[i]) ** 2).reshape(1, -1)
        total *= self.loss
        p = total.mean(axis=0)
        return DetectorOutput(values=p, unit="W", optical_power=p, positions=pos,
                              shape=(len(pos),), times=ts,
                              meta={"n_modes": self.n_modes_estimate,
                                    "time_resolved": total})

    def info(self) -> dict:
        return {**super().info(), "core_um": self.core_diameter * 1e6,
                "NA": self.numerical_aperture, "modes": self.n_modes_estimate}


@register("detector", "custom")
@dataclass
class CustomDetector(Detector):
    """Wrap an arbitrary ``func(fields, t) -> array`` as a detection screen.

    The escape hatch for a technology this package does not model: a streak
    camera, a photon-counting SPAD array, a nonlinear OO element, a waveguide
    grating coupler bank.
    """

    name: str = "custom"
    func: object = None
    positions_in: object = ((0.0, 0.0),)
    unit: str = "W"

    @property
    def positions(self) -> np.ndarray:  # type: ignore[override]
        return np.asarray(self.positions_in, dtype=float).reshape(-1, 2)

    def detect(self, fields: FieldSet, t=None) -> DetectorOutput:
        v = np.asarray(self.func(fields, t), dtype=float)
        return DetectorOutput(values=v, unit=self.unit, positions=self.positions,
                              shape=v.shape)
