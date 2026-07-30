"""Detection screen abstraction (plugin kind ``"detector"``).

A detector turns the optical field arriving at a plane into numbers.  The
technology choice changes the answer qualitatively, not just quantitatively:

* a **camera** integrates irradiance over a pixel and over an exposure -- spatially
  resolved, slow, shot- and read-noise limited;
* a **photodiode array** is fast and resolves the temporal waveform, but only
  over its (few) active areas;
* a **single-mode fibre** array is a *coherent* detector: coupling is an overlap
  integral, so it selects one incident direction and mode, and it is an OO
  element -- the light can go on to another stage;
* a **multimode fibre** array accepts any direction inside its NA and mixes the
  modes, which is the pragmatic way to collect a wide fan-out.

Screens can also be replaced by another :class:`ModulatorArray`, in which case no
detector is involved at all -- see :mod:`fsphotonix.system`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import engine as eng
from ..fields import FieldSet
from ..grid import Grid
from ..units import Q_E, photon_energy


@dataclass
class DetectorOutput:
    """Result of a detection: electronic values plus the optical truth behind them."""

    values: np.ndarray  # the detector's own output (DN, volts, amps, watts)
    unit: str = "W"
    optical_power: np.ndarray | None = None  # W per element, noiseless
    positions: np.ndarray | None = None  # (N, 2) element centres [m]
    times: np.ndarray | None = None  # for time-resolved detectors
    shape: tuple[int, ...] | None = None
    saturated: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    def vector(self) -> np.ndarray:
        """Flat 1-D view -- what a feedback controller consumes."""
        return np.asarray(self.values, dtype=float).ravel()

    def image(self) -> np.ndarray:
        """2-D view when the detector has a rectangular layout."""
        v = np.asarray(self.values, dtype=float)
        if self.shape and len(self.shape) == 2:
            return v.reshape(self.shape)
        return v

    def snr_db(self) -> np.ndarray:
        """Per-element SNR estimate from the noiseless reference."""
        if self.optical_power is None:
            return np.full(np.asarray(self.values).shape, np.nan)
        sig = np.asarray(self.optical_power, dtype=float)
        v = np.asarray(self.values, dtype=float)
        scale = sig.sum() / max(abs(v).sum(), 1e-300)
        noise = np.abs(v * scale - sig)
        return 20 * np.log10(np.maximum(sig, 1e-300) / np.maximum(noise, 1e-300))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        v = np.asarray(self.values, dtype=float)
        return (f"DetectorOutput({self.shape or v.shape}, unit={self.unit}, "
                f"min={v.min():.4g}, max={v.max():.4g})")


class Detector:
    """Base class for detection-screen technologies."""

    name: str = "detector"
    #: element centres, filled in by subclasses
    positions: np.ndarray

    def detect(self, fields: FieldSet, t=None) -> DetectorOutput:  # pragma: no cover
        raise NotImplementedError

    def __call__(self, fields: FieldSet, **kw) -> DetectorOutput:
        return self.detect(fields, **kw)

    # ---- shared helpers ---------------------------------------------------- #
    @property
    def n_elements(self) -> int:
        return int(np.asarray(self.positions).reshape(-1, 2).shape[0])

    def info(self) -> dict:
        return {"name": self.name, "type": type(self).__name__,
                "elements": self.n_elements}

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}({self.name!r}, {self.n_elements} elements)"


# --------------------------------------------------------------------------- #
# shared numerics
# --------------------------------------------------------------------------- #
def time_samples(fields: FieldSet, window: float, n: int) -> np.ndarray:
    """Time grid covering ``window``, at least Nyquist for the fastest waveform.

    ``n <= 1`` is an explicit request for a single (t = 0) sample -- "do not resolve
    time" -- and is honoured as given; the Nyquist floor only applies when the
    caller has asked for a time series.
    """
    if window <= 0 or int(n) <= 1:
        return np.array([0.0])
    periods = [ch.waveform.period for ch in fields.channels if ch.waveform.period > 0]
    fastest = min(periods) if periods else window
    slowest = max(periods) if periods else window
    # An exposure spanning many modulation periods has the same *average*
    # irradiance as one period of the slowest waveform, so sample that instead of
    # the whole window: a 2 us camera exposure over a 25 Gbaud pattern needs 16
    # samples, not the 400 000 that Nyquist over the full window would demand.
    span = min(window, slowest)
    need = int(np.clip(np.ceil(8 * span / fastest), 1, 4096))
    return np.linspace(0.0, span, max(int(n), need), endpoint=False)


def mean_irradiance(fields: FieldSet, window: float, n_time: int = 32,
                    beat_bandwidth: float | None = None, max_samples: int = 16):
    """Exposure-averaged irradiance [W/m^2] on the field grid.

    Averaged in chunks of at most ``max_samples`` time steps, so a long time axis
    over a large screen never materialises an ``(nt, ny, nx)`` cube.
    """
    t = time_samples(fields, window, n_time)
    if t.size == 1:
        return fields.intensity(0.0, beat_bandwidth=beat_bandwidth)
    acc = eng.xp.zeros(fields.grid.shape, dtype=eng.REAL)
    for i in range(0, t.size, max_samples):
        block = t[i : i + max_samples]
        acc = acc + eng.xp.sum(fields.intensity(block, beat_bandwidth=beat_bandwidth),
                               axis=0)
    return acc / t.size


def block_sum(a, by: int, bx: int, oy: int = 0, ox: int = 0, ny: int | None = None,
              nx: int | None = None):
    """Sum an array over (by, bx) blocks starting at offset (oy, ox)."""
    h, w = a.shape
    ny = (h - oy) // by if ny is None else ny
    nx = (w - ox) // bx if nx is None else nx
    ny = max(min(ny, (h - oy) // by), 0)
    nx = max(min(nx, (w - ox) // bx), 0)
    if ny == 0 or nx == 0:
        return eng.xp.zeros((max(ny, 0), max(nx, 0)), dtype=a.dtype)
    v = a[oy : oy + ny * by, ox : ox + nx * bx]
    return v.reshape(ny, by, nx, bx).sum(axis=(1, 3))


def apply_bandwidth(signal: np.ndarray, t: np.ndarray, bandwidth: float) -> np.ndarray:
    """First-order (RC) low-pass along the last axis, applied in the frequency domain."""
    if not np.isfinite(bandwidth) or bandwidth <= 0 or t.size < 4:
        return signal
    dt = float(np.mean(np.diff(t)))
    f = np.fft.fftfreq(t.size, dt)
    h = 1.0 / (1.0 + 1j * f / bandwidth)
    return np.real(np.fft.ifft(np.fft.fft(signal, axis=-1) * h, axis=-1))


def shot_noise_electrons(mean_e: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Poisson shot noise, with a Gaussian approximation for large counts."""
    out = np.asarray(mean_e, dtype=float)
    small = out < 1e6
    res = np.empty_like(out)
    if small.any():
        res[small] = rng.poisson(np.maximum(out[small], 0.0))
    if (~small).any():
        m = out[~small]
        res[~small] = m + rng.standard_normal(m.shape) * np.sqrt(m)
    return res


def photons_from_energy(energy_j: np.ndarray, wavelength: float) -> np.ndarray:
    return np.asarray(energy_j, dtype=float) / photon_energy(wavelength)


def as_detector(spec) -> Detector:
    from ..registry import build

    if isinstance(spec, Detector):
        return spec
    return build(spec, "detector")
