"""Temporal modulation: digital, analog and mixed pulse trains.

A :class:`Waveform` is the *complex amplitude envelope* ``a(t)`` of one optical
channel.  The full field of a channel is ``u(x, y) * a(t)``: the simulator keeps
space and time factorised, which is what lets it scale to large apertures --
propagation is applied once to ``u`` and the (cheap) time axis is only expanded
at the detector.  This is the standard quasi-monochromatic approximation and it
holds while the modulation bandwidth is small compared with the optical carrier.

Normalisation convention: waveforms are scaled so that the mean of ``|a(t)|^2``
over :attr:`Waveform.period` is 1.  A source's ``power`` is therefore its
*average* optical power.  Set ``normalize=False`` to keep raw amplitudes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from . import engine as eng
from .registry import register


class Waveform:
    """Base class: map time [s] to complex amplitude (dimensionless)."""

    #: characteristic repetition period [s] used for normalisation / plotting
    period: float = 1.0
    normalize: bool = True

    def envelope(self, t):  # pragma: no cover - abstract
        raise NotImplementedError

    # ---- public API -------------------------------------------------------- #
    def sample(self, t):
        """Complex amplitude at time(s) ``t``, honouring :attr:`normalize`."""
        a = self.envelope(eng.asarray(np.atleast_1d(np.asarray(t, dtype=float))))
        a = eng.asarray(a, dtype=eng.COMPLEX)
        if self.normalize:
            a = a * self._norm()
        return a

    def _norm(self) -> float:
        if not hasattr(self, "_norm_cache"):
            t = np.linspace(0.0, self.period, 4096, endpoint=False)
            raw = eng.to_numpy(eng.asarray(self.envelope(eng.asarray(t))))
            mean_p = float(np.mean(np.abs(raw) ** 2))
            self._norm_cache = 1.0 / np.sqrt(mean_p) if mean_p > 0 else 1.0
        return self._norm_cache

    def mean_power(self, n: int = 4096) -> float:
        t = np.linspace(0.0, self.period, n, endpoint=False)
        return float(np.mean(np.abs(eng.to_numpy(self.sample(t))) ** 2))

    def time_axis(self, n_periods: float = 1.0, n: int = 1024):
        return np.linspace(0.0, n_periods * self.period, int(n), endpoint=False)

    # ---- composition ------------------------------------------------------- #
    def __mul__(self, other):
        if isinstance(other, Waveform):
            return Product([self, other])
        return Scaled(self, complex(other))

    __rmul__ = __mul__

    def __add__(self, other):
        if isinstance(other, Waveform):
            return Sum([self, other])
        return Sum([self, Constant(amplitude=complex(other))])

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(period={self.period:.3g}s)"


# --------------------------------------------------------------------------- #
# Continuous wave and simple analog forms
# --------------------------------------------------------------------------- #
@register("waveform", "cw", "constant")
@dataclass
class Constant(Waveform):
    """Unmodulated carrier."""

    amplitude: complex = 1.0
    phase: float = 0.0
    period: float = 1e-9
    normalize: bool = False

    def envelope(self, t):
        return eng.xp.full(t.shape, self.amplitude * np.exp(1j * self.phase), dtype=eng.COMPLEX)


CW = Constant


@register("waveform", "sinusoid", "analog_tone")
@dataclass
class Sinusoid(Waveform):
    """Analog intensity tone: ``P(t) = 1 + m*sin(2 pi f t + phi)``."""

    frequency: float = 1e9
    depth: float = 1.0  # modulation index m in [0, 1]
    phase: float = 0.0
    bias: float = 1.0
    normalize: bool = True

    def __post_init__(self):
        self.period = 1.0 / self.frequency

    def envelope(self, t):
        p = self.bias + self.depth * eng.xp.sin(2 * np.pi * self.frequency * t + self.phase)
        return eng.xp.sqrt(eng.xp.clip(p, 0.0, None)).astype(eng.COMPLEX)


@register("waveform", "chirp")
@dataclass
class Chirp(Waveform):
    """Linearly swept analog tone (useful for transfer-function probing)."""

    f_start: float = 1e8
    f_stop: float = 1e10
    duration: float = 1e-6
    depth: float = 1.0

    def __post_init__(self):
        self.period = self.duration

    def envelope(self, t):
        tt = eng.xp.mod(t, self.duration)
        k = (self.f_stop - self.f_start) / self.duration
        phase = 2 * np.pi * (self.f_start * tt + 0.5 * k * tt**2)
        p = 1.0 + self.depth * eng.xp.sin(phase)
        return eng.xp.sqrt(eng.xp.clip(p, 0.0, None)).astype(eng.COMPLEX)


@register("waveform", "arbitrary", "custom")
@dataclass
class Arbitrary(Waveform):
    """Wrap any callable ``f(t) -> complex amplitude``."""

    func: Callable = lambda t: np.ones_like(t)
    period: float = 1e-9
    normalize: bool = False

    def envelope(self, t):
        return eng.asarray(self.func(t), dtype=eng.COMPLEX)


@register("waveform", "sampled")
@dataclass
class Sampled(Waveform):
    """Arbitrary waveform generator: replay samples at ``sample_rate``, looping."""

    samples: Sequence[complex] = (1.0,)
    sample_rate: float = 1e11
    interpolate: bool = True
    normalize: bool = False

    def __post_init__(self):
        self._s = np.asarray(self.samples, dtype=complex)
        self.period = len(self._s) / self.sample_rate

    def envelope(self, t):
        tn = eng.to_numpy(t) * self.sample_rate
        n = len(self._s)
        if self.interpolate:
            i0 = np.floor(tn).astype(int) % n
            i1 = (i0 + 1) % n
            frac = tn - np.floor(tn)
            out = self._s[i0] * (1 - frac) + self._s[i1] * frac
        else:
            out = self._s[np.floor(tn).astype(int) % n]
        return eng.asarray(out, dtype=eng.COMPLEX)


# --------------------------------------------------------------------------- #
# Digital pulse trains
# --------------------------------------------------------------------------- #
def _raised_edges(level_of_symbol, tn, rise: float):
    """Sample a symbol sequence with finite-rise-time transitions.

    ``tn`` is time in symbol units; ``rise`` is the 0-100% transition time as a
    fraction of the symbol period.  Uses a raised-cosine edge, which is a good
    stand-in for a bandwidth-limited driver.
    """
    n = len(level_of_symbol)
    idx = np.floor(tn).astype(int) % n
    cur = level_of_symbol[idx]
    if rise <= 0:
        return cur
    prev = level_of_symbol[(idx - 1) % n]
    frac = tn - np.floor(tn)
    w = np.clip(frac / rise, 0.0, 1.0)
    blend = 0.5 * (1 - np.cos(np.pi * w))
    return prev + (cur - prev) * blend


@register("waveform", "nrz", "ook")
@dataclass
class NRZ(Waveform):
    """Non-return-to-zero on/off keying with a finite extinction ratio.

    ``extinction_ratio_db`` is the electrical-domain P1/P0 ratio; a real
    transceiver is typically 6-13 dB.
    """

    bits: Sequence[int] = (1, 0, 1, 1, 0, 0, 1, 0)
    baud: float = 25e9
    extinction_ratio_db: float = 10.0
    rise_fraction: float = 0.2
    normalize: bool = True

    def __post_init__(self):
        b = np.asarray(self.bits, dtype=float)
        er = 10.0 ** (self.extinction_ratio_db / 10.0)
        p1, p0 = 1.0, 1.0 / er
        self._levels = np.sqrt(p0 + (p1 - p0) * b)  # amplitude levels
        self.period = len(b) / self.baud

    def envelope(self, t):
        tn = eng.to_numpy(t) * self.baud
        return eng.asarray(_raised_edges(self._levels, tn, self.rise_fraction), dtype=eng.COMPLEX)

    @classmethod
    def prbs(cls, n_bits: int = 127, seed: int = 0, **kw) -> "NRZ":
        rng = np.random.default_rng(seed)
        return cls(bits=rng.integers(0, 2, n_bits), **kw)


@register("waveform", "pam")
@dataclass
class PAM(Waveform):
    """Multi-level pulse amplitude modulation (PAM4 by default).

    ``linear_in`` selects whether the symbol levels are equispaced in optical
    *power* (as for a directly modulated laser) or in *amplitude*.
    """

    symbols: Sequence[int] = (0, 1, 2, 3, 3, 1, 0, 2)
    levels: int = 4
    baud: float = 25e9
    rise_fraction: float = 0.2
    linear_in: str = "power"
    normalize: bool = True

    def __post_init__(self):
        s = np.asarray(self.symbols, dtype=float) / max(1, self.levels - 1)
        self._levels = np.sqrt(s) if self.linear_in == "power" else s
        self.period = len(s) / self.baud

    def envelope(self, t):
        tn = eng.to_numpy(t) * self.baud
        return eng.asarray(_raised_edges(self._levels, tn, self.rise_fraction), dtype=eng.COMPLEX)


@register("waveform", "rz", "pulse_train")
@dataclass
class PulseTrain(Waveform):
    """Return-to-zero / mode-locked pulse train."""

    rep_rate: float = 10e9
    fwhm: float = 5e-12
    shape: str = "gaussian"  # gaussian | sech | square | lorentzian
    chirp: float = 0.0  # rad/s^2 quadratic phase (linear chirp parameter)
    normalize: bool = True

    def __post_init__(self):
        self.period = 1.0 / self.rep_rate

    def envelope(self, t):
        tt = eng.xp.mod(t + 0.5 * self.period, self.period) - 0.5 * self.period
        if self.shape == "gaussian":
            t0 = self.fwhm / (2 * np.sqrt(np.log(2)))
            a = eng.xp.exp(-((tt / t0) ** 2) / 2)
        elif self.shape == "sech":
            t0 = self.fwhm / 1.7627
            a = 1.0 / eng.xp.cosh(tt / t0)
        elif self.shape == "square":
            a = (eng.xp.abs(tt) <= self.fwhm / 2).astype(eng.REAL)
        elif self.shape == "lorentzian":
            a = 1.0 / (1.0 + (2 * tt / self.fwhm) ** 2)
        else:
            raise ValueError(f"unknown pulse shape {self.shape!r}")
        a = a.astype(eng.COMPLEX)
        if self.chirp:
            a = a * eng.xp.exp(1j * self.chirp * tt**2)
        return a


@register("waveform", "burst")
@dataclass
class Burst(Waveform):
    """Gate another waveform on/off -- packetised or duty-cycled operation."""

    carrier: Waveform = field(default_factory=Constant)
    on_time: float = 1e-9
    off_time: float = 1e-9
    normalize: bool = False

    def __post_init__(self):
        self.period = self.on_time + self.off_time

    def envelope(self, t):
        gate = (eng.xp.mod(t, self.period) < self.on_time).astype(eng.REAL)
        return self.carrier.sample(eng.to_numpy(t)) * gate


@register("waveform", "mixed")
@dataclass
class Mixed(Waveform):
    """Digital envelope carrying an analog sub-carrier (mixed-signal drive)."""

    digital: Waveform = field(default_factory=NRZ)
    analog: Waveform = field(default_factory=Sinusoid)
    analog_weight: float = 0.3
    normalize: bool = True

    def __post_init__(self):
        self.period = max(self.digital.period, self.analog.period)

    def envelope(self, t):
        tn = eng.to_numpy(t)
        d = self.digital.sample(tn)
        a = self.analog.sample(tn)
        w = self.analog_weight
        return d * ((1 - w) + w * a)


# --------------------------------------------------------------------------- #
# Composition & impairments
# --------------------------------------------------------------------------- #
@dataclass
class Sum(Waveform):
    terms: Sequence[Waveform] = ()
    normalize: bool = False

    def __post_init__(self):
        self.period = max((w.period for w in self.terms), default=1.0)

    def envelope(self, t):
        tn = eng.to_numpy(t)
        out = eng.zeros(t.shape)
        for w in self.terms:
            out = out + w.sample(tn)
        return out


@dataclass
class Product(Waveform):
    terms: Sequence[Waveform] = ()
    normalize: bool = False

    def __post_init__(self):
        self.period = max((w.period for w in self.terms), default=1.0)

    def envelope(self, t):
        tn = eng.to_numpy(t)
        out = eng.xp.ones(t.shape, dtype=eng.COMPLEX)
        for w in self.terms:
            out = out * w.sample(tn)
        return out


@dataclass
class Scaled(Waveform):
    inner: Waveform = field(default_factory=Constant)
    factor: complex = 1.0
    normalize: bool = False

    def __post_init__(self):
        self.period = self.inner.period

    def envelope(self, t):
        return self.inner.sample(eng.to_numpy(t)) * self.factor


@register("waveform", "noisy")
@dataclass
class Impaired(Waveform):
    """Add relative intensity noise and laser phase noise to any waveform.

    ``rin_dbc_hz`` is the single-sided RIN power spectral density; ``linewidth``
    drives a random-walk phase (Wiener process) sampled on the requested times.
    """

    inner: Waveform = field(default_factory=Constant)
    rin_dbc_hz: float = -140.0
    linewidth: float = 1e6
    bandwidth: float = 50e9
    seed: int = 0
    normalize: bool = False

    def __post_init__(self):
        self.period = self.inner.period
        self._rng = np.random.default_rng(self.seed)

    def envelope(self, t):
        tn = eng.to_numpy(t)
        a = eng.to_numpy(self.inner.sample(tn))
        # intensity noise: sigma_P/P = sqrt(RIN * BW)
        rin = 10.0 ** (self.rin_dbc_hz / 10.0)
        sigma = np.sqrt(max(rin * self.bandwidth, 0.0))
        a = a * (1.0 + 0.5 * sigma * self._rng.standard_normal(a.shape))
        if self.linewidth > 0 and a.size > 1:
            dt = np.gradient(tn) if tn.size > 1 else np.array([0.0])
            dphi = np.sqrt(np.maximum(2 * np.pi * self.linewidth * np.abs(dt), 0.0))
            a = a * np.exp(1j * np.cumsum(dphi * self._rng.standard_normal(a.shape)))
        return eng.asarray(a, dtype=eng.COMPLEX)


def as_waveform(spec) -> Waveform:
    """Coerce ``None`` / str / dict / Waveform into a :class:`Waveform`."""
    from .registry import build

    if spec is None:
        return Constant()
    if isinstance(spec, Waveform):
        return spec
    return build(spec, "waveform")
