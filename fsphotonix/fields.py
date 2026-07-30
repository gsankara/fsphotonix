"""Optical field representation.

A :class:`Channel` is one spatial mode at one wavelength with one temporal
envelope.  A :class:`FieldSet` is the bag of channels travelling through the
system together.

Coherence bookkeeping is explicit and matters a great deal for this class of
machine.  Channels that share a ``coherence`` tag interfere (their complex
amplitudes add); channels with different tags only add in *intensity*.  A single
laser split N ways -> one tag.  A bank of COTS transceivers on a 100 GHz grid ->
N different tags, because the ~100 GHz beat notes are far outside any camera or
photodiode bandwidth and average to zero.  Set
``FieldSet.allow_cross_wavelength_beat`` if you deliberately want to model the
beat term (e.g. a coherent comb with a fast detector).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import count
from typing import Iterable, Iterator, Sequence

import numpy as np

from . import engine as eng
from .grid import Grid
from .units import C0
from .waveform import Constant, Waveform

_uid = count()


@dataclass
class Channel:
    """One spatial mode: ``E(x, y, t) = u(x, y) * a(t) * exp(-i 2 pi nu t)``.

    ``u`` has units of sqrt(W)/m so that ``sum |u|^2 dx dy`` is watts.
    """

    u: "np.ndarray"
    wavelength: float
    coherence: str = ""
    waveform: Waveform = field(default_factory=Constant)
    label: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.coherence:
            self.coherence = f"grp{next(_uid)}"
        self.u = eng.asarray(self.u, dtype=eng.COMPLEX)

    # ---- properties -------------------------------------------------------- #
    @property
    def frequency(self) -> float:
        return C0 / self.wavelength

    @property
    def k(self) -> float:
        return 2 * np.pi / self.wavelength

    def power(self, grid: Grid) -> float:
        """Average optical power in this channel [W]."""
        p = float(eng.to_numpy(eng.xp.sum(eng.xp.abs(self.u) ** 2)).item()) * grid.cell_area
        return p * self.waveform.mean_power()

    def with_u(self, u) -> "Channel":
        return replace(self, u=u)

    def scaled(self, factor) -> "Channel":
        return replace(self, u=self.u * factor)

    def copy(self) -> "Channel":
        return replace(self, u=eng.xp.copy(self.u), meta=dict(self.meta))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Channel({self.label or '-'}, lam={self.wavelength*1e9:.3f}nm, "
            f"coh={self.coherence}, {self.u.shape})"
        )


@dataclass
class FieldSet:
    """Channels sharing one transverse plane."""

    grid: Grid
    channels: list[Channel] = field(default_factory=list)
    z: float = 0.0
    allow_cross_wavelength_beat: bool = False
    meta: dict = field(default_factory=dict)

    # ---- container protocol ------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.channels)

    def __iter__(self) -> Iterator[Channel]:
        return iter(self.channels)

    def __getitem__(self, i):
        return self.channels[i]

    def add(self, channel: Channel) -> "FieldSet":
        self.channels.append(channel)
        return self

    def extend(self, channels: Iterable[Channel]) -> "FieldSet":
        self.channels.extend(channels)
        return self

    # ---- construction ------------------------------------------------------ #
    @classmethod
    def merge(cls, sets: Sequence["FieldSet"]) -> "FieldSet":
        """Combine field sets defined on the same grid."""
        sets = [s for s in sets if s is not None]
        if not sets:
            raise ValueError("nothing to merge")
        g = sets[0].grid
        for s in sets[1:]:
            if s.grid.shape != g.shape or abs(s.grid.dx - g.dx) > 1e-18:
                raise ValueError("cannot merge field sets on different grids")
        out = cls(grid=g, z=sets[0].z, allow_cross_wavelength_beat=any(
            s.allow_cross_wavelength_beat for s in sets))
        for s in sets:
            out.channels.extend(s.channels)
        return out

    def like(self, channels: list[Channel], grid: Grid | None = None, z: float | None = None) -> "FieldSet":
        return FieldSet(
            grid=grid if grid is not None else self.grid,
            channels=channels,
            z=self.z if z is None else z,
            allow_cross_wavelength_beat=self.allow_cross_wavelength_beat,
            meta=dict(self.meta),
        )

    def copy(self) -> "FieldSet":
        return self.like([c.copy() for c in self.channels])

    # ---- transforms -------------------------------------------------------- #
    def map(self, fn, grid: Grid | None = None, z: float | None = None) -> "FieldSet":
        """Apply ``fn(channel) -> Channel`` (or ``-> list[Channel]``) to all channels."""
        out: list[Channel] = []
        for ch in self.channels:
            r = fn(ch)
            out.extend(r if isinstance(r, list) else [r])
        return self.like(out, grid=grid, z=z)

    def multiply(self, mask_fn) -> "FieldSet":
        """Multiply every channel by ``mask_fn(wavelength) -> array``."""
        cache: dict[float, "np.ndarray"] = {}

        def _apply(ch: Channel) -> Channel:
            m = cache.get(ch.wavelength)
            if m is None:
                m = cache[ch.wavelength] = mask_fn(ch.wavelength)
            return ch.with_u(ch.u * m)

        return self.map(_apply)

    # ---- observables ------------------------------------------------------- #
    @property
    def wavelengths(self) -> list[float]:
        seen: list[float] = []
        for ch in self.channels:
            if not any(abs(ch.wavelength - w) < 1e-15 for w in seen):
                seen.append(ch.wavelength)
        return sorted(seen)

    def total_power(self) -> float:
        return float(sum(ch.power(self.grid) for ch in self.channels))

    def coherence_groups(self) -> dict[str, list[Channel]]:
        groups: dict[str, list[Channel]] = {}
        for ch in self.channels:
            groups.setdefault(ch.coherence, []).append(ch)
        return groups

    def intensity(self, t=None, beat_bandwidth: float | None = None):
        """Instantaneous irradiance [W/m^2], shape ``(ny, nx)`` or ``(nt, ny, nx)``.

        Channels inside a coherence group add coherently, including any optical
        frequency offset between them when that offset is resolvable by
        ``beat_bandwidth`` (or when ``allow_cross_wavelength_beat`` is set).
        Groups add incoherently.
        """
        scalar_t = t is None or np.isscalar(t)
        tarr = np.array([0.0]) if t is None else np.atleast_1d(np.asarray(t, dtype=float))
        nt = tarr.size
        out = eng.xp.zeros((nt,) + self.grid.shape, dtype=eng.REAL)

        for chans in self.coherence_groups().values():
            nu_ref = chans[0].frequency
            acc = eng.xp.zeros((nt,) + self.grid.shape, dtype=eng.COMPLEX)
            for ch in chans:
                a = ch.waveform.sample(tarr)  # (nt,)
                dnu = ch.frequency - nu_ref
                resolvable = self.allow_cross_wavelength_beat or (
                    beat_bandwidth is not None and abs(dnu) <= beat_bandwidth
                )
                if dnu != 0.0:
                    if resolvable:
                        a = a * eng.asarray(np.exp(-2j * np.pi * dnu * tarr), dtype=eng.COMPLEX)
                    else:
                        # unresolvable beat: contributes only its own intensity
                        out = out + (eng.xp.abs(a).reshape(nt, 1, 1) ** 2) * (
                            eng.xp.abs(ch.u) ** 2
                        )
                        continue
                acc = acc + a.reshape(nt, 1, 1) * ch.u
            out = out + eng.xp.abs(acc) ** 2
        return out[0] if scalar_t else out

    def mean_intensity(self, n_time: int = 64):
        """Time-averaged irradiance over the longest waveform period [W/m^2]."""
        periods = [ch.waveform.period for ch in self.channels] or [1.0]
        t = np.linspace(0.0, max(periods), int(n_time), endpoint=False)
        return eng.xp.mean(self.intensity(t), axis=0)

    def centroid(self) -> tuple[float, float]:
        """Intensity centroid ``(x, y)`` [m] -- handy for alignment checks."""
        I = eng.to_numpy(self.mean_intensity(8))
        X, Y = (eng.to_numpy(a) for a in self.grid.meshgrid())
        s = I.sum()
        if s <= 0:
            return (0.0, 0.0)
        return (float((I * X).sum() / s), float((I * Y).sum() / s))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"FieldSet(z={self.z*1e3:.4g}mm, {len(self.channels)} channels, "
            f"{len(self.coherence_groups())} coherence groups, {self.grid})"
        )
