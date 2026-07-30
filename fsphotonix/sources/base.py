"""Light source abstraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ..fields import Channel, FieldSet
from ..grid import Grid
from ..waveform import Waveform, as_waveform
from .profiles import BeamProfile, as_profile


class LightSource:
    """Base class: emit a :class:`~fsphotonix.fields.FieldSet` onto a grid.

    Subclasses implement :meth:`channels`.  A source is also the natural handle
    for feedback: :meth:`set_drive` lets a controller write new amplitudes or
    waveforms without rebuilding the source.
    """

    name: str = "source"

    def channels(self, grid: Grid) -> list[Channel]:  # pragma: no cover - abstract
        raise NotImplementedError

    def emit(self, grid: Grid, z: float = 0.0) -> FieldSet:
        return FieldSet(grid=grid, channels=self.channels(grid), z=z)

    # ---- feedback hooks ---------------------------------------------------- #
    def set_drive(self, values) -> None:
        """Set per-emitter drive levels (linear optical amplitude scale, >= 0)."""
        raise NotImplementedError(f"{type(self).__name__} has no electrical drive input")

    @property
    def n_emitters(self) -> int:
        return 1

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}({self.name!r}, {self.n_emitters} emitters)"


@dataclass
class Emitter:
    """One physical emitter: a wavelength, a power, a profile and a waveform."""

    wavelength: float = 1550e-9
    power: float = 1e-3
    profile: BeamProfile | dict | str | None = None
    waveform: Waveform | dict | str | None = None
    coherence: str | None = None
    label: str = ""
    drive: float = 1.0  # electrical drive scale on optical POWER
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.profile = as_profile(self.profile)
        self.waveform = as_waveform(self.waveform)

    def channel(self, grid: Grid) -> Channel:
        p = max(self.power * self.drive, 0.0)
        return Channel(
            u=self.profile.field(grid, self.wavelength, power=p),
            wavelength=self.wavelength,
            coherence=self.coherence or f"emitter:{id(self)}",
            waveform=self.waveform,
            label=self.label,
            meta=dict(self.meta),
        )


class EmitterBank(LightSource):
    """A source built from an explicit list of :class:`Emitter` objects.

    All the concrete sources in this package are thin constructors over this
    class, so a custom technology only needs to produce an emitter list.
    """

    def __init__(self, emitters: Sequence[Emitter], name: str = "bank"):
        self.emitters = list(emitters)
        self.name = name

    @property
    def n_emitters(self) -> int:
        return len(self.emitters)

    def channels(self, grid: Grid) -> list[Channel]:
        return [e.channel(grid) for e in self.emitters if e.drive > 0 and e.power > 0]

    def set_drive(self, values) -> None:
        import numpy as np

        v = np.atleast_1d(np.asarray(values, dtype=float)).ravel()
        if v.size == 1:
            v = np.full(len(self.emitters), v[0])
        if v.size != len(self.emitters):
            raise ValueError(f"expected {len(self.emitters)} drive values, got {v.size}")
        for e, val in zip(self.emitters, v):
            e.drive = float(max(val, 0.0))

    def get_drive(self):
        import numpy as np

        return np.array([e.drive for e in self.emitters], dtype=float)

    def set_waveforms(self, waveforms) -> None:
        """Retarget the temporal drive of every emitter (digital feedback path)."""
        if isinstance(waveforms, (Waveform, dict, str)):
            waveforms = [waveforms] * len(self.emitters)
        for e, w in zip(self.emitters, waveforms):
            e.waveform = as_waveform(w)

    def __getitem__(self, i) -> Emitter:
        return self.emitters[i]
