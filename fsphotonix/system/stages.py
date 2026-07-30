"""Planes of the optical system.

A system is an ordered list of stages at increasing positions ``z`` along the
optical axis.  Reflective folds are handled by *unfolding* the path: ``z`` is
optical path length, not a physical coordinate, which keeps the maths flat while
still letting reflective SLMs and mirrors appear anywhere in the chain.

The key composability property the brief asks for -- "the detection screen can be
replaced by another light modulator array" -- is simply a matter of putting a
:class:`ModulatorStage` where a :class:`DetectorStage` would otherwise be, with no
other change.  Both are stages; a plane may even carry both (a beam-splitter tap
that measures *and* passes the light on).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..detectors.base import Detector, DetectorOutput, as_detector
from ..fields import FieldSet
from ..grid import Grid
from ..modulators.array import ModulatorArray
from ..propagation.base import Propagator, as_propagator
from ..propagation.spot import SpotPropagator
from ..sources.base import LightSource


class Stage:
    """Base class for anything that sits at a plane."""

    name: str = "stage"
    z: float = 0.0
    grid: Grid | None = None
    #: propagator used to *reach* this stage from the previous one
    in_propagator: Propagator | None = None

    def process(self, fields: FieldSet, t=None) -> FieldSet:
        """Transform the incident field.  Default: pass through untouched."""
        return fields

    def measure(self, fields: FieldSet, t=None) -> DetectorOutput | None:
        return None

    def target_grid(self, incoming: Grid) -> Grid:
        return self.grid if self.grid is not None else incoming

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}({self.name!r}, z={self.z*1e3:.4g}mm)"


@dataclass
class SourceStage(Stage):
    """Where the light comes from."""

    source: LightSource = None
    grid: Grid | None = None
    z: float = 0.0
    name: str = "source"
    in_propagator: Propagator | None = None

    def emit(self, t=None) -> FieldSet:
        if self.grid is None:
            raise ValueError("SourceStage needs a grid")
        return self.source.emit(self.grid, z=self.z)

    def process(self, fields: FieldSet, t=None) -> FieldSet:
        """Add this source's light to whatever is already propagating.

        Lets a system inject light at more than one plane -- a pump beam, a bias
        beam, or the second arm of an interferometric readout.
        """
        return FieldSet.merge([fields, self.emit(t)]) if fields is not None else self.emit(t)


@dataclass
class ModulatorStage(Stage):
    """A programmable diffractive plane.

    ``spot_mode`` switches the *outgoing* propagation to the beamlet propagator,
    which is how very large panels are simulated: see
    :class:`fsphotonix.propagation.spot.SpotPropagator`.
    """

    modulator: ModulatorArray = None
    z: float = 0.0
    grid: Grid | None = None
    name: str = "modulator"
    in_propagator: Propagator | None = None
    spot_mode: bool = False
    spot_options: dict = field(default_factory=dict)

    def process(self, fields: FieldSet, t=None) -> FieldSet:
        return self.modulator.apply(fields)

    @property
    def defers_to_propagator(self) -> bool:
        """In spot mode the modulator and the following hop are one operator.

        The beamlet propagator reads the *incident* field and the hologram's
        analytic orders together, so the plane must not also multiply the field by
        its transmittance -- that would apply the grating twice.  The field
        recorded at this plane is therefore the incident one.
        """
        return bool(self.spot_mode)

    @property
    def reflective(self) -> bool:
        return bool(self.modulator.device.reflective)

    def spot_propagator(self, z: float, out_grid: Grid) -> SpotPropagator:
        return SpotPropagator(z=z, out_grid=out_grid, **self.spot_options)


@dataclass
class DetectorStage(Stage):
    """A detection screen.

    ``pass_through`` keeps the light propagating past the plane (a partially
    reflecting tap, or a fibre array that only intercepts part of the field), so a
    single plane can both measure and feed the next stage.
    """

    detector: Detector = None
    z: float = 0.0
    grid: Grid | None = None
    name: str = "detector"
    in_propagator: Propagator | None = None
    pass_through: bool = False
    tap_fraction: float = 1.0  # fraction of power sent to the detector
    detect_options: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.detector is not None:
            self.detector = as_detector(self.detector)
        if self.grid is None and hasattr(self.detector, "grid"):
            try:
                self.grid = self.detector.grid()
            except Exception:  # pragma: no cover - detectors without a natural grid
                self.grid = None

    def measure(self, fields: FieldSet, t=None) -> DetectorOutput:
        f = fields
        if self.tap_fraction != 1.0:
            f = fields.map(lambda ch: ch.scaled(np.sqrt(max(self.tap_fraction, 0.0))))
        return self.detector.detect(f, t=t, **self.detect_options)

    def process(self, fields: FieldSet, t=None) -> FieldSet:
        if not self.pass_through:
            return fields
        keep = np.sqrt(max(1.0 - self.tap_fraction, 0.0))
        return fields.map(lambda ch: ch.scaled(keep))


@dataclass
class ElementStage(Stage):
    """A passive bulk element at a plane: lens, stop, aberration screen, relay."""

    element: Propagator = None
    z: float = 0.0
    grid: Grid | None = None
    name: str = "element"
    in_propagator: Propagator | None = None

    def __post_init__(self):
        self.element = as_propagator(self.element)

    def process(self, fields: FieldSet, t=None) -> FieldSet:
        return self.element.propagate(fields, z=0.0)

    def target_grid(self, incoming: Grid) -> Grid:
        if self.grid is not None:
            return self.grid
        tg = getattr(self.element, "target_grid", None)
        return tg(incoming) if callable(tg) else incoming


@dataclass
class FunctionStage(Stage):
    """Arbitrary user hook at a plane: ``func(fields, t) -> fields``.

    The general-purpose plug-in point for optics this package does not model --
    a nonlinear saturable absorber, a polarisation element, a custom OO device.
    """

    func: object = None
    z: float = 0.0
    grid: Grid | None = None
    name: str = "function"
    in_propagator: Propagator | None = None

    def process(self, fields: FieldSet, t=None) -> FieldSet:
        return self.func(fields, t)
