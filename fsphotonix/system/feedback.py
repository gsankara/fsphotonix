"""Feedback loops: local, global, nested, single- or multi-stage.

An optical machine of this kind is defined as much by its loops as by its optics.
The pieces:

:class:`Actuator`
    writes a control vector onto a physical degree of freedom -- source drive
    levels, per-region modulator piston or gain, or a whole re-routing of the
    holograms.
:class:`Loop`
    one sense -> process -> actuate path, with its own update ``period`` (so a
    slow global loop and a fast local loop can coexist), electronic ``delay``
    (loop latency in iterations) and an optional device rate limit.
:class:`Machine`
    the scheduler.  Each iteration it runs the optics once, then applies whichever
    loops are due.  Loops are applied in ``priority`` order, so a local loop
    nested inside a global one behaves as expected.

This is the structure behind an analog iterative machine (Microsoft's AIM being
the well-known example): a fixed optical matrix-vector multiply, an electronic
nonlinearity plus annealed noise in the return path, and a global loop iterated
thousands of times.  The same machinery expresses a multi-stage digital pipeline
or several independent local servo loops holding an alignment.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from ..detectors.base import DetectorOutput
from ..modulators.array import ModulatorArray
from ..sources.base import EmitterBank
from .logic import LogicBlock, Pipeline, as_logic
from .optical_system import OpticalSystem, Trace
from .stages import DetectorStage, ModulatorStage, SourceStage


# --------------------------------------------------------------------------- #
# Actuators
# --------------------------------------------------------------------------- #
class Actuator:
    """Writes a control vector to hardware."""

    name: str = "actuator"
    size: int = 0

    def write(self, values: np.ndarray) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def read(self) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}({self.name!r}, size={self.size})"


@dataclass
class PistonActuator(Actuator):
    """Per-region phase offset [rad] on a modulator -- fast and analog."""

    modulator: ModulatorArray = None
    scale: float = np.pi
    name: str = "piston"

    @property
    def size(self) -> int:
        return len(self.modulator.regions)

    def write(self, values):
        self.modulator.set_pistons(np.asarray(values, dtype=float).ravel() * self.scale)

    def read(self):
        return self.modulator.get_pistons() / (self.scale or 1.0)


@dataclass
class GainActuator(Actuator):
    """Per-region amplitude transmission in [0, 1]."""

    modulator: ModulatorArray = None
    name: str = "gain"

    @property
    def size(self) -> int:
        return len(self.modulator.regions)

    def write(self, values):
        self.modulator.set_gains(np.clip(np.asarray(values, dtype=float).ravel(), 0.0, 1.0))

    def read(self):
        return self.modulator.get_gains()


@dataclass
class DriveActuator(Actuator):
    """Per-emitter optical power scale on a source bank."""

    source: EmitterBank = None
    max_drive: float = 1.0
    name: str = "drive"

    @property
    def size(self) -> int:
        return self.source.n_emitters

    def write(self, values):
        v = np.clip(np.asarray(values, dtype=float).ravel(), 0.0, self.max_drive)
        self.source.set_drive(v)

    def read(self):
        return self.source.get_drive()


@dataclass
class WeightActuator(Actuator):
    """Per-region sub-region weights -- reprogram the optical matrix itself."""

    modulator: ModulatorArray = None
    name: str = "weights"

    @property
    def size(self) -> int:
        return sum(len(r.subregions) for r in self.modulator.regions)

    def write(self, values):
        v = np.asarray(values, dtype=float).ravel()
        k = 0
        for r in self.modulator.regions:
            n = len(r.subregions)
            self.modulator.set_weights(r, v[k : k + n])
            k += n

    def read(self):
        return np.array([s.weight for r in self.modulator.regions for s in r.subregions])


@dataclass
class FunctionActuator(Actuator):
    """Arbitrary hook: ``func(values)``."""

    func: Callable = None
    size: int = 0
    name: str = "function"

    def write(self, values):
        self.func(np.asarray(values, dtype=float))

    def read(self):
        return np.zeros(self.size)


def make_actuator(system: OpticalSystem, target: str, mode: str = "piston", **kw) -> Actuator:
    """Build an actuator for a named stage: ``piston``, ``gain``, ``drive``, ``weights``."""
    st = system[target]
    if mode == "piston":
        return PistonActuator(modulator=st.modulator, name=f"{target}.piston", **kw)
    if mode == "gain":
        return GainActuator(modulator=st.modulator, name=f"{target}.gain", **kw)
    if mode == "weights":
        return WeightActuator(modulator=st.modulator, name=f"{target}.weights", **kw)
    if mode == "drive":
        return DriveActuator(source=st.source, name=f"{target}.drive", **kw)
    raise ValueError(f"unknown actuation mode {mode!r}")


# --------------------------------------------------------------------------- #
# Loops
# --------------------------------------------------------------------------- #
@dataclass
class Loop:
    """One feedback path.

    Parameters
    ----------
    sense : name of the detector stage to read (or a callable ``(trace) -> vector``).
    actuator : where the result is written.
    controller : a :class:`~fsphotonix.system.logic.LogicBlock` / pipeline / callable
        mapping the measurement to an actuation vector.
    period : apply every ``period`` iterations (a slow outer loop uses a large value).
    delay : electronic latency, in iterations, between sensing and actuating.
    priority : lower runs first within one iteration.
    scope : ``'local'`` or ``'global'`` -- documentation only, but reported in the log.
    rate_limit : maximum change per update, in actuator units.
    """

    sense: object = None
    actuator: Actuator = None
    controller: object = None
    name: str = "loop"
    period: int = 1
    phase: int = 0
    delay: int = 0
    priority: int = 0
    scope: str = "global"
    rate_limit: float | None = None
    enabled: bool = True
    target_index: object = None  # optional element selection from the measurement

    def __post_init__(self):
        self.controller = as_logic(self.controller) if self.controller is not None else None
        self._queue: deque = deque()
        self._last: np.ndarray | None = None
        self.history: list[dict] = []

    # ---- plumbing ---------------------------------------------------------- #
    def measure(self, trace: Trace) -> np.ndarray:
        if callable(self.sense):
            v = np.asarray(self.sense(trace), dtype=float).ravel()
        else:
            v = trace.outputs[self.sense].vector()
        if self.target_index is not None:
            v = v[np.asarray(self.target_index, dtype=int)]
        return v

    def due(self, k: int) -> bool:
        return self.enabled and ((k - self.phase) % max(self.period, 1) == 0) and k >= self.phase

    def step(self, k: int, trace: Trace) -> dict | None:
        if not self.due(k):
            return None
        y = self.measure(trace)
        self._queue.append(y)
        if len(self._queue) <= self.delay:
            return {"iteration": k, "loop": self.name, "status": "filling_delay"}
        y_used = self._queue.popleft()
        u = np.asarray(self.controller(y_used) if self.controller else y_used, dtype=float).ravel()
        if self.actuator is None:
            return {"iteration": k, "loop": self.name, "measurement": y_used, "actuation": u}
        size = self.actuator.size
        if u.size != size:
            u = _fit(u, size)
        if self.rate_limit is not None and self._last is not None:
            u = self._last + np.clip(u - self._last, -self.rate_limit, self.rate_limit)
        self.actuator.write(u)
        self._last = u
        rec = {"iteration": k, "loop": self.name, "scope": self.scope,
               "measurement": y_used, "actuation": u}
        return rec

    def reset(self) -> None:
        self._queue.clear()
        self._last = None
        self.history.clear()
        if isinstance(self.controller, LogicBlock):
            self.controller.reset()


def _fit(u: np.ndarray, size: int) -> np.ndarray:
    """Resize a control vector to the actuator size (tile or truncate)."""
    if u.size == size:
        return u
    if u.size == 1:
        return np.full(size, float(u[0]))
    if u.size > size:
        return u[:size]
    return np.resize(u, size)


# --------------------------------------------------------------------------- #
# Machine
# --------------------------------------------------------------------------- #
@dataclass
class History:
    """Per-iteration record of a run."""

    records: list[dict] = field(default_factory=list)
    metrics: list[dict] = field(default_factory=list)

    def add(self, rec: dict) -> None:
        self.records.append(rec)

    def series(self, key: str) -> np.ndarray:
        return np.array([r[key] for r in self.metrics if key in r])

    def last(self) -> dict:
        return self.records[-1] if self.records else {}

    def __len__(self) -> int:
        return len(self.records)


class Machine:
    """An optical system plus its feedback loops, iterated.

    ``metric`` is an optional ``(trace, iteration) -> dict`` hook recorded every
    iteration; use it to log an energy, a bit-error count, or a convergence norm.
    """

    def __init__(self, system: OpticalSystem, loops: Sequence[Loop] = (),
                 metric: Callable | None = None, name: str = "machine",
                 keep_fields: bool = False):
        self.system = system
        self.loops = sorted(loops, key=lambda l: l.priority)
        self.metric = metric
        self.name = name
        self.keep_fields = keep_fields
        self.history = History()
        self.iteration = 0

    # ---- control ----------------------------------------------------------- #
    def reset(self) -> None:
        self.iteration = 0
        self.history = History()
        for l in self.loops:
            l.reset()

    def add_loop(self, loop: Loop) -> Loop:
        self.loops.append(loop)
        self.loops.sort(key=lambda l: l.priority)
        return loop

    def step(self, t=None) -> dict:
        """One iteration: run the optics, then apply every loop that is due."""
        k = self.iteration
        trace = self.system.run(t=t, keep_fields=self.keep_fields)
        rec: dict = {"iteration": k, "loops": []}
        for loop in self.loops:
            out = loop.step(k, trace)
            if out is not None:
                loop.history.append(out)
                rec["loops"].append(out)
        if self.metric is not None:
            m = dict(self.metric(trace, k))
            m["iteration"] = k
            self.history.metrics.append(m)
            rec["metric"] = m
        if self.keep_fields:
            rec["trace"] = trace
        else:
            rec["readout"] = {n: o.vector() for n, o in trace.outputs.items()}
        self.history.add(rec)
        self.iteration += 1
        return rec

    def run(self, iterations: int = 100, t=None, callback: Callable | None = None,
            stop: Callable | None = None) -> History:
        for _ in range(int(iterations)):
            rec = self.step(t=t)
            if callback is not None:
                callback(rec)
            if stop is not None and stop(rec):
                break
        return self.history

    # ---- reporting --------------------------------------------------------- #
    def loop_report(self) -> list[dict]:
        return [{
            "name": l.name, "scope": l.scope, "period": l.period, "delay": l.delay,
            "priority": l.priority,
            "sense": l.sense if isinstance(l.sense, str) else "callable",
            "actuator": l.actuator.name if l.actuator else None,
            "controller_latency": getattr(l.controller, "latency", 0),
            "updates": len(l.history),
        } for l in self.loops]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"Machine({self.name!r}, {len(self.system)} planes, "
                f"{len(self.loops)} loops, iteration={self.iteration})")


# --------------------------------------------------------------------------- #
# convenience builders
# --------------------------------------------------------------------------- #
def local_loop(system: OpticalSystem, sense: str, target: str, mode: str = "piston",
               controller=None, **kw) -> Loop:
    """A loop that closes around one plane -- e.g. holding an alignment."""
    return Loop(sense=sense, actuator=make_actuator(system, target, mode),
                controller=controller, scope="local",
                name=kw.pop("name", f"{sense}->{target}.{mode}"), **kw)


def global_loop(system: OpticalSystem, sense: str, target: str, mode: str = "drive",
                controller=None, **kw) -> Loop:
    """A loop that closes around the whole machine -- the iterative solver path."""
    return Loop(sense=sense, actuator=make_actuator(system, target, mode),
                controller=controller, scope="global",
                name=kw.pop("name", f"{sense}=>{target}.{mode}"), **kw)
