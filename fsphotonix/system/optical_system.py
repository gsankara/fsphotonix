"""Cascade engine: walk the planes, propagating between them.

Any number of modulator planes can be chained (multi-stage diffractive logic), any
plane can be tapped by a detector, and any plane's sampling grid is independent of
the others -- the propagators resample as needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from ..detectors.base import DetectorOutput
from ..fields import FieldSet
from ..grid import Grid
from ..propagation.base import Propagator, as_propagator
from ..propagation.kernels import Auto
from .stages import DetectorStage, ModulatorStage, SourceStage, Stage


@dataclass
class Trace:
    """Everything the simulator saw on one pass."""

    fields: dict[str, FieldSet] = field(default_factory=dict)
    outputs: dict[str, DetectorOutput] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    spot_lists: dict[str, object] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def __getitem__(self, name: str):
        if name in self.outputs:
            return self.outputs[name]
        return self.fields[name]

    @property
    def final(self) -> FieldSet:
        return self.fields[self.order[-1]]

    def vector(self, name: str | None = None) -> np.ndarray:
        """Flat readout of one detector (the last one by default)."""
        if name is None:
            if not self.outputs:
                raise KeyError("this system has no detector stages")
            name = list(self.outputs)[-1]
        return self.outputs[name].vector()

    def power_budget(self) -> dict[str, float]:
        return {k: v.total_power() for k, v in self.fields.items()}

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Trace(planes={self.order}, detectors={list(self.outputs)})"


class OpticalSystem:
    """An ordered chain of planes.

    Parameters
    ----------
    stages : the planes.  Sorted by ``z`` unless ``sort=False``.
    propagator : default propagation kernel between planes (``'auto'`` chooses per
        hop from the sampling regime).
    """

    def __init__(self, stages: Sequence[Stage], propagator=None, name: str = "system",
                 sort: bool = True):
        self.stages: list[Stage] = sorted(stages, key=lambda s: s.z) if sort else list(stages)
        self.default_propagator = propagator
        self.name = name
        self._check()

    def _check(self) -> None:
        if not self.stages:
            raise ValueError("a system needs at least one stage")
        if not isinstance(self.stages[0], SourceStage):
            raise ValueError("the first stage must be a SourceStage")
        seen: set[str] = set()
        for s in self.stages:
            if s.name in seen:
                raise ValueError(f"duplicate stage name {s.name!r}")
            seen.add(s.name)

    # ---- access ------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.stages)

    def __getitem__(self, key) -> Stage:
        if isinstance(key, str):
            for s in self.stages:
                if s.name == key:
                    return s
            raise KeyError(key)
        return self.stages[key]

    def stage_names(self) -> list[str]:
        return [s.name for s in self.stages]

    def of_type(self, cls) -> list[Stage]:
        return [s for s in self.stages if isinstance(s, cls)]

    @property
    def modulators(self) -> list[ModulatorStage]:
        return self.of_type(ModulatorStage)  # type: ignore[return-value]

    @property
    def detectors(self) -> list[DetectorStage]:
        return self.of_type(DetectorStage)  # type: ignore[return-value]

    def add(self, stage: Stage) -> Stage:
        self.stages.append(stage)
        self.stages.sort(key=lambda s: s.z)
        self._check()
        return stage

    # ---- the pass ---------------------------------------------------------- #
    def run(self, t=None, upto: str | int | None = None, keep_fields: bool = True) -> Trace:
        """Propagate through the system once and collect everything."""
        tr = Trace()
        stop = self._stop_index(upto)
        fields: FieldSet | None = None
        prev: Stage | None = None

        for i, st in enumerate(self.stages[: stop + 1]):
            if fields is None:
                fields = st.emit(t)  # guaranteed SourceStage by _check()
            else:
                fields = self._hop(prev, st, fields, tr)
                if not getattr(st, "defers_to_propagator", False):
                    fields = st.process(fields, t)
            out = st.measure(fields, t)
            if out is not None:
                tr.outputs[st.name] = out
            if isinstance(st, DetectorStage) and not st.pass_through:
                fields_after = st.process(fields, t)
            else:
                fields_after = fields
            if keep_fields:
                tr.fields[st.name] = fields
            tr.order.append(st.name)
            fields = fields_after
            prev = st
        tr.meta["t"] = t
        return tr

    def _stop_index(self, upto) -> int:
        if upto is None:
            return len(self.stages) - 1
        if isinstance(upto, int):
            return upto
        return self.stage_names().index(upto)

    def _hop(self, prev: Stage, nxt: Stage, fields: FieldSet, tr: Trace) -> FieldSet:
        """Propagate from ``prev`` to ``nxt``."""
        dz = nxt.z - prev.z
        out_grid = nxt.target_grid(fields.grid)

        if isinstance(prev, ModulatorStage) and prev.spot_mode:
            sp = prev.spot_propagator(dz, out_grid)
            f = sp.propagate(prev.modulator, fields, out_grid=out_grid, z=dz)
            tr.spot_lists[prev.name] = f.meta.get("spot_list")
            return f

        prop = nxt.in_propagator or self.default_propagator
        p = as_propagator(prop, z=dz, out_grid=out_grid)
        if dz == 0:
            return fields if out_grid is fields.grid else p.propagate(fields, z=0.0,
                                                                     out_grid=out_grid)
        return p.propagate(fields, z=dz, out_grid=out_grid)

    # ---- convenience ------------------------------------------------------- #
    def readout(self, t=None, detector: str | None = None) -> np.ndarray:
        return self.run(t, keep_fields=False).vector(detector)

    def sampling_report(self) -> list[dict]:
        """Per-hop sampling diagnostics -- run this before trusting a big model."""
        from ..propagation.kernels import suggest

        out = []
        for a, b in zip(self.stages[:-1], self.stages[1:]):
            g = a.grid or b.grid
            if g is None:
                continue
            lam = getattr(getattr(a, "modulator", None), "default_wavelength", 1550e-9)
            rep = suggest(g, lam, b.z - a.z, b.grid)
            rep.update({"from": a.name, "to": b.name, "dz_mm": (b.z - a.z) * 1e3})
            out.append(rep)
        return out

    def summary(self) -> dict:
        return {
            "name": self.name,
            "stages": [{"name": s.name, "type": type(s).__name__, "z_mm": s.z * 1e3,
                        "grid": repr(s.grid) if s.grid else None} for s in self.stages],
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        chain = " -> ".join(f"{s.name}@{s.z*1e3:.3g}mm" for s in self.stages)
        return f"OpticalSystem({self.name!r}: {chain})"
