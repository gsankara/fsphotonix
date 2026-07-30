"""Propagator abstraction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..fields import Channel, FieldSet
from ..grid import Grid


class Propagator:
    """Free-space (or optical-element) transfer between two planes.

    ``out_grid`` lets the receiving plane be sampled completely differently from
    the sending plane -- a 5 mm modulator at 2 um sampling can illuminate a 50 mm
    screen at 20 um sampling with no zero padding, which is what keeps large
    systems affordable.
    """

    z: float = 0.0
    out_grid: Grid | None = None

    def kernel(self, channel: Channel, in_grid: Grid, out_grid: Grid):  # pragma: no cover
        raise NotImplementedError

    def target_grid(self, in_grid: Grid, wavelength: float | None = None) -> Grid:
        return self.out_grid if self.out_grid is not None else in_grid

    def propagate(self, fields: FieldSet, z: float | None = None,
                  out_grid: Grid | None = None) -> FieldSet:
        zz = self.z if z is None else z
        og = out_grid if out_grid is not None else self.target_grid(fields.grid)
        chans = [self.propagate_channel(ch, fields.grid, og, zz) for ch in fields.channels]
        return fields.like(chans, grid=og, z=fields.z + zz)

    def propagate_channel(self, ch: Channel, in_grid: Grid, out_grid: Grid, z: float) -> Channel:
        return ch.with_u(self.apply(ch.u, in_grid, out_grid, ch.wavelength, z))

    def apply(self, u, in_grid: Grid, out_grid: Grid, wavelength: float, z: float):
        raise NotImplementedError

    def __call__(self, fields: FieldSet, **kw) -> FieldSet:
        return self.propagate(fields, **kw)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(z={self.z*1e3:.4g}mm)"


def as_propagator(spec, z: float | None = None, out_grid: Grid | None = None) -> Propagator:
    """Coerce a spec into a propagator, defaulting to the auto-selected kernel."""
    from ..registry import build
    from .kernels import Auto

    if spec is None:
        p = Auto()
    elif isinstance(spec, Propagator):
        p = spec
    else:
        p = build(spec, "propagator")
    if z is not None:
        p.z = z
    if out_grid is not None:
        p.out_grid = out_grid
    return p
