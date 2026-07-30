"""Declarative system description (YAML or JSON).

Every plugin is reachable by name, so a whole machine -- sources, holograms,
devices, screens, loops -- is a data file.  That is what makes parameter sweeps and
reproducible experiments practical.

Example::

    wavelength: 1550e-9
    grids:
      panel:  {width: 6.0e-3, pitch: 2.0e-6}
      screen: {width: 20.0e-3, pitch: 20.0e-6}
    stages:
      - {name: src,   type: source,    grid: panel,
         source: {type: cots_transceiver, n: 8, baud: 25.0e9}}
      - {name: panel, type: modulator, grid: panel, z: 0.0,
         device: {type: ipms_piston},
         layout: {nx: 8, ny: 1, pitch: 600.0e-6},
         route: {targets: [[-2.0e-3, 0], [0, 0], [2.0e-3, 0]], z: 0.08}}
      - {name: cam,   type: detector,  z: 0.08, grid: screen,
         detector: {type: camera, nx: 256, ny: 256, pitch: 20.0e-6}}
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from .detectors.base import as_detector
from .grid import Grid
from .modulators.array import ModulatorArray
from .modulators.devices import as_device
from .propagation.base import as_propagator
from .registry import build, create
from .system.feedback import Loop, Machine, make_actuator
from .system.logic import as_logic
from .system.optical_system import OpticalSystem
from .system.stages import DetectorStage, ElementStage, ModulatorStage, SourceStage


_NUMBER = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


def _coerce_numbers(obj):
    """Turn numeric-looking strings into numbers, recursively.

    YAML 1.1 (which PyYAML implements) only recognises an exponent as a float when
    it carries an explicit sign, so ``100.0e9`` parses as a *string* while
    ``100.0e+9`` parses as a float.  Optical configs are full of such values, so
    rather than making every user learn that rule the loader coerces anything that
    looks like a number.
    """
    if isinstance(obj, dict):
        return {k: _coerce_numbers(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_coerce_numbers(v) for v in obj]
    if isinstance(obj, str) and _NUMBER.match(obj.strip()):
        text = obj.strip()
        try:
            return int(text) if _NUMBER.match(text).group(2) is None and "." not in text \
                else float(text)
        except ValueError:  # pragma: no cover - guarded by the regex
            return obj
    return obj


def load_text(text: str) -> dict:
    """Parse YAML if available, else JSON, coercing numeric strings."""
    try:
        import yaml  # noqa: PLC0415

        data = yaml.safe_load(text)
    except ImportError:  # pragma: no cover - yaml is optional
        data = json.loads(text)
    return _coerce_numbers(data)


def load_file(path: str | Path) -> dict:
    return load_text(Path(path).read_text(encoding="utf-8"))


def make_grid(spec: dict | Grid) -> Grid:
    if isinstance(spec, Grid):
        return spec
    s = dict(spec)
    if "width" in s:
        return Grid.from_extent(float(s.pop("width")),
                                float(s.pop("height")) if "height" in s else None,
                                float(s.pop("pitch", 1e-6)), **_floats(s))
    return Grid(**_floats(s))


def _floats(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        out[k] = float(v) if isinstance(v, (int, float)) and k not in ("nx", "ny") else v
    for k in ("nx", "ny"):
        if k in out:
            out[k] = int(out[k])
    return out


def build_system(cfg: dict) -> OpticalSystem:
    """Instantiate an :class:`OpticalSystem` from a config mapping."""
    lam = float(cfg.get("wavelength", 1550e-9))
    grids = {k: make_grid(v) for k, v in (cfg.get("grids") or {}).items()}

    def g(ref):
        if ref is None:
            return None
        return grids[ref] if isinstance(ref, str) else make_grid(ref)

    stages = []
    for spec in cfg["stages"]:
        s = dict(spec)
        kind = s.pop("type")
        name = s.pop("name", kind)
        z = float(s.pop("z", 0.0))
        grid = g(s.pop("grid", None))
        prop = s.pop("propagator", None)
        in_prop = as_propagator(prop) if prop is not None else None

        if kind == "source":
            src = build(s.pop("source"), "source")
            stages.append(SourceStage(source=src, grid=grid, z=z, name=name,
                                      in_propagator=in_prop))
        elif kind == "modulator":
            dev = as_device(s.pop("device", None))
            layout = s.pop("layout", {"nx": 1, "ny": 1, "pitch": 1e-3})
            mod = ModulatorArray.tiled(device=dev, default_wavelength=lam,
                                       background=s.pop("background", "absorb"),
                                       weight_mode=s.pop("weight_mode", "area"),
                                       name=name, **_floats(dict(layout)))
            route = s.pop("route", None)
            if route:
                r = dict(route)
                targets = [tuple(float(v) for v in t) for t in r.pop("targets")]
                rz = float(r.pop("z", z))
                mod.broadcast(targets, rz, wavelength=lam, **r)
            matrix = s.pop("matrix", None)
            if matrix:
                m = dict(matrix)
                mod.set_matrix(np.asarray(m.pop("weights"), dtype=float),
                               np.asarray(m.pop("outputs"), dtype=float),
                               float(m.pop("z", z)), **m)
            stages.append(ModulatorStage(modulator=mod, grid=grid, z=z, name=name,
                                         in_propagator=in_prop,
                                         spot_mode=bool(s.pop("spot_mode", False)),
                                         spot_options=s.pop("spot_options", {})))
        elif kind == "detector":
            det = as_detector(s.pop("detector"))
            stages.append(DetectorStage(detector=det, grid=grid, z=z, name=name,
                                        in_propagator=in_prop,
                                        pass_through=bool(s.pop("pass_through", False)),
                                        tap_fraction=float(s.pop("tap_fraction", 1.0))))
        elif kind == "element":
            stages.append(ElementStage(element=build(s.pop("element"), "propagator"),
                                       grid=grid, z=z, name=name, in_propagator=in_prop))
        else:
            raise ValueError(f"unknown stage type {kind!r}")
    return OpticalSystem(stages, propagator=cfg.get("propagator"),
                         name=cfg.get("name", "system"))


def build_machine(cfg: dict, system: OpticalSystem | None = None) -> Machine:
    """Instantiate a :class:`Machine` (system + feedback loops) from a config."""
    sysm = system or build_system(cfg)
    loops = []
    for spec in cfg.get("loops", []) or []:
        s = dict(spec)
        target = s.pop("actuate")
        mode = s.pop("mode", "piston")
        ctrl = s.pop("controller", None)
        loops.append(Loop(
            sense=s.pop("sense"),
            actuator=make_actuator(sysm, target, mode),
            controller=as_logic(ctrl) if ctrl is not None else None,
            name=s.pop("name", f"{target}.{mode}"),
            period=int(s.pop("period", 1)),
            delay=int(s.pop("delay", 0)),
            priority=int(s.pop("priority", 0)),
            scope=s.pop("scope", "global"),
            rate_limit=s.pop("rate_limit", None),
        ))
    return Machine(sysm, loops=loops, name=cfg.get("name", "machine"))


def dump_summary(cfg_or_system) -> str:
    """Human-readable inventory of a config or a built system."""
    sysm = cfg_or_system if isinstance(cfg_or_system, OpticalSystem) else build_system(
        cfg_or_system)
    lines = [f"system: {sysm.name}"]
    for s in sysm.stages:
        lines.append(f"  {s.z*1e3:9.3f} mm  {type(s).__name__:15s} {s.name}")
        mod = getattr(s, "modulator", None)
        if mod is not None:
            lines.append(f"{'':14s}{mod!r}")
        det = getattr(s, "detector", None)
        if det is not None:
            lines.append(f"{'':14s}{det!r}")
    return "\n".join(lines)
