"""Plugin registry.

Every replaceable technology in the simulator is registered here under a
``kind`` (the abstraction slot) and a ``name`` (the specific technology).  A
researcher plugs in custom hardware by subclassing the relevant base class and
decorating it:

    from fsphotonix.registry import register
    from fsphotonix.modulators.gratings.base import Grating

    @register("grating", "my_metasurface")
    class MyMetasurface(Grating):
        ...

After that the new device is reachable from YAML/JSON configs and from
``fsphotonix.registry.create("grating", "my_metasurface", ...)`` without editing
any simulator code.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Iterable

#: Abstraction slots that ship with the simulator.  Custom kinds are allowed.
KINDS = (
    "source",  # light sources / transceivers
    "waveform",  # temporal modulation (digital / analog / mixed)
    "profile",  # transverse beam profile
    "geometry",  # aperture / sub-region shape
    "grating",  # holographic grating / CGH
    "device",  # SLM or micro-mirror hardware model
    "propagator",  # free-space propagation kernel
    "detector",  # detection screen technology (OE or OO)
    "controller",  # feedback controller / digital logic stage
)

_REGISTRY: dict[str, dict[str, type]] = defaultdict(dict)
_ALIASES: dict[str, dict[str, str]] = defaultdict(dict)


class PluginError(KeyError):
    """Raised when a plugin name cannot be resolved."""


def register(kind: str, name: str, *aliases: str) -> Callable[[type], type]:
    """Class decorator registering ``cls`` as ``kind``/``name``."""

    def _wrap(cls: type) -> type:
        key = name.lower()
        if key in _REGISTRY[kind] and _REGISTRY[kind][key] is not cls:
            raise PluginError(f"{kind}/{name} already registered by {_REGISTRY[kind][key]!r}")
        _REGISTRY[kind][key] = cls
        for alias in aliases:
            _ALIASES[kind][alias.lower()] = key
        cls.plugin_kind = kind  # type: ignore[attr-defined]
        cls.plugin_name = name  # type: ignore[attr-defined]
        return cls

    return _wrap


def get(kind: str, name: str) -> type:
    """Look up a registered class."""
    table = _REGISTRY.get(kind)
    if not table:
        raise PluginError(f"no plugins registered for kind {kind!r} (known: {sorted(_REGISTRY)})")
    key = name.lower()
    key = _ALIASES[kind].get(key, key)
    try:
        return table[key]
    except KeyError:
        raise PluginError(
            f"unknown {kind} {name!r}; available: {sorted(table)}"
        ) from None


def create(kind: str, name: str, /, **kwargs: Any) -> Any:
    """Instantiate a registered plugin."""
    return get(kind, name)(**kwargs)


def available(kind: str | None = None) -> dict[str, list[str]] | list[str]:
    """List registered plugin names, optionally for a single kind."""
    if kind is not None:
        return sorted(_REGISTRY.get(kind, {}))
    return {k: sorted(v) for k, v in sorted(_REGISTRY.items())}


def build(spec: Any, kind: str, **defaults: Any) -> Any:
    """Build an object from a declarative spec.

    ``spec`` may already be an instance (returned unchanged), a string naming a
    plugin, or a mapping ``{"type": name, ...kwargs}``.  Used by the config
    loader and by constructors that accept "either an object or a shorthand".
    """
    if spec is None:
        return None
    if isinstance(spec, str):
        return create(kind, spec, **defaults)
    if isinstance(spec, dict):
        params = dict(defaults)
        params.update({k: v for k, v in spec.items() if k != "type"})
        name = spec.get("type")
        if name is None:
            raise PluginError(f"spec for kind {kind!r} needs a 'type' key: {spec!r}")
        return create(kind, name, **params)
    return spec  # already an instance / duck-typed object


def describe(kinds: Iterable[str] | None = None) -> str:
    """Human-readable plugin inventory."""
    lines = []
    for kind in kinds or sorted(_REGISTRY):
        names = sorted(_REGISTRY[kind])
        lines.append(f"{kind:11s} : {', '.join(names) if names else '-'}")
    return "\n".join(lines)
