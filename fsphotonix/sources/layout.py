"""Emitter position / pointing layouts.

Separated from the source models so that any technology (laser, VCSEL, fibre
array, transceiver bank) can be arranged in any geometry.
"""

from __future__ import annotations

import numpy as np


def grid_positions(nx: int, ny: int = 1, pitch_x: float = 1e-3, pitch_y: float | None = None,
                   x0: float = 0.0, y0: float = 0.0) -> np.ndarray:
    """``(N, 2)`` array of centred rectangular-lattice positions [m]."""
    py = pitch_x if pitch_y is None else pitch_y
    xs = (np.arange(nx) - (nx - 1) / 2) * pitch_x + x0
    ys = (np.arange(ny) - (ny - 1) / 2) * py + y0
    X, Y = np.meshgrid(xs, ys)
    return np.stack([X.ravel(), Y.ravel()], axis=1)


def line_positions(n: int, pitch: float = 1e-3, angle: float = 0.0,
                   x0: float = 0.0, y0: float = 0.0) -> np.ndarray:
    """Linear array (e.g. a 1xN fibre array or VCSEL bar) rotated by ``angle``."""
    s = (np.arange(n) - (n - 1) / 2) * pitch
    return np.stack([x0 + s * np.cos(angle), y0 + s * np.sin(angle)], axis=1)


def ring_positions(n: int, radius: float = 1e-3, phase: float = 0.0,
                   x0: float = 0.0, y0: float = 0.0) -> np.ndarray:
    a = phase + 2 * np.pi * np.arange(n) / max(n, 1)
    return np.stack([x0 + radius * np.cos(a), y0 + radius * np.sin(a)], axis=1)


def hex_positions(rings: int = 2, pitch: float = 1e-3,
                  x0: float = 0.0, y0: float = 0.0) -> np.ndarray:
    """Hexagonal close-packed layout, ``rings`` rings around the centre."""
    pts = [(0.0, 0.0)]
    for r in range(1, rings + 1):
        for i in range(6 * r):
            a = 2 * np.pi * i / (6 * r)
            pts.append((r * pitch * np.cos(a), r * pitch * np.sin(a)))
    p = np.asarray(pts, dtype=float)
    p[:, 0] += x0
    p[:, 1] += y0
    return p


def fan_angles(positions: np.ndarray, target: tuple[float, float, float]) -> np.ndarray:
    """Pointing angles ``(theta_x, theta_y)`` that aim each emitter at ``target``.

    ``target`` is ``(x, y, z)`` relative to the emitter plane.  Use this to make a
    source array converge on one modulator region, or (with a per-emitter target)
    to fan out.
    """
    tx, ty, tz = target
    dx = tx - positions[:, 0]
    dy = ty - positions[:, 1]
    return np.stack([np.arctan2(dx, tz), np.arctan2(dy, tz)], axis=1)


def layout(kind: str = "grid", **kw) -> np.ndarray:
    """Dispatch helper for configs: ``layout('grid', nx=8, ny=8, pitch_x=...)``."""
    fns = {
        "grid": grid_positions,
        "line": line_positions,
        "ring": ring_positions,
        "hex": hex_positions,
    }
    if kind not in fns:
        raise ValueError(f"unknown layout {kind!r}; choose from {sorted(fns)}")
    return fns[kind](**kw)
