"""Dammann gratings: binary-phase fan-out into an array of equal-intensity spots.

A Dammann grating is a binary (0/pi) phase grating whose transition points inside
one period are chosen so that the first ``N`` diffraction orders carry equal
power.  The transition points are found numerically at construction time (and
cached), so any ``N`` works rather than only the handful printed in the classic
tables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from ... import engine as eng
from ...registry import register
from .base import Grating

# --------------------------------------------------------------------------- #
# Design: piecewise-constant binary phase, analytic Fourier coefficients
# --------------------------------------------------------------------------- #
def _profile_coeffs(transitions: np.ndarray, orders: np.ndarray) -> np.ndarray:
    """Fourier coefficients ``c_m`` of a 0/pi profile with the given transitions.

    ``transitions`` are fractional positions in (0, 1); the phase starts at 0 and
    flips at every transition.
    """
    t = np.concatenate(([0.0], np.sort(transitions), [1.0]))
    sign = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(len(t) - 1)])
    c = np.zeros(len(orders), dtype=complex)
    for i, m in enumerate(orders):
        if m == 0:
            c[i] = np.sum(sign * np.diff(t))
        else:
            e = np.exp(-2j * np.pi * m * t)
            c[i] = np.sum(sign * (e[1:] - e[:-1])) / (-2j * np.pi * m)
    return c


def _cost(params: np.ndarray, orders: np.ndarray) -> float:
    t = _params_to_transitions(params)
    eta = np.abs(_profile_coeffs(t, orders)) ** 2
    total = eta.sum()
    if total <= 1e-12:
        return 10.0
    mean = eta.mean()
    uniformity = np.sum((eta - mean) ** 2) / (mean**2)
    return 30.0 * uniformity + (1.0 - total)


def _params_to_transitions(params: np.ndarray) -> np.ndarray:
    """Map unconstrained reals to sorted transitions in (0, 1)."""
    w = np.exp(np.clip(params, -8.0, 8.0))
    frac = np.cumsum(w) / (w.sum() + 1.0)  # strictly increasing, all < 1
    return np.clip(frac, 1e-4, 1 - 1e-4)


def _target_orders(n_orders: int) -> np.ndarray:
    """The ``n_orders`` orders a fan-out of that size is designed to fill.

    Odd counts are centred on the zero order; even counts are symmetric pairs
    about it (a 0/pi profile has a symmetric spectrum, and even fan-outs
    conventionally suppress the zero order).
    """
    if n_orders % 2 == 1:
        h = (n_orders - 1) // 2
        return np.arange(-h, h + 1)
    h = n_orders // 2
    return np.concatenate([np.arange(-h, 0), np.arange(1, h + 1)])


@lru_cache(maxsize=128)
def dammann_transitions(n_orders: int, n_transitions: int | None = None,
                        restarts: int = 10, seed: int = 0) -> tuple[float, ...]:
    """Optimised transition points for an ``n_orders``-spot Dammann grating."""
    if n_orders < 2:
        return (0.5,)
    orders = _target_orders(n_orders)
    k = n_transitions or max(2, 2 * ((n_orders + 1) // 2))
    rng = np.random.default_rng(seed)
    best, best_cost = None, np.inf
    try:
        from scipy.optimize import minimize  # noqa: PLC0415

        have_scipy = True
    except Exception:  # pragma: no cover - scipy is optional
        have_scipy = False

    for _ in range(restarts):
        x0 = rng.normal(0.0, 1.0, k)
        if have_scipy:
            res = minimize(_cost, x0, args=(orders,), method="Nelder-Mead",
                           options={"maxiter": 1500, "xatol": 1e-7, "fatol": 1e-9})
            x, c = res.x, float(res.fun)
        else:  # simple random-restart hill climb
            x, c = x0, _cost(x0, orders)
            step = 0.4
            for _ in range(3000):
                cand = x + rng.normal(0, step, k)
                cc = _cost(cand, orders)
                if cc < c:
                    x, c = cand, cc
                else:
                    step *= 0.999
        if c < best_cost:
            best, best_cost = x, c
    return tuple(float(v) for v in _params_to_transitions(best))


@register("grating", "dammann", "fanout")
@dataclass
class DammannGrating(Grating):
    """Binary-phase fan-out grating.

    ``n_x`` / ``n_y`` are the number of equal-intensity spots along each axis
    (set ``n_y=1`` for a 1-D fan-out).  ``period`` is the grating spacing, which
    together with the wavelength and range sets the spot pitch on the screen:
    ``pitch = lambda * z / period``.
    """

    period: float = 40e-6
    period_y: float | None = None
    n_x: int = 3
    n_y: int = 1
    angle: float = 0.0
    transitions: tuple[float, ...] | None = None
    transitions_y: tuple[float, ...] | None = None
    design_wavelength: float = 1550e-9
    achromatic: bool = False
    efficiency_scale: float = 1.0

    def __post_init__(self):
        if self.transitions is None:
            self.transitions = dammann_transitions(self.n_x)
        if self.n_y > 1 and self.transitions_y is None:
            self.transitions_y = dammann_transitions(self.n_y)
        if self.period_y is None:
            self.period_y = self.period

    def _binary(self, s, period, transitions):
        frac = eng.to_numpy(eng.xp.mod(s / period, 1.0))
        t = np.sort(np.asarray(transitions, dtype=float))
        idx = np.searchsorted(t, frac, side="right")
        return eng.asarray(np.where(idx % 2 == 0, 0.0, np.pi), dtype=eng.REAL)

    def phase(self, xl, yl):
        c, s = np.cos(self.angle), np.sin(self.angle)
        u = xl * c + yl * s
        p = self._binary(u, self.period, self.transitions)
        if self.n_y > 1:
            v = -xl * s + yl * c
            p = p + self._binary(v, self.period_y, self.transitions_y)
        return p

    def unit_cell(self):
        c, s = np.cos(self.angle), np.sin(self.angle)
        a1 = (self.period * c, self.period * s)
        if self.n_y > 1:
            return (a1, (-self.period_y * s, self.period_y * c))
        return (a1, None)

    # ---- design helpers ---------------------------------------------------- #
    @classmethod
    def for_spot_pitch(cls, pitch: float, z: float, wavelength: float = 1550e-9,
                       n_x: int = 3, n_y: int = 1, **kw) -> "DammannGrating":
        """Grating giving screen spots separated by ``pitch`` at range ``z``."""
        period = wavelength * z / pitch
        return cls(period=period, n_x=n_x, n_y=n_y, design_wavelength=wavelength, **kw)

    def spot_pitch(self, z: float, wavelength: float | None = None) -> float:
        lam = wavelength or self.design_wavelength
        return lam * z / self.period

    def uniformity(self, wavelength: float | None = None) -> float:
        """``(max-min)/(max+min)`` over the designed orders -- lower is better."""
        orders = [o for o in self.orders(wavelength, max_order=max(self.n_x, self.n_y))]
        want = [o for o in orders if abs(o.mx) <= self.n_x // 2 and abs(o.my) <= self.n_y // 2]
        if not want:
            return 1.0
        e = np.array([o.efficiency for o in want])
        return float((e.max() - e.min()) / (e.max() + e.min()))
