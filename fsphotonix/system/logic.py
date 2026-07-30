"""Electronic post-processing: single- and multi-stage digital logic.

Between the detector and the next optical stage sits electronics.  Its depth is a
first-class design variable in this class of machine: a purely analog loop
(detector -> amplifier -> modulator) is fast but noisy, while inserting one or
several *digital* stages (quantise -> logic -> re-drive) buys precision and
programmability at the cost of latency.

A :class:`LogicBlock` is a pure function of its input plus optional internal
state.  :class:`Pipeline` chains them, and :class:`Register` inserts a clock cycle
of delay so that multi-stage pipelines have realistic latency.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from ..registry import register


class LogicBlock:
    """Base class: ``__call__(x) -> y``, optionally stateful."""

    name: str = "logic"
    latency: int = 0  # clock cycles this block adds

    def __call__(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def reset(self) -> None:
        pass

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}({self.name!r})"


@register("controller", "gain")
@dataclass
class Gain(LogicBlock):
    """Scale and offset -- the analog trans-impedance / VOA stage."""

    gain: float = 1.0
    offset: float = 0.0
    name: str = "gain"
    latency: int = 0

    def __call__(self, x):
        return np.asarray(x, dtype=float) * self.gain + self.offset


@register("controller", "normalize")
@dataclass
class Normalize(LogicBlock):
    """Divide by a reference: sum, max, or a monitored element.

    An optical machine has no absolute scale (source power drifts, the modulator
    loses light), so almost every real feedback path normalises before deciding.
    """

    mode: str = "max"  # max | sum | l2 | element | fixed
    index: int = 0
    reference: float = 1.0
    epsilon: float = 1e-12
    name: str = "normalize"
    latency: int = 0

    def __call__(self, x):
        v = np.asarray(x, dtype=float)
        if self.mode == "max":
            d = np.abs(v).max()
        elif self.mode == "sum":
            d = v.sum()
        elif self.mode == "l2":
            d = np.linalg.norm(v)
        elif self.mode == "element":
            d = v.ravel()[self.index]
        elif self.mode == "fixed":
            d = self.reference
        else:
            raise ValueError(f"unknown normalise mode {self.mode!r}")
        return v / (abs(d) + self.epsilon)


@register("controller", "threshold")
@dataclass
class Threshold(LogicBlock):
    """Comparator with optional hysteresis -- one bit of digital decision."""

    level: float = 0.5
    hysteresis: float = 0.0
    high: float = 1.0
    low: float = 0.0
    name: str = "threshold"
    latency: int = 1

    def __post_init__(self):
        self._state: np.ndarray | None = None

    def __call__(self, x):
        v = np.asarray(x, dtype=float)
        if self._state is None or self._state.shape != v.shape:
            self._state = np.zeros(v.shape, dtype=bool)
        up = self.level + self.hysteresis / 2
        dn = self.level - self.hysteresis / 2
        self._state = np.where(self._state, v > dn, v > up)
        return np.where(self._state, self.high, self.low)

    def reset(self):
        self._state = None


@register("controller", "quantize", "adc")
@dataclass
class Quantizer(LogicBlock):
    """ADC: clip to a range and quantise to ``bits``, with optional dither."""

    bits: int = 8
    vmin: float = 0.0
    vmax: float = 1.0
    dither: float = 0.0
    name: str = "adc"
    latency: int = 1

    def __call__(self, x):
        v = np.clip(np.asarray(x, dtype=float), self.vmin, self.vmax)
        n = 2**self.bits - 1
        s = (v - self.vmin) / max(self.vmax - self.vmin, 1e-300)
        if self.dither > 0:
            s = s + np.random.default_rng().normal(0, self.dither / n, s.shape)
        return np.round(np.clip(s, 0, 1) * n) / n * (self.vmax - self.vmin) + self.vmin


@register("controller", "nonlinearity", "activation")
@dataclass
class Nonlinearity(LogicBlock):
    """Point nonlinearity: the electronic activation of an optical layer."""

    kind: str = "tanh"  # tanh | relu | sigmoid | clip | sign | abs | square | custom
    scale: float = 1.0
    bias: float = 0.0
    func: Callable | None = None
    name: str = "activation"
    latency: int = 0

    def __call__(self, x):
        v = np.asarray(x, dtype=float) * self.scale + self.bias
        k = self.kind
        if self.func is not None or k == "custom":
            return np.asarray(self.func(v), dtype=float)
        if k == "tanh":
            return np.tanh(v)
        if k == "relu":
            return np.maximum(v, 0.0)
        if k == "sigmoid":
            return 1.0 / (1.0 + np.exp(-v))
        if k == "clip":
            return np.clip(v, -1.0, 1.0)
        if k == "sign":
            return np.sign(v)
        if k == "abs":
            return np.abs(v)
        if k == "square":
            return v * v
        raise ValueError(f"unknown nonlinearity {k!r}")


@register("controller", "lut")
@dataclass
class LUT(LogicBlock):
    """Look-up table on a quantised input -- an arbitrary digital function."""

    table: Sequence[float] = (0.0, 1.0)
    vmin: float = 0.0
    vmax: float = 1.0
    name: str = "lut"
    latency: int = 1

    def __call__(self, x):
        t = np.asarray(self.table, dtype=float)
        s = (np.asarray(x, dtype=float) - self.vmin) / max(self.vmax - self.vmin, 1e-300)
        idx = np.clip((s * (len(t) - 1)).round().astype(int), 0, len(t) - 1)
        return t[idx]


@register("controller", "matrix")
@dataclass
class MatrixMultiply(LogicBlock):
    """Electronic mixing matrix -- crossbar / routing between optical stages."""

    matrix: object = None
    name: str = "matmul"
    latency: int = 1

    def __call__(self, x):
        W = np.asarray(self.matrix, dtype=float)
        return W @ np.asarray(x, dtype=float).ravel()


@register("controller", "boolean")
@dataclass
class Boolean(LogicBlock):
    """Bitwise logic between an input vector and a stored mask/operand.

    With ``pairs`` given, gates are evaluated between element pairs, which is how
    a diffractive plane implementing many two-input gates in parallel is read out.
    """

    op: str = "and"  # and | or | xor | nand | nor | xnor | not
    operand: object = None
    pairs: object = None  # (M, 2) index pairs
    name: str = "boolean"
    latency: int = 1

    def __call__(self, x):
        a = np.asarray(x, dtype=float) > 0.5
        if self.pairs is not None:
            p = np.asarray(self.pairs, dtype=int).reshape(-1, 2)
            u, v = a.ravel()[p[:, 0]], a.ravel()[p[:, 1]]
        elif self.operand is not None:
            u, v = a, np.asarray(self.operand, dtype=bool)
        else:
            u = v = a
        table = {
            "and": u & v, "or": u | v, "xor": u ^ v,
            "nand": ~(u & v), "nor": ~(u | v), "xnor": ~(u ^ v), "not": ~u,
        }
        if self.op not in table:
            raise ValueError(f"unknown boolean op {self.op!r}")
        return table[self.op].astype(float)


@register("controller", "register", "delay")
@dataclass
class Register(LogicBlock):
    """Clocked pipeline register: ``depth`` cycles of latency."""

    depth: int = 1
    initial: float = 0.0
    name: str = "register"

    def __post_init__(self):
        self._q: deque = deque()
        self.latency = self.depth

    def __call__(self, x):
        v = np.asarray(x, dtype=float)
        self._q.append(v.copy())
        while len(self._q) > self.depth + 1:
            self._q.popleft()
        if len(self._q) <= self.depth:
            return np.full(v.shape, self.initial)
        return self._q.popleft()

    def reset(self):
        self._q.clear()


@register("controller", "integrator")
@dataclass
class Integrator(LogicBlock):
    """Leaky accumulator -- the analog memory of an iterative machine."""

    rate: float = 0.1
    leak: float = 0.0
    clip: tuple[float, float] | None = (-1.0, 1.0)
    name: str = "integrator"
    latency: int = 0

    def __post_init__(self):
        self._acc: np.ndarray | None = None

    def __call__(self, x):
        v = np.asarray(x, dtype=float)
        if self._acc is None or self._acc.shape != v.shape:
            self._acc = np.zeros(v.shape)
        self._acc = self._acc * (1 - self.leak) + self.rate * v
        if self.clip:
            self._acc = np.clip(self._acc, *self.clip)
        return self._acc.copy()

    def reset(self):
        self._acc = None

    @property
    def state(self):
        return None if self._acc is None else self._acc.copy()


@register("controller", "noise")
@dataclass
class NoiseInjector(LogicBlock):
    """Additive noise with an annealing schedule.

    Deliberate noise is what lets an analog iterative machine escape local minima;
    ``decay`` anneals it away as the loop converges.
    """

    sigma: float = 0.1
    decay: float = 0.99
    floor: float = 0.0
    seed: int = 0
    name: str = "noise"
    latency: int = 0

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)
        self._sigma = self.sigma

    def __call__(self, x):
        v = np.asarray(x, dtype=float)
        out = v + self._rng.standard_normal(v.shape) * self._sigma
        self._sigma = max(self._sigma * self.decay, self.floor)
        return out

    def reset(self):
        self._sigma = self.sigma

    @property
    def temperature(self) -> float:
        return self._sigma


@register("controller", "pipeline", "logic_pipeline")
@dataclass
class Pipeline(LogicBlock):
    """Chain of logic blocks; ``latency`` is the sum of the stages'."""

    blocks: Sequence[LogicBlock] = field(default_factory=tuple)
    name: str = "pipeline"

    def __post_init__(self):
        self.blocks = list(self.blocks)
        self.latency = int(sum(getattr(b, "latency", 0) for b in self.blocks))

    def __call__(self, x):
        v = x
        for b in self.blocks:
            v = b(v)
        return v

    def reset(self):
        for b in self.blocks:
            b.reset()

    def __or__(self, other: LogicBlock) -> "Pipeline":
        return Pipeline(blocks=[*self.blocks, other], name=self.name)

    def describe(self) -> str:
        return " -> ".join(f"{type(b).__name__}" for b in self.blocks)


def analog_path(gain: float = 1.0, nonlinearity: str = "tanh") -> Pipeline:
    """Zero-latency analog feedback path: normalise, amplify, saturate."""
    return Pipeline(blocks=[Normalize(mode="max"), Gain(gain=gain),
                            Nonlinearity(kind=nonlinearity)], name="analog")


def digital_path(bits: int = 8, stages: int = 1, gain: float = 1.0,
                 nonlinearity: str = "tanh") -> Pipeline:
    """``stages`` clocked digital stages, each quantise -> activate -> register."""
    blocks: list[LogicBlock] = [Normalize(mode="max"), Gain(gain=gain)]
    for i in range(max(stages, 1)):
        blocks += [Quantizer(bits=bits, vmin=-1.0, vmax=1.0),
                   Nonlinearity(kind=nonlinearity, name=f"act{i}"),
                   Register(depth=1, name=f"reg{i}")]
    return Pipeline(blocks=blocks, name=f"digital{stages}")


def as_logic(spec) -> LogicBlock:
    from ..registry import build

    if isinstance(spec, LogicBlock):
        return spec
    if callable(spec) and not isinstance(spec, (str, dict)):
        return Nonlinearity(kind="custom", func=spec)
    if isinstance(spec, (list, tuple)):
        return Pipeline(blocks=[as_logic(s) for s in spec])
    return build(spec, "controller")
