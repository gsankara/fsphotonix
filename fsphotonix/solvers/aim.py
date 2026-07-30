"""Optical matrix-vector multiply and the analog iterative machine built on it.

:class:`OpticalMVM` assembles a complete free-space multiplier: a bank of emitters
carrying the input vector, one modulator region per input whose holographic
sub-gratings fan that input out to every output with the requested weight, and a
detector array at the output plane.  Signed quantities are handled by dual-rail
encoding (a light beam cannot be negative), which the class sets up and
recombines for you:

* signed **inputs**  -> two emitters/regions per input (x+ and x-),
* signed **weights** -> two detector elements per output (y+ and y-),
* the readout subtracts the rails, so the result is a genuinely four-quadrant
  multiply.

:class:`AnalogIterativeMachine` closes an annealed feedback loop around that
multiplier to minimise an Ising / QUBO objective -- the architecture of Microsoft's
analog iterative machine.  The optical part is *never* short-circuited: every
iteration really does propagate light through the holograms and read a noisy
detector, so device limits (phase quantisation, fan-out uniformity, shot noise,
saturation, latency) show up in the convergence curve where they belong.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from .. import units as U
from ..detectors.photodiode import PhotodiodeArray
from ..fields import FieldSet
from ..grid import Grid
from ..modulators.array import ModulatorArray
from ..modulators.devices import as_device
from ..sources.lasers import LaserArray
from ..sources.profiles import SuperGaussian
from ..system.feedback import History, Loop, Machine
from ..system.logic import NoiseInjector, Nonlinearity, Normalize, Pipeline
from ..system.optical_system import OpticalSystem
from ..system.stages import DetectorStage, ModulatorStage, SourceStage


class OpticalMVM:
    """Free-space diffractive matrix-vector multiplier.

    Parameters
    ----------
    n_in, n_out : logical vector sizes (before dual-rail expansion).
    z : modulator-to-screen distance [m].
    region_pitch / region_size : geometry of the modulator panel.
    out_pitch : spacing of the detector elements on the screen.
    wavelength : design wavelength.
    device : modulator hardware plugin.
    detector : detector plugin instance (default: a fast photodiode array).
    signed_input / signed_weights : enable the corresponding dual rail.
    mode : ``'spot'`` (beamlet propagation -- fast, scales to large N) or
        ``'wave'`` (full diffraction on a sampled grid).
    """

    def __init__(
        self,
        n_in: int = 8,
        n_out: int = 8,
        z: float = 0.05,
        region_pitch: float = 400e-6,
        region_size: float | None = None,
        out_pitch: float = 250e-6,
        wavelength: float = 1550e-9,
        power_per_input: float = 1e-3,
        device=None,
        detector=None,
        signed_input: bool = True,
        signed_weights: bool = True,
        mode: str = "spot",
        oversample: int = 24,
        routing: str = "subregions",
        grating: str = "focuser",
        tiling: str = "grid",
        name: str = "mvm",
    ):
        self.n_in, self.n_out = int(n_in), int(n_out)
        self.z = z
        self.wavelength = wavelength
        self.signed_input = bool(signed_input)
        self.signed_weights = bool(signed_weights)
        self.mode = mode
        self.routing = routing
        self.grating = grating
        self.tiling = tiling
        self.name = name
        self.region_pitch = region_pitch
        self.region_size = region_size if region_size is not None else 0.85 * region_pitch
        self.out_pitch = out_pitch

        self.n_rails_in = 2 * self.n_in if self.signed_input else self.n_in
        self.n_rails_out = 2 * self.n_out if self.signed_weights else self.n_out

        # ---- panel layout: a near-square grid of regions ------------------- #
        self.panel_cols = int(np.ceil(np.sqrt(self.n_rails_in)))
        self.panel_rows = int(np.ceil(self.n_rails_in / self.panel_cols))
        self.device = as_device(device)

        self.modulator = ModulatorArray.tiled(
            nx=self.panel_cols, ny=self.panel_rows, pitch=region_pitch,
            region_size=self.region_size, device=self.device,
            background="absorb", weight_mode="area",
            default_wavelength=wavelength, name=f"{name}.panel",
        )
        # drop the surplus regions of the last row
        self.modulator.regions = self.modulator.regions[: self.n_rails_in]
        self.modulator.touch()
        positions = np.array([r.center for r in self.modulator.regions])

        # ---- sources: one emitter per input rail, aligned to its region ----- #
        self.source = LaserArray(
            n=self.n_rails_in, positions=positions,
            wavelength=wavelength, power=power_per_input,
            profile=SuperGaussian(diameter=self.region_size, order=8),
            linewidth=0.0, rin_dbc_hz=-300.0, name=f"{name}.src",
        )

        # ---- output rail positions on the screen --------------------------- #
        self.out_positions = self._output_positions()
        spot = 1.5 * wavelength * z / max(self.region_size / max(
            int(np.ceil(np.sqrt(self.n_rails_out))), 1), 1e-9)
        self.detector = detector or PhotodiodeArray(
            positions_in=self.out_positions,
            active_diameter=min(0.8 * out_pitch, max(spot, 20e-6)),
            responsivity=0.95, bandwidth=1e9, transimpedance=1e4,
            tia_input_noise=5e-12, output="watts", n_time=1,
            window=1e-9, name=f"{name}.det",
        )

        # ---- grids ---------------------------------------------------------- #
        panel_w = self.panel_cols * region_pitch * 1.2
        panel_h = self.panel_rows * region_pitch * 1.2
        if mode == "spot":
            pitch = self.region_size / max(oversample, 2)
        else:
            pitch = min(self.device.pitch, self.region_size / 32)
        self.panel_grid = Grid.from_extent(panel_w, panel_h, pitch)
        screen_w = max(np.ptp(self.out_positions[:, 0]) * 1.4, 4 * out_pitch)
        screen_h = max(np.ptp(self.out_positions[:, 1]) * 1.4, 4 * out_pitch)
        self.screen_grid = Grid.from_extent(screen_w, screen_h,
                                           max(min(out_pitch / 8, 40e-6), 2e-6))

        self.system = self._build_system()
        chk = self.check()
        if not chk["deflection_ok"]:
            warnings.warn(
                f"{name}: the layout needs {chk['max_deflection_needed_deg']:.2f} deg of "
                f"deflection but {type(self.device).__name__} can only reach "
                f"{chk['device_max_deflection_deg']:.2f} deg at {wavelength*1e9:.0f} nm. "
                "The excess grating frequency will alias on the device; increase z, "
                "shrink out_pitch/region_pitch, or use a finer-pitch modulator.",
                RuntimeWarning, stacklevel=2)
        self.W: np.ndarray | None = None
        self._transfer: np.ndarray | None = None
        self._scale: float = 1.0
        self._row_gain: np.ndarray | None = None

    # ---- construction ------------------------------------------------------ #
    def _output_positions(self) -> np.ndarray:
        """Screen positions of the output rails (y+ / y- interleaved)."""
        n = self.n_rails_out
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        xs = (np.arange(cols) - (cols - 1) / 2) * self.out_pitch
        ys = (np.arange(rows) - (rows - 1) / 2) * self.out_pitch
        pos = []
        for j in range(rows):
            for i in range(cols):
                if len(pos) < n:
                    pos.append((xs[i], ys[j]))
        return np.asarray(pos, dtype=float)

    def _build_system(self) -> OpticalSystem:
        src = SourceStage(source=self.source, grid=self.panel_grid, z=0.0, name="src")
        mod = ModulatorStage(modulator=self.modulator, z=0.0, grid=self.panel_grid,
                             name="panel", spot_mode=(self.mode == "spot"))
        det = DetectorStage(detector=self.detector, z=self.z, grid=self.screen_grid,
                            name="screen")
        return OpticalSystem([src, mod, det], name=self.name)

    # ---- programming ------------------------------------------------------- #
    def set_matrix(self, W) -> "OpticalMVM":
        """Program an ``n_out x n_in`` weight matrix onto the holograms."""
        W = np.asarray(W, dtype=float)
        if W.shape != (self.n_out, self.n_in):
            raise ValueError(f"expected matrix shape {(self.n_out, self.n_in)}, got {W.shape}")
        self.W = W
        Wx = self._expand(W)  # (n_rails_out, n_rails_in), non-negative
        col_sums = Wx.sum(axis=0)
        cmax = col_sums.max() if col_sums.size else 1.0
        for j, r in enumerate(self.modulator.regions):
            col = Wx[:, j]
            keep = np.nonzero(col > 0)[0]
            if keep.size == 0:
                r.enabled = False  # this input drives no output: park the region
                continue
            r.enabled = True
            targets = [(float(self.out_positions[m, 0]), float(self.out_positions[m, 1]),
                        float(col[m])) for m in keep]
            self.modulator.route(r, targets, self.z, wavelength=self.wavelength,
                                 mode=self.routing, grating=self.grating,
                                 tiling=self.tiling)
            # the hologram only redistributes its region's light, so each column is
            # power-normalised; restore the column sum by attenuating the region
            r.gain = float(np.sqrt(col_sums[j] / cmax)) if cmax > 0 else 1.0
        self.modulator.touch()
        self._transfer = None
        self._row_gain = None
        return self

    def _expand(self, W: np.ndarray) -> np.ndarray:
        """Dual-rail expansion of a signed matrix into non-negative optics."""
        Wp, Wm = np.maximum(W, 0.0), np.maximum(-W, 0.0)
        if self.signed_weights and self.signed_input:
            #      x+            x-
            # y+ [ W+            W- ]
            # y- [ W-            W+ ]
            top = np.concatenate([Wp, Wm], axis=1)
            bot = np.concatenate([Wm, Wp], axis=1)
            full = np.zeros((2 * self.n_out, 2 * self.n_in))
            full[0::2] = top
            full[1::2] = bot
            return full
        if self.signed_weights:
            full = np.zeros((2 * self.n_out, self.n_in))
            full[0::2] = Wp
            full[1::2] = Wm
            return full
        if self.signed_input:
            return np.concatenate([Wp, Wm], axis=1)
        return np.maximum(W, 0.0)

    def encode_input(self, x) -> np.ndarray:
        """Vector -> non-negative emitter drive levels."""
        x = np.asarray(x, dtype=float).ravel()
        if x.size != self.n_in:
            raise ValueError(f"expected {self.n_in} inputs, got {x.size}")
        if not self.signed_input:
            return np.maximum(x, 0.0)
        return np.concatenate([np.maximum(x, 0.0), np.maximum(-x, 0.0)])

    def decode_output(self, p) -> np.ndarray:
        """Detector powers -> signed output vector."""
        p = np.asarray(p, dtype=float).ravel()
        if not self.signed_weights:
            return p[: self.n_out]
        return p[0 : 2 * self.n_out : 2] - p[1 : 2 * self.n_out : 2]

    # ---- use --------------------------------------------------------------- #
    def multiply(self, x, calibrated: bool = True) -> np.ndarray:
        """Run light through the machine and return the (decoded) product."""
        self.source.set_drive(self.encode_input(x))
        trace = self.system.run(keep_fields=False)
        p = trace.outputs["screen"].vector()
        y = self.decode_output(p)
        if not calibrated:
            return y
        if self._row_gain is not None:
            return y / np.where(np.abs(self._row_gain) > 0, self._row_gain, 1.0)
        return y / self._scale if self._scale else y

    __call__ = multiply

    def transfer_matrix(self, probe: float = 1.0) -> np.ndarray:
        """Measure the *actual* end-to-end matrix by probing basis vectors.

        This is the honest version of ``W``: it folds in grating efficiency,
        fan-out non-uniformity, cross-talk between neighbouring outputs, vignetting
        and detector responsivity.  Comparing it with ``W`` is the main accuracy
        metric of a design.
        """
        if self._transfer is not None:
            return self._transfer
        T = np.zeros((self.n_out, self.n_in))
        for j in range(self.n_in):
            e = np.zeros(self.n_in)
            e[j] = probe
            T[:, j] = self.multiply(e, calibrated=False) / probe
        self._transfer = T
        return T

    def calibrate(self, mode: str = "rail") -> dict:
        """Fit the gain between the requested matrix and the measured one.

        ``mode='scalar'`` fits one number for the whole machine.  ``mode='rail'``
        (the default) fits one gain per output -- exactly the per-detector
        responsivity/trans-impedance trim any real instrument performs, and the
        thing that removes the systematic difference between the two rails of a
        dual-rail output.  Neither mode can hide cross-talk: the off-diagonal terms
        of the measured matrix are left untouched, and the reported residual is
        computed after the fit.
        """
        if self.W is None:
            raise RuntimeError("program a matrix with set_matrix() before calibrating")
        T = self.transfer_matrix()
        num = float(np.sum(T * self.W))
        den = float(np.sum(self.W * self.W))
        self._scale = num / den if den > 0 else 1.0
        if mode == "rail":
            g = np.ones(self.n_out)
            for m in range(self.n_out):
                d = float(self.W[m] @ self.W[m])
                g[m] = float(T[m] @ self.W[m]) / d if d > 0 else self._scale
            g[g == 0] = self._scale or 1.0
            self._row_gain = g
            fitted = T / g[:, None]
        elif mode == "scalar":
            self._row_gain = None
            fitted = T / (self._scale or 1.0)
        else:
            raise ValueError(f"unknown calibration mode {mode!r}")
        resid = fitted - self.W
        return {
            "mode": mode,
            "scale": self._scale,
            "row_gain": None if self._row_gain is None else self._row_gain.copy(),
            "rms_error": float(np.sqrt(np.mean(resid**2))),
            "max_error": float(np.abs(resid).max()),
            "relative_rms": float(np.sqrt(np.mean(resid**2))
                                  / max(np.sqrt(np.mean(self.W**2)), 1e-300)),
        }

    # ---- design checks ----------------------------------------------------- #
    def check(self) -> dict:
        """Feasibility report: deflection angles, sampling, fan-out per region."""
        corner = np.hypot(np.abs(self.out_positions[:, 0]).max()
                          + np.abs([r.center[0] for r in self.modulator.regions]).max(),
                          np.abs(self.out_positions[:, 1]).max()
                          + np.abs([r.center[1] for r in self.modulator.regions]).max())
        need = float(np.arctan2(corner, self.z))
        have = self.device.max_deflection(self.wavelength)
        fan = max((len(r.subregions) for r in self.modulator.regions), default=0)
        sub = self.region_size / max(np.sqrt(fan), 1)
        return {
            "max_deflection_needed_deg": np.degrees(need),
            "device_max_deflection_deg": np.degrees(have),
            "deflection_ok": need <= have,
            "max_fanout_per_region": fan,
            "subaperture_um": sub * 1e6,
            "pixels_per_subaperture": sub / self.device.pitch,
            "spot_size_um": self.wavelength * self.z / max(sub, 1e-12) * 1e6,
            "spot_vs_out_pitch": (self.wavelength * self.z / max(sub, 1e-12)) / self.out_pitch,
            "rails": {"in": self.n_rails_in, "out": self.n_rails_out},
            "panel_mm": (self.panel_cols * self.region_pitch * 1e3,
                         self.panel_rows * self.region_pitch * 1e3),
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"OpticalMVM({self.n_out}x{self.n_in}, mode={self.mode!r}, "
                f"z={self.z*1e3:.3g}mm, rails={self.n_rails_out}x{self.n_rails_in})")


# --------------------------------------------------------------------------- #
# Analog iterative machine
# --------------------------------------------------------------------------- #
@dataclass
class AIMResult:
    """Outcome of an AIM run."""

    spins: np.ndarray
    energy: float
    best_spins: np.ndarray
    best_energy: float
    energies: np.ndarray
    states: np.ndarray
    iterations: int

    def summary(self) -> dict:
        return {"energy": self.energy, "best_energy": self.best_energy,
                "iterations": self.iterations,
                "converged_at": int(np.argmin(self.energies))}


class AnalogIterativeMachine:
    """Annealed gradient-descent loop closed through an optical multiplier.

    Minimises the Ising energy ``E(s) = -0.5 s^T J s - h^T s`` over analog spins
    ``s`` in [-1, 1], reading ``J s`` optically each iteration:

        g = J s + h                (optical, noisy, quantised)
        s <- clip(s + step * g + annealed_noise)

    Setting ``optics=None`` substitutes an exact numeric multiply, which is the
    control experiment: run both and the difference is entirely attributable to
    the photonics.
    """

    def __init__(
        self,
        J,
        h=None,
        optics: OpticalMVM | None = None,
        step: float = 0.05,
        noise: float = 0.25,
        noise_decay: float = 0.995,
        noise_floor: float = 0.0,
        clip: float = 1.0,
        seed: int = 0,
        normalize: bool = True,
    ):
        self.J = np.asarray(J, dtype=float)
        if self.J.ndim != 2 or self.J.shape[0] != self.J.shape[1]:
            raise ValueError("J must be square")
        self.n = self.J.shape[0]
        self.h = np.zeros(self.n) if h is None else np.asarray(h, dtype=float).ravel()
        self.optics = optics
        self.step = step
        self.clip = clip
        self.normalize = normalize
        self.noise = NoiseInjector(sigma=noise, decay=noise_decay, floor=noise_floor,
                                   seed=seed)
        self._rng = np.random.default_rng(seed)
        if optics is not None:
            if optics.n_in != self.n or optics.n_out != self.n:
                raise ValueError(
                    f"optics is {optics.n_out}x{optics.n_in} but J is {self.n}x{self.n}")
            scale = np.abs(self.J).max() or 1.0
            optics.set_matrix(self.J / scale)
            optics.calibrate()
            self._jscale = scale
        else:
            self._jscale = 1.0

    # ---- objective --------------------------------------------------------- #
    def energy(self, s) -> float:
        s = np.asarray(s, dtype=float).ravel()
        return float(-0.5 * s @ self.J @ s - self.h @ s)

    def ising_energy(self, s) -> float:
        """Energy of the hard-thresholded (binary) spins."""
        return self.energy(np.sign(np.where(s == 0, 1.0, s)))

    def gradient(self, s) -> np.ndarray:
        if self.optics is None:
            return self.J @ s + self.h
        return self.optics.multiply(s) * self._jscale + self.h

    # ---- solve ------------------------------------------------------------- #
    def solve(self, iterations: int = 300, s0=None, record_states: bool = False,
              callback=None) -> AIMResult:
        s = (self._rng.uniform(-0.1, 0.1, self.n) if s0 is None
             else np.asarray(s0, dtype=float).ravel().copy())
        self.noise.reset()
        energies = np.empty(iterations)
        states = np.empty((iterations, self.n)) if record_states else np.empty((0, self.n))
        best_s, best_e = np.sign(s), np.inf

        for k in range(iterations):
            g = self.gradient(s)
            if self.normalize:
                m = np.abs(g).max()
                if m > 0:
                    g = g / m
            s = np.clip(self.noise(s + self.step * g), -self.clip, self.clip)
            e = self.ising_energy(s)
            energies[k] = e
            if record_states:
                states[k] = s
            if e < best_e:
                best_e, best_s = e, np.sign(np.where(s == 0, 1.0, s))
            if callback is not None:
                callback(k, s, e, self.noise.temperature)

        return AIMResult(spins=np.sign(np.where(s == 0, 1.0, s)), energy=self.ising_energy(s),
                         best_spins=best_s, best_energy=best_e, energies=energies,
                         states=states, iterations=iterations)

    def as_machine(self, iterations: int = 300) -> Machine:
        """Expose the same loop through the generic :class:`Machine` scheduler.

        Useful when the AIM loop has to coexist with other loops (a slow alignment
        servo, a source-power regulator) in one simulation.
        """
        if self.optics is None:
            raise RuntimeError("as_machine() requires an OpticalMVM")
        state = {"s": self._rng.uniform(-0.1, 0.1, self.n)}

        def controller(y: np.ndarray) -> np.ndarray:
            g = self.optics.decode_output(y) if y.size != self.n else y
            g = g * self._jscale + self.h
            m = np.abs(g).max()
            if self.normalize and m > 0:
                g = g / m
            s = np.clip(self.noise(state["s"] + self.step * g), -self.clip, self.clip)
            state["s"] = s
            return self.optics.encode_input(s)

        loop = Loop(sense="screen", actuator=_DriveAdapter(self.optics),
                    controller=controller, name="aim", scope="global")
        m = Machine(self.optics.system, loops=[loop],
                    metric=lambda tr, k: {"energy": self.ising_energy(state["s"]),
                                          "temperature": self.noise.temperature},
                    name="aim")
        m.state = state  # type: ignore[attr-defined]
        return m


@dataclass
class _DriveAdapter:
    """Actuator that writes already-encoded rail drives to the MVM source."""

    mvm: OpticalMVM
    name: str = "aim.drive"

    @property
    def size(self) -> int:
        return self.mvm.n_rails_in

    def write(self, values):
        self.mvm.source.set_drive(np.clip(np.asarray(values, dtype=float).ravel(), 0, None))

    def read(self):
        return self.mvm.source.get_drive()


# --------------------------------------------------------------------------- #
# problem builders
# --------------------------------------------------------------------------- #
def maxcut_ising(adjacency) -> tuple[np.ndarray, np.ndarray]:
    """Max-Cut on a weighted graph -> Ising ``(J, h)``.

    Cut value ``= (sum(W) - s^T W s / 2) / 2``, so minimising
    ``E = -0.5 s^T J s`` with ``J = -W`` maximises the cut.
    """
    W = np.asarray(adjacency, dtype=float)
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)
    return -W, np.zeros(W.shape[0])


def cut_value(adjacency, spins) -> float:
    W = np.asarray(adjacency, dtype=float)
    s = np.asarray(spins, dtype=float).ravel()
    return float((W.sum() - s @ W @ s) / 4.0)


def qubo_to_ising(Q, c=None) -> tuple[np.ndarray, np.ndarray, float]:
    """QUBO ``x^T Q x`` over x in {0,1} -> Ising ``(J, h, offset)`` over s in {-1,1}."""
    Q = np.asarray(Q, dtype=float)
    Q = 0.5 * (Q + Q.T)
    n = Q.shape[0]
    d = np.diag(Q).copy()
    Qo = Q - np.diag(d)
    J = -Qo / 2.0
    h = -(d / 2.0 + Qo.sum(axis=1) / 2.0)
    offset = float(d.sum() / 2 + Qo.sum() / 4)
    if c is not None:
        offset += float(c)
    return J, h, offset


def random_graph(n: int = 16, density: float = 0.5, seed: int = 0,
                 weights: str = "unit") -> np.ndarray:
    """Random undirected weighted graph adjacency matrix."""
    rng = np.random.default_rng(seed)
    A = (rng.random((n, n)) < density).astype(float)
    if weights == "gaussian":
        A = A * rng.standard_normal((n, n))
    elif weights == "uniform":
        A = A * rng.uniform(0.2, 1.0, (n, n))
    A = np.triu(A, 1)
    return A + A.T


def brute_force_maxcut(adjacency) -> tuple[float, np.ndarray]:
    """Exact Max-Cut by enumeration -- only for n <= 20, as a correctness check."""
    W = np.asarray(adjacency, dtype=float)
    n = W.shape[0]
    if n > 20:
        raise ValueError("brute force is only sensible up to n=20")
    best, best_s = -np.inf, None
    for m in range(2 ** (n - 1)):
        s = np.array([1.0] + [1.0 if (m >> i) & 1 else -1.0 for i in range(n - 1)])
        v = cut_value(W, s)
        if v > best:
            best, best_s = v, s
    return float(best), best_s
