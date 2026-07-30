"""Photodiode and avalanche-photodiode detection screens.

Unlike a camera these are *fast*: they resolve the temporal waveform, so this is
the detector to use when the machine's information is in the pulse train rather
than in an image, and the one that closes a high-bandwidth feedback loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import engine as eng
from ..fields import FieldSet
from ..grid import Grid
from ..registry import register
from ..units import K_B, Q_E, photon_energy
from .base import Detector, DetectorOutput, apply_bandwidth, time_samples


@register("detector", "photodiode", "pd", "pd_array")
@dataclass
class PhotodiodeArray(Detector):
    """Array of PIN photodiodes with a transimpedance front end.

    ``positions`` is an ``(N, 2)`` array of element centres, or ``None`` for a
    regular ``nx x ny`` grid.  Each element integrates irradiance over its active
    area, converts with ``responsivity``, low-passes at ``bandwidth``, and adds
    shot, dark and thermal (TIA) noise.

    Set ``apd_gain > 1`` for an avalanche photodiode: the signal is multiplied by
    M and the shot noise by ``sqrt(M^2 * F)`` with ``F`` the excess-noise factor,
    which is why an APD helps only until the excess noise catches the TIA noise.
    """

    name: str = "pd_array"
    nx: int = 8
    ny: int = 1
    pitch: float = 250e-6
    positions_in: object = None
    center: tuple[float, float] = (0.0, 0.0)
    active_diameter: float = 100e-6
    responsivity: float = 0.95  # A/W at 1550 nm for InGaAs
    bandwidth: float = 10e9
    dark_current: float = 5e-9  # A
    apd_gain: float = 1.0
    excess_noise_factor: float = 3.0
    transimpedance: float = 1e3  # Ohm
    tia_input_noise: float = 15e-12  # A/sqrt(Hz), input-referred
    temperature: float = 300.0
    load_resistance: float = 50.0
    n_time: int = 256
    window: float | None = None  # observation window [s]; None -> slowest waveform
    output: str = "volts"  # volts | amps | watts
    seed: int = 0

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    # ---- layout ------------------------------------------------------------ #
    @property
    def positions(self) -> np.ndarray:  # type: ignore[override]
        if self.positions_in is not None:
            return np.asarray(self.positions_in, dtype=float).reshape(-1, 2)
        xs = (np.arange(self.nx) - (self.nx - 1) / 2) * self.pitch + self.center[0]
        ys = (np.arange(self.ny) - (self.ny - 1) / 2) * self.pitch + self.center[1]
        X, Y = np.meshgrid(xs, ys)
        return np.stack([X.ravel(), Y.ravel()], axis=1)

    @property
    def active_area(self) -> float:
        return np.pi * (self.active_diameter / 2) ** 2

    # ---- detection --------------------------------------------------------- #
    def detect(self, fields: FieldSet, t=None, noiseless: bool = False) -> DetectorOutput:
        pos = self.positions
        grid = fields.grid
        window = self.window
        if window is None:
            periods = [ch.waveform.period for ch in fields.channels]
            window = max(periods) if periods else 1e-9
        ts = np.asarray(t, dtype=float).ravel() if t is not None else time_samples(
            fields, window, self.n_time)

        masks = _element_masks(grid, pos, self.active_diameter)
        # instantaneous irradiance, low-passed per element after area integration
        I = fields.intensity(ts, beat_bandwidth=self.bandwidth)  # (nt, ny, nx)
        In = eng.to_numpy(I).reshape(len(ts), -1)
        p_t = In @ masks.T * grid.cell_area  # (nt, N) watts on each element

        i_signal = self.responsivity * self.apd_gain * p_t
        i_filt = apply_bandwidth(i_signal.T, ts, self.bandwidth).T
        i_dark = self.dark_current

        if not noiseless:
            b = self.bandwidth if np.isfinite(self.bandwidth) else 1e12
            i_mean = np.maximum(i_filt, 0.0) + i_dark
            f = self.excess_noise_factor if self.apd_gain > 1 else 1.0
            var_shot = 2 * Q_E * i_mean * b * (self.apd_gain * f if self.apd_gain > 1 else 1.0)
            var_tia = (self.tia_input_noise**2) * b
            var_thermal = 4 * K_B * self.temperature / max(self.load_resistance, 1e-9) * b
            sigma = np.sqrt(var_shot + var_tia + var_thermal)
            i_filt = i_filt + self._rng.standard_normal(i_filt.shape) * sigma

        i_total = i_filt + i_dark
        if self.output == "amps":
            values, unit = i_total, "A"
        elif self.output == "watts":
            values, unit = p_t, "W"
        else:
            values, unit = i_total * self.transimpedance, "V"

        return DetectorOutput(
            values=values if len(ts) > 1 else values[0],
            unit=unit,
            optical_power=p_t.mean(axis=0),
            positions=pos,
            times=ts,
            shape=(len(pos),) if len(ts) == 1 else (len(ts), len(pos)),
            meta={"mean_current_A": float(np.mean(i_total)),
                  "bandwidth_Hz": self.bandwidth,
                  "mean_power_W": p_t.mean(axis=0)},
        )

    def mean_values(self, fields: FieldSet, **kw) -> np.ndarray:
        """Time-averaged output per element -- the usual feedback observable."""
        out = self.detect(fields, **kw)
        v = np.asarray(out.values, dtype=float)
        return v.mean(axis=0) if v.ndim == 2 else v

    # ---- reporting --------------------------------------------------------- #
    def nep(self) -> float:
        """Noise-equivalent power [W/sqrt(Hz)]."""
        i_n = np.sqrt(self.tia_input_noise**2 + 2 * Q_E * self.dark_current
                      + 4 * K_B * self.temperature / max(self.load_resistance, 1e-9))
        return float(i_n / (self.responsivity * self.apd_gain))

    def sensitivity_w(self, snr_db: float = 13.6) -> float:
        """Optical power for a target electrical SNR over the full bandwidth."""
        snr = 10 ** (snr_db / 20.0)
        return float(snr * self.nep() * np.sqrt(max(self.bandwidth, 1.0)))

    def info(self) -> dict:
        return {**super().info(), "bandwidth_Hz": self.bandwidth,
                "responsivity": self.responsivity, "apd_gain": self.apd_gain,
                "nep_W_rtHz": self.nep(), "active_diameter_um":
                self.active_diameter * 1e6}


@register("detector", "bolometer", "power_meter")
@dataclass
class PowerMeter(Detector):
    """Slow, calibrated total-power sensor over a set of areas.

    The simplest possible screen: no noise model beyond a fractional uncertainty.
    Useful for energy-conservation checks and quick optical budgets.
    """

    name: str = "power_meter"
    positions_in: object = ((0.0, 0.0),)
    diameter: float = 5e-3
    uncertainty: float = 0.0
    seed: int = 0

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    @property
    def positions(self) -> np.ndarray:  # type: ignore[override]
        return np.asarray(self.positions_in, dtype=float).reshape(-1, 2)

    def detect(self, fields: FieldSet, t=None, noiseless: bool = False) -> DetectorOutput:
        pos = self.positions
        masks = _element_masks(fields.grid, pos, self.diameter)
        I = eng.to_numpy(fields.mean_intensity()).ravel()
        p = masks @ I * fields.grid.cell_area
        v = p if noiseless or self.uncertainty <= 0 else p * (
            1 + self.uncertainty * self._rng.standard_normal(p.shape))
        return DetectorOutput(values=v, unit="W", optical_power=p, positions=pos,
                              shape=(len(pos),))


def _element_masks(grid: Grid, positions: np.ndarray, diameter: float) -> np.ndarray:
    """``(N, ny*nx)`` circular active-area masks, built once per geometry."""
    X, Y = (eng.to_numpy(a) for a in grid.meshgrid())
    r2 = (diameter / 2) ** 2
    out = np.zeros((len(positions), X.size))
    for i, (px, py) in enumerate(positions):
        out[i] = (((X - px) ** 2 + (Y - py) ** 2) <= r2).ravel().astype(float)
    return out
