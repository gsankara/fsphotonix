"""Camera detection screens: CCD / CMOS / high-speed sensor arrays."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import engine as eng
from ..fields import FieldSet
from ..grid import Grid
from ..registry import register
from ..units import photon_energy
from .base import (
    Detector,
    DetectorOutput,
    block_sum,
    mean_irradiance,
    shot_noise_electrons,
    time_samples,
)


@register("detector", "camera", "ccd", "cmos")
@dataclass
class CameraArray(Detector):
    """Pixelated imaging sensor.

    The full small-signal chain is modelled: quantum efficiency, fill factor,
    exposure integration, photon shot noise, dark current, pixel response
    non-uniformity, read noise, full-well saturation and ADC quantisation.  With
    ``shutter='rolling'`` each row is exposed at a different time, which matters
    when the machine's state is changing during the frame.

    Defaults are a fast global-shutter InGaAs/SWIR-style sensor; the visible-band
    silicon case is a matter of changing ``qe`` and the noise numbers.
    """

    name: str = "camera"
    nx: int = 128
    ny: int = 128
    pitch: float = 20e-6
    center: tuple[float, float] = (0.0, 0.0)
    fill_factor: float = 0.9
    qe: float = 0.7
    integration_time: float = 10e-6
    frame_rate: float = 1e4
    full_well_e: float = 30_000.0
    read_noise_e: float = 25.0
    dark_current_e_s: float = 5e3
    prnu: float = 0.01  # pixel response non-uniformity, fractional rms
    dsnu_e: float = 0.0  # dark signal non-uniformity [e-]
    adc_bits: int = 12
    gain_e_per_dn: float | None = None  # None -> map full well onto full scale
    shutter: str = "global"  # global | rolling
    n_time: int = 16
    output: str = "dn"  # dn | electrons | watts
    seed: int = 0

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)
        self._prnu_map = None

    # ---- layout ------------------------------------------------------------ #
    @property
    def positions(self) -> np.ndarray:  # type: ignore[override]
        xs = (np.arange(self.nx) - (self.nx - 1) / 2) * self.pitch + self.center[0]
        ys = (np.arange(self.ny) - (self.ny - 1) / 2) * self.pitch + self.center[1]
        X, Y = np.meshgrid(xs, ys)
        return np.stack([X.ravel(), Y.ravel()], axis=1)

    def grid(self, oversample: int = 4) -> Grid:
        """A field grid that samples this sensor properly (``oversample`` per pixel)."""
        return Grid(nx=self.nx * oversample, ny=self.ny * oversample,
                    dx=self.pitch / oversample, dy=self.pitch / oversample,
                    x0=self.center[0], y0=self.center[1])

    @property
    def active_area(self) -> float:
        return self.pitch**2 * self.fill_factor

    # ---- detection --------------------------------------------------------- #
    def detect(self, fields: FieldSet, t=None, integration_time: float | None = None,
               noiseless: bool = False) -> DetectorOutput:
        tint = self.integration_time if integration_time is None else integration_time
        grid = fields.grid
        by = max(1, int(round(self.pitch / grid.dy)))
        bx = max(1, int(round(self.pitch / grid.dx)))

        if self.shutter == "rolling":
            energy = self._rolling(fields, grid, tint, by, bx)
        else:
            I = mean_irradiance(fields, tint, self.n_time)
            energy = self._bin(I, grid, by, bx) * grid.cell_area * tint

        e_np = np.asarray(eng.to_numpy(energy), dtype=float)
        e_np *= self.fill_factor
        power = e_np / max(tint, 1e-300)

        lam = fields.wavelengths[0] if fields.wavelengths else 1550e-9
        photons = e_np / photon_energy(lam)
        electrons = photons * self.qe
        if not noiseless:
            electrons = self._add_noise(electrons, tint)
        sat = electrons >= self.full_well_e
        electrons = np.clip(electrons, 0.0, self.full_well_e)

        if self.output == "electrons":
            values, unit = electrons, "e-"
        elif self.output == "watts":
            values, unit = power, "W"
        else:
            g = self.gain_e_per_dn or (self.full_well_e / (2**self.adc_bits - 1))
            values = np.clip(np.floor(electrons / g), 0, 2**self.adc_bits - 1)
            unit = "DN"

        return DetectorOutput(
            values=values, unit=unit, optical_power=power, positions=self.positions,
            shape=values.shape, saturated=sat,
            meta={"integration_time": tint, "wavelength": lam,
                  "electrons": electrons, "well_fill": float(electrons.max()
                                                            / self.full_well_e)},
        )

    # ---- internals --------------------------------------------------------- #
    def _bin(self, I, grid: Grid, by: int, bx: int):
        """Sum irradiance samples into pixels, aligning the sensor to the grid."""
        # index of the sensor's first pixel corner on the field grid
        x0 = self.center[0] - self.nx * self.pitch / 2
        y0 = self.center[1] - self.ny * self.pitch / 2
        r0, c0 = grid.index_of(x0, y0)
        out = block_sum(I, by, bx, oy=r0, ox=c0, ny=self.ny, nx=self.nx)
        # pad if the field grid does not cover the whole sensor
        pad_y = self.ny - out.shape[0]
        pad_x = self.nx - out.shape[1]
        if pad_y > 0 or pad_x > 0:
            out = eng.xp.pad(out, ((0, max(pad_y, 0)), (0, max(pad_x, 0))))
        return out

    def _rolling(self, fields: FieldSet, grid: Grid, tint: float, by: int, bx: int):
        """Row-by-row exposure: row j starts at ``j / (ny * frame_rate)``."""
        row_dt = 1.0 / max(self.frame_rate * self.ny, 1e-9)
        out = eng.xp.zeros((self.ny, self.nx), dtype=eng.REAL)
        for j in range(self.ny):
            ts = time_samples(fields, tint, self.n_time) + j * row_dt
            I = eng.xp.zeros(grid.shape, dtype=eng.REAL)
            for i in range(0, ts.size, 16):
                I = I + eng.xp.sum(fields.intensity(ts[i : i + 16]), axis=0)
            I = I / ts.size
            row = self._bin(I, grid, by, bx)[j : j + 1, :]
            out[j : j + 1, :] = row * grid.cell_area * tint
        return out

    def _add_noise(self, electrons: np.ndarray, tint: float) -> np.ndarray:
        rng = self._rng
        if self._prnu_map is None or self._prnu_map.shape != electrons.shape:
            self._prnu_map = 1.0 + self.prnu * np.random.default_rng(
                self.seed + 1).standard_normal(electrons.shape)
        e = electrons * self._prnu_map
        e = e + self.dark_current_e_s * tint
        if self.dsnu_e > 0:
            e = e + np.abs(np.random.default_rng(self.seed + 2).normal(
                0, self.dsnu_e, electrons.shape))
        e = shot_noise_electrons(e, rng)
        if self.read_noise_e > 0:
            e = e + rng.standard_normal(e.shape) * self.read_noise_e
        return np.maximum(e, 0.0)

    # ---- reporting --------------------------------------------------------- #
    def sensitivity_w(self, snr: float = 1.0, wavelength: float = 1550e-9) -> float:
        """Optical power per pixel needed for a given SNR in one exposure."""
        t = self.integration_time
        ep = photon_energy(wavelength)
        # solve  q*S = snr * sqrt(q*S + dark + read^2)  for S (watts)
        a = (self.qe * t / ep) ** 2
        n0 = self.dark_current_e_s * t + self.read_noise_e**2
        q = self.qe * t / ep
        disc = (snr**2 * q) ** 2 + 4 * a * snr**2 * n0
        return float((snr**2 * q + np.sqrt(disc)) / (2 * a))

    def info(self) -> dict:
        return {**super().info(), "nx": self.nx, "ny": self.ny,
                "pitch_um": self.pitch * 1e6, "frame_rate": self.frame_rate,
                "integration_time_s": self.integration_time,
                "full_well_e": self.full_well_e, "adc_bits": self.adc_bits}
