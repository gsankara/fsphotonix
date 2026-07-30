"""Hardware models for the modulator plane (plugin kind ``"device"``).

A :class:`ModulatorDevice` takes the *ideal* complex transmittance requested by
the holograms and degrades it the way real hardware would: pixelation at the
device pitch, phase quantisation, limited stroke, inter-pixel cross-talk,
dead-space fill factor, surface flatness error, dead pixels and insertion loss.

The distinction that matters most for system design is what the device can
actually represent:

* analog phase (LCoS, MEMS piston such as the Fraunhofer IPMS arrays)
* binary amplitude only (a tilting micro-mirror / DMD), where a complex hologram
  must be encoded by Lee-type binary holography or accepted as amplitude-only
* analog amplitude (LC + polarisers), which throws away the phase

Reflective and transmissive variants differ in the optical path a piston or LC
layer sees: a reflective element gets the phase twice.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import engine as eng
from ..grid import Grid
from ..registry import register


def _block_average(a, by: int, bx: int):
    """Average an array over (by, bx) blocks and expand back to full size."""
    ny, nx = a.shape
    py, px = (-ny) % by, (-nx) % bx
    if py or px:
        a = eng.xp.pad(a, ((0, py), (0, px)), mode="edge")
    Ny, Nx = a.shape
    blocks = a.reshape(Ny // by, by, Nx // bx, bx).mean(axis=(1, 3))
    out = eng.xp.repeat(eng.xp.repeat(blocks, by, axis=0), bx, axis=1)
    return out[:ny, :nx]


def _gaussian_blur(a, sigma_y: float, sigma_x: float):
    """Separable Gaussian blur (FFT-based, periodic)."""
    if sigma_x <= 0 and sigma_y <= 0:
        return a
    ny, nx = a.shape
    fy = eng.fftfreq(ny, 1.0).reshape(-1, 1)
    fx = eng.fftfreq(nx, 1.0).reshape(1, -1)
    h = eng.xp.exp(-2 * (np.pi**2) * ((sigma_y * fy) ** 2 + (sigma_x * fx) ** 2))
    return eng.ifft2(eng.fft2(a) * h)


class ModulatorDevice:
    """Base class for modulator hardware."""

    name = "device"
    pitch: float = 8e-6
    reflective: bool = True
    fill_factor: float = 0.95
    efficiency: float = 1.0  # reflectance or transmittance of the active area
    phase_bits: int | None = None
    max_phase: float = 2 * np.pi  # available stroke [rad] at ``design_wavelength``
    design_wavelength: float = 1550e-9
    crosstalk: float = 0.0  # inter-pixel blur, in units of the device pitch
    flatness_rms: float = 0.0  # surface error [rad rms] at design wavelength
    dead_pixel_fraction: float = 0.0
    frame_rate: float = 1e3  # Hz -- how fast a feedback loop can update it
    settling_time: float = 0.0  # s
    seed: int = 0

    #: what the device can represent: 'phase' | 'binary_amplitude' | 'amplitude' | 'complex'
    modulation: str = "phase"

    # ---- main entry point -------------------------------------------------- #
    def render(self, target, grid: Grid, wavelength: float):
        """Degrade an ideal complex transmittance into what the hardware gives."""
        t = eng.asarray(target, dtype=eng.COMPLEX)
        t = self.represent(t, wavelength)
        t = self.pixelate(t, grid)
        t = self.apply_crosstalk(t, grid)
        t = t * self.aperture_mask(grid)
        t = t * self.static_errors(grid, wavelength)
        return t * np.sqrt(max(self.efficiency, 0.0))

    # ---- pipeline steps ---------------------------------------------------- #
    def represent(self, t, wavelength: float):
        """Project the ideal transmittance onto the device's modulation space."""
        scale = self.design_wavelength / wavelength if wavelength else 1.0
        if self.modulation == "complex":
            return t
        if self.modulation == "phase":
            p = eng.xp.angle(t)
            stroke = self.max_phase * scale
            if stroke < 2 * np.pi - 1e-9:
                p = eng.xp.clip(eng.xp.mod(p, 2 * np.pi), 0.0, stroke)
            else:
                p = eng.xp.mod(p, 2 * np.pi)
            p = self.quantize_phase(p, stroke)
            return eng.xp.exp(1j * p)
        if self.modulation == "amplitude":
            a = eng.xp.clip(eng.xp.abs(t), 0.0, 1.0)
            return self.quantize_amplitude(a).astype(eng.COMPLEX)
        if self.modulation == "binary_amplitude":
            a = eng.xp.abs(t)
            thr = float(eng.to_numpy(eng.xp.mean(a)))
            return (a >= thr).astype(eng.COMPLEX)
        raise ValueError(f"unknown modulation mode {self.modulation!r}")

    def quantize_phase(self, p, stroke: float):
        if not self.phase_bits:
            return p
        levels = 2**self.phase_bits
        step = stroke / (levels - 1)
        return eng.xp.round(p / step) * step

    def quantize_amplitude(self, a):
        if not self.phase_bits:
            return a
        levels = 2**self.phase_bits
        return eng.xp.round(a * (levels - 1)) / (levels - 1)

    def pixelate(self, t, grid: Grid):
        """Average onto device pixels.  No-op if the grid is coarser than a pixel."""
        by = max(1, int(round(self.pitch / grid.dy)))
        bx = max(1, int(round(self.pitch / grid.dx)))
        if by == 1 and bx == 1:
            return t
        return _block_average(t, by, bx)

    def apply_crosstalk(self, t, grid: Grid):
        if self.crosstalk <= 0:
            return t
        s_y = self.crosstalk * self.pitch / grid.dy
        s_x = self.crosstalk * self.pitch / grid.dx
        return _gaussian_blur(t, s_y, s_x)

    def aperture_mask(self, grid: Grid):
        """Per-pixel active-area mask -- the source of pixel replica orders."""
        key = ("ap", grid.shape, grid.dx, grid.dy, self.pitch, self.fill_factor)
        cached = getattr(self, "_ap_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        if self.fill_factor >= 0.999:
            m = eng.xp.ones(grid.shape, dtype=eng.REAL)
        else:
            X, Y = grid.meshgrid()
            f = np.sqrt(self.fill_factor)
            u = eng.xp.abs(eng.xp.mod(X / self.pitch + 0.5, 1.0) - 0.5)
            v = eng.xp.abs(eng.xp.mod(Y / self.pitch + 0.5, 1.0) - 0.5)
            m = ((u <= f / 2) & (v <= f / 2)).astype(eng.REAL)
        self._ap_cache = (key, m)
        return m

    def static_errors(self, grid: Grid, wavelength: float):
        """Frozen flatness error and dead pixels (deterministic per device)."""
        key = ("err", grid.shape, grid.dx, self.pitch, wavelength)
        cached = getattr(self, "_err_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        out = eng.xp.ones(grid.shape, dtype=eng.COMPLEX)
        if self.flatness_rms > 0 or self.dead_pixel_fraction > 0:
            by = max(1, int(round(self.pitch / grid.dy)))
            bx = max(1, int(round(self.pitch / grid.dx)))
            npy, npx = int(np.ceil(grid.ny / by)), int(np.ceil(grid.nx / bx))
            rng = np.random.default_rng(self.seed)
            e = np.ones((npy, npx), dtype=complex)
            if self.flatness_rms > 0:
                scale = self.design_wavelength / wavelength if wavelength else 1.0
                e *= np.exp(1j * rng.normal(0, self.flatness_rms * scale, (npy, npx)))
            if self.dead_pixel_fraction > 0:
                e *= (rng.random((npy, npx)) >= self.dead_pixel_fraction)
            big = np.repeat(np.repeat(e, by, axis=0), bx, axis=1)[: grid.ny, : grid.nx]
            out = eng.asarray(big, dtype=eng.COMPLEX)
        self._err_cache = (key, out)
        return out

    # ---- reporting --------------------------------------------------------- #
    def max_deflection(self, wavelength: float) -> float:
        """Largest steerable angle [rad]: two pixels per grating period."""
        return float(np.arcsin(min(1.0, wavelength / (2 * self.pitch))))

    def resolvable_spots(self, aperture: float, wavelength: float) -> int:
        """Space-bandwidth product of one aperture: how many spots it can address."""
        return max(1, int(aperture / self.pitch))

    def info(self) -> dict:
        return {
            "name": self.name,
            "modulation": self.modulation,
            "reflective": self.reflective,
            "pitch_um": self.pitch * 1e6,
            "fill_factor": self.fill_factor,
            "phase_bits": self.phase_bits,
            "max_phase_waves": self.max_phase / (2 * np.pi),
            "frame_rate_hz": self.frame_rate,
            "max_deflection_deg": np.degrees(self.max_deflection(self.design_wavelength)),
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}({self.name!r}, pitch={self.pitch*1e6:.2f}um)"


# --------------------------------------------------------------------------- #
# Concrete devices
# --------------------------------------------------------------------------- #
@register("device", "ideal")
@dataclass
class IdealModulator(ModulatorDevice):
    """Perfect complex modulator -- the reference for ablation studies."""

    name: str = "ideal"
    pitch: float = 1e-6
    reflective: bool = True
    fill_factor: float = 1.0
    efficiency: float = 1.0
    modulation: str = "complex"
    phase_bits: int | None = None
    max_phase: float = 2 * np.pi
    design_wavelength: float = 1550e-9
    crosstalk: float = 0.0
    flatness_rms: float = 0.0
    dead_pixel_fraction: float = 0.0
    frame_rate: float = 1e6
    settling_time: float = 0.0
    seed: int = 0


@register("device", "lcos", "lcos_slm", "phase_slm")
@dataclass
class LCoSPhaseSLM(ModulatorDevice):
    """Reflective liquid-crystal-on-silicon phase SLM.

    Defaults follow a typical 1550 nm telecom-grade panel: ~6-8 um pitch, 8-bit
    phase, 2*pi stroke, appreciable inter-pixel cross-talk (LC fringing field)
    and a few hundred Hz frame rate.
    """

    name: str = "lcos"
    pitch: float = 8e-6
    reflective: bool = True
    fill_factor: float = 0.93
    efficiency: float = 0.75  # incl. polariser + fringing losses
    modulation: str = "phase"
    phase_bits: int | None = 8
    max_phase: float = 2 * np.pi
    design_wavelength: float = 1550e-9
    crosstalk: float = 0.45
    flatness_rms: float = 0.05
    dead_pixel_fraction: float = 0.0
    frame_rate: float = 240.0
    settling_time: float = 4e-3
    seed: int = 0


@register("device", "transmissive_lc", "transmissive_slm")
@dataclass
class TransmissiveLCSLM(ModulatorDevice):
    """Transmissive LC phase SLM: single-pass phase, larger pitch, more loss."""

    name: str = "transmissive_lc"
    pitch: float = 32e-6
    reflective: bool = False
    fill_factor: float = 0.58  # transistor + bus lines block a lot
    efficiency: float = 0.55
    modulation: str = "phase"
    phase_bits: int | None = 8
    max_phase: float = 2 * np.pi
    design_wavelength: float = 1550e-9
    crosstalk: float = 0.25
    flatness_rms: float = 0.08
    dead_pixel_fraction: float = 1e-4
    frame_rate: float = 60.0
    settling_time: float = 15e-3
    seed: int = 0


@register("device", "amplitude_slm", "lc_amplitude")
@dataclass
class AmplitudeSLM(ModulatorDevice):
    """LC amplitude modulator (crossed polarisers).  Phase information is lost."""

    name: str = "amplitude_slm"
    pitch: float = 8e-6
    reflective: bool = False
    fill_factor: float = 0.9
    efficiency: float = 0.35
    modulation: str = "amplitude"
    phase_bits: int | None = 8
    max_phase: float = 0.0
    design_wavelength: float = 1550e-9
    crosstalk: float = 0.2
    flatness_rms: float = 0.0
    dead_pixel_fraction: float = 0.0
    frame_rate: float = 120.0
    settling_time: float = 8e-3
    seed: int = 0


@register("device", "ipms_piston", "mems_piston", "ipms")
@dataclass
class PistonMicroMirrorArray(ModulatorDevice):
    """MEMS piston micro-mirror phase array (Fraunhofer IPMS style).

    Each mirror translates along the optical axis by up to ``stroke`` metres.  In
    reflection the phase is ``4*pi*h/lambda``, so a quarter-wave of mechanical
    stroke buys a full 2*pi of phase.  Frame rates are orders of magnitude above
    LC, which is what makes a fast optical feedback loop possible; the trade is
    coarse pitch and a modest fill factor.
    """

    name: str = "ipms_piston"
    pitch: float = 16e-6
    stroke: float = 800e-9  # mechanical, metres -> a full 2*pi at 1550 nm
    reflective: bool = True
    fill_factor: float = 0.9
    efficiency: float = 0.9  # aluminium mirror reflectance
    modulation: str = "phase"
    phase_bits: int | None = 10
    design_wavelength: float = 1550e-9
    crosstalk: float = 0.05  # mechanical coupling only
    flatness_rms: float = 0.03
    dead_pixel_fraction: float = 1e-4
    frame_rate: float = 20e3
    settling_time: float = 20e-6
    seed: int = 0

    @property
    def max_phase(self) -> float:  # type: ignore[override]
        n = 2.0 if self.reflective else 1.0
        return 2 * np.pi * n * self.stroke / self.design_wavelength

    @max_phase.setter
    def max_phase(self, value: float) -> None:  # allow dataclass default handling
        n = 2.0 if self.reflective else 1.0
        self.stroke = value * self.design_wavelength / (2 * np.pi * n)


@register("device", "tilt_mirror", "analog_tilt")
@dataclass
class TiltMicroMirrorArray(ModulatorDevice):
    """Analog tilting micro-mirror array.

    Each mirror is a small flat facet with a controllable tilt, i.e. a
    *piecewise-linear* phase.  Steering is continuous but a single mirror cannot
    hold a general hologram, so the phase is projected onto a per-pixel ramp.
    """

    name: str = "tilt_mirror"
    pitch: float = 40e-6
    max_tilt: float = 8e-3  # mechanical half-angle [rad]
    reflective: bool = True
    fill_factor: float = 0.92
    efficiency: float = 0.9
    modulation: str = "phase"
    phase_bits: int | None = 12
    max_phase: float = 2 * np.pi
    design_wavelength: float = 1550e-9
    crosstalk: float = 0.02
    flatness_rms: float = 0.02
    dead_pixel_fraction: float = 0.0
    frame_rate: float = 5e3
    settling_time: float = 100e-6
    seed: int = 0

    def render(self, target, grid: Grid, wavelength: float):
        """Project the request onto a per-mirror linear ramp, then degrade as usual.

        The mean local phase gradient over each mirror is measured from the
        complex field (so it is wrap-safe), clamped to the mechanical tilt limit,
        and re-expanded as an exact ramp across the mirror.
        """
        t = eng.asarray(target, dtype=eng.COMPLEX)
        by = max(1, int(round(self.pitch / grid.dy)))
        bx = max(1, int(round(self.pitch / grid.dx)))
        if by > 1 or bx > 1:
            t = self._fit_ramps(t, grid, by, bx, wavelength)
        t = self.apply_crosstalk(t, grid)
        t = t * self.aperture_mask(grid)
        t = t * self.static_errors(grid, wavelength)
        return t * np.sqrt(max(self.efficiency, 0.0))

    def _fit_ramps(self, t, grid: Grid, by: int, bx: int, wavelength: float):
        # wrap-safe local phase differences
        dx_ph = eng.xp.zeros(t.shape, dtype=eng.REAL)
        dy_ph = eng.xp.zeros(t.shape, dtype=eng.REAL)
        dx_ph[:, :-1] = eng.xp.angle(t[:, 1:] * eng.xp.conj(t[:, :-1]))
        dy_ph[:-1, :] = eng.xp.angle(t[1:, :] * eng.xp.conj(t[:-1, :]))
        gx = _block_average(dx_ph, by, bx) / grid.dx  # rad/m
        gy = _block_average(dy_ph, by, bx) / grid.dy
        k = 2 * np.pi / wavelength
        gmax = k * (2.0 if self.reflective else 1.0) * np.sin(self.max_tilt)
        gx = eng.xp.clip(gx, -gmax, gmax)
        gy = eng.xp.clip(gy, -gmax, gmax)
        X, Y = grid.meshgrid()
        xl = (eng.xp.mod(X / self.pitch + 0.5, 1.0) - 0.5) * self.pitch
        yl = (eng.xp.mod(Y / self.pitch + 0.5, 1.0) - 0.5) * self.pitch
        ramp = eng.xp.exp(1j * (gx * xl + gy * yl))
        # de-ramp before averaging so the per-mirror piston is not washed out
        return _block_average(t * eng.xp.conj(ramp), by, bx) * ramp

    def max_deflection(self, wavelength: float) -> float:
        n = 2.0 if self.reflective else 1.0
        return float(n * self.max_tilt)


@register("device", "dmd", "micromirror", "binary_mirror")
@dataclass
class DMD(ModulatorDevice):
    """Binary tilting micro-mirror array (DLP-style).

    ``encoding`` decides how a complex hologram is squeezed onto a device that
    only has two states per mirror:

    ``amplitude``  threshold |t| -- simple, phase discarded
    ``lee``        Lee binary-carrier holography: the complex field is encoded on a
                   spatial carrier, so phase survives at the cost of a carrier
                   offset and ~10% efficiency
    ``error_diffusion``  halftone the amplitude with Floyd-Steinberg diffusion,
                   which pushes the quantisation noise to high angles

    The mirror tilt itself acts as a blazed facet: ``tilt_angle`` sets where the
    "on" order sits, and the simulator adds that facet phase so the DMD blazing
    condition appears naturally.
    """

    name: str = "dmd"
    pitch: float = 13.7e-6
    tilt_angle: float = 12 * np.pi / 180
    encoding: str = "amplitude"
    carrier_period_pixels: float = 4.0
    include_facet_tilt: bool = True
    reflective: bool = True
    fill_factor: float = 0.92
    efficiency: float = 0.88
    modulation: str = "binary_amplitude"
    phase_bits: int | None = None
    max_phase: float = 0.0
    design_wavelength: float = 1550e-9
    crosstalk: float = 0.0
    flatness_rms: float = 0.02
    dead_pixel_fraction: float = 1e-5
    frame_rate: float = 22e3
    settling_time: float = 15e-6
    seed: int = 0

    def represent(self, t, wavelength: float):
        a = eng.xp.abs(t)
        amax = float(eng.to_numpy(eng.xp.max(a)))
        if amax <= 0:
            return eng.xp.zeros_like(t)
        # a passive element cannot ask for more than unit transmittance; only
        # rescale when the request exceeds it, so Lee encoding sees true amplitudes
        if amax > 1.0:
            a = a / amax
        if self.encoding == "amplitude":
            b = (a >= 0.5).astype(eng.COMPLEX)
        elif self.encoding == "lee":
            b = self._lee(t, a)
        elif self.encoding == "error_diffusion":
            b = eng.asarray(_floyd_steinberg(eng.to_numpy(a)), dtype=eng.COMPLEX)
        else:
            raise ValueError(f"unknown DMD encoding {self.encoding!r}")
        return b

    def _lee(self, t, a):
        """Lee binary hologram: threshold cos(carrier - phase) against amplitude."""
        ny, nx = a.shape
        ix = eng.xp.arange(nx).reshape(1, -1)
        iy = eng.xp.arange(ny).reshape(-1, 1)
        carrier = 2 * np.pi * (ix + 0 * iy) / self.carrier_period_pixels
        ph = eng.xp.angle(t)
        return (eng.xp.cos(carrier - ph) >= eng.xp.cos(np.pi * eng.xp.clip(a, 0, 1))).astype(
            eng.COMPLEX
        )

    def aperture_mask(self, grid: Grid):
        m = super().aperture_mask(grid)
        if not self.include_facet_tilt:
            return m
        # the "on" mirror is a tilted facet: add its local blaze phase
        key = ("facet", grid.shape, grid.dx, self.pitch, self.tilt_angle)
        cached = getattr(self, "_facet_cache", None)
        if cached is None or cached[0] != key:
            X, _ = grid.meshgrid()
            xl = (eng.xp.mod(X / self.pitch + 0.5, 1.0) - 0.5) * self.pitch
            k = 2 * np.pi / self.design_wavelength
            facet = eng.xp.exp(1j * 2 * k * np.sin(self.tilt_angle) * xl)
            self._facet_cache = (key, facet)
        return m * self._facet_cache[1]

    def blaze_order(self, wavelength: float) -> float:
        """Diffraction order that coincides with the mirror blaze direction."""
        return float(2 * np.sin(self.tilt_angle) * self.pitch / wavelength)


def _floyd_steinberg(a: np.ndarray) -> np.ndarray:
    """Binary halftone with Floyd-Steinberg error diffusion."""
    img = np.array(a, dtype=float, copy=True)
    ny, nx = img.shape
    for y in range(ny):
        for x in range(nx):
            old = img[y, x]
            new = 1.0 if old >= 0.5 else 0.0
            img[y, x] = new
            err = old - new
            if x + 1 < nx:
                img[y, x + 1] += err * 7 / 16
            if y + 1 < ny:
                if x > 0:
                    img[y + 1, x - 1] += err * 3 / 16
                img[y + 1, x] += err * 5 / 16
                if x + 1 < nx:
                    img[y + 1, x + 1] += err * 1 / 16
    return np.clip(img, 0, 1)


def as_device(spec) -> ModulatorDevice:
    from ..registry import build

    if spec is None:
        return LCoSPhaseSLM()
    if isinstance(spec, ModulatorDevice):
        return spec
    return build(spec, "device")
