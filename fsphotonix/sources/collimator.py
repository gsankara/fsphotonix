"""Collimator / microlens arrays on the illumination side.

Real machines do not shine bare fibres at an SLM: each emitter sits behind a
lenslet that sets how large an area of the modulator it illuminates.  That area is
a *tunable* design parameter -- move the emitter off the lenslet's focal plane and
the footprint grows or shrinks -- and it is one of the dominant trade-offs in the
whole architecture:

* too small, and only part of the region's holographic aperture is lit, so the
  sub-gratings lose efficiency and the spots on the screen broaden;
* too large, and light spills across the guard gap onto the neighbouring region,
  where it is diffracted by *that* input's hologram.  It arrives at the wrong
  screen location as **illumination cross-talk**, indistinguishable at the detector
  from a genuine signal.

Two models are provided:

:class:`Collimator`
    the Gaussian-optics description of one emitter + lenslet.  It answers "what
    footprint do I get?" and, inverted, "what defocus do I need for the footprint I
    want?", including the truncation loss when the beam overfills the lenslet.
:class:`MicrolensArray`
    a wave-optics propagator that applies the actual per-lenslet phase and
    aperture, so lenslet diffraction, dead space between lenslets and inter-lenslet
    leakage appear in the field itself.

:class:`CollimatedArray` is the source built from the first of these -- the usual
starting point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .. import engine as eng
from ..grid import Grid
from ..propagation.base import Propagator
from ..registry import register
from ..waveform import Impaired, as_waveform
from .base import Emitter, EmitterBank
from .layout import grid_positions
from .profiles import BeamProfile, GaussianBeam


# --------------------------------------------------------------------------- #
# Gaussian-optics collimator
# --------------------------------------------------------------------------- #
@register("profile", "collimator")
@dataclass
class Collimator:
    """One emitter behind one lenslet, in the Gaussian beam approximation.

    Parameters
    ----------
    mfd : emitter mode-field diameter [m] (fibre facet, VCSEL aperture, ...).
    focal_length : lenslet focal length [m].
    object_distance : emitter-to-lens distance [m].  ``None`` means "at the focal
        plane", i.e. nominal collimation.
    defocus : shorthand for ``object_distance = focal_length + defocus``.  This is
        the tuning knob for the illuminated area.
    clear_aperture : lenslet clear aperture diameter [m]; the beam is truncated
        there, which costs power and adds diffraction rings.
    """

    mfd: float = 10.4e-6
    focal_length: float = 1.5e-3
    object_distance: float | None = None
    defocus: float = 0.0
    clear_aperture: float = 250e-6

    # ---- basic quantities -------------------------------------------------- #
    @property
    def d(self) -> float:
        return (self.focal_length + self.defocus if self.object_distance is None
                else self.object_distance)

    def source_waist(self) -> float:
        return self.mfd / 2

    def rayleigh(self, wavelength: float) -> float:
        w0 = self.source_waist()
        return np.pi * w0**2 / wavelength

    def magnification(self, wavelength: float) -> float:
        """Gaussian-beam magnification of the lenslet."""
        zr = self.rayleigh(wavelength)
        return float(self.focal_length / np.sqrt((self.d - self.focal_length) ** 2 + zr**2))

    def output_waist(self, wavelength: float) -> float:
        """1/e^2 radius of the waist formed behind the lenslet."""
        return float(self.magnification(wavelength) * self.source_waist())

    def waist_position(self, wavelength: float) -> float:
        """Distance from the lenslet to that waist [m] (negative = virtual)."""
        m = self.magnification(wavelength)
        return float(self.focal_length + m**2 * (self.d - self.focal_length))

    def beam_radius_at_lens(self, wavelength: float) -> float:
        zr = self.rayleigh(wavelength)
        return float(self.source_waist() * np.sqrt(1 + (self.d / zr) ** 2))

    # ---- what the modulator sees ------------------------------------------- #
    def radius_at(self, distance: float, wavelength: float) -> float:
        """1/e^2 beam radius a given distance behind the lenslet."""
        w0 = self.output_waist(wavelength)
        zr = np.pi * w0**2 / wavelength
        return float(w0 * np.sqrt(1 + ((distance - self.waist_position(wavelength)) / zr) ** 2))

    def illumination_diameter(self, distance: float, wavelength: float) -> float:
        """1/e^2 illuminated diameter on the modulator plane."""
        return 2 * self.radius_at(distance, wavelength)

    def divergence(self, wavelength: float) -> float:
        """Far-field half angle of the collimated beam [rad]."""
        return float(wavelength / (np.pi * self.output_waist(wavelength)))

    def truncation_transmission(self, wavelength: float) -> float:
        """Power fraction passing the lenslet clear aperture."""
        a = self.clear_aperture / 2
        w = self.beam_radius_at_lens(wavelength)
        if w <= 0:
            return 1.0
        return float(1.0 - np.exp(-2 * (a / w) ** 2))

    # ---- inverse design: pick the defocus for a wanted footprint ----------- #
    def footprint_for(self, defocus: float, distance: float, wavelength: float) -> float:
        """Illuminated diameter at ``distance`` for a trial defocus."""
        from dataclasses import replace

        return replace(self, defocus=float(defocus), object_distance=None
                       ).illumination_diameter(distance, wavelength)

    def solve_defocus(self, target_diameter: float, distance: float,
                      wavelength: float, branch: str = "diverging") -> float:
        """Defocus [m] that makes the footprint ``target_diameter`` at ``distance``.

        Sign convention: a **negative** defocus moves the emitter *inside* the focal
        length, so the lenslet forms a virtual waist behind itself and the beam is
        already diverging when it reaches the modulator -- the footprint then grows
        monotonically with ``|defocus|``.  A **positive** defocus pushes the real
        waist downstream, so at a fixed short throw the footprint first *shrinks*
        (the beam is still converging) and only grows again once the waist has moved
        past the modulator.

        ``branch='diverging'`` therefore searches negative defocus and is the
        well-behaved way to open up the illuminated area; ``'converging'`` searches
        the positive side, which reaches the same footprint with the opposite
        wavefront curvature at the panel -- a difference the holograms see, since
        they are designed for a given incident phase.  ``'auto'`` (the default used
        by :class:`CollimatedArray`) picks whichever side can reach the request: the
        nominally collimated footprint is a *lower* bound on the diverging side, so
        anything smaller has to come from the converging branch.
        """
        if branch == "auto":
            nominal = self.footprint_for(0.0, distance, wavelength)
            branch = "diverging" if target_diameter >= nominal else "converging"
        sign = -1.0 if branch == "diverging" else 1.0
        deltas = np.concatenate([[0.0], np.geomspace(1e-9, 200 * self.focal_length, 600)])
        vals = np.array([self.footprint_for(sign * d, distance, wavelength)
                         for d in deltas])

        # first interval that brackets the target (the curve is not monotonic on
        # the converging branch, so scan rather than assume)
        idx = None
        for i in range(len(deltas) - 1):
            a, b = vals[i], vals[i + 1]
            if (a - target_diameter) * (b - target_diameter) <= 0:
                idx = i
                break
        if idx is None:
            return float(sign * deltas[int(np.argmin(np.abs(vals - target_diameter)))])

        lo, hi = deltas[idx], deltas[idx + 1]
        rising = vals[idx + 1] >= vals[idx]
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            v = self.footprint_for(sign * mid, distance, wavelength)
            if (v < target_diameter) == rising:
                lo = mid
            else:
                hi = mid
        return float(sign * 0.5 * (lo + hi))

    def profile(self, distance: float, wavelength: float, x0: float = 0.0,
                y0: float = 0.0, theta_x: float = 0.0,
                theta_y: float = 0.0) -> GaussianBeam:
        """The :class:`~fsphotonix.sources.profiles.GaussianBeam` seen at the modulator."""
        return GaussianBeam(waist=self.output_waist(wavelength),
                            z_offset=distance - self.waist_position(wavelength),
                            x0=x0, y0=y0, theta_x=theta_x, theta_y=theta_y)

    def report(self, distance: float, wavelength: float) -> dict:
        return {
            "object_distance_um": self.d * 1e6,
            "defocus_um": (self.d - self.focal_length) * 1e6,
            "magnification": self.magnification(wavelength),
            "output_waist_um": self.output_waist(wavelength) * 1e6,
            "waist_position_um": self.waist_position(wavelength) * 1e6,
            "illumination_diameter_um": self.illumination_diameter(distance, wavelength) * 1e6,
            "divergence_mrad": self.divergence(wavelength) * 1e3,
            "beam_at_lens_um": self.beam_radius_at_lens(wavelength) * 2e6,
            "truncation_transmission": self.truncation_transmission(wavelength),
            "overfills_lenslet": self.beam_radius_at_lens(wavelength) * 2
            > self.clear_aperture,
        }


@register("source", "collimated_array", "collimator_array")
class CollimatedArray(EmitterBank):
    """Emitter array behind a collimator/lenslet array.

    The footprint on the modulator is set either directly
    (``illumination_diameter``, and the defocus is solved for) or by giving the
    ``collimator`` an explicit ``defocus``.  ``stray_fraction`` adds the broad,
    weakly-focused background that a real lenslet array leaks between its
    elements -- the other route by which one input's light reaches another input's
    hologram.

    ``distance`` is the lenslet-to-modulator separation; it is a pure design
    quantity here (the emitted field is evaluated *at the modulator plane*), so
    place this source at the same ``z`` as the modulator stage it feeds.
    """

    def __init__(
        self,
        n: int = 8,
        ny: int = 1,
        pitch: float = 400e-6,
        pitch_y: float | None = None,
        positions=None,
        distance: float = 5e-3,
        illumination_diameter: float | None = None,
        collimator: Collimator | dict | None = None,
        wavelength: float | Sequence[float] = 1550e-9,
        power: float | Sequence[float] = 1e-3,
        waveform=None,
        pointing=None,
        coherent: bool = False,
        stray_fraction: float = 0.0,
        stray_diameter_factor: float = 6.0,
        linewidth: float = 0.0,
        rin_dbc_hz: float = -300.0,
        branch: str = "auto",
        name: str = "collimated",
        seed: int = 0,
    ):
        if positions is None:
            positions = grid_positions(n, ny, pitch, pitch_y)
        positions = np.asarray(positions, dtype=float).reshape(-1, 2)
        count = len(positions)
        lam = np.broadcast_to(np.atleast_1d(np.asarray(wavelength, float)), (count,))
        pw = np.broadcast_to(np.atleast_1d(np.asarray(power, float)), (count,))
        point = (np.zeros((count, 2)) if pointing is None
                 else np.asarray(pointing, float).reshape(-1, 2))

        col = collimator if isinstance(collimator, Collimator) else Collimator(
            **(collimator or {"clear_aperture": 0.95 * pitch}))
        if illumination_diameter is not None:
            col = _with_defocus(col, col.solve_defocus(illumination_diameter, distance,
                                                       float(lam[0]), branch=branch))
        self.collimator = col
        self.distance = distance
        self.footprint = col.illumination_diameter(distance, float(lam[0]))
        self.truncation = col.truncation_transmission(float(lam[0]))

        wf_list = waveform if isinstance(waveform, (list, tuple)) else [waveform] * count
        emitters: list[Emitter] = []
        for i in range(count):
            wf = as_waveform(wf_list[i])
            if linewidth > 0 or rin_dbc_hz > -300:
                wf = Impaired(inner=wf, linewidth=linewidth, rin_dbc_hz=rin_dbc_hz,
                              seed=seed + i)
            coh = f"{name}:coherent" if coherent else f"{name}:{i}"
            p_main = float(pw[i]) * self.truncation * (1.0 - stray_fraction)
            emitters.append(Emitter(
                wavelength=float(lam[i]), power=p_main,
                profile=col.profile(distance, float(lam[i]),
                                    x0=positions[i][0], y0=positions[i][1],
                                    theta_x=point[i][0], theta_y=point[i][1]),
                waveform=wf, coherence=coh, label=f"{name}[{i}]",
                meta={"index": i, "x": positions[i][0], "y": positions[i][1],
                      "footprint": self.footprint},
            ))
            if stray_fraction > 0:
                emitters.append(Emitter(
                    wavelength=float(lam[i]),
                    power=float(pw[i]) * self.truncation * stray_fraction,
                    profile=GaussianBeam(
                        waist=stray_diameter_factor * self.footprint / 2,
                        x0=positions[i][0], y0=positions[i][1],
                        theta_x=point[i][0], theta_y=point[i][1]),
                    waveform=wf, coherence=coh, label=f"{name}[{i}].stray",
                    meta={"index": i, "stray": True},
                ))
        super().__init__(emitters, name=name)
        self.n_inputs = count

    # ---- reporting --------------------------------------------------------- #
    def set_drive(self, values) -> None:
        """Drive per *input*, applied to that input's main and stray beams."""
        v = np.atleast_1d(np.asarray(values, dtype=float)).ravel()
        if v.size == self.n_inputs and len(self.emitters) != self.n_inputs:
            for e in self.emitters:
                e.drive = float(max(v[e.meta["index"]], 0.0))
            return
        super().set_drive(values)

    def get_drive(self):
        if len(self.emitters) == self.n_inputs:
            return super().get_drive()
        out = np.zeros(self.n_inputs)
        for e in self.emitters:
            out[e.meta["index"]] = e.drive
        return out

    @property
    def n_emitters(self) -> int:
        return self.n_inputs if len(self.emitters) != self.n_inputs else len(self.emitters)

    def report(self, wavelength: float | None = None) -> dict:
        lam = wavelength or self.emitters[0].wavelength
        r = self.collimator.report(self.distance, lam)
        r["footprint_um"] = self.footprint * 1e6
        r["inputs"] = self.n_inputs
        return r


def _with_defocus(col: Collimator, defocus: float) -> Collimator:
    from dataclasses import replace

    return replace(col, defocus=defocus, object_distance=None)


# --------------------------------------------------------------------------- #
# Wave-optics lenslet array
# --------------------------------------------------------------------------- #
@register("propagator", "microlens_array", "lenslet_array", "collimator_plane")
@dataclass
class MicrolensArray(Propagator):
    """Per-lenslet phase and aperture applied to the field (a zero-distance element).

    Use this when the lenslet diffraction, its dead space, or the exact clipped
    wavefront matters -- for instance when asking how much of the cross-talk is
    lenslet leakage rather than Gaussian overspill.  ``leak`` is the amplitude
    transmission of the dead space between lenslets, i.e. unlensed straight-through
    light.
    """

    focal_length: float = 1.5e-3
    pitch: float = 400e-6
    pitch_y: float | None = None
    clear_aperture: float | None = None  # default: 0.95 * pitch
    shape: str = "circle"  # circle | rect
    leak: float = 0.0
    positions: object = None  # explicit lenslet centres, else a regular lattice
    z: float = 0.0
    out_grid: Grid | None = None

    def _mask_and_phase(self, grid: Grid, wavelength: float):
        key = (grid.shape, grid.dx, grid.dy, grid.x0, grid.y0, wavelength,
               self.focal_length, self.pitch, self.pitch_y, self.clear_aperture,
               self.shape, self.leak, id(self.positions))
        cached = getattr(self, "_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        k = 2 * np.pi / wavelength
        a = (self.clear_aperture if self.clear_aperture is not None
             else 0.95 * self.pitch)
        py = self.pitch_y or self.pitch
        X, Y = grid.meshgrid()
        if self.positions is None:
            xl = (eng.xp.mod(X / self.pitch + 0.5, 1.0) - 0.5) * self.pitch
            yl = (eng.xp.mod(Y / py + 0.5, 1.0) - 0.5) * py
            if self.shape == "circle":
                inside = (eng.xp.sqrt(xl**2 + yl**2) <= a / 2)
            else:
                inside = (eng.xp.abs(xl) <= a / 2) & (eng.xp.abs(yl) <= a / 2)
            t = eng.xp.where(inside, eng.xp.exp(-1j * k * (xl**2 + yl**2)
                                                / (2 * self.focal_length)),
                             eng.asarray(self.leak, dtype=eng.COMPLEX))
        else:
            pos = np.asarray(self.positions, dtype=float).reshape(-1, 2)
            t = eng.xp.full(grid.shape, self.leak, dtype=eng.COMPLEX)
            for px, py_ in pos:
                dx, dy = X - px, Y - py_
                if self.shape == "circle":
                    inside = eng.xp.sqrt(dx**2 + dy**2) <= a / 2
                else:
                    inside = (eng.xp.abs(dx) <= a / 2) & (eng.xp.abs(dy) <= a / 2)
                lens = eng.xp.exp(-1j * k * (dx**2 + dy**2) / (2 * self.focal_length))
                t = eng.xp.where(inside, lens, t)
        self._cache = (key, t)
        return t

    def apply(self, u, in_grid: Grid, out_grid: Grid, wavelength: float, z: float):
        return u * self._mask_and_phase(in_grid, wavelength)
