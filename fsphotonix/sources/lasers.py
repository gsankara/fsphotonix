"""Coherent light sources: single lasers, coherent arrays and WDM combs."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..registry import register
from ..units import itu_grid
from ..waveform import Constant, Impaired, as_waveform
from .base import Emitter, EmitterBank
from .layout import grid_positions
from .profiles import GaussianBeam, as_profile


@register("source", "coherent_laser", "laser")
class CoherentLaser(EmitterBank):
    """A single-frequency laser illuminating the plane.

    ``split`` > 1 models one laser split into several mutually **coherent**
    beamlets (a fibre splitter feeding a lenslet array, or a beam-splitter tree):
    all beamlets share one coherence tag so they interfere with each other.
    """

    def __init__(
        self,
        wavelength: float = 1550e-9,
        power: float = 10e-3,
        profile=None,
        waveform=None,
        linewidth: float = 0.0,
        rin_dbc_hz: float | None = None,
        split: int = 1,
        positions: Sequence[Sequence[float]] | None = None,
        pitch: float = 1e-3,
        weights: Sequence[float] | None = None,
        pointing: Sequence[Sequence[float]] | None = None,
        name: str = "laser",
        seed: int = 0,
    ):
        base_profile = as_profile(profile if profile is not None else GaussianBeam(waist=0.5e-3))
        wf = as_waveform(waveform)
        if linewidth > 0 or rin_dbc_hz is not None:
            wf = Impaired(
                inner=wf,
                linewidth=linewidth,
                rin_dbc_hz=-300.0 if rin_dbc_hz is None else rin_dbc_hz,
                seed=seed,
            )
        if positions is None:
            positions = grid_positions(split, 1, pitch) if split > 1 else [(0.0, 0.0)]
        positions = np.asarray(positions, dtype=float).reshape(-1, 2)
        n = len(positions)
        w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
        w = w / w.sum() if w.sum() > 0 else w
        point = np.zeros((n, 2)) if pointing is None else np.asarray(pointing, float).reshape(-1, 2)

        coh = f"{name}:coherent"
        emitters = []
        for i, (pos, wi) in enumerate(zip(positions, w)):
            prof = _relocate(base_profile, pos[0], pos[1], point[i][0], point[i][1])
            emitters.append(
                Emitter(
                    wavelength=wavelength,
                    power=power * float(wi),
                    profile=prof,
                    waveform=wf,
                    coherence=coh,
                    label=f"{name}[{i}]",
                )
            )
        super().__init__(emitters, name=name)


@register("source", "laser_array", "incoherent_array")
class LaserArray(EmitterBank):
    """An array of independent lasers -- mutually **incoherent** emitters.

    This is the right model for an array of separately packaged DFBs/VCSELs and
    for encoding a vector on source intensities (each emitter is one vector
    element, driven via :meth:`~fsphotonix.sources.base.EmitterBank.set_drive`).
    """

    def __init__(
        self,
        n: int = 8,
        ny: int = 1,
        pitch: float = 1e-3,
        pitch_y: float | None = None,
        wavelength: float | Sequence[float] = 1550e-9,
        power: float | Sequence[float] = 1e-3,
        profile=None,
        waveform=None,
        positions=None,
        pointing=None,
        linewidth: float = 1e6,
        rin_dbc_hz: float = -145.0,
        name: str = "laser_array",
        seed: int = 0,
    ):
        base_profile = as_profile(profile if profile is not None else GaussianBeam(waist=0.3e-3))
        if positions is None:
            positions = grid_positions(n, ny, pitch, pitch_y)
        positions = np.asarray(positions, dtype=float).reshape(-1, 2)
        count = len(positions)
        lam = np.broadcast_to(np.atleast_1d(np.asarray(wavelength, float)), (count,))
        pw = np.broadcast_to(np.atleast_1d(np.asarray(power, float)), (count,))
        point = np.zeros((count, 2)) if pointing is None else np.asarray(pointing, float).reshape(-1, 2)
        wf_list = waveform if isinstance(waveform, (list, tuple)) else [waveform] * count

        emitters = []
        for i in range(count):
            wf = as_waveform(wf_list[i])
            if linewidth > 0 or rin_dbc_hz > -300:
                wf = Impaired(inner=wf, linewidth=linewidth, rin_dbc_hz=rin_dbc_hz, seed=seed + i)
            prof = _relocate(base_profile, positions[i][0], positions[i][1],
                             point[i][0], point[i][1])
            emitters.append(
                Emitter(
                    wavelength=float(lam[i]),
                    power=float(pw[i]),
                    profile=prof,
                    waveform=wf,
                    coherence=f"{name}:{i}",  # independent -> incoherent
                    label=f"{name}[{i}]",
                    meta={"index": i, "x": positions[i][0], "y": positions[i][1]},
                )
            )
        super().__init__(emitters, name=name)


@register("source", "wdm_comb", "comb")
class WDMComb(EmitterBank):
    """Frequency comb / DWDM bank on the ITU grid.

    ``mutually_coherent=True`` (a real comb from one mode-locked laser) puts all
    lines in one coherence group, so cross-line beat notes exist and a detector
    with enough bandwidth will see them.  ``False`` models independent DWDM
    lasers on grid.
    """

    def __init__(
        self,
        n_lines: int = 8,
        spacing: float = 100e9,
        anchor_thz: float = 193.1,
        first_channel: int = 0,
        power: float = 1e-3,
        profile=None,
        waveform=None,
        mutually_coherent: bool = True,
        positions=None,
        name: str = "comb",
    ):
        base_profile = as_profile(profile if profile is not None else GaussianBeam(waist=0.5e-3))
        pos = (np.zeros((n_lines, 2)) if positions is None
               else np.asarray(positions, float).reshape(-1, 2))
        wf_list = waveform if isinstance(waveform, (list, tuple)) else [waveform] * n_lines
        coh = f"{name}:comb"
        emitters = []
        for i in range(n_lines):
            lam = itu_grid(first_channel + i, spacing=spacing, anchor_thz=anchor_thz)
            emitters.append(
                Emitter(
                    wavelength=lam,
                    power=power,
                    profile=_relocate(base_profile, pos[i][0], pos[i][1], 0.0, 0.0),
                    waveform=as_waveform(wf_list[i]),
                    coherence=coh if mutually_coherent else f"{name}:{i}",
                    label=f"{name}:ch{first_channel + i}",
                    meta={"channel": first_channel + i, "index": i},
                )
            )
        super().__init__(emitters, name=name)
        self.spacing = spacing


def _relocate(profile, x0: float, y0: float, theta_x: float, theta_y: float):
    """Copy a profile, *offsetting* its position and pointing by the emitter's.

    The profile's own ``x0``/``y0``/tilt stay meaningful (they describe the beam
    relative to its emitter), and the emitter's place in the array is added on top.
    """
    from dataclasses import is_dataclass, replace

    nx = float(profile.x0) + float(x0)
    ny = float(profile.y0) + float(y0)
    tx = float(profile.theta_x) + float(theta_x)
    ty = float(profile.theta_y) + float(theta_y)
    if is_dataclass(profile):
        return replace(profile, x0=nx, y0=ny, theta_x=tx, theta_y=ty)
    import copy

    p = copy.copy(profile)
    p.x0, p.y0, p.theta_x, p.theta_y = nx, ny, tx, ty
    return p
