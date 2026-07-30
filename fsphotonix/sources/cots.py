"""Commercial off-the-shelf transceiver sources.

The interesting regime for a cheap large-scale machine is a rack of ordinary
datacom optics rather than a lab laser: many emitters that are *mutually
incoherent* but sit inside one 100 GHz-wide slot (uncooled DFB / EML / VCSEL part
tolerance), each already carrying a real line-rate digital waveform.

Two models are provided:

* :class:`TransceiverBank` -- N transceivers whose centre frequencies scatter
  inside a specified span around one nominal channel.  Different tags => no
  interference between them, only intensity addition.  This is the
  "same 100 GHz range, not coherent" case.
* :class:`CoarseWDMBank` -- N transceivers on genuinely different wavelengths
  (CWDM 20 nm grid or a custom list), where grating dispersion actually walks the
  spots across the screen.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..registry import register
from ..units import C0
from ..waveform import NRZ, Impaired, as_waveform
from .base import Emitter, EmitterBank
from .layout import grid_positions
from .profiles import GaussianBeam, as_profile

#: CWDM grid (ITU-T G.694.2), nm
CWDM_NM = [1271, 1291, 1311, 1331, 1351, 1371, 1391, 1411, 1431, 1451,
           1471, 1491, 1511, 1531, 1551, 1571, 1591, 1611]


def _transceiver_waveform(i: int, baud: float, waveform, modulation: str,
                          extinction_ratio_db: float, prbs_bits: int,
                          linewidth: float, rin_dbc_hz: float, seed: int):
    from ..waveform import PAM

    if waveform is not None:
        wf = as_waveform(waveform)
    elif modulation.lower() in ("nrz", "ook"):
        wf = NRZ.prbs(prbs_bits, seed=seed + i, baud=baud,
                      extinction_ratio_db=extinction_ratio_db)
    elif modulation.lower() in ("pam4", "pam"):
        rng = np.random.default_rng(seed + i)
        wf = PAM(symbols=rng.integers(0, 4, prbs_bits), levels=4, baud=baud)
    else:
        raise ValueError(f"unknown modulation {modulation!r}")
    return Impaired(inner=wf, linewidth=linewidth, rin_dbc_hz=rin_dbc_hz, seed=seed + i,
                    bandwidth=baud)


@register("source", "cots_transceiver", "transceiver_bank")
class TransceiverBank(EmitterBank):
    """A bank of nominally identical COTS transceivers.

    Parameters
    ----------
    center_wavelength : nominal wavelength [m] (e.g. 1310 nm O-band).
    freq_span : full width [Hz] over which the individual centre frequencies are
        distributed -- default 100 GHz, i.e. all parts land in one DWDM slot but
        are mutually incoherent.
    spread : ``'uniform'`` spreads the parts deterministically across the span
        (worst-case dispersion study), ``'random'`` draws them, ``'identical'``
        puts them all exactly on the nominal frequency (still incoherent).
    """

    def __init__(
        self,
        n: int = 8,
        ny: int = 1,
        center_wavelength: float = 1310e-9,
        freq_span: float = 100e9,
        spread: str = "uniform",
        power: float | Sequence[float] = 1e-3,
        baud: float = 25e9,
        modulation: str = "nrz",
        waveform=None,
        extinction_ratio_db: float = 8.0,
        prbs_bits: int = 63,
        linewidth: float = 5e6,
        rin_dbc_hz: float = -140.0,
        pitch: float = 250e-6,
        pitch_y: float | None = None,
        positions=None,
        pointing=None,
        profile=None,
        mfd: float = 9.2e-6,
        collimated_waist: float | None = 0.3e-3,
        name: str = "cots",
        seed: int = 0,
    ):
        if profile is not None:
            base_profile = as_profile(profile)
        elif collimated_waist:
            base_profile = GaussianBeam(waist=collimated_waist)
        else:
            base_profile = GaussianBeam.from_fiber(mfd)

        if positions is None:
            positions = grid_positions(n, ny, pitch, pitch_y)
        positions = np.asarray(positions, float).reshape(-1, 2)
        count = len(positions)
        point = np.zeros((count, 2)) if pointing is None else np.asarray(pointing, float).reshape(-1, 2)
        pw = np.broadcast_to(np.atleast_1d(np.asarray(power, float)), (count,))

        f0 = C0 / center_wavelength
        rng = np.random.default_rng(seed)
        if spread == "uniform":
            offs = (np.linspace(-0.5, 0.5, count) if count > 1 else np.array([0.0])) * freq_span
        elif spread == "random":
            offs = rng.uniform(-0.5, 0.5, count) * freq_span
        elif spread == "identical":
            offs = np.zeros(count)
        else:
            raise ValueError(f"unknown spread {spread!r}")

        from .lasers import _relocate

        emitters = []
        for i in range(count):
            lam = C0 / (f0 + offs[i])
            emitters.append(
                Emitter(
                    wavelength=lam,
                    power=float(pw[i]),
                    profile=_relocate(base_profile, positions[i][0], positions[i][1],
                                      point[i][0], point[i][1]),
                    waveform=_transceiver_waveform(i, baud, waveform, modulation,
                                                   extinction_ratio_db, prbs_bits,
                                                   linewidth, rin_dbc_hz, seed),
                    coherence=f"{name}:{i}",  # separate packages -> incoherent
                    label=f"{name}[{i}]",
                    meta={"index": i, "freq_offset": float(offs[i]),
                          "x": positions[i][0], "y": positions[i][1]},
                )
            )
        super().__init__(emitters, name=name)
        self.center_wavelength = center_wavelength
        self.freq_span = freq_span
        self.baud = baud

    def frequency_offsets(self) -> np.ndarray:
        return np.array([e.meta.get("freq_offset", 0.0) for e in self.emitters])


@register("source", "cwdm_bank", "coarse_wdm")
class CoarseWDMBank(EmitterBank):
    """Transceivers on genuinely different wavelengths (CWDM or custom list).

    Because a diffractive deflector is dispersive, each wavelength lands in a
    different place on the screen; that is either a nuisance to be corrected or a
    free wavelength-routed fan-out, depending on the experiment.
    """

    def __init__(
        self,
        wavelengths: Sequence[float] | None = None,
        n: int = 4,
        first_cwdm: int = 0,
        power: float | Sequence[float] = 1e-3,
        baud: float = 25e9,
        modulation: str = "nrz",
        waveform=None,
        extinction_ratio_db: float = 8.0,
        prbs_bits: int = 63,
        linewidth: float = 5e6,
        rin_dbc_hz: float = -140.0,
        positions=None,
        pitch: float = 250e-6,
        pointing=None,
        profile=None,
        name: str = "cwdm",
        seed: int = 0,
    ):
        if wavelengths is None:
            wavelengths = [CWDM_NM[(first_cwdm + i) % len(CWDM_NM)] * 1e-9 for i in range(n)]
        wavelengths = [float(w) for w in wavelengths]
        count = len(wavelengths)
        if positions is None:
            positions = grid_positions(count, 1, pitch)
        positions = np.asarray(positions, float).reshape(-1, 2)
        point = np.zeros((count, 2)) if pointing is None else np.asarray(pointing, float).reshape(-1, 2)
        pw = np.broadcast_to(np.atleast_1d(np.asarray(power, float)), (count,))
        base_profile = as_profile(profile if profile is not None else GaussianBeam(waist=0.3e-3))

        from .lasers import _relocate

        emitters = []
        for i, lam in enumerate(wavelengths):
            emitters.append(
                Emitter(
                    wavelength=lam,
                    power=float(pw[i]),
                    profile=_relocate(base_profile, positions[i][0], positions[i][1],
                                      point[i][0], point[i][1]),
                    waveform=_transceiver_waveform(i, baud, waveform, modulation,
                                                   extinction_ratio_db, prbs_bits,
                                                   linewidth, rin_dbc_hz, seed),
                    coherence=f"{name}:{i}",
                    label=f"{name}[{lam*1e9:.0f}nm]",
                    meta={"index": i, "x": positions[i][0], "y": positions[i][1]},
                )
            )
        super().__init__(emitters, name=name)
