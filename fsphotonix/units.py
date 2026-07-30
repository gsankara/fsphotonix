"""Physical constants and unit helpers.

Internally the simulator is strictly SI: metres, seconds, watts, radians.  These
helpers exist so that scripts and configs can stay readable.
"""

from __future__ import annotations

C0 = 299_792_458.0  # m/s
H_PLANCK = 6.626_070_15e-34  # J s
Q_E = 1.602_176_634e-19  # C
K_B = 1.380_649e-23  # J/K

# length
nm = 1e-9
um = 1e-6
mm = 1e-3
cm = 1e-2
m = 1.0
inch = 25.4e-3

# time
fs = 1e-15
ps = 1e-12
ns = 1e-9
us = 1e-6
ms = 1e-3
s = 1.0

# frequency / rate
Hz = 1.0
kHz = 1e3
MHz = 1e6
GHz = 1e9
THz = 1e12
Gbps = 1e9
GBd = 1e9

# power
pW = 1e-12
nW = 1e-9
uW = 1e-6
mW = 1e-3
W = 1.0

# angle
import math as _math  # noqa: E402

deg = _math.pi / 180.0
mrad = 1e-3
urad = 1e-6

# common telecom wavelengths
LAMBDA_C_BAND = 1550e-9
LAMBDA_O_BAND = 1310e-9
LAMBDA_850 = 850e-9
LAMBDA_VIS = 632.8e-9


def freq_to_wavelength(f: float) -> float:
    """Optical frequency [Hz] -> vacuum wavelength [m]."""
    return C0 / f


def wavelength_to_freq(lam: float) -> float:
    """Vacuum wavelength [m] -> optical frequency [Hz]."""
    return C0 / lam


def itu_grid(channel: int, spacing: float = 100e9, anchor_thz: float = 193.1) -> float:
    """Wavelength [m] of an ITU-T DWDM channel offset from the 193.1 THz anchor.

    ``itu_grid(0)`` is 1552.52 nm; ``itu_grid(+1, 100*GHz)`` is one 100 GHz slot up.
    """
    return C0 / (anchor_thz * 1e12 + channel * spacing)


def dbm(p_watt: float) -> float:
    """Watts -> dBm."""
    import math

    return 10.0 * math.log10(max(p_watt, 1e-300) / 1e-3)


def from_dbm(p_dbm: float) -> float:
    """dBm -> watts."""
    return 10.0 ** (p_dbm / 10.0) * 1e-3


def db(ratio: float) -> float:
    import math

    return 10.0 * math.log10(max(ratio, 1e-300))


def from_db(value_db: float) -> float:
    return 10.0 ** (value_db / 10.0)


def photon_energy(lam: float) -> float:
    """Energy of one photon at wavelength ``lam`` [J]."""
    return H_PLANCK * C0 / lam


def photons_per_second(power: float, lam: float) -> float:
    return power / photon_energy(lam)


def diffraction_angle(order: int, wavelength: float, period: float) -> float:
    """First-principles grating deflection angle [rad] (normal incidence)."""
    import math

    s = order * wavelength / period
    if abs(s) > 1.0:
        return float("nan")  # evanescent order
    return math.asin(s)


def grating_period(angle: float, wavelength: float, order: int = 1) -> float:
    """Grating period [m] needed to deflect ``order`` to ``angle`` [rad]."""
    import math

    return order * wavelength / math.sin(angle)
