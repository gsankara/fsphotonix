"""Light sources.

Plugin kind ``"source"``.  Add a technology with::

    @register("source", "my_emitter")
    class MyEmitter(EmitterBank): ...
"""

from .base import Emitter, EmitterBank, LightSource
from .collimator import CollimatedArray, Collimator, MicrolensArray
from .cots import CWDM_NM, CoarseWDMBank, TransceiverBank
from .lasers import CoherentLaser, LaserArray, WDMComb
from .layout import (
    fan_angles,
    grid_positions,
    hex_positions,
    layout,
    line_positions,
    ring_positions,
)
from .profiles import (
    BeamProfile,
    CustomProfile,
    GaussianBeam,
    LaguerreGaussVortex,
    PlaneWave,
    Rectangular,
    SuperGaussian,
    as_profile,
)

__all__ = [
    "LightSource", "Emitter", "EmitterBank",
    "Collimator", "CollimatedArray", "MicrolensArray",
    "CoherentLaser", "LaserArray", "WDMComb",
    "TransceiverBank", "CoarseWDMBank", "CWDM_NM",
    "BeamProfile", "GaussianBeam", "PlaneWave", "SuperGaussian", "Rectangular",
    "LaguerreGaussVortex", "CustomProfile", "as_profile",
    "grid_positions", "line_positions", "ring_positions", "hex_positions",
    "fan_angles", "layout",
]
