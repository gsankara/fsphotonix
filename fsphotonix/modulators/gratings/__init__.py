"""Holographic grating library (plugin kind ``"grating"``)."""

from .base import (
    Grating,
    Order,
    as_grating,
    carrier_frequency,
    period_and_angle,
    target_to_angles,
)
from .cgh import MultiSpotHologram, PhaseMap, SpotTarget
from .composite import Apodizer, Cascade, Superposition
from .dammann import DammannGrating, dammann_transitions
from .lenses import Axicon, FresnelLens, SpotFocuser, ZonePlate
from .periodic import (
    BinaryPhaseGrating,
    BlazedGrating,
    CrossedGrating,
    FlatMirror,
    PhaseRamp,
    RonchiGrating,
    SinusoidalPhaseGrating,
)

__all__ = [
    "Grating", "Order", "as_grating",
    "carrier_frequency", "target_to_angles", "period_and_angle",
    "BlazedGrating", "PhaseRamp", "BinaryPhaseGrating", "RonchiGrating",
    "SinusoidalPhaseGrating", "CrossedGrating", "FlatMirror",
    "DammannGrating", "dammann_transitions",
    "FresnelLens", "SpotFocuser", "Axicon", "ZonePlate",
    "MultiSpotHologram", "PhaseMap", "SpotTarget",
    "Superposition", "Cascade", "Apodizer",
]
