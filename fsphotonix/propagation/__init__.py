"""Propagation kernels and bulk optics (plugin kind ``"propagator"``)."""

from .base import Propagator, as_propagator
from .kernels import (
    AngularSpectrum,
    Auto,
    Fraunhofer,
    FresnelTransfer,
    FresnelZoom,
    suggest,
)
from .optics import AfocalRelay, ApertureStop, FourierPlane, Misalignment, PhaseScreen, ThinLens
from .spot import Spot, SpotList, SpotPropagator

__all__ = [
    "Propagator", "as_propagator", "suggest",
    "AngularSpectrum", "FresnelTransfer", "FresnelZoom", "Fraunhofer", "Auto",
    "ThinLens", "ApertureStop", "FourierPlane", "AfocalRelay", "PhaseScreen",
    "Misalignment",
    "SpotPropagator", "Spot", "SpotList",
]
