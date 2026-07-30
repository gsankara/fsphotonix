"""Modulator plane: geometry, gratings, hardware models, and the array itself."""

from . import gratings
from .array import ModulatorArray, Region, SubRegion, incidence_from
from .devices import (
    DMD,
    AmplitudeSLM,
    IdealModulator,
    LCoSPhaseSLM,
    ModulatorDevice,
    PistonMicroMirrorArray,
    TiltMicroMirrorArray,
    TransmissiveLCSLM,
    as_device,
)
from .geometry import (
    Annulus,
    Aperture,
    Circle,
    Combined,
    CustomAperture,
    Ellipse,
    Hexagon,
    Polygon,
    Rect,
    Sector,
    Shifted,
    as_aperture,
    tile_region,
)
from .gratings import *  # noqa: F401,F403  (re-export the grating library)

__all__ = [
    "ModulatorArray", "Region", "SubRegion", "incidence_from",
    "ModulatorDevice", "IdealModulator", "LCoSPhaseSLM", "TransmissiveLCSLM",
    "AmplitudeSLM", "PistonMicroMirrorArray", "TiltMicroMirrorArray", "DMD",
    "as_device",
    "Aperture", "Rect", "Circle", "Ellipse", "Annulus", "Sector", "Hexagon",
    "Polygon", "CustomAperture", "Combined", "Shifted", "as_aperture", "tile_region",
    "gratings",
] + list(gratings.__all__)
