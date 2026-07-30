"""fsphotonix -- large-scale free-space photonic system simulator.

A physical-layer simulator for machines built out of light sources, holographic
modulator arrays, free-space propagation, detection screens and feedback loops.

Layer map
---------
``sources``      coherent lasers, COTS transceiver banks, WDM combs; digital,
                 analog and mixed pulse trains
``modulators``   illuminated regions carrying one or more holographic grating
                 sub-regions, rendered through a hardware model (LCoS, DMD,
                 MEMS piston/tilt micro-mirrors, transmissive LC, ideal)
``propagation``  angular spectrum, Fresnel (same-grid and arbitrary-grid zoom),
                 Fraunhofer, bulk optics, and a beamlet propagator for very
                 large systems
``detectors``    CCD/CMOS cameras, photodiode and APD arrays, SMF and MMF
                 coupling, power meters, custom
``system``       planes, the cascade engine, digital/analog electronics and
                 local/global feedback loops
``solvers``      optical matrix-vector multiply and the analog iterative machine

Every one of those slots is a plugin; see :mod:`fsphotonix.registry`.

Quick start
-----------
>>> import fsphotonix as fp
>>> panel = fp.ModulatorArray.tiled(nx=4, ny=1, pitch=800e-6, device="ipms_piston")
>>> panel.broadcast([(-2e-3, 0.0), (0.0, 0.0), (2e-3, 0.0)], z=0.08)  # 3-way fan-out
>>> sys_ = fp.build_relay(
...     source=fp.TransceiverBank(n=4, pitch=800e-6),
...     modulator=panel,
...     detector=fp.CameraArray(nx=192, ny=64, pitch=20e-6),
...     z=0.08)
>>> trace = sys_.run()
>>> trace.vector("screen").shape
(12288,)
"""

from __future__ import annotations

__version__ = "0.1.0"
__license__ = "BSD-3-Clause"
__author__ = "Ganesh Chennimalai Sankaran"

from . import analysis, engine, registry, units, viz
from .config import build_machine, build_system, load_file, load_text
from .detectors import (
    CameraArray,
    CustomDetector,
    Detector,
    DetectorOutput,
    MMFArray,
    PhotodiodeArray,
    PowerMeter,
    SMFArray,
)
from .fields import Channel, FieldSet
from .grid import Grid, sampling_report
from .modulators import (
    DMD,
    AmplitudeSLM,
    Annulus,
    Aperture,
    Apodizer,
    Axicon,
    BinaryPhaseGrating,
    BlazedGrating,
    Cascade,
    Circle,
    CrossedGrating,
    CustomAperture,
    DammannGrating,
    Ellipse,
    FlatMirror,
    FresnelLens,
    Grating,
    Hexagon,
    IdealModulator,
    LCoSPhaseSLM,
    ModulatorArray,
    ModulatorDevice,
    MultiSpotHologram,
    Order,
    PhaseMap,
    PhaseRamp,
    PistonMicroMirrorArray,
    Polygon,
    Rect,
    Region,
    RonchiGrating,
    Sector,
    SinusoidalPhaseGrating,
    SpotFocuser,
    SpotTarget,
    SubRegion,
    Superposition,
    TiltMicroMirrorArray,
    TransmissiveLCSLM,
    ZonePlate,
    incidence_from,
    tile_region,
)
from .propagation import (
    AfocalRelay,
    AngularSpectrum,
    ApertureStop,
    Auto,
    FourierPlane,
    Fraunhofer,
    FresnelTransfer,
    FresnelZoom,
    Misalignment,
    PhaseScreen,
    Propagator,
    Spot,
    SpotList,
    SpotPropagator,
    ThinLens,
    suggest,
)
from .registry import available, create, describe, register
from .solvers import (
    AnalogIterativeMachine,
    OpticalMVM,
    cut_value,
    maxcut_ising,
    qubo_to_ising,
    random_graph,
)
from .sources import (
    CoarseWDMBank,
    CoherentLaser,
    Emitter,
    EmitterBank,
    GaussianBeam,
    LaserArray,
    LightSource,
    PlaneWave,
    Rectangular,
    SuperGaussian,
    TransceiverBank,
    WDMComb,
    grid_positions,
    line_positions,
)
from .system import (
    DetectorStage,
    ElementStage,
    FunctionStage,
    Gain,
    History,
    Integrator,
    LogicBlock,
    Loop,
    Machine,
    ModulatorStage,
    NoiseInjector,
    Nonlinearity,
    Normalize,
    OpticalSystem,
    Pipeline,
    Quantizer,
    Register,
    SourceStage,
    Threshold,
    Trace,
    analog_path,
    digital_path,
    global_loop,
    local_loop,
    make_actuator,
)
from .units import GHz, THz, cm, deg, mW, mm, mrad, nm, um, us
from .waveform import (
    NRZ,
    PAM,
    Arbitrary,
    Burst,
    Chirp,
    Constant,
    Impaired,
    Mixed,
    PulseTrain,
    Sampled,
    Sinusoid,
    Waveform,
)


# --------------------------------------------------------------------------- #
# assembly shortcuts
# --------------------------------------------------------------------------- #
def build_relay(source, modulator, detector, z: float, panel_grid: Grid | None = None,
                screen_grid: Grid | None = None, source_z: float = 0.0,
                spot_mode: bool = False, name: str = "relay") -> OpticalSystem:
    """The canonical three-plane system: source -> modulator -> screen.

    Grids are inferred when not given: the panel from the modulator's region
    layout (8 samples across the smallest device pixel), the screen from the
    detector's own geometry.
    """
    if panel_grid is None:
        import numpy as _np

        centers = _np.array([r.center for r in modulator.regions]) if modulator.regions \
            else _np.zeros((1, 2))
        sizes = [max(r.aperture.size) for r in modulator.regions] or [1e-3]
        span_x = float(_np.ptp(centers[:, 0])) + max(sizes) * 1.5
        span_y = float(_np.ptp(centers[:, 1])) + max(sizes) * 1.5
        pitch = (max(sizes) / 8 if spot_mode
                 else min(modulator.device.pitch, max(sizes) / 24))
        panel_grid = Grid.from_extent(span_x, span_y, pitch)
    if screen_grid is None:
        screen_grid = detector.grid() if hasattr(detector, "grid") else Grid.from_extent(
            20e-3, 20e-3, 20e-6)
    return OpticalSystem([
        SourceStage(source=source, grid=panel_grid, z=source_z, name="src"),
        ModulatorStage(modulator=modulator, grid=panel_grid, z=0.0, name="panel",
                       spot_mode=spot_mode),
        DetectorStage(detector=detector, grid=screen_grid, z=z, name="screen"),
    ], name=name)


def build_cascade(source, modulators, detector, spacings, panel_grid: Grid | None = None,
                  screen_grid: Grid | None = None, spot_mode: bool = False,
                  name: str = "cascade") -> OpticalSystem:
    """Multi-stage diffractive chain: source -> mod1 -> mod2 -> ... -> screen.

    ``spacings`` has one entry per hop (``len(modulators)`` entries: the gaps
    between successive modulator planes and then to the screen).  Any intermediate
    plane is a modulator array acting on the light incident on it -- the
    "detection screen replaced by another modulator" case.
    """
    if len(spacings) != len(modulators):
        raise ValueError("need one spacing per hop (modulator gaps plus the final one)")
    if panel_grid is None:
        sizes = [max(r.aperture.size) for m in modulators for r in m.regions] or [1e-3]
        pitch = (max(sizes) / 8 if spot_mode
                 else min(min(m.device.pitch for m in modulators), max(sizes) / 24))
        panel_grid = Grid.from_extent(max(sizes) * 8, max(sizes) * 8, pitch)
    if screen_grid is None:
        screen_grid = detector.grid() if hasattr(detector, "grid") else Grid.from_extent(
            20e-3, 20e-3, 20e-6)
    stages: list = [SourceStage(source=source, grid=panel_grid, z=0.0, name="src")]
    z = 0.0
    for i, m in enumerate(modulators):
        stages.append(ModulatorStage(modulator=m, grid=panel_grid, z=z, name=f"panel{i}",
                                     spot_mode=spot_mode))
        z += float(spacings[i])
    stages.append(DetectorStage(detector=detector, grid=screen_grid, z=z, name="screen"))
    return OpticalSystem(stages, name=name)


__all__ = [
    "__version__", "engine", "registry", "units", "viz", "analysis",
    "Grid", "sampling_report", "Channel", "FieldSet",
    "Waveform", "Constant", "Sinusoid", "Chirp", "Arbitrary", "Sampled", "NRZ", "PAM",
    "PulseTrain", "Burst", "Mixed", "Impaired",
    "LightSource", "Emitter", "EmitterBank", "CoherentLaser", "LaserArray", "WDMComb",
    "TransceiverBank", "CoarseWDMBank", "GaussianBeam", "PlaneWave", "SuperGaussian",
    "Rectangular", "grid_positions", "line_positions",
    "ModulatorArray", "Region", "SubRegion", "ModulatorDevice", "IdealModulator",
    "LCoSPhaseSLM", "TransmissiveLCSLM", "AmplitudeSLM", "PistonMicroMirrorArray",
    "TiltMicroMirrorArray", "DMD",
    "Aperture", "Rect", "Circle", "Ellipse", "Annulus", "Sector", "Hexagon", "Polygon",
    "CustomAperture", "tile_region", "incidence_from",
    "Grating", "Order", "BlazedGrating", "PhaseRamp", "BinaryPhaseGrating",
    "RonchiGrating", "SinusoidalPhaseGrating", "CrossedGrating", "FlatMirror",
    "DammannGrating", "FresnelLens", "SpotFocuser", "Axicon", "ZonePlate",
    "MultiSpotHologram", "PhaseMap", "SpotTarget", "Superposition", "Cascade", "Apodizer",
    "Propagator", "AngularSpectrum", "FresnelTransfer", "FresnelZoom", "Fraunhofer",
    "Auto", "ThinLens", "ApertureStop", "FourierPlane", "AfocalRelay", "PhaseScreen",
    "Misalignment", "SpotPropagator", "Spot", "SpotList", "suggest",
    "Detector", "DetectorOutput", "CameraArray", "PhotodiodeArray", "PowerMeter",
    "SMFArray", "MMFArray", "CustomDetector",
    "OpticalSystem", "Trace", "SourceStage", "ModulatorStage", "DetectorStage",
    "ElementStage", "FunctionStage",
    "LogicBlock", "Pipeline", "Gain", "Normalize", "Threshold", "Quantizer",
    "Nonlinearity", "Register", "Integrator", "NoiseInjector", "analog_path",
    "digital_path", "Loop", "Machine", "History", "make_actuator", "local_loop",
    "global_loop",
    "OpticalMVM", "AnalogIterativeMachine", "maxcut_ising", "cut_value",
    "qubo_to_ising", "random_graph",
    "register", "create", "available", "describe",
    "build_system", "build_machine", "load_file", "load_text",
    "build_relay", "build_cascade",
    "nm", "um", "mm", "cm", "mW", "GHz", "THz", "deg", "mrad", "us",
]
