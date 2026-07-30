"""System assembly: planes, the cascade engine, electronics and feedback loops."""

from .feedback import (
    Actuator,
    DriveActuator,
    FunctionActuator,
    GainActuator,
    History,
    Loop,
    Machine,
    PistonActuator,
    WeightActuator,
    global_loop,
    local_loop,
    make_actuator,
)
from .logic import (
    LUT,
    Boolean,
    Gain,
    Integrator,
    LogicBlock,
    MatrixMultiply,
    NoiseInjector,
    Nonlinearity,
    Normalize,
    Pipeline,
    Quantizer,
    Register,
    Threshold,
    analog_path,
    as_logic,
    digital_path,
)
from .optical_system import OpticalSystem, Trace
from .stages import (
    DetectorStage,
    ElementStage,
    FunctionStage,
    ModulatorStage,
    SourceStage,
    Stage,
)

__all__ = [
    "Stage", "SourceStage", "ModulatorStage", "DetectorStage", "ElementStage",
    "FunctionStage", "OpticalSystem", "Trace",
    "LogicBlock", "Pipeline", "Gain", "Normalize", "Threshold", "Quantizer",
    "Nonlinearity", "LUT", "MatrixMultiply", "Boolean", "Register", "Integrator",
    "NoiseInjector", "analog_path", "digital_path", "as_logic",
    "Actuator", "PistonActuator", "GainActuator", "DriveActuator", "WeightActuator",
    "FunctionActuator", "make_actuator", "Loop", "Machine", "History",
    "local_loop", "global_loop",
]
