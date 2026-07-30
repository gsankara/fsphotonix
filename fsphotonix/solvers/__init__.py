"""Applications built on the optics: matrix multiply, iterative solvers."""

from .aim import (
    AIMResult,
    AnalogIterativeMachine,
    OpticalMVM,
    brute_force_maxcut,
    cut_value,
    maxcut_ising,
    qubo_to_ising,
    random_graph,
)

__all__ = [
    "OpticalMVM", "AnalogIterativeMachine", "AIMResult",
    "maxcut_ising", "cut_value", "qubo_to_ising", "random_graph",
    "brute_force_maxcut",
]
