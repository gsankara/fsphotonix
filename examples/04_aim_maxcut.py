"""Analog iterative machine: Max-Cut solved through a real optical multiply.

The loop, once per iteration:

    spins s  -> emitter drive levels (dual rail, because light is non-negative)
             -> holographic fan-out that applies J
             -> detector array, dual-rail subtraction  -> gradient J s
             -> electronic nonlinearity + annealed noise
             -> new spins

Nothing is short-circuited: every iteration propagates light through the
holograms, hits a noisy photodiode array and comes back.  Running the same problem
with ``optics=None`` gives the ideal-arithmetic control, so the difference between
the two curves is attributable entirely to the photonics.
"""

import sys
import time

import numpy as np

import fsphotonix as fp
from fsphotonix.solvers.aim import (
    AnalogIterativeMachine,
    OpticalMVM,
    brute_force_maxcut,
    cut_value,
    maxcut_ising,
    random_graph,
)

N = 8
ITER = 400


def main() -> None:
    A = random_graph(N, density=0.6, seed=4)
    J, h = maxcut_ising(A)
    best_cut, _ = brute_force_maxcut(A)
    print(f"Max-Cut on a random graph, n={N}, exact optimum = {best_cut:.4g}")

    optics = OpticalMVM(n_in=N, n_out=N, z=150e-3, region_pitch=1.2e-3,
                        out_pitch=1.5e-3, device="ipms_piston", mode="spot")
    print("\n-- optical multiplier feasibility --")
    for k, v in optics.check().items():
        print(f"  {k:28s} {v}")

    print("\n-- calibration against the requested matrix --")
    t0 = time.perf_counter()
    aim = AnalogIterativeMachine(J, h, optics=optics, step=0.09, noise=0.4,
                                 noise_decay=0.992, seed=1)
    cal = optics.calibrate()
    print(f"  relative rms error of the optical matrix: {cal['relative_rms']:.4f}")
    print(f"  worst element error:                      {cal['max_error']:.4f}")
    print(f"  (calibration took {time.perf_counter()-t0:.2f} s)")

    print("\n-- solving --")
    t0 = time.perf_counter()
    res_opt = aim.solve(ITER)
    t_opt = time.perf_counter() - t0

    ideal = AnalogIterativeMachine(J, h, optics=None, step=0.09, noise=0.4,
                                   noise_decay=0.992, seed=1)
    res_num = ideal.solve(ITER)

    for label, res, dt in (("optical ", res_opt, t_opt), ("numeric ", res_num, None)):
        cut = cut_value(A, res.best_spins)
        line = (f"  {label} best energy {res.best_energy:9.4f}   "
                f"cut {cut:8.4f}   {cut/best_cut:6.1%} of optimum")
        if dt is not None:
            line += f"   ({dt:.1f} s for {ITER} optical iterations)"
        print(line)

    print("\n-- what the photonics costs --")
    print(f"  gradient correlation with exact J.s: "
          f"{np.corrcoef(aim.gradient(res_opt.spins), J @ res_opt.spins + h)[0, 1]:.4f}")
    print(f"  final annealing temperature:         {aim.noise.temperature:.4f}")

    if "--plot" in sys.argv:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        fp.viz.show_convergence(res_opt, ax=ax[0],
                                reference=-(A.sum() / 2 - 2 * best_cut))
        ax[0].set_title("through the optics")
        fp.viz.show_convergence(res_num, ax=ax[1],
                                reference=-(A.sum() / 2 - 2 * best_cut))
        ax[1].set_title("ideal arithmetic")
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
