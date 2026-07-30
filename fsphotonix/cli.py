"""Command line interface: ``fsphotonix <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _cmd_plugins(args) -> int:
    from . import registry

    print(registry.describe())
    return 0


def _cmd_run(args) -> int:
    from . import config
    from .analysis import power_budget

    cfg = config.load_file(args.config)
    system = config.build_system(cfg)
    print(config.dump_summary(system))
    if args.check:
        for rep in system.sampling_report():
            print(f"  hop {rep['from']}->{rep['to']}: {rep['dz_mm']:.3g} mm, "
                  f"regime={rep['regime']}, kernel={rep['recommended']}, "
                  f"far-field spot={rep['far_field_spot_x']*1e6:.3g} um")
        return 0
    if args.iterations > 1:
        machine = config.build_machine(cfg, system)
        machine.run(args.iterations)
        print(f"ran {len(machine.history)} iterations, "
              f"{len(machine.loops)} loops: {machine.loop_report()}")
        rec = machine.history.last()
        out = rec.get("readout", {})
    else:
        trace = system.run()
        print(json.dumps(power_budget(trace), indent=2, default=float))
        out = {k: v.vector() for k, v in trace.outputs.items()}
    for name, v in out.items():
        v = np.asarray(v, dtype=float)
        print(f"{name}: n={v.size} min={v.min():.6g} max={v.max():.6g} sum={v.sum():.6g}")
    if args.save:
        np.savez(args.save, **{k: np.asarray(v) for k, v in out.items()})
        print(f"saved -> {args.save}")
    return 0


def _cmd_mvm(args) -> int:
    from .solvers.aim import OpticalMVM

    mvm = OpticalMVM(n_in=args.n_in, n_out=args.n_out, z=args.z, mode=args.mode)
    print(json.dumps(mvm.check(), indent=2, default=float))
    rng = np.random.default_rng(args.seed)
    W = rng.normal(0, 1, (args.n_out, args.n_in))
    mvm.set_matrix(W / np.abs(W).max())
    print(json.dumps(mvm.calibrate(), indent=2, default=float))
    x = rng.normal(0, 1, args.n_in)
    y_opt = mvm.multiply(x)
    y_ref = (W / np.abs(W).max()) @ x
    err = np.linalg.norm(y_opt - y_ref) / max(np.linalg.norm(y_ref), 1e-300)
    print(f"relative error of one optical multiply: {err:.4%}")
    return 0


def _cmd_aim(args) -> int:
    from .solvers.aim import (
        AnalogIterativeMachine,
        OpticalMVM,
        brute_force_maxcut,
        cut_value,
        maxcut_ising,
        random_graph,
    )

    A = random_graph(args.n, density=args.density, seed=args.seed)
    J, h = maxcut_ising(A)
    optics = None if args.numeric else OpticalMVM(n_in=args.n, n_out=args.n, z=args.z,
                                                  mode="spot")
    aim = AnalogIterativeMachine(J, h, optics=optics, step=args.step, noise=args.noise,
                                 seed=args.seed)
    res = aim.solve(args.iterations)
    print(f"best Ising energy: {res.best_energy:.6g}")
    print(f"cut value:         {cut_value(A, res.best_spins):.6g}")
    if args.n <= 18:
        best, _ = brute_force_maxcut(A)
        print(f"optimal cut:       {best:.6g}  ({cut_value(A, res.best_spins)/best:.2%})")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="fsphotonix",
                                description="free-space photonic system simulator")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("plugins", help="list all registered plugins")
    sp.set_defaults(func=_cmd_plugins)

    sr = sub.add_parser("run", help="run a system described by a YAML/JSON config")
    sr.add_argument("config")
    sr.add_argument("--iterations", type=int, default=1)
    sr.add_argument("--check", action="store_true", help="only report sampling diagnostics")
    sr.add_argument("--save", help="write detector outputs to an .npz file")
    sr.set_defaults(func=_cmd_run)

    sm = sub.add_parser("mvm", help="build and characterise an optical matrix multiplier")
    sm.add_argument("--n-in", type=int, default=8)
    sm.add_argument("--n-out", type=int, default=8)
    sm.add_argument("--z", type=float, default=0.05)
    sm.add_argument("--mode", default="spot", choices=["spot", "wave"])
    sm.add_argument("--seed", type=int, default=0)
    sm.set_defaults(func=_cmd_mvm)

    sa = sub.add_parser("aim", help="solve Max-Cut on the analog iterative machine")
    sa.add_argument("-n", type=int, default=12)
    sa.add_argument("--density", type=float, default=0.5)
    sa.add_argument("--iterations", type=int, default=300)
    sa.add_argument("--step", type=float, default=0.06)
    sa.add_argument("--noise", type=float, default=0.3)
    sa.add_argument("--z", type=float, default=0.05)
    sa.add_argument("--seed", type=int, default=0)
    sa.add_argument("--numeric", action="store_true",
                    help="bypass the optics (control experiment)")
    sa.set_defaults(func=_cmd_aim)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
