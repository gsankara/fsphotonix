"""Multi-stage diffractive machine with digital logic between the optical layers.

Two things the architecture requires, in one script:

1. the detection screen of stage 1 is *replaced by another modulator array*, which
   operates on the light incident on it and relays it to a third plane;
2. between the optical layers sits electronics, and its depth is a design variable
   -- a purely analog return path, or one/several clocked digital stages.

It also surfaces the two effects that stop a real cascade from being the plain
matrix product of its stages: the angle sensitivity of the second diffractive
plane, and coherent interference between the optical routes that share a detector.
"""

import sys

import numpy as np

import _figure
import fsphotonix as fp
from fsphotonix.analysis import power_budget, transfer_matrix

LAM = 1550e-9
Z1, Z2 = 40e-3, 40e-3
N = 4
PITCH = 900e-6

MID = np.stack([np.linspace(-1.8e-3, 1.8e-3, N), np.zeros(N)], axis=1)
OUT = np.stack([np.linspace(-2.4e-3, 2.4e-3, N), np.zeros(N)], axis=1)
W1 = np.abs(np.eye(N) + 0.4 * np.roll(np.eye(N), 1, axis=0))
W2 = np.abs(np.eye(N) + 0.3 * np.roll(np.eye(N), -1, axis=0))


def build(precompensate: bool = False):
    # ---- stage 1: N inputs, each fanned out to N intermediate positions -------
    p1 = fp.ModulatorArray.tiled(nx=N, ny=1, pitch=PITCH, gap=120e-6,
                                 device="ipms_piston", default_wavelength=LAM,
                                 name="stage1")
    p1.set_matrix(W1, MID, z=Z1, tiling="stripes", gap=20e-6)

    # ---- stage 2: a modulator *where the screen would have been* --------------
    p2 = fp.ModulatorArray.from_positions(MID, region_size=700e-6,
                                          device="ipms_piston",
                                          default_wavelength=LAM, name="stage2")
    if precompensate:
        # design each hologram for the incidence of its dominant feed
        src_centers = np.array([r.center for r in p1.regions])
        cmax = np.abs(W2).sum(axis=0).max()
        for j, r in enumerate(p2.regions):
            col = np.abs(W2[:, j])
            keep = np.nonzero(col > 0)[0]
            p2.route(r, [(float(OUT[k, 0]), float(OUT[k, 1]), float(col[k]))
                         for k in keep],
                     z=Z2, wavelength=LAM, tiling="stripes", gap=20e-6,
                     incident=fp.incidence_from(src_centers[j], r.center, Z1))
            r.gain = float(np.sqrt(col.sum() / cmax))
        p2.touch()
    else:
        p2.set_matrix(W2, OUT, z=Z2, tiling="stripes", gap=20e-6)

    src = fp.LaserArray(n=N, pitch=PITCH, power=2e-3, linewidth=0.0, rin_dbc_hz=-300,
                        profile=fp.SuperGaussian(diameter=700e-6, order=10))
    det = fp.PhotodiodeArray(positions_in=OUT, active_diameter=1.2e-3,
                             output="watts", n_time=1, bandwidth=5e9, name="screen")

    panel_grid = fp.Grid.from_extent(N * PITCH * 1.2, 1.4e-3, 40e-6)
    system = fp.OpticalSystem([
        fp.SourceStage(source=src, grid=panel_grid, z=0.0, name="src"),
        fp.ModulatorStage(modulator=p1, grid=panel_grid, z=0.0, name="stage1",
                          spot_mode=True),
        fp.ModulatorStage(modulator=p2, grid=panel_grid, z=Z1, name="stage2",
                          spot_mode=True),
        fp.DetectorStage(detector=det, grid=fp.Grid.from_extent(8e-3, 2e-3, 80e-6),
                         z=Z1 + Z2, name="screen"),
    ], name="two_stage")
    return system, p1, p2


def main() -> None:
    system, p1, p2 = build()
    print(system)
    print("planes:", " -> ".join(system.stage_names()))

    trace = system.run()
    print("\n-- power at each plane --")
    for k, v in power_budget(trace)["power_W"].items():
        print(f"  {k:8s} {v*1e3:8.4f} mW")

    want = W2 @ W1
    want = want / want.max()
    print("\ncascaded |W2| @ |W1| (normalised) -- the intensity-domain intent:")
    print(np.round(want, 3))

    for label, pre in (("normal-incidence design", False),
                       ("per-region angle pre-compensation", True)):
        sysm = build(precompensate=pre)[0]
        T = transfer_matrix(sysm, "src", "screen")
        T = T / T.max()
        err = np.sqrt(np.mean((T - want) ** 2)) / np.sqrt(np.mean(want**2))
        print(f"\nmeasured, {label} (rel. rms vs intent {err:.3f}):")
        print(np.round(T, 3))

    print(
        "\nWhat those two rows show: stage 2 is angle sensitive.  A thin diffractive\n"
        "element adds its grating vector to the incident one, so each hologram points\n"
        "correctly for exactly one incidence.  Region j of stage 2 is fed by two\n"
        "stage-1 regions at different angles, so compensating for one path necessarily\n"
        "mis-points the other -- which is why per-region pre-compensation does not\n"
        "simply win.  A cascade that needs every path on target wants either a\n"
        "telecentric relay between the planes (every beam arrives normally, an\n"
        "ElementStage here) or angle-multiplexed holograms: one extra sub-region per\n"
        "incident direction, each with its own incident= design.")

    print("\ncoherent path interference (scan one stage-2 region's piston):")
    scan = []
    for phi in np.linspace(0, 2 * np.pi, 9):
        p2[0].piston = float(phi)
        p2.touch()
        scan.append(system.run(keep_fields=False).vector("screen"))
    p2[0].piston = 0.0
    p2.touch()
    scan = np.array(scan)
    depth = np.ptp(scan, axis=0) / np.maximum(scan.mean(axis=0), 1e-30)
    print(f"  peak-to-peak modulation per output: {np.round(depth * 100, 1)} %")
    print("  (non-zero wherever two mutually coherent routes reach one detector)")

    print("\n-- electronic return path depth --")
    drive = fp.make_actuator(system, "src", "drive")
    for name, path in (("analog", fp.analog_path(gain=2.0)),
                       ("digital x1", fp.digital_path(bits=8, stages=1, gain=2.0)),
                       ("digital x3", fp.digital_path(bits=6, stages=3, gain=2.0))):
        drive.write(np.ones(N))
        loop = fp.Loop(sense="screen", actuator=drive, controller=path, name=name)
        m = fp.Machine(system, loops=[loop],
                       metric=lambda tr, k: {"total_W": float(
                           tr.outputs["screen"].vector().sum())})
        m.run(12)
        series = m.history.series("total_W")
        print(f"  {name:11s} latency={path.latency} cycles  settled {series[-1]*1e6:8.3f} uW"
              f"  (swing over last 5: {np.ptp(series[-5:])*1e6:.3f} uW)")
    drive.write(np.ones(N))

    if _figure.wanted():
        plt = _figure.pyplot()
        fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
        fp.viz.show_system(system, ax=axes[0])
        fp.viz.show_spots(trace.spot_lists["stage2"], ax=axes[1])
        fp.viz.show_detector(trace.outputs["screen"], ax=axes[2])
        _figure.finish(fig, "03_cascade")


if __name__ == "__main__":
    main()
