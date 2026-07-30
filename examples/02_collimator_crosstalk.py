"""Illumination design: collimator footprint vs guard gap vs cross-talk.

The input side is a coupled optimisation.  A lenslet in front of each emitter sets
how much of a modulator region is illuminated:

* under-fill and the hologram's aperture is wasted -- less light, and a broader
  spot on the screen because the effective sub-aperture is smaller;
* over-fill and light crosses the guard gap onto the *next* input's hologram,
  which diffracts it to that input's screen targets.  At the detector it is
  indistinguishable from signal.

This example sweeps the footprint at several guard-gap widths and prints the
capture/cross-talk trade-off, then shows the same trade on the second gap -- the
one between the sub-holograms of a single input.
"""

import sys

import numpy as np

import _figure
import fsphotonix as fp
from fsphotonix.analysis import cgh_gap_report, illumination_crosstalk, illumination_sweep
from fsphotonix.sources.collimator import CollimatedArray, Collimator

LAM = 1550e-9
PITCH = 500e-6  # region pitch on the panel
LENS_TO_PANEL = 6e-3
Z = 80e-3


def make_source(diameter: float, n: int = 5, stray: float = 0.0) -> CollimatedArray:
    return CollimatedArray(
        n=n, pitch=PITCH, distance=LENS_TO_PANEL,
        illumination_diameter=diameter,
        collimator=Collimator(mfd=10.4e-6, focal_length=1.5e-3,
                              clear_aperture=0.95 * PITCH),
        wavelength=LAM, power=1e-3, stray_fraction=stray, name="cots",
    )


def main() -> None:
    grid = fp.Grid.from_extent(6 * PITCH, PITCH * 2, 2e-6)
    diameters = np.array([150e-6, 250e-6, 350e-6, 450e-6, 600e-6, 800e-6])

    print("collimator at nominal focus:")
    c = Collimator(mfd=10.4e-6, focal_length=1.5e-3, clear_aperture=0.95 * PITCH)
    for k, v in c.report(LENS_TO_PANEL, LAM).items():
        print(f"  {k:28s} {v}")

    print("\nfootprint sweep, three guard gaps  (capture % / cross-talk dB / spill %)")
    header = "  footprint  " + "".join(f"{g*1e6:>26.0f}um gap" for g in (40e-6, 100e-6, 200e-6))
    print(header)
    curves = {}
    for gap in (40e-6, 100e-6, 200e-6):
        panel = fp.ModulatorArray.tiled(nx=5, ny=1, pitch=PITCH, gap=gap,
                                        device="ipms_piston", default_wavelength=LAM)
        curves[gap] = illumination_sweep(make_source, panel, grid, diameters)
    for i, d in enumerate(diameters):
        row = f"  {d*1e6:7.0f}um  "
        for gap in (40e-6, 100e-6, 200e-6):
            s = curves[gap]
            row += (f"   {s['capture'][i]*100:6.1f}% "
                    f"{s['crosstalk_db'][i]:7.1f}dB {s['spill'][i]*100:5.1f}%")
        print(row)

    # pick the footprint that keeps cross-talk under -20 dB at a 100 um gap
    s = curves[100e-6]
    ok = np.nonzero(s["crosstalk_db"] < -20)[0]
    best = int(ok[-1]) if ok.size else 0
    print(f"\nlargest footprint under -20 dB cross-talk at a 100 um gap: "
          f"{diameters[best]*1e6:.0f} um (capture {s['capture'][best]:.1%})")

    print("\nlenslet leakage is an independent route:")
    panel = fp.ModulatorArray.tiled(nx=5, ny=1, pitch=PITCH, gap=100e-6,
                                    device="ipms_piston", default_wavelength=LAM)
    for stray in (0.0, 0.02, 0.05):
        src = make_source(float(diameters[best]), stray=stray)
        mapping = ([i for i in range(5)] if stray == 0
                   else [i // 2 for i in range(10)])
        r = illumination_crosstalk(src.emit(grid), panel, mapping=mapping)
        print(f"  stray={stray:4.0%}  cross-talk {np.nanmean(r['crosstalk_db']):7.2f} dB")

    print("\nsecond gap: between the sub-holograms of one input")
    print("  intra gap   subaperture   spot@80mm   aperture loss")
    for sub_gap in (0.0, 20e-6, 50e-6):
        p = fp.ModulatorArray.tiled(nx=1, ny=1, pitch=PITCH, gap=100e-6,
                                    device="ipms_piston", default_wavelength=LAM)
        p.route(0, [(-4e-3, 0.0), (0.0, 0.0), (4e-3, 0.0)], z=Z, wavelength=LAM,
                tiling="stripes", gap=sub_gap)
        g = cgh_gap_report(p, z=Z, wavelength=LAM)
        print(f"  {sub_gap*1e6:6.0f}um {g['subaperture_um']:10.1f}um "
              f"{g['spot_size_um']:10.1f}um {g['aperture_loss_db']:10.2f} dB   "
              f"guard ok={g['detector_guard_ok']}")

    if _figure.wanted():
        plt = _figure.pyplot()
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for gap, s in curves.items():
            ax[0].plot(s["diameter_m"] * 1e6, s["capture"] * 100, "o-",
                       label=f"{gap*1e6:.0f} um gap")
            ax[1].plot(s["diameter_m"] * 1e6, s["crosstalk_db"], "o-",
                       label=f"{gap*1e6:.0f} um gap")
        ax[0].set_xlabel("illuminated footprint [um]")
        ax[0].set_ylabel("capture efficiency [%]")
        ax[1].set_xlabel("illuminated footprint [um]")
        ax[1].set_ylabel("illumination cross-talk [dB]")
        ax[1].axhline(-20, ls="--", c="k", lw=1)
        for a in ax:
            a.legend(fontsize=8)
            a.grid(alpha=0.3)
        _figure.finish(fig, "02_illumination_tradeoff")


if __name__ == "__main__":
    main()
