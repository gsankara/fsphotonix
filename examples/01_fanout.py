"""One illuminated region, several holographic sub-gratings, many screen spots.

The most basic object in the architecture: a region of the modulator is divided
into sub-apertures, each carrying its own off-axis focusing hologram, and each
sending its share of the light to a chosen point on the detection screen.

Run with ``--plot`` if matplotlib is installed.
"""

import sys

import numpy as np

import _figure
import fsphotonix as fp
from fsphotonix.analysis import cgh_gap_report, power_budget, spot_metrics

LAM = 1550e-9
Z = 80e-3  # modulator -> screen


def build(n_targets: int = 5, device: str = "ipms_piston", tiling: str = "grid"):
    region = 800e-6
    panel = fp.ModulatorArray.tiled(nx=1, ny=1, pitch=1e-3, region_size=region,
                                    device=device, default_wavelength=LAM)

    # a ring of targets on the screen, given in absolute screen coordinates
    ang = np.linspace(0, 2 * np.pi, n_targets, endpoint=False)
    ring = 2e-3  # must fit inside the sensor: 256 x 25 um = 6.4 mm across
    targets = [(ring * np.cos(a), ring * np.sin(a)) for a in ang]
    panel.route(0, targets, z=Z, wavelength=LAM, tiling=tiling, gap=20e-6)

    src = fp.CoherentLaser(wavelength=LAM, power=0.5e-3,
                           profile=fp.SuperGaussian(diameter=region, order=12))
    cam = fp.CameraArray(nx=256, ny=256, pitch=25e-6, integration_time=0.05e-6,
                         output="dn", full_well_e=100_000)

    panel_grid = fp.Grid.from_extent(1.2e-3, 1.2e-3, 0.6e-6)
    system = fp.OpticalSystem([
        fp.SourceStage(source=src, grid=panel_grid, z=0.0, name="src"),
        fp.ModulatorStage(modulator=panel, grid=panel_grid, z=0.0, name="panel"),
        fp.DetectorStage(detector=cam, grid=cam.grid(oversample=1), z=Z, name="screen"),
    ], name="fanout")
    return system, panel, targets


def main() -> None:
    system, panel, targets = build()
    print(system)
    print(f"device: {panel.device.info()}")

    gaps = cgh_gap_report(panel, z=Z, wavelength=LAM)
    print("\n-- hologram gaps --")
    for k in ("subaperture_um", "intra_input_gap_um", "pixels_per_subaperture",
              "spot_size_um", "target_pitch_um", "detector_guard_ok"):
        print(f"  {k:26s} {gaps[k]}")

    trace = system.run()
    print("\n-- power budget --")
    for k, v in power_budget(trace)["power_W"].items():
        print(f"  {k:8s} {v*1e3:8.4f} mW")

    # the metric radius must be a couple of spot widths, not an arbitrary number
    radius = 1.5 * gaps["spot_size_um"] * 1e-6
    m = spot_metrics(trace.fields["screen"], targets, radius=radius)
    print("\n-- spots --")
    print(f"  collected fraction {m['collected_fraction']:.3f}")
    print(f"  uniformity         {m['uniformity']:.3f}")
    print(f"  centroid error     {np.nanmax(m['centroid_error_m'])*1e6:.1f} um (worst)")
    print(f"  power per spot     {np.round(m['power_W']*1e6, 2)} uW")

    cam_out = trace.outputs["screen"]
    print(f"\ncamera: {cam_out}, well fill {cam_out.meta['well_fill']:.2%}")

    if _figure.wanted():
        plt = _figure.pyplot()
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
        fp.viz.show_hologram(panel, system["panel"].grid, LAM, axes=axes[:2])
        fp.viz.show_intensity(trace.fields["screen"], ax=axes[2], log=True,
                              title="screen (dB)")
        _figure.finish(fig, "01_fanout")


if __name__ == "__main__":
    main()
