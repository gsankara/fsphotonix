"""Plugging in custom technology: a metasurface grating and a SPAD screen.

Every replaceable part of the simulator is a registered plugin.  A researcher adds
a technology by subclassing the relevant base class and decorating it -- no change
to the simulator, and the new device is immediately reachable from YAML configs and
from every builder that accepts a plugin name.

This example adds three things and runs them in a real system:

* a **grating**: a polarisation-insensitive metasurface deflector with a measured
  wavelength-dependent efficiency curve, which no built-in class models;
* a **detector**: a photon-counting SPAD array with dead time and afterpulsing;
* a **logic block**: a custom feedback controller.
"""

import numpy as np

import _figure
import fsphotonix as fp
from fsphotonix.detectors.base import Detector, DetectorOutput
from fsphotonix.modulators.gratings.base import Grating, Order, carrier_frequency
from fsphotonix.registry import register
from fsphotonix.system.logic import LogicBlock
from fsphotonix.units import photon_energy

LAM = 1550e-9
Z = 70e-3


# --------------------------------------------------------------------------- #
# 1. a custom grating technology
# --------------------------------------------------------------------------- #
@register("grating", "metasurface")
class MetasurfaceDeflector(Grating):
    """Nano-pillar metasurface deflector with a measured efficiency spectrum.

    Unlike a wrapped phase ramp, a metasurface imposes its phase through pillar
    geometry rather than an optical path difference, so the phase profile carries no
    ``lambda0/lambda`` scaling (``achromatic=True``) -- but the deflection still
    obeys the grating equation, and the *efficiency* is resonant and therefore a
    strong function of wavelength.  That combination is the opposite of a blazed
    grating's and is exactly what a plugin slot is for.
    """

    def __init__(self, theta_x=0.03, theta_y=0.0, design_wavelength=LAM,
                 peak_efficiency=0.85, bandwidth=120e-9):
        self.design_wavelength = design_wavelength
        self.achromatic = True  # geometric phase: no lambda0/lambda scaling
        self.peak_efficiency = peak_efficiency
        self.bandwidth = bandwidth
        self.fx, self.fy = carrier_frequency(theta_x, theta_y, design_wavelength)
        self.efficiency_scale = peak_efficiency

    def efficiency(self, wavelength: float) -> float:
        detune = (wavelength - self.design_wavelength) / self.bandwidth
        return float(self.peak_efficiency * np.exp(-(detune**2)))

    def phase(self, xl, yl):
        return 2 * np.pi * (self.fx * xl + self.fy * yl)

    def amplitude(self, xl, yl):
        import fsphotonix.engine as eng

        return eng.xp.ones_like(eng.asarray(xl, dtype=eng.REAL))

    def unit_cell(self):
        return None

    def orders(self, wavelength=None, **kw):
        lam = wavelength or self.design_wavelength
        return [Order(1, 0, self.fx, self.fy, self.efficiency(lam))]


# --------------------------------------------------------------------------- #
# 2. a custom detection screen technology
# --------------------------------------------------------------------------- #
@register("detector", "spad")
class SPADArray(Detector):
    """Photon-counting SPAD array with dead time and afterpulsing.

    A regime none of the built-in screens covers: the output is a *count*, the
    response saturates through dead time rather than through a full well, and the
    noise is dominated by dark counts and afterpulses.
    """

    name = "spad"

    def __init__(self, positions, active_diameter=50e-6, pde=0.25,
                 dead_time=25e-9, dark_cps=200.0, afterpulse=0.02,
                 gate=1e-6, seed=0):
        self.positions_in = np.asarray(positions, dtype=float).reshape(-1, 2)
        self.active_diameter = active_diameter
        self.pde = pde
        self.dead_time = dead_time
        self.dark_cps = dark_cps
        self.afterpulse = afterpulse
        self.gate = gate
        self._rng = np.random.default_rng(seed)

    @property
    def positions(self):
        return self.positions_in

    def detect(self, fields, t=None) -> DetectorOutput:
        from fsphotonix.detectors.photodiode import _element_masks
        import fsphotonix.engine as eng

        masks = _element_masks(fields.grid, self.positions, self.active_diameter)
        I = eng.to_numpy(fields.mean_intensity()).ravel()
        power = masks @ I * fields.grid.cell_area
        lam = fields.wavelengths[0] if fields.wavelengths else LAM

        incident_cps = power / photon_energy(lam)
        detected = incident_cps * self.pde
        # dead-time paralysis and afterpulsing
        live = detected / (1.0 + detected * self.dead_time)
        rate = live * (1.0 + self.afterpulse) + self.dark_cps
        counts = self._rng.poisson(np.maximum(rate * self.gate, 0.0))
        return DetectorOutput(values=counts.astype(float), unit="counts",
                              optical_power=power, positions=self.positions,
                              shape=(len(self.positions),),
                              meta={"saturated_fraction": float(
                                  np.mean(detected * self.dead_time > 0.5)),
                                  "rate_cps": rate})


# --------------------------------------------------------------------------- #
# 3. a custom controller
# --------------------------------------------------------------------------- #
@register("controller", "winner_take_all")
class WinnerTakeAll(LogicBlock):
    """Drive only the strongest channel -- a trivial nonlinear decision layer."""

    name = "wta"
    latency = 1

    def __init__(self, k: int = 1):
        self.k = k

    def __call__(self, x):
        v = np.asarray(x, dtype=float).ravel()
        out = np.zeros_like(v)
        if v.size:
            out[np.argsort(v)[-self.k:]] = 1.0
        return out


# --------------------------------------------------------------------------- #
def main() -> None:
    print("plugins now registered:")
    for kind in ("grating", "detector", "controller"):
        print(f"  {kind:11s}: {', '.join(fp.available(kind))}")

    # --- build a system entirely out of plugin names ------------------------- #
    n = 3
    pitch = 900e-6
    outs = np.stack([np.linspace(-3e-3, 3e-3, n), np.zeros(n)], axis=1)
    panel = fp.ModulatorArray.tiled(nx=n, ny=1, pitch=pitch, gap=100e-6,
                                    device="ipms_piston", default_wavelength=LAM)
    for j, r in enumerate(panel.regions):
        angle = np.arctan2(outs[j, 0] - r.center[0], Z)
        r.subregions = [fp.SubRegion(grating=fp.create(
            "grating", "metasurface", theta_x=angle, design_wavelength=LAM))]
    panel.touch()

    src = fp.CoarseWDMBank(wavelengths=[1490e-9, 1550e-9, 1610e-9], power=1e-3,
                           positions=[(r.center[0], r.center[1]) for r in panel.regions],
                           profile=fp.SuperGaussian(diameter=700e-6, order=10))
    spad = fp.create("detector", "spad", positions=outs, active_diameter=400e-6,
                     gate=1e-6)

    grid = fp.Grid.from_extent(n * pitch * 1.2, 1.2e-3, 40e-6)
    system = fp.OpticalSystem([
        fp.SourceStage(source=src, grid=grid, z=0.0, name="src"),
        fp.ModulatorStage(modulator=panel, grid=grid, z=0.0, name="panel",
                          spot_mode=True),
        # a SPAD needs an attenuator in front of it: 70 dB puts a milliwatt-class
        # machine into the photon-counting regime instead of hard paralysis
        fp.DetectorStage(detector=spad, grid=fp.Grid.from_extent(9e-3, 2e-3, 60e-6),
                         z=Z, name="screen", tap_fraction=1e-7),
    ], name="custom")

    trace = system.run()
    out = trace.outputs["screen"]
    print(f"\nSPAD readout: {out}")
    print(f"  counts        {out.values}")
    print(f"  optical power {np.round(out.optical_power * 1e12, 3)} pW")
    print(f"  dead-time saturated fraction {out.meta['saturated_fraction']:.2f}")

    g = MetasurfaceDeflector(theta_x=0.03)
    print("\nthe metasurface's phase is geometric (no lambda0/lambda scaling), so the")
    print("deflection follows the grating equation while the efficiency is resonant:")
    for lam in (1490e-9, 1550e-9, 1610e-9):
        o = g.orders(lam)[0]
        print(f"  {lam*1e9:.0f} nm: efficiency {o.efficiency:.3f}, "
              f"deflection {np.degrees(o.angles(lam)[0]):.4f} deg")

    print("\ncustom controller in a loop:")
    wta = fp.create("controller", "winner_take_all", k=1)
    loop = fp.Loop(sense="screen", actuator=fp.make_actuator(system, "src", "drive"),
                   controller=wta, name="wta")
    m = fp.Machine(system, loops=[loop])
    m.run(4)
    print(f"  drives after 4 iterations: {src.get_drive()}")

    if _figure.wanted():
        plt = _figure.pyplot()
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        lams = np.linspace(1400e-9, 1700e-9, 200)
        ax[0].plot(lams * 1e9, [g.efficiency(l) for l in lams], lw=1.8)
        for lam in (1490e-9, 1550e-9, 1610e-9):
            ax[0].plot(lam * 1e9, g.efficiency(lam), "o", ms=7)
        ax[0].set_xlabel("wavelength [nm]")
        ax[0].set_ylabel("diffraction efficiency")
        ax[0].set_title("custom grating: resonant metasurface")
        ax[0].grid(alpha=0.3)

        ax[1].plot(lams * 1e9, np.degrees(np.arcsin(lams * g.fx)), lw=1.8, color="C1")
        ax[1].set_xlabel("wavelength [nm]")
        ax[1].set_ylabel("deflection [deg]")
        ax[1].set_title("geometric phase: angle still obeys the grating equation")
        ax[1].grid(alpha=0.3)
        _figure.finish(fig, "05_custom_plugin")


if __name__ == "__main__":
    main()
