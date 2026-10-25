"""Sub-aperture apodization: paying light for cross-talk you cannot guard away.

A hologram region split into sub-apertures has a hard edge, and a hard edge
has sinc far-field wings whose envelope falls only as ``1/u^2``.  Across a
lattice of detectors that are *all* being driven at once -- which is what a
dense matrix means -- those wings pile up on every site the channel does not
own, and they set a cross-talk floor that no amount of guard band escapes: the
guard moves the detector further out along a tail that decays too slowly to
help.  Tapering the sub-aperture's amplitude to zero at its rim removes the
edge instead of running away from it, and replaces the ``1/u^2`` tail with a
far faster roll-off.

The two results worth reading off the output:

1. **Tukey is the bargain.**  26.1 dB of cross-talk suppression for 2.94 dB of
   light.  Its flat centre passes most of the aperture untouched while the
   cosine skirt still reaches exactly zero at the rim, and it is reaching zero
   -- not being gentle -- that kills the wings.  Hann buys 11 dB more for
   another 5 dB of light, Blackman 9 dB more again for 1.8 dB.  The knee is
   visible in the middle panel of the figure.

2. **This is apodization *versus* a wider guard gap, not a free improvement on
   it.**  The window keeps the sub-aperture's size, so it pays in throughput
   (-8.2 dB for Hann) and in a main lobe ~1.6x wider, which eats the detector
   guard from the other side.  The geometric table prints both costs before any
   light is propagated, so the comparison is on one set of axes.

**This example must use a wave propagator (``spot_mode=False``).**  The beamlet
kernel replaces the true aperture transform with a Gaussian/sinc envelope, so it
reproduces spot positions, powers and widths faithfully but *not* the wings --
and the wings are the entire subject here.  Run this on beamlets and it will
report, wrongly, that apodization does nothing.

    python examples/07_apodization.py
    python examples/07_apodization.py --save docs/figures
"""

import numpy as np

import _figure
import fsphotonix as fp
from fsphotonix import engine as eng
from fsphotonix.analysis import apodization_report, apodization_sweep
from fsphotonix.capacity import window_metrics
from fsphotonix.modulators import IdealModulator, ModulatorArray
from fsphotonix.modulators.geometry import WINDOWS

LAM, Z = 1550e-9, 60e-3
REGION, PITCH = 400e-6, 1.2e-3

#: A 3x3 lattice of detector sites on 4 mm centres.  This input drives the four
#: corners; the five remaining sites belong to neighbouring channels, so power
#: landing there is cross-talk they would actually suffer.
SITES = [(i * 4e-3, j * 4e-3) for j in (-1, 0, 1) for i in (-1, 0, 1)]
TARGETS = [SITES[0], SITES[2], SITES[6], SITES[8]]
LEAKS = [s for s in SITES if s not in TARGETS]

RADIUS = 1.2e-3    # detector half-aperture the sweep integrates over
CUT_Y = -4e-3      # the figure's far-field cut runs along the bottom target row
WINDOW_ORDER = ("none", "tukey", "hann", "blackman")


def panel(window):
    """A single region fanned out to the four corner sites, tapered by ``window``."""
    arr = ModulatorArray.tiled(nx=1, ny=1, pitch=PITCH, region_size=REGION,
                               device=IdealModulator(), background="absorb",
                               default_wavelength=LAM)
    arr.route(0, TARGETS, z=Z, tiling="grid", apodize=window)
    return arr


def build(window):
    """``build(window) -> OpticalSystem``, the closure ``apodization_sweep`` wants."""
    src = fp.LaserArray(n=1, positions=[(0.0, 0.0)], wavelength=LAM, power=1e-3,
                        profile=fp.Rectangular(width=REGION), linewidth=0.0,
                        rin_dbc_hz=-300.0)
    det = fp.PowerMeter(positions_in=[(0.0, 0.0)], diameter=30e-3)
    return fp.build_relay(source=src, modulator=panel(window), detector=det, z=Z,
                          panel_grid=fp.Grid.from_extent(900e-6, 900e-6, 1.0e-6),
                          screen_grid=fp.Grid.from_extent(24e-3, 24e-3, 20e-6),
                          # a wave propagator: beamlets cannot show wings at all
                          spot_mode=False)


def main() -> None:
    print("-- layout --")
    print(f"  {'wavelength [nm]':26s}{LAM * 1e9:>12.1f}")
    print(f"  {'throw [mm]':26s}{Z * 1e3:>12.1f}")
    print(f"  {'region [um]':26s}{REGION * 1e6:>12.1f}")
    print(f"  {'lattice sites':26s}{len(SITES):>12d}")
    print(f"  {'driven by this input':26s}{len(TARGETS):>12d}  "
          f"(corners, 4 mm centres)")
    print(f"  {'owned by neighbours':26s}{len(LEAKS):>12d}")

    print("\n-- what the window costs, geometrically (no light moved yet) --")
    reports = {w: apodization_report(panel(w), z=Z, wavelength=LAM)
               for w in WINDOW_ORDER}
    print(f"  {'window':12s}{'sub-ap um':>11s}{'thru':>8s}{'loss dB':>9s}"
          f"{'broaden':>9s}{'spot um':>9s}{'spot/pitch':>12s}")
    for w, r in reports.items():
        print(f"  {w:12s}{r['subaperture_um']:>11.1f}{r['power_throughput']:>8.3f}"
              f"{r['apodization_loss_db']:>9.2f}{r['mainlobe_broadening']:>9.3f}"
              f"{r['apodized_spot_size_um']:>9.1f}{r['spot_vs_target_pitch']:>12.3f}")
    print("  (the window does not shrink the sub-aperture -- it keeps its size and "
          "passes less of it)")

    print("\n-- what the window buys, measured by propagating light --")
    sweep = apodization_sweep(build, windows=WINDOW_ORDER,
                              target_positions=TARGETS, leak_positions=LEAKS,
                              radius=RADIUS)
    print(f"  {'window':12s}{'signal uW':>11s}{'leak uW':>10s}{'xtalk dB':>10s}"
          f"{'suppress dB':>13s}{'thru dB':>9s}")
    for row in sweep["rows"]:
        print(f"  {row['window']:12s}{row['signal_W'] * 1e6:>11.3f}"
              f"{row['leak_W'] * 1e6:>10.4f}{row['crosstalk_db']:>10.1f}"
              f"{row['crosstalk_suppression_db']:>13.1f}"
              f"{row['throughput_vs_hard_db']:>9.2f}")
    by_window = {r["window"]: r for r in sweep["rows"]}
    tuk, han = by_window["tukey"], by_window["hann"]
    print(f"  (the knee: tukey trades {-tuk['throughput_vs_hard_db']:.2f} dB of light "
          f"for {tuk['crosstalk_suppression_db']:.1f} dB of suppression; hann's next "
          f"{han['crosstalk_suppression_db'] - tuk['crosstalk_suppression_db']:.1f} dB "
          f"costs a further "
          f"{tuk['throughput_vs_hard_db'] - han['throughput_vs_hard_db']:.2f} dB)")

    print("\n-- the wings themselves, along the bottom target row --")
    cuts = {w: far_field_cut(w) for w in WINDOW_ORDER}
    probes = [0.0, 8e-3, 12e-3]
    print(f"  {'window':12s}" + "".join(f"{f'x={p * 1e3:.0f} mm':>14s}"
                                        for p in probes))
    for w in WINDOW_ORDER:
        x, I = cuts[w]
        peak = float(I.max())
        cells = "".join(f"{10 * np.log10(max(_envelope_at(x, I, p), 1e-300) / peak):>14.1f}"
                        for p in probes)
        print(f"  {w:12s}{cells}")
    print("  (dB below the main lobe, on the sidelobe envelope; x = 4 mm is a target "
          "and x = 0 mm a neighbour's site)")

    if _figure.wanted():
        make_figure(reports, sweep, cuts)


def far_field_cut(window: str):
    """``(x, irradiance)`` along ``y = CUT_Y`` out of the propagated screen field."""
    fields = build(window).run().final
    I = eng.to_numpy(fields.mean_intensity())
    x = eng.to_numpy(fields.grid.x())
    y = eng.to_numpy(fields.grid.y())
    return x, I[int(np.argmin(np.abs(y - CUT_Y)))]


def _lobe_cells(x) -> int:
    """Sidelobe spacing ``lambda z / a`` in grid cells -- one envelope window."""
    dx = float(x[1] - x[0])
    return max(int(round(LAM * Z / (REGION / 2) / dx)), 3)


def _envelope(I, n: int):
    """Upper envelope of an oscillating cut: rolling max over one sidelobe."""
    pad = np.pad(I, n // 2, mode="edge")
    view = np.lib.stride_tricks.sliding_window_view(pad, n)
    return view.max(axis=-1)[:len(I)]


def _envelope_at(x, I, x0: float) -> float:
    env = _envelope(I, _lobe_cells(x))
    return float(env[int(np.argmin(np.abs(x - x0)))])


def make_figure(reports, sweep, cuts) -> None:
    plt = _figure.pyplot()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    colour = {"none": "#c44", "tukey": "#2a7fbf", "hann": "#2f9e44",
              "blackman": "#8a4fbf"}

    # --- 1. what "tapering the rim" physically is --------------------------- #
    t = np.linspace(-0.5, 0.5, 601)
    for w in WINDOW_ORDER:
        a = np.asarray(eng.to_numpy(WINDOWS[w](t)), dtype=float)
        thru, broad = window_metrics(w)
        ax[0].plot(t, a, color=colour[w], lw=2.0 if w == "tukey" else 1.6,
                   label=f"{w}  ({10 * np.log10(thru):+.1f} dB, {broad:.2f}x)")
    ax[0].axhline(0.0, color="k", lw=0.8, alpha=0.4)
    for e in (-0.5, 0.5):
        ax[0].axvline(e, color="k", ls=":", lw=0.8, alpha=0.5)
    ax[0].set_xlim(-0.56, 0.56)
    ax[0].set_ylim(-0.32, 1.10)  # headroom under the curves for the legend
    ax[0].set_yticks(np.arange(0.0, 1.01, 0.2))
    ax[0].set_xlabel("position across the sub-aperture  [$a$]")
    ax[0].set_ylabel("field amplitude")
    ax[0].set_title("the window: zero at the rim, not merely small")
    ax[0].legend(fontsize=7, title="(throughput, main-lobe width)",
                 title_fontsize=7, loc="lower center")
    ax[0].grid(alpha=0.3)

    # --- 2. the measured trade, and where its knee is ----------------------- #
    rows = sweep["rows"]
    cost = np.array([-r["throughput_vs_hard_db"] for r in rows])
    gain = np.array([r["crosstalk_suppression_db"] for r in rows])
    ax[1].plot(cost, gain, "-", color="0.5", lw=1.2, zorder=1)
    for r, c, g in zip(rows, cost, gain):
        ax[1].plot(c, g, "o", ms=9, color=colour[r["window"]], zorder=3)
    # label each point away from the line, alternating side so none collide
    offsets = {"none": (0.45, 1.0), "tukey": (0.45, -1.0),
               "hann": (0.45, -1.0), "blackman": (-0.5, 3.0)}
    for r, c, g in zip(rows, cost, gain):
        dx, dy = offsets[r["window"]]
        price = "the hard edge" if r["window"] == "none" \
            else f"{g:.1f} dB for {c:.2f} dB"
        ax[1].annotate(f"{r['window']}\n{price}", (c, g),
                       xytext=(c + dx, g + dy), fontsize=7.5,
                       color=colour[r["window"]], va="top" if dy < 0 else "bottom",
                       ha="left" if dx > 0 else "right")
    ax[1].set_xlim(-1.4, 13.0)
    ax[1].set_ylim(-9.0, 56.0)
    ax[1].set_xlabel("throughput paid  [dB of light lost]")
    ax[1].set_ylabel("cross-talk suppression  [dB]")
    ax[1].set_title("the knee is at tukey: most of the win, little of the price")
    ax[1].grid(alpha=0.3)

    # --- 3. the mechanism: the wings collapsing ----------------------------- #
    # Each cut is normalised to its own main lobe, because cross-talk is a ratio:
    # what matters is how far the wings sit below the signal, not their absolute
    # level (the absolute loss is panel 2's axis).
    x = cuts["none"][0]
    n = _lobe_cells(x)
    floor = 1e-9
    for w in WINDOW_ORDER:
        xw, I = cuts[w]
        rel = np.maximum(I / I.max(), floor)
        ax[2].semilogy(xw * 1e3, rel, color=colour[w], lw=0.7, alpha=0.3)
        ax[2].semilogy(xw * 1e3, np.maximum(_envelope(rel, n), floor),
                       color=colour[w], lw=1.9, label=w)
    # the 1/u^2 asymptote of a hard edge, anchored on its own envelope
    env = _envelope(cuts["none"][1] / cuts["none"][1].max(), n)
    x0 = 6e-3
    v0 = float(env[int(np.argmin(np.abs(x - x0)))])
    far = x[x > 4.9e-3]
    ax[2].semilogy(far * 1e3,
                   v0 * ((far - 4e-3) / (x0 - 4e-3)) ** -2.0,
                   "k:", lw=1.3, alpha=0.8, label="$\\propto 1/u^2$")
    for s in (-4, 0, 4, 8, 12):
        ax[2].axvline(s, color="0.75", lw=0.8, ls="--", zorder=0)
    ax[2].text(0.0, 3.0, "neighbour's\nsite", fontsize=7, ha="center",
               va="bottom", color="0.35")
    ax[2].text(4.0, 3.0, "target", fontsize=7, ha="center", va="bottom",
               color="0.35")
    ax[2].set_xlim(-1.0, 12.0)
    ax[2].set_ylim(floor, 60.0)
    ax[2].set_xlabel(f"x on the screen along y = {CUT_Y * 1e3:.0f} mm  [mm]")
    ax[2].set_ylabel("irradiance / own main lobe")
    ax[2].set_title("why: the $1/u^2$ sinc tail is what the guard cannot outrun")
    ax[2].legend(fontsize=7.5, ncol=2, loc="upper right")
    ax[2].grid(which="both", alpha=0.25)

    _figure.finish(fig, "07_apodization")


if __name__ == "__main__":
    main()
