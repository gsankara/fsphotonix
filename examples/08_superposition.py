"""Phase multiplexing: what the ``arg()`` of a sum actually costs.

One physical sub-region can carry N holograms at once.  Sum their complex
transmittances and keep only the phase, and the single element steers light to
every target of every term -- which is how a dense matrix column gets written on
an area that only pays for one.  The catch is the projection.  A sum of N
phase-only terms is *not* phase-only: it has an amplitude, and throwing that
amplitude away is a non-linear operation that scatters power into
intermodulation orders and speckle.  This example measures how much.

Until recently the code would not admit any of it.  ``Superposition.orders()``
apportioned each constituent's analytic orders by its power share, and since a
``SpotFocuser`` reports unit efficiency the shares summed to exactly 1.000 with
0.000 stray -- a phase-only superposition claiming to lose nothing to the one
mechanism it certainly loses to.  ``orders()`` now rasterises the hologram,
transforms it and integrates the power that actually arrives at each designed
direction, so every number below is measured.

The four results worth reading off the output:

1. **The loss is 25% and it was being reported as zero.**  Every row of the
   first table used to read ``total = 1.000, stray = 0.000``.  Measured, a
   9-target multiplex delivers 0.748 of its light to the targets it was asked
   for and scatters 0.252 elsewhere.  The first panel of the figure puts the old
   claim on the same axis as the measurement.

2. **Random carrier phases are not cosmetic.**  Without them every term peaks
   together at the aperture centre, the sum has a huge peak-to-average ratio,
   and the ``arg()`` projection throws away wildly different amounts of each
   term: 0.540 total at n=4 against 0.753 once the carriers are decorrelated.
   That is why both :class:`Superposition` and :class:`MultiSpotHologram` draw
   them, seeded so a design stays reproducible.

3. **WGS wins on uniformity, not on throughput.**  Across the fan-out sweep all
   three synthesis methods land within a few points of each other on efficiency
   (0.71-0.80), but the rms spread of the *delivered* weights separates them by
   an order of magnitude: 0.407 for the analytic sum at fan 25 against 0.051 for
   weighted Gerchberg-Saxton.  In a matrix-vector product every target is an
   intended output, so that spread is the coefficient error -- under two bits
   for the analytic sum, over four for WGS.  Throughput is a power budget;
   uniformity is correctness.

4. **Gerchberg-Saxton quantises target placement; the analytic sum does not.**
   GS optimises on its own transform's lattice, whose bins are spaced
   ``1/aperture`` -- one spot width.  A target half a spot off that lattice can
   only be approximated by the two bins straddling it, and the efficiency drops
   (0.795 -> 0.690 at fan 9).  The analytic superposition sets each target by
   exact phase and is flat to a few parts in a thousand across the same sweep.
   That is the trade the third panel shows, and it is why the method comparison
   above is run on a lattice deliberately aligned to GS's own.

    python examples/08_superposition.py
    python examples/08_superposition.py --save docs/figures
"""

import math

import numpy as np

import _figure
from fsphotonix.modulators.gratings.cgh import MultiSpotHologram, SpotTarget
from fsphotonix.modulators.gratings.composite import Superposition
from fsphotonix.modulators.gratings.lenses import SpotFocuser

LAM, Z, APERTURE = 1550e-9, 0.2, 250e-6
SPOT = LAM * Z / APERTURE  # one spot width on the screen: lambda z / a
PITCH = 3.0 * SPOT         # target lattice pitch, in spot widths

#: Measurement sampling.  192 across the aperture is plenty: ``measure_orders``
#: zero-pads by 4x, so the frequency bins are a quarter of a spot wide however
#: coarse this is, and the integration window is specified in spot widths.
SAMPLES, GS_SAMPLES, ITERATIONS, SEED = 192, 192, 60, 0

FANS = (4, 9, 16, 25, 36)
METHODS = ("superposition", "gs", "wgs")
OFFSETS = (0.0, 0.25, 0.5, 0.75, 1.0)  # sub-lattice shift, in spot widths
OFFSET_FAN = 9


def lattice(n: int, offset: float = 0.0):
    """``n`` targets on a near-square lattice of :data:`PITCH`, shifted in x.

    The lattice is centred on a *site* rather than on the block's centroid: for
    an odd number of columns the two agree, and for an even number this keeps
    every target on an integer multiple of the pitch instead of a half-integer
    one.  Since the pitch is a whole number of spot widths, that puts the entire
    fan on Gerchberg-Saxton's own transform lattice -- so the method comparison
    is not secretly also measuring the placement penalty that ``offset``
    isolates.
    """
    cols = int(math.ceil(math.sqrt(n)))
    c0 = (cols - 1) // 2
    return [(float((c - c0) * PITCH + offset), float((r - c0) * PITCH))
            for r, c in (divmod(i, cols) for i in range(n))]


def multiplex(n: int, randomize: bool = True) -> Superposition:
    """One sub-region carrying ``n`` focusing holograms, summed and phase-only."""
    return Superposition(
        gratings=[SpotFocuser.from_target(x, y, Z, LAM) for x, y in lattice(n)],
        weights=np.ones(n), aperture_size=(APERTURE, APERTURE),
        randomize_phases=randomize, seed=SEED)


def hologram(targets, method: str) -> MultiSpotHologram:
    """The same fan-out synthesised by ``method`` instead of summed analytically."""
    return MultiSpotHologram(
        targets=[SpotTarget(x, y, Z) for x, y in targets], range_z=Z,
        aperture_size=(APERTURE, APERTURE), method=method,
        iterations=ITERATIONS, gs_samples=GS_SAMPLES, seed=SEED)


def delivered(h) -> tuple[float, float]:
    """``(total efficiency, rms weight non-uniformity)`` of a measured fan-out."""
    e = np.array([o.efficiency for o in h.orders(LAM, samples=SAMPLES)], dtype=float)
    if e.size == 0 or e.mean() <= 0:
        return 0.0, 0.0
    return float(e.sum()), float(e.std() / e.mean())


def carrier_rows() -> list[dict]:
    """The honesty table: what the projection costs, with and without carriers."""
    out = []
    for n in (4, 9):
        for randomize in (False, True):
            s = multiplex(n, randomize)
            total, _ = delivered(s)
            out.append({"n": n, "carrier": "random" if randomize else "aligned",
                        "total": total,
                        "stray": s.stray_fraction(LAM, samples=SAMPLES),
                        "uniformity": s.uniformity(LAM, samples=SAMPLES),
                        "resolved": s.resolved(LAM, samples=SAMPLES)})
    return out


def method_rows() -> dict[str, list[tuple[float, float]]]:
    """``{method: [(efficiency, uniformity) per fan]}`` across :data:`FANS`."""
    return {m: [delivered(hologram(lattice(f), m)) for f in FANS] for m in METHODS}


def offset_rows() -> dict[str, list[float]]:
    """``{method: [efficiency per sub-lattice offset]}`` at :data:`OFFSET_FAN`."""
    return {m: [delivered(hologram(lattice(OFFSET_FAN, o * SPOT), m))[0]
                for o in OFFSETS] for m in METHODS}


def main() -> None:
    print("-- layout --")
    print(f"  {'wavelength [nm]':28s}{LAM * 1e9:>12.1f}")
    print(f"  {'throw [mm]':28s}{Z * 1e3:>12.1f}")
    print(f"  {'sub-region aperture [um]':28s}{APERTURE * 1e6:>12.1f}")
    print(f"  {'spot width [um]':28s}{SPOT * 1e6:>12.1f}  (lambda z / a)")
    print(f"  {'target pitch [spots]':28s}{PITCH / SPOT:>12.1f}")
    print(f"  {'measurement samples':28s}{SAMPLES:>12d}")

    print("\n-- the projection's cost, and the carrier phases that reduce it --")
    carrier = carrier_rows()
    print(f"  {'n':>3s}{'carrier':>10s}{'total eff':>12s}{'stray':>9s}"
          f"{'rms unif':>10s}{'resolved':>10s}")
    for r in carrier:
        print(f"  {r['n']:>3d}{r['carrier']:>10s}{r['total']:>12.3f}"
              f"{r['stray']:>9.3f}{r['uniformity']:>10.3f}"
              f"{str(r['resolved']):>10s}")
    print("  (every one of these rows used to read total = 1.000, stray = 0.000: "
          "orders()")
    print("   apportioned the constituents' analytic shares instead of measuring "
          "the sum)")

    print("\n-- synthesis method, on a lattice aligned to GS's own --")
    methods = method_rows()
    print(f"  {'fan':>5}" + "".join(f"{m:>22s}" for m in METHODS))
    for i, f in enumerate(FANS):
        cells = "".join(f"{methods[m][i][0]:>10.3f} /{methods[m][i][1]:>10.3f}"
                        for m in METHODS)
        print(f"  {f:>5}{cells}")
    print("  (efficiency / rms weight non-uniformity.  The methods are within a few "
          "points on")
    print("   efficiency and an order of magnitude apart on uniformity -- which is "
          "the one that")
    print("   is a coefficient error rather than a power budget)")

    print(f"\n-- where GS can put a target: fan {OFFSET_FAN}, lattice shifted in x --")
    offs = offset_rows()
    print(f"  {'offset [spots]':>16s}" + "".join(f"{m:>16s}" for m in METHODS))
    for i, o in enumerate(OFFSETS):
        print(f"  {o:>16.2f}" + "".join(f"{offs[m][i]:>16.3f}" for m in METHODS))
    dip = min(range(len(OFFSETS)), key=lambda i: offs["wgs"][i])
    print(f"  (GS's transform bins are spaced 1/aperture = one spot width, so a "
          f"target")
    print(f"   {OFFSETS[dip]:.2f} spots off that lattice can only be straddled: wgs "
          f"drops to {offs['wgs'][dip]:.3f}")
    print(f"   from {offs['wgs'][0]:.3f}, while the analytic sum sets its targets by "
          f"exact phase and")
    print(f"   stays within {max(offs['superposition']) - min(offs['superposition']):.3f} "
          f"across the whole sweep)")

    if _figure.wanted():
        make_figure(carrier, methods, offs)


def make_figure(carrier, methods, offs) -> None:
    plt = _figure.pyplot()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    colour = {"superposition": "#c44", "gs": "#2f9e44", "wgs": "#2a7fbf"}

    # --- 1. the honesty fix ------------------------------------------------- #
    # The old order table is a flat line at 1.0 with nothing stray; the bars are
    # what the same holograms actually deliver.  Drawing both on one axis is the
    # whole point of the panel.
    x = np.arange(len(carrier))
    w = 0.36
    eff = [r["total"] for r in carrier]
    stray = [r["stray"] for r in carrier]
    ax[0].bar(x - w / 2, eff, w, color="#2a7fbf", label="to the targets (measured)")
    ax[0].bar(x + w / 2, stray, w, color="#c44",
              label="stray: intermodulation + speckle")
    for xi, (e, s) in enumerate(zip(eff, stray)):
        # inside the bars, so the gap arrows above them stay readable
        ax[0].text(xi - w / 2, e - 0.045, f"{e:.3f}", ha="center", fontsize=7.5,
                   color="white")
        ax[0].text(xi + w / 2, s + 0.02, f"{s:.3f}", ha="center", fontsize=7.5,
                   color="#8e2a2a")
        # the gap the old table denied
        ax[0].annotate("", xy=(xi - w / 2, 1.0), xytext=(xi - w / 2, e),
                       arrowprops=dict(arrowstyle="<->", color="0.35", lw=0.9))
    ax[0].axhline(1.0, color="k", ls="--", lw=1.4)
    ax[0].text(len(carrier) - 0.5, 1.015,
               "what orders() used to report: total 1.000, stray 0.000",
               ha="right", va="bottom", fontsize=8, style="italic")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([f"n={r['n']}\n{r['carrier']}" for r in carrier],
                          fontsize=8)
    ax[0].set_ylim(0.0, 1.34)
    ax[0].set_ylabel("fraction of incident power")
    ax[0].set_title("a phase-only sum that claimed to lose nothing")
    ax[0].legend(fontsize=7.5, loc="upper left")
    ax[0].grid(axis="y", alpha=0.3)

    # --- 2. efficiency against uniformity, across fan-out ------------------- #
    # Horizontal spread is throughput, vertical spread is coefficient error: the
    # methods are stacked vertically and side by side, which is the result.
    floor = 1e-3
    # the three clusters sit close together, so each method's fan labels are
    # pushed to a different side of its markers rather than piling up on them
    # ``flip`` alternates a method's labels to either side of its markers:
    # converged WGS lands two of its fans on practically the same point, and
    # their labels would otherwise merge into one unreadable blob
    where = {"superposition": ((5.0, 2.0), "left", "bottom", False),
             "gs": ((-5.0, 2.5), "right", "bottom", False),
             "wgs": ((5.0, -3.0), "left", "top", True)}
    for m in METHODS:
        e = [v[0] for v in methods[m]]
        u = [max(v[1], floor) for v in methods[m]]
        ax[1].semilogy(e, u, "o-", color=colour[m], lw=1.3, ms=6, label=m)
        (dx, dy), ha, va, flip = where[m]
        for i, (f, ei, ui) in enumerate(zip(FANS, e, u)):
            left = flip and i % 2 == 1
            ax[1].annotate(f"{f}", (ei, ui), xytext=(-dx if left else dx, dy),
                           textcoords="offset points", fontsize=7,
                           color=colour[m], ha="right" if left else ha, va=va)
    for bits, name in ((2.0, "2 bits"), (4.0, "4 bits")):
        ax[1].axhline(2.0 ** -bits, color="k", ls=":", alpha=0.6)
        ax[1].text(0.702, 2.0 ** -bits * 1.12, name, fontsize=7)
    ax[1].set_xlim(0.700, 0.830)
    ax[1].set_ylim(0.018, 0.62)
    ax[1].set_xlabel("total efficiency  (point labels are fan-out)")
    ax[1].set_ylabel("rms weight non-uniformity")
    ax[1].set_title("WGS's win is vertical: the weights, not the light")
    ax[1].legend(fontsize=7.5, loc="lower left")
    ax[1].grid(which="both", alpha=0.25)

    # --- 3. GS's placement lattice ------------------------------------------ #
    for m in METHODS:
        ax[2].plot(OFFSETS, offs[m], "o-", color=colour[m], lw=1.6, ms=6, label=m)
    ax[2].axvline(0.5, color="0.75", lw=0.9, ls="--", zorder=0)
    lo = min(offs["wgs"])
    ax[2].annotate("half a spot off GS's\ntransform lattice",
                   xy=(0.5, lo), xytext=(0.5, lo - 0.045),
                   fontsize=7.5, ha="center", va="top", color="0.3",
                   arrowprops=dict(arrowstyle="->", color="0.5", lw=0.9))
    for o, side in ((0.0, "left"), (1.0, "right")):
        ax[2].text(o, max(offs["wgs"]) + 0.004, "on lattice", fontsize=7,
                   ha=side, va="bottom", color="0.35")
    ax[2].set_xticks(list(OFFSETS))
    ax[2].set_xlim(-0.08, 1.08)
    ax[2].set_ylim(lo - 0.075, max(max(v) for v in offs.values()) + 0.03)
    ax[2].set_xlabel("sub-lattice offset of every target  [spot widths]")
    ax[2].set_ylabel("total efficiency")
    ax[2].set_title(f"GS quantises placement, the analytic sum does not "
                    f"(fan {OFFSET_FAN})")
    ax[2].legend(fontsize=7.5, loc="lower left")
    ax[2].grid(alpha=0.3)

    _figure.finish(fig, "08_superposition")


if __name__ == "__main__":
    main()
