"""CCG scaling: how much optics does a WCSP of a given size actually need?

A weighted constraint satisfaction problem is not what an optical annealer sees.
The route to the machine runs WCSP -> constraint composite graph -> minimum-weight
vertex cover -> QUBO -> Ising, and every step but the first leaves the problem
size alone except the CCG, which inflates it.  So the component budget is set by
the CCG, not by the WCSP the user wrote down, and this example sizes the hardware
along that route instead of along the problem statement.

``ccg_wcsp_graph(n)`` builds the CCG of a boolean WCSP with a unary table per
variable and a binary table per pair: one vertex per variable, one auxiliary
vertex per table, one edge per (table, variable) incidence.  That gives
``N = 2n + n(n-1)/2`` vertices, ``E = n + n(n-1)`` edges and variable-vertex
degree ``n``.  The topology is exact; **the edge weights are representative, not
reduced from real cost tables** -- they are drawn from the span a real
MWVC-derived QUBO uses (``Q_ij = min(w_i, w_j) + 1``, so 1.1 to 1.7 for the
anchor instance below), because the weight assignment depends on how each table
is decomposed while the sizing depends on the sparsity and the degree, both of
which are exact.

The four results worth reading off the output:

1. **The CCG is the cost driver, and it is cubic.**  A 10-variable WCSP is a
   65-variable QUBO, because every one of the 45 binary tables adds its own
   vertex.  Worse, the variable vertices have degree ``n``, so the panel pays
   twice: ``2N`` illuminated regions *and* ``n`` sub-apertures inside each one.
   Hologram count goes as ``2N x n ~ n^3`` -- 54 at three variables, 132500 at
   fifty.

2. **Ten WCSP variables already needs a panel nobody sells.**  Hard-edged area
   division asks for 17.4 Mpx at ``n = 10``, against about 8 Mpx for the largest
   commercial SLMs (3840 x 2160 and up).  Five variables fit with room to
   spare at 0.85 Mpx; ten do not fit at all.  The crossing is between the two,
   and the first panel of the figure marks it.  This is the useful engineering
   conclusion of the whole example: on this architecture the interesting WCSPs
   are a panel-fabrication problem before they are an annealing problem.

3. **Apodization buys back a factor of four to five of panel, and at the top of
   the sweep it is the difference between a design and no design.**  A Tukey
   window on each sub-aperture suppresses the sidelobe that lands on the
   neighbouring detector, which lets the search close the spot guard and shrink
   the sub-aperture: 0.195 to 0.228 of the hard-edged pixel count at every size
   where both are designs.  At
   ``n = 50`` the hard-edged design misses the 4-bit target outright (3.59 bits,
   binding constraint ``accuracy_floor``) while the apodized one still makes
   4.17.  It is not free -- the same window costs throughput, and total optical
   power roughly doubles -- but power is a budget and accuracy is correctness.

4. **Lasers and detectors coincide; holograms do not follow either.**  The CCG's
   Ising matrix is square and dual-railed on both sides, so emitter count and
   detector count are both ``2N`` and the two lines of the second panel lie on
   top of each other (drawn with different markers so both are visible).  The
   hologram count carries the extra factor of the degree, and the divergence
   between the two is the whole reason the panel, not the laser bar, is what
   runs out first.

The third panel takes the *why* of the first: the panel is a product of three
growing factors -- regions, sub-apertures per region, and pixels per hologram --
and on a log axis a product is a sum of heights.  Hologram count and
pixels-per-hologram are plotted against the total they multiply to, which reads
more clearly than the power curve does, since power is monotone and
unsurprising while this shows two independent explosions stacking.

    python examples/09_ccg_scaling.py
    python examples/09_ccg_scaling.py --save docs/figures
"""

import numpy as np

import _figure
from fsphotonix.capacity import (
    Features,
    ProblemSpec,
    Requirements,
    baseline,
    estimate,
)
from fsphotonix.solvers.aim import ccg_wcsp_graph

#: WCSP variable counts to size.  ``estimate`` searches a grid of throws, output
#: pitches and spot guards per call, so the sweep is kept to the sizes that carry
#: the argument: the anchor, the last one that fits a real panel, and the three
#: that show the cubic.
SIZES = (3, 5, 10, 20, 30, 50)

#: The representative machine whose bill of materials gets printed in full.  Ten
#: is the interesting one: the smallest size in the sweep that no commercial
#: panel can hold.
SHOWN = 10

TARGET_BITS = 4.0
ITERATIONS = 300

#: Largest SLM you can buy as of writing, in megapixels: the 4K-class phase
#: panels are 4160 x 2464 or thereabouts.  Drawn as a line in the figure because
#: the crossing point is the engineering answer.
COMMERCIAL_MPX = 8.0

#: The hard-edged reference and the apodized variant, both area division.
HARD = Features(routing="subregions")
APOD = Features(routing="subregions", apodization="tukey")


def anchor() -> dict:
    """Check ``ccg_wcsp_graph(3)`` against a real reduced instance.

    The instance: a 3-variable boolean WCSP with a unary table per variable and
    a binary table per pair, pushed through CCG -> MWVC -> QUBO by hand.  Six
    tables become six auxiliary vertices, which with the three variables makes
    nine QUBO variables, and the nine (table, variable) incidences become nine
    couplers.  Each variable sits in three tables, so its vertex has degree
    three.

    Asserting it here is the point.  The construction below is otherwise just a
    formula; this is the one size where the formula can be held against
    something that was actually reduced.
    """
    A = ccg_wcsp_graph(3)
    nz = A != 0
    got = {"vertices": int(A.shape[0]), "edges": int(nz.sum()) // 2,
            "max_degree": int(nz.sum(axis=1).max()),
            "weight_lo": float(A[nz].min()), "weight_hi": float(A[nz].max())}
    want = {"vertices": 9, "edges": 9, "max_degree": 3}
    for k, v in want.items():
        assert got[k] == v, f"anchor mismatch: {k} = {got[k]}, expected {v}"
    assert 1.1 <= got["weight_lo"] <= got["weight_hi"] <= 1.7, got
    return got | want


def structure(n: int) -> dict:
    """Closed-form CCG size, for checking the built graph against the formula."""
    return {"vertices": 2 * n + n * (n - 1) // 2, "edges": n + n * (n - 1),
            "degree": n}


def size_rows() -> list[dict]:
    """One row per WCSP size: the CCG, then the machine it needs, twice."""
    rows = []
    for n in SIZES:
        A = ccg_wcsp_graph(n)
        nz = A != 0
        want = structure(n)
        assert A.shape[0] == want["vertices"], (n, A.shape, want)
        assert int(nz.sum()) // 2 == want["edges"], (n, int(nz.sum()) // 2, want)
        assert int(nz[:n].sum(axis=1).max()) == want["degree"], (n, want)

        problem = ProblemSpec.from_graph(A, iterations=ITERATIONS, label=f"ccg-{n}")
        req = Requirements(target_bits=TARGET_BITS)
        rows.append({"n": n, "problem": problem,
                     "hard": baseline(problem, req),
                     "apod": estimate(problem, req, APOD)})
    return rows


def main() -> None:
    a = anchor()
    print("-- the anchor: one real 3-variable WCSP, reduced by hand --")
    print("  6 tables (3 unary + 3 binary) -> 6 auxiliary vertices + 3 variable")
    print("  vertices, and 9 (table, variable) incidences -> 9 couplers.")
    print(f"  {'quantity':26s}{'ccg_wcsp_graph(3)':>20s}{'reduced instance':>20s}")
    for k in ("vertices", "edges", "max_degree"):
        print(f"  {k:26s}{a[k]:>20d}{a[k]:>20d}")
    span = f"{a['weight_lo']:.2f}-{a['weight_hi']:.2f}"
    print(f"  {'coupler weights':26s}{span:>20s}{'1.10-1.70':>20s}")
    print("  (asserted in the example, so the construction is anchored to an")
    print("   instance that was reduced rather than to a formula alone)")

    rows = size_rows()

    print("\n-- the CCG, not the WCSP, is what the machine holds --")
    print(f"  {'WCSP n':>7s}{'QUBO N':>8s}{'couplers':>10s}{'fan':>6s}"
          f"{'density':>9s}{'N / n':>8s}")
    for r in rows:
        p, s = r["problem"], structure(r["n"])
        print(f"  {r['n']:>7d}{p.n_in:>8d}{s['edges']:>10d}{p.max_fanout:>6d}"
              f"{p.density:>9.3f}{p.n_in / r['n']:>8.1f}")
    print("  (N = 2n + n(n-1)/2: a unary table per variable and a binary table per")
    print("   pair, each table its own vertex.  The fan-out is the variable degree n)")

    print(f"\n-- hardware per WCSP size "
          f"(dual rail, area division, {TARGET_BITS:.0f} bits) --")
    print(f"  {'WCSP n':>7s}{'QUBO N':>8s}{'fan':>5s}{'lasers':>8s}{'detect':>8s}"
          f"{'CGHs':>9s}{'px/CGH':>9s}{'SLM Mpx':>11s}{'panel mm':>15s}"
          f"{'power mW':>11s}  binding")
    for r in rows:
        p, h = r["problem"], r["hard"]
        flag = "" if h.feasible else "  <-- INFEASIBLE"
        print(f"  {r['n']:>7d}{p.n_in:>8d}{p.max_fanout:>5d}{h.lasers:>8d}"
              f"{h.detectors:>8d}{h.cgh_count:>9d}{h.pixels_per_cgh:>9d}"
              f"{h.megapixels:>11.2f}"
              f"{f'{h.slm_extent_x * 1e3:.1f}x{h.slm_extent_y * 1e3:.1f}':>15s}"
              f"{h.optical_power_total * 1e3:>11.1f}  {h.binding}{flag}")
    over = [r["n"] for r in rows if r["hard"].megapixels > COMMERCIAL_MPX]
    under = [r["n"] for r in rows if r["hard"].megapixels <= COMMERCIAL_MPX]
    print(f"  (the largest commercial SLM is about {COMMERCIAL_MPX:.0f} Mpx, so "
          f"n <= {max(under)} fits a panel you")
    print(f"   can buy and n >= {min(over)} does not.  Mpx goes as CGHs x px/CGH, "
          f"and both columns grow)")

    print("\n-- what a Tukey window on each sub-aperture buys back --")
    print(f"  {'WCSP n':>7s}{'hard Mpx':>11s}{'apod Mpx':>11s}{'ratio':>8s}"
          f"{'hard bits':>11s}{'apod bits':>11s}{'power x':>9s}  apodized binding")
    for r in rows:
        h, ap = r["hard"], r["apod"]
        print(f"  {r['n']:>7d}{h.megapixels:>11.2f}{ap.megapixels:>11.2f}"
              f"{ap.megapixels / h.megapixels:>8.3f}"
              f"{h.achievable_bits:>11.2f}{ap.achievable_bits:>11.2f}"
              f"{ap.optical_power_total / h.optical_power_total:>9.2f}  {ap.binding}")
    feas = [r for r in rows if not r["hard"].feasible and r["apod"].feasible]
    print("  (the window suppresses the sidelobe that lands on the neighbouring")
    print("   detector, so the search can close the spot guard -- a factor of four to")
    print("   five of panel, paid for in throughput.  The ratio relaxes at the top of")
    if feas:
        n = feas[0]["n"]
        print(f"   the sweep only because the n = {n} hard-edged row is no longer a")
        print(f"   design at all: it stalls at {feas[0]['hard'].achievable_bits:.2f} "
              f"bits against {TARGET_BITS:.0f} requested, and apodization")
        print("   is what keeps that size feasible)")
    else:
        print("   the sweep as the binding constraint changes)")

    shown = next(r for r in rows if r["n"] == SHOWN)
    print(f"\n-- one representative machine: WCSP n = {SHOWN}, "
          f"QUBO N = {shown['problem'].n_in} --")
    print(shown["hard"].components())
    print(f"\n{shown['hard'].table()}")

    if _figure.wanted():
        make_figure(rows)


def make_figure(rows) -> None:
    plt = _figure.pyplot()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    n = np.array([r["n"] for r in rows], dtype=float)
    hard = [r["hard"] for r in rows]
    apod = [r["apod"] for r in rows]
    bad = "#c44"

    def mark_infeasible(axis, reports, values):
        """Red ring on any design the accuracy floor rules out, as 06 does."""
        for rep, v in zip(reports, values):
            if not rep.feasible:
                axis.plot([rep_n(rep, rows)], [v], "o", ms=11, mfc="none",
                          mec=bad, mew=1.8, zorder=5)

    # --- 1. the panel, against what you can buy ----------------------------- #
    ax[0].semilogy(n, [r.megapixels for r in hard], "o-", color="#2a7fbf", lw=1.5,
                   ms=6, label="hard-edged sub-apertures")
    ax[0].semilogy(n, [r.megapixels for r in apod], "s-", color="#2f9e44", lw=1.5,
                   ms=6, label="Tukey-apodized")
    mark_infeasible(ax[0], hard, [r.megapixels for r in hard])
    mark_infeasible(ax[0], apod, [r.megapixels for r in apod])
    ax[0].axhspan(COMMERCIAL_MPX / 2.0, COMMERCIAL_MPX, color="0.75", alpha=0.45,
                  zorder=0)
    ax[0].axhline(COMMERCIAL_MPX, color="k", ls="--", lw=1.2, zorder=1)
    ax[0].text(n[-1], COMMERCIAL_MPX * 1.3,
               f"largest commercial SLM ~{COMMERCIAL_MPX:.0f} Mpx",
               fontsize=7.5, va="bottom", ha="right")
    # where the hard-edged curve crosses the band is the engineering answer
    cross = next((i for i, r in enumerate(hard)
                  if r.megapixels > COMMERCIAL_MPX), None)
    if cross:
        lo, hi = hard[cross - 1], hard[cross]
        ax[0].annotate(f"n={n[cross - 1]:.0f} fits ({lo.megapixels:.2f} Mpx),\n"
                       f"n={n[cross]:.0f} does not ({hi.megapixels:.1f} Mpx)",
                       xy=(n[cross], hi.megapixels),
                       xytext=(n[cross] + 5.0, hi.megapixels / 90.0),
                       fontsize=7.5, color="0.25",
                       arrowprops=dict(arrowstyle="->", color="0.5", lw=0.9))
    ax[0].text(0.02, 0.97, "red ring: ruled out by the accuracy floor",
               transform=ax[0].transAxes, fontsize=7, color=bad, va="top")
    ax[0].set_xlabel("WCSP variables  $n$")
    ax[0].set_ylabel("SLM size [Mpx]")
    ax[0].set_title("ten WCSP variables needs a panel nobody sells")
    ax[0].legend(fontsize=7.5, loc="lower right")
    ax[0].grid(True, which="both", alpha=0.3)

    # --- 2. component counts, and where they part company ------------------- #
    # lasers and detectors are both 2N -- the CCG's Ising matrix is square and
    # dual-railed on both sides -- so the two lines coincide exactly.  Drawn with
    # a wide hollow marker under a small filled one so neither hides.
    ax[1].semilogy(n, [r.lasers for r in hard], "o-", color="#2a7fbf", lw=1.5,
                   ms=11, mfc="none", mew=1.6, label="lasers / emitters  ($2N$)")
    ax[1].semilogy(n, [r.detectors for r in hard], "s-", color="#8e44ad", lw=1.0,
                   ms=5, label="detectors  ($2N$, identical)")
    ax[1].semilogy(n, [r.cgh_count for r in hard], "^-", color="#d9821b", lw=1.6,
                   ms=6, label="holograms  ($2N \\times$ fan)")
    for p, style, lab in ((2.0, "k:", "$\\propto n^2$"),
                          (3.0, "k-.", "$\\propto n^3$")):
        ref = hard[0].lasers * (n / n[0]) ** p
        ax[1].semilogy(n, ref, style, alpha=0.45, lw=1.0, label=lab)
    # the gap between sources and holograms at the top of the sweep, drawn as the
    # factor it is: the variable degree
    gap_lo, gap_hi = hard[-1].lasers, hard[-1].cgh_count
    ax[1].annotate("", xy=(n[-1], gap_lo), xytext=(n[-1], gap_hi),
                   arrowprops=dict(arrowstyle="<->", color="0.35", lw=1.0))
    ax[1].text(n[-1] - 1.5, (gap_lo * gap_hi) ** 0.5,
               f"$\\times${n[-1]:.0f}: the panel runs out\nbefore the laser bar "
               f"does", fontsize=7.5, color="0.25", ha="right", va="center")
    ax[1].set_xlabel("WCSP variables  $n$")
    ax[1].set_ylabel("count per machine")
    ax[1].set_title("sources scale with $N$, holograms with $N \\times$ degree")
    ax[1].legend(fontsize=7, loc="upper left")
    ax[1].grid(True, which="both", alpha=0.3)

    # --- 3. why: two explosions multiplying ---------------------------------- #
    # On a log axis a product is a sum of heights, so plotting both factors under
    # the total they make shows the panel blow-up decomposed rather than asserted.
    ax[2].semilogy(n, [r.cgh_count for r in hard], "^-", color="#d9821b", lw=1.6,
                   ms=6, label="holograms on the panel")
    ax[2].semilogy(n, [r.pixels_per_cgh for r in hard], "v-", color="#2f9e44",
                   lw=1.6, ms=6, label="pixels per hologram")
    total = np.array([r.megapixels * 1e6 for r in hard])
    ax[2].semilogy(n, total, "o--", color="#2a7fbf", lw=1.8, ms=6,
                   label="total panel pixels (their product)")
    mark_infeasible(ax[2], hard, list(total))
    # the two ends of the product, written out.  The first label is pushed
    # sideways into the empty band between the factors and their product rather
    # than stacked on the curve it describes
    def product_label(i: int) -> str:
        mpx = (f"{hard[i].megapixels:,.0f}" if hard[i].megapixels >= 10
               else f"{hard[i].megapixels:.2f}")
        return (f"{hard[i].cgh_count} $\\times$ {hard[i].pixels_per_cgh}"
                f"\n= {mpx} Mpx")

    ax[2].annotate(product_label(0), xy=(n[0], total[0]), xytext=(26, 0),
                   textcoords="offset points", fontsize=7, color="0.25",
                   ha="left", va="center",
                   arrowprops=dict(arrowstyle="->", color="0.5", lw=0.8))
    ax[2].annotate(product_label(len(n) - 1), xy=(n[-1], total[-1]),
                   xytext=(0, 15), textcoords="offset points", fontsize=7,
                   color="0.25", ha="right", va="bottom")
    ax[2].set_ylim(10 ** 1.2, 10 ** 12.0)
    ax[2].set_xlabel("WCSP variables  $n$")
    ax[2].set_ylabel("count")
    ax[2].set_title("the panel is a product, and both factors grow")
    ax[2].legend(fontsize=7.5, loc="lower right")
    ax[2].grid(True, which="both", alpha=0.3)

    _figure.finish(fig, "09_ccg_scaling")


def rep_n(report, rows) -> float:
    """The WCSP size a report was built for -- the label carries it."""
    return float(next(r["n"] for r in rows
                      if r["hard"] is report or r["apod"] is report))


if __name__ == "__main__":
    main()
