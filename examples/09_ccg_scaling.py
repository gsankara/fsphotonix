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

5. **What fits on a real panel is not panel pixels over pixels per hologram.**
   Pixels per CGH is not a constant -- 1389 at ``n = 3``, 90472 at ``n = 30`` --
   because the sub-aperture has to grow as the screen does, so dividing a fixed
   panel by a fixed per-CGH figure badly overstates what it holds.  Sizing each
   ``n`` at the panel's own pitch instead: a 16K-class panel at 3 um
   (15360 x 8640, 133 Mpx) holds ``n = 12`` and not ``n = 14``, which wants
   about 11100 x 10400.  **16:9 is the wrong shape** -- the region grid is near
   square, so a *square* panel with the same 133 Mpx (11520 x 11520) holds
   ``n = 14`` where the 16:9 one does not; the 4K-class panel fails ``n = 8``
   the same way, with pixels to spare and too few rows.  And **pitch buys
   reach as well as pixels**: steering is ``arcsin(lambda / 2p)`` -- 29.8,
   15.0, 11.2 and 5.6 deg at 1.56, 3, 4 and 8 um -- so a coarser panel needs
   more pixels for the same problem (``n = 14``: 86, 116, 146 Mpx at 1.56, 3,
   4 um), and at 8 um no size in the sweep reaches 4 bits at all.  More pixels
   still beat finer pitch when the factor is large: a 32K panel at 4 um has
   four times the 16K's pixels and holds ``n = 14``.

6. **A flat detector array taxes exactly the panels that steer widest.**  Off
   axis a flat screen stretches the spot ``1/cos^3`` radially (the planar
   panel's foreshortening, the longer range and the oblique landing together),
   1.51x at 29.4 deg.  At 3 um the corner is only 15 deg out, the stretch is
   1.11x and the bit margin absorbs it: the 3 um sizes do not move.  At 1.56 um
   with ``screen="flat"`` the search backs off to ~23 deg, the fine panel's
   extra reach goes unused, and its pixels still have to cover a larger
   hologram -- 34 Mpx at ``n = 10`` against 25 at 3 um, so **with a flat
   detector "pitch beats pixel count" reverses**.  A curved detector puts the
   1.56 um numbers back: it removes the screen's share of the stretch (not the
   planar panel's ``1/cos``), and the remaining 1.15x costs a few hundredths
   of a bit.

The third panel takes the *why* of the first: the panel is a product of three
growing factors -- regions, sub-apertures per region, and pixels per hologram --
and on a log axis a product is a sum of heights.  Hologram count and
pixels-per-hologram are plotted against the total they multiply to, which reads
more clearly than the power curve does, since power is monotone and
unsurprising while this shows two independent explosions stacking.

    python examples/09_ccg_scaling.py
    python examples/09_ccg_scaling.py --save docs/figures
"""

import math

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

#: Panels to fit against: ``(name, columns, rows, pitch)``.  Generic classes
#: rather than catalogue parts -- the 4K line is roughly what a phase LCoS
#: ships at today, the rest are what a fab could be asked for.  The square one
#: has the 16K's pixel budget in the shape the region grid actually wants.
PANELS = (
    ("4K-class", 4160, 2464, 3.74e-6),
    ("8K-class", 7680, 4320, 3.0e-6),
    ("16K-class", 15360, 8640, 3.0e-6),
    ("square 133 Mpx", 11520, 11520, 3.0e-6),
    ("32K-class", 30720, 17280, 4.0e-6),
    ("16K-class, fine", 15360, 8640, 1.56e-6),
)

#: WCSP sizes to try against each panel.  Denser than ``SIZES`` around the
#: crossings, because "which is the largest that fits" is the question.
PANEL_SIZES = (3, 5, 8, 10, 12, 14, 16, 20)

#: Pitches whose steering reach is printed, and the one swept for screen shape:
#: the obliquity of a flat screen only matters where the deflection is wide.
STEER_PITCHES = (1.56e-6, 3.0e-6, 4.0e-6, 8.0e-6)
WIDE_PITCH = 1.56e-6


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


def sized_at(pitch: float, screen: str = "ideal") -> dict:
    """``{n: report}`` for every ``PANEL_SIZES`` entry, designed for one pitch.

    ``min_device_pitch`` is the honest way to say "this panel": the search may
    still pick a coarser design pitch (the hologram is then oversampled on the
    real device), never a finer one.
    """
    out = {}
    for n in PANEL_SIZES:
        problem = ProblemSpec.from_graph(ccg_wcsp_graph(n), iterations=ITERATIONS,
                                         label=f"ccg-{n}")
        req = Requirements(target_bits=TARGET_BITS, min_device_pitch=pitch,
                           screen=screen)
        out[n] = baseline(problem, req)
    return out


def on_panel(rep, pitch: float) -> tuple[int, int]:
    """Pixels the design occupies when written at the panel's own pitch."""
    return (int(math.ceil(rep.slm_extent_x / pitch)),
            int(math.ceil(rep.slm_extent_y / pitch)))


def fits(rep, cols: int, rows: int, pitch: float) -> bool:
    """Feasible, and its footprint fits the panel in either orientation."""
    nx, ny = on_panel(rep, pitch)
    return rep.feasible and ((nx <= cols and ny <= rows) or (nx <= rows and ny <= cols))


def panel_study() -> dict:
    """Every panel against every size, once per distinct pitch."""
    by_pitch = {p: sized_at(p) for p in sorted({p for *_, p in PANELS})}
    for p in STEER_PITCHES:
        by_pitch.setdefault(p, sized_at(p))
    screens = {s: sized_at(WIDE_PITCH, s) for s in ("flat", "curved")}
    screens["ideal"] = by_pitch[WIDE_PITCH]
    screens_3um = sized_at(3.0e-6, "flat")
    return {"by_pitch": by_pitch, "screens": screens, "flat_3um": screens_3um}


def print_panels(rows, study: dict) -> None:
    by_pitch = study["by_pitch"]
    print(f"\n-- what fits on a real panel "
          f"(dual rail, area division, {TARGET_BITS:.0f} bits) --")
    print("  pixels per CGH is not a constant, so a panel cannot be divided by one:")
    print("  " + "   ".join(f"n={r['n']}: {r['hard'].pixels_per_cgh}"
                            for r in rows if r["n"] <= 30) + "  px/CGH")
    print("  (the sub-aperture grows with the screen it has to resolve.  Below, each")
    print("   size is designed at the panel's own pitch and counted on that panel)")
    print(f"\n  {'panel':<17s}{'pixels':>14s}{'Mpx':>7s}{'pitch':>8s}"
          f"{'largest n':>11s}   needs at the next size")
    for name, cols, rows, pitch in PANELS:
        reps = by_pitch[pitch]
        ok = [n for n in PANEL_SIZES if fits(reps[n], cols, rows, pitch)]
        best = max(ok) if ok else None
        nxt = next((n for n in PANEL_SIZES if best is None or n > best), None)
        need = ""
        if nxt is not None:
            nx, ny = on_panel(reps[nxt], pitch)
            need = (f"n={nxt}: {nx} x {ny} ({nx * ny / 1e6:.0f} Mpx)"
                    + ("" if reps[nxt].feasible else ", misses the bit target"))
        print(f"  {name:<17s}{f'{cols}x{rows}':>14s}{cols * rows / 1e6:>7.0f}"
              f"{pitch * 1e6:>6.2f}um{str(best):>11s}   {need}")
    print("  (the 16:9 panel runs out of rows, not pixels: the region grid is near")
    print("   square, so the same budget as 11520 x 11520 reaches one size further)")

    print("\n  pitch sets the reach before it sets the pixel count:")
    print(f"  {'pitch':>8s}{'steers':>9s}" + "".join(f"{f'n={n} Mpx':>11s}"
                                                   for n in (10, 14, 20)))
    for p in STEER_PITCHES:
        reps = by_pitch[p]
        cells = []
        for n in (10, 14, 20):
            nx, ny = on_panel(reps[n], p)
            cells.append(f"{nx * ny / 1e6:>10.0f}" + (" " if reps[n].feasible else "!"))
        print(f"  {p * 1e6:>6.2f}um{math.degrees(math.asin(1550e-9 / (2 * p))):>7.1f}deg"
              + "".join(cells))
    print("  ('!' misses the 4-bit target at any power -- binding constraint")
    print("   accuracy_floor -- so the pixel count beside it is not a design)")

    sc = study["screens"]

    def mpx(rep, pitch):
        nx, ny = on_panel(rep, pitch)
        return nx * ny / 1e6

    print(f"\n  detector shape: Mpx on a {WIDE_PITCH * 1e6:.2f} um panel, where the "
          f"steering is wide, and on a 3 um one")
    print(f"  {'WCSP n':>7s}" + "".join(f"{s:>9s}" for s in ("ideal", "curved", "flat"))
          + f"{'flat deg':>10s}{'flat stretch':>14s}{'3um ideal':>11s}{'3um flat':>10s}")
    for n in (10, 12, 14):
        cells = "".join(f"{mpx(sc[s][n], WIDE_PITCH):>9.1f}"
                        for s in ("ideal", "curved", "flat"))
        f = sc["flat"][n]
        print(f"  {n:>7d}{cells}{math.degrees(f.deflection_needed):>10.1f}"
              f"{f'{f.spot_stretch[0]:.2f} x {f.spot_stretch[1]:.2f}':>14s}"
              f"{mpx(by_pitch[3.0e-6][n], 3e-6):>11.1f}"
              f"{mpx(study['flat_3um'][n], 3e-6):>10.1f}")
    print("  (a flat array stretches the corner spot 1/cos^3 radially, so the search")
    print("   steers less -- the fine panel's extra reach goes unused while its finer")
    print("   pixels still have to cover a larger hologram, and it ends up *behind*")
    print("   the 3 um panel.  At 3 um the corner is only 15 deg out and nothing")
    print("   moves.  A curved detector removes the screen's share of the stretch,")
    print("   not the planar panel's 1/cos, and the bit margin absorbs the rest)")


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

    study = panel_study()
    print_panels(rows, study)

    if _figure.wanted():
        make_figure(rows, study)


def make_figure(rows, study) -> None:
    plt = _figure.pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.6))
    ax = axes.ravel()

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

    # --- 4. real panels: pixels needed at each pitch, against what exists --- #
    # Required pixels on the panel's own pitch, so the curves can be read
    # against a panel's pixel count directly.  Pixel count is necessary but not
    # sufficient -- the shape has to fit too -- which is what the n=14 note says.
    ns = np.array(PANEL_SIZES, dtype=float)
    colours = {1.56e-6: "#2a7fbf", 3.0e-6: "#2f9e44", 4.0e-6: "#d9821b"}
    for p, col in colours.items():
        reps = study["by_pitch"][p]
        mpx = [np.prod(on_panel(reps[n], p)) / 1e6 for n in PANEL_SIZES]
        ax[3].semilogy(ns, mpx, "o-", color=col, lw=1.5, ms=5,
                       label=f"{p * 1e6:.2f} um  "
                             f"(steers {math.degrees(math.asin(1550e-9 / (2 * p))):.0f} deg)")
        mark = [(n, m) for n, m in zip(PANEL_SIZES, mpx) if not reps[n].feasible]
        for n, m in mark:
            ax[3].plot([n], [m], "o", ms=11, mfc="none", mec=bad, mew=1.8, zorder=5)
    flat = study["screens"]["flat"]
    ax[3].semilogy(ns, [np.prod(on_panel(flat[n], WIDE_PITCH)) / 1e6 for n in PANEL_SIZES],
                   "o--", color=colours[WIDE_PITCH], lw=1.2, ms=4, mfc="white",
                   label=f"{WIDE_PITCH * 1e6:.2f} um, flat detector array")
    for name, cols, rows_, _ in PANELS:
        if name in ("4K-class", "16K-class", "32K-class"):
            ax[3].axhline(cols * rows_ / 1e6, color="0.4", ls=":", lw=1.0, zorder=1)
            ax[3].text(ns[0], cols * rows_ / 1e6 * 1.12, f"{name} {cols}x{rows_}",
                       fontsize=7, color="0.3", va="bottom")
    ax[3].annotate("n=14 at 3 um: 133 Mpx is enough\nas 11520 x 11520, not as 16:9",
                   xy=(14, np.prod(on_panel(study["by_pitch"][3.0e-6][14], 3e-6)) / 1e6),
                   xytext=(15.2, 14), fontsize=7.5, color="0.25",
                   arrowprops=dict(arrowstyle="->", color="0.5", lw=0.9))
    ax[3].set_xlabel("WCSP variables  $n$")
    ax[3].set_ylabel("panel pixels needed at that pitch [Mpx]")
    ax[3].set_title("pitch sets the reach, and 16:9 is the wrong shape")
    ax[3].legend(fontsize=7, loc="lower right")
    ax[3].grid(True, which="both", alpha=0.3)

    _figure.finish(fig, "09_ccg_scaling")


def rep_n(report, rows) -> float:
    """The WCSP size a report was built for -- the label carries it."""
    return float(next(r["n"] for r in rows
                      if r["hard"] is report or r["apod"] is report))


if __name__ == "__main__":
    main()
