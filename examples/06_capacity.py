"""Capacity estimation: what hardware does a given problem actually need?

Runs the inverse of a feasibility check.  Instead of asking whether a layout
works, it takes a Max-Cut instance and solves for the component budget -- lasers,
SLM pixels and pitch, detectors, throw, power -- then prices each architectural
feature against a conservative baseline.

The two results worth reading off the output:

1. **Area division pays a factor of fan-out in panel area; phase multiplexing
   does not.**  One sub-aperture per target makes each region ``sqrt(fan)``
   across, so a dense problem needs a panel tens of times larger than the same
   problem multiplexed through full apertures.

2. **Whether that saving is usable depends entirely on how the hologram is
   synthesised.**  A naive analytic superposition delivers its weights to about
   30% rms -- under two bits, which rules it out -- while weighted
   Gerchberg-Saxton delivers them to 1-2% *and* is more efficient.  The last
   panel of the figure measures that directly out of
   ``MultiSpotHologram.orders()`` rather than taking anyone's word for it.

    python examples/06_capacity.py
    python examples/06_capacity.py --save docs/figures
"""

import numpy as np

import _figure
from fsphotonix.capacity import (
    CGH_METHODS,
    Features,
    ProblemSpec,
    Requirements,
    baseline,
    compare,
    estimate,
    measure_routing,
)
from fsphotonix.solvers.aim import maxcut_ising, random_graph

N = 48
DENSITY = 0.5


def main() -> None:
    A = random_graph(N, density=DENSITY, seed=0)
    problem = ProblemSpec.from_graph(A, iterations=300, label=f"maxcut-{N}")
    req = Requirements(target_bits=4.0)

    print(f"-- problem --")
    for k, v in problem.summary().items():
        print(f"  {k:22s} {v}")

    print("\n-- baseline design (dual rail, area division, no apodization) --")
    base = baseline(problem, req)
    print(base.table())

    print("\n-- what each feature costs --")
    variants = [
        Features(routing="multiplex", cgh_method="wgs"),
        Features(routing="multiplex", cgh_method="gs"),
        Features(routing="multiplex", cgh_method="superposition"),
        Features(routing="superpose"),
        Features(routing="subregions", apodization="hann"),
        Features(routing="multiplex", cgh_method="wgs", apodization="hann"),
        Features(routing="subregions", time_multiplex=4),
        Features(encoding="unsigned", routing="multiplex", cgh_method="wgs"),
    ]
    res = compare(problem, variants, req)
    print(res["table"])

    print("\n-- the estimate, checked against the optical engine --")
    J, _ = maxcut_ising(A)
    pinned = baseline(problem, Requirements(target_bits=4.0, z=0.2))
    chk = pinned.to_mvm(matrix=J / np.abs(J).max()).check()
    rows = [("sub-aperture [um]", pinned.subaperture * 1e6, chk["subaperture_um"]),
            ("spot size [um]", pinned.spot_size * 1e6, chk["spot_size_um"]),
            ("pixels / sub-aperture", pinned.pixels_per_subaperture,
             chk["pixels_per_subaperture"]),
            ("deflection needed [deg]", np.degrees(pinned.deflection_needed),
             chk["max_deflection_needed_deg"])]
    print(f"  {'quantity':26s}{'estimate':>12s}{'OpticalMVM.check()':>20s}")
    for name, est, got in rows:
        print(f"  {name:26s}{est:>12.2f}{got:>20.2f}")
    print("  (the sizing uses the full panel and screen extents for the corner "
          "angle, so its deflection is deliberately the conservative one)")

    print("\n-- measured CGH synthesis penalty (what the coefficients come from) --")
    fans = [4, 8, 16, 32]
    measured: dict[str, list[dict]] = {}
    for method in ("superposition", "gs", "wgs"):
        measured[method] = [
            measure_routing(f, 250e-6, 1550e-9, 0.2, 2e-3, routing="multiplex",
                            samples=192, method=method) for f in fans]
    print(f"  {'fan':>5}" + "".join(f"{m:>22s}" for m in measured))
    for i, f in enumerate(fans):
        cells = "".join(f"{measured[m][i]['efficiency']:>10.3f} /"
                        f"{measured[m][i]['uniformity']:>10.3f}" for m in measured)
        print(f"  {f:>5}{cells}")
    print("  (efficiency / rms weight non-uniformity; table coefficients: "
          + ", ".join(f"{k}={v[1]:.3f}" for k, v in CGH_METHODS.items()) + ")")

    if _figure.wanted():
        make_figure(problem, req, base, res, fans, measured)


def make_figure(problem, req, base, res, fans, measured) -> None:
    plt = _figure.pyplot()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    # --- 1. panel size vs fan-out, the two routing families ----------------- #
    fan_axis = np.array([1, 2, 4, 8, 16, 32, 64])
    curves = {}
    for routing, style in (("subregions", "o-"), ("multiplex", "s-")):
        px = []
        for f in fan_axis:
            p = ProblemSpec.from_matrix(_fanout_matrix(48, int(f)))
            r = estimate(p, Requirements(target_bits=3.0, z=0.2, out_pitch=2e-3,
                                         spot_guard=0.3),
                         Features(routing=routing, cgh_method="wgs"))
            px.append(r.slm_pixels / 1e6)
        curves[routing] = px
        ax[0].loglog(fan_axis, px, style, label=routing)
    ax[0].loglog(fan_axis, curves["subregions"][0] * fan_axis, "k:", alpha=0.5,
                 label="$\\propto$ fan")
    ax[0].set_xlabel("fan-out per input")
    ax[0].set_ylabel("SLM size [Mpx]")
    ax[0].set_title("area division pays for fan-out")
    ax[0].legend()
    ax[0].grid(True, which="both", alpha=0.3)

    # --- 2. the feature comparison ------------------------------------------ #
    labels = [base.features.tag] + [v.tag for v, _, _ in res["variants"]]
    mpx = [base.megapixels] + [r.megapixels for _, r, _ in res["variants"]]
    bits = [base.achievable_bits] + [r.achievable_bits for _, r, _ in res["variants"]]
    ok = [base.feasible] + [r.feasible for _, r, _ in res["variants"]]
    y = np.arange(len(labels))
    ax[1].barh(y, mpx, color=["#2a7fbf" if f else "#c44" for f in ok])
    ax[1].set_yticks(y)
    ax[1].set_yticklabels([_short(s) for s in labels], fontsize=7)
    ax[1].invert_yaxis()
    ax[1].set_xscale("log")
    ax[1].set_xlabel("SLM size [Mpx]  (red = accuracy floor rules it out)")
    ax[1].set_title(f"feature cost at {req.target_bits:.0f} bits")
    for yi, (m, b) in enumerate(zip(mpx, bits)):
        ax[1].text(m, yi, f" {b:.1f} bits", va="center", fontsize=7)
    ax[1].grid(True, axis="x", alpha=0.3)

    # --- 3. the measured synthesis penalty ---------------------------------- #
    # a converged WGS can hit exactly zero spread on the measurement grid, which
    # a log axis cannot show -- floor it at 0.1% rather than distort the scale
    floor = 1e-3
    for method, mark in (("superposition", "o-"), ("gs", "^-"), ("wgs", "s-")):
        ax[2].semilogy(fans, [max(m["uniformity"], floor) for m in measured[method]],
                       mark, label=method)
    ax[2].set_ylim(floor / 2, 1.0)
    ax[2].axhline(2.0 ** -4, color="k", ls=":", alpha=0.6)
    ax[2].text(fans[0], 2.0 ** -4 * 1.15, "4-bit budget", fontsize=7)
    ax[2].set_xlabel("fan-out per hologram")
    ax[2].set_ylabel("rms weight non-uniformity")
    ax[2].set_title("multiplexing lives or dies on the synthesis")
    ax[2].legend()
    ax[2].grid(True, which="both", alpha=0.3)

    _figure.finish(fig, "06_capacity")


def _fanout_matrix(n: int, fan: int) -> np.ndarray:
    W = np.zeros((n, n))
    for j in range(n):
        W[(np.arange(fan) * 7 + j) % n, j] = 1.0
    return W


def _short(tag: str) -> str:
    return (tag.replace("dual_rail+", "").replace("unsigned+", "unsigned ")
            .replace("subregions", "area").replace("multiplex", "mux"))


if __name__ == "__main__":
    main()
