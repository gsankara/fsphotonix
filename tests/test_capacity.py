"""Capacity estimation: sizing a machine from a problem instance.

These are invariant tests rather than golden numbers: each one pins a *relation*
that has to hold for the sizing to mean anything -- that the panel scales with
fan-out under area division and not under multiplexing, that no amount of power
beats a cross-talk floor, that the model lands where the engine's own measured
accuracy lands.
"""

import math

import numpy as np
import pytest

from fsphotonix.capacity import (
    CGH_METHODS,
    Features,
    FeatureModel,
    ProblemSpec,
    Requirements,
    _capture,
    _crosstalk_ratio,
    _window_profile,
    baseline,
    compare,
    estimate,
    measure_routing,
)
from fsphotonix.solvers.aim import maxcut_ising, random_graph


# --------------------------------------------------------------------------- #
# problem specs
# --------------------------------------------------------------------------- #
def test_problem_spec_reads_the_sparsity_not_just_the_shape():
    W = np.zeros((8, 8))
    W[0, 0] = W[1, 0] = W[3, 2] = 1.0  # column 0 drives two outputs, column 2 one
    p = ProblemSpec.from_matrix(W)
    assert (p.n_out, p.n_in) == (8, 8)
    assert p.max_fanout == 2  # the busiest column, not n_out
    assert p.nnz == 3
    assert p.density == pytest.approx(3 / 64)


def test_problem_spec_detects_signed_weights_and_dynamic_range():
    assert not ProblemSpec.from_matrix(np.ones((4, 4))).signed_weights
    assert ProblemSpec.from_matrix(np.array([[1.0, -1.0]])).signed_weights
    p = ProblemSpec.from_matrix(np.array([[1.0, 8.0]]))
    assert p.dynamic_range_bits == pytest.approx(3.0)  # 8/1 = 2^3


def test_ising_problem_keeps_the_graph_sparsity():
    A = random_graph(24, density=0.3, seed=3)
    p = ProblemSpec.from_graph(A, iterations=200)
    assert p.n_in == p.n_out == 24
    assert p.iterations == 200
    # J = -W, so a vertex's fan-out is its degree
    assert p.max_fanout == int((np.abs(A) > 0).sum(axis=0).max())


# --------------------------------------------------------------------------- #
# the window / far-field machinery
# --------------------------------------------------------------------------- #
def test_capture_matches_the_analytic_sinc_squared():
    """A hard-edged aperture must reproduce the sinc^2 encircled-energy curve."""
    u = np.linspace(-200, 200, 2_000_001)
    s = np.sinc(u) ** 2
    for half in (0.5, 1.0, 2.0):
        want = s[np.abs(u) <= half].sum() / s.sum()
        assert _capture("none", 0.0, half) == pytest.approx(want, abs=0.01)


def test_apodization_trades_throughput_for_sidelobes():
    """The whole point of a window: less light, cleaner wings."""
    t_rect, b_rect = (_window_profile("none")[0], _window_profile("none")[1])
    t_hann, b_hann = (_window_profile("hann")[0], _window_profile("hann")[1])
    assert t_rect == pytest.approx(1.0)
    assert t_hann == pytest.approx(0.375, abs=0.01)  # mean(w^2) for Hann
    assert b_rect == pytest.approx(1.0)
    assert b_hann > 1.5  # Hann broadens the main lobe

    # at the same guard fraction the apodized aperture leaks far less
    leak_rect, _ = _crosstalk_ratio("none", 1.0 / 0.3, 0.4 / 0.3)
    leak_hann, _ = _crosstalk_ratio("hann", b_hann / 0.3, 0.4 * b_hann / 0.3)
    assert leak_hann < leak_rect / 5


def test_leakage_falls_as_the_guard_opens():
    prev = None
    for guard in (0.9, 0.6, 0.3, 0.15):
        leak, _ = _crosstalk_ratio("none", 1.0 / guard, 0.4 / guard)
        if prev is not None:
            assert leak < prev
        prev = leak


# --------------------------------------------------------------------------- #
# the sizing relations -- the reason the module exists
# --------------------------------------------------------------------------- #
def _problem(n=32, density=0.5, seed=0):
    return ProblemSpec.from_graph(random_graph(n, density=density, seed=seed))


def test_area_division_pays_for_fanout_and_multiplexing_does_not():
    """The headline relation, with the geometry pinned so only fan-out varies.

    One sub-aperture per target makes the region ``sqrt(fan)`` across, so the
    panel area carries a full factor of ``fan``.  Phase multiplexing addresses
    every target through the full aperture, so its region is the sub-aperture
    itself and the panel is set by the number of *inputs* alone.

    The geometry has to be pinned to see this cleanly: left free, the accuracy
    budget feeds back into the choice of guard fraction -- a sparse problem
    suffers more off-support stray light, so the optimiser buys a tighter guard
    and a *larger* aperture, which moves the panel for reasons that have nothing
    to do with the area/multiplex distinction.
    """
    sparse = ProblemSpec.from_matrix(_fixed_fanout_matrix(n=32, fan=2))
    dense = ProblemSpec.from_matrix(_fixed_fanout_matrix(n=32, fan=32))
    req = Requirements(target_bits=3.0, z=0.2, out_pitch=2e-3, spot_guard=0.3)

    div_s = estimate(sparse, req, Features(routing="subregions"))
    div_d = estimate(dense, req, Features(routing="subregions"))
    mux_s = estimate(sparse, req, Features(routing="multiplex"))
    mux_d = estimate(dense, req, Features(routing="multiplex"))

    # the sub-aperture is set by the spot/pitch constraint, so it is common to all
    assert div_d.subaperture == pytest.approx(mux_d.subaperture, rel=1e-6)

    # area division: region across scales as sqrt(fan) -> sqrt(32/2) = 4
    assert div_d.region_size / div_s.region_size == pytest.approx(4.0, rel=0.01)
    assert div_d.slm_pixels > 10 * div_s.slm_pixels

    # multiplexing: one sub-region at the full aperture, whatever the fan-out
    assert mux_d.region_size == pytest.approx(mux_s.region_size, rel=1e-6)
    assert mux_d.slm_pixels == pytest.approx(mux_s.slm_pixels, rel=1e-6)
    assert mux_d.subregions_per_region == 1
    assert div_d.subregions_per_region == 32

    # and for the dense problem that is a large saving
    assert mux_d.slm_pixels < div_d.slm_pixels / 10


def _fixed_fanout_matrix(n: int, fan: int) -> np.ndarray:
    """``n x n`` matrix where every column drives exactly ``fan`` outputs."""
    W = np.zeros((n, n))
    for j in range(n):
        rows = (np.arange(fan) * 7 + j) % n
        W[rows, j] = 1.0
    return W


def test_dual_rail_doubles_the_component_count():
    p = _problem(16)
    req = Requirements(target_bits=3.0)
    dual = estimate(p, req, Features(encoding="dual_rail"))
    single = estimate(p, req, Features(encoding="unsigned"))
    assert dual.lasers == 2 * single.lasers
    assert dual.detectors == 2 * single.detectors
    assert dual.lasers == 2 * p.n_in


def test_time_multiplexing_trades_rate_for_panel():
    p = _problem(32)
    req = Requirements(target_bits=3.0)
    one = estimate(p, req, Features(time_multiplex=1))
    four = estimate(p, req, Features(time_multiplex=4))
    assert four.slm_pixels < one.slm_pixels
    assert four.update_rate == pytest.approx(one.update_rate / 4)


def test_sparser_problems_are_cheaper_to_build():
    req = Requirements(target_bits=3.0)
    a = estimate(_problem(32, density=0.15, seed=1), req)
    b = estimate(_problem(32, density=0.9, seed=1), req)
    assert a.slm_pixels < b.slm_pixels


# --------------------------------------------------------------------------- #
# the accuracy floor
# --------------------------------------------------------------------------- #
def test_an_unreachable_bit_depth_is_reported_not_bought_with_power():
    """A cross-talk floor is not a power problem, and must not look like one."""
    p = _problem(32)
    rep = estimate(p, Requirements(target_bits=14.0))
    assert not rep.feasible
    assert rep.binding == "accuracy_floor"
    assert rep.achievable_bits < 14.0
    assert rep.error_static >= rep.error_budget


def test_more_bits_costs_more_power_at_a_fixed_design_point():
    """Only meaningful with the geometry pinned.

    Left free the search minimises *pixels*, so a different bit target picks a
    different design point and the power is not monotonic -- which is correct
    behaviour, not a bug.
    """
    p = _problem(16)
    f = Features()
    fixed = dict(z=0.2, out_pitch=2e-3, spot_guard=0.25)
    lo = estimate(p, Requirements(target_bits=2.0, **fixed), f)
    hi = estimate(p, Requirements(target_bits=3.5, **fixed), f)
    assert lo.subaperture == pytest.approx(hi.subaperture, rel=1e-9)
    assert hi.optical_power_total > lo.optical_power_total


def test_optimised_cgh_synthesis_is_what_makes_multiplexing_usable():
    """WGS beats a naive analytic superposition by well over a bit of depth.

    Tested on a *dense* problem on purpose: there every spot in the fan is an
    intended output, so no stray light falls off-support and the weight
    non-uniformity is the whole error -- which is exactly the term the
    synthesis method controls. On a sparse problem the off-support stray
    dominates instead and the choice of synthesis matters much less.
    """
    p = ProblemSpec.from_matrix(_fixed_fanout_matrix(n=32, fan=32))
    naive = Features(routing="multiplex", cgh_method="superposition")
    wgs = Features(routing="multiplex", cgh_method="wgs")

    # with the geometry pinned the advantage shows up as bit depth
    pinned = Requirements(target_bits=3.0, z=0.2, out_pitch=2e-3, spot_guard=0.15)
    assert (estimate(p, pinned, wgs).achievable_bits
            > estimate(p, pinned, naive).achievable_bits + 1.0)

    # left free it shows up as a smaller panel instead: once the target is met
    # the search stops buying accuracy and spends the margin on a looser guard,
    # which shrinks the sub-aperture. Same bits, less glass.
    free = Requirements(target_bits=3.0)
    r_naive, r_wgs = estimate(p, free, naive), estimate(p, free, wgs)
    assert r_wgs.slm_pixels < r_naive.slm_pixels
    assert r_wgs.achievable_bits >= free.target_bits


def test_unoptimised_superposition_is_the_worst_routing():
    p = _problem(32)
    req = Requirements(target_bits=3.0)
    sup = estimate(p, req, Features(routing="superpose"))
    mux = estimate(p, req, Features(routing="multiplex", cgh_method="superposition"))
    assert sup.achievable_bits < mux.achievable_bits


# --------------------------------------------------------------------------- #
# agreement with the rest of the package
# --------------------------------------------------------------------------- #
def test_the_error_model_lands_in_the_measured_accuracy_band():
    """Cross-check against reality.

    ``docs/PHYSICS.md`` reports the optical matrix reproducing the requested one
    to 1-5% rms after per-rail calibration.  A sized baseline design has to come
    out in that band: much better and this module is flattering itself, much
    worse and it would reject designs the engine demonstrably achieves.
    """
    for n in (8, 16, 32):
        rep = baseline(_problem(n), Requirements(target_bits=4.0))
        assert 0.01 <= rep.error_static <= 0.06, (n, rep.error_static)


def test_geometry_agrees_with_the_optical_engine():
    """The sized design, handed to OpticalMVM, must describe the same optics."""
    A = random_graph(12, density=0.6, seed=0)
    J, _ = maxcut_ising(A)
    p = ProblemSpec.from_graph(A)
    rep = baseline(p, Requirements(target_bits=3.0, z=0.15))

    chk = rep.to_mvm(matrix=J / np.abs(J).max()).check()
    assert chk["subaperture_um"] == pytest.approx(rep.subaperture * 1e6, rel=0.02)
    assert chk["spot_size_um"] == pytest.approx(rep.spot_size * 1e6, rel=0.02)
    assert chk["pixels_per_subaperture"] == pytest.approx(
        rep.pixels_per_subaperture, rel=0.02)
    assert chk["rails"] == {"in": rep.lasers, "out": rep.detectors}
    # the sizing deliberately over-estimates the corner angle (it uses the full
    # panel and screen extents), so it must be conservative, never optimistic
    assert chk["max_deflection_needed_deg"] <= math.degrees(rep.deflection_needed) + 1e-6


def test_derived_pitch_can_always_steer_what_the_layout_needs():
    """The pitch is an output, so steering is satisfied by construction."""
    for n in (8, 32, 64):
        rep = baseline(_problem(n), Requirements(target_bits=3.0))
        assert rep.deflection_available >= rep.deflection_needed - 1e-9
        assert rep.pixels_per_subaperture >= 16 - 1e-6


# --------------------------------------------------------------------------- #
# measurement path
# --------------------------------------------------------------------------- #
def test_measured_routing_never_reports_more_power_than_it_was_given():
    """MultiSpotHologram.orders() double-counts when targets are unresolvable.

    Its integration windows are one spot width wide and spaced one spot width
    apart, so targets closer than ~3 spot widths overlap and the same energy is
    counted twice -- which is why measure_routing forces a wider lattice. Guard
    that, because an efficiency above 1 would silently flatter every estimate.
    """
    for fan in (4, 16, 32):
        m = measure_routing(fan, 250e-6, 1550e-9, 0.2, 2e-3, routing="multiplex",
                            samples=160, method="wgs")
        assert 0.0 < m["efficiency"] <= 1.0
        assert m["u_sep"] >= 3.0 - 1e-9
        assert not m["window_overlap"]


def test_measurement_confirms_the_ordering_of_the_cgh_methods():
    """The table of coefficients has to match what the synthesis actually does."""
    unif = {}
    for method in ("superposition", "wgs"):
        m = measure_routing(16, 250e-6, 1550e-9, 0.2, 2e-3, routing="multiplex",
                            samples=160, method=method)
        unif[method] = m["uniformity"]
    assert unif["wgs"] < unif["superposition"]
    assert CGH_METHODS["wgs"][1] < CGH_METHODS["superposition"][1]


def test_measure_true_overrides_the_model():
    p = _problem(16)
    f = Features(routing="multiplex", cgh_method="wgs")
    rep = estimate(p, Requirements(target_bits=3.0), f, measure=True)
    assert any("measured" in n for n in rep.notes)


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def test_compare_prices_every_variant_against_the_baseline():
    p = _problem(32)
    res = compare(p, [Features(routing="multiplex"),
                      Features(encoding="unsigned")],
                  Requirements(target_bits=3.0))
    assert res["baseline"].features == Features.baseline()
    assert len(res["variants"]) == 2
    mux_deltas = res["variants"][0][2]
    assert mux_deltas["slm_pixels"] < 1.0  # multiplexing shrinks the panel
    assert "baseline" in res["table"]


def test_report_summary_and_table_are_complete():
    rep = baseline(_problem(16), Requirements(target_bits=3.0))
    s = rep.summary()
    for k in ("lasers", "detectors", "slm_megapixels", "slm_pitch_um",
              "achievable_bits", "optical_power_mw", "binding"):
        assert k in s
    assert rep.lasers > 0 and rep.detectors > 0 and rep.slm_pixels > 0
    assert "capacity:" in rep.table()


def test_rejected_feature_names_fail_loudly():
    with pytest.raises(ValueError):
        Features(routing="sideways")
    with pytest.raises(ValueError):
        Features(cgh_method="magic")
    with pytest.raises(ValueError):
        Features(encoding="trinary")
    with pytest.raises(ValueError):
        FeatureModel().window("gaussian_ish")
