"""Phase multiplexing: superposing gratings, and reporting what that costs.

One physical sub-region can carry N holograms at once by summing them and
keeping only the phase.  The projection is lossy -- it scatters power into
intermodulation orders and speckle -- and the whole value of the class is that
it says so.  These tests pin that it does.
"""

import numpy as np
import pytest

from fsphotonix.modulators.gratings.base import Order, measure_orders
from fsphotonix.modulators.gratings.cgh import MultiSpotHologram, SpotTarget
from fsphotonix.modulators.gratings.composite import Cascade, Superposition
from fsphotonix.modulators.gratings.lenses import SpotFocuser
from fsphotonix.modulators.gratings.periodic import BlazedGrating

LAM, Z, APERTURE = 1550e-9, 0.2, 250e-6
SPOT = LAM * Z / APERTURE  # one spot width on the screen


def _fan(n: int, pitch_spots: float = 3.0, offset: float = 0.0):
    """``n`` targets on a near-square lattice, spaced in spot widths."""
    cols = int(np.ceil(np.sqrt(n)))
    return [((i % cols - (cols - 1) / 2) * pitch_spots * SPOT + offset * SPOT,
             (i // cols - (cols - 1) / 2) * pitch_spots * SPOT)
            for i in range(n)]


def _superposition(n: int, **kw) -> Superposition:
    gs = [SpotFocuser.from_target(x, y, Z, LAM) for x, y in _fan(n)]
    return Superposition(gratings=gs, aperture_size=(APERTURE, APERTURE), **kw)


# --------------------------------------------------------------------------- #
# the honesty fix
# --------------------------------------------------------------------------- #
def test_a_phase_only_superposition_admits_losing_power():
    """The bug this file exists for.

    ``orders()`` used to apportion each constituent's *analytic* orders by its
    power share.  ``SpotFocuser`` reports unit efficiency, so the shares summed
    to exactly 1.0: a phase-only projection claiming to lose nothing to
    intermodulation, which is the one thing it certainly does lose.
    """
    for n in (4, 9, 16):
        sp = _superposition(n)
        total = sum(o.efficiency for o in sp.orders(LAM))
        assert total < 0.95, f"n={n} still claims {total:.3f}"
        assert total > 0.3
        assert sp.stray_fraction(LAM) == pytest.approx(1.0 - total, abs=1e-6)


def test_orders_never_claim_more_power_than_was_supplied():
    """Disjoint assignment, so no bin is counted for two directions at once."""
    for n in (4, 9, 16, 25):
        for pitch in (1.0, 1.5, 3.0):
            gs = [SpotFocuser.from_target(x, y, Z, LAM)
                  for x, y in _fan(n, pitch_spots=pitch)]
            sp = Superposition(gratings=gs, aperture_size=(APERTURE, APERTURE))
            total = sum(o.efficiency for o in sp.orders(LAM))
            assert 0.0 <= total <= 1.0, (n, pitch, total)


def test_one_order_per_constituent():
    sp = _superposition(9)
    assert len(sp.orders(LAM)) == 9


def test_unresolvable_targets_are_flagged_not_silently_split():
    """Closer than the aperture can separate, the split is arbitrary."""
    tight = Superposition(
        gratings=[SpotFocuser.from_target(x, y, Z, LAM)
                  for x, y in _fan(4, pitch_spots=1.0)],
        aperture_size=(APERTURE, APERTURE))
    wide = Superposition(
        gratings=[SpotFocuser.from_target(x, y, Z, LAM)
                  for x, y in _fan(4, pitch_spots=6.0)],
        aperture_size=(APERTURE, APERTURE))
    assert not tight.resolved(LAM)
    assert wide.resolved(LAM)


# --------------------------------------------------------------------------- #
# random carrier phases
# --------------------------------------------------------------------------- #
def test_random_carrier_phases_recover_efficiency():
    """Without them every term peaks together and the projection wastes power.

    This is why ``MultiSpotHologram`` draws them, and the reason ``Superposition``
    now does too.
    """
    aligned = _superposition(4, randomize_phases=False)
    random = _superposition(4, randomize_phases=True)
    e_aligned = sum(o.efficiency for o in aligned.orders(LAM))
    e_random = sum(o.efficiency for o in random.orders(LAM))
    assert e_random > e_aligned + 0.1


def test_carrier_phases_are_seeded_and_reproducible():
    a = _superposition(9, seed=7)
    b = _superposition(9, seed=7)
    c = _superposition(9, seed=8)
    ea = [o.efficiency for o in a.orders(LAM)]
    eb = [o.efficiency for o in b.orders(LAM)]
    ec = [o.efficiency for o in c.orders(LAM)]
    assert ea == pytest.approx(eb)
    assert ea != pytest.approx(ec)


# --------------------------------------------------------------------------- #
# agreement with the synthesised CGH
# --------------------------------------------------------------------------- #
def test_superposition_matches_the_analytic_cgh_synthesis():
    """Both are a random-phase sum of focusing terms, so they must agree.

    ``Superposition`` composes arbitrary gratings while ``MultiSpotHologram``
    synthesises from targets, but for a plain spot fan-out they are the same
    hologram -- and before the carrier-phase fix they were not.
    """
    for n in (4, 9, 16):
        pos = _fan(n)
        sp = Superposition(gratings=[SpotFocuser.from_target(x, y, Z, LAM)
                                     for x, y in pos],
                           aperture_size=(APERTURE, APERTURE))
        cgh = MultiSpotHologram(targets=[SpotTarget(x=x, y=y, z=Z) for x, y in pos],
                                range_z=Z, aperture_size=(APERTURE, APERTURE),
                                method="superposition", seed=0)
        e_sp = sum(o.efficiency for o in sp.orders(LAM, samples=192))
        e_cgh = sum(o.efficiency for o in cgh.orders(LAM, samples=192))
        assert e_sp == pytest.approx(e_cgh, rel=0.05), n


def test_optimised_synthesis_delivers_better_weights():
    """WGS beats the analytic sum where it matters: the delivered weights."""
    pos = _fan(25)
    tg = [SpotTarget(x=x, y=y, z=Z) for x, y in pos]
    spread = {}
    for method in ("superposition", "wgs"):
        h = MultiSpotHologram(targets=tg, range_z=Z,
                              aperture_size=(APERTURE, APERTURE), method=method,
                              iterations=60, gs_samples=192, seed=0)
        e = np.array([o.efficiency for o in h.orders(LAM, samples=192)])
        spread[method] = e.std() / e.mean()
    assert spread["wgs"] < spread["superposition"] / 3


def test_gerchberg_saxton_quantises_target_placement():
    """A limitation worth knowing, not a bug to paper over.

    GS can only put energy on its transform's lattice, spaced ``1/aperture`` --
    one spot width -- so a target half a spot off that lattice is approximated
    and loses efficiency.  The analytic superposition places targets by exact
    phase and is flat against the same offset.
    """
    eff = {}
    for method in ("wgs", "superposition"):
        for offset in (0.0, 0.5):
            tg = [SpotTarget(x=(2 + offset + i * 3) * SPOT, y=0.0, z=Z)
                  for i in range(3)]
            h = MultiSpotHologram(targets=tg, range_z=Z,
                                  aperture_size=(APERTURE, APERTURE), method=method,
                                  iterations=60, gs_samples=192, seed=0)
            eff[(method, offset)] = sum(o.efficiency
                                        for o in h.orders(LAM, samples=192))
    # the optimiser pays for being off its lattice; the analytic sum does not
    assert eff[("wgs", 0.5)] < eff[("wgs", 0.0)] - 0.03
    assert eff[("superposition", 0.5)] == pytest.approx(
        eff[("superposition", 0.0)], rel=0.03)


# --------------------------------------------------------------------------- #
# uniformity
# --------------------------------------------------------------------------- #
def test_uniformity_is_zero_for_a_symmetric_fan_and_grows_with_count():
    assert _superposition(4).uniformity(LAM) < 0.05
    assert _superposition(16).uniformity(LAM) > _superposition(4).uniformity(LAM)


def test_orders_come_back_in_constituent_order():
    """Entry i must belong to gratings[i], or weights cannot be checked at all.

    This branch used to sort by efficiency, which silently permuted the
    correspondence: a fan-out requested as 1:2:3:4 came back anti-correlated
    with what was asked for.
    """
    pos = _fan(4)
    weights = [1.0, 2.0, 3.0, 4.0]
    sp = Superposition(gratings=[SpotFocuser.from_target(x, y, Z, LAM)
                                 for x, y in pos],
                       weights=weights, aperture_size=(APERTURE, APERTURE))
    out = sp.orders(LAM)
    assert [o.mx for o in out] == list(range(4))


def test_uniformity_measures_the_requested_weights_not_just_the_spread():
    """A deliberately unequal fan-out must not be called non-uniform for it."""
    pos = _fan(4)
    weights = [1.0, 2.0, 3.0, 4.0]
    sp = Superposition(gratings=[SpotFocuser.from_target(x, y, Z, LAM)
                                 for x, y in pos],
                       weights=weights, aperture_size=(APERTURE, APERTURE))
    eff = np.array([o.efficiency for o in sp.orders(LAM)])
    # the delivered shares should track the requested ones, not be flat
    want = np.array(weights, dtype=float)
    want /= want.sum()
    got = eff / eff.sum()
    assert np.corrcoef(got, want)[0, 1] > 0.9


# --------------------------------------------------------------------------- #
# the periodic branch still works
# --------------------------------------------------------------------------- #
def test_a_periodic_superposition_still_uses_the_unit_cell():
    """Constituents sharing a lattice keep the exact analytic route."""
    g1 = BlazedGrating(period=20e-6, design_wavelength=LAM)
    g2 = BlazedGrating(period=10e-6, design_wavelength=LAM)
    sp = Superposition(gratings=[g1, g2], design_wavelength=LAM)
    assert sp.unit_cell() is not None
    out = sp.orders(LAM)
    assert out and all(isinstance(o, Order) for o in out)
    assert sum(o.efficiency for o in out) <= 1.0 + 1e-9


def test_cascade_is_untouched_by_the_superposition_change():
    c = Cascade(gratings=[BlazedGrating(period=20e-6, design_wavelength=LAM)],
                design_wavelength=LAM)
    assert sum(o.efficiency for o in c.orders(LAM)) <= 1.0 + 1e-9


# --------------------------------------------------------------------------- #
# the shared measurement helper
# --------------------------------------------------------------------------- #
def test_measure_orders_is_disjoint_even_for_coincident_directions():
    """Two directions at the same place must not each claim the same power."""
    g = SpotFocuser.from_target(3 * SPOT, 0.0, Z, LAM)
    got = measure_orders(g, (APERTURE, APERTURE),
                         [(1.0 / (LAM * Z) * 3 * SPOT,) * 1 + (0.0,)] * 2,
                         LAM, samples=192)
    assert sum(got["efficiency"]) <= 1.0 + 1e-9
    assert not got["resolved"]


def test_measure_orders_is_insensitive_to_sub_bin_placement():
    """Zero-padding removes the scalloping that made this swing by 2x."""
    totals = []
    for offset in (0.0, 0.25, 0.5, 0.75):
        gs = [SpotFocuser.from_target(x, y, Z, LAM)
              for x, y in _fan(4, pitch_spots=3.0, offset=offset)]
        sp = Superposition(gratings=gs, aperture_size=(APERTURE, APERTURE))
        totals.append(sum(o.efficiency for o in sp.orders(LAM)))
    assert max(totals) - min(totals) < 0.1, totals
