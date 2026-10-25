"""Sub-aperture apodization: tapering the edge instead of guarding it.

A hard-edged sub-aperture has ``sinc`` far-field wings that fall only as
``1/u^2``, which across a driven detector lattice sets a cross-talk floor no
guard band escapes.  These tests pin the taper's three consequences -- less
light, a wider spot, far cleaner wings -- and that the first two are the price
of the third, measured by propagation rather than asserted.
"""

import numpy as np
import pytest

import fsphotonix as fp
from fsphotonix import engine as eng
from fsphotonix.analysis import apodization_report, apodization_sweep
from fsphotonix.capacity import window_metrics
from fsphotonix.grid import Grid
from fsphotonix.modulators.array import ModulatorArray
from fsphotonix.modulators.devices import IdealModulator
from fsphotonix.modulators.geometry import (
    WINDOWS,
    Apodized,
    Circle,
    Rect,
    apodize_aperture,
    tile_region,
    window_function,
)

LAM, Z = 1550e-9, 60e-3
REGION, PITCH = 400e-6, 1.2e-3

_SITES = [(i * 4e-3, j * 4e-3) for j in (-1, 0, 1) for i in (-1, 0, 1)]
_TARGETS = [_SITES[0], _SITES[2], _SITES[6], _SITES[8]]
_LEAKS = [s for s in _SITES if s not in _TARGETS]


# --------------------------------------------------------------------------- #
# the windows themselves
# --------------------------------------------------------------------------- #
def test_windows_peak_at_the_centre_and_vanish_at_the_rim():
    t = eng.asarray(np.linspace(-0.5, 0.5, 257))
    for name in WINDOWS:
        w = eng.to_numpy(window_function(name)(t))
        assert w.max() == pytest.approx(1.0, abs=1e-9)
        assert w.min() >= -1e-12
        assert w[128] == pytest.approx(1.0, abs=1e-9)
        if name != "none":
            assert w[0] < 1e-6 and w[-1] < 1e-6


def test_more_taper_passes_less_light():
    thru = [Apodized(inner=Rect(width=REGION), window=w).power_throughput()
            for w in ("none", "tukey", "hann", "blackman")]
    assert thru == sorted(thru, reverse=True)
    assert thru[0] == pytest.approx(1.0, abs=1e-3)
    # a square aperture windows both axes: Hann passes 0.375^2, not 0.375
    assert thru[2] == pytest.approx(0.375**2, rel=0.05)


def test_rendered_throughput_matches_the_predicted_one():
    """The geometry and the capacity model must not drift apart."""
    for name in ("tukey", "hann", "blackman"):
        rendered = Apodized(inner=Rect(width=REGION), window=name).power_throughput()
        predicted, _ = window_metrics(name)
        assert rendered == pytest.approx(predicted, rel=0.05), name


def test_unknown_window_fails_loudly():
    with pytest.raises(ValueError):
        window_function("gaussian_ish")
    with pytest.raises(ValueError):
        Apodized(inner=Rect(), window="nope")
    with pytest.raises(ValueError):
        Apodized(inner=Rect(), mode="sideways")


# --------------------------------------------------------------------------- #
# the aperture wrapper
# --------------------------------------------------------------------------- #
def test_apodization_tapers_amplitude_without_moving_the_support():
    """A window must not move the edge, or the sub-apertures stop tiling."""
    hard = Rect(width=REGION)
    soft = Apodized(inner=hard, window="hann")
    assert soft.bounds() == hard.bounds()
    assert soft.size == hard.size

    g = Grid.from_extent(600e-6, 600e-6, 2e-6)
    mh = eng.to_numpy(hard.mask(g))
    ms = eng.to_numpy(soft.mask(g))
    assert ms.max() == pytest.approx(1.0, abs=0.02)
    assert (ms[mh > 0] <= mh[mh > 0] + 1e-9).all()
    assert (ms[mh == 0] == 0).all()
    assert ms.sum() < mh.sum()


def test_taper_falls_monotonically_towards_the_rim():
    g = Grid.from_extent(600e-6, 600e-6, 2e-6)
    m = eng.to_numpy(Apodized(inner=Rect(width=REGION), window="hann").mask(g))
    row = m[m.shape[0] // 2]
    inside = np.nonzero(row > 0)[0]
    half = row[inside[0]: inside[0] + len(inside) // 2]
    assert np.all(np.diff(half) >= -1e-9)


def test_round_apertures_get_a_radial_window():
    assert Apodized(inner=Circle(diameter=REGION)).radial
    assert not Apodized(inner=Rect(width=REGION)).radial
    assert Apodized(inner=Circle(diameter=REGION), mode="separable").radial is False


def test_no_window_is_a_passthrough():
    r = Rect(width=REGION)
    assert apodize_aperture(r, None) is r
    assert apodize_aperture(r, "none") is r
    assert isinstance(apodize_aperture(r, "hann"), Apodized)


# --------------------------------------------------------------------------- #
# plumbing through the routing API
# --------------------------------------------------------------------------- #
def test_tile_region_apodizes_every_sub_aperture():
    subs = tile_region(Rect(width=REGION), 4, mode="grid", apodize="hann")
    assert len(subs) == 4
    assert all(isinstance(ap, Apodized) for ap, _ in subs)
    plain = tile_region(Rect(width=REGION), 4, mode="grid")
    # the window must not change where the tiles sit
    assert [off for _, off in subs] == [off for _, off in plain]


@pytest.mark.parametrize("mode", ["subregions", "multiplex", "superpose"])
def test_route_accepts_a_window_in_every_mode(mode):
    arr = _panel()
    arr.route(0, _TARGETS, z=Z, tiling="grid", mode=mode, apodize="hann")
    assert arr.gap_report(z=Z)["apodization"] == "hann"
    aps = [s.aperture for s in arr[0].subregions]
    assert all(isinstance(a, Apodized) for a in aps)


def test_the_rendered_transmittance_keeps_the_taper():
    """The render path must not normalise the window back to a hard edge.

    ``ideal_transmittance`` divides the accumulated sub-aperture field by
    ``maximum(wacc, 1.0)``.  For non-overlapping tapers ``wacc`` is below 1 near
    the edges, so the divisor stays at 1 and the taper survives -- this pins
    that, because the alternative would silently undo the whole feature.
    """
    g = Grid.from_extent(600e-6, 600e-6, 1e-6)
    power = {}
    for window in (None, "hann"):
        arr = _panel()
        arr.route(0, _TARGETS, z=Z, tiling="grid", apodize=window)
        a = np.abs(eng.to_numpy(arr.ideal_transmittance(g, LAM)))
        power[window] = float((a**2).sum())
        assert a.max() == pytest.approx(1.0, abs=0.02)

    predicted, _ = window_metrics("hann")
    assert power["hann"] / power[None] == pytest.approx(predicted, rel=0.05)


# --------------------------------------------------------------------------- #
# what it buys, measured
# --------------------------------------------------------------------------- #
def test_apodization_report_prices_the_window():
    arr = _panel()
    arr.route(0, _TARGETS, z=Z, tiling="grid", apodize="hann")
    rep = apodization_report(arr, z=Z, wavelength=LAM)
    assert rep["apodization"] == "hann"
    assert rep["power_throughput"] == pytest.approx(0.375**2, rel=0.05)
    assert rep["apodization_loss_db"] < 0
    assert rep["mainlobe_broadening"] > 1.5
    assert rep["apodized_spot_size_um"] > rep["spot_size_um"]


def test_apodization_really_suppresses_crosstalk_and_really_costs_light():
    """The trade, through actual wave propagation.

    One input fans out to four sites of a 3x3 lattice; the other five are the
    sites a neighbouring channel would own, so any power there is cross-talk.
    """
    res = apodization_sweep(_build_system, windows=(None, "tukey", "hann"),
                            target_positions=_TARGETS, leak_positions=_LEAKS,
                            radius=1.2e-3)
    rows = {r["window"]: r for r in res["rows"]}

    assert rows["none"]["crosstalk_db"] > -30
    assert rows["tukey"]["crosstalk_suppression_db"] > 2
    assert (rows["hann"]["crosstalk_suppression_db"]
            > rows["tukey"]["crosstalk_suppression_db"] > 0)

    assert rows["none"]["throughput_vs_hard_db"] == pytest.approx(0.0, abs=1e-9)
    assert rows["hann"]["throughput_vs_hard_db"] < \
        rows["tukey"]["throughput_vs_hard_db"] < 0


def test_a_beamlet_sweep_cannot_see_the_wings():
    """Guard the docstring's warning.

    The beamlet kernel replaces the true aperture transform with an envelope, so
    it reproduces spot positions and powers but not the wings -- and the wings
    are the whole subject.  Run on beamlets the sweep under-reports the
    suppression badly; this pins the discrepancy so nobody quietly rebuilds the
    sweep on the fast path.
    """
    wave = apodization_sweep(_build_system, windows=(None, "hann"),
                             target_positions=_TARGETS, leak_positions=_LEAKS,
                             radius=1.2e-3)["rows"][1]["crosstalk_suppression_db"]
    spot = apodization_sweep(lambda w: _build_system(w, spot_mode=True),
                             windows=(None, "hann"),
                             target_positions=_TARGETS, leak_positions=_LEAKS,
                             radius=1.2e-3)["rows"][1]["crosstalk_suppression_db"]
    assert wave > spot + 10


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _panel() -> ModulatorArray:
    return ModulatorArray.tiled(nx=1, ny=1, pitch=PITCH, region_size=REGION,
                                device=IdealModulator(), background="absorb",
                                default_wavelength=LAM)


def _build_system(window, spot_mode: bool = False):
    arr = _panel()
    arr.route(0, _TARGETS, z=Z, tiling="grid", apodize=window)
    src = fp.LaserArray(n=1, positions=[(0.0, 0.0)], wavelength=LAM, power=1e-3,
                        profile=fp.Rectangular(width=REGION), linewidth=0.0,
                        rin_dbc_hz=-300.0)
    det = fp.PowerMeter(positions_in=[(0.0, 0.0)], diameter=30e-3)
    return fp.build_relay(source=src, modulator=arr, detector=det, z=Z,
                          panel_grid=Grid.from_extent(900e-6, 900e-6, 1.0e-6),
                          screen_grid=Grid.from_extent(24e-3, 24e-3, 20e-6),
                          spot_mode=spot_mode)
