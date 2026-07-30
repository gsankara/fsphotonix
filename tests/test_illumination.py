"""Collimator arrays, tunable illumination area, and the two hologram gaps.

These cover the coupled design problem on the input side: a lenslet sets how much
of a region is lit, the guard gap between regions decides how much of that light
lands on the neighbouring input's hologram, and the guard gap *inside* a region
decides how big each sub-grating (and therefore each screen spot) is.
"""

import numpy as np
import pytest

import fsphotonix as fp
from fsphotonix import engine as eng
from fsphotonix.analysis import (
    cgh_gap_report,
    illumination_crosstalk,
    illumination_map,
    illumination_sweep,
)
from fsphotonix.sources.collimator import CollimatedArray, Collimator, MicrolensArray

LAM = 1550e-9


# --------------------------------------------------------------------------- #
# collimator optics
# --------------------------------------------------------------------------- #
def test_focal_plane_emitter_gives_the_textbook_collimated_waist():
    c = Collimator(mfd=10.4e-6, focal_length=1.5e-3, defocus=0.0)
    want = LAM * c.focal_length / (np.pi * (10.4e-6 / 2))
    assert c.output_waist(LAM) == pytest.approx(want, rel=0.02)
    # and it stays collimated over a short throw
    d0 = c.illumination_diameter(0.0, LAM)
    d1 = c.illumination_diameter(5e-3, LAM)
    assert d1 / d0 == pytest.approx(1.0, rel=0.35)


def test_defocus_tunes_the_illuminated_area():
    c = Collimator(mfd=10.4e-6, focal_length=1.5e-3, clear_aperture=400e-6)
    L = 6e-3
    # negative defocus = emitter inside the focal length = diverging at the panel
    grow = [c.footprint_for(x, L, LAM) for x in (0.0, -40e-6, -100e-6, -200e-6)]
    assert grow == sorted(grow)
    # positive defocus pushes the waist downstream, so the footprint shrinks first
    shrink = [c.footprint_for(x, L, LAM) for x in (0.0, 40e-6, 100e-6, 200e-6)]
    assert shrink == sorted(shrink, reverse=True)


def test_solve_defocus_hits_the_requested_footprint():
    c = Collimator(mfd=10.4e-6, focal_length=1.5e-3, clear_aperture=500e-6)
    L = 6e-3
    for want in (400e-6, 800e-6):
        delta = c.solve_defocus(want, L, LAM)
        got = Collimator(mfd=c.mfd, focal_length=c.focal_length,
                         clear_aperture=c.clear_aperture,
                         defocus=delta).illumination_diameter(L, LAM)
        assert got == pytest.approx(want, rel=0.02)


def test_overfilling_the_lenslet_costs_power():
    tight = Collimator(mfd=10.4e-6, focal_length=1.5e-3, clear_aperture=400e-6)
    small = Collimator(mfd=10.4e-6, focal_length=1.5e-3, clear_aperture=40e-6)
    assert tight.truncation_transmission(LAM) > 0.97
    assert small.truncation_transmission(LAM) < 0.95
    assert small.report(6e-3, LAM)["overfills_lenslet"]


def test_collimated_array_footprint_matches_the_request():
    src = CollimatedArray(n=4, pitch=500e-6, distance=6e-3,
                          illumination_diameter=350e-6, wavelength=LAM, power=1e-3)
    assert src.footprint == pytest.approx(350e-6, rel=0.03)
    grid = fp.Grid.from_extent(3e-3, 1e-3, 4e-6)
    f = src.emit(grid)
    assert f.total_power() == pytest.approx(4e-3 * src.truncation, rel=0.02)


def test_microlens_array_focuses_a_plane_wave_per_lenslet():
    """Wave-optics lenslet array: a flat input becomes a grid of foci."""
    f = 2e-3
    pitch = 200e-6
    grid = fp.Grid.from_extent(800e-6, 800e-6, 1e-6)
    u = fp.sources.PlaneWave().field(grid, LAM, power=1e-3)
    fs = fp.FieldSet(grid=grid, channels=[
        fp.Channel(u=u, wavelength=LAM, coherence="a")])
    lensed = MicrolensArray(focal_length=f, pitch=pitch,
                            clear_aperture=0.95 * pitch).propagate(fs, z=0.0)
    out = fp.FresnelZoom(z=f, out_grid=grid).propagate(lensed)
    I = eng.to_numpy(out.mean_intensity(1))
    row = I[grid.ny // 2]
    peaks = [i for i in range(1, len(row) - 1)
             if row[i] > row[i - 1] and row[i] > row[i + 1] and row[i] > 0.2 * row.max()]
    xs = eng.to_numpy(grid.x())[peaks]
    assert len(xs) >= 3
    assert np.median(np.diff(xs)) == pytest.approx(pitch, rel=0.15)


# --------------------------------------------------------------------------- #
# illumination cross-talk driven by the footprint and the inter-input gap
# --------------------------------------------------------------------------- #
def make_panel(gap: float, pitch: float = 500e-6, n: int = 3):
    return fp.ModulatorArray.tiled(nx=n, ny=1, pitch=pitch, gap=gap,
                                   device="ideal", default_wavelength=LAM)


def test_gap_sets_the_region_size():
    p = make_panel(gap=100e-6, pitch=500e-6)
    assert p[0].aperture.size[0] == pytest.approx(400e-6)
    assert p.region_gap[0] == pytest.approx(100e-6)
    with pytest.raises(ValueError):
        fp.ModulatorArray.tiled(nx=2, pitch=500e-6, gap=100e-6, region_size=300e-6)


def test_tight_illumination_is_captured_wide_illumination_crosses_over():
    pitch = 500e-6
    panel = make_panel(gap=100e-6, pitch=pitch)
    grid = fp.Grid.from_extent(3e-3, 1e-3, 4e-6)
    res = {}
    for d in (250e-6, 900e-6):
        src = CollimatedArray(n=3, pitch=pitch, distance=6e-3,
                              illumination_diameter=d, wavelength=LAM, power=1e-3)
        res[d] = illumination_crosstalk(src.emit(grid), panel)
    tight, wide = res[250e-6], res[900e-6]
    assert np.mean(tight["capture_efficiency"]) > np.mean(wide["capture_efficiency"])
    assert np.nanmean(tight["crosstalk_db"]) < np.nanmean(wide["crosstalk_db"]) - 6
    assert np.nanmean(tight["crosstalk_db"]) < -15


def test_wider_guard_gap_suppresses_illumination_crosstalk():
    pitch = 500e-6
    grid = fp.Grid.from_extent(3e-3, 1e-3, 4e-6)
    src = CollimatedArray(n=3, pitch=pitch, distance=6e-3,
                          illumination_diameter=600e-6, wavelength=LAM, power=1e-3)
    f = src.emit(grid)
    xt = [np.nanmean(illumination_crosstalk(f, make_panel(gap=g, pitch=pitch))
                     ["crosstalk_db"]) for g in (40e-6, 200e-6)]
    assert xt[1] < xt[0] - 3  # a bigger guard band buys cross-talk margin


def test_guard_gap_costs_capture_efficiency():
    """The other side of the trade: guard band means light thrown away."""
    pitch = 500e-6
    grid = fp.Grid.from_extent(3e-3, 1e-3, 4e-6)
    src = CollimatedArray(n=3, pitch=pitch, distance=6e-3,
                          illumination_diameter=600e-6, wavelength=LAM, power=1e-3)
    f = src.emit(grid)
    cap = [np.mean(illumination_crosstalk(f, make_panel(gap=g, pitch=pitch))
                   ["capture_efficiency"]) for g in (40e-6, 200e-6)]
    assert cap[1] < cap[0]


def test_illumination_map_accounts_for_all_the_power():
    pitch = 500e-6
    panel = make_panel(gap=120e-6, pitch=pitch)
    grid = fp.Grid.from_extent(3e-3, 1e-3, 4e-6)
    src = CollimatedArray(n=3, pitch=pitch, distance=6e-3,
                          illumination_diameter=400e-6, wavelength=LAM, power=1e-3)
    f = src.emit(grid)
    m = illumination_map(f, panel)
    on_panel = m["power_W"].sum(axis=1)
    assert np.all(on_panel + m["spill_W"] <= m["incident_W"] * 1.001 + 1e-15)
    assert np.all(m["spill_fraction"] >= 0)


def test_illumination_sweep_shows_the_tradeoff_curve():
    pitch = 500e-6
    panel = make_panel(gap=100e-6, pitch=pitch)
    grid = fp.Grid.from_extent(3e-3, 1e-3, 5e-6)
    sweep = illumination_sweep(
        lambda d: CollimatedArray(n=3, pitch=pitch, distance=6e-3,
                                  illumination_diameter=d, wavelength=LAM, power=1e-3),
        panel, grid, diameters=[200e-6, 400e-6, 700e-6, 1000e-6])
    assert np.all(np.diff(sweep["crosstalk_db"]) > 0)  # cross-talk only grows
    assert sweep["capture"][0] > sweep["capture"][-1]


def test_lenslet_leakage_is_an_independent_crosstalk_route():
    pitch = 500e-6
    panel = make_panel(gap=150e-6, pitch=pitch)
    grid = fp.Grid.from_extent(3e-3, 1e-3, 4e-6)
    clean = CollimatedArray(n=3, pitch=pitch, distance=6e-3,
                            illumination_diameter=300e-6, wavelength=LAM, power=1e-3)
    leaky = CollimatedArray(n=3, pitch=pitch, distance=6e-3,
                            illumination_diameter=300e-6, wavelength=LAM, power=1e-3,
                            stray_fraction=0.05, stray_diameter_factor=8.0)
    a = np.nanmean(illumination_crosstalk(clean.emit(grid), panel)["crosstalk_db"])
    b = np.nanmean(illumination_crosstalk(leaky.emit(grid), panel,
                                          mapping=[0, 0, 1, 1, 2, 2])["crosstalk_db"])
    assert b > a + 3


# --------------------------------------------------------------------------- #
# the gap between the CGHs of one input
# --------------------------------------------------------------------------- #
def test_subregion_gap_shrinks_the_subapertures():
    z = 0.08
    targets = [(-3e-3, 0.0), (0.0, 0.0), (3e-3, 0.0)]
    reports = []
    for gap in (0.0, 60e-6):
        panel = make_panel(gap=100e-6, pitch=500e-6, n=1)
        panel.route(0, targets, z=z, wavelength=LAM, tiling="stripes", gap=gap)
        reports.append(cgh_gap_report(panel, z=z, wavelength=LAM))
    assert reports[1]["subaperture_um"] < reports[0]["subaperture_um"]
    assert reports[1]["spot_size_um"] > reports[0]["spot_size_um"]
    assert reports[1]["aperture_loss_db"] < reports[0]["aperture_loss_db"] + 1e-9


def test_gap_report_flags_an_infeasible_fanout():
    """Too many targets per region -> spots wider than the target pitch."""
    z = 0.1
    panel = make_panel(gap=60e-6, pitch=500e-6, n=1)
    dense = [(x, 0.0) for x in np.linspace(-1e-3, 1e-3, 9)]
    panel.route(0, dense, z=z, wavelength=LAM, tiling="grid")
    bad = cgh_gap_report(panel, z=z, wavelength=LAM)
    assert bad["detector_guard_ok"] is False

    sparse = [(x, 0.0) for x in np.linspace(-8e-3, 8e-3, 2)]
    panel.route(0, sparse, z=z, wavelength=LAM, tiling="stripes")
    good = cgh_gap_report(panel, z=z, wavelength=LAM)
    assert good["detector_guard_ok"] is True


def test_gap_report_geometry_matches_the_layout():
    panel = make_panel(gap=120e-6, pitch=600e-6, n=4)
    panel.broadcast([(-4e-3, 0.0), (4e-3, 0.0)], z=0.08, wavelength=LAM,
                    tiling="stripes", gap=40e-6)
    r = panel.gap_report(z=0.08, wavelength=LAM)
    assert r["inter_input_gap_um"] == pytest.approx(120.0, rel=0.05)
    assert r["region_size_um"][0] == pytest.approx(480.0, rel=0.01)
    assert r["subregions_per_region"][0] == 2
    assert r["target_pitch_um"] == pytest.approx(8000.0, rel=0.01)
