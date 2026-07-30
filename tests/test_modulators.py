"""Modulator array: geometry, routing, hardware models."""

import numpy as np
import pytest

from fsphotonix import engine as eng
from fsphotonix.fields import Channel, FieldSet
from fsphotonix.grid import Grid
from fsphotonix.modulators import (
    DMD,
    Annulus,
    Circle,
    Hexagon,
    IdealModulator,
    LCoSPhaseSLM,
    ModulatorArray,
    PistonMicroMirrorArray,
    Rect,
    Region,
    Sector,
    SubRegion,
    tile_region,
)
from fsphotonix.modulators.gratings import BlazedGrating, FlatMirror
from fsphotonix.sources.profiles import SuperGaussian

LAM = 1550e-9


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def test_aperture_areas():
    assert Rect(width=2e-3, height=1e-3).area() == pytest.approx(2e-6)
    assert Circle(diameter=2e-3).area() == pytest.approx(np.pi * 1e-6)
    assert Annulus(outer_diameter=2e-3, inner_diameter=1e-3).area() == pytest.approx(
        np.pi * (1e-6 - 0.25e-6))
    assert Hexagon(across_flats=1e-3).area() == pytest.approx(np.sqrt(3) / 2 * 1e-6)


def test_aperture_mask_covers_expected_fraction():
    g = Grid.from_extent(2e-3, 2e-3, 2e-6)
    m = eng.to_numpy(Circle(diameter=1e-3).mask(g))
    frac = m.mean() * g.width * g.height
    assert frac == pytest.approx(np.pi * 0.25e-6, rel=0.01)


def test_sector_covers_a_quarter():
    g = Grid.from_extent(2e-3, 2e-3, 2e-6)
    full = eng.to_numpy(Circle(diameter=1e-3).mask(g)).sum()
    quarter = eng.to_numpy(
        Sector(outer_diameter=1e-3, sweep=np.pi / 2).mask(g)).sum()
    assert quarter / full == pytest.approx(0.25, rel=0.03)


@pytest.mark.parametrize("mode", ["stripes", "grid", "sectors", "rings", "interleave"])
def test_tiling_partitions_the_region(mode):
    ap = Rect(width=1e-3, height=1e-3)
    subs = tile_region(ap, 4, mode=mode)
    assert len(subs) == 4
    g = Grid.from_extent(1.4e-3, 1.4e-3, 2e-6)
    total = np.zeros(g.shape)
    for sub, off in subs:
        total += eng.to_numpy(sub.mask(g, *off)) * eng.to_numpy(ap.mask(g))
    # tiles should not overlap and should cover most of the region
    assert total.max() <= 1.001
    covered = total.sum() / eng.to_numpy(ap.mask(g)).sum()
    assert covered > 0.7


# --------------------------------------------------------------------------- #
# array construction and routing
# --------------------------------------------------------------------------- #
def test_tiled_array_layout():
    m = ModulatorArray.tiled(nx=4, ny=2, pitch=1e-3, region_size=800e-6)
    assert len(m) == 8
    c = np.array([r.center for r in m.regions])
    assert np.ptp(c[:, 0]) == pytest.approx(3e-3)
    assert np.ptp(c[:, 1]) == pytest.approx(1e-3)


def test_route_creates_one_subregion_per_target():
    m = ModulatorArray.tiled(nx=2, ny=1, pitch=1e-3, region_size=800e-6,
                             device=IdealModulator())
    targets = [(-3e-3, 0.0), (0.0, 0.0), (3e-3, 0.0), (0.0, 3e-3)]
    m.route(0, targets, z=0.08, wavelength=LAM)
    assert len(m[0].subregions) == 4


def test_route_absolute_targets_account_for_region_position():
    """A region off-axis must deflect differently to reach the same screen point."""
    m = ModulatorArray.tiled(nx=2, ny=1, pitch=4e-3, region_size=1e-3,
                             device=IdealModulator())
    z = 0.1
    m.broadcast([(0.0, 0.0)], z=z, wavelength=LAM, mode="subregions", grating="blazed")
    for r in m.regions:
        o = r.subregions[0].grating.orders(LAM)[0]
        x, y = o.screen_position(LAM, z)
        assert r.center[0] + x == pytest.approx(0.0, abs=1e-9)


def test_set_matrix_programs_columns():
    m = ModulatorArray.tiled(nx=3, ny=1, pitch=1e-3, region_size=800e-6,
                             device=IdealModulator())
    W = np.array([[1.0, 0.0, 0.5], [0.0, 2.0, 0.5]])
    outs = np.array([[-2e-3, 0.0], [2e-3, 0.0]])
    m.set_matrix(W, outs, z=0.08)
    assert len(m[0].subregions) == 1  # only output 0 is fed
    assert len(m[2].subregions) == 2
    assert m[1].subregions[0].weight == pytest.approx(2.0)


def test_transmittance_is_cached_and_invalidated():
    m = ModulatorArray.tiled(nx=2, ny=1, pitch=1e-3, region_size=800e-6,
                             device=IdealModulator())
    g = Grid.from_extent(3e-3, 1.5e-3, 4e-6)
    a = m.transmittance(g, LAM)
    b = m.transmittance(g, LAM)
    assert a is b
    m.set_pistons([0.3, 0.0])
    c = m.transmittance(g, LAM)
    assert c is not b


def test_piston_applies_uniform_phase():
    m = ModulatorArray.tiled(nx=1, ny=1, pitch=1e-3, region_size=800e-6,
                             device=IdealModulator())
    g = Grid.from_extent(1.2e-3, 1.2e-3, 4e-6)
    t0 = eng.to_numpy(m.transmittance(g, LAM))
    m.set_pistons([np.pi / 2])
    t1 = eng.to_numpy(m.transmittance(g, LAM))
    inside = np.abs(t0) > 0.5
    ratio = t1[inside] / t0[inside]
    assert np.allclose(ratio, 1j, atol=1e-6)


def test_gain_scales_amplitude():
    m = ModulatorArray.tiled(nx=1, ny=1, pitch=1e-3, region_size=800e-6,
                             device=IdealModulator())
    g = Grid.from_extent(1.2e-3, 1.2e-3, 4e-6)
    m.set_gains([0.25])
    t = eng.to_numpy(m.transmittance(g, LAM))
    assert np.abs(t).max() == pytest.approx(0.25, rel=1e-6)


def test_apply_conserves_or_reduces_power():
    m = ModulatorArray.tiled(nx=2, ny=1, pitch=1e-3, region_size=900e-6,
                             device=IdealModulator())
    g = Grid.from_extent(3e-3, 1.5e-3, 3e-6)
    u = SuperGaussian(diameter=2.5e-3).field(g, LAM, power=1e-3)
    f0 = FieldSet(grid=g, channels=[Channel(u=u, wavelength=LAM, coherence="c")])
    f1 = m.apply(f0)
    assert 0 < f1.total_power() < f0.total_power()


# --------------------------------------------------------------------------- #
# hardware models
# --------------------------------------------------------------------------- #
def test_lcos_quantises_phase():
    dev = LCoSPhaseSLM(phase_bits=2, crosstalk=0.0, fill_factor=1.0, efficiency=1.0,
                       flatness_rms=0.0, pitch=4e-6)
    g = Grid.from_extent(200e-6, 200e-6, 4e-6)
    X, Y = g.meshgrid()
    ideal = eng.xp.exp(1j * eng.xp.mod(2 * np.pi * X / 40e-6, 2 * np.pi))
    out = eng.to_numpy(dev.render(ideal, g, LAM))
    levels = np.unique(np.round(np.angle(out), 4))
    assert len(levels) <= 4


def test_piston_stroke_limits_phase():
    dev = PistonMicroMirrorArray(stroke=100e-9, phase_bits=None, crosstalk=0.0,
                                 fill_factor=1.0, efficiency=1.0, flatness_rms=0.0,
                                 dead_pixel_fraction=0.0, pitch=4e-6)
    # 2*2pi*h/lambda at 1550 nm with h=100 nm -> 0.81 rad of stroke
    assert dev.max_phase == pytest.approx(2 * np.pi * 2 * 100e-9 / LAM, rel=1e-9)
    g = Grid.from_extent(200e-6, 200e-6, 4e-6)
    X, Y = g.meshgrid()
    ideal = eng.xp.exp(1j * eng.xp.mod(2 * np.pi * X / 40e-6, 2 * np.pi))
    out = eng.to_numpy(dev.render(ideal, g, LAM))
    assert np.angle(out).max() <= dev.max_phase + 1e-6


def test_reflective_piston_doubles_phase():
    r = PistonMicroMirrorArray(stroke=200e-9, reflective=True)
    t = PistonMicroMirrorArray(stroke=200e-9, reflective=False)
    assert r.max_phase == pytest.approx(2 * t.max_phase)


def test_dmd_is_binary_amplitude():
    dev = DMD(encoding="amplitude", fill_factor=1.0, efficiency=1.0, flatness_rms=0.0,
              dead_pixel_fraction=0.0, include_facet_tilt=False, pitch=4e-6)
    g = Grid.from_extent(200e-6, 200e-6, 4e-6)
    X, Y = g.meshgrid()
    ideal = (0.5 + 0.5 * eng.xp.cos(2 * np.pi * X / 40e-6)).astype(eng.COMPLEX)
    out = eng.to_numpy(dev.render(ideal, g, LAM))
    assert set(np.unique(np.round(np.abs(out), 6))) <= {0.0, 1.0}


def test_dmd_lee_encoding_keeps_phase_information():
    plain = DMD(encoding="amplitude", pitch=4e-6)
    lee = DMD(encoding="lee", pitch=4e-6)
    g = Grid.from_extent(200e-6, 200e-6, 4e-6)
    X, Y = g.meshgrid()
    ideal = 0.6 * eng.xp.exp(1j * 2 * np.pi * X / 60e-6)
    a = eng.to_numpy(plain.render(ideal, g, LAM))
    b = eng.to_numpy(lee.render(ideal, g, LAM))
    # the Lee hologram writes the phase into a binary fringe pattern; a plain
    # amplitude threshold of a uniform-amplitude field is featureless
    assert np.std(np.abs(b)) > 10 * np.std(np.abs(a))


def test_fill_factor_creates_dead_space():
    dev = LCoSPhaseSLM(fill_factor=0.5, crosstalk=0.0, efficiency=1.0,
                       flatness_rms=0.0, pitch=8e-6, phase_bits=None)
    # the dead-space mask is sampled on the field grid, so it needs a good few
    # samples per device pixel before it reproduces the nominal fill factor
    g = Grid.from_extent(400e-6, 400e-6, 8e-6 / 32)
    m = eng.to_numpy(dev.aperture_mask(g))
    assert m.mean() == pytest.approx(0.5, rel=0.1)


def test_device_max_deflection():
    dev = LCoSPhaseSLM(pitch=8e-6)
    assert np.degrees(dev.max_deflection(LAM)) == pytest.approx(
        np.degrees(np.arcsin(LAM / 16e-6)), rel=1e-9)


def test_crosstalk_smooths_the_pattern():
    sharp = LCoSPhaseSLM(crosstalk=0.0, fill_factor=1.0, flatness_rms=0.0, pitch=8e-6)
    soft = LCoSPhaseSLM(crosstalk=1.0, fill_factor=1.0, flatness_rms=0.0, pitch=8e-6)
    g = Grid.from_extent(400e-6, 400e-6, 2e-6)
    X, Y = g.meshgrid()
    ideal = eng.xp.exp(1j * np.pi * (eng.xp.mod(X / 16e-6, 1.0) < 0.5))
    a = eng.to_numpy(sharp.render(ideal, g, LAM))
    b = eng.to_numpy(soft.render(ideal, g, LAM))
    assert np.std(np.abs(np.diff(b, axis=1))) < np.std(np.abs(np.diff(a, axis=1)))
