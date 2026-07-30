"""Grating order efficiencies and deflection geometry."""

import numpy as np
import pytest

from fsphotonix import engine as eng
from fsphotonix.fields import Channel, FieldSet
from fsphotonix.grid import Grid
from fsphotonix.modulators.gratings import (
    BinaryPhaseGrating,
    BlazedGrating,
    CrossedGrating,
    DammannGrating,
    FresnelLens,
    MultiSpotHologram,
    RonchiGrating,
    SinusoidalPhaseGrating,
    SpotFocuser,
    Superposition,
    target_to_angles,
)
from fsphotonix.propagation import FresnelZoom
from fsphotonix.sources.profiles import Rectangular

LAM = 1550e-9


def eff(orders, mx, my=0):
    for o in orders:
        if (o.mx, o.my) == (mx, my):
            return o.efficiency
    return 0.0


def test_blazed_puts_everything_in_first_order():
    g = BlazedGrating(period=20e-6, design_wavelength=LAM)
    o = g.orders()
    assert eff(o, 1) == pytest.approx(1.0, abs=0.02)
    assert eff(o, 0) < 1e-3


def test_blazed_quantisation_matches_sinc_law():
    for levels in (2, 4, 8, 16):
        g = BlazedGrating(period=32e-6, levels=levels, design_wavelength=LAM)
        want = np.sinc(1.0 / levels) ** 2
        assert eff(g.orders(), 1) == pytest.approx(want, rel=0.02)
        assert g.ideal_efficiency() == pytest.approx(want, rel=1e-6)


def test_blazed_partial_stroke():
    g = BlazedGrating(period=20e-6, blaze_depth=0.6, design_wavelength=LAM)
    assert eff(g.orders(), 1) == pytest.approx(np.sinc(0.6 - 1.0) ** 2, rel=0.02)


def test_binary_phase_symmetric_orders():
    g = BinaryPhaseGrating(period=20e-6, design_wavelength=LAM)
    o = g.orders()
    assert eff(o, 1) == pytest.approx(4 / np.pi**2, rel=0.02)
    assert eff(o, -1) == pytest.approx(4 / np.pi**2, rel=0.02)
    assert eff(o, 0) < 1e-6


def test_ronchi_amplitude_orders():
    g = RonchiGrating(period=20e-6, design_wavelength=LAM)
    o = g.orders()
    assert eff(o, 0) == pytest.approx(0.25, rel=0.02)
    assert eff(o, 1) == pytest.approx(1 / np.pi**2, rel=0.03)


def test_sinusoidal_bessel_orders():
    from scipy.special import jv

    depth = 1.8
    g = SinusoidalPhaseGrating(period=20e-6, depth=depth, design_wavelength=LAM)
    o = g.orders()
    for m in (0, 1, 2):
        assert eff(o, m) == pytest.approx(jv(m, depth) ** 2, rel=0.03)


def test_deflection_angle_follows_grating_equation():
    theta = np.radians(2.0)
    g = BlazedGrating.from_deflection(theta, 0.0, LAM)
    assert g.period == pytest.approx(LAM / np.sin(theta), rel=1e-9)
    o = [x for x in g.orders() if x.mx == 1][0]
    assert o.angles(LAM)[0] == pytest.approx(theta, rel=1e-6)


def test_dispersion_walks_orders_with_wavelength():
    g = BlazedGrating(period=20e-6, design_wavelength=LAM)
    a1 = [o for o in g.orders(LAM) if o.mx == 1][0].angles(LAM)[0]
    a2 = [o for o in g.orders(1310e-9) if o.mx == 1][0].angles(1310e-9)[0]
    assert a2 < a1  # shorter wavelength deflects less
    assert np.sin(a2) / np.sin(a1) == pytest.approx(1310 / 1550, rel=1e-6)


def test_dammann_uniform_fanout():
    for n in (3, 5):
        g = DammannGrating(period=60e-6, n_x=n, design_wavelength=LAM)
        assert g.uniformity() < 0.2
        want = [o for o in g.orders(max_order=n) if abs(o.mx) <= n // 2]
        assert len(want) == n
        assert sum(o.efficiency for o in want) > 0.6


def test_crossed_grating_2d_lattice():
    g = CrossedGrating(period_x=20e-6, period_y=30e-6, kind="binary",
                       design_wavelength=LAM)
    o = g.orders(max_order=2)
    got = {(x.mx, x.my) for x in o if x.efficiency > 0.02}
    assert (1, 1) in got and (-1, -1) in got


def test_spot_focuser_targets_the_requested_point():
    z = 0.08
    g = SpotFocuser(target_x=3e-3, target_y=-1e-3, range_z=z, design_wavelength=LAM)
    o = g.orders()[0]
    x, y = o.screen_position(LAM, z)
    assert x == pytest.approx(3e-3, rel=1e-6)
    assert y == pytest.approx(-1e-3, rel=1e-6)


def test_target_to_angles_is_exact_not_paraxial():
    tx, ty = target_to_angles(30e-3, 0.0, 40e-3)
    assert np.sin(tx) == pytest.approx(30 / 50, rel=1e-12)


def test_multispot_hologram_hits_all_targets():
    z = 0.06
    targets = [(-2e-3, 0.0), (0.0, 0.0), (2e-3, 0.0), (0.0, 2e-3)]
    g = MultiSpotHologram(targets=targets, range_z=z, design_wavelength=LAM,
                          aperture_size=(240e-6, 240e-6))
    o = g.orders()
    assert len(o) == len(targets)
    e = np.array([x.efficiency for x in o])
    assert e.min() > 0.05  # every target actually gets light
    assert g.uniformity() < 0.6


def test_superposition_reaches_all_constituent_orders():
    z = 0.06
    gs = [SpotFocuser.from_target(dx, 0.0, z, LAM) for dx in (-2e-3, 2e-3)]
    s = Superposition(gratings=gs)
    o = s.orders()
    assert len(o) == 2
    assert all(x.efficiency > 0.2 for x in o)


def test_fresnel_lens_chromatic_focal_shift():
    lens = FresnelLens(focal_length=0.1, design_wavelength=LAM)
    assert lens.focus_shift(1310e-9) > 0  # shorter wavelength focuses further out
    assert lens.focus_shift(LAM) == pytest.approx(0.0)


def test_wave_optics_agrees_with_analytic_order_positions():
    """Propagate a real field through a blazed grating and find the spot."""
    z = 0.05
    d = 400e-6
    period = 20e-6
    g = BlazedGrating(period=period, design_wavelength=LAM)
    grid = Grid.from_extent(600e-6, 600e-6, 0.5e-6)
    X, Y = grid.meshgrid()
    u = Rectangular(width=d, height=d).field(grid, LAM, power=1e-3)
    u = u * g.complex_transmittance(X, Y, LAM)
    f0 = FieldSet(grid=grid, channels=[Channel(u=u, wavelength=LAM, coherence="c")])
    want_x = [o for o in g.orders() if o.mx == 1][0].screen_position(LAM, z)[0]
    screen = Grid.from_extent(1.2e-3, 0.4e-3, 4e-6, x0=want_x)
    out = FresnelZoom(z=z, out_grid=screen).propagate(f0)
    I = eng.to_numpy(out.mean_intensity(1))
    xs = eng.to_numpy(screen.x())
    peak_x = xs[int(np.argmax(I.sum(axis=0)))]
    assert peak_x == pytest.approx(want_x, abs=3 * LAM * z / d)
