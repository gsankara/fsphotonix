"""End-to-end systems: fan-out, cascades, spot propagation, WDM dispersion."""

import numpy as np
import pytest

import fsphotonix as fp
from fsphotonix import engine as eng
from fsphotonix.analysis import crosstalk_db, power_budget, spot_metrics, transfer_matrix

LAM = 1550e-9


def make_fanout_system(n_targets=3, z=0.06, spot_mode=False, device="ideal"):
    pitch = 1.2e-3
    panel = fp.ModulatorArray.tiled(nx=1, ny=1, pitch=pitch, region_size=600e-6,
                                    device=device, default_wavelength=LAM)
    targets = [((i - (n_targets - 1) / 2) * 3e-3, 0.0) for i in range(n_targets)]
    panel.route(0, targets, z=z, wavelength=LAM, tiling="stripes")
    src = fp.CoherentLaser(wavelength=LAM, power=1e-3,
                           profile=fp.SuperGaussian(diameter=600e-6, order=12))
    cam = fp.CameraArray(nx=180, ny=24, pitch=40e-6, output="watts",
                         integration_time=1e-6, read_noise_e=0.0, prnu=0.0,
                         dark_current_e_s=0.0, full_well_e=1e15, fill_factor=1.0)
    grid = fp.Grid.from_extent(1.2e-3, 1.2e-3, 0.5e-6 if not spot_mode else 40e-6)
    screen = fp.Grid.from_extent(180 * 40e-6, 24 * 40e-6, 40e-6)
    return fp.OpticalSystem([
        fp.SourceStage(source=src, grid=grid, z=0.0, name="src"),
        fp.ModulatorStage(modulator=panel, grid=grid, z=0.0, name="panel",
                          spot_mode=spot_mode),
        fp.DetectorStage(detector=cam, grid=screen, z=z, name="screen"),
    ]), panel, targets


def test_three_way_fanout_lands_on_target():
    z = 0.06
    system, panel, targets = make_fanout_system(3, z)
    trace = system.run()
    fields = trace.fields["screen"]
    m = spot_metrics(fields, [(t[0], t[1]) for t in targets], radius=400e-6)
    assert m["collected_fraction"] > 0.5
    assert np.all(m["centroid_error_m"] < 60e-6)
    # the flat-top illumination still rolls off at the edges, so the outer stripes
    # collect a little less than the centre one
    assert m["uniformity"] < 0.3


def test_power_budget_is_monotonic():
    system, _, _ = make_fanout_system(3, 0.06)
    tr = system.run()
    b = power_budget(tr)
    assert b["power_W"]["src"] >= b["power_W"]["panel"] > 0
    assert all(v <= 0.2 for v in b["hop_loss_dB"].values())


def test_spot_mode_agrees_with_wave_optics():
    """The beamlet propagator must land the spots where diffraction does."""
    z = 0.06
    wave, panel_w, targets = make_fanout_system(3, z, spot_mode=False)
    spot, panel_s, _ = make_fanout_system(3, z, spot_mode=True)
    tw = wave.run()
    ts = spot.run()
    pw = spot_metrics(tw.fields["screen"], targets, radius=400e-6)["power_W"]
    ps = spot_metrics(ts.fields["screen"], targets, radius=400e-6)["power_W"]
    assert np.all(ps > 0)
    # same relative distribution to within the beamlet approximation
    rw, rs = pw / pw.sum(), ps / ps.sum()
    assert np.max(np.abs(rw - rs)) < 0.12
    assert ps.sum() == pytest.approx(pw.sum(), rel=0.35)


def test_spot_list_is_analytic_and_cheap():
    z = 0.06
    system, panel, targets = make_fanout_system(4, z, spot_mode=True)
    tr = system.run()
    spots = tr.spot_lists["panel"]
    assert len(spots) >= 4
    got = spots.bin_to(targets, radius=300e-6)
    assert np.all(got > 0)
    t = spots.table()
    assert t["power"].sum() > 0


def test_many_regions_scale_in_spot_mode():
    """100 regions x 2 targets = 200 beamlets, no giant FFT anywhere."""
    z = 0.2
    n_side, pitch, region = 10, 500e-6, 440e-6
    panel = fp.ModulatorArray.tiled(nx=n_side, ny=n_side, pitch=pitch,
                                    region_size=region, device="ipms_piston",
                                    default_wavelength=LAM)
    targets = [(-6e-3, 0.0), (6e-3, 0.0)]
    # a panel can only steer as far as two device pixels per grating period; ask for
    # more and the written grating aliases instead of pointing where it was told
    reach = 6e-3 + n_side * pitch / 2
    assert np.arctan2(reach, z) < panel.device.max_deflection(LAM)
    panel.broadcast(targets, z=z, wavelength=LAM, mode="subregions", tiling="grid",
                    grating="blazed")
    src = fp.LaserArray(n=n_side, ny=n_side, pitch=pitch, power=100e-6, linewidth=0.0,
                        rin_dbc_hz=-300,
                        profile=fp.SuperGaussian(diameter=region, order=10))
    grid = fp.Grid.from_extent(n_side * pitch * 1.1, n_side * pitch * 1.1, 50e-6)
    det = fp.PhotodiodeArray(positions_in=targets, active_diameter=4e-3,
                             output="watts", n_time=1, dark_current=0.0)
    screen = fp.Grid.from_extent(20e-3, 8e-3, 200e-6)
    system = fp.OpticalSystem([
        fp.SourceStage(source=src, grid=grid, z=0.0, name="src"),
        fp.ModulatorStage(modulator=panel, grid=grid, z=0.0, name="panel",
                          spot_mode=True),
        fp.DetectorStage(detector=det, grid=screen, z=z, name="screen"),
    ])
    tr = system.run()
    spots = tr.spot_lists["panel"]
    assert len(spots) >= n_side**2 * 2
    # the analytic table is the primary output: bin it without rendering anything
    binned = spots.bin_to(targets, radius=3e-3)
    assert np.all(binned > 0)
    assert float((binned.max() - binned.min()) / binned.max()) < 0.35
    v = tr.outputs["screen"].vector()
    assert np.all(v > 0)


def test_device_steering_limit_aliases_an_over_driven_grating():
    """Asking for more deflection than two pixels per period allows must not
    silently succeed: the written grating folds, as it would on real hardware."""
    z = 0.1
    coarse = fp.ModulatorArray.tiled(nx=1, ny=1, pitch=1e-3, region_size=600e-6,
                                     device="ipms_piston", default_wavelength=LAM)
    reach = 12e-3  # 6.8 deg, well past the 2.8 deg the 16 um pitch can write
    assert np.arctan2(reach, z) > coarse.device.max_deflection(LAM)
    coarse.route(0, [(reach, 0.0)], z=z, wavelength=LAM, grating="blazed")
    src = fp.CoherentLaser(power=1e-3, profile=fp.SuperGaussian(diameter=600e-6))
    grid = fp.Grid.from_extent(1.2e-3, 1.2e-3, 20e-6)
    sp = fp.SpotPropagator(z=z, out_grid=fp.Grid.from_extent(30e-3, 4e-3, 50e-6))
    landed = max((s for s in sp.spots(coarse, src.emit(grid)) if s.order != (0, 0)),
                 key=lambda s: s.power)
    assert abs(landed.x - reach) > 2e-3           # it did not go where it was asked
    off = fp.SpotPropagator(z=z, enforce_device_limit=False,
                            out_grid=fp.Grid.from_extent(30e-3, 4e-3, 50e-6))
    ideal = max((s for s in off.spots(coarse, src.emit(grid)) if s.order != (0, 0)),
                key=lambda s: s.power)
    assert ideal.x == pytest.approx(reach, rel=0.02)  # the request itself was fine


def test_wdm_channels_disperse_across_the_screen():
    """The same hologram sends different wavelengths to different places."""
    z = 0.08
    panel = fp.ModulatorArray.tiled(nx=1, ny=1, pitch=1e-3, region_size=500e-6,
                                    device="ideal", default_wavelength=LAM)
    panel.route(0, [(4e-3, 0.0)], z=z, wavelength=LAM, grating="blazed")
    src = fp.CoarseWDMBank(wavelengths=[1450e-9, 1550e-9, 1650e-9], power=1e-3,
                           positions=[(0, 0)] * 3,
                           profile=fp.SuperGaussian(diameter=500e-6))
    grid = fp.Grid.from_extent(1e-3, 1e-3, 40e-6)
    screen = fp.Grid.from_extent(16e-3, 2e-3, 40e-6)
    system = fp.OpticalSystem([
        fp.SourceStage(source=src, grid=grid, z=0.0, name="src"),
        fp.ModulatorStage(modulator=panel, grid=grid, z=0.0, name="panel",
                          spot_mode=True),
        fp.DetectorStage(detector=fp.PowerMeter(positions_in=[(0, 0)], diameter=1e-3),
                         grid=screen, z=z, name="screen"),
    ])
    tr = system.run()
    spots = tr.spot_lists["panel"]
    xs = {}
    for s in sorted(spots, key=lambda s: s.power):
        if s.order != (0, 0):
            xs[round(s.wavelength * 1e9)] = s.x  # brightest wins (sorted ascending)
    assert len(xs) == 3
    assert xs[1450] < xs[1550] < xs[1650]  # longer wavelength deflects further
    assert abs(xs[1550] - 4e-3) < 50e-6  # the design wavelength is on target


def test_two_stage_cascade_runs():
    """The 'detection screen replaced by another modulator array' case."""
    z1, z2 = 0.04, 0.04
    p1 = fp.ModulatorArray.tiled(nx=1, ny=1, pitch=1e-3, region_size=500e-6,
                                 device="ideal", default_wavelength=LAM)
    p1.route(0, [(-1.5e-3, 0.0), (1.5e-3, 0.0)], z=z1, wavelength=LAM,
             tiling="stripes")
    p2 = fp.ModulatorArray.from_positions([(-1.5e-3, 0.0), (1.5e-3, 0.0)],
                                          region_size=900e-6, device="ideal",
                                          default_wavelength=LAM)
    p2.broadcast([(0.0, 0.0)], z=z2, wavelength=LAM, grating="blazed")
    src = fp.CoherentLaser(power=1e-3, profile=fp.SuperGaussian(diameter=500e-6))
    panel_grid = fp.Grid.from_extent(5e-3, 2e-3, 2e-6)
    screen = fp.Grid.from_extent(2e-3, 1e-3, 5e-6)
    system = fp.OpticalSystem([
        fp.SourceStage(source=src, grid=panel_grid, z=0.0, name="src"),
        fp.ModulatorStage(modulator=p1, grid=panel_grid, z=0.0, name="stage1"),
        fp.ModulatorStage(modulator=p2, grid=panel_grid, z=z1, name="stage2"),
        fp.DetectorStage(detector=fp.PowerMeter(positions_in=[(0.0, 0.0)],
                                                diameter=600e-6),
                         grid=screen, z=z1 + z2, name="screen"),
    ])
    tr = system.run()
    assert tr.order == ["src", "stage1", "stage2", "screen"]
    recombined = float(tr.outputs["screen"].values[0])
    assert recombined > 0
    # both arms contribute: disabling one halves the recombined power
    p2[0].enabled = False
    p2.touch()
    half = float(system.run().outputs["screen"].values[0])
    assert 0 < half < recombined


def test_transfer_matrix_is_diagonal_for_a_shuffle():
    """Program a permutation and check the measured matrix reproduces it."""
    z = 0.08
    n = 4
    pitch = 700e-6
    panel = fp.ModulatorArray.tiled(nx=n, ny=1, pitch=pitch, region_size=600e-6,
                                    device="ideal", default_wavelength=LAM)
    outs = np.stack([np.linspace(-3e-3, 3e-3, n), np.zeros(n)], axis=1)
    perm = np.roll(np.eye(n), 1, axis=0)
    panel.set_matrix(perm, outs, z=z)
    src = fp.LaserArray(n=n, pitch=pitch, power=1e-3, linewidth=0.0, rin_dbc_hz=-300,
                        profile=fp.SuperGaussian(diameter=600e-6, order=10))
    grid = fp.Grid.from_extent(n * pitch * 1.3, 1e-3, 40e-6)
    det = fp.PhotodiodeArray(positions_in=outs, active_diameter=600e-6,
                             output="watts", n_time=1, dark_current=0.0,
                             tia_input_noise=0.0)
    screen = fp.Grid.from_extent(9e-3, 2e-3, 50e-6)
    system = fp.OpticalSystem([
        fp.SourceStage(source=src, grid=grid, z=0.0, name="src"),
        fp.ModulatorStage(modulator=panel, grid=grid, z=0.0, name="panel",
                          spot_mode=True),
        fp.DetectorStage(detector=det, grid=screen, z=z, name="screen"),
    ])
    T = transfer_matrix(system, "src", "screen")
    T = T / T.max()
    assert np.all(np.argmax(T, axis=0) == np.argmax(perm, axis=0))
    assert np.all(crosstalk_db(T) < -15)


def test_sampling_report_flags_the_regime():
    system, _, _ = make_fanout_system(3, 0.06)
    reps = system.sampling_report()
    assert reps
    assert all("recommended" in r for r in reps)


# --------------------------------------------------------------------------- #
# cascade incidence and beamlet direction
# --------------------------------------------------------------------------- #
def test_beamlets_carry_their_propagation_direction():
    """A downstream diffractive plane needs the incident angle, not just the spot."""
    z = 0.05
    panel = fp.ModulatorArray.tiled(nx=1, ny=1, pitch=1e-3, region_size=600e-6,
                                    device="ideal", default_wavelength=LAM)
    panel.route(0, [(3e-3, 0.0)], z=z, wavelength=LAM, grating="blazed")
    src = fp.CoherentLaser(power=1e-3, profile=fp.SuperGaussian(diameter=600e-6))
    grid = fp.Grid.from_extent(1.2e-3, 1.2e-3, 20e-6)
    sp = fp.SpotPropagator(z=z, out_grid=fp.Grid.from_extent(8e-3, 2e-3, 20e-6))
    spots = [s for s in sp.spots(panel, src.emit(grid)) if s.order != (0, 0)]
    assert spots
    s = max(spots, key=lambda s: s.power)
    # the beamlet's own direction must match where it landed
    assert s.alpha == pytest.approx(np.sin(np.arctan2(3e-3, z)), rel=1e-6)
    assert s.angles()[0] == pytest.approx(np.arctan2(3e-3, z), rel=1e-6)


def test_incidence_from_geometry():
    a, b = fp.incidence_from((0.0, 0.0), (3e-3, 4e-3), 12e-3)
    r = np.sqrt(3e-3**2 + 4e-3**2 + 12e-3**2)
    assert a == pytest.approx(3e-3 / r) and b == pytest.approx(4e-3 / r)


def test_incident_precompensation_repoints_an_off_axis_hologram():
    """A hologram fed off-axis mis-points unless it is designed for that incidence."""
    z = 0.06
    tilt = 0.02  # rad of incidence
    alpha = float(np.sin(tilt))
    # the grid must be able to represent that tilt, or the measured incident angle
    # aliases and the compensation is applied with the wrong sign
    grid = fp.Grid.from_extent(1.2e-3, 1.2e-3, 8e-6)
    assert grid.max_angle(LAM) > tilt
    src = fp.CoherentLaser(power=1e-3, wavelength=LAM,
                           profile=fp.SuperGaussian(diameter=600e-6, theta_x=tilt))
    fields = src.emit(grid)
    sp = fp.SpotPropagator(z=z, out_grid=fp.Grid.from_extent(20e-3, 4e-3, 40e-6))

    def landing(incident):
        panel = fp.ModulatorArray.tiled(nx=1, ny=1, pitch=1e-3, region_size=600e-6,
                                        device="ideal", default_wavelength=LAM)
        panel.route(0, [(2e-3, 0.0)], z=z, wavelength=LAM, grating="blazed",
                    incident=incident)
        s = [x for x in sp.spots(panel, fields) if x.order != (0, 0)]
        return max(s, key=lambda x: x.power).x

    naive = landing(None)
    fixed = landing((alpha, 0.0))
    assert abs(naive - 2e-3) > 0.8e-3      # off by roughly z * alpha
    assert abs(fixed - 2e-3) < 60e-6       # back on target
