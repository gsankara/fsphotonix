"""Detection screens: cameras, photodiodes, fibre coupling."""

import numpy as np
import pytest

from fsphotonix import engine as eng
from fsphotonix.detectors import CameraArray, MMFArray, PhotodiodeArray, PowerMeter, SMFArray
from fsphotonix.fields import Channel, FieldSet
from fsphotonix.grid import Grid
from fsphotonix.sources.profiles import GaussianBeam, PlaneWave
from fsphotonix.units import photon_energy
from fsphotonix.waveform import NRZ, Constant, Sinusoid

LAM = 1550e-9


def uniform_field(grid, power, waveform=None, coherence="a", wavelength=LAM):
    u = np.ones(grid.shape, dtype=complex)
    u = u * np.sqrt(power / (float(np.sum(np.abs(u) ** 2)) * grid.cell_area))
    return FieldSet(grid=grid, channels=[
        Channel(u=u, wavelength=wavelength, coherence=coherence,
                waveform=waveform or Constant())])


# --------------------------------------------------------------------------- #
# camera
# --------------------------------------------------------------------------- #
def test_camera_collects_the_incident_power():
    g = Grid.from_extent(400e-6, 400e-6, 2.5e-6)
    f = uniform_field(g, 1e-6)
    cam = CameraArray(nx=20, ny=20, pitch=20e-6, fill_factor=1.0, output="watts",
                      integration_time=1e-6, qe=1.0, read_noise_e=0.0,
                      dark_current_e_s=0.0, prnu=0.0, full_well_e=1e12)
    out = cam.detect(f, noiseless=True)
    assert out.values.shape == (20, 20)
    assert out.values.sum() == pytest.approx(1e-6, rel=0.02)


def test_camera_electron_count_matches_photon_budget():
    g = Grid.from_extent(80e-6, 80e-6, 2e-6)
    p = 1e-9
    f = uniform_field(g, p)
    tint = 1e-6
    cam = CameraArray(nx=4, ny=4, pitch=20e-6, fill_factor=1.0, qe=0.8,
                      integration_time=tint, output="electrons", read_noise_e=0.0,
                      dark_current_e_s=0.0, prnu=0.0, full_well_e=1e12)
    out = cam.detect(f, noiseless=True)
    want = p * tint / photon_energy(LAM) * 0.8
    assert out.values.sum() == pytest.approx(want, rel=0.02)


def test_camera_saturates_and_quantises():
    g = Grid.from_extent(80e-6, 80e-6, 2e-6)
    f = uniform_field(g, 1e-3)
    cam = CameraArray(nx=4, ny=4, pitch=20e-6, full_well_e=1000.0, adc_bits=8,
                      integration_time=1e-6, output="dn")
    out = cam.detect(f)
    assert out.saturated.all()
    assert out.values.max() <= 255


def test_camera_shot_noise_scales_as_sqrt_n():
    g = Grid.from_extent(80e-6, 80e-6, 2e-6)
    cam = CameraArray(nx=4, ny=4, pitch=20e-6, fill_factor=1.0, qe=1.0,
                      integration_time=1e-6, output="electrons", read_noise_e=0.0,
                      dark_current_e_s=0.0, prnu=0.0, full_well_e=1e12, seed=3)
    ratios = []
    for p in (1e-13, 1e-11):
        out = cam.detect(uniform_field(g, p))
        mean = out.values.mean()
        ratios.append(out.values.std() / np.sqrt(max(mean, 1e-30)))
    assert 0.3 < ratios[0] < 3.0 and 0.3 < ratios[1] < 3.0


def test_camera_grid_helper_matches_geometry():
    cam = CameraArray(nx=32, ny=16, pitch=10e-6)
    g = cam.grid(oversample=4)
    assert g.nx == 128 and g.ny == 64
    assert g.width == pytest.approx(32 * 10e-6)


# --------------------------------------------------------------------------- #
# photodiodes
# --------------------------------------------------------------------------- #
def test_photodiode_current_matches_responsivity():
    g = Grid.from_extent(400e-6, 400e-6, 2e-6)
    f = uniform_field(g, 1e-3)
    pd = PhotodiodeArray(positions_in=[(0.0, 0.0)], active_diameter=100e-6,
                         responsivity=0.9, output="amps", bandwidth=1e9, n_time=1,
                         dark_current=0.0)
    out = pd.detect(f, noiseless=True)
    # power on the active area = incident irradiance x pi r^2
    irr = 1e-3 / (400e-6) ** 2
    want = 0.9 * irr * np.pi * (50e-6) ** 2
    assert float(np.asarray(out.values).ravel()[0]) == pytest.approx(want, rel=0.05)


def test_photodiode_resolves_the_waveform():
    g = Grid.from_extent(200e-6, 200e-6, 4e-6)
    f = uniform_field(g, 1e-3, waveform=Sinusoid(frequency=1e9, depth=1.0))
    pd = PhotodiodeArray(positions_in=[(0.0, 0.0)], active_diameter=150e-6,
                         bandwidth=20e9, n_time=128, output="amps", dark_current=0.0)
    out = pd.detect(f, noiseless=True)
    v = np.asarray(out.values, dtype=float).ravel()
    assert v.std() / v.mean() > 0.3  # 1 GHz tone inside a 20 GHz detector


def test_photodiode_bandwidth_filters_fast_modulation():
    g = Grid.from_extent(200e-6, 200e-6, 4e-6)
    f = uniform_field(g, 1e-3, waveform=Sinusoid(frequency=20e9, depth=1.0))
    fast = PhotodiodeArray(positions_in=[(0.0, 0.0)], active_diameter=150e-6,
                           bandwidth=100e9, n_time=256, output="amps", dark_current=0.0)
    slow = PhotodiodeArray(positions_in=[(0.0, 0.0)], active_diameter=150e-6,
                           bandwidth=1e9, n_time=256, output="amps", dark_current=0.0)
    a = np.asarray(fast.detect(f, noiseless=True).values, dtype=float).ravel()
    b = np.asarray(slow.detect(f, noiseless=True).values, dtype=float).ravel()
    assert b.std() < 0.3 * a.std()


def test_apd_gain_raises_the_signal():
    g = Grid.from_extent(200e-6, 200e-6, 4e-6)
    f = uniform_field(g, 1e-7)
    pin = PhotodiodeArray(positions_in=[(0.0, 0.0)], apd_gain=1.0, output="amps",
                          n_time=1, dark_current=0.0)
    apd = PhotodiodeArray(positions_in=[(0.0, 0.0)], apd_gain=10.0, output="amps",
                          n_time=1, dark_current=0.0)
    a = float(np.asarray(pin.detect(f, noiseless=True).values).ravel()[0])
    b = float(np.asarray(apd.detect(f, noiseless=True).values).ravel()[0])
    assert b / a == pytest.approx(10.0, rel=1e-6)
    assert apd.nep() < pin.nep()


def test_power_meter_totals():
    g = Grid.from_extent(1e-3, 1e-3, 4e-6)
    f = uniform_field(g, 2e-3)
    pm = PowerMeter(positions_in=[(0.0, 0.0)], diameter=2e-3)
    assert float(pm.detect(f).values[0]) == pytest.approx(2e-3, rel=0.02)


# --------------------------------------------------------------------------- #
# fibre coupling
# --------------------------------------------------------------------------- #
def test_smf_couples_a_matched_mode_almost_perfectly():
    mfd = 40e-6
    g = Grid.from_extent(200e-6, 200e-6, 0.5e-6)
    u = GaussianBeam(waist=mfd / 2).field(g, LAM, power=1e-3)
    f = FieldSet(grid=g, channels=[Channel(u=u, wavelength=LAM, coherence="a")])
    smf = SMFArray(positions_in=[(0.0, 0.0)], mfd=mfd, insertion_loss_db=0.0,
                   mode_scaling=False)
    eta = smf.coupling_efficiency(f)[0]
    assert eta == pytest.approx(1.0, rel=0.02)


def test_smf_rejects_a_tilted_wavefront():
    mfd = 40e-6
    g = Grid.from_extent(200e-6, 200e-6, 0.25e-6)
    smf = SMFArray(positions_in=[(0.0, 0.0)], mfd=mfd, insertion_loss_db=0.0,
                   mode_scaling=False)
    etas = []
    for theta in (0.0, 0.05):
        u = GaussianBeam(waist=mfd / 2, theta_x=theta).field(g, LAM, power=1e-3)
        f = FieldSet(grid=g, channels=[Channel(u=u, wavelength=LAM, coherence="a")])
        etas.append(smf.coupling_efficiency(f)[0])
    assert etas[1] < 0.1 * etas[0]  # single-mode coupling is direction selective


def test_smf_rejects_a_displaced_beam():
    mfd = 40e-6
    g = Grid.from_extent(400e-6, 400e-6, 1e-6)
    smf = SMFArray(positions_in=[(0.0, 0.0)], mfd=mfd, insertion_loss_db=0.0,
                   mode_scaling=False)
    on = GaussianBeam(waist=mfd / 2).field(g, LAM, power=1e-3)
    off = GaussianBeam(waist=mfd / 2, x0=60e-6).field(g, LAM, power=1e-3)
    e_on = smf.coupling_efficiency(FieldSet(grid=g, channels=[
        Channel(u=on, wavelength=LAM, coherence="a")]))[0]
    e_off = smf.coupling_efficiency(FieldSet(grid=g, channels=[
        Channel(u=off, wavelength=LAM, coherence="a")]))[0]
    assert e_off < 0.05 * e_on


def test_mmf_accepts_tilts_inside_its_na():
    g = Grid.from_extent(400e-6, 400e-6, 1e-6)
    mmf = MMFArray(positions_in=[(0.0, 0.0)], core_diameter=100e-6,
                   numerical_aperture=0.2, insertion_loss_db=0.0)
    etas = []
    for theta in (0.0, 0.05, 0.4):  # 0.4 rad = sin 0.39 > NA
        u = GaussianBeam(waist=30e-6, theta_x=theta).field(g, LAM, power=1e-3)
        f = FieldSet(grid=g, channels=[Channel(u=u, wavelength=LAM, coherence="a")])
        etas.append(mmf.coupling_efficiency(f)[0])
    assert etas[1] > 0.5 * etas[0]  # tolerant inside the NA
    assert etas[2] < 0.1 * etas[0]  # hard cut-off outside it


def test_mmf_is_position_tolerant_relative_to_smf():
    g = Grid.from_extent(400e-6, 400e-6, 1e-6)
    off = 30e-6
    u = GaussianBeam(waist=20e-6, x0=off).field(g, LAM, power=1e-3)
    f = FieldSet(grid=g, channels=[Channel(u=u, wavelength=LAM, coherence="a")])
    mmf = MMFArray(positions_in=[(0.0, 0.0)], core_diameter=100e-6,
                   numerical_aperture=0.2, insertion_loss_db=0.0)
    smf = SMFArray(positions_in=[(0.0, 0.0)], mfd=40e-6, insertion_loss_db=0.0,
                   mode_scaling=False)
    assert mmf.coupling_efficiency(f)[0] > 5 * smf.coupling_efficiency(f)[0]


def test_fiber_array_layout_and_loss():
    smf = SMFArray(nx=4, ny=2, pitch=250e-6, insertion_loss_db=3.0)
    assert smf.n_elements == 8
    assert smf.loss == pytest.approx(0.5, rel=0.01)


def test_single_time_sample_is_honoured():
    """n_time=1 means 'do not resolve time', not 'apply a Nyquist floor'."""
    g = Grid.from_extent(200e-6, 200e-6, 4e-6)
    f = uniform_field(g, 1e-3, waveform=NRZ.prbs(63, baud=25e9))
    pd = PhotodiodeArray(positions_in=[(0.0, 0.0), (50e-6, 0.0)],
                         active_diameter=40e-6, n_time=1, output="watts",
                         dark_current=0.0)
    out = pd.detect(f, noiseless=True)
    assert np.asarray(out.values).shape == (2,)
    assert out.vector().size == 2


def test_long_exposure_does_not_blow_up_memory():
    """A 2 us exposure over a 25 Gbaud pattern must not build a 400k-sample cube."""
    from fsphotonix.detectors.base import time_samples

    g = Grid.from_extent(200e-6, 200e-6, 8e-6)
    f = uniform_field(g, 1e-3, waveform=NRZ.prbs(63, baud=25e9))
    ts = time_samples(f, window=2e-6, n=16)
    assert ts.size <= 64
    # one slowest-waveform period is enough to get the average right
    assert ts[-1] <= f.channels[0].waveform.period
    cam = CameraArray(nx=8, ny=8, pitch=25e-6, integration_time=2e-6, output="watts",
                      fill_factor=1.0, qe=1.0, read_noise_e=0.0, prnu=0.0,
                      dark_current_e_s=0.0, full_well_e=1e15)
    out = cam.detect(f, noiseless=True)
    assert out.values.sum() > 0
