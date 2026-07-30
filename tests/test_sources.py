"""Sources, waveforms and coherence bookkeeping."""

import numpy as np
import pytest

from fsphotonix import engine as eng
from fsphotonix.fields import Channel, FieldSet
from fsphotonix.grid import Grid
from fsphotonix.sources import (
    CoarseWDMBank,
    CoherentLaser,
    GaussianBeam,
    LaserArray,
    PlaneWave,
    SuperGaussian,
    TransceiverBank,
    WDMComb,
    grid_positions,
)
from fsphotonix.units import C0
from fsphotonix.waveform import NRZ, PAM, Constant, Mixed, PulseTrain, Sinusoid

LAM = 1550e-9
GRID = Grid.from_extent(4e-3, 4e-3, 8e-6)


# --------------------------------------------------------------------------- #
# waveforms
# --------------------------------------------------------------------------- #
def test_waveform_normalisation_is_mean_power_one():
    for w in (Sinusoid(frequency=1e9, depth=0.8),
              NRZ(bits=[1, 0, 1, 1, 0], baud=25e9),
              PAM(symbols=[0, 1, 2, 3], baud=25e9),
              PulseTrain(rep_rate=10e9, fwhm=5e-12)):
        assert w.mean_power() == pytest.approx(1.0, rel=0.05)


def test_nrz_extinction_ratio():
    w = NRZ(bits=[1, 0], baud=10e9, extinction_ratio_db=10.0, rise_fraction=0.0,
            normalize=False)
    t = np.array([0.5, 1.5]) / 10e9
    p = np.abs(eng.to_numpy(w.sample(t))) ** 2
    assert p[0] / p[1] == pytest.approx(10.0, rel=0.02)


def test_pulse_train_duty_cycle():
    w = PulseTrain(rep_rate=10e9, fwhm=10e-12, shape="square", normalize=False)
    t = np.linspace(0, 1 / 10e9, 1000, endpoint=False)
    duty = np.mean(np.abs(eng.to_numpy(w.sample(t))) ** 2)
    assert duty == pytest.approx(0.1, rel=0.05)


def test_mixed_waveform_carries_both():
    w = Mixed(digital=NRZ(bits=[1, 1, 1, 1], baud=10e9),
              analog=Sinusoid(frequency=2e9), analog_weight=0.5)
    t = np.linspace(0, w.period, 512, endpoint=False)
    v = np.abs(eng.to_numpy(w.sample(t)))
    assert v.std() > 0.05  # the analog subcarrier is visible


# --------------------------------------------------------------------------- #
# beam profiles
# --------------------------------------------------------------------------- #
def test_profiles_are_power_normalised():
    for p in (GaussianBeam(waist=400e-6), PlaneWave(diameter=2e-3),
              SuperGaussian(diameter=2e-3)):
        u = p.field(GRID, LAM, power=3e-3)
        pw = float(np.sum(np.abs(eng.to_numpy(u)) ** 2)) * GRID.cell_area
        assert pw == pytest.approx(3e-3, rel=0.01)


def test_pointing_creates_a_phase_ramp():
    theta = 5e-3
    u = eng.to_numpy(PlaneWave(theta_x=theta).field(GRID, LAM, power=1e-3))
    row = np.angle(u[GRID.ny // 2])
    grad = np.gradient(np.unwrap(row)) / GRID.dx
    assert np.mean(grad) == pytest.approx(2 * np.pi * np.sin(theta) / LAM, rel=1e-3)


def test_gaussian_from_divergence():
    g = GaussianBeam.from_divergence(10e-3, LAM)
    assert g.waist == pytest.approx(LAM / (np.pi * 10e-3))


# --------------------------------------------------------------------------- #
# sources
# --------------------------------------------------------------------------- #
def test_coherent_split_shares_one_coherence_group():
    s = CoherentLaser(power=8e-3, split=4, pitch=1e-3, wavelength=LAM)
    f = s.emit(GRID)
    assert len(f) == 4
    assert len(f.coherence_groups()) == 1
    assert f.total_power() == pytest.approx(8e-3, rel=0.02)


def test_laser_array_is_mutually_incoherent():
    s = LaserArray(n=4, pitch=1e-3, power=1e-3, linewidth=0.0, rin_dbc_hz=-300)
    f = s.emit(GRID)
    assert len(f.coherence_groups()) == 4


def test_drive_scales_power_linearly():
    s = LaserArray(n=3, pitch=1e-3, power=2e-3, linewidth=0.0, rin_dbc_hz=-300)
    s.set_drive([1.0, 0.5, 0.0])
    f = s.emit(GRID)
    assert len(f) == 2  # the zero-drive emitter is dropped
    assert f.total_power() == pytest.approx(2e-3 * 1.5, rel=0.02)


def test_cots_bank_spreads_inside_one_100ghz_slot():
    s = TransceiverBank(n=8, center_wavelength=1310e-9, freq_span=100e9, baud=25e9)
    offs = s.frequency_offsets()
    assert offs.min() == pytest.approx(-50e9, rel=1e-6)
    assert offs.max() == pytest.approx(+50e9, rel=1e-6)
    f = s.emit(GRID)
    assert len(f.coherence_groups()) == 8  # separate packages never interfere
    lams = np.array([c.wavelength for c in f.channels])
    assert np.ptp(lams) / 1310e-9 < 1e-3  # still "the same" wavelength


def test_wdm_comb_on_itu_grid():
    s = WDMComb(n_lines=4, spacing=100e9, mutually_coherent=True)
    f = s.emit(GRID)
    assert len(f.coherence_groups()) == 1
    freqs = np.array([C0 / c.wavelength for c in f.channels])
    assert np.allclose(np.diff(freqs), 100e9, rtol=1e-9)


def test_cwdm_bank_uses_distinct_wavelengths():
    s = CoarseWDMBank(n=4, first_cwdm=0)
    lams = np.array([e.wavelength for e in s.emitters])
    assert np.allclose(np.diff(lams) * 1e9, 20.0, atol=0.01)


# --------------------------------------------------------------------------- #
# coherence arithmetic
# --------------------------------------------------------------------------- #
def test_same_group_interferes_opposite_group_does_not():
    g = Grid.from_extent(100e-6, 100e-6, 5e-6)
    u = np.ones(g.shape, dtype=complex) * 1e-2

    coh = FieldSet(grid=g, channels=[
        Channel(u=u, wavelength=LAM, coherence="a", waveform=Constant()),
        Channel(u=u, wavelength=LAM, coherence="a", waveform=Constant()),
    ])
    inc = FieldSet(grid=g, channels=[
        Channel(u=u, wavelength=LAM, coherence="a", waveform=Constant()),
        Channel(u=u, wavelength=LAM, coherence="b", waveform=Constant()),
    ])
    p_one = float(np.sum(np.abs(u) ** 2)) * g.cell_area
    # constructive interference gives 4x; incoherent addition gives 2x
    assert coh.total_power() == pytest.approx(2 * p_one, rel=1e-9)  # per-channel sum
    assert float(eng.to_numpy(coh.intensity()).sum()) == pytest.approx(
        4 * float(np.sum(np.abs(u) ** 2)), rel=1e-9)
    assert float(eng.to_numpy(inc.intensity()).sum()) == pytest.approx(
        2 * float(np.sum(np.abs(u) ** 2)), rel=1e-9)


def test_unresolvable_beat_is_dropped():
    """Two wavelengths tagged coherent but 100 GHz apart: a slow detector sees only
    the sum of intensities."""
    g = Grid.from_extent(50e-6, 50e-6, 5e-6)
    u = np.ones(g.shape, dtype=complex)
    f = FieldSet(grid=g, channels=[
        Channel(u=u, wavelength=1550e-9, coherence="a"),
        Channel(u=u, wavelength=C0 / (C0 / 1550e-9 + 100e9), coherence="a"),
    ])
    slow = float(eng.to_numpy(f.intensity(0.0, beat_bandwidth=1e9)).sum())
    fast = float(eng.to_numpy(f.intensity(0.0, beat_bandwidth=1e12)).sum())
    assert slow == pytest.approx(2 * u.size, rel=1e-9)
    assert fast == pytest.approx(4 * u.size, rel=1e-9)


def test_field_centroid_tracks_offset_beam():
    s = CoherentLaser(profile=GaussianBeam(waist=300e-6, x0=1e-3, y0=-0.5e-3))
    f = s.emit(GRID)
    cx, cy = f.centroid()
    assert cx == pytest.approx(1e-3, rel=0.02)
    assert cy == pytest.approx(-0.5e-3, rel=0.02)
