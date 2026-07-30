"""Propagation kernels against analytic diffraction results."""

import numpy as np
import pytest

from fsphotonix import engine as eng
from fsphotonix.fields import Channel, FieldSet
from fsphotonix.grid import Grid
from fsphotonix.propagation import (
    AngularSpectrum,
    Fraunhofer,
    FresnelTransfer,
    FresnelZoom,
    ThinLens,
)
from fsphotonix.sources.profiles import GaussianBeam, Rectangular

LAM = 1550e-9


def gaussian_field(grid, w0, power=1.0):
    u = GaussianBeam(waist=w0).field(grid, LAM, power=power)
    return FieldSet(grid=grid, channels=[Channel(u=u, wavelength=LAM, coherence="c")])


def measure_waist(fields):
    """1/e^2 intensity radius from the second moment of the irradiance."""
    I = eng.to_numpy(fields.mean_intensity(1))
    X, Y = (eng.to_numpy(a) for a in fields.grid.meshgrid())
    s = I.sum()
    cx = (I * X).sum() / s
    var = (I * (X - cx) ** 2).sum() / s
    return 2.0 * np.sqrt(var)  # w = 2*sigma for a Gaussian intensity


@pytest.mark.parametrize("kernel", [AngularSpectrum, FresnelTransfer])
def test_gaussian_diffraction_same_grid(kernel):
    w0 = 200e-6
    zr = np.pi * w0**2 / LAM
    g = Grid.from_extent(4e-3, 4e-3, 4e-6)
    f0 = gaussian_field(g, w0, power=2e-3)
    z = 0.6 * zr
    out = kernel(z=z).propagate(f0)
    want = w0 * np.sqrt(1 + (z / zr) ** 2)
    assert measure_waist(out) == pytest.approx(want, rel=0.03)
    assert out.total_power() == pytest.approx(f0.total_power(), rel=0.01)


def test_fresnel_zoom_gaussian_far_field():
    """The zoom kernel handles a big output grid at long range."""
    w0 = 100e-6
    zr = np.pi * w0**2 / LAM
    g = Grid.from_extent(2e-3, 2e-3, 2e-6)
    f0 = gaussian_field(g, w0, power=1e-3)
    z = 40 * zr
    want = w0 * np.sqrt(1 + (z / zr) ** 2)
    screen = Grid.from_extent(8 * want, 8 * want, want / 12)
    out = FresnelZoom(z=z, out_grid=screen).propagate(f0)
    assert measure_waist(out) == pytest.approx(want, rel=0.05)
    assert out.total_power() == pytest.approx(f0.total_power(), rel=0.05)


def test_fraunhofer_slit_first_null():
    d = 200e-6
    z = 0.3
    g = Grid.from_extent(1e-3, 1e-3, 1e-6)
    u = Rectangular(width=d, height=d).field(g, LAM, power=1e-3)
    f0 = FieldSet(grid=g, channels=[Channel(u=u, wavelength=LAM, coherence="c")])
    expect_null = LAM * z / d
    screen = Grid.from_extent(12 * expect_null, 12 * expect_null, expect_null / 40)
    out = Fraunhofer(z=z, out_grid=screen).propagate(f0)
    I = eng.to_numpy(out.mean_intensity(1))
    row = I[out.grid.ny // 2]
    x = eng.to_numpy(out.grid.x())
    # first local minimum to the right of the peak = the first diffraction null
    peak = int(np.argmax(row))
    null = peak
    for i in range(peak + 1, len(row) - 1):
        if row[i] <= row[i + 1]:
            null = i
            break
    assert abs(x[null]) == pytest.approx(expect_null, rel=0.08)
    assert out.total_power() == pytest.approx(f0.total_power(), rel=0.05)


def test_lens_focuses_to_diffraction_limit():
    d = 1e-3
    f = 0.05
    g = Grid.from_extent(1.6e-3, 1.6e-3, 1e-6)
    u = Rectangular(width=d, height=d).field(g, LAM, power=1e-3)
    f0 = FieldSet(grid=g, channels=[Channel(u=u, wavelength=LAM, coherence="c")])
    lensed = ThinLens(focal_length=f).propagate(f0, z=0.0)
    spot = LAM * f / d
    screen = Grid.from_extent(16 * spot, 16 * spot, spot / 20)
    out = FresnelZoom(z=f, out_grid=screen).propagate(lensed)
    I = eng.to_numpy(out.mean_intensity(1))
    row = I[out.grid.ny // 2]
    x = eng.to_numpy(out.grid.x())
    half = row.max() / 2
    above = x[row >= half]
    fwhm = above.max() - above.min()
    assert fwhm == pytest.approx(0.886 * spot, rel=0.15)


def test_backward_propagation_recovers_field():
    g = Grid.from_extent(2e-3, 2e-3, 4e-6)
    f0 = gaussian_field(g, 150e-6, power=1e-3)
    z = 0.02
    fwd = AngularSpectrum(z=z).propagate(f0)
    back = AngularSpectrum(z=-z).propagate(fwd)
    a = eng.to_numpy(f0.channels[0].u)
    b = eng.to_numpy(back.channels[0].u)
    err = np.linalg.norm(b - a) / np.linalg.norm(a)
    assert err < 0.02


def test_zero_distance_is_identity():
    g = Grid.from_extent(1e-3, 1e-3, 4e-6)
    f0 = gaussian_field(g, 100e-6)
    out = FresnelZoom(z=0.0).propagate(f0)
    assert np.allclose(eng.to_numpy(out.channels[0].u), eng.to_numpy(f0.channels[0].u))
