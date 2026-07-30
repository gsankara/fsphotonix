"""Chirp-z / zoom-DFT correctness and grid bookkeeping."""

import numpy as np
import pytest

from fsphotonix import engine as eng
from fsphotonix.grid import Grid


def test_czt_matches_direct_sum():
    rng = np.random.default_rng(0)
    a = rng.normal(size=32) + 1j * rng.normal(size=32)
    m, w, a0 = 20, np.exp(-0.13j), np.exp(0.07j)
    n = np.arange(a.size)
    want = np.array([np.sum(a * a0**n * w ** (n * k)) for k in range(m)])
    got = eng.to_numpy(eng.czt(a, m, w=w, a0=a0))
    assert np.allclose(got, want, rtol=1e-9, atol=1e-9)


def test_czt_reduces_to_fft():
    rng = np.random.default_rng(1)
    a = rng.normal(size=64) + 1j * rng.normal(size=64)
    got = eng.to_numpy(eng.czt(a, 64, w=np.exp(-2j * np.pi / 64), a0=1.0))
    assert np.allclose(got, np.fft.fft(a), rtol=1e-9, atol=1e-8)


def test_dtft_axis_arbitrary_frequencies():
    rng = np.random.default_rng(2)
    a = rng.normal(size=48) + 1j * rng.normal(size=48)
    x0, dx = -3.0, 0.25
    x = x0 + np.arange(a.size) * dx
    f = np.linspace(-0.7, 0.4, 13)
    want = np.array([np.sum(a * np.exp(-2j * np.pi * fi * x)) for fi in f])
    got = eng.to_numpy(eng.dtft_axis(a, x0, dx, f))
    assert np.allclose(got, want, rtol=1e-8, atol=1e-8)


def test_zoom_dft2_matches_direct():
    rng = np.random.default_rng(3)
    u = rng.normal(size=(12, 10)) + 1j * rng.normal(size=(12, 10))
    x = (np.arange(10) - 5) * 0.3
    y = (np.arange(12) - 6) * 0.2
    fx = np.linspace(-0.4, 0.4, 7)
    fy = np.linspace(-0.2, 0.5, 5)
    want = np.zeros((5, 7), dtype=complex)
    for i, fyi in enumerate(fy):
        for j, fxi in enumerate(fx):
            ph = np.exp(-2j * np.pi * (fxi * x[None, :] + fyi * y[:, None]))
            want[i, j] = np.sum(u * ph)
    got = eng.to_numpy(eng.zoom_dft2(u, x, y, fx, fy))
    assert np.allclose(got, want, rtol=1e-8, atol=1e-8)


def test_grid_geometry():
    g = Grid.from_extent(2e-3, 1e-3, 5e-6)
    assert g.nx == 400 and g.ny == 200
    assert g.width == pytest.approx(2e-3)
    assert g.cell_area == pytest.approx(25e-12)
    r, c = g.index_of(0.0, 0.0)
    assert (r, c) == (g.ny // 2, g.nx // 2)
    x = eng.to_numpy(g.x())
    assert x[g.nx // 2] == pytest.approx(0.0)


def test_grid_offcenter_and_rescale():
    g = Grid.from_extent(1e-3, 1e-3, 1e-6, x0=3e-3, y0=-1e-3)
    x = eng.to_numpy(g.x())
    assert x[g.nx // 2] == pytest.approx(3e-3)
    g2 = g.rescaled(4.0)
    assert g2.dx == pytest.approx(4e-6) and g2.nx == g.nx
