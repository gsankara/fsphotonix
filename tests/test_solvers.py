"""Optical matrix-vector multiply and the analog iterative machine."""

import numpy as np
import pytest

from fsphotonix.solvers.aim import (
    AnalogIterativeMachine,
    OpticalMVM,
    brute_force_maxcut,
    cut_value,
    maxcut_ising,
    qubo_to_ising,
    random_graph,
)


# --------------------------------------------------------------------------- #
# problem encodings
# --------------------------------------------------------------------------- #
def test_maxcut_ising_energy_tracks_the_cut():
    A = random_graph(8, density=0.6, seed=1)
    J, h = maxcut_ising(A)
    rng = np.random.default_rng(0)
    for _ in range(20):
        s = rng.choice([-1.0, 1.0], 8)
        e = -0.5 * s @ J @ s - h @ s
        assert cut_value(A, s) == pytest.approx((A.sum() / 2 - e) / 2, rel=1e-9)


def test_qubo_to_ising_preserves_the_objective():
    rng = np.random.default_rng(2)
    n = 6
    Q = rng.normal(0, 1, (n, n))
    Q = 0.5 * (Q + Q.T)
    J, h, off = qubo_to_ising(Q)
    for _ in range(20):
        x = rng.integers(0, 2, n).astype(float)
        s = 2 * x - 1
        want = x @ Q @ x
        got = -0.5 * s @ J @ s - h @ s + off
        assert got == pytest.approx(want, abs=1e-9)


def test_brute_force_matches_a_known_graph():
    # a 4-cycle: the optimal cut splits alternating vertices, value 4
    A = np.zeros((4, 4))
    for i in range(4):
        A[i, (i + 1) % 4] = A[(i + 1) % 4, i] = 1.0
    best, s = brute_force_maxcut(A)
    assert best == pytest.approx(4.0)


# --------------------------------------------------------------------------- #
# the optical multiplier
# --------------------------------------------------------------------------- #
def test_mvm_geometry_is_feasible():
    mvm = OpticalMVM(n_in=4, n_out=4, z=0.06, mode="spot")
    chk = mvm.check()
    assert chk["deflection_ok"]
    assert chk["rails"] == {"in": 8, "out": 8}
    assert chk["pixels_per_subaperture"] > 4


def test_mvm_reproduces_a_signed_matrix():
    rng = np.random.default_rng(0)
    n = 4
    mvm = OpticalMVM(n_in=n, n_out=n, z=0.06, out_pitch=1.5e-3, region_pitch=1.2e-3,
                     mode="spot")
    W = rng.normal(0, 1, (n, n))
    W /= np.abs(W).max()
    mvm.set_matrix(W)
    cal = mvm.calibrate()
    assert cal["relative_rms"] < 0.05
    x = rng.normal(0, 1, n)
    y = mvm.multiply(x)
    assert np.corrcoef(y, W @ x)[0, 1] > 0.995


def test_mvm_is_linear_and_signed():
    n = 3
    mvm = OpticalMVM(n_in=n, n_out=n, z=0.06, out_pitch=1.5e-3, region_pitch=1.2e-3,
                     mode="spot")
    W = np.array([[1.0, -1.0, 0.5], [-0.5, 1.0, 0.0], [0.25, 0.25, -1.0]])
    mvm.set_matrix(W)
    mvm.calibrate()
    a, b = np.array([1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.5])
    ya, yb = mvm.multiply(a), mvm.multiply(b)
    yab = mvm.multiply(a + b)
    assert np.allclose(yab, ya + yb, atol=0.05)
    # a sign flip in the input flips the output
    assert np.allclose(mvm.multiply(-a), -ya, atol=0.05)


def test_mvm_transfer_matrix_matches_the_request():
    n = 4
    mvm = OpticalMVM(n_in=n, n_out=n, z=0.06, out_pitch=1.5e-3, region_pitch=1.2e-3,
                     mode="spot")
    W = np.eye(n) - 0.5 * np.roll(np.eye(n), 1, axis=0)
    mvm.set_matrix(W)
    mvm.calibrate()
    T = mvm.transfer_matrix() / mvm._row_gain[:, None]
    assert np.abs(T - W).max() < 0.05


# --------------------------------------------------------------------------- #
# the iterative machine
# --------------------------------------------------------------------------- #
def test_aim_numeric_control_solves_small_maxcut():
    A = random_graph(10, density=0.6, seed=3)
    J, h = maxcut_ising(A)
    aim = AnalogIterativeMachine(J, h, optics=None, step=0.08, noise=0.35, seed=1)
    res = aim.solve(600)
    best, _ = brute_force_maxcut(A)
    assert cut_value(A, res.best_spins) >= 0.9 * best
    assert res.best_energy <= res.energies[0]


def test_aim_energy_decreases():
    A = random_graph(12, density=0.5, seed=5)
    J, h = maxcut_ising(A)
    aim = AnalogIterativeMachine(J, h, optics=None, step=0.06, noise=0.3, seed=2)
    res = aim.solve(400)
    early = res.energies[:50].mean()
    late = res.energies[-50:].mean()
    assert late < early


def test_aim_through_the_optics_finds_a_good_cut():
    """The full loop: light really propagates every iteration."""
    n = 6
    A = random_graph(n, density=0.7, seed=7)
    J, h = maxcut_ising(A)
    optics = OpticalMVM(n_in=n, n_out=n, z=0.06, out_pitch=1.5e-3,
                        region_pitch=1.2e-3, mode="spot")
    aim = AnalogIterativeMachine(J, h, optics=optics, step=0.1, noise=0.4, seed=0)
    res = aim.solve(150)
    best, _ = brute_force_maxcut(A)
    assert cut_value(A, res.best_spins) >= 0.85 * best


def test_optical_and_numeric_gradients_agree():
    n = 5
    A = random_graph(n, density=0.8, seed=11)
    J, h = maxcut_ising(A)
    optics = OpticalMVM(n_in=n, n_out=n, z=0.06, out_pitch=1.5e-3,
                        region_pitch=1.2e-3, mode="spot")
    aim = AnalogIterativeMachine(J, h, optics=optics, seed=0)
    rng = np.random.default_rng(4)
    s = rng.uniform(-1, 1, n)
    g_opt = aim.gradient(s)
    g_num = J @ s + h
    assert np.corrcoef(g_opt, g_num)[0, 1] > 0.97


def test_aim_records_states_when_asked():
    A = random_graph(6, density=0.6, seed=9)
    J, h = maxcut_ising(A)
    aim = AnalogIterativeMachine(J, h, optics=None, seed=0)
    res = aim.solve(30, record_states=True)
    assert res.states.shape == (30, 6)
    assert res.summary()["iterations"] == 30
