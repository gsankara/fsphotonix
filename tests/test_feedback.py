"""Electronics, actuators and feedback loops."""

import numpy as np
import pytest

import fsphotonix as fp
from fsphotonix.system.logic import (
    Boolean,
    Gain,
    Integrator,
    LUT,
    MatrixMultiply,
    NoiseInjector,
    Nonlinearity,
    Normalize,
    Pipeline,
    Quantizer,
    Register,
    Threshold,
    analog_path,
    digital_path,
)

LAM = 1550e-9


# --------------------------------------------------------------------------- #
# logic blocks
# --------------------------------------------------------------------------- #
def test_threshold_hysteresis_latches():
    t = Threshold(level=0.5, hysteresis=0.2)
    assert t(np.array([0.55]))[0] == 0.0   # below the upper trip point 0.6
    assert t(np.array([0.7]))[0] == 1.0    # crosses up
    assert t(np.array([0.45]))[0] == 1.0   # stays latched above 0.4
    assert t(np.array([0.3]))[0] == 0.0    # falls out


def test_quantiser_levels():
    q = Quantizer(bits=2, vmin=0.0, vmax=1.0)
    out = q(np.linspace(0, 1, 9))
    assert len(np.unique(out)) <= 4


def test_register_delays_by_its_depth():
    r = Register(depth=2, initial=0.0)
    seq = [np.array([1.0]), np.array([2.0]), np.array([3.0]), np.array([4.0])]
    got = [float(r(x)[0]) for x in seq]
    assert got == [0.0, 0.0, 1.0, 2.0]


def test_pipeline_latency_is_the_sum():
    p = Pipeline(blocks=[Gain(2.0), Register(depth=3), Quantizer(bits=8)])
    assert p.latency == 4


def test_digital_path_has_more_latency_than_analog():
    assert analog_path().latency == 0
    assert digital_path(stages=3).latency >= 3


def test_normalise_modes():
    v = np.array([1.0, 2.0, 5.0])
    assert Normalize(mode="max")(v).max() == pytest.approx(1.0)
    assert Normalize(mode="sum")(v).sum() == pytest.approx(1.0)
    assert np.linalg.norm(Normalize(mode="l2")(v)) == pytest.approx(1.0)


def test_boolean_gates_on_index_pairs():
    b = Boolean(op="xor", pairs=[[0, 1], [1, 2]])
    assert np.allclose(b(np.array([1.0, 0.0, 1.0])), [1.0, 1.0])


def test_lut_and_matrix_blocks():
    assert np.allclose(LUT(table=[0, 10, 20, 30])(np.array([0.0, 1.0])), [0.0, 30.0])
    W = np.array([[1.0, 1.0], [1.0, -1.0]])
    assert np.allclose(MatrixMultiply(matrix=W)(np.array([3.0, 1.0])), [4.0, 2.0])


def test_integrator_accumulates_and_clips():
    it = Integrator(rate=0.5, clip=(-1, 1))
    for _ in range(10):
        out = it(np.array([1.0]))
    assert out[0] == pytest.approx(1.0)


def test_noise_injector_anneals():
    n = NoiseInjector(sigma=1.0, decay=0.5, seed=0)
    for _ in range(4):
        n(np.zeros(3))
    assert n.temperature == pytest.approx(1.0 * 0.5**4)


def test_nonlinearities():
    v = np.array([-2.0, 0.0, 2.0])
    assert np.allclose(Nonlinearity(kind="relu")(v), [0, 0, 2])
    assert np.allclose(Nonlinearity(kind="sign")(v), [-1, 0, 1])
    assert Nonlinearity(kind="tanh")(v)[2] == pytest.approx(np.tanh(2.0))


# --------------------------------------------------------------------------- #
# actuators and loops
# --------------------------------------------------------------------------- #
def build_machine_parts(n=4, z=0.06):
    pitch = 600e-6
    panel = fp.ModulatorArray.tiled(nx=n, ny=1, pitch=pitch, gap=80e-6,
                                    device="ipms_piston", default_wavelength=LAM)
    outs = np.stack([np.linspace(-3e-3, 3e-3, n), np.zeros(n)], axis=1)
    panel.set_matrix(np.eye(n), outs, z=z)
    src = fp.LaserArray(n=n, pitch=pitch, power=1e-3, linewidth=0.0, rin_dbc_hz=-300,
                        profile=fp.SuperGaussian(diameter=500e-6, order=10))
    grid = fp.Grid.from_extent(n * pitch * 1.2, 800e-6, 60e-6)
    det = fp.PhotodiodeArray(positions_in=outs, active_diameter=800e-6,
                             output="watts", n_time=1, dark_current=0.0,
                             tia_input_noise=0.0)
    screen = fp.Grid.from_extent(9e-3, 2e-3, 100e-6)
    system = fp.OpticalSystem([
        fp.SourceStage(source=src, grid=grid, z=0.0, name="src"),
        fp.ModulatorStage(modulator=panel, grid=grid, z=0.0, name="panel",
                          spot_mode=True),
        fp.DetectorStage(detector=det, grid=screen, z=z, name="screen"),
    ])
    return system, panel, src


def test_actuators_read_back_what_they_write():
    system, panel, src = build_machine_parts()
    pis = fp.make_actuator(system, "panel", "piston")
    pis.write(np.array([0.1, 0.2, 0.3, 0.4]))
    assert np.allclose(pis.read(), [0.1, 0.2, 0.3, 0.4], atol=1e-9)
    dr = fp.make_actuator(system, "src", "drive")
    dr.write(np.array([1.0, 0.5, 0.25, 0.0]))
    assert np.allclose(dr.read(), [1.0, 0.5, 0.25, 0.0])
    g = fp.make_actuator(system, "panel", "gain")
    g.write(np.array([0.5, 0.5, 0.5, 0.5]))
    assert np.allclose(g.read(), 0.5)


def test_drive_actuator_changes_the_readout():
    system, panel, src = build_machine_parts()
    base = system.run(keep_fields=False).vector("screen")
    src.set_drive([1.0, 0.0, 1.0, 0.0])
    off = system.run(keep_fields=False).vector("screen")
    assert off[1] < 1e-3 * base[1]
    assert off[0] == pytest.approx(base[0], rel=1e-6)


def test_power_servo_converges_to_its_setpoint():
    """A global loop that drives all outputs to a common target power."""
    system, panel, src = build_machine_parts()
    target = 0.4 * system.run(keep_fields=False).vector("screen").max()
    state = {"u": np.ones(4)}

    def controller(y):
        err = target - np.asarray(y, dtype=float).ravel()
        state["u"] = np.clip(state["u"] + 0.5 * err / max(target, 1e-12), 0.0, 1.0)
        return state["u"]

    loop = fp.Loop(sense="screen", actuator=fp.make_actuator(system, "src", "drive"),
                   controller=controller, name="servo", scope="global")
    m = fp.Machine(system, loops=[loop],
                   metric=lambda tr, k: {"err": float(np.max(np.abs(
                       tr.outputs["screen"].vector() - target)) / target)})
    m.run(40)
    assert m.history.series("err")[-1] < 0.05
    assert m.history.series("err")[-1] < m.history.series("err")[0]


def test_loop_period_and_delay_are_respected():
    system, panel, src = build_machine_parts()
    calls = []

    def spy(y):
        calls.append(len(calls))
        return np.ones(4)

    loop = fp.Loop(sense="screen", actuator=fp.make_actuator(system, "src", "drive"),
                   controller=spy, period=3, delay=2, name="slow")
    m = fp.Machine(system, loops=[loop])
    m.run(12)
    # due on iterations 0,3,6,9 -> 4 senses, the first 2 absorbed by the delay line
    assert len(calls) == 2
    assert loop.history[-1]["iteration"] == 9


def test_nested_local_and_global_loops_coexist():
    system, panel, src = build_machine_parts()
    fast = fp.Loop(sense="screen",
                   actuator=fp.make_actuator(system, "panel", "piston"),
                   controller=lambda y: np.zeros(4), period=1, priority=0,
                   name="fast", scope="local")
    slow = fp.Loop(sense="screen",
                   actuator=fp.make_actuator(system, "src", "drive"),
                   controller=lambda y: np.full(4, 0.5), period=5, priority=1,
                   name="slow", scope="global")
    m = fp.Machine(system, loops=[fast, slow])
    m.run(11)
    assert len(fast.history) == 11
    assert len(slow.history) == 3  # iterations 0, 5, 10
    rep = m.loop_report()
    assert {r["scope"] for r in rep} == {"local", "global"}
    assert np.allclose(src.get_drive(), 0.5)


def test_rate_limit_bounds_the_step():
    system, panel, src = build_machine_parts()
    loop = fp.Loop(sense="screen", actuator=fp.make_actuator(system, "src", "drive"),
                   controller=lambda y: np.ones(4), rate_limit=0.1, name="slew")
    m = fp.Machine(system, loops=[loop])
    m.step()
    first = src.get_drive().copy()
    m.step()
    assert np.all(np.abs(src.get_drive() - first) <= 0.1 + 1e-12)


def test_digital_pipeline_in_the_loop_adds_latency_not_error():
    system, panel, src = build_machine_parts()
    pipe = fp.Pipeline(blocks=[fp.Normalize(mode="max"), fp.Register(depth=2)])
    loop = fp.Loop(sense="screen", actuator=fp.make_actuator(system, "src", "drive"),
                   controller=pipe, name="digital")
    m = fp.Machine(system, loops=[loop])
    m.run(6)
    assert loop.controller.latency == 2
    assert np.all(src.get_drive() >= 0)
