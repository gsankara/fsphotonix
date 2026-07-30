"""Declarative configs and the plugin registry."""

import numpy as np
import pytest

import fsphotonix as fp
from fsphotonix import config
from fsphotonix.registry import PluginError, available, create, describe


def test_registry_inventory_covers_every_slot():
    inv = available()
    for kind in ("source", "waveform", "profile", "geometry", "grating", "device",
                 "propagator", "detector", "controller"):
        assert inv[kind], f"no plugins registered for {kind}"
    assert "blazed" in inv["grating"]
    assert "ipms_piston" in inv["device"]
    assert "cots_transceiver" in inv["source"]
    assert isinstance(describe(), str)


def test_unknown_plugin_lists_the_alternatives():
    with pytest.raises(PluginError) as e:
        create("grating", "not_a_grating")
    assert "available" in str(e.value)


def test_aliases_resolve():
    assert type(create("grating", "blaze")) is type(create("grating", "blazed"))
    assert type(create("detector", "ccd")) is type(create("detector", "camera"))


def test_yaml_numeric_strings_are_coerced():
    """YAML 1.1 parses 100.0e9 as a string; the loader must not care."""
    cfg = config.load_text("a: 100.0e9\nb: 1.0e-6\nc: [2.5e9, 3]\nd: hello\n")
    assert cfg["a"] == pytest.approx(1e11)
    assert cfg["b"] == pytest.approx(1e-6)
    assert cfg["c"][0] == pytest.approx(2.5e9)
    assert cfg["c"][1] == 3
    assert cfg["d"] == "hello"


CFG = """
name: test_system
wavelength: 1550.0e-9
grids:
  panel:  {width: 3.0e-3, height: 1.0e-3, pitch: 50.0e-6}
  screen: {width: 12.0e-3, height: 2.0e-3, pitch: 100.0e-6}
stages:
  - name: src
    type: source
    grid: panel
    source: {type: laser_array, n: 3, pitch: 800.0e-6, power: 1.0e-3,
             linewidth: 0.0, rin_dbc_hz: -300.0}
  - name: panel
    type: modulator
    grid: panel
    z: 0.0
    spot_mode: true
    device: {type: ipms_piston}
    layout: {nx: 3, ny: 1, pitch: 800.0e-6, gap: 100.0e-6}
    route:
      z: 0.06
      targets: [[-3.0e-3, 0.0], [3.0e-3, 0.0]]
      tiling: stripes
      gap: 20.0e-6
  - name: screen
    type: detector
    grid: screen
    z: 0.06
    detector: {type: photodiode, positions_in: [[-3.0e-3, 0.0], [3.0e-3, 0.0]],
               active_diameter: 1.0e-3, output: watts, n_time: 1}
loops:
  - name: servo
    sense: screen
    actuate: src
    mode: drive
    period: 2
    controller:
      - {type: normalize, mode: max}
      - {type: gain, gain: 0.9}
"""


def test_build_system_from_config():
    cfg = config.load_text(CFG)
    system = config.build_system(cfg)
    assert system.stage_names() == ["src", "panel", "screen"]
    assert len(system["panel"].modulator.regions) == 3
    assert len(system["panel"].modulator[0].subregions) == 2
    assert system["panel"].modulator.region_gap[0] == pytest.approx(100e-6)
    trace = system.run()
    v = trace.vector("screen")
    assert v.size == 2 and np.all(v > 0)


def test_build_machine_from_config():
    cfg = config.load_text(CFG)
    m = config.build_machine(cfg)
    assert len(m.loops) == 1
    assert m.loops[0].period == 2
    m.run(6)
    rep = m.loop_report()
    assert rep[0]["updates"] == 3  # iterations 0, 2, 4
    assert "test_system" in config.dump_summary(config.build_system(cfg))


def test_shipped_configs_build():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "configs"
    for path in sorted(root.glob("*.yaml")):
        cfg = config.load_file(path)
        system = config.build_system(cfg)
        assert len(system) >= 3, path.name
        assert system.sampling_report(), path.name
