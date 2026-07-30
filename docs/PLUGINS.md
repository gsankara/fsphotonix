# Writing plugins

Every replaceable technology is registered under a `kind` (the abstraction slot) and
a `name`. Adding one requires no change to the simulator: subclass the base, decorate
it, and the new device is reachable from `create()`, from YAML configs, and from every
builder that accepts a plugin name.

```python
from fsphotonix.registry import register, create, available, describe

print(describe())            # inventory of every slot
print(available("grating"))  # names in one slot
g = create("grating", "blazed", period=20e-6)
```

| kind | base class | must implement |
|---|---|---|
| `source` | `sources.base.LightSource` (usually `EmitterBank`) | `channels(grid)` |
| `waveform` | `waveform.Waveform` | `envelope(t)`, `period` |
| `profile` | `sources.profiles.BeamProfile` | `amplitude(grid, λ)` |
| `geometry` | `modulators.geometry.Aperture` | `signed_inside(xl, yl)`, `bounds()` |
| `grating` | `modulators.gratings.base.Grating` | `phase` and/or `amplitude`; `unit_cell()` or `orders()` |
| `device` | `modulators.devices.ModulatorDevice` | usually just field overrides |
| `propagator` | `propagation.base.Propagator` | `apply(u, in_grid, out_grid, λ, z)` |
| `detector` | `detectors.base.Detector` | `detect(fields, t)`, `positions` |
| `controller` | `system.logic.LogicBlock` | `__call__(x)` |

Custom kinds are allowed — `register("my_kind", "thing")` works.

## A grating

Supply the design-wavelength phase and either a lattice (so the base class can
compute orders numerically) or an explicit order list.

```python
import numpy as np
from fsphotonix import engine as eng
from fsphotonix.modulators.gratings.base import Grating, Order
from fsphotonix.registry import register

@register("grating", "chirped_blaze")
class ChirpedBlaze(Grating):
    """A blazed grating whose period varies across the aperture."""

    def __init__(self, period=20e-6, chirp=0.1, design_wavelength=1550e-9):
        self.period, self.chirp = period, chirp
        self.design_wavelength = design_wavelength
        self.achromatic = False        # phase scales as lambda0/lambda
        self.efficiency_scale = 1.0

    def phase(self, xl, yl):
        local = self.period * (1 + self.chirp * xl / self.period)
        return 2 * np.pi * eng.xp.mod(xl / local, 1.0)

    def unit_cell(self):
        return None                    # aperiodic -> override orders()

    def orders(self, wavelength=None, **kw):
        lam = wavelength or self.design_wavelength
        return [Order(1, 0, 1.0 / self.period, 0.0, 0.9)]
```

Three things to get right:

* **`achromatic`** — `False` for an optical-path-difference element (the phase scales
  as `λ₀/λ`), `True` for a geometric-phase element such as a metasurface.
* **`unit_cell()`** — return `((ax, ay), None)` for a 1-D grating, `((a1),(a2))` for a
  2-D lattice, or `None` if aperiodic. When it returns a lattice you get correct
  orders for free, including all the higher ones.
* **`orders()`** — needed for the beamlet propagator and for every design report.
  The base class memoises it, so it may be as expensive as it needs to be.

## A detector

```python
import numpy as np
from fsphotonix.detectors.base import Detector, DetectorOutput
from fsphotonix.registry import register

@register("detector", "lock_in")
class LockInArray(Detector):
    """Homodyne readout at a reference frequency."""

    name = "lock_in"

    def __init__(self, positions, diameter=100e-6, f_ref=1e9, n_time=256):
        self.positions_in = np.asarray(positions, float).reshape(-1, 2)
        self.diameter, self.f_ref, self.n_time = diameter, f_ref, n_time

    @property
    def positions(self):
        return self.positions_in

    def detect(self, fields, t=None):
        from fsphotonix import engine as eng
        from fsphotonix.detectors.photodiode import _element_masks

        ts = np.linspace(0, 8 / self.f_ref, self.n_time, endpoint=False)
        masks = _element_masks(fields.grid, self.positions, self.diameter)
        I = eng.to_numpy(fields.intensity(ts)).reshape(len(ts), -1)
        p = I @ masks.T * fields.grid.cell_area                    # (nt, N)
        ref = np.exp(-2j * np.pi * self.f_ref * ts)[:, None]
        amp = np.abs((p * ref).mean(axis=0)) * 2
        return DetectorOutput(values=amp, unit="W", optical_power=p.mean(axis=0),
                              positions=self.positions, shape=(len(self.positions),),
                              times=ts)
```

Note `fields.intensity(ts)` — the coherence bookkeeping (which channels interfere,
which only add in intensity) is already handled there; do not re-derive it.

## A modulator device

Devices are dataclasses of field values plus optional pipeline overrides. Most
technologies need no code at all:

```python
from dataclasses import dataclass
from fsphotonix.modulators.devices import ModulatorDevice
from fsphotonix.registry import register

@register("device", "fast_lcos")
@dataclass
class FastLCoS(ModulatorDevice):
    name: str = "fast_lcos"
    pitch: float = 4.5e-6
    reflective: bool = True
    fill_factor: float = 0.96
    efficiency: float = 0.82
    modulation: str = "phase"
    phase_bits: int | None = 10
    max_phase: float = 2 * 3.141592653589793
    design_wavelength: float = 1550e-9
    crosstalk: float = 0.30
    flatness_rms: float = 0.04
    dead_pixel_fraction: float = 0.0
    frame_rate: float = 1440.0
    settling_time: float = 700e-6
    seed: int = 0
```

Override `represent()` to change what the device can express (see `DMD` for Lee
holography and error diffusion, `TiltMicroMirrorArray` for per-mirror ramp fitting),
or `render()` to replace the whole pipeline.

## A controller / logic block

```python
import numpy as np
from fsphotonix.system.logic import LogicBlock
from fsphotonix.registry import register

@register("controller", "pid")
class PID(LogicBlock):
    name = "pid"
    latency = 1

    def __init__(self, setpoint=0.5, kp=0.5, ki=0.05, kd=0.0):
        self.setpoint, self.kp, self.ki, self.kd = setpoint, kp, ki, kd
        self.reset()

    def reset(self):
        self._i = 0.0
        self._prev = None

    def __call__(self, x):
        e = self.setpoint - np.asarray(x, float).ravel()
        self._i = self._i + e
        d = 0.0 if self._prev is None else e - self._prev
        self._prev = e
        return self.kp * e + self.ki * self._i + self.kd * d
```

`latency` is counted in clock cycles and summed by `Pipeline`, so a controller that
claims latency is reported honestly in `Machine.loop_report()`.

Any plain callable also works as a controller — `as_logic` wraps it.

## Reaching a plugin from a config

Once registered, the name works in YAML:

```yaml
stages:
  - name: panel
    type: modulator
    device: {type: fast_lcos}
    layout: {nx: 8, ny: 8, pitch: 500.0e-6}
loops:
  - sense: screen
    actuate: panel
    mode: piston
    controller: {type: pid, setpoint: 0.4, kp: 0.6}
```

## Backend

Write array code through `fsphotonix.engine` (`eng.xp`, `eng.fft2`, `eng.asarray`,
`eng.to_numpy`) rather than importing numpy directly, and the plugin will work
unchanged on the CUDA backend:

```python
import fsphotonix as fp
fp.engine.use_backend("cupy")
fp.engine.use_precision(single=True)
```
