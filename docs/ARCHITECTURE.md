# Architecture

## The data flow

```
LightSource ──► FieldSet ──► ModulatorArray ──► Propagator ──► FieldSet ──► Detector ──► DetectorOutput
   emitters      channels      regions ×          kernel or                   screen        vector
                              sub-regions        beamlets                                     │
                                   ▲                                                          │
                              Actuator ◄──── LogicBlock pipeline ◄──── Loop ◄──────────────────┘
                                                (analog / digital)
```

A `FieldSet` is the single currency between layers: a grid plus a list of channels.
Every layer is a function of a `FieldSet` and returns one, so layers compose without
knowing about each other.

## Core types

| type | module | role |
|---|---|---|
| `Grid` | `grid.py` | uniform 2-D sampling of a transverse plane; independently chosen per plane |
| `Channel` | `fields.py` | one spatial mode: `u(x,y)`, wavelength, coherence tag, temporal envelope |
| `FieldSet` | `fields.py` | the channels at one plane, plus the coherence arithmetic |
| `Waveform` | `waveform.py` | complex temporal envelope `a(t)` |
| `Aperture` | `modulators/geometry.py` | a support mask in its own local coordinates |
| `Grating` | `modulators/gratings/base.py` | complex transmittance **and** an analytic order list |
| `ModulatorDevice` | `modulators/devices.py` | how hardware degrades an ideal hologram |
| `Propagator` | `propagation/base.py` | plane-to-plane transfer, with a free output grid |
| `Detector` | `detectors/base.py` | field → numbers |
| `Stage` | `system/stages.py` | a plane in the system |
| `LogicBlock` | `system/logic.py` | one electronic operation, with a latency |
| `Loop` | `system/feedback.py` | sense → process → actuate, with period/delay/priority |

## Why space and time are factorised

A channel is `u(x, y) · a(t)`. Propagation touches only `u`; the time axis is
expanded only inside a detector. Without this, a 2048² grid over 4096 time samples
would be 137 GB. With it, propagation costs one transform per channel and the
detector chunks time (`mean_irradiance`), so a 25 Gbaud pattern on a megapixel
screen fits in a few hundred MB.

The price is the quasi-monochromatic approximation, which is unarguable here: the
modulation bandwidth is 10⁻⁴ of the carrier.

## Why coherence is explicit, not inferred

The simulator cannot guess whether two beams interfere. A laser split N ways does;
N separately packaged transceivers do not, even at the same nominal wavelength,
because their ~100 GHz beat is far outside any detector bandwidth. Getting this
wrong changes answers by a factor of two (amplitude vs intensity addition), so each
source declares a `coherence` tag per channel and `FieldSet.intensity` obeys it.
`beat_bandwidth` lets a detector decide which frequency offsets it can actually
resolve.

## Why every plane has its own grid

The panel is millimetres at micron sampling; the screen is centimetres at tens of
microns. Forcing one grid on both would need either absurd sample counts or heavy
zero padding. The zoom (chirp-z) Fresnel kernel evaluates the diffraction integral
directly on the receiving plane's grid, so each plane is sampled for its own needs
and — importantly — every wavelength lands on the *same physical* screen grid, which
is what makes chromatic walk-off observable instead of being absorbed into a
per-wavelength rescaling.

## Two propagation regimes, one interface

`ModulatorStage(spot_mode=False)` multiplies the field by the rendered hologram and
hands it to a wave kernel: fully general, cost set by the grid.

`ModulatorStage(spot_mode=True)` treats each sub-region as a beamlet and uses the
grating's analytic orders: cost O(sub-regions × orders), independent of screen
resolution, and it produces an analytic spot *table* alongside the rendered field.

Because the beamlet propagator consumes the modulator and the following hop
together, the stage declares `defers_to_propagator` and the engine skips the
transmittance multiply — otherwise the grating would be applied twice. The field
recorded at a spot-mode plane is therefore the *incident* one.

## Modulator hierarchy

```
ModulatorArray            the panel: a device + a list of regions
 └─ Region                one illuminated patch: aperture, piston, gain, enable
     └─ SubRegion         one hologram: grating, sub-aperture, offset, weight
```

Two ways to fan one input out to many screen points:

* **area division** — several `SubRegion`s, each with its own deflecting or focusing
  hologram (`route(mode="subregions")`). Clean and low noise; each sub-aperture is
  smaller, so each screen spot is wider (`λz/a`).
* **phase multiplexing** — one `SubRegion` holding a `MultiSpotHologram`
  (`route(mode="multiplex")`) or a `Superposition`. Uses the full aperture, so the
  spots stay tight, at the cost of intermodulation orders and speckle that
  `orders()` reports honestly.

`weight_mode` decides how a sub-region's weight becomes physics: `area` shrinks its
aperture (the only honest option on a phase-only device), `amplitude` scales the
field (needs an amplitude device).

## Rendering pipeline

```
holograms                                  device                       field
─────────                                  ──────                       ─────
region mask × sub-region mask × grating  →  represent (phase/amp/binary)
                                         →  quantise, clamp stroke
                                         →  block-average to device pixels
                                         →  inter-pixel cross-talk
                                         →  fill-factor aperture mask
                                         →  flatness error, dead pixels
                                         →  insertion loss             →  ×  u(x,y)
```

`ideal_transmittance()` is the first column, `transmittance()` the whole thing, and
the result is cached per (grid, wavelength, version). Any mutation calls `touch()`,
which bumps the version — so a feedback loop that writes pistons every iteration
re-renders exactly once per change.

## The cascade engine

`OpticalSystem.run()` walks stages in order of optical path length `z`:

1. the first `SourceStage` emits;
2. for each subsequent stage, `_hop` propagates from the previous plane onto this
   plane's grid (beamlet propagator if the previous stage asked for it, otherwise the
   stage's own kernel or the system default);
3. the stage processes the field (`ModulatorArray.apply`, an element's transfer, a
   user hook), unless it defers to the propagator;
4. the stage measures if it is a detector;
5. everything is recorded in a `Trace`.

Reflective folds are handled by *unfolding*: `z` is optical path length, not a
physical coordinate, so a reflective SLM is just a plane in the chain.

**Replacing the detection screen with another modulator array is putting a
`ModulatorStage` where the `DetectorStage` was.** Nothing else changes. That is the
multi-stage case, and `DetectorStage(pass_through=True, tap_fraction=...)` gives the
beam-splitter tap that both measures a plane and feeds the next one.

## Feedback scheduling

```
for k in iterations:
    trace = system.run()                     # the optics, once
    for loop in loops (by priority):         # local before global, say
        if loop.due(k):                      # its own period
            y = loop.measure(trace)          # from a named detector, or a callable
            y = delay_line.push_pop(y)       # its own latency
            u = loop.controller(y)           # LogicBlock pipeline
            u = slew_limit(u)                # its own rate limit
            loop.actuator.write(u)           # piston / gain / drive / weights
```

`period` and `priority` are what let a 20 kHz local alignment servo and a slow
global solver loop coexist in one simulation, and `delay` plus the controller's own
`latency` make the loop's real latency visible rather than assumed away.

## Design tensions the structure exposes

Each of these is a coupling the code makes measurable rather than hiding.

| tension | knobs | reporting |
|---|---|---|
| illuminated footprint vs illumination cross-talk | `Collimator.defocus` / `illumination_diameter`, `tiled(gap=)` | `illumination_crosstalk`, `illumination_sweep` |
| sub-hologram guard gap vs spot size | `route(gap=)`, targets per region | `cgh_gap_report` (`aperture_loss_db`, `detector_guard_ok`) |
| fan-out per region vs sub-aperture size | number of targets, `tiling` | `gap_report`, `OpticalMVM.check` |
| column normalisation vs throughput | `set_matrix(normalize_columns=)` | `Region.gain`, `power_budget` |
| device stroke/bits vs diffraction efficiency | `phase_bits`, `stroke`, `levels` | `Grating.orders`, `diffraction_efficiency` |
| electronic depth vs loop latency | `analog_path` / `digital_path(stages=)` | `Pipeline.latency`, `Machine.loop_report` |
| cascade angle sensitivity | `incident=` / `incidence_from` | `transfer_matrix` vs the intended product |
| WDM reuse of one hologram | wavelength list | `chromatic_walkoff` |

## Extension points

Nine plugin slots (`source`, `waveform`, `profile`, `geometry`, `grating`, `device`,
`propagator`, `detector`, `controller`), plus `FunctionStage` for an arbitrary hook
at a plane and `CustomDetector` / `CustomAperture` / `CustomProfile` / `Arbitrary`
for one-off callables. See `docs/PLUGINS.md`.

Backend swapping is global: `engine.use_backend("cupy")` and
`engine.use_precision(single=True)`. Plugins that go through `engine.xp` inherit it.
