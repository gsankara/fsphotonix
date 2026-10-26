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
| panel area vs achievable bit depth | `Features.routing`, `cgh_method` | `capacity.estimate`, `capacity.compare` |
| spot size vs detector guard | `Requirements.spot_guard`, `apodization` | `CapacityReport.achievable_bits` |
| fan-out per hologram vs delivered weight accuracy | `route(mode=)`, `MultiSpotHologram.method` | `Superposition.uniformity`, `stray_fraction`, `resolved` |
| sub-aperture taper vs throughput | `route(apodize=)`, `Apodized(window=)`, `tile_region(apodize=)` | `apodization_report` (`power_throughput`, `apodization_loss_db`), `gap_report()["apodization"]` |
| far-field wing suppression vs main-lobe width | choice of `WINDOWS` entry, `mode=` | `apodization_report` (`mainlobe_broadening`, `spot_vs_target_pitch`), `apodization_sweep` |

## Sizing a machine: `capacity.py`

Every report above answers *"is this layout feasible?"*. `capacity.py` answers the
inverse — *"given a problem instance, what hardware does it take?"* — and is the
only module that derives geometry rather than checking it. The SLM comes out as a
required pixel count and pitch, deliberately not restricted to a catalogue part.

```python
from fsphotonix.capacity import ProblemSpec, Requirements, Features, baseline, compare

problem = ProblemSpec.from_graph(adjacency)          # or .from_matrix / .from_ising
print(baseline(problem, Requirements(target_bits=4)).table())
print(compare(problem, [Features(routing="multiplex", cgh_method="wgs")])["table"])
```

Two free parameters — the throw `z` and the detector pitch `out_pitch`, plus the
guard fraction — fix everything else, and both pull against themselves, so the
module searches rather than evaluates a formula. A small `out_pitch` keeps the
screen and the deflection angle small but forces a *larger* sub-aperture; a large
one shrinks the panel but demands a finer pitch to steer that far.

A target bit depth is an error budget split between terms that geometry and
features set (aperture leakage, phase quantisation, weight non-uniformity,
off-support stray) and one bought with laser power. **When the static terms alone
exceed the budget the design is infeasible at any power**, and the report says so
and quotes the depth it can actually reach — which is the useful answer.

Two results fall out, both in `examples/06_capacity.py`:

* **Area division pays a factor of fan-out in panel area and phase multiplexing
  does not** — one sub-aperture per target makes each region `sqrt(fan)` across,
  while a multiplexed region is the sub-aperture itself. For a dense problem that
  is a 10-50x difference in panel size.
* **Whether that saving is usable is decided by the hologram synthesis, not by
  the architecture.** A naive analytic superposition delivers its weights to
  ~30% rms (under two bits); weighted Gerchberg-Saxton delivers them to 1-2% and
  is *more* efficient. `measure=True` re-derives the coefficients by synthesising
  the hologram at the design point instead of trusting the table.

The sizing is cross-checked two ways: `CapacityReport.to_mvm(matrix=...)` hands
the derived geometry to `OpticalMVM` so `check()` can confirm it, and the error
model is pinned by test to land inside the 1-5% rms band that `PHYSICS.md` §8
reports for the real optical matrix.

`CapacityReport.components()` itemises the answer stage by stage -- emitters and
their power, a collimating lenslet per emitter with the pitch the region layout
demands, the SLM as rows x columns of pixels at a derived pitch, how many
holograms and how many pixels each one is written on, the detector grid, and the
features in force at each stage. The panel is *not* forced square: 18 rails tile
as 5x4, and a 2:1 panel is a different piece of glass from a square one with the
same pixel count.

Problems arrive as a matrix, an Ising coupling matrix, a graph, or a `.qubo`
file -- `read_qubo` parses the usual DIMACS-style layout and
`ProblemSpec.from_qubo_file` sizes the Ising problem the optics would actually
run, since `h` is applied electronically and costs no optics.

One check is kept deliberately apart from the error budget. `Features.validate`
asks whether a feature set can represent the problem *at all*: light is
non-negative, so `encoding="unsigned"` against signed weights is not an
inaccurate machine but a machine computing a different function, and the report
says `binding="encoding_mismatch"` rather than quietly sizing something cheaper.
The sign never lives in the hologram -- a CGH can hold a negative amplitude as a
π phase shift, but the detector is square-law and destroys it, so the hologram
carries `|W|` as a power share (`weight_mode`) and the sign lives in which rail
the light lands on.

## Tapering a sub-aperture instead of guarding it

A hard-edged sub-aperture has sinc far-field wings that fall only as `1/u²`. Across
a lattice of simultaneously driven detectors that sets a cross-talk floor no guard
band can escape: opening the gap wider shrinks the sub-aperture, and a smaller
sub-aperture widens *every* spot as `λz/a`, so the cross-talk reappears at the
detector instead. Tapering the sub-aperture's amplitude smoothly to zero at its rim
replaces those wings with a much faster roll-off.

This is apodization *versus* a guard gap, not a free improvement on it. Both remove
the same thing — the phase discontinuity at the edge a sub-hologram shares with its
neighbour. The gap removes it by leaving dead space, and pays in sub-aperture size.
The window removes it by making the field vanish there, keeps the sub-aperture's
size, and pays in transmitted power and a broader main lobe.

```python
from fsphotonix.modulators.geometry import WINDOWS, Apodized, apodize_aperture
from fsphotonix.analysis import apodization_report, apodization_sweep

modulator.route(region, targets, z=0.25, mode="subregions", apodize="hann")
modulator.gap_report()["apodization"]          # the window actually in force
apodization_report(modulator, z=0.25, wavelength=1.55e-6)
```

`Apodized(inner, window, mode)` wraps any aperture and tapers amplitude *without
moving the support*, so apodized sub-apertures still tile their region exactly as
hard-edged ones do. `mode` is `auto` / `separable` / `radial`, and `auto` picks
radial for `Circle`, `Annulus` and `Sector`. `apodize_aperture` does the same for a
loose aperture, `tile_region(apodize=)` for a whole tiling, and `route(apodize=)`
for routing modes `subregions`, `multiplex` and `superpose` alike.

`WINDOWS` — `none`, `tukey`, `hann`, `blackman`, each a function of a normalised
coordinate `t ∈ [−1/2, 1/2]` — lives in `geometry.py` precisely so that
`capacity.py` imports it rather than keeping a copy, which is what stops a window's
predicted cost and its rendered optical effect from drifting apart. They had already
drifted once: a 2-D separable aperture applies the window on *both* axes, so Hann
passes `0.375² = 0.141` of the incident power, not the 1-D `0.375` the capacity
model first assumed.

`apodization_report(modulator, z, wavelength)` is the knob's geometric price —
`power_throughput`, `apodization_loss_db`, `mainlobe_broadening`,
`apodized_spot_size_um` and the resulting `spot_vs_target_pitch`.
`apodization_sweep(build, windows, target_positions, leak_positions, radius)` is the
measured answer: `build(window) -> OpticalSystem`, then propagate and compare the
power landing on the sites an input is meant to drive against the power landing on
sites it is not. For one input fanning to four sites of a 3×3 lattice, the other
five belonging to a neighbouring channel:

| window | cross-talk | suppression vs hard edge | throughput cost | 2-D power throughput | main-lobe broadening |
|---|---|---|---|---|---|
| `none` | −19.3 dB | — | 0 dB | 1.000 | 1.00 |
| `tukey` | −45.5 dB | 26.1 dB | −2.94 dB | 0.476 | 1.03 |
| `hann` | −56.7 dB | 37.3 dB | −8.16 dB | 0.141 | 1.62 |
| `blackman` | −65.7 dB | 46.4 dB | −9.96 dB | 0.093 | 1.83 |

**Tukey is the bargain** — 26 dB of suppression for 2.9 dB of light, because its
flat centre preserves throughput while the taper still reaches zero at the rim. Hann
buys 11 dB more for another 5 dB.

This has to be measured with a **wave** propagator. `PHYSICS.md` §2 is explicit that
the beamlet kernel replaces the true aperture transform with a Gaussian or sinc
envelope: it reproduces spot positions, powers and widths but not the wings — and
the wings are the whole subject here. A sweep built on beamlets reports that
apodization does almost nothing.

The render path was checked to preserve the taper rather than assumed to.
`ModulatorArray.ideal_transmittance` divides the accumulated sub-aperture field by
`maximum(wacc, 1.0)`; for non-overlapping tapers `wacc` stays below 1 near the
edges, so the divisor stays 1 and the window survives. The rendered power ratio
comes out at exactly the window's 2-D throughput.

## Phase multiplexing, and what the projection costs

One sub-region can carry N holograms at once: sum them and keep only the phase.
`Superposition` composes arbitrary gratings that way, `MultiSpotHologram`
synthesises the same thing from a target list, and `route(mode="superpose")` and
`route(mode="multiplex")` reach them. It is the area-efficient alternative to
area division — the panel stops scaling with fan-out — and the whole question is
what the `arg()` projection costs.

It is not free, and the code used to say it was. `Superposition.orders()`
apportioned each constituent's *analytic* orders by its power share; since
`SpotFocuser` reports unit efficiency, the shares summed to exactly 1.0 — a
phase-only projection claiming to lose nothing to intermodulation, which is the
one thing it certainly does lose. `orders()` now **measures**: it rasterises the
hologram over `aperture_size`, transforms it, and integrates the power actually
arriving at each constituent's direction. `stray_fraction()` is what reached no
designed direction at all, `uniformity()` the rms spread of delivered against
requested weights, and `resolved()` false when two targets sit closer than the
aperture can separate.

```python
sp = Superposition(gratings=gs, aperture_size=(a, a))   # route() passes the real one
sum(o.efficiency for o in sp.orders(lam))   # ~0.75, not 1.0
sp.stray_fraction(lam), sp.uniformity(lam), sp.resolved(lam)
```

`orders()` comes back in **constituent order**, so entry *i* belongs to
`gratings[i]` and lines up with `weights[i]`. The aperiodic branch used to sort
by efficiency, which silently permuted that correspondence — a fan-out requested
as 1:2:3:4 came back anti-correlated with what was asked for.

**Random carrier phases are not a detail.** Without them every term peaks
together at the origin, the sum has a huge peak-to-average ratio, and the
projection discards wildly different amounts of each term. `Superposition` now
draws them, seeded, exactly as `MultiSpotHologram` always did:

| n | carrier | efficiency | stray | uniformity |
|---|---|---|---|---|
| 4 | aligned | 0.540 | 0.460 | 0.001 |
| 4 | random | **0.753** | 0.247 | 0.026 |
| 9 | aligned | 0.736 | 0.264 | 0.248 |
| 9 | random | 0.748 | 0.252 | **0.209** |

(targets on a lattice at three spot widths; `examples/08_superposition.py`
reproduces every number here.)

With that fixed, `Superposition` and `MultiSpotHologram(method="superposition")`
measure identically, as two spellings of one hologram should.

**Which synthesis, then.** For a matrix-vector product every target is an
intended output, so the error is the spread in delivered weights rather than
cross-talk, and that is where the optimiser earns its keep:

| fan | `superposition` | `gs` | `wgs` |
|---|---|---|---|
| 9 | 0.748 / 0.209 | 0.797 / 0.049 | 0.795 / **0.028** |
| 25 | 0.745 / 0.407 | 0.789 / 0.096 | 0.788 / **0.051** |

(efficiency / rms weight non-uniformity). WGS wins on the weights by roughly an
order of magnitude and barely at all on throughput — so prefer it for what it
delivers, not for what it transmits.

One limitation worth knowing rather than papering over: **Gerchberg-Saxton
quantises target placement.** It can only put energy on its transform's lattice,
spaced `1/aperture` — one spot width — so a target half a spot off that lattice
is approximated and loses efficiency -- 0.795 on-lattice against 0.690 half a
spot off it -- while the analytic superposition sets its targets by exact phase
and stays within 0.006 across the same sweep. Put targets on the lattice where
the layout allows it.

This is easy to trip over by accident. A near-square block of `n` targets centred
on its own *centroid* puts every target at a half-integer multiple of the pitch
whenever `ceil(sqrt(n))` is even -- exactly half a spot off the lattice -- so a
fan-out sweep over 4, 9, 16, 25 shows GS efficiency alternating by a quarter and
looks like a fan-out effect when it is really an alignment one. Centring the
block on a *site* instead leaves odd counts untouched and brings the even ones
back onto the lattice.

### Measuring orders at all

`gratings.base.measure_orders` is shared by both classes, and two things make it
trustworthy — both learned by getting them wrong first.

*The transform is zero-padded.* An unpadded transform of an `a`-wide aperture has
bins spaced exactly `1/a`, one spot width, **regardless of the sample count**,
because more samples extend the frequency range rather than refine it. A
direction landing halfway between bins then spreads across them and a one-bin
window collects about half its power, so the measured efficiency swung by 2x with
where targets happened to fall. Padding makes the bins finer than the spot, and
the window is specified in spot widths.

*Bins are assigned, not boxed.* Each bin goes to its nearest requested direction
and is counted once, so efficiencies are disjoint by construction and cannot sum
above one. A fixed box around each direction double-counts every bin two close
directions share — which is how an order list came to report a total of 2.6 for a
16-spot lattice.

## Extension points

Nine plugin slots (`source`, `waveform`, `profile`, `geometry`, `grating`, `device`,
`propagator`, `detector`, `controller`), plus `FunctionStage` for an arbitrary hook
at a plane and `CustomDetector` / `CustomAperture` / `CustomProfile` / `Arbitrary`
for one-off callables. See `docs/PLUGINS.md`.

Backend swapping is global: `engine.use_backend("cupy")` and
`engine.use_precision(single=True)`. Plugins that go through `engine.xp` inherit it.
