# Physics and numerics

Conventions, approximations and their validity limits. Everything is SI: metres,
seconds, watts, radians.

## 1. Field representation

A **channel** is one spatial mode at one wavelength with one temporal envelope:

```
E(x, y, t) = u(x, y) · a(t) · exp(−i 2π ν t)
```

`u` has units of √W/m so that `Σ|u|² dx dy` is watts. Space and time are kept
factorised, which is the quasi-monochromatic approximation: valid while the
modulation bandwidth is small compared with the optical carrier (25 GHz on a
193 THz carrier — a ratio of 10⁻⁴). It is what makes the simulator scale, because
propagation is applied once to `u` and the time axis is only expanded at the
detector.

Waveforms are normalised so that `⟨|a(t)|²⟩ = 1` over their period, so a source's
`power` is its **average** optical power.

### Coherence

Channels sharing a `coherence` tag add as complex amplitudes; different tags add in
intensity. Within a tag, an optical frequency offset Δν contributes a beat term
`exp(−i2πΔν t)` only when it is resolvable — `beat_bandwidth` (the detector's) or
the explicit `allow_cross_wavelength_beat` flag. Otherwise only its own intensity is
added, which is the correct time-average of an unresolvable beat.

This is why `TransceiverBank` gives every part its own tag: eight uncooled DFBs
inside one 100 GHz slot are mutually incoherent as far as a 10 GHz photodiode is
concerned.

## 2. Propagation

### Angular spectrum (`angular_spectrum`)

```
H(fx, fy) = exp(i 2π z √(1/λ² − fx² − fy²))
```

Evanescent components (`fx² + fy² > 1/λ²`) are dropped unless `evanescent=True`.
Non-paraxial and exact for a uniform grid, same grid in and out. Band-limited per
Matsushima: frequencies whose rays leave the window within `z` are removed, at
`u_lim = 1/(λ√((2Δu z)² + 1))`, which suppresses the FFT's periodic wrap-around.

### Fresnel transfer function (`fresnel_tf`)

```
H = exp(i k z) exp(−i π λ z (fx² + fy²))
```

Paraxial, same grid, well sampled for `z ≲ N dx²/λ`.

### Zoom Fresnel (`fresnel`) — the workhorse

The Fresnel integral is evaluated directly on an **arbitrary uniform output grid**:

```
U₂(x₂,y₂) = exp(ikz)/(iλz) · exp(ik(x₂²+y₂²)/2z) · Σ u₁ exp(ik(x₁²+y₁²)/2z)
                                                     · exp(−i2π(x₂x₁+y₂y₁)/λz) dx₁dy₁
```

The inner sum is a DTFT at frequencies `fx = x₂/(λz)`, computed with a Bluestein
chirp-z transform (`engine.zoom_dft2`) in O((N+M) log(N+M)). Consequences:

* the screen's pitch and extent are free parameters — no zero padding needed to put
  a 5 mm panel at 2 µm sampling onto a 50 mm screen at 20 µm sampling;
* **every wavelength lands on the same physical grid**, so chromatic walk-off is
  visible rather than hidden inside a per-wavelength rescaling.

`z < 0` works throughout (the feedback code uses it).

### Fraunhofer / Fourier plane

The same zoom transform without the input chirp. `fourier_plane` drops the output
quadratic phase as well, giving the exact `x₂ = λ f fx` mapping of a lens's back
focal plane.

### Beamlet ("spot") propagation

For each sub-region, the code measures the incident power over the (anti-aliased)
sub-aperture mask and the mean incident tilt from wrap-safe phase differences, looks
up the grating's analytic diffraction orders, and deposits one spot per order:

```
position  = sub-region centre + z·α/γ,     (α, β, γ) direction cosines, exact
power     = P_incident · η_order · share · gain² · device efficiency
waist     = 0.75 · λ z / D_subaperture     (matched to the sinc main-lobe FWHM)
```

Cost is O(sub-regions × orders) and independent of screen resolution.

**Exact**: spot positions (non-paraxial), powers, on-axis spot widths, and the wavelength
dependence of all three. Beamlets within one coherence group are summed as complex
amplitudes, so interference between overlapping spots is modelled, and each beamlet
carries its propagation direction so a downstream diffractive plane sees the right
incidence.

**Approximate**: the wings (a Gaussian or sinc envelope replaces the true aperture
transform), and the wavefront within one beamlet (a single tilt plus a piston, so a
strongly aberrated sub-aperture is not resolved). Use a wave kernel when the wings
or the aberrations are the answer.

**Off axis the width is the on-axis one.** The waist is `0.75 λz/D` at every
angle, but a sub-aperture on a planar panel steered to θ onto a parallel screen
gives a spot `1/cos³θ` wider radially and `1/cos θ` tangentially — the panel
foreshortens the aperture, the screen is further away and is struck obliquely.
Measured with `angular_spectrum` and by direct Rayleigh–Sommerfeld summation to
about 1% up to 29.4°, where it is 1.51 × 1.15. On a screen curved about the
panel it is `1/cos θ` × 1. `capacity.obliquity_stretch` has the factors.

### Choosing a kernel

`propagation.suggest(grid, λ, z, out_grid)` reports the critical distance
`z_c = N dx²/λ`, the Fresnel number, the maximum representable angle, the far-field
spot size, and the recommended kernel. `auto` applies that choice per hop.

## 3. Gratings

A grating supplies the complex transmittance of one sub-region in local
coordinates, and its discrete order list.

### Wavelength scaling

A phase realised as an optical path difference (SLM voltage, mirror piston, etched
relief) gives

```
φ(λ) = φ_design · λ_design / λ
```

That one line produces all the chromatic behaviour of the machine. Set
`achromatic=True` for a geometric-phase element (a metasurface) whose phase profile
does not scale — its deflection still obeys the grating equation, but its efficiency
spectrum is its own business.

### Order efficiencies

The base class computes orders numerically from one unit cell (1-D or 2-D lattice),
memoised per instance. Closed forms are used where they exist and are verified
against the numerics in the tests:

| grating | order efficiency |
|---|---|
| blazed, depth `d` (×2π) | `η_m = sinc²(d − m)` → 100% into m=1 at d=1 |
| blazed quantised to `L` levels | extra factor `sinc²(1/L)` (2 levels → 40.5%) |
| binary phase, π step | `η_±1 = (2/π)² = 40.5%`, `η_0 = 0` |
| Ronchi amplitude, 50% duty | `η_0 = 25%`, `η_±1 = 1/π² = 10.1%` |
| sinusoidal phase, depth `m` | `η_n = J_n(m)²` |
| Dammann, N spots | transition points optimised numerically for uniformity |

For aperiodic elements (lenses, CGHs) `orders()` is overridden: `MultiSpotHologram`
rasterises itself over its aperture, transforms, and integrates the energy in a
diffraction-limited window around each designed target — so the reported
efficiencies include the real cross-talk and speckle loss of the phase-only
projection.

### Deflection geometry

`target_to_angles(dx, dy, z)` normalises over the full 3-D vector, so wide-angle
fan-outs land where they are asked to rather than where a paraxial approximation
would put them. `Order.screen_position` is likewise exact.

Routing targets are **absolute screen coordinates** by default; the required
deflection is worked out from each sub-region's own position on the panel.

## 4. Illumination and the two gaps

### Collimator

Gaussian beam through a lenslet, with `zR = π w_s²/λ`:

```
m  = f / √((d − f)² + zR²)          magnification
w_out = m · w_s                     waist behind the lenslet
s'    = f + m²(d − f)               its position
w(L)  = w_out √(1 + ((L − s')/zR_out)²)
```

The illuminated footprint at the panel is `2w(L)`. Sign convention: **negative**
defocus (emitter inside the focal length) makes the waist virtual and the beam
already diverging at the panel, so the footprint grows monotonically with |defocus|.
**Positive** defocus pushes a real waist downstream, so at a fixed short throw the
footprint first *shrinks* and only grows once the waist has passed the panel.
`solve_defocus(..., branch="auto")` picks the reachable side.

Truncation at the lenslet's clear aperture of radius `a` transmits
`1 − exp(−2(a/w)²)`.

### Inter-input gap

`ModulatorArray.tiled(gap=...)` sets `region_size = pitch − gap`. Illumination that
overspills it lands on the neighbouring input's hologram and is diffracted to *that*
input's screen targets — illumination cross-talk, indistinguishable from signal at
the detector. Measured by `analysis.illumination_crosstalk`, aggregated per input
(several channels can serve one input: a main beam plus lenslet stray light, or
several WDM lines).

### Intra-input gap

`route(..., gap=...)` leaves dead space between the sub-holograms of one input,
which suppresses the scatter from the phase discontinuity at a shared edge but
shrinks each sub-aperture `a` — and therefore widens every screen spot as `λz/a`.
`analysis.cgh_gap_report` reports the aperture loss in dB and whether the resulting
spot still fits the target pitch.

## 5. Devices

The rendering pipeline is: project onto the device's modulation space (phase /
amplitude / binary amplitude / complex) → quantise → clamp to the available stroke →
block-average onto device pixels → inter-pixel cross-talk (Gaussian blur in units of
the pitch) → fill-factor aperture mask → frozen flatness error and dead pixels →
insertion loss.

The fill-factor mask is what creates the pixel replica orders, and it is sampled on
the field grid — with only a few samples per device pixel the *area* is quantised, so
use ≥16 samples per pixel when the exact throughput matters.

Reflective phase elements see the path twice: a piston `h` gives `φ = 4π h/λ`, so a
quarter-wave of mechanical stroke buys a full 2π. Transmissive elements get `2π h/λ`.

Maximum steerable angle is set by two pixels per grating period:
`θ_max = arcsin(λ / 2p)`.

## 6. Detectors

**Camera**: `energy = Σ I·dA·t_int · fill_factor` → photons → electrons via QE →
PRNU → dark current → Poisson shot noise (Gaussian above 10⁶ e⁻) → read noise →
full-well clip → ADC. Rolling shutter exposes row *j* at `j/(ny·frame_rate)`.

**Photodiode**: irradiance integrated over the active area, `i = R·M·P`, then a
first-order RC low-pass at `bandwidth` applied in the frequency domain along the
time axis. Noise variance is
`2qI·B·(M·F)` (shot, with APD excess noise) `+ i_TIA²·B + 4kT B/R_L` (thermal).

**SMF**: `P = |∫ u ψ* dA|²` with `ψ` the normalised LP01 Gaussian mode. Because it
is an overlap integral it is intrinsically selective in position *and* angle — a
tilted wavefront simply does not couple, which the tests confirm.

**MMF**: the field is windowed by the core aperture, transformed to angular space,
truncated at `|f| ≤ NA/λ`, and the remaining power is collected. That reproduces the
two facts that matter: a big core is forgiving about position, and the NA is a hard
limit on the collectable angular spread. Modal scrambling destroys the relative
phase, so MMF outputs add incoherently.

## 7. Dual-rail encoding

Light cannot be negative, so a signed matrix-vector product needs rails. With both
signed inputs and signed weights, `OpticalMVM` builds

```
        x⁺      x⁻                       y⁺ = W⁺x⁺ + W⁻x⁻
y⁺  [  W⁺      W⁻  ]                     y⁻ = W⁻x⁺ + W⁺x⁻
y⁻  [  W⁻      W⁺  ]                     y⁺ − y⁻ = (W⁺−W⁻)(x⁺−x⁻) = Wx
```

with `W± = max(±W, 0)`, `x± = max(±x, 0)`. The rails are physically different
regions and detectors, so they have genuinely different losses; `calibrate("rail")`
fits one gain per output, which is the per-detector responsivity trim any real
instrument performs. It cannot hide cross-talk, because the off-diagonal terms of
the measured matrix are left untouched and the residual is computed after the fit.

## 8. Numerical accuracy achieved

From the test suite, against closed form:

* chirp-z / zoom DFT vs direct summation: `rtol < 1e-8`
* Gaussian beam waist after diffraction: within 3% (5% through the zoom kernel)
* single-slit first null: within 8%; focal FWHM vs `0.886 λf/D`: within 15%
* energy conservation: within 1% (same grid), 5% (zoom kernel)
* grating order efficiencies vs `sinc²` / `(2/π)²` / `J_n²`: within 2–3%
* backward propagation round trip: 2% field error
* beamlet vs wave-optics relative spot distribution: within 2 percentage points
* optical matrix vs requested matrix: 1–5% rms after per-rail calibration
