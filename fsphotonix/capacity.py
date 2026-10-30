"""Capacity estimation: from a problem instance to a component budget.

:meth:`fsphotonix.solvers.aim.OpticalMVM.check` and
:meth:`fsphotonix.modulators.array.ModulatorArray.gap_report` answer *"is this
layout feasible?"*.  This module answers the inverse, which is the question a
system proposal actually has to answer: **given a problem instance, what
hardware does it take?**  How many lasers, how big an SLM and at what pixel
pitch, how many detectors, over what throw, at what optical power.

The deliberate asymmetry with the rest of the package: nothing here is
restricted to a catalogue part.  The SLM comes out as a required pixel count and
a required pitch, and it is the caller's job to decide whether such a panel can
be bought, printed or must be subdivided.

How the sizing closes
---------------------
Two free parameters, ``z`` (throw) and ``out_pitch`` (detector spacing), fix
everything else::

    a       = broadening * lambda*z / (guard * out_pitch)   spot fits the detector pitch
    region  = a*sqrt(fan)   area division                   one sub-aperture per target
            = a             phase multiplexing              full aperture, every target
    L_panel = ceil(sqrt(regions)) * region / (1 - gap)
    L_screen= ceil(sqrt(rails_out)) * out_pitch
    theta   = arctan( (L_panel + L_screen)/2 * sqrt(2) / z )
    p       = min( lambda / (2 sin theta),  a / K )          steering vs pixels/sub-aperture
    N_pix   = (L_panel / p)^2

``lambda z / a`` is the spot on axis only.  Off axis a flat panel foreshortens
the sub-aperture to ``a cos(theta)``, and a flat screen both sits further away
(``z / cos``) and is struck obliquely, so the worst spot is stretched by
``1/cos^3`` radially and ``1/cos`` tangentially -- 1.51 x 1.15 at 29.4 deg,
measured by wave propagation (``tests/test_capacity.py``).
``Requirements.screen`` decides which spot the cross-talk term is evaluated on
-- that worst one, a curved detector's, or the uniform ideal -- and
:func:`obliquity_stretch` has all three.

Both free parameters pull against themselves, which is why this is an
optimisation and not a formula. A small ``out_pitch`` keeps the screen and the
deflection angle small (coarse pixels allowed) but forces a *large*
sub-aperture, so the panel grows; a large ``out_pitch`` shrinks the panel but
demands a finer pitch to steer that far. :func:`estimate` searches the plane and
returns the cheapest feasible point under the chosen objective.

**The result the feature flags exist to expose**: under area division the panel
area carries a factor of ``fan``, and under phase multiplexing it does not --
one sub-region addresses every target through the full aperture.  That factor is
paid for in intermodulation and speckle from the phase-only projection, which
puts a *ceiling on achievable bit depth* no amount of laser power can lift.
:func:`baseline` gives the conservative reference and :func:`compare` puts the
variants side by side, so the saving and its price are one table.

Accuracy budget
---------------
A target effective bit depth ``b`` is an error budget ``eps = 2^-b`` on each
output, which the design has to split::

    eps^2  >=  eps_crosstalk^2 + eps_quantisation^2 + eps_uniformity^2 + eps_noise^2
               \\_________________ static, geometry and feature _______/   \\_ power _/

The static terms come from the feature set and the device; only the last is
bought with optical power.  If the static terms alone exceed the budget the
design is **infeasible at any power**, and the report says so and quotes the
bit depth it can actually reach -- a far more useful answer than a power number.

Model honesty
-------------
:class:`FeatureModel` holds the efficiency and cross-talk coefficients. The
window numbers are the standard 1-D separable values; the phase-multiplexing
penalties are first-order placeholders, and
``FeatureModel.measured(...)`` replaces them by **running the simulator's own
CGH code** (``MultiSpotHologram.orders()`` rasterises, transforms and integrates
a diffraction-limited window per target, so it reports real intermodulation and
speckle loss). Prefer the measured path whenever the fan-out is small enough to
afford it, and treat the defaults as a sketch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from functools import lru_cache

import numpy as np

from . import engine as eng
from .detectors.photodiode import PhotodiodeArray
from .modulators.devices import ModulatorDevice, as_device
from .modulators.geometry import WINDOWS as _WINDOWS

__all__ = [
    "ProblemSpec",
    "Features",
    "Requirements",
    "FeatureModel",
    "CapacityReport",
    "estimate",
    "baseline",
    "compare",
    "measure_routing",
    "window_metrics",
    "read_qubo",
    "obliquity_stretch",
    "SCREENS",
]


# --------------------------------------------------------------------------- #
# the problem instance
# --------------------------------------------------------------------------- #
def read_qubo(path) -> np.ndarray:
    """Read a ``.qubo`` file into a dense upper-triangular ``Q``.

    The format is the usual one::

        c any comment
        p qubo 0 n n m
        i i value      one line per variable  (the diagonal, Q_ii)
        i j value      one line per coupler   (i < j, Q_ij)

    with ``H(x) = sum_i Q_ii x_i + sum_{i<j} Q_ij x_i x_j`` over x in {0, 1}.
    Only the entries present are set; everything else stays zero.
    """
    n = None
    entries: list[tuple[int, int, float]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line or line.startswith("c"):
                continue
            if line.startswith("p"):
                parts = line.split()
                if len(parts) < 4 or parts[1] != "qubo":
                    raise ValueError(f"{path}:{lineno}: bad header {line!r}")
                n = int(parts[3])
                continue
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(f"{path}:{lineno}: expected 'i j value', got {line!r}")
            entries.append((int(parts[0]), int(parts[1]), float(parts[2])))
    if n is None:
        raise ValueError(f"{path}: no 'p qubo' header line")
    Q = np.zeros((n, n))
    for i, j, v in entries:
        if not (0 <= i < n and 0 <= j < n):
            raise ValueError(f"{path}: index ({i}, {j}) outside 0..{n - 1}")
        Q[i, j] = v
    return Q


@dataclass(frozen=True)
class ProblemSpec:
    """What the optics has to compute, stripped to the numbers that size it.

    ``max_fanout`` is the largest number of outputs any single input drives --
    the nonzeros in the busiest *column*.  It, not ``n_out``, is what sets the
    sub-aperture division, so a sparse problem is genuinely cheaper to build.
    """

    n_in: int
    n_out: int
    max_fanout: int
    nnz: int
    signed_input: bool = True
    signed_weights: bool = True
    dynamic_range_bits: float = 0.0
    column_balance: float = 1.0  # mean(|col|sum) / max(|col|sum); 1.0 = balanced
    iterations: int | None = None
    label: str = "problem"

    @property
    def density(self) -> float:
        return self.nnz / max(self.n_in * self.n_out, 1)

    @classmethod
    def from_matrix(cls, W, signed_input: bool = True, iterations: int | None = None,
                    label: str = "matrix") -> "ProblemSpec":
        """Size from an explicit ``n_out x n_in`` weight matrix."""
        W = np.asarray(W, dtype=float)
        if W.ndim != 2:
            raise ValueError("W must be 2-D")
        A = np.abs(W)
        nz = A > 0
        per_col = nz.sum(axis=0)
        col_sums = A.sum(axis=0)
        nonzero = A[nz]
        drange = (float(np.log2(nonzero.max() / nonzero.min()))
                  if nonzero.size and nonzero.min() > 0 else 0.0)
        balance = (float(col_sums[col_sums > 0].mean() / col_sums.max())
                   if col_sums.max() > 0 else 1.0)
        return cls(n_in=W.shape[1], n_out=W.shape[0],
                   max_fanout=int(max(per_col.max(), 1)), nnz=int(nz.sum()),
                   signed_input=signed_input,
                   signed_weights=bool((W < 0).any()),
                   dynamic_range_bits=drange, column_balance=balance,
                   iterations=iterations, label=label)

    @classmethod
    def from_ising(cls, J, h=None, iterations: int = 300,
                   label: str = "ising") -> "ProblemSpec":
        """Size from an Ising coupling matrix -- the AIM case.

        The optical job each iteration is ``J s``, so the multiplier is
        ``n x n``; ``iterations`` carries through to the time-to-solution.
        """
        J = np.asarray(J, dtype=float)
        if J.ndim != 2 or J.shape[0] != J.shape[1]:
            raise ValueError("J must be square")
        spec = cls.from_matrix(J, iterations=iterations, label=label)
        return replace(spec, label=label, iterations=iterations)

    @classmethod
    def from_qubo(cls, Q, iterations: int = 300, label: str = "qubo") -> "ProblemSpec":
        """Size from a QUBO matrix ``H(x) = x^T Q x`` over x in {0, 1}.

        The optics computes ``J s`` each iteration, so the machine is sized from
        the Ising coupling matrix, not from ``Q`` directly. The linear field
        ``h`` falls out of the same conversion and is applied electronically in
        the feedback path -- it costs no optics and so does not appear here.
        """
        from .solvers.aim import qubo_to_ising

        J, _h, _offset = qubo_to_ising(np.asarray(Q, dtype=float))
        return cls.from_ising(J, iterations=iterations, label=label)

    @classmethod
    def from_qubo_file(cls, path, iterations: int = 300,
                       label: str | None = None) -> "ProblemSpec":
        """Size from a ``.qubo`` file in the usual DIMACS-style layout.

        ``p qubo 0 n n m`` then one ``i i value`` line per variable and one
        ``i j value`` line per coupler, ``c`` for comments. Returns the spec for
        the Ising problem the optics would actually run; use :func:`read_qubo`
        if you want ``Q`` itself.
        """
        import os

        Q = read_qubo(path)
        return cls.from_qubo(Q, iterations=iterations,
                             label=label or os.path.basename(str(path)))

    @classmethod
    def from_graph(cls, adjacency, iterations: int = 300,
                   label: str = "maxcut") -> "ProblemSpec":
        """Size from a Max-Cut graph (``J = -W``, so the sparsity is the graph's)."""
        from .solvers.aim import maxcut_ising

        J, _ = maxcut_ising(adjacency)
        return cls.from_ising(J, iterations=iterations, label=label)

    def summary(self) -> dict:
        return {"label": self.label, "n_in": self.n_in, "n_out": self.n_out,
                "max_fanout": self.max_fanout, "nnz": self.nnz,
                "density": self.density, "signed_input": self.signed_input,
                "signed_weights": self.signed_weights,
                "dynamic_range_bits": self.dynamic_range_bits,
                "column_balance": self.column_balance, "iterations": self.iterations}


# --------------------------------------------------------------------------- #
# the features whose cost we want to see
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Features:
    """Architecture choices, each of which the report prices separately.

    ``encoding``
        ``dual_rail`` -- two rails per signed quantity, the honest four-quadrant
        multiply (and what :class:`OpticalMVM` builds).  ``unsigned`` assumes the
        problem is non-negative and halves both component counts.
    ``routing``
        ``subregions`` -- area division, one sub-aperture per target.  Clean, and
        the panel pays a factor of ``fan`` in area.
        ``multiplex`` -- one phase-multiplexed CGH over all targets
        (``MultiSpotHologram``): full aperture, so the panel is independent of
        fan-out, at the cost of intermodulation and speckle.
        ``superpose`` -- complex superposition of per-target gratings projected
        back to phase-only (``Superposition``): same area win, a little worse
        than a synthesised CGH because nothing optimises the projection.
    ``apodization``
        ``None`` for a hard-edged sub-aperture, or a window name to taper it.
        Buys sidelobe (cross-talk) suppression and costs throughput and spot
        width -- the trade that TODO 1 will measure properly.
    ``time_multiplex``
        Split each input's fan-out across ``t`` sequential passes: the panel
        shrinks (fan per pass drops) and the update rate divides by ``t``.
    """

    encoding: str = "dual_rail"
    routing: str = "subregions"
    apodization: str | None = None
    time_multiplex: int = 1
    cgh_method: str = "wgs"

    def __post_init__(self):
        if self.encoding not in ("dual_rail", "unsigned"):
            raise ValueError(f"unknown encoding {self.encoding!r}")
        if self.routing not in ("subregions", "multiplex", "superpose"):
            raise ValueError(f"unknown routing {self.routing!r}")
        if self.cgh_method not in CGH_METHODS:
            raise ValueError(f"unknown cgh_method {self.cgh_method!r}; "
                             f"have {sorted(CGH_METHODS)}")
        if self.time_multiplex < 1:
            raise ValueError("time_multiplex must be >= 1")

    @property
    def multiplexed(self) -> bool:
        """True when one sub-region addresses every target through a full aperture."""
        return self.routing in ("multiplex", "superpose")

    @property
    def tag(self) -> str:
        bits = [self.encoding, self.routing]
        if self.routing == "multiplex":
            bits[-1] = f"{self.routing}/{self.cgh_method}"
        if self.apodization:
            bits.append(f"apod={self.apodization}")
        if self.time_multiplex > 1:
            bits.append(f"tdm={self.time_multiplex}")
        return "+".join(bits)

    def validate(self, problem: "ProblemSpec") -> list[str]:
        """Reasons this feature set cannot represent ``problem`` at all.

        Separate from the error budget on purpose. A design that is merely
        inaccurate still computes the right function badly; one that cannot
        encode the sign of a weight computes a different function, and no amount
        of laser power or panel area repairs that.
        """
        bad = []
        if self.encoding == "unsigned":
            if problem.signed_weights:
                bad.append(
                    "encoding='unsigned' cannot represent this problem: its "
                    "weights are signed, and light is non-negative. Use "
                    "'dual_rail', or pre-condition the matrix.")
            if problem.signed_input:
                bad.append(
                    "encoding='unsigned' drops the input rail, but this "
                    "problem's inputs are signed.")
        return bad

    @classmethod
    def baseline(cls) -> "Features":
        """The conservative reference: dual rail, area division, nothing clever."""
        return cls()


@dataclass(frozen=True)
class Requirements:
    """Performance asked of the machine, and the engineering margins."""

    wavelength: float = 1550e-9
    #: Effective bit depth per output.  The default is deliberately modest:
    #: ``docs/PHYSICS.md`` records the simulator's own measured matrix accuracy as
    #: 1-5% rms after per-rail calibration, i.e. 4.3 to 6.6 bits, so asking for
    #: much more than 5 is asking for better than the engine has ever shown.
    target_bits: float = 5.0
    z: float | None = None  # None -> searched
    out_pitch: float | None = None  # None -> searched
    detector_bandwidth: float = 1e9
    #: Spot size as a fraction of the detector pitch.  ``None`` searches it,
    #: which is usually what you want: it is the one knob that trades panel area
    #: against optical cross-talk, so pinning it pins the achievable bit depth.
    spot_guard: float | None = None
    min_spot_guard: float = 0.15
    max_spot_guard: float = 0.95
    guard_points: int = 17
    #: a hologram needs this many device pixels across one sub-aperture
    min_pixels_per_subaperture: int = 16
    #: detector active width as a fraction of the output pitch -- it sets both
    #: how much of a channel's own spot is collected and how much of its
    #: neighbour's leaks in, so it is a real design knob, not a detail
    detector_active_frac: float = 0.8
    #: guard band between neighbouring inputs, as a fraction of the region pitch
    inter_input_gap_frac: float = 0.15
    #: search bounds and fabrication floors
    min_out_pitch: float = 50e-6
    max_out_pitch: float = 20e-3
    min_z: float = 10e-3
    max_z: float = 1.0
    min_device_pitch: float = 0.5e-6
    #: Largest corner deflection to allow.  A thin diffractive element's
    #: efficiency and the scalar diffraction treatment both degrade badly beyond
    #: a few tens of degrees, and a design that wants 80 deg is not a design.
    #: Note this is a limit on the *geometry*; the pixel pitch needed to reach it
    #: is an output, which is the whole point of sizing rather than selecting.
    max_deflection_deg: float = 30.0
    max_panel: float | None = None
    max_screen: float | None = None
    #: Shape of the detector surface, which decides how much the off-axis spots
    #: grow (:func:`obliquity_stretch`).  ``flat`` is a planar photodiode array,
    #: the part you can buy.  ``curved`` is a sphere of radius ``z`` centred on
    #: the panel: normal incidence at constant range, which removes the screen's
    #: share but not the panel's foreshortening.  ``ideal`` is a uniform
    #: ``lambda z / a`` everywhere -- what this module always assumed, and what
    #: the beamlet propagator still assumes (its waist is ``0.75 lambda z / D``
    #: at every angle), so it stays the default while the two agree and every
    #: ideal report carries a note quoting what a flat array would cost.
    #: Sizing for ``flat`` on CCG designs moves the optimum from ~30 to ~23 deg,
    #: grows the panel 13-42% and costs most of the half-bit margin.
    screen: str = "ideal"
    objective: str = "slm_pixels"  # slm_pixels | panel_area | optical_power
    search_points: int = 96

    def __post_init__(self):
        if self.objective not in ("slm_pixels", "panel_area", "optical_power"):
            raise ValueError(f"unknown objective {self.objective!r}")
        if self.screen not in SCREENS:
            raise ValueError(f"unknown screen {self.screen!r}; have {SCREENS}")


# --------------------------------------------------------------------------- #
# the coefficients -- a model, and labelled as one
# --------------------------------------------------------------------------- #
SCREENS = ("flat", "curved", "ideal")


def obliquity_stretch(theta: float, screen: str = "flat") -> tuple[float, float]:
    """``(radial, tangential)`` growth of a spot steered to ``theta``, over ``lambda z/a``.

    Two surfaces contribute, and only one of them can be curved away.  The
    *panel* is planar silicon, so a sub-aperture seen from direction ``theta``
    is foreshortened to ``a cos(theta)`` in the plane of deflection and its
    angular spread grows by ``1/cos`` there -- whatever the detector does.  A
    *flat screen* then adds two more factors: the spot sits at range
    ``z / cos`` instead of ``z``, and it lands obliquely, smeared by another
    ``1/cos`` radially.  Altogether::

        flat     1/cos^3  radial,  1/cos  tangential
        curved   1/cos    radial,  1      tangential   (sphere of radius z)
        ideal    1                 1                   (the old assumption)

    These are the wave-optics answer, not a ray sketch: an exact spherical
    ``SpotFocuser`` propagated by the non-paraxial angular-spectrum kernel and
    by direct Rayleigh-Sommerfeld summation reproduces all four to about 1%
    from 0 to 29.4 deg, and ``tests/test_capacity.py`` pins the flat case.  So
    a curved detector does *not* make the uniform-spot model true -- it leaves
    the panel's ``1/cos``, 1.15x at 29.4 deg -- and a flat one is worse than a
    ``1/cos^2`` estimate suggests: 1.51x, not 1.32x.
    """
    if screen not in SCREENS:
        raise ValueError(f"unknown screen {screen!r}; have {SCREENS}")
    c = math.cos(theta)
    if screen == "ideal" or c <= 0:
        return 1.0, 1.0
    if screen == "curved":
        return 1.0 / c, 1.0
    return 1.0 / c**3, 1.0 / c


def _axis_stretch(radial: float, tangential: float, cx: float,
                  cy: float) -> tuple[float, float]:
    """Spot growth along the detector lattice's two axes.

    The worst spot is stretched along the line from its region to its detector,
    which at a corner is a diagonal, while the cross-talk model is separable
    along the lattice.  Each axis takes the spot's rms extent along it,
    ``sqrt((e_r c)^2 + (e_t s)^2)``; against a Rayleigh-Sommerfeld cut at 45
    deg that comes out about 4% wide, i.e. slightly conservative.
    """
    return (math.sqrt((radial * cx) ** 2 + (tangential * cy) ** 2),
            math.sqrt((radial * cy) ** 2 + (tangential * cx) ** 2))


#: Sub-aperture amplitude windows.  Imported from
#: :mod:`fsphotonix.modulators.geometry` rather than redefined here, so the cost
#: this module *predicts* for a window and the taper the optics actually renders
#: come from one definition.  They did drift once: a 2-D separable aperture
#: passes ``mean(w^2)`` per axis, so Hann passes 0.375**2 = 0.141 of the incident
#: power, not the 1-D 0.375 this module first assumed.
#:
#: Everything else about a window -- its throughput, how much it broadens the
#: spot, how much it leaks into the neighbouring detector -- is *computed* from
#: the window below rather than tabulated, so the three stay consistent and a new
#: window needs one line in the geometry table.
WINDOWS = _WINDOWS

_WINDOW_SAMPLES = 256
_WINDOW_PAD = 16384
_window_cache: dict[str, tuple] = {}


def _window_profile(name: str) -> tuple:
    """``(throughput, broadening, u_axis, power)`` for one window, cached.

    ``power`` is the window's far-field intensity against ``u = X a / (lambda z)``
    -- screen position in units of the hard-edged first-null radius -- so ``u=1``
    is where a rect aperture puts its first zero.  ``throughput`` is
    ``mean(w^2)``, the fraction of incident power the taper passes, and
    ``broadening`` is the FWHM ratio against a hard edge.
    """
    key = (name or "none").lower()
    if key in _window_cache:
        return _window_cache[key]
    if key not in WINDOWS:
        raise ValueError(f"unknown apodization {name!r}; have {sorted(WINDOWS)}")
    n, N = _WINDOW_SAMPLES, _WINDOW_PAD
    t = (np.arange(n) - (n - 1) / 2) / n
    # the geometry table writes windows against eng.xp so they render on either
    # backend; this module wants plain numpy to transform
    w = eng.to_numpy(WINDOWS[key](eng.asarray(t))).astype(float)
    # a square aperture applies the window along *both* axes, so the power it
    # passes is mean(w^2) once per axis -- Hann passes 0.141, not 0.375
    throughput = float(np.mean(w**2)) ** 2

    H = np.fft.rfft(w, N)
    power = np.abs(H) ** 2
    # DFT bin k sits at u = k * n / N
    u = np.arange(power.size) * (n / N)

    half = power[0] / 2.0
    idx = int(np.argmax(power < half))
    fwhm = u[idx] if idx > 0 else u[1]
    if key == "none":
        _ref = fwhm
    else:
        _ref = _window_profile("none")[4]
    out = (throughput, float(fwhm / _ref), u, power, float(fwhm))
    _window_cache[key] = out
    return out


def _capture(name: str, center_u: float, half_u: float) -> float:
    """Power fraction landing within ``+/- half_u`` of ``center_u``, in u units.

    One-dimensional and normalised to the window's total far-field power, so for
    a square aperture the two axes multiply.
    """
    _, _, u, power, _ = _window_profile(name)
    # rfft gives only u >= 0 and the pattern is even, so every bin but DC stands
    # for two points of the real axis -- in the numerator as well as the total
    total = power[0] + 2.0 * power[1:].sum()
    if total <= 0:
        return 0.0
    lo, hi = center_u - half_u, center_u + half_u
    pos = (u >= lo) & (u <= hi)
    neg = (-u >= lo) & (-u <= hi)
    neg[0] = False  # DC belongs to one side only
    return float((power[pos].sum() + power[neg].sum()) / total)


def window_metrics(name: str | None) -> tuple[float, float]:
    """``(power throughput, main-lobe broadening)`` of an apodization window.

    The public way in: :mod:`fsphotonix.analysis` uses it so that a measured
    apodization report and a predicted capacity estimate quote the same numbers.
    """
    throughput, broadening, _, _, _ = _window_profile(name or "none")
    return throughput, broadening


@lru_cache(maxsize=16384)
def _crosstalk_ratio(name: str, sep_u: float, half_u: float,
                     reach: int = 3, stretch_x: float = 1.0,
                     stretch_y: float = 1.0) -> tuple[float, float]:
    """``(leak/signal, signal)`` for a square detector lattice of pitch ``sep_u``.

    Sums every neighbour out to ``reach`` lattice steps in both axes -- the
    design assumption that all of them are driven at once, which is the case for
    a dense matrix. Returns the relative power error and the capture fraction of
    a channel's own detector, which is itself a real optical loss.

    Cached, and worth understanding why it can be: the sub-aperture is sized as
    ``a = broadening * lambda z / (guard * out_pitch)``, so the neighbour
    separation in ``u`` is ``broadening / guard`` -- it depends only on the
    window and the guard fraction, never on the throw or the detector pitch.
    That is also why ``spot_guard`` has to be a searched design parameter rather
    than a fixed margin: it is the *only* knob that moves optical cross-talk.

    ``stretch_x``/``stretch_y`` grow the spot along each axis (an oblique spot,
    see :func:`obliquity_stretch`).  The detectors do not grow with it, so in
    the spot's own units both the neighbour and the collecting window shrink.
    """
    sx = {i: _capture(name, abs(i) * sep_u / stretch_x, half_u / stretch_x)
          for i in range(-reach, reach + 1)}
    sy = {i: _capture(name, abs(i) * sep_u / stretch_y, half_u / stretch_y)
          for i in range(-reach, reach + 1)}
    signal = sx[0] * sy[0]
    if signal <= 0:
        return float("inf"), 0.0
    leak = 0.0
    for i in range(-reach, reach + 1):
        for j in range(-reach, reach + 1):
            if i == 0 and j == 0:
                continue
            leak += sx[i] * sy[j]
    return leak / signal, signal


#: ``cgh synthesis method -> (efficiency, relative rms weight non-uniformity)``
#: for a phase-multiplexed hologram.  **These are measured**, not guessed: see
#: ``examples/06_capacity.py``, which reads them back out of
#: :meth:`MultiSpotHologram.orders` over a range of fan-outs.
#:
#: The uniformity column is the single most consequential number in this module.
#: A naive analytic superposition delivers its weights to about 30% rms, barely
#: 1.7 bits, which is what makes phase multiplexing look unusable; weighted
#: Gerchberg-Saxton delivers them to 4%. Choosing to multiplex is therefore not
#: really a choice about area -- it is a choice about whether the hologram is
#: optimised.
#:
#: The efficiency column is near-flat across the three, and that is a correction
#: to an earlier, over-flattering reading of it. Measured honestly (disjoint,
#: zero-padded windows: see ``gratings.base.measure_orders``) all three land
#: around 0.75-0.79, and GS/WGS can fall *below* the analytic sum when targets
#: are off-lattice: GS places energy only on the ``1/aperture`` grid, one spot
#: width, so a target half a spot off it loses about 8 points of efficiency in
#: one axis and more in two, while the analytic superposition places targets
#: continuously by exact phase and is flat against offset. Prefer WGS for the
#: weights it delivers, not for its throughput, and put targets on the lattice
#: where the layout allows it.
CGH_METHODS: dict[str, tuple[float, float]] = {
    "superposition": (0.75, 0.300),
    "gs": (0.79, 0.070),
    "wgs": (0.79, 0.040),
}


@dataclass(frozen=True)
class FeatureModel:
    """Efficiency and non-uniformity coefficients per feature.

    The window behaviour is computed from the window itself and is trustworthy.
    The routing coefficients are drawn from :data:`CGH_METHODS`, which came out
    of the simulator, but at one aperture and one set of fan-outs -- so for a
    design that matters, pass ``measure=True`` to :func:`estimate` and have the
    hologram synthesised at the actual design point instead.
    """

    #: diffraction efficiency of one off-axis focusing hologram (measured ~0.84)
    subregion_efficiency: float = 0.84
    #: relative rms spread of per-target efficiency under area division
    #: (measured ~0.06-0.07, near flat in fan-out: it is the variation of a
    #: single-spot hologram's efficiency with deflection angle, not a fan-out
    #: effect)
    subregion_uniformity: float = 0.07
    #: The ``Superposition`` grating used to lack randomised carrier phases, so it
    #: sat below even the analytic ``superposition`` row of CGH_METHODS. It now
    #: draws them, and measures identically to that row -- hence 1.0. Kept as a
    #: knob because a constituent set that is not a plain spot fan-out can still
    #: project worse than the measured reference.
    superpose_efficiency_penalty: float = 1.0
    superpose_uniformity_penalty: float = 1.0
    #: Fraction of the weight non-uniformity that survives calibration.
    #: ``OpticalMVM.calibrate("rail")`` fits one gain per output, which takes out
    #: the part of the spread that is common to an output row and leaves the
    #: per-(output, input) variation -- so the raw measured spread overstates the
    #: end-to-end error. Half is the deliberately unflattering assumption; it is
    #: also what makes this model land inside the 1-5% rms band that
    #: ``docs/PHYSICS.md`` reports for the real thing.
    calibration_residual: float = 0.5
    #: measured override, keyed by (routing, cgh_method, fan)
    measurements: dict = field(default_factory=dict)

    def window(self, name: str | None) -> tuple[float, float]:
        """``(power throughput, main-lobe broadening)`` of an apodization window."""
        throughput, broadening, _, _, _ = _window_profile(name or "none")
        return throughput, broadening

    def routing_penalty(self, features: Features, fan: int,
                        rails_out: int) -> dict:
        """Efficiency, weight non-uniformity and stray cross-talk of the routing.

        The two routing families are deliberately asymmetric, because the light
        each one *loses* goes somewhere different.

        **Area division** loses power to its grating's other diffraction orders,
        which leave at quite different angles and miss the detector array
        altogether, so its near-neighbour leakage is set by the aperture
        diffraction pattern alone -- computed by :func:`_crosstalk_ratio`, not
        guessed here.

        **Phase multiplexing** loses power to intermodulation orders that sit
        *among* the designed spots. Note what that does and does not mean: for a
        dense column every spot in the fan is an **intended** output, so the
        error is the spread in the delivered weights -- the non-uniformity --
        and not cross-talk at all. Stray light only becomes cross-talk where it
        falls on an output this input is not supposed to drive, so the stray term
        is scaled by the fraction of the output array outside the column's
        support. A dense problem therefore pays in uniformity; a sparse one pays
        in both. The stray is assumed to stay among the designed spots rather
        than spreading over the whole plane, which is the conservative reading;
        ``measure=True`` replaces it with an integrated number.
        """
        off_support = max(1.0 - fan / max(rails_out, 1), 0.0)
        hit = self.measurements.get((features.routing, features.cgh_method, int(fan)))
        if hit is not None:
            got = dict(hit)
            if "stray" in got:
                # the measurement knows the stray fraction; only the caller knows
                # how much of the output array it can actually corrupt
                eff = max(got.get("efficiency", 1.0), 1e-12)
                got["crosstalk"] = got["stray"] / eff * off_support
            got["uniformity"] = got.get("uniformity", 0.0) * self.calibration_residual
            return got
        if features.multiplexed:
            if features.routing == "superpose":
                # the Superposition grating is the analytic sum with no optimiser
                eff, unif = CGH_METHODS["superposition"]
                eff *= self.superpose_efficiency_penalty
                unif *= self.superpose_uniformity_penalty
            else:
                eff, unif = CGH_METHODS[features.cgh_method]
            return {"efficiency": eff,
                    "uniformity": unif * self.calibration_residual,
                    "crosstalk": (1.0 - eff) / eff * off_support,
                    "source": "model"}
        return {"efficiency": self.subregion_efficiency,
                "uniformity": self.subregion_uniformity * self.calibration_residual,
                "crosstalk": 0.0,
                "source": "model"}

    def measured(self, features: Features, fan: int, aperture: float,
                 wavelength: float, z: float, out_pitch: float,
                 samples: int = 256, u_sep: float | None = None) -> "FeatureModel":
        """Return a copy whose ``(routing, fan)`` entry came from the simulator."""
        got = measure_routing(fan, aperture, wavelength, z, out_pitch,
                              routing=features.routing, samples=samples,
                              u_sep=u_sep, method=features.cgh_method)
        m = dict(self.measurements)
        m[(features.routing, features.cgh_method, int(fan))] = got
        return replace(self, measurements=m)


#: Target separation, in spot widths, used when measuring a routing penalty.
#: :meth:`MultiSpotHologram.orders` integrates a +/-1-bin window per target and
#: its frequency bins are spaced ``1/a`` -- one spot width -- *regardless of the
#: sample count*, since more samples extend the frequency range rather than
#: refine it. Targets closer than about 3 spot widths therefore have overlapping
#: integration windows and the same energy is counted more than once, which is
#: how that method comes to report efficiencies summing above 1. Measuring at
#: this separation keeps the windows disjoint.
_MEASURE_MIN_USEP = 3.0


def measure_routing(fan: int, aperture: float, wavelength: float, z: float,
                    out_pitch: float, routing: str = "multiplex",
                    samples: int = 256, seed: int = 0,
                    u_sep: float | None = None, method: str = "wgs") -> dict:
    """Measure a routing choice with the package's own hologram synthesis.

    Lays ``fan`` targets on a near-square lattice at range ``z``, synthesises the
    hologram over an ``aperture``-wide sub-region, and reads back
    :meth:`MultiSpotHologram.orders`, which rasterises, transforms and integrates
    a diffraction-limited window per target -- so the efficiency it reports
    already includes the intermodulation and speckle loss of the phase-only
    projection, and the spread across targets is the real non-uniformity.

    The lattice pitch is forced to at least ``_MEASURE_MIN_USEP`` spot widths
    (see above), widening it past the design's own ``out_pitch`` if need be. That
    is sound for the two numbers taken from here -- efficiency and uniformity,
    which are properties of the projection rather than of the spacing -- and the
    returned ``u_sep`` says what was actually used. Near-neighbour leakage is
    *not* taken from this measurement; it comes from the aperture window's own
    transform in :func:`_crosstalk_ratio`, where the design separation applies.

    For ``routing="subregions"`` each target gets its own single-spot hologram on
    the aperture that area division actually leaves it (``sqrt(fan)`` times
    smaller), which is the honest comparison.
    """
    from .modulators.gratings.cgh import MultiSpotHologram, SpotTarget

    fan = int(max(fan, 1))
    spot = wavelength * z / max(aperture, 1e-30)
    want = max(u_sep or 0.0, _MEASURE_MIN_USEP) * spot
    pitch = max(out_pitch, want)
    cols = int(math.ceil(math.sqrt(fan)))
    tg: list[SpotTarget] = []
    for i in range(fan):
        r, c = divmod(i, cols)
        tg.append(SpotTarget(x=(c - (cols - 1) / 2) * pitch,
                             y=(r - (cols - 1) / 2) * pitch, z=z, weight=1.0))

    if routing == "subregions":
        # one target per sub-aperture: measure a single-spot hologram on the
        # sub-aperture that the area division actually leaves it
        sub = aperture / math.sqrt(fan)
        effs = []
        for t in tg:
            h = MultiSpotHologram(targets=[t], range_z=z, aperture_size=(sub, sub),
                                  seed=seed)
            effs.append(sum(o.efficiency for o in h.orders(wavelength, samples=samples)))
        e = np.asarray(effs, dtype=float)
        return {"efficiency": float(min(e.mean(), 1.0)),
                "uniformity": float(e.std() / e.mean()) if e.mean() > 0 else 0.0,
                "crosstalk": 0.0, "stray": 0.0,
                "u_sep": pitch / spot, "source": "measured"}

    # the Superposition grating has no optimiser at all, so measure it as the
    # analytic sum; a multiplexed CGH gets whichever synthesis was asked for
    synth = "superposition" if routing == "superpose" else method
    h = MultiSpotHologram(targets=tg, range_z=z, aperture_size=(aperture, aperture),
                          method=synth, iterations=60, gs_samples=max(samples, 128),
                          seed=seed)
    orders = h.orders(wavelength, samples=samples)
    e = np.asarray([o.efficiency for o in orders], dtype=float)
    raw = float(e.sum())
    overlap = raw > 1.0
    total = min(raw, 1.0)
    # Power that landed in no designed spot is intermodulation and speckle. Report
    # it as a stray *fraction*; how much of it becomes an output error depends on
    # how much of the output array lies outside the column's support, which is the
    # caller's problem, not the hologram's -- see FeatureModel.routing_penalty.
    stray = max(1.0 - total, 0.0)
    return {"efficiency": total,
            "uniformity": float(e.std() / e.mean()) if e.mean() > 0 else 0.0,
            "crosstalk": float(stray / total) if total > 0 else float("inf"),
            "stray": stray, "u_sep": pitch / spot,
            "raw_efficiency_sum": raw, "window_overlap": overlap,
            "source": "measured"}


# --------------------------------------------------------------------------- #
# the report
# --------------------------------------------------------------------------- #
@dataclass
class CapacityReport:
    """A sized machine, plus what limited it."""

    problem: ProblemSpec
    features: Features
    requirements: Requirements
    feasible: bool
    binding: str
    # components
    lasers: int
    detectors: int
    regions: int
    subregions_per_region: int
    slm_pixels: int
    slm_pixels_x: int
    slm_pixels_y: int
    slm_extent: float
    slm_extent_x: float
    slm_extent_y: float
    slm_pitch: float
    panel_cols: int
    panel_rows: int
    screen_cols: int
    screen_rows: int
    screen_extent_x: float
    screen_extent_y: float
    region_pitch: float
    subaperture_px: float
    region_px: float
    # geometry
    z: float
    out_pitch: float
    region_size: float
    subaperture: float
    spot_size: float
    screen_extent: float
    deflection_needed: float
    deflection_available: float
    pixels_per_subaperture: float
    # budget
    efficiency: float
    optical_power_total: float
    power_per_input: float
    power_per_detector: float
    achievable_bits: float
    error_static: float
    error_budget: float
    # throughput
    update_rate: float
    mac_rate: float
    time_to_solution: float | None
    #: ``(radial, tangential)`` growth of the worst (corner) spot over
    #: ``spot_size``, from :func:`obliquity_stretch` for ``requirements.screen``
    spot_stretch: tuple[float, float] = (1.0, 1.0)
    notes: list[str] = field(default_factory=list)

    @property
    def megapixels(self) -> float:
        return self.slm_pixels / 1e6

    def summary(self) -> dict:
        return {
            "label": self.problem.label,
            "features": self.features.tag,
            "feasible": self.feasible,
            "binding": self.binding,
            "lasers": self.lasers,
            "detectors": self.detectors,
            "regions": self.regions,
            "subregions_per_region": self.subregions_per_region,
            "slm_megapixels": self.megapixels,
            "slm_pixels_x": self.slm_pixels_x,
            "slm_extent_mm": self.slm_extent * 1e3,
            "slm_pitch_um": self.slm_pitch * 1e6,
            "z_mm": self.z * 1e3,
            "out_pitch_um": self.out_pitch * 1e6,
            "region_size_um": self.region_size * 1e6,
            "subaperture_um": self.subaperture * 1e6,
            "spot_size_um": self.spot_size * 1e6,
            "screen": self.requirements.screen,
            "spot_stretch_radial": self.spot_stretch[0],
            "spot_stretch_tangential": self.spot_stretch[1],
            "screen_extent_mm": self.screen_extent * 1e3,
            "deflection_needed_deg": math.degrees(self.deflection_needed),
            "deflection_available_deg": math.degrees(self.deflection_available),
            "pixels_per_subaperture": self.pixels_per_subaperture,
            "efficiency": self.efficiency,
            "optical_power_mw": self.optical_power_total * 1e3,
            "power_per_input_mw": self.power_per_input * 1e3,
            "power_per_detector_uw": self.power_per_detector * 1e6,
            "achievable_bits": self.achievable_bits,
            "target_bits": self.requirements.target_bits,
            "update_rate_hz": self.update_rate,
            "mac_rate_gmacs": self.mac_rate / 1e9,
            "time_to_solution_ms": (None if self.time_to_solution is None
                                    else self.time_to_solution * 1e3),
        }

    def table(self) -> str:
        """Human-readable block, in the same spirit as the other reports."""
        p, r = self.problem, self.requirements
        ok = "feasible" if self.feasible else f"INFEASIBLE ({self.binding})"
        lines = [
            f"capacity: {p.label} {p.n_out}x{p.n_in} "
            f"(fan {p.max_fanout}, density {p.density:.2f})  [{self.features.tag}]",
            f"  {ok}; binding constraint: {self.binding}",
            f"  lasers {self.lasers}   detectors {self.detectors}   "
            f"regions {self.regions} x {self.subregions_per_region} subregions",
            f"  SLM    {self.slm_pixels_x} x {self.slm_pixels_y} px "
            f"({self.megapixels:.2f} Mpx) at {self.slm_pitch*1e6:.2f} um pitch "
            f"= {self.slm_extent_x*1e3:.2f} x {self.slm_extent_y*1e3:.2f} mm",
            f"  throw  z = {self.z*1e3:.1f} mm; screen {self.screen_extent*1e3:.2f} mm "
            f"at {self.out_pitch*1e6:.0f} um pitch",
            f"  optics sub-aperture {self.subaperture*1e6:.1f} um -> spot "
            f"{self.spot_size*1e6:.1f} um "
            f"({self.spot_size/self.out_pitch:.2f} x out_pitch); "
            f"{self.pixels_per_subaperture:.0f} px/sub-aperture",
            f"  screen {r.screen}: corner spot {self.spot_stretch[0]:.2f} x "
            f"{self.spot_stretch[1]:.2f} the axial one (radial x tangential)",
            f"  steer  need {math.degrees(self.deflection_needed):.2f} deg, "
            f"have {math.degrees(self.deflection_available):.2f} deg",
            f"  budget {self.achievable_bits:.1f} bits achievable vs "
            f"{r.target_bits:.1f} requested "
            f"(static error {self.error_static:.4f}, budget {self.error_budget:.4f})",
            f"  power  {self.optical_power_total*1e3:.2f} mW total, "
            f"{self.power_per_input*1e3:.3f} mW/input, "
            f"{self.power_per_detector*1e6:.2f} uW/detector "
            f"(efficiency {self.efficiency:.3f})",
            f"  rate   {self.update_rate:.3g} updates/s -> "
            f"{self.mac_rate/1e6:.1f} MMAC/s"
            + (f"; time to solution {self.time_to_solution*1e3:.1f} ms"
               if self.time_to_solution is not None else ""),
        ]
        lines += [f"  note   {n}" for n in self.notes]
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.table()

    # ---- bill of materials ------------------------------------------------ #
    @property
    def cgh_count(self) -> int:
        """Distinct holograms on the panel: one per sub-region, per region."""
        return self.regions * self.subregions_per_region

    @property
    def pixels_per_cgh(self) -> int:
        """Device pixels one hologram is written on."""
        return int(max(self.subaperture_px, 0.0) ** 2)

    @property
    def pixels_per_region(self) -> int:
        return int(max(self.region_px, 0.0) ** 2)

    def bill_of_materials(self) -> dict:
        """Component counts and sizes, stage by stage.

        What `table()` summarises, itemised: what you would have to buy or
        fabricate, and which modelled feature each stage is exercising. Counts
        are per machine, and every length is in metres (the ``components()``
        renderer converts for reading).

        The collimator row is a lenslet per emitter. :class:`OpticalMVM` builds
        the illumination as a profiled :class:`LaserArray` rather than explicit
        lenslets, so this row is a sizing statement rather than something the
        engine instantiates -- it is what the panel's region pitch demands of
        whatever optic actually delivers the light.
        """
        req, feat = self.requirements, self.features
        dual = feat.encoding == "dual_rail"
        return {
            "source": [
                ("laser / emitter", self.lasers,
                 f"{self.power_per_input*1e3:.3g} mW each at "
                 f"{req.wavelength*1e9:.0f} nm"),
                ("input rails", self.lasers,
                 "dual rail (x+ and x-)" if dual else "unsigned, single rail"),
            ],
            "illumination": [
                ("collimating lenslet", self.lasers,
                 f"pitch {self.region_pitch*1e6:.1f} um, clear aperture "
                 f">= {self.region_size*1e6:.1f} um"),
                ("lenslet array", 1,
                 f"{self.panel_cols} x {self.panel_rows} on the panel grid"),
            ],
            "modulator": [
                ("SLM panel", 1,
                 f"{self.slm_pixels_x} x {self.slm_pixels_y} px "
                 f"({self.megapixels:.2f} Mpx) at {self.slm_pitch*1e6:.2f} um "
                 f"= {self.slm_extent_x*1e3:.2f} x {self.slm_extent_y*1e3:.2f} mm"),
                ("illuminated region", self.regions,
                 f"{self.panel_cols} x {self.panel_rows} grid, "
                 f"{self.region_size*1e6:.1f} um wide, "
                 f"{self.pixels_per_region} px each"),
                ("hologram (CGH)", self.cgh_count,
                 f"{self.subregions_per_region} per region, "
                 f"{self.subaperture_px:.0f} x {self.subaperture_px:.0f} px "
                 f"= {self.pixels_per_cgh} px each"),
                ("routing", "-",
                 f"{feat.routing}"
                 + (f" / {feat.cgh_method}" if feat.routing == "multiplex" else "")),
                ("sub-aperture window", "-", feat.apodization or "none (hard edge)"),
            ],
            "propagation": [
                ("free-space throw", 1, f"z = {self.z*1e3:.1f} mm"),
                ("steering demand", "-",
                 f"{math.degrees(self.deflection_needed):.2f} deg needed, "
                 f"{math.degrees(self.deflection_available):.2f} deg available "
                 f"at {self.slm_pitch*1e6:.2f} um pitch"),
            ],
            "detection": [
                ("detector element", self.detectors,
                 f"{self.screen_cols} x {self.screen_rows} grid at "
                 f"{self.out_pitch*1e6:.0f} um pitch, "
                 f"{req.detector_active_frac*self.out_pitch*1e6:.0f} um active"),
                ("output rails", self.detectors,
                 "dual rail (y+ and y-), subtracted" if dual else "unsigned"),
                ("screen", 1,
                 f"{self.screen_extent_x*1e3:.1f} x "
                 f"{self.screen_extent_y*1e3:.1f} mm, "
                 f"{req.screen}, spot {self.spot_size*1e6:.0f} um "
                 f"({self.spot_size/self.out_pitch:.2f} x pitch) on axis, "
                 f"{self.spot_stretch[0]:.2f} x {self.spot_stretch[1]:.2f} "
                 f"that at the corner"),
            ],
            "electronics": [
                ("update rate", "-", f"{self.update_rate:.3g} Hz"
                 + (f" (time-multiplexed x{feat.time_multiplex})"
                    if feat.time_multiplex > 1 else "")),
                ("effective depth", "-",
                 f"{self.achievable_bits:.1f} bits achievable, "
                 f"{req.target_bits:.1f} requested"),
                ("iterations", self.problem.iterations or "-",
                 (f"time to solution {self.time_to_solution*1e3:.1f} ms"
                  if self.time_to_solution is not None else "single pass")),
            ],
        }

    def components(self) -> str:
        """Render :meth:`bill_of_materials` as a readable block."""
        bom = self.bill_of_materials()
        lines = [f"components: {self.problem.label} "
                 f"{self.problem.n_out}x{self.problem.n_in} [{self.features.tag}]",
                 f"  {'stage':<13}{'component':<22}{'count':>8}  detail"]
        lines.append("  " + "-" * 76)
        for stage, items in bom.items():
            for k, (name, count, detail) in enumerate(items):
                lines.append(f"  {stage if k == 0 else '':<13}{name:<22}"
                             f"{str(count):>8}  {detail}")
        return '\n'.join(lines)

    def to_mvm(self, matrix=None, **kw):
        """Instantiate the :class:`OpticalMVM` this report describes.

        The point is to *check the estimate against the engine*: build it,
        program it, call ``check()``, and the geometry should agree. Nothing else
        in this module touches the optical engine, so this is the seam where a
        sizing claim becomes a measurement.

        Pass ``matrix`` to program it. Do not skip that and then read
        ``check()``: an unprogrammed array still carries one ``FlatMirror``
        sub-region per region, so ``check()`` reports a fan-out of 1 and a
        sub-aperture ``sqrt(fan)`` times too large.
        """
        import dataclasses

        from .solvers.aim import OpticalMVM

        dev = as_device(kw.pop("device", None))
        if dataclasses.is_dataclass(dev):
            # the whole point is the *derived* pitch, not the catalogue one
            dev = replace(dev, pitch=self.slm_pitch)
        params = dict(
            n_in=self.problem.n_in, n_out=self.problem.n_out, z=self.z,
            region_pitch=self.region_size / (1 - self.requirements.inter_input_gap_frac),
            region_size=self.region_size, out_pitch=self.out_pitch,
            wavelength=self.requirements.wavelength,
            device=dev,
            # an infeasible design has no power number; fall back to 1 mW so the
            # geometry can still be checked
            power_per_input=(self.power_per_input
                             if self.power_per_input == self.power_per_input else 1e-3),
            signed_input=(self.features.encoding == "dual_rail"
                          and self.problem.signed_input),
            signed_weights=(self.features.encoding == "dual_rail"
                            and self.problem.signed_weights),
            routing=("multiplex" if self.features.routing == "multiplex"
                     else "superpose" if self.features.routing == "superpose"
                     else "subregions"),
            mode="spot",
        )
        params.update(kw)
        mvm = OpticalMVM(**params)
        if matrix is not None:
            mvm.set_matrix(np.asarray(matrix, dtype=float))
        return mvm


# --------------------------------------------------------------------------- #
# the estimator
# --------------------------------------------------------------------------- #
def _rail_counts(problem: ProblemSpec, features: Features) -> tuple[int, int]:
    dual = features.encoding == "dual_rail"
    nri = problem.n_in * (2 if dual and problem.signed_input else 1)
    nro = problem.n_out * (2 if dual and problem.signed_weights else 1)
    return nri, nro


def _required_power(detector: PhotodiodeArray, snr: float) -> float:
    """Optical power per element for a target current SNR, exact in the PD model.

    Solves ``i_s = snr * sigma(i_s)`` with the detector's own noise terms --
    shot (signal and dark, with APD excess noise), TIA and thermal -- rather
    than the thermal-limited approximation of
    :meth:`PhotodiodeArray.sensitivity_w`, because at the powers a fan-out
    machine runs at the signal shot noise is not negligible.
    """
    from .units import K_B, Q_E

    b = detector.bandwidth if np.isfinite(detector.bandwidth) else 1e12
    f = detector.excess_noise_factor if detector.apd_gain > 1 else 1.0
    shot_coeff = 2 * Q_E * b * (detector.apd_gain * f if detector.apd_gain > 1 else 1.0)
    var0 = (shot_coeff * detector.dark_current
            + detector.tia_input_noise**2 * b
            + 4 * K_B * detector.temperature / max(detector.load_resistance, 1e-9) * b)
    s2 = snr * snr
    # i^2 - s2*shot*i - s2*var0 = 0
    i_s = 0.5 * (s2 * shot_coeff + math.sqrt((s2 * shot_coeff) ** 2 + 4 * s2 * var0))
    return i_s / max(detector.responsivity * detector.apd_gain, 1e-30)


def _evaluate(problem: ProblemSpec, req: Requirements, features: Features,
              device: ModulatorDevice, detector: PhotodiodeArray,
              model: FeatureModel, z: float, out_pitch: float,
              guard: float) -> dict | None:
    """Size one candidate ``(z, out_pitch, guard)``.  None if hard-infeasible."""
    nri, nro = _rail_counts(problem, features)
    fan_total = min(problem.max_fanout, nro)
    fan = int(math.ceil(fan_total / features.time_multiplex))
    fan = max(fan, 1)

    throughput, broadening = model.window(features.apodization)

    # --- transverse geometry ------------------------------------------------ #
    a = broadening * req.wavelength * z / (guard * out_pitch)
    region = a if features.multiplexed else a * math.sqrt(fan)
    region_pitch = region / (1.0 - req.inter_input_gap_frac)
    # Near-square arrangements, but *not* forced square: the last row of a
    # panel is usually short, and a 2:1 panel is a different piece of glass
    # from a square one with the same pixel count.
    panel_cols = int(math.ceil(math.sqrt(nri)))
    panel_rows = int(math.ceil(nri / panel_cols))
    screen_cols = int(math.ceil(math.sqrt(nro)))
    screen_rows = int(math.ceil(nro / screen_cols))
    l_panel_x = panel_cols * region_pitch
    l_panel_y = panel_rows * region_pitch
    l_screen_x = screen_cols * out_pitch
    l_screen_y = screen_rows * out_pitch
    l_panel, l_screen = max(l_panel_x, l_panel_y), max(l_screen_x, l_screen_y)
    if req.max_panel is not None and l_panel > req.max_panel:
        return None
    if req.max_screen is not None and l_screen > req.max_screen:
        return None

    # --- steering and pixel pitch ------------------------------------------- #
    # worst case is a corner region reaching the opposite corner output, which is
    # how OpticalMVM.check() measures it too
    corner = math.hypot((l_panel_x + l_screen_x) / 2.0,
                        (l_panel_y + l_screen_y) / 2.0)
    theta_need = math.atan2(corner, z)
    if theta_need > math.radians(req.max_deflection_deg):
        return None
    # the same corner ray carries the worst spot, stretched along its own
    # direction across the screen
    e_rad, e_tan = obliquity_stretch(theta_need, req.screen)
    cx = (l_panel_x + l_screen_x) / 2.0 / corner if corner > 0 else 1.0
    cy = (l_panel_y + l_screen_y) / 2.0 / corner if corner > 0 else 0.0
    e_x, e_y = _axis_stretch(e_rad, e_tan, cx, cy)
    sin_need = math.sin(theta_need)
    p_steer = req.wavelength / (2.0 * sin_need) if sin_need > 0 else float("inf")
    p_sampling = a / max(req.min_pixels_per_subaperture, 1)
    p = min(p_steer, p_sampling)
    if p < req.min_device_pitch:
        return None
    npx_x = int(math.ceil(l_panel_x / p))
    npx_y = int(math.ceil(l_panel_y / p))
    if npx_x <= 0 or npx_y <= 0:
        return None
    # what one hologram actually occupies on the device
    sub_px = a / p
    region_px = region / p

    # --- error budget ------------------------------------------------------- #
    pen = model.routing_penalty(features, fan, nro)
    eps_budget = 2.0 ** (-req.target_bits)
    # the neighbour sits one out_pitch away; in units of the hard-edged first-null
    # radius lambda*z/a that is out_pitch * a / (lambda z), and the detector
    # collects over its own active width
    u_sep = broadening / guard  # == out_pitch * a / (lambda z), see _crosstalk_ratio
    u_half = 0.5 * req.detector_active_frac * u_sep
    # stretches rounded so the cache still hits across the search plane; 1%
    # of a spot width is far below anything the budget can resolve
    leak, capture = _crosstalk_ratio(features.apodization or "none",
                                     round(u_sep, 9), round(u_half, 9), 3,
                                     round(e_x, 2), round(e_y, 2))
    eps_xt = math.hypot(pen["crosstalk"], leak)
    bits = device.phase_bits
    eps_quant = ((2 * math.pi / (2 ** bits)) / (2 * math.sqrt(3.0))) if bits else 0.0
    eps_static = math.sqrt(eps_xt**2 + eps_quant**2 + pen["uniformity"] ** 2)
    achievable = -math.log2(eps_static) if eps_static > 0 else float("inf")

    # --- power -------------------------------------------------------------- #
    # ``capture`` is the fraction of a spot its own detector actually collects:
    # a real loss, and the one apodization makes worse as it widens the spot
    efficiency = (pen["efficiency"] * throughput * capture * device.efficiency
                  * device.fill_factor * problem.column_balance)
    if eps_static < eps_budget:
        eps_noise = math.sqrt(eps_budget**2 - eps_static**2)
    else:
        # the static floor already blows the budget, so there is no noise
        # allowance left to solve for.  Quote the power that puts shot and TIA
        # noise a factor of three below that floor instead: it is the power this
        # design wants, even though it will not reach the requested bit depth.
        eps_noise = eps_static / 3.0
    p_det = _required_power(detector, 1.0 / max(eps_noise, 1e-12))
    p_in = p_det * fan / max(efficiency, 1e-12)
    p_tot = p_in * nri

    # --- throughput --------------------------------------------------------- #
    settle = 1.0 / device.settling_time if device.settling_time > 0 else float("inf")
    rate = min(device.frame_rate, settle) / features.time_multiplex
    mac = 2.0 * problem.nnz * rate  # one multiply and one add per nonzero
    tts = (problem.iterations / rate) if problem.iterations else None

    # --- what bound it ------------------------------------------------------ #
    if eps_static >= eps_budget:
        binding = "accuracy_floor"
    elif p_steer <= p_sampling:
        binding = "steering_pitch"
    else:
        binding = "pixels_per_subaperture"

    return {
        "feasible": eps_static < eps_budget,
        "binding": binding,
        "z": z, "out_pitch": out_pitch, "guard": guard,
        "lasers": nri, "detectors": nro, "regions": nri,
        "subregions": 1 if features.multiplexed else fan,
        "slm_pixels": npx_x * npx_y, "slm_pixels_x": npx_x,
        "slm_pixels_y": npx_y, "slm_extent": l_panel,
        "slm_extent_x": l_panel_x, "slm_extent_y": l_panel_y,
        "panel_cols": panel_cols, "panel_rows": panel_rows,
        "screen_cols": screen_cols, "screen_rows": screen_rows,
        "screen_extent_x": l_screen_x, "screen_extent_y": l_screen_y,
        "region_pitch": region_pitch,
        "subaperture_px": sub_px, "region_px": region_px,
        "slm_pitch": p,
        "region_size": region, "subaperture": a,
        "spot_size": broadening * req.wavelength * z / a,
        "spot_stretch": (e_rad, e_tan),
        "screen_extent": l_screen,
        "theta_need": theta_need,
        "theta_have": float(np.arcsin(min(1.0, req.wavelength / (2 * p)))),
        "pixels_per_subaperture": a / p,
        "efficiency": efficiency,
        "power_total": p_tot, "power_in": p_in, "power_det": p_det,
        "achievable_bits": achievable, "eps_static": eps_static,
        "eps_budget": eps_budget, "leak": leak, "capture": capture,
        "eps_quant": eps_quant,
        "rate": rate, "mac": mac, "tts": tts,
        "fan": fan, "penalty": pen,
    }


def estimate(problem: ProblemSpec, requirements: Requirements | None = None,
             features: Features | None = None, device=None, detector=None,
             model: FeatureModel | None = None,
             measure: bool = False) -> CapacityReport:
    """Size the machine for ``problem``, searching ``z`` and ``out_pitch``.

    ``measure=True`` replaces the modelled routing penalty with one obtained by
    actually synthesising the hologram (see :func:`measure_routing`) at the
    chosen design point, then re-solves the budget with it.  That costs a couple
    of FFTs per call and is what turns a sketch into a defensible number.
    """
    req = requirements or Requirements()
    feat = features or Features.baseline()
    dev = as_device(device)
    det = detector or PhotodiodeArray(bandwidth=req.detector_bandwidth, output="watts",
                                      n_time=1)
    mdl = model or FeatureModel()

    zs = ([req.z] if req.z is not None
          else list(np.geomspace(req.min_z, req.max_z, max(req.search_points // 3, 8))))
    ps = ([req.out_pitch] if req.out_pitch is not None
          else list(np.geomspace(req.min_out_pitch, req.max_out_pitch,
                                 req.search_points)))

    primary = {"slm_pixels": lambda c: c["slm_pixels"],
               "panel_area": lambda c: c["slm_extent"] ** 2,
               "optical_power": lambda c: (c["power_total"] if c["power_total"] ==
                                           c["power_total"]
                                           else float("inf"))}[req.objective]

    def key(c):
        # When ``pixels_per_subaperture`` binds, the pixel count works out to
        # sqrt(regions)*sqrt(fan)*K/(1-gap) -- independent of z, out_pitch and
        # guard, so the objective ties across the whole search plane. Break the
        # tie towards the design that is easier to actually build: the coarsest
        # device pitch, which is also the smallest deflection.
        return (primary(c), -c["slm_pitch"])

    gs = ([req.spot_guard] if req.spot_guard is not None
          else list(np.linspace(req.min_spot_guard, req.max_spot_guard,
                                req.guard_points)))

    best = None
    best_any = None
    for g in gs:
        for z in zs:
            for op in ps:
                c = _evaluate(problem, req, feat, dev, det, mdl, float(z), float(op),
                              float(g))
                if c is None:
                    continue
                # the fallback keeps the best *achievable bit depth* rather than the
                # smallest panel: when nothing meets the target, how close it got is
                # the useful answer
                if best_any is None or (c["achievable_bits"],
                                        -c["slm_pixels"]) > (best_any["achievable_bits"],
                                                             -best_any["slm_pixels"]):
                    best_any = c
                if c["feasible"] and (best is None or key(c) < key(best)):
                    best = c
    chosen = best or best_any
    if chosen is None:
        raise ValueError(
            "no geometry satisfied the hard limits; relax max_panel/max_screen, "
            "min_device_pitch, or widen the z / out_pitch search range")

    notes: list[str] = []
    blocking = feat.validate(problem)
    notes.extend(blocking)
    if measure:
        _, broad = mdl.window(feat.apodization)
        mdl = mdl.measured(feat, chosen["fan"], chosen["subaperture"],
                           req.wavelength, chosen["z"], chosen["out_pitch"],
                           u_sep=broad / chosen["guard"])
        re_solved = _evaluate(problem, req, feat, dev, det, mdl,
                              chosen["z"], chosen["out_pitch"], chosen["guard"])
        if re_solved is not None:
            chosen = re_solved
        notes.append("routing penalty measured by hologram synthesis, not modelled")
    elif chosen["penalty"].get("source") == "model":
        notes.append("routing penalty is a first-order model; pass measure=True "
                     "to synthesise the hologram instead")
    if req.screen == "ideal":
        fr, ft = obliquity_stretch(chosen["theta_need"], "flat")
        notes.append(
            f"screen='ideal' assumes a uniform lambda z / a spot; a flat detector "
            f"array stretches the corner one {fr:.2f} x {ft:.2f} -- size with "
            f"screen='flat' (or 'curved') for the hardware you would build")

    if not chosen["feasible"]:
        notes.append(
            f"static error {chosen['eps_static']:.4f} exceeds the "
            f"{chosen['eps_budget']:.4f} budget: {chosen['achievable_bits']:.1f} bits "
            f"is the ceiling at any optical power")
    if problem.dynamic_range_bits > req.target_bits:
        notes.append(
            f"the matrix spans {problem.dynamic_range_bits:.1f} bits of dynamic range "
            f"but only {req.target_bits:.1f} are budgeted; small weights will be lost")
    if problem.column_balance < 0.5:
        notes.append(
            f"column sums are unbalanced (balance {problem.column_balance:.2f}); "
            "normalising columns throws away most of the light -- consider "
            "normalize_columns=False and reading the true map back")
    if feat.multiplexed and chosen["fan"] > 64:
        notes.append(f"fan-out {chosen['fan']} is a lot for one phase-only hologram; "
                     "the speckle term is the least trustworthy part of this estimate")

    return CapacityReport(
        problem=problem, features=feat, requirements=req,
        feasible=chosen["feasible"] and not blocking,
        binding="encoding_mismatch" if blocking else chosen["binding"],
        lasers=chosen["lasers"], detectors=chosen["detectors"],
        regions=chosen["regions"], subregions_per_region=chosen["subregions"],
        slm_pixels=chosen["slm_pixels"], slm_pixels_x=chosen["slm_pixels_x"],
        slm_pixels_y=chosen["slm_pixels_y"],
        slm_extent=chosen["slm_extent"], slm_pitch=chosen["slm_pitch"],
        slm_extent_x=chosen["slm_extent_x"], slm_extent_y=chosen["slm_extent_y"],
        panel_cols=chosen["panel_cols"], panel_rows=chosen["panel_rows"],
        screen_cols=chosen["screen_cols"], screen_rows=chosen["screen_rows"],
        screen_extent_x=chosen["screen_extent_x"],
        screen_extent_y=chosen["screen_extent_y"],
        region_pitch=chosen["region_pitch"],
        subaperture_px=chosen["subaperture_px"], region_px=chosen["region_px"],
        z=chosen["z"], out_pitch=chosen["out_pitch"],
        region_size=chosen["region_size"], subaperture=chosen["subaperture"],
        spot_size=chosen["spot_size"], screen_extent=chosen["screen_extent"],
        deflection_needed=chosen["theta_need"],
        deflection_available=chosen["theta_have"],
        pixels_per_subaperture=chosen["pixels_per_subaperture"],
        efficiency=chosen["efficiency"],
        optical_power_total=chosen["power_total"],
        power_per_input=chosen["power_in"],
        power_per_detector=chosen["power_det"],
        achievable_bits=chosen["achievable_bits"],
        error_static=chosen["eps_static"], error_budget=chosen["eps_budget"],
        update_rate=chosen["rate"], mac_rate=chosen["mac"],
        time_to_solution=chosen["tts"], spot_stretch=chosen["spot_stretch"],
        notes=notes,
    )


def baseline(problem: ProblemSpec, requirements: Requirements | None = None,
             **kw) -> CapacityReport:
    """Size the conservative reference design: dual rail, area division, no tricks.

    This is the number every feature has to beat, and the denominator in
    :func:`compare`.
    """
    return estimate(problem, requirements, Features.baseline(), **kw)


def compare(problem: ProblemSpec, variants, requirements: Requirements | None = None,
            **kw) -> dict:
    """Price a list of :class:`Features` against the baseline.

    Returns ``{"baseline": report, "variants": [(features, report, deltas)],
    "table": str}`` where the deltas are ratios against the baseline, so a
    ``slm_pixels`` of 0.05 means the variant needs one twentieth of the panel.
    """
    base = baseline(problem, requirements, **kw)
    rows = []
    for v in variants:
        rep = estimate(problem, requirements, v, **kw)
        deltas = {
            "slm_pixels": rep.slm_pixels / max(base.slm_pixels, 1),
            "slm_extent": rep.slm_extent / max(base.slm_extent, 1e-30),
            "optical_power": (rep.optical_power_total / base.optical_power_total
                              if base.optical_power_total == base.optical_power_total
                              and rep.optical_power_total == rep.optical_power_total
                              and base.optical_power_total > 0 else float("nan")),
            "achievable_bits": rep.achievable_bits - base.achievable_bits,
            "update_rate": rep.update_rate / max(base.update_rate, 1e-30),
        }
        rows.append((v, rep, deltas))

    head = (f"{'features':<44}{'Mpx':>9}{'x base':>9}{'bits':>7}"
            f"{'power mW':>11}{'rate Hz':>11}  binding")
    lines = [f"capacity comparison: {problem.label} "
             f"{problem.n_out}x{problem.n_in}, fan {problem.max_fanout}", head,
             "-" * len(head)]

    def row(tag, rep, ratio):
        pw = (f"{rep.optical_power_total*1e3:>11.2f}"
              if rep.optical_power_total == rep.optical_power_total else f"{'--':>11}")
        return (f"{tag:<44}{rep.megapixels:>9.2f}{ratio:>9.2f}"
                f"{rep.achievable_bits:>7.1f}{pw}{rep.update_rate:>11.0f}  "
                f"{rep.binding}{'' if rep.feasible else ' !'}")

    lines.append(row(base.features.tag + "  (baseline)", base, 1.0))
    for v, rep, d in rows:
        lines.append(row(v.tag, rep, d["slm_pixels"]))
    lines.append("")
    lines.append("'!' marks a design the accuracy floor rules out at any power.")
    return {"baseline": base, "variants": rows, "table": "\n".join(lines)}
