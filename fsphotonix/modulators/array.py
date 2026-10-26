"""The modulator plane: an array of illuminated regions, each carrying holograms.

Structure, mirroring the physical device:

``ModulatorArray``
    the panel.  Owns a :class:`~fsphotonix.modulators.devices.ModulatorDevice`
    (LCoS, DMD, MEMS piston, ...) and a list of regions.
``Region``
    one illuminated patch of the panel, with its own aperture geometry.
``SubRegion``
    a piece of a region carrying one grating.  A region with several sub-regions
    fans its light out to several screen locations at once; a region with one
    sub-region holding a :class:`~fsphotonix.modulators.gratings.cgh.MultiSpotHologram`
    does the same job by phase multiplexing instead of area division.

Everything is addressable at run time (``set_targets``, ``set_weights``,
``set_pistons``, ``set_enabled``), which is what the feedback loops actuate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from .. import engine as eng
from ..fields import Channel, FieldSet
from ..grid import Grid
from ..registry import register
from .devices import ModulatorDevice, as_device
from .geometry import Aperture, Rect, apodize_aperture, as_aperture, tile_region
from .gratings.base import Grating, Order, as_grating
from .gratings.cgh import MultiSpotHologram, SpotTarget
from .gratings.lenses import SpotFocuser
from .gratings.periodic import BlazedGrating, FlatMirror


@dataclass
class SubRegion:
    """One grating written on part of a region."""

    grating: Grating | None = None
    aperture: Aperture | None = None  # None -> inherit the region aperture
    offset: tuple[float, float] = (0.0, 0.0)
    weight: float = 1.0  # relative power share, see ModulatorArray.weight_mode
    enabled: bool = True
    label: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.grating = as_grating(self.grating)
        if self.aperture is not None:
            self.aperture = as_aperture(self.aperture)


@dataclass
class Region:
    """One illuminated patch of the modulator."""

    center: tuple[float, float] = (0.0, 0.0)
    aperture: Aperture | None = None
    subregions: list[SubRegion] = field(default_factory=list)
    piston: float = 0.0  # uniform phase offset [rad] -- a cheap analog actuator
    gain: float = 1.0  # amplitude scale in [0, 1]
    enabled: bool = True
    label: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.aperture = as_aperture(self.aperture if self.aperture is not None else Rect())
        if not self.subregions:
            self.subregions = [SubRegion(grating=FlatMirror())]

    @property
    def size(self) -> tuple[float, float]:
        return self.aperture.size

    def area(self) -> float:
        return self.aperture.area()

    def orders(self, wavelength: float, **kw) -> list[tuple[SubRegion, Order]]:
        """All diffraction orders of all enabled sub-regions."""
        out = []
        total_w = sum(s.weight for s in self.subregions if s.enabled) or 1.0
        for s in self.subregions:
            if not s.enabled or s.grating is None:
                continue
            share = s.weight / total_w
            for o in s.grating.orders(wavelength, **kw):
                out.append((s, Order(o.mx, o.my, o.fx, o.fy, o.efficiency * share)))
        return out


class ModulatorArray:
    """A programmable diffractive plane.

    Parameters
    ----------
    device : hardware model (plugin kind ``device``).
    regions : list of :class:`Region`.
    background : what the un-patterned panel does -- ``'absorb'`` (dark),
        ``'reflect'`` (specular, which is realistic and puts a bright zero order
        on the screen) or ``'transparent'``.
    weight_mode : how ``SubRegion.weight`` is realised.  ``'area'`` scales the
        sub-region's aperture (the only honest option for a phase-only device);
        ``'amplitude'`` scales the field directly (needs an amplitude device);
        ``'none'`` ignores weights.
    """

    def __init__(
        self,
        device=None,
        regions: Sequence[Region] | None = None,
        background: str = "absorb",
        weight_mode: str = "area",
        name: str = "modulator",
        default_wavelength: float = 1550e-9,
    ):
        self.device: ModulatorDevice = as_device(device)
        self.regions: list[Region] = list(regions or [])
        self.background = background
        self.weight_mode = weight_mode
        self.name = name
        self.default_wavelength = default_wavelength
        #: filled in by :meth:`tiled`; ``None`` for irregular layouts
        self.region_pitch: tuple[float, float] | None = None
        self.region_gap: tuple[float, float] | None = None
        #: guard gap last used between the sub-regions of one input
        self.subregion_gap: float = 0.0
        #: apodization window last applied to the sub-apertures, if any
        self.subregion_apodization: str | None = None
        self._version = 0
        self._cache: dict = {}

    # ---- construction ------------------------------------------------------ #
    @classmethod
    def tiled(
        cls,
        nx: int,
        ny: int = 1,
        pitch: float = 1e-3,
        pitch_y: float | None = None,
        region_size: float | None = None,
        gap: float | None = None,
        shape: str = "rect",
        device=None,
        x0: float = 0.0,
        y0: float = 0.0,
        **kw,
    ) -> "ModulatorArray":
        """Regular ``nx x ny`` grid of identical regions.

        ``gap`` is the guard band left *between* neighbouring regions, i.e. between
        the holograms belonging to different inputs: ``region_size = pitch - gap``.
        It is the knob that trades panel fill (and therefore optical throughput and
        sub-aperture size) against tolerance to illumination overspill from the
        neighbouring input.  Give either ``gap`` or ``region_size``, not both.
        """
        py = pitch if pitch_y is None else pitch_y
        if gap is not None:
            if region_size is not None:
                raise ValueError("give either gap or region_size, not both")
            region_size = min(pitch, py) - gap
            if region_size <= 0:
                raise ValueError(f"gap {gap} leaves no room inside pitch {min(pitch, py)}")
        rs = region_size if region_size is not None else 0.9 * min(pitch, py)
        ap = as_aperture({"type": shape, **({"diameter": rs} if shape in
                                            ("circle", "circular", "disk") else
                                            {"width": rs, "height": rs})})
        regions = []
        for j in range(ny):
            for i in range(nx):
                cx = (i - (nx - 1) / 2) * pitch + x0
                cy = (j - (ny - 1) / 2) * py + y0
                regions.append(Region(center=(cx, cy), aperture=ap,
                                      label=f"r{j*nx+i}", meta={"i": i, "j": j,
                                                                "index": j * nx + i}))
        arr = cls(device=device, regions=regions, **kw)
        arr.region_pitch = (pitch, py)
        arr.region_gap = (pitch - rs, py - rs)
        return arr

    @classmethod
    def from_positions(cls, positions, region_size: float = 0.5e-3,
                       shape: str = "rect", device=None, **kw) -> "ModulatorArray":
        """Regions centred on arbitrary ``(N, 2)`` positions."""
        pos = np.asarray(positions, dtype=float).reshape(-1, 2)
        ap = as_aperture({"type": shape, **({"diameter": region_size} if shape in
                                           ("circle", "circular", "disk") else
                                           {"width": region_size, "height": region_size})})
        regions = [Region(center=(float(p[0]), float(p[1])), aperture=ap,
                          label=f"r{i}", meta={"index": i}) for i, p in enumerate(pos)]
        return cls(device=device, regions=regions, **kw)

    def touch(self) -> None:
        """Invalidate the rasterisation cache (call after any mutation)."""
        self._version += 1
        self._cache.clear()

    # ---- addressing -------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.regions)

    def __getitem__(self, key) -> Region:
        if isinstance(key, str):
            for r in self.regions:
                if r.label == key:
                    return r
            raise KeyError(key)
        return self.regions[key]

    def add_region(self, region: Region) -> Region:
        self.regions.append(region)
        self.touch()
        return region

    # ---- programming the holograms ----------------------------------------- #
    def route(
        self,
        region: int | str | Region,
        targets: Sequence,
        z: float,
        wavelength: float | None = None,
        mode: str = "subregions",
        grating: str = "focuser",
        tiling: str = "grid",
        weights: Sequence[float] | None = None,
        gap: float = 0.0,
        apodize: str | None = None,
        apodize_mode: str = "auto",
        absolute: bool = True,
        incident=None,
        **grating_kw,
    ):
        """Point one region's light at a list of screen locations.

        ``targets`` are ``(x, y)`` (or ``(x, y, weight)``) screen coordinates at
        range ``z``.  They are **absolute screen coordinates** by default; the
        required deflection is worked out from each sub-region's own position on
        the panel.  Pass ``absolute=False`` to give offsets relative to the
        sub-region instead.  ``mode``:

        ``subregions``   divide the region into one sub-aperture per target
                         (spatial multiplexing; clean, area-limited)
        ``multiplex``    one sub-region carrying a phase-multiplexed CGH over all
                         targets (area-efficient, noisier)
        ``superpose``    complex superposition of individual gratings, projected
                         to phase-only

        ``grating`` picks the per-target element: ``focuser`` (off-axis focusing
        hologram, spot size set by the sub-aperture), ``blazed``, ``ramp``.

        ``apodize`` names an amplitude window (``hann``, ``blackman``, ``tukey``)
        that tapers each sub-aperture to zero at its rim.  A hard-edged
        sub-aperture has ``sinc`` far-field wings falling only as ``1/u^2``,
        which across a lattice of driven detectors sets a cross-talk floor that
        no guard band escapes; a window replaces those wings with a much faster
        roll-off.  It is the alternative to ``gap``, not an addition to it: the
        gap removes the shared-edge discontinuity by leaving dead space, the
        window by making the field vanish there.  Both cost -- the gap shrinks
        the sub-aperture, the window passes less light (Hann: 0.141 on a square
        aperture) and widens every spot by about 1.6x, which eats the detector
        guard from the other side.  :func:`fsphotonix.analysis.apodization_sweep`
        measures which one wins for a given layout.

        ``incident`` is the direction cosine pair ``(alpha, beta)`` of the light
        arriving at this region.  A thin diffractive element *adds* its grating
        vector to the incident one, so a hologram designed for normal incidence
        mis-points by roughly ``z * alpha`` when fed off-axis -- which is exactly
        what happens to the second plane of a cascade.  Passing ``incident``
        pre-compensates the design; :func:`incidence_from` computes it from the
        previous stage's geometry.
        """
        r = self[region] if not isinstance(region, Region) else region
        lam = wavelength or self.default_wavelength
        tg = [_coerce_target(t, z) for t in targets]
        if weights is not None:
            tg = [SpotTarget(t.x, t.y, t.z, float(w), t.label) for t, w in zip(tg, weights)]

        self.subregion_gap = float(gap)
        self.subregion_apodization = apodize
        if mode == "subregions":
            subs = tile_region(r.aperture, len(tg), mode=tiling, gap=gap,
                               apodize=apodize, apodize_mode=apodize_mode)
            r.subregions = []
            for (ap, off), t in zip(subs, tg):
                origin = (r.center[0] + off[0], r.center[1] + off[1]) if absolute else (0.0, 0.0)
                g = _compensate(_make_grating(grating, _rebase(t, origin), lam,
                                              **grating_kw), incident, lam)
                r.subregions.append(SubRegion(grating=g, aperture=ap, offset=off,
                                              weight=t.weight, label=t.label,
                                              meta={"target": (t.x, t.y, t.z)}))
        elif mode == "multiplex":
            w, h = r.aperture.size
            origin = r.center if absolute else (0.0, 0.0)
            g = _compensate(
                MultiSpotHologram(targets=[_rebase(t, origin) for t in tg], range_z=z,
                                  design_wavelength=lam, aperture_size=(w, h),
                                  **grating_kw), incident, lam)
            # one sub-region using the whole region, so the window goes on the
            # region aperture itself
            ap = (apodize_aperture(r.aperture, apodize, apodize_mode)
                  if apodize else None)
            r.subregions = [SubRegion(grating=g, aperture=ap, label="cgh",
                                      meta={"targets": [(t.x, t.y, t.z) for t in tg]})]
        elif mode == "superpose":
            from .gratings.composite import Superposition

            origin = r.center if absolute else (0.0, 0.0)
            gs = [_make_grating(grating, _rebase(t, origin), lam, **grating_kw) for t in tg]
            # the measured orders are only meaningful on the aperture the
            # hologram is actually written on, so hand it the region's size
            g = _compensate(Superposition(gratings=gs, weights=[t.weight for t in tg],
                                          aperture_size=r.aperture.size,
                                          design_wavelength=lam), incident, lam)
            ap = (apodize_aperture(r.aperture, apodize, apodize_mode)
                  if apodize else None)
            r.subregions = [SubRegion(grating=g, aperture=ap, label="superposed",
                                      meta={"targets": [(t.x, t.y, t.z) for t in tg]})]
        else:
            raise ValueError(f"unknown routing mode {mode!r}")
        self.touch()
        return r

    def route_all(self, targets_per_region: Sequence[Sequence], z: float, **kw):
        """Vectorised :meth:`route` over every region."""
        for r, tg in zip(self.regions, targets_per_region):
            self.route(r, tg, z, **kw)
        return self

    def broadcast(self, targets: Sequence, z: float, **kw):
        """Point *every* region at the same list of screen locations.

        The fan-in/fan-out pattern of an optical matrix-vector product: region i
        carries input i, and every region illuminates every output.
        """
        for r in self.regions:
            self.route(r, targets, z, **kw)
        return self

    def set_matrix(self, matrix, output_positions, z: float, mode: str = "subregions",
                   normalize_columns: bool = True, **kw):
        """Program an ``M x N`` weight matrix.

        Region ``n`` (holding input ``n``) sends power ``|matrix[m, n]|`` to output
        ``m``.  Signed matrices need a dual-rail detector arrangement -- see
        :class:`fsphotonix.solvers.aim.OpticalMVM`, which builds that for you.

        A hologram can only *redistribute* the light its region receives, so the
        sub-region weights are shares and every column is intrinsically normalised
        to the region's incident power.  With ``normalize_columns=True`` the column
        sums are restored by attenuating the regions (``Region.gain``), so the
        programmed map really is ``matrix`` up to one global scale; the price is the
        light thrown away in every column that is weaker than the strongest one.
        Set it to ``False`` to keep full throughput and accept a column-normalised
        matrix (then read the true map back with
        :func:`fsphotonix.analysis.transfer_matrix`).
        """
        W = np.asarray(matrix, dtype=float)
        m, n = W.shape
        if n != len(self.regions):
            raise ValueError(f"matrix has {n} columns but there are {len(self.regions)} regions")
        pos = np.asarray(output_positions, dtype=float).reshape(-1, 2)
        if len(pos) != m:
            raise ValueError(f"matrix has {m} rows but {len(pos)} output positions")
        col_sums = np.abs(W).sum(axis=0)
        cmax = col_sums.max() if col_sums.size else 1.0
        for j, r in enumerate(self.regions):
            col = np.abs(W[:, j])
            keep = np.nonzero(col > 0)[0]
            if keep.size == 0:
                r.subregions = [SubRegion(grating=FlatMirror(reflectance=0.0), label="off")]
                r.gain = 0.0
                continue
            tg = [SpotTarget(float(pos[i, 0]), float(pos[i, 1]), z, float(col[i]),
                             f"out{i}") for i in keep]
            self.route(r, tg, z, mode=mode, **kw)
            if normalize_columns and cmax > 0:
                # gain acts on amplitude, so power scales as gain^2
                r.gain = float(np.sqrt(col_sums[j] / cmax))
        self.touch()
        return self

    # ---- run-time actuation ------------------------------------------------ #
    def set_pistons(self, values) -> None:
        """Per-region uniform phase [rad] -- the fastest analog actuator."""
        v = np.atleast_1d(np.asarray(values, dtype=float)).ravel()
        if v.size == 1:
            v = np.full(len(self.regions), v[0])
        for r, x in zip(self.regions, v):
            r.piston = float(x)
        self.touch()

    def get_pistons(self) -> np.ndarray:
        return np.array([r.piston for r in self.regions], dtype=float)

    def set_gains(self, values) -> None:
        """Per-region amplitude scale in [0, 1]."""
        v = np.atleast_1d(np.asarray(values, dtype=float)).ravel()
        if v.size == 1:
            v = np.full(len(self.regions), v[0])
        for r, x in zip(self.regions, v):
            r.gain = float(np.clip(x, 0.0, 1.0))
        self.touch()

    def get_gains(self) -> np.ndarray:
        return np.array([r.gain for r in self.regions], dtype=float)

    def set_enabled(self, mask) -> None:
        m = np.atleast_1d(np.asarray(mask)).ravel()
        for r, x in zip(self.regions, m):
            r.enabled = bool(x)
        self.touch()

    def set_weights(self, region, weights) -> None:
        """Per-sub-region power share inside one region."""
        r = self[region] if not isinstance(region, Region) else region
        w = np.atleast_1d(np.asarray(weights, dtype=float)).ravel()
        for s, x in zip(r.subregions, w):
            s.weight = float(max(x, 0.0))
        self.touch()

    def set_all_weights(self, matrix) -> None:
        """``set_weights`` for every region at once (row per region)."""
        for r, row in zip(self.regions, matrix):
            self.set_weights(r, row)

    # ---- rasterisation ----------------------------------------------------- #
    def ideal_transmittance(self, grid: Grid, wavelength: float):
        """The complex map the holograms *ask* for, before hardware degradation."""
        bg = {"absorb": 0.0, "reflect": 1.0, "transparent": 1.0}.get(self.background)
        if bg is None:
            raise ValueError(f"unknown background {self.background!r}")
        num = eng.xp.full(grid.shape, bg, dtype=eng.COMPLEX)
        den = eng.xp.zeros(grid.shape, dtype=eng.REAL)
        X, Y = grid.meshgrid()

        for r in self.regions:
            if not r.enabled or r.gain <= 0:
                continue
            rmask = r.aperture.mask(grid, r.center[0], r.center[1])
            if float(eng.to_numpy(eng.xp.max(rmask))) <= 0:
                continue  # region outside this grid
            active = [s for s in r.subregions if s.enabled and s.grating is not None]
            if not active:
                continue
            tot_w = sum(s.weight for s in active) or 1.0
            acc = eng.xp.zeros(grid.shape, dtype=eng.COMPLEX)
            wacc = eng.xp.zeros(grid.shape, dtype=eng.REAL)
            for s in active:
                cx = r.center[0] + s.offset[0]
                cy = r.center[1] + s.offset[1]
                ap = s.aperture
                if ap is None:
                    smask = eng.xp.ones(grid.shape, dtype=eng.REAL)
                else:
                    smask = _scaled_mask(ap, grid, cx, cy, s.weight, tot_w,
                                         self.weight_mode, len(active))
                amp = 1.0
                if self.weight_mode == "amplitude":
                    amp = np.sqrt(max(s.weight, 0.0) / tot_w * len(active))
                t = s.grating.complex_transmittance(X - cx, Y - cy, wavelength)
                acc = acc + smask * amp * t
                wacc = wacc + smask
            # overlapping sub-apertures average rather than interfere
            norm = eng.xp.maximum(wacc, 1.0)
            region_t = acc / norm * r.gain * np.exp(1j * r.piston)
            num = num * (1 - rmask) + rmask * region_t
            den = den + rmask
        return num

    def transmittance(self, grid: Grid, wavelength: float | None = None):
        """Hardware-accurate complex transmittance (cached per grid/wavelength)."""
        lam = wavelength or self.default_wavelength
        key = (id(grid), grid.shape, grid.dx, grid.dy, grid.x0, grid.y0,
               round(lam, 18), self._version)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        t = self.device.render(self.ideal_transmittance(grid, lam), grid, lam)
        if len(self._cache) > 32:
            self._cache.clear()
        self._cache[key] = t
        return t

    # ---- optical action ---------------------------------------------------- #
    def apply(self, fields: FieldSet) -> FieldSet:
        """Multiply an incident field set by this plane's transmittance."""
        grid = fields.grid
        return fields.multiply(lambda lam: self.transmittance(grid, lam))

    def __call__(self, fields: FieldSet) -> FieldSet:
        return self.apply(fields)

    # ---- analysis ---------------------------------------------------------- #
    def region_orders(self, wavelength: float | None = None, **kw):
        """``[(region, subregion, order), ...]`` for the fast spot propagator."""
        lam = wavelength or self.default_wavelength
        out = []
        for r in self.regions:
            if not r.enabled or r.gain <= 0:
                continue
            for s, o in r.orders(lam, **kw):
                out.append((r, s, o))
        return out

    def gap_report(self, z: float | None = None, wavelength: float | None = None) -> dict:
        """Guard-band geometry: the two gaps that set cross-talk on this panel.

        *Between inputs* -- the gap between neighbouring regions.  Illumination that
        overspills it is picked up by the next input's hologram and lands at that
        input's screen targets, so this gap is the panel's defence against
        illumination cross-talk.

        *Within one input* -- the gap between the sub-region holograms of a single
        region.  It costs aperture (and therefore efficiency) but suppresses the
        cross-modulation between abutting sub-gratings, whose phase discontinuity at
        a shared edge scatters light into unintended orders.

        Both gaps also fix the sub-aperture size, and hence the spot size
        ``lambda*z/a`` on the screen: a bigger guard band means a blurrier spot, so
        the gap cannot be increased without eventually re-introducing cross-talk at
        the *detector*.  The returned ``spot_vs_target_pitch`` is that trade-off in
        one number.
        """
        lam = wavelength or self.default_wavelength
        centers = np.array([r.center for r in self.regions]) if self.regions else np.zeros((1, 2))
        sizes = np.array([r.aperture.size for r in self.regions]) if self.regions \
            else np.zeros((1, 2))
        nn = _nearest_neighbour_distance(centers)
        edge_gap = float(np.min(nn - sizes[:, 0].mean())) if len(centers) > 1 else float("inf")

        sub_sizes, sub_gaps, target_pitch = [], [], []
        for r in self.regions:
            offs = np.array([s.offset for s in r.subregions], dtype=float)
            for s in r.subregions:
                ap = s.aperture if s.aperture is not None else r.aperture
                sub_sizes.append(min(ap.size))
            if len(offs) > 1:
                d = _nearest_neighbour_distance(offs)
                sub_gaps.append(float(np.min(d)) - float(np.mean(sub_sizes[-len(offs):])))
            tg = [s.meta.get("target") for s in r.subregions if s.meta.get("target")]
            if len(tg) > 1:
                p = np.array([[t[0], t[1]] for t in tg], dtype=float)
                target_pitch.append(float(np.min(_nearest_neighbour_distance(p))))
        a = float(np.mean(sub_sizes)) if sub_sizes else float(sizes.min())
        spot = lam * (z or 0.0) / max(a, 1e-12)
        tp = float(np.mean(target_pitch)) if target_pitch else float("nan")
        return {
            "regions": len(self.regions),
            "region_size_um": (float(sizes[:, 0].mean() * 1e6), float(sizes[:, 1].mean() * 1e6)),
            "region_pitch_um": (None if self.region_pitch is None
                                else tuple(p * 1e6 for p in self.region_pitch)),
            "inter_input_gap_um": edge_gap * 1e6,
            "inter_input_gap_declared_um": (None if self.region_gap is None
                                            else tuple(g * 1e6 for g in self.region_gap)),
            "subregions_per_region": [len(r.subregions) for r in self.regions][:8],
            "subaperture_um": a * 1e6,
            "intra_input_gap_um": (float(np.mean(sub_gaps)) * 1e6 if sub_gaps
                                  else self.subregion_gap * 1e6),
            "apodization": self.subregion_apodization,
            "pixels_per_subaperture": a / self.device.pitch,
            "spot_size_um": spot * 1e6,
            "target_pitch_um": tp * 1e6,
            "spot_vs_target_pitch": spot / tp if tp == tp and tp > 0 else float("nan"),
        }

    def fill_fraction(self, grid: Grid) -> float:
        """Fraction of the grid covered by enabled regions."""
        tot = eng.xp.zeros(grid.shape, dtype=eng.REAL)
        for r in self.regions:
            if r.enabled:
                tot = eng.xp.maximum(tot, r.aperture.mask(grid, *r.center))
        return float(eng.to_numpy(eng.xp.mean(tot)))

    def summary(self) -> dict:
        return {
            "name": self.name,
            "regions": len(self.regions),
            "subregions": sum(len(r.subregions) for r in self.regions),
            "background": self.background,
            "weight_mode": self.weight_mode,
            "device": self.device.info(),
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"ModulatorArray({self.name!r}, {len(self.regions)} regions, "
                f"{sum(len(r.subregions) for r in self.regions)} subregions, "
                f"{self.device!r})")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _coerce_target(t, z: float) -> SpotTarget:
    if isinstance(t, SpotTarget):
        return t
    v = list(t)
    if len(v) == 2:
        return SpotTarget(float(v[0]), float(v[1]), z, 1.0)
    if len(v) == 3:
        return SpotTarget(float(v[0]), float(v[1]), z, float(v[2]))
    return SpotTarget(float(v[0]), float(v[1]), float(v[2]), float(v[3]))


def incidence_from(source: Sequence[float], target_center: Sequence[float],
                   z: float) -> tuple[float, float]:
    """Direction cosines of a ray from ``source`` (x, y) to ``target_center`` over ``z``.

    Use it to tell a downstream plane what angle its light arrives at::

        a = incidence_from(stage1_region_center, stage2_region_center, gap_z)
        stage2.route(region, targets, z=z2, incident=a)
    """
    dx = float(target_center[0]) - float(source[0])
    dy = float(target_center[1]) - float(source[1])
    r = float(np.sqrt(dx * dx + dy * dy + z * z))
    return (dx / r, dy / r) if r > 0 else (0.0, 0.0)


def _compensate(grating: Grating, incident, lam: float) -> Grating:
    """Subtract the incident carrier so the design points where it was asked to."""
    if incident is None:
        return grating
    a, b = float(incident[0]), float(incident[1])
    if a == 0.0 and b == 0.0:
        return grating
    from .gratings.composite import Cascade
    from .gratings.periodic import PhaseRamp

    tilt = PhaseRamp(fx=-a / lam, fy=-b / lam, design_wavelength=lam)
    return Cascade(gratings=[grating, tilt], design_wavelength=lam)


def _nearest_neighbour_distance(points: np.ndarray) -> np.ndarray:
    """Distance from each point to its closest neighbour."""
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(p) < 2:
        return np.array([np.inf])
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return d.min(axis=1)


def _rebase(t: SpotTarget, origin: tuple[float, float]) -> SpotTarget:
    """Express an absolute screen target relative to a sub-region's own position."""
    return SpotTarget(t.x - origin[0], t.y - origin[1], t.z, t.weight, t.label)


def _make_grating(kind: str, t: SpotTarget, lam: float, **kw) -> Grating:
    if kind in ("focuser", "spot_focuser", "off_axis_lens"):
        return SpotFocuser(target_x=t.x, target_y=t.y, range_z=t.z,
                           design_wavelength=lam, **kw)
    if kind in ("blazed", "blaze"):
        return BlazedGrating.from_target(t.x, t.y, t.z, lam, **kw)
    if kind in ("ramp", "phase_ramp", "ideal"):
        from .gratings.periodic import PhaseRamp

        return PhaseRamp.from_target(t.x, t.y, t.z, lam, **kw)
    from ..registry import create

    return create("grating", kind, design_wavelength=lam, **kw)


def _scaled_mask(ap: Aperture, grid: Grid, cx: float, cy: float, weight: float,
                 tot_w: float, weight_mode: str, n_active: int):
    """Rasterise a sub-aperture, shrinking it if weights are realised by area."""
    if weight_mode != "area" or tot_w <= 0:
        return ap.mask(grid, cx, cy)
    share = max(weight, 0.0) / tot_w * n_active  # 1.0 == equal share
    if abs(share - 1.0) < 1e-9:
        return ap.mask(grid, cx, cy)
    s = float(np.sqrt(np.clip(share, 0.0, 1.0)))  # linear shrink for area ratio
    if s <= 0:
        return eng.xp.zeros(grid.shape, dtype=eng.REAL)
    X, Y = grid.meshgrid()
    return ap.mask_at((X - cx) / s, (Y - cy) / s)
