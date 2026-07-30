"""Beamlet ("spot list") propagation for very large systems.

The wave-optics propagators cost one transform over the whole aperture per
wavelength.  That is fine up to a few thousand samples per side, but a panel with
10^4 regions each fanning out to 10^3 screen locations is hopeless that way.

This module takes the other route.  Every sub-region is treated as one beamlet:
its incident power and mean tilt are measured, its grating's discrete diffraction
orders are looked up analytically, and each order is deposited on the screen as a
single diffraction-limited spot.  Cost is O(sub-regions x orders) and independent
of screen resolution -- the spot *table* is the primary output and can be used
without ever rasterising an image.

Accuracy: exact for the position, the power and the spot width of each order;
approximate in the wings (a Gaussian or sinc envelope replaces the true aperture
transform) and in the phase (each beamlet carries a single piston).  Interference
between beamlets that land on top of each other *is* modelled, because the spots
are summed as complex amplitudes within a coherence group.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

import numpy as np

from .. import engine as eng
from ..fields import Channel, FieldSet
from ..grid import Grid

if TYPE_CHECKING:  # pragma: no cover
    from ..modulators.array import ModulatorArray, Region, SubRegion


@dataclass
class Spot:
    """One beamlet landing on the screen."""

    x: float
    y: float
    power: float
    wavelength: float
    waist_x: float
    waist_y: float
    phase: float = 0.0
    #: direction cosines of the beamlet's propagation, needed by the *next*
    #: diffractive stage (a hologram adds its grating vector to the incident one)
    alpha: float = 0.0
    beta: float = 0.0
    coherence: str = ""
    channel: int = 0
    region: str = ""
    subregion: str = ""
    order: tuple[int, int] = (0, 0)

    @property
    def peak_irradiance(self) -> float:
        return 2 * self.power / (np.pi * self.waist_x * self.waist_y)

    def angles(self) -> tuple[float, float]:
        return (float(np.arcsin(np.clip(self.alpha, -1, 1))),
                float(np.arcsin(np.clip(self.beta, -1, 1))))


class SpotList(list):
    """A list of :class:`Spot` with reporting helpers."""

    def total_power(self) -> float:
        return float(sum(s.power for s in self))

    def table(self) -> np.ndarray:
        """Structured array of the whole spot list (for export / analysis)."""
        dt = [("x", "f8"), ("y", "f8"), ("power", "f8"), ("wavelength", "f8"),
              ("waist_x", "f8"), ("waist_y", "f8"), ("alpha", "f8"), ("beta", "f8"),
              ("channel", "i4"), ("mx", "i4"), ("my", "i4")]
        return np.array([(s.x, s.y, s.power, s.wavelength, s.waist_x, s.waist_y,
                          s.alpha, s.beta, s.channel, s.order[0], s.order[1])
                         for s in self], dtype=dt)

    def bin_to(self, positions, radius: float) -> np.ndarray:
        """Total power within ``radius`` of each of ``positions`` -- a virtual
        detector array evaluated without ever making an image."""
        pos = np.asarray(positions, dtype=float).reshape(-1, 2)
        out = np.zeros(len(pos))
        if not self:
            return out
        sx = np.array([s.x for s in self])
        sy = np.array([s.y for s in self])
        sp = np.array([s.power for s in self])
        for i, (px, py) in enumerate(pos):
            m = (sx - px) ** 2 + (sy - py) ** 2 <= radius**2
            out[i] = float(sp[m].sum())
        return out

    def summary(self) -> dict:
        if not self:
            return {"spots": 0, "power": 0.0}
        t = self.table()
        return {
            "spots": len(self),
            "power": float(t["power"].sum()),
            "x_range": (float(t["x"].min()), float(t["x"].max())),
            "y_range": (float(t["y"].min()), float(t["y"].max())),
            "wavelengths": sorted(set(np.round(t["wavelength"], 15).tolist())),
        }


@dataclass
class SpotPropagator:
    """Beamlet propagation from a modulator plane to a screen.

    ``envelope`` selects the per-spot profile: ``gaussian`` (default, smooth and
    fast) or ``sinc`` (the true far-field of a hard rectangular sub-aperture,
    including side lobes).
    """

    z: float = 0.1
    out_grid: Grid | None = None
    envelope: str = "gaussian"
    waist_factor: float = 0.75  # matches the sinc main-lobe FWHM
    max_order: int = 4
    min_efficiency: float = 1e-4
    mask_supersample: int = 3
    carry_direction: bool = True
    enforce_device_limit: bool = True
    include_background: bool = True
    render_radius: float = 4.0  # how many waists to render around each spot

    # ---- spot table -------------------------------------------------------- #
    def spots(self, modulator: "ModulatorArray", fields: FieldSet,
              z: float | None = None) -> SpotList:
        zz = self.z if z is None else z
        out = SpotList()
        grid = fields.grid
        for ci, ch in enumerate(fields.channels):
            lam = ch.wavelength
            for r in modulator.regions:
                if not r.enabled or r.gain <= 0:
                    continue
                active = [s for s in r.subregions if s.enabled and s.grating is not None]
                if not active:
                    continue
                tot_w = sum(s.weight for s in active) or 1.0
                for s in active:
                    cx = r.center[0] + s.offset[0]
                    cy = r.center[1] + s.offset[1]
                    ap = s.aperture if s.aperture is not None else r.aperture
                    p_in, fx_in, fy_in, ph_in = _sample_patch(
                        ch, grid, ap, cx, cy, r.aperture, r.center,
                        supersample=self.mask_supersample)
                    if p_in <= 0:
                        continue
                    # the sub-aperture mask already accounts for the area split, so
                    # only the *relative* weight (1.0 == equal share) applies here
                    share = s.weight * len(active) / tot_w
                    p_in *= share * r.gain**2 * modulator.device.efficiency
                    dx_ap, dy_ap = ap.size
                    wx = max(self.waist_factor * lam * abs(zz) / max(dx_ap, 1e-9), 1e-12)
                    wy = max(self.waist_factor * lam * abs(zz) / max(dy_ap, 1e-9), 1e-12)
                    fnyq_x = 1.0 / (2 * modulator.device.pitch) if self.enforce_device_limit \
                        else None
                    for o in s.grating.orders(lam, max_order=self.max_order):
                        if o.efficiency < self.min_efficiency:
                            continue
                        gx, gy = _fold(o.fx, fnyq_x), _fold(o.fy, fnyq_x)
                        # the incident tilt is free-space propagation and is not
                        # sampled by the device, so only the written grating folds
                        fx = fx_in + gx
                        fy = fy_in + gy
                        a, b = lam * fx, lam * fy
                        g2 = 1.0 - a * a - b * b
                        if g2 <= 0:
                            continue  # evanescent / beyond the horizon
                        g = np.sqrt(g2)
                        out.append(Spot(
                            x=cx + zz * a / g,
                            y=cy + zz * b / g,
                            power=p_in * o.efficiency,
                            wavelength=lam,
                            waist_x=wx / max(g, 1e-3),
                            waist_y=wy / max(g, 1e-3),
                            phase=ph_in + r.piston + 2 * np.pi * abs(zz) / lam,
                            alpha=float(a),
                            beta=float(b),
                            coherence=ch.coherence,
                            channel=ci,
                            region=r.label,
                            subregion=s.label,
                            order=(o.mx, o.my),
                        ))
        return out

    # ---- rendering --------------------------------------------------------- #
    def propagate(self, modulator: "ModulatorArray", fields: FieldSet,
                  out_grid: Grid | None = None, z: float | None = None) -> FieldSet:
        """Render the spot list back into a :class:`FieldSet` on the screen grid."""
        zz = self.z if z is None else z
        og = out_grid or self.out_grid
        if og is None:
            raise ValueError("SpotPropagator needs an output grid to render a field")
        spots = self.spots(modulator, fields, zz)
        chans: list[Channel] = []
        for ci, ch in enumerate(fields.channels):
            mine = [s for s in spots if s.channel == ci]
            u = self.render(mine, og)
            chans.append(Channel(u=u, wavelength=ch.wavelength, coherence=ch.coherence,
                                 waveform=ch.waveform, label=ch.label,
                                 meta={**ch.meta, "spots": len(mine)}))
        fs = fields.like(chans, grid=og, z=fields.z + zz)
        fs.meta = {**fields.meta, "spot_list": spots}
        return fs

    def render(self, spots: Sequence[Spot], grid: Grid):
        """Sum beamlet amplitudes onto a grid (complex, so they can interfere)."""
        u = eng.zeros(grid.shape)
        if not spots:
            return u
        xs = eng.to_numpy(grid.x())
        ys = eng.to_numpy(grid.y())
        for s in spots:
            i0 = int(np.searchsorted(xs, s.x - self.render_radius * s.waist_x))
            i1 = int(np.searchsorted(xs, s.x + self.render_radius * s.waist_x))
            j0 = int(np.searchsorted(ys, s.y - self.render_radius * s.waist_y))
            j1 = int(np.searchsorted(ys, s.y + self.render_radius * s.waist_y))
            i0, i1 = max(i0, 0), min(max(i1, i0 + 1), grid.nx)
            j0, j1 = max(j0, 0), min(max(j1, j0 + 1), grid.ny)
            if i0 >= grid.nx or j0 >= grid.ny or i1 <= 0 or j1 <= 0:
                continue
            X = eng.asarray(xs[i0:i1].reshape(1, -1) - s.x, dtype=eng.REAL)
            Y = eng.asarray(ys[j0:j1].reshape(-1, 1) - s.y, dtype=eng.REAL)
            amp = np.sqrt(2 * s.power / (np.pi * s.waist_x * s.waist_y))
            if self.envelope == "gaussian":
                env = eng.xp.exp(-((X / s.waist_x) ** 2 + (Y / s.waist_y) ** 2))
            elif self.envelope == "sinc":
                # main-lobe-matched sinc^2 in intensity -> sinc in amplitude
                env = eng.xp.sinc(X / (s.waist_x / 0.6)) * eng.xp.sinc(Y / (s.waist_y / 0.6))
            else:
                raise ValueError(f"unknown envelope {self.envelope!r}")
            term = amp * env * np.exp(1j * s.phase)
            if self.carry_direction and (s.alpha or s.beta):
                # keep the beamlet's tilt in the rendered wavefront: the next
                # diffractive plane adds its grating vector to the *incident* one,
                # so a cascade is wrong without it
                k = 2 * np.pi / s.wavelength
                term = term * eng.xp.exp(1j * k * (s.alpha * X + s.beta * Y))
            u[j0:j1, i0:i1] = u[j0:j1, i0:i1] + term
        return u

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"SpotPropagator(z={self.z*1e3:.4g}mm, envelope={self.envelope!r})"


# --------------------------------------------------------------------------- #
def _fold(f: float, f_nyquist: float | None) -> float:
    """Alias a written spatial frequency onto the device's sampling lattice.

    A modulator cannot write a grating finer than two pixels per period.  Asking for
    more does not simply fail: the pattern aliases, and the light goes to the folded
    order instead of the requested one.  The wave-optics path gets this for free from
    pixelation, so the beamlet path folds explicitly to stay consistent with it --
    and so that a design which over-drives the steering range is caught in the
    simulation rather than only in ``check()``.
    """
    if f_nyquist is None or f_nyquist <= 0:
        return f
    span = 2.0 * f_nyquist
    return (f + f_nyquist) % span - f_nyquist


def _antialiased_mask(ap, region_ap, region_center, xs, ys, cx: float, cy: float,
                      grid: Grid, k: int):
    """Sub-aperture (optionally clipped by the region) averaged over a k x k sub-grid."""
    offs = (np.arange(k) + 0.5) / k - 0.5
    acc = None
    for oy in offs:
        for ox in offs:
            x = xs + ox * grid.dx
            y = ys + oy * grid.dy
            m = ap.mask_at(x - cx, y - cy)
            if region_ap is not None:
                m = m * region_ap.mask_at(x - region_center[0], y - region_center[1])
            acc = m if acc is None else acc + m
    return acc / (k * k)


def _sample_patch(ch: Channel, grid: Grid, ap, cx: float, cy: float,
                  region_ap, region_center, supersample: int = 3
                  ) -> tuple[float, float, float, float]:
    """Incident power, mean spatial frequency and mean phase over a sub-aperture.

    Only the sub-aperture's bounding box is touched, so this stays cheap even with
    thousands of regions on a large grid.

    ``supersample`` anti-aliases the aperture edge by evaluating the mask on a
    k x k sub-grid inside every cell.  Without it, a hard-edged sub-aperture only a
    handful of samples across has its *area* quantised to whole cells, and regions
    sitting at different sub-cell offsets then collect visibly different power --
    a numerical artefact that otherwise shows up as a fake asymmetry between
    nominally identical channels (for instance the two rails of a dual-rail
    encoding).
    """
    xmin, xmax, ymin, ymax = ap.bounds()
    r0, c0 = grid.index_of(cx + xmin, cy + ymin)
    r1, c1 = grid.index_of(cx + xmax, cy + ymax)
    r0, r1 = min(r0, r1), max(r0, r1) + 1
    c0, c1 = min(c0, c1), max(c0, c1) + 1
    if r1 <= r0 or c1 <= c0:
        return (0.0, 0.0, 0.0, 0.0)
    u = ch.u[r0:r1, c0:c1]
    xs = grid.x()[c0:c1].reshape(1, -1)
    ys = grid.y()[r0:r1].reshape(-1, 1)
    m = _antialiased_mask(ap, region_ap, region_center, xs, ys, cx, cy, grid,
                          max(int(supersample), 1))
    w = u * m
    p = float(eng.to_numpy(eng.xp.sum(eng.xp.abs(w) ** 2)).item()) * grid.cell_area
    p *= ch.waveform.mean_power()
    if p <= 0:
        return (0.0, 0.0, 0.0, 0.0)
    # mean tilt from wrap-safe phase differences, weighted by local intensity
    fx = fy = 0.0
    if w.shape[1] > 1:
        d = w[:, 1:] * eng.xp.conj(w[:, :-1])
        s = eng.to_numpy(eng.xp.sum(d))
        fx = float(np.angle(s)) / (2 * np.pi * grid.dx)
    if w.shape[0] > 1:
        d = w[1:, :] * eng.xp.conj(w[:-1, :])
        s = eng.to_numpy(eng.xp.sum(d))
        fy = float(np.angle(s)) / (2 * np.pi * grid.dy)
    phase = float(np.angle(eng.to_numpy(eng.xp.sum(w))))
    return (p, fx, fy, phase)
