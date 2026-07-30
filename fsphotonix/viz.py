"""Plotting helpers (matplotlib optional).

Every function returns the matplotlib axes so plots can be composed; nothing here
is imported by the simulation core, so the package works headless.
"""

from __future__ import annotations

import numpy as np

from . import engine as eng
from .fields import FieldSet
from .grid import Grid


def _plt():
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError("plotting needs matplotlib: pip install matplotlib") from e
    return plt


def show_intensity(fields: FieldSet, ax=None, log: bool = False, floor_db: float = -40,
                   title: str = "", mm: bool = True, cmap: str = "inferno"):
    """Time-averaged irradiance of a field set."""
    plt = _plt()
    ax = ax or plt.subplots(figsize=(5, 4.2))[1]
    I = eng.to_numpy(fields.mean_intensity())
    ext = np.array(fields.grid.extent) * (1e3 if mm else 1.0)
    if log:
        m = I.max() or 1.0
        I = 10 * np.log10(np.maximum(I / m, 10 ** (floor_db / 10)))
    im = ax.imshow(I, origin="lower", extent=ext, cmap=cmap, aspect="equal")
    ax.set_xlabel("x [mm]" if mm else "x [m]")
    ax.set_ylabel("y [mm]" if mm else "y [m]")
    ax.set_title(title or f"irradiance @ z={fields.z*1e3:.3g} mm")
    plt.colorbar(im, ax=ax, label="dB rel. peak" if log else "W/m$^2$")
    return ax


def show_phase(array, grid: Grid, ax=None, title: str = "phase", mm: bool = True):
    """Wrapped phase of a complex map (a hologram, or a field)."""
    plt = _plt()
    ax = ax or plt.subplots(figsize=(5, 4.2))[1]
    p = np.angle(eng.to_numpy(array))
    ext = np.array(grid.extent) * (1e3 if mm else 1.0)
    im = ax.imshow(p, origin="lower", extent=ext, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    ax.set_xlabel("x [mm]" if mm else "x [m]")
    ax.set_ylabel("y [mm]" if mm else "y [m]")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="rad")
    return ax


def show_hologram(modulator, grid: Grid, wavelength: float | None = None, axes=None):
    """Amplitude and phase of the rendered modulator plane, side by side."""
    plt = _plt()
    t = eng.to_numpy(modulator.transmittance(grid, wavelength))
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ext = np.array(grid.extent) * 1e3
    im0 = axes[0].imshow(np.abs(t), origin="lower", extent=ext, cmap="gray")
    axes[0].set_title("amplitude")
    plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(np.angle(t), origin="lower", extent=ext, cmap="twilight",
                         vmin=-np.pi, vmax=np.pi)
    axes[1].set_title("phase [rad]")
    plt.colorbar(im1, ax=axes[1])
    for a in axes:
        a.set_xlabel("x [mm]")
        a.set_ylabel("y [mm]")
    return axes


def show_spots(spots, ax=None, screen: Grid | None = None, scale: float = 200.0,
               color_by: str = "wavelength"):
    """Scatter plot of a beamlet spot list, marker area proportional to power."""
    plt = _plt()
    ax = ax or plt.subplots(figsize=(5.2, 4.6))[1]
    if not len(spots):
        ax.set_title("no spots")
        return ax
    t = spots.table() if hasattr(spots, "table") else np.asarray(spots)
    p = t["power"]
    s = scale * p / (p.max() or 1.0)
    c = t[color_by] * (1e9 if color_by == "wavelength" else 1.0)
    sc = ax.scatter(t["x"] * 1e3, t["y"] * 1e3, s=np.maximum(s, 2), c=c, cmap="viridis")
    plt.colorbar(sc, ax=ax, label="wavelength [nm]" if color_by == "wavelength" else color_by)
    if screen is not None:
        e = np.array(screen.extent) * 1e3
        ax.add_patch(plt.Rectangle((e[0], e[2]), e[1] - e[0], e[3] - e[2],
                                   fill=False, ls="--", ec="0.5"))
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(f"{len(t)} beamlets, {p.sum()*1e3:.4g} mW")
    # A fan-out along one axis leaves the other with (almost) no spread; with an
    # equal aspect that collapses the axes to zero height and makes the layout
    # transform singular, so give every axis a floor on its span.
    _pad_limits(ax, t["x"] * 1e3, t["y"] * 1e3)
    ax.set_aspect("equal", adjustable="datalim")
    return ax


def _pad_limits(ax, xs, ys, floor_frac: float = 0.15) -> None:
    """Set axis limits with a margin, never letting a span reach zero."""
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    span = max(np.ptp(xs), np.ptp(ys), 1e-12)
    pad = max(0.05 * span, 1e-9)
    half = max(span, floor_frac * span) / 2 + pad
    cx, cy = 0.5 * (xs.min() + xs.max()), 0.5 * (ys.min() + ys.max())
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)


def show_detector(out, ax=None, title: str = ""):
    """Detector output as an image (2-D layouts) or a stem plot (1-D)."""
    plt = _plt()
    ax = ax or plt.subplots(figsize=(5, 4))[1]
    v = np.asarray(out.values, dtype=float)
    if v.ndim == 2 and min(v.shape) > 1:
        im = ax.imshow(v, origin="lower", cmap="viridis")
        plt.colorbar(im, ax=ax, label=out.unit)
    else:
        ax.stem(np.arange(v.size), v.ravel())
        ax.set_ylabel(out.unit)
        ax.set_xlabel("element")
    ax.set_title(title or f"{out.unit} readout")
    return ax


def show_convergence(result, ax=None, reference: float | None = None):
    """Energy versus iteration for an AIM run."""
    plt = _plt()
    ax = ax or plt.subplots(figsize=(5.4, 3.6))[1]
    e = np.asarray(result.energies if hasattr(result, "energies") else result, dtype=float)
    ax.plot(e, lw=1.2, label="Ising energy")
    ax.plot(np.minimum.accumulate(e), lw=1.6, label="best so far")
    if reference is not None:
        ax.axhline(reference, ls="--", c="k", lw=1, label="optimum")
    ax.set_xlabel("iteration")
    ax.set_ylabel("energy")
    ax.legend(fontsize=8)
    return ax


def show_system(system, ax=None):
    """Schematic of the plane layout along the optical axis."""
    plt = _plt()
    ax = ax or plt.subplots(figsize=(8, 2.4))[1]
    colors = {"SourceStage": "#d97706", "ModulatorStage": "#2563eb",
              "DetectorStage": "#16a34a", "ElementStage": "#7c3aed",
              "FunctionStage": "#64748b"}
    seen: dict[float, int] = {}
    for s in system.stages:
        z = round(s.z * 1e3, 6)
        # planes can share a z (a source and the panel it illuminates); stagger
        # their labels so they stay legible
        level = seen.get(z, 0)
        seen[z] = level + 1
        h = 1.0
        ax.plot([z, z], [-h, h], lw=6, color=colors.get(type(s).__name__, "0.4"),
                solid_capstyle="butt", alpha=1.0 if level == 0 else 0.55)
        ax.text(z, h * (1.15 + 0.45 * level), s.name, ha="center", va="bottom",
                fontsize=8, rotation=45)
    ax.set_ylim(-1.6, 2.4 + 0.45 * max(max(seen.values()) - 1, 0))
    ax.set_xlabel("optical path z [mm]")
    ax.set_yticks([])
    ax.set_title(f"{system.name}: {len(system)} planes")
    return ax
