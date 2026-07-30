"""Sampling grids for transverse planes."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from . import engine as eng


@dataclass(frozen=True)
class Grid:
    """A uniform 2-D sampling grid, indexed ``[y, x]``.

    ``x0``/``y0`` shift the grid centre, which is how off-axis detection screens
    and tiled modulator panels are described.
    """

    nx: int
    ny: int
    dx: float
    dy: float | None = None
    x0: float = 0.0
    y0: float = 0.0

    def __post_init__(self) -> None:
        if self.dy is None:
            object.__setattr__(self, "dy", self.dx)
        if self.nx < 1 or self.ny < 1:
            raise ValueError("grid must have at least one sample per axis")

    # ---- construction helpers -------------------------------------------- #
    @classmethod
    def from_extent(cls, width: float, height: float | None = None, pitch: float = 1e-6, **kw) -> "Grid":
        """Grid covering ``width`` x ``height`` at the given sample pitch."""
        height = width if height is None else height
        nx = max(1, int(round(width / pitch)))
        ny = max(1, int(round(height / pitch)))
        return cls(nx=nx, ny=ny, dx=pitch, dy=pitch, **kw)

    @classmethod
    def from_samples(cls, n: int, width: float, **kw) -> "Grid":
        return cls(nx=n, ny=n, dx=width / n, dy=width / n, **kw)

    def like(self, **changes) -> "Grid":
        return replace(self, **changes)

    def centered_on(self, x0: float, y0: float) -> "Grid":
        return replace(self, x0=x0, y0=y0)

    def rescaled(self, factor: float) -> "Grid":
        """Same sample count, pitch multiplied by ``factor`` (a magnified plane)."""
        return replace(self, dx=self.dx * factor, dy=self.dy * factor)

    # ---- geometry --------------------------------------------------------- #
    @property
    def shape(self) -> tuple[int, int]:
        return (self.ny, self.nx)

    @property
    def size(self) -> int:
        return self.nx * self.ny

    @property
    def width(self) -> float:
        return self.nx * self.dx

    @property
    def height(self) -> float:
        return self.ny * self.dy

    @property
    def cell_area(self) -> float:
        return self.dx * self.dy

    def x(self):
        """Column coordinates [m]."""
        return eng.asarray((np.arange(self.nx) - self.nx // 2) * self.dx + self.x0, dtype=eng.REAL)

    def y(self):
        """Row coordinates [m]."""
        return eng.asarray((np.arange(self.ny) - self.ny // 2) * self.dy + self.y0, dtype=eng.REAL)

    def meshgrid(self):
        """``(X, Y)`` arrays of shape ``(ny, nx)``."""
        return eng.xp.meshgrid(self.x(), self.y())

    def rho(self):
        X, Y = self.meshgrid()
        return eng.xp.sqrt(X * X + Y * Y)

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """``(xmin, xmax, ymin, ymax)`` suitable for ``imshow(extent=...)``."""
        x = eng.to_numpy(self.x())
        y = eng.to_numpy(self.y())
        return (
            float(x[0] - self.dx / 2),
            float(x[-1] + self.dx / 2),
            float(y[0] - self.dy / 2),
            float(y[-1] + self.dy / 2),
        )

    # ---- frequency space -------------------------------------------------- #
    def fx(self):
        """Spatial frequency along x [1/m], FFT order."""
        return eng.fftfreq(self.nx, self.dx)

    def fy(self):
        return eng.fftfreq(self.ny, self.dy)

    def freq_meshgrid(self):
        return eng.xp.meshgrid(self.fx(), self.fy())

    @property
    def nyquist(self) -> tuple[float, float]:
        return (1.0 / (2 * self.dx), 1.0 / (2 * self.dy))

    def max_angle(self, wavelength: float) -> float:
        """Largest propagation angle [rad] the grid can represent."""
        f = min(self.nyquist)
        return float(np.arcsin(min(1.0, f * wavelength)))

    # ---- indexing --------------------------------------------------------- #
    def index_of(self, x: float, y: float) -> tuple[int, int]:
        """Nearest ``(row, col)`` for a physical coordinate."""
        col = int(round((x - self.x0) / self.dx)) + self.nx // 2
        row = int(round((y - self.y0) / self.dy)) + self.ny // 2
        return (int(np.clip(row, 0, self.ny - 1)), int(np.clip(col, 0, self.nx - 1)))

    def contains(self, x: float, y: float) -> bool:
        xmin, xmax, ymin, ymax = self.extent
        return xmin <= x <= xmax and ymin <= y <= ymax

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Grid({self.ny}x{self.nx}, pitch={self.dx*1e6:.3g}x{self.dy*1e6:.3g} um, "
            f"extent={self.width*1e3:.3g}x{self.height*1e3:.3g} mm, "
            f"center=({self.x0*1e3:.3g},{self.y0*1e3:.3g}) mm)"
        )


def sampling_report(grid: Grid, wavelength: float, z: float) -> dict:
    """Diagnostics used by :func:`fsphotonix.propagation.suggest`.

    ``fresnel_number`` and the critical distance ``z_c = N*dx^2/lambda`` decide
    which propagation kernel is well sampled.
    """
    zc_x = grid.nx * grid.dx**2 / wavelength
    zc_y = grid.ny * grid.dy**2 / wavelength
    return {
        "z": z,
        "z_critical_x": zc_x,
        "z_critical_y": zc_y,
        "fresnel_number": (grid.width / 2) ** 2 / (wavelength * z) if z else float("inf"),
        "max_angle_deg": np.degrees(grid.max_angle(wavelength)),
        "far_field_spot_x": wavelength * z / grid.width if z else 0.0,
        "far_field_spot_y": wavelength * z / grid.height if z else 0.0,
        "regime": "near" if z < 0.5 * zc_x else ("transition" if z < 2 * zc_x else "far"),
    }
