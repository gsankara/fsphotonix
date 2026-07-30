"""Shared figure handling for the examples.

Every example takes the same two flags:

    python examples/01_fanout.py --plot                # open a window
    python examples/01_fanout.py --save docs/figures   # write a PNG instead

``--save`` selects a non-interactive backend before pyplot is imported, so the
examples double as the script that regenerates the figures checked into the repo.
"""

from __future__ import annotations

import sys
from pathlib import Path

SAVE_FLAG = "--save"
PLOT_FLAG = "--plot"


def save_dir() -> Path | None:
    """Directory given after ``--save``, or ``None`` if the flag is absent."""
    if SAVE_FLAG not in sys.argv:
        return None
    i = sys.argv.index(SAVE_FLAG)
    target = sys.argv[i + 1] if i + 1 < len(sys.argv) else "docs/figures"
    return Path(target)


def wanted() -> bool:
    """True when the example should build its figure at all."""
    return PLOT_FLAG in sys.argv or SAVE_FLAG in sys.argv


def pyplot():
    """Import pyplot, forcing a headless backend when we are only saving."""
    import matplotlib

    if save_dir() is not None and PLOT_FLAG not in sys.argv:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def finish(fig, name: str, dpi: int = 130) -> None:
    """Save the figure under ``--save``, show it under ``--plot``, or both."""
    try:
        fig.tight_layout()
    except Exception:  # a degenerate axis can make the layout solver fail
        pass
    out = save_dir()
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{name}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"  figure -> {path}")
    if PLOT_FLAG in sys.argv:
        pyplot().show()
    else:
        pyplot().close(fig)
