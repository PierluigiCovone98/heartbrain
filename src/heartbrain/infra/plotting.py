"""Plotting module.

Visualization helpers for inspecting experiment signals. Unlike the persistence
package (which stores data to be reloaded), these produce images: 
they live under ``output/`` rather than ``data/``.
"""
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
# Non-interactive backend: render to file, no display needed.
matplotlib.use("Agg")

from heartbrain.infra import _paths


# Constant
_SUBDIR = "coupled"  # Output subdirectory for coupled-run figures.
_EXTENSION = ".png"


def plot_time_series(heart_series,
                     brain_series,
                     name: str,
                     heart_label: str = "heart  x(t)",
                     brain_label: str = "brain  s(t)") -> Path:
    """Plot heart and brain time series as two stacked panels and save to disk.

    The two signals share the time axis (x) but keep independent amplitude axes
    (y), one panel each, so their phase alignment can be compared by eye.

    Parameters
    ----------
    heart_series : array-like
        The heart signal ``x(t)``, one value per step.
    brain_series : array-like
        The network signal ``s(t)``, one value per step. Must match the length
        of ``heart_series``.
    name : str
        Bare file name (no directory, no extension). The figure is saved to
        ``output/coupled/<name>.png``.
    heart_label : str
        Legend/axis label for the heart panel.
    brain_label : str
        Legend/axis label for the brain panel.

    Returns
    -------
    Path
        The path of the saved figure.

    Raises
    ------
    ValueError
        If the two series have different lengths.
    """
    if len(heart_series) != len(brain_series):
        raise ValueError(
            f"Series length mismatch: heart has {len(heart_series)}, "
            f"brain has {len(brain_series)}."
        )

    steps = range(len(heart_series))

    # Two panels, shared x (time), independent y (amplitude).
    fig, (ax_heart, ax_brain) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

    ax_heart.plot(steps, heart_series, color="crimson", linewidth=1.0)
    ax_heart.set_ylabel(heart_label)
    ax_heart.grid(True, alpha=0.3)

    ax_brain.plot(steps, brain_series, color="steelblue", linewidth=1.0)
    ax_brain.set_ylabel(brain_label)
    ax_brain.set_xlabel("step")
    ax_brain.grid(True, alpha=0.3)

    fig.suptitle(name)
    fig.tight_layout()

    # Resolve output path and ensure the directory exists.
    out_dir = _paths.output_subdir(_SUBDIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (name + _EXTENSION)

    fig.savefig(path, dpi=120)
    plt.close(fig)  # Free the figure; important when called in a loop.

    return path
