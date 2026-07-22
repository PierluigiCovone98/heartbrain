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
_EXTENSION = ".png"


def plot_time_series(heart_series,
                     brain_series,
                     name: str,
                     subdir : str,
                     start: int | None = None,
                     end: int | None = None,
                     heart_label: str = "heart  x(t)",
                     brain_label: str = "brain  s(t)") -> Path:
    """Plot heart and brain time series as two stacked panels and save to disk.
 
    The two signals share the time axis (x) but keep independent amplitude axes
    (y), one panel each, so their phase alignment can be compared by eye. Passing
    ``start``/``end`` zooms both panels to the same window (the x-axis keeps the
    original step indices), which is needed to read the two-scale relationship on
    long runs where the full plot compresses everything into a solid band.
 
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
    subdir : str
        Output subdirectory under ``output/`` (e.g. ``"sweep"``).
    start : int | None
        First step of the window (inclusive). ``None`` means from the beginning.
    end : int | None
        Last step of the window (exclusive). ``None`` means until the end.
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
 
    # Resolve the window; both panels share it so they stay aligned in time.
    lo = start if start is not None else 0
    hi = end if end is not None else len(heart_series)
    heart_window = heart_series[lo:hi]
    brain_window = brain_series[lo:hi]
    steps = range(lo, lo + len(heart_window))  # keep original step indices
 
    # Two panels, shared x (time), independent y (amplitude).
    fig, (ax_heart, ax_brain) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
 
    ax_heart.plot(steps, heart_window, color="crimson", linewidth=1.0)
    ax_heart.set_ylabel(heart_label)
    ax_heart.grid(True, alpha=0.3)
 
    ax_brain.plot(steps, brain_window, color="steelblue", linewidth=1.0)
    ax_brain.set_ylabel(brain_label)
    ax_brain.set_xlabel("step")
    ax_brain.grid(True, alpha=0.3)
 
    fig.suptitle(name)
    fig.tight_layout()
 
    # Resolve output path and ensure the directory exists.
    out_dir = _paths.output_subdir(subdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (name + _EXTENSION)
 
    fig.savefig(path, dpi=120)
    plt.close(fig)  # Free the figure; important when called in a loop.
 
    return path


def plot_single_series(series,
                       name: str,
                       subdir: str,
                       start: int | None = None,
                       end: int | None = None,
                       label: str = "s(t)",
                       color: str = "steelblue") -> Path:
    """Plot a single time series, optionally zoomed to a time window, and save it.
 
    Meant for inspecting one signal in detail — e.g. the network's projected
    scalar ``s(t)`` — where a full-length plot compresses thousands of steps into
    a solid band. Passing ``start``/``end`` slices the series to a readable window
    so individual oscillations separate. The x-axis keeps the original step
    indices, so a zoomed window still shows *where* in the run it comes from.
 
    Parameters
    ----------
    series : array-like
        The signal to plot, one value per step.
    name : str
        Bare file name (no directory, no extension). Saved to
        ``output/coupled/<name>.png``.
    subdir : str
        Output subdirectory under ``output/`` (e.g. ``"sweep"``).
    start : int | None
        First step of the window (inclusive). ``None`` means from the beginning.
    end : int | None
        Last step of the window (exclusive). ``None`` means until the end.
    label : str
        Y-axis label for the signal.
    color : str
        Line color.
 
    Returns
    -------
    Path
        The path of the saved figure.
    """
    # Resolve the window; Python slice semantics handle None on both ends.
    lo = start if start is not None else 0
    hi = end if end is not None else len(series)
    window = series[lo:hi]
    steps = range(lo, lo + len(window))  # keep original step indices on the x-axis
 
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, window, color=color, linewidth=1.0)
    ax.set_ylabel(label)
    ax.set_xlabel("step")
    ax.grid(True, alpha=0.3)
    fig.suptitle(name)
    fig.tight_layout()
 
    out_dir = _paths.output_subdir(subdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (name + _EXTENSION)
 
    fig.savefig(path, dpi=120)
    plt.close(fig)
 
    return path


def plot_curve(x_values,
               y_values,
               name: str,
               subdir: str,
               x_label: str = "x",
               y_label: str = "y",
               color: str = "steelblue",
               marker: str = "o") -> Path:
    """Plot a single y-vs-x curve and save it to disk.
 
    A general curve plotter: it draws ``y_values`` against ``x_values`` as a line
    with markers. It carries no assumption about what the axes mean — the caller
    labels them — so the same function serves any sweep result (e.g. a locking
    value against coupling strength, a period against gain, and so on).
 
    Markers are shown because sweep points are often unevenly spaced (denser
    where the interesting transition is), and the markers make the actual
    sampling visible rather than hidden inside a smooth line.
 
    Parameters
    ----------
    x_values : array-like
        The horizontal coordinates (e.g. the swept parameter).
    y_values : array-like
        The vertical coordinates (e.g. the measured quantity). Same length as
        ``x_values``.
    name : str
        Bare file name (no directory, no extension). Saved to
        ``output/<subdir>/<name>.png``.
    subdir : str
        Output subdirectory under ``output/`` (e.g. ``"sweep"``).
    x_label, y_label : str
        Axis labels.
    color : str
        Line and marker color.
    marker : str
        Marker style for the sampled points.
 
    Returns
    -------
    Path
        The path of the saved figure.
 
    Raises
    ------
    ValueError
        If the two coordinate arrays have different lengths.
    """
    if len(x_values) != len(y_values):
        raise ValueError(
            f"Length mismatch: x has {len(x_values)}, y has {len(y_values)}."
        )
 
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_values, y_values, color=color, marker=marker, markersize=4, linewidth=1.0)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    fig.suptitle(name)
    fig.tight_layout()
 
    out_dir = _paths.output_subdir(subdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (name + _EXTENSION)
 
    fig.savefig(path, dpi=120)
    plt.close(fig)
 
    return path


def plot_two_panels(x_values,
                    top_values,
                    bottom_values,
                    name: str,
                    subdir: str,
                    x_label: str = "x",
                    top_label: str = "top",
                    bottom_label: str = "bottom",
                    top_color: str = "crimson",
                    bottom_color: str = "steelblue",
                    marker: str = "o") -> Path:
    """Plot two quantities against a shared x-axis, as two stacked panels.

    A generic two-panel plotter: the panels share the x-axis but keep
    independent y-axes, so two quantities on different scales can be compared
    point by point. It assumes nothing about what the axes mean — the caller
    labels them — so it serves any shared-x comparison (e.g. a reference signal
    and a measured quantity, both resolved over the same phase bins).

    Parameters
    ----------
    x_values : array-like
        The shared horizontal coordinates.
    top_values, bottom_values : array-like
        The quantities for the top and bottom panels. Same length as
        ``x_values``.
    name : str
        Bare file name (no directory, no extension). Saved to
        ``output/<subdir>/<name>.png``.
    subdir : str
        Output subdirectory under ``output/``.
    x_label : str
        Shared x-axis label.
    top_label, bottom_label : str
        Y-axis labels for the two panels.
    top_color, bottom_color : str
        Line/marker colors.
    marker : str
        Marker style.

    Returns
    -------
    Path
        The path of the saved figure.

    Raises
    ------
    ValueError
        If the arrays have different lengths.
    """
    if not (len(x_values) == len(top_values) == len(bottom_values)):
        raise ValueError(
            f"Length mismatch: x={len(x_values)}, top={len(top_values)}, "
            f"bottom={len(bottom_values)}."
        )

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

    ax_top.plot(x_values, top_values, color=top_color, marker=marker, markersize=4, linewidth=1.0)
    ax_top.set_ylabel(top_label)
    ax_top.grid(True, alpha=0.3)

    ax_bottom.plot(x_values, bottom_values, color=bottom_color, marker=marker, markersize=4, linewidth=1.0)
    ax_bottom.set_ylabel(bottom_label)
    ax_bottom.set_xlabel(x_label)
    ax_bottom.grid(True, alpha=0.3)

    fig.suptitle(name)
    fig.tight_layout()

    out_dir = _paths.output_subdir(subdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (name + _EXTENSION)

    fig.savefig(path, dpi=120)
    plt.close(fig)

    return path