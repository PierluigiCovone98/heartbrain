"""Experiments module.

Saves and loads the conditions of an experiment: the parameters needed to
reproduce it. An experiment is persisted as two files sharing a base name inside
its own directory ``root/data/experiments/<name>/``:

- ``<name>.json`` — scalar parameters and metadata (seed, sizes, gains, the name
  of the baseline network used, ...). Human-readable.
- ``<name>.npz`` — array-valued conditions (e.g. the input ``x``), which do not
  belong in JSON.

Callers split their data into two groups — scalars and arrays — via the
``make_params`` / ``make_arrays`` helpers, keeping the division explicit rather
than guessed from types.
"""
import json

import numpy as np

from heartbrain.infra import _paths


# Constants
_SUBDIR = "experiments"     # Directory holding all saved experiments.
_PARAMS_EXTENSION = ".json"  # Scalar parameters / metadata.
_ARRAYS_EXTENSION = ".npz"   # Array-valued conditions.


# === Grouping helpers ===
#
# The caller is responsible for splitting its data into scalars and arrays.
# These two functions give that split a named, controlled place to happen (and a
# natural spot to add validation later).
# Native Python types that ``json`` can serialize as scalar parameters.
# Note: ``bool`` is a subclass of ``int``, so it is covered by ``int``.
_JSON_SCALAR_TYPES = (str, int, float, bool, type(None))


def make_params(**scalars: object) -> dict[str, object]:
    """Collect scalar parameters / metadata into a single dict.

    Values must be native Python scalars (``str``, ``int``, ``float``, ``bool``,
    ``None``) so they serialize cleanly to JSON. Non-native types — notably numpy
    scalars such as ``np.int64`` or ``np.float64`` — are rejected here, at the
    entry point, with a clear error, rather than failing deep inside ``json.dump``
    at save time.

    Raises
    ------
    TypeError
        If any value is not a native JSON-serializable scalar.
    """
    for key, value in scalars.items():
        if type(value) not in _JSON_SCALAR_TYPES:
            raise TypeError(
                f"Parameter '{key}' has non-native type {type(value).__name__}; "
                f"expected a Python scalar (str, int, float, bool, None). "
                f"Convert it explicitly (e.g. int(...) / float(...)) before saving."
            )
    return dict(scalars)


def make_arrays(**arrays: np.ndarray) -> dict[str, np.ndarray]:
    """Collect array-valued conditions into a single dict.

    Example: ``make_arrays(x=x, h0=h0)``.
    """
    return dict(arrays)


# === Public API ===
def save_experiment(name: str,
                    params: dict[str, object],
                    arrays: dict[str, np.ndarray]) -> None:
    """Save an experiment's conditions to ``data/experiments/<name>/``.

    Writes scalars to ``<name>.json`` and arrays to ``<name>.npz`` inside the
    experiment's own directory. The two files are handled behind this single
    call; the caller does not deal with formats.

    Parameters
    ----------
    name : str
        Experiment name; also the directory and the base file name.
    params : dict[str, object]
        Scalar parameters / metadata (see ``make_params``).
    arrays : dict[str, np.ndarray]
        Array-valued conditions (see ``make_arrays``).

    Raises
    ------
    FileExistsError
        If the experiment directory already exists (no silent overwrite).
    """
    exp_dir = _paths.data_subdir(_SUBDIR) / name

    if exp_dir.exists():
        raise FileExistsError(f"An experiment named '{name}' already exists.")

    exp_dir.mkdir(parents=True, exist_ok=True)

    _save_params_json(exp_dir / (name + _PARAMS_EXTENSION), params)
    _save_arrays_npz(exp_dir / (name + _ARRAYS_EXTENSION), arrays)


def load_experiment(name: str) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Load an experiment's conditions from ``data/experiments/<name>/``.

    Returns the two groups separately, symmetrically to how they were saved:
    scalars and arrays are not remerged.

    Parameters
    ----------
    name : str
        Experiment name.

    Returns
    -------
    tuple[dict[str, object], dict[str, np.ndarray]]
        ``(params, arrays)``.

    Raises
    ------
    FileNotFoundError
        If no experiment with this name exists.
    """
    exp_dir = _paths.data_subdir(_SUBDIR) / name

    if not exp_dir.exists():
        raise FileNotFoundError(f"No experiment named '{name}' found.")

    params = _load_params_json(exp_dir / (name + _PARAMS_EXTENSION))
    arrays = _load_arrays_npz(exp_dir / (name + _ARRAYS_EXTENSION))

    return params, arrays


# === Format-specific helpers (private) ===
def _save_params_json(path, params: dict[str, object]) -> None:
    """Write scalar parameters to a JSON file (indented for readability)."""
    with open(path, "w") as f:
        json.dump(params, f, indent=4)


def _load_params_json(path) -> dict[str, object]:
    """Read scalar parameters from a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def _save_arrays_npz(path, arrays: dict[str, np.ndarray]) -> None:
    """Write array-valued conditions to an ``.npz`` archive."""
    np.savez(path,**arrays)   # type: ignore[arg-type]


def _load_arrays_npz(path) -> dict[str, np.ndarray]:
    """Read array-valued conditions from an ``.npz`` archive into a plain dict.

    ``np.load`` is lazy and keeps the file open; we materialize the arrays into a
    regular dict so the caller gets plain arrays and no open file handle lingers.
    """
    with np.load(path) as data:
        return {key: data[key] for key in data.files}
