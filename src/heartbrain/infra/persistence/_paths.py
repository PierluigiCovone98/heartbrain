"""Path utilities shared across the persistence package.

Single place that knows how a data path is composed: the project root, the
``data`` directory, the per-category subdirectory, the file name and its
extension. All persistence modules (``networks``, ``experiments``, ...) build
their paths through here, so path logic lives in exactly one spot.
"""
from pathlib import Path

# Parent directory (under the project root) holding all saved data.
_DATA_DIR = Path("data")


def _find_project_root(marker: str = "pyproject.toml") -> Path:
    """Locate the project root by climbing until a marker file is found.

    Walks upward from this module's location and returns the first ancestor
    directory that contains ``marker``. Anchoring to a marker (rather than a
    fixed number of parent steps) keeps the path correct regardless of how deep
    this module sits in the package, so moving the file does not break it.

    Parameters
    ----------
    marker : str
        Name of the file that identifies the project root (default:
        ``pyproject.toml``).

    Returns
    -------
    Path
        The project root directory.

    Raises
    ------
    RuntimeError
        If no ancestor directory contains ``marker``.
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / marker).exists():
            return parent
    raise RuntimeError(f"Project root ('{marker}') not found.")


def data_subdir(subdir: str) -> Path:
    """Return the absolute path of a data subdirectory ``<root>/data/<subdir>``.

    Does not create anything on disk; it only composes the path.

    Parameters
    ----------
    subdir : str
        Category subdirectory under ``data`` (e.g. ``"saved_networks"``).

    Returns
    -------
    Path
        The absolute subdirectory path.
    """
    return _find_project_root() / _DATA_DIR / subdir


def resolve_path(subdir: str, name: str, extension: str = "") -> Path:
    """Compose the absolute path of a data file ``<root>/data/<subdir>/<name><extension>``.

    Extension handling lives here (not in the caller), so every persistence
    module builds file paths the same way. Pass ``name`` bare, without extension.

    Parameters
    ----------
    subdir : str
        Category subdirectory under ``data`` (e.g. ``"saved_networks"``).
    name : str
        Bare file name, without directory or extension.
    extension : str
        File extension including the leading dot (e.g. ``".npz"``); empty for
        a directory or an extension-less name.

    Returns
    -------
    Path
        The absolute file path.
    """
    return data_subdir(subdir) / (name + extension)


def ensure_parent(path: Path) -> None:
    """Create the parent directory of ``path`` if it does not exist.

    Idempotent: does nothing if the directory is already there.
    """
    path.parent.mkdir(parents=True, exist_ok=True)