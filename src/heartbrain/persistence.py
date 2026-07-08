"""Persistence module.

Saves and loads baseline ``VanillaRNN`` instances to and from disk, keeping the
network itself a pure mathematical object.

A saved network is the *identity* of the network — its static baseline weights
(``W_xh``, ``W_hh_baseline``, ``b_h_baseline``) plus the initial state ``h0``.
Derived quantities (the scaled ``W_hh``, the perturbed ``b_h``, the evolving
state) are not stored: they are reconstructed at load time as a clean, not-yet-
advanced network, exactly as ``VanillaRNN.from_weights`` produces them.

Networks are stored as ``.npz`` archives under ``data/saved_networks/`` at the
project root. Callers provide only a name; the extension and the directory are
handled here, so every network is saved uniformly and in one place regardless of
where the experiment is launched from.
"""
from pathlib import Path

import numpy as np

from heartbrain.brain import VanillaRNN


# Directory holding all saved networks, anchored to this file's location so the
# path is stable no matter the working directory the experiment is launched from.
# ``persistence.py`` lives in the source package; we climb to the project root
# (two parents up) and descend into the data directory.
# Please notice that if location of the current module is different from the 
# declared source package, be careful that the following constant point to the 
# correct directory
_SAVE_DIR = Path(__file__).resolve().parent.parent / "data" / "saved_networks"

# Uniform extension for every saved network. Callers never specify it.
_EXTENSION = ".npz"


def _resolve_path(name: str) -> Path:
    """Turn a bare network name into its full path ``.../<name>.npz``.
    Notice that the extension is manually added but Numpy add it too.
    """
    return _SAVE_DIR / (name + _EXTENSION)


def save_network(rnn: VanillaRNN, name: str) -> None:
    """Save a network's baseline identity to ``data/saved_networks/<name>.npz``.

    Reads the network's static weights through its public getters (which already
    return copies) and writes them to a uniform ``.npz`` archive.

    Parameters
    ----------
    rnn : VanillaRNN
        The network to persist. Only its baseline identity is stored.
    name : str
        Bare file name, without directory or extension.

    Raises
    ------
    FileExistsError
        If a network with this name already exists (no silent overwrite).
    """
    path = _resolve_path(name)

    if path.exists():
        raise FileExistsError(f"A saved network named '{name}' already exists.")

    # Create the data directory on first use; harmless if it already exists.
    _SAVE_DIR.mkdir(parents=True, exist_ok=True)

    np.savez(
        path,
        W_xh=rnn.W_xh,
        W_hh_baseline=rnn.W_hh_baseline,
        b_h_baseline=rnn.b_h_baseline,
        h0=rnn.initial_state,
    )


def load_network(name: str) -> VanillaRNN:
    """Load a baseline network from ``data/saved_networks/<name>.npz``.

    Reads the stored weights and rebuilds a clean, not-yet-advanced network via
    ``VanillaRNN.from_weights``. The returned network is in its baseline state:
    ``g`` is not applied and no step has been taken.

    Parameters
    ----------
    name : str
        Bare file name, without directory or extension.

    Returns
    -------
    VanillaRNN
        The reconstructed baseline network.

    Raises
    ------
    FileNotFoundError
        If no network with this name exists.
    """
    path = _resolve_path(name)

    if not path.exists():
        raise FileNotFoundError(f"No saved network named '{name}' found.")

    # Remember: ``np.load`` is a lazy loader.
    data = np.load(path)

    return VanillaRNN.from_weights(
        W_xh=data["W_xh"],
        W_hh_baseline=data["W_hh_baseline"],
        b_h_baseline=data["b_h_baseline"],
        h0=data["h0"],
    )