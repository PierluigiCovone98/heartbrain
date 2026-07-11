"""Networks module.

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
import numpy as np

from heartbrain.brain import VanillaRNN
from heartbrain.infra import _paths


# Constants
_SUBDIR = "saved_networks"  # Directory holding all saved networks
_EXTENSION = ".npz"         # Uniform extension for every saved network.
   

def save_network(rnn: VanillaRNN, name: str) -> None:
    """Save a network's baseline identity to ``root/data/saved_networks/<name>.npz``.

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
    path = _paths.resolve_path(subdir=_SUBDIR, name=name, extension=_EXTENSION)

    if path.exists():
        raise FileExistsError(f"A saved network named '{name}' already exists.")

    # Create the data directory on first use; harmless if it already exists.
    _paths.ensure_parent(path)

    np.savez(
        path,
        W_xh=rnn.W_xh,
        W_hh_baseline=rnn.W_hh_baseline,
        b_h_baseline=rnn.b_h_baseline,
        h0=rnn.initial_state,
    )


def load_network(name: str) -> VanillaRNN:
    """Load a baseline network from ``root/data/saved_networks/<name>.npz``.

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
    path = _paths.resolve_path(subdir=_SUBDIR, name=name, extension=_EXTENSION)

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