"""Coupling module.

Collects the functions that route signals between framework components,
each in its own direction.

The heart lives in a 2-D phase space (its state is ``(x, y)``); the brain's
bias lives in an N-dimensional space. The coupling maps ``2 -> N`` through a
projection matrix ``K`` of shape ``(N, 2)``: row ``i`` encodes how neuron ``i``
"feels" the heart, i.e. how much it weighs the heart's position ``x`` and
velocity ``y``. Following the project convention, ``K`` is split into a fixed
random baseline ``K_baseline`` and a scalar gain ``k_hb`` that sets the coupling
intensity, exactly as ``W_hh = g * W_hh_baseline`` in the brain.
"""
import numpy as np

from heartbrain import brain


def create_K_baseline(rng: np.random.Generator, N: int, fan_in: int) -> np.ndarray:
    """Random scaled initialization of the coupling baseline matrix.

    Draws ``K_baseline`` from a normal distribution scaled by ``1/sqrt(fan_in)``,
    the same rule used for the brain's weights. This keeps the coupling in the
    same statistical register as ``W_xh`` and ``W_hh`` so that later, the effect
    of the heart can be attributed to the coupling intensity ``k_hb`` rather than
    to a mis-scaled matrix.

    Parameters
    ----------
    rng : np.random.Generator
        Generator used to draw the matrix. The point in the stream where this is
        called affects reproducibility, so the call order must be fixed.
    N : int
        Dimension of the brain's hidden state (rows of ``K``).
    fan_in : int
        Dimension of the source space (columns of ``K``); ``2`` for the heart.

    Returns
    -------
    np.ndarray
        The baseline coupling matrix, shape ``(N, fan_in)``.
    """
    return rng.normal( size=(N, fan_in), scale=brain.weight_std(input_size=fan_in) )


def extract_K_baseline_x(K_baseline: np.ndarray) -> np.ndarray:
    """Project the coupling baseline matrix ``K_baseline`` onto its first column."""
    return K_baseline[:,0]


def build_K(k_hb: float, K_baseline: np.ndarray) -> np.ndarray:
    """Build the coupling matrix ``K`` by scaling the baseline by the gain.

    Parameters
    ----------
    k_hb : float
        Coupling intensity; how strongly the heart influences the brain.
    K_baseline : np.ndarray
        Baseline coupling matrix, shape ``(N, fan_in)``, scaled by ``1/sqrt(fan_in)``.

    Returns
    -------
    np.ndarray
        The scaled coupling matrix ``K = k_hb * K_baseline``, shape ``(N, fan_in)``.
    """
    return k_hb * K_baseline


def heart_to_brain_bias_perturbation(K: np.ndarray, heart_state: tuple[float, float]) -> np.ndarray:
    """Project the heart state into an N-dimensional bias perturbation.

    Parameters
    ----------
    K : np.ndarray
        Scaled coupling matrix, shape ``(N, 2)``.
    heart_state : tuple[float, float]
        The heart's ``(x, y)`` state.

    Returns
    -------
    np.ndarray
        The bias perturbation, shape ``(N,)``.
    """
    return K @ np.array(heart_state)


def create_D_baseline(rng: np.random.Generator, N: int) -> np.ndarray:
    """Create the fixed random direction the brain state is projected onto for
    the brain->heart coupling.

    This is the N->1 projection that turns the network state into the single scalar
    the heart receives (``sigma = D · h``).

    The scale is ``1/sqrt(N)``: ``sigma`` is a sum of N random terms, which would grow
    like sqrt(N) with the dimension; normalizing by ``1/sqrt(N)`` keeps the projection
    magnitude controlled regardless of network size.

    This direction must be **independent** of ``K_baseline_x`` (the heart->brain
    channel): the two coupling directions are physically distinct pathways
    (afferent vs efferent), and sharing one would couple the two channels by
    construction, making it impossible to attribute a bidirectional effect to the
    interaction rather than to the overlap.

    Parameters
    ----------
    rng : np.random.Generator
        Random generator. Use a seed distinct from the one behind
        ``K_baseline``, to keep the two channels independent.
    N : int
        Number of network units (the dimension of the brain state).

    Returns
    -------
    np.ndarray
        The projection direction, shape ``(N,)``.
    """
    return rng.normal( size=N, scale=brain.weight_std(input_size=N) )