"""Coupled-system module.

TODO: expand when brain→heart is added.
Orchestrates full simulations of the ``heart-brain`` coupled system over time. This
sits one level above ``heartbrain/coupling``: where ``coupling`` provides the per-instant
building blocks (how one signal is projected into the other at a single step),
this module runs the loop — advancing both components for many steps and
collecting their signals.

The functions are named by ``coupling direction``, because the system is built up
one direction at a time. ``run_heart_to_brain`` is the uni-directional case
where the heart drives the network and receives nothing back. The reverse
direction and the bidirectional case are separate problems, with their own loop
structure, and will get their own functions when they are actually needed. 

Each function here is a pure orchestrator: it receives components already built
and initialized by the caller and only advances them. It holds no state of its
own between calls. Building, scaling, and resetting the components — everything
that must vary between runs — is the caller's responsibility.
"""
import numpy as np

from heartbrain import brain, heart, coupling


def run_heart_to_brain(rnn: brain.VanillaRNN,
                       oscillator: heart.Heart,
                       x: np.ndarray,
                       K: np.ndarray,
                       K_baseline_x: np.ndarray,
                       n_steps: int,
                       dt: float,
                       sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Run one heart -> brain coupled simulation, returning the two signals.

    Advances the coupled system for ``n_steps``: at each step the heart state
    perturbs the network bias (the directed heart→brain coupling), then both
    components step forward.

    The function is a pure orchestrator: it does not build or reset its
    components. The caller passes an ``rnn`` and an ``oscillator`` already in
    their intended initial state (built, scaled, and reset as needed) and owns
    everything that must vary between runs — which is what lets callers compare,
    for instance, two networks differing only by ``h0``. Nothing carries over
    between calls except through the objects the caller supplies.

    Parameters
    ----------
    rnn : brain.VanillaRNN
        The network, already built and gain-scaled, in its initial state.
    oscillator : heart.Heart
        The heart oscillator, in its initial state.
    x : np.ndarray
        The fixed input vector driving the network, shape ``(input_size,)``.
    K : np.ndarray
        The scaled coupling matrix ``k_hb * K_baseline``, shape ``(N, 2)``.
    K_baseline_x : np.ndarray
        The unscaled coupling direction the network state is projected onto to
        produce the scalar signal ``s(t)``, shape ``(N,)``.
    n_steps : int
        Number of simulation steps.
    dt : float
        Heart integration step size.
    sigma : float
        Brain -> heart signal for the heart's step. Zero here, since the coupling
        is uni-directional (heart -> brain only): the heart receives nothing back
        from the network. Kept as an explicit parameter, rather than hard-wired
        to zero, to stay faithful to the heart's equation (which carries a
        ``sigma`` term) and readable at the call site.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(heart_series, brain_series)``: the heart signal ``x(t)`` and the
        network's projected signal ``s(t)``, one value per step, of length
        ``n_steps``.
    """
    # Preparing output
    heart_time_series = np.zeros(n_steps)
    brain_time_series = np.zeros(n_steps)

    # We lose the last state; don't care on a high number of steps.
    for t in range(n_steps):
        
        heart_state = oscillator.get_state()

        # "h_state[0]" beacuse "h_state := (x,y)".
        heart_time_series[t] = heart_state[0]
        # s(t) = k_baseline_x @ b_state
        brain_time_series[t] = rnn.project_state_onto(direction=K_baseline_x)

        # Forward step of the system
        perturbation = coupling.heart_to_brain_bias_perturbation(K=K, heart_state=heart_state)
        rnn.apply_bias_perturbation(perturbation)
        rnn.step(x=x)
        oscillator.step(sigma=sigma, dt=dt)

    return (heart_time_series, brain_time_series)
