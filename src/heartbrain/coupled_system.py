"""Coupled-system module.

Orchestrates full simulations of the ``heart-brain`` coupled system over time. This
sits one level above ``heartbrain/coupling``: where ``coupling`` provides the per-instant
building blocks (how one signal is projected into the other at a single step),
this module runs the loop — advancing both components for many steps and
collecting their signals.

The functions are named by ``coupling direction``, because the system is built up
one direction at a time. ``run_heart_to_brain`` is the uni-directional case where
the heart drives the network and receives nothing back; ``run_brain_to_heart`` is
its mirror, where the chaotic network drives the heart. The two are separate
functions rather than one parametrized loop because the loops genuinely differ:
in the heart -> brain case the forcing term ``sigma`` is a fixed parameter (zero),
while in the brain -> heart case it is dynamic — recomputed every step from the
network's current state. The bidirectional case, where both channels are on at
once, is a further separate problem and will get its own function when needed.

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


def run_brain_to_heart(rnn: brain.VanillaRNN,
                       oscillator: heart.Heart,
                       x: np.ndarray,
                       D_baseline: np.ndarray,
                       k_bh: float,
                       n_steps: int,
                       dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Run one brain -> heart coupled simulation, returning the two signals.

    The mirror of ``run_heart_to_brain``: here the network drives the heart and
    receives nothing back (the heart -> brain channel is off). At each step the
    network state is projected onto ``D_baseline`` and scaled by ``k_bh`` to form
    the scalar ``sigma`` that forces the heart, then both components step forward.

    Unlike the heart -> brain case, where ``sigma`` was a fixed parameter (zero),
    here ``sigma`` is *dynamic*: recomputed every step from the network's current
    state, since that is what the heart receives. This is why the reverse
    direction is a separate function rather than a parameter change — the loop
    itself differs.

    On what is recorded: the network series is its projection onto the *unscaled*
    ``D_baseline`` — what the network sends along the brain -> heart channel at
    unit intensity, independent of ``k_bh`` — mirroring how ``run_heart_to_brain``
    records ``s(t)`` on ``K_baseline_x`` without ``k_hb``. The ``k_bh`` scaling is
    applied only where the signal enters the heart, not to the recorded series, so
    runs at different ``k_bh`` stay comparable (the intensity belongs in the plot
    legend, not baked into the curve).

    The function is a pure orchestrator: it does not build or reset its
    components. The caller passes an ``rnn`` and an ``oscillator`` already in
    their intended initial state and owns everything that must vary between runs.

    Parameters
    ----------
    rnn : brain.VanillaRNN
        The network, already built and gain-scaled, in its initial state.
    oscillator : heart.Heart
        The heart oscillator, in its initial state.
    x : np.ndarray
        The fixed input vector driving the network, shape ``(input_size,)``.
    D_baseline : np.ndarray
        The unscaled brain -> heart projection direction, shape ``(N,)``. The
        network state is projected onto this to produce the signal; the heart
        then receives it scaled by ``k_bh``.
    k_bh : float
        Brain -> heart coupling strength. Scales the projection before it enters
        the heart. Zero means no coupling: the heart runs free.
    n_steps : int
        Number of simulation steps.
    dt : float
        Heart integration step size.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(heart_series, brain_series)``: the heart signal ``x(t)`` and the
        network's projection onto ``D_baseline`` (unscaled), one value per step,
        of length ``n_steps``. Here the heart series is the one of interest — it
        is the component being driven — while the brain series shows what the
        network is sending, at unit intensity.
    """
    # Preparing output
    heart_time_series = np.zeros(n_steps)
    brain_time_series = np.zeros(n_steps)

    # We lose the last state; don't care on a high number of steps.
    for t in range(n_steps):

        heart_state = oscillator.get_state()

        # "heart_state[0]" because "heart_state := (x, y)".
        heart_time_series[t] = heart_state[0]
        # The network projected onto the unscaled brain→heart channel: what it
        # sends at unit intensity, independent of k_bh (mirrors s(t) on K_baseline_x).
        brain_time_series[t] = rnn.project_state_onto(direction=D_baseline)

        # The heart receives that projection scaled by the coupling strength.
        sigma = k_bh * brain_time_series[t]

        # Forward step of the system
        oscillator.step(sigma=sigma, dt=dt)
        rnn.step(x=x)

    return (heart_time_series, brain_time_series)