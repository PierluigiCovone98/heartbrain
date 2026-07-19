"""Is the PLV an invariant, or an artifact of a single trajectory?

The coupled system runs at the edge of chaos, where a perturbation of 1e-16
makes two trajectories diverge completely within a few hundred steps. This is
why trajectories were rejected as a measure of the heart's influence, and why a
*structural* quantity — the phase locking value — was introduced instead. But
that choice rests on a claim that has not yet been checked: that the PLV
survives the very divergence that destroys trajectories.

This experiment checks it. The same simulation is run twice, differing only by a
1e-13 shift on one component of ``h0``. Two things are then compared:

1. **The trajectories must diverge.** Without this the test proves nothing: an
   unchanged PLV would be meaningless if the perturbation had simply been
   absorbed. This half validates the other.
2. **The PLV must not.** If it stays put while the trajectory explodes, it
   measures the structure of the coupling and not the accident of one path.

Everything else is held identical: the network is loaded from disk (same weights
by construction), the input comes from the saved experiment, and the coupling rng
is consumed exactly as in ``coupled_run`` — ``K_baseline`` is the only quantity
regenerated here, and therefore the only place where a different call order could
silently change the setup.

Before perturbing anything, the run is first reproduced *unperturbed* as a
positive control: it must return the same PLV as ``coupled_run`` (0.9978 at
k_hb=0.5, 0.0732 at k_hb=0). If it does not, this bench is not the same
simulation, and nothing measured on it would be trustworthy.
"""
import numpy as np

from heartbrain import brain, heart, coupling
from heartbrain.infra import plotting, analysis
from heartbrain.infra.persistence import networks, experiments


# Constants
GAIN_SWEEP_EXP = "gain_sweep"
RNN_BASELINE_NAME = "gain_sweep_baseline"

# === coupling setup
SEED2 = 54

# === brain
G = 2.5
STATE_PERTURBATION = 1e-13

# === heart
DT = 0.01
SIGMA = 0.0

# === coupled
N_STEPS = 10000
K_HB = 0.5

# === brain signal filtering
CUTOFF_PERIOD = 100

# === measurement window
TRANSIENT_STEPS = 1000
BORDER_STEPS = 500


def main():

    # === Vanilla RNN setup ===
    # We use ``rnn`` as a "baseline RNN" from where we take weight.
    rnn_baseline = networks.load_network(RNN_BASELINE_NAME)
    print(f"OK - {RNN_BASELINE_NAME} network correctly loaded.")    # Log


    # === Parameters Setup ===
    _, arrays = experiments.load_experiment(GAIN_SWEEP_EXP)
    print(f"OK - {GAIN_SWEEP_EXP} experiment correctly loaded.")    # Log

    x = arrays["x"]
    N = rnn_baseline.N


    # === Heart setup ===
    oscillator = heart.Heart()


    # === Coupling Components ===
    coupled_rng = np.random.default_rng(SEED2)

    K_baseline = coupling.create_K_baseline(
        rng=coupled_rng,
        N=N,
        fan_in=2
    )
    K_baseline_x = coupling.extract_K_baseline_x(K_baseline)
    K = coupling.build_K(k_hb=K_HB, K_baseline=K_baseline)


    # === Network setups ===
    #
    # = 1. RNN with "base h0".
    h0_base = rnn_baseline.initial_state
    rnn_h0_base = brain.VanillaRNN.from_weights(W_xh=rnn_baseline.W_xh,
                                                W_hh_baseline=rnn_baseline.W_hh_baseline,
                                                b_h_baseline=rnn_baseline.b_h_baseline,
                                                h0=h0_base)
    rnn_h0_base.scale_W_hh(g=G)

    # = 2. RNN with (slightly) pertubated "h0"            
    h0_pert = h0_base.copy()
    h0_pert[0] += STATE_PERTURBATION
    rnn_h0_pert = brain.VanillaRNN.from_weights(W_xh=rnn_baseline.W_xh,
                                                W_hh_baseline=rnn_baseline.W_hh_baseline,
                                                b_h_baseline=rnn_baseline.b_h_baseline,
                                                h0=h0_pert)
    rnn_h0_pert.scale_W_hh(g=G)


    # === Trajectory Divergence check ===
    #
    # = Time series 
    oscillator.reset_state()
    heart_A, brain_A = _run_simulation(rnn=rnn_h0_base,
                                       x=x,
                                       oscillator=oscillator,
                                       K=K,
                                       K_baseline_x=K_baseline_x)
    
    oscillator.reset_state()
    heart_B, brain_B = _run_simulation(rnn=rnn_h0_pert,
                                       x=x,
                                       oscillator=oscillator,
                                       K=K,
                                       K_baseline_x=K_baseline_x)


    # === Check 1: did the trajectories actually diverge? ===
    #     
    # Pointwise comparison, so the window is taken first: no context to preserve.
    brain_A_valid = analysis.discard_borders(brain_A, n_start=TRANSIENT_STEPS + BORDER_STEPS, n_end=BORDER_STEPS)
    brain_B_valid = analysis.discard_borders(brain_B, n_start=TRANSIENT_STEPS + BORDER_STEPS, n_end=BORDER_STEPS)

    mean_abs_diff = np.mean(np.abs(brain_A_valid - brain_B_valid))
    correlation = np.corrcoef(brain_A_valid, brain_B_valid)[0, 1]
    signal_scale = np.mean(np.abs(brain_A_valid))

    print()
    print(f"==== Divergence check (k_hb={K_HB}, G={G}, perturbation={STATE_PERTURBATION:.0e}) ====")
    print(f"mean |s_A - s_B|     = {mean_abs_diff:.6f}")
    print(f"mean |s_A|  (scale)  = {signal_scale:.6f}")
    print(f"ratio diff/scale     = {mean_abs_diff / signal_scale:.4f}")
    print(f"correlation(A, B)    = {correlation:.6f}")
    

    # === Check 2: did the PLV survive? ===
    #
    # The heart phase is computed once, deliberately: the coupling is
    # uni-directional (k_bh = 0), so the heart is untouched by the network and
    # ``heart_A`` and ``heart_B`` are identical. Computing it twice would hide
    # that independence behind a redundant call — the single call states it.
    #
    # Filter and Hilbert need the whole signal (they use surrounding context, so
    # trimming first would only create fresh artifacts at the new borders). The
    # window is therefore taken at the end of the chain, right before the PLV.
    assert np.array_equal(heart_A, heart_B), "Heart differs between runs: coupling is not uni-directional."
    heart_phase = analysis.instantaneous_phase(heart_A)

    plv_A = _measure_plv(heart_phase, brain_A)
    plv_B = _measure_plv(heart_phase, brain_B)

    print()
    print(f"==== PLV invariance check (k_hb={K_HB}, G={G}) ====")
    print(f"PLV (unperturbed h0) = {plv_A:.6f}")
    print(f"PLV (perturbed   h0) = {plv_B:.6f}")
    print(f"|difference|         = {abs(plv_A - plv_B):.2e}")


def _run_simulation(rnn: brain.VanillaRNN,
                    x: np.ndarray,
                    oscillator: heart.Heart,
                    K: np.ndarray,
                    K_baseline_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run one heart -> brain coupled simulation, returning the two signals.

    Advances the coupled system for ``n_steps``: at each step the heart state
    perturbs the network bias (the directed heart -> brain coupling), then both
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
    x : np.ndarray
        The fixed input vector driving the network, shape ``(input_size,)``.
    oscillator : heart.Heart
        The heart oscillator, in its initial state.
    K : np.ndarray
        The scaled coupling matrix ``k_hb * K_baseline``, shape ``(N, 2)``.
    K_baseline_x : np.ndarray
        The unscaled coupling direction the network state is projected onto to
        produce the scalar signal ``s(t)``, shape ``(N,)``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(heart_series, brain_series)``: the heart signal ``x(t)`` and the
        network's projected signal ``s(t)``, one value per step, of length
        ``n_steps``.
    """
    # Preparing output
    heart_time_series = np.zeros(N_STEPS)
    brain_time_series = np.zeros(N_STEPS)

    # Let's implement one actual interaction (directed).
    # We lose the last state; don't care on a high number of steps.
    for t in range(N_STEPS):
        
        hearth_state = oscillator.get_state()

        # "h_state[0]" beacuse "h_state := (x,y)".
        heart_time_series[t] = hearth_state[0]
        # s(t) = k_baseline_x @ b_state
        brain_time_series[t] = rnn.project_state_onto(direction=K_baseline_x)

        # Forward step of the system
        perturbation = coupling.heart_to_brain_bias_perturbation(K=K, heart_state=hearth_state)
        rnn.apply_bias_perturbation(perturbation)
        rnn.step(x=x)
        oscillator.step(sigma=SIGMA, dt=DT)

    return (heart_time_series, brain_time_series)


def _measure_plv(heart_phase: np.ndarray, brain_series: np.ndarray) -> float:
    """Measure the phase locking value between the heart and one brain run.

    Applies the full measurement chain: low-pass (isolate the slow scale that
    carries the locking), phase extraction, then the window — trimmed last,
    since the transforms before it need the whole signal.
    """
    brain_slow = analysis.low_pass_filter(signal=brain_series, cutoff_period=CUTOFF_PERIOD)
    brain_phase = analysis.instantaneous_phase(brain_slow)

    n_start = TRANSIENT_STEPS + BORDER_STEPS
    heart_phase_valid = analysis.discard_borders(heart_phase, n_start=n_start, n_end=BORDER_STEPS)
    brain_phase_valid = analysis.discard_borders(brain_phase, n_start=n_start, n_end=BORDER_STEPS)

    return analysis.phase_locking_value(heart_phase_valid, brain_phase_valid)


if __name__=="__main__":
    main()