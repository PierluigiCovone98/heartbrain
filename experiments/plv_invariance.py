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
positive control: it must return the same PLV as ``coupled_run`` (0.9945 at
k_hb=0.5, 0.0264 at k_hb=0). If it does not, this bench is not the same
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
G = 2.04

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
    rnn = networks.load_network(RNN_BASELINE_NAME)
    print(f"OK - {RNN_BASELINE_NAME} network correctly loaded.")    # Log


    # === Parameters Setup ===
    _, arrays = experiments.load_experiment(GAIN_SWEEP_EXP)
    print(f"OK - {GAIN_SWEEP_EXP} experiment correctly loaded.")    # Log

    x = arrays["x"]
    N = rnn.N
    rnn.scale_W_hh(g=G)


    # === Heart setup ===
    h = heart.Heart()


    # === Coupling Components ===
    coupled_rng = np.random.default_rng(SEED2)

    K_baseline = coupling.create_K_baseline(
        rng=coupled_rng,
        N=N,
        fan_in=2
    )
    K_baseline_x = coupling.extract_K_baseline_x(K_baseline)
    K = coupling.build_K(k_hb=K_HB, K_baseline=K_baseline)

    # Temporal series
    heart_time_series = np.zeros(N_STEPS)
    brain_time_series = np.zeros(N_STEPS)

    # Let's implement one actual interaction (directed).
    # We lose the last state; don't care on a high number of steps.
    for t in range(N_STEPS):
        
        h_state = h.get_state()

        # "h_state[0]" beacuse "h_state := (x,y)".
        heart_time_series[t] = h_state[0]
        # s(t) = k_baseline_x @ b_state
        brain_time_series[t] = rnn.project_state_onto(direction=K_baseline_x)

        # Forward step of the system
        perturbation = coupling.heart_to_brain_bias_perturbation(K=K, heart_state=h_state)
        rnn.apply_bias_perturbation(perturbation)
        rnn.step(x=x)
        h.step(sigma=SIGMA, dt=DT)


    # === Brain time series filtering ====
    brain_slow = analysis.low_pass_filter(signal=brain_time_series, cutoff_period=CUTOFF_PERIOD)


    # === Phase extraction ===
    heart_phase = analysis.instantaneous_phase(heart_time_series)
    brain_phase = analysis.instantaneous_phase(brain_slow)

    # === Coherence ===
    heart_phase_valid = analysis.discard_borders(heart_phase, n_start= TRANSIENT_STEPS + BORDER_STEPS, n_end=BORDER_STEPS)
    brain_phase_valid = analysis.discard_borders(brain_phase, n_start= TRANSIENT_STEPS + BORDER_STEPS, n_end=BORDER_STEPS)
    plv = analysis.phase_locking_value(heart_phase_valid, brain_phase_valid)
    print(f"PLV (k_hb={K_HB}) = {plv:.4f}")


if __name__=="__main__":
    main()