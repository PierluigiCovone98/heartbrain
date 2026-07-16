"""Raw run of the coupled components of the framework.

More details later...
"""
import numpy as np

from heartbrain import brain, heart, coupling
from heartbrain.infra import plotting, analysis
from heartbrain.infra.persistence import networks, experiments


# Constants
GAIN_SWEEP_EXP = "gain_sweep"
RNN_BASELINE_NAME = "gain_sweep_baseline"

# Temporary choice:
# to avoid statistichal correlation I have to use:
#   np.random.SeedSequence(SEED).spawn(n)
SEED2 = 54

# === brain
G = 2.5   # Edge of chaos !!!!

# === heart
DT = 0.01
SIGMA = 0.0

# === coupled
N_STEPS = 10000
K_HB = 0.0

# === brain signal filtering
CUTOFF_PERIOD = 100

# === measurement window
TRANSIENT_STEPS = 1000
BORDER_STEPS = 500


def main():

    # === Vanilla RNN setup ===
    # Load the VanillaRNN instance studied in the ``gain_sweep`` experiment.
    rnn = networks.load_network(RNN_BASELINE_NAME)
    print(f"OK - {RNN_BASELINE_NAME} network correctly loaded.")    # Log


    # === Parameters Setup ===
    _, arrays = experiments.load_experiment(GAIN_SWEEP_EXP)
    print(f"OK - {GAIN_SWEEP_EXP} experiment correctly loaded.")    # Log

    # To align this esperiment to the network studied in the "gain_sweep" one.
    x = arrays["x"]
    N = rnn.N
    rnn.scale_W_hh(g=G)


    # === Heart setup ===
    h = heart.Heart()


    # === Coupling Components ===
    # 
    # First we define those parameters that are required for
    # the interaction of heart and brain components.
    coupled_rng = np.random.default_rng(SEED2)

    K_baseline = coupling.create_K_baseline(
        rng=coupled_rng,
        N=N,
        fan_in=2
    )
    K_baseline_x = coupling.extract_K_baseline_x(K_baseline)
    K = coupling.build_K(k_hb=K_HB, K_baseline=K_baseline)

    # Temporal series: 
    # pre-allocate memory such that we certainly know how
    # many steps there are. 
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

    # Plotting1
    # plotting.plot_time_series(heart_time_series, brain_time_series, name="chaotic_coupled_run_10K")
    # plotting.plot_single_series(brain_time_series, "chaotic_coupled_run_10K_brain_zoom1",start=2000, end=2601)
    # plotting.plot_single_series(brain_time_series, "chaotic_coupled_run_10K_brain_zoom2",start=2000, end=2101)
    
    # plotting.plot_time_series(heart_time_series, brain_time_series, name="chaotic_coupled_run_10K_win", start=2000, end=3200)
    # plotting.plot_time_series(heart_time_series, brain_time_series, name="chaotic_coupled_run_10K_khb_03", start=2000, end=3200)
    # plotting.plot_time_series(heart_time_series, brain_time_series, name="chaotic_coupled_run_10K_khb_05", start=2000, end=3200)
    # plotting.plot_time_series(heart_time_series, brain_time_series, name="chaotic_coupled_run_10K_khb_1", start=2000, end=3200)
    # plotting.plot_time_series(heart_time_series, brain_time_series, name="chaotic_coupled_run_10K_khb_2", start=2000, end=3200)

    # === Brain time series filtering ====
    brain_slow = analysis.low_pass_filter(signal=brain_time_series, cutoff_period=CUTOFF_PERIOD)

    # Plotting2
    # plotting.plot_time_series(heart_time_series, brain_slow, name="chaotic_coupled_run_10K_khb_05_filtered1", start=2000, end=3200, brain_label="brain  s_slow(t)")


    # # === Phase extraction ===
    heart_phase = analysis.instantaneous_phase(heart_time_series)
    brain_phase = analysis.instantaneous_phase(brain_slow)

    # Plotting3: the two instantaneous phases (sawtooth)
    # plotting.plot_time_series(heart_phase, brain_phase,
    #                           name="chaotic_coupled_run_10K_khb_05_filtered1_phases1_zoomout",
    #                           start=2000, end=3200,
    #                           heart_label="phase heart",
    #                           brain_label="phase brain")


    # === Coherence ===
    heart_phase_valid = analysis.discard_borders(heart_phase, n_start= TRANSIENT_STEPS + BORDER_STEPS, n_end=BORDER_STEPS)
    brain_phase_valid = analysis.discard_borders(brain_phase, n_start= TRANSIENT_STEPS + BORDER_STEPS, n_end=BORDER_STEPS)
    plv = analysis.phase_locking_value(heart_phase_valid, brain_phase_valid)
    print(f"PLV (k_hb={K_HB}) = {plv:.4f}")


if __name__=="__main__":
    main()