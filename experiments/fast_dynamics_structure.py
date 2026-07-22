"""How is the network's fast dynamics structured across the heart's cycle?

At a fixed k_hb, isolates the network's fast component (high-pass) and resolves
its properties against heart phase, to see whether the cycle modulates them —
and if so, how. Starts with amplitude by phase; chaoticity by phase follows.
"""
import numpy as np

from heartbrain import heart, coupling, coupled_system
from heartbrain.infra import analysis, plotting
from heartbrain.infra.persistence import networks, experiments


# Constants
GAIN_SWEEP_EXP = "gain_sweep"
RNN_BASELINE_NAME = "gain_sweep_baseline"

# === coupling componens
SEED2 = 54

# === brain
G = 2.5

# === heart
DT = 0.01
SIGMA = 0.0

# === coupled
N_STEPS = 10000
K_HB = 1.0
SUBDIR = "structure"

# === brain signal filtering
CUTOFF_PERIOD = 100

# === measurement window
TRANSIENT_STEPS = 1000
BORDER_STEPS = 500

# === resolve by phase
N_BINS = 36


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
    heart_series, brain_series = coupled_system.run_heart_to_brain(rnn=rnn,
                                                                   oscillator=h,
                                                                   x=x,
                                                                   K=K,
                                                                   K_baseline_x=K_baseline_x,
                                                                   n_steps=N_STEPS,
                                                                   dt=DT,
                                                                             sigma=SIGMA)


    heart_phase = analysis.instantaneous_phase(heart_series)

    # Extract the fast component from the ``brain_series``
    brain_fast = analysis.high_pass_filter(brain_series, cutoff_period=CUTOFF_PERIOD)
    amplitude = analysis.instantaneous_amplitude(brain_fast)

    # trim borders (Hilbert + filter artifacts) BEFORE resolving
    heart_phase = analysis.discard_borders(heart_phase, n_start=TRANSIENT_STEPS + BORDER_STEPS, n_end=BORDER_STEPS)
    amplitude   = analysis.discard_borders(amplitude,   n_start=TRANSIENT_STEPS + BORDER_STEPS, n_end=BORDER_STEPS)

    amp_mean, amp_std = analysis.resolve_by_phase(amplitude, heart_phase, n_bins=N_BINS)

    # Resolve the heart's own x(t) by phase too, as a readable reference: it maps
    # each abstract phase value back to where it sits in the heartbeat.
    heart_valid = analysis.discard_borders(heart_series, n_start=TRANSIENT_STEPS + BORDER_STEPS, n_end=BORDER_STEPS)
    heart_by_phase, _ = analysis.resolve_by_phase(heart_valid, heart_phase, n_bins=N_BINS)
    
    
    # Phase-bin centers for the x-axis: midpoint of each [-pi, pi] bin.
    bin_centers = _phase_bin_centers(n_bins=N_BINS)

    # plotting.plot_two_panels(x_values=bin_centers,
    #                          top_values=heart_by_phase,
    #                          bottom_values=amp_mean,
    #                          name="fast_amplitude_by_phase_khb_1_twopanels",
    #                          subdir=SUBDIR,
    #                          x_label="heart phase (rad)",
    #                          top_label="heart x (mean)",
    #                          bottom_label="fast amplitude (mean)")

    plotting.plot_two_panels(x_values=bin_centers,
                             top_values=amp_std,
                             bottom_values=amp_mean,
                             name="fast_amp_std_vs_amp_mean_by_phase_khb_1",
                             subdir=SUBDIR,
                             x_label="heart phase (rad)",
                             top_label="fast amplitude (std)",
                             bottom_label="fast amplitude (mean)")


def _phase_bin_centers(n_bins: int) -> np.ndarray:
    """Return the phase value at the center of each of ``n_bins`` bins.

    ``resolve_by_phase`` returns one value per bin but not where each bin sits in
    phase; this reconstructs those positions — the midpoint of each equal bin
    over ``[-π, π]`` — for use as the x-axis when plotting a phase-resolved
    quantity.

    Parameters
    ----------
    n_bins : int
        Number of phase bins (must match the value passed to
        ``resolve_by_phase``).

    Returns
    -------
    np.ndarray
        The ``n_bins`` bin-center phases, in radians, ascending.
    """
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    return (edges[:-1] + edges[1:]) / 2


if __name__=="__main__":
    main()