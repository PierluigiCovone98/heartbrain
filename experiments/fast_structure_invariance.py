"""Are ``amplitude by phase`` and ``reproducibility by phase`` structural invariant?

This experiment is designed to state if the two directions of the ``second axis`` are
invariant as the ``PLV``. 
"""
import numpy as np

from heartbrain import heart, coupling, coupled_system, brain
from heartbrain.infra import analysis, plotting
from heartbrain.infra.persistence import networks, experiments

# Constants
GAIN_SWEEP_EXP = "gain_sweep"
RNN_BASELINE_NAME = "gain_sweep_baseline"

# === coupling componens
SEED2 = 54

# === brain
G = 2.5
STATE_PERTURBATION = 1e-13

# === heart
DT = 0.01
SIGMA = 0.0

# === coupled
N_STEPS = 50000
K_HB = 1.0
SUBDIR = "structure"

# === brain signal filtering
CUTOFF_PERIOD = 100

# === measurement window
TRANSIENT_STEPS = 1000
BORDER_STEPS = 500

# === resolve by phase
N_BINS = 36

# === reproducibility by phase
N_WINDOWS = 18
WINDOW_LENGTH = 33
NULL_SEED = 99


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

    
    # === Networks setup ===
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


    # === Two runs, differing only by h0 ===
    # The fast dynamics itself will not survive the perturbation — the network is
    # chaotic, so the two trajectories diverge completely. The question is
    # whether the *measures* built on it do: if they hold while the trajectory
    # underneath them does not, they describe the structure of the coupling
    # rather than the accident of one path.
    oscillator.reset_state()
    amp_A, repro_A = _measure_structure(rnn=rnn_h0_base,
                                        oscillator=oscillator,
                                        x=x,
                                        K=K,
                                        K_baseline_x=K_baseline_x)

    oscillator.reset_state()
    amp_B, repro_B = _measure_structure(rnn=rnn_h0_pert,
                                        oscillator=oscillator,
                                        x=x,
                                        K=K,
                                        K_baseline_x=K_baseline_x)


    # === Invariance check ===
    #
    # Each measure is compared against its own scale: the amplitude against its
    # typical value, the reproducibility against its full range — a shift of 0.1
    # means something different for a quantity spanning 0..4 than for one
    # spanning 0..1.
    amp_max_diff = np.nanmax(np.abs(amp_A - amp_B))
    amp_scale = np.nanmean(amp_A)

    repro_max_diff = np.nanmax(np.abs(repro_A - repro_B))
    repro_range = np.nanmax(repro_A) - np.nanmin(repro_A)

    print()
    print(f"==== Second-axis invariance (k_hb={K_HB}, G={G}, perturbation={STATE_PERTURBATION:.0e}) ====")
    print(f"amplitude       max|A-B| = {amp_max_diff:.6f}   scale = {amp_scale:.4f}   "
          f"relative = {amp_max_diff / amp_scale:.4f}")
    print(f"reproducibility max|A-B| = {repro_max_diff:.6f}   range = {repro_range:.4f}   "
          f"relative = {repro_max_diff / repro_range:.4f}")
    print(f"correlation(amp_A, amp_B)     = {np.corrcoef(amp_A, amp_B)[0, 1]:.6f}")
    print(f"correlation(repro_A, repro_B) = {np.corrcoef(repro_A, repro_B)[0, 1]:.6f}")


    # === Plots: the two curves, run against run ===
    window_phases = _phase_bin_centers(n_bins=N_WINDOWS)

    plotting.plot_two_panels(x_values=window_phases,
                             top_values=repro_A,
                             bottom_values=repro_B,
                             name="invariance_reproducibility_by_phase_khb_1",
                             subdir=SUBDIR,
                             x_label="heart phase (rad)",
                             top_label="B (h0 base)",
                             bottom_label="B (h0 perturbed)")

    plotting.plot_two_panels(x_values=window_phases,
                             top_values=amp_A,
                             bottom_values=amp_B,
                             name="invariance_amplitude_by_phase_khb_1",
                             subdir=SUBDIR,
                             x_label="heart phase (rad)",
                             top_label="amplitude (h0 base)",
                             bottom_label="amplitude (h0 perturbed)")
    

def _measure_structure(rnn: brain.VanillaRNN,
                       oscillator: heart.Heart,
                       x: np.ndarray,
                       K: np.ndarray,
                       K_baseline_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run once and return the two phase-resolved measures.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(amplitude_by_phase, reproducibility_by_phase)``, both of length
        ``N_WINDOWS``.
    """
    heart_series, brain_series = coupled_system.run_heart_to_brain(rnn=rnn,
                                                                   oscillator=oscillator,
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
    n_start = TRANSIENT_STEPS + BORDER_STEPS
    heart_phase = analysis.discard_borders(heart_phase, n_start=n_start, n_end=BORDER_STEPS)
    amplitude = analysis.discard_borders(amplitude, n_start=n_start, n_end=BORDER_STEPS)
    brain_fast_valid = analysis.discard_borders(brain_fast, n_start=n_start, n_end=BORDER_STEPS)

    # We do not need amplitude std
    amp_by_phase, _ = analysis.resolve_by_phase(amplitude, heart_phase, n_bins=N_WINDOWS)


    window_phases = _phase_bin_centers(n_bins=N_WINDOWS)
    reproducibility = np.zeros(N_WINDOWS)
    null_rng = np.random.default_rng(NULL_SEED)

    for i, target_phase in enumerate(window_phases):
        windows = analysis.extract_windows_at_phase(signal=brain_fast_valid,
                                                    phase=heart_phase,
                                                    target_phase=target_phase,
                                                    window_length=WINDOW_LENGTH)
        reproducibility[i], _ = analysis.cycle_reproducibility(windows=windows, rng=null_rng)

    return (amp_by_phase, reproducibility)


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