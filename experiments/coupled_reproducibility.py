"""Is the network's fast dynamics still chaotic once the loop is coupled?

In the isolated heart -> brain case, the fast component of s(t) was intermittent:
reproducibility B, resolved by heart phase, had plateaus (B ~ 1, the dynamics
repeat) and valleys (B ~ 0, each cycle different — live chaos). Closing the loop
made the coupled system look increasingly periodic by eye as coupling grew. This
asks whether that apparent periodicity is real suppression of the chaos, or only
a periodic envelope over a still-chaotic content.

It runs the bidirectional system once at (k_hb, k_bh) and measures B(phase) on the
fast component of s(t) — the same quantity studied in isolation, so the reading
is on familiar ground. Read in absolute, no baseline comparison: valleys in
B(phase) mean the chaos is still alive; B high everywhere means the loop has
ordered the fast dynamics into repetition.

Three things differ from the isolated case and colour the reading, without
breaking the measure: the phase comes from a heart influenced by the loop; the slow
component of s(t) is a ramp, not a sinusoid; and the segmenting phase is itself
part of the loop (circularity). The null control (B on random, non-adjacent
cycles) is computed too: in the loop it may reveal temporal structure — e.g.
repetition only with the adjacent cycle, not across the run — that the isolated
case did not show.
"""
import numpy as np

from heartbrain import heart, coupling, coupled_system
from heartbrain.infra import analysis, plotting
from heartbrain.infra.persistence import networks, experiments


# Constants
GAIN_SWEEP_EXP = "gain_sweep"
RNN_BASELINE_NAME = "gain_sweep_baseline"

# Distinct seeds for the two channels.
SEED2 = 54
SEED3 = 77

# === brain
G = 2.5

# === heart
DT = 0.01

# === coupled
N_STEPS = 10000
K_HB = 1.0
K_BH = 1.0
SUBDIR = "coupled_reproducibility"

# === fast dynamics / reproducibility (same as the isolated case)
CUTOFF_PERIOD = 100
TRANSIENT_STEPS = 1000
BORDER_STEPS = 500
N_WINDOWS = 18
WINDOW_LENGTH = 33
NULL_SEED = 99


def _phase_bin_centers(n_bins: int) -> np.ndarray:
    """Return the phase at the center of each of ``n_bins`` equal bins over [-π, π]."""
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    return (edges[:-1] + edges[1:]) / 2.0


def main():

    # === Vanilla RNN setup ===
    rnn = networks.load_network(RNN_BASELINE_NAME)
    print(f"OK - {RNN_BASELINE_NAME} network correctly loaded.")    # Log

    _, arrays = experiments.load_experiment(GAIN_SWEEP_EXP)
    print(f"OK - {GAIN_SWEEP_EXP} experiment correctly loaded.")    # Log

    x = arrays["x"]
    N = rnn.N
    rnn.scale_W_hh(g=G)


    # === Heart setup ===
    h = heart.Heart()


    # === Coupling components ===
    heart_to_brain_rng = np.random.default_rng(SEED2)
    K_baseline = coupling.create_K_baseline(rng=heart_to_brain_rng, N=N, fan_in=2)
    K_baseline_x = coupling.extract_K_baseline_x(K_baseline)
    K = coupling.build_K(k_hb=K_HB, K_baseline=K_baseline)

    brain_to_heart_rng = np.random.default_rng(SEED3)
    D_baseline = coupling.create_D_baseline(rng=brain_to_heart_rng, N=N)


    # === Bidirectional run ===
    heart_series, brain_signal_series, _ = coupled_system.run_coupled(rnn=rnn,
                                                                      oscillator=h,
                                                                      x=x,
                                                                      K=K,
                                                                      K_baseline_x=K_baseline_x,
                                                                      D_baseline=D_baseline,
                                                                      k_bh=K_BH,
                                                                      n_steps=N_STEPS,
                                                                      dt=DT)


    # === Fast component of s(t), and heart phase ===
    heart_phase = analysis.instantaneous_phase(heart_series)
    brain_fast = analysis.high_pass_filter(brain_signal_series, cutoff_period=CUTOFF_PERIOD)

    # Trim borders on both, aligned, to drop Hilbert/filter edge artifacts.
    n_start = TRANSIENT_STEPS + BORDER_STEPS
    heart_phase = analysis.discard_borders(heart_phase, n_start=n_start, n_end=BORDER_STEPS)
    brain_fast = analysis.discard_borders(brain_fast, n_start=n_start, n_end=BORDER_STEPS)


    # === Reproducibility B, resolved by heart phase ===
    window_phases = _phase_bin_centers(n_bins=N_WINDOWS)
    reproducibility = np.zeros(N_WINDOWS)
    reproducibility_null = np.zeros(N_WINDOWS)
    null_rng = np.random.default_rng(NULL_SEED)

    print(f"\n(k_hb={K_HB}, k_bh={K_BH})")    # Log
    print(f"{'phase':>8} {'B':>8} {'B_null':>8} {'n_win':>7}")    # Log
    for i, target_phase in enumerate(window_phases):
        windows = analysis.extract_windows_at_phase(signal=brain_fast,
                                                    phase=heart_phase,
                                                    target_phase=target_phase,
                                                    window_length=WINDOW_LENGTH)

        reproducibility[i], reproducibility_null[i] = analysis.cycle_reproducibility(
            windows=windows,
            rng=null_rng)

        print(f"{target_phase:>8.3f} {reproducibility[i]:>8.3f} "
              f"{reproducibility_null[i]:>8.3f} {windows.shape[0]:>7}")    # Log


    # === Plot: B and B_null vs heart phase ===
    plotting.plot_two_panels(x_values=window_phases,
                             top_values=reproducibility,
                             bottom_values=reproducibility_null,
                             name="coupled_reproducibility_khb_1_kbh_1",
                             subdir=SUBDIR,
                             x_label="heart phase (rad)",
                             top_label="B (consecutive cycles)",
                             bottom_label="B (null: random cycles)")


if __name__ == "__main__":
    main()