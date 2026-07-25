"""Are the two second-axis measures — amplitude and reproducibility — redundant?

The second axis turned out to have two directions: how strongly the network's
fast component swings (amplitude) and whether that dynamics repeats identically
from cycle to cycle (reproducibility, B). At k_hb=1 they came apart — high
amplitude could be either reproducible or not — which is a hint of independence,
but only at one point. Redundancy is a property that can depend on the state:
two measures may separate at one k_hb and move together everywhere else.

This experiment resolves both measures by heart phase at many values of k_hb, and
asks whether they stay distinct *across* those states. For every (phase, k_hb)
pair it holds an amplitude and a B; the correlation over all pairs, together with
a scatter of one against the other, tells whether the two are one axis or two.

The reproducibility measure is compared against a null (correlation of random,
non-adjacent cycles), kept here so the redundancy question rests on the same B
that the inspector (the ``fast_dynamics_structure`` experiment) produced.
"""
import numpy as np

from heartbrain import brain, heart, coupling
from heartbrain.infra import analysis, plotting
from heartbrain.infra.persistence import networks, experiments
from heartbrain.coupled_system import run_heart_to_brain


# Constants
GAIN_SWEEP_EXP = "gain_sweep"
RNN_BASELINE_NAME = "gain_sweep_baseline"

# === coupling setup
SEED2 = 54

# === brain
G = 2.5

# === heart
DT = 0.01
SIGMA = 0.0

# === coupled
N_STEPS = 50000

# === brain signal filtering
CUTOFF_PERIOD = 100

# === measurement window
TRANSIENT_STEPS = 1000
BORDER_STEPS = 500

# === phase resolution of the two measures
N_WINDOWS = 18
WINDOW_LENGTH = 33
NULL_SEED = 99

# === k_hb sweep for the redundancy test
K_MIN = 0.0
K_MAX = 2.0
N_KHB = 18
SUBDIR = "redundancy"


def _measure_structure(rnn: brain.VanillaRNN,
                       oscillator: heart.Heart,
                       x: np.ndarray,
                       K: np.ndarray,
                       K_baseline_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run once and return the two phase-resolved measures of the fast dynamics.

    Isolates the network's fast component, then resolves by heart phase both its
    amplitude (how strongly it swings) and its reproducibility (how much a window
    at a given phase repeats from one cycle to the next).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(amplitude_by_phase, reproducibility_by_phase)``, each of length
        ``N_WINDOWS``.
    """
    heart_series, brain_series = run_heart_to_brain(rnn=rnn,
                                                    oscillator=oscillator,
                                                    x=x,
                                                    K=K,
                                                    K_baseline_x=K_baseline_x,
                                                    n_steps=N_STEPS,
                                                    dt=DT,
                                                    sigma=SIGMA)

    heart_phase = analysis.instantaneous_phase(heart_series)
    brain_fast = analysis.high_pass_filter(brain_series, cutoff_period=CUTOFF_PERIOD)
    amplitude = analysis.instantaneous_amplitude(brain_fast)

    n_start = TRANSIENT_STEPS + BORDER_STEPS
    heart_phase = analysis.discard_borders(heart_phase, n_start=n_start, n_end=BORDER_STEPS)
    amplitude = analysis.discard_borders(amplitude, n_start=n_start, n_end=BORDER_STEPS)
    brain_fast = analysis.discard_borders(brain_fast, n_start=n_start, n_end=BORDER_STEPS)

    amplitude_by_phase, _ = analysis.resolve_by_phase(amplitude, heart_phase, n_bins=N_WINDOWS)

    window_phases = _phase_bin_centers(n_bins=N_WINDOWS)
    reproducibility = np.zeros(N_WINDOWS)
    null_rng = np.random.default_rng(NULL_SEED)
    for i, target_phase in enumerate(window_phases):
        windows = analysis.extract_windows_at_phase(signal=brain_fast,
                                                    phase=heart_phase,
                                                    target_phase=target_phase,
                                                    window_length=WINDOW_LENGTH)
        reproducibility[i], _ = analysis.cycle_reproducibility(windows=windows, rng=null_rng)

    return (amplitude_by_phase, reproducibility)


def _phase_bin_centers(n_bins: int) -> np.ndarray:
    """Return the phase at the center of each of ``n_bins`` equal bins over [-pi, pi].

    Must be called with the same ``n_bins`` used to resolve the measures, so the
    phase axis and the measured values line up.
    """
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    return (edges[:-1] + edges[1:]) / 2


def main():

    # === Vanilla RNN setup ===
    rnn = networks.load_network(RNN_BASELINE_NAME)
    rnn.scale_W_hh(g=G)
    print(f"OK - {RNN_BASELINE_NAME} network loaded and scaled (g={G}).")    # Log

    _, arrays = experiments.load_experiment(GAIN_SWEEP_EXP)
    x = arrays["x"]
    N = rnn.N

    oscillator = heart.Heart()

    coupled_rng = np.random.default_rng(SEED2)
    K_baseline = coupling.create_K_baseline(rng=coupled_rng, N=N, fan_in=2)
    K_baseline_x = coupling.extract_K_baseline_x(K_baseline)


    # === Sweep k_hb, collecting both phase-resolved measures at each value ===
    # Each row is one k_hb; each column one phase bin. The reused network and
    # heart are reset before every run (state and perturbed bias to baseline;
    # the gain-scaled W_hh stays, since g is fixed and only K changes with k_hb).
    khb_values = np.linspace(K_MIN, K_MAX, N_KHB)
    amplitude_matrix = np.zeros((N_KHB, N_WINDOWS))
    reproducibility_matrix = np.zeros((N_KHB, N_WINDOWS))

    for i, k_hb in enumerate(khb_values):
        rnn.reset_state()
        rnn.reset_bias()
        oscillator.reset_state()

        K = coupling.build_K(k_hb=k_hb, K_baseline=K_baseline)
        amp, repro = _measure_structure(rnn, oscillator, x, K, K_baseline_x)

        amplitude_matrix[i] = amp
        reproducibility_matrix[i] = repro
        print(f"k_hb = {k_hb:.3f}   done")    # Log


    # === Redundancy test: correlate the two measures across all (phase, k_hb) ===
    flat_amp = amplitude_matrix.flatten()
    flat_repro = reproducibility_matrix.flatten()
    valid = ~(np.isnan(flat_amp) | np.isnan(flat_repro))

    correlation = np.corrcoef(flat_amp[valid], flat_repro[valid])[0, 1]

    print()
    print(f"==== Redundancy of the two second-axis measures ====")
    print(f"pairs (phase x k_hb)          = {int(valid.sum())}")
    print(f"correlation(amplitude, B)     = {correlation:.4f}")


    # === Scatter: B against amplitude, one point per (phase, k_hb) ===
    # Pearson only sees a linear link; the scatter reveals a nonlinear one (e.g.
    # B high only above an amplitude threshold) that the single number would miss.
    plotting.plot_scatter(x_values=flat_amp[valid],
                          y_values=flat_repro[valid],
                          name="amplitude_vs_reproducibility",
                          subdir=SUBDIR,
                          x_label="fast amplitude (mean)",
                          y_label="reproducibility (B)")

    # === B distribution: is it bimodal (threshold-like) rather than continuous? ===
    # A near-binary B cannot be predicted by a continuous amplitude — this is a
    # non-redundancy argument on its own, and often clearer than the correlation.
    valid_repro = flat_repro[valid]
    n_low = int(np.sum(valid_repro < 0.2))
    n_mid = int(np.sum((valid_repro >= 0.2) & (valid_repro <= 0.8)))
    n_high = int(np.sum(valid_repro > 0.8))
    total = valid_repro.size

    print()
    print(f"B distribution (bimodality check):")
    print(f"  B < 0.2        = {n_low:4d}  ({100*n_low/total:.1f}%)")
    print(f"  0.2 <= B <= 0.8 = {n_mid:4d}  ({100*n_mid/total:.1f}%)")
    print(f"  B > 0.8        = {n_high:4d}  ({100*n_high/total:.1f}%)")


if __name__ == "__main__":
    main()
