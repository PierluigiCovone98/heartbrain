"""Does the causal heart-brain alignment fluctuate, or is it a metronome?

The three-factor rule needs a modulator r(t) available *inside the loop*, at step
t, from past data only. The PLV cannot serve: ``instantaneous_phase`` is a Hilbert
transform over the whole signal, non-causal by construction, and it requires a
narrow-band signal, which forces the low-pass that discards the chaotic component
— the part that actually varies. Measured at (1,1), that filtered locking is
saturated: 0.9938 with sd 0.0003 over 115 cycles. Nothing to modulate there.

The candidate measured here is the plainest causal alternative: the running
correlation between the heart signal x(t) and the network signal s(t), over a
trailing window of one cardiac period. No Hilbert, no filter, no future — the
whole network dynamics, chaos included.

s(t) is the *afferent* channel (the network projected onto K_baseline_x), not the
efferent projection onto D_baseline. Two reasons. The efferent is where the
network acts on the heart, so a modulator built on it could be driven up by the
rule itself — learn, send a more aligned signal, entrain the heart, learn more —
and the heart would stop being the one quantity plasticity cannot rewrite. And
s(t) is the signal every other measure of the project already lives on, including
the probe responses of the lag curve.

Two things are read, and the second matters more.

The distribution of r: mean, excursion, sd, shape over time. If it is as flat as
the PLV, this candidate falls too.

The split of its variance into a phase-locked part and a cycle-to-cycle residual.
Prediction, registered before running: r should be high on the plateaus of B and
low in the valleys, i.e. largely locked to cardiac phase. If *all* of its variance
is phase-locked, r is a metronome — a gate that opens at the same phases every
cycle regardless of what the system is doing — and it modulates timing, not
resonance. The residual is the part a learning rule could actually use.
"""
import numpy as np

from heartbrain import heart, coupling, coupled_system
from heartbrain.infra import analysis, plotting
from heartbrain.infra.persistence import networks, experiments


# Constants
GAIN_SWEEP_EXP = "gain_sweep"
RNN_BASELINE_NAME = "gain_sweep_baseline"

SEED2 = 54
SEED3 = 77

# === brain
G = 2.5

# === heart
DT = 0.01

# === coupled
N_STEPS = 90000
K_HB = 1.0
K_BH = 1.0
SUBDIR = "coupled"

# === discarded transient (same trimming as the PLV chain, for comparability)
TRANSIENT_STEPS = 1000
BORDER_STEPS = 500

# === modulator
# Cardiac period at (1,1), measured: 762.5 +- 5.9 steps.
CARDIAC_PERIOD = 762
R_WINDOW = CARDIAC_PERIOD        # trailing window of r(t): one cardiac cycle
PHASE_BINS = 24                  # bins used to split the variance of r by phase


def running_correlation(signal_a, signal_b, window_length):
    """Pearson correlation of two signals over a trailing window, per step.

    Causal by construction: the value at t uses only the ``window_length`` samples
    ending at t. Computed from running sums so the cost does not grow with the
    window.

    The first ``window_length - 1`` entries are NaN — no full window yet.
    """
    n = len(signal_a)
    out = np.full(n, np.nan)

    csum_a = np.concatenate(([0.0], np.cumsum(signal_a)))
    csum_b = np.concatenate(([0.0], np.cumsum(signal_b)))
    csum_aa = np.concatenate(([0.0], np.cumsum(signal_a * signal_a)))
    csum_bb = np.concatenate(([0.0], np.cumsum(signal_b * signal_b)))
    csum_ab = np.concatenate(([0.0], np.cumsum(signal_a * signal_b)))

    w = window_length
    end = np.arange(w, n + 1)
    start = end - w

    sum_a = csum_a[end] - csum_a[start]
    sum_b = csum_b[end] - csum_b[start]
    sum_aa = csum_aa[end] - csum_aa[start]
    sum_bb = csum_bb[end] - csum_bb[start]
    sum_ab = csum_ab[end] - csum_ab[start]

    cov = sum_ab - sum_a * sum_b / w
    var_a = sum_aa - sum_a * sum_a / w
    var_b = sum_bb - sum_b * sum_b / w

    denominator = np.sqrt(var_a * var_b)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[w - 1:] = np.where(denominator > 0, cov / denominator, np.nan)

    return out


def main():

    # === Vanilla RNN setup ===
    rnn = networks.load_network(RNN_BASELINE_NAME)
    rnn.scale_W_hh(g=G)
    print(f"OK - {RNN_BASELINE_NAME} network loaded and scaled (g={G}).")    # Log

    _, arrays = experiments.load_experiment(GAIN_SWEEP_EXP)
    x = arrays["x"]
    N = rnn.N

    oscillator = heart.Heart()


    # === Coupling components ===
    heart_to_brain_rng = np.random.default_rng(SEED2)
    K_baseline = coupling.create_K_baseline(rng=heart_to_brain_rng, N=N, fan_in=2)
    K_baseline_x = coupling.extract_K_baseline_x(K_baseline)
    K = coupling.build_K(k_hb=K_HB, K_baseline=K_baseline)

    brain_to_heart_rng = np.random.default_rng(SEED3)
    D_baseline = coupling.create_D_baseline(rng=brain_to_heart_rng, N=N)


    # === Coupled run at the operating point ===
    heart_series, brain_signal_series, _ = coupled_system.run_coupled(
        rnn=rnn,
        oscillator=oscillator,
        x=x,
        K=K,
        K_baseline_x=K_baseline_x,
        D_baseline=D_baseline,
        k_bh=K_BH,
        n_steps=N_STEPS,
        dt=DT,
    )
    print(f"OK - coupled run done (k_hb={K_HB}, k_bh={K_BH}, n_steps={N_STEPS}).")    # Log


    # === The candidate modulator: causal, unfiltered, one cycle of history ===
    r = running_correlation(heart_series, brain_signal_series, window_length=R_WINDOW)

    # Drop the transient and the still-empty windows at the start.
    n_start = max(TRANSIENT_STEPS + BORDER_STEPS, R_WINDOW - 1)
    r_valid = r[n_start:len(r) - BORDER_STEPS]
    heart_valid = heart_series[n_start:len(r) - BORDER_STEPS]

    print(f"\nr(t) over {len(r_valid)} steps (window = {R_WINDOW}, 1 cardiac cycle)")    # Log
    print(f"  mean  = {np.nanmean(r_valid):+.4f}")    # Log
    print(f"  sd    = {np.nanstd(r_valid):.4f}")    # Log
    print(f"  min   = {np.nanmin(r_valid):+.4f}")    # Log
    print(f"  max   = {np.nanmax(r_valid):+.4f}")    # Log
    print(f"  range = {np.nanmax(r_valid) - np.nanmin(r_valid):.4f}")    # Log


    # === Is r a metronome? Split its variance by cardiac phase ===
    # The phase proxy atan2(y, x) needs the heart's y, which run_coupled does not
    # return, so the phase is taken from the recorded x alone via its Hilbert
    # angle. This is a post-hoc read of the *diagnostic* only — r itself stays
    # causal, and nothing here enters the modulator.
    heart_phase = analysis.instantaneous_phase(heart_valid)

    bin_edges = np.linspace(-np.pi, np.pi, PHASE_BINS + 1)
    bin_index = np.clip(np.digitize(heart_phase, bin_edges) - 1, 0, PHASE_BINS - 1)

    phase_means = np.array([np.nanmean(r_valid[bin_index == b]) if np.any(bin_index == b)
                            else np.nan
                            for b in range(PHASE_BINS)])

    # Variance of r explained by the phase, and what is left cycle to cycle.
    r_phase_predicted = phase_means[bin_index]
    total_variance = np.nanvar(r_valid)
    residual_variance = np.nanvar(r_valid - r_phase_predicted)
    explained_fraction = 1.0 - residual_variance / total_variance

    print(f"\nvariance of r: total = {total_variance:.6f}")    # Log
    print(f"  explained by cardiac phase = {explained_fraction * 100:.1f}%")    # Log
    print(f"  residual (cycle to cycle)  = {residual_variance:.6f} "
          f"(sd {np.sqrt(residual_variance):.4f})")    # Log
    print("  a modulator whose variance is all phase-explained is a metronome;")    # Log
    print("  the residual is the part a learning rule could use.")    # Log


    # === Plots ===
    plotting.plot_curve(x_values=np.arange(len(r_valid)),
                        y_values=r_valid,
                        name="modulator_r_over_time",
                        subdir=SUBDIR,
                        x_label="step (after transient)",
                        y_label="r(t) = running corr(heart, s)")

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    plotting.plot_curve(x_values=bin_centers,
                        y_values=phase_means,
                        name="modulator_r_by_phase",
                        subdir=SUBDIR,
                        x_label="cardiac phase (rad)",
                        y_label="mean r(t)")


if __name__ == "__main__":
    main()
