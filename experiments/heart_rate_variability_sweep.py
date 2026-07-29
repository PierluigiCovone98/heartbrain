"""How much does the network's chaos make the heart's rhythm variable, vs k_bh?

Sweeps the brain -> heart coupling strength and, at each value, measures the
variability of the heart's beat period under the *pure-chaos* forcing (the
projection with its constant bias removed). Isolating the chaos matters: the bias
would shift or, past a threshold, suppress the oscillation, leaving no rhythm to
measure — whereas the chaos keeps the cycle alive and only adds irregularity.

Two variability measures are reported per k_bh: the standard deviation of the
periods (overall spread) and the RMS of successive differences (beat-to-beat),
the analogues of SDNN and RMSSD. At k_bh = 0 the heart is a clean limit cycle, so
the variability there is not zero but a small floor from measuring integer-step
periods on a non-integer true period — the null against which real variability is
judged.

Reading, kept deliberately structural (not clinical): the curve says how robust
the heart's limit cycle is to chaotic forcing — how much chaos it absorbs before
the rhythm loses regularity, and how. The "HRV" analogy is descriptive only; the
clinical association (low HRV = pathology) does not transfer, because here the
variability comes from injected chaos, not from an autonomic control loop. That
loop only exists once the system is bidirectional.

The mean of D·h is measured once, before the sweep: with the heart -> brain
channel off, the network runs independently of the heart, so its projection — and
that mean — is the same for every k_bh.
"""
import numpy as np

from heartbrain import brain, heart, coupling
from heartbrain.infra import analysis, plotting
from heartbrain.infra.persistence import networks, experiments
from heartbrain.coupled_system import run_brain_to_heart


# Constants
GAIN_SWEEP_EXP = "gain_sweep"
RNN_BASELINE_NAME = "gain_sweep_baseline"

# Seed for the brain -> heart channel, distinct from the heart -> brain seed.
SEED3 = 77

# === brain
G = 2.5

# === heart
DT = 0.01

# === coupled
N_STEPS = 50000

# === measurement
TRANSIENT_STEPS = 1000
BORDER_STEPS = 500
TARGET_PHASE = 0.0

# === k_bh sweep
K_MIN = 0.0
K_MAX = 4.0
N_KBH = 18
SUBDIR = "hrv"


def main():

    # === Vanilla RNN setup ===
    rnn = networks.load_network(RNN_BASELINE_NAME)
    rnn.scale_W_hh(g=G)
    print(f"OK - {RNN_BASELINE_NAME} network loaded and scaled (g={G}).")    # Log

    _, arrays = experiments.load_experiment(GAIN_SWEEP_EXP)
    x = arrays["x"]
    N = rnn.N

    oscillator = heart.Heart()

    coupled_rng = np.random.default_rng(SEED3)
    D_baseline = coupling.create_D_baseline(rng=coupled_rng, N=N)


    # === Measure mean(D·h) once (coupling off) ===
    # In this direction the heart does not affect the network, so the network's
    # projection — and its mean — is identical for every k_bh. Reuse the run with
    # k_bh = 0: the returned brain series is the unscaled projection D·h.
    _, projection = run_brain_to_heart(rnn=rnn, oscillator=oscillator, x=x,
                                       D_baseline=D_baseline, k_bh=0.0,
                                       n_steps=N_STEPS, dt=DT)
    mean_projection = float(projection.mean())
    print(f"mean(D·h) = {mean_projection:.4f}")    # Log


    # === Sweep k_bh, measuring period variability under pure-chaos forcing ===
    # The coupled loop is written out here (not via run_brain_to_heart) because
    # the forcing must be centered: sigma = k_bh * (D·h - mean), the chaos alone.
    kbh_values = np.linspace(K_MIN, K_MAX, N_KBH)
    period_std = np.zeros(N_KBH)
    period_rmssd = np.zeros(N_KBH)

    n_start = TRANSIENT_STEPS + BORDER_STEPS

    print(f"\n{'k_bh':>8} {'n_periods':>10} {'std':>10} {'rmssd':>10}")    # Log
    for i, k_bh in enumerate(kbh_values):
        rnn.reset_state()
        rnn.reset_bias()
        oscillator.reset_state()

        heart_series = np.zeros(N_STEPS)
        for t in range(N_STEPS):
            heart_state = oscillator.get_state()
            heart_series[t] = heart_state[0]

            projection_t = rnn.project_state_onto(direction=D_baseline)
            sigma = k_bh * (projection_t - mean_projection)

            oscillator.step(sigma=sigma, dt=DT)
            rnn.step(x=x)

        heart_valid = analysis.discard_borders(heart_series, n_start=n_start, n_end=BORDER_STEPS)
        heart_phase = analysis.instantaneous_phase(heart_valid)
        periods = analysis.extract_periods(heart_phase, target_phase=TARGET_PHASE)
        period_std[i], period_rmssd[i] = analysis.period_variability(periods)

        print(f"{k_bh:>8.3f} {len(periods):>10} "
              f"{period_std[i]:>10.3f} {period_rmssd[i]:>10.3f}")    # Log


    # === Plot: the two variability measures vs k_bh, shared x-axis ===
    plotting.plot_two_panels(x_values=kbh_values,
                             top_values=period_std,
                             bottom_values=period_rmssd,
                             name="heart_rate_variability_vs_kbh",
                             subdir=SUBDIR,
                             x_label="k_bh",
                             top_label="period std (SDNN-like)",
                             bottom_label="period rmssd (RMSSD-like)")


if __name__ == "__main__":
    main()
