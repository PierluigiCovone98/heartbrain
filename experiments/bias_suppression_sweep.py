"""At what k_bh does the constant bias suppress the heart's oscillation?

The brain -> heart forcing splits into a constant bias and a chaotic fluctuation.
The bias alone, past some strength, pushes the Van der Pol off its limit cycle
onto a fixed point — the oscillation dies (an inverse Hopf bifurcation). 
This sweep locates that threshold.

It drives the heart with the bias alone (sigma = k_bh * mean, constant) across a
range of k_bh, and measures the oscillation amplitude on the *stationary tail*:
above threshold the heart spirals slowly onto its fixed point, so the amplitude
must be read after that transient has fully settled, not over the whole run. The
amplitude is large while the cycle lives and collapses to ~0 past the threshold;
the point of collapse is the bifurcation.

Only the positive side is swept: the effect is symmetric in the sign of k_bh (a
negative bias shifts the fixed point the other way but suppresses at the same
magnitude).

The mean of D·h is measured once: with the heart -> brain channel off, the
network runs independently of the heart, so its projection is the same for every
k_bh.
"""
import numpy as np

from heartbrain import heart, coupling
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
# Generous transient: above threshold the heart spirals slowly onto its fixed
# point, so the amplitude is read only on the well-settled tail.
TAIL_START = 30000

# === k_bh sweep (positive side only; effect symmetric in sign)
K_MIN = 0.5
K_MAX = 2.0
N_KBH = 31
SUBDIR = "suppression"


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
    _, projection = run_brain_to_heart(rnn=rnn, oscillator=oscillator, x=x,
                                       D_baseline=D_baseline, k_bh=0.0,
                                       n_steps=N_STEPS, dt=DT)
    mean_projection = float(projection.mean())
    print(f"mean(D·h) = {mean_projection:.4f}")    # Log


    # === Sweep k_bh under the constant-bias forcing ===
    # sigma = k_bh * mean is constant, so it is computed once per k_bh, outside
    # the step loop. The network still runs (it is deterministic and unused here
    # beyond keeping the loop symmetric), but only the heart matters.
    kbh_values = np.linspace(K_MIN, K_MAX, N_KBH)
    tail_amplitude = np.zeros(N_KBH)

    print(f"\n{'k_bh':>8} {'tail_amplitude':>16}")    # Log
    for i, k_bh in enumerate(kbh_values):
        rnn.reset_state()
        rnn.reset_bias()
        oscillator.reset_state()

        sigma = k_bh * mean_projection   # constant bias forcing

        heart_series = np.zeros(N_STEPS)
        for t in range(N_STEPS):
            heart_series[t] = oscillator.get_state()[0]
            oscillator.step(sigma=sigma, dt=DT)
            rnn.step(x=x)

        # Amplitude on the settled tail: large while the cycle lives, ~0 once
        # the oscillation has collapsed onto the fixed point.
        tail = heart_series[TAIL_START:]
        amplitude = analysis.instantaneous_amplitude(tail - tail.mean())
        tail_amplitude[i] = float(np.mean(amplitude))

        print(f"{k_bh:>8.3f} {tail_amplitude[i]:>16.4f}")    # Log


    # === Plot: oscillation amplitude vs k_bh (collapses at the threshold) ===
    plotting.plot_curve(x_values=kbh_values,
                        y_values=tail_amplitude,
                        name="bias_suppression_vs_kbh",
                        subdir=SUBDIR,
                        x_label="k_bh",
                        y_label="oscillation amplitude (tail)")


if __name__ == "__main__":
    main()