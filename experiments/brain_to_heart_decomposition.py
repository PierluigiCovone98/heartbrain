"""Decompose the brain -> heart forcing into its components, one at a time.

The forcing the network sends the heart, ``D·h``, splits into a constant part and a
fluctuating part: ``D·h = mean + (D·h - mean)``. 
These two act on different registers of the oscillator — the constant biases
its equilibrium (and, if large enough, pushes it off the limit cycle entirely),
while the fluctuation adds irregularity without destroying the rhythm.
This experiment drives the heart with one piece at a time to see each effect
in isolation.

The FORCING constant selects which piece:
  - "full"  -> sigma = k_bh * (D·h)          the real signal (both parts)
  - "chaos" -> sigma = k_bh * (D·h - mean)   fluctuation only, constant removed
  - "bias"  -> sigma = k_bh * mean           constant only, fluctuation removed

A first free run (coupling off) measures mean(D·h); this is valid for the driven
run because, with the heart -> brain channel off, the network evolves identically
regardless of k_bh. The three modes together against the full signal give a
visual decomposition: what the whole forcing does, what the chaos alone does and 
what the bias alone does.

Neither part is an artifact. A non-zero baseline is a legitimate part of what the
network sends (like a resting autonomic tone), and the fluctuation is the genuine
chaotic drive. Splitting them here is an analytic move to attribute each effect,
not a correction to the model.

Biological note: the two effects (seems) to have distinct physiological analogues.
A strong constant drive suppressing the pacemaker's own oscillation echoes
overdrive suppression / loss of sinus rhythm; a fluctuating drive adding 
beat-to-beat variability without abolishing the rhythm echoes heart-rate 
variability (and, in the extreme, arrhythmia).
"""
import numpy as np

from heartbrain import heart, coupling, coupled_system
from heartbrain.infra import plotting
from heartbrain.infra.persistence import networks, experiments


# Constants
GAIN_SWEEP_EXP = "gain_sweep"
RNN_BASELINE_NAME = "gain_sweep_baseline"

# Seed for the brain -> heart channel, distinct from the heart -> brain seed
# (SEED2 = 54) so the two coupling directions are independent.
SEED3 = 77

# === brain
G = 2.5   # Edge of chaos

# === heart
DT = 0.01

# === coupled
N_STEPS = 10000
K_BH = 0.5
SUBDIR = "brain_to_heart"

# Which component of the forcing drives the heart:
#   "full"    -> sigma = k_bh * (D·h)              [the real signal]
#   "chaos"   -> sigma = k_bh * (D·h - mean)       [fluctuation only, bias removed]
#   "bias"    -> sigma = k_bh * mean               [constant only, fluctuation removed]
FORCING = "bias"


def main():

    # === Vanilla RNN setup ===
    # Load the VanillaRNN instance studied in the ``gain_sweep`` experiment.
    rnn = networks.load_network(RNN_BASELINE_NAME)
    print(f"OK - {RNN_BASELINE_NAME} network correctly loaded.")    # Log


    # === Parameters Setup ===
    _, arrays = experiments.load_experiment(GAIN_SWEEP_EXP)
    print(f"OK - {GAIN_SWEEP_EXP} experiment correctly loaded.")    # Log

    # To align this experiment to the network studied in the "gain_sweep" one.
    x = arrays["x"]
    N = rnn.N
    rnn.scale_W_hh(g=G)


    # === Heart setup ===
    h = heart.Heart()


    # === Coupling component (brain -> heart channel) ===
    # A single projection direction, from its own seed to stay independent of the
    # heart -> brain channel. The network state projected onto it, scaled by k_bh,
    # is the scalar the heart receives.
    coupled_rng = np.random.default_rng(SEED3)
    D_baseline = coupling.create_D_baseline(rng=coupled_rng, N=N)


    # === First pass: measure the mean of the (unscaled) projection D·h ===
    # Run with the coupling OFF (k_bh=0): the network runs free and the returned
    # projection series is D·h. Its mean is the constant bias we want to isolate.
    # (Valid for the second pass too: in this direction the heart does not affect
    # the network, so the network's trajectory — and thus D·h — is identical
    # regardless of k_bh.)
    _, projection = coupled_system.run_brain_to_heart(rnn=rnn, oscillator=h, x=x,
                                                      D_baseline=D_baseline, k_bh=0.0,
                                                      n_steps=N_STEPS, dt=DT)
    mean_projection = projection.mean()
    print(f"mean(D·h) = {mean_projection:.4f}")

    # Reset both components to their initial state before the coupled pass.
    rnn.reset_state()
    rnn.reset_bias()
    h.reset_state()


    # === Second pass: coupled run with the selected forcing component ===
    # forcing starts as the full projection D·h; the FORCING mode then reduces it
    # to the chaos part (minus mean) or the bias part (the mean alone). sigma is
    # that component scaled by k_bh. What is recorded as the network series is the
    # forcing component actually driving the heart..
    heart_time_series = np.zeros(N_STEPS)
    brain_time_series = np.zeros(N_STEPS)

    for t in range(N_STEPS):
        heart_state = h.get_state()
        heart_time_series[t] = heart_state[0]

        # ``forcing`` is the case "FULL"
        forcing = rnn.project_state_onto(direction=D_baseline)

        if FORCING == "chaos":
            forcing -= mean_projection
        elif FORCING == "bias":
            forcing = mean_projection

        sigma = K_BH * forcing
        brain_time_series[t] = forcing

        h.step(sigma=sigma, dt=DT)
        rnn.step(x=x)


    # === Look first: heart x(t) against what the network sends ===
    plotting.plot_time_series(heart_time_series, brain_time_series,
                              name=f"brain_to_heart_{FORCING}_kbh_05",
                              subdir=SUBDIR,
                              start=2000, end=7000,
                              brain_label=f"network projection ({FORCING})")


if __name__ == "__main__":
    main()
