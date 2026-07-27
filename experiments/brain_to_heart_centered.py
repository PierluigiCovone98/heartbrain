"""Diagnostic: is the heart's collapse at large k_bh driven by the chaos or the bias?

The brain -> heart forcing sigma = k_bh * (D·h) has a non-zero mean, because the
projection D·h sits around ~1 rather than 0. So the forcing is a constant bias
plus a chaotic fluctuation. At large k_bh the heart's limit cycle collapses — but
is that the chaos destroying the oscillation, or just the constant bias pushing
the oscillator off its cycle?

This isolates the two: it measures the mean of D·h in a first free run, then runs
the coupled system forcing the heart with only the *centered* projection
(D·h - mean) — the chaotic part with the constant bias removed. If the limit
cycle survives, the collapse was the bias; if it collapses anyway, the chaos.

The bias is not an artifact — a non-zero baseline is a legitimate part of what the
network sends (like a resting autonomic tone). Removing it here is a diagnostic
move to attribute the effect, not a correction to the model.
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
K_BH = 1.5
SUBDIR = "brain_to_heart"


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


    # === Second pass: coupled run with the CENTERED forcing ===
    # sigma = k_bh * (D·h - mean): the heart receives only the fluctuating
    # (chaotic) part of the projection, with the constant bias removed. If the
    # heart's limit cycle survives this, the collapse seen at large k_bh was
    # driven by the bias; if it collapses anyway, by the chaos.
    heart_time_series = np.zeros(N_STEPS)
    brain_time_series = np.zeros(N_STEPS)

    for t in range(N_STEPS):
        heart_state = h.get_state()
        heart_time_series[t] = heart_state[0]

        projection_t = rnn.project_state_onto(direction=D_baseline)
        centered_projection = projection_t - mean_projection
        brain_time_series[t] = centered_projection

        sigma = K_BH * centered_projection    

        h.step(sigma=sigma, dt=DT)
        rnn.step(x=x)


    # === Look first: heart x(t) against what the network sends ===
    plotting.plot_time_series(heart_time_series, brain_time_series,
                              name="brain_to_heart_kbh_1p5",
                              subdir=SUBDIR,
                              start=2000, end=7000,
                              brain_label="network projection (centered)")


if __name__ == "__main__":
    main()
