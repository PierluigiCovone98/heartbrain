"""Raw run of the brain -> heart directed coupling.

The mirror of ``coupled_run``, in the opposite direction: here the chaotic
network drives the heart (the heart -> brain channel is off, k_hb = 0). At a
fixed k_bh it plots the heart signal x(t) against what the network is sending, so
one can *see* whether a chaotic forcing changes the heart's dynamics — and if so,
how. Nothing is measured yet: this is the look-first stage. What to measure, if
anything, is decided only after seeing what the heart does (it might stay on its
limit cycle, shift frequency or amplitude, or become irregular — none of this is
known in advance).

The projection direction D_baseline is drawn from a seed distinct from the one
behind K_baseline, so the brain -> heart channel is independent of the heart ->
brain one (they are physically distinct pathways).
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
K_BH = 0.0
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


    # === Temporal series ===
    heart_time_series, brain_time_series = coupled_system.run_brain_to_heart(rnn=rnn,
                                                                             oscillator=h,
                                                                             x=x,
                                                                             D_baseline=D_baseline,
                                                                             k_bh=K_BH,
                                                                             n_steps=N_STEPS,
                                                                             dt=DT)


    # === Look first: heart x(t) against what the network sends ===
    plotting.plot_time_series(heart_time_series, brain_time_series,
                              name="brain_to_heart_kbh_0",
                              subdir=SUBDIR,
                              start=2000, end=7000,
                              brain_label="network projection (unit intensity)")


if __name__ == "__main__":
    main()
