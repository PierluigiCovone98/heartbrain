"""Raw run of the bidirectional heart <-> brain coupling.

Both channels on at once: the heart perturbs the network (heart -> brain, via K)
and the network forces the heart (brain -> heart, via D_baseline scaled by k_bh).
At a fixed (k_hb, k_bh) it plots the heart signal x(t), what the network sends the
heart (projection onto D_baseline), and the network signal studied throughout
(s(t), projection onto K_baseline_x). Nothing is measured yet: this is the
look-first stage.

The two channels use directions from distinct seeds (K_baseline from SEED2,
D_baseline from SEED3), so the afferent and efferent pathways are independent.

Reading, with the closed loop, is against the single-direction baselines: does
the heart differ from how it behaved at this k_bh in isolation, and does the
network differ from how it behaved at this k_hb in isolation? What the loop adds,
beyond the sum of the two monologues, is the thing to look for.
"""
import numpy as np

from heartbrain import heart, coupling, coupled_system
from heartbrain.infra import plotting
from heartbrain.infra.persistence import networks, experiments


# Constants
GAIN_SWEEP_EXP = "gain_sweep"
RNN_BASELINE_NAME = "gain_sweep_baseline"

# Distinct seeds for the two channels: heart -> brain (K) and brain -> heart (D).
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
SUBDIR = "bidirectional"


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
    # heart -> brain: matrix K (scaled here by k_hb), and the observation
    # direction K_baseline_x for recording s(t).
    heart_to_brain_rng = np.random.default_rng(SEED2)
    K_baseline = coupling.create_K_baseline(rng=heart_to_brain_rng, N=N, fan_in=2)
    K_baseline_x = coupling.extract_K_baseline_x(K_baseline)
    K = coupling.build_K(k_hb=K_HB, K_baseline=K_baseline)

    # brain -> heart: projection direction D_baseline (scaled by k_bh inside the
    # loop), from its own seed to stay independent of the heart -> brain channel.
    brain_to_heart_rng = np.random.default_rng(SEED3)
    D_baseline = coupling.create_D_baseline(rng=brain_to_heart_rng, N=N)


    # === Temporal series (closed loop) ===
    heart_series, brain_signal_series, brain_projection_series = coupled_system.run_coupled(
        rnn=rnn,
        oscillator=h,
        x=x,
        K=K,
        K_baseline_x=K_baseline_x,
        D_baseline=D_baseline,
        k_bh=K_BH,
        n_steps=N_STEPS,
        dt=DT)


    # === Look first: heart, what the network sends, and the network signal ===
    steps = np.arange(N_STEPS)
    window = slice(2000, 7000)

    plotting.plot_three_panels(x_values=steps[window],
                               top_values=heart_series[window],
                               middle_values=brain_projection_series[window],
                               bottom_values=brain_signal_series[window],
                               name="bidirectional_khb_1_kbh_1",
                               subdir=SUBDIR,
                               x_label="step",
                               top_label="heart x(t)",
                               middle_label="network -> heart (proj. on D)",
                               bottom_label="network s(t) (proj. on K_x)")


if __name__ == "__main__":
    main()
