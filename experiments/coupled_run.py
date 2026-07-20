"""Raw run of the coupled components of the framework.

More details later...
"""
import numpy as np

from heartbrain import heart, coupling, coupled_system
from heartbrain.infra import analysis
from heartbrain.infra.persistence import networks, experiments


# Constants
GAIN_SWEEP_EXP = "gain_sweep"
RNN_BASELINE_NAME = "gain_sweep_baseline"

# Temporary choice:
# to avoid statistichal correlation I have to use:
#   np.random.SeedSequence(SEED).spawn(n)
SEED2 = 54

# === brain
G = 2.5   # Edge of chaos !!!!

# === heart
DT = 0.01
SIGMA = 0.0

# === coupled
N_STEPS = 10000
K_HB = 0.5
SUBDIR = "coupled"

# === brain signal filtering
CUTOFF_PERIOD = 100

# === measurement window
TRANSIENT_STEPS = 1000
BORDER_STEPS = 500


def main():

    # === Vanilla RNN setup ===
    # Load the VanillaRNN instance studied in the ``gain_sweep`` experiment.
    rnn = networks.load_network(RNN_BASELINE_NAME)
    print(f"OK - {RNN_BASELINE_NAME} network correctly loaded.")    # Log


    # === Parameters Setup ===
    _, arrays = experiments.load_experiment(GAIN_SWEEP_EXP)
    print(f"OK - {GAIN_SWEEP_EXP} experiment correctly loaded.")    # Log

    # To align this esperiment to the network studied in the "gain_sweep" one.
    x = arrays["x"]
    N = rnn.N
    rnn.scale_W_hh(g=G)


    # === Heart setup ===
    h = heart.Heart()


    # === Coupling Components ===
    # 
    # First we define those parameters that are required for
    # the interaction of heart and brain components.
    coupled_rng = np.random.default_rng(SEED2)

    K_baseline = coupling.create_K_baseline(
        rng=coupled_rng,
        N=N,
        fan_in=2
    )
    K_baseline_x = coupling.extract_K_baseline_x(K_baseline)
    K = coupling.build_K(k_hb=K_HB, K_baseline=K_baseline)

    # Temporal series: 
    heart_time_series, brain_time_series = coupled_system.run_heart_to_brain(rnn=rnn,
                                                                             oscillator=h,
                                                                             x=x,
                                                                             K=K,
                                                                             K_baseline_x=K_baseline_x,
                                                                             n_steps=N_STEPS,
                                                                             dt=DT,
                                                                             sigma=SIGMA)

    # Phase Locking Values
    plv = analysis.measure_plv(heart_series=heart_time_series,
                               brain_series=brain_time_series,
                               cutoff_period=CUTOFF_PERIOD,
                               transient_steps=TRANSIENT_STEPS,
                               border_steps=BORDER_STEPS)
    print(f"PLV (k_hb={K_HB}) = {plv:.4f}")


if __name__=="__main__":
    main()