"""Raw run of the coupled components of the framework.

More details later...
"""
import numpy as np

from heartbrain import brain, heart, coupling
from heartbrain.infra.persistence import networks, experiments

# Constants
GAIN_SWEEP_EXP = "gain_sweep"
RNN_BASELINE_NAME = "gain_sweep_baseline"

# Temporary choice:
# to avoid statistichal correlation I have to use:
#   np.random.SeedSequence(SEED).spawn(n)
SEED2 = 54

# === brain
INPUT_SIZE = 3
N =64
G = 2.04    # Edge of chaos (?)

# === heart
DT = 0.01
SIGMA = 0.0

# === coupled
N_STEPS = 10
K_HB = 0.0


def main():

    # === Previous Experiment infos Restoring ===
    _, arrays = experiments.load_experiment(GAIN_SWEEP_EXP)
    print(f"OK - {GAIN_SWEEP_EXP} experiment correctly loaded.")    # Log


    # === Vanilla RNN setup ===
    #
    # Load the VanillaRNN instance studied in the ``gain_sweep`` experiment.
    rnn = networks.load_network(RNN_BASELINE_NAME)
    print(f"OK - {RNN_BASELINE_NAME} network correctly loaded.")    # Log


    # === Parameter setup ====
    #
    # To align this esperiment to the network studied in the "gain_sweep" one.
    x = arrays["x"]
    
    rnn.scale_W_hh(g=G)


    # === Heart setup ===
    h = heart.Heart()


    # === Coupling Them ===
    # 
    # First we define those parameters that
    # are required fot the interaction of
    # heart and brain components.
    # Notice that I handwrote the value ``2``
    # for the seek of simplicity.
    coupled_rng = np.random.default_rng(SEED2)

    K_baseline = coupling.create_K_baseline(
        rng=coupled_rng,
        N=N,
        fan_in=2
    )
    K = coupling.build_K(k_hb=K_HB, K_baseline=K_baseline)

    # At this point the heart state is in its limit cycle.
    # Let's implement one actual interaction (directed).
    for _ in range(N_STEPS):
        h_state = h.get_state()
        perturbation = coupling.heart_to_brain_bias_perturbation(K=K, heart_state=h_state)
        rnn.apply_bias_perturbation(perturbation)
        rnn.step(x=x)
        h.step(sigma=SIGMA, dt=DT)

if __name__=="__main__":
    main()