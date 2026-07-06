"""Raw run of the coupled components of the framework.

More details later...
"""
import numpy as np

from heartbrain import brain, heart, coupling

# Constants
SEED = 42

# === brain
INPUT_SIZE = 3
N =64
G = 2.04    # Edge of chaos (?)

# === heart
DT = 0.01
HEART_RUNS = 500
SIGMA = 0.0

# === coupled
K_HB = 0.0


def main():

    # Two different normal random number generators
    # does not bind future reproductions of the experiment
    # to specific "call sequences".
    brain_rng = np.random.default_rng(SEED)
    coupled_rng = np.random.default_rng(SEED)

    # === Vanilla RNN setup ===
    #
    # We use the same order used in the ``gain_sweep``
    # experiment such that the dynamics for a given 
    # value of ``g`` is well known.  
    x = brain_rng.normal(size=INPUT_SIZE)
    h0 = brain.init_h0(hidden_size=N)
    rnn = brain.VanillaRNN(
        hidden_size=N,
        input_size=INPUT_SIZE,
        h0=h0,
        rng=brain_rng
    )
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
    K_baseline = coupling.create_K_baseline(
        rng=coupled_rng,
        N=N,
        fan_in=2
    )
    K = coupling.build_K(k_hb=K_HB, K_baseline=K_baseline)

    # Try one interaction from the heart 
    # to the brain. Let's assume that the heart
    # is the first to born, so it "runs" for 
    # few steps.
    xs, ys = np.empty(HEART_RUNS), np.empty(HEART_RUNS)
    for i in range(HEART_RUNS):
        xs[i], ys[i] = h.current_x, h.current_y
        h.step(sigma=SIGMA, dt=DT)


    # At this point the heart state is in its limit cycle.
    # Let's implement one actual interaction (directed).
    h_state = h.get_state()
    perturbation = coupling.heart_to_brain_bias_perturbation(K=K, heart_state=h_state)
    rnn.apply_bias_perturbation(perturbation)
    rnn.step(x=x)
    h.step(sigma=SIGMA, dt=DT)

if __name__=="__main__":
    main()