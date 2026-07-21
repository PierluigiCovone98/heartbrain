"""How does the heart->brain locking value grow with the coupling strength?

The invariance experiment established two points: at ``k_hb = 0`` the phase
locking value is ~0 (no coupling, no locking), and at ``k_hb = 0.5`` it is
~0.998 (near-perfect locking). Two points are not a curve: they leave open *how*
the system moves between them — whether the locking rises gradually or through a
sharp threshold, and where entrainment sets in. Since ``k_hb = 0.5`` already
gives 0.998, the whole interesting transition happens between 0 and 0.5.

This experiment sweeps ``k_hb`` and measures the PLV at each value, tracing the
full curve. The sampling is denser near zero (where the transition lives) and
sparser toward the edges (where the PLV saturates), and it is symmetric about
zero: negative values are included so that the response to ``+k_hb`` and
``-k_hb`` can be compared point by point — the open question of whether the sign
produces a genuinely different dynamics or merely a phase shift.

The same network and heart instances are reused across the sweep, so both are
reset before every run: the state and the (perturbed) bias are returned to
baseline, while the gain-scaled ``W_hh`` is left untouched — the gain is fixed
for the whole sweep, only ``K`` changes with ``k_hb``.
"""
import numpy as np

from heartbrain import heart, coupling
from heartbrain.infra import plotting, analysis
from heartbrain.infra.persistence import networks, experiments
from heartbrain.coupled_system import run_heart_to_brain


# Constants
GAIN_SWEEP_EXP = "gain_sweep"
RNN_BASELINE_NAME = "gain_sweep_baseline"

# === coupling setup
SEED2 = 54

# === brain
G = 2.5

# === heart
DT = 0.01
SIGMA = 0.0

# === coupled
N_STEPS = 10000

# === brain signal filtering
CUTOFF_PERIOD = 100

# === measurement window
TRANSIENT_STEPS = 1000
BORDER_STEPS = 500

# === sweep
K_MAX = 2.0
N_POINTS = 51
DENSITY = 2.0            # sinh warp: larger = denser near zero
SWEEP_SUBDIR = "sweep"


def _make_khb_values(k_max: float, n_points: int, density: float) -> np.ndarray:
    """Generate the k_hb values to sweep, dense near zero and symmetric.

    Uniform samples in [-1, 1] are warped through ``sinh``, which is odd (so the
    result is symmetric about zero and passes through it exactly, when
    ``n_points`` is odd) and concentrates points near the center. ``density``
    controls how strong that concentration is: larger values pack more points
    into the transition region near zero, leaving the saturated edges sparse.

    Parameters
    ----------
    k_max : float
        The sweep spans ``[-k_max, +k_max]``.
    n_points : int
        Number of values. Odd includes zero exactly.
    density : float
        Warp strength; larger means denser near zero.

    Returns
    -------
    np.ndarray
        The k_hb values, ascending, symmetric about zero.
    """
    t = np.linspace(-1.0, 1.0, n_points)
    return k_max * np.sinh(density * t) / np.sinh(density)


def main():

    # === Vanilla RNN setup ===
    # We load the network from disk.
    rnn = networks.load_network(RNN_BASELINE_NAME)
    print(f"OK - {RNN_BASELINE_NAME} network correctly loaded.")    # Log


    # === Parameters Setup ===
    _, arrays = experiments.load_experiment(GAIN_SWEEP_EXP)
    print(f"OK - {GAIN_SWEEP_EXP} experiment correctly loaded.")    # Log

    x = arrays["x"]
    N = rnn.N


    # === Heart setup ===
    oscillator = heart.Heart()


    # === Network setup ===
    rnn.scale_W_hh(g=G)


    # === Coupling components ===
    coupled_rng = np.random.default_rng(SEED2)
    K_baseline = coupling.create_K_baseline(rng=coupled_rng, N=N, fan_in=2)
    K_baseline_x = coupling.extract_K_baseline_x(K_baseline)


    # === Sweep ===
    khb_values = _make_khb_values(k_max=K_MAX, n_points=N_POINTS, density=DENSITY)
    plv_values = np.zeros(N_POINTS)

    for i, k_hb in enumerate(khb_values):
        # Reset the reused components to baseline before each run. State and the
        # perturbed bias are restored; the gain-scaled W_hh is left as is, since
        # the gain is fixed for the whole sweep and only K changes with k_hb.
        rnn.reset_state()
        rnn.reset_bias()
        oscillator.reset_state()

        K = coupling.build_K(k_hb=k_hb, K_baseline=K_baseline)

        heart_series, brain_series = run_heart_to_brain(rnn=rnn,
                                                        oscillator=oscillator,
                                                        x=x,
                                                        K=K,
                                                        K_baseline_x=K_baseline_x,
                                                        n_steps=N_STEPS,
                                                        dt=DT,
                                                        sigma=SIGMA)

        plv_values[i] = analysis.measure_plv(heart_series=heart_series,
                                             brain_series=brain_series,
                                             cutoff_period=CUTOFF_PERIOD,
                                             transient_steps=TRANSIENT_STEPS,
                                             border_steps=BORDER_STEPS)

        print(f"k_hb = {k_hb:+.4f}   PLV = {plv_values[i]:.4f}")    # Log


    # === Plot the curve ===
    path = plotting.plot_curve(x_values=khb_values,
                               y_values=plv_values,
                               name="khb_sweep",
                               subdir=SWEEP_SUBDIR,
                               x_label="k_hb",
                               y_label="PLV")
    print(f"OK - curve saved to {path}")    # Log


if __name__ == "__main__":
    main()
