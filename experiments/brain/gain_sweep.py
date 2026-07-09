"""The aim of the following experiment is to find the critical threshold 
where the Vanilla RNN changes its dynamics from a qualitative point of view.

We fix the following parameters:
    - rng seed
    - input ``x``
    - initial state ``h_0`` (set to zero by convention)
    - layer ``W_xh``
    - bias ``b_h``.

We define a base weight matrix W_base for the recurrent connection, generated 
as a standard scaled initialization with 
        sigma_base = 1 / sqrt(hidden_size=N). 
This corresponds to the case g = 1 (i.e., no rescaling).

Then, for each value of g in the sweep, we scale the W_base matrix by a factor 
of g and run the cell for N_STEPS steps. For example, when g = 1.25: 
        
        W_(g=1.25) = 1.25 * W_base.

Results are stored in a list of (g, trajectory) tuples and then logged.
"""
import numpy as np

from heartbrain import brain
from heartbrain.infra import persistence
from heartbrain.infra.experiment_logger import ExperimentLogger

from typing import TextIO


# Constants
RNN_BASELINE_NAME = "gain_sweep_baseline"

SEED = 42

INPUT_SIZE = 3
N = 64
            
G_MIN = 2.04
G_MAX = 2.14
NUM_G = 25  # step of 0.005

# Notice:
#   When the system starts to have a chaoitic dynamics
#   it is important to study the trajectory for many steps.
N_STEPS = 1000

# Period detection
PERIOD_EPSILON = 1e-4       # threshold for "same state"
WARMUP_STEPS = 50           # steps to skip before checking period


def main():
    
    rng = np.random.default_rng(SEED)


    # === Set up invariants ===
    x = rng.normal(size=INPUT_SIZE)
    h0 = brain.init_h0(hidden_size=N)
  

    # === Weights Initialization (Object Version) ===
    rnn = brain.VanillaRNN(
        hidden_size=N,
        input_size=INPUT_SIZE,
        h0=h0,
        rng=rng
    )


    # === Save the baseline RNN ===
    try:
        persistence.save_network(rnn=rnn, name=RNN_BASELINE_NAME)
        print(f"OK - {RNN_BASELINE_NAME} correctly saved.")
    except FileExistsError as fee:
        print(fee)


    # === Sweep g ===
    g_values = np.linspace(start=G_MIN, stop=G_MAX, num=NUM_G)


    # === Run The Cell (Object Version) ===
    g_trajectories = []
    for g in g_values:

        # Required to do not carry last state form previous "g"
        rnn.reset_state()
        
        rnn.scale_W_hh(g)

        trajectory = _run_single_experiment_object_version(x=x, rnn=rnn, n_steps=N_STEPS)
        g_trajectories.append( (g, trajectory) )


    # === Set ups experiment logger ===
    logger = ExperimentLogger(
        experiment_type="gain_sweep",
        parameters={ 
            "SEED": SEED, 
            "N": N,
            "G_MIN": G_MIN,
            "G_MAX" : G_MAX,
            "NUM_G": NUM_G,
            "N_STEPS": N_STEPS
            },
        component="brain"
    )


    # Logs
    with logger.open() as log:
        _print_header(file=log)
        _print_summary(g_trajectories, file=log)
          

def _run_single_experiment_object_version(x: np.ndarray, rnn: brain.VanillaRNN, n_steps: int) -> list[np.ndarray]:
    """Run the cell for n_steps steps with fixed weights encapsulated in the brain.VanillaRNN object.
    Returns the full trajectory as a list of states, starting with h0.
    """
    trajectory = [rnn.state]
    for _ in range(1, N_STEPS):
        rnn.step(x)
        trajectory.append(rnn.state)
    return trajectory


# === Log functions ===
def _print_header(file: TextIO | None = None) -> None:
    """Print experiment parameters."""
    print("==== Experiment parameters ====", file=file)
    print(f"N        = {N}", file=file)
    print(f"N_STEPS  = {N_STEPS}", file=file)
    print(f"G_MIN    = {G_MIN}", file=file)
    print(f"G_MAX    = {G_MAX}", file=file)
    print(f"NUM_G    = {NUM_G}", file=file)
    print(f"SEED     = {SEED}", file=file)


def _print_summary(g_trajectories: list[tuple[float, list[np.ndarray]]],
                   file: TextIO | None = None) -> None:
    """Print summary: (g, period, convergence status, last |dh|)."""
    print("\n==== Summary ====", file=file)
    for g, trajectory in g_trajectories:
        last_dh = np.linalg.norm(trajectory[-1] - trajectory[-2])
        converged = last_dh < PERIOD_EPSILON
        conv_status = "converged" if converged else "NOT converged"
        
        period = _detect_period(trajectory)
        period_str = str(period) if period is not None else "?"
        
        print(f"g = {g:.4f}   period = {period_str:>3s}   "
              f"[{conv_status}]   last |dh| = {last_dh:.6f}",
              file=file)


def _print_trajectory(index: int, total: int, g: float, trajectory: list[np.ndarray], file: TextIO | None = None) -> None:
    """Print one experiment's trajectory with a header line."""
    print(f"\n==== Experiment {index+1}/{total} — g = {g:.4f} ====", file=file)
    for t, h in enumerate(trajectory):
        h_str = ", ".join(f"{hi:+.4f}" for hi in h)
        if t == 0:
            distance_str = "-"
        else:
            distance_str = f"{np.linalg.norm(h - trajectory[t-1]):.6f}"
        print(f"t={t:3d}   h = [{h_str}]   |dh| = {distance_str}", file=file)


# === Period Detection ===
def _detect_period(trajectory: list[np.ndarray]) -> int | None:
    """Detect the period of the cycle at the end of the trajectory.
    
    For each candidate ``k``, verifies that all ``k`` components of the last
    period match the corresponding components of the second-to-last period,
    AND that the second-to-last period matches the third-to-last period.
    Requiring three consecutive periods to match makes the detection robust
    against sporadic coincidences.
    
    Returns the smallest matching ``k``, or None if no period is found within
    the available range.
    """
    usable_length = len(trajectory) - WARMUP_STEPS
    k_max = usable_length // 3
    
    for k in range(1, k_max + 1):
        all_match = True
        for j in range(k):
            # Compare last period with second-to-last period at position j.
            if np.linalg.norm(trajectory[-1 - j] - trajectory[-1 - j - k]) > PERIOD_EPSILON:
                all_match = False
                break
            # Compare second-to-last period with third-to-last period at position j.
            if np.linalg.norm(trajectory[-1 - j - k] - trajectory[-1 - j - 2*k]) > PERIOD_EPSILON:
                all_match = False
                break
        if all_match:
            return k
    return None


if __name__ == '__main__':
    main()