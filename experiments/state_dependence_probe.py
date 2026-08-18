"""Does the pre-probe state still predict the response when sampled further back?

The earlier probe (``state_dependence_probe``) sampled the network state at the
trigger, immediately before the impulse, and asked whether near states give near
responses. It answered a narrower question than intended: a correlation at lag
zero says *a state matters*, which any nonlinear system satisfies — it does not
say the state *carries the past*. Chaos and memory are indistinguishable there.

Sampling the state further back separates them. If the response only depends on
where the system happens to be, then knowing h(t-tau) helps only insofar as
h(t-tau) predicts h(t) — and at g=2.5 that predictive power decays. The curve
falls to zero. If something persists across tau — a slow variable that integrated
the history and still modulates processing — the curve keeps a tail. The height at
tau=0 says "there is a state"; the *shape* of the curve says "the state remembers".

The probes, the impulses and the responses are identical to the earlier
experiment: one run yields the whole curve, since only the sampled state changes
with the lag while the responses stay fixed. This makes the points mutually
comparable by construction — no across-run variability to control for.

The heart is the only slow structure here, so the state at t-tau is sampled at a
cardiac phase that shifts with tau. The curve is therefore expected to oscillate
with the cardiac cycle rather than fall smoothly; the spacing of its maxima is the
cardiac period, and the damping of the oscillation is the accumulated phase
jitter. Both are read off the curve rather than measured separately.

Read in absolute, against the null at every lag. Pre-registered expectation
(written before running): the curve decays to zero, and it oscillates.
"""
import numpy as np
from collections import deque

from heartbrain import heart, coupling
from heartbrain.infra import analysis, plotting
from heartbrain.infra.persistence import networks, experiments


# Constants
GAIN_SWEEP_EXP = "gain_sweep"
RNN_BASELINE_NAME = "gain_sweep_baseline"

SEED2 = 54
SEED3 = 77

# === brain
G = 2.5

# === heart
DT = 0.01

# === coupled
N_STEPS = 90000
K_HB = 1.0
K_BH = 1.0

# === warmup (reach the attractor before probing)
WARMUP_STEPS = 2000

# === probe
TARGET_PHASE = 1.2      # a fast/chaotic phase (valley of B at (1,1))
IMPULSE_DX = 0.5         # fixed perturbation added to x during the impulse
IMPULSE_LEN = 5          # impulse duration in steps
RESPONSE_LEN = 300       # response window after the impulse onset (~half a cycle)
MIN_GAP = 1500           # minimum steps between probes (different states)

# === lags
# How far back the pre-probe state is sampled. Capped below MIN_GAP: beyond it
# the sampled state would fall inside the tail of the previous probe, and would
# no longer be a spontaneous state.
LAGS = list(range(0, 1401, 50))
MAX_LAG = max(LAGS)

NULL_SEED = 99


def main():

    # === Vanilla RNN setup ===
    rnn = networks.load_network(RNN_BASELINE_NAME)
    rnn.scale_W_hh(g=G)
    print(f"OK - {RNN_BASELINE_NAME} network loaded and scaled (g={G}).")    # Log

    _, arrays = experiments.load_experiment(GAIN_SWEEP_EXP)
    x = arrays["x"]
    N = rnn.N

    oscillator = heart.Heart()


    # === Coupling components ===
    heart_to_brain_rng = np.random.default_rng(SEED2)
    K_baseline = coupling.create_K_baseline(rng=heart_to_brain_rng, N=N, fan_in=2)
    K_baseline_x = coupling.extract_K_baseline_x(K_baseline)
    K = coupling.build_K(k_hb=K_HB, K_baseline=K_baseline)

    brain_to_heart_rng = np.random.default_rng(SEED3)
    D_baseline = coupling.create_D_baseline(rng=brain_to_heart_rng, N=N)


    # === Warmup: advance the coupled system onto its attractor ===
    for _ in range(WARMUP_STEPS):
        heart_state = oscillator.get_state()
        projection = rnn.project_state_onto(direction=D_baseline)
        perturbation = coupling.heart_to_brain_bias_perturbation(K=K, heart_state=heart_state)
        rnn.apply_bias_perturbation(perturbation)
        rnn.step(x=x)
        oscillator.step(sigma=K_BH * projection, dt=DT)


    # === Probed run: fire a fixed impulse at TARGET_PHASE, record states + response ===
    # Same trigger rule as state_dependence_probe: an upward crossing of the phase
    # proxy atan2(y, x), at least MIN_GAP steps after the last fire, with room left
    # for the full response. What changes here is only *which* pre-probe states are
    # kept: one per lag, pulled from a rolling window, instead of the trigger state
    # alone.
    state_history = deque(maxlen=MAX_LAG + 1)    # newest last; [-1 - lag] is lag steps back
    states_by_lag = {lag: [] for lag in LAGS}
    responses = []

    last_fire = -MIN_GAP
    previous_shift = None

    in_probe = False
    probe_onset = None
    response_buffer = []

    skipped_probes = 0

    for t in range(N_STEPS):
        heart_x, heart_y = oscillator.get_state()

        phase_proxy = np.arctan2(heart_y, heart_x)
        shifted = np.angle(np.exp(1j * (phase_proxy - TARGET_PHASE)))

        brain_signal = rnn.project_state_onto(direction=K_baseline_x)

        # The rolling window must already hold the current state when the trigger
        # is evaluated, so that state_history[-1] is the lag-0 state.
        state_history.append(rnn.state.copy())

        # Collect the response of an active probe.
        if in_probe:
            response_buffer.append(brain_signal)
            if len(response_buffer) >= RESPONSE_LEN:
                responses.append(np.array(response_buffer))
                in_probe = False
                response_buffer = []

        # Trigger a new probe on an upward crossing of the target phase.
        if (not in_probe) and previous_shift is not None:
            crossed = (previous_shift < 0) and (shifted >= 0)
            if crossed and (t - last_fire) >= MIN_GAP and (t + RESPONSE_LEN < N_STEPS):
                if len(state_history) > MAX_LAG:
                    in_probe = True
                    probe_onset = t
                    last_fire = t
                    response_buffer = []
                    # Pre-probe states, one per lag. Copied at append time, since
                    # ``state`` is the live array and the network keeps evolving.
                    for lag in LAGS:
                        states_by_lag[lag].append(state_history[-1 - lag])
                else:
                    # Not enough history yet (early probes after warmup): skipping
                    # keeps the number of states equal to the number of responses.
                    skipped_probes += 1

        previous_shift = shifted

        # Advance. During a probe's impulse window, x carries the fixed perturbation.
        driving_x = x
        if in_probe and (t - probe_onset) < IMPULSE_LEN:
            driving_x = x + IMPULSE_DX

        projection = rnn.project_state_onto(direction=D_baseline)
        perturbation = coupling.heart_to_brain_bias_perturbation(K=K, heart_state=(heart_x, heart_y))
        rnn.apply_bias_perturbation(perturbation)
        rnn.step(x=driving_x)
        oscillator.step(sigma=K_BH * projection, dt=DT)

    responses = np.array(responses)
    n_probes = len(responses)
    print(f"probes fired: {n_probes} (skipped for short history: {skipped_probes}, "
          f"response_len={responses.shape[1]}, pairs={n_probes * (n_probes - 1) // 2})")    # Log


    # === Measure: how far back does the state still predict the response? ===
    # One table per lag: pairwise state distances against pairwise response
    # distances, correlated. The response column is identical at every lag — only
    # the state column moves — so any change along the curve is attributable to
    # the lag alone.
    print(f"\n(k_hb={K_HB}, k_bh={K_BH}, target_phase={TARGET_PHASE})")    # Log
    print(f"{'lag':>6}  {'dependence':>11}  {'null':>8}")    # Log

    dependences = []
    nulls = []

    for lag in LAGS:
        states_lag = np.array(states_by_lag[lag])[:n_probes]

        dependence = analysis.state_response_dependence(states=states_lag,
                                                        responses=responses)
        null_rng = np.random.default_rng(NULL_SEED)    # fixed: same floor at every lag
        dependence_null = analysis.state_response_dependence_null(states=states_lag,
                                                                  responses=responses,
                                                                  rng=null_rng)
        dependences.append(dependence)
        nulls.append(dependence_null)
        print(f"{lag:>6}  {dependence:>+11.4f}  {dependence_null:>+8.4f}")    # Log

    dependences = np.array(dependences)
    print(f"\nlag 0 = {dependences[0]:+.4f} (baseline: 0.134 at phase 1.5)")    # Log
    print(f"max = {dependences.max():+.4f} at lag {LAGS[int(dependences.argmax())]}")    # Log
    print(f"min = {dependences.min():+.4f} at lag {LAGS[int(dependences.argmin())]}")    # Log

    plotting.plot_curve(x_values=LAGS,
                        y_values=dependences,
                        name="state_dependence_vs_lag",
                        subdir="lag",
                        x_label="lag (steps before the probe)",
                        y_label="state-response dependence")

    plotting.plot_curve(x_values=LAGS,
                        y_values=nulls,
                        name="state_dependence_vs_lag_null",
                        subdir="lag",
                        x_label="lag (steps before the probe)",
                        y_label="null floor",
                        color="grey")


if __name__ == "__main__":
    main()
