"""Is the response to a fixed probe predictable from the internal state?

Once the loop is closed, the coupled system is chaotic: the same input, given at
different moments, gives different responses. But "responses vary" is satisfied by
chaos alone. The sharper question is whether the response is *predictable from the
state* — do near states give near responses (state-dependence, a transient memory)
or arbitrary ones (chaos)?

The probe is a fixed impulse on x (same magnitude, same duration), fired many
times at the *same heart phase*, chosen in a fast/chaotic region of the cycle
(where reproducibility B is low — a plateau would give a trivially repeatable
response). Firing at one fixed phase neutralizes the phase as an explanation: any
residual dependence of the response on the state is then beyond the phase, the
candidate for memory. For each probe the pre-probe network state h and the
response (s(t) over the following steps) are recorded; the measure correlates
pairwise state distances with pairwise response distances across probes.

The loop is written out here rather than using run_coupled, because the probe must
inject an impulse mid-run and the pre-probe state must be captured at the trigger
— neither of which the closed orchestrator allows.

Read in absolute: a correlation near 1 means the response follows the state
(transient state-dependence); near 0 means the response does not follow the state
(chaos). No persistent memory can live here anyway — D and K are fixed, there is
no slow state — so this probes only the transient, state-carried dependence, as a
baseline for when a memory substrate is later added.
"""
import numpy as np

from heartbrain import heart, coupling
from heartbrain.infra import analysis
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
TARGET_PHASE = 1.5       # a fast/chaotic phase (valley of B at (1,1))
IMPULSE_DX = 0.5         # fixed perturbation added to x during the impulse
IMPULSE_LEN = 5          # impulse duration in steps
RESPONSE_LEN = 300       # response window after the impulse onset (~half a cycle)
MIN_GAP = 1500           # minimum steps between probes (different states)


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


    # === Probed run: fire a fixed impulse at TARGET_PHASE, record state + response ===
    # Phase proxy atan2(y, x) is available every step (the Hilbert phase is only
    # known post hoc). A probe fires when this proxy crosses TARGET_PHASE and at
    # least MIN_GAP steps have passed since the last fire, with room left for the
    # full response.
    states = []
    responses = []

    last_fire = -MIN_GAP
    previous_shift = None

    in_probe = False
    probe_onset = None
    response_buffer = []

    for t in range(N_STEPS):
        heart_x, heart_y = oscillator.get_state()

        phase_proxy = np.arctan2(heart_y, heart_x)
        shifted = np.angle(np.exp(1j * (phase_proxy - TARGET_PHASE)))

        brain_signal = rnn.project_state_onto(direction=K_baseline_x)

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
                in_probe = True
                probe_onset = t
                last_fire = t
                response_buffer = []
                # Pre-probe state: the network state at the trigger, before the
                # impulse-affected step advances it. Copied, since ``state`` is the
                # live array and the network keeps evolving.
                states.append(rnn.state.copy())

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

    states = np.array(states)
    responses = np.array(responses)
    print(f"probes fired: {len(states)} (state_dim={states.shape[1]}, response_len={responses.shape[1]})")    # Log


    # === Measure: is the response predictable from the pre-probe state? ===
    dependence = analysis.state_response_dependence(states=states, responses=responses)
    print(f"\n(k_hb={K_HB}, k_bh={K_BH}, target_phase={TARGET_PHASE})")    # Log
    print(f"state_response_dependence = {dependence:.4f}")    # Log
    print("  ~1 -> response follows the state (transient state-dependence)")    # Log
    print("  ~0 -> response does not follow the state (chaos)")    # Log

    null_rng = np.random.default_rng(99)
    dependence_null = analysis.state_response_dependence_null(states=states,
                                                              responses=responses,
                                                              rng=null_rng)
    print(f"state_response_dependence (null) = {dependence_null:.4f}")    # Log
    print(f"  true={dependence:.4f} vs null={dependence_null:.4f}: "
          f"{'above floor' if dependence > dependence_null + 0.05 else 'indistinguishable from chance'}")    # Log


if __name__ == "__main__":
    main()
