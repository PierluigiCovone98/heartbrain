"""Why does the lag curve look the way it does? Same run, four readouts.

The state-dependence-vs-lag curve did not decay: it collapsed within one lag
step, resurged almost fully at lag ~300-450, again (lower) at ~700-800, with a
possible third bump at ~1050-1300. Two readings are compatible with that shape:

  (H-mask)  phase mask — lags falling on plateau phases sample nearly-identical
            states; the distance column degenerates and the correlation dies as
            an artifact of *where* the state is read, not of forgetting. Memory
            would then live only in the envelope of the peaks.
  (H-recur) phase recurrence — lags near k*T sample the same phase as the
            trigger; the correlation there reflects content that persists
            across whole cycles. The valleys would be real decorrelation.

This script reruns the exact same experiment (same loop, same constants; the
run is deterministic, so the dependence column MUST reproduce the existing
table digit for digit — that is the validation gate) and adds four readouts,
all from the single run:

  A. Cardiac period at (1,1): every upward crossing of the target phase, not
     just the triggering ones -> cycle lengths: mean, std, CV, percentiles,
     and the predicted recurrence lags k*T +/- sqrt(k)*std to lay against the
     observed peaks. (Was explicitly unmeasured until now.)
  B. Null with dispersion: N_SHUFFLES probe-level shuffles per lag -> null
     mean AND std -> a z-score and empirical p for every point. The same
     permutation set is used at every lag, extending the original fixed-seed
     logic: differences between lags cannot come from shuffle randomness.
  C. Phase of the sampled state per lag: circular mean and resultant length R.
     If the peaks sit at k*T, the circular mean there must return to ~1.2;
     if R decays with lag (period jitter accumulating), that alone can
     produce the decaying envelope without any forgetting in the substrate.
  D. Coefficient of variation of the pairwise state distances per lag
     (Euclidean). If the CV collapses exactly in the valleys of the curve,
     the valleys are the mask (H-mask); if it stays flat, they are real.

Runtime: the null adds ~N_SHUFFLES x len(LAGS) calls to the dependence
measure (~1-2 min on top of the run). Nothing else changes.
"""
import numpy as np
from collections import deque

from heartbrain import heart, coupling
from heartbrain.infra import analysis, plotting
from heartbrain.infra.persistence import networks, experiments


# Constants — identical to state_dependence_probe.py. Do not touch: the
# validation gate (exact reproduction of the dependence column) depends on it.
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
TARGET_PHASE = 1.2
IMPULSE_DX = 0.5
IMPULSE_LEN = 5
RESPONSE_LEN = 300
MIN_GAP = 1500

# === lags
LAGS = list(range(0, 1401, 50))
MAX_LAG = max(LAGS)

# === null
NULL_SEED = 99           # same seed family as the original
N_SHUFFLES = 1000        # raised from the original's floor-only estimate:
                         # the std needs samples


def euclidean_pdist(X):
    """Condensed pairwise Euclidean distances (diagnostic D only; the
    dependence measure itself always goes through analysis.*)."""
    iu = np.triu_indices(len(X), 1)
    d = X[iu[0]] - X[iu[1]]
    return np.sqrt((d * d).sum(axis=1))


def circular_mean_R(phases):
    z = np.exp(1j * np.asarray(phases)).mean()
    return float(np.angle(z)), float(np.abs(z))


def main():

    # === Vanilla RNN setup (verbatim from state_dependence_probe) ===
    rnn = networks.load_network(RNN_BASELINE_NAME)
    rnn.scale_W_hh(g=G)
    print(f"OK - {RNN_BASELINE_NAME} network loaded and scaled (g={G}).")    # Log

    _, arrays = experiments.load_experiment(GAIN_SWEEP_EXP)
    x = arrays["x"]
    N = rnn.N

    oscillator = heart.Heart()

    # === Coupling components (verbatim) ===
    heart_to_brain_rng = np.random.default_rng(SEED2)
    K_baseline = coupling.create_K_baseline(rng=heart_to_brain_rng, N=N, fan_in=2)
    K_baseline_x = coupling.extract_K_baseline_x(K_baseline)
    K = coupling.build_K(k_hb=K_HB, K_baseline=K_baseline)

    brain_to_heart_rng = np.random.default_rng(SEED3)
    D_baseline = coupling.create_D_baseline(rng=brain_to_heart_rng, N=N)

    # === Warmup (verbatim) ===
    for _ in range(WARMUP_STEPS):
        heart_state = oscillator.get_state()
        projection = rnn.project_state_onto(direction=D_baseline)
        perturbation = coupling.heart_to_brain_bias_perturbation(K=K, heart_state=heart_state)
        rnn.apply_bias_perturbation(perturbation)
        rnn.step(x=x)
        oscillator.step(sigma=K_BH * projection, dt=DT)

    # === Probed run: identical loop, plus three recordings that do not touch
    # the dynamics: (i) every upward crossing of the target phase, for the
    # period; (ii) the phase proxy in a rolling window mirroring the state
    # window, for the phase-at-lag readout; (iii) nothing else.
    state_history = deque(maxlen=MAX_LAG + 1)
    phase_history = deque(maxlen=MAX_LAG + 1)
    states_by_lag = {lag: [] for lag in LAGS}
    phases_by_lag = {lag: [] for lag in LAGS}
    responses = []

    crossing_times = []          # ALL upward crossings, probing or not

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

        state_history.append(rnn.state.copy())
        phase_history.append(phase_proxy)

        if in_probe:
            response_buffer.append(brain_signal)
            if len(response_buffer) >= RESPONSE_LEN:
                responses.append(np.array(response_buffer))
                in_probe = False
                response_buffer = []

        # Same crossing rule as the trigger, but recorded unconditionally:
        # this is the period measurement.
        crossed = (previous_shift is not None) and (previous_shift < 0) and (shifted >= 0)
        if crossed:
            crossing_times.append(t)

        # Trigger (verbatim semantics: evaluated only when not in_probe).
        if (not in_probe) and crossed:
            if (t - last_fire) >= MIN_GAP and (t + RESPONSE_LEN < N_STEPS):
                if len(state_history) > MAX_LAG:
                    in_probe = True
                    probe_onset = t
                    last_fire = t
                    response_buffer = []
                    for lag in LAGS:
                        states_by_lag[lag].append(state_history[-1 - lag])
                        phases_by_lag[lag].append(phase_history[-1 - lag])
                else:
                    skipped_probes += 1

        previous_shift = shifted

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
    print("VALIDATION: the dependence column below must match the existing table "
          "digit for digit (deterministic run). If it does not, stop and diff.")    # Log

    # === A. Cardiac period at the operating point ===
    crossing_times = np.array(crossing_times)
    periods = np.diff(crossing_times).astype(float)
    T_mean = periods.mean()
    T_std = periods.std()
    q5, q25, q50, q75, q95 = np.percentile(periods, [5, 25, 50, 75, 95])
    print(f"\n=== A. cardiac period at (k_hb={K_HB}, k_bh={K_BH}), "
          f"phase-{TARGET_PHASE} crossings ===")    # Log
    print(f"cycles: {len(periods)}")    # Log
    print(f"T mean = {T_mean:.2f} steps   std = {T_std:.3f}   CV = {T_std / T_mean:.5f}")    # Log
    print(f"min/max = {periods.min():.0f}/{periods.max():.0f}   "
          f"p5/p25/p50/p75/p95 = {q5:.0f}/{q25:.0f}/{q50:.0f}/{q75:.0f}/{q95:.0f}")    # Log
    print("expected recurrence lags (k*T, spread ~ sqrt(k)*std):")    # Log
    for k in range(1, 5):
        print(f"  k={k}:  {k * T_mean:8.1f}  +/- {np.sqrt(k) * T_std:.1f}")    # Log

    # === B+C+D per lag ===
    # Null: probe-level shuffle done by permuting the response rows and calling
    # the SAME measure — the metric and the correlation are whatever
    # analysis.state_response_dependence uses, by construction. The permutation
    # set is generated once (fixed seed) and reused at every lag.
    perm_rng = np.random.default_rng(NULL_SEED)
    perms = [perm_rng.permutation(n_probes) for _ in range(N_SHUFFLES)]

    print(f"\n(k_hb={K_HB}, k_bh={K_BH}, target_phase={TARGET_PHASE}, "
          f"n_shuffles={N_SHUFFLES})")    # Log
    print(f"{'lag':>6}  {'dependence':>11}  {'null_mu':>8}  {'null_sd':>8}  "
          f"{'z':>7}  {'p_emp':>7}  {'ph_mean':>8}  {'ph_R':>6}  "
          f"{'dS_cv':>7}  {'cycles':>7}")    # Log

    dependences, zs, null_sds, ph_Rs, dS_cvs = [], [], [], [], []

    for lag in LAGS:
        states_lag = np.array(states_by_lag[lag])[:n_probes]
        phases_lag = np.array(phases_by_lag[lag])[:n_probes]

        dependence = analysis.state_response_dependence(states=states_lag,
                                                        responses=responses)

        null = np.empty(N_SHUFFLES)
        for i, p in enumerate(perms):
            null[i] = analysis.state_response_dependence(states=states_lag,
                                                         responses=responses[p])
        null_mu = null.mean()
        null_sd = null.std()
        z = (dependence - null_mu) / null_sd if null_sd > 0 else np.nan
        p_emp = (1 + np.sum(np.abs(null - null_mu) >= abs(dependence - null_mu))) \
                / (1 + N_SHUFFLES)

        ph_mean, ph_R = circular_mean_R(phases_lag)

        dS = euclidean_pdist(states_lag)
        dS_cv = dS.std() / dS.mean()

        cycles = lag / T_mean

        dependences.append(dependence)
        zs.append(z)
        null_sds.append(null_sd)
        ph_Rs.append(ph_R)
        dS_cvs.append(dS_cv)

        print(f"{lag:>6}  {dependence:>+11.4f}  {null_mu:>+8.4f}  {null_sd:>8.4f}  "
              f"{z:>+7.2f}  {p_emp:>7.4f}  {ph_mean:>+8.3f}  {ph_R:>6.3f}  "
              f"{dS_cv:>7.4f}  {cycles:>7.2f}")    # Log

    dependences = np.array(dependences)
    zs = np.array(zs)
    print(f"\nmax = {dependences.max():+.4f} at lag {LAGS[int(dependences.argmax())]}"
          f" (z = {zs[int(dependences.argmax())]:+.1f})")    # Log
    print(f"min = {dependences.min():+.4f} at lag {LAGS[int(dependences.argmin())]}"
          f" (z = {zs[int(dependences.argmin())]:+.1f})")    # Log

    plotting.plot_curve(x_values=LAGS,
                        y_values=zs,
                        name="state_dependence_vs_lag_zscore",
                        subdir="lag",
                        x_label="lag (steps before the probe)",
                        y_label="z vs null")

    plotting.plot_curve(x_values=LAGS,
                        y_values=null_sds,
                        name="state_dependence_vs_lag_null_sd",
                        subdir="lag",
                        x_label="lag (steps before the probe)",
                        y_label="null sd",
                        color="grey")

    plotting.plot_curve(x_values=LAGS,
                        y_values=ph_Rs,
                        name="sampled_phase_R_vs_lag",
                        subdir="lag",
                        x_label="lag (steps before the probe)",
                        y_label="circular R of sampled phase")

    plotting.plot_curve(x_values=LAGS,
                        y_values=dS_cvs,
                        name="state_distance_cv_vs_lag",
                        subdir="lag",
                        x_label="lag (steps before the probe)",
                        y_label="CV of pairwise state distances")


if __name__ == "__main__":
    main()
