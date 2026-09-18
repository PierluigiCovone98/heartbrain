# heartbrain

A bidirectionally coupled system made of a Van der Pol oscillator (a functional analogue of the heartbeat, called `heart` in the code) and a 64-neuron vanilla recurrent network at the edge of chaos (called `brain`). The two components exchange signals through two asymmetric channels, `oscillator → RNN` (strength `k_hb`) and `RNN → oscillator` (strength `k_bh`), while the network weights stay fixed.

The system is studied as a dynamical substrate: the aim is to find internal quantities that could modulate the network's learning without an external loss function.

> **Status: paused.** The project was developed full-time for about six months and has been paused since the beginning of August 2026. The measures and code described below are verified and can be reused. The learning rule, which was the step these measures were preparing for, was never built. See [Status](#status).

## Research question

The project asks the following question:

**Can a multi-component system with internal state produce quantities informative enough to modulate the weight updates of its network, without relying on an external objective?**

Two questions had to be answered before any update rule could be written:

1. **Which internal properties of the coupled system are structurally stable and informative?** In other words, which ones depend on the structure of the system rather than on the accident of one chaotic trajectory?
2. **Does the coupled substrate with fixed weights already depend on its own history in some way, or is it blind to its past?**

If both answers were constructive, they would point to a class of systems where training and use are not separate phases, and where internal organisation is driven by the system's own quantities instead of an externally defined objective.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[experiments,dev]"
```

## Running an experiment

**1. Create the baseline network (REQUIRED).** Almost every experiment loads a saved network and its input from disk. `data/` is not versioned, so this step has to be run first on a fresh clone:

```bash
python experiments/brain/gain_sweep.py
```

This writes `data/saved_networks/gain_sweep_baseline.npz`.

**2. Run any experiment from the repository root:**

```bash
python experiments/experiment_name.py
```

- **Parameters** are module-level constants at the top of each script (`K_HB`, `K_BH`, `G`, `N_STEPS`, seeds, …). The scripts have no command-line interface: to explore another operating point, edit the constants.
- **Figures** go to `output/<subdir>/`. The subdirectory is set by the script's `SUBDIR` constant, and `output/` is not versioned.
- **Numbers** are printed to stdout. The module docstring of each script states its question, its method and, where one was made, its pre-registered expectation.

**Tests:**

```bash
pytest
```

## Repository map

### The system — `src/heartbrain/`

| File | Content |
|---|---|
| [heart.py](src/heartbrain/heart.py) | `Heart`: Van der Pol oscillator, driven by an external forcing `sigma` |
| [integrators.py](src/heartbrain/integrators.py) | RK4 step used by the heart |
| [brain.py](src/heartbrain/brain.py) | `VanillaRNN`: recurrent network, gain-scaled recurrent matrix, resettable state |
| [coupling.py](src/heartbrain/coupling.py) | Coupling channels: `K` (heart → brain, perturbs the network bias) and `D` (brain → heart projection) |
| [coupled_system.py](src/heartbrain/coupled_system.py) | Simulation loops: `run_heart_to_brain`, `run_brain_to_heart`, `run_coupled` (bidirectional) |

### The measures — `src/heartbrain/infra/analysis.py`

| Function(s) | Purpose |
|---|---|
| `low_pass_filter`, `high_pass_filter` | Split the network signal into slow and fast components |
| `instantaneous_phase`, `instantaneous_amplitude` | Hilbert-based phase and envelope (non-causal) |
| `measure_plv`, `phase_locking_value`, `plv_over_windows` | Phase Locking Value: global, or on consecutive windows |
| `resolve_by_phase` | Average a quantity per cardiac-phase bin |
| `extract_windows_at_phase`, `cycle_reproducibility` | Reproducibility B across consecutive cycles, plus a null |
| `extract_periods`, `period_variability` | Cardiac period and its variability (SDNN / RMSSD analogues) |
| `state_response_dependence`, `state_response_dependence_null` | Does the pre-probe state predict the response to an impulse? |

Other infrastructure in `src/heartbrain/infra/`: `plotting.py` (figures), `persistence/` (saving and loading networks and experiment conditions), `experiment_logger.py`, `_paths.py`.

### The experiments — `experiments/`

The experimental work first characterises each coupling direction on its own, then the bidirectional case.

**Isolated components**

| Script | Principle |
|---|---|
| [brain/gain_sweep.py](experiments/brain/gain_sweep.py) | Sweeps the recurrent gain `g` to find the chaotic regime. Saves the baseline network. The RNN is chaotic at `g = 2.5`. |
| [heart_only.py](experiments/heart_only.py) | Autonomous heart: phase portrait and convergence to the limit cycle |

**Oscillator → RNN (`k_bh = 0`)**

| Script | Principle |
|---|---|
| [coupled_run.py](experiments/coupled_run.py) | Raw run at a fixed `k_hb` |
| [khb_sweep.py](experiments/khb_sweep.py) | PLV as a function of `k_hb`, symmetric around zero |
| [plv_invariance.py](experiments/plv_invariance.py) | Is the PLV invariant under a 1e-13 perturbation of the initial state? |
| [fast_dynamics_structure.py](experiments/fast_dynamics_structure.py) | Amplitude and reproducibility of the network's fast component, resolved by cardiac phase |
| [fast_structure_invariance.py](experiments/fast_structure_invariance.py) | Invariance test for the two phase-resolved curves |
| [amplitude_vs_reproducibility.py](experiments/amplitude_vs_reproducibility.py) | Are amplitude and B redundant across many `k_hb`? |

**RNN → oscillator (`k_hb = 0`)**

| Script | Principle |
|---|---|
| [brain_to_heart_run.py](experiments/brain_to_heart_run.py) | Raw run at a fixed `k_bh` |
| [brain_to_heart_decomposition.py](experiments/brain_to_heart_decomposition.py) | Drives the heart with the full forcing, its constant bias alone, or its chaotic fluctuation alone |
| [bias_suppression_sweep.py](experiments/bias_suppression_sweep.py) | Threshold at which the bias suppresses the oscillation (inverse Hopf) |
| [heart_rate_variability_sweep.py](experiments/heart_rate_variability_sweep.py) | Beat-period variability under pure-chaos forcing, vs `k_bh` |

**Bidirectional**

| Script | Principle |
|---|---|
| [coupled_bidirectional_run.py](experiments/coupled_bidirectional_run.py) | Raw run at a fixed `(k_hb, k_bh)`. The `(k_hb, k_bh)` plane was explored by rerunning it at hand-picked points |
| [coupled_reproducibility.py](experiments/coupled_reproducibility.py) | Is the fast dynamics still chaotic once the loop is closed? |

## Measures: three structural invariants

The three quantities below were identified in the `oscillator → RNN` setting and verified as **structural invariants**. The test runs two identical simulations whose initial network states differ by 1e-13. The trajectories diverge strongly, because the regime is chaotic, and a structural quantity must stay put anyway.

- **PLV (Phase Locking Value).** A scalar measure of phase synchronisation between the heart and a projection of the network: the global relation between the two systems. At `k_hb = 0.5`, the trajectories diverge by 36% while the PLV changes by 4.59 × 10⁻⁵.
- **Envelope amplitude by cardiac phase.** An 18-bin curve giving the typical amplitude of the network's fast dynamics in each phase of the cardiac cycle. It shows intermittency: chaotic bursts and calm windows, locked to specific phases of the beat. The correlation between the curves before and after perturbation is 0.9995.
- **Reproducibility B by cardiac phase.** An 18-bin curve measuring how identically the network's dynamics repeats between consecutive beats at each phase. The correlation before and after perturbation is 0.9964.

**Non-redundancy.** Amplitude and B correlate at about −0.53 across conditions, and 72% of B's variance is independent of amplitude. PLV is a scalar, so it cannot carry the information of an 18-degree-of-freedom curve (a dimensional argument).

## Status

Paused since August 2026. The three invariants above are verified. With fixed weights, closing the loop stabilises the dynamics rather than enriching it.

### Open

- **Memory in the fixed-weight substrate.** At lag 0 the response depends only negligibly on the prior state (~2% of variance); the behaviour at longer lags is unresolved.
- **A causal modulator.** The invariants rely on a non-causal Hilbert transform, while a learning rule needs a coherence signal computed online from past data. At the bidirectional operating point (1, 1) the PLV is saturated (0.9938 ± 0.0003), so it cannot serve.
- **A top-down view.** The measures were built bottom-up, tied to specific versions of the system. Organising principles that transfer across variants are missing.

### Next

Rather than resuming from the last experiment, the intention is to restart with a different, top-down approach, supported by tools for inspecting the internal dynamics of models.
