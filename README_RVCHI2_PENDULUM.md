# Neural chi-square robust Q-learning on Pendulum

This guide describes the neural Pendulum implementation released with
*Finite-Time Convergence of Single-Trajectory Chi-Square Robust Q-Learning
With Linear Function Approximation* by Saptarshi Mandal, Yashaswini Murthy,
and R. Srikant.

It complements the command-focused
[Pendulum reproducibility workflow](reproducibility/paper_experiments/pendulum/README.md).
The [root README](README.md) explains installation, companion-data extraction,
and the combined MiniCliff/Pendulum release.

## Scope and claim boundary

The released Pendulum study is an empirical neural-control extension of the
paper's chi-square robust Q-learning framework. The finite-time theorem in the
paper concerns fixed linear features. Replay buffers, target networks, learned
neural features, Adam/AdamW, exponential moving averages, and online neural
control are practical components of this experiment; the theorem is not
claimed to cover those additions.

The release supports two different reproducibility tasks:

- **Artifact reconstruction:** authenticate the preserved ten-seed reporting
  bundle and frozen executable capsule, then reconstruct the paper aggregates
  and figures without training or collecting new trajectories.
- **Fresh retraining:** optionally train five development seeds and, only after
  a new development authorization passes, ten reporting seeds. This is an
  independent seeded comparison, not a promise of a byte-identical historical
  trajectory.

The source tag contains the implementation and frozen configurations. The
larger authenticated records are distributed separately in
`pendulum-companion-data.tar.gz`; see
[`docs/COMPANION_DATA.md`](docs/COMPANION_DATA.md).

## Experiment at a glance

The task is a custom NumPy implementation of Pendulum-v1-style dynamics with
a hidden actuator fault. At each step, a Bernoulli fault can reverse the sign
of the commanded torque. The nominal fault probability is `0.10`, and the
Pearson chi-square radius is `0.25`.

The implementation uses the convention
`D_chi2(Ber(p) || Ber(0.10)) = (p - 0.10)^2 / (0.10 * 0.90)`. Requiring this
divergence to be at most `0.25` gives the simplex-clipped fault-rate interval
`[0, 0.25]`. The plotted profiles hold one common fault probability across all
state-action pairs, so they form a one-dimensional homogeneous slice of the
broader state-action-rectangular ambiguity class.

| Item | Frozen value |
|---|---|
| State | Angle and angular velocity |
| Observation | `(cos(theta), sin(theta), angular_velocity)` |
| Neural-network input (scale `(1, 1, 8)`) | `(cos(theta), sin(theta), angular_velocity/8)` |
| Discrete torque commands | `(-2, 0, 2)` |
| Fault | Reverse the commanded torque sign |
| Nominal fault probability | `0.10` |
| Gravity, mass, length | `10`, `1`, `1` |
| Time step | `0.05` |
| Torque and speed limits | `2` and `8` |
| Initial angle | Uniform on `[-pi, pi)` |
| Initial angular velocity | Uniform on `[-1, 1)` |
| Raw reward | `-(wrap(angle)^2 + 0.1 velocity^2 + 0.001 applied_torque^2)` |
| Training reward scale | `0.01` |
| Horizon | `200` steps |
| Termination | No physical termination; time-limit truncation at the horizon |
| Time-limit Bellman treatment | Bootstrap through truncation |

The implementation does not instantiate Gymnasium's Pendulum environment.
Gymnasium is included in the frozen dependency set, but the executed dynamics
are defined in [`src/rvchi2_dqn/envs.py`](src/rvchi2_dqn/envs.py).
Because this task has no physical termination, its reported failure probability
is always zero; the configured competence failure-rate threshold of `1.0` is
therefore nonbinding.

## Compared methods

| Configuration name | Paper label | Description |
|---|---|---|
| `nominal` | Nominal DDQN | Double-DQN continuation trained from transitions sampled under the nominal fault kernel. |
| `exact` | Exact inner problem | Enumerates healthy and reversed-torque successors and evaluates the exact binary chi-square robust continuation. Used only in the seed-31 supplement. |
| `affine` | RVChi2-A | Learns affine variational location and positive-scale heads on the bounded Q-network representation, with explicit projections. |
| `full_nn` | RVChi2-N | Learns independent neural variational location and positive-scale networks from normalized observations. |

All enabled methods start from the same nominally pretrained Q-network, the
same Q-optimizer state, and deep copies of the same replay prefix. Each branch
then receives its own environment, exploration generator, replay sampler,
optimizer, and newly collected data. Environment-step and Q-update budgets are
matched across branches.

## Released study specifications

| Study file | Phase and seeds | Methods | Role |
|---|---|---|---|
| [`development_pendulum_full_nn.json`](configs/rvchi2_dqn/v1/development_pendulum_full_nn.json) | Development, `31-35` | nominal, affine, full-NN | Development authorization study |
| [`reporting_pendulum_full_nn.json`](configs/rvchi2_dqn/v1/reporting_pendulum_full_nn.json) | Reporting, `101-110` | nominal, affine, full-NN | Paper-facing reporting panel |
| [`appendix_pendulum_exact_inner_seed31_v1.json`](configs/rvchi2_dqn/v1/appendix_pendulum_exact_inner_seed31_v1.json) | Development, seed `31` | nominal, exact | One-seed exact-inner comparison |
| [`ablation_pendulum_full_nn_lr1e4_seed31.json`](configs/rvchi2_dqn/v1/ablation_pendulum_full_nn_lr1e4_seed31.json) | Development, seed `31` | nominal, affine, full-NN | One-seed neural supplement |

The last file's historical `ablation` name does not introduce an additional
numerical override: it resolves to the same methods and training settings as
development seed 31.

The official `full-rerun` command executes only the five-seed development and
ten-seed reporting studies. It does not execute either one-seed supplement.

### How sparse study files become complete configurations

The short JSON files intentionally contain only the study identity, reserved
seed panel, enabled methods, continuation policy, and two Q-learning rates.
The runner resolves a complete per-seed configuration in this order:

1. generic defaults from `ExperimentConfig`;
2. Pendulum-specific defaults from `task_defaults("pendulum")`;
3. the `development` or `reporting` phase profile; and
4. the JSON `overrides` object.

The resolution and validation logic is in
[`src/rvchi2_dqn/config.py`](src/rvchi2_dqn/config.py). Every completed run
writes all inherited and overridden values to its own `config.json`; the
companion archive therefore contains the exact resolved configuration for
each historical reporting seed and each released seed-31 supplement.

## Environment and release setup

Run commands from the repository root. Python 3.10.13 was used for the release
audit:

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements-freeze.txt
```

The principal frozen versions are Torch 2.11.0, NumPy 2.2.6, Gymnasium 1.2.3,
Pandas 2.3.3, and Matplotlib 3.10.9. The audited platform was macOS arm64 with
Apple's Accelerate BLAS. Fresh neural trajectories on other hardware or
native-library builds are not promised to be bit-identical.

Download the version-matched release assets, verify their checksums, and
extract `pendulum-companion-data.tar.gz` at the repository root before using
the authenticated wrapper. The Pendulum archive contains 158 files totaling
36,446,249 extracted bytes; its content inventory SHA-256 is
`d1c278aaeaab7224fc3a10f64af0a9c4ab33784adc2dd2c62be5373c616ef01d`.

## Verify or reconstruct the released results

### Read-only verification

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  reproducibility/paper_experiments/pendulum/reproduce.py \
  verify --require-live-correspondence
```

This authenticates the frozen 16-file executable capsule, canonical reporting
tree, ten per-seed manifests, development authorization chain, two seed-31
supplements, dependency record, and byte correspondence of the convenient live
source tree. It does not create an output directory.

The focused release test can be run separately:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  discover -s tests \
  -p 'test_pendulum_paper_reproducibility.py' -v
```

This test suite exercises authentication, inventory-tamper rejection,
output-path guards, the expensive-run confirmation guard, and snapshot
immutability. It does not train a neural agent. The release does not include a
separate low-budget Pendulum training specification; the documented fresh
training jobs below use the frozen scientific budgets.

### Tables and figures without training

The destination must be fresh and nonexistent:

```bash
PEND_PARENT="$(.venv/bin/python -c \
  'import tempfile; print(tempfile.mkdtemp(prefix="pendulum-parent."))')"
PEND_OUT="$PEND_PARENT/rebuilt"

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  reproducibility/paper_experiments/pendulum/reproduce.py \
  reproduce --output "$PEND_OUT"
```

This command copies authenticated raw seeds 101-110 into an isolated staging
tree, reconstructs seven aggregate files, renders the figures, reauthenticates
the canonical inputs, and publishes the output atomically. It performs no
optimizer update, environment rollout, checkpoint selection, or tuning.

The primary figure is:

```text
$PEND_OUT/figures/robustness_profiles.pdf
```

All seven numerical aggregates must be byte-identical to the authenticated
canonical files, and all six PNGs must have identical decoded pixels. PDF
container bytes may differ because plotting libraries can embed creation
metadata.

## Complete resolved hyperparameters

The following values apply to both the main development and reporting studies.
Those phases differ in seeds and authorization role, not numerical training
settings.

### Robust problem, targets, and evaluation

| Parameter | Value |
|---|---:|
| Nominal fault probability | `0.10` |
| Pearson chi-square radius | `0.25` |
| Discount factor | `0.99` |
| Certified evaluation probabilities | `0, 0.05, 0.10, 0.15, 0.20, 0.25` |
| Out-of-distribution probabilities | `0.35, 0.50` |
| Variational floor `ell` | `0.001` |
| Eta constraint | Affine row L2 bound `18`; full-NN output-magnitude bound `18` |
| Affine rho bound | `24` |
| Affine initial scale `u` | `0.04` |
| Full-NN initial scale `u` | `0.05` |
| Successor-Q and Bellman-target clipping interval | `[-16.4, 0]` |
| Development-gate limit on each method's maximum robust-block mean target-clipping fraction | `0.01` |
| Affine eta-projection and scale-floor diagnostic thresholds | `0.01` each |
| Evaluation episodes per method and probability | `100` |
| Evaluation horizon | `200` |
| Intermediate evaluation interval | Every `5` outer blocks |
| Support-grid points per state coordinate | `41` |
| Legacy `heldout_probe_count` | `2,048` (validated but unused) |
| Nominal competence raw-return threshold | `-225` |
| Nominal competence failure-rate threshold | `1.0` |
| Minimum affine-minus-nominal raw-return difference at `p=0.10` | `-100` |
| Artifact/configuration schema version | `2` |

### Q-learning, replay, and exploration

| Parameter | Value |
|---|---:|
| Q hidden widths | `(128, 128)` |
| Replay capacity | `1,000,000` transitions |
| Batch size | `256` |
| Learning starts | `20,000` transitions |
| Nominal pretraining steps | `150,000` |
| Nominal Q updates per eligible environment step | `1` (`130,001` total) |
| Nominal target-network update interval | `1,000` environment steps |
| Nominal pretraining Q learning rate | `2.5e-4` |
| Robust-stage Q learning rate | `1e-4` |
| Q weight decay | `0` |
| Q and auxiliary gradient clipping | L2 norm `10` |
| Pretraining epsilon | Linear `1.0` to `0.05` over `120,000` steps |
| Robust-branch epsilon | `0.05` |
| Coverage-rescue random-action fraction | `0` |

### Robust-stage optimization

| Parameter | Value |
|---|---:|
| Outer blocks | `30` |
| Collection per block and method | `3,000` transitions |
| Q updates per block and method | `1,000` |
| Auxiliary updates per block and learned variational method | `5,000` |
| Auxiliary learning rate | `0.001` |
| Full-NN auxiliary weight decay | `1e-5` |
| Deployed-auxiliary EMA decay | `0.99` |
| Device | CPU |
| Torch intra-operation threads | `1` |
| Deterministic Torch algorithms | Enabled |
| Continue after failed early nominal gate | `true` for all four released studies |
| Stop after nominal gate | `false` |

Each method branch therefore adds 90,000 transitions to the shared 150,000
transition prefix and performs 30,000 robust Q updates. Each learned
variational branch performs 150,000 auxiliary updates. The final replay size is
240,000, below the one-million-entry capacity, so no stored transition is
overwritten.

## Networks and initialization

| Component | Architecture | Initialization and constraints |
|---|---|---|
| Q network | `3 -> 128 tanh -> 128 tanh`; normalized `[1,h]/sqrt(129)` feature; bias-free `129 -> 3` head | Hidden weights orthogonal with gain `sqrt(2)`; hidden biases zero; Q head orthogonal with gain `0.01` |
| Affine eta | Three action-specific rows over the 129-dimensional bounded Q feature | Initialized from the pretrained Q-head weights divided by `gamma`, then projected to row norm at most `18` |
| Affine positive scale | Three action-specific rows over 257 split-sign features | Nonnegative, row norm at most `24`, intercept-only initialization giving constant `u=0.04` |
| Full-NN eta | Independent `3 -> 128 tanh -> 128 tanh -> 3` MLP | Hidden weights orthogonal with gain `sqrt(2)`, zero biases; output gain `0.01`; deployed value `18*tanh(raw)` |
| Full-NN positive scale | Independent `3 -> 128 tanh -> 128 tanh -> 3` MLP | Hidden weights orthogonal with gain `sqrt(2)` and zero biases; output weights zero; output bias chosen so `0.001 + softplus(raw) = 0.05` |

Normalized observations are clipped to `[-5, 5]` by a network-constructor
default. This clip is inactive for every valid scaled Pendulum observation but
remains part of the frozen constructor semantics. Training floating-point
tensors are explicitly float32.

### Optimizers and losses

| Parameters | Optimizer |
|---|---|
| Nominal-pretraining Q | Adam, learning rate `2.5e-4`, weight decay `0` |
| Robust-stage Q | Deep-copied Adam state, learning rate reset to `1e-4`, weight decay `0` |
| Affine auxiliary | Adam, learning rate `0.001`, weight decay `0` |
| Full-NN auxiliary | AdamW, learning rate `0.001`, weight decay `1e-5` |

The frozen Torch 2.11 defaults supply unoverridden Adam/AdamW settings,
including betas `(0.9, 0.999)`, epsilon `1e-8`, and AMSGrad disabled. No
learning-rate scheduler is used. Q updates minimize mean Smooth-L1 loss with
beta `1`; auxiliary updates maximize the sampled variational objective. The
deployed auxiliary begins as an exact frozen copy of its initialized online
auxiliary and receives an EMA update after every auxiliary optimizer step.

The configuration field `auxiliary_weight_decay=1e-5` applies only to the
full-NN AdamW optimizer. The affine optimizer is constructed without a
weight-decay argument and therefore uses zero weight decay.

### Replay representation and sampling

Replay is uniform and non-prioritized, and minibatches are sampled with
replacement. Observations and rewards are stored as float32, actions as int64,
and termination, truncation, and fault indicators as booleans. Each branch
receives the same transition prefix but a freshly seeded replay sampler.

## Training sequence

For each seed, the runner performs the following steps:

1. Seed Python, NumPy, and Torch and enable deterministic CPU Torch behavior.
2. Initialize the Q-network and collect/train a 150,000-transition nominal
   prefix using DDQN targets.
3. Evaluate the pretrained nominal policy at the nominal fault probability.
   In the released development/reporting configurations, the early competence
   result is recorded but `continue_after_failed_pretraining_gate=true` makes
   this early gate diagnostic rather than stopping the run.
4. Deep-copy the pretrained Q-network and replay prefix into each enabled
   branch, construct a branch Adam optimizer, load a deep copy of the
   pretrained Adam state, and then reset only its learning rate.
5. For each of 30 blocks, collect 3,000 new transitions per method, freeze one
   target-Q copy per branch, perform 5,000 auxiliary updates for RVChi2-A and
   RVChi2-N, then perform 1,000 Q updates for every branch.
6. Evaluate greedy policies every five blocks at fault probabilities `0.10`
   and `0.25` using common random numbers.
7. Run the final eight-probability sweep, calculate diagnostics, and write the
   complete per-seed artifact bundle.

The nominal branch uses DDQN targets. The exact branch enumerates the two
actuator modes. RVChi2-A and RVChi2-N form sampled variational targets from
their EMA-deployed auxiliary heads.

## Randomness and common-random-number design

For run seed `S`, the frozen code sets the following seed arguments and active
random-number streams. Some constructor/global seeds are defensive or are
superseded before a draw; the table distinguishes those cases rather than
claiming that every number defines an independently consumed stream.

| Component | Seed | Executed role |
|---|---:|---|
| Python and legacy NumPy globals | `S` | Seeded defensively; the current scientific path makes no later draw from either global generator |
| Torch global generator | `S` | Drives neural initialization; no separately derived per-branch Torch generator is created |
| Prefix environment constructor | `S*10000 + 11` | Superseded by the explicit first-reset seed before any environment draw |
| Prefix first reset/environment stream | `S*10000 + 12` | Draws the initial state and subsequent environment faults/resets |
| Prefix action generator | `S*10000 + 13` | Active epsilon-greedy exploration stream |
| Prefix replay sampler | `S*10000 + 21` | Active minibatch-sampling stream |
| Method branch base, index `i` | `S*100000 + 1000 + 100*i` | Base for the four branch arguments below |
| Branch replay sampler | Base `+1` | Active minibatch-sampling stream |
| Branch environment constructor | Base `+2` | Superseded by the explicit reset seed before any environment draw |
| Branch reset/environment stream | Base `+3` | Draws the branch state and subsequent environment faults/resets |
| Branch action generator | Base `+4` | Active epsilon-greedy exploration stream |
| Evaluation-panel generator | `S*1000000 + 77` | Draws reset seeds and the common fault-uniform matrix |
| Deterministic mode enumerator | `S*100000 + 909` | Passed to the environment constructor, but mode enumeration consumes no RNG |

The main-study method indices are nominal `0`, affine `1`, and full-NN `2`.
The exact supplement uses nominal `0` and exact `1`.

Every evaluation call constructs 100 reset seeds and a `100 x 200` matrix of
fault uniforms. The same panel is reused across methods and fault
probabilities, yielding paired comparisons. Reconstructing the generator for
each checkpoint also repeats the panel over training time.

## Evaluation, diagnostics, and authorization

The final sweep evaluates greedy policies at probabilities
`0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50`. The first six are marked as
inside the certified interval; the last two are out-of-distribution stress
tests. The paper-facing performance variable is undiscounted raw return, while
discounted return is retained as supporting evidence.

Calibration uses a deterministic grid of 41 angles, 41 velocities, and three
actions: 5,043 state-action queries. Sparse support means replay count at most
one. The near-decision-boundary stratum contains states whose nominal Q-action
margin is at or below the empirical 10th-percentile threshold; ties can make
this set larger than exactly 10 percent. Replay-based policy diagnostics use at
most 20,000 evenly spaced query rows.

Backup calibration uses the target Q-network frozen before the final block's Q
updates together with the EMA-deployed auxiliary after that block's auxiliary
updates. It is therefore not a comparison against the post-update final Q
network.

`heldout_probe_count=2048` is defined and validated in the configuration class
but is not consumed by the executed calibration path. It must not be
interpreted as a 2,048-sample held-out evaluation; the actual diagnostic is the
5,043-point deterministic grid described above.
“Held-out” in the frozen appendix-figure title is a historical label for this
deterministic-grid calculation; the content-addressed plotter is left
unchanged.

Seed-level aggregate means use two-sided 95 percent Student-t intervals with
the sample standard deviation (`ddof=1`) and frozen tabulated critical values
for sample sizes through ten. Paired AUC is the unnormalized trapezoidal
integral of method-minus-nominal raw return over `[0.10, 0.25]`.

### Development authorization gate

Reporting is fail-closed. A five-seed development run creates
`reporting_freeze_manifest.json` only when all ten required checks pass:

- final nominal CSV evidence authenticates against each seed summary;
- final nominal competence passes on all five seeds;
- affine-minus-nominal raw-return advantage at `p=0.25` is positive on at
  least four seeds;
- mean affine boundary advantage at `p=0.25` is positive;
- mean unnormalized trapezoidal paired affine AUC over `[0.10, 0.25]` is
  positive;
- affine-minus-nominal raw return at `p=0.10` is at least `-100` for every
  seed;
- branch environment-step budgets match;
- branch Q-update budgets match;
- every method's maximum robust-block mean target-clipping fraction is at or
  below `0.01`; and
- all required numerical values are finite.

Diagnostic-only checks require, on every development seed, affine
selected-action zero-support fraction at most `0.01`, occupied
affine-versus-nominal policy disagreement at least `0.05`, supported affine
calibration Pearson correlation at least `0.90`, supported normalized MAE at
most `0.10`, mean affine eta-projection fraction at most `0.01`, and mean
affine scale-floor fraction at most `0.01`. These diagnostics do not authorize
or block reporting.

The guarded `full-rerun` passes its newly generated development freeze to the
reporting stage. The direct runner accepts any otherwise-valid freeze and does
not enforce freshness; use the wrapper when claiming a fresh authorized
15-seed rerun.

## Optional fresh 15-seed rerun

This is expensive and is not needed to reconstruct the released plots. The
output path must not exist:

```bash
PEND_FULL_PARENT="$(.venv/bin/python -c \
  'import tempfile; print(tempfile.mkdtemp(prefix="pendulum-full-parent."))')"
PEND_FULL_OUT="$PEND_FULL_PARENT/full-rerun"

PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  reproducibility/paper_experiments/pendulum/reproduce.py \
  full-rerun \
  --output "$PEND_FULL_OUT" \
  --max-parallel 1 \
  --confirm-expensive-rerun RUN_THE_FULL_15_SEED_TUNED_STUDY
```

The wrapper first authenticates all Pendulum inputs and principal package
versions. It then trains development seeds 31-35. Only a passing fresh gate
allows reporting seeds 101-110 to run. The phases are sequential;
`--max-parallel` controls concurrent seeds within each phase. One worker is the
safest starting point for memory use and auditability.

An interrupted output cannot be resumed by this wrapper. Keep the partial tree
for diagnosis and start any new attempt at another fresh output path. Terminal
output can remain quiet for long periods because child output is captured in
the final report.

Successful output has this structure:

```text
$PEND_FULL_OUT/
  development_full_nn_v1/
    raw/seed_0031/ ... raw/seed_0035/
    aggregated/
    manifest.json
    reporting_freeze_manifest.json
  reporting_full_nn_v1/
    raw/seed_0101/ ... raw/seed_0110/
    aggregated/
    figures/
    manifest.json
  matplotlib-cache/
  pendulum_reproduction_report.json
```

## Optional fresh seed-31 supplements

These commands execute the authenticated frozen runner directly. Unlike the
main wrapper, the direct runner has no literal expensive-run confirmation,
principal-version guard, or canonical-path guard. These are full-budget,
expensive training jobs; there is no low-budget public Pendulum training
configuration.

Run strict verification first and confirm that its JSON report contains
`runtime.principal_versions_match: true`; verification reports package drift
but does not fail solely because of that drift. Set `PYTHONHASHSEED=0` and use
fresh output paths outside `results/`. Never point the direct runner or plotter
at an extracted companion `results/...` tree, and never add `--skip-existing`
there: those programs write aggregates, manifests, tables, and figures and do
not have the wrapper's protected historical-path check.

```bash
PEND_SNAPSHOT=reproducibility/paper_experiments/pendulum/frozen_snapshot
PEND_SUPP_PARENT="$(.venv/bin/python -c \
  'import tempfile; print(tempfile.mkdtemp(prefix="pendulum-supplements-parent."))')"

EXACT_OUT="$PEND_SUPP_PARENT/exact-inner-seed31"
PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  "$PEND_SNAPSHOT/scripts/run_rvchi2_dqn.py" \
  --config "$PEND_SNAPSHOT/configs/rvchi2_dqn/v1/appendix_pendulum_exact_inner_seed31_v1.json" \
  --output-root "$EXACT_OUT" \
  --max-parallel 1

FULL_NN_OUT="$PEND_SUPP_PARENT/full-nn-seed31"
PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  "$PEND_SNAPSHOT/scripts/run_rvchi2_dqn.py" \
  --config "$PEND_SNAPSHOT/configs/rvchi2_dqn/v1/ablation_pendulum_full_nn_lr1e4_seed31.json" \
  --output-root "$FULL_NN_OUT" \
  --max-parallel 1
```

A single development seed cannot issue a reporting freeze, so its
`development_gate.json` correctly reports `evaluated: false`.

The frozen plotter supports the nominal/affine/full-NN supplement:

```bash
PEND_MPL="$(.venv/bin/python -c \
  'import tempfile; print(tempfile.mkdtemp(prefix="pendulum-mpl."))')"
MPLCONFIGDIR="$PEND_MPL" PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  "$PEND_SNAPSHOT/scripts/plot_rvchi2_dqn.py" \
  "$FULL_NN_OUT" --representative-seed 31
```

Do not run that plotter on the nominal/exact supplement: the released plotting
layout expects nominal, affine, and full-NN checkpoints. The exact run still
writes its raw bundle and aggregate CSV/JSON evidence.

## Output records

Each completed seed directory contains:

```text
config.json
learning_metrics.csv
evaluation_episodes.csv
evaluation_summary.csv
checkpoint_evaluation_episodes.csv
checkpoint_evaluation_summary.csv
summary.json
checkpoints.pt
backup_calibration.npz
metadata.json
manifest.json
```

The manifest authenticates the other ten files, the resolved configuration,
scientific-source hashes, schema, and completion status. Checkpoints retain the
pretrained base nominal Q state and every branch's final Q and Q-optimizer
state. Learned variational branches additionally retain the online auxiliary,
EMA-deployed auxiliary, auxiliary optimizer, and the last frozen Q-target state
used for auxiliary calibration.

An aggregated study directory contains:

```text
aggregated/final_by_seed.csv
aggregated/robustness_profile.csv
aggregated/paired_by_seed.csv
aggregated/paired_profile.csv
aggregated/seed_inference.json
aggregated/development_gate.json
manifest.json
```

The paper plotter additionally creates `aggregated/main_table.csv`, six
PDF/PNG figure pairs, and `figures/figure_manifest.json`.

## Reproducibility limitations

Historical seed bundles preserve resolved configurations, final model and
optimizer states, evaluation episodes and summaries, learning traces,
calibration arrays, source hashes, package versions, and manifests. They do
not preserve complete replay arrays, environment/cursor state, every training
transition, or Python/NumPy/Torch RNG states.

Consequently:

- exact authentication and plot reconstruction from the preserved artifacts
  are supported;
- fresh training from the specified seeds is supported;
- exact mid-run continuation is not supported; and
- byte-identical checkpoints or trajectories across machines are not claimed.

The dependency versions are pinned, but wheel hashes, the complete operating
system image, CPU/native-library build details, Torch inter-operation threads,
and all external thread-control variables are not fully frozen. These details
can affect a neural trajectory even when deterministic Torch algorithms are
enabled.

## Implementation map

| File | Responsibility |
|---|---|
| [`src/rvchi2_dqn/config.py`](src/rvchi2_dqn/config.py) | Defaults, task profiles, reserved seeds, validation |
| [`src/rvchi2_dqn/envs.py`](src/rvchi2_dqn/envs.py) | NumPy dynamics, fault channel, deterministic mode enumeration |
| [`src/rvchi2_dqn/operators.py`](src/rvchi2_dqn/operators.py) | Exact binary and sampled variational robust operators |
| [`src/rvchi2_dqn/networks.py`](src/rvchi2_dqn/networks.py) | Q network, affine/full-NN heads, projections, EMA |
| [`src/rvchi2_dqn/replay.py`](src/rvchi2_dqn/replay.py) | Uniform replay storage, cloning, sampling |
| [`src/rvchi2_dqn/diagnostics.py`](src/rvchi2_dqn/diagnostics.py) | Support, calibration, and auxiliary diagnostics |
| [`src/rvchi2_dqn/trainer.py`](src/rvchi2_dqn/trainer.py) | Per-seed training, evaluation, seeding, artifact publication |
| [`scripts/run_rvchi2_dqn.py`](scripts/run_rvchi2_dqn.py) | Multi-seed orchestration, aggregation, development freeze |
| [`scripts/plot_rvchi2_dqn.py`](scripts/plot_rvchi2_dqn.py) | Paper tables and figures |
| [`reproducibility/paper_experiments/pendulum/reproduce.py`](reproducibility/paper_experiments/pendulum/reproduce.py) | Authentication, protected reconstruction, guarded full rerun |

The conventional live files are provided for inspection. Reproduction and the
guarded main rerun execute the authenticated copies under
`reproducibility/paper_experiments/pendulum/frozen_snapshot/`, which remain
authoritative by default. Supplying `--require-live-correspondence` to
verification additionally requires the convenient live and frozen copies to
be byte-identical.

## Citation, license, and contact

Citation metadata is in [`CITATION.cff`](CITATION.cff). Software is licensed
under Apache-2.0; narrative documentation, figures, and released scientific
records are licensed under CC BY 4.0. The exact path-based boundary is in
[`LICENSE_SCOPE.md`](LICENSE_SCOPE.md).

Contact: `smandal4@illinois.edu` (primary) or
`smandal32153@gmail.com` (fallback).

For papers and review responses, cite the versioned source tag together with
its matching companion assets. Publish later retraining evidence under a new
commit, tag, and release rather than replacing the historical evidence.
