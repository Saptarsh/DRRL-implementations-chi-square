# Variational Chi-Square DRRL: Multi-Decision Tabular Experiment

This is the paper-facing tabular flow for the two-stage algorithm in
`variational_algorithm.txt`.  Its primary task is a continuing 4x6 MiniCliff
gridworld, not the earlier one-decision risky corridor.

## Why this task

The grid is

```text
F F F F F F
F F F F F F
F F F F F F
S C C C C G
```

There are 24 states, four actions at every state, and 19 ordinary states at
which the policy makes repeated navigation decisions. `S` is the start, `C`
denotes four cliff marker states, and `G` is a goal marker state. Movement uses
actions up/right/down/left. With slip probability `p`, the requested action is
executed with probability `1-p`, and each of the other three actions is
executed with probability `p/3`.

Cliffs and the goal are explicit one-step states. Ordinary states have reward
`-0.01`, cliff states have reward `-1`, and the goal has reward `+1`; every
marker state then resets deterministically to `S`. Thus rewards are genuinely
fixed functions `r(s,a)`, as in the theory, while the controlled experiment
changes only the transition kernel.

The defaults are

```text
gamma = 0.9, nominal slip p0 = 0.10, chi-square radius delta = 0.10.
```

The exact nominal and robust oracle policies agree at the start but differ at
three reachable, noninitial decision states: `(2,0)`, `(2,1)`, and `(2,2)`.
The nominal policy turns right above the cliff sooner; the robust policy keeps
moving upward before traversing. This makes the experiment a test of learning
a state-dependent policy, rather than selecting one route once at time zero.

## Theory-aligned data stream

The learner observes one continuing nominal trajectory. Its fixed behavior
policy is

```text
0.8 * Uniform(actions) + 0.2 * GoalDirected(actions).
```

Consequently, the preferred action has probability `0.4` and every other
action has probability `0.2`. The trajectory cursor is initialized once and is
never restarted between Stage 1, the Q stage, or outer blocks. The nominal
Q-learning baseline consumes every one of those same transitions, so both
methods have an identical trajectory budget.

For the default behavior chain,

```text
min_(s,a) d(s,a) = 9.859869e-4.
```

The default `beta0=650` therefore gives
`p_Q = 2 min_(s,a)d(s,a) beta0 = 1.28178 > 1`, while
`h_q=1300=2 beta0`. Every run records the exact stationary residual, expected
and observed rare-pair coverage, `p_Q`, and whether the clean-rate condition is
satisfied.

## Exact references and metrics

The nominal optimal Q-function, unsmoothed chi-square robust optimal
Q-function, and finite-floor fixed point are computed by contraction-based
dynamic programming. The chi-square inner problem uses the analytic finite
active-set solution; no eta grid, smoothing, or generative sampler is used.

Each outer block records:

- `||Q_hat_t - Q_chi*||_inf` versus all transitions read;
- robust and floor Bellman residuals;
- exact return and regret of the learned robust and nominal policies at the
  in-ball evaluation slip `p=0.1875`;
- exact worst-case return of the learned robust policy;
- full-grid, discounted-occupancy-weighted, and three-separating-state policy
  agreement with the exact oracles;
- discounted cliff occupancy;
- Stage-1 target error, gradients, `x/u` amplification, projection hits, and
  state-action coverage.

The zero Q-table greedily selects `up` everywhere and has poor start return: a
good learned return requires the complete sequence of state-dependent moves.

## Controlled transition perturbations

The evaluation sweep increases the slip probability while preserving all
nominal supports and keeping rewards fixed. It computes the actual rowwise
quantity

```text
max_(s,a) D_chi(P_p(.|s,a) || P_p0(.|s,a)),
```

after probabilities leading to the same grid cell have been aggregated. The
upper slip boundary of the training ambiguity set is

```text
p_delta = p0 + sqrt(delta p0 (1-p0)) = 0.194868.
```

The default grid contains both in-set evaluations and a few explicitly marked
out-of-set stress points. At `p=0.1875`, exact evaluation gives approximately
`0.1618` for the robust oracle and `0.0593` for the nominal oracle; the exact
test-kernel optimum is `0.1630`.

## Floor sensitivity

The paired-seed floor sweep uses

```text
ell in {0.003, 0.01, 0.03, 0.10, 0.30, 1.0}.
```

It reports total robust-Q error, error to the exact unconstrained finite-floor
fixed point, exact floor bias, and the analytic ceiling

```text
gamma sqrt(1+delta) ell / (2(1-gamma)).
```

These norms obey a triangle inequality but are not an additive decomposition.
Gradient size, `x/u`, projection rates, and across-seed variation expose the
predicted small-floor instability. A single Stage-1 step size is held fixed
across the sweep. The shared tabular parameter radii are
`R_eta=12` and `R_scale=10`; they contain the exact finite-floor optimizer even
at `ell=1`. Each run checks this before learning and fails with a diagnostic if
custom radii would confound floor bias with parameter-class truncation.

## Files

```text
src/variational_tabular_envs.py
    MiniCliff construction, behavior chain, persistent trajectory, and kernel
    divergence checks.

src/train_variational_chi2_gridworld.py
    Two-stage learner, nominal baseline, exact solvers, and one-seed artifacts.

scripts/run_variational_chi2_gridworld_paper.py
    Profiles, multi-seed aggregation, validated caching, and paper figures.

tests/test_variational_tabular_envs.py
tests/test_variational_chi2_gridworld.py
    Environment, exact-reference, policy-separation, and trajectory tests.
```

## Run it

Verify everything first:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

Run an artifact/plot smoke test:

```bash
.venv/bin/python scripts/run_variational_chi2_gridworld_paper.py \
  --profile smoke \
  --output-root paper_variational_chi2_gridworld/smoke \
  --max-parallel 1
```

The smoke profile is only a plumbing check (two tiny 4,000-transition runs);
its learned policies are not intended for interpretation.

Run the multi-seed preview:

```bash
.venv/bin/python scripts/run_variational_chi2_gridworld_paper.py \
  --profile quick \
  --output-root paper_variational_chi2_gridworld/quick \
  --max-parallel 3
```

The quick profile runs three paired seeds at
`ell in {0.01, 0.1, 0.5}`. Each run uses 20 blocks and one million cumulative
transitions. The full profile uses 60 blocks and six million transitions per
run.

For the main convergence and perturbation experiment, isolate the selected
floor from the sensitivity sweep:

```bash
.venv/bin/python scripts/run_variational_chi2_gridworld_paper.py \
  --profile full \
  --ells 0.1 \
  --focus-ell 0.1 \
  --n-seeds 25 \
  --output-root paper_variational_chi2_gridworld/main \
  --max-parallel 4
```

Then run the paired floor study:

```bash
.venv/bin/python scripts/run_variational_chi2_gridworld_paper.py \
  --profile full \
  --ells 0.003,0.01,0.03,0.1,0.3,1.0 \
  --n-seeds 20 \
  --output-root paper_variational_chi2_gridworld/ell_sensitivity \
  --max-parallel 4
```

Add `--skip-existing` to resume. A run is reused only after its environment,
algorithm, CSV schemas, array shapes, finite values, and all implementation
hashes have been validated. The three numerical artifacts also authenticate
each other through SHA-256 hashes in `metadata.json`, and metadata is promoted
last so an interrupted overwrite cannot be mistaken for a complete run.

## Output layout

```text
<output-root>/
  raw/ell_<value>/seed_<seed>/
    metrics.csv
    perturbation_metrics.csv
    metadata.json
    arrays.npz
  aggregated/
    learning_curves.csv
    perturbation_summary.csv
    ell_summary.csv
  figures/
    convergence_policy_agreement.{png,pdf}
    perturbation_performance.{png,pdf}
    ell_bias_stability.{png,pdf}
    policy_maps.{png,pdf}
  manifest.json
```

The plotted profiles use the practical shared Stage-1 step size `0.024`. It is
an empirical choice and is not presented as the theorem's conservative
finite-sample prescription. `--stage1-step-mode theory` remains available for
running the literal theorem-derived value. No per-floor tuning, warm starts,
gradient clipping, smoothing, or trajectory restarts are used.
