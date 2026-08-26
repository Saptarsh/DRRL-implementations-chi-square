# Paper experiment reproducibility

This directory is the tag-facing entry point for the two experiments currently
used in the paper.  Run every command from the repository root with the
frozen release-audit environment (`.venv/bin/python`, Python 3.10.13). For
MiniCliff, this environment was authenticated during the later release audit;
the original run did not record its exact dependency versions.

The workflows deliberately separate three questions:

1. **Verify** authenticates the frozen raw artifacts, source/config hashes,
   seed panels, and authorization records without writing to canonical data.
2. **Regenerate** copies authenticated raw artifacts to a fresh directory,
   reconstructs the aggregate tables, and renders the figures.  No training is
   performed and the historical result directories are never passed to code
   that writes.
3. **Train/full-rerun** repeats the frozen tuned training protocol from the
   recorded seeds in a new directory.  These commands are expensive and
   require an explicit acknowledgement string.

## Fast verification and plot regeneration

MiniCliff:

```bash
.venv/bin/python reproducibility/paper_experiments/minicliff/minicliff_reproduce.py verify

.venv/bin/python reproducibility/paper_experiments/minicliff/minicliff_reproduce.py \
  regenerate --output-root /tmp/minicliff-paper-reproduction
```

MiniCliff regeneration is strict by default. A compatible non-frozen runtime
may add `--allow-runtime-drift`; source/data authentication and exact numerical
aggregate equality remain mandatory. The supplied PNG must also match a fresh
render from those verified aggregates, while equality to the canonical PNG is
reported instead of required when drift is present. This flag is unavailable
to training.

Pendulum:

```bash
.venv/bin/python reproducibility/paper_experiments/pendulum/reproduce.py verify

.venv/bin/python reproducibility/paper_experiments/pendulum/reproduce.py \
  reproduce --output /tmp/pendulum-paper-reproduction
```

Expected primary outputs:

```text
/tmp/minicliff-paper-reproduction/paper_derivative/tabular_tac_composite.pdf
/tmp/pendulum-paper-reproduction/figures/robustness_profiles.pdf
```

The reconstructed numerical aggregates must match the canonical aggregates
exactly. Under the frozen release-audit runtime, PNG figures must have
identical decoded pixels. Compatible-runtime MiniCliff regeneration requires
the supplied PNG to equal a fresh render from verified aggregates and reports
its canonical pixel comparison separately. PDF file hashes may differ because
Matplotlib can embed creation timestamps or other container metadata.

## Full tuned retraining

The complete commands and safeguards are documented in the task-specific
READMEs:

- [`minicliff/README.md`](minicliff/README.md)
- [`pendulum/README.md`](pendulum/README.md)

The detailed scientific and implementation guide for the neural study is
[`README_RVCHI2_PENDULUM.md`](../../README_RVCHI2_PENDULUM.md).

Do not point a training or regeneration command at the canonical paper result
directories.  Both wrappers reject overlapping or pre-existing output paths.

Approximate scale of the frozen jobs:

- MiniCliff main: 25 runs × 6,000,000 transitions = 150 million transitions;
  floor sweep: 120 runs × 6,000,000 = 720 million transitions.
- Pendulum: five development and ten reporting seeds; each seed uses 150,000
  nominal-prefix transitions followed by three 90,000-step method branches,
  plus Q/auxiliary update and evaluation budgets.

## Reproducibility boundary

The preserved raw artifacts are sufficient to authenticate and reconstruct
the paper statistics and figures.  MiniCliff fresh training is deterministic
from its recorded seeds under the frozen NumPy/PCG64 environment, but its
historical RNG state and sampled trajectory were not archived.  Pendulum did
not archive replay arrays, environment states, or RNG states; therefore its
full rerun begins from the recorded seeds and is not promised to reproduce a
bit-identical training trajectory on different hardware or library builds.
