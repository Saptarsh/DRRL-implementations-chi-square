# MiniCliff paper reproducibility capsule

This directory freezes and verifies the paper-facing `4x6` continuing
MiniCliff experiment. It deliberately excludes the older one-decision risky
corridor. The historical result directories are always read-only inputs:

- `paper_variational_chi2_gridworld/main` — `ell=0.1`, seeds `1..25`;
- `paper_variational_chi2_gridworld/ell_sensitivity` — six floors and seeds
  `1..20`.

The capsule contains the exact scientific source, runner, plotter, tests,
theory note, and dependency files used by those artifacts. This is important
because the corresponding live files and result directory were not tracked in
the repository when the studies were produced.

## What verification proves

Run from the repository root with the frozen environment:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  reproducibility/paper_experiments/minicliff/minicliff_reproduce.py verify
```

The command fails closed unless it can:

1. authenticate the exact 609-file, 12,005,948-byte canonical evidence tree;
2. authenticate the frozen source snapshot and its correspondence to live
   files;
3. reconstruct all 145 run specifications and validate each raw bundle;
4. recompute all learning, perturbation, and floor aggregates plus both modal
   policy archives exactly; and
5. prove that the duplicated `ell=0.1`, seeds `1..20` raw bundles are
   byte-identical between the main and floor studies.

The historical intervals are mean `+/- 1.96 SEM`; they are not Student-t
intervals. Policy evaluation is exact finite-MDP evaluation, not Monte Carlo.

`--allow-live-source-drift` permits an archival verification from the frozen
snapshot after live files change. It never relaxes snapshot or data hashes.

## Regenerate aggregates and plots without training

Choose a destination that does not exist and is outside the canonical result
tree and this capsule:

```bash
REPRO_ROOT="$(.venv/bin/python -c \
  'import tempfile; print(tempfile.mkdtemp(prefix="minicliff-parent."))')/rebuilt"

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  reproducibility/paper_experiments/minicliff/minicliff_reproduce.py \
  regenerate --output-root "$REPRO_ROOT"

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  reproducibility/paper_experiments/minicliff/minicliff_reproduce.py \
  verify-output --output-root "$REPRO_ROOT"
```

The command above is the strict frozen release-audit-runtime path. On a
compatible non-frozen runtime, regeneration can be requested explicitly with:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  reproducibility/paper_experiments/minicliff/minicliff_reproduce.py \
  regenerate --output-root "$REPRO_ROOT" --allow-runtime-drift
```

This option is limited to regeneration from authenticated raw artifacts. It
records every runtime mismatch and still requires exact source, data, raw-copy,
and numerical aggregate checks. The supplied PNG must match a fresh render
from those verified aggregates. Equality to the canonical PNG is reported
rather than required only when a real runtime mismatch exists. The option is
not available to `train`.

`regenerate` authenticates canonical inputs, copies only raw bundles into a
sibling staging directory, invokes the frozen runner with validated reuse,
rebuilds aggregate CSVs/NPZs and plots, compares scientific aggregates with
canonical values, evaluates decoded composite-PNG pixel identity, writes an
exhaustive manifest, fsyncs, and atomically publishes. `verify-output` also
rerenders portable outputs from their verified aggregates and requires the
supplied PNG to match. The workflow authenticates the canonical trees again
before publication. It never trains and grants no new selection or reporting
authority.

PDF bytes are not expected to match: historical Matplotlib PDFs contain
wall-clock creation metadata. Historical PDF bytes remain authenticated by
the canonical tree; regenerated PDF bytes are bound in the new manifest.

## Optional full tuned rerun

This is intentionally difficult to invoke. The main study reads 150 million
transitions; the complete floor sweep reads 720 million. Use a new destination:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  reproducibility/paper_experiments/minicliff/minicliff_reproduce.py train \
  --study main \
  --output-root /path/to/new/minicliff-main-rerun \
  --acknowledge-expensive-training I_UNDERSTAND_THIS_RUN_IS_EXPENSIVE
```

Use `--study ell_sensitivity` for the sweep or `--study all` for both. Training
and strict regeneration require exact Python, platform, NumPy, Matplotlib,
PCG64, Accelerate BLAS identity, and `pip freeze` hash. This is the environment
in which the archived results were audited and regenerated; the original
tabular run did not itself record these dependency versions. The compatible
runtime option described above never relaxes the training gate.
Because the historical run did not save RNG state or raw trajectories, a fresh run is
a reproducibility attempt rather than a guaranteed bitwise reconstruction.

## Tests

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  discover -s reproducibility/paper_experiments/minicliff/tests \
  -p 'test_*.py' -v
```

The suite covers source closure, complete raw/aggregate verification,
canonical immutability, coordinated tampering, path/symlink isolation,
publication races, expensive-run authorization, regeneration, and immutable
output reauthentication.

## Tag and companion-data release

The source tag must include every file in this capsule. Check that with:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  reproducibility/paper_experiments/minicliff/minicliff_reproduce.py tag-audit
```

The canonical result directory is ignored by Git and therefore is not carried
by a source tag. Release it as a companion archive (or force-track it) with
these extraction invariants:

```text
main:            116 files, 2,862,826 bytes,
                 inventory 5fd528c52932d5eb0d42429bda757162f93091953d64b773bdfd985278affe87
ell_sensitivity: 493 files, 9,143,122 bytes,
                 inventory 6a30e08a0606602c70c154b01c9077efba30305d5f67de77958724621314412b
combined:        609 files, 12,005,948 bytes,
                 inventory 8d02a0b0ea68f1fe1cae82c35ff1418410f982dc7a0aa36b90d8edecba6df864
```

Do not describe a tag as a complete reproducibility release until both the
capsule is tracked and a durable companion archive/DOI with the combined
inventory digest is published. `tag-audit` therefore reports the current
tracked-state honestly rather than silently treating ignored local data as a
tag artifact.
