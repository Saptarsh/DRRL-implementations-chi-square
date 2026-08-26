# Pendulum paper reproducibility workflow

This directory provides a tag-contained, non-mutating workflow for the
paper-facing Pendulum experiment. The canonical evidence is the three-method
reporting bundle at:

```text
results/rvchi2_dqn/v2/pendulum/reporting_full_nn_v1
```

For the task definition, compared methods, network initialization, complete
resolved hyperparameters, seed derivations, and one-seed supplemental commands,
see the top-level
[`README_RVCHI2_PENDULUM.md`](../../../README_RVCHI2_PENDULUM.md). This file
remains the command-focused guide to authentication, paper-figure
reconstruction, and the guarded main rerun.

That bundle contains nominal DDQN, RVChi2-A, and RVChi2-N. Its nominal and
affine rows are exactly the same numerical results as the earlier affine-only
`reporting_frozen_v2` study. The later bundle is canonical because its
145-entry paper inventory presently authenticates every raw leaf, aggregate,
figure, configuration, authorization record, scientific source, and plotting
source.

The executable capsule is `frozen_snapshot/`: exact authenticated copies of
the historical runner and plotter, the eight scientific modules plus package
initializer, the development/reporting/exact-inner/full-NN configurations,
and `requirements-freeze.txt`. Reproduction and optional retraining execute
these snapshot bytes, not mutable live files elsewhere in the checkout.

## 1. Read-only verification

From the repository root, using the recorded environment:

```bash
.venv/bin/python reproducibility/paper_experiments/pendulum/reproduce.py verify
```

The default trusts the authenticated snapshot even if live code later evolves.
To additionally require current live files to match it exactly:

```bash
.venv/bin/python reproducibility/paper_experiments/pendulum/reproduce.py \
  verify --require-live-correspondence
```

This verifies:

- the exact SHA-256 of the 145-entry canonical paper inventory;
- every path, byte size, and SHA-256 declared by that inventory;
- the exact canonical reporting-tree inventory (no unlisted files);
- every per-seed raw manifest and its ten artifacts for seeds 101--110;
- the five-seed development authorization gate and reporting freeze;
- the exact 16-file executable snapshot, including runner, plotter,
  development/reporting/supplement configs, scientific modules, package
  initializer, and dependency lock;
- optional live-source correspondence without making live bytes authoritative;
- the canonical study/raw source maps against the snapshot source map; and
- the one-seed exact-inner and full-NN appendix raw bundles.

Verification is read-only. It reports platform drift separately from package
version drift. Aggregation from saved raw data is portable across platforms,
but fresh neural training on another platform is not promised to be bit-exact.

## 2. Regenerate paper aggregates and figures

Choose a **new path that does not exist**:

```bash
.venv/bin/python reproducibility/paper_experiments/pendulum/reproduce.py \
  reproduce \
  --output /tmp/rvchi2-pendulum-paper-reproduction
```

The command:

1. authenticates all frozen inputs;
2. computes the canonical tree digest;
3. copies only `raw/seed_0101` through `raw/seed_0110` into a fresh staging
   tree;
4. derives a portable reporting-freeze copy whose only semantic change is the
   absolute development-gate path;
5. invokes the frozen snapshot runner with `--skip-existing` to reconstruct all
   aggregates from the copied raw seeds;
6. invokes the frozen snapshot plotter in the copied tree;
7. requires all seven numerical aggregate files to be byte-identical to the
   canonical versions;
8. requires every regenerated PNG to be pixel-identical to the canonical
   image (PDF container bytes are recorded but may differ because of embedded
   creation metadata);
9. hashes every regenerated figure and writes
   `pendulum_reproduction_report.json`; and
10. reauthenticates the canonical inputs and requires its before/after tree
   digest to be unchanged before atomically publishing the output.

The temporary Matplotlib font cache is removed before publication; it is not a
scientific artifact.

The canonical result directory is never passed to code that writes. Existing
output paths, the canonical path, descendants of it, and ancestors containing
it are rejected.

The primary regenerated plot is:

```text
/tmp/rvchi2-pendulum-paper-reproduction/figures/robustness_profiles.pdf
```

The hard checks are byte identity of the seven aggregate files and decoded
pixel identity of every PNG. PDF byte hashes may change because renderers can
embed timestamps or other metadata.

## 3. Optional full tuned rerun — expensive

This is not needed to regenerate the paper. It retrains five development seeds
and, only if they issue a fresh reporting freeze, all ten reporting seeds. The
literal confirmation is intentional:

```bash
.venv/bin/python reproducibility/paper_experiments/pendulum/reproduce.py \
  full-rerun \
  --output /path/to/a/new/pendulum-full-rerun \
  --max-parallel 1 \
  --confirm-expensive-rerun RUN_THE_FULL_15_SEED_TUNED_STUDY
```

The command executes the frozen runner and frozen development/reporting
specifications, retains
the reserved seed panels 31--35 and 101--110, and uses the newly generated
development freeze to authorize the new reporting stage. It never resumes or
mixes partial per-seed output. Use a machine with ample wall time and disk
space; each seed includes 150,000 nominal pretraining transitions, three
90,000-step branches, 30,000 Q updates per branch, and 150,000 auxiliary
updates per variational branch.

## Reproducibility boundary

The historical bundles preserve resolved configs, final neural and optimizer
states, per-episode evaluations, learning traces, deterministic-grid
exact/learned backup calibration, source hashes, and package versions. They do
**not** preserve full
replay arrays, environment states, or Python/NumPy/Torch RNG states. Therefore:

- paper statistics and plots can be independently regenerated from saved raw
  artifacts;
- final checkpoints can be independently evaluated; but
- an exact mid-run continuation or byte-identical replay reconstruction is
  impossible. Full training reproduction starts from the recorded seeds.

The original run used Python 3.10.13, Torch 2.11.0, NumPy 2.2.6, Gymnasium
1.2.3, Pandas 2.3.3, Matplotlib 3.10.9, deterministic CPU Torch, and one Torch
thread on macOS arm64. `requirements-freeze.txt` is now explicitly bound by
this workflow, although it was not part of the historical per-run manifests.

## Tag and companion-data release

Audit whether every capsule byte is tracked and clean:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  reproducibility/paper_experiments/pendulum/reproduce.py tag-audit
```

`tag-audit` returns exit status 2 while the capsule is untracked or dirty and
reports the exact files. It also deliberately reports
`companion_data_required: true`: a source tag contains the workflow and frozen
code/configuration, but the canonical neural artifacts remain external data.
A complete reproducibility release must therefore publish a durable companion
archive (or explicitly track those data) containing:

- `reporting_full_nn_v1`, authenticated by its 145-entry inventory SHA-256
  `cbd473f2a0fcbfb165cd3e6ecbd1852638c03adb1f1687d9243ef2f99a8b133c`;
- the development gate and reporting-freeze authorization files; and
- the exact-inner and full-NN seed-31 appendix bundles named in
  `frozen_inputs.json`.

The minimum frozen companion scope is 158 files / 36,446,249 bytes. Its
content-addressed inventory digest is
`d1c278aaeaab7224fc3a10f64af0a9c4ab33784adc2dd2c62be5373c616ef01d`
under the canonicalization documented in `frozen_inputs.json`.

Do not describe a repository tag alone as a complete data release. The tag
and extracted companion archive must both pass `verify`.
