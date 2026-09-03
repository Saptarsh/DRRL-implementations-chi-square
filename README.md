# Chi-square robust Q-learning reproducibility package

This repository accompanies
*Finite-Time Convergence of Single-Trajectory Chi-Square Robust Q-Learning
With Linear Function Approximation* by Saptarshi Mandal, Yashaswini Murthy,
and R. Srikant.

If you use this code, please cite our arXiv paper [Paper Link](https://arxiv.org/abs/2510.01721).

It contains the executable source and frozen configurations for the two
paper-facing experiments:

- a tabular `4 x 6` MiniCliff study; and
- a [neural Pendulum study](README_RVCHI2_PENDULUM.md).

The complete declared companion-data scope required by these workflows is
distributed as two versioned assets on the repository's
[Releases page](https://github.com/Saptarsh/DRRL-implementations-chi-square/releases):

- `minicliff-companion-data.tar.gz`
- `pendulum-companion-data.tar.gz`

Small copies of the two primary paper figures are available in
[`paper_outputs/`](paper_outputs/). They are browsing conveniences, not a
substitute for the companion archives.

## How the release works

In plain language, the Git tag preserves the code and the exact experimental
instructions. The two release assets preserve the larger raw records,
checkpoints, tables, and figures. Extracting both assets at the repository root
restores the paths expected by the verification scripts. The scripts first
authenticate those files and then rebuild the tables and plots in a new output
directory; they never write into the historical evidence trees.

Plot regeneration uses the archived per-seed records and performs no fresh
training or trajectory rollout. Full retraining is a separate, optional, and
expensive workflow guarded by literal confirmation strings.

## 1. Clone and install the release-audit environment

Python 3.10.13 was used for the release audit. From the repository root:

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements-freeze.txt
```

The dependency lock records NumPy 2.2.6, Torch 2.11.0, Gymnasium 1.2.3,
Pandas 2.3.3, and Matplotlib 3.10.9. The strict audit used Python 3.10.13 on
macOS arm64 with Apple's Accelerate BLAS. For MiniCliff, this is an
authenticated release-audit environment, not proof of the dependency versions
that produced the original historical run. Package installation can require
platform-specific build prerequisites.

## 2. Download and extract companion data

Download both archives, their two `.json` sidecars, and `SHA256SUMS` from the
same versioned GitHub Release. Put them in one download directory, verify the
compressed files, and extract the archives at this repository root:

```bash
cd /path/to/downloads
shasum -a 256 -c SHA256SUMS
# Linux alternative: sha256sum -c SHA256SUMS

cd /path/to/DRRL-implementations-chi-square
tar -xzf /path/to/downloads/minicliff-companion-data.tar.gz
tar -xzf /path/to/downloads/pendulum-companion-data.tar.gz
```

The archive SHA-256 values authenticate the compressed downloads. Separate
extracted-tree digests authenticate the file content after extraction; see
[`docs/COMPANION_DATA.md`](docs/COMPANION_DATA.md).

## 3. Verify the release inputs

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  discover -s reproducibility/paper_experiments/minicliff/tests \
  -p 'test_*.py' -v

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  discover -s tests -p 'test_pendulum_paper_reproducibility.py' -v

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  discover -s tests -p 'test_release_export.py' -v

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  reproducibility/paper_experiments/minicliff/minicliff_reproduce.py verify

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  reproducibility/paper_experiments/pendulum/reproduce.py verify \
  --require-live-correspondence
```

The verification commands are read-only. Successful reports authenticate the
frozen source capsules, live-source correspondence, complete selected data
inventories, per-run manifests, and historical authorization chain.

## 4. Rebuild tables and figures without training

Every output path must be fresh and nonexistent.

MiniCliff, strict release-audit-runtime mode:

```bash
MINI_PARENT="$(.venv/bin/python -c \
  'import tempfile; print(tempfile.mkdtemp(prefix="minicliff-parent."))')"
MINI_OUT="$MINI_PARENT/rebuilt"

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  reproducibility/paper_experiments/minicliff/minicliff_reproduce.py \
  regenerate --output-root "$MINI_OUT"

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  reproducibility/paper_experiments/minicliff/minicliff_reproduce.py \
  verify-output --output-root "$MINI_OUT"
```

The strict mode requires the complete frozen release-audit runtime identity. On a
compatible non-frozen runtime, add `--allow-runtime-drift` to the `regenerate`
command. That opt-in still requires exact source, raw-data, and numerical
aggregate checks; it records the runtime differences, requires the supplied
figure to match a fresh render from those verified aggregates, and reports
canonical figure pixel equality instead of requiring it. The option is
intentionally unavailable to training. Pendulum retains its strict
pixel-identity requirement.

Pendulum:

```bash
PEND_PARENT="$(.venv/bin/python -c \
  'import tempfile; print(tempfile.mkdtemp(prefix="pendulum-parent."))')"
PEND_OUT="$PEND_PARENT/rebuilt"

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  reproducibility/paper_experiments/pendulum/reproduce.py \
  reproduce --output "$PEND_OUT"
```

Primary rebuilt PDFs:

```text
$MINI_OUT/paper_derivative/tabular_tac_composite.pdf
$PEND_OUT/figures/robustness_profiles.pdf
```

Under the frozen release-audit runtime, numerical aggregates must match the authenticated
historical aggregates and PNG pixels must match the canonical images. PDF
byte identity is not claimed because renderers may embed creation or other
container metadata.

## Reproducibility boundary

The release supports exact authentication and reconstruction of the paper's
statistics and plots from preserved scientific artifacts. This is not the same
claim as independently repeating every training trajectory. MiniCliff did not
archive its historical RNG state or sampled trajectory. Pendulum did not
archive replay arrays, environment states, or Python, NumPy, and Torch RNG
states. A fresh full rerun begins from the recorded seeds and should be treated
as an independent retraining comparison, not as a replacement for the paper's
historical evidence.

Full-rerun commands, safeguards, and compute scale are documented in the
task-specific READMEs:

- [`reproducibility/paper_experiments/minicliff/README.md`](reproducibility/paper_experiments/minicliff/README.md)
- [`reproducibility/paper_experiments/pendulum/README.md`](reproducibility/paper_experiments/pendulum/README.md)

The scientific task, method definitions, neural initialization, complete
resolved Pendulum hyperparameters, seed derivations, and supplemental-run
commands are collected in
[`README_RVCHI2_PENDULUM.md`](README_RVCHI2_PENDULUM.md).

## Source layout

The top-level `src/`, `scripts/`, `configs/`, and selected test files are the
conventional live source tree. Each experiment capsule also carries an
authenticated `frozen_snapshot/` duplicate. Reproduction executes the frozen
bytes; strict verification confirms that the convenient live copies match.
This intentional duplication makes the historical execution surface both
auditable and easy to inspect.

The MiniCliff capsule mentions its historical smoke/quick profiles and the
earlier risky-corridor task only as source context. This release contains no
quick-profile or risky-corridor result data. The authenticated companion scope
is limited to the paper-facing MiniCliff `main` and `ell_sensitivity` studies
and the exact Pendulum closure declared in `frozen_inputs.json`.

## Licensing, citation, and contact

Software is licensed under Apache-2.0. Data, figures, and narrative
documentation are licensed under CC BY 4.0. The exact boundary is described in
[`LICENSE_SCOPE.md`](LICENSE_SCOPE.md). Citation metadata is in
[`CITATION.cff`](CITATION.cff).

Contact: `smandal4@illinois.edu` (primary) or `smandal32153@gmail.com`
(fallback).

For a paper or review, cite a versioned release tag rather than the moving
default branch. Later retraining evidence should be published as a new commit,
tag, and release so the original cited capsule remains unchanged.
