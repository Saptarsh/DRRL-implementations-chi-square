# Reproducibility statement

## What this release establishes

Using the frozen source capsules and release-audit environment, the workflows
authenticate the content-addressed MiniCliff and Pendulum source capsules and
their complete declared companion-data closures. The MiniCliff dependency
record describes the later release audit; the original run did not record its
exact dependency environment. From preserved per-run artifacts, the workflows
independently reconstruct the numerical aggregate tables and render the paper
figures in fresh output directories while confirming that the canonical input
trees remain unchanged.

For the audited release candidate:

- MiniCliff validates 25 main-study runs and 120 floor-sensitivity runs,
  reconstructs all declared aggregates, and checks the cross-study reuse of
  `ell=0.1`, seeds 1 through 20.
- Pendulum validates the ten-seed reporting bundle, its development
  authorization chain, and the two seed-31 supplemental raw bundles.
- All declared numerical aggregate files match their historical counterparts.
- Under the frozen release-audit runtime, regenerated PNGs have identical
  decoded pixels. Pendulum retains this strict pixel-identity gate; MiniCliff's
  explicit runtime-drift mode instead requires a fresh render from verified
  aggregates and reports its equality to the canonical image.
- PDF byte identity is not claimed because renderers may embed creation or
  other container metadata. In the audited release-candidate run, normalizing
  the embedded `CreationDate` was sufficient to recover byte identity.
- Canonical result trees are read-only inputs and are authenticated again after
  reproduction.

This stage performs analysis and figure regeneration from saved artifacts. It
does not perform fresh MiniCliff training, neural training, environment
trajectory collection, checkpoint selection, hyperparameter tuning, or seed
selection.

## What requires a fresh full rerun

A complete retraining attempt is needed to test whether the frozen
training/environment/replay/optimizer implementation, starting from the
recorded seeds, produces statistically comparable new outcomes. Such a run is
optional for using the archived evidence to reconstruct the paper plots and is
not a prerequisite for this release.

Fresh training is not promised to be byte-identical to the historical study:

- MiniCliff did not preserve the historical RNG state or sampled trajectory.
- Pendulum did not preserve replay arrays, environment states, or the Python,
  NumPy, and Torch RNG states needed for exact mid-run continuation.
- library, operating-system, BLAS, and hardware differences can change a fresh
  trajectory or rendered container metadata.

Fresh results must therefore be labeled an independent retraining comparison.
They must use a new output directory and must not replace, overwrite, or be
presented as the historical paper evidence.

## Suggested concise description

> We publish content-addressed source capsules and complete declared companion
> data for the MiniCliff and Pendulum experiments. The release independently
> authenticates the archived per-seed records and reconstructs the paper's
> aggregate tables and figures without fresh training. A complete retraining
> protocol is also provided, but exact historical trajectories are not claimed
> because replay state and full RNG state were not archived.

## Versioning

The paper and review response should cite a versioned source tag and its two
matching companion assets. Any later full-retraining evidence or documentation
correction should be published in a new commit, tag, and release. Existing tags
and assets should remain unchanged so the cited scientific record stays
resolvable.
