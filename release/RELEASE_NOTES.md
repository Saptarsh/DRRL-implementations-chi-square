# Initial reproducibility release

This release publishes the frozen MiniCliff and Pendulum experiment capsules
for *Finite-Time Convergence of Single-Trajectory Chi-Square Robust Q-Learning
With Linear Function Approximation*.

Included in the source tag:

- authenticated frozen source/configuration snapshots and byte-identical live
  source counterparts;
- strict verification and no-training figure-regeneration workflows;
- optional, confirmation-gated full-retraining entry points;
- focused reproducibility and deterministic-export tests;
- hash-bound convenience copies of the two primary paper figures; and
- citation, licensing, companion-data, and reproducibility documentation.

Release assets:

- `minicliff-companion-data.tar.gz` — 609 files, 12,005,948 extracted bytes;
- `pendulum-companion-data.tar.gz` — 158 files, 36,446,249 extracted bytes;
- one JSON authentication sidecar per archive; and
- `SHA256SUMS` for the compressed archives.

The release candidate was tested by extracting both archives into an isolated
source tree, running both strict input verifiers and all focused release tests,
regenerating both paper-facing plot sets without training, and reauthenticating
the MiniCliff output. All declared aggregate comparisons passed, all audited
PNG comparisons were pixel-identical, and the canonical input trees remained
unchanged.

Fresh full retraining is intentionally not part of release construction. The
recorded seeds and complete tuned protocols are supplied for a later independent
retraining comparison, but exact historical trajectories are not promised
because full RNG, replay, and environment state were not archived.

Software is available under Apache-2.0. Data, figures, and narrative
documentation are available under CC BY 4.0 as detailed in
[`LICENSE_SCOPE.md`](../LICENSE_SCOPE.md).
