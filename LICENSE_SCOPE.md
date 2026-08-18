# License scope

This repository uses separate licenses for executable software and research
content.

## Software: Apache License 2.0

The root [`LICENSE`](LICENSE) applies to source code, executable scripts,
configuration files, tests, dependency specifications, and release tooling.
This includes the conventional top-level source tree and the executable
capsules under `reproducibility/paper_experiments/`.

Authenticated software, configuration, test, and dependency files inside
`frozen_snapshot/`, and their byte-identical top-level counterparts,
intentionally do not carry inserted license headers: changing those bytes
would invalidate the recorded source hashes. Their license is declared here
instead. Research and narrative files in the snapshots follow the CC BY 4.0
scope below.

## Research content: Creative Commons Attribution 4.0 International

The following material is licensed under CC BY 4.0, whose full text is in
[`LICENSES/CC-BY-4.0.txt`](LICENSES/CC-BY-4.0.txt):

- narrative Markdown and text documentation;
- the frozen theory note `variational_algorithm.txt`;
- figures and figure captions in `paper_outputs/`;
- citation and descriptive release metadata, including `CITATION.cff`,
  `paper_outputs/manifest.json`, and the checksum and JSON metadata under
  `release/`;
- scientific data, metadata, manifests, tables, checkpoints, and figures in
  the two separately distributed companion archives.

Reasonable attribution should identify Saptarshi Mandal, Yashaswini Murthy,
and R. Srikant; name the associated work, *Finite-Time Convergence of
Single-Trajectory Chi-Square Robust Q-Learning With Linear Function
Approximation*; link to this repository; and identify CC BY 4.0. See
[`CITATION.cff`](CITATION.cff) for machine-readable citation metadata.

## Exclusions and third-party material

The complete paper manuscript is not distributed in this repository and is
not covered by this license statement. The authenticated MiniCliff companion
archive retains one historical experiment-section excerpt,
`paper_variational_chi2_gridworld/main/tabular_experiment_subsection.tex`;
that excerpt is research content covered by CC BY 4.0. Package names in the
dependency files and any third-party software installed from them remain
subject to their respective licenses. No license here grants rights in
third-party trademarks.
