# Companion data

The source tag is intentionally small. The complete declared companion-data
scope required by these workflows is published as two versioned GitHub Release
assets, each preserving repository-relative paths.

## Asset names and extracted inventories

| Asset | Files | Extracted bytes | Extracted inventory SHA-256 |
|---|---:|---:|---|
| `minicliff-companion-data.tar.gz` | 609 | 12,005,948 | `8d02a0b0ea68f1fe1cae82c35ff1418410f982dc7a0aa36b90d8edecba6df864` |
| `pendulum-companion-data.tar.gz` | 158 | 36,446,249 | `d1c278aaeaab7224fc3a10f64af0a9c4ab33784adc2dd2c62be5373c616ef01d` |

The compressed archive SHA-256 values are release-specific and are recorded in
the accompanying `SHA256SUMS` and `.tar.gz.json` sidecars. A compressed-file
hash authenticates the downloaded container. The extracted inventory hash
authenticates the member names, content hashes, and sizes after extraction.
They are different quantities and should not be substituted for one another.

| Asset | Compressed bytes | Compressed SHA-256 |
|---|---:|---|
| `minicliff-companion-data.tar.gz` | 5,836,123 | `9da4975969bd6cabd8978ef9c4e97dc25ee61787cb8d3631ed156a7e2a42351f` |
| `pendulum-companion-data.tar.gz` | 30,145,268 | `f8cb86e08198179ea99bac86fbb352f0aeb1338b94d47dec1a3afb8bc7c3b7c5` |

Tag-bound copies of the sidecars and checksum list are kept in
[`release/`](../release/).

## Exact MiniCliff scope

The MiniCliff archive contains only:

```text
paper_variational_chi2_gridworld/main
paper_variational_chi2_gridworld/ell_sensitivity
```

The authenticated sub-inventories are:

| Study | Files | Bytes | Inventory SHA-256 |
|---|---:|---:|---|
| `main` | 116 | 2,862,826 | `5fd528c52932d5eb0d42429bda757162f93091953d64b773bdfd985278affe87` |
| `ell_sensitivity` | 493 | 9,143,122 | `6a30e08a0606602c70c154b01c9077efba30305d5f67de77958724621314412b` |

The historical `quick` directory and older tabular studies are not included.

## Exact Pendulum scope

The Pendulum archive contains exactly the closure declared by
`reproducibility/paper_experiments/pendulum/frozen_inputs.json`:

1. the complete `reporting_full_nn_v1` tree;
2. `development_full_nn_v1/aggregated/development_gate.json` and
   `development_full_nn_v1/reporting_freeze_manifest.json`;
3. the study manifest and complete `raw/seed_0031` bundle for
   `exact_inner_seed31_v1`; and
4. the study manifest and complete `raw/seed_0031` bundle for
   `ablation_full_nn_seed31_v1`.

The full development tree, adjacent supplemental aggregates, and unrelated
Pendulum studies are excluded.

Four authenticated Pendulum JSON fields retain historical local absolute
paths. These fields record where the original configuration or authorization
input was located. They are inert provenance, not credentials, and remain
byte-exact because editing them would invalidate the frozen hashes. Runtime
execution uses the repository-relative frozen configuration copies.

The scientific records consist of CSV, JSON, NPZ, PyTorch checkpoint, PDF, PNG,
and TeX files. Conventional console `.log` files were not part of the
historical scientific record and are not required by the workflows.

## Download, verify, and extract

Download these five assets from one versioned release:

```text
minicliff-companion-data.tar.gz
minicliff-companion-data.tar.gz.json
pendulum-companion-data.tar.gz
pendulum-companion-data.tar.gz.json
SHA256SUMS
```

Then run:

```bash
cd /path/to/downloads
shasum -a 256 -c SHA256SUMS
# Linux alternative: sha256sum -c SHA256SUMS

cd /path/to/DRRL-implementations-chi-square
tar -xzf /path/to/downloads/minicliff-companion-data.tar.gz
tar -xzf /path/to/downloads/pendulum-companion-data.tar.gz
```

There is no enclosing directory inside either archive. Extraction at the
repository root creates the exact paths expected by the verification wrappers.
Do not rename, edit, or reserialize authenticated members.

## Maintainer archive construction

The exporter first runs both strict input-verification workflows. It derives
the Pendulum selection from the frozen manifest rather than archiving a broad
`results/` directory. Tar members are sorted, limited to regular files, and
written with normalized metadata. The completed archives are reopened and
checked member by member, and the source trees are reauthenticated before
atomic publication.

Use a fresh destination outside the checkout:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  tools/build_companion_archives.py \
  --repo-root . \
  --output-dir /fresh/nonexistent/companion-assets
```

The builder refuses existing output directories, source overlap, symlinked
payload entries, special nodes, missing or extra authenticated data, and
content drift. Rebuilding unchanged inputs twice in the same tested environment
must produce byte-identical archives. The hashes published with the versioned
release remain authoritative across compression-library implementations.
