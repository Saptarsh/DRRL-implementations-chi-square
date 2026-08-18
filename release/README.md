# Release asset metadata

The files in this directory are tag-bound copies of the metadata uploaded with
the companion archives. The archives themselves remain GitHub Release assets
and are not stored in Git history.

| Archive | Compressed bytes | SHA-256 |
|---|---:|---|
| `minicliff-companion-data.tar.gz` | 5,836,123 | `9da4975969bd6cabd8978ef9c4e97dc25ee61787cb8d3631ed156a7e2a42351f` |
| `pendulum-companion-data.tar.gz` | 30,145,268 | `f8cb86e08198179ea99bac86fbb352f0aeb1338b94d47dec1a3afb8bc7c3b7c5` |

`SHA256SUMS` is suitable for `shasum -a 256 -c` on macOS or
`sha256sum -c SHA256SUMS` on Linux. Each JSON sidecar also
records the archive size, compressed SHA-256, extracted inventory, frozen
specification path, and frozen specification SHA-256.

The release upload must use byte-identical copies of these three metadata files
and the two archives from the validated local asset directory.

Before publication, enable GitHub release immutability in the repository
settings if an immutable release is desired. Create the release as a draft,
attach all five assets, verify their names and hashes, and only then publish it.

[`RELEASE_NOTES.md`](RELEASE_NOTES.md) is the source for the corresponding
GitHub Release description.
