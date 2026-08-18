#!/usr/bin/env python3
"""Authenticate and reproduce the paper-facing Pendulum experiment safely.

The default ``verify`` command is read-only.  ``reproduce`` copies only the
authenticated raw reporting bundles into a fresh staging directory, invokes
the tag-contained frozen study snapshot with ``--skip-existing``, renders
figures in that copy, and atomically publishes the new directory.  The
canonical results tree is never passed to code that writes.  Live scientific
files are optional correspondence checks, not executable dependencies.

The ``full-rerun`` command is deliberately opt-in and computationally
expensive.  It repeats the tuned five-seed development and ten-seed reporting
studies in a fresh output root; it is not needed to regenerate paper plots from
the preserved raw data.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageChops


HERE = Path(__file__).resolve().parent
CAPSULE_ROOT = HERE
SNAPSHOT_ROOT = HERE / "frozen_snapshot"
REPO_ROOT = HERE.parents[2]
INPUT_SPEC_PATH = HERE / "frozen_inputs.json"
REPORT_NAME = "pendulum_reproduction_report.json"
EXPENSIVE_CONFIRMATION = "RUN_THE_FULL_15_SEED_TUNED_STUDY"


class VerificationError(RuntimeError):
    """Raised when a content-addressed input or protocol invariant fails."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VerificationError(f"Cannot read JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise VerificationError(f"Expected a JSON object in {path}.")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_regular_file(path: Path, label: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise VerificationError(f"{label} is absent, non-regular, or a symlink: {path}")


def _verify_file(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    _assert_regular_file(path, label)
    actual = _sha256(path)
    if actual != expected_sha256:
        raise VerificationError(
            f"{label} SHA-256 differs: expected {expected_sha256}, observed {actual}: {path}"
        )
    return {"path": str(path), "sha256": actual, "size_bytes": path.stat().st_size}


def _resolve_repo_path(relative: str, *, repo_root: Path = REPO_ROOT) -> Path:
    candidate = (repo_root / relative).resolve()
    root = repo_root.resolve()
    if candidate != root and root not in candidate.parents:
        raise VerificationError(f"Frozen path escapes the repository: {relative!r}")
    return candidate


def _resolve_snapshot_path(relative: str, *, snapshot_root: Path = SNAPSHOT_ROOT) -> Path:
    candidate = (snapshot_root / relative).resolve()
    root = snapshot_root.resolve()
    if candidate != root and root not in candidate.parents:
        raise VerificationError(f"Frozen snapshot path escapes its root: {relative!r}")
    return candidate


def load_input_spec(path: Path = INPUT_SPEC_PATH) -> dict[str, Any]:
    spec = _read_json(path)
    if spec.get("schema") != "rvchi2_dqn.pendulum_reproducibility_inputs.v1":
        raise VerificationError("Unsupported Pendulum reproducibility input schema.")
    return spec


def verify_snapshot(
    spec: Mapping[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
    snapshot_root: Path = SNAPSHOT_ROOT,
    require_live_correspondence: bool = False,
) -> dict[str, Any]:
    """Authenticate the complete executable snapshot and optionally live bytes."""

    expected = spec.get("source_snapshot")
    correspondence = spec.get("live_source_correspondence")
    if not isinstance(expected, dict) or not expected:
        raise VerificationError("Frozen source snapshot inventory is absent or malformed.")
    if not isinstance(correspondence, dict) or set(correspondence) != set(expected):
        raise VerificationError("Live-source correspondence does not cover the snapshot exactly.")
    if snapshot_root.is_symlink() or not snapshot_root.is_dir():
        raise VerificationError(f"Frozen snapshot root is absent or a symlink: {snapshot_root}")
    unsafe = sorted(
        str(path)
        for path in snapshot_root.rglob("*")
        if path.is_symlink() or (not path.is_file() and not path.is_dir())
    )
    if unsafe:
        raise VerificationError(f"Frozen snapshot contains symlinks or special nodes: {unsafe}")
    actual = {
        path.relative_to(snapshot_root).as_posix()
        for path in snapshot_root.rglob("*")
        if path.is_file()
    }
    if actual != set(expected):
        raise VerificationError(
            "Frozen snapshot inventory differs; "
            f"missing={sorted(set(expected) - actual)}, "
            f"unexpected={sorted(actual - set(expected))}"
        )

    snapshot_records: list[dict[str, Any]] = []
    live_records: list[dict[str, Any]] = []
    for relative in sorted(expected):
        expected_hash = expected[relative]
        if not isinstance(expected_hash, str):
            raise VerificationError(f"Snapshot hash is malformed for {relative}.")
        frozen = _resolve_snapshot_path(relative, snapshot_root=snapshot_root)
        snapshot_records.append(
            {
                "relative_path": relative,
                **_verify_file(frozen, expected_hash, f"frozen snapshot {relative}"),
            }
        )
        live_relative = correspondence[relative]
        if not isinstance(live_relative, str):
            raise VerificationError(f"Live-source path is malformed for {relative}.")
        live = _resolve_repo_path(live_relative, repo_root=repo_root)
        live_exists = live.is_file() and not live.is_symlink()
        live_hash = _sha256(live) if live_exists else None
        matches = live_hash == expected_hash
        live_records.append(
            {
                "snapshot_relative_path": relative,
                "live_relative_path": live_relative,
                "live_exists_as_regular_file": live_exists,
                "live_sha256": live_hash,
                "matches_frozen_snapshot": matches,
            }
        )
        if require_live_correspondence and not matches:
            raise VerificationError(
                f"Live-source drift for {live_relative}; frozen snapshot remains authoritative."
            )
    digest = hashlib.sha256(
        json.dumps(expected, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "status": "verified",
        "snapshot_root": str(snapshot_root.resolve()),
        "snapshot_file_count": len(snapshot_records),
        "snapshot_inventory_sha256": digest,
        "snapshot_files": snapshot_records,
        "live_correspondence_required": require_live_correspondence,
        "all_live_files_match": all(row["matches_frozen_snapshot"] for row in live_records),
        "live_correspondence": live_records,
        "authority_note": (
            "Frozen snapshot bytes are authoritative. Live correspondence is optional unless "
            "--require-live-correspondence is requested."
        ),
    }


def verify_inventory(
    repo_root: Path,
    inventory_path: Path,
    expected_inventory_sha256: str,
    canonical_root: Path,
    *,
    path_overrides: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    """Verify every declared file and the exact canonical reporting inventory."""

    repo_root = repo_root.resolve()
    inventory_path = inventory_path.resolve()
    canonical_root = canonical_root.resolve()
    inventory_record = _verify_file(
        inventory_path, expected_inventory_sha256, "canonical paper inventory"
    )
    inventory = _read_json(inventory_path)
    entries = inventory.get("file_inventory", {}).get("files")
    declared_count = inventory.get("file_inventory", {}).get("file_count")
    if not isinstance(entries, list) or declared_count != len(entries) or not entries:
        raise VerificationError("Canonical inventory has an invalid file list or count.")

    seen: set[str] = set()
    verified: list[dict[str, Any]] = []
    canonical_declared: set[Path] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise VerificationError(f"Inventory entry {index} is not an object.")
        relative = entry.get("path")
        expected = entry.get("sha256")
        expected_size = entry.get("size_bytes")
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise VerificationError(f"Inventory entry {index} lacks path or SHA-256.")
        if relative in seen:
            raise VerificationError(f"Duplicate inventory path: {relative}")
        seen.add(relative)
        logical_path = _resolve_repo_path(relative, repo_root=repo_root)
        path = (
            Path(path_overrides[relative]).resolve()
            if path_overrides is not None and relative in path_overrides
            else logical_path
        )
        row = _verify_file(path, expected, f"inventory entry {relative}")
        if expected_size != row["size_bytes"]:
            raise VerificationError(
                f"Inventory size differs for {relative}: expected {expected_size}, "
                f"observed {row['size_bytes']}"
            )
        verified.append(
            {
                "relative_path": relative,
                "authenticated_from_snapshot": path != logical_path,
                **row,
            }
        )
        if logical_path == canonical_root or canonical_root in logical_path.parents:
            canonical_declared.add(logical_path)

    actual_canonical = {
        path.resolve()
        for path in canonical_root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    unsafe_nodes = sorted(
        str(path)
        for path in canonical_root.rglob("*")
        if path.is_symlink() or (not path.is_file() and not path.is_dir())
    )
    if unsafe_nodes:
        raise VerificationError(
            f"Canonical reporting tree contains symlinks or special nodes: {unsafe_nodes}"
        )
    expected_canonical = canonical_declared | {inventory_path.resolve()}
    if actual_canonical != expected_canonical:
        missing = sorted(str(path) for path in expected_canonical - actual_canonical)
        extra = sorted(str(path) for path in actual_canonical - expected_canonical)
        raise VerificationError(
            f"Canonical reporting tree inventory differs; missing={missing}, extra={extra}"
        )

    return {
        "inventory": inventory_record,
        "declared_file_count": len(verified),
        "canonical_tree_file_count": len(actual_canonical),
        "authorization_status": inventory.get("authorization", {}).get("status"),
        "deep_validation_status": inventory.get("deep_validation", {}).get("status"),
    }


def _verify_raw_manifest(raw_manifest_path: Path) -> dict[str, Any]:
    manifest = _read_json(raw_manifest_path)
    declared = manifest.get("files_sha256")
    if not isinstance(declared, dict) or not declared:
        raise VerificationError(f"Raw manifest has no artifact map: {raw_manifest_path}")
    raw_dir = raw_manifest_path.parent
    expected_names = set(declared) | {raw_manifest_path.name}
    actual_names = {
        path.name
        for path in raw_dir.iterdir()
        if path.is_file() and not path.is_symlink()
    }
    if actual_names != expected_names:
        raise VerificationError(
            f"Raw bundle inventory differs at {raw_dir}; expected={sorted(expected_names)}, "
            f"observed={sorted(actual_names)}"
        )
    for name, expected in declared.items():
        _verify_file(raw_dir / name, str(expected), f"raw artifact {raw_dir.name}/{name}")
    return {
        "path": str(raw_manifest_path),
        "sha256": _sha256(raw_manifest_path),
        "artifact_count": len(declared),
        "source_sha256": manifest.get("source_sha256"),
        "status": manifest.get("status"),
    }


def _runtime_record() -> dict[str, str]:
    packages = {
        "torch": "torch",
        "numpy": "numpy",
        "gymnasium": "gymnasium",
        "pandas": "pandas",
        "matplotlib": "matplotlib",
    }
    record = {"python": platform.python_version(), "platform": platform.platform()}
    for label, distribution in packages.items():
        try:
            record[label] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            record[label] = "missing"
    return record


def _runtime_comparison(recorded: Mapping[str, Any]) -> dict[str, Any]:
    current = _runtime_record()
    version_keys = ("python", "torch", "numpy", "gymnasium", "pandas", "matplotlib")
    mismatches = {
        key: {"expected": str(recorded[key]), "observed": current.get(key)}
        for key in version_keys
        if current.get(key) != str(recorded[key])
    }
    platform_matches = current["platform"] == str(recorded.get("platform"))
    return {
        "current": current,
        "recorded": dict(recorded),
        "version_mismatches": mismatches,
        "principal_versions_match": not mismatches,
        "platform_matches_recorded_run": platform_matches,
        "platform_note": (
            "Platform identity is reported, not required for raw-data aggregation. "
            "A full neural retraining on a different platform may not be bit-exact."
        ),
    }


def verify_companion_inventory(
    spec: Mapping[str, Any], *, repo_root: Path = REPO_ROOT
) -> dict[str, Any]:
    """Authenticate the minimum external-data closure as one inventory."""

    canonical = spec["canonical_reporting"]
    canonical_root = _resolve_repo_path(canonical["root"], repo_root=repo_root)
    paths = [path for path in canonical_root.rglob("*") if path.is_file()]
    paths.extend(
        _resolve_repo_path(canonical[key], repo_root=repo_root)
        for key in ("development_gate", "reporting_freeze")
    )
    for item in spec["supplemental_studies"]:
        supplement_root = _resolve_repo_path(item["root"], repo_root=repo_root)
        paths.append(supplement_root / "manifest.json")
        raw_root = _resolve_repo_path(item["raw_manifest"], repo_root=repo_root).parent
        paths.extend(path for path in raw_root.iterdir() if path.is_file())
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(set(paths)):
        _assert_regular_file(path, "companion-data entry")
        relative = path.relative_to(repo_root.resolve()).as_posix()
        records[relative] = {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
    digest = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    frozen = spec["companion_data_inventory"]
    observed = {
        "file_count": len(records),
        "total_bytes": sum(row["size_bytes"] for row in records.values()),
        "inventory_sha256": digest,
    }
    for key, value in observed.items():
        if value != frozen[key]:
            raise VerificationError(
                f"Companion-data {key} differs: expected {frozen[key]}, observed {value}."
            )
    return {"status": "verified", **observed, "scope": frozen["scope"]}


def verify_all_inputs(
    *,
    repo_root: Path = REPO_ROOT,
    spec_path: Path = INPUT_SPEC_PATH,
    require_live_correspondence: bool = False,
) -> dict[str, Any]:
    spec = load_input_spec(spec_path)
    snapshot = verify_snapshot(
        spec,
        repo_root=repo_root,
        require_live_correspondence=require_live_correspondence,
    )
    snapshot_overrides = {
        live_relative: _resolve_snapshot_path(snapshot_relative)
        for snapshot_relative, live_relative in spec["live_source_correspondence"].items()
    }
    canonical = spec["canonical_reporting"]
    canonical_root = _resolve_repo_path(canonical["root"], repo_root=repo_root)
    inventory_path = _resolve_repo_path(canonical["inventory"], repo_root=repo_root)
    inventory = verify_inventory(
        repo_root,
        inventory_path,
        canonical["inventory_sha256"],
        canonical_root,
        path_overrides=snapshot_overrides,
    )
    _verify_file(
        canonical_root / "manifest.json",
        canonical["study_manifest_sha256"],
        "canonical reporting study manifest",
    )

    study = _read_json(canonical_root / "manifest.json")
    raw_records = []
    raw_runs = study.get("raw_runs")
    if raw_runs != [f"raw/seed_{seed:04d}" for seed in range(101, 111)]:
        raise VerificationError("Canonical reporting manifest does not contain seeds 101--110.")
    for relative in raw_runs:
        raw_records.append(_verify_raw_manifest(canonical_root / relative / "manifest.json"))
    expected_sources = study.get("source_sha256")
    source_map = spec.get("scientific_source_map")
    if not isinstance(source_map, dict) or not source_map:
        raise VerificationError("Frozen scientific source map is absent or malformed.")
    frozen_sources = {
        name: spec["source_snapshot"][relative]
        for name, relative in source_map.items()
    }
    if expected_sources != frozen_sources:
        raise VerificationError(
            "Canonical reporting source map differs from the frozen snapshot source map."
        )
    if not isinstance(expected_sources, dict) or any(
        row["source_sha256"] != expected_sources for row in raw_records
    ):
        raise VerificationError(
            "Canonical raw reporting bundles do not share the study source map."
        )

    fixed_records = []
    for path_key, digest_key, label in (
        ("development_gate", "development_gate_sha256", "development gate"),
        ("reporting_freeze", "reporting_freeze_sha256", "reporting freeze"),
    ):
        fixed_records.append(
            _verify_file(
                _resolve_repo_path(canonical[path_key], repo_root=repo_root),
                canonical[digest_key],
                label,
            )
        )
    for path_key, digest_key, label in (
        ("development_config", "development_config_sha256", "development config"),
        ("reporting_config", "reporting_config_sha256", "reporting config"),
    ):
        relative = canonical[path_key]
        fixed_records.append(
            _verify_file(
                snapshot_overrides[relative],
                canonical[digest_key],
                f"frozen snapshot {label}",
            )
        )

    gate = _read_json(_resolve_repo_path(canonical["development_gate"], repo_root=repo_root))
    freeze = _read_json(_resolve_repo_path(canonical["reporting_freeze"], repo_root=repo_root))
    if not (
        gate.get("development_gate_passed") is True
        and gate.get("reporting_authorized") is True
        and freeze.get("status") == "reporting_authorized"
        and freeze.get("development_gate_sha256") == canonical["development_gate_sha256"]
        and freeze.get("development_seeds") == list(range(31, 36))
        and freeze.get("reporting_seeds") == list(range(101, 111))
        and freeze.get("source_sha256") == expected_sources
    ):
        raise VerificationError("Development authorization chain is not valid.")

    executable_records = []
    for path_key, digest_key in (("runner", "runner_sha256"), ("plotter", "plotter_sha256")):
        relative = spec["executables"][path_key]
        executable_records.append(
            _verify_file(
                snapshot_overrides[relative],
                spec["executables"][digest_key],
                f"frozen snapshot Pendulum {path_key}",
            )
        )
    dependency_relative = spec["dependency_lock"]["path"]
    dependency_lock = _verify_file(
        snapshot_overrides[dependency_relative],
        spec["dependency_lock"]["sha256"],
        "frozen snapshot dependency lock",
    )

    supplements = []
    for item in spec["supplemental_studies"]:
        supplement_root = _resolve_repo_path(item["root"], repo_root=repo_root)
        study_record = _verify_file(
            supplement_root / "manifest.json",
            item["study_manifest_sha256"],
            f"{item['label']} study manifest",
        )
        raw_manifest = _resolve_repo_path(item["raw_manifest"], repo_root=repo_root)
        _verify_file(
            raw_manifest,
            item["raw_manifest_sha256"],
            f"{item['label']} raw manifest",
        )
        raw_record = _verify_raw_manifest(raw_manifest)
        if raw_record["source_sha256"] != expected_sources:
            raise VerificationError(
                f"{item['label']} was not produced by the frozen scientific source map."
            )
        config_record = _verify_file(
            snapshot_overrides[item["config"]],
            item["config_sha256"],
            f"frozen snapshot {item['label']} config",
        )
        supplements.append(
            {
                "label": item["label"],
                "study_manifest": study_record,
                "raw_bundle": raw_record,
                "config": config_record,
            }
        )

    runtime = _runtime_comparison(spec["recorded_runtime"])
    companion_inventory = verify_companion_inventory(spec, repo_root=repo_root)
    return {
        "schema": "rvchi2_dqn.pendulum_input_verification.v1",
        "status": "verified" if runtime["principal_versions_match"] else "verified_with_runtime_drift",
        "frozen_snapshot": snapshot,
        "canonical_root": str(canonical_root),
        "inventory": inventory,
        "raw_reporting_bundles": raw_records,
        "authorization_files": fixed_records,
        "executables": executable_records,
        "dependency_lock": dependency_lock,
        "supplemental_studies": supplements,
        "companion_data_inventory": companion_inventory,
        "runtime": runtime,
        "known_limitations": [
            "Replay arrays, environment states, and RNG states were not saved in the historical runs.",
            "Exact mid-run resume is impossible; a full reproduction retrains from deterministic seeds.",
            "The historical Git worktree was dirty, so content hashes—not the Git revision—are authoritative.",
            "The artifact inventory is content-addressed but not externally signed.",
        ],
    }


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        path for path in root.rglob("*") if path.is_file() and not path.is_symlink()
    )
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _png_pixels_equal(left: Path, right: Path) -> bool:
    """Compare decoded RGBA pixels rather than PNG container metadata."""

    with Image.open(left) as left_image, Image.open(right) as right_image:
        left_rgba = left_image.convert("RGBA")
        right_rgba = right_image.convert("RGBA")
        return left_rgba.size == right_rgba.size and ImageChops.difference(
            left_rgba, right_rgba
        ).getbbox() is None


def assert_fresh_output(output: Path, canonical_root: Path) -> Path:
    resolved = output.expanduser().resolve()
    canonical = canonical_root.resolve()
    if resolved == canonical or canonical in resolved.parents or resolved in canonical.parents:
        raise VerificationError(
            f"Output must not equal, contain, or lie inside the canonical tree: {resolved}"
        )
    if resolved.exists():
        raise VerificationError(f"Output must be a fresh, nonexistent path: {resolved}")
    return resolved


def _portable_freeze(
    source: Path, destination: Path, development_gate: Path
) -> dict[str, Any]:
    freeze = _read_json(source)
    original_gate_path = freeze.get("development_gate_path")
    portable_gate_path = str(development_gate.resolve())
    freeze["development_gate_path"] = portable_gate_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "source_path": str(source),
        "source_sha256": _sha256(source),
        "portable_relative_path": "reproduction_inputs/portable_reporting_freeze.json",
        "portable_sha256": _sha256(destination),
        "original_development_gate_path": original_gate_path,
        "portable_development_gate_path": portable_gate_path,
        "transformation": (
            "development_gate_path normalized to the authenticated gate in the current clone; "
            "all authorization/config/source fields retained"
        ),
    }


def _run(
    command: Sequence[str], *, cwd: Path, environment: Mapping[str, str]
) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=dict(environment),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise VerificationError(
            "Child command failed with exit code "
            f"{completed.returncode}: {_command_strings([command])[0]}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return {
        "command": [str(value) for value in command],
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _command_strings(commands: Iterable[Sequence[str]]) -> list[list[str]]:
    return [[str(value) for value in command] for command in commands]


def reproduce_from_raw(
    output: Path,
    *,
    repo_root: Path = REPO_ROOT,
    spec_path: Path = INPUT_SPEC_PATH,
) -> dict[str, Any]:
    verification_before = verify_all_inputs(repo_root=repo_root, spec_path=spec_path)
    if not verification_before["runtime"]["principal_versions_match"]:
        raise VerificationError(
            "Principal runtime versions differ; use the recorded environment before reproduction."
        )
    spec = load_input_spec(spec_path)
    canonical = spec["canonical_reporting"]
    canonical_root = _resolve_repo_path(canonical["root"], repo_root=repo_root)
    destination = assert_fresh_output(output, canonical_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canonical_before = _tree_digest(canonical_root)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    ).resolve()
    published = False
    try:
        source_raw = canonical_root / "raw"
        shutil.copytree(source_raw, staging / "raw")
        portable_freeze_path = staging / "reproduction_inputs" / "portable_reporting_freeze.json"
        portable_freeze = _portable_freeze(
            _resolve_repo_path(canonical["reporting_freeze"], repo_root=repo_root),
            portable_freeze_path,
            _resolve_repo_path(canonical["development_gate"], repo_root=repo_root),
        )
        runner = _resolve_snapshot_path(spec["executables"]["runner"])
        plotter = _resolve_snapshot_path(spec["executables"]["plotter"])
        config = _resolve_snapshot_path(canonical["reporting_config"])
        commands = [
            [
                sys.executable,
                str(runner),
                "--config",
                str(config),
                "--output-root",
                str(staging),
                "--skip-existing",
                "--freeze-manifest",
                str(portable_freeze_path),
            ],
            [sys.executable, str(plotter), str(staging), "--representative-seed", "101"],
        ]
        environment = os.environ.copy()
        environment.setdefault("PYTHONHASHSEED", "0")
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        runtime_directory = staging / "reproduction_runtime"
        mpl_cache = runtime_directory / "matplotlib-cache"
        mpl_cache.mkdir(parents=True, exist_ok=True)
        environment["MPLCONFIGDIR"] = str(mpl_cache)
        execution = [
            _run(command, cwd=SNAPSHOT_ROOT, environment=environment)
            for command in commands
        ]
        # Font caches are execution scratch, not scientific artifacts.
        shutil.rmtree(runtime_directory)

        aggregate_comparison = []
        for relative in spec["aggregate_files"]:
            canonical_path = canonical_root / relative
            reproduced_path = staging / relative
            _assert_regular_file(reproduced_path, f"reproduced aggregate {relative}")
            canonical_sha = _sha256(canonical_path)
            reproduced_sha = _sha256(reproduced_path)
            equal = canonical_sha == reproduced_sha
            aggregate_comparison.append(
                {
                    "path": relative,
                    "canonical_sha256": canonical_sha,
                    "reproduced_sha256": reproduced_sha,
                    "byte_identical": equal,
                }
            )
            if not equal:
                raise VerificationError(f"Reproduced aggregate differs: {relative}")

        figure_hashes = {}
        for relative in spec["required_figure_files"]:
            path = staging / relative
            _assert_regular_file(path, f"reproduced figure {relative}")
            record: dict[str, Any] = {
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            canonical_figure = canonical_root / relative
            if relative.endswith(".png"):
                record["pixel_identical_to_canonical"] = _png_pixels_equal(
                    canonical_figure, path
                )
                if not record["pixel_identical_to_canonical"]:
                    raise VerificationError(
                        f"Reproduced PNG pixels differ from canonical: {relative}"
                    )
            elif relative.endswith(".pdf"):
                record["byte_identical_to_canonical"] = (
                    _sha256(canonical_figure) == record["sha256"]
                )
                record["comparison_note"] = (
                    "PDF container bytes may differ because Matplotlib embeds creation metadata."
                )
            figure_hashes[relative] = record

        verification_after = verify_all_inputs(repo_root=repo_root, spec_path=spec_path)
        canonical_after = _tree_digest(canonical_root)
        if canonical_before != canonical_after:
            raise VerificationError("Canonical reporting tree changed during reproduction.")
        report = {
            "schema": "rvchi2_dqn.pendulum_reproduction_report.v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "reproduced_from_authenticated_raw",
            "output": str(destination),
            "canonical_root": str(canonical_root),
            "canonical_inventory_sha256": spec["canonical_reporting"]["inventory_sha256"],
            "canonical_tree_sha256_before": canonical_before,
            "canonical_tree_sha256_after": canonical_after,
            "canonical_tree_unchanged": True,
            "frozen_snapshot_inventory_sha256": verification_after["frozen_snapshot"][
                "snapshot_inventory_sha256"
            ],
            "executed_from_frozen_snapshot": True,
            "raw_seed_count": len(list((staging / "raw").glob("seed_*"))),
            "commands": _command_strings(commands),
            "execution": execution,
            "portable_freeze": portable_freeze,
            "aggregate_comparison": aggregate_comparison,
            "all_aggregates_byte_identical": True,
            "reproduced_figure_hashes": figure_hashes,
            "figure_note": (
                "Every canonical PNG is required to be pixel-identical. Aggregate files "
                "are required to be byte-identical. PDF bytes may differ only because "
                "Matplotlib embeds creation metadata."
            ),
            "runtime": verification_after["runtime"],
        }
        (staging / REPORT_NAME).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(staging, destination)
        published = True
        return report
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


def full_tuned_rerun(
    output: Path,
    *,
    confirmation: str,
    max_parallel: int,
    repo_root: Path = REPO_ROOT,
    spec_path: Path = INPUT_SPEC_PATH,
) -> dict[str, Any]:
    if confirmation != EXPENSIVE_CONFIRMATION:
        raise VerificationError(
            "Full tuned rerun is disabled unless --confirm-expensive-rerun is exactly "
            f"{EXPENSIVE_CONFIRMATION!r}."
        )
    if max_parallel < 1:
        raise VerificationError("max-parallel must be positive.")
    verification = verify_all_inputs(repo_root=repo_root, spec_path=spec_path)
    if not verification["runtime"]["principal_versions_match"]:
        raise VerificationError("Principal runtime versions differ from the frozen runs.")
    spec = load_input_spec(spec_path)
    canonical = spec["canonical_reporting"]
    canonical_root = _resolve_repo_path(canonical["root"], repo_root=repo_root)
    root = assert_fresh_output(output, canonical_root)
    development = root / "development_full_nn_v1"
    reporting = root / "reporting_full_nn_v1"
    runner = _resolve_snapshot_path(spec["executables"]["runner"])
    plotter = _resolve_snapshot_path(spec["executables"]["plotter"])
    development_config = _resolve_snapshot_path(canonical["development_config"])
    reporting_config = _resolve_snapshot_path(canonical["reporting_config"])
    commands = [
        [
            sys.executable,
            str(runner),
            "--config",
            str(development_config),
            "--output-root",
            str(development),
            "--max-parallel",
            str(max_parallel),
        ],
        [
            sys.executable,
            str(runner),
            "--config",
            str(reporting_config),
            "--output-root",
            str(reporting),
            "--max-parallel",
            str(max_parallel),
            "--freeze-manifest",
            str(development / "reporting_freeze_manifest.json"),
        ],
        [sys.executable, str(plotter), str(reporting), "--representative-seed", "101"],
    ]
    # Creating this root is itself the explicit start of the expensive rerun.  The
    # existing runner refuses all pre-existing per-seed outputs, so interrupted
    # work is never silently mixed with a later invocation.
    root.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment.setdefault("PYTHONHASHSEED", "0")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["MPLCONFIGDIR"] = str(root / "matplotlib-cache")
    Path(environment["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    execution = [
        _run(command, cwd=SNAPSHOT_ROOT, environment=environment)
        for command in commands
    ]
    report = {
        "schema": "rvchi2_dqn.pendulum_full_rerun_report.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed_full_tuned_rerun",
        "warning": (
            "This is a fresh 5-development-seed plus 10-reporting-seed neural retraining, "
            "not plot regeneration. Hardware/platform differences can change neural results."
        ),
        "commands": _command_strings(commands),
        "execution": execution,
        "executed_from_frozen_snapshot": True,
        "frozen_snapshot_root": str(SNAPSHOT_ROOT),
        "frozen_snapshot_inventory_sha256": verification["frozen_snapshot"][
            "snapshot_inventory_sha256"
        ],
        "frozen_input_inventory_sha256": canonical["inventory_sha256"],
        "runtime": verification["runtime"],
    }
    (root / REPORT_NAME).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def tag_audit(
    *, repo_root: Path = REPO_ROOT, spec_path: Path = INPUT_SPEC_PATH
) -> dict[str, Any]:
    """Report whether the source capsule can be carried intact by a Git tag."""

    spec = load_input_spec(spec_path)
    snapshot = verify_snapshot(spec, repo_root=repo_root)
    files = sorted(
        path
        for path in CAPSULE_ROOT.rglob("*")
        if path.is_file()
    )
    reproducibility_test = repo_root / "tests" / "test_pendulum_paper_reproducibility.py"
    _assert_regular_file(reproducibility_test, "Pendulum reproducibility test")
    files.append(reproducibility_test)
    files = sorted(set(files))
    untracked: list[str] = []
    for path in files:
        relative = path.relative_to(repo_root).as_posix()
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "ls-files", "--error-unmatch", relative],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if completed.returncode != 0:
            untracked.append(relative)
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "status",
            "--porcelain=v1",
            "--",
            CAPSULE_ROOT.relative_to(repo_root).as_posix(),
            reproducibility_test.relative_to(repo_root).as_posix(),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise VerificationError(
            f"Could not audit capsule cleanliness against Git: {completed.stderr.strip()}"
        )
    dirty = sorted(
        line[3:]
        for line in completed.stdout.splitlines()
        if len(line) >= 4 and not line.startswith("??")
    )
    canonical = spec["canonical_reporting"]
    companions = [
        {
            "label": "paper-facing reporting_full_nn_v1 bundle",
            "root": canonical["root"],
            "inventory": canonical["inventory"],
            "inventory_sha256": canonical["inventory_sha256"],
        },
        {
            "label": "development authorization chain",
            "paths": [canonical["development_gate"], canonical["reporting_freeze"]],
            "sha256": [
                canonical["development_gate_sha256"],
                canonical["reporting_freeze_sha256"],
            ],
        },
    ]
    companions.extend(
        {
            "label": item["label"],
            "root": item["root"],
            "study_manifest_sha256": item["study_manifest_sha256"],
            "raw_manifest": item["raw_manifest"],
            "raw_manifest_sha256": item["raw_manifest_sha256"],
        }
        for item in spec["supplemental_studies"]
    )
    return {
        "schema": "rvchi2_dqn.pendulum_tag_audit.v1",
        "tag_ready": not untracked and not dirty,
        "capsule_files_checked": len(files),
        "untracked_capsule_files": untracked,
        "dirty_tracked_capsule_files": dirty,
        "frozen_snapshot_file_count": snapshot["snapshot_file_count"],
        "frozen_snapshot_inventory_sha256": snapshot["snapshot_inventory_sha256"],
        "companion_data_required": True,
        "companion_data_inventory": dict(spec["companion_data_inventory"]),
        "companion_data": companions,
        "note": (
            "A source tag carries this workflow, frozen code/configuration, and dependency "
            "lock. Canonical neural-result trees are external companion data and a complete "
            "reproducibility release must publish them with the declared content hashes."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify", help="Read-only verification of all frozen inputs.")
    verify.add_argument(
        "--require-live-correspondence",
        action="store_true",
        help="also require every current live code/config file to equal its frozen snapshot",
    )
    reproduce = subparsers.add_parser(
        "reproduce", help="Regenerate aggregates and figures from copied raw bundles."
    )
    reproduce.add_argument("--output", type=Path, required=True)
    rerun = subparsers.add_parser(
        "full-rerun", help="OPTIONAL/EXPENSIVE: retrain all 5+10 tuned seeds from scratch."
    )
    rerun.add_argument("--output", type=Path, required=True)
    rerun.add_argument("--max-parallel", type=int, default=1)
    rerun.add_argument("--confirm-expensive-rerun", default="")
    subparsers.add_parser("tag-audit", help="check capsule tracking and companion-data status")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "verify":
            result = verify_all_inputs(
                require_live_correspondence=args.require_live_correspondence
            )
        elif args.command == "reproduce":
            result = reproduce_from_raw(args.output)
        elif args.command == "full-rerun":
            result = full_tuned_rerun(
                args.output,
                confirmation=args.confirm_expensive_rerun,
                max_parallel=args.max_parallel,
            )
        else:
            result = tag_audit()
    except VerificationError as error:
        print(json.dumps({"status": "failed", "error": str(error)}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("tag_ready", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())
