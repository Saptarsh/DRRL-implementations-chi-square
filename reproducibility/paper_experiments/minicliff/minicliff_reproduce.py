#!/usr/bin/env python3
"""Authenticate and reproduce the frozen paper-facing MiniCliff studies.

Canonical results are inputs only.  Every command authenticates them before use,
records their inventory again afterward, and refuses output paths that overlap the
historical MiniCliff result tree or this capsule.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageChops


# Loading the frozen runner must never dirty the source snapshot with pyc files.
sys.dont_write_bytecode = True


CAPSULE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CAPSULE_ROOT.parents[2]
SNAPSHOT_ROOT = CAPSULE_ROOT / "frozen_snapshot"
SPEC_PATH = CAPSULE_ROOT / "frozen_spec.json"
HISTORICAL_ROOT = REPO_ROOT / "paper_variational_chi2_gridworld"
REPRODUCTION_MANIFEST = "reproduction_manifest.json"
STRICT_REGENERATION_SCHEMA = "minicliff.regenerated_from_authenticated_raw.v1"
PORTABLE_REGENERATION_SCHEMA = (
    "minicliff.regenerated_from_authenticated_raw.portable.v1"
)
TRAINING_SCHEMA = "minicliff.fresh_training_reproduction.v1"
REGENERATION_SCHEMAS = frozenset(
    {STRICT_REGENERATION_SCHEMA, PORTABLE_REGENERATION_SCHEMA}
)


class ReproducibilityError(RuntimeError):
    """Fail-closed validation error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReproducibilityError(f"Cannot load JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReproducibilityError(f"Expected a JSON object in {path}.")
    return payload


def load_spec() -> dict[str, Any]:
    spec = load_json(SPEC_PATH)
    if spec.get("schema_version") != "minicliff.paper_reproducibility.v1":
        raise ReproducibilityError("Unsupported or missing frozen specification schema.")
    if spec.get("study_id") != "variational_chi2_minicliff_4x6":
        raise ReproducibilityError("Unexpected frozen study identity.")
    return spec


def inventory(root: Path, *, prefix: str | None = None) -> dict[str, dict[str, Any]]:
    if root.is_symlink() or not root.is_dir():
        raise ReproducibilityError(f"Expected a real directory, not a link: {root}")
    entries: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ReproducibilityError(f"Symlinks are forbidden in authenticated trees: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ReproducibilityError(f"Non-regular entry in authenticated tree: {path}")
        key = f"{prefix}/{relative}" if prefix else relative
        entries[key] = {"sha256": sha256_file(path), "size": path.stat().st_size}
    return entries


def subset_inventory(
    entries: Mapping[str, Mapping[str, Any]], prefix: str
) -> dict[str, dict[str, Any]]:
    return {key: dict(value) for key, value in entries.items() if key.startswith(prefix)}


def verify_snapshot(spec: Mapping[str, Any], *, require_live_match: bool = True) -> None:
    expected = spec.get("source_snapshot")
    correspondence = spec.get("live_source_correspondence")
    if not isinstance(expected, dict) or not isinstance(correspondence, dict):
        raise ReproducibilityError("Frozen source closure is malformed.")
    actual_files = {
        path.relative_to(SNAPSHOT_ROOT).as_posix()
        for path in SNAPSHOT_ROOT.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    if actual_files != set(expected):
        raise ReproducibilityError(
            "Frozen source inventory differs: "
            f"missing={sorted(set(expected) - actual_files)}, "
            f"unexpected={sorted(actual_files - set(expected))}"
        )
    for relative, expected_hash in expected.items():
        path = SNAPSHOT_ROOT / relative
        if path.is_symlink() or not path.is_file():
            raise ReproducibilityError(f"Missing regular frozen source: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ReproducibilityError(
                f"Frozen source hash mismatch for {relative}: {actual_hash} != {expected_hash}"
            )
        if require_live_match:
            live_relative = correspondence.get(relative)
            if not isinstance(live_relative, str):
                raise ReproducibilityError(f"No live-source mapping for {relative}.")
            live_path = REPO_ROOT / live_relative
            if live_path.is_symlink() or not live_path.is_file():
                raise ReproducibilityError(f"Missing regular live source: {live_path}")
            if sha256_file(live_path) != expected_hash:
                raise ReproducibilityError(
                    f"Live source drift for {live_relative}; use the frozen snapshot for this study."
                )


def study_root(spec: Mapping[str, Any], label: str) -> Path:
    relative = spec["repository_roots"][label]
    root = (REPO_ROOT / relative).resolve()
    expected = (REPO_ROOT / relative).absolute()
    if root != expected:
        raise ReproducibilityError(f"Canonical {label} root resolves through a link: {expected}")
    return root


def verify_inventory(
    root: Path, expected: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    entries = inventory(root)
    count = len(entries)
    total_bytes = sum(int(value["size"]) for value in entries.values())
    digest = canonical_hash(entries)
    raw = subset_inventory(entries, "raw/")
    if count != int(expected["file_count"]):
        raise ReproducibilityError(f"File-count mismatch at {root}: {count}")
    if total_bytes != int(expected["total_bytes"]):
        raise ReproducibilityError(f"Byte-count mismatch at {root}: {total_bytes}")
    if digest != expected["inventory_sha256"]:
        raise ReproducibilityError(f"Full-tree inventory hash mismatch at {root}: {digest}")
    if len(raw) != int(expected["raw_file_count"]):
        raise ReproducibilityError(f"Raw file-count mismatch at {root}: {len(raw)}")
    if canonical_hash(raw) != expected["raw_inventory_sha256"]:
        raise ReproducibilityError(f"Raw-tree inventory hash mismatch at {root}.")
    manifest_path = root / "manifest.json"
    if sha256_file(manifest_path) != expected["manifest_sha256"]:
        raise ReproducibilityError(f"Manifest hash mismatch at {manifest_path}.")
    return entries


def verify_combined_inventory(
    spec: Mapping[str, Any], inventories: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> None:
    combined: dict[str, dict[str, Any]] = {}
    for label in ("main", "ell_sensitivity"):
        for relative, value in inventories[label].items():
            combined[f"{label}/{relative}"] = dict(value)
    if len(combined) != int(spec["combined_canonical_file_count"]):
        raise ReproducibilityError("Combined canonical file count differs.")
    if sum(int(value["size"]) for value in combined.values()) != int(
        spec["combined_canonical_total_bytes"]
    ):
        raise ReproducibilityError("Combined canonical byte count differs.")
    if canonical_hash(combined) != spec["combined_inventory_sha256"]:
        raise ReproducibilityError("Combined canonical inventory hash differs.")


def expected_resolved(spec: Mapping[str, Any], label: str) -> dict[str, Any]:
    study = spec["canonical_inputs"][label]
    algorithm = spec["algorithm"]
    return {
        "ells": study["ells"],
        "seeds": study["seeds"],
        "focus_ell": study["focus_ell"],
        "environment": {
            "behavior_goal_bias": spec["environment"]["behavior_goal_bias"],
            "gamma": spec["environment"]["gamma"],
            "nominal_slip_probability": spec["environment"]["nominal_slip_probability"],
        },
        "algorithm_fixed_fields": {
            "beta0": algorithm["beta0"],
            "chi2_delta": algorithm["chi2_delta"],
            "dp_max_iterations": algorithm["dp_max_iterations"],
            "dp_tolerance": algorithm["dp_tolerance"],
            "eta_l2_radius": algorithm["eta_l2_radius"],
            "evaluation_slip_probability": algorithm["evaluation_slip_probability"],
            "h_q": algorithm["h_q"],
            "nominal_lr_exponent": algorithm["nominal_lr_exponent"],
            "nominal_lr_scale": algorithm["nominal_lr_scale"],
            "outer_blocks": algorithm["outer_blocks"],
            "perturbation_grid": algorithm["perturbation_grid"],
            "q_stage_samples": algorithm["q_stage_samples"],
            "scale_l2_radius": algorithm["scale_l2_radius"],
            "stage1_samples": algorithm["stage1_samples"],
            "stage1_step_mode": algorithm["stage1_step_mode"],
            "stage1_stepsize": algorithm["stage1_stepsize"],
            "theory_step_multiplier": algorithm["theory_step_multiplier"],
        },
        "max_parallel": study["max_parallel"],
        "ci_multiplier": 1.96,
    }


def verify_manifest_semantics(
    spec: Mapping[str, Any], label: str, manifest: Mapping[str, Any]
) -> None:
    expected = spec["canonical_inputs"][label]
    if manifest.get("schema_version") != 1 or manifest.get("status") != "complete":
        raise ReproducibilityError(f"{label} manifest is not a completed schema-1 study.")
    if manifest.get("profile") != expected["profile"]:
        raise ReproducibilityError(f"{label} profile differs from the frozen profile.")
    if manifest.get("resolved") != expected_resolved(spec, label):
        raise ReproducibilityError(f"{label} resolved configuration differs from the freeze.")
    if manifest.get("source_sha256") != {
        "environment": spec["source_snapshot"]["src/variational_tabular_envs.py"],
        "shared_exact_solver": spec["source_snapshot"]["src/train_variational_chi2_tabular.py"],
        "trainer": spec["source_snapshot"]["src/train_variational_chi2_gridworld.py"],
    }:
        raise ReproducibilityError(f"{label} scientific source map differs.")
    if manifest.get("runner_sha256") != spec["source_snapshot"][
        "scripts/run_variational_chi2_gridworld_paper.py"
    ]:
        raise ReproducibilityError(f"{label} runner hash differs.")
    if manifest.get("failures") != []:
        raise ReproducibilityError(f"{label} records failures.")
    runs = manifest.get("runs")
    if not isinstance(runs, list) or len(runs) != len(expected["ells"]) * len(
        expected["seeds"]
    ):
        raise ReproducibilityError(f"{label} run panel is incomplete.")


def load_frozen_runner() -> Any:
    src = str((SNAPSHOT_ROOT / "src").resolve())
    scripts = str((SNAPSHOT_ROOT / "scripts").resolve())
    for path in (scripts, src):
        if path not in sys.path:
            sys.path.insert(0, path)
    for module_name in (
        "variational_tabular_envs",
        "train_variational_chi2_tabular",
        "train_variational_chi2_gridworld",
        "run_variational_chi2_gridworld_paper",
    ):
        loaded = sys.modules.get(module_name)
        if loaded is not None:
            location = Path(getattr(loaded, "__file__", "")).resolve()
            if SNAPSHOT_ROOT.resolve() not in location.parents:
                raise ReproducibilityError(
                    f"Module {module_name} was already loaded from non-frozen path {location}."
                )
    runner = importlib.import_module("run_variational_chi2_gridworld_paper")
    location = Path(runner.__file__).resolve()
    if location != (SNAPSHOT_ROOT / "scripts/run_variational_chi2_gridworld_paper.py").resolve():
        raise ReproducibilityError(f"Loaded runner from unexpected path: {location}")
    return runner


def build_specs(runner: Any, spec: Mapping[str, Any], label: str, root: Path) -> list[Any]:
    resolved = expected_resolved(spec, label)
    environment = runner.MiniCliffConfig(**resolved["environment"])
    fields = dict(resolved["algorithm_fixed_fields"])
    fields["perturbation_grid"] = tuple(fields["perturbation_grid"])
    return runner.build_specs(root, resolved["ells"], resolved["seeds"], environment, fields)


def numeric_rows_equal(
    expected: Sequence[Mapping[str, Any]], actual_path: Path
) -> bool:
    with actual_path.open(newline="", encoding="utf-8") as handle:
        actual = list(csv.DictReader(handle))
    if len(expected) != len(actual):
        return False
    for expected_row, actual_row in zip(expected, actual):
        if set(expected_row) != set(actual_row):
            return False
        for key, value in expected_row.items():
            try:
                actual_value = float(actual_row[key])
                expected_value = float(value)
            except (TypeError, ValueError):
                return False
            if math.isnan(expected_value):
                if not math.isnan(actual_value):
                    return False
            elif actual_value != expected_value:
                return False
    return True


def arrays_equal(path_a: Path, path_b: Path) -> bool:
    with np.load(path_a, allow_pickle=False) as a, np.load(path_b, allow_pickle=False) as b:
        if set(a.files) != set(b.files):
            return False
        return all(
            a[name].dtype == b[name].dtype
            and a[name].shape == b[name].shape
            and np.array_equal(a[name], b[name], equal_nan=True)
            for name in a.files
        )


def png_pixels_equal(path_a: Path, path_b: Path) -> bool:
    """Compare decoded pixels rather than format/container metadata."""

    with Image.open(path_a) as first, Image.open(path_b) as second:
        first_rgba = first.convert("RGBA")
        second_rgba = second.convert("RGBA")
        return first_rgba.size == second_rgba.size and ImageChops.difference(
            first_rgba, second_rgba
        ).getbbox() is None


def verify_raw_and_aggregates(
    runner: Any, spec: Mapping[str, Any], label: str, root: Path
) -> dict[str, Any]:
    source_hashes = runner.expected_source_hashes()
    specs = build_specs(runner, spec, label, root)
    for run_spec in specs:
        runner.validate_saved_run(run_spec, source_hashes)
    metrics, perturbations, array_runs = runner.load_raw(specs, source_hashes)
    ci = float(expected_resolved(spec, label)["ci_multiplier"])
    learning = runner.aggregate_learning(metrics, ci)
    perturbation_summary = runner.aggregate_perturbations(perturbations, ci)
    ell_summary = runner.aggregate_ell_summary(metrics, ci)
    checks = {
        "learning_curves_exact": numeric_rows_equal(
            learning, root / "aggregated/learning_curves.csv"
        ),
        "perturbation_summary_exact": numeric_rows_equal(
            perturbation_summary, root / "aggregated/perturbation_summary.csv"
        ),
        "ell_summary_exact": numeric_rows_equal(
            ell_summary, root / "aggregated/ell_summary.csv"
        ),
    }
    modal = runner.build_modal_arrays(array_runs, float(spec["canonical_inputs"][label]["focus_ell"]))
    modal_path = root / "aggregated/modal_policies_focus_ell.npz"
    with np.load(modal_path, allow_pickle=False) as archive:
        checks["modal_schema_exact"] = set(archive.files) == {"focus_ell", *modal}
        checks["modal_arrays_exact"] = checks["modal_schema_exact"] and all(
            np.array_equal(archive[name], value, equal_nan=True)
            for name, value in modal.items()
        )
        checks["modal_focus_ell_exact"] = bool(
            float(archive["focus_ell"])
            == float(spec["canonical_inputs"][label]["focus_ell"])
        )
    if not all(checks.values()):
        raise ReproducibilityError(f"{label} aggregate reconstruction failed: {checks}")
    return {
        "runs_validated": len(specs),
        "learning_rows": len(learning),
        "perturbation_rows": len(perturbation_summary),
        "ell_rows": len(ell_summary),
        "checks": checks,
    }


def verify_cross_study_reuse(spec: Mapping[str, Any]) -> None:
    main = study_root(spec, "main") / "raw/ell_0p1"
    sweep = study_root(spec, "ell_sensitivity") / "raw/ell_0p1"
    for seed in range(1, 21):
        left = main / f"seed_{seed}"
        right = sweep / f"seed_{seed}"
        for name in ("metadata.json", "metrics.csv", "perturbation_metrics.csv"):
            if left.joinpath(name).read_bytes() != right.joinpath(name).read_bytes():
                raise ReproducibilityError(
                    f"Shared ell=0.1 evidence differs for seed {seed}: {name}"
                )
        if left.joinpath("arrays.npz").read_bytes() != right.joinpath("arrays.npz").read_bytes():
            raise ReproducibilityError(f"Shared ell=0.1 arrays differ for seed {seed}.")


def runtime_environment() -> dict[str, Any]:
    pip_freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    freeze_hash = (
        hashlib.sha256(pip_freeze.stdout).hexdigest()
        if pip_freeze.returncode == 0
        else None
    )
    blas = np.__config__.CONFIG.get("Build Dependencies", {}).get("blas", {})
    result: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "numpy_bit_generator": type(np.random.default_rng(0).bit_generator).__name__,
        "numpy_blas_name": blas.get("name"),
        "pip_freeze_sha256": freeze_hash,
    }
    for package in ("matplotlib",):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def frozen_runtime_requirements(spec: Mapping[str, Any]) -> dict[str, Any]:
    expected = spec["audit_environment"]
    return {
        "python": expected["python"],
        "platform": expected["platform"],
        "numpy": expected["numpy"],
        "matplotlib": expected["matplotlib"],
        "numpy_bit_generator": spec["rng"]["expected_bit_generator_on_frozen_dependency"],
        "numpy_blas_name": "accelerate",
        "pip_freeze_sha256": spec["source_snapshot"]["requirements-freeze.txt"],
    }


def frozen_runtime_mismatches(
    spec: Mapping[str, Any], actual: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Return every recorded-runtime field that differs from ``actual``."""

    return {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in frozen_runtime_requirements(spec).items()
        if actual.get(key) != value
    }


def require_frozen_runtime(spec: Mapping[str, Any]) -> dict[str, Any]:
    actual = runtime_environment()
    mismatches = frozen_runtime_mismatches(spec, actual)
    if mismatches:
        raise ReproducibilityError(
            f"Runtime differs from the frozen audit environment: {mismatches}"
        )
    return actual


def authenticate(*, require_live_match: bool = True) -> dict[str, Any]:
    spec = load_spec()
    verify_snapshot(spec, require_live_match=require_live_match)
    roots = {label: study_root(spec, label) for label in ("main", "ell_sensitivity")}
    inventories: dict[str, dict[str, dict[str, Any]]] = {}
    manifests: dict[str, dict[str, Any]] = {}
    for label, root in roots.items():
        inventories[label] = verify_inventory(root, spec["canonical_inputs"][label])
        manifests[label] = load_json(root / "manifest.json")
        verify_manifest_semantics(spec, label, manifests[label])
    verify_combined_inventory(spec, inventories)
    runner = load_frozen_runner()
    reconstruction = {
        label: verify_raw_and_aggregates(runner, spec, label, roots[label])
        for label in ("main", "ell_sensitivity")
    }
    verify_cross_study_reuse(spec)
    final_inventories = {
        label: verify_inventory(roots[label], spec["canonical_inputs"][label])
        for label in roots
    }
    if inventories != final_inventories:
        raise ReproducibilityError("Canonical evidence changed during authentication.")
    return {
        "schema_version": "minicliff.authentication_report.v1",
        "authenticated": True,
        "spec_sha256": sha256_file(SPEC_PATH),
        "workflow_sha256": sha256_file(Path(__file__).resolve()),
        "canonical_inputs": {
            label: {
                "root": str(roots[label]),
                "manifest_sha256": spec["canonical_inputs"][label]["manifest_sha256"],
                "inventory_sha256": spec["canonical_inputs"][label]["inventory_sha256"],
                **reconstruction[label],
            }
            for label in roots
        },
        "combined_inventory_sha256": spec["combined_inventory_sha256"],
        "cross_study_ell_0p1_seeds_1_through_20_exact": True,
        "source_snapshot_exact": True,
        "live_source_match": require_live_match,
        "runtime_environment": runtime_environment(),
        "historical_environment_caveat": spec["historical_execution_environment"],
    }


def path_overlaps(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def resolve_new_destination(path: Path) -> Path:
    requested = path.expanduser().absolute()
    if requested.exists() or requested.is_symlink():
        raise ReproducibilityError(f"Output must not already exist: {requested}")
    destination = requested.resolve(strict=False)
    protected = (HISTORICAL_ROOT.resolve(), CAPSULE_ROOT.resolve(), REPO_ROOT.resolve())
    # A destination may be a child of the repository, but never an ancestor of it;
    # the two narrower protected roots may not overlap in either direction.
    if destination == REPO_ROOT.resolve() or destination in REPO_ROOT.resolve().parents:
        raise ReproducibilityError("Output cannot be the repository or its ancestor.")
    for root in protected[:2]:
        if path_overlaps(destination, root):
            raise ReproducibilityError(f"Output overlaps protected path {root}: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def runner_command(spec: Mapping[str, Any], label: str, output_root: Path, *, reuse: bool) -> list[str]:
    study = spec["canonical_inputs"][label]
    command = [
        sys.executable,
        str(SNAPSHOT_ROOT / "scripts/run_variational_chi2_gridworld_paper.py"),
        "--profile",
        "full",
        "--ells",
        ",".join(format(float(value), ".12g") for value in study["ells"]),
        "--focus-ell",
        format(float(study["focus_ell"]), ".12g"),
        "--n-seeds",
        str(len(study["seeds"])),
        "--base-seed",
        str(study["seeds"][0]),
        "--output-root",
        str(output_root),
        "--max-parallel",
        str(study["max_parallel"]),
        "--fig-dpi",
        "200",
        "--ci-multiplier",
        "1.96",
    ]
    if reuse:
        command.append("--skip-existing")
    return command


def paper_derivative_command(
    main_root: Path, sweep_root: Path, output_dir: Path
) -> list[str]:
    """Build the paper composite from explicit, already validated study roots."""

    return [
        sys.executable,
        str(SNAPSHOT_ROOT / "scripts/plot_variational_chi2_gridworld_tac.py"),
        "--main-root",
        str(main_root),
        "--sweep-root",
        str(sweep_root),
        "--output-dir",
        str(output_dir),
        "--stem",
        "tabular_tac_composite",
        "--dpi",
        "400",
    ]


def run_checked(command: Sequence[str], *, mpl_config: Path) -> str:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["MPLCONFIGDIR"] = str(mpl_config)
    completed = subprocess.run(
        list(command),
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode != 0:
        raise ReproducibilityError(
            f"Command failed ({completed.returncode}): {' '.join(command)}\n{completed.stdout}"
        )
    return completed.stdout


def fsync_tree(root: Path) -> None:
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
    directories = sorted(
        (item for item in root.rglob("*") if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    )
    for directory in [*directories, root]:
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def compare_regenerated_aggregates(candidate: Path, canonical: Path) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for name in ("learning_curves.csv", "perturbation_summary.csv", "ell_summary.csv"):
        checks[name] = candidate.joinpath("aggregated", name).read_bytes() == canonical.joinpath(
            "aggregated", name
        ).read_bytes()
    checks["modal_policies_values"] = arrays_equal(
        candidate / "aggregated/modal_policies_focus_ell.npz",
        canonical / "aggregated/modal_policies_focus_ell.npz",
    )
    return checks


def payload_inventory(root: Path) -> dict[str, dict[str, Any]]:
    values = inventory(root)
    values.pop(REPRODUCTION_MANIFEST, None)
    return values


def publish_staged(stage: Path, destination: Path) -> None:
    fsync_tree(stage)
    if destination.exists() or destination.is_symlink():
        raise ReproducibilityError(
            f"Output appeared before atomic publication; refusing overwrite: {destination}"
        )
    os.replace(stage, destination)
    descriptor = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def regenerate(output: Path, *, allow_runtime_drift: bool = False) -> dict[str, Any]:
    authentication = authenticate()
    spec = load_spec()
    actual_runtime = runtime_environment()
    runtime_mismatches = frozen_runtime_mismatches(spec, actual_runtime)
    if runtime_mismatches and not allow_runtime_drift:
        raise ReproducibilityError(
            "Runtime differs from the frozen audit environment: "
            f"{runtime_mismatches}"
        )
    output_schema = (
        PORTABLE_REGENERATION_SCHEMA
        if runtime_mismatches
        else STRICT_REGENERATION_SCHEMA
    )
    destination = resolve_new_destination(output)
    stage = destination.with_name(f".{destination.name}.stage-{uuid.uuid4().hex}")
    if stage.exists():
        raise ReproducibilityError(f"Unexpected staging collision: {stage}")
    before = authentication["combined_inventory_sha256"]
    stage.mkdir()
    commands: list[list[str]] = []
    outputs: list[str] = []
    try:
        for label in ("main", "ell_sensitivity"):
            source_raw = study_root(spec, label) / "raw"
            candidate_root = stage / label
            shutil.copytree(source_raw, candidate_root / "raw", copy_function=shutil.copy2)
        with tempfile.TemporaryDirectory(prefix="minicliff-mpl-") as mpl:
            mpl_path = Path(mpl)
            for label in ("main", "ell_sensitivity"):
                command = runner_command(spec, label, stage / label, reuse=True)
                commands.append(command)
                outputs.append(run_checked(command, mpl_config=mpl_path))
            plot_command = paper_derivative_command(
                stage / "main",
                stage / "ell_sensitivity",
                stage / "paper_derivative",
            )
            commands.append(plot_command)
            outputs.append(run_checked(plot_command, mpl_config=mpl_path))
        comparisons = {
            label: compare_regenerated_aggregates(
                stage / label, study_root(spec, label)
            )
            for label in ("main", "ell_sensitivity")
        }
        if not all(all(values.values()) for values in comparisons.values()):
            raise ReproducibilityError(f"Regenerated aggregate mismatch: {comparisons}")
        derivative_png = stage / "paper_derivative/tabular_tac_composite.png"
        canonical_png = study_root(spec, "main") / "figures/tabular_tac_composite.png"
        derivative_pixels_identical = png_pixels_equal(derivative_png, canonical_png)
        if not derivative_pixels_identical and not runtime_mismatches:
            raise ReproducibilityError(
                "Regenerated MiniCliff composite PNG differs from the canonical pixels."
            )
        raw_checks = {}
        for label in ("main", "ell_sensitivity"):
            candidate_raw = subset_inventory(inventory(stage / label), "raw/")
            raw_checks[label] = canonical_hash(candidate_raw) == spec["canonical_inputs"][label][
                "raw_inventory_sha256"
            ]
        if not all(raw_checks.values()):
            raise ReproducibilityError(f"Copied raw input drift: {raw_checks}")
        manifest = {
            "schema_version": output_schema,
            "training_performed": False,
            "reporting_or_selection_authority": False,
            "source_authentication": authentication,
            "commands": commands,
            "command_stdout": outputs,
            "aggregate_matches_canonical_bytes_or_values": comparisons,
            "paper_composite_png_pixel_identical_to_canonical": (
                derivative_pixels_identical
            ),
            "paper_composite_pdf_note": (
                "PDF container bytes may differ because Matplotlib embeds creation metadata."
            ),
            "raw_inventory_matches_canonical": raw_checks,
            "payload_inventory": payload_inventory(stage),
            "runtime_environment": actual_runtime,
            "runtime_matches_frozen": not runtime_mismatches,
            "frozen_runtime_mismatches": runtime_mismatches,
        }
        manifest["payload_inventory_sha256"] = canonical_hash(manifest["payload_inventory"])
        write_json_atomic(stage / REPRODUCTION_MANIFEST, manifest)
        after_report = authenticate()
        if before != after_report["combined_inventory_sha256"]:
            raise ReproducibilityError("Canonical evidence changed during regeneration.")
        publish_staged(stage, destination)
        return load_json(destination / REPRODUCTION_MANIFEST)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def train(output: Path, study_choice: str, acknowledgement: str) -> dict[str, Any]:
    if acknowledgement != "I_UNDERSTAND_THIS_RUN_IS_EXPENSIVE":
        raise ReproducibilityError(
            "Training requires --acknowledge-expensive-training "
            "I_UNDERSTAND_THIS_RUN_IS_EXPENSIVE"
        )
    authentication = authenticate()
    spec = load_spec()
    frozen_runtime = require_frozen_runtime(spec)
    destination = resolve_new_destination(output)
    stage = destination.with_name(f".{destination.name}.stage-{uuid.uuid4().hex}")
    stage.mkdir()
    labels = (
        ("main", "ell_sensitivity") if study_choice == "all" else (study_choice,)
    )
    commands: list[list[str]] = []
    outputs: list[str] = []
    try:
        with tempfile.TemporaryDirectory(prefix="minicliff-mpl-") as mpl:
            mpl_path = Path(mpl)
            for label in labels:
                command = runner_command(spec, label, stage / label, reuse=False)
                commands.append(command)
                outputs.append(run_checked(command, mpl_config=mpl_path))
            if set(labels) == {"main", "ell_sensitivity"}:
                plot_command = paper_derivative_command(
                    stage / "main",
                    stage / "ell_sensitivity",
                    stage / "paper_derivative",
                )
                commands.append(plot_command)
                outputs.append(run_checked(plot_command, mpl_config=mpl_path))
        comparisons = {
            label: compare_regenerated_aggregates(stage / label, study_root(spec, label))
            for label in labels
        }
        manifest = {
            "schema_version": TRAINING_SCHEMA,
            "training_performed": True,
            "reporting_or_selection_authority": False,
            "study_choice": study_choice,
            "source_authentication": authentication,
            "commands": commands,
            "command_stdout": outputs,
            "aggregate_matches_historical_bytes_or_values": comparisons,
            "exact_historical_environment_was_not_archived": True,
            "payload_inventory": payload_inventory(stage),
            "runtime_environment": frozen_runtime,
        }
        manifest["payload_inventory_sha256"] = canonical_hash(manifest["payload_inventory"])
        write_json_atomic(stage / REPRODUCTION_MANIFEST, manifest)
        authenticate()
        publish_staged(stage, destination)
        return load_json(destination / REPRODUCTION_MANIFEST)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def verify_output(output: Path) -> dict[str, Any]:
    authentication = authenticate()
    root = output.resolve()
    if root.is_symlink() or not root.is_dir():
        raise ReproducibilityError(f"Expected a real reproduction directory: {root}")
    if path_overlaps(root, HISTORICAL_ROOT.resolve()):
        raise ReproducibilityError("A reproduction output may not overlap historical results.")
    manifest_path = root / REPRODUCTION_MANIFEST
    manifest = load_json(manifest_path)
    output_schema = manifest.get("schema_version")
    if output_schema not in REGENERATION_SCHEMAS | {TRAINING_SCHEMA}:
        raise ReproducibilityError("Unknown reproduction-output schema.")
    if manifest.get("reporting_or_selection_authority") is not False:
        raise ReproducibilityError("Reproduction output claims unauthorized reporting authority.")
    stored = manifest.get("payload_inventory")
    if not isinstance(stored, dict):
        raise ReproducibilityError("Reproduction manifest has no payload inventory.")
    actual = payload_inventory(root)
    if actual != stored:
        raise ReproducibilityError("Reproduction payload inventory differs from its manifest.")
    digest = canonical_hash(actual)
    if digest != manifest.get("payload_inventory_sha256"):
        raise ReproducibilityError("Reproduction payload inventory digest differs.")
    source = manifest.get("source_authentication")
    if not isinstance(source, dict):
        raise ReproducibilityError("Reproduction has no source-authentication record.")
    if source.get("spec_sha256") != sha256_file(SPEC_PATH):
        raise ReproducibilityError("Reproduction binds a different frozen specification.")
    if source.get("workflow_sha256") != sha256_file(Path(__file__).resolve()):
        raise ReproducibilityError("Reproduction binds a different workflow implementation.")
    if source.get("combined_inventory_sha256") != authentication[
        "combined_inventory_sha256"
    ]:
        raise ReproducibilityError("Reproduction binds different canonical evidence.")
    spec = load_spec()
    is_regeneration = output_schema in REGENERATION_SCHEMAS
    if is_regeneration and manifest.get("training_performed") is not False:
        raise ReproducibilityError("A regeneration output must record no training.")
    if not is_regeneration and manifest.get("training_performed") is not True:
        raise ReproducibilityError("A training output must record that training occurred.")
    recorded_runtime = manifest.get("runtime_environment")
    if not isinstance(recorded_runtime, Mapping):
        raise ReproducibilityError("Reproduction has no runtime-environment record.")
    recorded_runtime_mismatches = frozen_runtime_mismatches(spec, recorded_runtime)
    if is_regeneration:
        if manifest.get("frozen_runtime_mismatches") != recorded_runtime_mismatches:
            raise ReproducibilityError(
                "Regeneration runtime-mismatch record is internally inconsistent."
            )
        if manifest.get("runtime_matches_frozen") is not (
            not recorded_runtime_mismatches
        ):
            raise ReproducibilityError(
                "Regeneration frozen-runtime status is internally inconsistent."
            )
        if (
            output_schema == PORTABLE_REGENERATION_SCHEMA
            and not recorded_runtime_mismatches
        ):
            raise ReproducibilityError(
                "Portable regeneration schema requires a recorded runtime mismatch."
            )
        if output_schema == STRICT_REGENERATION_SCHEMA and recorded_runtime_mismatches:
            raise ReproducibilityError(
                "Strict regeneration schema cannot contain runtime drift."
            )
    runner = load_frozen_runner()
    if is_regeneration:
        labels = ("main", "ell_sensitivity")
        expected_top = {
            "main",
            "ell_sensitivity",
            "paper_derivative",
            REPRODUCTION_MANIFEST,
        }
    else:
        study_choice = manifest.get("study_choice")
        if study_choice not in {"main", "ell_sensitivity", "all"}:
            raise ReproducibilityError("Fresh-training output has an invalid study choice.")
        labels = (
            ("main", "ell_sensitivity")
            if study_choice == "all"
            else (str(study_choice),)
        )
        expected_top = {*labels, REPRODUCTION_MANIFEST}
        if study_choice == "all":
            expected_top.add("paper_derivative")
    actual_top = {path.name for path in root.iterdir()}
    if actual_top != expected_top:
        raise ReproducibilityError(
            f"Unexpected reproduction top-level inventory: {sorted(actual_top)}"
        )
    reconstructed: dict[str, Any] = {}
    canonical_comparisons: dict[str, dict[str, bool]] = {}
    expected_study_entries = {"raw", "aggregated", "figures", "manifest.json"}
    expected_aggregates = {
        "learning_curves.csv",
        "perturbation_summary.csv",
        "ell_summary.csv",
        "modal_policies_focus_ell.npz",
    }
    expected_figures = {
        "convergence_policy_agreement.png",
        "convergence_policy_agreement.pdf",
        "perturbation_performance.png",
        "perturbation_performance.pdf",
        "ell_bias_stability.png",
        "ell_bias_stability.pdf",
        "policy_maps.png",
        "policy_maps.pdf",
    }
    for label in labels:
        study = root / label
        if {path.name for path in study.iterdir()} != expected_study_entries:
            raise ReproducibilityError(f"Unexpected {label} study inventory.")
        if {path.name for path in (study / "aggregated").iterdir()} != expected_aggregates:
            raise ReproducibilityError(f"Unexpected {label} aggregate inventory.")
        if {path.name for path in (study / "figures").iterdir()} != expected_figures:
            raise ReproducibilityError(f"Unexpected {label} figure inventory.")
        verify_manifest_semantics(spec, label, load_json(study / "manifest.json"))
        reconstructed[label] = verify_raw_and_aggregates(runner, spec, label, study)
        canonical_comparisons[label] = compare_regenerated_aggregates(
            study, study_root(spec, label)
        )
        if is_regeneration:
            raw = subset_inventory(inventory(study), "raw/")
            if len(raw) != spec["canonical_inputs"][label]["raw_file_count"] or canonical_hash(
                raw
            ) != spec["canonical_inputs"][label]["raw_inventory_sha256"]:
                raise ReproducibilityError(
                    f"Regenerated {label} raw evidence differs from canonical input."
                )
    derivative_pixels_identical: bool | None = None
    derivative_matches_fresh_render: bool | None = None
    if "paper_derivative" in expected_top:
        derivative = root / "paper_derivative"
        if {path.name for path in derivative.iterdir()} != {
            "tabular_tac_composite.png",
            "tabular_tac_composite.pdf",
        }:
            raise ReproducibilityError("Unexpected paper-derivative inventory.")
        derivative_pixels_identical = png_pixels_equal(
            derivative / "tabular_tac_composite.png",
            study_root(spec, "main") / "figures/tabular_tac_composite.png",
        )
        if is_regeneration and manifest.get(
            "paper_composite_png_pixel_identical_to_canonical"
        ) is not derivative_pixels_identical:
            raise ReproducibilityError(
                "Stored paper-derivative PNG comparison differs from verification."
            )
        if output_schema == PORTABLE_REGENERATION_SCHEMA:
            with tempfile.TemporaryDirectory(
                prefix="minicliff-verify-render-"
            ) as temporary:
                temporary_root = Path(temporary)
                mpl_config = temporary_root / "matplotlib-cache"
                mpl_config.mkdir()
                fresh_derivative = temporary_root / "paper_derivative"
                run_checked(
                    paper_derivative_command(
                        root / "main",
                        root / "ell_sensitivity",
                        fresh_derivative,
                    ),
                    mpl_config=mpl_config,
                )
                derivative_matches_fresh_render = png_pixels_equal(
                    derivative / "tabular_tac_composite.png",
                    fresh_derivative / "tabular_tac_composite.png",
                )
            if not derivative_matches_fresh_render:
                raise ReproducibilityError(
                    "Portable paper-derivative PNG differs from a fresh render "
                    "of the verified output aggregates."
                )
        if (
            not derivative_pixels_identical
            and output_schema != PORTABLE_REGENERATION_SCHEMA
        ):
            raise ReproducibilityError("Paper-derivative PNG pixels differ from canonical.")
    return {
        "schema_version": "minicliff.reproduction_output_verification.v1",
        "verified": True,
        "output_root": str(root),
        "output_schema": output_schema,
        "training_performed": manifest["training_performed"],
        "reporting_or_selection_authority": False,
        "payload_file_count": len(actual),
        "payload_inventory_sha256": digest,
        "canonical_combined_inventory_sha256": authentication[
            "combined_inventory_sha256"
        ],
        "raw_and_aggregate_reconstruction": reconstructed,
        "aggregate_matches_historical_bytes_or_values": canonical_comparisons,
        "runtime_matches_frozen": not recorded_runtime_mismatches,
        "frozen_runtime_mismatches": recorded_runtime_mismatches,
        "paper_composite_png_pixel_identical_to_canonical": (
            derivative_pixels_identical
        ),
        "paper_composite_png_matches_fresh_render": (
            derivative_matches_fresh_render
        ),
    }


def tag_audit() -> dict[str, Any]:
    spec = load_spec()
    verify_snapshot(spec)
    files = sorted(
        path
        for path in CAPSULE_ROOT.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )
    untracked: list[str] = []
    tracked: list[str] = []
    for path in files:
        relative = path.relative_to(REPO_ROOT).as_posix()
        completed = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "ls-files", "--error-unmatch", relative],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if completed.returncode != 0:
            untracked.append(relative)
        else:
            tracked.append(relative)
    dirty: list[str] = []
    if tracked:
        completed = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--", *tracked],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        if completed.returncode != 0:
            raise ReproducibilityError("Could not audit capsule cleanliness against Git.")
        dirty = sorted(
            line[3:] for line in completed.stdout.splitlines() if len(line) >= 4
        )
    return {
        "schema_version": "minicliff.tag_audit.v1",
        "tag_ready": not untracked and not dirty,
        "capsule_files_checked": len(files),
        "untracked_capsule_files": untracked,
        "dirty_tracked_capsule_files": dirty,
        "companion_data_required": True,
        "companion_data_combined_inventory_sha256": spec["combined_inventory_sha256"],
        "note": (
            "A source tag contains the workflow and frozen code. The ignored 12 MB canonical "
            "result trees must be distributed as a companion archive whose extracted inventory "
            "matches the frozen digest."
        ),
    }


def print_report(report: Mapping[str, Any]) -> None:
    print(json.dumps(report, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify_parser = subparsers.add_parser("verify", help="authenticate canonical evidence")
    verify_parser.add_argument(
        "--allow-live-source-drift",
        action="store_true",
        help="authenticate against the frozen snapshot even if corresponding live files drifted",
    )
    rebuild = subparsers.add_parser(
        "regenerate", help="copy authenticated raw inputs and rebuild aggregates/plots"
    )
    rebuild.add_argument("--output-root", type=Path, required=True)
    rebuild.add_argument(
        "--allow-runtime-drift",
        action="store_true",
        help=(
            "permit no-training regeneration in a compatible non-frozen runtime; "
            "aggregate equality remains required and figure pixel equality is reported"
        ),
    )
    training = subparsers.add_parser(
        "train", help="run the frozen tuned training protocol in a new output root"
    )
    training.add_argument("--output-root", type=Path, required=True)
    training.add_argument(
        "--study", choices=("main", "ell_sensitivity", "all"), default="main"
    )
    training.add_argument("--acknowledge-expensive-training", required=True)
    output_verification = subparsers.add_parser(
        "verify-output", help="authenticate an already published reproduction directory"
    )
    output_verification.add_argument("--output-root", type=Path, required=True)
    subparsers.add_parser("tag-audit", help="check whether every capsule file is tracked")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        if arguments.command == "verify":
            report = authenticate(require_live_match=not arguments.allow_live_source_drift)
        elif arguments.command == "regenerate":
            report = regenerate(
                arguments.output_root,
                allow_runtime_drift=arguments.allow_runtime_drift,
            )
        elif arguments.command == "train":
            report = train(
                arguments.output_root,
                arguments.study,
                arguments.acknowledge_expensive_training,
            )
        elif arguments.command == "verify-output":
            report = verify_output(arguments.output_root)
        else:
            report = tag_audit()
        print_report(report)
        return 0 if report.get("tag_ready", True) else 2
    except ReproducibilityError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
