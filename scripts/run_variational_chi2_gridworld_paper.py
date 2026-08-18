#!/usr/bin/env python3
"""Run, aggregate, and plot the variational chi-square MiniCliff study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from train_variational_chi2_gridworld import (  # noqa: E402
    MiniCliffAlgorithmConfig,
    run_experiment,
    save_run,
    validate_algorithm_config,
)
from variational_tabular_envs import (  # noqa: E402
    MiniCliffConfig,
    validate_minicliff_config,
)


@dataclass(frozen=True)
class Profile:
    n_seeds: int
    ells: Tuple[float, ...]
    outer_blocks: int
    stage1_samples: int
    q_stage_samples: int


PROFILES: Dict[str, Profile] = {
    "smoke": Profile(
        n_seeds=1,
        ells=(0.03, 0.30),
        outer_blocks=2,
        stage1_samples=1_000,
        q_stage_samples=1_000,
    ),
    "quick": Profile(
        n_seeds=3,
        ells=(0.01, 0.10, 0.50),
        outer_blocks=20,
        stage1_samples=25_000,
        q_stage_samples=25_000,
    ),
    "full": Profile(
        n_seeds=20,
        ells=(0.003, 0.01, 0.03, 0.10, 0.30, 1.0),
        outer_blocks=60,
        stage1_samples=50_000,
        q_stage_samples=50_000,
    ),
}


SOURCE_FILES = {
    "trainer": SRC_ROOT / "train_variational_chi2_gridworld.py",
    "environment": SRC_ROOT / "variational_tabular_envs.py",
    "shared_exact_solver": SRC_ROOT / "train_variational_chi2_tabular.py",
}

NPZ_SCHEMA: Dict[str, Tuple[Tuple[int, ...], str]] = {
    "robust_q": ((24, 4), "float"),
    "nominal_q": ((24, 4), "float"),
    "robust_reference_q": ((24, 4), "float"),
    "floor_reference_q": ((24, 4), "float"),
    "nominal_reference_q": ((24, 4), "float"),
    "robust_policy": ((24,), "integer"),
    "nominal_policy": ((24,), "integer"),
    "robust_oracle_policy": ((24,), "integer"),
    "nominal_oracle_policy": ((24,), "integer"),
    "state_action_counts": ((24, 4), "integer"),
    "last_eta_bar": ((24, 4), "float"),
    "last_u_bar": ((24, 4), "float"),
    "floor_oracle_eta": ((24, 4), "float"),
    "floor_oracle_u": ((24, 4), "float"),
    "behavior_policy": ((24, 4), "float"),
    "behavior_stationary_state": ((24,), "float"),
    "behavior_stationary_state_action": ((24, 4), "float"),
    "state_coordinates": ((24, 2), "integer"),
    "decision_state_mask": ((24,), "bool"),
    "oracle_separating_state_mask": ((24,), "bool"),
}

REQUIRED_METRIC_COLUMNS = {
    "outer_block",
    "transitions",
    "robust_q_sup_error",
    "floor_q_sup_error",
    "floor_reference_bias",
    "floor_bias_bound",
    "robust_policy_perturbed_gap",
    "nominal_policy_perturbed_gap",
    "robust_policy_oracle_agreement",
    "robust_policy_separating_state_agreement",
    "stage1_gradient_rms",
    "stage1_gradient_max",
    "stage1_ratio_x_over_u_p95",
    "eta_projection_fraction",
    "scale_projection_fraction",
    "scale_floor_fraction",
}

REQUIRED_PERTURBATION_COLUMNS = {
    "slip_probability",
    "max_row_chi2_distance",
    "support_preserved",
    "inside_chi2_radius",
    "optimal_return",
    "robust_policy_return",
    "nominal_policy_return",
    "oracle_robust_policy_return",
    "oracle_nominal_policy_return",
    "robust_policy_gap",
    "nominal_policy_gap",
    "robust_policy_advantage",
    "robust_policy_discounted_cliff_occupancy",
    "nominal_policy_discounted_cliff_occupancy",
}


@dataclass(frozen=True)
class RunSpec:
    ell: float
    seed: int
    environment: MiniCliffConfig
    algorithm: MiniCliffAlgorithmConfig
    output_dir: Path


def parse_float_tuple(text: str) -> Tuple[float, ...]:
    try:
        values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated floating-point values.") from exc
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated value.")
    return values


def float_slug(value: float) -> str:
    return format(value, ".12g").replace("-", "m").replace("+", "").replace(".", "p")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_source_hashes() -> Dict[str, str]:
    return {name: sha256_file(path) for name, path in SOURCE_FILES.items()}


def algorithm_metadata(config: MiniCliffAlgorithmConfig) -> Dict[str, object]:
    payload: Dict[str, object] = asdict(config)
    payload["perturbation_grid"] = list(config.perturbation_grid)
    return payload


def config_mismatches(
    expected: Mapping[str, object], actual: Mapping[str, object], prefix: str
) -> List[str]:
    mismatches: List[str] = []
    for key, expected_value in expected.items():
        label = f"{prefix}.{key}"
        if key not in actual:
            mismatches.append(f"{label}: missing")
            continue
        actual_value = actual[key]
        if isinstance(expected_value, dict) and isinstance(actual_value, dict):
            mismatches.extend(config_mismatches(expected_value, actual_value, label))
        elif actual_value != expected_value:
            mismatches.append(f"{label}: expected {expected_value!r}, found {actual_value!r}")
    return mismatches


def read_numeric_csv(path: Path) -> List[Dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        rows: List[Dict[str, float]] = []
        for line_number, row in enumerate(reader, start=2):
            try:
                numeric_row = {key: float(value) for key, value in row.items()}
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Non-numeric value in {path}:{line_number}") from exc
            if not all(math.isfinite(value) for value in numeric_row.values()):
                raise ValueError(f"Non-finite value in {path}:{line_number}")
            rows.append(numeric_row)
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def validate_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        found_keys = set(archive.files)
        expected_keys = set(NPZ_SCHEMA)
        if found_keys != expected_keys:
            raise ValueError(
                f"NPZ schema mismatch in {path}; missing={sorted(expected_keys - found_keys)}, "
                f"unexpected={sorted(found_keys - expected_keys)}"
            )
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}

    for name, (shape, kind) in NPZ_SCHEMA.items():
        array = arrays[name]
        if array.shape != shape:
            raise ValueError(f"{path}:{name} has shape {array.shape}, expected {shape}.")
        if kind == "float" and not np.issubdtype(array.dtype, np.floating):
            raise ValueError(f"{path}:{name} must have floating dtype, found {array.dtype}.")
        if kind == "integer" and not np.issubdtype(array.dtype, np.integer):
            raise ValueError(f"{path}:{name} must have integer dtype, found {array.dtype}.")
        if kind == "bool" and not np.issubdtype(array.dtype, np.bool_):
            raise ValueError(f"{path}:{name} must have bool dtype, found {array.dtype}.")
        if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
            raise ValueError(f"{path}:{name} contains non-finite values.")

    for policy_name in (
        "robust_policy",
        "nominal_policy",
        "robust_oracle_policy",
        "nominal_oracle_policy",
    ):
        policy = arrays[policy_name]
        if np.any((policy < 0) | (policy >= 4)):
            raise ValueError(f"{path}:{policy_name} contains an invalid action.")
    if int(np.count_nonzero(arrays["decision_state_mask"])) != 19:
        raise ValueError(f"{path}: decision_state_mask must contain exactly 19 states.")
    if int(np.count_nonzero(arrays["oracle_separating_state_mask"])) < 2:
        raise ValueError(f"{path}: expected multiple oracle-separating decision states.")
    if not np.array_equal(
        arrays["oracle_separating_state_mask"] & arrays["decision_state_mask"],
        arrays["oracle_separating_state_mask"],
    ):
        raise ValueError(f"{path}: separating states must be decision states.")
    return arrays


def validate_saved_run(spec: RunSpec, source_hashes: Mapping[str, str]) -> Dict[str, object]:
    paths = {
        "metadata": spec.output_dir / "metadata.json",
        "metrics": spec.output_dir / "metrics.csv",
        "perturbations": spec.output_dir / "perturbation_metrics.csv",
        "arrays": spec.output_dir / "arrays.npz",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ValueError(
            f"Cannot reuse incomplete run {spec.output_dir}; missing: {', '.join(missing)}"
        )
    with paths["metadata"].open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    expected = {
        "source_sha256": dict(source_hashes),
        "environment": asdict(spec.environment),
        "algorithm": algorithm_metadata(spec.algorithm),
        "n_states": 24,
        "n_actions": 4,
        "n_decision_states": 19,
    }
    mismatches = config_mismatches(expected, metadata, "metadata")
    if mismatches:
        detail = "\n  - ".join(mismatches[:16])
        raise ValueError(
            f"Existing run does not match current sources/configuration: {spec.output_dir}\n"
            f"  - {detail}"
        )
    artifact_hashes = metadata.get("artifact_sha256")
    authenticated_names = ("metrics.csv", "perturbation_metrics.csv", "arrays.npz")
    if not isinstance(artifact_hashes, dict) or set(artifact_hashes) != set(
        authenticated_names
    ):
        raise ValueError(f"Missing or invalid artifact hashes in {paths['metadata']}.")
    for name in authenticated_names:
        actual_hash = sha256_file(spec.output_dir / name)
        if artifact_hashes[name] != actual_hash:
            raise ValueError(
                f"Artifact hash mismatch for {spec.output_dir / name}; "
                "the saved run may be incomplete or mixed across executions."
            )
    metrics = read_numeric_csv(paths["metrics"])
    perturbations = read_numeric_csv(paths["perturbations"])
    missing_metrics = REQUIRED_METRIC_COLUMNS - metrics[0].keys()
    missing_perturbations = REQUIRED_PERTURBATION_COLUMNS - perturbations[0].keys()
    if missing_metrics or missing_perturbations:
        raise ValueError(
            f"CSV schema mismatch in {spec.output_dir}; "
            f"missing metrics={sorted(missing_metrics)}, "
            f"missing perturbations={sorted(missing_perturbations)}"
        )
    if len(metrics) != spec.algorithm.outer_blocks + 1:
        raise ValueError(
            f"Expected {spec.algorithm.outer_blocks + 1} learning rows, found {len(metrics)} "
            f"in {paths['metrics']}."
        )
    if len(perturbations) != len(spec.algorithm.perturbation_grid):
        raise ValueError(
            f"Expected {len(spec.algorithm.perturbation_grid)} perturbation rows, found "
            f"{len(perturbations)} in {paths['perturbations']}."
        )
    for expected_block, row in enumerate(metrics):
        expected_transitions = expected_block * (
            spec.algorithm.stage1_samples + spec.algorithm.q_stage_samples
        )
        if row["outer_block"] != float(expected_block) or row["transitions"] != float(
            expected_transitions
        ):
            raise ValueError(
                f"Unexpected learning index in {paths['metrics']} at data row "
                f"{expected_block + 1}: block={row['outer_block']}, "
                f"transitions={row['transitions']}."
            )
    saved_slips = tuple(row["slip_probability"] for row in perturbations)
    if saved_slips != tuple(spec.algorithm.perturbation_grid):
        raise ValueError(
            f"Perturbation grid in {paths['perturbations']} does not match the configuration."
        )
    validate_npz(paths["arrays"])
    return metadata


def execute_run(spec: RunSpec) -> Dict[str, object]:
    result = run_experiment(spec.environment, spec.algorithm)
    save_run(result, spec.output_dir)
    return {
        "ell": spec.ell,
        "seed": spec.seed,
        "final_q_error": result.metrics[-1]["robust_q_sup_error"],
        "final_agreement": result.metrics[-1]["robust_policy_oracle_agreement"],
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    """Publish a JSON file with a same-directory atomic replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def summarize(values: Iterable[float], ci_multiplier: float) -> Dict[str, float]:
    array = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if array.size == 0:
        return {
            "mean": math.nan,
            "std": math.nan,
            "sem": math.nan,
            "interval_halfwidth": math.nan,
        }
    standard_deviation = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    standard_error = standard_deviation / math.sqrt(float(array.size))
    return {
        "mean": float(np.mean(array)),
        "std": standard_deviation,
        "sem": standard_error,
        "interval_halfwidth": ci_multiplier * standard_error,
    }


def add_statistics(
    target: Dict[str, object], metric: str, values: Iterable[float], ci_multiplier: float
) -> None:
    for statistic, value in summarize(values, ci_multiplier).items():
        target[f"{metric}_{statistic}"] = value


def aggregate_learning(
    rows: Sequence[Dict[str, float]], ci_multiplier: float
) -> List[Dict[str, object]]:
    groups: Dict[Tuple[float, int], List[Dict[str, float]]] = {}
    for row in rows:
        groups.setdefault((row["ell"], int(round(row["outer_block"]))), []).append(row)
    metrics = sorted(
        set.intersection(*(set(row) for row in rows))
        - {"ell", "seed", "outer_block", "transitions"}
    )
    output: List[Dict[str, object]] = []
    for (ell, block), group in sorted(groups.items()):
        transitions = np.asarray([row["transitions"] for row in group])
        if not np.allclose(transitions, transitions[0]):
            raise ValueError(f"Transition counts differ for ell={ell}, block={block}.")
        aggregate: Dict[str, object] = {
            "ell": ell,
            "outer_block": block,
            "transitions": float(transitions[0]),
            "n_seeds": len(group),
        }
        for metric in metrics:
            add_statistics(aggregate, metric, (row[metric] for row in group), ci_multiplier)
        output.append(aggregate)
    return output


def aggregate_perturbations(
    rows: Sequence[Dict[str, float]], ci_multiplier: float
) -> List[Dict[str, object]]:
    groups: Dict[Tuple[float, float], List[Dict[str, float]]] = {}
    for row in rows:
        groups.setdefault((row["ell"], row["slip_probability"]), []).append(row)
    metrics = sorted(
        set.intersection(*(set(row) for row in rows))
        - {"ell", "seed", "slip_probability"}
    )
    output: List[Dict[str, object]] = []
    for (ell, slip), group in sorted(groups.items()):
        aggregate: Dict[str, object] = {
            "ell": ell,
            "slip_probability": slip,
            "n_seeds": len(group),
        }
        for metric in metrics:
            add_statistics(aggregate, metric, (row[metric] for row in group), ci_multiplier)
        output.append(aggregate)
    return output


def aggregate_ell_summary(
    rows: Sequence[Dict[str, float]], ci_multiplier: float
) -> List[Dict[str, object]]:
    final: Dict[Tuple[float, int], Dict[str, float]] = {}
    for row in rows:
        key = (row["ell"], int(row["seed"]))
        if key not in final or row["outer_block"] > final[key]["outer_block"]:
            final[key] = row
    groups: Dict[float, List[Dict[str, float]]] = {}
    for (ell, _), row in final.items():
        groups.setdefault(ell, []).append(row)
    metrics = sorted(
        set.intersection(*(set(row) for row in final.values()))
        - {"ell", "seed", "outer_block", "transitions"}
    )
    output: List[Dict[str, object]] = []
    for ell, group in sorted(groups.items()):
        aggregate: Dict[str, object] = {
            "ell": ell,
            "outer_block": int(max(row["outer_block"] for row in group)),
            "transitions": float(max(row["transitions"] for row in group)),
            "n_seeds": len(group),
        }
        for metric in metrics:
            add_statistics(aggregate, metric, (row[metric] for row in group), ci_multiplier)
        output.append(aggregate)
    return output


def load_raw(
    specs: Sequence[RunSpec], source_hashes: Mapping[str, str]
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]], List[Dict[str, object]]]:
    metrics: List[Dict[str, float]] = []
    perturbations: List[Dict[str, float]] = []
    array_runs: List[Dict[str, object]] = []
    for spec in specs:
        validate_saved_run(spec, source_hashes)
        for row in read_numeric_csv(spec.output_dir / "metrics.csv"):
            metrics.append({"ell": spec.ell, "seed": float(spec.seed), **row})
        for row in read_numeric_csv(spec.output_dir / "perturbation_metrics.csv"):
            perturbations.append({"ell": spec.ell, "seed": float(spec.seed), **row})
        array_runs.append(
            {
                "ell": spec.ell,
                "seed": spec.seed,
                "arrays": validate_npz(spec.output_dir / "arrays.npz"),
            }
        )
    return metrics, perturbations, array_runs


def modal_policy(policies: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    stack = np.asarray(policies, dtype=np.int64)
    if stack.ndim != 2 or stack.shape[1] != 24:
        raise ValueError(f"Expected a seed-by-24 policy stack, found {stack.shape}.")
    modes = np.empty(24, dtype=np.int64)
    ties = np.zeros(24, dtype=bool)
    for state in range(24):
        counts = np.bincount(stack[:, state], minlength=4)
        winners = np.flatnonzero(counts == counts.max())
        modes[state] = int(winners[0])
        ties[state] = len(winners) > 1
    return modes, ties


def build_modal_arrays(
    array_runs: Sequence[Dict[str, object]], focus_ell: float
) -> Dict[str, np.ndarray]:
    selected = [
        run
        for run in array_runs
        if math.isclose(float(run["ell"]), focus_ell, rel_tol=1e-12, abs_tol=1e-15)
    ]
    if not selected:
        raise ValueError(f"No saved arrays found for focus ell={focus_ell:g}.")
    arrays = [run["arrays"] for run in selected]
    nominal_oracle = np.asarray(arrays[0]["nominal_oracle_policy"])
    robust_oracle = np.asarray(arrays[0]["robust_oracle_policy"])
    coordinates = np.asarray(arrays[0]["state_coordinates"])
    decision_mask = np.asarray(arrays[0]["decision_state_mask"])
    separating_mask = np.asarray(arrays[0]["oracle_separating_state_mask"])
    for archive in arrays[1:]:
        for name, reference in (
            ("nominal_oracle_policy", nominal_oracle),
            ("robust_oracle_policy", robust_oracle),
            ("state_coordinates", coordinates),
            ("decision_state_mask", decision_mask),
            ("oracle_separating_state_mask", separating_mask),
        ):
            if not np.array_equal(archive[name], reference):
                raise ValueError(f"Oracle/layout array {name} differs across seeds.")
    modal_nominal, nominal_ties = modal_policy(
        [np.asarray(archive["nominal_policy"]) for archive in arrays]
    )
    modal_robust, robust_ties = modal_policy(
        [np.asarray(archive["robust_policy"]) for archive in arrays]
    )
    return {
        "nominal_oracle_policy": nominal_oracle,
        "robust_oracle_policy": robust_oracle,
        "modal_nominal_policy": modal_nominal,
        "modal_robust_policy": modal_robust,
        "modal_nominal_tie_mask": nominal_ties,
        "modal_robust_tie_mask": robust_ties,
        "state_coordinates": coordinates,
        "decision_state_mask": decision_mask,
        "oracle_separating_state_mask": separating_mask,
    }


def save_figure(fig: object, directory: Path, stem: str, dpi: int) -> List[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / f"{stem}.png", directory / f"{stem}.pdf"]
    fig.savefig(paths[0], dpi=dpi, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    return paths


def ell_colors(ells: Sequence[float]) -> Dict[float, object]:
    import matplotlib.pyplot as plt

    ordered = sorted(ells)
    positions = np.linspace(0.08, 0.92, len(ordered))
    return {ell: plt.cm.viridis(position) for ell, position in zip(ordered, positions)}


def plot_convergence(
    rows: Sequence[Dict[str, object]], focus_ell: float, figures: Path, dpi: int
) -> List[Path]:
    import matplotlib.pyplot as plt

    ells = sorted({float(row["ell"]) for row in rows})
    colors = ell_colors(ells)
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.25))
    for ell in ells:
        selected = sorted(
            (
                row
                for row in rows
                if float(row["ell"]) == ell and float(row["transitions"]) > 0.0
            ),
            key=lambda row: float(row["transitions"]),
        )
        x = np.asarray([row["transitions"] for row in selected], dtype=float)
        label = rf"$\ell={ell:g}$"
        for axis, metric in (
            (axes[0], "robust_q_sup_error"),
            (axes[1], "robust_policy_oracle_agreement"),
        ):
            y = np.asarray([row[f"{metric}_mean"] for row in selected], dtype=float)
            ci = np.asarray(
                [row[f"{metric}_interval_halfwidth"] for row in selected], dtype=float
            )
            axis.plot(x, y, color=colors[ell], linewidth=2.0, label=label)
            axis.fill_between(x, y - ci, y + ci, color=colors[ell], alpha=0.18, linewidth=0)

    focus_rows = sorted(
        (
            row
            for row in rows
            if math.isclose(float(row["ell"]), focus_ell) and float(row["transitions"]) > 0.0
        ),
        key=lambda row: float(row["transitions"]),
    )
    x = np.asarray([row["transitions"] for row in focus_rows], dtype=float)
    for metric, label, color in (
        ("robust_policy_perturbed_gap", "Variational robust", "#1f77b4"),
        ("nominal_policy_perturbed_gap", "Nominal Q-learning", "#d62728"),
    ):
        y = np.asarray([row[f"{metric}_mean"] for row in focus_rows], dtype=float)
        ci = np.asarray(
            [row[f"{metric}_interval_halfwidth"] for row in focus_rows], dtype=float
        )
        axes[2].plot(x, y, color=color, linewidth=2.1, label=label)
        axes[2].fill_between(x, y - ci, y + ci, color=color, alpha=0.18, linewidth=0)

    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$\|\widehat Q_t-Q^*_{\chi}\|_\infty$")
    axes[0].set_title("Robust Q convergence")
    axes[0].legend(frameon=False, fontsize=8.5)
    axes[1].set_xscale("log")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].axhline(1.0, color="black", linestyle=":", linewidth=1.0)
    axes[1].set_ylabel("Agreement with robust oracle")
    axes[1].set_title("Policy agreement (19 decision states)")
    axes[2].set_xscale("log")
    axes[2].axhline(0.0, color="black", linewidth=0.9, alpha=0.5)
    axes[2].set_ylabel("Gap to stress-kernel optimum")
    axes[2].set_title(rf"Stress policy gaps ($\ell={focus_ell:g}$)")
    axes[2].legend(frameon=False, fontsize=8.5)
    for axis in axes:
        axis.set_xlabel("Cumulative transitions")
        axis.grid(True, alpha=0.25)
    fig.tight_layout()
    paths = save_figure(fig, figures, "convergence_policy_agreement", dpi)
    plt.close(fig)
    return paths


def shade_ambiguity(
    axis: object, nominal_slip: float, boundary: float, x_min: float, x_max: float
) -> None:
    if boundary < x_min or nominal_slip > x_max:
        return
    axis.axvspan(
        max(nominal_slip, x_min),
        min(boundary, x_max),
        color="#2ca02c",
        alpha=0.07,
    )
    if boundary <= x_max:
        axis.axvline(boundary, color="#2ca02c", linestyle="-.", linewidth=1.25)


def plot_perturbations(
    rows: Sequence[Dict[str, object]],
    focus_ell: float,
    nominal_slip: float,
    boundary: float,
    figures: Path,
    dpi: int,
) -> List[Path]:
    import matplotlib.pyplot as plt

    selected = sorted(
        (row for row in rows if math.isclose(float(row["ell"]), focus_ell)),
        key=lambda row: float(row["slip_probability"]),
    )
    if not selected:
        raise ValueError(f"No perturbation rows found for focus ell={focus_ell:g}.")
    x = np.asarray([row["slip_probability"] for row in selected], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.2), sharex=True)

    for metric, label, color in (
        ("robust_policy_return", "Variational robust", "#1f77b4"),
        ("nominal_policy_return", "Nominal Q-learning", "#d62728"),
    ):
        y = np.asarray([row[f"{metric}_mean"] for row in selected], dtype=float)
        ci = np.asarray(
            [row[f"{metric}_interval_halfwidth"] for row in selected], dtype=float
        )
        axes[0, 0].plot(x, y, color=color, linewidth=2.1, label=label)
        axes[0, 0].fill_between(x, y - ci, y + ci, color=color, alpha=0.18)
    axes[0, 0].plot(
        x,
        [row["optimal_return_mean"] for row in selected],
        color="black",
        linestyle="--",
        linewidth=1.7,
        label="Test-kernel optimum",
    )
    axes[0, 0].plot(
        x,
        [row["oracle_robust_policy_return_mean"] for row in selected],
        color="#1f77b4",
        linestyle=":",
        linewidth=1.6,
        label="Exact robust oracle",
    )
    axes[0, 0].set_ylabel("Exact discounted return")
    axes[0, 0].set_title("Returns")
    axes[0, 0].legend(frameon=False, fontsize=8.5)

    for metric, label, color in (
        ("robust_policy_gap", "Variational robust", "#1f77b4"),
        ("nominal_policy_gap", "Nominal Q-learning", "#d62728"),
    ):
        y = np.asarray([row[f"{metric}_mean"] for row in selected], dtype=float)
        ci = np.asarray(
            [row[f"{metric}_interval_halfwidth"] for row in selected], dtype=float
        )
        axes[0, 1].plot(x, y, color=color, linewidth=2.1, label=label)
        axes[0, 1].fill_between(x, y - ci, y + ci, color=color, alpha=0.18)
    axes[0, 1].axhline(0.0, color="black", linewidth=0.9, alpha=0.5)
    axes[0, 1].set_ylabel("Gap to test-kernel optimum")
    axes[0, 1].set_title("Policy gaps")
    axes[0, 1].legend(frameon=False, fontsize=8.5)

    advantage = np.asarray([row["robust_policy_advantage_mean"] for row in selected])
    advantage_ci = np.asarray(
        [row["robust_policy_advantage_interval_halfwidth"] for row in selected]
    )
    axes[1, 0].plot(x, advantage, color="#9467bd", linewidth=2.1)
    axes[1, 0].fill_between(x, advantage - advantage_ci, advantage + advantage_ci, color="#9467bd", alpha=0.18)
    axes[1, 0].axhline(0.0, color="black", linewidth=0.9, alpha=0.6)
    axes[1, 0].set_ylabel(r"$J_p(\pi_{\rm rob})-J_p(\pi_{\rm nom})$")
    axes[1, 0].set_title("Robust-policy advantage")

    for metric, label, color in (
        ("robust_policy_discounted_cliff_occupancy", "Variational robust", "#1f77b4"),
        ("nominal_policy_discounted_cliff_occupancy", "Nominal Q-learning", "#d62728"),
    ):
        y = np.asarray([row[f"{metric}_mean"] for row in selected], dtype=float)
        ci = np.asarray(
            [row[f"{metric}_interval_halfwidth"] for row in selected], dtype=float
        )
        axes[1, 1].plot(x, y, color=color, linewidth=2.1, label=label)
        axes[1, 1].fill_between(x, np.maximum(y - ci, 0.0), y + ci, color=color, alpha=0.18)
    axes[1, 1].set_ylabel("Normalized discounted occupancy")
    axes[1, 1].set_title("Cliff-marker occupancy")
    axes[1, 1].legend(frameon=False, fontsize=8.5)

    for axis in axes.flat:
        shade_ambiguity(axis, nominal_slip, boundary, float(x.min()), float(x.max()))
        axis.set_xlabel("Slip probability")
        axis.grid(True, alpha=0.25)
    fig.suptitle(
        rf"Fixed-support transition perturbations ($\ell={focus_ell:g}$; "
        rf"$p_\chi={boundary:.4f}$)",
        y=1.01,
    )
    fig.tight_layout()
    paths = save_figure(fig, figures, "perturbation_performance", dpi)
    plt.close(fig)
    return paths


def plot_ell_sensitivity(
    rows: Sequence[Dict[str, object]], figures: Path, dpi: int
) -> List[Path]:
    import matplotlib.pyplot as plt

    selected = sorted(rows, key=lambda row: float(row["ell"]))
    x = np.asarray([row["ell"] for row in selected], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.2))

    for metric, label, color in (
        ("robust_q_sup_error", "Total error to $Q^*_{\\chi}$", "#1f77b4"),
        ("floor_q_sup_error", "Error to floor fixed point", "#ff7f0e"),
    ):
        y = np.asarray([row[f"{metric}_mean"] for row in selected], dtype=float)
        axes[0, 0].plot(x, y, marker="o", color=color, linewidth=2.0, label=label)
    axes[0, 0].plot(
        x,
        [row["floor_reference_bias_mean"] for row in selected],
        marker="s",
        color="#2ca02c",
        linestyle="--",
        label="Exact floor bias",
    )
    axes[0, 0].plot(
        x,
        [row["floor_bias_bound_mean"] for row in selected],
        color="black",
        linestyle=":",
        label="Analytic bias bound",
    )
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel("Final sup-norm error")
    axes[0, 0].set_title("Error and floor bias")
    axes[0, 0].legend(frameon=False, fontsize=8.3)

    for metric, label, color, style in (
        ("stage1_gradient_rms", "Gradient RMS", "#9467bd", "-"),
        ("stage1_gradient_max", "Gradient maximum", "#8c564b", "--"),
        ("stage1_ratio_x_over_u_p95", r"95% quantile of $x/u$", "#7f7f7f", ":"),
    ):
        axes[0, 1].plot(
            x,
            [max(float(row[f"{metric}_mean"]), 1e-16) for row in selected],
            marker="o",
            color=color,
            linestyle=style,
            linewidth=1.8,
            label=label,
        )
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_ylabel("Stage-1 diagnostic")
    axes[0, 1].set_title("Gradient and denominator stability")
    axes[0, 1].legend(frameon=False, fontsize=8.3)

    for metric, label, color in (
        ("eta_projection_fraction", r"$\eta$ projection", "#1f77b4"),
        ("scale_projection_fraction", "Scale projection", "#ff7f0e"),
        ("scale_floor_fraction", "Nonnegativity floor", "#2ca02c"),
    ):
        axes[1, 0].plot(
            x,
            [row[f"{metric}_mean"] for row in selected],
            marker="o",
            color=color,
            linewidth=1.8,
            label=label,
        )
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_ylim(-0.03, 1.03)
    axes[1, 0].set_ylabel("Fraction of Stage-1 updates")
    axes[1, 0].set_title("Constraint activity")
    axes[1, 0].legend(frameon=False, fontsize=8.3)

    for metric, label, color in (
        ("robust_q_sup_error_std", "Q-error SD", "#1f77b4"),
        ("robust_policy_perturbed_gap_std", "Stress-gap SD", "#d62728"),
        ("robust_policy_oracle_agreement_std", "Agreement SD", "#9467bd"),
    ):
        axes[1, 1].plot(
            x,
            [row[metric] for row in selected],
            marker="o",
            linewidth=1.8,
            color=color,
            label=label,
        )
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_ylabel("Across-seed standard deviation")
    axes[1, 1].set_title("Empirical variability")
    axes[1, 1].legend(frameon=False, fontsize=8.3)

    for axis in axes.flat:
        axis.set_xlabel(r"Floor $\ell$")
        axis.grid(True, alpha=0.25)
    fig.tight_layout()
    paths = save_figure(fig, figures, "ell_bias_stability", dpi)
    plt.close(fig)
    return paths


def draw_policy_map(
    axis: object,
    policy: np.ndarray,
    tie_mask: np.ndarray,
    title: str,
    coordinates: np.ndarray,
    decision_mask: np.ndarray,
    separating_mask: np.ndarray,
) -> None:
    import matplotlib.patches as patches

    arrows = ("↑", "→", "↓", "←")
    rows = int(np.max(coordinates[:, 0])) + 1
    columns = int(np.max(coordinates[:, 1])) + 1
    marker_states = np.flatnonzero(~decision_mask)
    goal_state = max(marker_states, key=lambda state: int(coordinates[state, 1]))
    start_state = int(np.flatnonzero((coordinates[:, 0] == rows - 1) & (coordinates[:, 1] == 0))[0])
    for state, (row_value, column_value) in enumerate(coordinates):
        row, column = int(row_value), int(column_value)
        if state == goal_state:
            facecolor, text = "#c7e9c0", "G"
        elif not decision_mask[state]:
            facecolor, text = "#fcbba1", "C"
        else:
            facecolor = "white"
            text = "?" if tie_mask[state] else arrows[int(policy[state])]
        if separating_mask[state]:
            facecolor = "#efe6ff"
        rectangle = patches.Rectangle(
            (column, row),
            1,
            1,
            facecolor=facecolor,
            edgecolor="#666666",
            linewidth=0.8,
            hatch="///" if tie_mask[state] and decision_mask[state] else None,
        )
        axis.add_patch(rectangle)
        if separating_mask[state]:
            axis.add_patch(
                patches.Rectangle(
                    (column + 0.035, row + 0.035),
                    0.93,
                    0.93,
                    fill=False,
                    edgecolor="#b218b2",
                    linewidth=2.2,
                )
            )
        if state == start_state:
            axis.add_patch(
                patches.Rectangle(
                    (column + 0.075, row + 0.075),
                    0.85,
                    0.85,
                    fill=False,
                    edgecolor="#2166ac",
                    linewidth=1.5,
                    linestyle="--",
                )
            )
        axis.text(column + 0.5, row + 0.54, text, ha="center", va="center", fontsize=16)
    axis.set_xlim(0, columns)
    axis.set_ylim(rows, 0)
    axis.set_aspect("equal")
    axis.set_xticks(np.arange(columns) + 0.5, labels=np.arange(columns))
    axis.set_yticks(np.arange(rows) + 0.5, labels=np.arange(rows))
    axis.tick_params(length=0, labelsize=8)
    axis.set_title(title, fontsize=10.5)


def plot_policy_maps(
    arrays: Mapping[str, np.ndarray], focus_ell: float, figures: Path, dpi: int
) -> List[Path]:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    decision = arrays["decision_state_mask"]
    separating = arrays["oracle_separating_state_mask"]
    nominal_ties = arrays["modal_nominal_tie_mask"] & decision
    robust_ties = arrays["modal_robust_tie_mask"] & decision
    nominal_untied = decision & ~nominal_ties
    robust_untied = decision & ~robust_ties
    nominal_agreement = float(
        np.mean(
            arrays["modal_nominal_policy"][nominal_untied]
            == arrays["nominal_oracle_policy"][nominal_untied]
        )
    )
    robust_agreement = float(
        np.mean(
            arrays["modal_robust_policy"][robust_untied]
            == arrays["robust_oracle_policy"][robust_untied]
        )
    )
    no_ties = np.zeros_like(decision)
    panels = (
        (arrays["nominal_oracle_policy"], no_ties, "Nominal oracle"),
        (arrays["robust_oracle_policy"], no_ties, "Robust oracle"),
        (
            arrays["modal_nominal_policy"],
            nominal_ties,
            f"Modal learned nominal ({nominal_agreement:.0%} on untied states; "
            f"{int(nominal_ties.sum())} ties)",
        ),
        (
            arrays["modal_robust_policy"],
            robust_ties,
            f"Modal learned robust ({robust_agreement:.0%} on untied states; "
            f"{int(robust_ties.sum())} ties)",
        ),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.7))
    for axis, (policy, tie_mask, title) in zip(axes.flat, panels):
        draw_policy_map(
            axis,
            policy,
            tie_mask,
            title,
            arrays["state_coordinates"],
            decision,
            separating,
        )
    legend_handles = [
        patches.Patch(facecolor="#efe6ff", edgecolor="#b218b2", label="Oracle-separating state"),
        patches.Patch(facecolor="#fcbba1", edgecolor="#666666", label="Cliff marker"),
        patches.Patch(facecolor="#c7e9c0", edgecolor="#666666", label="Goal marker"),
        patches.Patch(facecolor="white", edgecolor="#2166ac", linestyle="--", label="Start"),
        patches.Patch(facecolor="white", edgecolor="#666666", hatch="///", label="Modal tie"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=5, frameon=False, fontsize=8.2)
    fig.suptitle(rf"MiniCliff greedy policies at focus $\ell={focus_ell:g}$", y=0.99)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    paths = save_figure(fig, figures, "policy_maps", dpi)
    plt.close(fig)
    return paths


def choose_focus_ell(ells: Sequence[float], requested: Optional[float]) -> float:
    if requested is not None:
        matches = [
            ell
            for ell in ells
            if math.isclose(ell, requested, rel_tol=1e-12, abs_tol=1e-15)
        ]
        if not matches:
            raise ValueError(f"--focus-ell={requested:g} is not in --ells={tuple(ells)}")
        return matches[0]
    # ell=0.10 is the tuned paper-facing operating point.  Prefer it for
    # perturbation and policy panels whenever a sweep contains it; smaller
    # floors remain visible in the convergence and sensitivity panels.
    for preferred in (0.10, 0.03):
        for ell in ells:
            if math.isclose(ell, preferred, rel_tol=1e-12, abs_tol=1e-15):
                return ell
    return sorted(ells)[len(ells) // 2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--profile", choices=tuple(PROFILES), default="quick")
    parser.add_argument("--output-root", default="paper_variational_chi2_gridworld")
    parser.add_argument("--ells", type=parse_float_tuple, default=None)
    parser.add_argument("--n-seeds", type=int, default=None)
    parser.add_argument("--base-seed", type=int, default=1)
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--outer-blocks", type=int, default=None)
    parser.add_argument("--stage1-samples", type=int, default=None)
    parser.add_argument("--q-stage-samples", type=int, default=None)
    parser.add_argument("--focus-ell", type=float, default=None)
    parser.add_argument("--fig-dpi", type=int, default=200)
    parser.add_argument("--ci-multiplier", type=float, default=1.96)
    parser.add_argument("--no-plots", action="store_true")

    environment_defaults = MiniCliffConfig()
    for field in fields(MiniCliffConfig):
        default = getattr(environment_defaults, field.name)
        parser.add_argument(
            "--" + field.name.replace("_", "-"), type=type(default), default=default
        )

    algorithm_defaults = MiniCliffAlgorithmConfig()
    profile_fields = {"seed", "ell", "outer_blocks", "stage1_samples", "q_stage_samples"}
    for field in fields(MiniCliffAlgorithmConfig):
        if field.name in profile_fields:
            continue
        default = getattr(algorithm_defaults, field.name)
        if field.name == "perturbation_grid":
            parser.add_argument(
                "--perturbation-grid",
                type=parse_float_tuple,
                default=default,
                help="comma-separated fixed-support slip probabilities",
            )
        else:
            parser.add_argument(
                "--" + field.name.replace("_", "-"), type=type(default), default=default
            )
    return parser


def resolve_configs(
    args: argparse.Namespace,
) -> Tuple[Profile, Tuple[float, ...], Tuple[int, ...], MiniCliffConfig, Dict[str, object]]:
    profile = PROFILES[args.profile]
    ells = tuple(profile.ells if args.ells is None else args.ells)
    if len(set(ells)) != len(ells):
        raise ValueError("--ells contains duplicate values.")
    if any(not math.isfinite(ell) or not (0.0 < ell <= 1.0) for ell in ells):
        raise ValueError("Every ell must be finite and lie in (0, 1].")
    n_seeds = profile.n_seeds if args.n_seeds is None else args.n_seeds
    if n_seeds < 1 or args.max_parallel < 1:
        raise ValueError("--n-seeds and --max-parallel must be positive.")
    if (
        args.fig_dpi < 1
        or not math.isfinite(args.ci_multiplier)
        or args.ci_multiplier < 0.0
    ):
        raise ValueError("--fig-dpi must be positive and --ci-multiplier nonnegative.")
    seeds = tuple(args.base_seed + offset for offset in range(n_seeds))
    environment = MiniCliffConfig(
        **{field.name: getattr(args, field.name) for field in fields(MiniCliffConfig)}
    )
    validate_minicliff_config(environment)
    algorithm_fields: Dict[str, object] = {}
    for field in fields(MiniCliffAlgorithmConfig):
        if field.name in {"seed", "ell"}:
            continue
        if field.name == "outer_blocks":
            value = profile.outer_blocks if args.outer_blocks is None else args.outer_blocks
        elif field.name == "stage1_samples":
            value = profile.stage1_samples if args.stage1_samples is None else args.stage1_samples
        elif field.name == "q_stage_samples":
            value = profile.q_stage_samples if args.q_stage_samples is None else args.q_stage_samples
        else:
            value = getattr(args, field.name)
        algorithm_fields[field.name] = value
    validate_algorithm_config(
        MiniCliffAlgorithmConfig(seed=seeds[0], ell=ells[0], **algorithm_fields)
    )
    return profile, ells, seeds, environment, algorithm_fields


def build_specs(
    output_root: Path,
    ells: Sequence[float],
    seeds: Sequence[int],
    environment: MiniCliffConfig,
    algorithm_fields: Mapping[str, object],
) -> List[RunSpec]:
    specs: List[RunSpec] = []
    for ell in ells:
        for seed in seeds:
            algorithm = MiniCliffAlgorithmConfig(seed=seed, ell=ell, **algorithm_fields)
            validate_algorithm_config(algorithm)
            specs.append(
                RunSpec(
                    ell=ell,
                    seed=seed,
                    environment=environment,
                    algorithm=algorithm,
                    output_dir=output_root / "raw" / f"ell_{float_slug(ell)}" / f"seed_{seed}",
                )
            )
    return specs


def main() -> None:
    args = build_parser().parse_args()
    profile, ells, seeds, environment, algorithm_fields = resolve_configs(args)
    output_root = Path(args.output_root).resolve()
    aggregated_dir = output_root / "aggregated"
    figures_dir = output_root / "figures"
    output_root.mkdir(parents=True, exist_ok=True)
    specs = build_specs(output_root, ells, seeds, environment, algorithm_fields)
    source_hashes = expected_source_hashes()
    focus_ell = choose_focus_ell(ells, args.focus_ell)

    print(
        f"profile={args.profile} runs={len(specs)} seeds={len(seeds)} "
        f"ells={','.join(format(ell, 'g') for ell in ells)} focus_ell={focus_ell:g} "
        f"max_parallel={args.max_parallel}",
        flush=True,
    )
    boundary = environment.nominal_slip_probability + math.sqrt(
        float(algorithm_fields["chi2_delta"])
        * environment.nominal_slip_probability
        * (1.0 - environment.nominal_slip_probability)
    )
    manifest_path = output_root / "manifest.json"
    manifest: Dict[str, object] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "profile": args.profile,
        "profile_defaults": asdict(profile),
        "resolved": {
            "ells": list(ells),
            "seeds": list(seeds),
            "focus_ell": focus_ell,
            "environment": asdict(environment),
            "algorithm_fixed_fields": dict(algorithm_fields),
            "max_parallel": args.max_parallel,
            "ci_multiplier": args.ci_multiplier,
        },
        "source_sha256": source_hashes,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "ambiguity_upper_slip_probability": boundary,
        "runs": [],
        "failures": [],
        "command": sys.argv,
    }
    # Replace any old complete manifest before doing work.  An interruption can
    # therefore never leave stale output advertised as the current command's
    # completed result.
    write_json_atomic(manifest_path, manifest)

    pending: List[RunSpec] = []
    records: List[Dict[str, object]] = []
    for spec in specs:
        if args.skip_existing and spec.output_dir.exists():
            validate_saved_run(spec, source_hashes)
            records.append(
                {
                    "ell": spec.ell,
                    "seed": spec.seed,
                    "status": "skipped_validated",
                    "output_dir": str(spec.output_dir.relative_to(output_root)),
                }
            )
            print(f"validated existing ell={spec.ell:g} seed={spec.seed}", flush=True)
        else:
            pending.append(spec)

    failures: List[Dict[str, object]] = []

    def record_success(spec: RunSpec, result: Mapping[str, object], status: str) -> None:
        records.append(
            {
                "ell": spec.ell,
                "seed": spec.seed,
                "status": status,
                "output_dir": str(spec.output_dir.relative_to(output_root)),
                "final_q_error": result["final_q_error"],
                "final_agreement": result["final_agreement"],
            }
        )

    def run_sequential(run_specs: Sequence[RunSpec], status: str) -> None:
        for completed, spec in enumerate(run_specs, start=1):
            try:
                result = execute_run(spec)
                record_success(spec, result, status)
                print(
                    f"[{completed}/{len(run_specs)}] ell={spec.ell:g} seed={spec.seed} "
                    f"error={result['final_q_error']:.5g} "
                    f"agreement={result['final_agreement']:.3f}",
                    flush=True,
                )
            except Exception as exc:  # finish independent requested runs
                failures.append({"ell": spec.ell, "seed": spec.seed, "error": repr(exc)})
                print(f"FAILED ell={spec.ell:g} seed={spec.seed}: {exc}", file=sys.stderr)

    if args.max_parallel == 1:
        run_sequential(pending, "completed")
    elif pending:
        try:
            executor_context = ProcessPoolExecutor(
                max_workers=min(args.max_parallel, len(pending))
            )
        except (OSError, PermissionError) as exc:
            print(
                f"process parallelism unavailable ({exc}); using one worker",
                file=sys.stderr,
                flush=True,
            )
            run_sequential(pending, "completed_sequential_fallback")
        else:
            with executor_context as executor:
                futures = {executor.submit(execute_run, spec): spec for spec in pending}
                for completed, future in enumerate(as_completed(futures), start=1):
                    spec = futures[future]
                    try:
                        result = future.result()
                        record_success(spec, result, "completed")
                        print(
                            f"[{completed}/{len(pending)}] ell={spec.ell:g} seed={spec.seed} "
                            f"error={result['final_q_error']:.5g} "
                            f"agreement={result['final_agreement']:.3f}",
                            flush=True,
                        )
                    except Exception as exc:  # finish independent requested runs
                        failures.append(
                            {"ell": spec.ell, "seed": spec.seed, "error": repr(exc)}
                        )
                        print(
                            f"FAILED ell={spec.ell:g} seed={spec.seed}: {exc}",
                            file=sys.stderr,
                        )

    manifest["runs"] = sorted(
        records, key=lambda row: (float(row["ell"]), int(row["seed"]))
    )
    manifest["failures"] = failures
    if failures:
        manifest["status"] = "failed"
        write_json_atomic(manifest_path, manifest)
        raise RuntimeError(f"{len(failures)} run(s) failed; see {manifest_path}.")

    manifest["status"] = "postprocessing"
    write_json_atomic(manifest_path, manifest)

    raw_metrics, raw_perturbations, array_runs = load_raw(specs, source_hashes)
    learning = aggregate_learning(raw_metrics, args.ci_multiplier)
    perturbations = aggregate_perturbations(raw_perturbations, args.ci_multiplier)
    ell_summary = aggregate_ell_summary(raw_metrics, args.ci_multiplier)
    modal_arrays = build_modal_arrays(array_runs, focus_ell)

    learning_path = aggregated_dir / "learning_curves.csv"
    perturbation_path = aggregated_dir / "perturbation_summary.csv"
    ell_path = aggregated_dir / "ell_summary.csv"
    modal_path = aggregated_dir / "modal_policies_focus_ell.npz"
    write_csv(learning_path, learning)
    write_csv(perturbation_path, perturbations)
    write_csv(ell_path, ell_summary)
    aggregated_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(modal_path, focus_ell=np.asarray(focus_ell), **modal_arrays)

    figure_paths: List[Path] = []
    if not args.no_plots:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/drrl-sim-matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        figure_paths.extend(plot_convergence(learning, focus_ell, figures_dir, args.fig_dpi))
        figure_paths.extend(
            plot_perturbations(
                perturbations,
                focus_ell,
                environment.nominal_slip_probability,
                boundary,
                figures_dir,
                args.fig_dpi,
            )
        )
        figure_paths.extend(plot_ell_sensitivity(ell_summary, figures_dir, args.fig_dpi))
        figure_paths.extend(plot_policy_maps(modal_arrays, focus_ell, figures_dir, args.fig_dpi))

    manifest["aggregated_files"] = [
        str(path.relative_to(output_root))
        for path in (learning_path, perturbation_path, ell_path, modal_path)
    ]
    manifest["figure_files"] = [str(path.relative_to(output_root)) for path in figure_paths]
    manifest["modal_policy_tie_counts"] = {
        "nominal_decision_states": int(
            np.count_nonzero(
                modal_arrays["modal_nominal_tie_mask"]
                & modal_arrays["decision_state_mask"]
            )
        ),
        "robust_decision_states": int(
            np.count_nonzero(
                modal_arrays["modal_robust_tie_mask"]
                & modal_arrays["decision_state_mask"]
            )
        ),
    }
    manifest["status"] = "complete"
    write_json_atomic(manifest_path, manifest)
    print(f"output_root={output_root}")
    print(f"manifest={manifest_path}")
    print(f"figures={len(figure_paths)} focus_ell={focus_ell:g}")


if __name__ == "__main__":
    main()
