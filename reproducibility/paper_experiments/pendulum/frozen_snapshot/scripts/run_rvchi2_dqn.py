#!/usr/bin/env python3
"""Run isolated RVChi2-DQN smoke, development, or frozen reporting studies."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import fields, replace
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rvchi2_dqn.artifacts import (  # noqa: E402
    artifact_hashes,
    sha256_file,
    utc_now,
    write_csv_atomic,
    write_json_atomic,
)
from rvchi2_dqn.config import (  # noqa: E402
    DEVELOPMENT_SEEDS,
    REPORTING_SEEDS,
    SCIENTIFIC_SOURCE_FILES,
    SMOKE_SEEDS,
    ExperimentConfig,
    task_defaults,
    validate_config,
)
from rvchi2_dqn.trainer import run_experiment  # noqa: E402


SOURCE_PATHS = {
    name: ROOT / relative
    for name, relative in SCIENTIFIC_SOURCE_FILES.items()
}
FREEZE_SCHEMA_VERSION = 2
RUN_ARTIFACT_FILES = frozenset(
    {
        "config.json",
        "learning_metrics.csv",
        "evaluation_episodes.csv",
        "evaluation_summary.csv",
        "checkpoint_evaluation_episodes.csv",
        "checkpoint_evaluation_summary.csv",
        "summary.json",
        "checkpoints.pt",
        "backup_calibration.npz",
        "metadata.json",
    }
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _source_hashes() -> dict[str, str]:
    return {name: sha256_file(path) for name, path in SOURCE_PATHS.items()}


def _parse_seeds(value: str | None, phase: str, configured: Sequence[int] | None) -> tuple[int, ...]:
    if value:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    elif configured is not None:
        seeds = tuple(int(item) for item in configured)
    else:
        seeds = {
            "smoke": (1,),
            "development": tuple(sorted(DEVELOPMENT_SEEDS)),
            "reporting": tuple(sorted(REPORTING_SEEDS)),
        }[phase]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be a nonempty unique list.")
    allowed = {
        "smoke": SMOKE_SEEDS,
        "development": DEVELOPMENT_SEEDS,
        "reporting": REPORTING_SEEDS,
    }[phase]
    if any(seed not in allowed for seed in seeds):
        raise ValueError(f"One or more seeds are reserved outside phase {phase!r}.")
    return seeds


def _coerce_overrides(overrides: Mapping[str, Any]) -> dict[str, Any]:
    field_names = {field.name for field in fields(ExperimentConfig)}
    unknown = set(overrides) - field_names
    if unknown:
        raise ValueError(f"Unknown config overrides: {sorted(unknown)}")
    result = dict(overrides)
    for name in (
        "hidden_dims",
        "certified_probabilities",
        "ood_probabilities",
        "enabled_methods",
    ):
        if name in result:
            result[name] = tuple(result[name])
    return result


def resolve_study(
    specification: Mapping[str, Any],
    *,
    seed_override: str | None = None,
) -> tuple[str, str, tuple[int, ...], dict[str, Any]]:
    task = str(specification.get("task", "lqr")).lower()
    phase = str(specification.get("phase", "smoke")).lower()
    if phase not in {"smoke", "development", "reporting"}:
        raise ValueError("phase must be smoke, development, or reporting.")
    seeds_value = specification.get("seeds")
    if seeds_value is not None and not isinstance(seeds_value, list):
        raise ValueError("Config seeds must be a JSON list.")
    seeds = _parse_seeds(seed_override, phase, seeds_value)
    overrides = specification.get("overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError("overrides must be a JSON object.")
    resolved_overrides = _coerce_overrides(overrides)
    for protected in ("task", "phase", "seed", "schema_version"):
        if protected in resolved_overrides:
            raise ValueError(f"{protected} cannot be supplied as an override.")
    return task, phase, seeds, resolved_overrides


def _config_for_seed(
    task: str, phase: str, seed: int, overrides: Mapping[str, Any]
) -> ExperimentConfig:
    config = replace(task_defaults(task, phase=phase, seed=seed), **overrides)
    validate_config(config)
    return config


def _config_without_seed_and_phase(config: ExperimentConfig) -> dict[str, Any]:
    payload = config.to_dict()
    payload.pop("seed")
    payload.pop("phase")
    return json.loads(json.dumps(payload))


def _json_config(config: ExperimentConfig) -> dict[str, Any]:
    return json.loads(json.dumps(config.to_dict()))


def _validate_existing_run(path: Path, config: ExperimentConfig) -> dict[str, Any]:
    if not (path / "manifest.json").is_file():
        raise ValueError(f"Existing run is incomplete: {path}")
    manifest = _read_json(path / "manifest.json")
    hashes = manifest.get("files_sha256")
    if (
        not isinstance(hashes, dict)
        or set(hashes) != RUN_ARTIFACT_FILES
        or any(
            not (path / name).is_file() or sha256_file(path / name) != digest
            for name, digest in hashes.items()
        )
    ):
        raise ValueError(f"Existing run authentication failed: {path}")

    expected_config = _json_config(config)
    stored_config = _read_json(path / "config.json")
    metadata = _read_json(path / "metadata.json")
    summary = _read_json(path / "summary.json")
    expected_sources = _source_hashes()
    if manifest.get("config") != expected_config or stored_config != expected_config:
        raise ValueError(f"Existing run config differs: {path}")
    if (
        manifest.get("artifact_schema_version") != config.schema_version
        or metadata.get("artifact_schema_version") != config.schema_version
        or summary.get("result_schema_version") != config.schema_version
    ):
        raise ValueError(f"Existing run schema differs: {path}")
    if (
        manifest.get("source_sha256") != expected_sources
        or metadata.get("source_sha256") != expected_sources
    ):
        raise ValueError(f"Existing run source hashes differ: {path}")
    if manifest.get("status") != summary.get("status"):
        raise ValueError(f"Existing run status differs: {path}")
    return summary


def _run_seed(arguments: tuple[ExperimentConfig, str]) -> tuple[int, str, dict[str, Any]]:
    config, output = arguments
    result = run_experiment(config, Path(output))
    return config.seed, str(result.output_dir), result.summary


def _critical_975(degrees_of_freedom: int) -> float | None:
    # Exact familiar Student-t values are sufficient for the predeclared n<=10.
    values = {
        1: 12.7062047364,
        2: 4.3026527297,
        3: 3.1824463053,
        4: 2.7764451052,
        5: 2.5705818356,
        6: 2.4469118511,
        7: 2.3646242510,
        8: 2.3060041352,
        9: 2.2621571629,
    }
    return values.get(degrees_of_freedom)


def _mean_ci(values: Sequence[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"n": 0, "mean": None, "ci95_low": None, "ci95_high": None}
    mean = float(np.mean(array))
    if array.size < 2:
        return {"n": int(array.size), "mean": mean, "ci95_low": None, "ci95_high": None}
    critical = _critical_975(array.size - 1)
    if critical is None:
        critical = 1.96
    half = float(critical * np.std(array, ddof=1) / math.sqrt(array.size))
    return {
        "n": int(array.size),
        "mean": mean,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
    }


def _development_auxiliary_checks(
    summaries: Sequence[Mapping[str, Any]],
    config: ExperimentConfig,
) -> dict[str, bool]:
    """Check persistent affine projection and scale-floor collapse."""

    eta_values: list[float] = []
    floor_values: list[float] = []
    for summary in summaries:
        health = summary.get("auxiliary_health", {}).get("affine", {})
        eta = health.get("eta_projection_fraction_mean")
        floor = health.get("u_floor_fraction_mean")
        if eta is None or floor is None:
            return {
                "affine_eta_projection_below_failure_threshold": False,
                "affine_u_floor_below_failure_threshold": False,
            }
        eta_values.append(float(eta))
        floor_values.append(float(floor))
    return {
        "affine_eta_projection_below_failure_threshold": bool(summaries)
        and all(
            math.isfinite(value)
            and value <= config.auxiliary_eta_projection_failure_rate
            for value in eta_values
        ),
        "affine_u_floor_below_failure_threshold": bool(summaries)
        and all(
            math.isfinite(value)
            and value <= config.auxiliary_u_floor_failure_rate
            for value in floor_values
        ),
    }


def _nominal_gate_for_authorization(
    summary: Mapping[str, Any],
    evaluation_rows: Sequence[Mapping[str, Any]],
    config: ExperimentConfig,
) -> Mapping[str, Any]:
    """Authenticate final nominal competence against its unique frozen CSV row."""

    final_gate = summary.get("final_nominal_gate")
    if not isinstance(final_gate, Mapping):
        return {}

    def is_true(value: Any) -> bool:
        return value is True or (
            isinstance(value, str) and value.strip().lower() == "true"
        )

    def close(left: float, right: float) -> bool:
        return math.isclose(left, right, rel_tol=0.0, abs_tol=1.0e-12)

    matching_rows: list[Mapping[str, Any]] = []
    try:
        for row in evaluation_rows:
            if row.get("method") != "nominal" or not is_true(
                row.get("is_final_frozen_sweep")
            ):
                continue
            probability = float(row["fault_probability"])
            if math.isfinite(probability) and close(
                probability, config.nominal_fault_probability
            ):
                matching_rows.append(row)
        if len(matching_rows) != 1:
            return {}

        evidence = matching_rows[0]
        observed_return = float(evidence["mean_raw_return"])
        observed_failure = float(evidence["failure_probability"])
        evidence_block = float(evidence["checkpoint_block"])
        gate_return = float(final_gate["observed_raw_return"])
        gate_failure = float(final_gate["observed_failure_probability"])
        gate_probability = float(final_gate["fault_probability"])
        gate_block = float(final_gate["checkpoint_block"])
        required_return = float(final_gate["required_raw_return"])
        required_failure = float(
            final_gate["required_max_failure_probability"]
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return {}

    numeric_values = (
        observed_return,
        observed_failure,
        evidence_block,
        gate_return,
        gate_failure,
        gate_probability,
        gate_block,
        required_return,
        required_failure,
    )
    if not all(math.isfinite(value) for value in numeric_values):
        return {}
    if not 0.0 <= observed_failure <= 1.0:
        return {}
    expected_pass = bool(
        observed_return >= config.nominal_competence_return
        and observed_failure <= config.nominal_competence_failure_rate
    )
    if not (
        final_gate.get("evaluation_stage") == "final_frozen_sweep"
        and close(evidence_block, float(config.outer_blocks))
        and close(gate_block, float(config.outer_blocks))
        and close(gate_probability, config.nominal_fault_probability)
        and close(required_return, config.nominal_competence_return)
        and close(required_failure, config.nominal_competence_failure_rate)
        and close(gate_return, observed_return)
        and close(gate_failure, observed_failure)
        and final_gate.get("passed") is expected_pass
    ):
        return {}
    return {
        "passed": expected_pass,
        "observed_raw_return": observed_return,
        "observed_failure_probability": observed_failure,
        "evidence_validated": True,
    }


def _development_check_sections(
    summaries: Sequence[Mapping[str, Any]],
    evaluation_rows_by_run: Sequence[Sequence[Mapping[str, Any]]],
    affine_effects: Sequence[float],
    affine_aucs: Sequence[float],
    config: ExperimentConfig,
) -> tuple[dict[str, bool], dict[str, bool]]:
    """Separate reporting requirements from visible mechanism diagnostics."""

    if len(evaluation_rows_by_run) == len(summaries):
        final_gates = [
            _nominal_gate_for_authorization(summary, rows, config)
            for summary, rows in zip(summaries, evaluation_rows_by_run)
        ]
    else:
        final_gates = [{} for _ in summaries]
    evidence_pass = bool(summaries) and all(
        gate.get("evidence_validated") is True for gate in final_gates
    )
    nominal_pass = evidence_pass and all(
        gate.get("passed") is True for gate in final_gates
    )
    sign_pass = sum(value > 0.0 for value in affine_effects) >= 4
    mean_effect_pass = bool(affine_effects) and float(np.mean(affine_effects)) > 0.0
    auc_pass = bool(affine_aucs) and float(np.mean(affine_aucs)) > 0.0
    costs = [
        float(summary["methods"]["affine"]["nominal_kernel_cost_vs_nominal"])
        for summary in summaries
    ]
    cost_pass = bool(costs) and all(
        value >= -config.max_nominal_cost for value in costs
    )
    matched_environment_steps = bool(summaries) and all(
        summary.get("equal_branch_environment_steps") is True
        for summary in summaries
    )
    matched_q_updates = bool(summaries) and all(
        summary.get("equal_branch_q_updates") is True for summary in summaries
    )
    clipping_pass = bool(summaries) and all(
        summary.get("target_clipping_gate_passed") is True
        for summary in summaries
    )
    required_numeric_values = [
        *[float(value) for value in affine_effects],
        *[float(value) for value in affine_aucs],
        *costs,
    ]
    clip_maps_valid = True
    for summary, gate in zip(summaries, final_gates):
        required_numeric_values.extend(
            (
                float(gate.get("observed_raw_return", float("nan"))),
                float(gate.get("observed_failure_probability", float("nan"))),
            )
        )
        clip_map = summary.get("max_block_mean_target_clip_fraction")
        if not isinstance(clip_map, Mapping) or not clip_map:
            clip_maps_valid = False
        else:
            required_numeric_values.extend(
                float(value) for value in clip_map.values()
            )
    finite_pass = (
        bool(required_numeric_values)
        and len(affine_effects) == len(summaries)
        and len(affine_aucs) == len(summaries)
        and evidence_pass
        and clip_maps_valid
        and all(math.isfinite(value) for value in required_numeric_values)
    )

    required_checks = {
        "all_final_nominal_evidence_authenticated": evidence_pass,
        "all_final_nominal_competence": nominal_pass,
        "affine_positive_boundary_on_at_least_four_seeds": sign_pass,
        "affine_positive_mean_boundary_effect": mean_effect_pass,
        "affine_positive_mean_certified_auc": auc_pass,
        "nominal_cost_within_predeclared_limit": cost_pass,
        "equal_branch_environment_step_budgets": matched_environment_steps,
        "equal_branch_q_update_budgets": matched_q_updates,
        "target_clipping_below_failure_threshold": clipping_pass,
        "all_required_numerics_finite": finite_pass,
    }

    support_pass = bool(summaries) and all(
        summary["replay_support"]["affine"]["selected_action_zero_bin_fraction"]
        is not None
        and summary["replay_support"]["affine"]["selected_action_zero_bin_fraction"]
        <= 0.01
        for summary in summaries
    )
    disagreement_pass = bool(summaries) and all(
        summary["replay_support"]["affine"][
            "occupied_policy_disagreement_with_nominal"
        ]
        >= 0.05
        for summary in summaries
    )
    calibration_pass = bool(summaries) and all(
        summary["backup_calibration"]["affine"]["by_support"]["supported"][
            "pearson"
        ]
        is not None
        and summary["backup_calibration"]["affine"]["by_support"]["supported"][
            "pearson"
        ]
        >= 0.90
        and summary["backup_calibration"]["affine"]["by_support"]["supported"][
            "normalized_mae"
        ]
        is not None
        and summary["backup_calibration"]["affine"]["by_support"]["supported"][
            "normalized_mae"
        ]
        <= 0.10
        for summary in summaries
    )
    diagnostic_checks = {
        "selected_action_zero_support_at_most_one_percent": support_pass,
        "occupied_policy_disagreement_at_least_five_percent": disagreement_pass,
        "occupied_backup_calibration_engineering_target": calibration_pass,
        **_development_auxiliary_checks(summaries, config),
    }
    return required_checks, diagnostic_checks


def _development_gate_decision(
    required_checks: Mapping[str, bool],
    diagnostic_checks: Mapping[str, bool],
) -> dict[str, Any]:
    """Apply the explicit claim gate while retaining all diagnostic outcomes."""

    required_passed = bool(required_checks) and all(required_checks.values())
    diagnostics_passed = bool(diagnostic_checks) and all(
        diagnostic_checks.values()
    )
    compatibility_checks = {
        **required_checks,
        **diagnostic_checks,
        "all_nominal_competence": required_checks.get(
            "all_final_nominal_competence", False
        ),
    }
    return {
        "required_checks": dict(required_checks),
        "diagnostic_checks": dict(diagnostic_checks),
        "checks": compatibility_checks,
        "all_diagnostics_passed": diagnostics_passed,
        "development_gate_passed": required_passed,
        "reporting_authorized": required_passed,
    }


def _aggregate(
    raw_runs: Sequence[tuple[int, Path, dict[str, Any]]],
    output_root: Path,
    phase: str,
    config_template: ExperimentConfig,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    evaluation_rows_by_run: list[list[dict[str, Any]]] = []
    effects: dict[str, list[float]] = {method: [] for method in ("exact", "affine", "full_nn")}
    aucs: dict[str, list[float]] = {method: [] for method in effects}
    for seed, run_path, summary in raw_runs:
        evaluation_rows: list[dict[str, Any]] = []
        import csv

        with (run_path / "evaluation_summary.csv").open(newline="", encoding="utf-8") as handle:
            evaluation_rows = list(csv.DictReader(handle))
        evaluation_rows_by_run.append(evaluation_rows)
        for row in evaluation_rows:
            rows.append(
                {
                    "seed": seed,
                    "method": row["method"],
                    "fault_probability": float(row["fault_probability"]),
                    "inside_certified_interval": row["inside_certified_interval"] == "True",
                    "ood_stress_test": row["ood_stress_test"] == "True",
                    "mean_raw_return": float(row["mean_raw_return"]),
                    "mean_discounted_return": float(row["mean_discounted_return"]),
                    "failure_probability": float(row["failure_probability"]),
                    "mean_length": float(row["mean_length"]),
                }
            )
        if "methods" in summary:
            for method in effects:
                if method not in summary["methods"]:
                    continue
                effects[method].append(
                    float(summary["methods"][method]["paired_boundary_advantage_vs_nominal"])
                )
                aucs[method].append(
                    float(summary["methods"][method]["paired_advantage_auc_p010_to_p025"])
                )

    aggregated_dir = output_root / "aggregated"
    aggregated_dir.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(aggregated_dir / "final_by_seed.csv", rows)
    profile_rows: list[dict[str, Any]] = []
    groups: dict[tuple[str, float], list[float]] = {}
    for row in rows:
        groups.setdefault((row["method"], row["fault_probability"]), []).append(
            row["mean_raw_return"]
        )
    for (method, probability), values in sorted(groups.items()):
        profile_rows.append(
            {
                "method": method,
                "fault_probability": probability,
                **_mean_ci(values),
            }
        )
    write_csv_atomic(aggregated_dir / "robustness_profile.csv", profile_rows)
    nominal_lookup = {
        (int(row["seed"]), float(row["fault_probability"])): float(row["mean_raw_return"])
        for row in rows
        if row["method"] == "nominal"
    }
    paired_groups: dict[tuple[str, float], list[float]] = {}
    paired_seed_rows: list[dict[str, Any]] = []
    for row in rows:
        if row["method"] == "nominal":
            continue
        key = (int(row["seed"]), float(row["fault_probability"]))
        difference = float(row["mean_raw_return"]) - nominal_lookup[key]
        paired_seed_rows.append(
            {
                "seed": row["seed"],
                "method": row["method"],
                "fault_probability": row["fault_probability"],
                "paired_return_advantage": difference,
            }
        )
        paired_groups.setdefault(
            (str(row["method"]), float(row["fault_probability"])), []
        ).append(difference)
    paired_profile_rows = [
        {
            "method": method,
            "fault_probability": probability,
            **_mean_ci(values),
        }
        for (method, probability), values in sorted(paired_groups.items())
    ]
    write_csv_atomic(aggregated_dir / "paired_by_seed.csv", paired_seed_rows)
    write_csv_atomic(aggregated_dir / "paired_profile.csv", paired_profile_rows)

    inference = {
        method: {
            "boundary_advantage": _mean_ci(effects[method]),
            "paired_advantage_auc_p010_to_p025": _mean_ci(aucs[method]),
        }
        for method in effects
    }
    gate: dict[str, Any] = {
        "evaluated": (
            phase == "development"
            and len(raw_runs) == 5
            and all("affine" in summary.get("methods", {}) for _, _, summary in raw_runs)
        ),
        "development_gate_passed": False,
        "reporting_authorized": False,
    }
    if gate["evaluated"]:
        summaries = [summary for _, _, summary in raw_runs]
        affine_effects = effects["affine"]
        affine_auc = aucs["affine"]
        required_checks, diagnostic_checks = _development_check_sections(
            summaries,
            evaluation_rows_by_run,
            affine_effects,
            affine_auc,
            config_template,
        )
        gate["auxiliary_health_thresholds"] = {
            "eta_projection_fraction_mean_max": (
                config_template.auxiliary_eta_projection_failure_rate
            ),
            "u_floor_fraction_mean_max": (
                config_template.auxiliary_u_floor_failure_rate
            ),
        }
        gate.update(
            _development_gate_decision(required_checks, diagnostic_checks)
        )
    write_json_atomic(aggregated_dir / "seed_inference.json", inference)
    write_json_atomic(aggregated_dir / "development_gate.json", gate)
    return {
        "phase": phase,
        "seeds": [seed for seed, _, _ in raw_runs],
        "seed_inference": inference,
        "development_gate": gate,
        "aggregated_files": [
            "aggregated/final_by_seed.csv",
            "aggregated/robustness_profile.csv",
            "aggregated/paired_by_seed.csv",
            "aggregated/paired_profile.csv",
            "aggregated/seed_inference.json",
            "aggregated/development_gate.json",
        ],
    }


def _validate_reporting_freeze(
    path: Path,
    config_template: ExperimentConfig,
) -> dict[str, Any]:
    freeze = _read_json(path)
    if (
        freeze.get("schema_version") != FREEZE_SCHEMA_VERSION
        or freeze.get("status") != "reporting_authorized"
        or freeze.get("development_gate_passed") is not True
    ):
        raise ValueError("Freeze manifest does not authorize reporting.")
    if freeze.get("source_sha256") != _source_hashes():
        raise ValueError("Frozen scientific source hashes differ from current code.")
    if freeze.get("frozen_config_without_seed_and_phase") != _config_without_seed_and_phase(config_template):
        raise ValueError("Reporting config differs from the frozen development config.")
    if freeze.get("development_seeds") != sorted(DEVELOPMENT_SEEDS):
        raise ValueError("Freeze manifest does not identify exactly seeds 31--35.")
    if freeze.get("reporting_seeds") != sorted(REPORTING_SEEDS):
        raise ValueError("Freeze manifest does not reserve exactly seeds 101--110.")
    try:
        gate_path = Path(freeze["development_gate_path"])
    except (KeyError, TypeError) as error:
        raise ValueError("Freeze manifest lacks a development gate path.") from error
    if not gate_path.is_file() or sha256_file(gate_path) != freeze["development_gate_sha256"]:
        raise ValueError("Frozen development gate is absent or changed.")
    gate = _read_json(gate_path)
    required_checks = gate.get("required_checks")
    if not (
        gate.get("development_gate_passed") is True
        and gate.get("reporting_authorized") is True
        and isinstance(required_checks, Mapping)
        and bool(required_checks)
        and all(value is True for value in required_checks.values())
    ):
        raise ValueError("Frozen development gate does not authorize reporting.")
    return freeze


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", default=None, help="Comma-separated phase-valid seeds.")
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--freeze-manifest", type=Path, default=None)
    args = parser.parse_args()

    specification = _read_json(args.config)
    task, phase, seeds, overrides = resolve_study(
        specification, seed_override=args.seeds
    )
    if args.max_parallel < 1:
        raise ValueError("max-parallel must be positive.")
    configs = [_config_for_seed(task, phase, seed, overrides) for seed in seeds]
    template = configs[0]
    if phase == "reporting":
        if args.freeze_manifest is None:
            raise ValueError("Reporting requires --freeze-manifest from a passing development study.")
        _validate_reporting_freeze(args.freeze_manifest, template)
    elif args.freeze_manifest is not None:
        raise ValueError("freeze-manifest is only valid for reporting.")

    raw_root = args.output_root / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    completed: list[tuple[int, Path, dict[str, Any]]] = []
    pending: list[tuple[ExperimentConfig, str]] = []
    for config in configs:
        run_path = raw_root / f"seed_{config.seed:04d}"
        if run_path.exists():
            if not args.skip_existing:
                raise FileExistsError(f"Run already exists: {run_path}")
            completed.append((config.seed, run_path, _validate_existing_run(run_path, config)))
        else:
            pending.append((config, str(run_path)))

    if pending:
        workers = min(args.max_parallel, len(pending))
        if workers == 1:
            results = [_run_seed(item) for item in pending]
        else:
            results = []
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(_run_seed, item): item[0].seed for item in pending}
                for future in as_completed(futures):
                    results.append(future.result())
        completed.extend((seed, Path(path), summary) for seed, path, summary in results)
    completed.sort(key=lambda item: item[0])
    config_by_seed = {config.seed: config for config in configs}
    completed = [
        (
            seed,
            path,
            _validate_existing_run(path, config_by_seed[seed]),
        )
        for seed, path, _ in completed
    ]

    study = _aggregate(completed, args.output_root, phase, template)
    study.update(
        {
            "created_at_utc": utc_now(),
            "task": task,
            "config_path": str(args.config.resolve()),
            "source_sha256": _source_hashes(),
            "resolved_config_without_seed": _config_without_seed_and_phase(template),
            "raw_runs": [str(path.relative_to(args.output_root)) for _, path, _ in completed],
        }
    )
    write_json_atomic(args.output_root / "manifest.json", study)

    gate = study["development_gate"]
    if phase == "development" and gate["development_gate_passed"]:
        gate_path = (args.output_root / "aggregated" / "development_gate.json").resolve()
        freeze = {
            "schema_version": FREEZE_SCHEMA_VERSION,
            "status": "reporting_authorized",
            "created_at_utc": utc_now(),
            "development_gate_passed": True,
            "development_seeds": sorted(DEVELOPMENT_SEEDS),
            "reporting_seeds": sorted(REPORTING_SEEDS),
            "development_gate_path": str(gate_path),
            "development_gate_sha256": sha256_file(gate_path),
            "frozen_config_without_seed_and_phase": _config_without_seed_and_phase(template),
            "source_sha256": _source_hashes(),
        }
        write_json_atomic(args.output_root / "reporting_freeze_manifest.json", freeze)

    print(json.dumps({"output_root": str(args.output_root), **study["development_gate"]}, indent=2))


if __name__ == "__main__":
    main()
