#!/usr/bin/env python3
"""Generate paper-ready RVChi2-DQN profile, mechanism, and table artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rvchi2_dqn.artifacts import write_csv_atomic, write_json_atomic  # noqa: E402
from rvchi2_dqn.networks import QNetwork  # noqa: E402


COLORS = {
    "nominal": "#333333",
    "exact": "#377eb8",
    "affine": "#d95f02",
    "full_nn": "#1b9e77",
}
LABELS = {
    "nominal": "Nominal DDQN",
    "exact": "Exact-inner",
    "affine": r"RV$\chi^2$-A",
    "full_nn": r"RV$\chi^2$-N",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _plot_segmented_profile(
    axis: plt.Axes,
    rows: pd.DataFrame,
    *,
    value: str,
    low: str,
    high: str,
    label: str,
    color: str,
) -> None:
    rows = rows.sort_values("fault_probability")
    inside = rows[rows["fault_probability"] <= 0.25]
    outside = rows[rows["fault_probability"] >= 0.25]
    axis.plot(inside["fault_probability"], inside[value], color=color, marker="o", label=label)
    if inside[low].notna().any():
        axis.fill_between(
            inside["fault_probability"], inside[low], inside[high], color=color, alpha=0.16
        )
    if len(outside) > 1:
        axis.plot(
            outside["fault_probability"], outside[value], color=color, marker="o", linestyle="--"
        )
        if outside[low].notna().any():
            axis.fill_between(
                outside["fault_probability"], outside[low], outside[high], color=color, alpha=0.10
            )


def plot_profiles(study_root: Path, figures: Path) -> None:
    manifest = _read_json(study_root / "manifest.json")
    profile = pd.read_csv(study_root / "aggregated" / "robustness_profile.csv")
    paired = pd.read_csv(study_root / "aggregated" / "paired_profile.csv")
    figure, axes = plt.subplots(2, 1, figsize=(5.4, 5.6), sharex=True)
    for axis in axes:
        axis.axvspan(0.0, 0.25, color="#d9eaf7", alpha=0.35, zorder=-5, label="certified interval")
        axis.axvline(0.10, color="#777777", linestyle=":", linewidth=1)
        axis.axvline(0.25, color="#777777", linestyle="--", linewidth=1)
        axis.grid(alpha=0.2)
    for method in ("nominal", "exact", "affine", "full_nn"):
        rows = profile[profile["method"] == method]
        if rows.empty:
            continue
        _plot_segmented_profile(
            axes[0], rows, value="mean", low="ci95_low", high="ci95_high",
            label=LABELS[method], color=COLORS[method]
        )
    for method in ("exact", "affine", "full_nn"):
        rows = paired[paired["method"] == method]
        if rows.empty:
            continue
        _plot_segmented_profile(
            axes[1], rows, value="mean", low="ci95_low", high="ci95_high",
            label=LABELS[method], color=COLORS[method]
        )
    axes[0].set_ylabel("Raw episodic return")
    phase = str(manifest.get("phase", "study"))
    seed_count = len(manifest.get("seeds", ()))
    if phase == "reporting":
        title = f"Frozen reporting robustness profile ({seed_count} training seeds)"
    elif phase == "development":
        title = f"Development robustness profile ({seed_count} training seeds)"
    else:
        title = f"{phase.capitalize()} robustness profile ({seed_count} training seeds)"
    axes[0].set_title(title)
    axes[0].legend(ncol=2, frameon=False)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Paired method − nominal")
    axes[1].set_xlabel("Actuator reversal probability $p$")
    axes[1].set_xlim(0.0, 0.50)
    figure.tight_layout()
    for extension in ("pdf", "png"):
        figure.savefig(figures / f"robustness_profiles.{extension}", bbox_inches="tight")
    plt.close(figure)


def _student_t_critical(count: int) -> float:
    values = {
        2: 12.7062047364,
        3: 4.3026527297,
        4: 3.1824463053,
        5: 2.7764451052,
        6: 2.5705818356,
        7: 2.4469118511,
        8: 2.3646242510,
        9: 2.3060041352,
        10: 2.2621571629,
    }
    return values.get(count, 1.96)


def _mean_ci95(values: np.ndarray) -> tuple[float, float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(finite))
    if finite.size < 2:
        return mean, float("nan"), float("nan")
    half_width = float(
        _student_t_critical(int(finite.size))
        * np.std(finite, ddof=1)
        / np.sqrt(finite.size)
    )
    return mean, mean - half_width, mean + half_width


def _profile_auc_rows(
    rows: pd.DataFrame,
    *,
    value_column: str,
    lower_probability: float,
    upper_probability: float,
) -> pd.DataFrame:
    selected = rows[
        (rows["fault_probability"] >= lower_probability)
        & (rows["fault_probability"] <= upper_probability)
    ]
    areas: list[dict[str, float | int]] = []
    for seed, profile in selected.groupby("seed"):
        profile = profile.sort_values("fault_probability")
        probabilities = profile["fault_probability"].to_numpy(dtype=np.float64)
        values = profile[value_column].to_numpy(dtype=np.float64)
        if (
            probabilities.size < 2
            or not np.isclose(probabilities[0], lower_probability)
            or not np.isclose(probabilities[-1], upper_probability)
        ):
            continue
        areas.append(
            {
                "seed": int(seed),
                "profile_auc": float(np.trapezoid(values, x=probabilities)),
            }
        )
    return pd.DataFrame(areas, columns=("seed", "profile_auc"))


def _profile_auc_by_seed(
    rows: pd.DataFrame,
    *,
    value_column: str,
    lower_probability: float,
    upper_probability: float,
) -> np.ndarray:
    return _profile_auc_rows(
        rows,
        value_column=value_column,
        lower_probability=lower_probability,
        upper_probability=upper_probability,
    )["profile_auc"].to_numpy(dtype=np.float64)


def plot_checkpoint_learning_curves(study_root: Path, figures: Path) -> None:
    manifest = _read_json(study_root / "manifest.json")
    frames: list[pd.DataFrame] = []
    for seed in manifest["seeds"]:
        path = (
            study_root
            / "raw"
            / f"seed_{int(seed):04d}"
            / "checkpoint_evaluation_summary.csv"
        )
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame["seed"] = int(seed)
        frames.append(frame)
    if not frames:
        return
    checkpoints = pd.concat(frames, ignore_index=True)
    checkpoints = checkpoints[
        np.isclose(checkpoints["fault_probability"], 0.10)
        | np.isclose(checkpoints["fault_probability"], 0.25)
    ]
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), sharex=True)
    for axis, probability in zip(axes, (0.10, 0.25)):
        selected = checkpoints[np.isclose(checkpoints["fault_probability"], probability)]
        for method in ("nominal", "exact", "affine", "full_nn"):
            rows = selected[selected["method"] == method]
            if rows.empty:
                continue
            grouped = rows.groupby("checkpoint_block")["mean_raw_return"]
            blocks = np.asarray(sorted(grouped.groups), dtype=np.int64)
            means: list[float] = []
            lows: list[float] = []
            highs: list[float] = []
            for block in blocks:
                values = grouped.get_group(block).to_numpy(dtype=np.float64)
                mean = float(np.mean(values))
                half = (
                    0.0
                    if values.size < 2
                    else float(
                        _student_t_critical(int(values.size))
                        * np.std(values, ddof=1)
                        / np.sqrt(values.size)
                    )
                )
                means.append(mean)
                lows.append(mean - half)
                highs.append(mean + half)
            axis.plot(
                blocks,
                means,
                marker="o",
                color=COLORS[method],
                label=LABELS[method],
            )
            if len(frames) > 1:
                axis.fill_between(blocks, lows, highs, color=COLORS[method], alpha=0.14)
        axis.set_title(f"Evaluation at p={probability:.2f}")
        axis.set_xlabel("Outer block")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Raw episodic return")
    axes[0].legend(frameon=False)
    figure.suptitle("Fixed-checkpoint learning curves")
    figure.tight_layout()
    for extension in ("pdf", "png"):
        figure.savefig(figures / f"checkpoint_learning_curves.{extension}", bbox_inches="tight")
    plt.close(figure)


def _network_from_checkpoint(config: dict[str, Any], state: dict[str, torch.Tensor]) -> QNetwork:
    if config["task"] == "lqr":
        observation_dim, scale = 2, (2.0, 2.0)
    else:
        observation_dim, scale = 3, (1.0, 1.0, 8.0)
    network = QNetwork(
        observation_dim,
        3,
        tuple(config["hidden_dims"]),
        scale,
    )
    network.load_state_dict(state)
    network.eval()
    return network


def _policy_grid(task: str, points: int = 121) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
    if task == "lqr":
        horizontal = np.linspace(-2.0, 2.0, points)
        vertical = np.linspace(-2.0, 2.0, points)
        x, y = np.meshgrid(horizontal, vertical, indexing="xy")
        observations = np.column_stack((x.reshape(-1), y.reshape(-1))).astype(np.float32)
        return x, y, observations, "Position", "Velocity"
    horizontal = np.linspace(-np.pi, np.pi, points)
    vertical = np.linspace(-8.0, 8.0, points)
    x, y = np.meshgrid(horizontal, vertical, indexing="xy")
    observations = np.column_stack(
        (np.cos(x).reshape(-1), np.sin(x).reshape(-1), y.reshape(-1))
    ).astype(np.float32)
    return x, y, observations, r"Angle $\theta$", "Angular velocity"


def _occupancy_xy(task: str, observations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if task == "lqr":
        return observations[:, 0], observations[:, 1]
    return np.arctan2(observations[:, 1], observations[:, 0]), observations[:, 2]


def plot_mechanism(study_root: Path, figures: Path, representative_seed: int) -> None:
    run = study_root / "raw" / f"seed_{representative_seed:04d}"
    config = _read_json(run / "config.json")
    checkpoint = torch.load(run / "checkpoints.pt", map_location="cpu", weights_only=False)
    arrays = np.load(run / "backup_calibration.npz")
    summary = _read_json(run / "summary.json")
    mechanism_methods = ("nominal", "affine", "full_nn")
    networks = {
        method: _network_from_checkpoint(config, checkpoint["methods"][method]["q"])
        for method in mechanism_methods
    }
    x, y, observations, xlabel, ylabel = _policy_grid(config["task"])
    with torch.no_grad():
        tensor = torch.as_tensor(observations)
        policy_actions = {
            method: torch.argmax(network(tensor), dim=1).numpy().reshape(x.shape)
            for method, network in networks.items()
        }
    affine_disagreement = policy_actions["nominal"] != policy_actions["affine"]
    full_nn_disagreement = policy_actions["nominal"] != policy_actions["full_nn"]
    occupancy = arrays["nominal__occupancy_observations"]
    occupancy_x, occupancy_y = _occupancy_xy(config["task"], occupancy)

    figure, axes = plt.subplots(2, 3, figsize=(12.0, 7.3))
    panels = (
        (axes[0, 0], policy_actions["nominal"], "Nominal DDQN greedy action"),
        (
            axes[0, 1],
            policy_actions["affine"],
            "RV$\\chi^2$-A (affine auxiliary)\nGreedy action",
        ),
        (
            axes[0, 2],
            policy_actions["full_nn"],
            "RV$\\chi^2$-N (full-NN auxiliary)\nGreedy action",
        ),
        (
            axes[1, 0],
            affine_disagreement.astype(float),
            r"RV$\chi^2$-A vs. nominal disagreement",
        ),
        (
            axes[1, 1],
            full_nn_disagreement.astype(float),
            r"RV$\chi^2$-N vs. nominal disagreement",
        ),
    )
    for axis, values, title in panels:
        axis.pcolormesh(x, y, values, shading="auto", cmap="viridis")
        if occupancy_x.size > 10:
            histogram, x_edges, y_edges = np.histogram2d(occupancy_x, occupancy_y, bins=35)
            positive = histogram[histogram > 0]
            if positive.size:
                levels = np.unique(np.quantile(positive, [0.50, 0.80, 0.95]))
                if levels.size:
                    axis.contour(
                        0.5 * (x_edges[:-1] + x_edges[1:]),
                        0.5 * (y_edges[:-1] + y_edges[1:]),
                        histogram.T,
                        levels=levels,
                        colors="white",
                        linewidths=0.7,
                    )
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)

    calibration_axis = axes[1, 2]
    for method, marker in (("affine", "o"), ("full_nn", "x")):
        if f"{method}__exact" not in arrays or f"{method}__learned" not in arrays:
            continue
        exact = arrays[f"{method}__exact"]
        learned = arrays[f"{method}__learned"]
        statistics = (
            summary.get("backup_calibration", {})
            .get(method, {})
            .get("by_support", {})
            .get("supported", {})
        )
        pearson = statistics.get("pearson")
        normalized_mae = statistics.get("normalized_mae")
        annotation = LABELS[method]
        if pearson is not None and normalized_mae is not None:
            annotation += f" (r={float(pearson):.3f}, NMAE={float(normalized_mae):.3f})"
        stride = max(1, exact.size // 2_000)
        calibration_axis.scatter(
            exact[::stride], learned[::stride], s=9, alpha=0.35,
            marker=marker, color=COLORS[method], label=annotation
        )
    limits = calibration_axis.get_xlim()
    lower = min(limits[0], calibration_axis.get_ylim()[0])
    upper = max(limits[1], calibration_axis.get_ylim()[1])
    calibration_axis.plot([lower, upper], [lower, upper], color="black", linewidth=0.8)
    calibration_axis.set_xlim(lower, upper)
    calibration_axis.set_ylim(lower, upper)
    calibration_axis.set_xlabel("Exact two-mode robust continuation")
    calibration_axis.set_ylabel("Learned variational continuation")
    calibration_axis.set_title("Held-out backup calibration")
    calibration_axis.legend(frameon=False, fontsize=8)
    figure.suptitle(
        f"State-dependent mechanism (training seed {representative_seed})",
        y=0.985,
    )
    figure.subplots_adjust(
        left=0.065,
        right=0.985,
        bottom=0.08,
        top=0.86,
        wspace=0.30,
        hspace=0.42,
    )
    for extension in ("pdf", "png"):
        figure.savefig(figures / f"state_policy_and_calibration_seed_{representative_seed}.{extension}", bbox_inches="tight")
    plt.close(figure)


def _save_figure_pair(figure: plt.Figure, figures: Path, stem: str) -> list[str]:
    files: list[str] = []
    for extension in ("pdf", "png"):
        filename = f"{stem}.{extension}"
        figure.savefig(figures / filename, bbox_inches="tight")
        files.append(filename)
    plt.close(figure)
    return files


def plot_seed_paired_effects(study_root: Path, figures: Path) -> list[str]:
    path = study_root / "aggregated" / "paired_by_seed.csv"
    if not path.is_file():
        return []
    paired = pd.read_csv(path)
    manifest = _read_json(study_root / "manifest.json")
    methods = [
        method
        for method in ("exact", "affine", "full_nn")
        if not paired[paired["method"] == method].empty
    ]
    if not methods:
        return []

    panels: list[tuple[str, str, dict[str, pd.DataFrame]]] = []
    for probability, title in ((0.10, r"Nominal kernel $p_0=0.10$"), (0.25, r"Boundary $p=0.25$")):
        values = {
            method: paired[
                (paired["method"] == method)
                & np.isclose(paired["fault_probability"], probability)
            ][["seed", "paired_return_advantage"]].rename(
                columns={"paired_return_advantage": "effect"}
            )
            for method in methods
        }
        panels.append((title, "Paired return advantage", values))
    auc_values = {
        method: _profile_auc_rows(
            paired[paired["method"] == method],
            value_column="paired_return_advantage",
            lower_probability=0.10,
            upper_probability=0.25,
        ).rename(columns={"profile_auc": "effect"})
        for method in methods
    }
    panels.append((r"Paired AUC, $p\in[0.10,0.25]$", "Paired advantage AUC", auc_values))

    all_seeds = sorted(int(seed) for seed in paired["seed"].unique())
    figure, axes = plt.subplots(1, 3, figsize=(9.0, 3.0))
    offsets = np.linspace(-0.10, 0.10, len(methods)) if len(methods) > 1 else np.zeros(1)
    for axis, (title, ylabel, values_by_method) in zip(axes, panels):
        axis.axhline(0.0, color="black", linewidth=0.8)
        for offset, method in zip(offsets, methods):
            values = values_by_method[method].dropna().sort_values("seed")
            if values.empty:
                continue
            seeds = values["seed"].to_numpy(dtype=np.float64)
            effects = values["effect"].to_numpy(dtype=np.float64)
            mean, low, high = _mean_ci95(effects)
            axis.scatter(
                seeds + offset,
                effects,
                s=24,
                color=COLORS[method],
                alpha=0.78,
                label=LABELS[method],
                zorder=3,
            )
            axis.plot(
                [seeds.min() - 0.25, seeds.max() + 0.25],
                [mean, mean],
                color=COLORS[method],
                linewidth=1.4,
            )
            if np.isfinite(low) and np.isfinite(high):
                axis.fill_between(
                    [seeds.min() - 0.25, seeds.max() + 0.25],
                    [low, low],
                    [high, high],
                    color=COLORS[method],
                    alpha=0.10,
                    zorder=0,
                )
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(all_seeds)
        axis.tick_params(axis="x", rotation=45)
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    if len(all_seeds) == 1:
        figure.suptitle(
            f"Paired effects for {manifest.get('phase', 'study')} seed {all_seeds[0]} "
            "(95% CI unavailable for n=1)"
        )
    else:
        figure.suptitle(
            f"Individual {manifest.get('phase', 'study')} seed paired effects "
            "(line/band: mean and 95% CI)"
        )
    figure.tight_layout()
    return _save_figure_pair(figure, figures, "appendix_seed_paired_effects")


def _plot_seed_strata(
    axis: plt.Axes,
    summaries: pd.DataFrame,
    *,
    category: str,
    categories: list[Any],
    labels: list[str],
) -> None:
    for position, (value, label) in enumerate(zip(categories, labels)):
        selected = summaries[summaries[category] == value].sort_values("seed")
        errors = selected["absolute_error"].to_numpy(dtype=np.float64)
        if errors.size == 0:
            continue
        jitter = np.linspace(-0.12, 0.12, errors.size) if errors.size > 1 else np.zeros(1)
        mean, low, high = _mean_ci95(errors)
        axis.scatter(
            position + jitter,
            errors,
            s=22,
            color="#777777",
            alpha=0.75,
            zorder=2,
        )
        error = None
        if np.isfinite(low) and np.isfinite(high):
            error = np.asarray([[mean - low], [high - mean]])
        axis.errorbar(
            position,
            mean,
            yerr=error,
            fmt="D",
            markersize=5,
            capsize=3,
            color=COLORS["affine"],
            zorder=3,
        )
    axis.set_xticks(range(len(labels)), labels)
    axis.grid(axis="y", alpha=0.2)


def plot_backup_error_diagnostics(study_root: Path, figures: Path) -> list[str]:
    manifest = _read_json(study_root / "manifest.json")
    frames: list[pd.DataFrame] = []
    required = {
        "affine__actions",
        "affine__exact",
        "affine__learned",
        "affine__policy_margins",
        "affine__support_counts",
    }
    for seed in manifest["seeds"]:
        path = study_root / "raw" / f"seed_{int(seed):04d}" / "backup_calibration.npz"
        if not path.is_file():
            continue
        with np.load(path) as arrays:
            if not required.issubset(arrays.files):
                continue
            exact = arrays["affine__exact"].astype(np.float64, copy=False)
            learned = arrays["affine__learned"].astype(np.float64, copy=False)
            frames.append(
                pd.DataFrame(
                    {
                        "seed": int(seed),
                        "action": arrays["affine__actions"].astype(np.int64, copy=False),
                        "supported": arrays["affine__support_counts"] > 0,
                        "policy_margin": arrays["affine__policy_margins"].astype(
                            np.float64, copy=False
                        ),
                        "absolute_error": np.abs(learned - exact),
                    }
                )
            )
    if not frames:
        return []
    calibration = pd.concat(frames, ignore_index=True)

    action_summary = (
        calibration.groupby(["seed", "action"], as_index=False)["absolute_error"].mean()
    )
    support_summary = (
        calibration.groupby(["seed", "supported"], as_index=False)["absolute_error"].mean()
    )
    quantile_edges = np.unique(
        np.quantile(calibration["policy_margin"].to_numpy(dtype=np.float64), np.linspace(0, 1, 9))
    )
    if quantile_edges.size >= 3:
        calibration["margin_bin"] = np.digitize(
            calibration["policy_margin"], quantile_edges[1:-1], right=True
        )
    else:
        calibration["margin_bin"] = 0
    margin_summary = (
        calibration.groupby(["seed", "margin_bin"], as_index=False)["absolute_error"].mean()
    )
    margin_locations = calibration.groupby("margin_bin")["policy_margin"].median()

    figure, axes = plt.subplots(1, 3, figsize=(9.0, 3.0))
    _plot_seed_strata(
        axes[0],
        action_summary,
        category="action",
        categories=[0, 1, 2],
        labels=["Action 0", "Action 1", "Action 2"],
    )
    axes[0].set_ylabel(r"Mean $|\widehat{B}-B_{exact}|$")
    axes[0].set_title("Error by action")
    _plot_seed_strata(
        axes[1],
        support_summary,
        category="supported",
        categories=[True, False],
        labels=["Supported", "Unsupported"],
    )
    axes[1].set_ylabel(r"Mean $|\widehat{B}-B_{exact}|$")
    axes[1].set_title("Error by replay support")

    margin_axis = axes[2]
    for margin_bin, selected in margin_summary.groupby("margin_bin"):
        errors = selected["absolute_error"].to_numpy(dtype=np.float64)
        location = float(margin_locations.loc[margin_bin])
        margin_axis.scatter(
            np.full(errors.size, location),
            errors,
            s=12,
            color="#777777",
            alpha=0.35,
        )
    bins = sorted(int(value) for value in margin_summary["margin_bin"].unique())
    locations: list[float] = []
    means: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    for margin_bin in bins:
        errors = margin_summary[margin_summary["margin_bin"] == margin_bin][
            "absolute_error"
        ].to_numpy(dtype=np.float64)
        mean, low, high = _mean_ci95(errors)
        locations.append(float(margin_locations.loc[margin_bin]))
        means.append(mean)
        lows.append(low)
        highs.append(high)
    margin_axis.plot(locations, means, marker="o", color=COLORS["affine"])
    if np.all(np.isfinite(lows)) and np.all(np.isfinite(highs)):
        margin_axis.fill_between(locations, lows, highs, color=COLORS["affine"], alpha=0.14)
    if locations and min(locations) > 0.0:
        margin_axis.set_xscale("log")
    margin_axis.set_xlabel(r"Policy margin $Q_{(1)}-Q_{(2)}$")
    margin_axis.set_ylabel(r"Mean $|\widehat{B}-B_{exact}|$")
    margin_axis.set_title("Error vs. policy margin")
    margin_axis.grid(alpha=0.2)
    seeds = sorted(int(seed) for seed in calibration["seed"].unique())
    if len(seeds) == 1:
        figure.suptitle(
            f"Held-out affine backup error for {manifest.get('phase', 'study')} seed "
            f"{seeds[0]} (seed CI unavailable for n=1)"
        )
    else:
        figure.suptitle(
            f"Held-out affine backup error across {len(seeds)} "
            f"{manifest.get('phase', 'study')} seeds "
            "(dots: seed means; bars/bands: 95% CI)"
        )
    figure.tight_layout()
    return _save_figure_pair(figure, figures, "appendix_affine_backup_error")


def _plot_block_metric(
    axis: plt.Axes,
    rows: pd.DataFrame,
    *,
    method: str,
    value_column: str,
    label: str,
    color: str,
    linestyle: str = "-",
    marker: str | None = None,
    shade: bool = True,
) -> bool:
    if value_column not in rows.columns:
        return False
    selected = rows[rows["method"] == method][["block", "seed", value_column]].dropna()
    if selected.empty:
        return False
    blocks: list[int] = []
    means: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    for block, values in selected.groupby("block"):
        mean, low, high = _mean_ci95(values[value_column].to_numpy(dtype=np.float64))
        blocks.append(int(block))
        means.append(mean)
        lows.append(low)
        highs.append(high)
    order = np.argsort(blocks)
    x = np.asarray(blocks, dtype=np.int64)[order]
    y = np.asarray(means, dtype=np.float64)[order]
    low_values = np.asarray(lows, dtype=np.float64)[order]
    high_values = np.asarray(highs, dtype=np.float64)[order]
    axis.plot(
        x,
        y,
        color=color,
        linestyle=linestyle,
        marker=marker,
        markevery=max(1, len(x) // 6) if marker is not None else None,
        linewidth=1.4,
        label=label,
    )
    if shade and np.all(np.isfinite(low_values)) and np.all(np.isfinite(high_values)):
        axis.fill_between(x, low_values, high_values, color=color, alpha=0.10)
    return True


def _mark_unavailable(axis: plt.Axes, available: bool) -> None:
    if not available:
        axis.text(
            0.5,
            0.5,
            "Unavailable in saved artifacts",
            ha="center",
            va="center",
            transform=axis.transAxes,
            color="#777777",
        )


def plot_auxiliary_health(study_root: Path, figures: Path) -> list[str]:
    manifest = _read_json(study_root / "manifest.json")
    frames: list[pd.DataFrame] = []
    for seed in manifest["seeds"]:
        path = study_root / "raw" / f"seed_{int(seed):04d}" / "learning_metrics.csv"
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        if frame.empty or not {"phase", "method", "block"}.issubset(frame.columns):
            continue
        frame = frame[frame["phase"] == "robust_outer"].copy()
        if frame.empty:
            continue
        frame["seed"] = int(seed)
        frames.append(frame)
    if not frames:
        return []
    learning = pd.concat(frames, ignore_index=True)
    auxiliary_methods = [
        method
        for method in ("affine", "full_nn")
        if not learning[learning["method"] == method].empty
    ]
    if not auxiliary_methods:
        return []

    figure, axes = plt.subplots(2, 2, figsize=(8.2, 6.0), sharex=True)

    eta_axis = axes[0, 0]
    eta_available = False
    for method in auxiliary_methods:
        if method == "full_nn":
            mean_column = "auxiliary_eta_saturation_fraction"
            max_column = "auxiliary_eta_saturation_fraction_max"
            constraint_label = "saturation"
        else:
            mean_column = "auxiliary_eta_projected_action_fraction"
            max_column = "auxiliary_eta_projected_action_fraction_max"
            constraint_label = "projection"
        eta_available |= _plot_block_metric(
            eta_axis,
            learning,
            method=method,
            value_column=mean_column,
            label=f"{LABELS[method]} mean {constraint_label}",
            color=COLORS[method],
        )
        eta_available |= _plot_block_metric(
            eta_axis,
            learning,
            method=method,
            value_column=max_column,
            label=f"{LABELS[method]} block max",
            color=COLORS[method],
            linestyle=":",
            shade=False,
        )
    eta_axis.set_ylabel("Fraction")
    eta_axis.set_title(r"$\eta$ constraint activity")
    eta_axis.set_ylim(bottom=0.0)
    eta_axis.grid(alpha=0.2)
    _mark_unavailable(eta_axis, eta_available)
    if eta_available:
        eta_columns = [
            column
            for column in (
                "auxiliary_eta_projected_action_fraction",
                "auxiliary_eta_projected_action_fraction_max",
                "auxiliary_eta_saturation_fraction",
                "auxiliary_eta_saturation_fraction_max",
            )
            if column in learning.columns
        ]
        eta_values = learning[eta_columns].to_numpy(dtype=np.float64)
        finite_eta = eta_values[np.isfinite(eta_values)]
        if finite_eta.size and np.max(np.abs(finite_eta)) == 0.0:
            eta_axis.set_ylim(0.0, 0.01)
            eta_axis.set_yticks([0.0])
            eta_axis.text(
                0.5,
                0.5,
                "No eta constraint activity observed",
                ha="center",
                va="center",
                transform=eta_axis.transAxes,
                color="#555555",
            )
        eta_axis.legend(frameon=False, fontsize=7)

    u_axis = axes[0, 1]
    u_available = False
    quantiles = (
        ("auxiliary_u_p10", "p10", "--"),
        ("auxiliary_u_p50", "p50", "-"),
        ("auxiliary_u_p90", "p90", "-."),
    )
    for method in auxiliary_methods:
        for column, quantile, linestyle in quantiles:
            u_available |= _plot_block_metric(
                u_axis,
                learning,
                method=method,
                value_column=column,
                label=f"{LABELS[method]} u {quantile}",
                color=COLORS[method],
                linestyle=linestyle,
                shade=quantile == "p50",
            )
    u_axis.set_ylabel("Auxiliary u")
    u_axis.set_title("u quantiles and floor activity")
    u_axis.set_ylim(bottom=0.0)
    u_axis.grid(alpha=0.2)
    floor_axis = u_axis.twinx()
    floor_available = False
    for method in auxiliary_methods:
        floor_available |= _plot_block_metric(
            floor_axis,
            learning,
            method=method,
            value_column="auxiliary_u_floor_fraction",
            label=f"{LABELS[method]} u-floor fraction",
            color="#6a3d9a" if method == "affine" else COLORS[method],
            linestyle=":",
            shade=False,
        )
    floor_axis.set_ylabel("u-floor fraction")
    floor_axis.set_ylim(bottom=0.0)
    floor_axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    _mark_unavailable(u_axis, u_available or floor_available)
    handles, labels = u_axis.get_legend_handles_labels()
    floor_handles, floor_labels = floor_axis.get_legend_handles_labels()
    if handles or floor_handles:
        u_axis.legend(handles + floor_handles, labels + floor_labels, frameon=False, fontsize=7)

    gradient_axis = axes[1, 0]
    gradient_available = False
    for method in auxiliary_methods:
        gradient_available |= _plot_block_metric(
            gradient_axis,
            learning,
            method=method,
            value_column="auxiliary_gradient_norm",
            label=f"{LABELS[method]} mean",
            color=COLORS[method],
        )
        gradient_available |= _plot_block_metric(
            gradient_axis,
            learning,
            method=method,
            value_column="auxiliary_gradient_norm_max",
            label=f"{LABELS[method]} block max",
            color=COLORS[method],
            linestyle=":",
            shade=False,
        )
    gradient_axis.set_xlabel("Outer block")
    gradient_axis.set_ylabel("Gradient norm")
    gradient_axis.set_title("Auxiliary gradient health")
    gradient_axis.grid(alpha=0.2)
    if gradient_available:
        positive = learning[
            [
                column
                for column in ("auxiliary_gradient_norm", "auxiliary_gradient_norm_max")
                if column in learning.columns
            ]
        ].to_numpy(dtype=np.float64)
        if positive.size and np.all(positive[np.isfinite(positive)] > 0.0):
            gradient_axis.set_yscale("log")
        gradient_axis.legend(frameon=False, fontsize=7)
    _mark_unavailable(gradient_axis, gradient_available)

    clipping_axis = axes[1, 1]
    clipping_available = False
    for method in ("nominal", "exact", "affine", "full_nn"):
        if learning[learning["method"] == method].empty:
            continue
        clipping_available |= _plot_block_metric(
            clipping_axis,
            learning,
            method=method,
            value_column="target_clip_fraction",
            label=LABELS[method],
            color=COLORS[method],
        )
    clipping_axis.set_xlabel("Outer block")
    clipping_axis.set_ylabel("Target clip fraction")
    clipping_axis.set_title("Bellman-target clipping")
    clipping_axis.set_ylim(bottom=0.0)
    clipping_axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    clipping_axis.grid(alpha=0.2)
    _mark_unavailable(clipping_axis, clipping_available)
    if clipping_available:
        clip_values = learning["target_clip_fraction"].to_numpy(dtype=np.float64)
        finite_clipping = clip_values[np.isfinite(clip_values)]
        if finite_clipping.size and np.max(np.abs(finite_clipping)) == 0.0:
            clipping_axis.set_ylim(0.0, 0.01)
            clipping_axis.set_yticks([0.0])
            clipping_axis.text(
                0.5,
                0.5,
                "No target clipping observed",
                ha="center",
                va="center",
                transform=clipping_axis.transAxes,
                color="#555555",
            )
        clipping_axis.legend(frameon=False, fontsize=7)

    seeds = sorted(int(seed) for seed in learning["seed"].unique())
    if len(seeds) == 1:
        figure.suptitle(
            f"Auxiliary and target health for {manifest.get('phase', 'study')} seed {seeds[0]}"
        )
    else:
        figure.suptitle(
            f"Auxiliary and target health across {len(seeds)} "
            f"{manifest.get('phase', 'study')} seeds (mean and 95% CI)"
        )
    figure.tight_layout()
    return _save_figure_pair(figure, figures, "appendix_auxiliary_health")


def write_main_table(study_root: Path) -> None:
    final = pd.read_csv(study_root / "aggregated" / "final_by_seed.csv")
    profile = pd.read_csv(study_root / "aggregated" / "robustness_profile.csv")
    paired = pd.read_csv(study_root / "aggregated" / "paired_by_seed.csv")
    paired_profile = pd.read_csv(study_root / "aggregated" / "paired_profile.csv")

    def profile_row(frame: pd.DataFrame, method: str, probability: float) -> pd.Series:
        selected = frame[
            (frame["method"] == method)
            & np.isclose(frame["fault_probability"], probability)
        ]
        if len(selected) != 1:
            raise ValueError(
                f"Expected one aggregate row for {method=} and {probability=}; "
                f"found {len(selected)}."
            )
        return selected.iloc[0]

    rows: list[dict[str, Any]] = []
    for method in ("nominal", "exact", "affine", "full_nn"):
        subset = final[final["method"] == method]
        if subset.empty:
            continue
        p0 = profile_row(profile, method, 0.10)
        boundary = profile_row(profile, method, 0.25)
        certified_auc = _profile_auc_by_seed(
            subset,
            value_column="mean_raw_return",
            lower_probability=0.0,
            upper_probability=0.25,
        )
        certified_auc_mean, certified_auc_low, certified_auc_high = _mean_ci95(
            certified_auc
        )
        if method == "nominal":
            paired_p0_mean = paired_p0_low = paired_p0_high = 0.0
            paired_boundary_mean = paired_boundary_low = paired_boundary_high = 0.0
            paired_auc = np.zeros(int(subset["seed"].nunique()), dtype=np.float64)
        else:
            paired_p0 = profile_row(paired_profile, method, 0.10)
            paired_boundary = profile_row(paired_profile, method, 0.25)
            paired_p0_mean = float(paired_p0["mean"])
            paired_p0_low = float(paired_p0["ci95_low"])
            paired_p0_high = float(paired_p0["ci95_high"])
            paired_boundary_mean = float(paired_boundary["mean"])
            paired_boundary_low = float(paired_boundary["ci95_low"])
            paired_boundary_high = float(paired_boundary["ci95_high"])
            paired_auc = _profile_auc_by_seed(
                paired[paired["method"] == method],
                value_column="paired_return_advantage",
                lower_probability=0.10,
                upper_probability=0.25,
            )
        paired_auc_mean, paired_auc_low, paired_auc_high = _mean_ci95(paired_auc)
        disagreement: list[float] = []
        environment_steps: list[int] = []
        q_updates: list[int] = []
        auxiliary_updates: list[int] = []
        for seed in sorted(subset["seed"].unique()):
            summary = _read_json(study_root / "raw" / f"seed_{int(seed):04d}" / "summary.json")
            config = _read_json(study_root / "raw" / f"seed_{int(seed):04d}" / "config.json")
            replay_support = summary.get("replay_support", {}).get(method, {})
            disagreement.append(
                float(replay_support.get("occupied_policy_disagreement_with_nominal", np.nan))
            )
            environment_steps.append(config["nominal_pretrain_steps"] + config["outer_blocks"] * config["collection_steps_per_block"])
            q_updates.append(summary["nominal_pretraining_q_updates"] + config["outer_blocks"] * config["q_updates_per_block"])
            auxiliary_updates.append(
                config["outer_blocks"] * config["auxiliary_updates_per_block"]
                if method in {"affine", "full_nn"}
                else 0
            )
        rows.append(
            {
                "method": LABELS[method],
                "seed_count": int(subset["seed"].nunique()),
                "p0_fault_probability": 0.10,
                "p0_raw_return_mean": float(p0["mean"]),
                "p0_raw_return_ci95_low": float(p0["ci95_low"]),
                "p0_raw_return_ci95_high": float(p0["ci95_high"]),
                "boundary_fault_probability": 0.25,
                "boundary_raw_return_mean": float(boundary["mean"]),
                "boundary_raw_return_ci95_low": float(boundary["ci95_low"]),
                "boundary_raw_return_ci95_high": float(boundary["ci95_high"]),
                "certified_profile_auc_mean": certified_auc_mean,
                "certified_profile_auc_ci95_low": certified_auc_low,
                "certified_profile_auc_ci95_high": certified_auc_high,
                "paired_p0_advantage_method_minus_nominal_mean": paired_p0_mean,
                "paired_p0_advantage_method_minus_nominal_ci95_low": paired_p0_low,
                "paired_p0_advantage_method_minus_nominal_ci95_high": paired_p0_high,
                "paired_boundary_advantage_method_minus_nominal_mean": paired_boundary_mean,
                "paired_boundary_advantage_method_minus_nominal_ci95_low": paired_boundary_low,
                "paired_boundary_advantage_method_minus_nominal_ci95_high": paired_boundary_high,
                "paired_advantage_auc_p010_to_p025_mean": paired_auc_mean,
                "paired_advantage_auc_p010_to_p025_ci95_low": paired_auc_low,
                "paired_advantage_auc_p010_to_p025_ci95_high": paired_auc_high,
                "occupied_policy_disagreement_mean": float(np.mean(disagreement)),
                "environment_steps": int(environment_steps[0]),
                "q_updates": int(q_updates[0]),
                "auxiliary_updates": int(auxiliary_updates[0]),
                "total_gradient_updates": int(q_updates[0] + auxiliary_updates[0]),
            }
        )
    write_csv_atomic(study_root / "aggregated" / "main_table.csv", rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_root", type=Path)
    parser.add_argument("--representative-seed", type=int, default=None)
    args = parser.parse_args()
    _style()
    manifest = _read_json(args.study_root / "manifest.json")
    seed = args.representative_seed or int(manifest["seeds"][0])
    figures = args.study_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plot_profiles(args.study_root, figures)
    plot_checkpoint_learning_curves(args.study_root, figures)
    plot_mechanism(args.study_root, figures, seed)
    appendix_files = [
        *plot_seed_paired_effects(args.study_root, figures),
        *plot_backup_error_diagnostics(args.study_root, figures),
        *plot_auxiliary_health(args.study_root, figures),
    ]
    write_main_table(args.study_root)
    write_json_atomic(
        figures / "figure_manifest.json",
        {
            "representative_seed": seed,
            "certified_interval": [0.0, 0.25],
            "ood_stress_probabilities": [0.35, 0.50],
            "appendix_diagnostic_files": appendix_files,
            "files": sorted(
                path.name
                for path in figures.iterdir()
                if path.is_file() and path.name != "figure_manifest.json"
            ),
        },
    )


if __name__ == "__main__":
    main()
