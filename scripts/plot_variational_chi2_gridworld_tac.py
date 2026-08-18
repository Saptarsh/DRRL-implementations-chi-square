#!/usr/bin/env python3
"""Build the compact TAC tabular figure from completed MiniCliff artifacts.

This is a plotting-only postprocessor.  It reads the 20-seed floor sweep for
the multi-floor convergence panel and the 25-seed main experiment for the
fixed-floor perturbation and policy panels.  It never launches training.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAIN_ROOT = REPO_ROOT / "paper_variational_chi2_gridworld" / "main"
DEFAULT_SWEEP_ROOT = REPO_ROOT / "paper_variational_chi2_gridworld" / "ell_sensitivity"


def read_manifest(root: Path) -> Dict[str, object]:
    path = root / "manifest.json"
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("status") != "complete":
        raise ValueError(f"Expected a complete manifest at {path}.")
    return manifest


def read_numeric_csv(path: Path) -> List[Dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        rows: List[Dict[str, float]] = []
        for line_number, row in enumerate(reader, start=2):
            try:
                numeric = {key: float(value) for key, value in row.items()}
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Non-numeric value in {path}:{line_number}") from exc
            if not all(math.isfinite(value) for value in numeric.values()):
                raise ValueError(f"Non-finite value in {path}:{line_number}")
            rows.append(numeric)
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def require_columns(rows: Sequence[Mapping[str, float]], columns: Sequence[str], label: str) -> None:
    missing = set(columns) - set(rows[0])
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def resolved(manifest: Mapping[str, object]) -> Mapping[str, object]:
    value = manifest.get("resolved")
    if not isinstance(value, dict):
        raise ValueError("Manifest has no resolved configuration.")
    return value


def fixed_fields(manifest: Mapping[str, object]) -> Mapping[str, object]:
    value = resolved(manifest).get("algorithm_fixed_fields")
    if not isinstance(value, dict):
        raise ValueError("Manifest has no resolved algorithm configuration.")
    return value


def validate_studies(
    main_manifest: Mapping[str, object], sweep_manifest: Mapping[str, object]
) -> Tuple[float, float, float, int, int]:
    main_resolved = resolved(main_manifest)
    sweep_resolved = resolved(sweep_manifest)
    if main_resolved.get("environment") != sweep_resolved.get("environment"):
        raise ValueError("Main and floor-sweep environment configurations differ.")
    if fixed_fields(main_manifest) != fixed_fields(sweep_manifest):
        raise ValueError("Main and floor-sweep fixed algorithm configurations differ.")

    focus_ell = float(main_resolved["focus_ell"])
    main_ells = tuple(float(value) for value in main_resolved["ells"])
    sweep_ells = tuple(float(value) for value in sweep_resolved["ells"])
    if len(main_ells) != 1 or not math.isclose(main_ells[0], focus_ell):
        raise ValueError("The main study must contain only its focus floor.")
    if not any(math.isclose(value, focus_ell) for value in sweep_ells):
        raise ValueError("The focus floor is absent from the floor sweep.")

    environment = main_resolved["environment"]
    if not isinstance(environment, dict):
        raise ValueError("Invalid environment configuration in main manifest.")
    nominal_slip = float(environment["nominal_slip_probability"])
    delta = float(fixed_fields(main_manifest)["chi2_delta"])
    boundary = float(main_manifest["ambiguity_upper_slip_probability"])
    expected_boundary = nominal_slip + math.sqrt(
        delta * nominal_slip * (1.0 - nominal_slip)
    )
    if not math.isclose(boundary, expected_boundary, rel_tol=1e-12, abs_tol=1e-14):
        raise ValueError("The recorded ambiguity boundary is inconsistent with delta and p0.")

    main_seeds = len(tuple(main_resolved["seeds"]))
    sweep_seeds = len(tuple(sweep_resolved["seeds"]))
    return focus_ell, nominal_slip, boundary, main_seeds, sweep_seeds


def load_modal_arrays(path: Path, focus_ell: float) -> Dict[str, np.ndarray]:
    required = {
        "focus_ell",
        "nominal_oracle_policy",
        "robust_oracle_policy",
        "modal_nominal_policy",
        "modal_robust_policy",
        "modal_nominal_tie_mask",
        "modal_robust_tie_mask",
        "state_coordinates",
        "decision_state_mask",
        "oracle_separating_state_mask",
    }
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != required:
            raise ValueError(
                f"Unexpected modal-policy schema in {path}; "
                f"missing={sorted(required - set(archive.files))}, "
                f"extra={sorted(set(archive.files) - required)}"
            )
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    if not math.isclose(float(arrays["focus_ell"]), focus_ell):
        raise ValueError("Modal-policy floor does not match the main focus floor.")
    for name in (
        "nominal_oracle_policy",
        "robust_oracle_policy",
        "modal_nominal_policy",
        "modal_robust_policy",
    ):
        if arrays[name].shape != (24,):
            raise ValueError(f"{path}:{name} must have shape (24,).")
    return arrays


def ell_colors(ells: Sequence[float]) -> Dict[float, object]:
    import matplotlib.pyplot as plt

    positions = np.linspace(0.06, 0.94, len(ells))
    return {ell: plt.cm.viridis(position) for ell, position in zip(ells, positions)}


def shade_ambiguity(axis: object, nominal_slip: float, boundary: float) -> None:
    axis.axvspan(nominal_slip, boundary, color="#2ca02c", alpha=0.075, linewidth=0)
    axis.axvline(boundary, color="#2ca02c", linestyle="-.", linewidth=1.0)
    axis.text(
        boundary - 0.002,
        0.035,
        r"$p_\chi$",
        color="#208020",
        ha="right",
        va="bottom",
        transform=axis.get_xaxis_transform(),
        fontsize=6.2,
    )


def plot_convergence(axis: object, rows: Sequence[Mapping[str, float]]) -> None:
    available_ells = {float(row["ell"]) for row in rows}
    ells = [ell for ell in (0.1, 0.3, 1.0) if ell in available_ells]
    if len(ells) != 3:
        raise ValueError("The convergence panel requires ell in {0.1, 0.3, 1.0}.")
    colors = ell_colors(ells)
    for ell in ells:
        selected = sorted(
            (
                row
                for row in rows
                if math.isclose(float(row["ell"]), ell)
                and float(row["transitions"]) > 0.0
            ),
            key=lambda row: float(row["transitions"]),
        )
        x = np.asarray([row["transitions"] for row in selected], dtype=float)
        y = np.asarray([row["robust_q_sup_error_mean"] for row in selected], dtype=float)
        ci = np.asarray(
            [row["robust_q_sup_error_interval_halfwidth"] for row in selected],
            dtype=float,
        )
        axis.plot(x, y, color=colors[ell], label=rf"{ell:g}")
        axis.fill_between(
            x,
            np.maximum(y - ci, np.finfo(float).tiny),
            y + ci,
            color=colors[ell],
            alpha=0.16,
            linewidth=0,
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_ylim(0.1, 1.1)
    axis.set_xticks((1e5, 1e6, 6e6), labels=("0.1", "1", "6"))
    axis.set_xlabel("Cumulative transitions (millions)")
    axis.set_ylabel(r"$\|\widehat Q_t-Q^*_{\chi}\|_\infty$")
    axis.set_title(r"(a) Robust $Q$ convergence", loc="left", fontweight="bold")
    axis.legend(
        title=r"Floor $\ell$",
        ncol=1,
        loc="best",
        frameon=False,
        fontsize=6.2,
        title_fontsize=6.4,
        handlelength=1.7,
        labelspacing=0.3,
    )


def perturbation_arrays(
    rows: Sequence[Mapping[str, float]], focus_ell: float
) -> Tuple[List[Mapping[str, float]], np.ndarray]:
    selected = sorted(
        (row for row in rows if math.isclose(float(row["ell"]), focus_ell)),
        key=lambda row: float(row["slip_probability"]),
    )
    if not selected:
        raise ValueError(f"No perturbation rows found for ell={focus_ell:g}.")
    return selected, np.asarray([row["slip_probability"] for row in selected], dtype=float)


def plot_returns(
    axis: object,
    rows: Sequence[Mapping[str, float]],
    focus_ell: float,
    nominal_slip: float,
    boundary: float,
) -> None:
    selected, x = perturbation_arrays(rows, focus_ell)
    for metric, label, color in (
        ("robust_policy_return", "Variational robust", "#1f77b4"),
        ("nominal_policy_return", "Nominal Q-learning", "#d62728"),
    ):
        y = np.asarray([row[f"{metric}_mean"] for row in selected], dtype=float)
        ci = np.asarray(
            [row[f"{metric}_interval_halfwidth"] for row in selected], dtype=float
        )
        axis.errorbar(
            x,
            y,
            yerr=ci,
            color=color,
            marker="o",
            markersize=2.5,
            capsize=1.8,
            capthick=0.7,
            elinewidth=0.7,
            label=label,
            zorder=3,
        )
    axis.plot(
        x,
        [row["optimal_return_mean"] for row in selected],
        color="black",
        linestyle="--",
        label=r"$P_p$ optimum",
    )
    axis.plot(
        x,
        [row["oracle_robust_policy_return_mean"] for row in selected],
        color="#1f77b4",
        linestyle=":",
        linewidth=1.45,
        label="Robust oracle",
    )
    shade_ambiguity(axis, nominal_slip, boundary)
    axis.set_xlabel(r"Slip probability $p$")
    axis.set_ylabel("Start-state return")
    axis.set_title(r"(b) Return under $P_p$", loc="left", fontweight="bold")
    axis.legend(
        ncol=2,
        loc="upper right",
        frameon=False,
        fontsize=5.35,
        handlelength=1.6,
        columnspacing=0.7,
        labelspacing=0.25,
    )


def plot_gaps(
    axis: object,
    rows: Sequence[Mapping[str, float]],
    focus_ell: float,
    nominal_slip: float,
    boundary: float,
) -> None:
    selected, x = perturbation_arrays(rows, focus_ell)
    for metric, label, color in (
        ("robust_policy_gap", "Variational robust", "#1f77b4"),
        ("nominal_policy_gap", "Nominal Q-learning", "#d62728"),
    ):
        y = np.asarray([row[f"{metric}_mean"] for row in selected], dtype=float)
        ci = np.asarray(
            [row[f"{metric}_interval_halfwidth"] for row in selected], dtype=float
        )
        axis.errorbar(
            x,
            y,
            yerr=ci,
            color=color,
            marker="o",
            markersize=2.5,
            capsize=1.8,
            capthick=0.7,
            elinewidth=0.7,
            label=label,
            zorder=3,
        )
    axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.55)
    shade_ambiguity(axis, nominal_slip, boundary)
    axis.set_xlabel(r"Slip probability $p$")
    axis.set_ylabel(r"$V^*_{P_p}(s_0)-V^{\pi}_{P_p}(s_0)$")
    axis.set_title(r"(c) Gap to $P_p$ optimum", loc="left", fontweight="bold")
    axis.legend(
        loc="upper left",
        frameon=False,
        fontsize=5.5,
        handlelength=1.6,
        labelspacing=0.25,
    )


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
    start_state = int(
        np.flatnonzero(
            (coordinates[:, 0] == rows - 1) & (coordinates[:, 1] == 0)
        )[0]
    )
    for state, (row_value, column_value) in enumerate(coordinates):
        row, column = int(row_value), int(column_value)
        if state == goal_state:
            facecolor, text = "#c7e9c0", "G"
        elif not decision_mask[state]:
            facecolor, text = "#fcbba1", "C"
        else:
            facecolor = "#efe6ff" if separating_mask[state] else "white"
            text = "?" if tie_mask[state] else arrows[int(policy[state])]
        axis.add_patch(
            patches.Rectangle(
                (column, row),
                1,
                1,
                facecolor=facecolor,
                edgecolor="#666666",
                linewidth=0.48,
                hatch="///" if tie_mask[state] and decision_mask[state] else None,
            )
        )
        if separating_mask[state]:
            axis.add_patch(
                patches.Rectangle(
                    (column + 0.04, row + 0.04),
                    0.92,
                    0.92,
                    fill=False,
                    edgecolor="#b218b2",
                    linewidth=1.0,
                )
            )
        if state == start_state:
            axis.add_patch(
                patches.Rectangle(
                    (column + 0.08, row + 0.08),
                    0.84,
                    0.84,
                    fill=False,
                    edgecolor="#2166ac",
                    linewidth=0.85,
                    linestyle="--",
                )
            )
        axis.text(
            column + 0.5,
            row + 0.54,
            text,
            ha="center",
            va="center",
            fontsize=7.5,
        )
    axis.set_xlim(0, columns)
    axis.set_ylim(rows, 0)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(title, fontsize=6.7, pad=2.0)
    for spine in axis.spines.values():
        spine.set_visible(False)


def build_figure(
    learning_rows: Sequence[Mapping[str, float]],
    perturbation_rows: Sequence[Mapping[str, float]],
    modal_arrays: Mapping[str, np.ndarray],
    focus_ell: float,
    nominal_slip: float,
    boundary: float,
    main_seeds: int,
    sweep_seeds: int,
    output_dir: Path,
    stem: str,
    dpi: int,
) -> List[Path]:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/drrl-sim-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 6.8,
            "axes.titlesize": 7.8,
            "axes.labelsize": 6.8,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 5.6,
            "lines.linewidth": 1.35,
            "axes.linewidth": 0.65,
            "grid.linewidth": 0.45,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(7.15, 4.7))
    outer = fig.add_gridspec(
        2,
        2,
        left=0.082,
        right=0.985,
        top=0.965,
        bottom=0.14,
        hspace=0.43,
        wspace=0.31,
    )
    convergence_axis = fig.add_subplot(outer[0, 0])
    returns_axis = fig.add_subplot(outer[0, 1])
    gaps_axis = fig.add_subplot(outer[1, 0])

    plot_convergence(convergence_axis, learning_rows)
    plot_returns(returns_axis, perturbation_rows, focus_ell, nominal_slip, boundary)
    plot_gaps(gaps_axis, perturbation_rows, focus_ell, nominal_slip, boundary)
    for axis in (convergence_axis, returns_axis, gaps_axis):
        axis.grid(True, alpha=0.24)
        axis.tick_params(width=0.55, length=2.5)

    policy_grid = outer[1, 1].subgridspec(1, 2, wspace=0.1)

    decision = modal_arrays["decision_state_mask"].astype(bool)
    separating = modal_arrays["oracle_separating_state_mask"].astype(bool)
    panels = (
        (
            modal_arrays["modal_nominal_policy"],
            modal_arrays["modal_nominal_tie_mask"].astype(bool),
            r"$\bf{(d)}$ Learned nominal",
        ),
        (
            modal_arrays["modal_robust_policy"],
            modal_arrays["modal_robust_tie_mask"].astype(bool),
            "Learned robust",
        ),
    )
    for column, (policy, ties, title) in enumerate(panels):
        axis = fig.add_subplot(policy_grid[0, column])
        draw_policy_map(
            axis,
            np.asarray(policy),
            np.asarray(ties),
            title,
            modal_arrays["state_coordinates"],
            decision,
            separating,
        )

    legend_handles = [
        patches.Patch(
            facecolor="#efe6ff",
            edgecolor="#b218b2",
            label="Oracle-separating state",
        ),
        patches.Patch(facecolor="#fcbba1", edgecolor="#666666", label="Cliff"),
        patches.Patch(facecolor="#c7e9c0", edgecolor="#666666", label="Goal"),
        patches.Patch(
            facecolor="white",
            edgecolor="#2166ac",
            linestyle="--",
            label="Start",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.018),
        ncol=4,
        frameon=False,
        fontsize=6.1,
        handlelength=1.5,
        columnspacing=1.3,
    )
    fig.text(
        0.995,
        0.006,
        rf"Mean and 95% CI: $n={sweep_seeds}$ in (a), $n={main_seeds}$ in (b,c); "
        rf"error bars in (b,c) are often smaller than markers; $\ell={focus_ell:g}$ in (b–d).",
        ha="right",
        va="bottom",
        fontsize=5.4,
        color="#444444",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    fig.savefig(paths[0], dpi=dpi, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(paths[1], bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--main-root", type=Path, default=DEFAULT_MAIN_ROOT)
    parser.add_argument("--sweep-root", type=Path, default=DEFAULT_SWEEP_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--stem", default="tabular_tac_composite")
    parser.add_argument("--dpi", type=int, default=400)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.dpi < 1:
        raise ValueError("--dpi must be positive.")
    main_root = args.main_root.resolve()
    sweep_root = args.sweep_root.resolve()
    output_dir = (
        (main_root / "figures") if args.output_dir is None else args.output_dir.resolve()
    )

    main_manifest = read_manifest(main_root)
    sweep_manifest = read_manifest(sweep_root)
    focus_ell, nominal_slip, boundary, main_seeds, sweep_seeds = validate_studies(
        main_manifest, sweep_manifest
    )
    learning_rows = read_numeric_csv(sweep_root / "aggregated" / "learning_curves.csv")
    perturbation_rows = read_numeric_csv(
        main_root / "aggregated" / "perturbation_summary.csv"
    )
    require_columns(
        learning_rows,
        (
            "ell",
            "transitions",
            "robust_q_sup_error_mean",
            "robust_q_sup_error_interval_halfwidth",
        ),
        "learning curves",
    )
    require_columns(
        perturbation_rows,
        (
            "ell",
            "slip_probability",
            "optimal_return_mean",
            "robust_policy_return_mean",
            "robust_policy_return_interval_halfwidth",
            "nominal_policy_return_mean",
            "nominal_policy_return_interval_halfwidth",
            "oracle_robust_policy_return_mean",
            "robust_policy_gap_mean",
            "robust_policy_gap_interval_halfwidth",
            "nominal_policy_gap_mean",
            "nominal_policy_gap_interval_halfwidth",
        ),
        "perturbation summary",
    )
    modal_arrays = load_modal_arrays(
        main_root / "aggregated" / "modal_policies_focus_ell.npz", focus_ell
    )

    paths = build_figure(
        learning_rows,
        perturbation_rows,
        modal_arrays,
        focus_ell,
        nominal_slip,
        boundary,
        main_seeds,
        sweep_seeds,
        output_dir,
        args.stem,
        args.dpi,
    )

    evaluation_slip = float(fixed_fields(main_manifest)["evaluation_slip_probability"])
    selected, _ = perturbation_arrays(perturbation_rows, focus_ell)
    evaluation_row = min(
        selected, key=lambda row: abs(float(row["slip_probability"]) - evaluation_slip)
    )
    if not math.isclose(float(evaluation_row["slip_probability"]), evaluation_slip):
        raise ValueError("The evaluation slip is absent from the perturbation summary.")
    decision = modal_arrays["decision_state_mask"].astype(bool)
    nominal_matches = int(
        np.count_nonzero(
            modal_arrays["modal_nominal_policy"][decision]
            == modal_arrays["nominal_oracle_policy"][decision]
        )
    )
    robust_matches = int(
        np.count_nonzero(
            modal_arrays["modal_robust_policy"][decision]
            == modal_arrays["robust_oracle_policy"][decision]
        )
    )
    print(f"main_seeds={main_seeds} sweep_seeds={sweep_seeds} focus_ell={focus_ell:g}")
    print(f"p0={nominal_slip:.8g} p_chi={boundary:.8g}")
    print(
        f"test_kernel_optimum_at_p={evaluation_slip:g}: "
        f"{evaluation_row['optimal_return_mean']:.10g}"
    )
    print(
        f"modal_policy_decision_state_matches: nominal={nominal_matches}/19 "
        f"robust={robust_matches}/19"
    )
    for path in paths:
        print(f"wrote={path}")


if __name__ == "__main__":
    main()
