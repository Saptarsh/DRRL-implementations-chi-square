#!/usr/bin/env python3
"""Variational chi-square robust Q-learning on continuing MiniCliff.

This is the paper-facing, multi-decision tabular implementation.  It uses
one-hot features, one persistent nominal behavior trajectory, exact robust
dynamic programming only for evaluation, and a nominal Q-learning baseline
that consumes every transition observed by the robust learner.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import uuid
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from train_variational_chi2_tabular import (
    chi2_robust_expectation,
    floor_variational_solution,
    greedy_policy,
    project_l2,
    value_iteration,
)
from variational_tabular_envs import (
    MiniCliffConfig,
    PersistentTabularTrajectory,
    TabularMDP,
    exact_stationary_distributions,
    make_minicliff_behavior_policy,
    make_minicliff_mdp,
    rowwise_chi2_divergence,
    validate_minicliff_config,
)


Array = np.ndarray
AMBIGUITY_BOUNDARY = 0.1 + math.sqrt(0.1 * 0.1 * 0.9)


@dataclass(frozen=True)
class MiniCliffAlgorithmConfig:
    """Learning and exact-evaluation settings for one independent run."""

    seed: int = 1
    chi2_delta: float = 0.10
    ell: float = 0.10
    outer_blocks: int = 40
    stage1_samples: int = 50_000
    q_stage_samples: int = 50_000

    # The theorem-derived option is intentionally available separately from
    # the shared practical constant used for the empirical sweep.
    stage1_step_mode: str = "constant"
    stage1_stepsize: float = 0.024
    theory_step_multiplier: float = 1.0

    # With the default behavior distribution, beta0=650 gives
    # p_Q=2*min_i d_i*beta0 > 1 and h_q=2*beta0.
    beta0: float = 650.0
    h_q: float = 1_300.0

    # Zero selects the conservative automatic radii.
    eta_l2_radius: float = 12.0
    scale_l2_radius: float = 10.0

    nominal_lr_exponent: float = 0.60
    nominal_lr_scale: float = 1.0

    evaluation_slip_probability: float = 0.1875
    perturbation_grid: Tuple[float, ...] = (
        0.10,
        0.125,
        0.15,
        0.175,
        0.1875,
        AMBIGUITY_BOUNDARY,
        0.225,
        0.25,
        0.30,
    )
    dp_tolerance: float = 1e-11
    dp_max_iterations: int = 20_000


@dataclass
class GridRunResult:
    metrics: List[Dict[str, float]]
    perturbation_metrics: List[Dict[str, float]]
    metadata: Dict[str, object]
    arrays: Dict[str, Array]


def validate_algorithm_config(config: MiniCliffAlgorithmConfig) -> None:
    def is_integer(value: object) -> bool:
        return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))

    if not is_integer(config.seed) or config.seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    if not math.isfinite(config.chi2_delta) or config.chi2_delta <= 0.0:
        raise ValueError("chi2_delta must be finite and positive.")
    if not math.isfinite(config.ell) or not (0.0 < config.ell <= 1.0):
        raise ValueError("ell must be finite and lie in (0, 1].")
    counts = (config.outer_blocks, config.stage1_samples, config.q_stage_samples)
    if any(not is_integer(count) or count < 1 for count in counts):
        raise ValueError("outer_blocks, stage1_samples, and q_stage_samples must be positive.")
    if config.stage1_step_mode not in {"constant", "theory"}:
        raise ValueError("stage1_step_mode must be 'constant' or 'theory'.")
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in (config.stage1_stepsize, config.theory_step_multiplier)
    ):
        raise ValueError("Stage-1 step-size parameters must be positive.")
    if (
        not math.isfinite(config.beta0)
        or not math.isfinite(config.h_q)
        or config.beta0 <= 0.0
        or config.h_q < max(1.0, 2.0 * config.beta0)
    ):
        raise ValueError("Require beta0 > 0 and h_q >= max(1, 2*beta0).")
    if any(
        not math.isfinite(radius) or radius < 0.0
        for radius in (config.eta_l2_radius, config.scale_l2_radius)
    ):
        raise ValueError("Parameter radii must be nonnegative; zero selects automatic radii.")
    if not math.isfinite(config.nominal_lr_exponent) or not (
        0.5 < config.nominal_lr_exponent <= 1.0
    ):
        raise ValueError("nominal_lr_exponent must lie in (0.5, 1].")
    if not math.isfinite(config.nominal_lr_scale) or config.nominal_lr_scale <= 0.0:
        raise ValueError("nominal_lr_scale must be positive.")
    if (
        not math.isfinite(config.dp_tolerance)
        or config.dp_tolerance <= 0.0
        or not is_integer(config.dp_max_iterations)
        or config.dp_max_iterations < 1
    ):
        raise ValueError("DP tolerance and maximum iterations must be positive.")
    probabilities = (config.evaluation_slip_probability,) + tuple(config.perturbation_grid)
    if any(not math.isfinite(probability) or not (0.0 < probability < 1.0) for probability in probabilities):
        raise ValueError("Every evaluation slip probability must lie strictly in (0, 1).")
    if not config.perturbation_grid:
        raise ValueError("perturbation_grid must contain at least one value.")
    if len(set(config.perturbation_grid)) != len(config.perturbation_grid):
        raise ValueError("perturbation_grid must not contain duplicate values.")


def nominal_bellman(q_values: Array, mdp: TabularMDP) -> Array:
    values = np.max(q_values, axis=1)
    return mdp.rewards + mdp.gamma * np.einsum(
        "sat,t->sa", mdp.transitions, values, optimize=True
    )


def robust_bellman(q_values: Array, mdp: TabularMDP, delta: float) -> Array:
    values = np.max(q_values, axis=1)
    result = np.empty_like(q_values, dtype=np.float64)
    for state in range(mdp.n_states):
        for action in range(mdp.n_actions):
            result[state, action] = mdp.rewards[state, action] + mdp.gamma * chi2_robust_expectation(
                values, mdp.transitions[state, action], delta
            )
    return result


def floor_bellman(q_values: Array, mdp: TabularMDP, delta: float, ell: float) -> Array:
    values = np.max(q_values, axis=1)
    result = np.empty_like(q_values, dtype=np.float64)
    for state in range(mdp.n_states):
        for action in range(mdp.n_actions):
            value, _, _ = floor_variational_solution(
                values, mdp.transitions[state, action], delta, ell
            )
            result[state, action] = mdp.rewards[state, action] + mdp.gamma * value
    return result


def exact_references(
    mdp: TabularMDP, config: MiniCliffAlgorithmConfig
) -> Tuple[Array, Array, Array]:
    shape = mdp.rewards.shape
    nominal = value_iteration(
        lambda q: nominal_bellman(q, mdp),
        shape,
        config.dp_tolerance,
        config.dp_max_iterations,
    )
    robust = value_iteration(
        lambda q: robust_bellman(q, mdp, config.chi2_delta),
        shape,
        config.dp_tolerance,
        config.dp_max_iterations,
    )
    floor = value_iteration(
        lambda q: floor_bellman(q, mdp, config.chi2_delta, config.ell),
        shape,
        config.dp_tolerance,
        config.dp_max_iterations,
    )
    return nominal, robust, floor


def exact_policy_value(mdp: TabularMDP, policy: Array) -> Array:
    states = np.arange(mdp.n_states)
    policy = np.asarray(policy, dtype=np.int64)
    system = np.eye(mdp.n_states) - mdp.gamma * mdp.transitions[states, policy]
    return np.linalg.solve(system, mdp.rewards[states, policy])


def exact_robust_policy_value(
    mdp: TabularMDP,
    policy: Array,
    delta: float,
    tolerance: float,
    max_iterations: int,
) -> Array:
    policy = np.asarray(policy, dtype=np.int64)
    values = np.zeros(mdp.n_states, dtype=np.float64)
    # The active-set inner solve is accurate to roughly machine precision but
    # can leave a tiny row-selection jitter near ties.  A 1e-10 fixed-policy
    # tolerance is far below every plotted scale and avoids mistaking that
    # numerical floor for a failure of the gamma-contraction.
    effective_tolerance = max(tolerance, 1e-10)
    last_difference = math.inf
    for _ in range(max_iterations):
        updated = np.empty_like(values)
        for state in range(mdp.n_states):
            action = int(policy[state])
            updated[state] = mdp.rewards[state, action] + mdp.gamma * chi2_robust_expectation(
                values, mdp.transitions[state, action], delta
            )
        last_difference = float(np.max(np.abs(updated - values)))
        if last_difference <= effective_tolerance:
            return updated
        values = updated
    raise RuntimeError(
        f"Robust fixed-policy evaluation did not converge; successive difference={last_difference:.3e}."
    )


def discounted_state_occupancy(mdp: TabularMDP, policy: Array) -> Array:
    """Return normalized discounted occupancy from the designated start."""
    states = np.arange(mdp.n_states)
    transition = mdp.transitions[states, np.asarray(policy, dtype=np.int64)]
    start = np.zeros(mdp.n_states, dtype=np.float64)
    start[mdp.start_state] = 1.0 - mdp.gamma
    occupancy = np.linalg.solve(
        np.eye(mdp.n_states, dtype=np.float64) - mdp.gamma * transition.T,
        start,
    )
    occupancy = np.maximum(occupancy, 0.0)
    return occupancy / occupancy.sum()


def decision_state_mask(mdp: TabularMDP) -> Array:
    mask = np.ones(mdp.n_states, dtype=bool)
    mask[list(mdp.marker_states)] = False
    return mask


def policy_agreement_metrics(
    mdp: TabularMDP,
    policy: Array,
    oracle_policy: Array,
    separating_mask: Array,
) -> Tuple[float, float, float]:
    decision_mask = decision_state_mask(mdp)
    agreements = np.asarray(policy) == np.asarray(oracle_policy)
    ordinary_agreement = float(np.mean(agreements[decision_mask]))
    occupancy = discounted_state_occupancy(mdp, oracle_policy)
    decision_mass = float(occupancy[decision_mask].sum())
    weighted_agreement = float(
        np.dot(occupancy[decision_mask], agreements[decision_mask]) / max(decision_mass, 1e-15)
    )
    separating_count = int(np.count_nonzero(separating_mask))
    separating_agreement = (
        float(np.mean(agreements[separating_mask])) if separating_count else 1.0
    )
    return ordinary_agreement, weighted_agreement, separating_agreement


def exact_test_optimum(mdp: TabularMDP, config: MiniCliffAlgorithmConfig) -> Tuple[float, Array]:
    q_star = value_iteration(
        lambda q: nominal_bellman(q, mdp),
        mdp.rewards.shape,
        config.dp_tolerance,
        config.dp_max_iterations,
    )
    return float(np.max(q_star[mdp.start_state])), q_star


def automatic_parameter_radii(
    mdp: TabularMDP, config: MiniCliffAlgorithmConfig
) -> Tuple[float, float]:
    c_delta = math.sqrt(1.0 + config.chi2_delta)
    b_q = 1.0 / (1.0 - mdp.gamma)
    eta_coordinate_bound = (
        (c_delta + 1.0) * b_q + 0.5 * c_delta * config.ell
    ) / (c_delta - 1.0)
    n_pairs = mdp.n_states * mdp.n_actions
    eta_radius = config.eta_l2_radius or math.sqrt(n_pairs) * eta_coordinate_bound
    scale_radius = config.scale_l2_radius or math.sqrt(n_pairs) * (
        eta_coordinate_bound + b_q
    )
    return float(eta_radius), float(scale_radius)


def stage1_stepsize(
    mdp: TabularMDP,
    config: MiniCliffAlgorithmConfig,
    eta_radius: float,
    scale_radius: float,
) -> float:
    if config.stage1_step_mode == "constant":
        return float(config.stage1_stepsize)
    c_delta = math.sqrt(1.0 + config.chi2_delta)
    b_q = 1.0 / (1.0 - mdp.gamma)
    h_bound = eta_radius + b_q
    gradient_bound = math.sqrt(
        (1.0 + c_delta * h_bound / config.ell) ** 2
        + 0.25 * c_delta**2 * (1.0 + h_bound**2 / config.ell**2) ** 2
    )
    radius = math.sqrt(eta_radius**2 + scale_radius**2)
    n_samples = config.stage1_samples
    mixing_factor = 1.0 + math.log(n_samples + 1.0) + math.log(n_samples + 1.0) ** 2
    return float(
        config.theory_step_multiplier
        * radius
        / (gradient_bound * math.sqrt(n_samples * mixing_factor))
    )


def update_nominal_q(
    q_values: Array,
    visit_counts: Array,
    state: int,
    action: int,
    reward: float,
    next_state: int,
    mdp: TabularMDP,
    config: MiniCliffAlgorithmConfig,
) -> None:
    visit_counts[state, action] += 1
    learning_rate = config.nominal_lr_scale / float(
        visit_counts[state, action]
    ) ** config.nominal_lr_exponent
    learning_rate = min(1.0, learning_rate)
    target = reward + mdp.gamma * float(np.max(q_values[next_state]))
    q_values[state, action] += learning_rate * (target - q_values[state, action])


def diagnostic_row(
    outer_block: int,
    transitions_read: int,
    robust_q: Array,
    nominal_q: Array,
    mdp: TabularMDP,
    evaluation_mdp: TabularMDP,
    config: MiniCliffAlgorithmConfig,
    robust_reference: Array,
    floor_reference: Array,
    nominal_reference: Array,
    robust_oracle_policy: Array,
    nominal_oracle_policy: Array,
    evaluation_optimal_return: float,
    separating_mask: Array,
    block_diagnostics: Dict[str, float],
) -> Dict[str, float]:
    robust_policy = greedy_policy(robust_q)
    nominal_policy = greedy_policy(nominal_q)
    robust_evaluation = exact_policy_value(evaluation_mdp, robust_policy)
    nominal_evaluation = exact_policy_value(evaluation_mdp, nominal_policy)
    robust_worst_case = exact_robust_policy_value(
        mdp,
        robust_policy,
        config.chi2_delta,
        config.dp_tolerance,
        config.dp_max_iterations,
    )
    robust_agreement = policy_agreement_metrics(
        mdp, robust_policy, robust_oracle_policy, separating_mask
    )
    nominal_agreement = policy_agreement_metrics(
        mdp, nominal_policy, nominal_oracle_policy, separating_mask
    )
    robust_occupancy = discounted_state_occupancy(evaluation_mdp, robust_policy)
    nominal_occupancy = discounted_state_occupancy(evaluation_mdp, nominal_policy)
    robust_return = float(robust_evaluation[evaluation_mdp.start_state])
    nominal_return = float(nominal_evaluation[evaluation_mdp.start_state])
    robust_optimal_value = float(np.max(robust_reference[mdp.start_state]))
    decision_mask = decision_state_mask(mdp)

    row = {
        "outer_block": float(outer_block),
        "transitions": float(transitions_read),
        "robust_q_sup_error": float(np.max(np.abs(robust_q - robust_reference))),
        "floor_q_sup_error": float(np.max(np.abs(robust_q - floor_reference))),
        "floor_reference_bias": float(np.max(np.abs(floor_reference - robust_reference))),
        "floor_bias_bound": float(
            mdp.gamma
            * math.sqrt(1.0 + config.chi2_delta)
            * config.ell
            / (2.0 * (1.0 - mdp.gamma))
        ),
        "robust_bellman_residual": float(
            np.max(np.abs(robust_q - robust_bellman(robust_q, mdp, config.chi2_delta)))
        ),
        "floor_bellman_residual": float(
            np.max(np.abs(robust_q - floor_bellman(robust_q, mdp, config.chi2_delta, config.ell)))
        ),
        "nominal_q_sup_error": float(np.max(np.abs(nominal_q - nominal_reference))),
        "robust_start_action": float(robust_policy[mdp.start_state]),
        "nominal_start_action": float(nominal_policy[mdp.start_state]),
        "robust_policy_perturbed_return": robust_return,
        "nominal_policy_perturbed_return": nominal_return,
        "perturbed_optimal_return": evaluation_optimal_return,
        "robust_policy_perturbed_gap": max(0.0, evaluation_optimal_return - robust_return),
        "nominal_policy_perturbed_gap": max(0.0, evaluation_optimal_return - nominal_return),
        "robust_policy_advantage": robust_return - nominal_return,
        "robust_policy_worst_case_return": float(robust_worst_case[mdp.start_state]),
        "robust_policy_worst_case_gap": max(
            0.0, robust_optimal_value - float(robust_worst_case[mdp.start_state])
        ),
        "robust_policy_oracle_agreement": robust_agreement[0],
        "robust_policy_occupancy_weighted_agreement": robust_agreement[1],
        "robust_policy_separating_state_agreement": robust_agreement[2],
        "nominal_policy_oracle_agreement": nominal_agreement[0],
        "nominal_policy_occupancy_weighted_agreement": nominal_agreement[1],
        "nominal_policy_separating_state_agreement": nominal_agreement[2],
        "learned_policy_disagreement_count": float(
            np.count_nonzero(robust_policy[decision_mask] != nominal_policy[decision_mask])
        ),
        "robust_policy_discounted_cliff_occupancy": float(
            robust_occupancy[list(mdp.cliff_states)].sum()
        ),
        "nominal_policy_discounted_cliff_occupancy": float(
            nominal_occupancy[list(mdp.cliff_states)].sum()
        ),
    }
    row.update(block_diagnostics)
    return row


def _source_hashes() -> Dict[str, str]:
    directory = Path(__file__).resolve().parent
    paths = {
        "trainer": Path(__file__).resolve(),
        "environment": directory / "variational_tabular_envs.py",
        "shared_exact_solver": directory / "train_variational_chi2_tabular.py",
    }
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}


def run_experiment(
    environment: MiniCliffConfig,
    config: MiniCliffAlgorithmConfig,
) -> GridRunResult:
    """Run one seed of the multi-decision tabular experiment."""
    validate_minicliff_config(environment)
    validate_algorithm_config(config)
    rng = np.random.default_rng(config.seed)
    mdp = make_minicliff_mdp(environment)
    behavior = make_minicliff_behavior_policy(mdp)
    stationary = exact_stationary_distributions(mdp, behavior)
    nominal_reference, robust_reference, floor_reference = exact_references(mdp, config)
    nominal_oracle_policy = greedy_policy(nominal_reference)
    robust_oracle_policy = greedy_policy(robust_reference)
    decision_mask = decision_state_mask(mdp)
    separating_mask = decision_mask & (nominal_oracle_policy != robust_oracle_policy)
    if int(np.count_nonzero(separating_mask)) < 2:
        raise RuntimeError("MiniCliff no longer has multiple oracle-separating decision states.")

    evaluation_mdp = make_minicliff_mdp(
        environment, slip_probability=config.evaluation_slip_probability
    )
    if not np.array_equal(evaluation_mdp.rewards, mdp.rewards):
        raise RuntimeError("Slip perturbations must leave the reward table fixed.")
    evaluation_optimal_return, _ = exact_test_optimum(evaluation_mdp, config)

    eta_radius, scale_radius = automatic_parameter_radii(mdp, config)
    alpha_n = stage1_stepsize(mdp, config, eta_radius, scale_radius)
    b_q = 1.0 / (1.0 - mdp.gamma)
    c_delta = math.sqrt(1.0 + config.chi2_delta)
    n_pairs = mdp.n_states * mdp.n_actions

    floor_values = np.max(floor_reference, axis=1)
    floor_oracle_eta = np.empty(n_pairs, dtype=np.float64)
    floor_oracle_rho = np.empty(n_pairs, dtype=np.float64)
    for state in range(mdp.n_states):
        for action in range(mdp.n_actions):
            index = state * mdp.n_actions + action
            _, eta_star, u_star = floor_variational_solution(
                floor_values,
                mdp.transitions[state, action],
                config.chi2_delta,
                config.ell,
            )
            floor_oracle_eta[index] = eta_star
            floor_oracle_rho[index] = max(0.0, u_star - config.ell)
    floor_eta_norm = float(np.linalg.norm(floor_oracle_eta))
    floor_rho_norm = float(np.linalg.norm(floor_oracle_rho))
    floor_inside_balls = float(
        floor_eta_norm <= eta_radius + 1e-10
        and floor_rho_norm <= scale_radius + 1e-10
    )
    if not bool(floor_inside_balls):
        raise ValueError(
            "The exact finite-floor optimizer is outside the configured parameter balls: "
            f"eta norm {floor_eta_norm:.6g} versus radius {eta_radius:.6g}, "
            f"scale norm {floor_rho_norm:.6g} versus radius {scale_radius:.6g}. "
            "Increase --eta-l2-radius/--scale-l2-radius or use zero for automatic radii."
        )

    trajectory = PersistentTabularTrajectory(mdp, behavior, rng)
    robust_raw_q = np.zeros((mdp.n_states, mdp.n_actions), dtype=np.float64)
    robust_q = np.zeros_like(robust_raw_q)
    nominal_q = np.zeros_like(robust_raw_q)
    nominal_visits = np.zeros_like(robust_raw_q, dtype=np.int64)

    empty_diagnostics = {
        "stage1_stepsize": alpha_n,
        "stage1_gradient_rms": 0.0,
        "stage1_gradient_max": 0.0,
        "stage1_ratio_x_over_u_p95": 0.0,
        "stage1_population_target_sup_error": 0.0,
        "stage1_population_target_mean_error": 0.0,
        "floor_oracle_eta_l2_norm": floor_eta_norm,
        "floor_oracle_scale_rho_l2_norm": floor_rho_norm,
        "floor_oracle_inside_parameter_balls": floor_inside_balls,
        "eta_projection_fraction": 0.0,
        "scale_projection_fraction": 0.0,
        "scale_floor_fraction": 0.0,
        "eta_norm": 0.0,
        "scale_rho_norm": 0.0,
        "min_state_action_visits": 0.0,
    }
    metrics = [
        diagnostic_row(
            0,
            0,
            robust_q,
            nominal_q,
            mdp,
            evaluation_mdp,
            config,
            robust_reference,
            floor_reference,
            nominal_reference,
            robust_oracle_policy,
            nominal_oracle_policy,
            evaluation_optimal_return,
            separating_mask,
            empty_diagnostics,
        )
    ]

    last_eta_bar = np.zeros(n_pairs, dtype=np.float64)
    last_u_bar = np.full(n_pairs, config.ell, dtype=np.float64)
    for outer_block in range(1, config.outer_blocks + 1):
        frozen_q = np.clip(robust_raw_q, -b_q, b_q)
        frozen_values = np.max(frozen_q, axis=1)
        eta = np.zeros(n_pairs, dtype=np.float64)
        rho = np.zeros(n_pairs, dtype=np.float64)
        eta_sum = np.zeros(n_pairs, dtype=np.float64)
        rho_sum = np.zeros(n_pairs, dtype=np.float64)
        ratio_samples = np.empty(config.stage1_samples, dtype=np.float64)
        gradient_square_sum = 0.0
        gradient_max = 0.0
        eta_projection_hits = 0
        scale_projection_hits = 0
        scale_floor_hits = 0

        for sample_index in range(config.stage1_samples):
            state, action, reward, next_state = trajectory.next()
            update_nominal_q(
                nominal_q,
                nominal_visits,
                state,
                action,
                reward,
                next_state,
                mdp,
                config,
            )
            index = state * mdp.n_actions + action
            eta_sum += eta
            rho_sum += rho
            old_eta = float(eta[index])
            old_u = config.ell + float(rho[index])
            positive_part = max(old_eta - float(frozen_values[next_state]), 0.0)
            ratio_samples[sample_index] = positive_part / old_u
            eta_gradient = 1.0 - c_delta * positive_part / old_u
            scale_gradient = 0.5 * c_delta * ((positive_part / old_u) ** 2 - 1.0)
            gradient_norm = math.hypot(eta_gradient, scale_gradient)
            if not np.isfinite(gradient_norm):
                raise FloatingPointError("Non-finite Stage-1 gradient.")
            gradient_square_sum += gradient_norm * gradient_norm
            gradient_max = max(gradient_max, gradient_norm)

            eta[index] = old_eta + alpha_n * eta_gradient
            raw_rho = float(rho[index]) + alpha_n * scale_gradient
            scale_floor_hits += int(raw_rho < 0.0)
            rho[index] = max(raw_rho, 0.0)
            eta, eta_hit = project_l2(eta, eta_radius)
            rho, scale_hit = project_l2(rho, scale_radius)
            eta_projection_hits += int(eta_hit)
            scale_projection_hits += int(scale_hit)

        eta_bar = eta_sum / float(config.stage1_samples)
        rho_bar = rho_sum / float(config.stage1_samples)
        u_bar = config.ell + rho_bar
        last_eta_bar = eta_bar.copy()
        last_u_bar = u_bar.copy()

        learned_population_next = np.empty_like(robust_q)
        exact_floor_next = np.empty_like(robust_q)
        for state in range(mdp.n_states):
            for action in range(mdp.n_actions):
                index = state * mdp.n_actions + action
                positive_parts = np.maximum(float(eta_bar[index]) - frozen_values, 0.0)
                second_moment = float(
                    mdp.transitions[state, action] @ (positive_parts * positive_parts)
                )
                learned_population_next[state, action] = float(eta_bar[index]) - 0.5 * c_delta * (
                    second_moment / float(u_bar[index]) + float(u_bar[index])
                )
                exact_floor_next[state, action], _, _ = floor_variational_solution(
                    frozen_values,
                    mdp.transitions[state, action],
                    config.chi2_delta,
                    config.ell,
                )
        population_error = np.abs(learned_population_next - exact_floor_next)

        q_work = np.zeros_like(robust_raw_q)
        for q_step in range(config.q_stage_samples):
            state, action, reward, next_state = trajectory.next()
            update_nominal_q(
                nominal_q,
                nominal_visits,
                state,
                action,
                reward,
                next_state,
                mdp,
                config,
            )
            index = state * mdp.n_actions + action
            positive_part = max(float(eta_bar[index]) - float(frozen_values[next_state]), 0.0)
            robust_next_sample = float(eta_bar[index]) - 0.5 * c_delta * (
                positive_part * positive_part / float(u_bar[index]) + float(u_bar[index])
            )
            target = reward + mdp.gamma * robust_next_sample
            beta_m = config.beta0 / (float(q_step) + config.h_q)
            q_work[state, action] += beta_m * (target - q_work[state, action])

        robust_raw_q = q_work
        robust_q = np.clip(robust_raw_q, -b_q, b_q)
        block_diagnostics = {
            "stage1_stepsize": alpha_n,
            "stage1_gradient_rms": math.sqrt(gradient_square_sum / config.stage1_samples),
            "stage1_gradient_max": gradient_max,
            "stage1_ratio_x_over_u_p95": float(np.quantile(ratio_samples, 0.95)),
            "stage1_population_target_sup_error": float(np.max(population_error)),
            "stage1_population_target_mean_error": float(np.mean(population_error)),
            "floor_oracle_eta_l2_norm": floor_eta_norm,
            "floor_oracle_scale_rho_l2_norm": floor_rho_norm,
            "floor_oracle_inside_parameter_balls": floor_inside_balls,
            "eta_projection_fraction": eta_projection_hits / float(config.stage1_samples),
            "scale_projection_fraction": scale_projection_hits / float(config.stage1_samples),
            "scale_floor_fraction": scale_floor_hits / float(config.stage1_samples),
            "eta_norm": float(np.linalg.norm(eta_bar)),
            "scale_rho_norm": float(np.linalg.norm(rho_bar)),
            "min_state_action_visits": float(np.min(trajectory.state_action_counts)),
        }
        metrics.append(
            diagnostic_row(
                outer_block,
                trajectory.transitions_read,
                robust_q,
                nominal_q,
                mdp,
                evaluation_mdp,
                config,
                robust_reference,
                floor_reference,
                nominal_reference,
                robust_oracle_policy,
                nominal_oracle_policy,
                evaluation_optimal_return,
                separating_mask,
                block_diagnostics,
            )
        )

    robust_policy = greedy_policy(robust_q)
    nominal_policy = greedy_policy(nominal_q)
    perturbation_metrics: List[Dict[str, float]] = []
    for slip_probability in config.perturbation_grid:
        perturbed_mdp = make_minicliff_mdp(environment, slip_probability=slip_probability)
        if not np.array_equal(perturbed_mdp.rewards, mdp.rewards):
            raise RuntimeError("Perturbation unexpectedly changed rewards.")
        divergence = rowwise_chi2_divergence(mdp.transitions, perturbed_mdp.transitions)
        optimal_return, _ = exact_test_optimum(perturbed_mdp, config)
        robust_values = exact_policy_value(perturbed_mdp, robust_policy)
        nominal_values = exact_policy_value(perturbed_mdp, nominal_policy)
        oracle_robust_values = exact_policy_value(perturbed_mdp, robust_oracle_policy)
        oracle_nominal_values = exact_policy_value(perturbed_mdp, nominal_oracle_policy)
        robust_occupancy = discounted_state_occupancy(perturbed_mdp, robust_policy)
        nominal_occupancy = discounted_state_occupancy(perturbed_mdp, nominal_policy)
        robust_return = float(robust_values[mdp.start_state])
        nominal_return = float(nominal_values[mdp.start_state])
        perturbation_metrics.append(
            {
                "slip_probability": float(slip_probability),
                "max_row_chi2_distance": divergence.maximum_divergence,
                "support_preserved": float(divergence.support_preserved),
                "inside_chi2_radius": float(divergence.within_radius(config.chi2_delta)),
                "optimal_return": optimal_return,
                "robust_policy_return": robust_return,
                "nominal_policy_return": nominal_return,
                "oracle_robust_policy_return": float(oracle_robust_values[mdp.start_state]),
                "oracle_nominal_policy_return": float(oracle_nominal_values[mdp.start_state]),
                "robust_policy_gap": max(0.0, optimal_return - robust_return),
                "nominal_policy_gap": max(0.0, optimal_return - nominal_return),
                "robust_policy_advantage": robust_return - nominal_return,
                "robust_policy_discounted_cliff_occupancy": float(
                    robust_occupancy[list(mdp.cliff_states)].sum()
                ),
                "nominal_policy_discounted_cliff_occupancy": float(
                    nominal_occupancy[list(mdp.cliff_states)].sum()
                ),
            }
        )

    minimum_sa_probability = stationary.minimum_state_action_probability
    q_gain_p = 2.0 * minimum_sa_probability * config.beta0
    oracle_difference_states = np.flatnonzero(separating_mask)
    metadata: Dict[str, object] = {
        "source_sha256": _source_hashes(),
        "environment": asdict(environment),
        "algorithm": {**asdict(config), "perturbation_grid": list(config.perturbation_grid)},
        "environment_name": mdp.name,
        "n_states": mdp.n_states,
        "n_actions": mdp.n_actions,
        "n_decision_states": int(np.count_nonzero(decision_mask)),
        "stage1_stepsize_used": alpha_n,
        "eta_l2_radius_used": eta_radius,
        "scale_l2_radius_used": scale_radius,
        "behavior_stationary_residual": stationary.residual,
        "behavior_min_state_probability": float(np.min(stationary.state_probabilities)),
        "behavior_min_state_action_probability": minimum_sa_probability,
        "expected_min_visits_per_stage1_block": minimum_sa_probability * config.stage1_samples,
        "expected_min_visits_per_q_block": minimum_sa_probability * config.q_stage_samples,
        "q_gain_p": q_gain_p,
        "clean_q_rate_condition_satisfied": bool(q_gain_p > 1.0),
        "floor_oracle_eta_l2_norm": floor_eta_norm,
        "floor_oracle_scale_rho_l2_norm": floor_rho_norm,
        "floor_oracle_inside_parameter_balls": bool(floor_inside_balls),
        "ambiguity_upper_slip_probability": float(
            environment.nominal_slip_probability
            + math.sqrt(
                config.chi2_delta
                * environment.nominal_slip_probability
                * (1.0 - environment.nominal_slip_probability)
            )
        ),
        "oracle_policy_difference_count": int(len(oracle_difference_states)),
        "oracle_policy_difference_states": oracle_difference_states.tolist(),
        "oracle_policy_difference_coordinates": [
            list(mdp.coordinate_of(int(state))) for state in oracle_difference_states
        ],
        "oracle_nominal_start_action": int(nominal_oracle_policy[mdp.start_state]),
        "oracle_robust_start_action": int(robust_oracle_policy[mdp.start_state]),
        "oracle_nominal_start_q": nominal_reference[mdp.start_state].tolist(),
        "oracle_robust_start_q": robust_reference[mdp.start_state].tolist(),
        "floor_reference_sup_bias": float(np.max(np.abs(floor_reference - robust_reference))),
        "floor_bias_bound": float(
            mdp.gamma * c_delta * config.ell / (2.0 * (1.0 - mdp.gamma))
        ),
        "total_transitions": trajectory.transitions_read,
        "final_min_state_action_visits": int(np.min(trajectory.state_action_counts)),
    }
    arrays = {
        "robust_q": robust_q,
        "nominal_q": nominal_q,
        "robust_reference_q": robust_reference,
        "floor_reference_q": floor_reference,
        "nominal_reference_q": nominal_reference,
        "robust_policy": robust_policy,
        "nominal_policy": nominal_policy,
        "robust_oracle_policy": robust_oracle_policy,
        "nominal_oracle_policy": nominal_oracle_policy,
        "state_action_counts": trajectory.state_action_counts,
        "last_eta_bar": last_eta_bar.reshape(mdp.n_states, mdp.n_actions),
        "last_u_bar": last_u_bar.reshape(mdp.n_states, mdp.n_actions),
        "floor_oracle_eta": floor_oracle_eta.reshape(mdp.n_states, mdp.n_actions),
        "floor_oracle_u": (config.ell + floor_oracle_rho).reshape(
            mdp.n_states, mdp.n_actions
        ),
        "behavior_policy": behavior,
        "behavior_stationary_state": stationary.state_probabilities,
        "behavior_stationary_state_action": stationary.state_action_probabilities,
        "state_coordinates": mdp.state_coordinates,
        "decision_state_mask": decision_mask,
        "oracle_separating_state_mask": separating_mask,
    }
    return GridRunResult(
        metrics=metrics,
        perturbation_metrics=perturbation_metrics,
        metadata=metadata,
        arrays=arrays,
    )


def write_csv(path: Path, rows: Sequence[Dict[str, float]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_run(result: GridRunResult, output_dir: Path) -> None:
    """Atomically publish a mutually authenticated set of run artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    token = f"{os.getpid()}-{uuid.uuid4().hex}"
    filenames = {
        "metrics.csv": output_dir / "metrics.csv",
        "perturbation_metrics.csv": output_dir / "perturbation_metrics.csv",
        "arrays.npz": output_dir / "arrays.npz",
        "metadata.json": output_dir / "metadata.json",
    }
    temporary = {
        name: path.with_name(f".{path.name}.{token}.tmp") for name, path in filenames.items()
    }
    try:
        write_csv(temporary["metrics.csv"], result.metrics)
        write_csv(temporary["perturbation_metrics.csv"], result.perturbation_metrics)
        with temporary["arrays.npz"].open("wb") as handle:
            np.savez_compressed(handle, **result.arrays)

        artifact_hashes = {
            name: hashlib.sha256(temporary[name].read_bytes()).hexdigest()
            for name in ("metrics.csv", "perturbation_metrics.csv", "arrays.npz")
        }
        metadata = {**result.metadata, "artifact_sha256": artifact_hashes}
        temporary["metadata.json"].write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        # Metadata is the commit record and is therefore promoted last.  If a
        # process is interrupted earlier, its old hashes cannot authenticate a
        # mixture of old and new data artifacts.
        for name in ("metrics.csv", "perturbation_metrics.csv", "arrays.npz"):
            os.replace(temporary[name], filenames[name])
        os.replace(temporary["metadata.json"], filenames["metadata.json"])
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)


def parse_float_tuple(text: str) -> Tuple[float, ...]:
    values = tuple(float(piece.strip()) for piece in text.split(",") if piece.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated value.")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    environment_defaults = MiniCliffConfig()
    for field in fields(MiniCliffConfig):
        default = getattr(environment_defaults, field.name)
        parser.add_argument("--" + field.name.replace("_", "-"), type=type(default), default=default)
    algorithm_defaults = MiniCliffAlgorithmConfig()
    for field in fields(MiniCliffAlgorithmConfig):
        default = getattr(algorithm_defaults, field.name)
        argument = "--" + field.name.replace("_", "-")
        if field.name == "perturbation_grid":
            parser.add_argument(argument, type=parse_float_tuple, default=default)
        else:
            parser.add_argument(argument, type=type(default), default=default)
    parser.add_argument("--output-dir", default="runs_variational_chi2_gridworld/direct")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    environment = MiniCliffConfig(
        **{field.name: getattr(args, field.name) for field in fields(MiniCliffConfig)}
    )
    config = MiniCliffAlgorithmConfig(
        **{field.name: getattr(args, field.name) for field in fields(MiniCliffAlgorithmConfig)}
    )
    result = run_experiment(environment, config)
    output_dir = Path(args.output_dir).resolve()
    save_run(result, output_dir)
    final = result.metrics[-1]
    print(f"output_dir={output_dir}")
    print(f"transitions={int(final['transitions'])}")
    print(f"robust_q_sup_error={final['robust_q_sup_error']:.8g}")
    print(f"robust_policy_perturbed_gap={final['robust_policy_perturbed_gap']:.8g}")
    print(f"nominal_policy_perturbed_gap={final['nominal_policy_perturbed_gap']:.8g}")


if __name__ == "__main__":
    main()
