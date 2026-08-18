#!/usr/bin/env python3
"""Tabular experiments for variational chi-square robust Q-learning.

This file implements Algorithm 1 from ``variational_algorithm.txt`` in the
tabular, one-hot-feature case.  It deliberately does not reuse the older
eta/Z1/Z2 implementation: the variational scale makes those moment critics and
the smoothing parameter unnecessary.

The learning process sees only one continuing nominal trajectory.  The nominal
transition table is used separately for exact evaluation, including an
unsmoothed robust-DP reference and controlled transition perturbations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class CorridorConfig:
    """Continuing-reset risky-shortcut/safe-detour MDP."""

    # The compact default has ten states.  It mixes markedly faster than the
    # longer NN corridor while preserving the same safe-versus-risky choice.
    risky_len: int = 2
    safe_len: int = 5
    nominal_crash_prob: float = 0.10
    step_reward: float = -0.03
    goal_reward: float = 0.50
    crash_reward: float = -0.50
    gamma: float = 0.90
    behavior_risky_prob: float = 0.50


@dataclass(frozen=True)
class AlgorithmConfig:
    """Learning and evaluation settings for one independent run."""

    seed: int = 1
    chi2_delta: float = 0.50
    ell: float = 0.03
    outer_blocks: int = 30
    stage1_samples: int = 10_000
    q_stage_samples: int = 5_000

    # ``constant`` is useful for simulations. ``theory`` uses the exact
    # conservative step from the theorem in variational_algorithm.txt.
    stage1_step_mode: str = "constant"
    stage1_stepsize: float = 0.008
    theory_step_multiplier: float = 1.0

    # beta_m = beta0 / (m + h_q), exactly as in the manuscript.
    beta0: float = 75.0
    h_q: float = 150.0

    # Zero selects safe automatic global L2/Frobenius radii.
    eta_l2_radius: float = 5.0
    scale_l2_radius: float = 5.0

    # Same nominal transitions, updated once per observed transition.
    nominal_lr_exponent: float = 0.60
    nominal_lr_scale: float = 1.0

    evaluation_crash_prob: float = 0.30
    perturbation_grid: Tuple[float, ...] = (
        0.10,
        0.125,
        0.15,
        0.175,
        0.20,
        0.225,
        0.25,
        0.275,
        0.30,
        0.325,
        0.35,
        0.375,
        0.40,
    )
    dp_tolerance: float = 1e-11
    dp_max_iterations: int = 20_000


@dataclass(frozen=True)
class CorridorMDP:
    transitions: Array
    rewards: Array
    start_state: int
    goal_state: int
    crash_state: int
    risky_states: Tuple[int, ...]
    safe_states: Tuple[int, ...]
    # Keep the adverse/risky action first.  With the stable argmax tie break,
    # the all-zero initialization then starts from the non-robust decision and
    # the policy diagnostic records a genuine learned switch to action 1.
    action_names: Tuple[str, str] = ("aggressive_risky", "cautious_safe")

    @property
    def n_states(self) -> int:
        return int(self.transitions.shape[0])

    @property
    def n_actions(self) -> int:
        return int(self.transitions.shape[1])


@dataclass
class RunResult:
    metrics: List[Dict[str, float]]
    perturbation_metrics: List[Dict[str, float]]
    metadata: Dict[str, object]
    arrays: Dict[str, Array]


def validate_configs(env: CorridorConfig, cfg: AlgorithmConfig) -> None:
    if env.risky_len < 1 or env.safe_len < 1:
        raise ValueError("Both route lengths must be positive.")
    if not (0.0 < env.nominal_crash_prob < 1.0):
        raise ValueError("nominal_crash_prob must be strictly between 0 and 1.")
    if not (0.0 < env.behavior_risky_prob < 1.0):
        raise ValueError("behavior_risky_prob must be strictly between 0 and 1.")
    if not (0.0 < env.gamma < 1.0):
        raise ValueError("gamma must lie in (0, 1).")
    if max(abs(env.step_reward), abs(env.goal_reward), abs(env.crash_reward)) > 1.0:
        raise ValueError("Rewards must lie in [-1, 1] to match the stated clipping bound.")
    if cfg.chi2_delta <= 0.0:
        raise ValueError("chi2_delta must be positive.")
    if not (0.0 < cfg.ell <= 1.0):
        raise ValueError("ell must lie in (0, 1].")
    if cfg.outer_blocks < 1 or cfg.stage1_samples < 1 or cfg.q_stage_samples < 1:
        raise ValueError("outer_blocks, stage1_samples, and q_stage_samples must be positive.")
    if cfg.stage1_step_mode not in {"constant", "theory"}:
        raise ValueError("stage1_step_mode must be 'constant' or 'theory'.")
    if cfg.stage1_stepsize <= 0.0 or cfg.theory_step_multiplier <= 0.0:
        raise ValueError("Stage-1 step-size parameters must be positive.")
    if cfg.beta0 <= 0.0 or cfg.h_q < max(1.0, 2.0 * cfg.beta0):
        raise ValueError("Require beta0 > 0 and h_q >= max(1, 2*beta0).")
    if cfg.eta_l2_radius < 0.0 or cfg.scale_l2_radius < 0.0:
        raise ValueError("Parameter radii must be nonnegative (zero selects automatic radii).")
    if not (0.5 < cfg.nominal_lr_exponent <= 1.0):
        raise ValueError("nominal_lr_exponent must lie in (0.5, 1].")
    if cfg.nominal_lr_scale <= 0.0:
        raise ValueError("nominal_lr_scale must be positive.")
    if cfg.dp_tolerance <= 0.0 or cfg.dp_max_iterations < 1:
        raise ValueError("DP tolerance and maximum iterations must be positive.")
    if not cfg.perturbation_grid:
        raise ValueError("perturbation_grid must contain at least one probability.")
    if len(set(cfg.perturbation_grid)) != len(cfg.perturbation_grid):
        raise ValueError("perturbation_grid must not contain duplicate probabilities.")
    probabilities = (cfg.evaluation_crash_prob,) + tuple(cfg.perturbation_grid)
    if any(prob <= 0.0 or prob >= 1.0 for prob in probabilities):
        raise ValueError("All evaluation crash probabilities must lie strictly in (0, 1).")


def make_corridor_mdp(env: CorridorConfig, crash_prob: Optional[float] = None) -> CorridorMDP:
    """Build the continuing MDP, preserving support for every p in (0, 1)."""
    p_crash = env.nominal_crash_prob if crash_prob is None else float(crash_prob)
    if not (0.0 < p_crash < 1.0):
        raise ValueError("crash_prob must lie strictly in (0, 1).")

    start = 0
    risky_states = tuple(range(1, 1 + env.risky_len))
    safe_states = tuple(range(1 + env.risky_len, 1 + env.risky_len + env.safe_len))
    goal = 1 + env.risky_len + env.safe_len
    crash = goal + 1
    n_states = crash + 1
    n_actions = 2

    transitions = np.zeros((n_states, n_actions, n_states), dtype=np.float64)
    rewards = np.full((n_states, n_actions), env.step_reward, dtype=np.float64)

    transitions[start, 0, risky_states[0]] = 1.0
    transitions[start, 1, safe_states[0]] = 1.0

    for route_index, state in enumerate(risky_states):
        good_next = goal if route_index == len(risky_states) - 1 else risky_states[route_index + 1]
        for action in range(n_actions):
            transitions[state, action, good_next] = 1.0 - p_crash
            transitions[state, action, crash] = p_crash

    for route_index, state in enumerate(safe_states):
        next_state = goal if route_index == len(safe_states) - 1 else safe_states[route_index + 1]
        transitions[state, :, next_state] = 1.0

    transitions[goal, :, start] = 1.0
    transitions[crash, :, start] = 1.0
    rewards[goal, :] = env.goal_reward
    rewards[crash, :] = env.crash_reward

    if not np.allclose(transitions.sum(axis=-1), 1.0, atol=1e-12):
        raise RuntimeError("Invalid transition kernel: rows do not sum to one.")
    return CorridorMDP(
        transitions=transitions,
        rewards=rewards,
        start_state=start,
        goal_state=goal,
        crash_state=crash,
        risky_states=risky_states,
        safe_states=safe_states,
    )


def behavior_action_probabilities(mdp: CorridorMDP, env: CorridorConfig) -> Array:
    probabilities = np.full((mdp.n_states, mdp.n_actions), 0.5, dtype=np.float64)
    probabilities[mdp.start_state] = (env.behavior_risky_prob, 1.0 - env.behavior_risky_prob)
    return probabilities


def behavior_stationary_distribution(mdp: CorridorMDP, env: CorridorConfig) -> Tuple[Array, Array]:
    """Return exact-enough stationary state and state-action probabilities."""
    policy = behavior_action_probabilities(mdp, env)
    state_kernel = np.einsum("sa,sat->st", policy, mdp.transitions, optimize=True)
    distribution = np.full(mdp.n_states, 1.0 / mdp.n_states, dtype=np.float64)
    for _ in range(100_000):
        updated = distribution @ state_kernel
        if np.max(np.abs(updated - distribution)) < 1e-15:
            distribution = updated
            break
        distribution = updated
    distribution /= distribution.sum()
    return distribution, distribution[:, None] * policy


def project_l2(vector: Array, radius: float) -> Tuple[Array, bool]:
    """Project on an L2 ball and report whether the radial boundary was hit."""
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm):
        raise FloatingPointError("Non-finite parameter norm before projection.")
    if norm <= radius or norm == 0.0:
        return vector, False
    return vector * (radius / norm), True


def chi2_worst_case_distribution(values: Array, probabilities: Array, delta: float) -> Array:
    """Solve a finite-support chi-square inner problem without scipy or a grid.

    The returned vector includes zeros outside the nominal support.  For an
    active ambiguity constraint it is q_j proportional to
    p_j (eta - values_j)_+, with eta obtained by bisection on the exact dual
    derivative.
    """
    values = np.asarray(values, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or probabilities.shape != values.shape:
        raise ValueError("values and probabilities must be one-dimensional with equal shape.")
    if delta < 0.0:
        raise ValueError("delta must be nonnegative.")
    if np.any(probabilities < -1e-15):
        raise ValueError("probabilities must be nonnegative.")

    support = probabilities > 1e-15
    if not np.any(support):
        raise ValueError("probabilities have empty support.")
    p = probabilities[support].copy()
    p /= p.sum()
    v = values[support]
    result = np.zeros_like(probabilities, dtype=np.float64)

    if delta <= 1e-15:
        result[support] = p
        return result

    minimum = float(np.min(v))
    minimizers = np.isclose(v, minimum, rtol=0.0, atol=1e-13 * max(1.0, abs(minimum)))
    minimum_mass = float(p[minimizers].sum())
    maximum_divergence_at_minimum = 1.0 / minimum_mass - 1.0
    if delta >= maximum_divergence_at_minimum - 1e-13:
        q = np.zeros_like(p)
        q[minimizers] = p[minimizers] / minimum_mass
        result[support] = q
        return result

    # On an interval with active set A={j:v_j<eta}, the derivative equation
    # sqrt(Z) = sqrt(1+delta) E[(eta-V)_+] has the closed-form solution
    #
    #   eta = mean_A + std_A / sqrt((1+delta) P(A) - 1).
    #
    # Checking the finitely many sorted active sets is both faster and more
    # accurate than an eta grid or a generic scalar optimizer.
    order = np.argsort(v, kind="stable")
    sorted_v = v[order]
    sorted_p = p[order]
    cumulative_p = np.cumsum(sorted_p)
    cumulative_pv = np.cumsum(sorted_p * sorted_v)
    cumulative_pv2 = np.cumsum(sorted_p * sorted_v * sorted_v)
    eta: Optional[float] = None
    end = 0
    while end < len(sorted_v):
        # Include a complete tie group in the candidate active set.  Points at
        # eta itself receive zero q mass, so either convention is equivalent.
        group_value = sorted_v[end]
        while end + 1 < len(sorted_v) and math.isclose(
            float(sorted_v[end + 1]), float(group_value), rel_tol=0.0, abs_tol=1e-13
        ):
            end += 1
        active_mass = float(cumulative_p[end])
        denominator_squared = (1.0 + delta) * active_mass - 1.0
        if denominator_squared > 1e-15:
            active_mean = float(cumulative_pv[end] / active_mass)
            active_second = float(cumulative_pv2[end] / active_mass)
            active_variance = max(active_second - active_mean * active_mean, 0.0)
            candidate = active_mean + math.sqrt(active_variance / denominator_squared)
            lower = float(sorted_v[end])
            upper = float(sorted_v[end + 1]) if end + 1 < len(sorted_v) else math.inf
            tolerance = 1e-12 * max(1.0, abs(candidate), abs(lower))
            if candidate >= lower - tolerance and candidate <= upper + tolerance:
                eta = candidate
                break
        end += 1

    if eta is None:
        # Degenerate roundoff fallback.  This path is not expected for a
        # positive radius below the point-mass threshold.
        c_delta = math.sqrt(1.0 + delta)

        def derivative(candidate: float) -> float:
            positive_part = np.maximum(candidate - v, 0.0)
            first = float(p @ positive_part)
            second = float(p @ (positive_part * positive_part))
            if second <= 0.0:
                return 1.0
            return 1.0 - c_delta * first / math.sqrt(second)

        scale = max(float(np.ptp(v)), float(np.max(np.abs(v))), 1.0)
        low = minimum
        high = float(np.max(v)) + scale
        while derivative(high) >= 0.0:
            high = minimum + 2.0 * (high - minimum)
        for _ in range(100):
            midpoint = 0.5 * (low + high)
            if derivative(midpoint) > 0.0:
                low = midpoint
            else:
                high = midpoint
        eta = 0.5 * (low + high)
    positive_part = np.maximum(eta - v, 0.0)
    normalizer = float(p @ positive_part)
    if normalizer <= 0.0:
        raise RuntimeError("Degenerate chi-square dual solution.")
    q = p * positive_part / normalizer
    q /= q.sum()
    result[support] = q
    return result


def chi2_robust_expectation(values: Array, probabilities: Array, delta: float) -> float:
    q = chi2_worst_case_distribution(values, probabilities, delta)
    return float(q @ np.asarray(values, dtype=np.float64))


def floor_variational_solution(
    values: Array,
    probabilities: Array,
    delta: float,
    ell: float,
) -> Tuple[float, float, float]:
    """Return the exact floor-surrogate value, dual eta, and scale u.

    Its derivative is
        1 - sqrt(1+delta) E[(eta-V)_+] / max(ell, sqrt(Z)).
    The derivative is monotone, so a scalar bisection is sufficient.
    """
    if delta <= 0.0:
        raise ValueError("The floor reference is defined here for delta > 0.")
    if ell <= 0.0:
        raise ValueError("ell must be positive.")
    values = np.asarray(values, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    support = probabilities > 1e-15
    if not np.any(support):
        raise ValueError("probabilities have empty support.")
    p = probabilities[support].copy()
    p /= p.sum()
    v = values[support]
    c_delta = math.sqrt(1.0 + delta)

    def moments(eta: float) -> Tuple[float, float]:
        positive_part = np.maximum(eta - v, 0.0)
        first = float(p @ positive_part)
        second = float(p @ (positive_part * positive_part))
        return first, second

    def derivative(eta: float) -> float:
        first, second = moments(eta)
        return 1.0 - c_delta * first / max(ell, math.sqrt(max(second, 0.0)))

    minimum = float(np.min(v))
    scale = max(float(np.ptp(v)), float(np.max(np.abs(v))), ell, 1.0)
    low = minimum
    high = float(np.max(v)) + scale
    expansions = 0
    while derivative(high) >= 0.0:
        high = minimum + 2.0 * (high - minimum)
        expansions += 1
        if expansions > 200 or not np.isfinite(high):
            raise RuntimeError("Could not bracket the floor-surrogate maximizer.")

    for _ in range(100):
        midpoint = 0.5 * (low + high)
        if derivative(midpoint) > 0.0:
            low = midpoint
        else:
            high = midpoint
    eta = 0.5 * (low + high)
    _, second = moments(eta)
    u = max(ell, math.sqrt(max(second, 0.0)))
    value = float(eta - 0.5 * c_delta * (second / u + u))
    return value, float(eta), float(u)


def floor_robust_expectation(values: Array, probabilities: Array, delta: float, ell: float) -> float:
    """Evaluate the exact unconstrained floor-surrogate robust expectation."""
    value, _, _ = floor_variational_solution(values, probabilities, delta, ell)
    return value


def nominal_bellman(q_values: Array, mdp: CorridorMDP, gamma: float) -> Array:
    values = np.max(q_values, axis=1)
    return mdp.rewards + gamma * np.einsum("sat,t->sa", mdp.transitions, values, optimize=True)


def robust_bellman(q_values: Array, mdp: CorridorMDP, gamma: float, delta: float) -> Array:
    values = np.max(q_values, axis=1)
    result = np.empty_like(q_values, dtype=np.float64)
    for state in range(mdp.n_states):
        for action in range(mdp.n_actions):
            result[state, action] = mdp.rewards[state, action] + gamma * chi2_robust_expectation(
                values, mdp.transitions[state, action], delta
            )
    return result


def floor_robust_bellman(q_values: Array, mdp: CorridorMDP, gamma: float, delta: float, ell: float) -> Array:
    values = np.max(q_values, axis=1)
    result = np.empty_like(q_values, dtype=np.float64)
    for state in range(mdp.n_states):
        for action in range(mdp.n_actions):
            result[state, action] = mdp.rewards[state, action] + gamma * floor_robust_expectation(
                values, mdp.transitions[state, action], delta, ell
            )
    return result


def value_iteration(
    operator,
    shape: Tuple[int, int],
    tolerance: float,
    max_iterations: int,
) -> Array:
    q_values = np.zeros(shape, dtype=np.float64)
    for _ in range(max_iterations):
        updated = np.asarray(operator(q_values), dtype=np.float64)
        if not np.all(np.isfinite(updated)):
            raise FloatingPointError("Non-finite value-iteration iterate.")
        if float(np.max(np.abs(updated - q_values))) <= tolerance:
            return updated
        q_values = updated
    residual = float(np.max(np.abs(operator(q_values) - q_values)))
    raise RuntimeError(f"Value iteration did not converge; residual={residual:.3e}.")


def exact_references(env: CorridorConfig, cfg: AlgorithmConfig, mdp: CorridorMDP) -> Tuple[Array, Array, Array]:
    shape = mdp.rewards.shape
    nominal_q = value_iteration(
        lambda q: nominal_bellman(q, mdp, env.gamma),
        shape,
        cfg.dp_tolerance,
        cfg.dp_max_iterations,
    )
    robust_q = value_iteration(
        lambda q: robust_bellman(q, mdp, env.gamma, cfg.chi2_delta),
        shape,
        cfg.dp_tolerance,
        cfg.dp_max_iterations,
    )
    floor_q = value_iteration(
        lambda q: floor_robust_bellman(q, mdp, env.gamma, cfg.chi2_delta, cfg.ell),
        shape,
        cfg.dp_tolerance,
        cfg.dp_max_iterations,
    )
    return nominal_q, robust_q, floor_q


def greedy_policy(q_values: Array) -> Array:
    """Deterministic greedy policy with stable action-index tie breaking."""
    return np.argmax(np.asarray(q_values), axis=1).astype(np.int64)


def exact_policy_value(mdp: CorridorMDP, policy: Array, gamma: float) -> Array:
    states = np.arange(mdp.n_states)
    policy = np.asarray(policy, dtype=np.int64)
    transition_policy = mdp.transitions[states, policy]
    reward_policy = mdp.rewards[states, policy]
    system = np.eye(mdp.n_states, dtype=np.float64) - gamma * transition_policy
    return np.linalg.solve(system, reward_policy)


def exact_robust_policy_value(
    mdp: CorridorMDP,
    policy: Array,
    gamma: float,
    delta: float,
    tolerance: float,
    max_iterations: int,
) -> Array:
    policy = np.asarray(policy, dtype=np.int64)
    values = np.zeros(mdp.n_states, dtype=np.float64)
    for _ in range(max_iterations):
        updated = np.empty_like(values)
        for state in range(mdp.n_states):
            action = int(policy[state])
            updated[state] = mdp.rewards[state, action] + gamma * chi2_robust_expectation(
                values, mdp.transitions[state, action], delta
            )
        if float(np.max(np.abs(updated - values))) <= tolerance:
            return updated
        values = updated
    raise RuntimeError("Robust policy evaluation did not converge.")


def exact_optimal_start_return(env: CorridorConfig, cfg: AlgorithmConfig, crash_prob: float) -> Tuple[float, Array]:
    mdp = make_corridor_mdp(env, crash_prob)
    q_star = value_iteration(
        lambda q: nominal_bellman(q, mdp, env.gamma),
        mdp.rewards.shape,
        cfg.dp_tolerance,
        cfg.dp_max_iterations,
    )
    return float(np.max(q_star[mdp.start_state])), q_star


class ContinuingBehaviorTrajectory:
    """A single cursor that is never restarted at a stage boundary."""

    def __init__(self, mdp: CorridorMDP, env: CorridorConfig, rng: np.random.Generator):
        self.mdp = mdp
        self.env = env
        self.rng = rng
        self.state = mdp.start_state
        self.transitions_read = 0
        self.state_action_counts = np.zeros((mdp.n_states, mdp.n_actions), dtype=np.int64)

    def next(self) -> Tuple[int, int, float, int]:
        state = self.state
        if state == self.mdp.start_state:
            action = 0 if self.rng.random() < self.env.behavior_risky_prob else 1
        else:
            action = int(self.rng.integers(self.mdp.n_actions))
        next_state = int(self.rng.choice(self.mdp.n_states, p=self.mdp.transitions[state, action]))
        reward = float(self.mdp.rewards[state, action])
        self.state_action_counts[state, action] += 1
        self.transitions_read += 1
        self.state = next_state
        return state, action, reward, next_state


def automatic_parameter_radii(
    mdp: CorridorMDP,
    env: CorridorConfig,
    cfg: AlgorithmConfig,
) -> Tuple[float, float]:
    """Safe global radii able to hold independent tabular coordinates."""
    c_delta = math.sqrt(1.0 + cfg.chi2_delta)
    b_q = 1.0 / (1.0 - env.gamma)
    b_eta_floor = ((c_delta + 1.0) * b_q + 0.5 * c_delta * cfg.ell) / (c_delta - 1.0)
    n_pairs = mdp.n_states * mdp.n_actions
    eta_radius = cfg.eta_l2_radius or math.sqrt(n_pairs) * b_eta_floor
    max_scale = b_eta_floor + b_q
    # rho = u - ell; this loose bound avoids making the projection an
    # unintended source of approximation error.
    scale_radius = cfg.scale_l2_radius or math.sqrt(n_pairs) * max_scale
    return float(eta_radius), float(scale_radius)


def stage1_stepsize(
    mdp: CorridorMDP,
    env: CorridorConfig,
    cfg: AlgorithmConfig,
    eta_radius: float,
    scale_radius: float,
) -> float:
    if cfg.stage1_step_mode == "constant":
        return float(cfg.stage1_stepsize)
    c_delta = math.sqrt(1.0 + cfg.chi2_delta)
    b_q = 1.0 / (1.0 - env.gamma)
    h_bound = eta_radius + b_q
    g_ell = math.sqrt(
        (1.0 + c_delta * h_bound / cfg.ell) ** 2
        + 0.25 * c_delta**2 * (1.0 + h_bound**2 / cfg.ell**2) ** 2
    )
    radius_w = math.sqrt(eta_radius**2 + scale_radius**2)
    n_samples = cfg.stage1_samples
    lambda_n = 1.0 + math.log(n_samples + 1.0) + math.log(n_samples + 1.0) ** 2
    return float(cfg.theory_step_multiplier * radius_w / (g_ell * math.sqrt(n_samples * lambda_n)))


def _update_nominal_q(
    q_values: Array,
    visit_counts: Array,
    state: int,
    action: int,
    reward: float,
    next_state: int,
    env: CorridorConfig,
    cfg: AlgorithmConfig,
) -> None:
    visit_counts[state, action] += 1
    learning_rate = cfg.nominal_lr_scale / float(visit_counts[state, action]) ** cfg.nominal_lr_exponent
    learning_rate = min(1.0, learning_rate)
    target = reward + env.gamma * float(np.max(q_values[next_state]))
    q_values[state, action] += learning_rate * (target - q_values[state, action])


def _diagnostic_row(
    outer_block: int,
    transitions_read: int,
    robust_q: Array,
    nominal_q: Array,
    mdp: CorridorMDP,
    env: CorridorConfig,
    cfg: AlgorithmConfig,
    robust_reference: Array,
    floor_reference: Array,
    nominal_reference: Array,
    evaluation_mdp: CorridorMDP,
    evaluation_optimal_return: float,
    robust_oracle_policy: Array,
    nominal_oracle_policy: Array,
    block_diagnostics: Dict[str, float],
) -> Dict[str, float]:
    robust_policy = greedy_policy(robust_q)
    nominal_policy = greedy_policy(nominal_q)
    robust_eval_return = float(exact_policy_value(evaluation_mdp, robust_policy, env.gamma)[evaluation_mdp.start_state])
    nominal_eval_return = float(exact_policy_value(evaluation_mdp, nominal_policy, env.gamma)[evaluation_mdp.start_state])
    robust_train_return = float(exact_policy_value(mdp, robust_policy, env.gamma)[mdp.start_state])
    nominal_train_return = float(exact_policy_value(mdp, nominal_policy, env.gamma)[mdp.start_state])
    robust_worst_case = float(
        exact_robust_policy_value(
            mdp,
            robust_policy,
            env.gamma,
            cfg.chi2_delta,
            cfg.dp_tolerance,
            cfg.dp_max_iterations,
        )[mdp.start_state]
    )
    robust_optimal_value = float(np.max(robust_reference[mdp.start_state]))
    row = {
        "outer_block": float(outer_block),
        "transitions": float(transitions_read),
        "robust_q_sup_error": float(np.max(np.abs(robust_q - robust_reference))),
        "floor_q_sup_error": float(np.max(np.abs(robust_q - floor_reference))),
        "floor_reference_bias": float(np.max(np.abs(floor_reference - robust_reference))),
        "floor_bias_bound": float(
            env.gamma * math.sqrt(1.0 + cfg.chi2_delta) * cfg.ell / (2.0 * (1.0 - env.gamma))
        ),
        "robust_bellman_residual": float(
            np.max(np.abs(robust_q - robust_bellman(robust_q, mdp, env.gamma, cfg.chi2_delta)))
        ),
        "floor_bellman_residual": float(
            np.max(
                np.abs(
                    robust_q
                    - floor_robust_bellman(robust_q, mdp, env.gamma, cfg.chi2_delta, cfg.ell)
                )
            )
        ),
        "nominal_q_sup_error": float(np.max(np.abs(nominal_q - nominal_reference))),
        "robust_start_action": float(robust_policy[mdp.start_state]),
        "nominal_start_action": float(nominal_policy[mdp.start_state]),
        "robust_policy_nominal_return": robust_train_return,
        "nominal_policy_nominal_return": nominal_train_return,
        "robust_policy_perturbed_return": robust_eval_return,
        "nominal_policy_perturbed_return": nominal_eval_return,
        "perturbed_optimal_return": evaluation_optimal_return,
        "robust_policy_perturbed_gap": max(0.0, evaluation_optimal_return - robust_eval_return),
        "nominal_policy_perturbed_gap": max(0.0, evaluation_optimal_return - nominal_eval_return),
        "robust_policy_advantage": robust_eval_return - nominal_eval_return,
        "robust_policy_worst_case_return": robust_worst_case,
        "robust_policy_worst_case_gap": max(0.0, robust_optimal_value - robust_worst_case),
        "oracle_robust_perturbed_return": float(
            exact_policy_value(evaluation_mdp, robust_oracle_policy, env.gamma)[evaluation_mdp.start_state]
        ),
        "oracle_nominal_perturbed_return": float(
            exact_policy_value(evaluation_mdp, nominal_oracle_policy, env.gamma)[evaluation_mdp.start_state]
        ),
    }
    row.update(block_diagnostics)
    return row


def run_experiment(env: CorridorConfig, cfg: AlgorithmConfig) -> RunResult:
    """Run one seed and return all paper-facing data in memory."""
    validate_configs(env, cfg)
    rng = np.random.default_rng(cfg.seed)
    mdp = make_corridor_mdp(env)
    nominal_reference, robust_reference, floor_reference = exact_references(env, cfg, mdp)
    robust_oracle_policy = greedy_policy(robust_reference)
    nominal_oracle_policy = greedy_policy(nominal_reference)
    evaluation_mdp = make_corridor_mdp(env, cfg.evaluation_crash_prob)
    evaluation_optimal_return, _ = exact_optimal_start_return(env, cfg, cfg.evaluation_crash_prob)

    eta_radius, scale_radius = automatic_parameter_radii(mdp, env, cfg)
    alpha_n = stage1_stepsize(mdp, env, cfg, eta_radius, scale_radius)
    b_q = 1.0 / (1.0 - env.gamma)
    c_delta = math.sqrt(1.0 + cfg.chi2_delta)
    n_pairs = mdp.n_states * mdp.n_actions

    # Check whether the independent pointwise optimizers at the unconstrained
    # floor fixed point fit inside the learner's shared compact parameter sets.
    # This does not replace the manuscript's uniform structural-error term, but
    # it distinguishes an active projection restriction at the reported oracle.
    floor_values = np.max(floor_reference, axis=1)
    floor_oracle_eta = np.empty(n_pairs, dtype=np.float64)
    floor_oracle_rho = np.empty(n_pairs, dtype=np.float64)
    for state in range(mdp.n_states):
        for action in range(mdp.n_actions):
            index = state * mdp.n_actions + action
            _, eta_star, u_star = floor_variational_solution(
                floor_values,
                mdp.transitions[state, action],
                cfg.chi2_delta,
                cfg.ell,
            )
            floor_oracle_eta[index] = eta_star
            floor_oracle_rho[index] = max(0.0, u_star - cfg.ell)
    floor_oracle_eta_norm = float(np.linalg.norm(floor_oracle_eta))
    floor_oracle_rho_norm = float(np.linalg.norm(floor_oracle_rho))
    floor_oracle_inside_balls = float(
        floor_oracle_eta_norm <= eta_radius + 1e-10
        and floor_oracle_rho_norm <= scale_radius + 1e-10
    )

    trajectory = ContinuingBehaviorTrajectory(mdp, env, rng)
    robust_raw_q = np.zeros((mdp.n_states, mdp.n_actions), dtype=np.float64)
    robust_q = np.clip(robust_raw_q, -b_q, b_q)
    nominal_q = np.zeros_like(robust_q)
    nominal_visits = np.zeros_like(robust_q, dtype=np.int64)

    empty_diagnostics = {
        "stage1_stepsize": alpha_n,
        "stage1_gradient_rms": 0.0,
        "stage1_gradient_max": 0.0,
        "stage1_population_target_sup_error": 0.0,
        "stage1_population_target_mean_error": 0.0,
        "floor_oracle_eta_l2_norm": floor_oracle_eta_norm,
        "floor_oracle_scale_rho_l2_norm": floor_oracle_rho_norm,
        "floor_oracle_inside_parameter_balls": floor_oracle_inside_balls,
        "eta_projection_fraction": 0.0,
        "scale_projection_fraction": 0.0,
        "scale_floor_fraction": 0.0,
        "eta_norm": 0.0,
        "scale_rho_norm": 0.0,
        "min_state_action_visits": 0.0,
    }
    metrics = [
        _diagnostic_row(
            0,
            0,
            robust_q,
            nominal_q,
            mdp,
            env,
            cfg,
            robust_reference,
            floor_reference,
            nominal_reference,
            evaluation_mdp,
            evaluation_optimal_return,
            robust_oracle_policy,
            nominal_oracle_policy,
            empty_diagnostics,
        )
    ]

    last_eta_bar = np.zeros(n_pairs, dtype=np.float64)
    last_u_bar = np.full(n_pairs, cfg.ell, dtype=np.float64)

    for outer_block in range(1, cfg.outer_blocks + 1):
        frozen_q = np.clip(robust_raw_q, -b_q, b_q)
        frozen_values = np.max(frozen_q, axis=1)

        # One-hot tabular specialization of (nu, Theta).  Since Theta starts
        # diagonal and receives only e_i e_i^T gradients, rho stores its
        # diagonal exactly and u_i = ell + rho_i.
        eta = np.zeros(n_pairs, dtype=np.float64)
        rho = np.zeros(n_pairs, dtype=np.float64)
        eta_sum = np.zeros_like(eta)
        rho_sum = np.zeros_like(rho)
        gradient_square_sum = 0.0
        gradient_max = 0.0
        eta_projection_hits = 0
        scale_projection_hits = 0
        scale_floor_hits = 0

        for _ in range(cfg.stage1_samples):
            state, action, reward, next_state = trajectory.next()
            _update_nominal_q(
                nominal_q,
                nominal_visits,
                state,
                action,
                reward,
                next_state,
                env,
                cfg,
            )
            index = state * mdp.n_actions + action

            # Average exactly the pre-update iterates k=0,...,N-1.
            eta_sum += eta
            rho_sum += rho

            old_eta = float(eta[index])
            old_u = cfg.ell + float(rho[index])
            positive_part = max(old_eta - float(frozen_values[next_state]), 0.0)
            eta_gradient = 1.0 - c_delta * positive_part / old_u
            scale_gradient = 0.5 * c_delta * ((positive_part / old_u) ** 2 - 1.0)
            gradient_norm = math.hypot(eta_gradient, scale_gradient)
            if not np.isfinite(gradient_norm):
                raise FloatingPointError("Non-finite Stage-1 gradient.")
            gradient_square_sum += gradient_norm * gradient_norm
            gradient_max = max(gradient_max, gradient_norm)

            # Both gradients use the same old iterate; updates are simultaneous.
            eta_candidate = eta.copy()
            rho_candidate = rho.copy()
            eta_candidate[index] = old_eta + alpha_n * eta_gradient
            raw_rho = float(rho[index]) + alpha_n * scale_gradient
            if raw_rho < 0.0:
                scale_floor_hits += 1
            rho_candidate[index] = max(raw_rho, 0.0)
            eta, eta_hit = project_l2(eta_candidate, eta_radius)
            rho, scale_hit = project_l2(rho_candidate, scale_radius)
            eta_projection_hits += int(eta_hit)
            scale_projection_hits += int(scale_hit)

        eta_bar = eta_sum / float(cfg.stage1_samples)
        rho_bar = rho_sum / float(cfg.stage1_samples)
        u_bar = cfg.ell + rho_bar
        last_eta_bar = eta_bar.copy()
        last_u_bar = u_bar.copy()

        learned_population_next = np.empty((mdp.n_states, mdp.n_actions), dtype=np.float64)
        exact_floor_next = np.empty_like(learned_population_next)
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
                exact_floor_next[state, action] = floor_robust_expectation(
                    frozen_values,
                    mdp.transitions[state, action],
                    cfg.chi2_delta,
                    cfg.ell,
                )
        stage1_population_error = np.abs(learned_population_next - exact_floor_next)

        # The theory resets the Q regressor to zero in every outer block.
        q_work = np.zeros_like(robust_raw_q)
        for q_step in range(cfg.q_stage_samples):
            state, action, reward, next_state = trajectory.next()
            _update_nominal_q(
                nominal_q,
                nominal_visits,
                state,
                action,
                reward,
                next_state,
                env,
                cfg,
            )
            index = state * mdp.n_actions + action
            positive_part = max(float(eta_bar[index]) - float(frozen_values[next_state]), 0.0)
            robust_next_sample = float(eta_bar[index]) - 0.5 * c_delta * (
                positive_part * positive_part / float(u_bar[index]) + float(u_bar[index])
            )
            target = reward + env.gamma * robust_next_sample
            beta_m = cfg.beta0 / (float(q_step) + cfg.h_q)
            q_work[state, action] += beta_m * (target - q_work[state, action])

        robust_raw_q = q_work
        robust_q = np.clip(robust_raw_q, -b_q, b_q)
        block_diagnostics = {
            "stage1_stepsize": alpha_n,
            "stage1_gradient_rms": math.sqrt(gradient_square_sum / cfg.stage1_samples),
            "stage1_gradient_max": gradient_max,
            "stage1_population_target_sup_error": float(np.max(stage1_population_error)),
            "stage1_population_target_mean_error": float(np.mean(stage1_population_error)),
            "floor_oracle_eta_l2_norm": floor_oracle_eta_norm,
            "floor_oracle_scale_rho_l2_norm": floor_oracle_rho_norm,
            "floor_oracle_inside_parameter_balls": floor_oracle_inside_balls,
            "eta_projection_fraction": eta_projection_hits / float(cfg.stage1_samples),
            "scale_projection_fraction": scale_projection_hits / float(cfg.stage1_samples),
            "scale_floor_fraction": scale_floor_hits / float(cfg.stage1_samples),
            "eta_norm": float(np.linalg.norm(eta_bar)),
            "scale_rho_norm": float(np.linalg.norm(rho_bar)),
            "min_state_action_visits": float(np.min(trajectory.state_action_counts)),
        }
        metrics.append(
            _diagnostic_row(
                outer_block,
                trajectory.transitions_read,
                robust_q,
                nominal_q,
                mdp,
                env,
                cfg,
                robust_reference,
                floor_reference,
                nominal_reference,
                evaluation_mdp,
                evaluation_optimal_return,
                robust_oracle_policy,
                nominal_oracle_policy,
                block_diagnostics,
            )
        )

    robust_policy = greedy_policy(robust_q)
    nominal_policy = greedy_policy(nominal_q)
    perturbation_metrics: List[Dict[str, float]] = []
    for crash_probability in cfg.perturbation_grid:
        perturbed_mdp = make_corridor_mdp(env, crash_probability)
        optimal_return, _ = exact_optimal_start_return(env, cfg, crash_probability)
        robust_return = float(exact_policy_value(perturbed_mdp, robust_policy, env.gamma)[mdp.start_state])
        nominal_return = float(exact_policy_value(perturbed_mdp, nominal_policy, env.gamma)[mdp.start_state])
        oracle_robust_return = float(
            exact_policy_value(perturbed_mdp, robust_oracle_policy, env.gamma)[mdp.start_state]
        )
        oracle_nominal_return = float(
            exact_policy_value(perturbed_mdp, nominal_oracle_policy, env.gamma)[mdp.start_state]
        )
        chi2_distance = (crash_probability - env.nominal_crash_prob) ** 2 / (
            env.nominal_crash_prob * (1.0 - env.nominal_crash_prob)
        )
        perturbation_metrics.append(
            {
                "crash_probability": float(crash_probability),
                "chi2_distance_from_nominal": float(chi2_distance),
                "inside_chi2_radius": float(chi2_distance <= cfg.chi2_delta + 1e-12),
                "optimal_return": optimal_return,
                "robust_policy_return": robust_return,
                "nominal_policy_return": nominal_return,
                "oracle_robust_policy_return": oracle_robust_return,
                "oracle_nominal_policy_return": oracle_nominal_return,
                "robust_policy_gap": max(0.0, optimal_return - robust_return),
                "nominal_policy_gap": max(0.0, optimal_return - nominal_return),
                "robust_policy_advantage": robust_return - nominal_return,
                "robust_start_action": float(robust_policy[mdp.start_state]),
                "nominal_start_action": float(nominal_policy[mdp.start_state]),
            }
        )

    state_stationary, sa_stationary = behavior_stationary_distribution(mdp, env)
    q_gain_p = 2.0 * float(np.min(sa_stationary)) * cfg.beta0
    ambiguity_upper_crash_probability = env.nominal_crash_prob + math.sqrt(
        cfg.chi2_delta * env.nominal_crash_prob * (1.0 - env.nominal_crash_prob)
    )
    metadata: Dict[str, object] = {
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": asdict(env),
        "algorithm": {**asdict(cfg), "perturbation_grid": list(cfg.perturbation_grid)},
        "n_states": mdp.n_states,
        "n_actions": mdp.n_actions,
        "stage1_stepsize_used": alpha_n,
        "eta_l2_radius_used": eta_radius,
        "scale_l2_radius_used": scale_radius,
        "behavior_min_state_probability": float(np.min(state_stationary)),
        "behavior_min_state_action_probability": float(np.min(sa_stationary)),
        "q_gain_p": q_gain_p,
        "clean_q_rate_condition_satisfied": bool(q_gain_p > 1.0),
        "floor_oracle_eta_l2_norm": floor_oracle_eta_norm,
        "floor_oracle_scale_rho_l2_norm": floor_oracle_rho_norm,
        "floor_oracle_inside_parameter_balls": bool(floor_oracle_inside_balls),
        "ambiguity_upper_crash_probability": float(ambiguity_upper_crash_probability),
        "oracle_nominal_start_action": int(nominal_oracle_policy[mdp.start_state]),
        "oracle_robust_start_action": int(robust_oracle_policy[mdp.start_state]),
        "oracle_nominal_start_q": nominal_reference[mdp.start_state].tolist(),
        "oracle_robust_start_q": robust_reference[mdp.start_state].tolist(),
        "floor_reference_sup_bias": float(np.max(np.abs(floor_reference - robust_reference))),
        "floor_bias_bound": float(
            env.gamma * c_delta * cfg.ell / (2.0 * (1.0 - env.gamma))
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
        "floor_oracle_u": (cfg.ell + floor_oracle_rho).reshape(mdp.n_states, mdp.n_actions),
    }
    return RunResult(metrics=metrics, perturbation_metrics=perturbation_metrics, metadata=metadata, arrays=arrays)


def write_csv(path: Path, rows: Sequence[Dict[str, float]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_run(result: RunResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "metrics.csv", result.metrics)
    write_csv(output_dir / "perturbation_metrics.csv", result.perturbation_metrics)
    (output_dir / "metadata.json").write_text(
        json.dumps(result.metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(output_dir / "arrays.npz", **result.arrays)


def parse_float_tuple(text: str) -> Tuple[float, ...]:
    values = tuple(float(piece.strip()) for piece in text.split(",") if piece.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated float.")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", default="runs_variational_chi2_tabular/single_run")
    env_defaults = CorridorConfig()
    algorithm_defaults = AlgorithmConfig()
    for field in fields(CorridorConfig):
        default = getattr(env_defaults, field.name)
        parser.add_argument("--" + field.name.replace("_", "-"), type=type(default), default=default)
    for field in fields(AlgorithmConfig):
        if field.name == "perturbation_grid":
            parser.add_argument(
                "--perturbation-grid",
                type=parse_float_tuple,
                default=algorithm_defaults.perturbation_grid,
                help="comma-separated risky-route crash probabilities",
            )
            continue
        default = getattr(algorithm_defaults, field.name)
        parser.add_argument("--" + field.name.replace("_", "-"), type=type(default), default=default)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    env_field_names = {field.name for field in fields(CorridorConfig)}
    algorithm_field_names = {field.name for field in fields(AlgorithmConfig)}
    raw = vars(args)
    env = CorridorConfig(**{name: raw[name] for name in env_field_names})
    cfg = AlgorithmConfig(**{name: raw[name] for name in algorithm_field_names})
    result = run_experiment(env, cfg)
    output_dir = Path(args.output_dir).resolve()
    save_run(result, output_dir)
    final = result.metrics[-1]
    print(f"output_dir={output_dir}")
    print(
        "final: "
        f"transitions={int(final['transitions'])}, "
        f"robust_q_sup_error={final['robust_q_sup_error']:.6g}, "
        f"robust_action={int(final['robust_start_action'])}, "
        f"nominal_action={int(final['nominal_start_action'])}, "
        f"stress_advantage={final['robust_policy_advantage']:.6g}"
    )


if __name__ == "__main__":
    main()
