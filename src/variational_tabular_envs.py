"""Reusable tabular environments for variational robust-Q experiments.

The MiniCliff task in this module deliberately uses state-action rewards
``r(s, a)``.  Cliff and goal outcomes are explicit one-step marker states:
they carry their fixed reward and then reset deterministically to the start.
Changing the slip probability therefore changes only the transition kernel,
not the reward table.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import numpy as np


Array = np.ndarray

N_ROWS = 4
N_COLS = 6
N_STATES = N_ROWS * N_COLS

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
ACTION_NAMES: Tuple[str, ...] = ("up", "right", "down", "left")
ACTION_DELTAS: Tuple[Tuple[int, int], ...] = (
    (-1, 0),
    (0, 1),
    (1, 0),
    (0, -1),
)

STEP_REWARD = -0.01
CLIFF_REWARD = -1.0
GOAL_REWARD = 1.0


@dataclass(frozen=True)
class MiniCliffConfig:
    """Configuration shared by nominal and perturbed MiniCliff kernels."""

    gamma: float = 0.9
    nominal_slip_probability: float = 0.1
    behavior_goal_bias: float = 0.2


@dataclass(frozen=True)
class TabularMDP:
    """Finite tabular MDP with metadata needed by the paper experiments."""

    transitions: Array
    rewards: Array
    start_state: int
    state_coordinates: Array
    cliff_states: Tuple[int, ...]
    goal_state: int
    n_rows: int
    n_cols: int
    action_names: Tuple[str, ...]
    gamma: float
    slip_probability: float
    behavior_goal_bias: float
    name: str = "minicliff_4x6_continuing"

    @property
    def n_states(self) -> int:
        return int(self.transitions.shape[0])

    @property
    def n_actions(self) -> int:
        return int(self.transitions.shape[1])

    @property
    def marker_states(self) -> Tuple[int, ...]:
        return self.cliff_states + (self.goal_state,)

    def state_at(self, row: int, column: int) -> int:
        if not (0 <= row < self.n_rows and 0 <= column < self.n_cols):
            raise IndexError(f"Grid coordinate {(row, column)} is outside the MDP.")
        return int(row * self.n_cols + column)

    def coordinate_of(self, state: int) -> Tuple[int, int]:
        if not (0 <= state < self.n_states):
            raise IndexError(f"State {state} is outside [0, {self.n_states}).")
        row, column = self.state_coordinates[state]
        return int(row), int(column)


@dataclass(frozen=True)
class StationaryDistributions:
    """Exact linear-system stationary distributions and their residuals."""

    state_probabilities: Array
    state_action_probabilities: Array
    state_residual: float
    state_action_residual: float

    @property
    def residual(self) -> float:
        return max(self.state_residual, self.state_action_residual)

    @property
    def minimum_state_action_probability(self) -> float:
        return float(np.min(self.state_action_probabilities))


@dataclass(frozen=True)
class KernelChiSquareReport:
    """Rowwise ``D_chi(candidate || nominal)`` and support diagnostics."""

    row_divergences: Array
    support_violations: Array

    @property
    def support_preserved(self) -> bool:
        return not bool(np.any(self.support_violations))

    @property
    def maximum_divergence(self) -> float:
        return float(np.max(self.row_divergences))

    def within_radius(self, radius: float, tolerance: float = 1e-12) -> bool:
        if not math.isfinite(radius) or radius < 0.0:
            raise ValueError("radius must be finite and nonnegative.")
        return self.support_preserved and self.maximum_divergence <= radius + tolerance


def validate_minicliff_config(config: MiniCliffConfig) -> None:
    """Validate assumptions needed by the nominal learning experiment."""

    if not math.isfinite(config.gamma) or not (0.0 < config.gamma < 1.0):
        raise ValueError("gamma must be finite and lie strictly between zero and one.")
    if not math.isfinite(config.nominal_slip_probability) or not (
        0.0 < config.nominal_slip_probability < 1.0
    ):
        raise ValueError(
            "nominal_slip_probability must be finite and lie strictly between zero and one."
        )
    if not math.isfinite(config.behavior_goal_bias) or not (
        0.0 <= config.behavior_goal_bias < 1.0
    ):
        raise ValueError(
            "behavior_goal_bias must be finite and lie in [0, 1) so behavior has full support."
        )


def _validate_slip_probability(slip_probability: float) -> float:
    slip_probability = float(slip_probability)
    if not math.isfinite(slip_probability) or not (0.0 <= slip_probability <= 1.0):
        raise ValueError("slip_probability must be finite and lie in [0, 1].")
    return slip_probability


def _state_index(row: int, column: int) -> int:
    return row * N_COLS + column


def _executed_action_probabilities(action: int, slip_probability: float) -> Array:
    probabilities = np.full(len(ACTION_NAMES), slip_probability / 3.0, dtype=np.float64)
    probabilities[action] = 1.0 - slip_probability
    return probabilities


def make_minicliff_mdp(
    config: MiniCliffConfig = MiniCliffConfig(),
    slip_probability: Optional[float] = None,
) -> TabularMDP:
    """Construct the nominal task or a fixed-reward slip perturbation.

    States use row-major indices on a 4x6 grid.  The start is ``(3, 0)``,
    cliffs are ``(3, 1)`` through ``(3, 4)``, and the goal is ``(3, 5)``.
    Ordinary movement is clipped at walls.  Marker-state rows ignore slip and
    return deterministically to the start.
    """

    validate_minicliff_config(config)
    actual_slip = _validate_slip_probability(
        config.nominal_slip_probability if slip_probability is None else slip_probability
    )

    coordinates = np.array(
        [(row, column) for row in range(N_ROWS) for column in range(N_COLS)],
        dtype=np.int64,
    )
    start_state = _state_index(N_ROWS - 1, 0)
    cliff_states = tuple(_state_index(N_ROWS - 1, column) for column in range(1, N_COLS - 1))
    goal_state = _state_index(N_ROWS - 1, N_COLS - 1)
    marker_states = set(cliff_states + (goal_state,))

    transitions = np.zeros((N_STATES, len(ACTION_NAMES), N_STATES), dtype=np.float64)
    rewards = np.full((N_STATES, len(ACTION_NAMES)), STEP_REWARD, dtype=np.float64)
    rewards[list(cliff_states), :] = CLIFF_REWARD
    rewards[goal_state, :] = GOAL_REWARD

    for state, (row_value, column_value) in enumerate(coordinates):
        if state in marker_states:
            transitions[state, :, start_state] = 1.0
            continue

        row, column = int(row_value), int(column_value)
        for action in range(len(ACTION_NAMES)):
            execution_probabilities = _executed_action_probabilities(action, actual_slip)
            for executed_action, probability in enumerate(execution_probabilities):
                delta_row, delta_column = ACTION_DELTAS[executed_action]
                next_row = min(max(row + delta_row, 0), N_ROWS - 1)
                next_column = min(max(column + delta_column, 0), N_COLS - 1)
                next_state = _state_index(next_row, next_column)
                transitions[state, action, next_state] += probability

    mdp = TabularMDP(
        transitions=transitions,
        rewards=rewards,
        start_state=start_state,
        state_coordinates=coordinates,
        cliff_states=cliff_states,
        goal_state=goal_state,
        n_rows=N_ROWS,
        n_cols=N_COLS,
        action_names=ACTION_NAMES,
        gamma=config.gamma,
        slip_probability=actual_slip,
        behavior_goal_bias=config.behavior_goal_bias,
    )
    validate_minicliff_mdp(mdp)
    return mdp


def validate_minicliff_mdp(mdp: TabularMDP, tolerance: float = 1e-12) -> None:
    """Validate geometry, stochastic rows, rewards, and marker resets."""

    expected_transition_shape = (N_STATES, len(ACTION_NAMES), N_STATES)
    expected_reward_shape = (N_STATES, len(ACTION_NAMES))
    if mdp.transitions.shape != expected_transition_shape:
        raise ValueError(
            f"transitions must have shape {expected_transition_shape}, got {mdp.transitions.shape}."
        )
    if mdp.rewards.shape != expected_reward_shape:
        raise ValueError(f"rewards must have shape {expected_reward_shape}, got {mdp.rewards.shape}.")
    if not np.all(np.isfinite(mdp.transitions)) or np.any(mdp.transitions < -tolerance):
        raise ValueError("transitions must be finite and nonnegative.")
    if not np.allclose(mdp.transitions.sum(axis=-1), 1.0, rtol=0.0, atol=tolerance):
        raise ValueError("every transition row must sum to one.")
    if not np.all(np.isfinite(mdp.rewards)) or np.max(np.abs(mdp.rewards)) > 1.0 + tolerance:
        raise ValueError("rewards must be finite and lie in [-1, 1].")

    expected_coordinates = np.array(
        [(row, column) for row in range(N_ROWS) for column in range(N_COLS)],
        dtype=np.int64,
    )
    expected_start = _state_index(N_ROWS - 1, 0)
    expected_cliffs = tuple(_state_index(N_ROWS - 1, column) for column in range(1, N_COLS - 1))
    expected_goal = _state_index(N_ROWS - 1, N_COLS - 1)
    if mdp.n_rows != N_ROWS or mdp.n_cols != N_COLS:
        raise ValueError("MiniCliff geometry must be exactly 4x6.")
    if mdp.state_coordinates.shape != (N_STATES, 2) or not np.array_equal(
        mdp.state_coordinates, expected_coordinates
    ):
        raise ValueError("state_coordinates must contain the complete row-major 4x6 grid.")
    if mdp.start_state != expected_start:
        raise ValueError("start_state must be the bottom-left grid state.")
    if mdp.cliff_states != expected_cliffs or mdp.goal_state != expected_goal:
        raise ValueError("cliff and goal marker states do not match the MiniCliff layout.")
    if mdp.action_names != ACTION_NAMES:
        raise ValueError(f"action_names must be {ACTION_NAMES}.")
    if not math.isfinite(mdp.gamma) or not (0.0 < mdp.gamma < 1.0):
        raise ValueError("mdp.gamma must lie strictly between zero and one.")
    _validate_slip_probability(mdp.slip_probability)
    if not math.isfinite(mdp.behavior_goal_bias) or not (0.0 <= mdp.behavior_goal_bias < 1.0):
        raise ValueError("mdp.behavior_goal_bias must lie in [0, 1).")

    ordinary_mask = np.ones(N_STATES, dtype=bool)
    ordinary_mask[list(mdp.marker_states)] = False
    if not np.allclose(mdp.rewards[ordinary_mask], STEP_REWARD, rtol=0.0, atol=tolerance):
        raise ValueError("ordinary-state rewards must equal STEP_REWARD.")
    if not np.allclose(mdp.rewards[list(mdp.cliff_states)], CLIFF_REWARD, rtol=0.0, atol=tolerance):
        raise ValueError("cliff-marker rewards must equal CLIFF_REWARD.")
    if not np.allclose(mdp.rewards[mdp.goal_state], GOAL_REWARD, rtol=0.0, atol=tolerance):
        raise ValueError("goal-marker rewards must equal GOAL_REWARD.")

    expected_reset = np.zeros(N_STATES, dtype=np.float64)
    expected_reset[mdp.start_state] = 1.0
    if not np.allclose(
        mdp.transitions[list(mdp.marker_states)], expected_reset[None, None, :], rtol=0.0, atol=tolerance
    ):
        raise ValueError("every marker-state action must reset deterministically to start.")


def goal_directed_actions(mdp: TabularMDP) -> Array:
    """Return a deterministic safe-route action for every state.

    From start the route goes up, then right above the cliff row, and finally
    down in the last column.  Actions at marker states are immaterial because
    their transition rows reset; ``up`` is used as a stable convention.
    """

    validate_minicliff_mdp(mdp)
    actions = np.full(mdp.n_states, UP, dtype=np.int64)
    markers = set(mdp.marker_states)
    for state in range(mdp.n_states):
        if state in markers:
            actions[state] = UP
            continue
        row, column = mdp.coordinate_of(state)
        if state == mdp.start_state:
            actions[state] = UP
        elif column < mdp.n_cols - 1:
            actions[state] = RIGHT
        else:
            actions[state] = DOWN
    return actions


def validate_behavior_policy(
    mdp: TabularMDP,
    behavior_policy: Array,
    *,
    require_full_support: bool = True,
    tolerance: float = 1e-12,
) -> Array:
    """Validate and return a float64 behavior-policy matrix."""

    policy = np.asarray(behavior_policy, dtype=np.float64)
    if policy.shape != (mdp.n_states, mdp.n_actions):
        raise ValueError(
            f"behavior_policy must have shape {(mdp.n_states, mdp.n_actions)}, got {policy.shape}."
        )
    if not np.all(np.isfinite(policy)) or np.any(policy < -tolerance):
        raise ValueError("behavior_policy must be finite and nonnegative.")
    if not np.allclose(policy.sum(axis=1), 1.0, rtol=0.0, atol=tolerance):
        raise ValueError("every behavior-policy row must sum to one.")
    if require_full_support and np.any(policy <= 0.0):
        raise ValueError("behavior_policy must assign positive probability to every action.")
    return policy


def make_minicliff_behavior_policy(
    mdp: TabularMDP,
    goal_bias: Optional[float] = None,
) -> Array:
    """Return ``(1-bias)*Uniform + bias*GoalDirected`` at every state."""

    bias = mdp.behavior_goal_bias if goal_bias is None else float(goal_bias)
    if not math.isfinite(bias) or not (0.0 <= bias < 1.0):
        raise ValueError("goal_bias must be finite and lie in [0, 1).")
    policy = np.full(
        (mdp.n_states, mdp.n_actions),
        (1.0 - bias) / mdp.n_actions,
        dtype=np.float64,
    )
    preferred_actions = goal_directed_actions(mdp)
    policy[np.arange(mdp.n_states), preferred_actions] += bias
    return validate_behavior_policy(mdp, policy, require_full_support=True)


def exact_stationary_distributions(
    mdp: TabularMDP,
    behavior_policy: Optional[Array] = None,
) -> StationaryDistributions:
    """Solve exactly (up to floating point) for behavior stationary mass."""

    policy = (
        make_minicliff_behavior_policy(mdp)
        if behavior_policy is None
        else validate_behavior_policy(mdp, behavior_policy, require_full_support=True)
    )
    state_kernel = np.einsum("sa,sat->st", policy, mdp.transitions, optimize=True)

    # Add the normalization equation to the singular stationary system and
    # use least squares instead of relying on power-iteration aperiodicity.
    stationary_system = np.vstack(
        [state_kernel.T - np.eye(mdp.n_states, dtype=np.float64), np.ones(mdp.n_states)]
    )
    stationary_rhs = np.concatenate([np.zeros(mdp.n_states), np.ones(1)])
    state_probabilities, *_ = np.linalg.lstsq(stationary_system, stationary_rhs, rcond=None)
    if np.min(state_probabilities) < -1e-11 or not np.all(np.isfinite(state_probabilities)):
        raise RuntimeError("Stationary linear system produced an invalid distribution.")
    state_probabilities = np.maximum(state_probabilities, 0.0)
    state_probabilities /= state_probabilities.sum()
    state_action_probabilities = state_probabilities[:, None] * policy

    next_state_probabilities = state_probabilities @ state_kernel
    next_state_action_probabilities = next_state_probabilities[:, None] * policy
    state_residual = float(np.max(np.abs(next_state_probabilities - state_probabilities)))
    state_action_residual = float(
        np.max(np.abs(next_state_action_probabilities - state_action_probabilities))
    )
    return StationaryDistributions(
        state_probabilities=state_probabilities,
        state_action_probabilities=state_action_probabilities,
        state_residual=state_residual,
        state_action_residual=state_action_residual,
    )


class PersistentTabularTrajectory:
    """A behavior trajectory whose cursor persists across caller-defined stages."""

    def __init__(
        self,
        mdp: TabularMDP,
        behavior_policy: Array,
        rng: np.random.Generator,
        initial_state: Optional[int] = None,
    ) -> None:
        self.mdp = mdp
        self.behavior_policy = validate_behavior_policy(
            mdp, behavior_policy, require_full_support=True
        ).copy()
        self.rng = rng
        self.state = mdp.start_state if initial_state is None else int(initial_state)
        if not (0 <= self.state < mdp.n_states):
            raise ValueError(f"initial_state must lie in [0, {mdp.n_states}).")
        self.transitions_read = 0
        self.state_action_counts = np.zeros((mdp.n_states, mdp.n_actions), dtype=np.int64)

    def next(self) -> Tuple[int, int, float, int]:
        state = self.state
        action = int(self.rng.choice(self.mdp.n_actions, p=self.behavior_policy[state]))
        next_state = int(self.rng.choice(self.mdp.n_states, p=self.mdp.transitions[state, action]))
        reward = float(self.mdp.rewards[state, action])
        self.state_action_counts[state, action] += 1
        self.transitions_read += 1
        self.state = next_state
        return state, action, reward, next_state

    step = next


def _validate_transition_kernel(kernel: Array, name: str, tolerance: float) -> Array:
    kernel = np.asarray(kernel, dtype=np.float64)
    if kernel.ndim != 3 or kernel.shape[0] != kernel.shape[2]:
        raise ValueError(f"{name} must have shape (S, A, S), got {kernel.shape}.")
    if not np.all(np.isfinite(kernel)) or np.any(kernel < -tolerance):
        raise ValueError(f"{name} must be finite and nonnegative.")
    if not np.allclose(kernel.sum(axis=-1), 1.0, rtol=0.0, atol=tolerance):
        raise ValueError(f"every row of {name} must sum to one.")
    return kernel


def rowwise_chi_square_divergence(
    nominal_kernel: Array,
    candidate_kernel: Array,
    *,
    support_tolerance: float = 1e-15,
    stochastic_tolerance: float = 1e-12,
) -> KernelChiSquareReport:
    """Compare kernels using ``D_chi(candidate row || nominal row)``.

    A row receives infinite divergence when the candidate puts more than
    ``support_tolerance`` mass outside nominal support.
    """

    if not math.isfinite(support_tolerance) or support_tolerance < 0.0:
        raise ValueError("support_tolerance must be finite and nonnegative.")
    nominal = _validate_transition_kernel(nominal_kernel, "nominal_kernel", stochastic_tolerance)
    candidate = _validate_transition_kernel(candidate_kernel, "candidate_kernel", stochastic_tolerance)
    if candidate.shape != nominal.shape:
        raise ValueError(
            f"candidate_kernel shape {candidate.shape} does not match nominal shape {nominal.shape}."
        )

    support = nominal > support_tolerance
    support_violations = np.any((~support) & (candidate > support_tolerance), axis=-1)
    safe_nominal = np.where(support, nominal, 1.0)
    differences = np.where(support, candidate - nominal, 0.0)
    row_divergences = np.sum(differences * differences / safe_nominal, axis=-1)
    row_divergences = np.asarray(row_divergences, dtype=np.float64)
    row_divergences[support_violations] = np.inf
    return KernelChiSquareReport(
        row_divergences=row_divergences,
        support_violations=support_violations,
    )


# Concise alias for callers that use the chi2 spelling elsewhere in the repo.
rowwise_chi2_divergence = rowwise_chi_square_divergence


__all__ = [
    "ACTION_DELTAS",
    "ACTION_NAMES",
    "CLIFF_REWARD",
    "DOWN",
    "GOAL_REWARD",
    "KernelChiSquareReport",
    "LEFT",
    "MiniCliffConfig",
    "N_COLS",
    "N_ROWS",
    "N_STATES",
    "PersistentTabularTrajectory",
    "RIGHT",
    "STEP_REWARD",
    "StationaryDistributions",
    "TabularMDP",
    "UP",
    "exact_stationary_distributions",
    "goal_directed_actions",
    "make_minicliff_behavior_policy",
    "make_minicliff_mdp",
    "rowwise_chi2_divergence",
    "rowwise_chi_square_divergence",
    "validate_behavior_policy",
    "validate_minicliff_config",
    "validate_minicliff_mdp",
]
