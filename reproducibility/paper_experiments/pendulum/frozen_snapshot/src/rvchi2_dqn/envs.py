"""Small online control environments for RVChi2-DQN.

The classes in this module intentionally depend only on NumPy.  They expose a
Gymnasium-like ``reset``/``step`` interface, but also expose deterministic
``transition`` and ``enumerate_modes`` methods for exact two-mode diagnostics.

Terminal values deserve special care.  Ordinary Gymnasium termination has a
zero continuation value.  The established continuing-LQR benchmark instead
uses an explicit failure-state continuation value of ``-10``.  That value is
returned separately as ``terminal_value``; it is not folded into the immediate
reward.  Consumers should therefore use :func:`terminal_continuation` before
forming either nominal or robust continuation targets.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from numbers import Integral
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.floating[Any]]
STANDARD_TERMINAL_VALUE = 0.0
LQR_FAILURE_TERMINAL_VALUE = -10.0


def _validate_probability(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and lie in [0, 1].")
    return value


def _validate_seed(seed: int | None) -> int | None:
    if seed is None:
        return None
    if not isinstance(seed, Integral) or isinstance(seed, (bool, np.bool_)):
        raise TypeError("seed must be an integer or None.")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be nonnegative.")
    return seed


def _action_index(action: int, n_actions: int) -> int:
    if not isinstance(action, Integral) or isinstance(action, (bool, np.bool_)):
        raise TypeError("action must be an integer action index.")
    action = int(action)
    if not 0 <= action < n_actions:
        raise ValueError(f"action must lie in [0, {n_actions - 1}].")
    return action


def _fault_from_uniform(
    rng: np.random.Generator,
    probability: float,
    fault_uniform: float | None,
) -> tuple[bool, float]:
    if fault_uniform is None:
        uniform = float(rng.random())
    else:
        uniform = float(fault_uniform)
        if not math.isfinite(uniform) or not 0.0 <= uniform < 1.0:
            raise ValueError("fault_uniform must be finite and lie in [0, 1).")
    return uniform < probability, uniform


def terminal_continuation(
    next_values: FloatArray | Sequence[float] | float,
    terminated: NDArray[np.bool_] | Sequence[bool] | bool,
    terminal_values: FloatArray | Sequence[float] | float = STANDARD_TERMINAL_VALUE,
) -> NDArray[np.float64]:
    """Resolve continuation values without conflating termination and truncation.

    A standard terminal transition supplies ``terminal_values=0``.  An LQR box
    exit supplies ``terminal_values=-10``.  Time-limit truncation is deliberately
    absent from this function: a truncation continues to bootstrap from the
    supplied next value.
    """

    values = np.asarray(next_values, dtype=np.float64)
    terminal = np.asarray(terminated, dtype=bool)
    terminal_values_array = np.asarray(terminal_values, dtype=np.float64)
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(terminal_values_array)):
        raise ValueError("Continuation values must be finite.")
    return np.asarray(
        np.where(terminal, terminal_values_array, values), dtype=np.float64
    )


@dataclass(frozen=True)
class ModeOutcome:
    """One deterministic actuator-mode outcome before any reset."""

    state: NDArray[np.float64]
    observation: NDArray[np.float32]
    scaled_reward: float
    raw_reward: float
    terminated: bool
    truncated: bool
    terminal_value: float
    fault: bool
    commanded_action: float
    applied_action: float


@dataclass(frozen=True)
class LQRConfig:
    """Frozen continuing double-integrator task with discrete commands."""

    actions: tuple[float, ...] = (-1.0, 0.0, 1.0)
    fault_probability: float = 0.10
    reward_scale: float = 1.0
    horizon: int = 200
    time_step: float = 0.10
    position_bound: float = 2.0
    velocity_bound: float = 2.0
    position_weight: float = 1.0
    velocity_weight: float = 0.10
    action_weight: float = 0.01
    initial_position_min: float = 0.8
    initial_position_max: float = 1.2
    initial_velocity_bound: float = 0.10
    failure_terminal_value: float = LQR_FAILURE_TERMINAL_VALUE

    def __post_init__(self) -> None:
        if self.actions != (-1.0, 0.0, 1.0):
            raise ValueError("The continuing-LQR actions are frozen to (-1, 0, 1).")
        _validate_probability(self.fault_probability, "fault_probability")
        positive = (
            self.reward_scale,
            self.time_step,
            self.position_bound,
            self.velocity_bound,
            self.position_weight,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("LQR scales, bounds, dt, and position weight must be positive.")
        nonnegative = (self.velocity_weight, self.action_weight)
        if any(not math.isfinite(value) or value < 0.0 for value in nonnegative):
            raise ValueError("LQR velocity and action weights must be nonnegative.")
        if not isinstance(self.horizon, Integral) or isinstance(self.horizon, bool) or self.horizon < 1:
            raise ValueError("horizon must be a positive integer.")
        if not (
            0.0
            <= self.initial_position_min
            < self.initial_position_max
            < self.position_bound
        ):
            raise ValueError("Initial position magnitudes must be ordered inside the box.")
        if not (
            math.isfinite(self.initial_velocity_bound)
            and 0.0 <= self.initial_velocity_bound < self.velocity_bound
        ):
            raise ValueError("initial_velocity_bound must lie inside the velocity box.")
        if (
            not math.isfinite(self.failure_terminal_value)
            or self.failure_terminal_value >= 0.0
        ):
            raise ValueError("failure_terminal_value must be finite and negative.")


class ReversalLQREnv:
    """Online nominal-kernel version of the established continuing LQR task."""

    observation_shape = (2,)
    n_actions = 3

    def __init__(self, config: LQRConfig | None = None, *, seed: int | None = None):
        self.config = config if config is not None else LQRConfig()
        self._rng = np.random.default_rng(_validate_seed(seed))
        self._state: NDArray[np.float64] | None = None
        self._elapsed_steps = 0
        self._needs_reset = True

    @property
    def state(self) -> NDArray[np.float64]:
        if self._state is None:
            raise RuntimeError("Call reset before accessing state.")
        return self._state.copy()

    @property
    def elapsed_steps(self) -> int:
        return self._elapsed_steps

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(_validate_seed(seed))
        if options is not None and set(options) - {"state"}:
            raise ValueError("Only the reset option 'state' is supported.")
        supplied_state = None if options is None else options.get("state")
        if supplied_state is None:
            # Match the established oracle/trainer reset draw order exactly.
            sign = float(self._rng.choice(np.asarray((-1.0, 1.0))))
            position = sign * float(
                self._rng.uniform(
                    self.config.initial_position_min,
                    self.config.initial_position_max,
                )
            )
            velocity = float(
                self._rng.uniform(
                    -self.config.initial_velocity_bound,
                    self.config.initial_velocity_bound,
                )
            )
            state = np.asarray((position, velocity), dtype=np.float64)
        else:
            state = np.asarray(supplied_state, dtype=np.float64)
            if state.shape != (2,) or not np.all(np.isfinite(state)):
                raise ValueError("LQR reset state must be a finite length-two vector.")
            if (
                abs(float(state[0])) > self.config.position_bound
                or abs(float(state[1])) > self.config.velocity_bound
            ):
                raise ValueError("LQR reset state must lie inside the closed state box.")
            state = state.copy()
        self._state = state
        self._elapsed_steps = 0
        self._needs_reset = False
        observation = state.astype(np.float32, copy=True)
        return observation, {"state": state.copy()}

    def transition(
        self,
        state: FloatArray | Sequence[float],
        action: int,
        *,
        fault: bool,
    ) -> ModeOutcome:
        state_array = np.asarray(state, dtype=np.float64)
        if state_array.shape != (2,) or not np.all(np.isfinite(state_array)):
            raise ValueError("LQR state must be a finite length-two vector.")
        action_index = _action_index(action, self.n_actions)
        command = float(self.config.actions[action_index])
        is_fault = bool(fault)
        applied = -command if is_fault else command
        position, velocity = (float(state_array[0]), float(state_array[1]))
        raw_reward = -self.config.time_step * (
            self.config.position_weight * position**2
            + self.config.velocity_weight * velocity**2
            + self.config.action_weight * command**2
        )
        next_position = (
            position
            + self.config.time_step * velocity
            + 0.5 * self.config.time_step**2 * applied
        )
        next_velocity = velocity + self.config.time_step * applied
        terminated = bool(
            abs(next_position) > self.config.position_bound
            or abs(next_velocity) > self.config.velocity_bound
        )
        next_state = np.asarray((next_position, next_velocity), dtype=np.float64)
        return ModeOutcome(
            state=next_state,
            observation=next_state.astype(np.float32),
            scaled_reward=float(self.config.reward_scale * raw_reward),
            raw_reward=float(raw_reward),
            terminated=terminated,
            truncated=False,
            terminal_value=(
                float(self.config.failure_terminal_value)
                if terminated
                else STANDARD_TERMINAL_VALUE
            ),
            fault=is_fault,
            commanded_action=command,
            applied_action=applied,
        )

    def enumerate_modes(
        self, state: FloatArray | Sequence[float], action: int
    ) -> tuple[ModeOutcome, ModeOutcome]:
        """Return ``(healthy, reversed)`` outcomes without consuming RNG state."""

        return (
            self.transition(state, action, fault=False),
            self.transition(state, action, fault=True),
        )

    def step(
        self, action: int, *, fault_uniform: float | None = None
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        if self._needs_reset or self._state is None:
            raise RuntimeError("Call reset before step, and reset after an episode ends.")
        fault, uniform = _fault_from_uniform(
            self._rng, self.config.fault_probability, fault_uniform
        )
        outcome = self.transition(self._state, action, fault=fault)
        self._elapsed_steps += 1
        truncated = bool(
            not outcome.terminated and self._elapsed_steps >= self.config.horizon
        )
        outcome = replace(outcome, truncated=truncated)
        self._state = outcome.state.copy()
        self._needs_reset = outcome.terminated or outcome.truncated
        info = {
            "fault": outcome.fault,
            "fault_uniform": uniform,
            "commanded_action": outcome.commanded_action,
            "applied_action": outcome.applied_action,
            "raw_reward": outcome.raw_reward,
            "scaled_reward": outcome.scaled_reward,
            "terminal_value": outcome.terminal_value,
            "time_limit_bootstrap": outcome.truncated,
            "state": outcome.state.copy(),
        }
        return (
            outcome.observation.copy(),
            outcome.scaled_reward,
            outcome.terminated,
            outcome.truncated,
            info,
        )


@dataclass(frozen=True)
class PendulumConfig:
    """Discretized Pendulum-v1 with a supported torque-reversal mode."""

    actions: tuple[float, ...] = (-2.0, 0.0, 2.0)
    fault_probability: float = 0.10
    reward_scale: float = 0.01
    horizon: int = 200
    gravity: float = 10.0
    mass: float = 1.0
    length: float = 1.0
    time_step: float = 0.05
    max_torque: float = 2.0
    max_speed: float = 8.0

    def __post_init__(self) -> None:
        if self.actions != (-2.0, 0.0, 2.0):
            raise ValueError("Pendulum actions are frozen to (-2, 0, 2).")
        _validate_probability(self.fault_probability, "fault_probability")
        positive = (
            self.reward_scale,
            self.gravity,
            self.mass,
            self.length,
            self.time_step,
            self.max_torque,
            self.max_speed,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("Pendulum scales and physical constants must be positive.")
        if not isinstance(self.horizon, Integral) or isinstance(self.horizon, bool) or self.horizon < 1:
            raise ValueError("horizon must be a positive integer.")


def wrap_angle(angle: FloatArray | float) -> NDArray[np.float64]:
    return np.asarray((np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi, dtype=np.float64)


class ReversalPendulumEnv:
    """NumPy Pendulum-v1 dynamics with three commanded torque actions."""

    observation_shape = (3,)
    n_actions = 3

    def __init__(
        self, config: PendulumConfig | None = None, *, seed: int | None = None
    ):
        self.config = config if config is not None else PendulumConfig()
        self._rng = np.random.default_rng(_validate_seed(seed))
        self._state: NDArray[np.float64] | None = None
        self._elapsed_steps = 0
        self._needs_reset = True

    @staticmethod
    def observation_from_state(
        state: FloatArray | Sequence[float],
    ) -> NDArray[np.float32]:
        state_array = np.asarray(state, dtype=np.float64)
        if state_array.shape != (2,) or not np.all(np.isfinite(state_array)):
            raise ValueError("Pendulum state must be a finite (angle, velocity) vector.")
        return np.asarray(
            (math.cos(float(state_array[0])), math.sin(float(state_array[0])), state_array[1]),
            dtype=np.float32,
        )

    @property
    def state(self) -> NDArray[np.float64]:
        if self._state is None:
            raise RuntimeError("Call reset before accessing state.")
        return self._state.copy()

    @property
    def elapsed_steps(self) -> int:
        return self._elapsed_steps

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(_validate_seed(seed))
        if options is not None and set(options) - {"state"}:
            raise ValueError("Only the reset option 'state' is supported.")
        supplied_state = None if options is None else options.get("state")
        if supplied_state is None:
            state = self._rng.uniform(
                low=np.asarray((-np.pi, -1.0)),
                high=np.asarray((np.pi, 1.0)),
            ).astype(np.float64)
        else:
            state = np.asarray(supplied_state, dtype=np.float64)
            if state.shape != (2,) or not np.all(np.isfinite(state)):
                raise ValueError("Pendulum reset state must be a finite length-two vector.")
            if abs(float(state[1])) > self.config.max_speed:
                raise ValueError("Pendulum reset velocity must respect max_speed.")
            state = state.copy()
        state[0] = float(wrap_angle(state[0]))
        self._state = state
        self._elapsed_steps = 0
        self._needs_reset = False
        return self.observation_from_state(state), {"state": state.copy()}

    def transition(
        self,
        state: FloatArray | Sequence[float],
        action: int,
        *,
        fault: bool,
    ) -> ModeOutcome:
        state_array = np.asarray(state, dtype=np.float64)
        if state_array.shape != (2,) or not np.all(np.isfinite(state_array)):
            raise ValueError("Pendulum state must be a finite length-two vector.")
        action_index = _action_index(action, self.n_actions)
        command = float(self.config.actions[action_index])
        is_fault = bool(fault)
        applied = float(np.clip(-command if is_fault else command, -self.config.max_torque, self.config.max_torque))
        angle = float(state_array[0])
        velocity = float(state_array[1])
        # This is Gymnasium Pendulum-v1's reward.  Under exact sign reversal,
        # applied**2 == command**2, so reward is identical in both modes.
        raw_reward = -(
            float(wrap_angle(angle)) ** 2
            + 0.1 * velocity**2
            + 0.001 * applied**2
        )
        next_velocity = velocity + (
            3.0 * self.config.gravity / (2.0 * self.config.length) * math.sin(angle)
            + 3.0 / (self.config.mass * self.config.length**2) * applied
        ) * self.config.time_step
        next_velocity = float(
            np.clip(next_velocity, -self.config.max_speed, self.config.max_speed)
        )
        next_angle = float(wrap_angle(angle + next_velocity * self.config.time_step))
        next_state = np.asarray((next_angle, next_velocity), dtype=np.float64)
        return ModeOutcome(
            state=next_state,
            observation=self.observation_from_state(next_state),
            scaled_reward=float(self.config.reward_scale * raw_reward),
            raw_reward=float(raw_reward),
            terminated=False,
            truncated=False,
            terminal_value=STANDARD_TERMINAL_VALUE,
            fault=is_fault,
            commanded_action=command,
            applied_action=applied,
        )

    def enumerate_modes(
        self, state: FloatArray | Sequence[float], action: int
    ) -> tuple[ModeOutcome, ModeOutcome]:
        """Return ``(healthy, reversed)`` outcomes without consuming RNG state."""

        return (
            self.transition(state, action, fault=False),
            self.transition(state, action, fault=True),
        )

    def step(
        self, action: int, *, fault_uniform: float | None = None
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        if self._needs_reset or self._state is None:
            raise RuntimeError("Call reset before step, and reset after an episode ends.")
        fault, uniform = _fault_from_uniform(
            self._rng, self.config.fault_probability, fault_uniform
        )
        outcome = self.transition(self._state, action, fault=fault)
        self._elapsed_steps += 1
        truncated = bool(self._elapsed_steps >= self.config.horizon)
        outcome = replace(outcome, truncated=truncated)
        self._state = outcome.state.copy()
        self._needs_reset = outcome.truncated
        info = {
            "fault": outcome.fault,
            "fault_uniform": uniform,
            "commanded_action": outcome.commanded_action,
            "applied_action": outcome.applied_action,
            "raw_reward": outcome.raw_reward,
            "scaled_reward": outcome.scaled_reward,
            "terminal_value": outcome.terminal_value,
            "time_limit_bootstrap": outcome.truncated,
            "state": outcome.state.copy(),
        }
        return (
            outcome.observation.copy(),
            outcome.scaled_reward,
            outcome.terminated,
            outcome.truncated,
            info,
        )


__all__ = [
    "LQRConfig",
    "LQR_FAILURE_TERMINAL_VALUE",
    "ModeOutcome",
    "PendulumConfig",
    "ReversalLQREnv",
    "ReversalPendulumEnv",
    "STANDARD_TERMINAL_VALUE",
    "terminal_continuation",
    "wrap_angle",
]
