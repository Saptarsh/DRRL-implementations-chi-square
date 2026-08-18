"""A compact, cloneable NumPy ring replay for online RVChi2-DQN."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from numbers import Integral
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ReplayBatch:
    """One detached replay batch.

    ``terminated`` and ``truncated`` remain separate.  ``terminal_values`` is
    zero for ordinary terminal transitions and may be nonzero for a task such
    as the continuing LQR benchmark with its explicit failure continuation.
    """

    observations: NDArray[np.float32]
    actions: NDArray[np.int64]
    scaled_rewards: NDArray[np.float32]
    raw_rewards: NDArray[np.float32]
    next_observations: NDArray[np.float32]
    terminated: NDArray[np.bool_]
    truncated: NDArray[np.bool_]
    terminal_values: NDArray[np.float32]
    faults: NDArray[np.bool_]
    indices: NDArray[np.int64]

    def __len__(self) -> int:
        return int(self.actions.shape[0])


class ReplayBuffer:
    """Fixed-capacity uniform replay with exact deep cloning.

    Cloning copies both stored transitions and sampler RNG state.  Thus two
    method branches can begin from an identical replay prefix and, until their
    buffers diverge, draw identical minibatches.  Every returned batch owns its
    arrays, preventing accidental mutation of replay storage.
    """

    def __init__(
        self,
        capacity: int,
        observation_shape: int | Sequence[int],
        *,
        seed: int | None = None,
    ) -> None:
        if not isinstance(capacity, Integral) or isinstance(capacity, bool) or capacity < 1:
            raise ValueError("capacity must be a positive integer.")
        if isinstance(observation_shape, Integral) and not isinstance(observation_shape, bool):
            shape = (int(observation_shape),)
        else:
            shape = tuple(int(value) for value in observation_shape)
        if not shape or any(value < 1 for value in shape):
            raise ValueError("observation_shape must contain positive dimensions.")
        if seed is not None:
            if not isinstance(seed, Integral) or isinstance(seed, bool) or seed < 0:
                raise ValueError("seed must be a nonnegative integer or None.")
            seed = int(seed)

        self.capacity = int(capacity)
        self.observation_shape = shape
        self._rng = np.random.default_rng(seed)
        self._observations = np.empty((self.capacity, *shape), dtype=np.float32)
        self._actions = np.empty(self.capacity, dtype=np.int64)
        self._scaled_rewards = np.empty(self.capacity, dtype=np.float32)
        self._raw_rewards = np.empty(self.capacity, dtype=np.float32)
        self._next_observations = np.empty((self.capacity, *shape), dtype=np.float32)
        self._terminated = np.empty(self.capacity, dtype=bool)
        self._truncated = np.empty(self.capacity, dtype=bool)
        self._terminal_values = np.empty(self.capacity, dtype=np.float32)
        self._faults = np.empty(self.capacity, dtype=bool)
        self._size = 0
        self._next_index = 0
        self._total_added = 0

    def __len__(self) -> int:
        return self._size

    @property
    def full(self) -> bool:
        return self._size == self.capacity

    @property
    def next_index(self) -> int:
        return self._next_index

    @property
    def total_added(self) -> int:
        return self._total_added

    def _observation(self, value: Any, name: str) -> NDArray[np.float32]:
        array = np.asarray(value, dtype=np.float32)
        if array.shape != self.observation_shape:
            raise ValueError(
                f"{name} must have shape {self.observation_shape}, got {array.shape}."
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite.")
        return array

    def add(
        self,
        observation: Any,
        action: int,
        scaled_reward: float,
        raw_reward: float,
        next_observation: Any,
        terminated: bool,
        truncated: bool,
        terminal_value: float,
        fault: bool,
    ) -> None:
        """Append one transition, overwriting the oldest when full."""

        obs = self._observation(observation, "observation")
        next_obs = self._observation(next_observation, "next_observation")
        if not isinstance(action, Integral) or isinstance(action, (bool, np.bool_)):
            raise TypeError("action must be an integer action index.")
        action = int(action)
        if action < 0:
            raise ValueError("action must be nonnegative.")
        scalar_values = (float(scaled_reward), float(raw_reward), float(terminal_value))
        if any(not math.isfinite(value) for value in scalar_values):
            raise ValueError("Rewards and terminal_value must be finite.")

        index = self._next_index
        self._observations[index] = obs
        self._actions[index] = action
        self._scaled_rewards[index] = scalar_values[0]
        self._raw_rewards[index] = scalar_values[1]
        self._next_observations[index] = next_obs
        self._terminated[index] = bool(terminated)
        self._truncated[index] = bool(truncated)
        self._terminal_values[index] = scalar_values[2]
        self._faults[index] = bool(fault)

        self._next_index = (index + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
        self._total_added += 1

    def _chronological_indices(self) -> NDArray[np.int64]:
        if self._size == 0:
            return np.empty(0, dtype=np.int64)
        if self._size < self.capacity:
            return np.arange(self._size, dtype=np.int64)
        return np.concatenate(
            (
                np.arange(self._next_index, self.capacity, dtype=np.int64),
                np.arange(0, self._next_index, dtype=np.int64),
            )
        )

    def _batch(self, indices: NDArray[np.int64]) -> ReplayBatch:
        return ReplayBatch(
            observations=self._observations[indices].copy(),
            actions=self._actions[indices].copy(),
            scaled_rewards=self._scaled_rewards[indices].copy(),
            raw_rewards=self._raw_rewards[indices].copy(),
            next_observations=self._next_observations[indices].copy(),
            terminated=self._terminated[indices].copy(),
            truncated=self._truncated[indices].copy(),
            terminal_values=self._terminal_values[indices].copy(),
            faults=self._faults[indices].copy(),
            indices=indices.astype(np.int64, copy=True),
        )

    def as_batch(self) -> ReplayBatch:
        """Return all stored transitions from oldest to newest."""

        return self._batch(self._chronological_indices())

    def sample(
        self,
        batch_size: int,
        *,
        replace: bool = True,
        rng: np.random.Generator | None = None,
    ) -> ReplayBatch:
        """Uniformly sample stored transitions.

        Sampling is over physical ring slots, which is equivalent to sampling
        the logical transitions uniformly.  Returned ``indices`` identify those
        slots and are useful for reproducibility diagnostics.
        """

        if self._size == 0:
            raise ValueError("Cannot sample an empty replay buffer.")
        if not isinstance(batch_size, Integral) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        batch_size = int(batch_size)
        if not replace and batch_size > self._size:
            raise ValueError("Cannot sample more than replay size without replacement.")
        generator = self._rng if rng is None else rng
        if not isinstance(generator, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator or None.")
        available = self._chronological_indices()
        logical_indices = generator.choice(
            self._size, size=batch_size, replace=replace
        )
        storage_indices = available[np.asarray(logical_indices, dtype=np.int64)]
        return self._batch(storage_indices)

    def clone(self, *, seed: int | None = None) -> "ReplayBuffer":
        """Return a deep replay copy, optionally with a fresh sampler seed."""

        clone = ReplayBuffer(
            self.capacity,
            self.observation_shape,
            seed=0 if seed is None else seed,
        )
        clone._observations[:] = self._observations
        clone._actions[:] = self._actions
        clone._scaled_rewards[:] = self._scaled_rewards
        clone._raw_rewards[:] = self._raw_rewards
        clone._next_observations[:] = self._next_observations
        clone._terminated[:] = self._terminated
        clone._truncated[:] = self._truncated
        clone._terminal_values[:] = self._terminal_values
        clone._faults[:] = self._faults
        clone._size = self._size
        clone._next_index = self._next_index
        clone._total_added = self._total_added
        if seed is None:
            clone._rng.bit_generator.state = copy.deepcopy(self._rng.bit_generator.state)
        return clone


__all__ = ["ReplayBatch", "ReplayBuffer"]
