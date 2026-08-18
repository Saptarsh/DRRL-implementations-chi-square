"""NumPy-only diagnostics for RVChi2-DQN experiments.

The routines here intentionally return ordinary Python scalars and dictionaries
for summaries so they can be written by :mod:`rvchi2_dqn.artifacts` with
``allow_nan=False``.  In particular, correlations and normalized errors that
are mathematically undefined are reported as ``None`` rather than ``NaN``.

The calibration convention throughout is ``error = learned - exact``.  A
positive bias therefore means that the learned robust continuation is too
optimistic on average.
"""

from __future__ import annotations

from numbers import Integral
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


# Bounds are expressed in the observations consumed by the Q network.  LQR
# observes (position, velocity), while Pendulum observes (cos(theta),
# sin(theta), angular_velocity).  Tuples keep this public constant immutable.
TASK_STATE_BOUNDS: Mapping[str, tuple[tuple[float, ...], tuple[float, ...]]] = {
    "lqr": ((-2.0, -2.0), (2.0, 2.0)),
    "pendulum": ((-1.0, -1.0, -8.0), (1.0, 1.0, 8.0)),
}
TASK_ACTION_COUNTS: Mapping[str, int] = {"lqr": 3, "pendulum": 3}


def _finite_vector(values: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return np.asarray(array.reshape(-1), dtype=np.float64)


def _same_length(values: ArrayLike, size: int, name: str) -> NDArray[Any]:
    array = np.asarray(values).reshape(-1)
    if array.size != size:
        raise ValueError(f"{name} must contain {size} entries, got {array.size}.")
    return array


def _boolean_vector(values: ArrayLike, size: int, name: str) -> NDArray[np.bool_]:
    array = _same_length(values, size, name)
    # An untyped empty Python list is valid for an empty stratum even though
    # NumPy gives it floating dtype by default.
    if array.size == 0:
        return np.empty(0, dtype=bool)
    if array.dtype.kind != "b":
        raise TypeError(f"{name} must be a boolean array.")
    return np.asarray(array, dtype=bool)


def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def stable_average_ranks(values: ArrayLike) -> FloatArray:
    """Return conventional one-based average ranks with deterministic ties.

    NumPy does not provide a rank-data primitive and SciPy would be an
    unnecessary dependency for experiment diagnostics.  A stable mergesort
    makes the result deterministic, while equal-value groups receive the mean
    of their occupied ranks (for example, ``[10, 10, 20] -> [1.5, 1.5, 3]``).
    """

    vector = _finite_vector(values, "values")
    if vector.size == 0:
        return np.empty(0, dtype=np.float64)

    order = np.argsort(vector, kind="stable")
    ordered = vector[order]
    starts = np.concatenate(
        (np.asarray([0], dtype=np.int64), np.flatnonzero(ordered[1:] != ordered[:-1]) + 1)
    )
    ends = np.concatenate((starts[1:], np.asarray([vector.size], dtype=np.int64)))

    ordered_ranks = np.empty(vector.size, dtype=np.float64)
    for start, end in zip(starts, ends):
        # Rank positions are start + 1 through end (because end is exclusive).
        ordered_ranks[start:end] = 0.5 * (float(start) + 1.0 + float(end))

    ranks = np.empty(vector.size, dtype=np.float64)
    ranks[order] = ordered_ranks
    return ranks


def _safe_pearson(left: FloatArray, right: FloatArray) -> float | None:
    """Compute Pearson correlation, returning ``None`` when undefined."""

    if left.size != right.size:
        raise ValueError("Correlation inputs must have equal length.")
    if left.size < 2:
        return None

    left_centered = left - np.mean(left)
    right_centered = right - np.mean(right)
    left_scale = float(np.max(np.abs(left_centered)))
    right_scale = float(np.max(np.abs(right_centered)))
    if left_scale == 0.0 or right_scale == 0.0:
        return None

    # Scaling first avoids overflow in the sum of squares for large finite data.
    left_scaled = left_centered / left_scale
    right_scaled = right_centered / right_scale
    denominator = float(
        np.sqrt(np.dot(left_scaled, left_scaled) * np.dot(right_scaled, right_scaled))
    )
    if denominator == 0.0 or not np.isfinite(denominator):
        return None
    correlation = float(np.dot(left_scaled, right_scaled) / denominator)
    # Roundoff can put a theoretically valid coefficient just outside [-1, 1].
    return float(np.clip(correlation, -1.0, 1.0))


def backup_calibration_metrics(
    exact: ArrayLike,
    learned: ArrayLike,
) -> dict[str, int | float | None]:
    """Summarize exact-versus-learned robust continuation calibration.

    ``normalized_mae`` is the MAE divided by ``max(exact) - min(exact)``.
    It is undefined (``None``) for an empty or constant exact stratum.  Pearson
    and Spearman correlations are likewise ``None`` for fewer than two samples
    or whenever either input is constant.
    """

    exact_values = _finite_vector(exact, "exact")
    learned_values = _finite_vector(learned, "learned")
    if exact_values.size != learned_values.size:
        raise ValueError("exact and learned must contain the same number of values.")

    count = int(exact_values.size)
    empty: dict[str, int | float | None] = {
        "count": count,
        "pearson": None,
        "spearman": None,
        "mae": None,
        "normalized_mae": None,
        "bias": None,
        "exact_range": None,
    }
    if count == 0:
        return empty

    errors = learned_values - exact_values
    absolute_errors = np.abs(errors)
    exact_range = float(np.max(exact_values) - np.min(exact_values))
    mae = float(np.mean(absolute_errors))
    empty.update(
        {
            "pearson": _safe_pearson(exact_values, learned_values),
            "spearman": _safe_pearson(
                stable_average_ranks(exact_values),
                stable_average_ranks(learned_values),
            ),
            "mae": mae,
            "normalized_mae": None if exact_range == 0.0 else mae / exact_range,
            "bias": float(np.mean(errors)),
            "exact_range": exact_range,
        }
    )
    return empty


def _masked_metrics(
    exact: FloatArray,
    learned: FloatArray,
    mask: NDArray[np.bool_],
) -> dict[str, int | float | None]:
    return backup_calibration_metrics(exact[mask], learned[mask])


def stratified_backup_calibration(
    exact: ArrayLike,
    learned: ArrayLike,
    *,
    actions: ArrayLike | None = None,
    action_values: Sequence[Any] | None = None,
    visited: ArrayLike | None = None,
    support: ArrayLike | None = None,
    sparse: ArrayLike | None = None,
    decision_boundary: ArrayLike | None = None,
) -> dict[str, Any]:
    """Return overall and requested held-out backup-calibration strata.

    ``visited`` and ``support`` are aliases for the same scientific concept;
    callers may use whichever name matches their probe construction, but may
    not supply both.  Explicit ``action_values`` retain empty action strata,
    which is useful when every task is expected to have a fixed action set.

    Boolean groups use these stable labels:

    * visited: ``visited`` / ``unvisited``;
    * support: ``supported`` / ``unsupported``;
    * sparse: ``sparse`` / ``not_sparse``;
    * decision boundary: ``near_boundary`` / ``away_from_boundary``.
    """

    exact_values = _finite_vector(exact, "exact")
    learned_values = _finite_vector(learned, "learned")
    if exact_values.size != learned_values.size:
        raise ValueError("exact and learned must contain the same number of values.")
    if visited is not None and support is not None:
        raise ValueError("visited and support are aliases; supply only one.")

    size = exact_values.size
    result: dict[str, Any] = {
        "overall": backup_calibration_metrics(exact_values, learned_values),
        "by_action": {},
        "by_visited": {},
        "by_support": {},
        "by_sparse": {},
        "by_decision_boundary": {},
    }

    if actions is not None:
        action_array = _same_length(actions, size, "actions")
        if action_values is None:
            labels = [_python_scalar(value) for value in np.unique(action_array)]
        else:
            labels = [_python_scalar(value) for value in action_values]
            if len(set(labels)) != len(labels):
                raise ValueError("action_values must not contain duplicates.")
        result["by_action"] = {
            label: _masked_metrics(exact_values, learned_values, action_array == label)
            for label in labels
        }
    elif action_values is not None:
        raise ValueError("action_values requires actions.")

    if visited is not None:
        mask = _boolean_vector(visited, size, "visited")
        result["by_visited"] = {
            "visited": _masked_metrics(exact_values, learned_values, mask),
            "unvisited": _masked_metrics(exact_values, learned_values, ~mask),
        }
    if support is not None:
        mask = _boolean_vector(support, size, "support")
        result["by_support"] = {
            "supported": _masked_metrics(exact_values, learned_values, mask),
            "unsupported": _masked_metrics(exact_values, learned_values, ~mask),
        }
    if sparse is not None:
        mask = _boolean_vector(sparse, size, "sparse")
        result["by_sparse"] = {
            "sparse": _masked_metrics(exact_values, learned_values, mask),
            "not_sparse": _masked_metrics(exact_values, learned_values, ~mask),
        }
    if decision_boundary is not None:
        mask = _boolean_vector(decision_boundary, size, "decision_boundary")
        result["by_decision_boundary"] = {
            "near_boundary": _masked_metrics(exact_values, learned_values, mask),
            "away_from_boundary": _masked_metrics(exact_values, learned_values, ~mask),
        }
    return result


def task_state_bounds(task: str) -> tuple[FloatArray, FloatArray]:
    """Return copies of the Q-observation bounds for a supported task."""

    normalized_task = str(task).lower()
    try:
        lower, upper = TASK_STATE_BOUNDS[normalized_task]
    except KeyError as error:
        choices = ", ".join(sorted(TASK_STATE_BOUNDS))
        raise ValueError(f"Unknown task {task!r}; expected one of {choices}.") from error
    return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


def _state_matrix(states: ArrayLike, dimensions: int, name: str) -> FloatArray:
    array = np.asarray(states, dtype=np.float64)
    if array.ndim == 1:
        if array.size == 0:
            array = np.empty((0, dimensions), dtype=np.float64)
        elif array.size == dimensions:
            array = array.reshape(1, dimensions)
        else:
            raise ValueError(f"{name} must have {dimensions} columns.")
    if array.ndim != 2 or array.shape[1] != dimensions:
        raise ValueError(f"{name} must have shape (n, {dimensions}).")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return np.asarray(array, dtype=np.float64)


def normalize_task_states(
    states: ArrayLike,
    task: str,
    *,
    clip: bool = True,
) -> FloatArray:
    """Map task observations affinely to the unit box.

    Clipping is appropriate for a support audit: it keeps tiny simulator
    roundoff and terminal overshoot in the edge bins.  Callers that want strict
    bounds checking can set ``clip=False`` and inspect values outside ``[0, 1]``.
    """

    lower, upper = task_state_bounds(task)
    matrix = _state_matrix(states, lower.size, "states")
    normalized = (matrix - lower) / (upper - lower)
    if clip:
        normalized = np.clip(normalized, 0.0, 1.0)
    return np.asarray(normalized, dtype=np.float64)


def _bins_per_dimension(bins: int | Sequence[int], dimensions: int) -> IntArray:
    if isinstance(bins, Integral) and not isinstance(bins, (bool, np.bool_)):
        values = np.full(dimensions, int(bins), dtype=np.int64)
    else:
        raw = np.asarray(bins)
        if raw.ndim != 1 or raw.size != dimensions:
            raise ValueError(f"bins must be an integer or a length-{dimensions} sequence.")
        if raw.dtype.kind not in "iu" or raw.dtype.kind == "b":
            raise TypeError("bin counts must be integers.")
        values = np.asarray(raw, dtype=np.int64)
    if np.any(values <= 0):
        raise ValueError("bin counts must be positive.")
    return values


def _action_vector(actions: ArrayLike, size: int, n_actions: int, name: str) -> IntArray:
    raw = _same_length(actions, size, name)
    # ``np.asarray([])`` has floating dtype even though the empty sequence is a
    # perfectly valid action vector for an empty replay or probe set.
    if raw.size == 0:
        return np.empty(0, dtype=np.int64)
    if raw.dtype.kind not in "iu" or raw.dtype.kind == "b":
        raise TypeError(f"{name} must contain integer action indices.")
    result = np.asarray(raw, dtype=np.int64)
    if np.any(result < 0) or np.any(result >= n_actions):
        raise ValueError(f"{name} entries must lie in [0, {n_actions - 1}].")
    return result


def _flat_state_bins(normalized_states: FloatArray, bins: IntArray) -> IntArray:
    if normalized_states.shape[0] == 0:
        return np.empty(0, dtype=np.int64)
    coordinates = np.floor(normalized_states * bins).astype(np.int64)
    # A normalized coordinate exactly equal to one belongs to the last bin.
    coordinates = np.minimum(coordinates, bins - 1)
    return np.asarray(
        np.ravel_multi_index(coordinates.T, tuple(int(value) for value in bins)),
        dtype=np.int64,
    )


def replay_support_counts(
    replay_states: ArrayLike,
    replay_actions: ArrayLike,
    query_states: ArrayLike,
    query_actions: ArrayLike,
    *,
    task: str,
    bins: int | Sequence[int] = 11,
    n_actions: int | None = None,
) -> IntArray:
    """Return replay counts in each queried normalized state-action bin."""

    lower, _ = task_state_bounds(task)
    task_key = str(task).lower()
    if n_actions is None:
        n_actions = TASK_ACTION_COUNTS[task_key]
    if not isinstance(n_actions, Integral) or isinstance(n_actions, (bool, np.bool_)):
        raise TypeError("n_actions must be an integer.")
    n_actions = int(n_actions)
    if n_actions <= 0:
        raise ValueError("n_actions must be positive.")

    replay_matrix = _state_matrix(replay_states, lower.size, "replay_states")
    query_matrix = _state_matrix(query_states, lower.size, "query_states")
    replay_action_array = _action_vector(
        replay_actions, replay_matrix.shape[0], n_actions, "replay_actions"
    )
    query_action_array = _action_vector(
        query_actions, query_matrix.shape[0], n_actions, "query_actions"
    )
    bin_shape = _bins_per_dimension(bins, lower.size)
    state_bin_count = int(np.prod(bin_shape, dtype=np.int64))

    replay_bins = _flat_state_bins(normalize_task_states(replay_matrix, task), bin_shape)
    query_bins = _flat_state_bins(normalize_task_states(query_matrix, task), bin_shape)
    replay_state_actions = replay_bins * n_actions + replay_action_array
    counts = np.bincount(
        replay_state_actions,
        minlength=state_bin_count * n_actions,
    )
    return np.asarray(counts[query_bins * n_actions + query_action_array], dtype=np.int64)


def _frequency_dict(
    actions: IntArray,
    n_actions: int,
) -> tuple[dict[int, int], dict[int, float | None]]:
    counts_array = np.bincount(actions, minlength=n_actions)
    total = int(actions.size)
    counts = {action: int(counts_array[action]) for action in range(n_actions)}
    frequencies = {
        action: (None if total == 0 else float(counts_array[action] / total))
        for action in range(n_actions)
    }
    return counts, frequencies


def _optional_fraction(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def replay_support_audit(
    replay_states: ArrayLike,
    replay_actions: ArrayLike,
    query_states: ArrayLike,
    selected_actions: ArrayLike,
    *,
    task: str,
    bins: int | Sequence[int] = 11,
    sparse_threshold: int = 1,
    n_actions: int | None = None,
) -> dict[str, Any]:
    """Audit replay support for actions selected on an occupancy probe set.

    States are normalized with :func:`normalize_task_states` before binning.
    ``selected_action_zero_bin_fraction`` is the requested occupancy statistic:
    the fraction of query/occupied states whose selected action has no replay
    sample in the same state bin.  ``sparse_threshold`` additionally reports a
    predeclared low-count mask without changing the zero-support definition.
    """

    lower, _ = task_state_bounds(task)
    task_key = str(task).lower()
    if n_actions is None:
        n_actions = TASK_ACTION_COUNTS[task_key]
    if not isinstance(n_actions, Integral) or isinstance(n_actions, (bool, np.bool_)):
        raise TypeError("n_actions must be an integer.")
    n_actions = int(n_actions)
    if n_actions <= 0:
        raise ValueError("n_actions must be positive.")
    if not isinstance(sparse_threshold, Integral) or isinstance(
        sparse_threshold, (bool, np.bool_)
    ):
        raise TypeError("sparse_threshold must be an integer.")
    sparse_threshold = int(sparse_threshold)
    if sparse_threshold < 0:
        raise ValueError("sparse_threshold must be nonnegative.")

    replay_matrix = _state_matrix(replay_states, lower.size, "replay_states")
    query_matrix = _state_matrix(query_states, lower.size, "query_states")
    replay_action_array = _action_vector(
        replay_actions, replay_matrix.shape[0], n_actions, "replay_actions"
    )
    selected_action_array = _action_vector(
        selected_actions, query_matrix.shape[0], n_actions, "selected_actions"
    )
    bin_shape = _bins_per_dimension(bins, lower.size)
    support_counts = replay_support_counts(
        replay_matrix,
        replay_action_array,
        query_matrix,
        selected_action_array,
        task=task,
        bins=bin_shape,
        n_actions=n_actions,
    )

    zero = support_counts == 0
    sparse = support_counts <= sparse_threshold
    replay_counts, replay_frequencies = _frequency_dict(
        replay_action_array, n_actions
    )
    selected_counts, selected_frequencies = _frequency_dict(
        selected_action_array, n_actions
    )

    per_action: dict[int, dict[str, int | float | None]] = {}
    for action in range(n_actions):
        selected_mask = selected_action_array == action
        selected_count = int(np.count_nonzero(selected_mask))
        zero_count = int(np.count_nonzero(zero & selected_mask))
        sparse_count = int(np.count_nonzero(sparse & selected_mask))
        per_action[action] = {
            "query_count": selected_count,
            "zero_bin_count": zero_count,
            "zero_bin_fraction": _optional_fraction(zero_count, selected_count),
            "sparse_bin_count": sparse_count,
            "sparse_bin_fraction": _optional_fraction(sparse_count, selected_count),
        }

    replay_normalized_unclipped = normalize_task_states(replay_matrix, task, clip=False)
    query_normalized_unclipped = normalize_task_states(query_matrix, task, clip=False)
    replay_outside = np.any(
        (replay_normalized_unclipped < 0.0) | (replay_normalized_unclipped > 1.0),
        axis=1,
    )
    query_outside = np.any(
        (query_normalized_unclipped < 0.0) | (query_normalized_unclipped > 1.0),
        axis=1,
    )

    query_count = int(query_matrix.shape[0])
    zero_count = int(np.count_nonzero(zero))
    sparse_count = int(np.count_nonzero(sparse))
    return {
        "task": task_key,
        "replay_count": int(replay_matrix.shape[0]),
        "query_count": query_count,
        "bins_per_dimension": [int(value) for value in bin_shape],
        "sparse_threshold": sparse_threshold,
        "selected_action_zero_bin_count": zero_count,
        "selected_action_zero_bin_fraction": _optional_fraction(zero_count, query_count),
        "selected_action_sparse_bin_count": sparse_count,
        "selected_action_sparse_bin_fraction": _optional_fraction(
            sparse_count, query_count
        ),
        "selected_action_support_count_min": (
            None if query_count == 0 else int(np.min(support_counts))
        ),
        "selected_action_support_count_median": (
            None if query_count == 0 else float(np.median(support_counts))
        ),
        "selected_action_support_count_mean": (
            None if query_count == 0 else float(np.mean(support_counts))
        ),
        "replay_action_counts": replay_counts,
        "replay_action_frequencies": replay_frequencies,
        "selected_action_counts": selected_counts,
        "selected_action_frequencies": selected_frequencies,
        "per_selected_action": per_action,
        "replay_outside_bounds_count": int(np.count_nonzero(replay_outside)),
        "replay_outside_bounds_fraction": _optional_fraction(
            int(np.count_nonzero(replay_outside)), int(replay_matrix.shape[0])
        ),
        "query_outside_bounds_count": int(np.count_nonzero(query_outside)),
        "query_outside_bounds_fraction": _optional_fraction(
            int(np.count_nonzero(query_outside)), query_count
        ),
    }


def _profile_interval(
    probabilities: FloatArray,
    values: FloatArray,
    lower: float,
    upper: float,
) -> tuple[FloatArray, FloatArray]:
    """Slice profiles along their final axis, interpolating range endpoints."""

    interior = (probabilities > lower) & (probabilities < upper)
    selected_probabilities = np.concatenate(
        (np.asarray([lower]), probabilities[interior], np.asarray([upper]))
    )

    flat = values.reshape(-1, probabilities.size)
    selected_values = np.empty((flat.shape[0], selected_probabilities.size), dtype=np.float64)
    for row_index, row in enumerate(flat):
        selected_values[row_index] = np.interp(selected_probabilities, probabilities, row)
    return selected_probabilities, selected_values.reshape(
        values.shape[:-1] + (selected_probabilities.size,)
    )


def profile_auc(
    probabilities: ArrayLike,
    values: ArrayLike,
    *,
    p_min: float | None = None,
    p_max: float | None = None,
    normalize: bool = False,
) -> float | FloatArray:
    """Integrate one or more profiles over a certified probability interval.

    Profiles occupy the final axis.  Bounds that fall between evaluated
    probabilities are linearly interpolated.  With ``normalize=True``, divide
    the area by interval width so the result is the mean profile height.
    """

    probability_values = _finite_vector(probabilities, "probabilities")
    if probability_values.size < 2:
        raise ValueError("At least two profile probabilities are required.")
    if np.any(probability_values < 0.0) or np.any(probability_values > 1.0):
        raise ValueError("probabilities must lie in [0, 1].")
    if np.any(np.diff(probability_values) <= 0.0):
        raise ValueError("probabilities must be strictly increasing.")

    profile_values = np.asarray(values, dtype=np.float64)
    if profile_values.ndim == 0 or profile_values.shape[-1] != probability_values.size:
        raise ValueError("The final values axis must match probabilities.")
    if not np.all(np.isfinite(profile_values)):
        raise ValueError("values must contain only finite values.")

    lower = float(probability_values[0]) if p_min is None else float(p_min)
    upper = float(probability_values[-1]) if p_max is None else float(p_max)
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("p_min and p_max must be finite.")
    if lower < probability_values[0] or upper > probability_values[-1]:
        raise ValueError("The requested interval must lie inside the probability grid.")
    if lower >= upper:
        raise ValueError("p_min must be strictly smaller than p_max.")

    selected_probabilities, selected_values = _profile_interval(
        probability_values, profile_values, lower, upper
    )
    # np.trapezoid is available in the project's pinned NumPy and avoids the
    # deprecation warning attached to np.trapz in NumPy 2.x.
    areas = np.trapezoid(selected_values, x=selected_probabilities, axis=-1)
    if normalize:
        areas = areas / (upper - lower)
    if np.ndim(areas) == 0:
        return float(areas)
    return np.asarray(areas, dtype=np.float64)


def paired_profile_auc(
    probabilities: ArrayLike,
    learned_returns: ArrayLike,
    reference_returns: ArrayLike,
    *,
    p_min: float | None = None,
    p_max: float | None = None,
    normalize: bool = False,
) -> float | FloatArray:
    """Integrate paired ``learned - reference`` profile differences.

    Inputs must have exactly equal shape, preserving episode/seed pairing.  A
    vector returns one float; arrays such as ``(n_seeds, n_probabilities)``
    return one AUC per leading-index profile.
    """

    learned = np.asarray(learned_returns, dtype=np.float64)
    reference = np.asarray(reference_returns, dtype=np.float64)
    if learned.shape != reference.shape:
        raise ValueError("Paired profile arrays must have exactly equal shape.")
    if not np.all(np.isfinite(learned)) or not np.all(np.isfinite(reference)):
        raise ValueError("Paired profile returns must contain only finite values.")
    return profile_auc(
        probabilities,
        learned - reference,
        p_min=p_min,
        p_max=p_max,
        normalize=normalize,
    )


# Concise aliases for callers that name the complete output a report or audit.
backup_calibration_report = stratified_backup_calibration
support_audit = replay_support_audit


__all__ = [
    "TASK_ACTION_COUNTS",
    "TASK_STATE_BOUNDS",
    "backup_calibration_metrics",
    "backup_calibration_report",
    "normalize_task_states",
    "paired_profile_auc",
    "profile_auc",
    "replay_support_audit",
    "replay_support_counts",
    "stable_average_ranks",
    "stratified_backup_calibration",
    "support_audit",
    "task_state_bounds",
]
