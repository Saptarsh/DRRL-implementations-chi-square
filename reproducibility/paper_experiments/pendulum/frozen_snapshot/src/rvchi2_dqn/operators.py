"""Canonical chi-square operators for RVChi2-DQN.

The ambiguity set uses the Pearson direction

    D_chi2(q || p0) = sum_i (q_i - p0_i)^2 / p0_i,

with no factor of one half.  Consequently the variational coefficient is
``sqrt(1 + delta) / 2``.  This module keeps the convention and Gymnasium
termination semantics in one small, independently testable place.
"""

from __future__ import annotations

import math
from typing import TypeAlias

import numpy as np
import torch


Array: TypeAlias = np.ndarray
Tensor: TypeAlias = torch.Tensor


def _probability(value: float, name: str, *, strict: bool = False) -> float:
    result = float(value)
    lower_ok = result > 0.0 if strict else result >= 0.0
    upper_ok = result < 1.0 if strict else result <= 1.0
    if not math.isfinite(result) or not lower_ok or not upper_ok:
        interval = "strictly in (0, 1)" if strict else "in [0, 1]"
        raise ValueError(f"{name} must lie {interval}.")
    return result


def _radius(delta: float) -> float:
    result = float(delta)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("delta must be finite and nonnegative.")
    return result


def pearson_bernoulli_divergence(
    probability: float, nominal_probability: float
) -> float:
    """Return ``D_chi2(Ber(probability) || Ber(nominal_probability))``."""

    probability = _probability(probability, "probability")
    nominal_probability = _probability(
        nominal_probability, "nominal_probability", strict=True
    )
    return float(
        (probability - nominal_probability) ** 2
        / (nominal_probability * (1.0 - nominal_probability))
    )


def pearson_bernoulli_interval(
    nominal_probability: float, delta: float
) -> tuple[float, float]:
    """Return the simplex-clipped Bernoulli Pearson-chi-square ball."""

    nominal_probability = _probability(
        nominal_probability, "nominal_probability", strict=True
    )
    delta = _radius(delta)
    displacement = math.sqrt(
        delta * nominal_probability * (1.0 - nominal_probability)
    )
    return (
        max(0.0, nominal_probability - displacement),
        min(1.0, nominal_probability + displacement),
    )


def exact_binary_robust_continuation_numpy(
    healthy_values: Array | float,
    fault_values: Array | float,
    nominal_fault_probability: float,
    delta: float,
) -> Array:
    """Solve the exact two-mode robust continuation with NumPy.

    At zero radius this deliberately takes the nominal-expectation path rather
    than relying on a degenerate variational maximizer.
    """

    healthy, fault = np.broadcast_arrays(
        np.asarray(healthy_values, dtype=np.float64),
        np.asarray(fault_values, dtype=np.float64),
    )
    if not np.all(np.isfinite(healthy)) or not np.all(np.isfinite(fault)):
        raise ValueError("continuation values must be finite.")
    p0 = _probability(
        nominal_fault_probability, "nominal_fault_probability", strict=True
    )
    delta = _radius(delta)
    if delta == 0.0:
        return healthy + p0 * (fault - healthy)
    lower, upper = pearson_bernoulli_interval(p0, delta)
    adverse_probability = np.where(fault < healthy, upper, lower)
    return healthy + adverse_probability * (fault - healthy)


def exact_binary_robust_continuation_torch(
    healthy_values: Tensor,
    fault_values: Tensor,
    nominal_fault_probability: float,
    delta: float,
) -> Tensor:
    """Torch equivalent of :func:`exact_binary_robust_continuation_numpy`."""

    if not isinstance(healthy_values, torch.Tensor) or not isinstance(
        fault_values, torch.Tensor
    ):
        raise TypeError("healthy_values and fault_values must be torch tensors.")
    if (
        not healthy_values.is_floating_point()
        or not fault_values.is_floating_point()
    ):
        raise TypeError("continuation tensors must have floating-point dtype.")
    healthy, fault = torch.broadcast_tensors(healthy_values, fault_values)
    if not bool(torch.all(torch.isfinite(healthy)).item()) or not bool(
        torch.all(torch.isfinite(fault)).item()
    ):
        raise ValueError("continuation values must be finite.")
    p0 = _probability(
        nominal_fault_probability, "nominal_fault_probability", strict=True
    )
    delta = _radius(delta)
    if delta == 0.0:
        return healthy + p0 * (fault - healthy)
    lower, upper = pearson_bernoulli_interval(p0, delta)
    adverse_probability = torch.where(
        fault < healthy,
        torch.as_tensor(upper, dtype=healthy.dtype, device=healthy.device),
        torch.as_tensor(lower, dtype=healthy.dtype, device=healthy.device),
    )
    return healthy + adverse_probability * (fault - healthy)


def sampled_variational_g_numpy(
    eta: Array | float,
    u: Array | float,
    successor_value: Array | float,
    delta: float,
) -> Array:
    """Evaluate one sampled quadratic-over-linear variational continuation."""

    eta_array, u_array, value_array = np.broadcast_arrays(
        np.asarray(eta, dtype=np.float64),
        np.asarray(u, dtype=np.float64),
        np.asarray(successor_value, dtype=np.float64),
    )
    if (
        not np.all(np.isfinite(eta_array))
        or not np.all(np.isfinite(u_array))
        or not np.all(np.isfinite(value_array))
    ):
        raise ValueError("eta, u, and successor_value must be finite.")
    if np.any(u_array <= 0.0):
        raise ValueError("u must be strictly positive.")
    c_delta = math.sqrt(1.0 + _radius(delta))
    positive_part = np.maximum(eta_array - value_array, 0.0)
    return eta_array - 0.5 * c_delta * (
        np.square(positive_part) / u_array + u_array
    )


def sampled_variational_g(
    eta: Tensor, u: Tensor, successor_value: Tensor, delta: float
) -> Tensor:
    """Torch sampled variational continuation used by neural training."""

    if not all(
        isinstance(value, torch.Tensor) for value in (eta, u, successor_value)
    ):
        raise TypeError("eta, u, and successor_value must be torch tensors.")
    eta_value, u_value, continuation = torch.broadcast_tensors(
        eta, u, successor_value
    )
    if not all(
        value.is_floating_point()
        for value in (eta_value, u_value, continuation)
    ):
        raise TypeError("eta, u, and successor_value must have floating-point dtype.")
    if not all(
        bool(torch.all(torch.isfinite(value)).item())
        for value in (eta_value, u_value, continuation)
    ):
        raise ValueError("eta, u, and successor_value must be finite.")
    if bool(torch.any(u_value <= 0.0).item()):
        raise ValueError("u must be strictly positive.")
    c_delta = math.sqrt(1.0 + _radius(delta))
    positive_part = torch.relu(eta_value - continuation)
    return eta_value - 0.5 * c_delta * (
        positive_part.square() / u_value + u_value
    )


def two_mode_conditional_variational_expectation_numpy(
    eta: Array | float,
    u: Array | float,
    healthy_values: Array | float,
    fault_values: Array | float,
    nominal_fault_probability: float,
    delta: float,
) -> Array:
    """Return the nominal two-mode expectation of the sampled objective."""

    p0 = _probability(
        nominal_fault_probability, "nominal_fault_probability", strict=True
    )
    healthy_g = sampled_variational_g_numpy(eta, u, healthy_values, delta)
    fault_g = sampled_variational_g_numpy(eta, u, fault_values, delta)
    return (1.0 - p0) * healthy_g + p0 * fault_g


def two_mode_conditional_variational_expectation(
    eta: Tensor,
    u: Tensor,
    healthy_values: Tensor,
    fault_values: Tensor,
    nominal_fault_probability: float,
    delta: float,
) -> Tensor:
    """Torch nominal two-mode expectation of the sampled objective."""

    p0 = _probability(
        nominal_fault_probability, "nominal_fault_probability", strict=True
    )
    healthy_g = sampled_variational_g(eta, u, healthy_values, delta)
    fault_g = sampled_variational_g(eta, u, fault_values, delta)
    return (1.0 - p0) * healthy_g + p0 * fault_g


def resolve_successor_values(
    next_values: Tensor,
    terminated: Tensor,
    truncated: Tensor | None = None,
    *,
    terminal_value: float = 0.0,
) -> Tensor:
    """Resolve Gymnasium successor values before applying the robust transform.

    Physical termination substitutes ``terminal_value``.  A time-limit
    truncation does not mask the successor and therefore bootstraps normally.
    ``truncated`` is accepted explicitly so callers cannot accidentally collapse
    the two Gymnasium signals into one ``done`` flag.
    """

    if (
        not isinstance(next_values, torch.Tensor)
        or not next_values.is_floating_point()
    ):
        raise TypeError("next_values must be a floating-point torch tensor.")
    if not isinstance(terminated, torch.Tensor) or terminated.dtype != torch.bool:
        raise TypeError("terminated must be a boolean torch tensor.")
    values, terminated_mask = torch.broadcast_tensors(next_values, terminated)
    if truncated is not None:
        if not isinstance(truncated, torch.Tensor) or truncated.dtype != torch.bool:
            raise TypeError("truncated must be a boolean torch tensor when provided.")
        torch.broadcast_tensors(values, truncated)
    if not math.isfinite(float(terminal_value)):
        raise ValueError("terminal_value must be finite.")
    terminal = torch.as_tensor(
        terminal_value, dtype=values.dtype, device=values.device
    )
    return torch.where(terminated_mask, terminal, values)


def transition_variational_g(
    eta: Tensor,
    u: Tensor,
    next_values: Tensor,
    terminated: Tensor,
    truncated: Tensor | None,
    delta: float,
    *,
    terminal_value: float = 0.0,
) -> Tensor:
    """Resolve physical termination, then evaluate the complete ``g`` sample."""

    continuation = resolve_successor_values(
        next_values,
        terminated,
        truncated,
        terminal_value=terminal_value,
    )
    return sampled_variational_g(eta, u, continuation, delta)


# Familiar concise aliases used elsewhere in the repository.
chi2_bernoulli_divergence = pearson_bernoulli_divergence
certified_bernoulli_interval = pearson_bernoulli_interval


__all__ = [
    "certified_bernoulli_interval",
    "chi2_bernoulli_divergence",
    "exact_binary_robust_continuation_numpy",
    "exact_binary_robust_continuation_torch",
    "pearson_bernoulli_divergence",
    "pearson_bernoulli_interval",
    "resolve_successor_values",
    "sampled_variational_g",
    "sampled_variational_g_numpy",
    "transition_variational_g",
    "two_mode_conditional_variational_expectation",
    "two_mode_conditional_variational_expectation_numpy",
]
