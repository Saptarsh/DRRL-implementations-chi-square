"""Neural models for the isolated RVChi2-DQN implementation.

The primary auxiliary model is affine in action-specific parameters once the
Q encoder is frozen.  Its feature dictionaries use a fixed denominator,
rather than a state-dependent normalization:

``psi(h)  = [1, h] / sqrt(d + 1)``
``zeta(h) = [1, relu(h), relu(-h)] / sqrt(d + 1)``.

For a tanh representation both dictionaries have Euclidean norm at most one.
Consequently, per-action L2 projection gives direct output bounds while the
nonnegative projection of ``rho`` guarantees ``u >= ell``.  The affine
parameters are ordinary :class:`torch.nn.Parameter` objects so one optimizer,
including its Adam moments, can persist across target blocks.

The secondary model uses independent neural networks for eta and the positive
scale.  It is an engineering ablation and makes no concavity claim.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F


Tensor = torch.Tensor


def _positive_dimensions(hidden_dims: Sequence[int]) -> tuple[int, ...]:
    resolved = tuple(int(width) for width in hidden_dims)
    if not resolved or any(width < 1 for width in resolved):
        raise ValueError("hidden_dims must contain at least one positive width.")
    return resolved


def _observation_scale(
    observation_dim: int, observation_scale: Sequence[float] | None
) -> Tensor:
    if observation_dim < 1:
        raise ValueError("observation_dim must be positive.")
    values = (
        torch.ones(observation_dim, dtype=torch.float32)
        if observation_scale is None
        else torch.as_tensor(observation_scale, dtype=torch.float32)
    )
    if (
        values.shape != (observation_dim,)
        or not bool(torch.all(torch.isfinite(values)))
        or bool(torch.any(values <= 0.0))
    ):
        raise ValueError(
            "observation_scale must be finite, positive, and match observation_dim."
        )
    return values


def _validate_observations(observations: Tensor, observation_dim: int) -> None:
    if observations.ndim != 2 or observations.shape[1] != observation_dim:
        raise ValueError(
            f"observations must have shape (batch, {observation_dim}); "
            f"got {tuple(observations.shape)}."
        )


def _validate_actions(actions: Tensor, batch_size: int, n_actions: int) -> None:
    if actions.ndim != 1 or actions.shape[0] != batch_size:
        raise ValueError("actions must have shape (batch,).")
    if actions.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise ValueError("actions must use an integer dtype.")
    if actions.numel() and (
        int(torch.min(actions).item()) < 0
        or int(torch.max(actions).item()) >= n_actions
    ):
        raise ValueError("actions contain an out-of-range action index.")


def fixed_normalized_features(representation: Tensor) -> tuple[Tensor, Tensor]:
    """Return the fixed-scale signed and split-sign affine dictionaries.

    ``representation`` must be a two-dimensional batch.  When every component
    of ``h`` lies in ``[-1, 1]``, both returned rows have L2 norm at most one.
    The same ``sqrt(d + 1)`` denominator is correct for ``zeta`` because only
    one of ``relu(h_i)`` and ``relu(-h_i)`` is nonzero for each coordinate.
    """

    if representation.ndim != 2 or representation.shape[1] < 1:
        raise ValueError("representation must have shape (batch, positive_dim).")
    denominator = math.sqrt(float(representation.shape[1]) + 1.0)
    ones = torch.ones(
        (representation.shape[0], 1),
        dtype=representation.dtype,
        device=representation.device,
    )
    psi = torch.cat((ones, representation), dim=1) / denominator
    zeta = torch.cat(
        (ones, torch.relu(representation), torch.relu(-representation)), dim=1
    ) / denominator
    return psi, zeta


# A short alias is convenient in training and diagnostic code.
affine_features = fixed_normalized_features


class QNetwork(nn.Module):
    """Finite-action Q network with a bounded tanh encoder and linear head."""

    def __init__(
        self,
        observation_dim: int,
        n_actions: int,
        hidden_dims: Sequence[int] = (128, 128),
        observation_scale: Sequence[float] | None = None,
        *,
        observation_clip: float | None = 5.0,
    ) -> None:
        super().__init__()
        if n_actions < 2:
            raise ValueError("n_actions must be at least two.")
        hidden_dims = _positive_dimensions(hidden_dims)
        if observation_clip is not None and (
            not math.isfinite(observation_clip) or observation_clip <= 0.0
        ):
            raise ValueError("observation_clip must be finite and positive or None.")

        self.observation_dim = int(observation_dim)
        self.n_actions = int(n_actions)
        self.hidden_dims = hidden_dims
        self.observation_clip = observation_clip
        self.register_buffer(
            "observation_scale",
            _observation_scale(self.observation_dim, observation_scale),
        )

        layers: list[nn.Module] = []
        input_dim = self.observation_dim
        for width in hidden_dims:
            layer = nn.Linear(input_dim, width)
            nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
            nn.init.zeros_(layer.bias)
            layers.extend((layer, nn.Tanh()))
            input_dim = width
        self.encoder = nn.Sequential(*layers)
        self.q_head = nn.Linear(self.representation_dim + 1, self.n_actions, bias=False)
        nn.init.orthogonal_(self.q_head.weight, gain=0.01)

    @property
    def representation_dim(self) -> int:
        return self.hidden_dims[-1]

    @property
    def q_feature_dim(self) -> int:
        return self.representation_dim + 1

    def normalize_observations(self, observations: Tensor) -> Tensor:
        _validate_observations(observations, self.observation_dim)
        normalized = observations / self.observation_scale
        if self.observation_clip is not None:
            normalized = torch.clamp(
                normalized, -self.observation_clip, self.observation_clip
            )
        return normalized

    def encode(self, observations: Tensor) -> Tensor:
        return self.encoder(self.normalize_observations(observations))

    def q_features(self, observations: Tensor) -> Tensor:
        psi, _ = fixed_normalized_features(self.encode(observations))
        return psi

    def forward(self, observations: Tensor) -> Tensor:
        return self.q_head(self.q_features(observations))

    def greedy_actions(self, observations: Tensor) -> Tensor:
        return torch.argmax(self(observations), dim=1)


@dataclass(frozen=True)
class ProjectionDiagnostics:
    """Projection activity from one post-optimizer affine projection."""

    eta_projected_action_fraction: float
    rho_nonnegative_projected_action_fraction: float
    rho_negative_element_fraction: float
    rho_radius_projected_action_fraction: float
    rho_any_projected_action_fraction: float
    eta_max_row_norm: float
    rho_max_row_norm: float

    def as_dict(self, prefix: str = "") -> dict[str, float]:
        return {
            f"{prefix}{name}": float(value)
            for name, value in self.__dict__.items()
        }


class AffineVariationalHeads(nn.Module):
    """Action-specific affine eta/u heads for a frozen bounded representation.

    Call :meth:`project_parameters_` immediately after every optimizer step.
    Projection is in-place, so parameter identities and optimizer moment state
    remain intact across macro-blocks.
    """

    def __init__(
        self,
        representation_dim: int,
        n_actions: int,
        *,
        ell: float,
        eta_l2_bound: float,
        rho_l2_bound: float,
        initial_u: float,
    ) -> None:
        super().__init__()
        if representation_dim < 1 or n_actions < 2:
            raise ValueError(
                "representation_dim must be positive and n_actions at least two."
            )
        for name, value in (
            ("ell", ell),
            ("eta_l2_bound", eta_l2_bound),
            ("rho_l2_bound", rho_l2_bound),
            ("initial_u", initial_u),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if initial_u <= ell:
            raise ValueError("initial_u must be strictly greater than ell.")

        self.representation_dim = int(representation_dim)
        self.n_actions = int(n_actions)
        self.ell = float(ell)
        self.eta_l2_bound = float(eta_l2_bound)
        self.rho_l2_bound = float(rho_l2_bound)
        self.initial_u = float(initial_u)
        self.eta_feature_dim = self.representation_dim + 1
        self.rho_feature_dim = 2 * self.representation_dim + 1

        intercept = (self.initial_u - self.ell) * math.sqrt(
            self.representation_dim + 1.0
        )
        if intercept > self.rho_l2_bound:
            raise ValueError(
                "rho_l2_bound is too small to represent initial_u with the "
                "intercept-only initialization."
            )
        self.nu = nn.Parameter(torch.zeros(self.n_actions, self.eta_feature_dim))
        self.rho = nn.Parameter(torch.zeros(self.n_actions, self.rho_feature_dim))
        with torch.no_grad():
            self.rho[:, 0].fill_(intercept)

    def feature_dictionaries(self, representation: Tensor) -> tuple[Tensor, Tensor]:
        if (
            representation.ndim != 2
            or representation.shape[1] != self.representation_dim
        ):
            raise ValueError(
                "representation does not match the configured representation_dim."
            )
        return fixed_normalized_features(representation)

    def all_outputs_from_features(
        self, psi: Tensor, zeta: Tensor
    ) -> tuple[Tensor, Tensor]:
        if (
            psi.ndim != 2
            or zeta.ndim != 2
            or psi.shape[0] != zeta.shape[0]
            or psi.shape[1] != self.eta_feature_dim
            or zeta.shape[1] != self.rho_feature_dim
        ):
            raise ValueError("psi/zeta shapes do not match the configured affine heads.")
        eta = psi @ self.nu.transpose(0, 1)
        u = self.ell + zeta @ self.rho.transpose(0, 1)
        return eta, u

    def all_outputs(self, representation: Tensor) -> tuple[Tensor, Tensor]:
        return self.all_outputs_from_features(*self.feature_dictionaries(representation))

    def outputs_from_features(
        self, psi: Tensor, zeta: Tensor, actions: Tensor
    ) -> tuple[Tensor, Tensor]:
        eta, u = self.all_outputs_from_features(psi, zeta)
        _validate_actions(actions, eta.shape[0], self.n_actions)
        indices = actions.to(dtype=torch.long)[:, None]
        return eta.gather(1, indices).squeeze(1), u.gather(1, indices).squeeze(1)

    def outputs(self, representation: Tensor, actions: Tensor) -> tuple[Tensor, Tensor]:
        psi, zeta = self.feature_dictionaries(representation)
        return self.outputs_from_features(psi, zeta, actions)

    forward = outputs

    @torch.no_grad()
    def initialize_eta_(self, weights: Tensor) -> ProjectionDiagnostics:
        """Copy action-wise affine eta weights without replacing the Parameter."""

        if weights.shape != self.nu.shape:
            raise ValueError(f"weights must have shape {tuple(self.nu.shape)}.")
        self.nu.copy_(weights.to(device=self.nu.device, dtype=self.nu.dtype))
        return self.project_parameters_()

    @torch.no_grad()
    def reset_scale_intercept_(self, initial_u: float | None = None) -> None:
        """Restore an exactly constant, intercept-only initial scale."""

        value = self.initial_u if initial_u is None else float(initial_u)
        if not math.isfinite(value) or value <= self.ell:
            raise ValueError("initial_u must be finite and strictly greater than ell.")
        intercept = (value - self.ell) * math.sqrt(self.representation_dim + 1.0)
        if intercept > self.rho_l2_bound:
            raise ValueError("rho_l2_bound is too small for the requested initial_u.")
        self.rho.zero_()
        self.rho[:, 0].fill_(intercept)

    @torch.no_grad()
    def project_parameters_(self) -> ProjectionDiagnostics:
        """Project each action row onto its own feasible L2 set in place."""

        if not bool(torch.all(torch.isfinite(self.nu))) or not bool(
            torch.all(torch.isfinite(self.rho))
        ):
            raise FloatingPointError("Cannot project non-finite affine parameters.")

        eta_norms = torch.linalg.vector_norm(self.nu, dim=1)
        eta_projected = eta_norms > self.eta_l2_bound
        eta_scale = torch.clamp(
            self.eta_l2_bound / eta_norms.clamp_min(torch.finfo(self.nu.dtype).tiny),
            max=1.0,
        )
        self.nu.mul_(eta_scale[:, None])

        negative_elements = self.rho < 0.0
        negative_actions = torch.any(negative_elements, dim=1)
        negative_element_fraction = float(
            negative_elements.to(dtype=torch.float32).mean().item()
        )
        self.rho.clamp_(min=0.0)
        rho_norms = torch.linalg.vector_norm(self.rho, dim=1)
        rho_radius_projected = rho_norms > self.rho_l2_bound
        rho_scale = torch.clamp(
            self.rho_l2_bound
            / rho_norms.clamp_min(torch.finfo(self.rho.dtype).tiny),
            max=1.0,
        )
        self.rho.mul_(rho_scale[:, None])
        rho_any_projected = negative_actions | rho_radius_projected

        eta_post = torch.linalg.vector_norm(self.nu, dim=1)
        rho_post = torch.linalg.vector_norm(self.rho, dim=1)
        return ProjectionDiagnostics(
            eta_projected_action_fraction=float(
                eta_projected.to(dtype=torch.float32).mean().item()
            ),
            rho_nonnegative_projected_action_fraction=float(
                negative_actions.to(dtype=torch.float32).mean().item()
            ),
            rho_negative_element_fraction=negative_element_fraction,
            rho_radius_projected_action_fraction=float(
                rho_radius_projected.to(dtype=torch.float32).mean().item()
            ),
            rho_any_projected_action_fraction=float(
                rho_any_projected.to(dtype=torch.float32).mean().item()
            ),
            eta_max_row_norm=float(eta_post.max().item()),
            rho_max_row_norm=float(rho_post.max().item()),
        )


def _make_mlp(
    input_dim: int, hidden_dims: Sequence[int], output_dim: int
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = input_dim
    for width in hidden_dims:
        layer = nn.Linear(current, width)
        nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
        nn.init.zeros_(layer.bias)
        layers.extend((layer, nn.Tanh()))
        current = width
    layers.append(nn.Linear(current, output_dim))
    return nn.Sequential(*layers)


def _inverse_softplus(value: float) -> float:
    if value > 20.0:
        return value + math.log1p(-math.exp(-value))
    return math.log(math.expm1(value))


class FullyNeuralVariationalHeads(nn.Module):
    """Independent eta and positive-scale MLPs for the neural ablation."""

    def __init__(
        self,
        observation_dim: int,
        n_actions: int,
        hidden_dims: Sequence[int] = (128, 128),
        observation_scale: Sequence[float] | None = None,
        *,
        ell: float,
        eta_bound: float,
        initial_u: float,
        observation_clip: float | None = 5.0,
    ) -> None:
        super().__init__()
        if n_actions < 2:
            raise ValueError("n_actions must be at least two.")
        hidden_dims = _positive_dimensions(hidden_dims)
        for name, value in (
            ("ell", ell),
            ("eta_bound", eta_bound),
            ("initial_u", initial_u),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if initial_u <= ell:
            raise ValueError("initial_u must be strictly greater than ell.")
        if observation_clip is not None and (
            not math.isfinite(observation_clip) or observation_clip <= 0.0
        ):
            raise ValueError("observation_clip must be finite and positive or None.")

        self.observation_dim = int(observation_dim)
        self.n_actions = int(n_actions)
        self.hidden_dims = hidden_dims
        self.ell = float(ell)
        self.eta_bound = float(eta_bound)
        self.initial_u = float(initial_u)
        self.observation_clip = observation_clip
        self.register_buffer(
            "observation_scale",
            _observation_scale(self.observation_dim, observation_scale),
        )

        self.eta_network = _make_mlp(
            self.observation_dim, hidden_dims, self.n_actions
        )
        self.raw_u_network = _make_mlp(
            self.observation_dim, hidden_dims, self.n_actions
        )
        eta_output = self.eta_network[-1]
        u_output = self.raw_u_network[-1]
        assert isinstance(eta_output, nn.Linear)
        assert isinstance(u_output, nn.Linear)
        nn.init.orthogonal_(eta_output.weight, gain=0.01)
        nn.init.zeros_(eta_output.bias)
        nn.init.zeros_(u_output.weight)
        nn.init.constant_(u_output.bias, _inverse_softplus(initial_u - ell))

    def normalize_observations(self, observations: Tensor) -> Tensor:
        _validate_observations(observations, self.observation_dim)
        normalized = observations / self.observation_scale
        if self.observation_clip is not None:
            normalized = torch.clamp(
                normalized, -self.observation_clip, self.observation_clip
            )
        return normalized

    def all_outputs(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        normalized = self.normalize_observations(observations)
        eta = self.eta_bound * torch.tanh(self.eta_network(normalized))
        u = self.ell + F.softplus(self.raw_u_network(normalized))
        return eta, u

    def outputs(self, observations: Tensor, actions: Tensor) -> tuple[Tensor, Tensor]:
        eta, u = self.all_outputs(observations)
        _validate_actions(actions, eta.shape[0], self.n_actions)
        indices = actions.to(dtype=torch.long)[:, None]
        return eta.gather(1, indices).squeeze(1), u.gather(1, indices).squeeze(1)

    forward = outputs


def frozen_ema_copy(module: nn.Module) -> nn.Module:
    """Return an evaluation-mode copy suitable for deployed EMA targets."""

    deployed = copy.deepcopy(module).eval()
    for parameter in deployed.parameters():
        parameter.requires_grad_(False)
    return deployed


@torch.no_grad()
def update_ema_(deployed: nn.Module, online: nn.Module, decay: float) -> None:
    """Update a same-structure deployed copy without replacing its tensors."""

    if not math.isfinite(decay) or not 0.0 <= decay < 1.0:
        raise ValueError("decay must lie in [0, 1).")
    deployed_parameters = dict(deployed.named_parameters())
    online_parameters = dict(online.named_parameters())
    if deployed_parameters.keys() != online_parameters.keys():
        raise ValueError("EMA modules do not have matching parameter names.")
    for name, target in deployed_parameters.items():
        source = online_parameters[name]
        if target.shape != source.shape:
            raise ValueError(f"EMA parameter shape mismatch for {name}.")
        target.mul_(decay).add_(source, alpha=1.0 - decay)

    deployed_buffers = dict(deployed.named_buffers())
    online_buffers = dict(online.named_buffers())
    if deployed_buffers.keys() != online_buffers.keys():
        raise ValueError("EMA modules do not have matching buffer names.")
    for name, target in deployed_buffers.items():
        source = online_buffers[name]
        if target.shape != source.shape:
            raise ValueError(f"EMA buffer shape mismatch for {name}.")
        target.copy_(source)


def gradient_l2_norm(parameters: Iterable[nn.Parameter]) -> float:
    """Return the global L2 norm of currently populated parameter gradients."""

    squared = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            gradient = parameter.grad.detach()
            squared += float(torch.sum(gradient * gradient).item())
    return math.sqrt(squared)


@torch.no_grad()
def summarize_auxiliary_outputs(
    eta: Tensor,
    u: Tensor,
    *,
    ell: float,
    eta_bound: float | None = None,
    next_values: Tensor | None = None,
    floor_tolerance: float = 1e-5,
    saturation_fraction: float = 0.98,
) -> dict[str, float]:
    """Summarize scale-floor, eta-saturation, and perspective diagnostics."""

    if eta.shape != u.shape or eta.numel() < 1:
        raise ValueError("eta and u must be nonempty tensors with equal shape.")
    if not math.isfinite(ell) or ell <= 0.0:
        raise ValueError("ell must be finite and positive.")
    if not math.isfinite(floor_tolerance) or floor_tolerance < 0.0:
        raise ValueError("floor_tolerance must be finite and nonnegative.")
    if not 0.0 < saturation_fraction < 1.0:
        raise ValueError("saturation_fraction must lie in (0, 1).")
    if not bool(torch.all(torch.isfinite(eta))) or not bool(
        torch.all(torch.isfinite(u))
    ):
        raise FloatingPointError("Auxiliary outputs contain non-finite values.")
    if bool(torch.any(u <= 0.0)):
        raise FloatingPointError("Auxiliary scale must be strictly positive.")

    eta_flat = eta.reshape(-1).to(dtype=torch.float64)
    u_flat = u.reshape(-1).to(dtype=torch.float64)
    result = {
        "eta_min": float(eta_flat.min().item()),
        "eta_mean": float(eta_flat.mean().item()),
        "eta_max": float(eta_flat.max().item()),
        "u_min": float(u_flat.min().item()),
        "u_mean": float(u_flat.mean().item()),
        "u_max": float(u_flat.max().item()),
        "u_p10": float(torch.quantile(u_flat, 0.10).item()),
        "u_p50": float(torch.quantile(u_flat, 0.50).item()),
        "u_p90": float(torch.quantile(u_flat, 0.90).item()),
        "u_p99": float(torch.quantile(u_flat, 0.99).item()),
        "u_floor_fraction": float(
            (u_flat <= ell + floor_tolerance).to(dtype=torch.float64).mean().item()
        ),
    }
    if eta_bound is not None:
        if not math.isfinite(eta_bound) or eta_bound <= 0.0:
            raise ValueError("eta_bound must be finite and positive when provided.")
        result["eta_saturation_fraction"] = float(
            (
                torch.abs(eta_flat) >= saturation_fraction * eta_bound
            ).to(dtype=torch.float64).mean().item()
        )
    if next_values is not None:
        if next_values.shape != eta.shape:
            raise ValueError("next_values must have the same shape as eta and u.")
        values = next_values.reshape(-1).to(dtype=torch.float64)
        if not bool(torch.all(torch.isfinite(values))):
            raise FloatingPointError("next_values contain non-finite values.")
        x = torch.relu(eta_flat - values)
        ratio = x / u_flat
        result.update(
            {
                "x_over_u_mean": float(ratio.mean().item()),
                "x2_over_u2_mean": float(torch.square(ratio).mean().item()),
            }
        )
    return result


# Explicit aliases make call sites read naturally while retaining concise class
# names for tests and interactive diagnostics.
RVChi2QNetwork = QNetwork
FullNeuralVariationalHeads = FullyNeuralVariationalHeads


__all__ = [
    "AffineVariationalHeads",
    "FullNeuralVariationalHeads",
    "FullyNeuralVariationalHeads",
    "ProjectionDiagnostics",
    "QNetwork",
    "RVChi2QNetwork",
    "affine_features",
    "fixed_normalized_features",
    "frozen_ema_copy",
    "gradient_l2_norm",
    "summarize_auxiliary_outputs",
    "update_ema_",
]
