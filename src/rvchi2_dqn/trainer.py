"""Gated online-replay trainer for RVChi2-DQN.

The implementation deliberately separates ordinary online Double DQN
pretraining from blockwise robust learning.  All branches start from the same
competent nominal checkpoint and an exact clone of its replay prefix.  They
then collect equal numbers of transitions under the nominal actuator kernel
into method-specific replay buffers.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import nn

from .artifacts import (
    artifact_hashes,
    git_provenance,
    library_versions,
    prepare_new_directory,
    sha256_file,
    torch_save_atomic,
    utc_now,
    write_csv_atomic,
    write_json_atomic,
    write_npz_atomic,
)
from .config import (
    ExperimentConfig,
    METHODS,
    SCIENTIFIC_SOURCE_FILES,
    validate_config,
)
from .diagnostics import (
    paired_profile_auc,
    replay_support_audit,
    replay_support_counts,
    stratified_backup_calibration,
)
from .envs import (
    LQRConfig,
    PendulumConfig,
    ReversalLQREnv,
    ReversalPendulumEnv,
)
from .networks import (
    AffineVariationalHeads,
    FullyNeuralVariationalHeads,
    QNetwork,
    frozen_ema_copy,
    gradient_l2_norm,
    summarize_auxiliary_outputs,
    update_ema_,
)
from .operators import (
    exact_binary_robust_continuation_torch,
    sampled_variational_g,
    two_mode_conditional_variational_expectation,
)
from .replay import ReplayBatch, ReplayBuffer


Tensor = torch.Tensor


@dataclass
class MethodBundle:
    name: str
    q: QNetwork
    optimizer: torch.optim.Optimizer
    replay: ReplayBuffer
    env: Any
    observation: np.ndarray
    collection_rng: np.random.Generator
    q_updates: int = 0
    environment_steps: int = 0
    auxiliary_updates: int = 0
    auxiliary: nn.Module | None = None
    deployed_auxiliary: nn.Module | None = None
    auxiliary_optimizer: torch.optim.Optimizer | None = None
    last_auxiliary_target: QNetwork | None = None
    diagnostic_sums: dict[str, float] = field(default_factory=dict)
    diagnostic_counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentResult:
    output_dir: Path
    summary: dict[str, Any]
    advanced_past_nominal_gate: bool


def _seed_everything(config: ExperimentConfig) -> None:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.torch_num_threads)
    if config.deterministic_torch:
        torch.use_deterministic_algorithms(True)


def _task_spec(config: ExperimentConfig) -> tuple[int, int, tuple[float, ...]]:
    if config.task == "lqr":
        return 2, 3, (2.0, 2.0)
    return 3, 3, (1.0, 1.0, 8.0)


def _make_env(
    config: ExperimentConfig,
    *,
    seed: int,
    fault_probability: float | None = None,
) -> ReversalLQREnv | ReversalPendulumEnv:
    probability = (
        config.nominal_fault_probability
        if fault_probability is None
        else float(fault_probability)
    )
    if config.task == "lqr":
        return ReversalLQREnv(
            LQRConfig(
                fault_probability=probability,
                horizon=config.evaluation_horizon,
            ),
            seed=seed,
        )
    return ReversalPendulumEnv(
        PendulumConfig(
            fault_probability=probability,
            horizon=config.evaluation_horizon,
        ),
        seed=seed,
    )


def _state_dict_hash(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def _scientific_source_hashes(repo_root: Path) -> dict[str, str]:
    """Hash every source file that defines or authenticates a scientific run."""

    return {
        name: sha256_file(repo_root / relative)
        for name, relative in SCIENTIFIC_SOURCE_FILES.items()
    }


def _tensor_batch(batch: ReplayBatch, device: torch.device) -> dict[str, Tensor]:
    return {
        "observations": torch.as_tensor(batch.observations, dtype=torch.float32, device=device),
        "actions": torch.as_tensor(batch.actions, dtype=torch.long, device=device),
        "rewards": torch.as_tensor(batch.scaled_rewards, dtype=torch.float32, device=device),
        "next_observations": torch.as_tensor(
            batch.next_observations, dtype=torch.float32, device=device
        ),
        "terminated": torch.as_tensor(batch.terminated, dtype=torch.bool, device=device),
        "truncated": torch.as_tensor(batch.truncated, dtype=torch.bool, device=device),
        "terminal_values": torch.as_tensor(
            batch.terminal_values, dtype=torch.float32, device=device
        ),
    }


def _bounded_values(q_values: Tensor, config: ExperimentConfig) -> tuple[Tensor, float]:
    bounded = torch.clamp(
        q_values,
        min=config.target_lower_bound,
        max=config.target_upper_bound,
    )
    fraction = float((bounded != q_values).to(torch.float32).mean().item())
    return bounded, fraction


def _terminal_continuation(
    next_values: Tensor, terminated: Tensor, terminal_values: Tensor
) -> Tensor:
    # Truncation is deliberately absent: it bootstraps through next_values.
    return torch.where(terminated, terminal_values, next_values)


def _clip_targets(raw_targets: Tensor, config: ExperimentConfig) -> tuple[Tensor, float]:
    targets = torch.clamp(
        raw_targets,
        min=config.target_lower_bound,
        max=config.target_upper_bound,
    )
    return targets, float((targets != raw_targets).to(torch.float32).mean().item())


def _double_dqn_targets(
    online: QNetwork,
    target: QNetwork,
    tensors: Mapping[str, Tensor],
    config: ExperimentConfig,
) -> tuple[Tensor, dict[str, float]]:
    with torch.no_grad():
        selected = torch.argmax(online(tensors["next_observations"]), dim=1)
        target_q, q_clip_fraction = _bounded_values(
            target(tensors["next_observations"]), config
        )
        next_values = target_q.gather(1, selected[:, None]).squeeze(1)
        continuation = _terminal_continuation(
            next_values, tensors["terminated"], tensors["terminal_values"]
        )
        raw_targets = tensors["rewards"] + config.gamma * continuation
        targets, target_clip_fraction = _clip_targets(raw_targets, config)
    return targets, {
        "successor_q_clip_fraction": q_clip_fraction,
        "target_clip_fraction": target_clip_fraction,
        "target_mean": float(targets.mean().item()),
    }


def _q_step(
    q: QNetwork,
    optimizer: torch.optim.Optimizer,
    tensors: Mapping[str, Tensor],
    targets: Tensor,
    config: ExperimentConfig,
) -> dict[str, float]:
    predictions = q(tensors["observations"]).gather(
        1, tensors["actions"][:, None]
    ).squeeze(1)
    loss = torch.nn.functional.smooth_l1_loss(predictions, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = float(
        torch.nn.utils.clip_grad_norm_(q.parameters(), config.q_gradient_clip).item()
    )
    optimizer.step()
    return {
        "q_loss": float(loss.detach().item()),
        "q_gradient_norm": gradient_norm,
        "absolute_td_mean": float(torch.abs(predictions.detach() - targets).mean().item()),
    }


def _epsilon(config: ExperimentConfig, step: int) -> float:
    fraction = min(max(step, 0) / max(config.epsilon_decay_steps, 1), 1.0)
    return config.epsilon_start + fraction * (config.epsilon_end - config.epsilon_start)


@torch.no_grad()
def _greedy_action(q: QNetwork, observation: np.ndarray, device: torch.device) -> int:
    tensor = torch.as_tensor(observation[None, :], dtype=torch.float32, device=device)
    return int(torch.argmax(q(tensor), dim=1).item())


def _choose_action(
    q: QNetwork,
    observation: np.ndarray,
    rng: np.random.Generator,
    n_actions: int,
    epsilon: float,
    rescue_fraction: float,
    device: torch.device,
) -> int:
    random_probability = 1.0 - (1.0 - epsilon) * (1.0 - rescue_fraction)
    if float(rng.random()) < random_probability:
        return int(rng.integers(n_actions))
    return _greedy_action(q, observation, device)


def _append_transition(
    replay: ReplayBuffer,
    observation: np.ndarray,
    action: int,
    step_result: tuple[np.ndarray, float, bool, bool, dict[str, Any]],
) -> tuple[np.ndarray, bool]:
    next_observation, scaled_reward, terminated, truncated, info = step_result
    replay.add(
        observation,
        action,
        scaled_reward,
        info["raw_reward"],
        next_observation,
        terminated,
        truncated,
        info["terminal_value"],
        info["fault"],
    )
    return next_observation, bool(terminated or truncated)


def _mean_rows(rows: Iterable[Mapping[str, float]]) -> dict[str, float]:
    materialized = list(rows)
    if not materialized:
        return {}
    keys = set.intersection(*(set(row) for row in materialized))
    return {
        key: float(np.mean([row[key] for row in materialized], dtype=np.float64))
        for key in sorted(keys)
    }


def _summarize_auxiliary_rows(
    rows: Iterable[Mapping[str, float]],
) -> dict[str, float]:
    """Retain block means and true within-block health maxima."""

    materialized = list(rows)
    summary = _mean_rows(materialized)
    for key in (
        "auxiliary_eta_projected_action_fraction",
        "auxiliary_eta_saturation_fraction",
        "auxiliary_u_floor_fraction",
        "auxiliary_gradient_norm",
    ):
        values = [float(row[key]) for row in materialized if key in row]
        if values:
            if not all(math.isfinite(value) for value in values):
                raise FloatingPointError(f"Non-finite auxiliary diagnostic {key!r}.")
            summary[f"{key}_max"] = max(values)
    return summary


def _auxiliary_health_summary(
    learning_rows: Iterable[Mapping[str, Any]],
    methods: Iterable[str],
) -> dict[str, dict[str, int | float | None]]:
    """Aggregate JSON-safe auxiliary health statistics over robust blocks."""

    materialized = list(learning_rows)
    result: dict[str, dict[str, int | float | None]] = {}
    specifications = {
        "eta_projection_fraction": "auxiliary_eta_projected_action_fraction",
        "eta_saturation_fraction": "auxiliary_eta_saturation_fraction",
        "u_floor_fraction": "auxiliary_u_floor_fraction",
        "gradient_norm": "auxiliary_gradient_norm",
    }
    for method in methods:
        rows = [
            row
            for row in materialized
            if row.get("phase") == "robust_outer" and row.get("method") == method
        ]
        health: dict[str, int | float | None] = {"block_count": len(rows)}
        for label, key in specifications.items():
            means = [float(row[key]) for row in rows if row.get(key) is not None]
            maxima = [
                float(row.get(f"{key}_max", row[key]))
                for row in rows
                if row.get(key) is not None
            ]
            if not all(math.isfinite(value) for value in means + maxima):
                raise FloatingPointError(f"Non-finite auxiliary health metric {key!r}.")
            health[f"{label}_mean"] = (
                None if not means else float(np.mean(means, dtype=np.float64))
            )
            health[f"{label}_max"] = None if not maxima else max(maxima)
        result[str(method)] = health
    return result


def _completed_status(phase: str) -> str:
    """Return an unambiguous successful terminal status for each seed phase."""

    statuses = {
        "smoke": "completed_smoke",
        "development": "completed_development_seed",
        "reporting": "completed_reporting_seed",
    }
    try:
        return statuses[phase]
    except KeyError as error:
        raise ValueError(f"Unknown experiment phase {phase!r}.") from error


def _nominal_competence_gate(
    aggregate_row: Mapping[str, Any],
    config: ExperimentConfig,
    *,
    smoke_bypass: bool | None = None,
) -> dict[str, Any]:
    """Evaluate one nominal p0 aggregate against the frozen competence rule."""

    observed_return = float(aggregate_row["mean_raw_return"])
    observed_failure = float(aggregate_row["failure_probability"])
    if not math.isfinite(observed_return) or not math.isfinite(observed_failure):
        raise FloatingPointError("Nominal competence aggregate must be finite.")
    gate: dict[str, Any] = {
        "passed": bool(
            observed_return >= config.nominal_competence_return
            and observed_failure <= config.nominal_competence_failure_rate
        ),
        "required_raw_return": config.nominal_competence_return,
        "required_max_failure_probability": config.nominal_competence_failure_rate,
        "observed_raw_return": observed_return,
        "observed_failure_probability": observed_failure,
    }
    if smoke_bypass is not None:
        gate["smoke_bypass"] = bool(smoke_bypass and not gate["passed"])
    return gate


def _pretrain_nominal(
    config: ExperimentConfig,
    q: QNetwork,
    replay: ReplayBuffer,
    device: torch.device,
) -> tuple[torch.optim.Optimizer, list[dict[str, Any]], int]:
    optimizer = torch.optim.Adam(
        q.parameters(),
        lr=(
            config.q_learning_rate
            if config.pretrain_q_learning_rate is None
            else config.pretrain_q_learning_rate
        ),
        weight_decay=config.q_weight_decay,
    )
    target = copy.deepcopy(q).eval()
    env = _make_env(config, seed=config.seed * 10_000 + 11)
    observation, _ = env.reset(seed=config.seed * 10_000 + 12)
    rng = np.random.default_rng(config.seed * 10_000 + 13)
    metrics: list[dict[str, Any]] = []
    window: list[dict[str, float]] = []
    updates = 0
    n_actions = q.n_actions

    for step in range(config.nominal_pretrain_steps):
        epsilon = _epsilon(config, step)
        action = _choose_action(
            q, observation, rng, n_actions, epsilon, 0.0, device
        )
        next_observation, ended = _append_transition(
            replay, observation, action, env.step(action)
        )
        observation = next_observation
        if ended:
            observation, _ = env.reset()

        if len(replay) >= config.learning_starts:
            for _ in range(config.nominal_updates_per_step):
                tensors = _tensor_batch(replay.sample(config.batch_size), device)
                targets, target_diagnostics = _double_dqn_targets(
                    q, target, tensors, config
                )
                row = _q_step(q, optimizer, tensors, targets, config)
                row.update(target_diagnostics)
                window.append(row)
                updates += 1
        if (step + 1) % config.nominal_target_update_interval == 0:
            target.load_state_dict(q.state_dict())
            metrics.append(
                {
                    "phase": "nominal_pretraining",
                    "block": (step + 1) // config.nominal_target_update_interval,
                    "environment_steps": step + 1,
                    "q_updates": updates,
                    "epsilon": epsilon,
                    "replay_size": len(replay),
                    **_mean_rows(window),
                }
            )
            window.clear()
    return optimizer, metrics, updates


def _method_env_seed(config: ExperimentConfig, method_index: int) -> int:
    return config.seed * 100_000 + 1_000 + method_index * 100


def _make_branches(
    config: ExperimentConfig,
    base_q: QNetwork,
    base_optimizer: torch.optim.Optimizer,
    prefix: ReplayBuffer,
    device: torch.device,
) -> dict[str, MethodBundle]:
    branches: dict[str, MethodBundle] = {}
    initial_optimizer_state = copy.deepcopy(base_optimizer.state_dict())
    for index, method in enumerate(config.enabled_methods):
        q = copy.deepcopy(base_q).to(device)
        optimizer = torch.optim.Adam(
            q.parameters(), lr=config.q_learning_rate, weight_decay=config.q_weight_decay
        )
        optimizer.load_state_dict(copy.deepcopy(initial_optimizer_state))
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = config.q_learning_rate
        replay = prefix.clone(seed=_method_env_seed(config, index) + 1)
        env = _make_env(config, seed=_method_env_seed(config, index) + 2)
        observation, _ = env.reset(seed=_method_env_seed(config, index) + 3)
        bundle = MethodBundle(
            name=method,
            q=q,
            optimizer=optimizer,
            replay=replay,
            env=env,
            observation=observation,
            collection_rng=np.random.default_rng(_method_env_seed(config, index) + 4),
        )
        if method == "affine":
            auxiliary = AffineVariationalHeads(
                q.representation_dim,
                q.n_actions,
                ell=config.ell,
                eta_l2_bound=config.eta_bound,
                rho_l2_bound=config.rho_bound,
                initial_u=config.affine_initial_u,
            ).to(device)
            with torch.no_grad():
                auxiliary.nu.copy_(q.q_head.weight / config.gamma)
                auxiliary.project_parameters_()
            bundle.auxiliary = auxiliary
            bundle.deployed_auxiliary = frozen_ema_copy(auxiliary)
            bundle.auxiliary_optimizer = torch.optim.Adam(
                auxiliary.parameters(), lr=config.auxiliary_learning_rate
            )
        elif method == "full_nn":
            _, _, observation_scale = _task_spec(config)
            auxiliary = FullyNeuralVariationalHeads(
                q.observation_dim,
                q.n_actions,
                config.hidden_dims,
                observation_scale,
                ell=config.ell,
                eta_bound=config.eta_bound,
                initial_u=config.full_initial_u,
            ).to(device)
            bundle.auxiliary = auxiliary
            bundle.deployed_auxiliary = frozen_ema_copy(auxiliary)
            bundle.auxiliary_optimizer = torch.optim.AdamW(
                auxiliary.parameters(),
                lr=config.auxiliary_learning_rate,
                weight_decay=config.auxiliary_weight_decay,
            )
        branches[method] = bundle
    return branches


def _collect_branch(
    bundle: MethodBundle,
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, float]:
    action_counts = np.zeros(bundle.q.n_actions, dtype=np.int64)
    fault_count = 0
    terminated_count = 0
    truncated_count = 0
    for _ in range(config.collection_steps_per_block):
        action = _choose_action(
            bundle.q,
            bundle.observation,
            bundle.collection_rng,
            bundle.q.n_actions,
            config.branch_epsilon_end,
            config.coverage_rescue_random_fraction,
            device,
        )
        action_counts[action] += 1
        step_result = bundle.env.step(action)
        fault_count += int(step_result[4]["fault"])
        terminated_count += int(step_result[2])
        truncated_count += int(step_result[3])
        next_observation, ended = _append_transition(
            bundle.replay, bundle.observation, action, step_result
        )
        bundle.observation = next_observation
        bundle.environment_steps += 1
        if ended:
            bundle.observation, _ = bundle.env.reset()
    return {
        "collection_fault_fraction": fault_count / config.collection_steps_per_block,
        "collection_termination_fraction": terminated_count / config.collection_steps_per_block,
        "collection_truncation_fraction": truncated_count / config.collection_steps_per_block,
        **{
            f"collection_action_{index}_fraction": float(count / config.collection_steps_per_block)
            for index, count in enumerate(action_counts)
        },
    }


def _sampled_continuations(
    target: QNetwork,
    tensors: Mapping[str, Tensor],
    config: ExperimentConfig,
) -> tuple[Tensor, float]:
    with torch.no_grad():
        next_q, clip_fraction = _bounded_values(
            target(tensors["next_observations"]), config
        )
        next_values = torch.max(next_q, dim=1).values
        continuation = _terminal_continuation(
            next_values, tensors["terminated"], tensors["terminal_values"]
        )
    return continuation, clip_fraction


def _auxiliary_step(
    bundle: MethodBundle,
    target: QNetwork,
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, float]:
    assert bundle.auxiliary is not None
    assert bundle.deployed_auxiliary is not None
    assert bundle.auxiliary_optimizer is not None
    tensors = _tensor_batch(bundle.replay.sample(config.batch_size), device)
    continuation, q_clip_fraction = _sampled_continuations(target, tensors, config)
    bundle.auxiliary_optimizer.zero_grad(set_to_none=True)

    if bundle.name == "affine":
        assert isinstance(bundle.auxiliary, AffineVariationalHeads)
        with torch.no_grad():
            representation = target.encode(tensors["observations"])
        eta, u = bundle.auxiliary(representation.detach(), tensors["actions"])
    else:
        assert isinstance(bundle.auxiliary, FullyNeuralVariationalHeads)
        eta, u = bundle.auxiliary(tensors["observations"], tensors["actions"])

    objective = sampled_variational_g(eta, u, continuation.detach(), config.chi2_delta)
    loss = -objective.mean()
    loss.backward()
    unclipped_gradient_norm = gradient_l2_norm(bundle.auxiliary.parameters())
    torch.nn.utils.clip_grad_norm_(bundle.auxiliary.parameters(), config.q_gradient_clip)
    bundle.auxiliary_optimizer.step()
    diagnostics: dict[str, float] = {
        "auxiliary_objective": float(objective.detach().mean().item()),
        "auxiliary_loss": float(loss.detach().item()),
        "auxiliary_gradient_norm": unclipped_gradient_norm,
        "auxiliary_successor_q_clip_fraction": q_clip_fraction,
    }
    if bundle.name == "affine":
        projection = bundle.auxiliary.project_parameters_()
        diagnostics.update(projection.as_dict(prefix="auxiliary_"))
    update_ema_(
        bundle.deployed_auxiliary,
        bundle.auxiliary,
        config.auxiliary_ema_decay,
    )
    with torch.no_grad():
        if bundle.name == "affine":
            deployed_eta, deployed_u = bundle.deployed_auxiliary(
                representation, tensors["actions"]
            )
        else:
            deployed_eta, deployed_u = bundle.deployed_auxiliary(
                tensors["observations"], tensors["actions"]
            )
        diagnostics.update(
            {
                f"auxiliary_{key}": value
                for key, value in summarize_auxiliary_outputs(
                    deployed_eta,
                    deployed_u,
                    ell=config.ell,
                    eta_bound=(config.eta_bound if bundle.name == "full_nn" else None),
                    next_values=continuation,
                ).items()
            }
        )
    bundle.auxiliary_updates += 1
    return diagnostics


def _observations_to_states(config: ExperimentConfig, observations: np.ndarray) -> np.ndarray:
    if config.task == "lqr":
        return np.asarray(observations, dtype=np.float64)
    observations = np.asarray(observations, dtype=np.float64)
    return np.column_stack(
        (np.arctan2(observations[:, 1], observations[:, 0]), observations[:, 2])
    )


def _enumerated_mode_arrays(
    config: ExperimentConfig,
    observations: np.ndarray,
    actions: np.ndarray,
) -> dict[str, np.ndarray]:
    env = _make_env(config, seed=config.seed * 100_000 + 909)
    states = _observations_to_states(config, observations)
    healthy_observations: list[np.ndarray] = []
    fault_observations: list[np.ndarray] = []
    healthy_terminated: list[bool] = []
    fault_terminated: list[bool] = []
    healthy_terminal_values: list[float] = []
    fault_terminal_values: list[float] = []
    for state, action in zip(states, actions):
        healthy, fault = env.enumerate_modes(state, int(action))
        if not math.isclose(
            healthy.scaled_reward, fault.scaled_reward, rel_tol=0.0, abs_tol=1e-12
        ):
            raise RuntimeError("The selected task has a mode-dependent immediate reward.")
        healthy_observations.append(healthy.observation)
        fault_observations.append(fault.observation)
        healthy_terminated.append(healthy.terminated)
        fault_terminated.append(fault.terminated)
        healthy_terminal_values.append(healthy.terminal_value)
        fault_terminal_values.append(fault.terminal_value)
    return {
        "healthy_observations": np.asarray(healthy_observations, dtype=np.float32),
        "fault_observations": np.asarray(fault_observations, dtype=np.float32),
        "healthy_terminated": np.asarray(healthy_terminated, dtype=bool),
        "fault_terminated": np.asarray(fault_terminated, dtype=bool),
        "healthy_terminal_values": np.asarray(healthy_terminal_values, dtype=np.float32),
        "fault_terminal_values": np.asarray(fault_terminal_values, dtype=np.float32),
    }


def _mode_values(
    target: QNetwork,
    mode_arrays: Mapping[str, np.ndarray],
    prefix: str,
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[Tensor, float]:
    observations = torch.as_tensor(
        mode_arrays[f"{prefix}_observations"], dtype=torch.float32, device=device
    )
    terminated = torch.as_tensor(
        mode_arrays[f"{prefix}_terminated"], dtype=torch.bool, device=device
    )
    terminal_values = torch.as_tensor(
        mode_arrays[f"{prefix}_terminal_values"], dtype=torch.float32, device=device
    )
    with torch.no_grad():
        q_values, clip_fraction = _bounded_values(target(observations), config)
        values = torch.max(q_values, dim=1).values
        values = _terminal_continuation(values, terminated, terminal_values)
    return values, clip_fraction


def _exact_targets(
    target: QNetwork,
    tensors: Mapping[str, Tensor],
    batch: ReplayBatch,
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[Tensor, dict[str, float]]:
    mode_arrays = _enumerated_mode_arrays(config, batch.observations, batch.actions)
    healthy, healthy_clip = _mode_values(target, mode_arrays, "healthy", config, device)
    fault, fault_clip = _mode_values(target, mode_arrays, "fault", config, device)
    with torch.no_grad():
        robust = exact_binary_robust_continuation_torch(
            healthy,
            fault,
            config.nominal_fault_probability,
            config.chi2_delta,
        )
        raw_targets = tensors["rewards"] + config.gamma * robust
        targets, target_clip = _clip_targets(raw_targets, config)
    return targets, {
        "successor_q_clip_fraction": 0.5 * (healthy_clip + fault_clip),
        "target_clip_fraction": target_clip,
        "target_mean": float(targets.mean().item()),
    }


def _variational_targets(
    bundle: MethodBundle,
    target: QNetwork,
    tensors: Mapping[str, Tensor],
    config: ExperimentConfig,
) -> tuple[Tensor, dict[str, float]]:
    if config.chi2_delta == 0.0:
        return _double_dqn_targets(bundle.q, target, tensors, config)
    assert bundle.deployed_auxiliary is not None
    continuation, q_clip_fraction = _sampled_continuations(target, tensors, config)
    with torch.no_grad():
        if bundle.name == "affine":
            representation = target.encode(tensors["observations"])
            eta, u = bundle.deployed_auxiliary(representation, tensors["actions"])
        else:
            eta, u = bundle.deployed_auxiliary(
                tensors["observations"], tensors["actions"]
            )
        conditional = sampled_variational_g(
            eta, u, continuation, config.chi2_delta
        )
        raw_targets = tensors["rewards"] + config.gamma * conditional
        targets, target_clip_fraction = _clip_targets(raw_targets, config)
    return targets, {
        "successor_q_clip_fraction": q_clip_fraction,
        "target_clip_fraction": target_clip_fraction,
        "target_mean": float(targets.mean().item()),
        "conditional_mean": float(conditional.mean().item()),
    }


def _train_q_block(
    bundle: MethodBundle,
    target: QNetwork,
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, float]:
    rows: list[dict[str, float]] = []
    for _ in range(config.q_updates_per_block):
        batch = bundle.replay.sample(config.batch_size)
        tensors = _tensor_batch(batch, device)
        if bundle.name == "nominal":
            targets, target_diagnostics = _double_dqn_targets(
                bundle.q, target, tensors, config
            )
        elif bundle.name == "exact":
            targets, target_diagnostics = _exact_targets(
                target, tensors, batch, config, device
            )
        else:
            targets, target_diagnostics = _variational_targets(
                bundle, target, tensors, config
            )
        row = _q_step(bundle.q, bundle.optimizer, tensors, targets, config)
        row.update(target_diagnostics)
        rows.append(row)
        bundle.q_updates += 1
    return _mean_rows(rows)


def _evaluate_one_policy(
    q: QNetwork,
    config: ExperimentConfig,
    probability: float,
    reset_seeds: np.ndarray,
    fault_uniforms: np.ndarray,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode, reset_seed in enumerate(reset_seeds):
        env = _make_env(
            config,
            seed=int(reset_seed),
            fault_probability=probability,
        )
        observation, _ = env.reset(seed=int(reset_seed))
        raw_return = 0.0
        discounted_return = 0.0
        discount = 1.0
        failure = False
        action_counts = np.zeros(q.n_actions, dtype=np.int64)
        length = 0
        for step in range(config.evaluation_horizon):
            action = _greedy_action(q, observation, device)
            action_counts[action] += 1
            next_observation, _, terminated, truncated, info = env.step(
                action, fault_uniform=float(fault_uniforms[episode, step])
            )
            raw_return += float(info["raw_reward"])
            discounted_return += discount * float(info["raw_reward"])
            if terminated:
                # Match the established LQR descriptive raw-return convention;
                # Bellman discounting applies gamma to the explicit successor.
                raw_return += float(info["terminal_value"])
                discounted_return += (
                    discount * config.gamma * float(info["terminal_value"])
                )
            length += 1
            observation = next_observation
            failure = failure or bool(terminated)
            if terminated or truncated:
                break
            discount *= config.gamma
        rows.append(
            {
                "episode": episode,
                "fault_probability": float(probability),
                "raw_return": raw_return,
                "discounted_return": discounted_return,
                "failure": int(failure),
                "length": length,
                **{
                    f"action_{index}_count": int(count)
                    for index, count in enumerate(action_counts)
                },
            }
        )
    return rows


def evaluate_profiles(
    networks: Mapping[str, QNetwork],
    config: ExperimentConfig,
    device: torch.device,
    *,
    probabilities: Iterable[float] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    probability_grid = tuple(
        sorted(
            set(
                config.certified_probabilities + config.ood_probabilities
                if probabilities is None
                else tuple(float(value) for value in probabilities)
            )
        )
    )
    rng = np.random.default_rng(config.seed * 1_000_000 + 77)
    reset_seeds = rng.integers(
        0, np.iinfo(np.int32).max, size=config.evaluation_episodes, dtype=np.int64
    )
    fault_uniforms = rng.random(
        (config.evaluation_episodes, config.evaluation_horizon)
    )
    episode_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    for probability in probability_grid:
        for method, network in networks.items():
            rows = _evaluate_one_policy(
                network,
                config,
                probability,
                reset_seeds,
                fault_uniforms,
                device,
            )
            for row in rows:
                episode_rows.append({"method": method, **row})
            aggregate_rows.append(
                {
                    "method": method,
                    "fault_probability": probability,
                    "inside_certified_interval": probability <= 0.25,
                    "ood_stress_test": probability > 0.25,
                    "mean_raw_return": float(np.mean([row["raw_return"] for row in rows])),
                    "mean_discounted_return": float(
                        np.mean([row["discounted_return"] for row in rows])
                    ),
                    "failure_probability": float(np.mean([row["failure"] for row in rows])),
                    "mean_length": float(np.mean([row["length"] for row in rows])),
                }
            )
    return episode_rows, aggregate_rows


def _probe_grid(config: ExperimentConfig) -> tuple[np.ndarray, np.ndarray]:
    points = config.support_grid_points
    if config.task == "lqr":
        first = np.linspace(-2.0, 2.0, points)
        second = np.linspace(-2.0, 2.0, points)
        x, velocity = np.meshgrid(first, second, indexing="ij")
        observations = np.column_stack((x.reshape(-1), velocity.reshape(-1))).astype(np.float32)
    else:
        angles = np.linspace(-np.pi, np.pi, points, endpoint=False)
        velocities = np.linspace(-8.0, 8.0, points)
        angle, velocity = np.meshgrid(angles, velocities, indexing="ij")
        observations = np.column_stack(
            (
                np.cos(angle).reshape(-1),
                np.sin(angle).reshape(-1),
                velocity.reshape(-1),
            )
        ).astype(np.float32)
    observations = np.repeat(observations, 3, axis=0)
    actions = np.tile(np.arange(3, dtype=np.int64), observations.shape[0] // 3)
    return observations, actions


def _calibration_for_method(
    bundle: MethodBundle,
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if bundle.name not in {"affine", "full_nn"}:
        raise ValueError("Calibration is defined for learned variational methods.")
    assert bundle.last_auxiliary_target is not None
    assert bundle.deployed_auxiliary is not None
    observations, actions = _probe_grid(config)
    mode_arrays = _enumerated_mode_arrays(config, observations, actions)
    healthy, _ = _mode_values(
        bundle.last_auxiliary_target, mode_arrays, "healthy", config, device
    )
    fault, _ = _mode_values(
        bundle.last_auxiliary_target, mode_arrays, "fault", config, device
    )
    observation_tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
    action_tensor = torch.as_tensor(actions, dtype=torch.long, device=device)
    with torch.no_grad():
        exact = exact_binary_robust_continuation_torch(
            healthy,
            fault,
            config.nominal_fault_probability,
            config.chi2_delta,
        )
        if bundle.name == "affine":
            representation = bundle.last_auxiliary_target.encode(observation_tensor)
            eta, u = bundle.deployed_auxiliary(representation, action_tensor)
        else:
            eta, u = bundle.deployed_auxiliary(observation_tensor, action_tensor)
        learned = two_mode_conditional_variational_expectation(
            eta,
            u,
            healthy,
            fault,
            config.nominal_fault_probability,
            config.chi2_delta,
        )
        target_q = bundle.last_auxiliary_target(observation_tensor)
        sorted_q = torch.sort(target_q, dim=1, descending=True).values
        margins = sorted_q[:, 0] - sorted_q[:, 1]
    replay = bundle.replay.as_batch()
    support_counts = replay_support_counts(
        replay.observations,
        replay.actions,
        observations,
        actions,
        task=config.task,
        bins=config.support_grid_points,
    )
    margin_array = margins.detach().cpu().numpy()
    threshold = float(np.quantile(margin_array, 0.10))
    report = stratified_backup_calibration(
        exact.detach().cpu().numpy(),
        learned.detach().cpu().numpy(),
        actions=actions,
        action_values=(0, 1, 2),
        support=support_counts > 0,
        sparse=support_counts <= 1,
        decision_boundary=margin_array <= threshold,
    )
    report["decision_boundary_margin_threshold"] = threshold
    return report, {
        "observations": observations,
        "actions": actions,
        "exact": exact.detach().cpu().numpy(),
        "learned": learned.detach().cpu().numpy(),
        "support_counts": support_counts,
        "policy_margins": margin_array,
        "eta": eta.detach().cpu().numpy(),
        "u": u.detach().cpu().numpy(),
    }


def _policy_and_support_diagnostics(
    branches: Mapping[str, MethodBundle],
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, Any]:
    nominal_replay = branches["nominal"].replay.as_batch()
    result: dict[str, Any] = {}
    for method, bundle in branches.items():
        replay = bundle.replay.as_batch()
        query_states = np.concatenate(
            (nominal_replay.observations, replay.observations), axis=0
        )
        if query_states.shape[0] > 20_000:
            indices = np.linspace(0, query_states.shape[0] - 1, 20_000).astype(np.int64)
            query_states = query_states[indices]
        with torch.no_grad():
            query_tensor = torch.as_tensor(query_states, dtype=torch.float32, device=device)
            selected = torch.argmax(bundle.q(query_tensor), dim=1).cpu().numpy()
            nominal_selected = torch.argmax(
                branches["nominal"].q(query_tensor), dim=1
            ).cpu().numpy()
        audit = replay_support_audit(
            replay.observations,
            replay.actions,
            query_states,
            selected,
            task=config.task,
            bins=config.support_grid_points,
        )
        audit["occupied_policy_disagreement_with_nominal"] = float(
            np.mean(selected != nominal_selected)
        )
        result[method] = audit
    return result


def _profile_summary(
    aggregate_rows: list[dict[str, Any]], config: ExperimentConfig
) -> dict[str, Any]:
    by_method: dict[str, dict[float, dict[str, Any]]] = {}
    for row in aggregate_rows:
        by_method.setdefault(str(row["method"]), {})[
            float(row["fault_probability"])
        ] = row
    probabilities = np.asarray(config.certified_probabilities, dtype=np.float64)
    summary: dict[str, Any] = {}
    nominal = by_method["nominal"]
    nominal_profile = np.asarray(
        [nominal[float(p)]["mean_raw_return"] for p in probabilities]
    )
    for method, rows in by_method.items():
        profile = np.asarray([rows[float(p)]["mean_raw_return"] for p in probabilities])
        summary[method] = {
            "nominal_kernel_raw_return": rows[0.10]["mean_raw_return"],
            "certified_boundary_raw_return": rows[0.25]["mean_raw_return"],
            "nominal_kernel_cost_vs_nominal": (
                rows[0.10]["mean_raw_return"] - nominal[0.10]["mean_raw_return"]
            ),
            "paired_boundary_advantage_vs_nominal": (
                rows[0.25]["mean_raw_return"] - nominal[0.25]["mean_raw_return"]
            ),
            "certified_profile_auc": float(
                np.trapezoid(profile, x=probabilities)
            ),
            "paired_advantage_auc_p010_to_p025": float(
                paired_profile_auc(
                    probabilities,
                    profile,
                    nominal_profile,
                    p_min=0.10,
                    p_max=0.25,
                )
            ),
        }
    return summary


def _save_run(
    output_dir: Path,
    config: ExperimentConfig,
    summary: dict[str, Any],
    learning_rows: list[dict[str, Any]],
    episode_rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
    checkpoint_episode_rows: list[dict[str, Any]],
    checkpoint_aggregate_rows: list[dict[str, Any]],
    branches: Mapping[str, MethodBundle] | None,
    calibration_arrays: Mapping[str, Mapping[str, np.ndarray]],
    base_q: QNetwork,
    replay_prefix: ReplayBuffer,
    initial_hash: str,
    source_hashes: Mapping[str, str],
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    if _scientific_source_hashes(repo_root) != dict(source_hashes):
        raise RuntimeError("Scientific source files changed while the run was active.")
    write_json_atomic(output_dir / "config.json", config.to_dict())
    write_csv_atomic(output_dir / "learning_metrics.csv", learning_rows)
    write_csv_atomic(output_dir / "evaluation_episodes.csv", episode_rows)
    write_csv_atomic(output_dir / "evaluation_summary.csv", aggregate_rows)
    write_csv_atomic(
        output_dir / "checkpoint_evaluation_episodes.csv", checkpoint_episode_rows
    )
    write_csv_atomic(
        output_dir / "checkpoint_evaluation_summary.csv", checkpoint_aggregate_rows
    )
    write_json_atomic(output_dir / "summary.json", summary)
    checkpoint: dict[str, Any] = {
        "base_nominal_q": base_q.state_dict(),
        "shared_initial_q_sha256": initial_hash,
    }
    if branches is not None:
        checkpoint["methods"] = {
            method: {
                "q": bundle.q.state_dict(),
                "q_optimizer": bundle.optimizer.state_dict(),
                "auxiliary": (
                    None if bundle.auxiliary is None else bundle.auxiliary.state_dict()
                ),
                "deployed_auxiliary": (
                    None
                    if bundle.deployed_auxiliary is None
                    else bundle.deployed_auxiliary.state_dict()
                ),
                "auxiliary_optimizer": (
                    None
                    if bundle.auxiliary_optimizer is None
                    else bundle.auxiliary_optimizer.state_dict()
                ),
                "last_auxiliary_target": (
                    None
                    if bundle.last_auxiliary_target is None
                    else bundle.last_auxiliary_target.state_dict()
                ),
            }
            for method, bundle in branches.items()
        }
    torch_save_atomic(output_dir / "checkpoints.pt", checkpoint)
    flattened_arrays: dict[str, np.ndarray] = {}
    for method, arrays in calibration_arrays.items():
        for name, value in arrays.items():
            flattened_arrays[f"{method}__{name}"] = value
    write_npz_atomic(output_dir / "backup_calibration.npz", **flattened_arrays)
    metadata = {
        "created_at_utc": utc_now(),
        "artifact_schema_version": config.schema_version,
        "source_sha256": dict(source_hashes),
        "git": git_provenance(repo_root),
        "libraries": library_versions(),
        "shared_initial_replay_size": len(replay_prefix),
        "shared_initial_replay_total_added": replay_prefix.total_added,
        "shared_initial_q_sha256": initial_hash,
        "environment_step_budget_per_branch": (
            0
            if branches is None
            else config.outer_blocks * config.collection_steps_per_block
        ),
        "q_update_budget_per_branch": (
            0 if branches is None else config.outer_blocks * config.q_updates_per_block
        ),
        "auxiliary_update_budget_per_variational_branch": (
            0
            if branches is None
            else config.outer_blocks * config.auxiliary_updates_per_block
        ),
        "practical_theorem_scope": (
            "Replay, target networks, learned neural features, Adam/AdamW, EMA, "
            "and online data collection are practical extensions not covered by "
            "the fixed-linear-feature finite-time theorem."
        ),
    }
    write_json_atomic(output_dir / "metadata.json", metadata)
    names = (
        "config.json",
        "learning_metrics.csv",
        "evaluation_episodes.csv",
        "evaluation_summary.csv",
        "checkpoint_evaluation_episodes.csv",
        "checkpoint_evaluation_summary.csv",
        "summary.json",
        "checkpoints.pt",
        "backup_calibration.npz",
        "metadata.json",
    )
    write_json_atomic(
        output_dir / "manifest.json",
        {
            "completed_at_utc": utc_now(),
            "artifact_schema_version": config.schema_version,
            "source_sha256": dict(source_hashes),
            "files_sha256": artifact_hashes(output_dir, names),
            "config": config.to_dict(),
            "status": summary["status"],
        },
    )


def run_experiment(config: ExperimentConfig, output_dir: Path | str) -> ExperimentResult:
    """Run one seed through the staged gates and save immutable artifacts."""

    validate_config(config)
    repo_root = Path(__file__).resolve().parents[2]
    source_hashes = _scientific_source_hashes(repo_root)
    _seed_everything(config)
    destination = prepare_new_directory(output_dir)
    device = torch.device(config.device)
    observation_dim, n_actions, observation_scale = _task_spec(config)
    base_q = QNetwork(
        observation_dim,
        n_actions,
        config.hidden_dims,
        observation_scale,
    ).to(device)
    replay_prefix = ReplayBuffer(
        config.replay_capacity,
        (observation_dim,),
        seed=config.seed * 10_000 + 21,
    )
    base_optimizer, learning_rows, pretrain_updates = _pretrain_nominal(
        config, base_q, replay_prefix, device
    )
    initial_hash = _state_dict_hash(base_q)
    nominal_episodes, nominal_aggregate = evaluate_profiles(
        {"nominal": base_q},
        config,
        device,
        probabilities=(config.nominal_fault_probability,),
    )
    nominal_row = nominal_aggregate[0]
    nominal_gate = _nominal_competence_gate(
        nominal_row,
        config,
        smoke_bypass=config.phase == "smoke",
    )
    nominal_gate["evaluation_stage"] = "nominal_pretraining"
    nominal_gate["fault_probability"] = config.nominal_fault_probability
    competence_passed = bool(nominal_gate["passed"])
    continuation_override = bool(
        config.phase in {"development", "reporting"}
        and config.continue_after_failed_pretraining_gate
        and not competence_passed
    )
    nominal_gate["diagnostic_only"] = bool(
        config.phase in {"development", "reporting"}
        and config.continue_after_failed_pretraining_gate
    )
    nominal_gate["continued_after_failure"] = continuation_override
    advance = config.phase == "smoke" or competence_passed or continuation_override
    summary: dict[str, Any] = {
        "result_schema_version": config.schema_version,
        "task": config.task,
        "phase": config.phase,
        "seed": config.seed,
        "status": (
            "nominal_gate_passed"
            if competence_passed
            else (
                "nominal_gate_failed_continuing"
                if advance
                else "nominal_gate_failed"
            )
        ),
        "nominal_gate": nominal_gate,
        "continued_after_failed_pretraining_gate": continuation_override,
        "nominal_pretraining_environment_steps": config.nominal_pretrain_steps,
        "nominal_pretraining_q_updates": pretrain_updates,
        "shared_initial_q_sha256": initial_hash,
    }
    if not advance or config.stop_after_nominal_gate:
        if advance and config.stop_after_nominal_gate:
            summary["status"] = (
                "nominal_gate_passed_stopped"
                if competence_passed
                else "nominal_gate_failed_stopped"
            )
        _save_run(
            destination,
            config,
            summary,
            learning_rows,
            nominal_episodes,
            nominal_aggregate,
            [],
            [],
            None,
            {},
            base_q,
            replay_prefix,
            initial_hash,
            source_hashes,
        )
        return ExperimentResult(destination, summary, False)

    branches = _make_branches(
        config, base_q, base_optimizer, replay_prefix, device
    )
    branch_initial_hashes = {
        method: _state_dict_hash(bundle.q) for method, bundle in branches.items()
    }
    if set(branch_initial_hashes.values()) != {initial_hash}:
        raise RuntimeError("Method branches did not start from the identical Q checkpoint.")

    all_episode_rows: list[dict[str, Any]] = []
    all_aggregate_rows: list[dict[str, Any]] = []
    for block in range(1, config.outer_blocks + 1):
        collection = {
            method: _collect_branch(bundle, config, device)
            for method, bundle in branches.items()
        }
        targets = {
            method: copy.deepcopy(bundle.q).eval()
            for method, bundle in branches.items()
        }
        for target in targets.values():
            for parameter in target.parameters():
                parameter.requires_grad_(False)

        auxiliary_metrics: dict[str, dict[str, float]] = {}
        if config.chi2_delta > 0.0:
            for method in (
                candidate
                for candidate in ("affine", "full_nn")
                if candidate in branches
            ):
                bundle = branches[method]
                rows = [
                    _auxiliary_step(bundle, targets[method], config, device)
                    for _ in range(config.auxiliary_updates_per_block)
                ]
                auxiliary_metrics[method] = _summarize_auxiliary_rows(rows)
                bundle.last_auxiliary_target = copy.deepcopy(targets[method]).eval()

        for method, bundle in branches.items():
            q_metrics = _train_q_block(bundle, targets[method], config, device)
            row: dict[str, Any] = {
                "phase": "robust_outer",
                "block": block,
                "method": method,
                "environment_steps": bundle.environment_steps,
                "q_updates": bundle.q_updates,
                "auxiliary_updates": bundle.auxiliary_updates,
                "replay_size": len(bundle.replay),
                **collection[method],
                **q_metrics,
            }
            row.update(auxiliary_metrics.get(method, {}))
            learning_rows.append(row)

        if block % config.evaluation_every_blocks == 0 or block == config.outer_blocks:
            episodes, aggregates = evaluate_profiles(
                {method: bundle.q for method, bundle in branches.items()},
                config,
                device,
                probabilities=(
                    config.nominal_fault_probability,
                    0.25,
                ),
            )
            for row in episodes:
                all_episode_rows.append({"checkpoint_block": block, **row})
            for row in aggregates:
                all_aggregate_rows.append({"checkpoint_block": block, **row})

    final_episodes, final_aggregates = evaluate_profiles(
        {method: bundle.q for method, bundle in branches.items()},
        config,
        device,
    )
    for row in final_episodes:
        row["checkpoint_block"] = config.outer_blocks
        row["is_final_frozen_sweep"] = True
    for row in final_aggregates:
        row["checkpoint_block"] = config.outer_blocks
        row["is_final_frozen_sweep"] = True
    final_nominal_rows = [
        row
        for row in final_aggregates
        if row["method"] == "nominal"
        and math.isclose(
            float(row["fault_probability"]),
            config.nominal_fault_probability,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ]
    if len(final_nominal_rows) != 1:
        raise RuntimeError("Final sweep did not contain exactly one nominal p0 row.")
    final_nominal_gate = _nominal_competence_gate(
        final_nominal_rows[0], config
    )
    final_nominal_gate["evaluation_stage"] = "final_frozen_sweep"
    final_nominal_gate["checkpoint_block"] = config.outer_blocks
    final_nominal_gate["fault_probability"] = config.nominal_fault_probability

    calibrations: dict[str, Any] = {}
    calibration_arrays: dict[str, Mapping[str, np.ndarray]] = {}
    if config.chi2_delta > 0.0:
        for method in (
            candidate
            for candidate in ("affine", "full_nn")
            if candidate in branches
        ):
            report, arrays = _calibration_for_method(
                branches[method], config, device
            )
            calibrations[method] = report
            calibration_arrays[method] = arrays
    support = _policy_and_support_diagnostics(branches, config, device)
    for method, bundle in branches.items():
        replay_snapshot = bundle.replay.as_batch()
        snapshot_observations = replay_snapshot.observations
        if snapshot_observations.shape[0] > 20_000:
            snapshot_indices = np.linspace(
                0, snapshot_observations.shape[0] - 1, 20_000
            ).astype(np.int64)
            snapshot_observations = snapshot_observations[snapshot_indices]
        with torch.no_grad():
            snapshot_actions = torch.argmax(
                bundle.q(
                    torch.as_tensor(
                        snapshot_observations, dtype=torch.float32, device=device
                    )
                ),
                dim=1,
            ).cpu().numpy()
        method_arrays = dict(calibration_arrays.get(method, {}))
        method_arrays["occupancy_observations"] = snapshot_observations
        method_arrays["occupancy_greedy_actions"] = snapshot_actions
        calibration_arrays[method] = method_arrays
    profile = _profile_summary(final_aggregates, config)
    target_clip_max = {
        method: max(
            (
                float(row.get("target_clip_fraction", 0.0))
                for row in learning_rows
                if row.get("phase") == "robust_outer" and row.get("method") == method
            ),
            default=0.0,
        )
        for method in branches
    }
    auxiliary_health = _auxiliary_health_summary(
        learning_rows,
        (
            method
            for method in ("affine", "full_nn")
            if method in branches
        ),
    )
    summary.update(
        {
            "status": _completed_status(config.phase),
            "methods": profile,
            "final_nominal_gate": final_nominal_gate,
            "backup_calibration": calibrations,
            "replay_support": support,
            "auxiliary_health": auxiliary_health,
            "max_block_mean_target_clip_fraction": target_clip_max,
            "target_clipping_gate_passed": all(
                value <= config.target_clip_failure_rate
                for value in target_clip_max.values()
            ),
            "branch_initial_q_sha256": branch_initial_hashes,
            "equal_branch_environment_steps": len(
                {bundle.environment_steps for bundle in branches.values()}
            )
            == 1,
            "equal_branch_q_updates": len(
                {bundle.q_updates for bundle in branches.values()}
            )
            == 1,
        }
    )
    _save_run(
        destination,
        config,
        summary,
        learning_rows,
        final_episodes,
        final_aggregates,
        all_episode_rows,
        all_aggregate_rows,
        branches,
        calibration_arrays,
        base_q,
        replay_prefix,
        initial_hash,
        source_hashes,
    )
    return ExperimentResult(destination, summary, True)


__all__ = [
    "ExperimentResult",
    "MethodBundle",
    "evaluate_profiles",
    "run_experiment",
]
