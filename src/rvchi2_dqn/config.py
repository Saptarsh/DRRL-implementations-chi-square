"""Versioned configuration and seed guards for RVChi2-DQN."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Any


SMOKE_SEEDS = frozenset({1, 2, 3})
DEVELOPMENT_SEEDS = frozenset({31, 32, 33, 34, 35})
REPORTING_SEEDS = frozenset(range(101, 111))
METHODS = ("nominal", "exact", "affine", "full_nn")
SCHEMA_VERSION = 2
SCIENTIFIC_SOURCE_FILES = {
    "artifacts": "src/rvchi2_dqn/artifacts.py",
    "config": "src/rvchi2_dqn/config.py",
    "operators": "src/rvchi2_dqn/operators.py",
    "envs": "src/rvchi2_dqn/envs.py",
    "replay": "src/rvchi2_dqn/replay.py",
    "networks": "src/rvchi2_dqn/networks.py",
    "diagnostics": "src/rvchi2_dqn/diagnostics.py",
    "trainer": "src/rvchi2_dqn/trainer.py",
    "runner": "scripts/run_rvchi2_dqn.py",
}


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete resolved settings for one independently seeded run."""

    task: str = "lqr"
    phase: str = "smoke"
    seed: int = 1
    schema_version: int = SCHEMA_VERSION
    enabled_methods: tuple[str, ...] = METHODS

    nominal_fault_probability: float = 0.10
    chi2_delta: float = 0.25
    gamma: float = 0.99
    certified_probabilities: tuple[float, ...] = (
        0.0,
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
    )
    ood_probabilities: tuple[float, ...] = (0.35, 0.50)

    hidden_dims: tuple[int, ...] = (128, 128)
    replay_capacity: int = 1_000_000
    batch_size: int = 256
    learning_starts: int = 20_000
    nominal_pretrain_steps: int = 100_000
    nominal_updates_per_step: int = 1
    nominal_target_update_interval: int = 1_000
    q_learning_rate: float = 2.5e-4
    pretrain_q_learning_rate: float | None = None
    q_weight_decay: float = 0.0
    q_gradient_clip: float = 10.0

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 80_000
    branch_epsilon_end: float = 0.05
    coverage_rescue_random_fraction: float = 0.0

    outer_blocks: int = 20
    collection_steps_per_block: int = 2_500
    q_updates_per_block: int = 1_000
    auxiliary_updates_per_block: int = 5_000
    auxiliary_learning_rate: float = 1.0e-3
    auxiliary_weight_decay: float = 1.0e-5
    auxiliary_ema_decay: float = 0.99
    affine_initial_u: float = 0.04
    full_initial_u: float = 0.05
    ell: float = 1.0e-4
    eta_bound: float = 15.0
    rho_bound: float = 20.0
    target_lower_bound: float = -15.0
    target_upper_bound: float = 0.0
    target_clip_failure_rate: float = 0.01
    auxiliary_eta_projection_failure_rate: float = 0.01
    auxiliary_u_floor_failure_rate: float = 0.01

    evaluation_episodes: int = 100
    evaluation_horizon: int = 200
    evaluation_every_blocks: int = 5
    heldout_probe_count: int = 2_048
    support_grid_points: int = 41
    nominal_competence_return: float = -1.7
    nominal_competence_failure_rate: float = 0.01
    max_nominal_cost: float = 0.25
    torch_num_threads: int = 1
    deterministic_torch: bool = True
    device: str = "cpu"
    stop_after_nominal_gate: bool = False
    continue_after_failed_pretraining_gate: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def task_defaults(task: str, *, phase: str = "smoke", seed: int = 1) -> ExperimentConfig:
    """Return task-specific physically bounded defaults before profile scaling."""

    task = task.lower()
    if task == "lqr":
        config = ExperimentConfig(task=task, phase=phase, seed=seed)
    elif task == "pendulum":
        config = ExperimentConfig(
            task=task,
            phase=phase,
            seed=seed,
            nominal_pretrain_steps=150_000,
            epsilon_decay_steps=120_000,
            outer_blocks=30,
            collection_steps_per_block=3_000,
            ell=1.0e-3,
            eta_bound=18.0,
            rho_bound=24.0,
            target_lower_bound=-16.4,
            nominal_competence_return=-225.0,
            nominal_competence_failure_rate=1.0,
            max_nominal_cost=100.0,
        )
    else:
        raise ValueError(f"Unknown task {task!r}; expected 'lqr' or 'pendulum'.")
    return apply_phase_profile(config, phase)


def apply_phase_profile(config: ExperimentConfig, phase: str) -> ExperimentConfig:
    """Resolve bounded budgets without changing the scientific task definition."""

    phase = phase.lower()
    if phase == "smoke":
        return replace(
            config,
            phase=phase,
            replay_capacity=5_000,
            batch_size=32,
            learning_starts=64,
            nominal_pretrain_steps=256,
            nominal_target_update_interval=64,
            epsilon_decay_steps=192,
            outer_blocks=2,
            collection_steps_per_block=64,
            q_updates_per_block=32,
            auxiliary_updates_per_block=64,
            evaluation_episodes=8,
            evaluation_horizon=30,
            evaluation_every_blocks=1,
            heldout_probe_count=64,
            support_grid_points=11,
        )
    if phase in {"development", "reporting"}:
        return replace(config, phase=phase)
    raise ValueError("phase must be smoke, development, or reporting.")


def validate_config(config: ExperimentConfig) -> None:
    if config.schema_version != SCHEMA_VERSION:
        raise ValueError("Unsupported RVChi2-DQN schema version.")
    if config.task not in {"lqr", "pendulum"}:
        raise ValueError("task must be lqr or pendulum.")
    if (
        not config.enabled_methods
        or config.enabled_methods[0] != "nominal"
        or len(set(config.enabled_methods)) != len(config.enabled_methods)
        or any(method not in METHODS for method in config.enabled_methods)
    ):
        raise ValueError("enabled_methods must be unique known methods beginning with nominal.")
    allowed = {
        "smoke": SMOKE_SEEDS,
        "development": DEVELOPMENT_SEEDS,
        "reporting": REPORTING_SEEDS,
    }
    if config.phase not in allowed or config.seed not in allowed[config.phase]:
        raise ValueError(f"Seed {config.seed} is reserved outside phase {config.phase!r}.")
    if not isinstance(config.stop_after_nominal_gate, bool):
        raise TypeError("stop_after_nominal_gate must be boolean.")
    if not isinstance(config.continue_after_failed_pretraining_gate, bool):
        raise TypeError("continue_after_failed_pretraining_gate must be boolean.")
    if config.stop_after_nominal_gate and config.continue_after_failed_pretraining_gate:
        raise ValueError(
            "stop_after_nominal_gate and continue_after_failed_pretraining_gate "
            "cannot both be enabled."
        )
    if config.phase == "smoke" and config.continue_after_failed_pretraining_gate:
        raise ValueError(
            "continue_after_failed_pretraining_gate is only valid in development "
            "or reporting."
        )
    if not 0.0 < config.nominal_fault_probability < 1.0:
        raise ValueError("nominal_fault_probability must lie in (0,1).")
    if not math.isfinite(config.chi2_delta) or config.chi2_delta < 0.0:
        raise ValueError("chi2_delta must be finite and nonnegative.")
    if not 0.0 < config.gamma < 1.0:
        raise ValueError("gamma must lie in (0,1).")
    if not math.isfinite(config.nominal_competence_return):
        raise ValueError("nominal_competence_return must be finite.")
    if not (
        math.isfinite(config.nominal_competence_failure_rate)
        and 0.0 <= config.nominal_competence_failure_rate <= 1.0
    ):
        raise ValueError(
            "nominal_competence_failure_rate must be finite and lie in [0,1]."
        )
    if not config.hidden_dims or any(width <= 0 for width in config.hidden_dims):
        raise ValueError("hidden_dims must be positive.")
    positive_ints = (
        config.replay_capacity,
        config.batch_size,
        config.nominal_pretrain_steps,
        config.nominal_target_update_interval,
        config.outer_blocks,
        config.collection_steps_per_block,
        config.q_updates_per_block,
        config.auxiliary_updates_per_block,
        config.evaluation_episodes,
        config.evaluation_horizon,
        config.heldout_probe_count,
    )
    if any(value <= 0 for value in positive_ints):
        raise ValueError("Training and evaluation counts must be positive.")
    if config.learning_starts < config.batch_size or config.learning_starts >= config.nominal_pretrain_steps:
        raise ValueError("learning_starts must be at least a batch and precede pretraining end.")
    if config.replay_capacity < config.learning_starts:
        raise ValueError("replay_capacity must cover learning_starts.")
    if config.nominal_updates_per_step < 1:
        raise ValueError("nominal_updates_per_step must be positive.")
    for value in (
        config.q_learning_rate,
        config.q_gradient_clip,
        config.auxiliary_learning_rate,
        config.affine_initial_u,
        config.full_initial_u,
        config.ell,
        config.eta_bound,
        config.rho_bound,
        config.max_nominal_cost,
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("Learning rates, bounds, floors, and clips must be positive.")
    if config.pretrain_q_learning_rate is not None and (
        not math.isfinite(config.pretrain_q_learning_rate)
        or config.pretrain_q_learning_rate <= 0.0
    ):
        raise ValueError("pretrain_q_learning_rate must be positive when provided.")
    if config.affine_initial_u <= config.ell or config.full_initial_u <= config.ell:
        raise ValueError("Initial auxiliary scales must exceed ell.")
    if config.target_lower_bound >= config.target_upper_bound:
        raise ValueError("target bounds must be ordered.")
    for value, name in (
        (config.target_clip_failure_rate, "target_clip_failure_rate"),
        (
            config.auxiliary_eta_projection_failure_rate,
            "auxiliary_eta_projection_failure_rate",
        ),
        (config.auxiliary_u_floor_failure_rate, "auxiliary_u_floor_failure_rate"),
    ):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be finite and lie in [0,1].")
    if not 0.0 <= config.coverage_rescue_random_fraction <= 1.0:
        raise ValueError("coverage rescue fraction must lie in [0,1].")
    if not 0.0 < config.auxiliary_ema_decay < 1.0:
        raise ValueError("EMA decay must lie in (0,1).")
    probabilities = config.certified_probabilities + config.ood_probabilities
    if any(not 0.0 <= probability <= 1.0 for probability in probabilities):
        raise ValueError("Evaluation probabilities must lie in [0,1].")
    if 0.10 not in config.certified_probabilities or 0.25 not in config.certified_probabilities:
        raise ValueError("Certified panel must include p0 and the exact upper boundary.")
