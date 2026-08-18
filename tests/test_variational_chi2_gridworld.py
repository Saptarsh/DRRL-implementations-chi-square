"""Regression tests for the multi-decision variational MiniCliff trainer."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import train_variational_chi2_gridworld as trainer  # noqa: E402
import variational_tabular_envs as envs  # noqa: E402


class ExactReferenceAndPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.environment = envs.MiniCliffConfig()
        cls.config = trainer.MiniCliffAlgorithmConfig(
            # The largest paper-sweep floor is the parameter-ball endpoint
            # most likely to expose an unintended class-truncation bias.
            ell=1.0,
            dp_tolerance=1e-10,
            dp_max_iterations=10_000,
        )
        cls.mdp = envs.make_minicliff_mdp(cls.environment)
        cls.nominal, cls.robust, cls.floor = trainer.exact_references(
            cls.mdp, cls.config
        )

    def test_exact_references_and_noninitial_policy_separation(self) -> None:
        nominal_residual = np.max(
            np.abs(self.nominal - trainer.nominal_bellman(self.nominal, self.mdp))
        )
        robust_residual = np.max(
            np.abs(
                self.robust
                - trainer.robust_bellman(
                    self.robust, self.mdp, self.config.chi2_delta
                )
            )
        )
        floor_residual = np.max(
            np.abs(
                self.floor
                - trainer.floor_bellman(
                    self.floor,
                    self.mdp,
                    self.config.chi2_delta,
                    self.config.ell,
                )
            )
        )
        self.assertLess(float(nominal_residual), 2e-9)
        self.assertLess(float(robust_residual), 2e-9)
        self.assertLess(float(floor_residual), 2e-9)

        nominal_policy = trainer.greedy_policy(self.nominal)
        robust_policy = trainer.greedy_policy(self.robust)
        differing_states = [
            state
            for state in range(self.mdp.n_states)
            if state not in self.mdp.marker_states
            and nominal_policy[state] != robust_policy[state]
        ]
        self.assertEqual(differing_states, [12, 13, 14])
        self.assertEqual(
            [self.mdp.coordinate_of(state) for state in differing_states],
            [(2, 0), (2, 1), (2, 2)],
        )
        self.assertNotIn(self.mdp.start_state, differing_states)

    def test_largest_sweep_floor_oracle_fits_parameter_balls(self) -> None:
        floor_values = np.max(self.floor, axis=1)
        eta = []
        rho = []
        for state in range(self.mdp.n_states):
            for action in range(self.mdp.n_actions):
                _, eta_star, u_star = trainer.floor_variational_solution(
                    floor_values,
                    self.mdp.transitions[state, action],
                    self.config.chi2_delta,
                    self.config.ell,
                )
                eta.append(eta_star)
                rho.append(max(0.0, u_star - self.config.ell))
        eta_norm = float(np.linalg.norm(eta))
        rho_norm = float(np.linalg.norm(rho))
        eta_radius, scale_radius = trainer.automatic_parameter_radii(
            self.mdp, self.config
        )

        self.assertGreater(eta_norm, 10.0)  # guards the original regression
        self.assertLessEqual(eta_norm, eta_radius)
        self.assertLessEqual(rho_norm, scale_radius)

    def test_robust_oracle_improves_under_an_in_ball_slip_perturbation(self) -> None:
        nominal_policy = trainer.greedy_policy(self.nominal)
        robust_policy = trainer.greedy_policy(self.robust)
        test_mdp = envs.make_minicliff_mdp(
            self.environment,
            slip_probability=self.config.evaluation_slip_probability,
        )
        report = envs.rowwise_chi2_divergence(
            self.mdp.transitions, test_mdp.transitions
        )
        self.assertTrue(report.support_preserved)
        self.assertTrue(report.within_radius(self.config.chi2_delta))
        np.testing.assert_array_equal(test_mdp.rewards, self.mdp.rewards)

        robust_return = trainer.exact_policy_value(test_mdp, robust_policy)[
            test_mdp.start_state
        ]
        nominal_return = trainer.exact_policy_value(test_mdp, nominal_policy)[
            test_mdp.start_state
        ]
        optimal_return, _ = trainer.exact_test_optimum(test_mdp, self.config)
        self.assertGreater(float(robust_return), float(nominal_return) + 0.09)
        self.assertLess(float(optimal_return - robust_return), 0.005)


class TrainerConfigurationAndSmokeTests(unittest.TestCase):
    def test_invalid_algorithm_settings_are_rejected(self) -> None:
        invalid = (
            trainer.MiniCliffAlgorithmConfig(ell=0.0),
            trainer.MiniCliffAlgorithmConfig(beta0=0.0),
            trainer.MiniCliffAlgorithmConfig(h_q=1.0),
            trainer.MiniCliffAlgorithmConfig(stage1_stepsize=float("nan")),
            trainer.MiniCliffAlgorithmConfig(beta0=float("nan")),
            trainer.MiniCliffAlgorithmConfig(eta_l2_radius=-1.0),
            trainer.MiniCliffAlgorithmConfig(scale_l2_radius=float("inf")),
            trainer.MiniCliffAlgorithmConfig(nominal_lr_scale=0.0),
            trainer.MiniCliffAlgorithmConfig(perturbation_grid=()),
            trainer.MiniCliffAlgorithmConfig(perturbation_grid=(0.1, 0.1)),
        )
        for config in invalid:
            with self.subTest(config=config), self.assertRaises(ValueError):
                trainer.validate_algorithm_config(config)

    def test_tiny_run_uses_one_cursor_and_saves_finite_diagnostics(self) -> None:
        environment = envs.MiniCliffConfig()
        config = trainer.MiniCliffAlgorithmConfig(
            seed=9,
            ell=0.1,
            outer_blocks=2,
            stage1_samples=40,
            q_stage_samples=30,
            stage1_stepsize=0.005,
            evaluation_slip_probability=0.15,
            perturbation_grid=(0.1, 0.15),
            dp_tolerance=1e-9,
            dp_max_iterations=5_000,
        )
        result = trainer.run_experiment(environment, config)
        expected = config.outer_blocks * (
            config.stage1_samples + config.q_stage_samples
        )

        self.assertEqual(result.metadata["total_transitions"], expected)
        self.assertEqual(int(result.arrays["state_action_counts"].sum()), expected)
        self.assertEqual(
            [int(row["transitions"]) for row in result.metrics],
            [0, config.stage1_samples + config.q_stage_samples, expected],
        )
        self.assertEqual(result.metadata["oracle_policy_difference_count"], 3)
        self.assertGreater(result.metadata["q_gain_p"], 1.0)
        self.assertTrue(result.metadata["clean_q_rate_condition_satisfied"])
        self.assertEqual(result.arrays["decision_state_mask"].sum(), 19)
        self.assertEqual(result.arrays["oracle_separating_state_mask"].sum(), 3)
        for name, array in result.arrays.items():
            self.assertTrue(np.all(np.isfinite(array)), f"non-finite entries in {name}")
        for row in result.metrics + result.perturbation_metrics:
            for name, value in row.items():
                self.assertTrue(math.isfinite(float(value)), f"non-finite metric {name}")


if __name__ == "__main__":
    unittest.main()
