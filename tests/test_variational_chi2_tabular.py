"""Focused regression tests for the variational chi-square tabular experiment."""

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

import train_variational_chi2_tabular as tabular  # noqa: E402


class ChiSquareInnerProblemTests(unittest.TestCase):
    def test_delta_zero_returns_the_nominal_distribution_and_expectation(self) -> None:
        values = np.array([-2.0, 0.5, 3.0], dtype=np.float64)
        probabilities = np.array([0.2, 0.5, 0.3], dtype=np.float64)

        worst_case = tabular.chi2_worst_case_distribution(values, probabilities, delta=0.0)

        np.testing.assert_allclose(worst_case, probabilities, rtol=0.0, atol=1e-14)
        self.assertAlmostEqual(
            tabular.chi2_robust_expectation(values, probabilities, delta=0.0),
            float(probabilities @ values),
            places=14,
        )

    def test_large_radius_concentrates_on_the_supported_minimizer(self) -> None:
        values = np.array([-10.0, -2.0, 1.0, 3.0], dtype=np.float64)
        # The lowest value is outside the nominal support and must remain
        # unreachable under q << p.  Concentrating on value -2 costs
        # 1 / 0.2 - 1 = 4 units of chi-square divergence.
        probabilities = np.array([0.0, 0.2, 0.5, 0.3], dtype=np.float64)

        worst_case = tabular.chi2_worst_case_distribution(values, probabilities, delta=5.0)

        np.testing.assert_allclose(worst_case, [0.0, 1.0, 0.0, 0.0], rtol=0.0, atol=1e-13)
        self.assertAlmostEqual(
            tabular.chi2_robust_expectation(values, probabilities, delta=5.0),
            -2.0,
            places=13,
        )

    def test_active_constraint_matches_the_analytic_two_point_solution(self) -> None:
        values = np.array([0.0, 1.0], dtype=np.float64)
        probabilities = np.array([0.5, 0.5], dtype=np.float64)
        delta = 0.25

        worst_case = tabular.chi2_worst_case_distribution(values, probabilities, delta)
        divergence = float(np.sum((worst_case - probabilities) ** 2 / probabilities))

        # min q_2 subject to 4(q_2 - 1/2)^2 <= 1/4 gives q=(3/4,1/4).
        np.testing.assert_allclose(worst_case, [0.75, 0.25], rtol=0.0, atol=2e-13)
        self.assertAlmostEqual(float(worst_case.sum()), 1.0, places=14)
        self.assertGreaterEqual(float(worst_case.min()), 0.0)
        self.assertAlmostEqual(divergence, delta, places=12)
        self.assertAlmostEqual(
            tabular.chi2_robust_expectation(values, probabilities, delta),
            0.25,
            places=12,
        )


class ConfigurationValidationTests(unittest.TestCase):
    def test_rejects_invalid_optimizer_and_evaluation_settings(self) -> None:
        env = tabular.CorridorConfig()
        invalid_configs = (
            tabular.AlgorithmConfig(eta_l2_radius=-1.0),
            tabular.AlgorithmConfig(scale_l2_radius=-1.0),
            tabular.AlgorithmConfig(nominal_lr_scale=0.0),
            tabular.AlgorithmConfig(dp_tolerance=0.0),
            tabular.AlgorithmConfig(dp_max_iterations=0),
            tabular.AlgorithmConfig(perturbation_grid=()),
            tabular.AlgorithmConfig(perturbation_grid=(0.1, 0.2, 0.1)),
        )
        for config in invalid_configs:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    tabular.validate_configs(env, config)


class ReferenceAndPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.env = tabular.CorridorConfig()
        cls.cfg = tabular.AlgorithmConfig(
            chi2_delta=0.5,
            ell=0.05,
            dp_tolerance=1e-11,
            dp_max_iterations=5_000,
        )
        cls.mdp = tabular.make_corridor_mdp(cls.env)
        (
            cls.nominal_reference,
            cls.robust_reference,
            cls.floor_reference,
        ) = tabular.exact_references(cls.env, cls.cfg, cls.mdp)

    def test_floor_reference_is_downward_biased_within_the_predicted_bound(self) -> None:
        c_delta = math.sqrt(1.0 + self.cfg.chi2_delta)
        scalar_values = np.full(3, 2.0, dtype=np.float64)
        scalar_probabilities = np.array([0.2, 0.5, 0.3], dtype=np.float64)
        exact_scalar = tabular.chi2_robust_expectation(
            scalar_values, scalar_probabilities, self.cfg.chi2_delta
        )
        floor_scalar = tabular.floor_robust_expectation(
            scalar_values, scalar_probabilities, self.cfg.chi2_delta, self.cfg.ell
        )
        self.assertLess(floor_scalar, exact_scalar)
        self.assertLessEqual(exact_scalar - floor_scalar, 0.5 * c_delta * self.cfg.ell + 1e-13)

        self.assertTrue(np.all(self.floor_reference <= self.robust_reference + 2e-10))
        reference_bias = float(np.max(np.abs(self.robust_reference - self.floor_reference)))
        fixed_point_bound = (
            self.env.gamma
            * c_delta
            * self.cfg.ell
            / (2.0 * (1.0 - self.env.gamma))
        )
        self.assertGreater(reference_bias, 0.0)
        self.assertLessEqual(reference_bias, fixed_point_bound + 2e-9)

        robust_residual = float(
            np.max(
                np.abs(
                    self.robust_reference
                    - tabular.robust_bellman(
                        self.robust_reference,
                        self.mdp,
                        self.env.gamma,
                        self.cfg.chi2_delta,
                    )
                )
            )
        )
        floor_residual = float(
            np.max(
                np.abs(
                    self.floor_reference
                    - tabular.floor_robust_bellman(
                        self.floor_reference,
                        self.mdp,
                        self.env.gamma,
                        self.cfg.chi2_delta,
                        self.cfg.ell,
                    )
                )
            )
        )
        self.assertLess(robust_residual, 2e-10)
        self.assertLess(floor_residual, 2e-10)

    def test_nominal_and_robust_oracles_separate_under_supported_stress(self) -> None:
        nominal_policy = tabular.greedy_policy(self.nominal_reference)
        robust_policy = tabular.greedy_policy(self.robust_reference)
        untrained_policy = tabular.greedy_policy(np.zeros_like(self.robust_reference))

        behavior = tabular.behavior_action_probabilities(self.mdp, self.env)
        behavior_kernel = np.einsum(
            "sa,sat->st", behavior, self.mdp.transitions, optimize=True
        )
        self.assertTrue(
            np.all(np.linalg.matrix_power(behavior_kernel, 50) > 0.0),
            "The default continuing behavior chain should be irreducible and aperiodic.",
        )

        self.assertEqual(
            int(nominal_policy[self.mdp.start_state]),
            0,
            "The nominal oracle should choose the short risky route.",
        )
        self.assertEqual(
            int(robust_policy[self.mdp.start_state]),
            1,
            "The robust oracle should choose the longer safe route.",
        )
        self.assertEqual(
            int(untrained_policy[self.mdp.start_state]),
            0,
            "Stable tie breaking should initially choose the risky route.",
        )

        stressed_crash_probability = 0.20
        stress_divergence = (
            (stressed_crash_probability - self.env.nominal_crash_prob) ** 2
            / (
                self.env.nominal_crash_prob
                * (1.0 - self.env.nominal_crash_prob)
            )
        )
        self.assertLessEqual(stress_divergence, self.cfg.chi2_delta)
        stressed_mdp = tabular.make_corridor_mdp(self.env, stressed_crash_probability)
        nominal_return = float(
            tabular.exact_policy_value(stressed_mdp, nominal_policy, self.env.gamma)[
                stressed_mdp.start_state
            ]
        )
        robust_return = float(
            tabular.exact_policy_value(stressed_mdp, robust_policy, self.env.gamma)[
                stressed_mdp.start_state
            ]
        )

        self.assertGreater(robust_return, nominal_return + 0.05)


class TinyExperimentSmokeTest(unittest.TestCase):
    def test_tiny_run_counts_every_transition_and_stays_finite(self) -> None:
        env = tabular.CorridorConfig(
            risky_len=1,
            safe_len=2,
            nominal_crash_prob=0.05,
            gamma=0.8,
        )
        cfg = tabular.AlgorithmConfig(
            seed=7,
            chi2_delta=0.5,
            ell=0.05,
            outer_blocks=2,
            stage1_samples=12,
            q_stage_samples=8,
            stage1_stepsize=0.01,
            beta0=2.0,
            h_q=4.0,
            evaluation_crash_prob=0.20,
            perturbation_grid=(0.05, 0.20),
            dp_tolerance=1e-9,
            dp_max_iterations=5_000,
        )

        result = tabular.run_experiment(env, cfg)
        expected_transitions = cfg.outer_blocks * (
            cfg.stage1_samples + cfg.q_stage_samples
        )

        self.assertEqual(result.metadata["total_transitions"], expected_transitions)
        self.assertEqual(int(result.metrics[-1]["transitions"]), expected_transitions)
        self.assertEqual(
            [int(row["transitions"]) for row in result.metrics],
            [0, cfg.stage1_samples + cfg.q_stage_samples, expected_transitions],
        )
        self.assertEqual(int(result.arrays["state_action_counts"].sum()), expected_transitions)

        for name, array in result.arrays.items():
            self.assertTrue(np.all(np.isfinite(array)), f"non-finite entries in {name}")
        for row in result.metrics + result.perturbation_metrics:
            for name, value in row.items():
                self.assertTrue(math.isfinite(float(value)), f"non-finite metric {name}")
        self.assertTrue(np.all(result.arrays["last_u_bar"] >= cfg.ell - 1e-15))


if __name__ == "__main__":
    unittest.main()
