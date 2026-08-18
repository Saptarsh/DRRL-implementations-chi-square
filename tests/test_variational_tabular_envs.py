"""Focused tests for the theory-aligned continuing MiniCliff task."""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import variational_tabular_envs as envs  # noqa: E402


class MiniCliffConstructionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = envs.MiniCliffConfig()
        self.mdp = envs.make_minicliff_mdp(self.config)

    def test_layout_rewards_and_marker_resets(self) -> None:
        self.assertEqual(self.mdp.transitions.shape, (24, 4, 24))
        self.assertEqual(self.mdp.rewards.shape, (24, 4))
        self.assertEqual(self.mdp.start_state, 18)
        self.assertEqual(self.mdp.cliff_states, (19, 20, 21, 22))
        self.assertEqual(self.mdp.goal_state, 23)
        self.assertEqual(self.mdp.coordinate_of(self.mdp.start_state), (3, 0))
        self.assertEqual(self.mdp.state_at(3, 5), self.mdp.goal_state)
        np.testing.assert_array_equal(
            self.mdp.state_coordinates,
            np.array([(r, c) for r in range(4) for c in range(6)]),
        )

        ordinary_states = sorted(set(range(24)) - set(self.mdp.marker_states))
        np.testing.assert_allclose(self.mdp.rewards[ordinary_states], envs.STEP_REWARD)
        np.testing.assert_allclose(self.mdp.rewards[list(self.mdp.cliff_states)], -1.0)
        np.testing.assert_allclose(self.mdp.rewards[self.mdp.goal_state], 1.0)

        expected_reset = np.zeros(24)
        expected_reset[self.mdp.start_state] = 1.0
        for marker in self.mdp.marker_states:
            for action in range(4):
                np.testing.assert_allclose(
                    self.mdp.transitions[marker, action], expected_reset, rtol=0.0, atol=1e-15
                )

    def test_nominal_slip_uses_intended_plus_three_equal_alternatives(self) -> None:
        start = self.mdp.start_state
        start_up = self.mdp.transitions[start, envs.UP]

        self.assertAlmostEqual(start_up[self.mdp.state_at(2, 0)], 0.9, places=14)
        self.assertAlmostEqual(start_up[self.mdp.state_at(3, 1)], 0.1 / 3.0, places=14)
        # Down and left both hit a wall and aggregate at start.
        self.assertAlmostEqual(start_up[start], 2.0 * 0.1 / 3.0, places=14)
        self.assertAlmostEqual(float(start_up.sum()), 1.0, places=14)

    def test_slip_perturbation_changes_only_the_kernel(self) -> None:
        perturbed = envs.make_minicliff_mdp(self.config, slip_probability=0.2)

        np.testing.assert_array_equal(perturbed.rewards, self.mdp.rewards)
        self.assertFalse(np.array_equal(perturbed.transitions, self.mdp.transitions))
        self.assertEqual(perturbed.start_state, self.mdp.start_state)
        self.assertEqual(perturbed.marker_states, self.mdp.marker_states)


class BehaviorAndStationarityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mdp = envs.make_minicliff_mdp()
        self.behavior = envs.make_minicliff_behavior_policy(self.mdp)

    def test_behavior_is_the_required_full_support_mixture(self) -> None:
        preferred = envs.goal_directed_actions(self.mdp)
        self.assertEqual(int(preferred[self.mdp.start_state]), envs.UP)
        self.assertEqual(int(preferred[self.mdp.state_at(2, 3)]), envs.RIGHT)
        self.assertEqual(int(preferred[self.mdp.state_at(2, 5)]), envs.DOWN)

        np.testing.assert_allclose(self.behavior.sum(axis=1), 1.0, rtol=0.0, atol=1e-15)
        self.assertAlmostEqual(float(self.behavior.min()), 0.2, places=14)
        for state, action in enumerate(preferred):
            self.assertAlmostEqual(self.behavior[state, action], 0.4, places=14)
            alternatives = np.delete(self.behavior[state], action)
            np.testing.assert_allclose(alternatives, 0.2, rtol=0.0, atol=1e-15)

    def test_exact_stationary_state_and_state_action_distributions(self) -> None:
        stationary = envs.exact_stationary_distributions(self.mdp, self.behavior)

        self.assertAlmostEqual(float(stationary.state_probabilities.sum()), 1.0, places=14)
        self.assertAlmostEqual(float(stationary.state_action_probabilities.sum()), 1.0, places=14)
        self.assertGreater(float(stationary.state_probabilities.min()), 0.0)
        self.assertGreater(stationary.minimum_state_action_probability, 0.0)
        np.testing.assert_allclose(
            stationary.state_action_probabilities,
            stationary.state_probabilities[:, None] * self.behavior,
            rtol=0.0,
            atol=2e-15,
        )
        self.assertLess(stationary.state_residual, 5e-14)
        self.assertLess(stationary.state_action_residual, 5e-14)
        self.assertLess(stationary.residual, 5e-14)

    def test_trajectory_cursor_persists_and_marker_reward_precedes_reset(self) -> None:
        rng = np.random.default_rng(17)
        cliff = self.mdp.cliff_states[0]
        trajectory = envs.PersistentTabularTrajectory(
            self.mdp,
            self.behavior,
            rng,
            initial_state=cliff,
        )

        state, action, reward, next_state = trajectory.next()
        self.assertEqual(state, cliff)
        self.assertEqual(reward, -1.0)
        self.assertEqual(next_state, self.mdp.start_state)
        self.assertEqual(trajectory.state, self.mdp.start_state)
        self.assertEqual(trajectory.transitions_read, 1)

        previous_next = next_state
        for _ in range(199):
            state, action, reward, next_state = trajectory.step()
            self.assertEqual(state, previous_next)
            self.assertEqual(reward, self.mdp.rewards[state, action])
            self.assertGreater(self.mdp.transitions[state, action, next_state], 0.0)
            previous_next = next_state
        self.assertEqual(trajectory.transitions_read, 200)
        self.assertEqual(int(trajectory.state_action_counts.sum()), 200)


class KernelComparisonAndValidationTests(unittest.TestCase):
    def test_rowwise_chi_square_report_checks_radius_and_support(self) -> None:
        nominal = envs.make_minicliff_mdp()
        perturbed = envs.make_minicliff_mdp(slip_probability=0.2)
        same_report = envs.rowwise_chi_square_divergence(
            nominal.transitions, nominal.transitions
        )
        np.testing.assert_array_equal(same_report.row_divergences, 0.0)
        self.assertTrue(same_report.support_preserved)
        self.assertTrue(same_report.within_radius(0.0))

        report = envs.rowwise_chi2_divergence(
            nominal.transitions, perturbed.transitions
        )
        self.assertTrue(report.support_preserved)
        self.assertTrue(np.all(np.isfinite(report.row_divergences)))
        self.assertAlmostEqual(report.maximum_divergence, 1.0 / 9.0, places=13)
        self.assertFalse(report.within_radius(0.05))
        self.assertTrue(report.within_radius(0.2))

        unsupported = nominal.transitions.copy()
        marker = nominal.cliff_states[0]
        unsupported[marker, envs.UP, nominal.start_state] = 0.9
        unsupported[marker, envs.UP, 0] = 0.1
        unsupported_report = envs.rowwise_chi_square_divergence(
            nominal.transitions, unsupported
        )
        self.assertFalse(unsupported_report.support_preserved)
        self.assertTrue(unsupported_report.support_violations[marker, envs.UP])
        self.assertTrue(math.isinf(unsupported_report.row_divergences[marker, envs.UP]))
        self.assertFalse(unsupported_report.within_radius(1e6))

    def test_invalid_configs_policies_and_mdps_are_rejected(self) -> None:
        invalid_configs = (
            envs.MiniCliffConfig(gamma=1.0),
            envs.MiniCliffConfig(nominal_slip_probability=0.0),
            envs.MiniCliffConfig(behavior_goal_bias=1.0),
        )
        for config in invalid_configs:
            with self.subTest(config=config), self.assertRaises(ValueError):
                envs.validate_minicliff_config(config)
        with self.assertRaises(ValueError):
            envs.make_minicliff_mdp(slip_probability=-0.1)

        mdp = envs.make_minicliff_mdp()
        invalid_policy = np.full((mdp.n_states, mdp.n_actions), 0.25)
        invalid_policy[0] = (1.0, 0.0, 0.0, 0.0)
        with self.assertRaises(ValueError):
            envs.validate_behavior_policy(mdp, invalid_policy, require_full_support=True)

        invalid_transitions = mdp.transitions.copy()
        invalid_transitions[0, 0] = 0.0
        invalid_mdp = replace(mdp, transitions=invalid_transitions)
        with self.assertRaises(ValueError):
            envs.validate_minicliff_mdp(invalid_mdp)


if __name__ == "__main__":
    unittest.main()
