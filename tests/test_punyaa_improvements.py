from __future__ import annotations

import unittest

import numpy as np

from punyaa.improve import bounded_blend, recency_weights


class ImprovementTests(unittest.TestCase):
    def test_half_life_uses_calendar_age_not_row_order(self):
        months = np.array([24, 12, 24, 0])
        weights = recency_weights(months, 12)
        self.assertAlmostEqual(weights.mean(), 1.0)
        self.assertAlmostEqual(weights[0] / weights[1], 2.0)
        self.assertAlmostEqual(weights[1] / weights[3], 2.0)
        self.assertEqual(weights[0], weights[2])

    def test_blend_rejects_directionally_worse_model(self):
        target = np.array([1.0, 2.0, 3.0])
        baseline = target + 1
        weight, prediction = bounded_blend(target, baseline, target + 2)
        self.assertEqual(weight, 0)
        np.testing.assert_array_equal(prediction, baseline)

    def test_blend_is_bounded_and_improves_opposing_errors(self):
        target = np.array([1.0, 2.0, 3.0])
        weight, prediction = bounded_blend(target, target - 1, target + 1)
        self.assertEqual(weight, 0.5)
        np.testing.assert_allclose(prediction, target)

    def test_blend_handles_identical_predictions(self):
        target = np.array([1.0, 2.0])
        weight, prediction = bounded_blend(target, target, target)
        self.assertEqual(weight, 0)
        np.testing.assert_array_equal(prediction, target)

    def test_invalid_recency_inputs_are_rejected(self):
        for months, half_life in [([], 12), ([1, 2], 0), ([1, np.nan], 12)]:
            with self.assertRaises(ValueError):
                recency_weights(np.array(months), half_life)


if __name__ == "__main__":
    unittest.main()
