from __future__ import annotations

import unittest

import numpy as np

from punyaa.modeling import (
    lightgbm_variant_parameters,
    make_submission,
    optimize_blend_weights,
)


class BlendTests(unittest.TestCase):
    def test_weights_are_bounded_and_sum_to_one(self) -> None:
        target = np.array([1.0, 2.0, 3.0, 4.0])
        predictions = np.column_stack(
            [target, target + 10, target + 20, np.full_like(target, 20.0)]
        )
        weights = optimize_blend_weights(target, predictions)
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertTrue((weights >= -1e-9).all())
        self.assertTrue((weights <= 0.85 + 1e-9).all())
        self.assertEqual(int(np.argmax(weights)), 0)

    def test_submission_schema(self) -> None:
        submission = make_submission(np.array([100.0, 2500.5, 10000.0]))
        self.assertEqual(submission.columns.tolist(), ["Id", "Predicted"])
        self.assertEqual(submission["Id"].tolist(), [0, 1, 2])
        self.assertEqual(submission["Predicted"].tolist(), [300.0, 2500.5, 9000.0])

    def test_lightgbm_variants_have_distinct_capacities(self) -> None:
        variants = lightgbm_variant_parameters(quick=True)
        self.assertEqual(set(variants), {"regularized", "balanced", "detailed"})
        self.assertLess(
            variants["regularized"]["num_leaves"],
            variants["detailed"]["num_leaves"],
        )


if __name__ == "__main__":
    unittest.main()
