"""Tests for the extrapolation and recalibration helpers."""

import unittest

import numpy as np
import pandas as pd

from punyaa.extrapolation import (
    MONOTONE_CORE,
    best_spread_scale,
    calibration_summary,
    cross_window_transfer,
    forward_transfer,
    horizon_diagnostics,
    monotone_constraints,
)
from punyaa.features import CATEGORICAL_FEATURES


COLUMNS = [
    "TOWN", "FLAT_TYPE", "FLOOR_AREA_SQM", "approval_date_num",
    "monthly_target_lag", "town_flat_type_history_mean",
    "property_flat_type_history_mean", "mrt_dist_nearest_km",
]


class MonotoneConstraintTests(unittest.TestCase):
    def test_vector_matches_column_order_and_length(self):
        vector = monotone_constraints(COLUMNS, MONOTONE_CORE)
        self.assertEqual(len(vector), len(COLUMNS))
        self.assertEqual(vector[COLUMNS.index("FLOOR_AREA_SQM")], 1)
        self.assertEqual(vector[COLUMNS.index("approval_date_num")], 1)

    def test_categorical_features_are_never_constrained(self):
        directions = {column: 1 for column in COLUMNS}
        vector = monotone_constraints(COLUMNS, directions)
        for column, value in zip(COLUMNS, vector):
            if column in CATEGORICAL_FEATURES:
                self.assertEqual(value, 0, column)

    def test_unconstrained_columns_default_to_zero(self):
        vector = monotone_constraints(COLUMNS, MONOTONE_CORE)
        self.assertEqual(vector[COLUMNS.index("mrt_dist_nearest_km")], 0)

    def test_unknown_constrained_column_is_rejected(self):
        with self.assertRaises(ValueError):
            monotone_constraints(COLUMNS, {"NOT_A_COLUMN": 1})

    def test_optional_benchmark_column_may_be_absent(self):
        monotone_constraints(COLUMNS, {"median_rent_benchmark": 1})


class CalibrationTests(unittest.TestCase):
    def test_perfect_prediction_needs_no_correction(self):
        actual = np.array([1000.0, 2000.0, 3000.0, 4000.0])
        summary = calibration_summary(actual, actual)
        self.assertAlmostEqual(summary["rmse"], 0.0)
        self.assertAlmostEqual(summary["affine_slope"], 1.0, places=6)
        self.assertAlmostEqual(summary["bias_prediction_minus_actual"], 0.0)

    def test_level_shift_removes_a_constant_offset(self):
        actual = np.array([1000.0, 2000.0, 3000.0, 4000.0])
        summary = calibration_summary(actual, actual + 50.0)
        self.assertAlmostEqual(summary["bias_prediction_minus_actual"], 50.0)
        self.assertAlmostEqual(summary["rmse_after_level_shift"], 0.0, places=6)

    def test_recalibration_never_increases_in_sample_error(self):
        rng = np.random.default_rng(5228)
        actual = rng.normal(3000, 600, 400)
        prediction = 0.8 * actual + 500 + rng.normal(0, 200, 400)
        summary = calibration_summary(actual, prediction)
        self.assertLessEqual(summary["rmse_after_affine"], summary["rmse"] + 1e-9)
        self.assertLessEqual(summary["rmse_after_level_shift"], summary["rmse"] + 1e-9)

    def test_mismatched_vectors_are_rejected(self):
        with self.assertRaises(ValueError):
            calibration_summary(np.zeros(3), np.zeros(4))


class ForwardTransferTests(unittest.TestCase):
    def setUp(self):
        self.months = pd.Series(["2024-05"] * 5 + ["2024-06"] * 5 + ["2024-07"] * 5)
        self.actual = np.linspace(2000, 4000, 15)

    def test_a_stable_offset_transfers_to_the_later_months(self):
        transfer = forward_transfer(self.months, self.actual, self.actual + 100.0)
        self.assertAlmostEqual(transfer["fitted_bias"], 100.0)
        self.assertGreater(transfer["level_shift_gain"], 0.0)
        self.assertEqual(transfer["applied_months"], ["2024-07"])

    def test_an_offset_that_only_exists_early_does_not_transfer(self):
        prediction = self.actual.copy()
        prediction[:10] += 300.0
        transfer = forward_transfer(self.months, self.actual, prediction)
        self.assertLess(transfer["level_shift_gain"], 0.0)

    def test_a_single_month_window_is_rejected(self):
        with self.assertRaises(ValueError):
            forward_transfer(pd.Series(["2024-05"] * 4), self.actual[:4], self.actual[:4])


class CrossWindowTransferTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(5228)
        self.actual = 3000 + rng.normal(0, 600, 4000)
        self.unbiased = self.actual + rng.normal(0, 300, 4000)

    def test_a_correction_fitted_on_a_biased_window_hurts_an_unbiased_one(self):
        biased = self.unbiased + 100.0
        result = cross_window_transfer(self.actual, biased, self.actual, self.unbiased)
        self.assertAlmostEqual(result["source_bias"], 100.0, delta=15.0)
        self.assertGreater(result["level_shift_change"], 0.0)

    def test_the_same_window_shows_the_in_sample_gain(self):
        biased = self.unbiased + 100.0
        result = cross_window_transfer(self.actual, biased, self.actual, biased)
        # RMSE falls from sqrt(300^2 + 100^2) to about 300 once the bias is removed.
        self.assertLess(result["level_shift_change"], -10.0)

    def test_mismatched_vectors_are_rejected(self):
        with self.assertRaises(ValueError):
            cross_window_transfer(self.actual[:10], self.unbiased[:9], self.actual, self.unbiased)

    def test_empty_window_is_rejected(self):
        with self.assertRaises(ValueError):
            cross_window_transfer([], [], self.actual, self.unbiased)


class SpreadScaleTests(unittest.TestCase):
    def test_recovers_a_planted_under_dispersion(self):
        rng = np.random.default_rng(7)
        actual = rng.normal(3000, 600, 5000)
        prediction = 3000 + 0.9 * (actual - 3000)
        result = best_spread_scale(actual, prediction)
        self.assertAlmostEqual(result["best_scale"], 1.1, delta=0.011)
        self.assertLess(result["change"], 0.0)

    def test_well_calibrated_predictions_keep_scale_one(self):
        rng = np.random.default_rng(7)
        actual = rng.normal(3000, 600, 5000)
        result = best_spread_scale(actual, actual)
        self.assertAlmostEqual(result["best_scale"], 1.0, places=6)
        self.assertAlmostEqual(result["change"], 0.0, places=6)

    def test_mismatched_vectors_are_rejected(self):
        with self.assertRaises(ValueError):
            best_spread_scale([1.0, 2.0], [1.0])


class HorizonDiagnosticTests(unittest.TestCase):
    def test_one_row_per_month_ordered_by_horizon(self):
        months = pd.Series(["2024-07", "2024-05", "2024-06", "2024-05"])
        actual = np.array([3000.0, 2000.0, 2500.0, 2100.0])
        table = horizon_diagnostics(months, actual, {"model": actual + 10.0})
        self.assertEqual(list(table["month"]), ["2024-05", "2024-06", "2024-07"])
        self.assertEqual(list(table["horizon"]), [1, 2, 3])
        self.assertEqual(list(table["rows"]), [2, 1, 1])
        self.assertTrue(np.allclose(table["model_bias"], 10.0))


if __name__ == "__main__":
    unittest.main()
