"""Tests for the moving-market backtest helpers."""

import unittest

import numpy as np
import pandas as pd

from punyaa.market_regimes import (
    error_summary,
    family_parameters,
    month_index,
    month_number,
    project_index,
    tail_slope,
    test_period_projection,
    window_masks,
)


def planted_frame(monthly_log_trend=0.01, months=12, rows_per_month=400, seed=5228):
    """Rents whose only calendar effect is a known log-linear trend."""
    rng = np.random.default_rng(seed)
    records = []
    property_effect = rng.normal(0, 0.15, 40)
    for step in range(months):
        for _ in range(rows_per_month):
            prop = int(rng.integers(0, 40))
            area = float(rng.choice([60.0, 75.0, 90.0, 110.0]))
            flat_type = ["3-room", "4-room", "5-room"][int(np.digitize(area, [70, 100]))]
            log_rent = (
                7.2 + monthly_log_trend * step + property_effect[prop]
                + 0.6 * np.log(area / 90.0) + rng.normal(0, 0.02)
            )
            records.append({
                "RENT_APPROVAL_DATE": f"2022-{step + 1:02d}", "TOWN": "Bishan",
                "BLOCK": str(prop), "STREET": "Bishan St 1", "FLAT_TYPE": flat_type,
                "FLAT_MODEL": "model a", "FLOOR_AREA_SQM": area,
                "MONTHLY_RENT": float(np.exp(log_rent)),
            })
    return pd.DataFrame(records)


class MonthNumberTests(unittest.TestCase):
    def test_counts_months_from_the_first_labelled_year(self):
        self.assertEqual(
            list(month_number(["2021-01", "2021-12", "2022-03"])), [0, 11, 14]
        )


class MonthIndexTests(unittest.TestCase):
    def test_recovers_a_planted_monthly_trend(self):
        frame = planted_frame(0.01)
        anchor = [int(value) for value in sorted(set(month_number(frame.RENT_APPROVAL_DATE)))[-3:]]
        index = month_index(frame, anchor)
        steps = np.diff(index.sort_index().to_numpy())
        self.assertAlmostEqual(float(steps.mean()), 0.01 * 1.0, delta=0.002)
        self.assertLess(float(np.abs(steps - 0.01).max()), 0.006)

    def test_anchor_months_average_to_zero(self):
        frame = planted_frame(0.01)
        anchor = [int(value) for value in sorted(set(month_number(frame.RENT_APPROVAL_DATE)))[-3:]]
        index = month_index(frame, anchor)
        self.assertAlmostEqual(float(index.loc[anchor].mean()), 0.0, places=10)

    def test_a_flat_market_gives_a_flat_index(self):
        frame = planted_frame(0.0)
        anchor = [int(value) for value in sorted(set(month_number(frame.RENT_APPROVAL_DATE)))[-3:]]
        index = month_index(frame, anchor)
        self.assertLess(float(index.abs().max()), 0.01)

    def test_unknown_anchor_month_is_rejected(self):
        with self.assertRaises(ValueError):
            month_index(planted_frame(months=3, rows_per_month=50), [99])

    def test_non_positive_rent_is_rejected(self):
        frame = planted_frame(months=3, rows_per_month=50)
        frame.loc[0, "MONTHLY_RENT"] = 0.0
        with self.assertRaises(ValueError):
            month_index(frame, [0, 1, 2])


class ProjectionTests(unittest.TestCase):
    def setUp(self):
        # A clean 0.02-per-month trend over months 0..11.
        self.index = pd.Series(0.02 * np.arange(12), index=np.arange(12))

    def test_tail_slope_recovers_the_trend(self):
        self.assertAlmostEqual(tail_slope(self.index), 0.02)

    def test_flat_projection_adds_nothing(self):
        np.testing.assert_array_equal(project_index(self.index, [12, 20], "flat"), [0.0, 0.0])

    def test_linear_projection_grows_with_the_horizon(self):
        np.testing.assert_allclose(project_index(self.index, [12, 14], "linear"), [0.02, 0.06])

    def test_damped_projection_is_bounded_and_below_linear(self):
        months = np.arange(12, 80)
        damped = project_index(self.index, months, "damped")
        linear = project_index(self.index, months, "linear")
        self.assertTrue((np.diff(damped) >= 0).all())
        self.assertLess(float(damped.max()), 0.02 * 0.9 / (1 - 0.9))
        self.assertTrue((damped[1:] < linear[1:]).all())
        self.assertAlmostEqual(float(damped[0]), 0.02 * 0.9)

    def test_months_inside_the_index_get_no_adjustment(self):
        np.testing.assert_array_equal(project_index(self.index, [3, 11], "linear"), [0.0, 0.0])

    def test_a_falling_market_projects_downward(self):
        falling = pd.Series(-0.01 * np.arange(12), index=np.arange(12))
        self.assertTrue((project_index(falling, [13, 15], "damped") < 0).all())

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            project_index(self.index, [12], "quadratic")

    def test_damp_must_be_a_proper_fraction(self):
        for bad in (0.0, 1.0, 1.5):
            with self.assertRaises(ValueError):
                project_index(self.index, [12], "damped", damp=bad)

    def test_slope_needs_two_months(self):
        with self.assertRaises(ValueError):
            tail_slope(pd.Series([0.1], index=[5]))


class TestPeriodProjectionTests(unittest.TestCase):
    def setUp(self):
        self.index, self.table = test_period_projection(planted_frame(0.01), windows=(3, 6), horizons=(1, 6))

    def test_index_is_anchored_on_the_last_three_labelled_months(self):
        self.assertAlmostEqual(float(self.index.iloc[-3:].mean()), 0.0, places=10)

    def test_slope_matches_the_planted_trend(self):
        self.assertAlmostEqual(float(self.table.slope_pct_per_month.mean()), 1.0, delta=0.4)

    def test_adjustment_grows_with_horizon_and_linear_exceeds_damped(self):
        subset = self.table[self.table.slope_window_months == 6]
        damped = subset[subset["mode"] == "damped"].sort_values("horizon_months")
        linear = subset[subset["mode"] == "linear"].sort_values("horizon_months")
        self.assertTrue((damped.adjustment_pct.diff().dropna() > 0).all())
        self.assertTrue((linear.adjustment_pct.to_numpy() >= damped.adjustment_pct.to_numpy()).all())

    def test_sgd_adjustment_is_pct_times_the_recent_rent_level(self):
        frame = planted_frame(0.01)
        numbers = month_number(frame.RENT_APPROVAL_DATE)
        recent = frame.loc[np.isin(numbers, sorted(set(numbers))[-3:]), "MONTHLY_RENT"].mean()
        levels = self.table.adjustment_sgd / self.table.adjustment_pct * 100.0
        np.testing.assert_allclose(levels.to_numpy(), recent, rtol=1e-9)


class WindowTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.to_datetime(
            pd.Series([f"2021-{m:02d}" for m in range(1, 13)] + [f"2022-{m:02d}" for m in range(1, 7)]),
            format="%Y-%m",
        )

    def test_fit_and_validation_rows_never_overlap(self):
        fit, validation = window_masks(self.dates, "2022-01", "2022-03")
        self.assertFalse((fit & validation).any())
        self.assertEqual(int(fit.sum()), 12)
        self.assertEqual(int(validation.sum()), 3)

    def test_validation_stays_inside_the_window(self):
        _, validation = window_masks(self.dates, "2021-05", "2021-07")
        self.assertEqual(list(self.dates[validation].dt.month), [5, 6, 7])

    def test_window_without_fitting_rows_is_rejected(self):
        with self.assertRaises(ValueError):
            window_masks(self.dates, "2021-01", "2021-03")

    def test_window_beyond_the_data_is_rejected(self):
        with self.assertRaises(ValueError):
            window_masks(self.dates, "2023-01", "2023-03")

    def test_backwards_window_is_rejected(self):
        with self.assertRaises(ValueError):
            window_masks(self.dates, "2022-03", "2022-01")


class ErrorSummaryTests(unittest.TestCase):
    def test_reports_signed_bias_by_edge_months(self):
        months = np.array(["a", "b", "c", "d", "e", "f"])
        actual = np.zeros(6)
        prediction = np.array([1.0, 1.0, 1.0, -2.0, -2.0, -2.0])
        summary = error_summary(prediction, actual, months)
        self.assertAlmostEqual(summary["bias_first3"], 1.0)
        self.assertAlmostEqual(summary["bias_last3"], -2.0)
        self.assertAlmostEqual(summary["bias"], -0.5)
        self.assertAlmostEqual(summary["rmse"], float(np.sqrt((3 * 1 + 3 * 4) / 6)))

    def test_non_finite_predictions_are_rejected(self):
        with self.assertRaises(ValueError):
            error_summary(np.array([np.nan, 1.0]), np.array([1.0, 1.0]), np.array(["a", "b"]))


class FamilyTests(unittest.TestCase):
    def test_unknown_family_is_rejected(self):
        with self.assertRaises(ValueError):
            family_parameters("xgboost", 2)

    def test_thread_count_is_passed_through(self):
        self.assertEqual(family_parameters("lightgbm", 3)["n_jobs"], 3)
        self.assertEqual(family_parameters("catboost", 3)["thread_count"], 3)


if __name__ == "__main__":
    unittest.main()
