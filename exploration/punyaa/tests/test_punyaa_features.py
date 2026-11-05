from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from punyaa.features import (
    HISTORY_FEATURES,
    CausalHistoryTransformer,
    StaticFeatureBuilder,
    normalize_category,
)


def make_frame(months: list[int], towns: list[str] | None = None) -> pd.DataFrame:
    if towns is None:
        towns = ["alpha"] * len(months)
    return pd.DataFrame(
        {
            "MONTH_INDEX": months,
            "TOWN": towns,
            "BLOCK": ["1"] * len(months),
            "STREET": ["main road"] * len(months),
            "FLAT_TYPE": ["4 room"] * len(months),
        }
    )


class CategoryCleaningTests(unittest.TestCase):
    def test_equivalent_spellings_are_normalized(self) -> None:
        values = pd.Series([" 4-Room ", "4_room", "4/room"])
        self.assertEqual(normalize_category(values).tolist(), ["4 room"] * 3)


class CausalHistoryTests(unittest.TestCase):
    def test_same_month_targets_are_excluded(self) -> None:
        frame = make_frame([1, 1, 2])
        target = pd.Series([1000.0, 5000.0, 3000.0])
        result = CausalHistoryTransformer(lag_months=1).fit_transform_train(frame, target)

        self.assertEqual(result.loc[0, "property_flat_type_history_count"], 0)
        self.assertEqual(result.loc[1, "property_flat_type_history_count"], 0)
        self.assertEqual(result.loc[2, "property_flat_type_history_count"], 2)

    def test_future_target_change_does_not_affect_past_rows(self) -> None:
        frame = make_frame([1, 2, 3, 4])
        first = CausalHistoryTransformer(1).fit_transform_train(
            frame, pd.Series([1000.0, 2000.0, 3000.0, 4000.0])
        )
        second = CausalHistoryTransformer(1).fit_transform_train(
            frame, pd.Series([1000.0, 2000.0, 3000.0, 9000.0])
        )
        pd.testing.assert_frame_equal(first.iloc[:3], second.iloc[:3])

    def test_reference_transform_excludes_overlapping_month(self) -> None:
        reference = make_frame([1, 2])
        transformer = CausalHistoryTransformer(1).fit(
            reference, pd.Series([1000.0, 9000.0])
        )
        result = transformer.transform(make_frame([2]))

        self.assertEqual(result.loc[0, "property_flat_type_history_count"], 1)
        self.assertAlmostEqual(result.loc[0, "monthly_target_lag"], 1000.0)

    def test_unseen_group_falls_back_to_prior_global_mean(self) -> None:
        reference = make_frame([1, 2], ["known", "known"])
        transformer = CausalHistoryTransformer(1).fit(
            reference, pd.Series([2000.0, 4000.0])
        )
        result = transformer.transform(make_frame([3], ["unseen"]))

        self.assertEqual(result.loc[0, "town_flat_type_history_count"], 0)
        self.assertAlmostEqual(result.loc[0, "town_flat_type_history_mean"], 3000.0)

    def test_expected_history_schema(self) -> None:
        result = CausalHistoryTransformer(1).fit_transform_train(
            make_frame([1, 2]), pd.Series([1000.0, 1200.0])
        )
        self.assertEqual(result.columns.tolist(), HISTORY_FEATURES)
        self.assertEqual(len(result), 2)
        self.assertTrue(np.isnan(result.loc[0, "monthly_target_lag"]))

    def test_recent_windows_use_calendar_month_boundaries(self) -> None:
        frame = make_frame(list(range(1, 31)))
        target = pd.Series(np.arange(1000.0, 4000.0, 100.0))
        result = CausalHistoryTransformer(24).fit_transform_train(frame, target)

        self.assertEqual(result.loc[29, "property_flat_type_recent_6_count"], 6)
        self.assertEqual(result.loc[29, "property_flat_type_recent_12_count"], 12)
        self.assertEqual(result.loc[29, "property_flat_type_recent_24_count"], 24)

    def test_recent_six_month_window_excludes_older_target(self) -> None:
        frame = make_frame(list(range(1, 31)))
        original = pd.Series(np.arange(1000.0, 4000.0, 100.0))
        changed = original.copy()
        changed.iloc[0] = 9000.0
        first = CausalHistoryTransformer(24).fit_transform_train(frame, original)
        second = CausalHistoryTransformer(24).fit_transform_train(frame, changed)

        self.assertAlmostEqual(
            first.loc[29, "property_flat_type_recent_6_mean"],
            second.loc[29, "property_flat_type_recent_6_mean"],
        )

    def test_optional_median_rent_preparation(self) -> None:
        raw = pd.DataFrame(
            {
                "quarter": ["2026-Q1", "2026-Q2"],
                "town": ["ANG MO KIO", "ANG MO KIO"],
                "flat_type": ["4-RM", "4-RM"],
                "median_rent": [3000, 3040],
            }
        )
        prepared = StaticFeatureBuilder._prepare_median_rent(raw)
        q3 = prepared.loc[prepared["quarter_key"].eq("2026-Q3")]

        self.assertEqual(len(q3), 1)
        self.assertAlmostEqual(float(q3.iloc[0]["median_rent_benchmark"]), 3080.0)


if __name__ == "__main__":
    unittest.main()
