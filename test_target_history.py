"""Tests for leakage-safe historical target features.

Run from the repository root with:
    python -m unittest -v test_target_history.py
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from train_rental_model import (
    HISTORY_SPECS,
    add_causal_target_history,
    add_reference_target_history,
)


GROUP_COLUMNS = [column for column, _ in HISTORY_SPECS]


def make_frame(months: list[int], groups: list[str] | None = None) -> pd.DataFrame:
    """Create the minimal input frame required by the history functions."""
    if groups is None:
        groups = ["group-a"] * len(months)
    data: dict[str, list[object]] = {"MONTH_INDEX": months}
    for column in GROUP_COLUMNS:
        data[column] = groups
    return pd.DataFrame(data)


class CausalHistoryTests(unittest.TestCase):
    def test_same_month_labels_are_excluded(self) -> None:
        frame = make_frame([0, 0, 1])
        target = pd.Series([1000.0, 5000.0, 3000.0])

        result = add_causal_target_history(frame, target)

        # Neither row in the first month can use its own label or the other row's
        # same-month label. The following month can use both earlier observations.
        self.assertEqual(result.loc[0, "HIST_COUNT_PROPERTY"], 0)
        self.assertEqual(result.loc[1, "HIST_COUNT_PROPERTY"], 0)
        self.assertEqual(result.loc[2, "HIST_COUNT_PROPERTY"], 2)
        self.assertAlmostEqual(result.loc[2, "PRIOR_GLOBAL_MEAN"], 3000.0)

    def test_future_targets_do_not_change_past_features(self) -> None:
        frame = make_frame([0, 1, 2, 3])
        original_target = pd.Series([1000.0, 2000.0, 3000.0, 4000.0])
        changed_future = pd.Series([1000.0, 2000.0, 3000.0, 9000.0])

        original = add_causal_target_history(frame, original_target)
        modified = add_causal_target_history(frame, changed_future)
        history_columns = [column for column in original if column not in frame]

        pd.testing.assert_frame_equal(
            original.loc[:2, history_columns],
            modified.loc[:2, history_columns],
        )

    def test_first_month_has_no_target_derived_prior(self) -> None:
        frame = make_frame([0, 1])
        result = add_causal_target_history(frame, pd.Series([1000.0, 9000.0]))

        self.assertTrue(np.isnan(result.loc[0, "PRIOR_GLOBAL_MEAN"]))
        self.assertTrue(np.isnan(result.loc[0, "HIST_MEAN_PROPERTY"]))
        self.assertAlmostEqual(result.loc[1, "PRIOR_GLOBAL_MEAN"], 1000.0)

    def test_row_order_does_not_change_month_based_history(self) -> None:
        frame = make_frame([0, 1, 2, 3])
        frame["row_id"] = [10, 11, 12, 13]
        target = pd.Series([1000.0, 2000.0, 3000.0, 4000.0])
        expected = add_causal_target_history(frame, target).sort_values("row_id")

        order = [2, 0, 3, 1]
        shuffled_frame = frame.iloc[order].reset_index(drop=True)
        shuffled_target = target.iloc[order].reset_index(drop=True)
        actual = add_causal_target_history(
            shuffled_frame, shuffled_target
        ).sort_values("row_id")

        columns = ["row_id", "PRIOR_GLOBAL_MEAN", "HIST_MEAN_PROPERTY", "HIST_COUNT_PROPERTY"]
        pd.testing.assert_frame_equal(
            expected[columns].reset_index(drop=True),
            actual[columns].reset_index(drop=True),
        )

    def test_rolling_count_uses_only_requested_previous_months(self) -> None:
        frame = make_frame(list(range(10)))
        target = pd.Series(np.arange(1000.0, 2000.0, 100.0))
        result = add_causal_target_history(frame, target)

        self.assertEqual(result.loc[9, "RECENT_6_COUNT_PROPERTY_FLAT_TYPE"], 6)
        self.assertEqual(result.loc[9, "RECENT_12_COUNT_PROPERTY_FLAT_TYPE"], 9)

    def test_expected_number_of_history_columns(self) -> None:
        frame = make_frame([0, 1])
        result = add_causal_target_history(frame, pd.Series([1000.0, 1100.0]))

        self.assertEqual(len(result.columns) - len(frame.columns), 29)


class ReferenceHistoryTests(unittest.TestCase):
    def test_unseen_group_uses_reference_global_mean(self) -> None:
        reference = make_frame([0, 1], ["known", "known"])
        target = pd.Series([2000.0, 4000.0])
        future = make_frame([2], ["unseen"])

        result = add_reference_target_history(reference, target, future)

        self.assertEqual(result.loc[0, "HIST_COUNT_PROPERTY"], 0)
        self.assertAlmostEqual(result.loc[0, "HIST_MEAN_PROPERTY"], 3000.0)
        self.assertAlmostEqual(result.loc[0, "PRIOR_GLOBAL_MEAN"], 3000.0)

    def test_future_mapping_uses_reference_labels_only(self) -> None:
        reference = make_frame([0, 1])
        target = pd.Series([2000.0, 4000.0])
        future = make_frame([2])

        result = add_reference_target_history(reference, target, future)
        smoothing = dict(HISTORY_SPECS)["PROPERTY"]
        expected = (6000.0 + smoothing * 3000.0) / (2 + smoothing)

        self.assertAlmostEqual(result.loc[0, "HIST_MEAN_PROPERTY"], expected)
        self.assertEqual(result.loc[0, "HIST_COUNT_PROPERTY"], 2)


if __name__ == "__main__":
    unittest.main()
