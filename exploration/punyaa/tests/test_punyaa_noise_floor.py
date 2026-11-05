"""Tests for the irreducible-noise estimate."""

import unittest

import numpy as np
import pandas as pd

from punyaa.noise_floor import FEATURE_KEY, noise_estimate, noise_interval, twin_mask


def make_frame(rents_by_group):
    rows = []
    for index, rents in enumerate(rents_by_group):
        for rent in rents:
            rows.append({
                "RENT_APPROVAL_DATE": "2024-05", "TOWN": "Bishan", "BLOCK": str(index),
                "STREET": "Bishan St 1", "FLAT_TYPE": "4-room", "FLAT_MODEL": "model a",
                "FLOOR_AREA_SQM": 90.0, "LEASE_COMMENCE_DATE": 1990, "MONTHLY_RENT": rent,
            })
    return pd.DataFrame(rows)


class NoiseEstimateTests(unittest.TestCase):
    def test_singleton_rows_are_excluded(self):
        frame = make_frame([[2000.0, 2200.0], [3000.0], [2500.0]])
        self.assertEqual(noise_estimate(frame)["rows_with_twin"], 2)
        self.assertEqual(list(twin_mask(frame)), [True, True, False, False])

    def test_unbiased_estimate_matches_the_pooled_sample_variance(self):
        frame = make_frame([[2000.0, 2200.0], [3000.0, 3400.0]])
        # Group variances (ddof=1) are 20000 and 80000; both groups have one degree of freedom.
        self.assertAlmostEqual(noise_estimate(frame)["unbiased_noise_sd"], np.sqrt(50000.0))

    def test_strict_bound_is_smaller_for_pairs_by_exactly_root_two(self):
        frame = make_frame([[2000.0, 2200.0], [3000.0, 3400.0]])
        estimate = noise_estimate(frame)
        self.assertAlmostEqual(
            estimate["realized_lower_bound"], estimate["unbiased_noise_sd"] / np.sqrt(2)
        )

    def test_recovers_a_known_noise_level_from_many_groups(self):
        rng = np.random.default_rng(5228)
        frame = make_frame([list(3000 + rng.normal(0, 450, 3)) for _ in range(3000)])
        self.assertAlmostEqual(noise_estimate(frame)["unbiased_noise_sd"], 450.0, delta=8.0)

    def test_text_variants_of_the_same_property_are_one_group(self):
        frame = make_frame([[2000.0, 2200.0]])
        frame.loc[1, "TOWN"] = "  BISHAN "
        frame.loc[1, "FLAT_TYPE"] = "4_room"
        self.assertEqual(noise_estimate(frame)["groups"], 1)

    def test_differing_floor_area_separates_rows(self):
        frame = make_frame([[2000.0, 2200.0]])
        frame.loc[1, "FLOOR_AREA_SQM"] = 91.0
        with self.assertRaises(ValueError):
            noise_estimate(frame)

    def test_interval_brackets_the_estimate_and_the_true_level(self):
        rng = np.random.default_rng(7)
        frame = make_frame([list(3000 + rng.normal(0, 450, 3)) for _ in range(800)])
        interval = noise_interval(frame, draws=500)
        point = noise_estimate(frame)["unbiased_noise_sd"]
        self.assertLess(interval["lower_95"], point)
        self.assertGreater(interval["upper_95"], point)
        self.assertLess(interval["lower_95"], 450.0)
        self.assertGreater(interval["upper_95"], 450.0)

    def test_interval_is_reproducible_for_a_seed(self):
        frame = make_frame([[2000.0, 2200.0, 2100.0], [3000.0, 3400.0], [2500.0, 2600.0]])
        self.assertEqual(noise_interval(frame, draws=200), noise_interval(frame, draws=200))

    def test_interval_rejects_too_few_draws(self):
        with self.assertRaises(ValueError):
            noise_interval(make_frame([[2000.0, 2200.0]]), draws=10)

    def test_missing_column_is_reported(self):
        with self.assertRaises(ValueError):
            noise_estimate(make_frame([[2000.0, 2200.0]]).drop(columns=["BLOCK"]))

    def test_key_covers_every_non_constant_raw_input(self):
        self.assertEqual(
            set(FEATURE_KEY),
            {"RENT_APPROVAL_DATE", "TOWN", "BLOCK", "STREET", "FLAT_TYPE", "FLAT_MODEL",
             "FLOOR_AREA_SQM", "LEASE_COMMENCE_DATE"},
        )


if __name__ == "__main__":
    unittest.main()
