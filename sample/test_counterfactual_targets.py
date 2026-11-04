"""Synthetic tests for paired uniform intervention targets; no models or GPU."""
import csv
from pathlib import Path
import random
import sys
import tempfile
import unittest

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sample.counterfactual_targets import load_training_age_range, make_counterfactual_parents


def parents(size):
    return {
        "view": F.one_hot(torch.arange(size) % 2, 2).float(),
        "density": F.one_hot(torch.arange(size) % 4, 4).float(),
        "scanner": F.one_hot(torch.arange(size) % 5, 5).float(),
        "age": torch.full((size, 1), .55),
    }


class UniformTargetTests(unittest.TestCase):
    def test_targets_are_uniform_for_every_source_and_include_source_class(self):
        count = 15000
        original = parents(count)
        for attribute, classes in [("view", 2), ("density", 4), ("scanner", 5)]:
            with self.subTest(attribute=attribute):
                result = make_counterfactual_parents(original, attribute, indices=range(count))
                source = original[attribute].argmax(1)
                target = result[attribute].argmax(1)
                self.assertTrue(torch.equal(result[attribute].sum(1), torch.ones(count)))
                for category in range(classes):
                    selected = target[source == category]
                    frequency = torch.bincount(selected, minlength=classes).float() / selected.numel()
                    self.assertLess(float((frequency - 1. / classes).abs().max()), .035)
                    self.assertGreater(int((selected == category).sum()), 0)

    def test_draws_are_independent_of_batch_order_global_rng_and_attribute_order(self):
        original = parents(17)
        indices = [27, 15, 804, 2, 90, 555, 11, 1, 19, 93, 716, 38, 71, 39, 87, 144, 7]
        for attribute in ["view", "density", "scanner", "age"]:
            with self.subTest(attribute=attribute):
                kwargs = dict(seed=39, age_range_years=(32., 84.))
                whole = make_counterfactual_parents(original, attribute, indices=indices, **kwargs)[attribute]
                pieces = []
                for start, end in [(0, 4), (4, 13), (13, 17)]:
                    batch = {key: value[start:end] for key, value in original.items()}
                    pieces.append(make_counterfactual_parents(batch, attribute,
                                  indices=indices[start:end], **kwargs)[attribute])
                self.assertTrue(torch.equal(whole, torch.cat(pieces)))
                permutation = torch.tensor([8, 1, 7, 4, 9, 10, 2, 12, 3, 16, 0, 15, 6, 13, 14, 11, 5])
                shuffled = {key: value[permutation] for key, value in original.items()}
                actual = make_counterfactual_parents(shuffled, attribute,
                         indices=[indices[index] for index in permutation.tolist()], **kwargs)[attribute]
                self.assertTrue(torch.equal(actual, whole[permutation]))
                python_before, torch_before = random.getstate(), torch.random.get_rng_state()
                make_counterfactual_parents(original, attribute, indices=indices, **kwargs)
                self.assertEqual(python_before, random.getstate())
                self.assertTrue(torch.equal(torch_before, torch.random.get_rng_state()))
                random.random()
                torch.rand(31)
                for other in reversed(["view", "density", "scanner", "age"]):
                    make_counterfactual_parents(original, other, indices=indices, **kwargs)
                repeated = make_counterfactual_parents(original, attribute, indices=indices, **kwargs)[attribute]
                self.assertTrue(torch.equal(whole, repeated))
                alternate = make_counterfactual_parents(original, attribute, indices=indices,
                            seed=40, age_range_years=(32., 84.))[attribute]
                self.assertFalse(torch.equal(whole, alternate))

    def test_conditions_are_cloned_and_only_requested_attribute_changes(self):
        original = parents(20)
        snapshot = {key: value.clone() for key, value in original.items()}
        for attribute in original:
            result = make_counterfactual_parents(original, attribute, age_range_years=(30., 90.))
            for key in original:
                self.assertTrue(torch.equal(original[key], snapshot[key]))
                self.assertNotEqual(result[key].data_ptr(), original[key].data_ptr())
                if key != attribute:
                    self.assertTrue(torch.equal(result[key], snapshot[key]))

    def test_age_uses_explicit_training_bounds_not_fixed_ten_year_edit(self):
        original = parents(500)
        result = make_counterfactual_parents(original, "age", indices=range(500), age_range_years=(37., 79.))["age"] * 100
        self.assertGreaterEqual(float(result.min()), 37.)
        self.assertLessEqual(float(result.max()), 79.)
        self.assertLess(float(result.min()), 40.)
        self.assertGreater(float(result.max()), 76.)
        self.assertGreater(int(torch.unique(result).numel()), 490)
        self.assertFalse(torch.allclose(result, torch.full_like(result, 65.)))
        fixed = make_counterfactual_parents(original, "age", age_range_years=(63., 63.))["age"]
        self.assertTrue(torch.equal(fixed, torch.full_like(fixed, .63)))

    def test_invalid_inputs_and_missing_age_bounds_fail(self):
        original = parents(2)
        for bounds in [None, (20,), "20,80", (80, 20), (-1, 80), (20, 101),
                       (float("nan"), 80), (20, float("inf")), (True, 80), ("bad", 80)]:
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                make_counterfactual_parents(original, "age", age_range_years=bounds)
        for indices in [[0], [-1, 2], [0., 1.], [True, 1], 3]:
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                make_counterfactual_parents(original, "view", indices=indices)
        with self.assertRaises(ValueError):
            make_counterfactual_parents(original, "unknown")
        with self.assertRaises(ValueError):
            make_counterfactual_parents(original, "view", seed=1.5)


class TrainingAgeRangeTests(unittest.TestCase):
    @staticmethod
    def write_csv(path, values, column="age"):
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([column])
            writer.writerows([[value] for value in values])

    def test_range_uses_training_only_and_is_cached_by_resolved_training_path(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.write_csv(root / "train.csv", [42, 67.5, 38, 91])
            self.write_csv(root / "test.csv", [0, 100])
            self.write_csv(root / "valid.csv", [1, 99])
            self.assertEqual(load_training_age_range(root), (38., 91.))
            self.write_csv(root / "train.csv", [20, 80])
            self.assertEqual(load_training_age_range(root / ".." / root.name), (38., 91.))

    def test_invalid_or_empty_training_ages_fail(self):
        for values, column in [([], "age"), ([40], "other"), ([""], "age"),
                               (["bad"], "age"), (["NaN"], "age"), (["inf"], "age"),
                               ([-1], "age"), ([101], "age")]:
            with self.subTest(values=values, column=column), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                self.write_csv(root / "train.csv", values, column)
                with self.assertRaises(ValueError):
                    load_training_age_range(root)


if __name__ == "__main__":
    unittest.main()
