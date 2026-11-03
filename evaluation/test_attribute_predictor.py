"""Synthetic-only checks; never train on real data or download model weights."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn

from evaluation.attribute_predictor import (ATTRIBUTES, CLASS_COUNTS, NORMALIZATION,
                                          SCHEMA, AttributePredictor, load_predictor,
                                          state_dict_sha256)
from evaluation.train_attribute_predictor import (AttributeDataset, atomic_checkpoint,
                                                classification_report, load_checked_splits,
                                                training_statistics)


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1)
        self.fc = nn.Linear(4, 1000)

    def forward(self, images):
        return self.fc(self.conv(images).mean((2, 3)))


class PredictorTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.addCleanup(self.directory.cleanup)

    def frame(self, split):
        return pd.DataFrame({"empi_anon": [f"{split}-a", f"{split}-b"],
                             "image_path": [f"{split}-a.png", f"{split}-b.png"],
                             "view": [0, 1], "density": [0, 3], "scanner": [0, 4],
                             "age": [40., 60.], "cview": [0, 1]})

    def write_splits(self, changes=None):
        for split in ("train", "valid", "test"):
            frame = self.frame(split)
            if changes:
                changes(split, frame)
            frame.to_csv(self.root / f"{split}.csv", index=False)

    def report(self):
        classes = {}
        for key, count in CLASS_COUNTS.items():
            matrix = np.zeros((count, count), dtype=np.int64)
            matrix[0, 0] = 2
            classes[key] = classification_report(matrix)
        age = {"mae_years": 1., "rmse_years": 2.}
        return {"n": 2, **classes, "age": age,
                "baselines": {"categorical_majority": copy.deepcopy(classes), "train_mean_age": age.copy()}}

    def metadata(self):
        return {"format_version": 1, "schema": SCHEMA, "backbone": "resnet18",
                "resolution": [32, 32], "input_range": [0, 1], "input_normalization": NORMALIZATION,
                "age_mean_years": 50., "age_std_years": 10.,
                "split_csv_sha256": {key: "0" * 64 for key in ("train", "valid", "test")},
                "source_sha256": {"test.py": "1" * 64}, "status": "complete",
                "debug_run": False, "validated_for_effectiveness": True,
                "validation_metrics": self.report(), "test_metrics": self.report()}

    def test_split_patient_and_image_overlap_all_pairs(self):
        for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
            for column in ("empi_anon", "image_path"):
                with self.subTest(left=left, right=right, column=column):
                    def change(split, frame):
                        if split in (left, right):
                            frame.loc[0, column] = "shared"
                    self.write_splits(change)
                    with self.assertRaisesRegex(ValueError, "overlap"):
                        load_checked_splits(self.root, self.root)

    def test_fractional_categorical_label_is_rejected(self):
        self.write_splits(lambda split, frame: frame.__setitem__("view", [.5, 1]) if split == "train" else None)
        with self.assertRaisesRegex(ValueError, "integer labels"):
            load_checked_splits(self.root, self.root)

    def test_age_outside_saved_image_schema_is_rejected(self):
        self.write_splits(lambda split, frame: frame.__setitem__("age", [50., 101.]) if split == "test" else None)
        with self.assertRaisesRegex(ValueError, "ages in years"):
            load_checked_splits(self.root, self.root)

    def test_training_statistics_use_supplied_training_rows(self):
        counts, weights, mean, std = training_statistics(self.frame("train"))
        self.assertEqual((mean, std), (50., 10.))
        self.assertEqual(counts["density"], [1, 0, 0, 1])
        self.assertEqual(weights["view"], [1., 1.])

    def test_metric_imbalance_is_visible(self):
        report = classification_report([[9, 0], [1, 0]])
        self.assertAlmostEqual(report["accuracy"], .9)
        self.assertAlmostEqual(report["balanced_accuracy"], .5)
        self.assertAlmostEqual(report["macro_f1"], 9 / 19)

    def test_dataset_never_replaces_failed_image(self):
        dataset = AttributeDataset(self.frame("train"), self.root, 32, 32)
        with patch("evaluation.train_attribute_predictor.get_sample", side_effect=FileNotFoundError("synthetic")) as read:
            with self.assertRaisesRegex(RuntimeError, "no replacement"):
                dataset[0]
            self.assertEqual(read.call_count, 1)

    def test_real_resnet18_forward_contract_without_pretrained_weights(self):
        torch.set_num_threads(1)
        model = AttributePredictor("resnet18", "none", (32, 32), 50., 10.).eval()
        nn.init.zeros_(model.heads["age"].weight)
        nn.init.zeros_(model.heads["age"].bias)
        with torch.inference_mode():
            result = model(torch.rand(2, 1, 32, 32))
        self.assertEqual(tuple(result), ATTRIBUTES)
        for key, count in CLASS_COUNTS.items():
            self.assertEqual(tuple(result[key].shape), (2, count))
        self.assertTrue(torch.equal(result["age"], torch.full((2,), 50.)))
        with self.assertRaisesRegex(ValueError, r"\[0,1\]"):
            model(torch.full((2, 1, 32, 32), 2.))

    def test_loader_requires_test_provenance_and_never_downloads(self):
        calls = []
        def constructor(*, weights):
            calls.append(weights)
            return TinyEncoder()
        with patch("evaluation.attribute_predictor.models.resnet18", side_effect=constructor):
            model = AttributePredictor("resnet18", "none", (32, 32), 50., 10.)
            path = self.root / "predictor.pt"
            atomic_checkpoint(path, model, self.metadata())
            loaded, metadata = load_predictor(path)
            self.assertFalse(loaded.training)
            self.assertFalse(any(parameter.requires_grad for parameter in loaded.parameters()))
            self.assertEqual(len(metadata["checkpoint_sha256"]), 64)
            self.assertEqual(metadata["state_dict_sha256"], state_dict_sha256(loaded.state_dict()))
            self.assertTrue(all(weight is None for weight in calls))
            for changes in ({"test_metrics": None}, {"debug_run": True}, {"status": "training"}):
                bad = dict(self.metadata(), **changes)
                atomic_checkpoint(path, model, bad)
                with self.assertRaises(ValueError):
                    load_predictor(path)
            good = atomic_checkpoint(path, model, self.metadata())
            checkpoint = torch.load(path, weights_only=True)
            checkpoint["state_dict"]["age_mean"] += 1
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                load_predictor(path)


if __name__ == "__main__":
    unittest.main()
