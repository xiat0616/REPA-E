"""Progress logging checks using synthetic metadata and forward-only inference."""
import ast
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from evaluation import train_attribute_predictor as trainer


class FlushSpy(io.StringIO):
    def __init__(self):
        super().__init__()
        self.flush_count = 0

    def flush(self):
        self.flush_count += 1
        super().flush()


class SyntheticRows(torch.utils.data.Dataset):
    def __len__(self):
        return 3

    def __getitem__(self, index):
        return torch.full((1, 32, 32), index / 2), {
            "view": torch.tensor(index % 2), "density": torch.tensor(index % 4),
            "scanner": torch.tensor(index % 5), "age": torch.tensor(40. + index)}


class ForwardOnlyPredictor(nn.Module):
    def forward(self, images):
        assert not torch.is_grad_enabled(), "Evaluation must remain forward-only"
        size = len(images)
        return {**{key: torch.zeros(size, count) for key, count in trainer.CLASS_COUNTS.items()},
                "age": torch.full((size,), 41.)}


class PredictorLoggingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.addCleanup(self.directory.cleanup)
        self.events = []

    def capture(self, event, **fields):
        self.events.append({"event": event, **fields})

    def write_splits(self):
        for split in ("train", "valid", "test"):
            pd.DataFrame({"empi_anon": [f"{split}-a", f"{split}-b"],
                          "image_path": [f"{split}-a.png", f"{split}-b.png"],
                          "view": [0, 1], "density": [0, 3], "scanner": [0, 4],
                          "age": [40., 60.], "cview": [0, 1]}).to_csv(self.root / f"{split}.csv", index=False)

    def test_startup_is_flushed_before_heavy_imports(self):
        source_path = Path(trainer.__file__)
        source = source_path.read_text()
        parsed = ast.parse(source)
        heavy_line = next(node.lineno for node in parsed.body
                          if isinstance(node, ast.Import) and any(alias.name in ("numpy", "pandas", "torch") for alias in node.names))
        # Execute only the stdlib prelude. NumPy/Torch and training never run.
        prelude = "\n".join(source.splitlines()[:heavy_line - 1])
        output = FlushSpy()
        with redirect_stdout(output):
            exec(compile(prelude, str(source_path), "exec"), {"__name__": "__main__", "__file__": str(source_path)})
        message = output.getvalue().strip()
        self.assertRegex(message, r"^\[\d{2}:\d{2}:\d{2} UTC\] Starting attribute predictor")
        self.assertNotIn('"event"', message)
        self.assertGreater(output.flush_count, 0)

    def test_config_and_device_are_visible_before_split_error(self):
        args = trainer.parser().parse_args(["--device", "cpu", "--cpu-threads", "1",
                                            "--output-dir", str(self.root / "output")])
        output = FlushSpy()
        with redirect_stdout(output), \
             patch.object(trainer, "load_checked_splits", side_effect=RuntimeError("synthetic split failure")) as load, \
             patch.object(trainer, "AttributePredictor") as model, \
             patch.object(trainer.torch.optim, "AdamW") as optimizer:
            with self.assertRaisesRegex(RuntimeError, "synthetic split failure"):
                trainer.train(args)
        messages = output.getvalue().splitlines()
        self.assertGreaterEqual(len(messages), 2)
        self.assertIn("Training ", messages[0])
        self.assertIn("ResNet50", messages[0])
        self.assertIn("Device: cpu", messages[-1])
        for message in messages:
            self.assertRegex(message, r"^\[\d{2}:\d{2}:\d{2} UTC\] ")
            self.assertNotIn('"event"', message)
        self.assertIs(load.call_args.kwargs["progress"], trainer.log_progress)
        self.assertFalse(model.called)
        self.assertFalse(optimizer.called)
        self.assertFalse((args.output_dir / "run.json").exists())

    def test_split_stages_preserve_checks_and_are_silent_by_default(self):
        self.write_splits()
        output = io.StringIO()
        with redirect_stdout(output):
            silent_frames, silent_hashes = trainer.load_checked_splits(self.root, self.root)
        self.assertEqual(output.getvalue(), "")
        frames, hashes = trainer.load_checked_splits(self.root, self.root, progress=self.capture)
        self.assertEqual(hashes, silent_hashes)
        self.assertEqual({key: len(frame) for key, frame in frames.items()}, {key: len(frame) for key, frame in silent_frames.items()})
        for split in ("train", "valid", "test"):
            events = [record["event"] for record in self.events if record.get("split") == split]
            for event in ("split_csv_hash_start", "split_csv_hash_complete", "split_csv_read_start",
                          "split_csv_read_complete", "split_labels_check_start", "split_labels_check_complete",
                          "path_resolution_start", "path_resolution_progress", "path_resolution_complete"):
                self.assertIn(event, events)
            progress = next(record for record in self.events if record["event"] == "path_resolution_progress" and record["split"] == split)
            self.assertEqual((progress["paths"], progress["total_paths"]), (2, 2))
        self.assertEqual(self.events[0]["event"], "split_checks_start")
        self.assertEqual(self.events[-1]["event"], "split_checks_complete")
        # Progress must not weaken patient or canonical-image overlap checks.
        frame = pd.read_csv(self.root / "valid.csv")
        frame.loc[0, "image_path"] = "subdir/../train-a.png"
        frame.to_csv(self.root / "valid.csv", index=False)
        with self.assertRaisesRegex(ValueError, "Image overlap"):
            trainer.load_checked_splits(self.root, self.root, progress=self.capture)

    def test_batch_reporting_first_interval_and_last_without_training(self):
        with patch.object(trainer.time, "monotonic", side_effect=[0., 1., 2., 32., 33.]):
            reporter = trainer.BatchProgress(self.capture, "train_progress", 30., epoch=2, epochs=20)
            reporter.update(1, 4, 4, 16, loss=1.2, mean_loss=1.)
            reporter.update(2, 4, 8, 16, loss=.8, mean_loss=.9)
            reporter.update(3, 4, 12, 16, loss=.6, mean_loss=.8)
            reporter.update(4, 4, 16, 16, loss=.4, mean_loss=.7)
        self.assertEqual([record["batch"] for record in self.events], [1, 3, 4])
        self.assertEqual(self.events[-1]["images"], 16)
        self.assertEqual(self.events[-1]["epoch"], 2)
        self.assertEqual(self.events[-1]["epochs"], 20)
        self.assertEqual(self.events[-1]["loss"], .4)
        self.assertEqual(self.events[-1]["mean_loss"], .7)
        self.assertAlmostEqual(self.events[-1]["images_per_second"], 16 / 33)
        self.assertAlmostEqual(self.events[0]["eta_seconds"], 3.)
        self.assertAlmostEqual(self.events[-1]["eta_seconds"], 0.)

    def test_training_progress_prints_current_loss_epoch_and_eta(self):
        output = FlushSpy()
        with redirect_stdout(output):
            trainer.log_progress("train_progress", epoch=2, epochs=20, batch=120, batches=3715,
                                 loss=1.2345, mean_loss=1.4567, images=7680, total_images=237746,
                                 elapsed_seconds=300., images_per_second=25.6, eta_seconds=1800.)
        message = output.getvalue().strip()
        self.assertRegex(message, r"^\[\d{2}:\d{2}:\d{2} UTC\] ")
        for text in ("Epoch 2/20", "Batch 120/3715", "loss=1.2345", "mean_loss=1.4567", "ETA 00:30:00"):
            self.assertIn(text, message)
        self.assertNotIn("{", message)
        self.assertNotIn("}", message)
        self.assertNotIn('"event"', message)
        self.assertGreater(output.flush_count, 0)

    def test_unknown_progress_total_has_no_numeric_eta(self):
        with patch.object(trainer.time, "monotonic", side_effect=[0., 2.]):
            reporter = trainer.BatchProgress(self.capture, "train_progress", 30., epoch=1, epochs=20)
            reporter.update(1, None, 4, None, loss=1.2, mean_loss=1.2)
        self.assertIsNone(self.events[-1]["eta_seconds"])

    def test_validation_prints_compact_metrics_without_raw_reports(self):
        metrics = {"n": 10,
                   **{key: {"accuracy": .8, "balanced_accuracy": .75, "macro_f1": .7,
                            "confusion_matrix": [[1]]} for key in trainer.CLASS_COUNTS},
                   "age": {"mae_years": 4.5, "rmse_years": 6.25},
                   "baselines": {"test_marker": "raw_baseline_report"}}
        output = FlushSpy()
        with redirect_stdout(output):
            trainer.log_progress("validation_complete", images=10, elapsed_seconds=12., metrics=metrics)
        message = output.getvalue()
        self.assertIn("Validation", message)
        for label in ("view", "density", "scanner", "age"):
            self.assertIn(label, message.lower())
        self.assertRegex(message, r"4\.5(?:0+)?")
        self.assertLessEqual(len(message.splitlines()), 2)
        self.assertNotIn("confusion_matrix", message)
        self.assertNotIn("raw_baseline_report", message)
        self.assertNotIn("{", message)
        self.assertNotIn('"event"', message)

    def test_evaluation_reports_progress_without_changing_metrics(self):
        torch.set_num_threads(1)
        loader = DataLoader(SyntheticRows(), batch_size=2, num_workers=0)
        model = ForwardOnlyPredictor()
        counts = {key: [2] + [1] * (count - 1) for key, count in trainer.CLASS_COUNTS.items()}
        expected = trainer.evaluate(model, loader, "cpu", counts, 50.)
        observed = trainer.evaluate(model, loader, "cpu", counts, 50., progress=self.capture,
                                    label="validation", log_interval_seconds=30.)
        self.assertEqual(observed, expected)
        self.assertEqual([record["event"] for record in self.events],
                         ["validation_start", "validation_progress", "validation_progress", "validation_complete"])
        self.assertEqual(self.events[-1]["metrics"], expected)
        self.assertEqual(self.events[-2]["images"], 3)

    def test_invalid_logging_intervals_fail_before_model_or_optimizer(self):
        for interval in ("0", "-1", "nan", "inf"):
            args = trainer.parser().parse_args(["--device", "cpu", "--output-dir", str(self.root / "output"),
                                                "--log-interval-seconds", interval])
            with self.subTest(interval=interval), patch.object(trainer, "AttributePredictor") as model:
                with self.assertRaisesRegex(ValueError, "positive and finite"):
                    trainer.train(args)
                self.assertFalse(model.called)


if __name__ == "__main__":
    unittest.main()
