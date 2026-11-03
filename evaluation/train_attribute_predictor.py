"""Train an independent EMBED image attribute predictor; user-launched only.

No predictor is trained by installing or importing this module. Example:
python evaluation/train_attribute_predictor.py --data-dir ... --split-dir ... --output-dir ...
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time


def _duration(seconds):
    if seconds is None or not math.isfinite(seconds):
        return "--:--:--"
    hours, remainder = divmod(int(max(0, seconds)), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def log_progress(event, **fields):
    """Print readable status; detailed configuration/metrics stay in JSON files."""
    if event in ("split_csv_hash_start", "split_csv_hash_complete",
                 "split_labels_check_start", "split_labels_check_complete",
                 "split_overlap_check_start"):
        return
    f = fields
    if event == "startup":
        message = f"Starting attribute predictor (PID {f['pid']})"
    elif event == "train_config":
        c = f['config']
        message = (f"Training {c['backbone'].replace('resnet', 'ResNet')} | epochs={c['epochs']} | "
                   f"batch_size={c['batch_size']} | lr={c['learning_rate']:g} | "
                   f"resolution={c['height']}x{c['width']} | output={c['output_dir']}")
    elif event == "device_ready":
        message = f"Device: {f['device']} | workers={f['num_workers']} | CPU threads={f['cpu_threads']}"
    elif event == "split_checks_start":
        message = "Checking training, validation and test datasets..."
    elif event == "split_csv_read_start":
        message = f"Reading {f['split']} metadata..."
    elif event == "split_csv_read_complete":
        message = f"Loaded {f['split']}: {f['rows']:,} images"
    elif event == "path_resolution_start":
        message = f"Checking {f['split']} image paths ({f['total_paths']:,} images)..."
    elif event == "path_resolution_progress":
        done, total, elapsed = f['paths'], f['total_paths'], f['elapsed_seconds']
        eta = max(0, total - done) * elapsed / done if done else None
        message = (f"Checking {f['split']} paths: {done:,}/{total:,} ({done / total:.1%}) | "
                   f"elapsed {_duration(elapsed)} | ETA {_duration(eta)}")
    elif event == "path_resolution_complete":
        message = f"{f['split'].capitalize()} paths checked in {_duration(f['elapsed_seconds'])}"
    elif event == "split_checks_complete":
        sizes = ' | '.join(f"{name}={count:,}" for name, count in f['split_sizes'].items())
        message = f"Dataset checks complete | {sizes}"
    elif event == "run_metadata_saved":
        message = f"Run directory: {f['output_dir']}" + (" | DEBUG run" if f['debug_run'] else "")
    elif event == "model_init_start":
        initialization = 'ImageNet-pretrained ' if f['pretrained'] == 'imagenet' else ''
        message = f"Loading {initialization}{f['backbone']} on {f['device']}..."
    elif event == "model_init_complete":
        message = f"Model ready in {_duration(f['elapsed_seconds'])}"
    elif event == "epoch_start":
        message = f"Epoch {f['epoch']}/{f['epochs']} | {f['batches']:,} batches | {f['total_images']:,} images"
    elif event in ("train_progress", "validation_progress", "test_progress"):
        label = f"Epoch {f['epoch']}/{f['epochs']}" if event == 'train_progress' else event.split('_')[0].capitalize()
        total = f['batches']
        percent = f" ({f['batch'] / total:.1%})" if total else ''
        message = f"{label} | Batch {f['batch']}/{total or '?'}{percent}"
        if event == "train_progress":
            message += f" | loss={f['loss']:.4f} | mean_loss={f['mean_loss']:.4f}"
        message += (f" | elapsed {_duration(f['elapsed_seconds'])} | "
                    f"{'epoch ' if event == 'train_progress' else ''}ETA {_duration(f['eta_seconds'])} | "
                    f"{f['images_per_second']:.1f} img/s")
    elif event in ("validation_start", "test_start"):
        message = f"{event.split('_')[0].capitalize()} started | {f['batches'] or '?'} batches"
    elif event in ("validation_complete", "test_complete"):
        metrics = f['metrics']
        scores = ' | '.join(f"{name} bal_acc={metrics[name]['balanced_accuracy']:.1%}"
                            for name in ('view', 'density', 'scanner'))
        message = (f"{event.split('_')[0].capitalize()} complete | {scores} | "
                   f"age MAE={metrics['age']['mae_years']:.2f} years | elapsed {_duration(f['elapsed_seconds'])}")
    elif event == "checkpoint_saved":
        label = 'best checkpoint' if f['status'] == 'best_validation_update' else 'evaluated checkpoint'
        message = f"Saved {label}: {f['path']} | epoch={f['epoch']} | validation score={f['selection_score']:.4f}"
    elif event == "epoch_complete":
        message = f"Epoch {f['epoch']} complete | mean_loss={f['train_loss']:.4f} | validation score={f['validation_score']:.4f}"
    elif event == "early_stopping":
        message = f"Early stopping: {f['stale_epochs']} epochs without improvement | best epoch={f['selected_epoch']}"
    elif event == "training_complete":
        message = f"Training complete | checkpoint={f['checkpoint']}" + (" | DEBUG run" if f['debug_run'] else "")
    elif event == "training_failed":
        message = f"Training failed: {f['error_type']}: {f['error']}"
    else:
        message = event.replace('_', ' ').capitalize()
    message = f"[{datetime.now(timezone.utc):%H:%M:%S} UTC] {message}"
    display = globals().get("tqdm")
    if display is None:
        print(message, flush=True)
    else:
        # Keep status lines readable alongside an interactive progress bar.
        display.write(message, file=sys.stdout)
        sys.stdout.flush()


if __name__ == "__main__":
    # Keep this before NumPy/Pandas/Torch imports: import or filesystem stalls
    # must still leave evidence that the scheduler launched this process.
    log_progress("startup", pid=os.getpid(), script=str(Path(__file__).absolute()))

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from evaluation.attribute_predictor import (ATTRIBUTES, CLASS_COUNTS, NORMALIZATION,
                                          SCHEMA, AttributePredictor, sha256_file,
                                          state_dict_sha256)
from data_handle.embed import DatasetConfig, build_image_transform, get_sample, seed_worker


def atomic_json(path, value):
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=".partial-", dir=Path(path).parent)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_checkpoint(path, model, metadata):
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    metadata = dict(metadata, state_dict_sha256=state_dict_sha256(state))
    fd, temporary = tempfile.mkstemp(prefix="partial-predictor-", suffix=".pt", dir=Path(path).parent)
    os.close(fd)
    try:
        torch.save({"state_dict": state, "metadata": metadata}, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return metadata


def load_checked_splits(split_dir, data_dir, progress=None, log_interval_seconds=30.0):
    """Check all full splits; optional callback receives (event, **fields)."""
    started = time.monotonic()
    if progress is not None:
        progress("split_checks_start", split_dir=str(split_dir), data_dir=str(data_dir))
    frames, hashes, patients, paths = {}, {}, {}, {}
    for name in ("train", "valid", "test"):
        path = Path(split_dir) / f"{name}.csv"
        stage_started = time.monotonic()
        if progress is not None:
            progress("split_csv_hash_start", split=name, path=str(path), stage="before_read")
        hashes[name] = sha256_file(path)
        if progress is not None:
            progress("split_csv_hash_complete", split=name, stage="before_read",
                     elapsed_seconds=time.monotonic() - stage_started)
            progress("split_csv_read_start", split=name, path=str(path))
        stage_started = time.monotonic()
        frame = pd.read_csv(path, dtype={"empi_anon": "string", "image_path": "string"}, low_memory=False)
        if progress is not None:
            progress("split_csv_read_complete", split=name, rows=len(frame),
                     elapsed_seconds=time.monotonic() - stage_started)
            progress("split_csv_hash_start", split=name, path=str(path), stage="after_read")
        stage_started = time.monotonic()
        if sha256_file(path) != hashes[name]:
            raise RuntimeError(f"{name}.csv changed while reading")
        if progress is not None:
            progress("split_csv_hash_complete", split=name, stage="after_read",
                     elapsed_seconds=time.monotonic() - stage_started)
            progress("split_labels_check_start", split=name, rows=len(frame))
        stage_started = time.monotonic()
        # cview is required by the repository get_sample helper, but is not predicted.
        required = {"empi_anon", "image_path", "cview", *ATTRIBUTES}
        missing = required - set(frame.columns)
        if missing or len(frame) == 0:
            raise ValueError(f"{name}.csv is empty or missing columns: {sorted(missing)}")
        for key in ("empi_anon", "image_path"):
            if frame[key].isna().any() or frame[key].str.strip().eq("").any():
                raise ValueError(f"{name}.csv contains empty {key}")
        for key, count in CLASS_COUNTS.items():
            values = pd.to_numeric(frame[key], errors="raise").to_numpy(dtype=np.float64)
            if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all() or (values < 0).any() or (values >= count).any():
                raise ValueError(f"{name}.csv {key} must be integer labels in [0,{count - 1}]")
            frame[key] = values.astype(np.int64)
        age = pd.to_numeric(frame["age"], errors="raise").to_numpy(dtype=np.float64)
        if not np.isfinite(age).all() or (age < 0).any() or (age > 100).any():
            raise ValueError(f"{name}.csv age must contain finite ages in years in [0,100]")
        frame["age"] = age
        patients[name] = set(frame["empi_anon"].str.strip())
        if progress is not None:
            progress("split_labels_check_complete", split=name, rows=len(frame),
                     patients=len(patients[name]), elapsed_seconds=time.monotonic() - stage_started)
            progress("path_resolution_start", split=name, total_paths=len(frame))
        stage_started = time.monotonic()
        last_logged = stage_started
        paths[name] = set()
        for index, value in enumerate(frame["image_path"], 1):
            # Preserve full canonicalization and overlap safety, including symlinks.
            paths[name].add(str((Path(data_dir) / str(value)).resolve()))
            if progress is not None:
                now = time.monotonic()
                if index % 10000 == 0 or index == len(frame) or now - last_logged >= log_interval_seconds:
                    progress("path_resolution_progress", split=name, paths=index, total_paths=len(frame),
                             elapsed_seconds=now - stage_started)
                    last_logged = now
        if len(paths[name]) != len(frame):
            raise ValueError(f"{name}.csv repeats an image path")
        frames[name] = frame.reset_index(drop=True)
        if progress is not None:
            progress("path_resolution_complete", split=name, paths=len(paths[name]),
                     elapsed_seconds=time.monotonic() - stage_started)
    if progress is not None:
        progress("split_overlap_check_start")
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        if patients[left] & patients[right]:
            raise ValueError(f"Patient overlap between {left} and {right}")
        if paths[left] & paths[right]:
            raise ValueError(f"Image overlap between {left} and {right}")
    if progress is not None:
        progress("split_checks_complete", split_sizes={name: len(frame) for name, frame in frames.items()},
                 elapsed_seconds=time.monotonic() - started)
    return frames, hashes


class AttributeDataset(Dataset):
    """Strict image reads: failures identify the row and never replace it."""
    def __init__(self, frame, data_dir, height, width):
        self.frame = frame.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = build_image_transform(DatasetConfig(img_height=height, img_width=width, img_channels=1))

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        try:
            image, labels = get_sample(self.data_dir, self.frame.iloc[index], normalize_age=False)
            image = self.transform(image)
            if image.ndim != 3 or image.shape[0] != 1 or not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                raise ValueError("Expected a finite 1xHxW image in [0,1]")
            target = {key: torch.tensor(labels[key], dtype=torch.long if key in CLASS_COUNTS else torch.float32)
                      for key in ATTRIBUTES}
            return image, target
        except Exception as error:
            raise RuntimeError(f"Image read/preprocessing failed at dataset row {index}; no replacement was sampled") from error


def training_statistics(frame):
    counts = {key: np.bincount(frame[key].to_numpy(dtype=np.int64), minlength=count).tolist()
              for key, count in CLASS_COUNTS.items()}
    weights = {}
    for key, values in counts.items():
        values = np.asarray(values, dtype=np.float64)
        weights[key] = np.divide(len(frame), len(values) * values,
                                 out=np.zeros_like(values), where=values > 0).tolist()
    mean = float(frame.age.mean())
    std = float(frame.age.to_numpy(dtype=np.float64).std())
    if not math.isfinite(std) or std <= 0:
        raise ValueError("Training ages must have positive finite standard deviation")
    return counts, weights, mean, std


def classification_report(matrix):
    matrix = np.asarray(matrix, dtype=np.int64)
    support, predicted = matrix.sum(axis=1), matrix.sum(axis=0)
    diagonal = matrix.diagonal().astype(np.float64)
    recall = np.divide(diagonal, support, out=np.zeros_like(diagonal), where=support > 0)
    f1 = np.divide(2 * diagonal, support + predicted, out=np.zeros_like(diagonal), where=(support + predicted) > 0)
    return {"confusion_matrix": matrix.tolist(), "support": support.tolist(),
            "accuracy": float(diagonal.sum() / matrix.sum()),
            "balanced_accuracy": float(recall[support > 0].mean()),
            "macro_f1": float(f1.mean())}


class BatchProgress:
    """Shared first/last/time-based reporting without affecting batch contents."""
    def __init__(self, callback, event, interval_seconds=30.0, **context):
        self.callback, self.event, self.interval_seconds = callback, event, interval_seconds
        self.context = context
        self.started = time.monotonic()
        self.last_logged = self.started

    def update(self, batch, batches, images, total_images, **fields):
        if self.callback is None:
            return
        now = time.monotonic()
        if batch == 1 or batch == batches or now - self.last_logged >= self.interval_seconds:
            elapsed = now - self.started
            rate = images / max(elapsed, 1e-9)
            eta = max(0, total_images - images) / rate if total_images is not None and rate > 0 else None
            self.callback(self.event, **self.context, batch=batch, batches=batches,
                          images=images, total_images=total_images,
                          elapsed_seconds=elapsed, eta_seconds=eta, images_per_second=rate, **fields)
            self.last_logged = now


@torch.inference_mode()
def evaluate(model, loader, device, class_counts, train_age_mean, progress=None,
             label="evaluation", log_interval_seconds=30.0):
    batches = len(loader) if hasattr(loader, "__len__") else None
    total_images = len(loader.dataset) if hasattr(loader, "dataset") else None
    reporter = BatchProgress(progress, f"{label}_progress", log_interval_seconds)
    if progress is not None:
        progress(f"{label}_start", batches=batches, total_images=total_images)
    model.eval()
    matrices = {key: np.zeros((count, count), dtype=np.int64) for key, count in CLASS_COUNTS.items()}
    baseline_matrices = {key: np.zeros_like(matrix) for key, matrix in matrices.items()}
    errors, baseline_errors = [], []
    n = 0
    for batch, (images, labels) in enumerate(loader, 1):
        predictions = model(images.to(device))
        n += len(images)
        for key, count in CLASS_COUNTS.items():
            target = labels[key].numpy()
            guess = predictions[key].argmax(1).cpu().numpy()
            matrices[key] += np.bincount(target * count + guess, minlength=count * count).reshape(count, count)
            majority = int(np.argmax(class_counts[key]))
            baseline_matrices[key] += np.bincount(target * count + majority, minlength=count * count).reshape(count, count)
        age = labels["age"].numpy().astype(np.float64)
        errors.extend((predictions["age"].cpu().numpy().astype(np.float64) - age).tolist())
        baseline_errors.extend((train_age_mean - age).tolist())
        reporter.update(batch, batches, n, total_images)
    if n == 0:
        raise ValueError("Cannot evaluate an empty split")
    def age_report(values):
        values = np.asarray(values, dtype=np.float64)
        return {"mae_years": float(np.abs(values).mean()), "rmse_years": float(np.sqrt(np.square(values).mean()))}
    report = {"n": n, **{key: classification_report(matrix) for key, matrix in matrices.items()},
            "age": age_report(errors),
            "baselines": {"categorical_majority": {key: classification_report(matrix) for key, matrix in baseline_matrices.items()},
                          "train_mean_age": age_report(baseline_errors)},
            "metric_conventions": {"balanced_accuracy": "mean recall over classes present in this split",
                                   "macro_f1": "mean F1 over every declared class; undefined F1=0"}}
    if progress is not None:
        progress(f"{label}_complete", images=n, elapsed_seconds=time.monotonic() - reporter.started, metrics=report)
    return report


def selection_score(metrics, train_age_std):
    # Four equally weighted errors; only validation data selects the checkpoint.
    return (sum(1 - metrics[key]["balanced_accuracy"] for key in CLASS_COUNTS)
            + metrics["age"]["mae_years"] / train_age_std) / len(ATTRIBUTES)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data-dir", type=Path, default=Path("/vol/biodata/data/Mammo/EMBED/pngs/1024x768"))
    result.add_argument("--split-dir", type=Path, default=Path("/vol/biomedic3/tx1215/mamo-flow/assets/embed_splits_v1"))
    result.add_argument("--output-dir", type=Path, required=True,
                        help="New run directory, or an existing directory containing only logs/")
    result.add_argument("--height", type=int, default=256)
    result.add_argument("--width", type=int, default=192)
    result.add_argument("--epochs", type=int, default=20)
    result.add_argument("--batch-size", type=int, default=64)
    result.add_argument("--num-workers", type=int, default=8)
    result.add_argument("--device", default="cuda")
    result.add_argument("--backbone", choices=("resnet50", "resnet18"), default="resnet50")
    result.add_argument("--pretrained", choices=("imagenet", "none"), default="imagenet")
    result.add_argument("--patience", type=int, default=5)
    result.add_argument("--seed", type=int, default=1234)
    result.add_argument("--cpu-threads", type=int, default=8)
    result.add_argument("--learning-rate", type=float, default=3e-4)
    result.add_argument("--weight-decay", type=float, default=1e-4)
    result.add_argument("--log-interval-seconds", type=float, default=30.0,
                        help="Progress interval in seconds; first and last batches are always logged")
    result.add_argument("--no-progress-bar", action="store_true",
                        help="Hide the training bar while retaining timestamped log messages")
    for split in ("train", "valid", "test"):
        result.add_argument(f"--max-{split}-samples", type=int, default=None,
                            help="Debug only; any sample cap makes the checkpoint ineligible for effectiveness")
    return result


def train(args):
    if not math.isfinite(args.log_interval_seconds) or args.log_interval_seconds <= 0:
        raise ValueError("--log-interval-seconds must be positive and finite")
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    log_progress("train_config", config=config)
    if min(args.height, args.width) < 32 or min(args.epochs, args.batch_size, args.patience, args.cpu_threads) <= 0 or args.num_workers < 0:
        raise ValueError("Invalid image size, epochs, batch size, patience, workers or CPU threads")
    if args.learning_rate <= 0 or not math.isfinite(args.learning_rate) or args.weight_decay < 0 or not math.isfinite(args.weight_decay):
        raise ValueError("Invalid optimizer configuration")
    for name in ("train", "valid", "test"):
        cap = getattr(args, f"max_{name}_samples")
        if cap is not None and cap < 2:
            raise ValueError("Debug sample caps must be >=2")
    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(
            path.name != "logs" or not path.is_dir() for path in args.output_dir.iterdir())):
        raise FileExistsError(f"Output directory must be new or contain only logs/: {args.output_dir}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; choose --device cpu for a debug run")
    log_progress("device_ready", device=str(device), cpu_threads=args.cpu_threads,
                 num_workers=args.num_workers, cuda_available=torch.cuda.is_available())
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(args.cpu_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    frames, split_hashes = load_checked_splits(args.split_dir, args.data_dir, progress=log_progress,
                                             log_interval_seconds=args.log_interval_seconds)
    full_sizes = {key: len(value) for key, value in frames.items()}
    debug_run = any(getattr(args, f"max_{key}_samples") is not None for key in frames)
    for index, key in enumerate(frames):
        cap = getattr(args, f"max_{key}_samples")
        if cap is not None:
            frames[key] = frames[key].sample(n=min(cap, len(frames[key])), random_state=args.seed + index).reset_index(drop=True)
    counts, weights, age_mean, age_std = training_statistics(frames["train"])
    if not debug_run and any(min(values) <= 0 for values in counts.values()):
        raise ValueError("Every declared categorical class must occur in the training split")
    source_files = ["evaluation/attribute_predictor.py", "evaluation/train_attribute_predictor.py", "data_handle/embed.py"]
    metadata = {"format_version": 1, "schema": SCHEMA, "backbone": args.backbone,
                "resolution": [args.height, args.width], "input_range": [0, 1],
                "input_normalization": NORMALIZATION, "age_mean_years": age_mean, "age_std_years": age_std,
                "split_csv_sha256": split_hashes, "source_sha256": {name: sha256_file(ROOT / name) for name in source_files},
                "class_counts": counts, "class_weights": weights, "full_split_sizes": full_sizes,
                "split_sizes": {key: len(value) for key, value in frames.items()}, "training_config": config,
                "selection_rule": "min mean(1-validation balanced_accuracy for each categorical head, validation age MAE / training age std)",
                "debug_run": debug_run, "validated_for_effectiveness": False, "status": "training",
                "quality_assessment": {"status": "requires_user_review",
                                       "note": "Protocol completion does not certify predictor quality. Review each held-out attribute metric against training-only baselines; generated-image domain shift can bias effectiveness."},
                "validation_metrics": None, "test_metrics": None, "selected_epoch": None,
                "completed_epochs": 0, "started_at": datetime.now(timezone.utc).isoformat(),
                "torch_version": str(torch.__version__)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Exclusive claim also protects two launchers that inspected the directory together.
    with (args.output_dir / ".training_started").open("x") as claim:
        claim.write(metadata["started_at"] + "\n")
    atomic_json(args.output_dir / "run.json", metadata)
    log_progress("run_metadata_saved", output_dir=str(args.output_dir), split_sizes=metadata["split_sizes"],
                 debug_run=debug_run)
    history = []
    try:
        generators = {key: torch.Generator().manual_seed(args.seed + index) for index, key in enumerate(frames)}
        loaders = {key: DataLoader(AttributeDataset(frame, args.data_dir, args.height, args.width),
                                   batch_size=args.batch_size, shuffle=key == "train", num_workers=args.num_workers,
                                   pin_memory=device.type == "cuda", worker_init_fn=seed_worker, generator=generators[key],
                                   drop_last=False, persistent_workers=args.num_workers > 0)
                   for key, frame in frames.items()}
        model_started = time.monotonic()
        log_progress("model_init_start", backbone=args.backbone, pretrained=args.pretrained,
                     pretrained_weights_may_download=args.pretrained == "imagenet",
                     torch_home=os.environ.get("TORCH_HOME"), device=str(device))
        model = AttributePredictor(args.backbone, args.pretrained, (args.height, args.width), age_mean, age_std).to(device)
        log_progress("model_init_complete", backbone=args.backbone, elapsed_seconds=time.monotonic() - model_started)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        losses = {key: nn.CrossEntropyLoss(weight=torch.tensor(weights[key], dtype=torch.float32, device=device)) for key in CLASS_COUNTS}
        best, stale = math.inf, 0
        checkpoint_path = args.output_dir / "best.pt"
        for epoch in range(1, args.epochs + 1):
            log_progress("epoch_start", epoch=epoch, epochs=args.epochs, batches=len(loaders["train"]),
                         total_images=len(frames["train"]))
            show_bar = sys.stderr.isatty() and not args.no_progress_bar
            reporter = BatchProgress(None if show_bar else log_progress, "train_progress",
                                     args.log_interval_seconds, epoch=epoch, epochs=args.epochs)
            model.train()
            total_loss, seen = 0.0, 0
            with tqdm(loaders["train"], total=len(loaders["train"]),
                      desc=f"Train epoch {epoch}/{args.epochs}", unit="batch",
                      file=sys.stderr, dynamic_ncols=True, ascii=True, miniters=1,
                      mininterval=min(1.0, args.log_interval_seconds) if sys.stderr.isatty()
                                  else args.log_interval_seconds,
                      disable=not show_bar) as batches:
                for batch, (images, labels) in enumerate(batches, 1):
                    images = images.to(device, non_blocking=device.type == "cuda")
                    labels = {key: value.to(device, non_blocking=device.type == "cuda") for key, value in labels.items()}
                    optimizer.zero_grad(set_to_none=True)
                    predictions = model(images)
                    loss = sum(losses[key](predictions[key], labels[key]) for key in CLASS_COUNTS)
                    loss = loss + nn.functional.smooth_l1_loss((predictions["age"] - age_mean) / age_std,
                                                              (labels["age"] - age_mean) / age_std)
                    if not torch.isfinite(loss):
                        raise ValueError("Training loss is NaN/Inf")
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
                    optimizer.step()
                    loss_value = float(loss.detach())
                    total_loss += loss_value * len(images)
                    seen += len(images)
                    batches.set_postfix(loss=f"{loss_value:.4f}", mean_loss=f"{total_loss / seen:.4f}", refresh=False)
                    reporter.update(batch, len(loaders["train"]), seen, len(frames["train"]),
                                    loss=loss_value, mean_loss=total_loss / seen)
            scheduler.step()
            validation = evaluate(model, loaders["valid"], device, counts, age_mean, progress=log_progress,
                                  label="validation", log_interval_seconds=args.log_interval_seconds)
            score = selection_score(validation, age_std)
            record = {"epoch": epoch, "train_loss": total_loss / seen,
                      "validation_score": score, "validation_metrics": validation,
                      "learning_rate": optimizer.param_groups[0]["lr"]}
            history.append(record)
            metadata["completed_epochs"] = epoch
            if score < best:
                best, stale = score, 0
                metadata.update(selected_epoch=epoch, selection_score=score, validation_metrics=validation)
                atomic_checkpoint(checkpoint_path, model, metadata)
                log_progress("checkpoint_saved", path=str(checkpoint_path), epoch=epoch, selection_score=score,
                             status="best_validation_update")
            else:
                stale += 1
            atomic_json(args.output_dir / "history.json", history)
            atomic_json(args.output_dir / "run.json", metadata)
            log_progress("epoch_complete", **{key: record[key] for key in ("epoch", "train_loss", "validation_score")})
            if stale >= args.patience:
                log_progress("early_stopping", epoch=epoch, selected_epoch=metadata["selected_epoch"],
                             selection_score=metadata["selection_score"], stale_epochs=stale, patience=args.patience)
                break
        # The test split is evaluated exactly once, after validation selection.
        selected = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(selected["state_dict"], strict=True)
        metadata["test_metrics"] = evaluate(model, loaders["test"], device, counts, age_mean, progress=log_progress,
                                            label="test", log_interval_seconds=args.log_interval_seconds)
        for key in frames:
            if sha256_file(args.split_dir / f"{key}.csv") != split_hashes[key]:
                raise RuntimeError(f"{key}.csv changed during training")
        metadata.update(status="complete", validated_for_effectiveness=not debug_run,
                        completed_at=datetime.now(timezone.utc).isoformat(), stop_reason="early_stopping" if stale >= args.patience else "epochs_completed")
        metadata = atomic_checkpoint(checkpoint_path, model, metadata)
        log_progress("checkpoint_saved", path=str(checkpoint_path), epoch=metadata["selected_epoch"],
                     selection_score=metadata["selection_score"], status="complete")
        atomic_json(args.output_dir / "run.json", metadata)
        log_progress("training_complete", checkpoint=str(checkpoint_path), debug_run=debug_run,
                     validated_for_effectiveness=metadata["validated_for_effectiveness"],
                     validation=metadata["validation_metrics"], test=metadata["test_metrics"])
    except BaseException as error:
        metadata.update(status="failed", validated_for_effectiveness=False, error_type=type(error).__name__,
                        error=str(error), failed_at=datetime.now(timezone.utc).isoformat())
        log_progress("training_failed", error_type=type(error).__name__, error=str(error))
        atomic_json(args.output_dir / "run.json", metadata)
        raise


if __name__ == "__main__":
    train(parser().parse_args())
