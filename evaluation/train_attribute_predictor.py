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

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

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


def load_checked_splits(split_dir, data_dir):
    """Check all full splits before applying optional debug subsampling."""
    frames, hashes, patients, paths = {}, {}, {}, {}
    for name in ("train", "valid", "test"):
        path = Path(split_dir) / f"{name}.csv"
        hashes[name] = sha256_file(path)
        frame = pd.read_csv(path, dtype={"empi_anon": "string", "image_path": "string"}, low_memory=False)
        if sha256_file(path) != hashes[name]:
            raise RuntimeError(f"{name}.csv changed while reading")
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
        paths[name] = {str((Path(data_dir) / str(value)).resolve()) for value in frame["image_path"]}
        if len(paths[name]) != len(frame):
            raise ValueError(f"{name}.csv repeats an image path")
        frames[name] = frame.reset_index(drop=True)
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        if patients[left] & patients[right]:
            raise ValueError(f"Patient overlap between {left} and {right}")
        if paths[left] & paths[right]:
            raise ValueError(f"Image overlap between {left} and {right}")
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


@torch.inference_mode()
def evaluate(model, loader, device, class_counts, train_age_mean):
    model.eval()
    matrices = {key: np.zeros((count, count), dtype=np.int64) for key, count in CLASS_COUNTS.items()}
    baseline_matrices = {key: np.zeros_like(matrix) for key, matrix in matrices.items()}
    errors, baseline_errors = [], []
    n = 0
    for images, labels in loader:
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
    if n == 0:
        raise ValueError("Cannot evaluate an empty split")
    def age_report(values):
        values = np.asarray(values, dtype=np.float64)
        return {"mae_years": float(np.abs(values).mean()), "rmse_years": float(np.sqrt(np.square(values).mean()))}
    return {"n": n, **{key: classification_report(matrix) for key, matrix in matrices.items()},
            "age": age_report(errors),
            "baselines": {"categorical_majority": {key: classification_report(matrix) for key, matrix in baseline_matrices.items()},
                          "train_mean_age": age_report(baseline_errors)},
            "metric_conventions": {"balanced_accuracy": "mean recall over classes present in this split",
                                   "macro_f1": "mean F1 over every declared class; undefined F1=0"}}


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
    for split in ("train", "valid", "test"):
        result.add_argument(f"--max-{split}-samples", type=int, default=None,
                            help="Debug only; any sample cap makes the checkpoint ineligible for effectiveness")
    return result


def train(args):
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
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(args.cpu_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    frames, split_hashes = load_checked_splits(args.split_dir, args.data_dir)
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
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
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
    history = []
    try:
        generators = {key: torch.Generator().manual_seed(args.seed + index) for index, key in enumerate(frames)}
        loaders = {key: DataLoader(AttributeDataset(frame, args.data_dir, args.height, args.width),
                                   batch_size=args.batch_size, shuffle=key == "train", num_workers=args.num_workers,
                                   pin_memory=device.type == "cuda", worker_init_fn=seed_worker, generator=generators[key],
                                   drop_last=False, persistent_workers=args.num_workers > 0)
                   for key, frame in frames.items()}
        model = AttributePredictor(args.backbone, args.pretrained, (args.height, args.width), age_mean, age_std).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        losses = {key: nn.CrossEntropyLoss(weight=torch.tensor(weights[key], dtype=torch.float32, device=device)) for key in CLASS_COUNTS}
        best, stale = math.inf, 0
        checkpoint_path = args.output_dir / "best.pt"
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss, seen = 0.0, 0
            for images, labels in loaders["train"]:
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
                total_loss += float(loss.detach()) * len(images)
                seen += len(images)
            scheduler.step()
            validation = evaluate(model, loaders["valid"], device, counts, age_mean)
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
            else:
                stale += 1
            atomic_json(args.output_dir / "history.json", history)
            atomic_json(args.output_dir / "run.json", metadata)
            print(json.dumps({key: record[key] for key in ("epoch", "train_loss", "validation_score")}), flush=True)
            if stale >= args.patience:
                break
        # The test split is evaluated exactly once, after validation selection.
        selected = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(selected["state_dict"], strict=True)
        metadata["test_metrics"] = evaluate(model, loaders["test"], device, counts, age_mean)
        for key in frames:
            if sha256_file(args.split_dir / f"{key}.csv") != split_hashes[key]:
                raise RuntimeError(f"{key}.csv changed during training")
        metadata.update(status="complete", validated_for_effectiveness=not debug_run,
                        completed_at=datetime.now(timezone.utc).isoformat(), stop_reason="early_stopping" if stale >= args.patience else "epochs_completed")
        metadata = atomic_checkpoint(checkpoint_path, model, metadata)
        atomic_json(args.output_dir / "run.json", metadata)
        print(f"Saved {'DEBUG (not eligible for effectiveness)' if debug_run else 'completed predictor'}: {checkpoint_path}", flush=True)
        print(json.dumps({"validation": metadata["validation_metrics"], "test": metadata["test_metrics"]}, allow_nan=False), flush=True)
    except BaseException as error:
        metadata.update(status="failed", validated_for_effectiveness=False, error_type=type(error).__name__,
                        error=str(error), failed_at=datetime.now(timezone.utc).isoformat())
        atomic_json(args.output_dir / "run.json", metadata)
        raise


if __name__ == "__main__":
    train(parser().parse_args())
