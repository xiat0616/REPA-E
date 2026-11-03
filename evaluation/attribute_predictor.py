"""Independent image-only attribute predictor and offline checkpoint loading.

Inputs are Bx1xHxW grayscale images in [0, 1]. Age predictions are in years.
ImageNet weights may be requested only when constructing a new training model.
"""
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from torchvision import models

ATTRIBUTES = ("view", "density", "scanner", "age")
CLASS_COUNTS = {"view": 2, "density": 4, "scanner": 5}
SCHEMA = {**{key: {"kind": "categorical", "classes": count}
             for key, count in CLASS_COUNTS.items()},
          "age": {"kind": "continuous", "unit": "years"}}
NORMALIZATION = "imagenet_rgb_after_gray_repeat"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_dict_sha256(state):
    """Canonical weight fingerprint, independent of torch.save ZIP metadata."""
    digest = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        if not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all():
            raise ValueError(f"Non-tensor or nonfinite predictor state: {key}")
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([key, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class AttributePredictor(nn.Module):
    def __init__(self, backbone="resnet50", pretrained="none", resolution=(256, 192),
                 age_mean_years=0.0, age_std_years=1.0):
        super().__init__()
        if backbone not in ("resnet18", "resnet50"):
            raise ValueError("backbone must be resnet18 or resnet50")
        if pretrained not in ("none", "imagenet"):
            raise ValueError("pretrained must be none or imagenet")
        if len(resolution) != 2 or any(type(x) is not int or x < 32 for x in resolution):
            raise ValueError("resolution must be [height, width], both >=32")
        if not math.isfinite(age_mean_years) or not math.isfinite(age_std_years) or age_std_years <= 0:
            raise ValueError("Invalid training age normalization")
        self.resolution = tuple(resolution)
        constructor = getattr(models, backbone)
        weight_enum = models.ResNet50_Weights if backbone == "resnet50" else models.ResNet18_Weights
        self.encoder = constructor(weights=weight_enum.DEFAULT if pretrained == "imagenet" else None)
        features = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()
        self.heads = nn.ModuleDict({key: nn.Linear(features, count) for key, count in CLASS_COUNTS.items()})
        self.heads["age"] = nn.Linear(features, 1)
        self.register_buffer("rgb_mean", torch.tensor([.485, .456, .406]).reshape(1, 3, 1, 1))
        self.register_buffer("rgb_std", torch.tensor([.229, .224, .225]).reshape(1, 3, 1, 1))
        self.register_buffer("age_mean", torch.tensor(float(age_mean_years)))
        self.register_buffer("age_std", torch.tensor(float(age_std_years)))

    def forward(self, images):
        if images.ndim != 4 or images.shape[1] != 1 or tuple(images.shape[-2:]) != self.resolution:
            raise ValueError(f"Expected Bx1x{self.resolution[0]}x{self.resolution[1]} images")
        if not images.is_floating_point() or not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
            raise ValueError("Predictor images must be finite floating-point values in [0,1]")
        features = self.encoder((images.float().repeat(1, 3, 1, 1) - self.rgb_mean) / self.rgb_std)
        predictions = {key: self.heads[key](features) for key in CLASS_COUNTS}
        predictions["age"] = self.heads["age"](features).squeeze(1) * self.age_std + self.age_mean
        if any(not torch.isfinite(value).all() for value in predictions.values()):
            raise ValueError("Predictor produced NaN/Inf")
        return predictions


def _validate_report(report, name):
    if not isinstance(report, dict) or type(report.get("n")) is not int or report["n"] <= 0:
        raise ValueError(f"Missing completed {name} report")
    for key, count in CLASS_COUNTS.items():
        item = report.get(key, {})
        matrix = item.get("confusion_matrix")
        if not isinstance(matrix, list) or len(matrix) != count or any(not isinstance(row, list) or len(row) != count for row in matrix):
            raise ValueError(f"Invalid {name} {key} confusion matrix")
        if any(type(value) is not int or value < 0 for row in matrix for value in row) or sum(map(sum, matrix)) != report["n"]:
            raise ValueError(f"Invalid {name} {key} counts")
        for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
            value = item.get(metric)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid {name} {key} {metric}")
    for metric in ("mae_years", "rmse_years"):
        value = report.get("age", {}).get(metric)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid {name} age {metric}")
    if not isinstance(report.get("baselines"), dict):
        raise ValueError(f"Missing {name} training-only baselines")
    baselines = report["baselines"]
    for key, count in CLASS_COUNTS.items():
        item = baselines.get("categorical_majority", {}).get(key, {})
        matrix = item.get("confusion_matrix", [])
        if len(matrix) != count or any(not isinstance(row, list) or len(row) != count for row in matrix):
            raise ValueError(f"Missing {name} {key} majority baseline")
        if any(type(value) is not int or value < 0 for row in matrix for value in row) or sum(map(sum, matrix)) != report["n"]:
            raise ValueError(f"Invalid {name} {key} majority baseline counts")
        for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
            value = item.get(metric)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid {name} {key} majority baseline")
    for metric in ("mae_years", "rmse_years"):
        value = baselines.get("train_mean_age", {}).get(metric)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Missing {name} train-mean-age baseline")


def load_predictor(path, device="cpu"):
    """Load completed, fully evaluated weights without network access.

    Debug/sample-capped runs and intermediate training checkpoints are refused.
    Returned metadata includes both file and canonical tensor fingerprints.
    """
    path = Path(path)
    before = sha256_file(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if sha256_file(path) != before:
        raise RuntimeError("Predictor checkpoint changed while loading")
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("metadata"), dict):
        raise ValueError("Predictor checkpoint requires metadata and state_dict")
    metadata = dict(checkpoint["metadata"])
    # Reject non-JSON or nonfinite metadata, including nested metric values.
    json.dumps(metadata, allow_nan=False)
    if metadata.get("format_version") != 1 or metadata.get("schema") != SCHEMA:
        raise ValueError("Unsupported predictor checkpoint schema")
    if metadata.get("status") != "complete" or metadata.get("debug_run") is not False or metadata.get("validated_for_effectiveness") is not True:
        raise ValueError("Predictor requires a completed, uncapped held-out evaluation; debug/incomplete checkpoints are not accepted")
    if metadata.get("input_range") != [0, 1] or metadata.get("input_normalization") != NORMALIZATION:
        raise ValueError("Unsupported predictor input preprocessing")
    for name in ("train", "valid", "test"):
        value = metadata.get("split_csv_sha256", {}).get(name, "")
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"Missing {name} split fingerprint")
    if not isinstance(metadata.get("source_sha256"), dict) or not metadata["source_sha256"]:
        raise ValueError("Missing predictor source provenance")
    for value in metadata["source_sha256"].values():
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Invalid predictor source fingerprint")
    _validate_report(metadata.get("validation_metrics"), "validation")
    _validate_report(metadata.get("test_metrics"), "test")
    model = AttributePredictor(backbone=metadata["backbone"], pretrained="none",
                               resolution=metadata["resolution"],
                               age_mean_years=metadata["age_mean_years"],
                               age_std_years=metadata["age_std_years"])
    state = checkpoint.get("state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("Missing predictor weights")
    fingerprint = state_dict_sha256(state)
    if fingerprint != metadata.get("state_dict_sha256"):
        raise ValueError("Predictor weight fingerprint mismatch")
    model.load_state_dict(state, strict=True)
    if not torch.equal(model.age_mean.cpu(), torch.tensor(float(metadata["age_mean_years"]))) or not torch.equal(model.age_std.cpu(), torch.tensor(float(metadata["age_std_years"]))):
        raise ValueError("Age normalization buffers disagree with checkpoint metadata")
    if not torch.equal(model.rgb_mean, torch.tensor([.485, .456, .406]).reshape(1, 3, 1, 1)) or not torch.equal(model.rgb_std, torch.tensor([.229, .224, .225]).reshape(1, 3, 1, 1)):
        raise ValueError("Input normalization buffers disagree with checkpoint metadata")
    model.to(device).eval().requires_grad_(False)
    metadata["checkpoint_sha256"] = before
    return model, metadata
