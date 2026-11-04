"""Uniform intervention targets shared by saved sampling and training previews.

Each draw is keyed by the protocol, seed, CSV row index and attribute. Model
randomness, batch boundaries and requested-attribute order cannot change it.
Categorical draws include the factual class; those draws are no-edit controls.
"""
import csv
from functools import lru_cache
import hashlib
import json
import math
from numbers import Integral
from pathlib import Path
import random

import torch


TARGET_PROTOCOL = "uniform_v1"
CLASS_COUNTS = {"view": 2, "density": 4, "scanner": 5}


def _age_bounds(bounds):
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        raise ValueError("Age targets require (minimum, maximum) training ages in years.")
    try:
        if any(isinstance(value, bool) for value in bounds):
            raise ValueError()
        low, high = map(float, bounds)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Training age bounds must be numeric and finite.") from error
    if not math.isfinite(low) or not math.isfinite(high) or not 0 <= low <= high <= 100:
        raise ValueError("Training age bounds must satisfy 0 <= minimum <= maximum <= 100 years.")
    return low, high


@lru_cache(maxsize=16)
def _training_age_range(train_csv):
    low, high = math.inf, -math.inf
    with Path(train_csv).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if "age" not in (reader.fieldnames or ()):
            raise ValueError("Training CSV must contain an age column: " + train_csv)
        for row_number, row in enumerate(reader, start=2):
            try:
                age = float(row["age"])
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError("Invalid training age at CSV row {}: {}".format(row_number, train_csv)) from error
            if not math.isfinite(age) or not 0 <= age <= 100:
                raise ValueError("Training ages must be finite and in [0,100] years: " + train_csv)
            low, high = min(low, age), max(high, age)
    if not math.isfinite(low):
        raise ValueError("Training CSV contains no ages: " + train_csv)
    return low, high


def load_training_age_range(split_dir):
    """Return cached min/max years from the resolved split_dir/train.csv only."""
    train_csv = (Path(split_dir).expanduser() / "train.csv").resolve()
    return _training_age_range(str(train_csv))


def _sample_indices(indices, size):
    if indices is None:
        return list(range(size))
    if torch.is_tensor(indices):
        indices = indices.detach().cpu().tolist()
    try:
        indices = list(indices)
    except TypeError as error:
        raise ValueError("Intervention indices must contain one CSV row index per image.") from error
    if len(indices) != size or any(isinstance(index, bool) or not isinstance(index, Integral)
                                  or index < 0 for index in indices):
        raise ValueError("Intervention indices must be nonnegative integers, one per image.")
    return [int(index) for index in indices]


def _generator(seed, index, attribute):
    key = json.dumps([TARGET_PROTOCOL, seed, index, attribute], separators=(",", ":"))
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest, "big"))


def make_counterfactual_parents(parents, attr, *, seed=1234, indices=None, age_range_years=None):
    """Clone conditions and draw one uniform target for the requested attribute.

    Pass stable CSV row indices for paired comparisons. ``indices=None`` retains
    compatibility with callers that only have a single ordered preview batch.
    Age bounds must be supplied explicitly in years from the training split.
    """
    if attr not in (*CLASS_COUNTS, "age"):
        raise ValueError("Unknown counterfactual attribute: " + str(attr))
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("Intervention seed must be an integer.")
    seed = int(seed)
    value = parents[attr]
    width = CLASS_COUNTS.get(attr, 1)
    if not torch.is_tensor(value) or value.ndim != 2 or value.shape[1] != width:
        raise ValueError("{} conditions must have shape Bx{}.".format(attr, width))
    indices = _sample_indices(indices, value.shape[0])
    changed = {key: tensor.clone() for key, tensor in parents.items()}
    if attr == "age":
        if not value.is_floating_point():
            raise ValueError("Age conditions must be floating-point values normalized by 100.")
        low, high = _age_bounds(age_range_years)
        draws = [_generator(seed, index, attr).uniform(low, high) / 100.0 for index in indices]
        changed[attr] = torch.tensor(draws, device=value.device, dtype=value.dtype).reshape(value.shape)
    else:
        targets = [_generator(seed, index, attr).randrange(width) for index in indices]
        target_tensor = torch.tensor(targets, device=value.device, dtype=torch.long).reshape(-1, 1)
        changed[attr].zero_().scatter_(1, target_tensor, 1)
    return changed
