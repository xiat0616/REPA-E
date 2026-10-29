"""Portable image-cache format shared by sampling and offline evaluation.

Images are 16-bit grayscale PNGs with a fixed 65535 scale. CSV paths are relative
so the completed cache can be moved without access to the source dataset/model.
"""
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import struct

FIELDS = ('index', 'task', 'attribute', 'source', 'target', 'file_path', 'sha256',
          'input_path', 'age', 'view', 'density', 'scanner')
ATTRIBUTES = ('view', 'density', 'scanner', 'age')
TASKS = ('real', 'generation', 'reconstruction_vae', 'reconstruction_cycle', 'counterfactual')
IMAGE_STORAGE = {'format': '16-bit grayscale PNG', 'dtype': 'uint16', 'scale': 65535,
                 'encoding': 'round(clamp(image,0,1)*65535)',
                 'decoding': 'uint16/65535; no per-image normalization',
                 'maximum_rounding_error': 0.5 / 65535}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def resolve_image_path(root, relative_path):
    root = Path(root).expanduser().resolve()
    relative = PurePosixPath(str(relative_path))
    if (not relative_path or relative.is_absolute() or '..' in relative.parts
            or '\\' in str(relative_path) or relative.suffix.lower() != '.png'):
        raise ValueError(f'Image path must be a relative PNG path inside the cache: {relative_path}')
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'Image path escapes its cache directory: {relative_path}')
    return path


def write_image(root, relative_path, image):
    import numpy as np
    import torch
    from PIL import Image

    if not torch.is_tensor(image):
        raise TypeError('Saved images must be torch tensors.')
    values = image.detach().float().cpu()
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 2 or not torch.isfinite(values).all():
        raise ValueError('Saved images must be finite grayscale HxW or 1xHxW tensors.')
    array = (values.clamp(0, 1) * 65535).round().numpy().astype(np.uint16)
    path = resolve_image_path(root, relative_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # The conversion is explicit, rather than PIL's display-oriented scaling.
    temporary = path.with_suffix('.png.tmp')
    Image.fromarray(array).save(temporary, format='PNG')
    temporary.replace(path)
    return sha256_file(path)


def read_image(root, relative_path, expected_sha256=None):
    import numpy as np
    import torch
    from PIL import Image

    path = resolve_image_path(root, relative_path)
    payload = path.read_bytes()
    if expected_sha256 is not None and hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError(f'Image checksum does not match samples.csv: {relative_path}')
    # Check the PNG header itself. Pillow may expose 16-bit PNGs as mode I or I;16.
    if (len(payload) < 29 or payload[:8] != b'\x89PNG\r\n\x1a\n'
            or payload[12:16] != b'IHDR' or payload[24] != 16 or payload[25] != 0):
        raise ValueError(f'Expected a 16-bit grayscale PNG: {relative_path}')
    width, height = struct.unpack('>II', payload[16:24])
    with Image.open(io.BytesIO(payload)) as image:
        array = np.asarray(image)
    if array.shape != (height, width) or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f'Invalid grayscale image dimensions or pixels: {relative_path}')
    if array.min() < 0 or array.max() > 65535:
        raise ValueError(f'Invalid uint16 pixels: {relative_path}')
    return torch.from_numpy(array.astype(np.float32) / 65535).unsqueeze(0)


def save_json(path, value):
    path = Path(path)
    temporary = path.with_name('.' + path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def write_records(root, records):
    path = Path(root) / 'samples.csv'
    temporary = path.with_name('.samples.csv.tmp')
    with temporary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)


def reference_images_hash(records):
    values = {str(int(row['index'])): row['sha256'] for row in records if row['task'] == 'real'}
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def load_samples(root):
    """Validate cache metadata/relationships without loading all PNG byte payloads."""
    import math

    root = Path(root).expanduser().resolve()
    config = json.loads((root / 'config.json').read_text())
    selection = json.loads((root / 'indices.json').read_text())
    if config.get('status') != 'complete':
        raise ValueError(f'Sampling cache is not complete: {root}')
    if config.get('image_storage') != IMAGE_STORAGE:
        raise ValueError('Unsupported image cache encoding. Expected fixed-scale 16-bit grayscale PNGs.')
    indices = selection.get('indices', [])
    if (not indices or any(type(index) is not int or index < 0 for index in indices)
            or len(set(indices)) != len(indices)):
        raise ValueError('indices.json needs nonempty, unique, nonnegative integer indices.')
    if (selection.get('split') != config.get('split')
            or selection.get('split_csv_sha256') != config.get('split_csv_sha256')
            or config.get('num_images') != len(indices)
            or config.get('indices_sha256') != hashlib.sha256(json.dumps(indices).encode()).hexdigest()):
        raise ValueError('indices.json does not match the sampling configuration.')
    with (root / 'samples.csv').open(newline='') as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or []) != FIELDS:
            raise ValueError('samples.csv does not have the supported image manifest schema.')
        records = list(reader)
    expected_indices = set(indices)
    lookup, real, paths = {}, {}, set()
    for row in records:
        if None in row or any(value is None for value in row.values()):
            raise ValueError('samples.csv contains a malformed row.')
        try:
            index = int(row['index'])
        except ValueError as error:
            raise ValueError('samples.csv indices must be integers.') from error
        if str(index) != row['index'] or index not in expected_indices:
            raise ValueError('samples.csv contains an index outside the selected cohort.')
        task, attribute = row['task'], row['attribute']
        if task not in TASKS or attribute not in (ATTRIBUTES if task == 'counterfactual' else ('none',)):
            raise ValueError('samples.csv contains an unsupported task/attribute combination.')
        key = (index, task, attribute)
        if key in lookup or row['file_path'] in paths:
            raise ValueError('samples.csv contains duplicate image records or paths.')
        lookup[key] = row
        paths.add(row['file_path'])
        for field in ('file_path', 'input_path'):
            path = resolve_image_path(root, row[field])
            if not path.is_file():
                raise FileNotFoundError(f'Cache image is missing: {path}')
        digest = row['sha256']
        if len(digest) != 64 or any(char not in '0123456789abcdef' for char in digest):
            raise ValueError('samples.csv contains an invalid SHA256 checksum.')
        try:
            age = float(row['age'])
            if not math.isfinite(age) or not 0 <= age <= 100:
                raise ValueError()
            for field, classes in (('view', 2), ('density', 4), ('scanner', 5)):
                if str(int(row[field])) != row[field] or not 0 <= int(row[field]) < classes:
                    raise ValueError()
        except ValueError as error:
            raise ValueError('samples.csv contains invalid age/view/density/scanner metadata.') from error
        if task == 'real':
            real[index] = row
        if task != 'counterfactual' and (row['source'] or row['target']):
            raise ValueError('Only counterfactual rows can specify source/target interventions.')
    if set(real) != expected_indices:
        raise ValueError('The cache must contain exactly one real image for every selected input.')
    arguments = config.get('arguments', {})
    tasks = set(arguments.get('tasks', ()))
    if not tasks or not tasks.issubset({'generation', 'reconstruction', 'counterfactual'}):
        raise ValueError('Sampling configuration has invalid requested tasks.')
    required = ['real']
    if 'generation' in tasks:
        required.append('generation')
    if tasks & {'reconstruction', 'counterfactual'}:
        required.extend(['reconstruction_vae', 'reconstruction_cycle'])
    for index in indices:
        if any((index, task, 'none') not in lookup for task in required):
            raise ValueError(f'Cache is missing required outputs for input {index}.')
    requested_attributes = set(arguments.get('attributes', ()))
    counts = {attribute: 0 for attribute in arguments.get('attributes', ())}
    for row in records:
        index = int(row['index'])
        factual = real[index]
        if row['input_path'] != factual['file_path']:
            raise ValueError('An output refers to the wrong factual input image.')
        if row['task'] == 'counterfactual':
            attribute = row['attribute']
            if 'counterfactual' not in tasks or attribute not in requested_attributes:
                raise ValueError('Cache contains an unrequested counterfactual.')
            try:
                source, target = float(row['source']), float(row['target'])
            except ValueError as error:
                raise ValueError('Counterfactual source/target values must be numeric.') from error
            if (not math.isfinite(source) or not math.isfinite(target)
                    or math.isclose(source, target, abs_tol=1e-6)
                    or not math.isclose(source, float(factual[attribute]), abs_tol=1e-4)
                    or not math.isclose(target, float(row[attribute]), abs_tol=1e-4)):
                raise ValueError('Counterfactual intervention metadata is inconsistent.')
            counts[attribute] += 1
        for attribute in ATTRIBUTES:
            if row['task'] != 'counterfactual' or row['attribute'] != attribute:
                if not math.isclose(float(row[attribute]), float(factual[attribute]), abs_tol=1e-4):
                    raise ValueError('Output metadata unexpectedly changes a factual attribute.')
    if counts != config.get('eligible_interventions', {}):
        raise ValueError('Counterfactual image counts do not match the sampling configuration.')
    if config.get('reference_images_sha256') != reference_images_hash(records):
        raise ValueError('Factual-image checksums do not match the sampling configuration.')
    return config, selection, records
