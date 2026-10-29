"""Frozen distribution-metric feature extractors and metadata grouping."""
import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F

from frechet_metrics import FeatureMoments


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class DinoFeatures(torch.nn.Module):
    """Local frozen DINOv2 CLS features; no teacher wrappers or downloads."""
    def __init__(self, model_name, repo, weights, device):
        super().__init__()
        if not Path(repo).is_dir() or not Path(weights).is_file():
            raise FileNotFoundError('DINO requires a local repository and pretrained weights; see --dino-repo/--dino-weights.')
        self.encoder = torch.hub.load(str(repo), model_name, source='local', pretrained=False)
        state = torch.load(weights, map_location='cpu', weights_only=False)
        self.encoder.load_state_dict(state, strict=True)
        self.encoder = self.encoder.float().to(device).eval().requires_grad_(False)
        self.dimension = self.encoder.embed_dim
        self.register_buffer('mean', torch.tensor([.485, .456, .406], device=device)[None, :, None, None])
        self.register_buffer('std', torch.tensor([.229, .224, .225], device=device)[None, :, None, None])

    @torch.no_grad()
    def forward(self, images):
        if images.ndim != 4 or images.shape[1] != 1 or not torch.isfinite(images).all():
            raise ValueError('Feature images must be finite Bx1xHxW grayscale tensors.')
        if images.min() < 0 or images.max() > 1:
            raise ValueError('Feature images must be in [0,1].')
        # Quantize real and generated images identically before feature resizing.
        images = (images.float() * 255).round().clamp(0, 255) / 255
        images = F.interpolate(images.repeat(1, 3, 1, 1), size=(224, 224), mode='bicubic',
                               align_corners=False, antialias=True)
        features = self.encoder.forward_features((images - self.mean) / self.std)['x_norm_clstoken']
        # Layer-normalized CLS is used directly, with no extra L2 normalization.
        if features.ndim != 2 or features.shape[1] != self.dimension or not torch.isfinite(features).all():
            raise ValueError('Invalid DINO CLS features.')
        return features.float()


class InceptionFeatures(torch.nn.Module):
    """Conventional FID pool3 features from the TensorFlow-compatible Inception."""
    dimension = 2048

    def __init__(self, weights, device):
        super().__init__()
        try:
            from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
        except ImportError as error:
            raise ImportError('Inception FID requires torch-fidelity==0.3.0; install requirements-uv.txt.') from error
        if not Path(weights).is_file():
            raise FileNotFoundError(f'FID Inception weights not found: {weights}. Set --inception-weights to the converted FID checkpoint.')
        self.encoder = FeatureExtractorInceptionV3(
            name='inception-v3-compat', features_list=['2048'],
            feature_extractor_weights_path=str(weights),
        ).float().to(device).eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, images):
        if images.ndim != 4 or images.shape[1] != 1 or not torch.isfinite(images).all():
            raise ValueError('Feature images must be finite Bx1xHxW grayscale tensors.')
        if images.min() < 0 or images.max() > 1:
            raise ValueError('Feature images must be in [0,1].')
        images = (images.float().repeat(1, 3, 1, 1) * 255).round().clamp(0, 255).to(torch.uint8)
        # The backend handles TF-compatible bilinear299 resizing and scaling.
        # No external resize or ImageNet normalization is applied here.
        features = self.encoder(images)[0]
        if features.shape != (images.shape[0], self.dimension) or not torch.isfinite(features).all():
            raise ValueError('Invalid Inception pool3 features.')
        return features.float()


def target_groups(values, attribute, age_bin_edges):
    if attribute != 'age':
        return values.argmax(1).long()
    years = values[:, 0] * 100
    edges = torch.tensor(age_bin_edges, device=years.device, dtype=years.dtype)
    groups = torch.bucketize(years.contiguous(), edges, right=True) - 1
    if (groups < 0).any() or (groups >= len(age_bin_edges) - 1).any():
        raise ValueError('An age is outside --age-bin-edges; extend the edges to cover it.')
    return groups


def group_label(attribute, group, edges):
    return f'[{edges[group]:g},{edges[group+1]:g})years' if attribute == 'age' else str(group)


def update_groups(container, features, values, attribute, args):
    groups = target_groups(values, attribute, args.age_bin_edges)
    for group in groups.unique().tolist():
        key = (attribute, int(group))
        container.setdefault(key, FeatureMoments(features.shape[1])).update(features[groups == group])

