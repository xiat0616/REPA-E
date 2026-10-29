"""Streaming float64 moments and Fréchet distance for arbitrary image features.

The extractor defines the metric: DINO features give a Fréchet DINO distance,
not the conventional Inception FID. Small samples can yield biased estimates;
the result reports counts and rank deficiency instead of hiding that limitation.
"""
import math

import torch


class FeatureMoments:
    """Chan/Welford moments; storage is O(D²), independent of image count."""
    def __init__(self, dimension):
        if isinstance(dimension, bool) or int(dimension) != dimension or dimension < 1:
            raise ValueError('Feature dimension must be a positive integer.')
        self.dimension = int(dimension)
        self._count = 0
        self._mean = torch.zeros(self.dimension, dtype=torch.float64)
        self._m2 = torch.zeros((self.dimension, self.dimension), dtype=torch.float64)

    @property
    def count(self):
        return self._count

    @torch.no_grad()
    def update(self, features):
        if torch.is_tensor(features):
            values = features.detach().to(device='cpu', dtype=torch.float64)
        else:
            values = torch.as_tensor(features, device='cpu', dtype=torch.float64)
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError(f'Expected Nx{self.dimension} features, got {tuple(values.shape)}.')
        if not torch.isfinite(values).all():
            raise ValueError('Features contain NaN/Inf.')
        batch_count = values.shape[0]
        if batch_count == 0:
            return
        batch_mean = values.mean(0)
        centered = values - batch_mean
        batch_m2 = centered.T @ centered
        total = self._count + batch_count
        if self._count:
            delta = batch_mean - self._mean
            new_mean = self._mean + delta * (batch_count / total)
            new_m2 = (self._m2 + batch_m2
                      + torch.outer(delta, delta) * (self._count * batch_count / total))
        else:
            new_mean, new_m2 = batch_mean, batch_m2
        if not torch.isfinite(new_mean).all() or not torch.isfinite(new_m2).all():
            raise ValueError('Feature moment accumulation overflowed float64.')
        self._count, self._mean, self._m2 = total, new_mean, new_m2

    def mean_covariance(self):
        if self.count < 2:
            raise ValueError('At least two feature vectors are required for sample covariance.')
        covariance = self._m2 / (self.count - 1)
        covariance = (covariance + covariance.T) * .5
        return self._mean.clone(), covariance

    def state_dict(self):
        """Return torch.save-compatible CPU tensors; covariance is None for n<2."""
        covariance = self.mean_covariance()[1] if self.count >= 2 else None
        return {'n': self.count, 'feature_dim': self.dimension,
                'mean': self._mean.clone() if self.count else None,
                'covariance': covariance}


def _psd_eigh(matrix, name):
    matrix = (matrix + matrix.T) * .5
    if not torch.isfinite(matrix).all():
        raise ValueError(f'{name} contains NaN/Inf.')
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    scale = float(eigenvalues.abs().max())
    # Relative tolerances preserve genuinely small-scale covariance matrices.
    negative_tolerance = scale * 1e-10
    if float(eigenvalues.min()) < -negative_tolerance:
        raise ValueError(f'{name} is materially non-positive-semidefinite: '
                         f'min eigenvalue={float(eigenvalues.min()):.6g}.')
    eigenvalues = eigenvalues.clamp_min(0)
    # Remove floating-point null-space noise before square roots. Without this,
    # rank-deficient products can turn O(eps) eigenvalues into O(sqrt(eps)) error.
    eigenvalues[eigenvalues <= scale * 1e-12] = 0
    return eigenvalues, eigenvectors


@torch.no_grad()
def frechet_distance(real, fake):
    """Squared mean distance + covariance Bures distance, using symmetric eigensolvers.

    Covariances are unbiased sample estimates (n-1 denominator). The numeric
    formula supports singular covariance; at least two images per set are needed.
    A computable low-sample score is still a biased, potentially noisy estimate.
    """
    if real.dimension != fake.dimension:
        raise ValueError('Real and generated feature dimensions differ.')
    dimension = real.dimension
    result = {'score': None, 'status': 'insufficient_samples',
              'n_real': real.count, 'n_generated': fake.count,
              'feature_dim': dimension,
              'rank_real': None, 'rank_generated': None,
              'low_sample_count': real.count <= dimension or fake.count <= dimension,
              'rank_deficient': real.count <= dimension or fake.count <= dimension}
    if real.count < 2 or fake.count < 2:
        return result
    mean_real, covariance_real = real.mean_covariance()
    mean_fake, covariance_fake = fake.mean_covariance()
    eigen_real, vectors_real = _psd_eigh(covariance_real, 'Real covariance')
    eigen_fake, vectors_fake = _psd_eigh(covariance_fake, 'Generated covariance')
    rank_real, rank_fake = int((eigen_real > 0).sum()), int((eigen_fake > 0).sum())
    result.update(rank_real=rank_real, rank_generated=rank_fake,
                  rank_deficient=rank_real < dimension or rank_fake < dimension)
    mean_distance = float((mean_real - mean_fake).square().sum())
    trace_real, trace_fake = float(eigen_real.sum()), float(eigen_fake.sum())
    if not all(math.isfinite(value) for value in (mean_distance, trace_real, trace_fake)):
        raise ValueError('Fréchet mean/covariance terms overflowed float64.')
    if rank_real == 0 or rank_fake == 0:
        trace_root = 0.
    elif torch.equal(covariance_real, covariance_fake):
        # Exact self-comparison should be zero even with a singular covariance.
        trace_root = trace_real
    else:
        # Form sqrt(C_real) C_fake sqrt(C_real) in the positive eigenspace of
        # C_real. Factoring C_fake also removes its numerical null-space noise.
        active_real, active_fake = eigen_real > 0, eigen_fake > 0
        overlap = vectors_real[:, active_real].T @ vectors_fake[:, active_fake]
        factor = (eigen_real[active_real].sqrt()[:, None] * overlap
                  * eigen_fake[active_fake].sqrt()[None, :])
        # Nuclear norm equals tr(sqrt(sqrt(C_real) C_fake sqrt(C_real))).
        # SVD avoids squaring the condition number or discarding genuine small
        # covariance modes through a second eigenvalue cutoff on factor@factor.T.
        trace_root = float(torch.linalg.svdvals(factor).sum())
    score = mean_distance + trace_real + trace_fake - 2 * trace_root
    cancellation_scale = mean_distance + trace_real + trace_fake + 2 * trace_root
    if not math.isfinite(score) or not math.isfinite(cancellation_scale):
        raise ValueError('Fréchet distance calculation overflowed float64.')
    if score < -1e-8 * cancellation_scale:
        raise ArithmeticError(f'Fréchet distance is materially negative: {score:.6g}.')
    result.update(score=max(0., score), status='ok')
    return result
