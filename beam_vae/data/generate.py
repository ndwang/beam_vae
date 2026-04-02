"""Analytic Gaussian frequency map generation and dataset creation.

Ported from old/beam_data.py with adaptive per-projection grids so that
generated maps are consistent with particles_to_frequency_maps().
"""

import argparse
import numpy as np

from .preprocessing import PLANE_MAP, PLANE_NAMES


def build_covariance(sigmas=None, corrs=None, generate_random=False, seed=None):
    """Build a 6x6 covariance matrix.

    Args:
        sigmas: (6,) array of per-dimension standard deviations.
        corrs: Flattened upper-triangle correlation coefficients.
        generate_random: If True, generate a random positive-definite covariance.
        seed: RNG seed.

    Returns:
        (6, 6) covariance matrix.
    """
    dim = 6
    if generate_random:
        rng = np.random.default_rng(seed)
        A = rng.normal(size=(dim, dim))
        Sigma = A @ A.T
        if sigmas is None:
            sigmas = rng.uniform(0.5, 2.0, size=dim)
        D = np.diag(sigmas / np.sqrt(np.diag(Sigma)))
        return D @ Sigma @ D
    else:
        sigmas = np.array(sigmas)
        if corrs is None:
            corrs = np.zeros(dim * (dim - 1) // 2)
        Sigma = np.diag(sigmas ** 2)
        idx = 0
        for i in range(dim):
            for j in range(i + 1, dim):
                Sigma[i, j] = corrs[idx] * sigmas[i] * sigmas[j]
                Sigma[j, i] = Sigma[i, j]
                idx += 1
        return Sigma


def gaussian_2d_density(X, Y, Sigma_2x2):
    """Compute normalized 2D Gaussian density on a meshgrid.

    Args:
        X, Y: 2D meshgrid arrays.
        Sigma_2x2: (2, 2) covariance sub-matrix.

    Returns:
        (bins, bins) density array normalized to sum=1.
    """
    inv_cov = np.linalg.inv(Sigma_2x2)
    det_cov = np.linalg.det(Sigma_2x2)
    pos = np.stack([X, Y], axis=-1)
    exp_term = np.einsum("...k,kl,...l->...", pos, inv_cov, pos)
    pdf = np.exp(-0.5 * exp_term) / (2 * np.pi * np.sqrt(det_cov))
    return pdf / pdf.sum()


def generate_frequency_maps_analytic(bins=64, Sigma=None, n_sigma=4, seed=None):
    """Generate 15-channel analytic Gaussian frequency maps with adaptive grids.

    Args:
        bins: Grid resolution per axis.
        Sigma: (6, 6) covariance matrix. If None, a random one is generated.
        n_sigma: Grid extent in units of per-dimension sigma.
        seed: RNG seed (used only when Sigma is None).

    Returns:
        maps: (15, bins, bins) array of normalized density maps.
        scales: (6,) array of per-dimension standard deviations.
        centroids: (6,) array of zeros (analytic maps are centered by construction).
    """
    if Sigma is None:
        Sigma = build_covariance(generate_random=True, seed=seed)

    scales = np.sqrt(np.diag(Sigma))  # (6,)
    centroids = np.zeros(6, dtype=np.float64)  # always centered

    maps = np.empty((len(PLANE_NAMES), bins, bins), dtype=np.float64)
    for ch, name in enumerate(PLANE_NAMES):
        i, j = PLANE_MAP[name]
        xi = np.linspace(-n_sigma * scales[i], n_sigma * scales[i], bins)
        xj = np.linspace(-n_sigma * scales[j], n_sigma * scales[j], bins)
        X, Y = np.meshgrid(xi, xj)
        Sigma_2x2 = Sigma[np.ix_([i, j], [i, j])]
        maps[ch] = gaussian_2d_density(X, Y, Sigma_2x2)

    return maps, scales, centroids


def generate_dataset(filename, n_samples=10000, bins=64, seed=42):
    """Generate a dataset of analytic Gaussian frequency maps.

    Saves two .npy files: one for maps (N, 15, bins, bins) and one for
    scales (N, 6).

    Args:
        filename: Base path (without extension). Produces <filename>_maps.npy
            and <filename>_scales.npy.
        n_samples: Number of samples to generate.
        bins: Grid resolution per axis.
        seed: Master RNG seed.
    """
    rng = np.random.default_rng(seed)
    all_maps = np.empty((n_samples, len(PLANE_NAMES), bins, bins), dtype=np.float32)
    all_scales = np.empty((n_samples, 6), dtype=np.float32)
    all_centroids = np.empty((n_samples, 6), dtype=np.float32)

    for idx in range(n_samples):
        sample_seed = int(rng.integers(0, 2**31))
        m, s, c = generate_frequency_maps_analytic(bins=bins, seed=sample_seed)
        all_maps[idx] = m.astype(np.float32)
        all_scales[idx] = s.astype(np.float32)
        all_centroids[idx] = c.astype(np.float32)

    maps_path = f"{filename}_maps.npy"
    scales_path = f"{filename}_scales.npy"
    centroids_path = f"{filename}_centroids.npy"
    np.save(maps_path, all_maps)
    np.save(scales_path, all_scales)
    np.save(centroids_path, all_centroids)
    print(f"Saved {n_samples} samples: {maps_path}, {scales_path}, {centroids_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate analytic Gaussian frequency map dataset")
    parser.add_argument("--output", "-o", type=str, default="data/dataset", help="Output base path (no extension)")
    parser.add_argument("--n-samples", "-n", type=int, default=10000, help="Number of samples")
    parser.add_argument("--bins", type=int, default=64, help="Grid resolution")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args()
    generate_dataset(args.output, args.n_samples, args.bins, args.seed)
