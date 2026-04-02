"""Preprocessing utilities for converting particle beam data to frequency maps."""

import numpy as np

# 15 unique 2D projections of 6D phase space
# Data ordering: (x, x', y, y', z, δ)
#   x  = horizontal position [m]
#   x' = horizontal angle (px/p_ref) [rad]
#   y  = vertical position [m]
#   y' = vertical angle (py/p_ref) [rad]
#   z  = longitudinal position, bunch-frame [m]
#   δ  = relative momentum deviation (pz-p_ref)/p_ref [dimensionless]
PLANE_MAP = {
    # On-diagonal phase spaces
    "x-x'": (0, 1),
    "y-y'": (2, 3),
    "z-d": (4, 5),
    # Position-position
    "x-y": (0, 2),
    "x-z": (0, 4),
    "y-z": (2, 4),
    # Cross-plane coupling
    "x-y'": (0, 3),
    "x-d": (0, 5),
    "x'-y": (1, 2),
    "x'-y'": (1, 3),
    "x'-z": (1, 4),
    "x'-d": (1, 5),
    "y-d": (2, 5),
    "y'-z": (3, 4),
    "y'-d": (3, 5),
}

# Sorted plane names define the canonical channel ordering
PLANE_NAMES = sorted(PLANE_MAP.keys())


def particles_to_frequency_maps(particles, bins=64, n_sigma=4):
    """Convert particle coordinates to 15-channel frequency maps with adaptive grids.

    Args:
        particles: (N, 6) array of particle coordinates [x, y, z, px, py, pz].
        bins: Number of histogram bins per axis.
        n_sigma: Grid extent in units of per-dimension standard deviation.

    Returns:
        maps: (15, bins, bins) array of normalized frequency maps.
        scales: (6,) array of per-dimension standard deviations.
        centroids: (6,) array of per-dimension means (beam orbit).
    """
    particles = np.asarray(particles)
    if particles.ndim != 2 or particles.shape[1] != 6:
        raise ValueError(f"Expected (N, 6) array, got {particles.shape}")

    centroids = np.mean(particles, axis=0)  # (6,)
    scales = np.std(particles, axis=0)  # (6,)

    # Center particles before histogramming so maps are always centered
    centered = particles - centroids

    maps = np.empty((len(PLANE_NAMES), bins, bins), dtype=np.float64)
    for ch, name in enumerate(PLANE_NAMES):
        i, j = PLANE_MAP[name]
        range_i = [-n_sigma * scales[i], n_sigma * scales[i]]
        range_j = [-n_sigma * scales[j], n_sigma * scales[j]]
        hist, _, _ = np.histogram2d(
            centered[:, i], centered[:, j],
            bins=bins, range=[range_i, range_j],
        )
        total = hist.sum()
        if total > 0:
            hist /= total
        maps[ch] = hist

    return maps, scales, centroids
