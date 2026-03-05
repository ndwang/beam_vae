"""Preprocessing utilities for converting particle beam data to frequency maps."""

import numpy as np

# 15 unique 2D projections of 6D phase space (x, y, z, px, py, pz)
PLANE_MAP = {
    # Position-Position
    "x-y": (0, 1),
    "x-z": (0, 2),
    "y-z": (1, 2),
    # Position-Momentum
    "x-px": (0, 3),
    "x-py": (0, 4),
    "x-pz": (0, 5),
    "y-px": (1, 3),
    "y-py": (1, 4),
    "y-pz": (1, 5),
    "z-px": (2, 3),
    "z-py": (2, 4),
    "z-pz": (2, 5),
    # Momentum-Momentum
    "px-py": (3, 4),
    "px-pz": (3, 5),
    "py-pz": (4, 5),
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
    """
    particles = np.asarray(particles)
    if particles.ndim != 2 or particles.shape[1] != 6:
        raise ValueError(f"Expected (N, 6) array, got {particles.shape}")

    scales = np.std(particles, axis=0)  # (6,)

    maps = np.empty((len(PLANE_NAMES), bins, bins), dtype=np.float64)
    for ch, name in enumerate(PLANE_NAMES):
        i, j = PLANE_MAP[name]
        range_i = [-n_sigma * scales[i], n_sigma * scales[i]]
        range_j = [-n_sigma * scales[j], n_sigma * scales[j]]
        hist, _, _ = np.histogram2d(
            particles[:, i], particles[:, j],
            bins=bins, range=[range_i, range_j],
        )
        total = hist.sum()
        if total > 0:
            hist /= total
        maps[ch] = hist

    return maps, scales
