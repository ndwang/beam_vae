"""Beam physics computations on frequency maps.

All functions operate on torch tensors and are differentiable,
so they can be used in physics-informed losses.
"""

import torch
import numpy as np

from .data.preprocessing import PLANE_MAP, PLANE_NAMES


# Conjugate phase space pairs for transverse Twiss
TRANSVERSE_PLANES = [
    (0, 1, "x"),   # x-x'
    (2, 3, "y"),   # y-y'
]


def _get_channel_index(dim_u, dim_v):
    """Find the channel index for a given coordinate pair."""
    pair = (dim_u, dim_v)
    for name in PLANE_NAMES:
        if PLANE_MAP[name] == pair:
            return PLANE_NAMES.index(name)
    raise ValueError(f"No channel found for coordinate pair ({dim_u}, {dim_v})")


def second_moments(maps, scales, dim_u, dim_v, n_sigma=4):
    """Compute second moments <u²>, <v²>, <uv> from a 2D frequency map.

    Args:
        maps: (N, 15, bins, bins) frequency maps (normalized to sum=1 per channel).
        scales: (N, 6) per-dimension standard deviations.
        dim_u: First coordinate index (0-5).
        dim_v: Second coordinate index (0-5).
        n_sigma: Grid extent used during histogram generation.

    Returns:
        uu, vv, uv: Each (N,) tensor of second moments.
    """
    ch = _get_channel_index(dim_u, dim_v)
    bins = maps.shape[-1]
    H = maps[:, ch]  # (N, bins, bins)

    # Pixel centers -> physical coordinates
    pix = (torch.arange(bins, device=maps.device, dtype=maps.dtype) + 0.5) / bins

    s_u = scales[:, dim_u]  # (N,)
    s_v = scales[:, dim_v]  # (N,)

    u = pix[None, :, None] * 2 * n_sigma * s_u[:, None, None] - n_sigma * s_u[:, None, None]
    v = pix[None, None, :] * 2 * n_sigma * s_v[:, None, None] - n_sigma * s_v[:, None, None]

    uu = (H * u ** 2).sum(dim=(1, 2))
    vv = (H * v ** 2).sum(dim=(1, 2))
    uv = (H * u * v).sum(dim=(1, 2))

    return uu, vv, uv


def emittance(maps, scales, dim_u, dim_v, n_sigma=4):
    """Compute RMS emittance for a conjugate coordinate pair.

    ε = sqrt(<u²><v²> - <uv>²)

    Args:
        maps: (N, 15, bins, bins) frequency maps.
        scales: (N, 6) per-dimension standard deviations.
        dim_u: Position coordinate index.
        dim_v: Momentum/angle coordinate index.
        n_sigma: Grid extent used during histogram generation.

    Returns:
        (N,) tensor of emittance values.
    """
    uu, vv, uv = second_moments(maps, scales, dim_u, dim_v, n_sigma)
    return torch.sqrt((uu * vv - uv ** 2).clamp(min=0))


def twiss(maps, scales, dim_u, dim_v, n_sigma=4):
    """Compute Twiss parameters (ε, α, β) for a conjugate pair.

    Args:
        maps: (N, 15, bins, bins) frequency maps.
        scales: (N, 6) per-dimension standard deviations.
        dim_u: Position coordinate index.
        dim_v: Momentum/angle coordinate index.
        n_sigma: Grid extent used during histogram generation.

    Returns:
        Dict with 'emit', 'alpha', 'beta' keys, each (N,) tensor.
    """
    uu, vv, uv = second_moments(maps, scales, dim_u, dim_v, n_sigma)
    emit2 = (uu * vv - uv ** 2).clamp(min=0)
    emit = torch.sqrt(emit2)
    emit_safe = emit.clamp(min=1e-30)
    return {
        "emit": emit,
        "alpha": -uv / emit_safe,
        "beta": uu / emit_safe,
    }


def transverse_twiss(maps, scales, n_sigma=4):
    """Compute Twiss parameters for both transverse planes.

    Args:
        maps: (N, 15, bins, bins) frequency maps.
        scales: (N, 6) per-dimension standard deviations.
        n_sigma: Grid extent used during histogram generation.

    Returns:
        Dict with keys like 'emit_x', 'alpha_x', 'beta_x', 'emit_y', etc.
    """
    result = {}
    for dim_u, dim_v, label in TRANSVERSE_PLANES:
        t = twiss(maps, scales, dim_u, dim_v, n_sigma)
        for key, val in t.items():
            result[f"{key}_{label}"] = val
    return result


def transverse_twiss_numpy(maps, scales, n_sigma=4):
    """Numpy convenience wrapper for transverse_twiss."""
    maps_t = torch.from_numpy(maps).float()
    scales_t = torch.from_numpy(scales).float()
    with torch.no_grad():
        result = transverse_twiss(maps_t, scales_t, n_sigma)
    return {k: v.numpy() for k, v in result.items()}
