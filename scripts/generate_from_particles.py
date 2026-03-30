#!/usr/bin/env python
"""Convert particle beam data files to frequency map datasets.

Reads particle arrays (N, 6) from .npy files and produces maps + scales .npy files
suitable for VAE training.

Usage:
    python scripts/generate_from_particles.py --input data/particles/ --output data/dataset
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocessing import particles_to_frequency_maps


def main():
    parser = argparse.ArgumentParser(description="Convert particle beam data to frequency maps")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Path to directory of .npy particle files or single .npy file")
    parser.add_argument("--output", "-o", type=str, default="data/particles_dataset",
                        help="Output base path (no extension). Produces <path>_maps.npy and <path>_scales.npy")
    parser.add_argument("--bins", type=int, default=64, help="Grid resolution per axis")
    parser.add_argument("--n-sigma", type=float, default=4.0, help="Grid extent in sigma units")
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.is_dir():
        particle_files = sorted(input_path.glob("*.npy"))
    else:
        particle_files = [input_path]

    if not particle_files:
        print(f"No .npy files found at {args.input}")
        sys.exit(1)

    all_maps = []
    all_scales = []
    all_centroids = []

    for pf in particle_files:
        particles = np.load(pf)
        if particles.ndim != 2 or particles.shape[1] != 6:
            print(f"Skipping {pf}: expected (N, 6), got {particles.shape}")
            continue
        maps, scales, centroids = particles_to_frequency_maps(particles, bins=args.bins, n_sigma=args.n_sigma)
        all_maps.append(maps.astype(np.float32))
        all_scales.append(scales.astype(np.float32))
        all_centroids.append(centroids.astype(np.float32))

    if not all_maps:
        print("No valid particle files processed")
        sys.exit(1)

    all_maps = np.stack(all_maps, axis=0)
    all_scales = np.stack(all_scales, axis=0)
    all_centroids = np.stack(all_centroids, axis=0)

    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    maps_path = f"{args.output}_maps.npy"
    scales_path = f"{args.output}_scales.npy"
    centroids_path = f"{args.output}_centroids.npy"
    np.save(maps_path, all_maps)
    np.save(scales_path, all_scales)
    np.save(centroids_path, all_centroids)
    print(f"Saved {len(all_maps)} samples: {maps_path}, {scales_path}, {centroids_path}")


if __name__ == "__main__":
    main()
