#!/usr/bin/env python
"""Generate synthetic Gaussian frequency map datasets.

Usage:
    python scripts/generate_analytic.py --output data/dataset --n-samples 10000
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beam_vae.data.generate import generate_dataset


def main():
    parser = argparse.ArgumentParser(description="Generate analytic Gaussian frequency map dataset")
    parser.add_argument("--output", "-o", type=str, default="data/dataset",
                        help="Output base path (no extension). Produces <path>_maps.npy and <path>_scales.npy")
    parser.add_argument("--n-samples", "-n", type=int, default=10000, help="Number of samples")
    parser.add_argument("--bins", type=int, default=64, help="Grid resolution per axis")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args()

    # Ensure output directory exists
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    generate_dataset(args.output, args.n_samples, args.bins, args.seed)


if __name__ == "__main__":
    main()
