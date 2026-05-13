#!/usr/bin/env python
"""Benchmark inference throughput and memory for a single VAE checkpoint.

Produces (in <run_dir>/benchmark/ by default):
  results.json    — all BenchmarkResult records (loadable by load_results)
  throughput.png  — samples/s vs batch size; encode and decode
  latency.png     — mean latency (ms) vs batch size
  memory.png      — peak GPU memory (MB) vs batch size

Usage:
    python scripts/benchmark.py runs/<run>/vae_best.pth
    python scripts/benchmark.py runs/<run>/vae_best.pth --batch-sizes 1 8 64 256
    python scripts/benchmark.py runs/<run>/vae_best.pth --dtype bfloat16
    python scripts/benchmark.py runs/<run>/vae_best.pth --output benchmark/my_run/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from beam_vae.utils.checkpoint import load_checkpoint
from beam_vae.benchmark import (BenchmarkResult, benchmark_model, save_results)

DEFAULT_BATCH_SIZES = [1, 4, 16, 64, 256]

_MODE_COLOR = {"encode": "C0", "decode": "C1"}
_MODE_LS    = {"encode": "-",  "decode": "--"}
_MODE_LABEL = {"encode": "Encode", "decode": "Decode"}


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark inference for a single checkpoint")
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=DEFAULT_BATCH_SIZES,
                   metavar="N")
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    p.add_argument("--n-warmup", type=int, default=20)
    p.add_argument("--n-repeat", type=int, default=50)
    p.add_argument("--output", type=Path, default=None,
                   help="Output directory (default: <run_dir>/benchmark/)")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def _plot_metric(results: list[BenchmarkResult], attr: str,
                 ylabel: str, title: str, path: Path) -> None:
    modes = sorted({r.mode for r in results})
    fig, ax = plt.subplots(figsize=(7, 4))
    for mode in modes:
        rs = sorted([r for r in results if r.mode == mode], key=lambda r: r.batch_size)
        ax.plot([r.batch_size for r in rs], [getattr(r, attr) for r in rs],
                color=_MODE_COLOR[mode], linestyle=_MODE_LS[mode],
                marker="o", label=_MODE_LABEL[mode])
    ax.set_xlabel("Batch size")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xscale("log", base=2)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def _print_table(results: list[BenchmarkResult]) -> None:
    header = f"\n{'Mode':<10} {'BS':>5} {'Latency(ms)':>15} {'Tput(samp/s)':>14} {'Mem(MB)':>9}"
    print(header)
    print("-" * len(header.rstrip()))
    for r in results:
        print(f"{r.mode:<10} {r.batch_size:>5}  "
              f"{r.latency_mean_ms:>8.2f} ±{r.latency_std_ms:>5.2f}  "
              f"{r.throughput_sps:>12.1f}  {r.peak_memory_mb:>7.0f}")


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype  = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    print(f"Device: {device}  dtype: {dtype}")

    model, model_name, label, config, ckpt = load_checkpoint(args.checkpoint, device)
    if dtype != torch.float32:
        model = model.to(dtype=dtype)
    run_name = args.checkpoint.parent.name
    print(f"Loaded: {label}  (epoch={ckpt.get('epoch','?')})")

    output_dir = args.output or (args.checkpoint.parent / "benchmark")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nBenchmarking ({args.n_warmup} warmup, {args.n_repeat} reps):")
    results = benchmark_model(
        model, model_name, run_name, label,
        batch_sizes=args.batch_sizes,
        device=device,
        dtype=dtype,
        n_warmup=args.n_warmup,
        n_repeat=args.n_repeat,
    )

    save_results(results, output_dir / "results.json")
    _print_table(results)

    _plot_metric(results, "throughput_sps",    "Samples / second",     f"Throughput — {label}", output_dir / "throughput.png")
    _plot_metric(results, "latency_mean_ms",   "Latency (ms)",         f"Latency — {label}",    output_dir / "latency.png")
    _plot_metric(results, "peak_memory_mb",    "Peak GPU memory (MB)", f"Memory — {label}",     output_dir / "memory.png")


if __name__ == "__main__":
    main()
