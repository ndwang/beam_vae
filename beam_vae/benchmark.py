"""Inference benchmarking utilities: timing, memory, synthetic batches.

Imported by scripts/benchmark.py.
"""

from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

MODES = ["encode", "decode"]


@dataclass
class BenchmarkResult:
    model_name: str
    run_name: str
    label: str
    mode: str              # "encode" | "decode"
    batch_size: int
    latent_dim: int
    latency_mean_ms: float
    latency_std_ms: float
    latency_p95_ms: float
    throughput_sps: float        # batch_size / (latency_mean_ms / 1000)
    peak_memory_mb: float

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> BenchmarkResult:
        return cls(**d)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def _time_cuda(fn: Callable, n_warmup: int, n_repeat: int) -> dict:
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(n_repeat):
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    t = np.array(times)
    return dict(mean_ms=float(t.mean()), std_ms=float(t.std()),
                p50_ms=float(np.percentile(t, 50)), p95_ms=float(np.percentile(t, 95)))


def _time_cpu(fn: Callable, n_warmup: int, n_repeat: int) -> dict:
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    t = np.array(times)
    return dict(mean_ms=float(t.mean()), std_ms=float(t.std()),
                p50_ms=float(np.percentile(t, 50)), p95_ms=float(np.percentile(t, 95)))


def time_fn(fn: Callable, n_warmup: int = 20, n_repeat: int = 50) -> dict:
    """Time fn() with CUDA events (falls back to perf_counter on CPU).

    Returns dict: mean_ms, std_ms, p50_ms, p95_ms.
    """
    if torch.cuda.is_available():
        return _time_cuda(fn, n_warmup, n_repeat)
    return _time_cpu(fn, n_warmup, n_repeat)


def measure_peak_memory(fn: Callable, device: torch.device) -> float:
    """Run fn() once and return peak GPU memory allocated in MB (0.0 on CPU)."""
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.reset_peak_memory_stats(device)
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated(device) / 1e6


# ---------------------------------------------------------------------------
# Synthetic batches
# ---------------------------------------------------------------------------

def make_encode_inputs(model, batch_size: int,
                       device: torch.device, dtype: torch.dtype):
    """Build random encoder inputs (image, scales, centroids)."""
    C = model.input_channels
    S = model.input_size
    x         = torch.rand(batch_size, C, S, S, device=device, dtype=dtype)
    scales    = torch.rand(batch_size, model.n_scales, device=device, dtype=dtype)
    centroids = torch.randn(batch_size, model.n_centroids, device=device, dtype=dtype)
    return x, scales, centroids


def make_decode_inputs(model, batch_size: int,
                       device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Build random decoder input (latent z ~ N(0, I))."""
    return torch.randn(batch_size, model.latent_dim, device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Forward callables
# ---------------------------------------------------------------------------

def _make_fn(model, mode: str, x, scales, centroids, z) -> Callable:
    if mode == "encode":
        return lambda: model.encode(x, scales, centroids)
    if mode == "decode":
        return lambda: model.decode(z)
    raise ValueError(f"Unknown mode: {mode}")


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def benchmark_model(
    model,
    model_name: str,
    run_name: str,
    label: str,
    batch_sizes: list[int],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    n_warmup: int = 20,
    n_repeat: int = 50,
) -> list[BenchmarkResult]:
    """Run timing + memory benchmark for encode/decode × batch sizes.

    Each (mode, batch_size) pair gets its own warmup so CUDA kernels are
    compiled for the exact input shape before timing begins.
    """
    latent_dim = model.latent_dim
    results: list[BenchmarkResult] = []

    for batch_size in batch_sizes:
        x, scales, centroids = make_encode_inputs(model, batch_size, device, dtype)
        z = make_decode_inputs(model, batch_size, device, dtype)
        for mode in MODES:
            fn = _make_fn(model, mode, x, scales, centroids, z)
            timing   = time_fn(fn, n_warmup, n_repeat)
            peak_mb  = measure_peak_memory(fn, device)
            tput     = batch_size / (timing["mean_ms"] / 1e3)

            results.append(BenchmarkResult(
                model_name=model_name,
                run_name=run_name,
                label=label,
                mode=mode,
                batch_size=batch_size,
                latent_dim=latent_dim,
                latency_mean_ms=timing["mean_ms"],
                latency_std_ms=timing["std_ms"],
                latency_p95_ms=timing["p95_ms"],
                throughput_sps=tput,
                peak_memory_mb=peak_mb,
            ))
            print(f"  {label:<22} {mode:<8} bs={batch_size:4d}  "
                  f"{timing['mean_ms']:8.2f} ± {timing['std_ms']:.2f} ms  "
                  f"{tput:9.1f} samp/s  {peak_mb:6.0f} MB")

    return results


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------

def save_results(results: list[BenchmarkResult], path: Path) -> None:
    with open(path, "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    print(f"Saved {path}")


def load_results(path: Path) -> list[BenchmarkResult]:
    with open(path) as f:
        return [BenchmarkResult.from_dict(d) for d in json.load(f)]
