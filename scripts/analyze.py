#!/usr/bin/env python
"""Analyze a trained VAE run.

Usage:
    python scripts/analyze.py runs/beta_0_260327_1910
    python scripts/analyze.py runs/beta_0_260327_1910 --only recon latent
    python scripts/analyze.py runs/beta_0_260327_1910 --n-samples 10000

Runs all analyses by default and saves plots to <run_dir>/analysis/.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from src.models import VAE2D, ResidualVAE2D
from src.data import FrequencyMapDataset
from src.physics import transverse_twiss_numpy
from src.utils.config import load_yaml


# ──────────────────────────────────────────────────────────────
# Common setup
# ──────────────────────────────────────────────────────────────

def load_run(run_dir):
    """Load config, model, and val dataset from a run directory."""
    run_dir = Path(run_dir)
    config_path = run_dir / "config.yaml"
    config = load_yaml(config_path)

    # Find best checkpoint
    best = list(run_dir.glob("*_best.pth"))
    if not best:
        raise FileNotFoundError(f"No *_best.pth checkpoint in {run_dir}")
    checkpoint_path = best[0]

    # Build model
    model_name = config.get("model", {}).get("name", "vae2d")
    model_cls = ResidualVAE2D if model_name == "residual_vae2d" else VAE2D
    model = model_cls(config)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    # Build norm_stats from model buffers (if they contain real stats)
    norm_stats = None
    if hasattr(model, 'scale_mean') and not torch.equal(model.scale_std, torch.ones_like(model.scale_std)):
        norm_stats = {
            'scale_mean': model.scale_mean,
            'scale_std': model.scale_std,
            'centroid_mean': model.centroid_mean,
            'centroid_std': model.centroid_std,
        }

    # Build val dataset
    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})
    seed = training_cfg.get("seed", 42)
    val_split = training_cfg.get("val_split", 0.1)

    full_dataset = FrequencyMapDataset(
        data_cfg["path"],
        data_cfg["scales_path"],
        data_cfg.get("centroids_path"),
        norm_stats=norm_stats,
    )
    n = len(full_dataset)
    val_size = int(val_split * n)
    train_size = n - val_size
    _, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    print(f"Run:        {run_dir.name}")
    print(f"Checkpoint: {checkpoint_path.name}")
    print(f"Dataset:    {n} total, {val_size} val")
    return config, model, val_dataset


@torch.no_grad()
def encode_samples(model, dataset, n_samples, batch_size=256):
    """Encode samples, returning maps, mu, logvar, scales, centroids as numpy arrays."""
    n = min(len(dataset), n_samples)
    subset = torch.utils.data.Subset(dataset, range(n))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
    has_norm = not torch.equal(model.scale_std, torch.ones_like(model.scale_std))
    all_maps, all_mu, all_logvar, all_scales, all_centroids = [], [], [], [], []
    for maps, scales, centroids in loader:
        mu, logvar = model.encode(maps, scales, centroids)
        all_maps.append(maps.numpy())
        all_mu.append(mu.cpu().numpy())
        all_logvar.append(logvar.cpu().numpy())
        if has_norm:
            all_scales.append(model.denormalize_scales(scales).cpu().numpy())
            all_centroids.append(model.denormalize_centroids(centroids).cpu().numpy())
        else:
            all_scales.append(scales.numpy())
            all_centroids.append(centroids.numpy())
    return (np.concatenate(all_maps), np.concatenate(all_mu), np.concatenate(all_logvar),
            np.concatenate(all_scales), np.concatenate(all_centroids))


@torch.no_grad()
def run_inference(model, dataset, n_samples, batch_size=256):
    """Run full forward pass, returning inputs, recons, pred/true scales and centroids."""
    n = min(len(dataset), n_samples)
    subset = torch.utils.data.Subset(dataset, range(n))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_maps, all_recons = [], []
    all_pred_s, all_true_s = [], []
    all_pred_c, all_true_c = [], []
    for maps, scales, centroids in loader:
        recon, pred_s, pred_c, mu, logvar = model(maps, scales, centroids)
        # Denormalize predictions and targets to physical space
        pred_s = model.denormalize_scales(pred_s)
        pred_c = model.denormalize_centroids(pred_c)
        if not torch.equal(model.scale_std, torch.ones_like(model.scale_std)):
            true_s = model.denormalize_scales(scales)
            true_c = model.denormalize_centroids(centroids)
        else:
            true_s = scales
            true_c = centroids
        all_maps.append(maps.numpy())
        all_recons.append(recon.cpu().numpy())
        all_pred_s.append(pred_s.cpu().numpy())
        all_true_s.append(true_s.cpu().numpy())
        all_pred_c.append(pred_c.cpu().numpy())
        all_true_c.append(true_c.cpu().numpy())
    return {
        "inputs": np.concatenate(all_maps),
        "recons": np.concatenate(all_recons),
        "pred_scales": np.concatenate(all_pred_s),
        "true_scales": np.concatenate(all_true_s),
        "pred_centroids": np.concatenate(all_pred_c),
        "true_centroids": np.concatenate(all_true_c),
    }


# ──────────────────────────────────────────────────────────────
# Analysis: Reconstruction
# ──────────────────────────────────────────────────────────────

def analyze_reconstruction(model, dataset, output_dir, n_vis=5, n_eval=50):
    """Per-channel MSE, spatial error heatmaps, side-by-side samples."""
    print("\n--- Reconstruction Quality ---")
    data = run_inference(model, dataset, n_eval)
    inputs, recons = data["inputs"], data["recons"]
    n_channels = inputs.shape[1]
    residuals = inputs - recons

    # Per-channel MSE
    per_ch_mse = (residuals ** 2).mean(axis=(0, 2, 3))
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(n_channels), per_ch_mse, color="steelblue", edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, per_ch_mse):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.2e}", ha="center", va="bottom", fontsize=7)
    ax.set_xlabel("Channel")
    ax.set_ylabel("MSE")
    ax.set_title(f"Per-Channel MSE ({inputs.shape[0]} val samples)")
    ax.set_xticks(range(n_channels))
    fig.tight_layout()
    fig.savefig(output_dir / "per_channel_mse.png", dpi=150)
    plt.close(fig)

    # Spatial error heatmap
    mean_abs_res = np.abs(residuals).mean(axis=0)
    ncols = 5
    nrows = int(np.ceil(n_channels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), constrained_layout=True)
    fig.suptitle(f"Spatial Mean Absolute Error ({inputs.shape[0]} val samples)")
    for ch in range(n_channels):
        ax = axes.flat[ch]
        im = ax.imshow(mean_abs_res[ch], cmap="hot", aspect="equal")
        ax.set_title(f"Ch {ch}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for idx in range(n_channels, len(axes.flat)):
        axes.flat[idx].set_visible(False)
    fig.savefig(output_dir / "spatial_error_heatmap.png", dpi=150)
    plt.close(fig)

    # Side-by-side for a few samples
    rng = np.random.default_rng(42)
    vis_idx = sorted(rng.choice(inputs.shape[0], size=min(n_vis, inputs.shape[0]), replace=False))
    for si in vis_idx:
        fig, axes = plt.subplots(n_channels, 3, figsize=(9, 3 * n_channels), constrained_layout=True)
        fig.suptitle(f"Sample {si}  |  Input / Reconstruction / Residual")
        inp, rec, res = inputs[si], recons[si], residuals[si]
        for ch in range(n_channels):
            vmin = min(inp[ch].min(), rec[ch].min())
            vmax = max(inp[ch].max(), rec[ch].max())
            axes[ch, 0].imshow(inp[ch], vmin=vmin, vmax=vmax, cmap="viridis")
            axes[ch, 0].set_ylabel(f"Ch {ch}", fontsize=8)
            axes[ch, 1].imshow(rec[ch], vmin=vmin, vmax=vmax, cmap="viridis")
            axes[ch, 2].imshow(res[ch], cmap="RdBu_r")
            for ax in axes[ch]:
                ax.set_xticks([])
                ax.set_yticks([])
        axes[0, 0].set_title("Input")
        axes[0, 1].set_title("Reconstruction")
        axes[0, 2].set_title("Residual")
        fig.savefig(output_dir / f"sidebyside_sample_{si}.png", dpi=120)
        plt.close(fig)

    print(f"  Overall MSE: {(residuals ** 2).mean():.2e}")
    print(f"  Best channel:  {per_ch_mse.argmin()} (MSE={per_ch_mse.min():.2e})")
    print(f"  Worst channel: {per_ch_mse.argmax()} (MSE={per_ch_mse.max():.2e})")


# ──────────────────────────────────────────────────────────────
# Analysis: Latent space structure
# ──────────────────────────────────────────────────────────────

def analyze_latent_space(model, dataset, output_dir, n_samples=5000):
    """PCA/UMAP of latent vectors colored by scales."""
    print("\n--- Latent Space Structure ---")
    all_maps, mu, _, scales, centroids = encode_samples(model, dataset, n_samples)
    n_scales = scales.shape[1]

    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(30, mu.shape[1]))
    coords = pca.fit_transform(mu)

    # PCA colored by each scale
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for i, ax in enumerate(axes.ravel()):
        if i >= n_scales:
            ax.set_visible(False)
            continue
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=scales[:, i],
                        cmap="viridis", s=4, alpha=0.6, rasterized=True)
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.set_title(f"Scale {i}")
        fig.colorbar(sc, ax=ax, shrink=0.8)
    fig.suptitle("PCA of Latent Space colored by scales")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_dir / "pca_by_scales.png", dpi=150)
    plt.close(fig)

    # PCA colored by Twiss parameters
    twiss = transverse_twiss_numpy(all_maps, scales)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for row, plane in enumerate(["x", "y"]):
        for col, param in enumerate(["emit", "alpha", "beta"]):
            ax = axes[row, col]
            key = f"{param}_{plane}"
            sc = ax.scatter(coords[:, 0], coords[:, 1], c=twiss[key],
                            cmap="viridis", s=4, alpha=0.6, rasterized=True)
            ax.set_xlabel("PC 1")
            ax.set_ylabel("PC 2")
            label = {"emit": "ε", "alpha": "α", "beta": "β"}[param]
            ax.set_title(f"{label}_{plane}")
            fig.colorbar(sc, ax=ax, shrink=0.8)
    fig.suptitle("PCA of Latent Space colored by Twiss parameters")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_dir / "pca_by_twiss.png", dpi=150)
    plt.close(fig)

    # PCA colored by each centroid
    n_centroids = centroids.shape[1]
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for i, ax in enumerate(axes.ravel()):
        if i >= n_centroids:
            ax.set_visible(False)
            continue
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=centroids[:, i],
                        cmap="coolwarm", s=4, alpha=0.6, rasterized=True)
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.set_title(f"Centroid {i}")
        fig.colorbar(sc, ax=ax, shrink=0.8)
    fig.suptitle("PCA of Latent Space colored by centroids")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_dir / "pca_by_centroids.png", dpi=150)
    plt.close(fig)

    # Explained variance
    evr = pca.explained_variance_ratio_
    n = len(evr)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(1, n + 1), evr, color="steelblue", label="Individual")
    ax.plot(range(1, n + 1), np.cumsum(evr), "o-", color="darkorange", markersize=4, label="Cumulative")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title(f"PCA Explained Variance (top {n} components)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "pca_explained_variance.png", dpi=150)
    plt.close(fig)

    print(f"  PC1: {evr[0]:.1%}, PC1-2: {evr[:2].sum():.1%}, PC1-10: {evr[:10].sum():.1%}")

    # UMAP (optional)
    try:
        from umap import UMAP
        print("  Computing UMAP...")
        embedding = UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.3).fit_transform(mu)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for i, ax in enumerate(axes):
            sc = ax.scatter(embedding[:, 0], embedding[:, 1], c=scales[:, i],
                            cmap="viridis", s=4, alpha=0.6, rasterized=True)
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.set_title(f"Scale {i}")
            fig.colorbar(sc, ax=ax, shrink=0.8)
        fig.suptitle("UMAP of Latent Space")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(output_dir / "umap_by_scales.png", dpi=150)
        plt.close(fig)
    except ImportError:
        print("  umap-learn not installed, skipping UMAP")


# ──────────────────────────────────────────────────────────────
# Analysis: Latent dimension utilization
# ──────────────────────────────────────────────────────────────

def analyze_latent_dims(model, dataset, output_dir, n_samples=5000):
    """Active vs dead dims, cumulative variance, logvar distribution."""
    print("\n--- Latent Dimension Utilization ---")
    _, mu, logvar, _, _ = encode_samples(model, dataset, n_samples)
    latent_dim = mu.shape[1]

    mu_var = np.var(mu, axis=0)
    logvar_mean = np.mean(logvar, axis=0)
    mu_var_sorted = np.sort(mu_var)[::-1]

    total_var = mu_var_sorted.sum()
    cumvar = np.cumsum(mu_var_sorted) / total_var * 100

    threshold = 0.01 * mu_var.max()
    n_active = int(np.sum(mu_var > threshold))
    dims_90 = int(np.searchsorted(cumvar, 90.0)) + 1
    dims_95 = int(np.searchsorted(cumvar, 95.0)) + 1
    dims_99 = int(np.searchsorted(cumvar, 99.0)) + 1

    print(f"  Active dims (var > 1% max): {n_active} / {latent_dim}")
    print(f"  Dims for 90/95/99% var:     {dims_90} / {dims_95} / {dims_99}")
    print(f"  Mean logvar:                {logvar_mean.mean():.4f}")

    fig_kw = dict(figsize=(10, 5), dpi=150)

    # Mu variance sorted
    fig, ax = plt.subplots(**fig_kw)
    ax.bar(range(latent_dim), mu_var_sorted, width=1.0, color="steelblue", edgecolor="none")
    ax.axhline(threshold, color="red", linestyle="--", linewidth=1, label=f"1% of max ({threshold:.4f})")
    ax.set_xlabel("Latent dimension (sorted)")
    ax.set_ylabel("Variance of mu")
    ax.set_title("Mu Variance per Latent Dimension")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "mu_variance_sorted.png")
    plt.close(fig)

    # Logvar mean sorted
    fig, ax = plt.subplots(**fig_kw)
    ax.bar(range(latent_dim), np.sort(logvar_mean), width=1.0, color="darkorange", edgecolor="none")
    ax.set_xlabel("Latent dimension (sorted)")
    ax.set_ylabel("Mean logvar")
    ax.set_title("Mean Log-Variance per Latent Dimension")
    fig.tight_layout()
    fig.savefig(output_dir / "logvar_mean_sorted.png")
    plt.close(fig)

    # Cumulative variance
    fig, ax = plt.subplots(**fig_kw)
    ax.plot(range(1, latent_dim + 1), cumvar, color="steelblue", linewidth=2)
    for pct, ndim, color in [(90, dims_90, "green"), (95, dims_95, "orange"), (99, dims_99, "red")]:
        ax.axhline(pct, color=color, linestyle=":", linewidth=1, alpha=0.7)
        ax.axvline(ndim, color=color, linestyle=":", linewidth=1, alpha=0.7)
        ax.annotate(f"{pct}% @ {ndim}", xy=(ndim, pct),
                    xytext=(ndim + latent_dim * 0.05, pct - 3),
                    fontsize=9, color=color, arrowprops=dict(arrowstyle="->", color=color))
    ax.set_xlabel("Number of dimensions")
    ax.set_ylabel("Cumulative variance (%)")
    ax.set_title("Cumulative Variance Explained")
    ax.set_xlim(1, latent_dim)
    ax.set_ylim(0, 102)
    fig.tight_layout()
    fig.savefig(output_dir / "cumulative_variance.png")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────
# Analysis: Scale & centroid predictions
# ──────────────────────────────────────────────────────────────

def _scatter_and_stats(pred, true, labels, title, output_dir, filename):
    """Shared scatter + MSE + error histogram for scale or centroid predictions."""
    n_dims = pred.shape[1]
    nrows = int(np.ceil(n_dims / 3))

    # Compute metrics
    mse = np.mean((pred - true) ** 2, axis=0)
    r2 = np.zeros(n_dims)
    for d in range(n_dims):
        ss_res = np.sum((true[:, d] - pred[:, d]) ** 2)
        ss_tot = np.sum((true[:, d] - true[:, d].mean()) ** 2)
        r2[d] = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Scatter
    fig, axes = plt.subplots(nrows, 3, figsize=(15, 5 * nrows))
    fig.suptitle(f"Predicted vs True {title}", fontsize=16, y=0.98)
    axes_flat = axes.flatten() if n_dims > 3 else [axes] if n_dims == 1 else axes
    if nrows == 1:
        axes_flat = axes
    else:
        axes_flat = axes.flatten()
    for d in range(n_dims):
        ax = axes_flat[d]
        ax.scatter(true[:, d], pred[:, d], alpha=0.3, s=5, color="steelblue")
        lims = [min(true[:, d].min(), pred[:, d].min()),
                max(true[:, d].max(), pred[:, d].max())]
        margin = (lims[1] - lims[0]) * 0.05 if lims[1] > lims[0] else 0.1
        lims = [lims[0] - margin, lims[1] + margin]
        ax.plot(lims, lims, "r--", linewidth=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{labels[d]} (R\u00b2={r2[d]:.4f})")
        ax.set_aspect("equal", adjustable="box")
    for d in range(n_dims, len(axes_flat)):
        axes_flat[d].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / f"scatter_{filename}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # MSE bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, mse, color="steelblue", edgecolor="black")
    for bar, val in zip(bars, mse):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.2e}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("MSE")
    ax.set_title(f"Per-Dimension MSE ({title})")
    fig.tight_layout()
    fig.savefig(output_dir / f"mse_{filename}.png", dpi=150)
    plt.close(fig)

    # Error distributions
    fig, axes = plt.subplots(nrows, 3, figsize=(15, 5 * nrows))
    fig.suptitle(f"Error Distribution ({title})", fontsize=16, y=0.98)
    if nrows == 1:
        axes_flat = axes
    else:
        axes_flat = axes.flatten()
    for d in range(n_dims):
        ax = axes_flat[d]
        errors = pred[:, d] - true[:, d]
        ax.hist(errors, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
        ax.axvline(0, color="red", linestyle="--")
        ax.set_xlabel("Error")
        ax.set_ylabel("Count")
        ax.set_title(f"{labels[d]} (mean={errors.mean():.2e}, std={errors.std():.2e})")
    for d in range(n_dims, len(axes_flat)):
        axes_flat[d].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / f"errors_{filename}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return mse, r2


def analyze_scales(output_dir, data):
    """Scale prediction scatter, MSE, error distributions, R²."""
    print("\n--- Scale Predictions ---")
    labels = [f"Scale {i}" for i in range(data["pred_scales"].shape[1])]
    mse, r2 = _scatter_and_stats(
        data["pred_scales"], data["true_scales"],
        labels, "Scales", output_dir, "scales",
    )
    for d in range(len(labels)):
        print(f"  {labels[d]}: MSE={mse[d]:.2e}, R\u00b2={r2[d]:.4f}")


def analyze_centroids(output_dir, data):
    """Centroid prediction scatter, MSE, error distributions, R²."""
    print("\n--- Centroid Predictions ---")
    n_dims = data["pred_centroids"].shape[1]
    labels = [f"Centroid {i}" for i in range(n_dims)]
    mse, r2 = _scatter_and_stats(
        data["pred_centroids"], data["true_centroids"],
        labels, "Centroids", output_dir, "centroids",
    )
    for d in range(n_dims):
        print(f"  {labels[d]}: MSE={mse[d]:.2e}, R\u00b2={r2[d]:.4f}")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

ALL_ANALYSES = ["recon", "latent", "dims", "scales", "centroids"]


def main():
    parser = argparse.ArgumentParser(description="Analyze a trained VAE run")
    parser.add_argument("run_dir", type=str, help="Path to run directory")
    parser.add_argument("--only", nargs="+", choices=ALL_ANALYSES,
                        help="Run only specific analyses")
    parser.add_argument("--n-samples", type=int, default=5000,
                        help="Max validation samples for encoding/inference")
    args = parser.parse_args()

    config, model, val_dataset = load_run(args.run_dir)
    output_dir = Path(args.run_dir) / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    analyses = args.only or ALL_ANALYSES

    if "recon" in analyses:
        analyze_reconstruction(model, val_dataset, output_dir)
    if "latent" in analyses:
        analyze_latent_space(model, val_dataset, output_dir, n_samples=args.n_samples)
    if "dims" in analyses:
        analyze_latent_dims(model, val_dataset, output_dir, n_samples=args.n_samples)

    # Run inference once for both scale and centroid analysis
    if "scales" in analyses or "centroids" in analyses:
        inference_data = run_inference(model, val_dataset, args.n_samples)
        if "scales" in analyses:
            analyze_scales(output_dir, inference_data)
        if "centroids" in analyses:
            analyze_centroids(output_dir, inference_data)

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
