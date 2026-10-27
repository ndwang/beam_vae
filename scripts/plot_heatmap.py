#!/usr/bin/env python
"""Generate the latent_dim × beta heatmap of val metrics.

Run from project root:
    python scripts/plot_heatmap.py
Output: runs/heatmap_dim_beta.png
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── data ──────────────────────────────────────────────────────────────────────
# Scan 8 — full latent_dim × beta grid at lr=5e-4 (the new stable default).
# Best-epoch (min val_total) values.
# Initial run 260510_2140: 20/28 cells; 8 crashed (torch _inductor cache race).
# Resubmit job 52849963 (260512_0804): completed all 8 remaining cells.

DIMS  = [64, 128, 256, 512]
BETAS = [1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4]

# (dim, beta) → (val_recon, val_scale, val_centroid)
DATA = {
    # dim=64
    (64,  1e-6): (1.01e-5, 3.55e-5, 7.62e-5),  # KL=20.0 (underregularized)
    (64,  2e-6): (8.9e-6,  1.77e-5, 4.97e-5),  # KL=1.74  ★ best scale+centroid
    (64,  5e-6): (9.6e-6,  3.74e-5, 1.23e-4),  # KL=1.40
    (64,  1e-5): (1.08e-5, 7.70e-5, 5.90e-4),
    (64,  2e-5): (1.08e-5, 1.35e-4, 1.39e-3),
    (64,  5e-5): (1.26e-5, 1.13e-3, 7.87e-3),
    (64,  1e-4): (1.18e-5, 4.20e-3, 2.50e-2),  # KL=0.46 (over-regularized)

    # dim=128
    (128, 1e-6): (1.18e-5, 2.04e-5, 3.68e-5),  # KL=26.7 (underregularized)
    (128, 2e-6): (9.0e-6,  2.02e-5, 3.36e-5),  # KL=15.9 (underregularized)
    (128, 5e-6): (1.08e-5, 1.82e-5, 4.72e-5),  # KL=0.77 — from 260505 lr=1e-3 (lr=5e-4 run didn't converge in 500ep)
    (128, 1e-5): (9.62e-6, 6.05e-5, 2.87e-4),  # KL=0.62 — from 260427 lr=1e-3
    (128, 2e-5): (9.7e-6,  3.75e-5, 1.00e-3),
    (128, 5e-5): (1.10e-5, 2.23e-4, 2.54e-3),
    (128, 1e-4): (1.12e-5, 1.09e-3, 9.87e-3),  # KL=0.31

    # dim=256
    (256, 1e-6): (9.9e-6,  2.51e-5, 7.72e-5),  # KL=2.06
    (256, 2e-6): (9.2e-6,  3.04e-5, 9.81e-5),  # KL=1.72 — previously collapsed at lr=1e-3
    (256, 5e-6): (8.9e-6,  3.34e-5, 5.83e-5),  # KL=0.59
    (256, 1e-5): (9.1e-6,  3.60e-5, 6.54e-5),
    (256, 2e-5): (8.9e-6,  4.38e-5, 2.16e-4),
    (256, 5e-5): (1.08e-5, 6.17e-5, 9.60e-4),
    (256, 1e-4): (1.12e-5, 2.50e-4, 2.10e-3),  # KL=0.20

    # dim=512
    (512, 1e-6): (1.09e-5, 3.53e-5, 8.75e-5),
    (512, 2e-6): (1.06e-5, 3.66e-5, 9.40e-5),
    (512, 5e-6): (1.10e-5, 2.79e-5, 1.03e-4),  # KL=0.48
    (512, 1e-5): (9.9e-6,  4.12e-5, 5.23e-5),
    (512, 2e-5): (9.5e-6,  3.49e-5, 4.54e-5),
    (512, 5e-5): (9.8e-6,  5.15e-5, 3.68e-4),
    (512, 1e-4): (9.2e-6,  7.13e-5, 7.58e-4),  # KL=0.13
}

SPECIAL = {}

# ── build arrays ──────────────────────────────────────────────────────────────
nrows, ncols = len(DIMS), len(BETAS)
recon    = np.full((nrows, ncols), np.nan)
scale    = np.full((nrows, ncols), np.nan)
centroid = np.full((nrows, ncols), np.nan)

for ri, d in enumerate(DIMS):
    for ci, b in enumerate(BETAS):
        if (d, b) in DATA:
            r, s, c = DATA[(d, b)]
            recon[ri, ci]    = r
            scale[ri, ci]    = s
            centroid[ri, ci] = c

# ── plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(24, 5))
fig.suptitle("Val loss heatmaps: latent_dim × beta @ lr=5e-4 (Scan 8)  (lower = better, log scale)",
             fontsize=13)

beta_labels = ["1e-6", "2e-6", "5e-6", "1e-5", "2e-5", "5e-5", "1e-4"]
dim_labels  = [str(d) for d in DIMS]

panels = [
    ("Reconstruction loss", recon),
    ("Scale loss",          scale),
    ("Centroid loss",       centroid),
]

for ax, (title, mat) in zip(axes, panels):
    valid = mat[~np.isnan(mat)]
    vmin = valid.min()
    vmax = np.percentile(valid, 85)
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    cmap = plt.cm.RdYlGn_r

    im = ax.imshow(mat, norm=norm, cmap=cmap, aspect="auto")
    plt.colorbar(im, ax=ax, extend="max")

    ax.set_xticks(range(ncols))
    ax.set_xticklabels(beta_labels)
    ax.set_yticks(range(nrows))
    ax.set_yticklabels(dim_labels)
    ax.set_xlabel("beta")
    ax.set_ylabel("latent_dim")
    ax.set_title(title)

    for ri, d in enumerate(DIMS):
            for ci, b in enumerate(BETAS):
                v = mat[ri, ci]
                special = SPECIAL.get((d, b))
                if special == "unreg":
                    ax.text(ci, ri, "unregularized", ha="center", va="center",
                            fontsize=6, color="#ff4444", fontweight="bold")
                elif special == "crash":
                    ax.text(ci, ri, "NaN", ha="center", va="center",
                            fontsize=8, color="gray", fontweight="bold")
                elif special == "t/o":
                    ax.text(ci, ri, "t/o", ha="center", va="center",
                            fontsize=8, color="gray", fontweight="bold")
                elif special == "pending":
                    ax.text(ci, ri, "pending", ha="center", va="center",
                            fontsize=7, color="gray", fontweight="bold",
                            fontstyle="italic")
                elif np.isnan(v):
                    ax.text(ci, ri, "—", ha="center", va="center",
                            fontsize=8, color="gray", fontweight="bold")
                else:
                    txt = f"{v:.1e}"
                    normed = (np.log10(v) - np.log10(vmin)) / (np.log10(vmax) - np.log10(vmin))
                    color = "white" if normed > 0.5 else "black"
                    ax.text(ci, ri, txt, ha="center", va="center",
                            fontsize=8, color=color, fontweight="bold")

plt.tight_layout()
out = "runs/heatmap_dim_beta_scan8.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
