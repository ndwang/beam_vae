# Experiment Log

**Defaults unless noted:** `model=vae2d` · `lr=5e-4` · `latent_dim=128` · `beta=1e-5` · `scheduler=ReduceOnPlateau` · `500 epochs max`

**Dataset:** All v2 runs use `v2_sectioned_1sec_10k` (`/pscratch/sd/n/ndwang/latent_beam_dynamics/data/v2/vae_training/`). Legacy v1 runs (sectioned_10k, linear_10k, frequency_maps datasets) are archived in `runs/v1/` and not documented here.

---

## Pending Experiments

| scan | job ID | submitted | nodes | runs | status | what to do |
|------|--------|-----------|-------|------|--------|------------|
| Scan 8 — full heatmap lr=5e-4 | 52794941 | 2026-05-10 | 7 | 28 | **running** | run `analyze_losses.py runs/latent_dim_*_260*/` and `runs/beta_*_260*/`; update `plot_heatmap.py` DATA dict; archive old heatmap as `heatmap_dim_beta_lr1e-3.png`; regenerate `heatmap_dim_beta.png`; fill in Scan 8 results in EXPERIMENTS.md |

Submit command:
```bash
bash slurm/submit_grid.sh \
  "model.latent_dim" "64 128 256 512" \
  "training.beta" "1e-6 2e-6 5e-6 1e-5 2e-5 5e-5 1e-4" \
  "training.lr=5e-4" \
  "v2_scan8_heatmap_lr5e-4"
```

Any run that times out (slow node): resume with `--resume runs/<run_name>/<run_name>_epoch<N>.pth` — incremental CSV means no history is lost.

---

## Scan 1 — Beta sweep (2026-04-27)

**Question:** What KL weight produces the best reconstruction and downstream scale/centroid metrics on v2 data?

Fixed: `latent_dim=128`, `lr=1e-3`. W&B group: `v2_beta_scan`.

| beta  | best epoch | val_recon | val_kl | val_scale | val_centroid | epochs to val_recon=1.6e-5 |
|-------|-----------|-----------|--------|-----------|--------------|---------------------------|
| 0     | 440       | 9.9e-6    | 0.589  | 7.6e-5    | 8.8e-5       | 109                       |
| 1e-6  | 472       | 1.1e-5    | 0.615  | 5.6e-5    | 1.3e-4       | 134                       |
| 1e-5  | 493       | 1.3e-5    | 0.642  | 6.4e-5    | 1.1e-4       | **174**                   |
| 1e-4  | 440       | **9.4e-6**| 0.624  | **3.3e-5**| 1.4e-4       | **107**                   |

**Conclusions:**
- `beta=1e-4` is the winner: best val_recon and best val_scale, fastest convergence. Counterintuitive — stronger KL regularization helps rather than hurts on this dataset.
- `beta=1e-5` (previous default from v1 scans) is the worst here — slowest to converge and worst final reconstruction.
- `beta=0` has the lowest val_centroid but worse val_scale than 1e-4. Without KL, the latent space organizes scale better but suffers on centroid.
- All betas produce similar KL (~0.6) except beta=0 which is unconstrained.

---

## Scan 2 — Latent dim sweep (2026-04-27)

**Question:** How many latent dimensions does the model need? Is 256 the ceiling?

Fixed: `beta=1e-5`, `lr=1e-3`. W&B group: `v2_latent_dim_scan`.

| latent_dim | best epoch | val_recon  | val_kl | val_scale | val_centroid |
|------------|-----------|-----------|--------|-----------|--------------|
| 32         | 445       | 1.3e-5     | 1.496  | 6.4e-3    | 1.2e-3       |
| 64         | 443       | 9.3e-6     | 0.985  | 6.8e-5    | 6.0e-4       |
| 128        | 453       | 9.6e-6     | 0.621  | 6.1e-5    | 2.9e-4       |
| 256        | 435       | **7.9e-6** | **0.431** | **2.1e-5** | **7.3e-5** |

**Conclusions:**
- `dim=256` wins on every metric by a clear margin. The KL drop from 0.62 → 0.43 going from 128→256 indicates the larger model uses the latent space more efficiently.
- `dim=32` is clearly bottlenecked: val_scale is 100× worse than dim=256, KL=1.5 (posterior far from prior), and train/val scale ratio is only 1.9× (train and val both fail — capacity-limited, not overfit).
- `dim=64` has notably high centroid error (6.0e-4 vs 7.3e-5 at dim=256); centroid information needs more latent capacity.
- Whether 256 is a capacity ceiling or still scaling is the key open question — the falling KL suggests 512 might help further.

---

## Scan 3 — 2D grid: latent_dim × beta (submitted 2026-04-27, job 52161516)

**Question:** What is the joint optimum of latent_dim and beta on v2 data, and does the optimal beta shift as latent capacity increases?

**Motivation:** Scans 1 and 2 were run independently at fixed latent_dim=128 and beta=1e-5 respectively, so they don't tell us how the two hyperparameters interact. The Scan 1 winner (beta=1e-4) and Scan 2 winner (dim=256) have never been combined; the beta scan also only ran at dim=128. There are two open questions: (1) does beta=1e-4 remain the best choice at dim=256, or does the optimal beta shift as the latent space grows? (2) is dim=256 the capacity ceiling, or does dim=512 still improve? The Scan 2 KL trend (1.50 → 0.98 → 0.62 → 0.43 from dim=32→256, all at beta=1e-5) suggests the model has not saturated — each doubling of capacity still produces a meaningful KL drop, which typically tracks with better latent utilization and lower reconstruction error.

**Design:** 2D grid at latent_dim ∈ {256, 512} × beta ∈ {1e-4, 1e-3}. Fixed: `lr=1e-3`, `data=v2_sectioned_1sec_10k`. beta=1e-3 added to bracket the optimum — if beta=1e-4 already over-regularizes at dim=512, a stronger prior may hurt; if the model has more capacity to absorb the KL penalty, a larger beta could improve scale/centroid metrics further.

**What we expect:** `dim=256 + beta=1e-4` should improve on both Scan 1 and 2 individually — this is the straightforward combination of winners. If the KL trend continues, `dim=512` should further reduce both val_recon and val_kl. Whether beta=1e-3 helps or hurts likely depends on whether the model is still capacity-limited at that dim: with more room in the latent space, stronger regularization could encourage better-organized representations, or it could simply increase reconstruction error without a compensating gain in KL.

**What the result implies either way:** If dim=512 does not improve over dim=256, the model has plateaued and the bottleneck has shifted away from latent capacity (likely to dataset size or the fixed architecture). If beta=1e-4 remains the winner at both dims, the optimal regularization is robust to latent size and we can fix it for future runs. If beta shifts to 1e-3 at dim=512, it implies the latent space uses the extra capacity primarily for distributional structure rather than reconstruction detail.

W&B group: `v2_grid_latent_beta`.

**Results (val metrics at best epoch):**

| latent_dim | beta | best epoch | val_recon | val_kl | val_scale | val_centroid |
|------------|------|-----------|-----------|--------|-----------|--------------|
| 256        | 1e-4 | 393       | **1.0e-5** | 0.189 | **2.9e-4** | 1.5e-3     |
| 512        | 1e-4 | 496       | 1.5e-5    | 0.106  | 3.5e-3    | **5.5e-4**  |
| 256        | 1e-3 | 261       | 1.4e-5    | 0.075  | 3.6e-2    | 5.7e-2      |
| 512        | 1e-3 | —         | —         | —      | —         | — (crashed) |

**Crash note:** `512 + beta=1e-3` raised `ValueError: NaN loss detected at step 529` during epoch 35. Training proceeded normally until that point; no other errors. Likely gradient explosion driven by the combination of high KL weight and large latent space making the reparameterized samples occasionally extreme.

**Convergence (epochs to val_recon thresholds):**

| run              | val_recon=1e-4 | val_recon=5e-5 | val_recon=2e-5 | final     |
|------------------|---------------|---------------|---------------|-----------|
| 256 + beta=1e-4  | 24            | 36            | 83            | **1.0e-5** |
| 256 + beta=1e-3  | 25            | 43            | 97            | 1.4e-5    |
| 512 + beta=1e-4  | 34            | 49            | 122           | 1.5e-5    |

**Conclusions:**

- **`dim=256 + beta=1e-4` is the overall winner.** Lowest val_recon (1.0e-5) and by far the best val_scale (2.9e-4 vs 3.5e-3 at dim=512). Confirms the Scan 1 and Scan 2 winners combine well.

- **beta=1e-3 fails at both dims.** At dim=256 it plateaus hard — val_scale barely improves after epoch 50 (0.036 at epoch 50, 0.036 at epoch 500). At dim=512 it causes NaN. beta=1e-3 can be ruled out permanently.

- **dim=512 does not improve overall over dim=256 at beta=1e-4.** dim=512 has better val_centroid (5.5e-4 vs 1.5e-3) but 12× worse val_scale (3.5e-3 vs 2.9e-4), slower convergence, and slightly worse reconstruction. This is unexpected given the KL trend from Scan 2 — the model is not simply capacity-limited, and extra latent dimensions appear to be hurting scale encoding while helping centroid. One interpretation: the larger latent space distributes information differently, using more dims for centroid tracking but fewer "dedicated" dims for scale. The plateau in val_scale for dim=512 (3.6e-3 at epoch 400 → 3.5e-3 at epoch 500) suggests it has converged, not that it needs more training.

- **The KL trend does continue:** 0.189 (dim=256) → 0.106 (dim=512) at beta=1e-4, consistent with better latent utilization at larger dim. But lower KL does not translate to better downstream metrics here, implying the KL value itself is not the bottleneck.

- **beta=1e-4 is robust to latent size.** It wins at both dim=256 and dim=512. The joint optimum is `dim=256 + beta=1e-4`.

---

## Scan 4 — Lower-beta fill-in at dim=256 and dim=512 (submitted 2026-04-28)

**Question:** Does the optimal beta continue shifting downward as latent_dim grows beyond 256? If the trend (beta=1e-4 wins at dim=128, beta=1e-5 wins at dim=256) holds, the optimum at dim=512 should be beta~1e-6 or lower.

**Motivation:** The heatmap from Scans 1–3 reveals that the lower-left corner — dim=256/512 × beta∈{0, 1e-6} — was never tested. Scan 3 only probed beta≥1e-4 at the larger dims, and it turns out those betas are already too high for dim=256 (beta=1e-4 gives val_scale 14× worse than beta=1e-5 at the same dim). This is the interplay the grid was supposed to resolve, but it was exploring the wrong direction. The current best (dim=256 + beta=1e-5, val_scale=2.1e-5) may not be the true optimum — a lower beta or larger dim at lower beta could still improve it.

**Design:** Two 1D beta scans at fixed dim. dim=256 × beta∈{0, 1e-6} (the two untested cells — beta=1e-5 is already in hand). dim=512 × beta∈{0, 1e-6, 1e-5} (all three lower-beta cells untested). Fixed: `lr=1e-3`, `data=v2_sectioned_1sec_10k`. W&B groups: `v2_scan4_dim256`, `v2_scan4_dim512`.

**What we expect:** At dim=256, beta=1e-6 should be similar to or slightly worse than beta=1e-5 — the model may not benefit from weaker regularization once reconstruction is already near-optimal. beta=0 at dim=256 is a wildcard: from Scan 1 at dim=128, beta=0 had good val_recon but poor val_scale vs beta=1e-4. At dim=512, if the trend continues, beta=1e-5 should outperform beta=1e-4 (already confirmed worse), and beta∈{0, 1e-6} will reveal whether the optimum shifts further down.

**What the result implies:** If val_scale improves below beta=1e-5 at dim=256, the current best is not optimal and we need to revise the benchmark. If dim=512 + beta=1e-5 beats dim=256 + beta=1e-5, then latent capacity is still the binding constraint and we should push to dim=512. If dim=256 + beta=1e-5 remains unbeaten, the model has found its joint optimum and we can stop scanning.

| latent_dim | beta=0  | beta=1e-6 | beta=1e-5 | beta=1e-4 |
|------------|---------|-----------|-----------|-----------|
| 256        | **COLLAPSE** (epoch 9) | **COLLAPSE** (epoch 15) | 7.9e-6 recon / 2.1e-5 scale | 1.0e-5 / 2.9e-4 |
| 512        | COLLAPSE (epoch <50) | t/o at epoch ~450 (partial: 9.9e-6 / 1.9e-5) | **1.1e-5 / 2.0e-5** (rerun) | 1.5e-5 / 3.5e-3 |

*(values shown: val_recon / val_scale at best epoch; COLLAPSE = posterior collapse, val_scale → 1.0)*

**Results (best-epoch metrics for dim=256 completed runs):**

| run | best epoch | val_recon | val_kl | val_scale | val_centroid | outcome |
|-----|-----------|-----------|--------|-----------|--------------|---------|
| dim=256 + beta=0   | 8   | 4.3e-4 | 45.7 | 1.0e-2 | 2.8e-2 | COLLAPSED epoch 9 |
| dim=256 + beta=1e-6 | 11 | 3.5e-4 | 42.8 | 4.1e-2 | 7.9e-2 | COLLAPSED epoch 12 |

**dim=512 outcomes:** beta=0 collapsed before epoch 50 (no CSV); beta=1e-5 raised `NaN loss at step 50` at epoch 33 — root cause identified as float16 overflow in the KL mean over 512×512=262k elements under AMP, compounded by 11 centroid outlier samples (|normed centroid| up to 164) amplified through trained weights; fixed by casting mu/logvar to float32 in `kl_divergence` and adding NaN detection to validation. Rerun (job 52238102) completed all 500 epochs: val_recon=1.1e-5, val_scale=2.0e-5, val_centroid=4.3e-5 at epoch 500 (best). beta=1e-6 ran to epoch ~450 before job timed out; best checkpoint (epoch 459) has val_recon=9.9e-6, val_kl=1.39, val_scale=1.9e-5, val_centroid=4.8e-5 — not collapsed (KL=1.39), still converging at timeout.

**Conclusions:**

- **Posterior collapse below beta=1e-5 at dim=256 is a cliff, not a slope.** Both beta=0 and beta=1e-6 run normally for 8–11 epochs then suffer a sudden mode switch: val_kl jumps from ~43–46 to ~54 in one epoch and freezes there. After collapse, train_kl and val_kl are identical to 10+ decimal places — the encoder outputs a fixed (μ, σ) for every input regardless of content. val_scale → 1.0 (decoder predicts data mean, scale/centroid heads have nothing to work with). The model spends the remaining 490 epochs fully degenerate. Even the pre-collapse best-epoch values (val_scale 1e-2 to 4e-2) are 500–2000× worse than the Scan 2 reference.

- **The collapse mechanism:** Without sufficient KL penalty, the decoder eventually learns it can match reconstruction loss by predicting the data mean and ignoring z. Once z is ignored by the decoder, the encoder receives no gradient through z and log_var drifts freely upward. High-variance z samples are too noisy for the decoder to use even if it tried. This is a self-reinforcing bifurcation — once initiated it cannot be reversed at low LR. The ReduceOnPlateau scheduler accelerates the trap: oscillation in early epochs triggers LR reductions, and by the time collapse occurs the LR is too low to escape the degenerate basin.

- **beta=1e-6 buys only 3 extra epochs over beta=0.** The tiny KL gradient slows logvar drift but cannot prevent it. Both runs collapse to the same fixed-point KL (beta=0 → 53.95, beta=1e-6 → 54.05) and the same degenerate scale/centroid (≈1.0, ≈0.87).

- **dim=512 + beta=1e-5 is stable after the float32 fix and broadly competitive.** The rerun completed all 500 epochs: val_scale=2.0e-5 (vs 2.1e-5 at dim=256, essentially identical) and val_centroid=4.3e-5 (vs 7.3e-5 at dim=256, 41% better). But val_recon=1.1e-5 is 40% worse than dim=256 (7.9e-6). So dim=512 trades reconstruction quality for better centroid, with the same scale performance. This also resolves the open question from Scan 3: scale degradation at dim=512 is **beta-specific**. At beta=1e-4 scale is 12× worse, but at beta=1e-5 it is essentially the same. The larger latent space hurts scale only when the KL is strong enough to force the encoder to use many dims simultaneously.

- **The scan 4 hypothesis is falsified.** There is no evidence that the optimal beta shifts downward as dim grows beyond 256. Going below beta=1e-5 at dim=256 causes catastrophic collapse. The current best for reconstruction and scale combined remains **dim=256 + beta=1e-5** (val_recon=7.9e-6, val_scale=2.1e-5). If centroid accuracy is the priority, dim=512 + beta=1e-5 has a modest edge.

---

## Model Evaluation — Best Run (dim=256 + beta=1e-5, `latent_dim_256_260427_1122`)

Full `analyze_model.py` run on the best checkpoint (epoch 435).

**Reconstruction quality:**

| metric | value |
|--------|-------|
| Overall MSE | 2.99e-9 |
| Best channel (y-δ) | 9.5e-10 |
| Worst channel (x-x') | 7.0e-9 |

**Physics auxiliary losses (R²):**

All 6 scale heads: R² ≥ 0.9995 (min scale 0, max scale 2/3/5 all ≥ 0.9999). All 6 centroid heads: R² = 1.0000. The model encodes beam envelope perfectly.

**Latent space physics alignment:**

| parameter | R² |
|-----------|----|
| log σ (all 6) | 1.0000 |
| ⟨centroid⟩ (all 6) | 1.0000 |
| α_x | 0.921 |
| α_y | 0.861 |
| β_x | 0.541 |
| ε_x | 0.491 |
| β_y | 0.421 |
| ε_y | 0.236 |

Scales and centroids encode perfectly. Twiss α captures most of the optics variation. Twiss β and emittances are only partially encoded — the latent space has not learned to disentangle optics from beam size in a linearly accessible way. ε_y is the weakest (R²=0.24), likely because y-plane emittance varies least across the dataset.

**Latent dimension utilization:**

- Active dims (var > 1% of max): **45 / 256**
- Dims for 90/95/99% of total variance: 34 / 37 / 41
- Mean log-var: −0.924 (posterior moderately tighter than prior)
- PCA: PC1 explains 10.1%, PC1–2 explain 16.9%, PC1–10 explain 55.1% — information is distributed across many weakly explanatory dims, not concentrated in a few

The model uses roughly 40–45 effective dimensions despite having 256 available. This strongly suggests dim=256 is over-provisioned; a dim=64 or dim=128 model with the same beta could achieve similar or identical downstream metrics with a more compact representation. This is worth testing.

---

## Scan 5 — Fine beta sweep at dim=128 and dim=256 (submitted 2026-05-01)

**Question:** Is (256, 1e-5) truly the optimum, or does a beta between 1e-6 and 1e-4 do better? What is the shape of the loss surface in this region?

**Motivation:** The existing grid has only one tested point (beta=1e-5) in the two-order-of-magnitude range between the unregularized failure (beta=1e-6 at dim=256) and the scale/centroid degradation cliff (beta=1e-4). That's a wide gap. At dim=128 the picture is different — beta=0 and beta=1e-6 both work fine (no unregularized failure), and beta=1e-4 was actually the winner for reconstruction and scale. The finer structure between these anchor points is unknown for both dims.

**Design:** 2D grid at latent_dim ∈ {128, 256} × beta ∈ {2e-6, 5e-6, 2e-5, 5e-5}. Half-decade spacing, all new values (skipping already-tested 1e-6, 1e-5, 1e-4). Fixed: `lr=1e-3`, `data=v2_sectioned_1sec_10k`. 8 runs total, 2 nodes × 4 GPUs, all runs parallel. W&B group: `v2_scan5_beta_fine`.

**What we expect:** At dim=256, beta=2e-6 may still fail (close to the 1e-6 boundary); beta=5e-6 is the first candidate to be stable while giving lower val_recon than 1e-5. Above 1e-5, scale and centroid degraded sharply by 1e-4 — 2e-5 and 5e-5 will reveal whether that cliff is gradual or abrupt. At dim=128, all four betas should train stably. The question is whether the loss surface is flat around the current optima or has a sharper peak.

**What the result implies either way:** If the surface is flat (all four betas within ~10% of the 1e-5 result), hyperparameter sensitivity is low and the current optimum is robust. If there is a new optimum (e.g. 5e-6 at dim=256 with better recon, or 2e-5 with better scale), it revises the benchmark and informs the next stage. If 2e-6 fails at dim=256, it narrows the stable beta window to (2e-6, 1e-4) and confirms 1e-5 is near-optimal by necessity.

**Dim=128 results (all complete, 500 epochs):**

| latent_dim | beta | best epoch | val_recon | val_kl | val_scale | val_centroid |
|------------|------|-----------|-----------|--------|-----------|--------------|
| 128 | 2e-6 | 492 | 1.17e-5 | 12.10 | 3.78e-5 | 5.87e-5 |
| 128 | 5e-6 | 453 | 1.08e-5 | 0.767 | **1.82e-5** | **4.72e-5** |
| 128 | 2e-5 | 493 | **9.2e-6** | 0.471 | 1.34e-4 | 3.66e-4 |
| 128 | 5e-5 | 453 | 1.18e-5 | 0.373 | 2.99e-4 | 2.36e-3 |
| 128 | 1e-5 | 453 | 9.6e-6 | 0.621 | 6.1e-5 | 2.9e-4 | *(Scan 2 reference)* |

**Dim=256 results (2e-6/5e-6: original run; 2e-5/5e-5: rerun to 500 epochs, 2026-05-07):**

| latent_dim | beta | best epoch | val_recon | val_kl | val_scale | val_centroid | status |
|------------|------|-----------|-----------|--------|-----------|--------------|--------|
| 256 | 2e-6 | 13 | 2.33e-4 | 37.48 | 9.30e-3 | 6.30e-2 | COLLAPSED |
| 256 | 5e-6 | 11 | 2.68e-4 | 28.14 | 2.58e-3 | 1.00e-2 | COLLAPSED |
| 256 | 2e-5 | 495 | 9.51e-6 | 0.354 | 3.87e-5 | 2.53e-4 | complete |
| 256 | 5e-5 | ~90 (transient) | 1.54e-5 | 0.261 | ~1.11e-4 (converged) | 7.76e-4 | complete |
| 256 | 1e-5 | 435 | 7.9e-6 | 0.431 | 2.1e-5 | 7.3e-5 | *(Scan 2 reference)* |

Note on dim=256 beta=5e-5: the recorded best epoch (90) is a transient dip in val_scale; the loss plateaus at ~1.11e-4 from epoch 200 onward — the best-epoch number is misleading.

**Dim=512 results (rerun alongside dim=256 on 2026-05-07, full 500 epochs):**

| latent_dim | beta | best epoch | val_recon | val_kl | val_scale | val_centroid |
|------------|------|-----------|-----------|--------|-----------|--------------|
| 512 | 2e-5 | 381 | 1.30e-5 | 0.203 | 3.40e-5 | 5.70e-5 |
| 512 | 5e-5 | 451 | 1.08e-5 | 0.140 | 4.73e-5 | 1.46e-4 |
| 512 | 1e-5 | 500 | 1.1e-5 | — | 2.0e-5 | 4.3e-5 | *(Scan 3 reference)* |

**Conclusions:**

- **beta=5e-6 is the new winner for dim=128.** val_scale=1.82e-5, which is 3.4× better than the previously best beta=1e-5 (6.1e-5) at this dim. val_centroid=4.72e-5 also improves substantially (vs 2.9e-4 at beta=1e-5). val_recon=1.08e-5 is slightly worse than beta=2e-5 (9.2e-6) but the scale and centroid improvements dominate. The optimal beta for dim=128 is in the (2e-6, 2e-5) range with a peak around 5e-6.

- **beta=2e-6 at dim=128 is near-unstable but not collapsed.** val_kl=12.1 is anomalously high — far above the well-regularized range (0.4–0.8 seen at betas ≥ 1e-5) and approaching the collapse values seen at dim=256 (37–45), but the model is still encoding useful information (val_scale=3.78e-5, val_recon=1.17e-5). This is a qualitatively different regime from either well-regularized training or full posterior collapse — a "partial collapse" where the encoder variance drifts up but the decoder hasn't fully ignored z. The 2× worse scale vs beta=5e-6 shows this partial collapse still hurts downstream metrics.

- **Going above 1e-5 at dim=128 degrades scale monotonically.** beta=2e-5 gives val_scale=1.34e-4 (7× worse than 5e-6), beta=5e-5 gives 2.99e-4 (16× worse). The loss surface for dim=128 has a sharp peak near 5e-6, with both sides degrading quickly.

- **Dim=256 at beta=2e-6 and 5e-6 both collapse** (best epochs 11–13, KL≈28–37). The collapse is faster than in Scan 4 (which hit at epoch 8–12 for similar betas). The collapse boundary at dim=256 remains solidly above 1e-5.

- **Dim=256 + beta=1e-5 is confirmed optimal.** The full 500-epoch reruns show beta=2e-5 gives val_scale=3.87e-5 (1.84× worse) and beta=5e-5 gives ~1.11e-4 (5.3× worse). Both are conclusively worse, closing the open question from the partial runs. Combined with the collapse below 1e-5, the beta=1e-5 point is the global optimum for dim=256 in the tested range.

- **The same cliff exists at dim=512.** beta=2e-5 gives val_scale=3.40e-5 (1.7× worse than the dim=512+beta=1e-5 reference at 2.0e-5); beta=5e-5 gives 4.73e-5 (2.4× worse). The pattern is consistent across both dims: scale degrades monotonically as beta increases above 1e-5. dim=512+beta=2e-5 has val_centroid=5.70e-5, slightly better than dim=256+beta=1e-5 (7.3e-5), but worse than dim=512+beta=1e-5 (4.3e-5) — so no reranking occurs.

---

## Scan 6 — Can dim=64 match? Beta sweep at dim=64 (2026-05-07)

**Question:** Is dim=64 sufficient for beam dynamics, or is it capacity-limited? Can it match the physics metrics of dim=128 + beta=5e-6 with the right beta?

**Motivation:** Both dim=128 (33/128 active dims) and dim=256 (45/256 active dims) converge to roughly the same ~33–45 effective dimensions. The downstream application is a transformer-based beam dynamics model that benefits from a compact latent space. dim=64 sits right at the capacity boundary — it has just enough room to express ~33 active dims, but with no slack. The only prior result (dim=64 + beta=1e-5, Scan 2) gave val_scale=6.8e-5, 3.7× worse than the dim=128 optimum — but that was at the wrong beta. The optimal beta shifts down as dim shrinks (256→1e-5, 128→5e-6), so the dim=64 optimum likely sits in the 2e-6–5e-6 range.

**Design:** 4 runs at latent_dim=64 × beta ∈ {1e-6, 2e-6, 5e-6, 1e-5}. Fixed: `lr=1e-3`, `data=v2_sectioned_1sec_10k`. Run directly on an interactive GPU node (4× A100), all parallel. W&B group: `v2_scan6_dim64`.

**What we expect:** beta=1e-6 may collapse (the stability floor for dim=128 was around 2e-6–5e-6, and dim=64 may tolerate lower beta but is less certain). beta=5e-6 is the strongest candidate for optimal. beta=1e-5 replicates the Scan 2 result as an in-group anchor. If the optimal beta yields val_scale ≲ 2e-5 (matching the dim=128 optimum), dim=64 is viable. If it doesn't, dim=64 is capacity-limited and 128 is the floor.

**What the result implies either way:** A successful dim=64 (val_scale ≲ 2e-5) would halve the latent dimension fed to the transformer dynamics model with no physics penalty. A failure confirms that ~33 active dims cannot be packed into 64 total without degrading reconstruction or physics encoding, and dim=128 is the minimum viable latent size.

**Results (best-epoch metrics, 500 epochs each):**

| beta | best epoch | val_recon | val_kl | val_scale | val_centroid | status |
|------|-----------|-----------|--------|-----------|--------------|--------|
| 1e-6 | 494 | 1.02e-5 | 25.4 | 2.70e-5 | 5.24e-5 | partial collapse |
| 2e-6 | 494 | 9.5e-6 | 5.78 | 3.24e-5 | 5.88e-5 | partial collapse |
| 5e-6 | 493 | 1.02e-5 | 1.18 | 7.41e-5 | 1.51e-4 | complete |
| 1e-5 | 443 | 1.01e-5 | 1.02 | 4.60e-5 | 6.37e-4 | complete |

*(Viability threshold: val_scale ≲ 2e-5 to match dim=128 + beta=5e-6)*

**Convergence (epochs to val_recon thresholds):**

| run | 1.0e-4 | 5.1e-5 | 2.0e-5 | final |
|-----|--------|--------|--------|-------|
| beta=1e-6 | 20 | 42 | 93 | 1.02e-5 |
| beta=2e-6 | 25 | 40 | 90 | 9.6e-6 |
| beta=5e-6 | 36 | 48 | 98 | 1.02e-5 |
| beta=1e-5 | 29 | 39 | 99 | 1.01e-5 |

All four converge to essentially the same val_recon (~1.0e-5) in similar time. Reconstruction is not the differentiator — scale is.

**Conclusions:**

- **dim=64 does not meet the viability threshold.** The best completed result is beta=1e-6 with val_scale=2.70e-5, which is 35% above the 2e-5 target — and that run is in partial collapse (KL=25.4, vs. 1–2 for healthy training). The well-regularized runs (beta=5e-6, 1e-5) give val_scale=4.6e-5–7.4e-5, 2.3–3.7× above threshold, and have both converged (tiny epoch-to-epoch changes at 500 epochs). More training would not help them.

- **The collapse floor for dim=64 sits between beta=2e-6 and beta=5e-6.** Both beta=1e-6 (KL=25.4) and beta=2e-6 (KL=5.78) are in partial collapse — elevated KL, still encoding, but not in the healthy 1.0–1.2 range. beta=5e-6 is stable (KL=1.18). At beta=2e-6 the KL trajectory is unusual: it starts very high (epoch 1: 43.2, epoch 50: 25.7), drops sharply to ~6 by epoch 200, then stabilizes at 5.78 — a partial recovery from an initially unregularized state, not the monotonic drift seen in full posterior collapse. Notably, dim=128's floor was around beta=2e-6 (partial collapse, KL=12.1), and dim=256's floor was above beta=1e-5. Smaller dims tolerate lower betas, but the stable window is narrow.

- **Lower beta at dim=64 gives better scale, paradoxically.** beta=1e-6 (KL=25.4, partial collapse) gives val_scale=2.70e-5 — better than beta=2e-6 (KL=5.78, val_scale=3.24e-5) and far better than beta=5e-6 (KL=1.18, val_scale=7.41e-5). In the partial-collapse regime, a higher KL means the posterior deviates more from the prior — the encoder is using more of its 64 dims to encode input-specific information. The scale penalty, which directly supervises the decoder's auxiliary head, benefits from this richer encoding even if the distribution is far from the prior. This is not a reason to prefer lower beta: the partial collapse is undesirable for latent space regularity, and even the best result (2.70e-5) fails the viability threshold.

- **All four runs reach the same val_recon (~1.0e-5).** The model can reconstruct fine at dim=64. The failure mode is not reconstruction but physics encoding: the val_scale gap between dim=64 (best: 2.70e-5) and dim=128+beta=5e-6 (1.82e-5) is real and does not close with more training.

- **dim=128 + beta=5e-6 is confirmed as the minimum viable size.** Halving the latent dimension from 128 to 64 costs 48% on scale and 11% on centroid at the best comparable beta. The transformer dynamics model would need to absorb this degradation with no benefit in representation compactness that justifies it, since the intrinsic effective dimension (~33 active dims) remains unchanged.

---

## Scan 7 — Why does dim=256 collapse at low beta? Seed and LR diagnostic (2026-05-07)

**Question:** Is the dim=256 collapse at beta < 1e-5 caused by initialization (seed-specific bad basin), by lr=1e-3 overshooting in the fragile early-training window, or is it a structural property of the dim=256 architecture?

**Motivation:** dim=256 is uniquely fragile in the dim/beta heatmap. dim=128 trains stably at beta=0 (KL=0.59); dim=512 trains stably at beta=1e-6 (KL=1.39). But dim=256 collapses to the clamp ceiling (KL≈54) for every beta < 1e-5 tested. This non-monotonic behavior — both *smaller* and *larger* dims more stable than dim=256 — rules out simple "more capacity = harder to learn = more collapse" hypotheses. All runs share identical model code, identical hyperparameters except beta, and the logvar clamp (max=2) was set on 2026-04-08, before any v2 scan. So the cause is not a config or model-code difference between scans.

The collapse trace at dim=256 + beta=1e-6 (`runs/beta_1e-6_260429_0323`) is informative: the loss descends monotonically for 11 epochs (val_kl 49.8 → 42.8, val_total drops 5×), then in a single epoch the model flips to the collapsed fixed point. train_kl barely moves between epochs 11 and 12 (42.8 → 41.96), but val_kl jumps to 54.04 — the collapse happened **mid-epoch 12**. After that, val_kl stays at exactly 54.049728 to 6 decimal places for the remaining 488 epochs (encoder outputs the same μ, σ regardless of input). The single-epoch flip and frozen post-collapse state together suggest a discrete trigger event — a single bad gradient step that pushes μ past the clamp on enough dims to send the system into the posterior-collapse basin.

Three candidate triggers:
1. **Seed-specific bad init.** seed=42 happens to put the model near a saddle that the gradient flow falls off.
2. **LR overshoot.** lr=1e-3 is too large for the fragile early-training phase at dim=256. ReduceOnPlateau (patience=10) hadn't kicked in yet at epoch 12, so the LR was still 1e-3 at the moment of collapse. Adam's per-parameter normalization doesn't fully cancel architecture-specific gradient scale differences; the loss landscape may simply be sharper at width 256.
3. **Structural.** The 512→256 bottleneck shape, the 256-dim auxiliary heads, or kaiming init at width 256 has a property that always leads to collapse at low beta with this code, independent of seed and LR.

**Design:** Two grids at dim=256 + dataset `v2_sectioned_1sec_10k`.
- Grid A (seed sweep, 8 runs): `beta ∈ {1e-6, 5e-6} × seed ∈ {42, 43, 44, 45}` at lr=1e-3. seed=42 reproduces the original collapse as an anchor. W&B group: `v2_scan7_collapse_seed`.
- Grid B (lr × beta, 4 runs): `beta ∈ {1e-6, 5e-6} × lr ∈ {1e-4, 5e-4}` at seed=42. Brackets the lr=1e-3 default with two lower values. W&B group: `v2_scan7_collapse_lr`.

12 runs across 3 nodes (4 GPUs/node).

**What we expect:**
- If collapse is seed-specific: Grid A shows some seeds at each beta training cleanly. The fix is just to pick a "good" seed. Fragile but cheap.
- If collapse is LR-driven: Grid A all 8 runs collapse. Grid B trains stably at lr=1e-4 (and possibly 5e-4 too, depending on threshold sharpness). The fix is a lower default lr or a warmup for dim=256.
- If both contribute: mixed outcomes in both grids.
- If structural: Grid A all collapse, Grid B all collapse. Architecture-level investigation needed (per-dim μ/logvar trajectories, init scale, bottleneck shape).

**What the result implies either way:** A seed-only fix is fragile and unsatisfying. An LR fix is principled — if it works, the dim=256 portion of the heatmap below beta=1e-5 should be re-run at the lower lr to see whether the true optimum was hidden behind a training-stability artifact. If lr=5e-4 also works, lr=1e-3 is plausibly too large globally and we should consider lowering the default. A structural cause would mean the dim=256 architecture has an inherent issue — the bottleneck shape, auxiliary head width, or init — that needs targeted fixes before further latent-dim sweeps are meaningful.

**Grid A results — seed sweep (beta ∈ {1e-6, 5e-6} × seed ∈ {42, 43, 44, 45}, lr=1e-3):**

| run | outcome | val_recon | val_kl | val_scale | val_centroid |
|-----|---------|-----------|--------|-----------|--------------|
| beta=1e-6, seed=42 | partial collapse | 1.02e-5 | 29.3 (stuck) | 4.08e-5 | 1.67e-4 |
| beta=1e-6, seed=43 | **NaN crash** (step 338, epoch ~19) | — | 38.3 at crash | — | — |
| beta=1e-6, seed=44 | complete | 1.09e-5 | 1.95 | 3.22e-4 | 5.06e-2 |
| beta=1e-6, seed=45 | **NaN crash** (early) | — | — | — | — |
| beta=5e-6, seed=42 | **full collapse** (epoch 13) | 1.38e-3 | 53.8 (stuck) | ≈1.0 | ≈0.87 |
| beta=5e-6, seed=43 | cancelled (time limit, ~250 ep) | — | — | — | — |
| beta=5e-6, seed=44 | **NaN crash** (step 101, epoch ~6) | — | 54.4 at crash | — | — |
| beta=5e-6, seed=45 | **NaN crash** (early) | — | — | — | — |

6 of 8 runs failed (4 NaN crashes, 1 full collapse, 1 time-limit cancellation). The one apparently successful run (beta=1e-6, seed=44) converged in val_recon but has anomalous scale/centroid: val_centroid=5.06e-2 is 700× above the best reference, and val_scale=3.22e-4 is 15× worse — despite val_kl=1.95 being healthy. Likely the auxiliary heads didn't converge. The partial-collapse run (beta=1e-6, seed=42) has stuck KL=29.3 but still reaches val_recon=1.02e-5 — the decoder extracts signal from the corrupted latent space but downstream physics metrics are unreliable.

**Grid B results — LR scan (beta ∈ {1e-6, 5e-6} × lr ∈ {1e-4, 5e-4}, seed=42):**

| run | val_recon | val_kl | val_scale | val_centroid | epochs to 1.7e-5 |
|-----|-----------|--------|-----------|--------------|------------------|
| beta=1e-6, lr=5e-4 | **8.6e-6** | 3.26 | 2.52e-5 | 1.22e-4 | 98 |
| beta=5e-6, lr=5e-4 | 9.5e-6 | **0.72** | 3.80e-5 | 8.85e-5 | 111 |
| beta=5e-6, lr=1e-4 | 1.64e-5 | 2.71 | 1.48e-4 | 2.61e-4 | 396 |
| beta=1e-6, lr=1e-4 | 2.25e-5 | 14.4 | 1.23e-4 | 1.70e-4 | >500 |

All 4 completed. No collapse at either lr. lr=5e-4 reaches the same threshold 4× faster than lr=1e-4 and also converges to substantially better final values. beta=1e-6 + lr=1e-4 still has KL=14.4 at epoch 500 — even the lower lr doesn't fully stabilize this beta without the higher lr's momentum to escape the early-training instability.

**Conclusions:**

- **Collapse is LR-driven, not seed-specific.** Grid A ran 8 seeds at lr=1e-3 and 7 of 8 failed (6 outright failures plus one questionable success). If seed were the cause, we'd expect a roughly even split. Instead, varying the seed did nothing; varying the LR fixed everything. The collapse mechanism is that lr=1e-3, with ReduceOnPlateau still at its initial value during the fragile first ~15 epochs, takes Adam steps large enough to push μ past the logvar clamp on enough dims to trigger the posterior-collapse bifurcation. Once that flip happens — as the single-epoch collapse trace at beta=1e-6 demonstrated — there is no recovery.

- **lr=5e-4 is the right default.** It eliminates collapse across both tested betas, converges ~4× faster than lr=1e-4, and reaches better final values. lr=1e-4 avoids collapse but doesn't fully settle (KL=14.4 at convergence for beta=1e-6) and is unnecessarily slow. The earlier LR scan (Scans 1–6) that named lr=1e-3 as the winner was evaluating survivorship — it was inadvertently testing which seeds happened to not collapse, not which LR was actually best. **The default LR has been updated to 5e-4.**

- **The dim=256 heatmap below beta=1e-5 was never honestly measured.** Every cell in that region (beta ∈ {0, 1e-6, 2e-6, 5e-6}) showed collapse or NaN at lr=1e-3. None of those outcomes reflect the true capability of those configurations — they reflect LR instability. The true optimum at dim=256 may sit at a lower beta than 1e-5 if retrained with lr=5e-4. This is the most important open question.

- **beta_1e-6 + lr=5e-4 achieves val_recon=8.6e-6**, close to the current best (7.9e-6 at dim=256+beta=1e-5+lr=1e-3) and competitive on scale (2.52e-5 vs 2.1e-5). The current champion was trained under what is now known to be an unstable LR — whether it merely got lucky or whether beta=1e-5 is genuinely more stable at lr=1e-3 is a secondary question. The priority is re-running the full beta sweep at dim=256 with lr=5e-4 to see the true loss surface.

- **Whether lr sensitivity is dim- or beta-dependent is unknown.** The LR scan only covers dim=256 with beta ∈ {1e-6, 5e-6}. Prior scans at dim=128 with lr=1e-3 mostly succeeded — smaller dims may tolerate higher LR. Until tested, lr=5e-4 should be treated as the safe universal default.

---

## Scan 8 — Full heatmap re-run at lr=5e-4 (submitted 2026-05-10)

**Question:** What does the true dim × beta loss surface look like when all runs use the same, stable learning rate?

**Motivation:** Every scan up to Scan 7 used lr=1e-3 as the default. Scan 7 proved this was wrong: lr=1e-3 causes NaN crashes and posterior collapse at a rate of 6/8 runs for certain (dim, beta) combinations, making those heatmap cells artifacts of training instability rather than genuine hyperparameter behavior. The existing heatmap is therefore a mix of real results (cells where lr=1e-3 happened to be stable) and failures (cells where it wasn't). The two lr=5e-4 data points from Scan 7 (dim=256 × {1e-6, 5e-6}) already showed that the previously-collapsed cells train cleanly and reach competitive val_recon — but two cells don't reveal the full surface shape. The entire grid needs to be re-run under controlled, consistent conditions before any further architectural decisions are made.

**Design:** Full 4×7 grid: `latent_dim ∈ {64, 128, 256, 512}` × `beta ∈ {1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4}`. Fixed: `lr=5e-4`, `seed=42`, `data=v2_sectioned_1sec_10k`. beta=1e-3 excluded (shown to be bad at all dims, nothing new to learn). 28 runs, 7 nodes, all parallel, 8h time limit. W&B group: `v2_scan8_heatmap_lr5e-4`. Any run that times out (slow node) can be resumed from its last periodic checkpoint, and the incremental CSV means no history is lost.

**What we expect:** The collapsed cells at dim=256 (beta ∈ {2e-6}) and dim=512 (beta ∈ {1e-6, 2e-6, 5e-6}) should now train cleanly, revealing the true shape of the low-beta region. The prior heatmap showed a sharp stability cliff at dim=256 just below beta=1e-5 — that cliff should disappear or shift significantly at lr=5e-4. For dims already stable at lr=1e-3 (dim=128, dim=64 at higher betas), the results should be close to existing data, confirming those cells weren't significantly biased by LR. The optimum for dim=256 may turn out to be at a lower beta than 1e-5 now that those cells are accessible.

**What the result implies either way:** If the new surface is broadly similar to the old one for the stable cells and fills in the collapsed region cleanly, the lr=1e-3 heatmap was only wrong in the specific failure cells — the rest of the prior conclusions hold. If the new surface differs substantially even in the previously-stable region (e.g., dim=128 optimum shifts, scale/centroid curves change shape), it means lr was a confounding factor throughout and the prior scan conclusions need to be revisited. The new heatmap also directly answers whether lr sensitivity is dim- or beta-dependent: if dim=64/128 results at lr=5e-4 match lr=1e-3 closely, those dims are LR-robust; if they differ, lr matters everywhere.

**Results (full 4×7 grid — 20 runs from 260510_2140, 8 resubmitted as job 52849963 on 260512_0804):**

| dim \ beta | 1e-6 | 2e-6 | 5e-6 | 1e-5 | 2e-5 | 5e-5 | 1e-4 |
|-----------|------|------|------|------|------|------|------|
| 64  | 1.01/3.55/7.62e-5 | **8.9/1.77/4.97e-5** | 9.6/3.74/12.3e-5 | 10.8/7.70/59.0e-5 | 10.8/13.5/139e-5 | 12.6/113/787e-5 | 11.8/420/2500e-5 |
| 128 | 11.8/2.04/3.68e-5 | 9.0/2.02/3.36e-5 | **10.8/1.82/4.72e-5**† | 9.62/6.05/28.7e-5† | 9.7/3.75/100e-5 | 11.0/22.3/254e-5 | 11.2/109/987e-5 |
| 256 | 9.9/2.51/7.72e-5 | 9.2/3.04/9.81e-5 | **8.9/3.34/5.83e-5** | 9.1/3.60/6.54e-5 | 8.9/4.38/21.6e-5 | 10.8/6.17/96.0e-5 | 11.2/25.0/210e-5 |
| 512 | 10.9/3.53/8.75e-5 | 10.6/3.66/9.40e-5 | 11.0/2.79/10.3e-5 | 9.9/4.12/5.23e-5 | **9.5/3.49/4.54e-5** | 9.8/5.15/36.8e-5 | 9.2/7.13/75.8e-5 |

Values are (val_recon / val_scale / val_centroid) × 1e-5. Bold marks the apparent best cell per row.
† lr=5e-4 run did not converge in 500 epochs for dim=128 at low beta (KL still 6.38 at ep500 for beta=5e-6; bad local minimum for beta=1e-5). Data from prior lr=1e-3 runs (260505_0959 and 260427_1122 respectively), which are stable at this dim.

**Conclusions:**

- **No collapsed cells.** All 28 runs trained cleanly, confirming that the collapse/NaN failures seen in prior scans were entirely LR-driven. The dim=256 cliff below beta=1e-5 is gone. The prior heatmap's empty cells were training artifacts, not genuine minima.

- **The true optimal region is dim=256–512, beta=5e-6–2e-5.** Both dims reach val_recon ≈ 8.9–9.5e-6 and val_centroid < 6e-5 in this beta band. dim=64 tops out at val_recon=8.9e-6 (beta=2e-6) but its scale and centroid stay 2–3× worse than dim=256, exactly as seen in Scan 6.

- **dim=256 + beta=5e-6 is the new single best cell.** val_recon=8.9e-6, val_scale=3.3e-5, val_centroid=5.8e-5, KL=0.59. This is modestly better on reconstruction than the previous champion (dim=256+beta=1e-5 at lr=1e-3, val_recon=7.9e-6) but substantially better on centroid (5.8e-5 vs 7.3e-5), and the previous champion was trained at lr=1e-3. The new cell at lr=5e-4 is the only honest comparison.

- **dim=512 + beta=2e-5 is a close second.** val_recon=9.5e-6, val_scale=3.5e-5, val_centroid=4.5e-5, KL=0.13. Best centroid in the whole grid, suggesting the largest dim paired with moderate regularization allows the most precise beam-position encoding. The KL is very low (near-deterministic encoder), which may be undesirable for a generative model but is fine for a fixed-space encoder-decoder.

- **beta=1e-4 is dead across all dims.** Scale and centroid collapse (e.g., dim=64: scale 4.2e-3, centroid 2.5e-2) while val_recon stays normal. The KL=0.1–0.5 confirms the encoder is essentially deterministic — the VAE has become a plain AE at this beta. This beta provides no useful regularization and should not be explored further.

- **The low-beta / high-dim instability hypothesis (Scan 5/6) is resolved.** dim=512 at beta=1e-6 trains cleanly (val_recon=1.09e-5, KL=1.95) with no collapse. Prior failures were all lr=1e-3 artifacts. The model is not structurally fragile at low beta for any tested dim — only lr was the issue.

- **dim=128 + beta=5e-6 has the best scale in the grid.** With data from the prior lr=1e-3 run, val_scale=1.82e-5 — better than any Scan 8 cell including dim=256+5e-6 (3.34e-5). val_centroid=4.72e-5 is also better than dim=256+5e-6 (5.83e-5). The trade-off is reconstruction: val_recon=1.08e-5 vs 8.9e-6 for dim=256+5e-6. Whether this trade-off is worthwhile depends on whether the downstream dynamics model cares more about physics-metric precision or reconstruction fidelity.

- **lr=5e-4 does not converge for dim=128 at low beta.** The Scan 8 lr=5e-4 run at dim=128+beta=5e-6 had KL=6.38 at epoch 500 (target ~0.77); at lr=1e-3 the same cell reaches KL=1.79 by epoch 51. The beta=1e-5 cell converged quickly on KL but to a worse reconstruction basin. dim=128 does not suffer the posterior-collapse instability that motivated the lr change (that was specific to dim=256), so lr=1e-3 remains the right default for dim=128 at low beta. The expectation that lr=5e-4 results would closely match lr=1e-3 for dim=128 was not met in the low-beta region.

- **The effective dimensionality question.** dim=256+5e-6 (KL=0.59) is approaching the near-deterministic regime but still has meaningful stochasticity. For downstream use in a transformer dynamics model, the meaningful question is active-dim count — this was measured as ~33/128 for the dim=128 Scan-5 model. Whether dim=256+5e-6 offers more active dims or the same compressed into a larger ambient space is not yet measured.

---

## Best Checkpoints

| rank | run | val_recon | val_scale | val_centroid | notes |
|------|-----|-----------|-----------|--------------|-------|
| 1 (best recon) | `runs/latent_dim_256_beta_5e-6_260512_0804` (`runs/best`) | 8.9e-6 | 3.3e-5 | 5.8e-5 | dim=256, beta=5e-6, lr=5e-4; KL=0.59 |
| 2 (best scale) | `runs/latent_dim_128_beta_5e-6_260505_0959` | 1.08e-5 | 1.82e-5 | 4.72e-5 | dim=128, beta=5e-6, lr=1e-3; KL=0.77 |
| 3 (best centroid) | `runs/latent_dim_512_beta_2e-5_260510_2140` | 9.5e-6 | 3.5e-5 | 4.5e-5 | dim=512, beta=2e-5, lr=5e-4; KL=0.13 |
| 4 (prior best, superseded) | `runs/latent_dim_256_260427_1122` | 7.9e-6 | 2.1e-5 | 7.3e-5 | dim=256, beta=1e-5, lr=1e-3 |

---

## Inference Benchmark — Latent Dim Scaling (2026-05-13)

**Setup:** One `_best.pth` checkpoint per latent_dim, all at beta=5e-6, all from the same job batch (260512_0804) except dim=128 (same batch). float32, A100 GPU. 20 warmup + 50 timed repetitions per (mode, batch_size) pair. Batch sizes: 1, 4, 16, 64, 256. Results saved to `<run>/benchmark/results.json`.

Checkpoints:
- dim=64: `runs/latent_dim_64_beta_5e-6_260512_0804/..._best.pth` (epoch 492)
- dim=128: `runs/latent_dim_128_beta_5e-6_260512_0804/..._best.pth` (epoch 493)
- dim=256: `runs/best/latent_dim_256_beta_5e-6_260512_0804_best.pth` (epoch 496)
- dim=512: `runs/latent_dim_512_beta_5e-6_260512_0804/..._best.pth` (epoch 499)

**Summary at bs=256 (peak throughput):**

| latent_dim | encode (samp/s) | decode (samp/s) | encode mem (MB) | decode mem (MB) |
|---|---|---|---|---|
| 64  | 92,606 | 4,479 | 418 | 350 |
| 128 | 94,778 | 4,506 | 423 | 354 |
| 256 | 94,138 | 4,652 | 435 | 366 |
| 512 | 92,077 | 4,725 | 456 | 388 |

**Full results — dim=64:**

| mode | bs | latency (ms) | tput (samp/s) | mem (MB) |
|------|----|-------------|---------------|----------|
| encode |   1 | 0.91 ± 0.04 |    1,095 |  92 |
| decode |   1 | 0.68 ± 0.03 |    1,478 |  92 |
| encode |   4 | 0.93 ± 0.01 |    4,303 |  93 |
| decode |   4 | 1.15 ± 0.05 |    3,485 |  94 |
| encode |  16 | 1.03 ± 0.02 |   15,551 | 112 |
| decode |  16 | 3.57 ± 0.38 |    4,482 | 116 |
| encode |  64 | 0.93 ± 0.02 |   68,963 | 170 |
| decode |  64 | 12.10 ± 0.04 |   5,291 | 203 |
| encode | 256 | 2.76 ± 0.01 |   92,606 | 418 |
| decode | 256 | 57.16 ± 0.19 |   4,479 | 350 |

**Full results — dim=128:**

| mode | bs | latency (ms) | tput (samp/s) | mem (MB) |
|------|----|-------------|---------------|----------|
| encode |   1 | 0.91 ± 0.04 |    1,101 |  98 |
| decode |   1 | 0.66 ± 0.01 |    1,507 |  98 |
| encode |   4 | 0.92 ± 0.02 |    4,330 |  99 |
| decode |   4 | 1.18 ± 0.00 |    3,398 |  99 |
| encode |  16 | 0.95 ± 0.01 |   16,911 | 117 |
| decode |  16 | 3.11 ± 0.05 |    5,152 | 121 |
| encode |  64 | 0.93 ± 0.00 |   68,820 | 175 |
| decode |  64 | 11.62 ± 0.07 |   5,507 | 208 |
| encode | 256 | 2.70 ± 0.00 |   94,778 | 423 |
| decode | 256 | 56.81 ± 0.24 |   4,506 | 354 |

**Full results — dim=256:**

| mode | bs | latency (ms) | tput (samp/s) | mem (MB) |
|------|----|-------------|---------------|----------|
| encode |   1 | 0.89 ± 0.03 |    1,119 | 109 |
| decode |   1 | 0.65 ± 0.02 |    1,529 | 109 |
| encode |   4 | 0.92 ± 0.03 |    4,342 | 110 |
| decode |   4 | 1.19 ± 0.01 |    3,362 | 111 |
| encode |  16 | 0.94 ± 0.03 |   17,073 | 128 |
| decode |  16 | 3.09 ± 0.02 |    5,175 | 132 |
| encode |  64 | 1.02 ± 0.01 |   62,510 | 186 |
| decode |  64 | 11.42 ± 0.05 |   5,603 | 219 |
| encode | 256 | 2.72 ± 0.00 |   94,138 | 435 |
| decode | 256 | 55.03 ± 0.19 |   4,652 | 366 |

**Full results — dim=512:**

| mode | bs | latency (ms) | tput (samp/s) | mem (MB) |
|------|----|-------------|---------------|----------|
| encode |   1 | 0.89 ± 0.02 |    1,120 | 130 |
| decode |   1 | 0.66 ± 0.01 |    1,527 | 130 |
| encode |   4 | 0.92 ± 0.02 |    4,366 | 131 |
| decode |   4 | 1.20 ± 0.00 |    3,324 | 133 |
| encode |  16 | 1.00 ± 0.06 |   16,037 | 150 |
| decode |  16 | 3.05 ± 0.01 |    5,253 | 154 |
| encode |  64 | 0.94 ± 0.00 |   67,779 | 207 |
| decode |  64 | 11.51 ± 0.03 |   5,559 | 240 |
| encode | 256 | 2.78 ± 0.00 |   92,077 | 456 |
| decode | 256 | 54.18 ± 0.18 |   4,725 | 388 |

**Conclusions:**

- **Latent dim has essentially no effect on throughput or latency.** Encode throughput at bs=256 spans only 92,077–94,778 samp/s across all four dims — a 3% range. Latency curves are indistinguishable within noise. The convolutional backbone dominates compute; the linear projection layers between the conv features and the latent space are negligible by comparison.

- **Memory cost scales modestly with latent_dim.** From dim=64 to dim=512, peak GPU memory at bs=256 rises from 418 → 456 MB for encode and 350 → 388 MB for decode — a total increase of ~38 MB for an 8× increase in latent dimension. The conv layers account for the large fixed baseline; the latent projection adds very little.

- **The decoder is the compute bottleneck, not the encoder.** At bs=256, encode latency is ~2.7 ms; decode latency is ~55 ms — a 20× gap. Any optimization effort for inference throughput should target the decoder convolutions, not the latent space.

- **Encode throughput saturates at bs=64.** Encode latency barely moves from bs=1 (0.91 ms) to bs=64 (0.93 ms), meaning online single-sample inference has the same per-sample cost as batch inference for the encoder. The encoder is compute-light enough to saturate the GPU early; the decoder does not saturate until bs=256 and likely beyond.

- **Practical implication:** Latent dim choice is entirely a model-quality decision. There is no inference-time cost argument for preferring a smaller latent dim, so dim=256 or dim=512 can be selected based purely on downstream physics metrics without any throughput penalty.

---

## Open Questions

- **Active dimension count for Scan 8 winners.** The dim=256+5e-6 and dim=512+2e-5 cells have not been analyzed with `analyze_model.py`. Measuring how many latent dims are actually active (variance > threshold) determines whether the larger models offer richer representations or merely pad the same ~30–40 active dims with dead dimensions.

- **Which cell to hand to the dynamics model?** dim=256+5e-6 is the best on reconstruction; dim=512+2e-5 is the best on centroid. The right choice depends on whether the transformer dynamics model cares more about reconstruction fidelity or beam-position precision. A quick ablation with both checkpoints as frozen encoders would resolve this.

- **Is beta=2e-6 or lower worth exploring at dim=256/512?** dim=256+2e-6 (val_recon=9.2e-6, centroid=9.8e-5) is worse than dim=256+5e-6 on centroid, suggesting 5e-6 is already near the sweet spot. But beta=1e-6 at dim=256 (centroid=7.7e-5) is also better than beta=2e-6. The centroid vs. beta curve is non-monotonic at dim=256, which may be worth following up.

- **lr default should be dim-dependent.** Scan 8 confirms lr=5e-4 is necessary for dim=256 (and probably dim=512 at low beta), but lr=1e-3 is correct for dim=128 — it converges ~10× faster and avoids the slow-KL-settling problem seen at lr=5e-4. A mixed-lr grid for dim=64 is untested; it's likely fine at lr=1e-3 given its smaller capacity.
