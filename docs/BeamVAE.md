# Physics Motivation

High-intensity charged particle beams are governed by collective effects, most notably space charge, which induce strong nonlinear forces that couple all degrees of freedom. Accurate simulation of space-charge-dominated beam dynamics typically requires self-consistent field solves on fine 3D grids, leading to computational costs that scale poorly with resolution, particle number, and the number of lattice elements. This cost becomes prohibitive for tasks that require repeated evaluations, such as online tuning, inverse design, uncertainty quantification, or optimization.

In many accelerator diagnostics and simulation workflows, the beam state is not represented by individual macroparticles but by low-dimensional density observables, such as transverse or longitudinal projections. These projected phase-space densities are non-negative, normalized, and often smooth, reflecting both physical conservation laws and measurement processes. Importantly, while the underlying 6D phase-space distribution is high-dimensional, the set of physically realizable beam distributions lies on a much lower-dimensional manifold constrained by beam optics, collective effects, and initial conditions.

The central hypothesis of this work is that the evolution of beam density distributions under space-charge-dominated dynamics can be efficiently represented in a learned latent space. By compressing physically valid density projections into a compact latent representation, one can retain the essential collective degrees of freedom while discarding redundant or noise-dominated variations. A variational autoencoder (VAE) provides a principled framework to learn such a representation, enforcing continuity, smoothness, and a well-behaved latent geometry that is suitable for downstream dynamical modeling.

# Data Representation

The inputs to the model are beam density projections obtained from simulation or diagnostics. Each sample consists of multiple projections (e.g., transverse and/or longitudinal), discretized on fixed grids. Each projection represents a probability density and is normalized such that the sum over all pixels equals one. This normalization reflects particle number conservation and ensures comparability across samples and lattice locations.

Because the data represent densities rather than arbitrary images, the model must respect non-negativity and normalization constraints, at least approximately. These physical constraints inform both architectural choices and loss design.

## Coordinate Convention

Particle coordinates are expressed in Courant-Snyder trace space before
building frequency maps:

| Index | Symbol | Description | Units |
|-------|--------|-------------|-------|
| 0 | x | Horizontal position | m |
| 1 | x' | Horizontal angle (px / p_ref) | rad |
| 2 | y | Vertical position | m |
| 3 | y' | Vertical angle (py / p_ref) | rad |
| 4 | z | Bunch-frame longitudinal position (−β·c·t, centered) | m |
| 5 | δ | Relative momentum deviation ((pz − p_ref) / p_ref) | 1 |

Here p_ref is the mean longitudinal momentum of alive particles at each
snapshot. This choice of coordinates has several advantages:

- **Removes energy dependence from scales.** In laboratory coordinates
  (x, y, z, px, py, pz), the momentum scales carry a factor of p_ref ≈ E/c,
  creating a ~10 order-of-magnitude gap between position and momentum
  dimensions. Trace-space coordinates collapse this to ~5 orders of
  magnitude, making the scale prediction head much better conditioned.

- **Natural for beam optics.** Twiss parameters (α, β, γ), emittance, and
  beam matching are all defined in (x, x') and (y, y') trace space. The
  latent representation therefore captures beam properties in the
  coordinates where optics is naturally formulated.

- **Physically meaningful longitudinal coordinate.** Bmad tracks particles at
  fixed longitudinal position s, so the raw z coordinate is identically zero
  for all particles at each snapshot. The arrival time t carries the
  longitudinal information; converting via z = −β·c·t recovers the
  bunch-frame position spread.

The 15 frequency map channels represent all unique pairwise 2D projections
of these 6 coordinates. The channel ordering is determined by sorting the
plane names alphabetically (see `src/data/preprocessing.py`).

Each frequency map channel is a normalized 2D histogram on an adaptive
grid of ±n_sigma (default 4) standard deviations per axis. The 6-component
scale vector records the per-dimension standard deviations, enabling
reconstruction of absolute beam sizes from the normalized maps.

# Model Overview

The model is a variational autoencoder composed of three main components:

1. An encoder that maps normalized density projections into a low-dimensional latent distribution.
2. A latent space regularized to follow a simple prior, enabling interpolation and robust generalization.
3. A decoder that reconstructs physically valid density projections from latent samples.

The VAE is trained end-to-end to minimize a combination of reconstruction error and latent regularization, balancing fidelity to the input distributions with smoothness and structure in the latent space.

# Encoder Architecture

The encoder takes as input a stack of density projections. Convolutional layers are used to exploit local spatial correlations and translational structure in the densities. These layers progressively reduce spatial resolution while increasing feature depth, producing a compact feature representation.

The final encoder layers map this representation to two vectors: the latent mean and the latent log-variance. Together, these define a multivariate Gaussian distribution in latent space. Sampling from this distribution during training introduces stochasticity that encourages the encoder to learn robust, distributed representations rather than memorizing individual samples.

From a physical perspective, the encoder learns to identify collective beam features such as size, asymmetry, halo structure, and correlations across projections, and to encode them as continuous latent variables.

# Latent Space and Prior

The latent space is regularized by a Kullback–Leibler divergence term that encourages the encoded latent distribution to match a simple prior, typically a standard normal distribution. This regularization has several important consequences:

* It enforces continuity, so that nearby points in latent space correspond to similar beam distributions.
* It enables meaningful interpolation between beam states.
* It prevents pathological encodings that would hinder generalization or downstream dynamical modeling.

Physically, the latent variables can be interpreted as abstract collective coordinates describing the beam state, analogous to generalized envelope parameters or higher-order moments, but learned directly from data rather than imposed analytically.

# Decoder Architecture

The decoder maps latent samples back to density projections. It mirrors the encoder structure, using transposed convolutions or upsampling layers to recover the original spatial resolution.

Because the outputs represent normalized density distributions, the decoder is designed to produce non-negative values and to enforce approximate normalization. This can be achieved, for example, by applying a softmax operation over each projection or by normalizing the output explicitly. These choices ensure that reconstructed outputs remain within the physically meaningful space of probability densities.

The decoder thus learns a nonlinear generative map from latent collective coordinates to physically plausible beam density projections.

# Loss Function

The training objective consists of two terms:

1. A reconstruction loss that measures the discrepancy between the input and reconstructed density projections. Depending on modeling choices, this may be formulated as mean squared error, cross-entropy, or another divergence appropriate for normalized densities.
2. A KL divergence term that regularizes the latent distribution toward the prior.

The relative weighting of these terms controls the trade-off between reconstruction accuracy and latent smoothness. From a physics standpoint, this trade-off determines how much fine-scale structure is retained versus how strongly the model emphasizes global, collective behavior.

# Role in Beam Dynamics Modeling

Once trained, the VAE serves as a compact, physics-informed representation of beam density distributions. The encoder provides a map from high-dimensional density data to a low-dimensional latent state, while the decoder enables reconstruction back to physical space.

This latent representation can be coupled to separate models that learn beam transport or inverse dynamics in latent space, dramatically reducing computational cost compared to full space-charge simulations. Because the latent space is continuous and regularized, it is well suited for sequence modeling, control, and optimization tasks relevant to accelerator operations.

In this way, the VAE acts as a learned surrogate for the high-dimensional beam distribution, preserving essential physics while enabling efficient downstream modeling.
