# Method Outline

This document keeps the paper-facing method story aligned with
`CCFA_VSR_NEW_MODEL_PROPOSAL.md`.

## Core Claim

Real-world arbitrary-scale VSR is modeled as reliability-calibrated conditional
transport over continuous spacetime trajectories.

## Guardrails From ST_VSR_Project

The old ST_VSR_Project failed because the generative prior improved some
no-reference scores while producing melted textures, unstable hallucination,
and structural errors. The new method must keep low-frequency structure in the
deterministic data-consistent path, treat rectified flow as an optional
high-frequency residual posterior, and gate all residual generation by
reliability/uncertainty. NIQE/MUSIQ/CLIPIQA are secondary diagnostics, not
checkpoint-selection targets.

## Notation To Finalize

- LR video: `Y_{1:T}`.
- HR clean video/function: `X_{1:T}` or `X(x,y,t)`.
- Reliability map: `R(x,y,t)`.
- Artifact map: `A(x,y,t)`.
- Texture/motion uncertainty: `U_tex`, `U_motion`.
- Evidence tokens: `e_i = (f_i, x_i, y_i, t_i, s_i, footprint_i)`.
- Soft trajectory transport plan: `P^{t -> t+k}_{i,j}`.
- Residual latent path: `z_tau = (1 - tau) z_0 + tau z_1`.

## Sections

### 1. Problem Formulation

Define arbitrary-scale real-world VSR as estimating a clean HR video function
from degraded LR observations under unknown degradation and motion uncertainty.
State why deterministic regression collapses degradation, motion, and texture
uncertainty into one point estimate.

### 2. Reliability-Calibrated Evidence

Describe multi-scale evidence tokenization and degradation-causal uncertainty.
The key paper point is that reliability is not a decorative mask; it controls
transport mass, data consistency, and posterior sampling strength.

### 3. OT/SB Trajectory Bridge

Define the local candidate graph, reliability-weighted source/target mass,
unbalanced mass for occlusion, and Sinkhorn transport plan. Explain the bridge
states as degraded-to-clean intermediate evidence for residual transport.

### 4. Trajectory Koopman-SSM Memory

Define scan policies and the trajectory-conditioned selective state update.
For the OT/SB policy, construct a per-anchor soft trajectory sequence by
normalizing the transport plan inside each target frame, aggregating expected
evidence along the soft path, and restoring the scanned state at the anchor
time. Keep bridge-temporal scan as an explicit ablation rather than the main
trajectory memory path.
Explain Koopman regularization as predictability in latent observable space,
not a claim that RGB video is linear.

### 5. Rectified Flow Residual Posterior

Separate low-frequency data-consistent base from high-frequency residual.
Train the conditional vector field on bridge-conditioned residual paths and
evaluate one-step, two-step, and four-step sampling.

### 6. Spacetime Wavelet Operator Decoder

Describe coordinate/footprint queries, operator mixing, wavelet frequency
split, and anti-aliasing gates for arbitrary scale.

### 7. Data Consistency And Calibration

State the projection objective and calibration losses that constrain low
frequency drift and hallucination risk.

## Figures To Produce

- Overall pipeline diagram.
- OT/SB soft trajectory graph with unmatched mass.
- Offline vs streaming memory diagram.
- Rectified-flow posterior sampling diagram.
- Wavelet/operator arbitrary-scale decoder diagram.
- Reliability/error calibration plot.
