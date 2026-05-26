# Critical Gap Closure

This note tracks the engineering gaps closed after comparing the project with
`CCFA_VSR_NEW_MODEL_PROPOSAL.md`.

## Closed In Code

- Multi-scale evidence tokenizer now injects low/high bands, temporal deltas,
  normalized spacetime coordinates, scale, and query footprint features.
- OT/SB bridge now applies local spatiotemporal candidate masks and exposes
  unmatched/occlusion mass derived from reliability.
- Trajectory memory now uses a lightweight gated selective state-space scan
  with Koopman prediction targets instead of a plain GRU-only placeholder.
- Wavelet operator decoder now includes hidden operator mixing, coordinate
  decoding, query coordinates, query footprint, and anti-aliasing gates.
- Evaluation now emits perceptual/temporal/calibration fields in addition to
  PSNR/SSIM/FPS: `tof`, `lpips`, `dists`, `reliability_ece`,
  `selective_reconstruction_auc`, and `vram_gb`.
- Training now includes an uncertainty calibration loss and frame-manifest
  sequential clip sampling for longer formal runs.
- Training runner now supports gradient accumulation, gradient clipping,
  scheduler state save/resume, periodic validation, and best-checkpoint
  selection from validation metrics.
- `configs/train/stage_b_frame_manifest_full.yaml` provides the first formal
  Stage-B frame-manifest training template beyond one-step smoke configs,
  including validation and cosine scheduling knobs.
- Benchmark tooling now has both a planner and an executable internal runner:
  `scripts/run_benchmark.py` can run selected CPU-safe TrajFlow rows and write
  selected plan, metrics, JSON, CSV, and Markdown summaries.
- Evaluation now reports metric backend status for official/proxy/missing
  LPIPS, DISTS, NIQE, VMAF, FVD, MUSIQ, and CLIPIQA adapters, and can collect
  posterior sample variance/PSNR/error statistics.
- Inference can export posterior sample frames for stochastic residual-flow
  inspection.
- Required ablation entry points now cover core module removal, unbalanced OT,
  reliability calibration, wavelet/operator/data-consistency removal, flow step
  counts, consistency distillation, and offline/streaming protocol comparisons.
- Evaluation and benchmark exports now include dependency-free MAC estimates
  for Conv/Linear modules, alongside parameters, FPS, latency, and VRAM.
- Full-manifest Stage C/D/E templates now exist for rectified-flow teacher
  training, one-step consistency distillation, and offline/streaming joint
  training.
- `scripts/audit_readiness.py` now exports a machine-readable CCF-A readiness
  report covering code modules, configs, data manifests, checkpoints, baseline
  registry completeness, official metric backends, and paper documents.
- Paper-facing method, related-work, submission-checklist, and formal-runbook
  documents now exist under `docs/paper/` and `docs/experiments/`.

## Still External-Resource Bound

The following cannot be completed honestly without datasets, pretrained weights,
or dedicated GPU time:

- Full Vimeo90K/REDS training and Vid4/UDM10/REDS4 benchmark metrics.
- Installed official metric packages/models or external binaries for final
  paper numbers; the code now records whether each backend is available and
  falls back only where a proxy is explicitly implemented.
- Reproduced BasicVSR++/RealBasicVSR/RealViformer/VSRM/SCST/DiffVSR baselines.
- Large-scale Stage C/D/E rectified-flow teacher/student training.
- Paper-level visual comparison figures and failure-case analysis.

Current CPU tests validate interfaces, shape contracts, losses, metrics, and
small end-to-end smoke paths, but do not replace full research training.
