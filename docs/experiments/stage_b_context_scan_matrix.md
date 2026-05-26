# Stage B Context Length x Scan Policy Matrix

## Goal

This ablation measures whether TrajFlow-VSR's trajectory memory benefits from
longer temporal context consistently across scan policies, or whether the gain
depends on OT/SB-aware ordering.

## Grid

The config `configs/ablation/stage_b_context_scan_matrix.yaml` expands a
Cartesian matrix with:

- context lengths: `3`, `7`, `15`, `31`, `63` frames
- scan policies: `raster`, `hilbert`, `content`, `temporal`, `ot_sb`

Variant names use `frames_N_x_policy`, for example `frames_7_x_ot_sb`.

## Outputs

Each variant writes isolated training artifacts, a final checkpoint, offline
evaluation metrics, and the aggregate files:

- `summary.json`
- `comparison.json`
- `comparison.csv`
- `comparison.md`

The comparison table is intended to become a paper-facing interaction table for
quality, temporal stability, and efficiency.

## CPU Smoke Command

Use a tiny subset while the GPU is occupied:

```bash
CUDA_VISIBLE_DEVICES="" uv run python scripts/run_ablation.py --config configs/ablation/stage_b_context_scan_matrix.yaml --variant frames_3_x_raster --variant frames_7_x_ot_sb --output-dir outputs/ablations/stage_b_context_scan_matrix_smoke --set runtime.dry_run=false --set runtime.device=cpu --set data.height=5 --set data.width=5 --set model.hidden_channels=8 --set model.transport.sinkhorn_iterations=3 --set model.transport.bridge_steps=2 --set optimizer.max_steps=1
```
