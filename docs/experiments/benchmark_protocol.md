# Benchmark Protocol Plan

`configs/benchmark/ccfa_core_benchmark.yaml` fixes the initial CCF-A benchmark
matrix for reproducible run planning.

The current grid covers:

- methods: TrajFlow Stage-B plus external baseline placeholders
- datasets: Vid4, UDM10, REDS4, SPMCS-30, RealVSR
- degradations: synthetic, mild-realistic, codec-motion
- scales: x2, x3, x4, x6
- protocols: offline and streaming

Export the plan without running evaluation:

```bash
uv run python scripts/export_benchmark_plan.py --config configs/benchmark/ccfa_core_benchmark.yaml --output-dir experiments/benchmark_plans --name ccfa_core_benchmark
```

The exporter writes JSON, CSV, and Markdown files with one row per planned run.
Internal TrajFlow rows contain CPU-safe evaluation commands. External baseline
rows point back to the baseline registry and the expected metric JSON path.

VideoLQ is tracked separately as a no-reference real-world qualitative set via
`configs/data/videolq.yaml` and `configs/infer/videolq.yaml`; it is not part of
the paired PSNR/SSIM benchmark matrix because it has no GT sequences. Use it for
qualitative comparisons and no-reference metrics once the official NIQE/MUSIQ/
CLIPIQA backends are installed.

Run a selected internal subset on CPU:

```bash
CUDA_VISIBLE_DEVICES="" uv run python scripts/run_benchmark.py --config configs/benchmark/ccfa_core_benchmark.yaml --method trajflow_stage_b --dataset vid4 --degradation synthetic --scale x2 --protocol offline --output-dir outputs/benchmark_runs/ccfa_core_smoke --set model.hidden_channels=8 --set model.transport.sinkhorn_iterations=4 --set model.transport.bridge_steps=3
```

If the configured trained checkpoint is not available yet, the runner records
`skipped_missing_checkpoint` instead of fabricating benchmark numbers. Once
`evaluation.checkpoint_path` points to a real checkpoint, the same command writes
per-run metrics plus `summary.json`, `summary.csv`, and `summary.md`.
The summary includes quality metrics, calibration metrics, FPS/latency/VRAM,
and model profile fields such as parameters and estimated MACs/GMACs.

For a dependency-light interface smoke run before datasets/checkpoints exist,
override the selected row back to synthetic data and allow missing checkpoints:

```bash
CUDA_VISIBLE_DEVICES="" uv run python scripts/run_benchmark.py --config configs/benchmark/ccfa_core_benchmark.yaml --method trajflow_stage_b --dataset vid4 --degradation synthetic --scale x2 --protocol offline --output-dir outputs/benchmark_runs/ccfa_core_synthetic_smoke --allow-missing-checkpoints --set data.name=synthetic --set data.frames=2 --set data.height=6 --set data.width=6 --set model.hidden_channels=8 --set model.transport.sinkhorn_iterations=4 --set model.transport.bridge_steps=3
```
