# TrajFlow-VSR

Reliability-calibrated conditional transport over spacetime trajectories for arbitrary-scale real-world video super-resolution.

## Quick Start

This project uses `uv` for Python and dependency management.

```bash
uv sync
uv run python scripts/train.py --config configs/train/stage_a_tokenizer.yaml --dry-run
uv run python scripts/train.py --config configs/train/stage_a_real_manifest.yaml --dry-run
uv run python scripts/train.py --config configs/train/stage_a_mixed.yaml --dry-run
uv run python scripts/train.py --config configs/train/stage_b_deterministic.yaml --dry-run
uv run python scripts/train.py --config configs/train/stage_b_frame_manifest_full.yaml --dry-run
uv run python scripts/train.py --config configs/train/stage_c_rectified_flow.yaml --dry-run
uv run python scripts/train.py --config configs/train/stage_c_rectified_flow_full.yaml --dry-run
uv run python scripts/train.py --config configs/train/stage_d_distill.yaml --dry-run
uv run python scripts/train.py --config configs/train/stage_d_distill_full.yaml --dry-run
uv run python scripts/train.py --config configs/train/stage_e_streaming.yaml --dry-run
uv run python scripts/train.py --config configs/train/stage_e_streaming_full.yaml --dry-run
uv run python scripts/evaluate.py --config configs/eval/offline.yaml --dry-run
uv run python scripts/evaluate.py --config configs/eval/streaming.yaml --dry-run
uv run python scripts/evaluate.py --config configs/eval/frame_manifest.yaml --dry-run
uv run python scripts/evaluate.py --config configs/eval/paper_official.yaml --dry-run
uv run python scripts/infer_video.py --config configs/infer/synthetic.yaml --dry-run
uv run python scripts/visualize_uncertainty.py --config configs/eval/visualization.yaml --dry-run
uv run python scripts/visualize_trajectory.py --config configs/eval/visualization.yaml --dry-run
uv run python scripts/run_ablation.py --config configs/ablation/stage_b_scan_policy_grid.yaml --dry-run
uv run python scripts/run_ablation.py --config configs/ablation/stage_b_context_scan_matrix.yaml --dry-run
uv run python scripts/run_ablation.py --config configs/ablation/stage_b_core_module_grid.yaml --dry-run
uv run python scripts/run_ablation.py --config configs/ablation/stage_c_flow_step_grid.yaml --dry-run
uv run python scripts/run_ablation.py --config configs/ablation/stage_d_distillation_grid.yaml --dry-run
uv run python scripts/run_ablation.py --config configs/ablation/stage_e_protocol_grid.yaml --dry-run
uv run python scripts/export_paper_table.py --input outputs/ablations/stage_b_context_scan_matrix/comparison.json --output-dir docs/paper/tables --name stage_b_context_scan_matrix
uv run python scripts/export_baseline_records.py --config configs/baselines/core_vsr_baselines.yaml --output-dir experiments/baselines --name core_vsr_baselines
uv run python scripts/export_baseline_metrics.py --config configs/baselines/core_vsr_baselines.yaml --output-dir experiments/baselines --name core_vsr_baseline_metrics
uv run python scripts/export_benchmark_plan.py --config configs/benchmark/ccfa_core_benchmark.yaml --output-dir experiments/benchmark_plans --name ccfa_core_benchmark
uv run python scripts/run_benchmark.py --config configs/benchmark/ccfa_core_benchmark.yaml --method trajflow_stage_b --max-runs 1 --dry-run
uv run python scripts/audit_readiness.py --output-dir outputs/readiness --name ccfa_readiness
uv run python scripts/degrade_data.py --hr-root data/raw/YOUR_HR_ROOT --lr-output-root data/processed/YOUR_DATASET_mild_real_x4 --manifest-output data/splits/YOUR_DATASET_mild_real_x4_train_manifest.json --profile mild_real --scale 4 --clip-length 5
CUDA_VISIBLE_DEVICES="" uv run python -m unittest discover -s tests
CUDA_VISIBLE_DEVICES="" uv run pytest
uv run ruff check src scripts tests
```

The project currently pins CUDA 12.8 PyTorch wheels:

```bash
torch==2.11.0+cu128
torchvision==0.26.0+cu128
torchaudio==2.11.0+cu128
```

While another job is using the GPU, keep all validation on CPU:

```bash
CUDA_VISIBLE_DEVICES="" uv run python scripts/train.py --config configs/train/stage_a_tokenizer.yaml --set runtime.dry_run=false --set runtime.device=cpu --set data.height=8 --set data.width=8 --set data.frames=2 --set model.hidden_channels=8 --set optimizer.max_steps=1
CUDA_VISIBLE_DEVICES="" uv run python scripts/train.py --config configs/train/stage_b_deterministic.yaml --set runtime.dry_run=false --set runtime.device=cpu --set data.height=6 --set data.width=6 --set data.frames=2 --set model.hidden_channels=8 --set model.transport.sinkhorn_iterations=4 --set model.transport.bridge_steps=3 --set optimizer.max_steps=1
CUDA_VISIBLE_DEVICES="" uv run python scripts/train.py --config configs/train/stage_c_rectified_flow.yaml --set runtime.dry_run=false --set runtime.device=cpu --set data.height=6 --set data.width=6 --set data.frames=2 --set model.hidden_channels=8 --set model.transport.sinkhorn_iterations=4 --set model.transport.bridge_steps=3 --set optimizer.max_steps=1
CUDA_VISIBLE_DEVICES="" uv run python scripts/train.py --config configs/train/stage_d_distill.yaml --set runtime.dry_run=false --set runtime.device=cpu --set data.height=6 --set data.width=6 --set data.frames=2 --set model.hidden_channels=8 --set model.transport.sinkhorn_iterations=4 --set model.transport.bridge_steps=3 --set model.flow.teacher_steps=2 --set optimizer.max_steps=1
CUDA_VISIBLE_DEVICES="" uv run python scripts/train.py --config configs/train/stage_e_streaming.yaml --set runtime.dry_run=false --set runtime.device=cpu --set data.height=6 --set data.width=6 --set data.frames=2 --set model.hidden_channels=8 --set model.transport.sinkhorn_iterations=4 --set model.transport.bridge_steps=3 --set model.flow.teacher_steps=2 --set optimizer.max_steps=2
CUDA_VISIBLE_DEVICES="" uv run python scripts/evaluate.py --config configs/eval/offline.yaml --set runtime.dry_run=false --set runtime.device=cpu --set data.height=6 --set data.width=6 --set data.frames=2 --set model.hidden_channels=8 --set model.transport.sinkhorn_iterations=4 --set model.transport.bridge_steps=3
CUDA_VISIBLE_DEVICES="" uv run python scripts/evaluate.py --config configs/eval/streaming.yaml --set runtime.dry_run=false --set runtime.device=cpu --set data.height=6 --set data.width=6 --set data.frames=2 --set model.hidden_channels=8 --set model.transport.sinkhorn_iterations=4 --set model.transport.bridge_steps=3
CUDA_VISIBLE_DEVICES="" uv run python scripts/evaluate.py --config configs/eval/frame_manifest.yaml --set runtime.dry_run=false --set runtime.device=cpu --set data.manifest_path=data/splits/YOUR_DATASET_test_manifest.json --set evaluation.clip_count=4 --set model.hidden_channels=8 --set model.transport.sinkhorn_iterations=4 --set model.transport.bridge_steps=3
CUDA_VISIBLE_DEVICES="" uv run python scripts/infer_video.py --config configs/infer/synthetic.yaml --set runtime.dry_run=false --set runtime.device=cpu --set data.height=6 --set data.width=6 --set data.frames=2 --set model.hidden_channels=8 --set model.transport.sinkhorn_iterations=4 --set model.transport.bridge_steps=3 --set inference.max_visualization_frames=1
CUDA_VISIBLE_DEVICES="" uv run python scripts/visualize_uncertainty.py --config configs/eval/visualization.yaml --set runtime.dry_run=false --set runtime.device=cpu --set data.height=6 --set data.width=6 --set data.frames=2 --set model.hidden_channels=8 --set model.transport.sinkhorn_iterations=4 --set model.transport.bridge_steps=3 --set visualization.max_frames=1
CUDA_VISIBLE_DEVICES="" uv run python scripts/visualize_trajectory.py --config configs/eval/visualization.yaml --set runtime.dry_run=false --set runtime.device=cpu --set data.height=6 --set data.width=6 --set data.frames=2 --set model.hidden_channels=8 --set model.transport.sinkhorn_iterations=4 --set model.transport.bridge_steps=3 --set visualization.max_frames=1
```

Training runs write `history.json` and `manifest.json` under `project.output_dir` by default. Checkpointing is controlled by the `checkpoint` config block or CLI flags:

```bash
CUDA_VISIBLE_DEVICES="" uv run python scripts/train.py --config configs/train/stage_b_deterministic.yaml --set runtime.dry_run=false --set runtime.device=cpu --set checkpoint.save_final=true --checkpoint-dir checkpoints/stage_b_deterministic
CUDA_VISIBLE_DEVICES="" uv run python scripts/train.py --config configs/train/stage_b_deterministic.yaml --resume checkpoints/stage_b_deterministic/final.pt --set runtime.dry_run=false --set runtime.device=cpu
```

Stage B-E can initialize tokenizer and uncertainty encoder weights from a Stage A checkpoint:

```bash
CUDA_VISIBLE_DEVICES="" uv run python scripts/train.py --config configs/train/stage_b_deterministic.yaml --set runtime.dry_run=false --set runtime.device=cpu --set model.pretrained.path=checkpoints/stage_a_tokenizer/final.pt --set model.pretrained.freeze_components='["tokenizer", "uncertainty"]'
```

Frame-sequence data can be indexed with:

```bash
uv run python scripts/prepare_data.py --root data/raw/YOUR_SEQUENCE_ROOT --output data/splits/train_manifest.json --dataset custom --split train --clip-length 5 --stride 1
uv run python scripts/prepare_data.py --root data/raw/YOUR_LR_SEQUENCE_ROOT --hr-root data/raw/YOUR_HR_SEQUENCE_ROOT --output data/splits/paired_train_manifest.json --dataset custom --split train --clip-length 5 --stride 1
uv run python scripts/degrade_data.py --hr-root data/raw/YOUR_HR_SEQUENCE_ROOT --lr-output-root data/processed/YOUR_DATASET_mild_real_x4 --manifest-output data/splits/YOUR_DATASET_mild_real_x4_train_manifest.json --dataset custom --split train --profile mild_real --scale 4 --clip-length 5 --stride 1
uv run python scripts/prepare_data.py --root data/raw/vimeo90k --output data/splits/vimeo90k_train_manifest.json --dataset vimeo90k --split train --layout vimeo90k --split-file sep_trainlist.txt --clip-length 7
uv run python scripts/prepare_data.py --root data/raw/REDS --output data/splits/reds_train_manifest.json --dataset reds --split train --layout reds --clip-length 15 --stride 5
uv run python scripts/prepare_data.py --root data/raw/SPMCS/BIx4 --hr-root data/raw/SPMCS/GT --output data/splits/spmcs_bix4_test_manifest.json --dataset spmcs --split test --layout generic --hr-layout generic --clip-length 31 --stride 31
uv run python scripts/prepare_data.py --root data/raw/RealVSR/LQ_test --hr-root data/raw/RealVSR/GT_test --output data/splits/realvsr_test_manifest.json --dataset realvsr --split test --layout generic --hr-layout generic --clip-length 50 --stride 50
uv run python scripts/prepare_data.py --root data/raw/VideoLQ/Input --output data/splits/videolq_real_test_manifest.json --dataset videolq --split real_test --layout generic --clip-length 0 --stride 1 --min-frames 1
```

`scripts/prepare_data.py` scans image-sequence directories into JSON manifests under `data/splits/`; Stage B-E can use these with `data.name=frame_manifest`. When `--hr-root` is provided, matching LR/HR sequence ids are stored in the same manifest and training reads the real HR target frames instead of synthesizing targets by interpolation. Formal training configs set `data.require_paired=true` so an unpaired manifest fails fast instead of silently training on interpolated pseudo-targets. `scripts/degrade_data.py` generates LR sequences from HR frames with `bicubic`, `mild_real`, `strong_real`, or `codec_motion` degradation profiles and can directly emit a paired manifest. Supported layout presets are `generic`, `vimeo90k`, `reds`, `vid4`, and `udm10`. VideoLQ is no-reference real-world data, so `configs/data/videolq.yaml`, `configs/eval/no_reference.yaml`, and `configs/infer/videolq.yaml` keep it outside paired PSNR/SSIM benchmark runs. `configs/train/stage_b_frame_manifest_full.yaml` is the first formal manifest-training template and samples clips sequentially across steps. Stage B-E include OT/SB trajectory, selective SSM/Koopman memory, wavelet/operator decoding, anti-aliasing, data consistency, and uncertainty calibration paths. Stage E alternates offline and streaming modes. Inference and visualization scripts use imageio/Pillow-backed PNG frame export plus JSON manifests by default, while PPM remains available for dependency-light checks.

Stage A can also pretrain from degraded paired manifests using `configs/train/stage_a_real_manifest.yaml`; the loader derives `clean_lr`, artifact maps, reliability maps, and degradation labels from the paired LR/HR frames and manifest degradation profile. `configs/train/stage_a_mixed.yaml` alternates synthetic degradation batches with real manifest batches and records `data_source` in training history.

Evaluation can run on paired manifests with `configs/eval/frame_manifest.yaml`; set `evaluation.clip_count` for multi-clip aggregation and `evaluation.checkpoint_path` to load a trained model checkpoint.

No-reference real-world evaluation runs through `configs/eval/no_reference.yaml`; it skips PSNR/SSIM/LPIPS/DISTS reference metrics and reports temporal activity, sharpness, blockiness, efficiency, and official no-reference metric backend status.

For the Phase 1 minimum validation, Stage B exposes scan-policy ablations through `model.memory.scan_policy` (`ot_sb`, `temporal`, `raster`, `hilbert`, `content`) and context-length ablations through `data.frames` (`3/7/15/31/63`). The grids are recorded in `configs/ablation/stage_b_scan_policy_grid.yaml`, `configs/ablation/stage_b_context_length_grid.yaml`, and the Cartesian interaction grid `configs/ablation/stage_b_context_scan_matrix.yaml`; the core-module grid `configs/ablation/stage_b_core_module_grid.yaml` covers w/o OT/SB, unbalanced OT, Koopman, reliability calibration, wavelet anti-aliasing, neural operator, and data consistency. Stage C/D/E add flow-step, consistency-distillation, and offline/streaming protocol grids. The ablation runner saves a final checkpoint per variant, runs the paired evaluation config, and writes `summary.json` plus `comparison.json/csv/md` ranking.

Evaluation supports `evaluation.metric_backend=official` and records metric backend status for official/proxy/missing LPIPS, DISTS, NIQE, VMAF, FVD, MUSIQ, and CLIPIQA adapters. `evaluation.posterior_samples=N` reports posterior sample variance/error summaries; `inference.posterior_samples=N` exports stochastic sample frames.

Evaluation also records dependency-free Conv/Linear MAC estimates in the model profile when `evaluation.profile_macs=true`, so benchmark exports can report parameters, MACs/GMACs, FPS, latency, and VRAM together.

Baseline reproducibility records are tracked in `configs/baselines/core_vsr_baselines.yaml`. Export them with `scripts/export_baseline_records.py` to keep command, commit/version, weight source, metric path, and TODO status visible before running external baselines. After external baseline metric JSON files are available, `scripts/export_baseline_metrics.py` collects them into JSON/CSV/Markdown comparison tables and marks missing files as `missing_metrics`.

The fixed CCF-A benchmark run matrix is tracked in `configs/benchmark/ccfa_core_benchmark.yaml`. `scripts/export_benchmark_plan.py` expands method x dataset x degradation x scale x protocol coverage into JSON/CSV/Markdown without running evaluation; `scripts/run_benchmark.py` executes selected internal TrajFlow rows on CPU and writes selected plan, metrics, JSON, CSV, and Markdown summaries. See `docs/experiments/critical_gap_closure.md` for the current boundary between implemented code and work that still requires datasets, pretrained weights, or GPU training.

`scripts/export_data_inventory.py` exports a manifest inventory and protocol audit without reading image tensors. `scripts/estimate_degradation.py` estimates offline degradation preprocessing time/storage from an existing manifest without writing LR frames.

`scripts/audit_readiness.py` exports a CCF-A readiness report covering code modules, configs, manifests, checkpoints, baseline registry completeness, official metric backends, and paper docs. It keeps external-resource gaps explicit instead of turning them into silent TODOs.

Run a CPU smoke subset of that grid with:

```bash
CUDA_VISIBLE_DEVICES="" uv run python scripts/run_ablation.py --config configs/ablation/stage_b_scan_policy_grid.yaml --variant raster --variant ot_sb --output-dir outputs/ablations/stage_b_scan_policy_smoke --set runtime.dry_run=false --set runtime.device=cpu --set data.frames=2 --set data.height=6 --set data.width=6 --set model.hidden_channels=8 --set model.transport.sinkhorn_iterations=4 --set model.transport.bridge_steps=3 --set optimizer.max_steps=1 --set checkpoint.save_final=true
CUDA_VISIBLE_DEVICES="" uv run python scripts/run_ablation.py --config configs/ablation/stage_b_context_length_grid.yaml --variant frames_3 --variant frames_7 --output-dir outputs/ablations/stage_b_context_length_smoke --set runtime.dry_run=false --set runtime.device=cpu --set data.height=5 --set data.width=5 --set model.hidden_channels=8 --set model.transport.sinkhorn_iterations=3 --set model.transport.bridge_steps=2 --set optimizer.max_steps=1
CUDA_VISIBLE_DEVICES="" uv run python scripts/run_ablation.py --config configs/ablation/stage_b_context_scan_matrix.yaml --variant frames_3_x_raster --variant frames_7_x_ot_sb --output-dir outputs/ablations/stage_b_context_scan_matrix_smoke --set runtime.dry_run=false --set runtime.device=cpu --set data.height=5 --set data.width=5 --set model.hidden_channels=8 --set model.transport.sinkhorn_iterations=3 --set model.transport.bridge_steps=2 --set optimizer.max_steps=1
CUDA_VISIBLE_DEVICES="" uv run python scripts/run_benchmark.py --config configs/benchmark/ccfa_core_benchmark.yaml --method trajflow_stage_b --dataset vid4 --degradation synthetic --scale x2 --protocol offline --output-dir outputs/benchmark_runs/ccfa_core_synthetic_smoke --allow-missing-checkpoints --set data.name=synthetic --set data.frames=2 --set data.height=6 --set data.width=6 --set model.hidden_channels=8 --set model.transport.sinkhorn_iterations=4 --set model.transport.bridge_steps=3
```

See `PROJECT_STRUCTURE.md` for the repository layout and staged implementation plan.
