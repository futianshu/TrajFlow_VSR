# Configs

Experiment configurations live here. Keep stage-level training configs, model variants, dataset paths, evaluation protocols, and ablation settings separate so each run can be reproduced with `uv run`.

Current training entry points:

- `configs/train/stage_a_tokenizer.yaml`
- `configs/train/stage_a_real_manifest.yaml`
- `configs/train/stage_a_mixed.yaml`
- `configs/train/stage_a_vimeo90k_mild_real_x4.yaml`
- `configs/train/stage_b_deterministic.yaml`
- `configs/train/stage_b_frame_manifest_full.yaml`
- `configs/train/stage_b_vimeo90k_mild_real_x4.yaml`
- `configs/train/stage_c_rectified_flow.yaml`
- `configs/train/stage_c_rectified_flow_full.yaml`
- `configs/train/stage_c_vimeo90k_mild_real_x4.yaml`
- `configs/train/stage_d_distill.yaml`
- `configs/train/stage_d_distill_full.yaml`
- `configs/train/stage_d_vimeo90k_mild_real_x4.yaml`
- `configs/train/stage_e_streaming.yaml`
- `configs/train/stage_e_streaming_full.yaml`
- `configs/train/stage_e_vimeo90k_mild_real_x4.yaml`

Each training config includes a `checkpoint` block with `output_dir`, `resume_path`, `save_every_steps`, `save_final`, and `save_history`.

Stage B-E configs include `model.pretrained`, which can load Stage A tokenizer/uncertainty weights and optionally freeze selected components.

Stage B-E configs also expose `model.memory.scan_policy` for the Phase 1 trajectory aggregation ablation. Supported values are `ot_sb`, `ot_sb_topk`, `ot_sb_hard`, `bridge_temporal`, `temporal`, `raster`, `hilbert`, and `content`; `ot_sb` builds a soft expectation trajectory sequence from the OT/SB transport plan, `ot_sb_topk` keeps only `model.memory.trajectory_topk` candidates per target frame before expectation, and `ot_sb_hard` uses a straight-through top-1 trajectory plan. `bridge_temporal` keeps the older reliability-gated bridge followed by fixed-pixel temporal scan. `model.memory.use_koopman=false` disables the Koopman prediction head for the w/o Koopman run. Transport configs expose `spatial_radius` and `temporal_radius` for local OT/SB candidate masking.

Controlled-motion transport sanity runs can also enable `losses.motion_transport` and `losses.transport_entropy`. These losses are intended for transport-plan debugging or curriculum warmup on synthetic known-motion clips, not as a default Stage B reconstruction objective.

Training configs may define a `curriculum` block. When `curriculum.enabled=true`, each phase can set `start_step`/`end_step`, `freeze_components`, `loss_multipliers`, `loss_overrides`, and `loss_schedules`; the runner resolves effective loss weights every step and records `curriculum_phase` plus `weight.*` entries in `history.json`.

Current data entry points:

- `configs/data/frame_manifest.yaml`
- `configs/data/degradation_mild_real.yaml`
- `configs/data/vimeo90k.yaml`
- `configs/data/reds.yaml`
- `configs/data/vid4.yaml`
- `configs/data/udm10.yaml`
- `configs/data/spmcs.yaml`
- `configs/data/realvsr.yaml`
- `configs/data/videolq.yaml`

Data templates support optional paired targets with `prepare.hr_root` and optional target resizing with `data.hr_height` / `data.hr_width`. VideoLQ is kept as a no-reference qualitative dataset and is paired with `configs/infer/videolq.yaml`, not the paired PSNR/SSIM benchmark matrix.

Current evaluation entry points:

- `configs/eval/offline.yaml`
- `configs/eval/streaming.yaml`
- `configs/eval/frame_manifest.yaml`
- `configs/eval/no_reference.yaml`
- `configs/eval/paper_official.yaml`
- `configs/eval/visualization.yaml`

Current inference entry points:

- `configs/infer/synthetic.yaml`
- `configs/infer/videolq.yaml`

Current baseline registry entry points:

- `configs/baselines/core_vsr_baselines.yaml`

Current benchmark protocol entry points:

- `configs/benchmark/ccfa_core_benchmark.yaml`

Current ablation entry points:

- `configs/ablation/stage_b_scan_policy_grid.yaml`
- `configs/ablation/stage_b_context_length_grid.yaml`
- `configs/ablation/stage_b_context_scan_matrix.yaml`
- `configs/ablation/stage_b_controlled_motion_scan.yaml`
- `configs/ablation/stage_b_transport_curriculum_grid.yaml`
- `configs/ablation/stage_b_transport_two_phase_grid.yaml`
- `configs/ablation/stage_b_core_module_grid.yaml`
- `configs/ablation/stage_c_flow_step_grid.yaml`
- `configs/ablation/stage_d_distillation_grid.yaml`
- `configs/ablation/stage_e_protocol_grid.yaml`

Run ablations with `scripts/run_ablation.py`; pass `--variant` to select a subset and `--set dotted.path=value` to apply shared overrides to every variant. Training ablations can define `evaluation_config` and `evaluation_overrides` so each variant is evaluated from its final checkpoint and summarized with quality/efficiency metrics. `selection.metrics`, `selection.directions`, and `selection.weights` control the generated `comparison.json/csv/md` ranking. Ablation configs can also define `matrix.axes` to expand Cartesian grids such as context length x scan policy without hand-writing every variant.

Export baseline registry records with `scripts/export_baseline_records.py`. Fields containing `TODO`, `TBD`, or `FILL_ME` are treated as incomplete so external baseline setup work remains visible. Once external baseline metric JSON files exist, collect them with `scripts/export_baseline_metrics.py`; missing metric files are kept in the comparison as `missing_metrics`.

Export benchmark run matrices with `scripts/export_benchmark_plan.py`. The benchmark plan fixes method x dataset x degradation x scale x protocol coverage and writes CPU-safe internal commands plus external baseline metric placeholders. Execute selected internal TrajFlow rows with `scripts/run_benchmark.py`; external baselines remain registry/metric-file driven until their third-party repos and weights are installed.

Run `scripts/export_data_inventory.py` before training to audit manifest roles, paired status, sequence/clip counts, and no-reference datasets. Run `scripts/estimate_degradation.py` to estimate offline LR degradation preprocessing time and storage from an existing manifest without writing data.

Run `scripts/audit_readiness.py` before paper freeze to export a machine-readable CCF-A readiness report for manifests, checkpoints, baseline completeness, official metric backend availability, and required paper docs.
