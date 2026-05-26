# Formal Experiment Runbook

This runbook is the non-smoke path from the current prototype to paper numbers.
It intentionally separates executable project entry points from external data
and GPU requirements.

## 1. Prepare Data Manifests

Create paired frame manifests for training and evaluation:

```bash
uv run python scripts/prepare_data.py --root data/raw/vimeo90k --output data/splits/vimeo90k_train_manifest.json --dataset vimeo90k --split train --layout vimeo90k --split-file sep_trainlist.txt --clip-length 7
uv run python scripts/prepare_data.py --root data/raw/REDS --output data/splits/reds_train_manifest.json --dataset reds --split train --layout reds --clip-length 15 --stride 5
uv run python scripts/prepare_data.py --root data/raw/Vid4 --output data/splits/vid4_test_manifest.json --dataset vid4 --split test --layout vid4 --clip-length 15 --stride 15
uv run python scripts/prepare_data.py --root data/raw/UDM10 --output data/splits/udm10_test_manifest.json --dataset udm10 --split test --layout udm10 --clip-length 15 --stride 15
uv run python scripts/prepare_data.py --root data/raw/SPMCS/BIx4 --hr-root data/raw/SPMCS/GT --output data/splits/spmcs_bix4_test_manifest.json --dataset spmcs --split test --layout generic --hr-layout generic --clip-length 31 --stride 31
uv run python scripts/prepare_data.py --root data/raw/RealVSR/LQ_test --hr-root data/raw/RealVSR/GT_test --output data/splits/realvsr_test_manifest.json --dataset realvsr --split test --layout generic --hr-layout generic --clip-length 50 --stride 50
```

Create no-reference real-world manifests for qualitative inference:

```bash
uv run python scripts/prepare_data.py --root data/raw/VideoLQ/Input --output data/splits/videolq_real_test_manifest.json --dataset videolq --split real_test --layout generic --clip-length 0 --stride 1 --min-frames 1
```

For realistic degradation training, generate paired LR/HR manifests:

```bash
uv run python scripts/degrade_data.py --hr-root data/raw/YOUR_HR_ROOT --lr-output-root data/processed/YOUR_DATASET_mild_real_x4 --manifest-output data/splits/YOUR_DATASET_mild_real_x4_train_manifest.json --profile mild_real --scale 4 --clip-length 7 --stride 1
```

Local data note, 2026-05-26: the old-model datasets are mounted at
`/home/ubuntu/data` (`/data/data`). The available assets include Vimeo90K,
REDS4, Vid4, UDM10, SPMCS, RealVSR, VideoLQ, and codec LR variants. Vimeo90K
uses the standard `vimeo_septuplet/sequences/<group>/<clip>/im1..im7.png`
layout with `sep_trainlist.txt` and `sep_testlist.txt`.

Pilot check before full degradation:

```bash
uv run python scripts/degrade_data.py --hr-root /home/ubuntu/data/OpenDataLab___Vimeo90K/raw/vimeo_septuplet --lr-output-root data/processed/vimeo90k_mild_real_x4_pilot --manifest-output data/splits/vimeo90k_mild_real_x4_pilot_train_manifest.json --dataset vimeo90k_mild_real_x4_pilot --split train --layout vimeo90k --sequence-glob 'sequences/00001/000[1-8]' --profile mild_real --scale 4 --clip-length 7 --stride 1 --min-frames 7 --overwrite
```

This writes an ignored local paired manifest with 8 sequences, 56 frames, and 8
clips. The degradation manifest writer preserves `sequence_glob` for both LR
and HR pairing, so subset pilots do not accidentally scan the full HR tree.

## 2. Stage Training

Run with GPU only when the machine is free. CPU commands are for validation,
not for final paper numbers.

```bash
uv run python scripts/train.py --config configs/train/stage_a_tokenizer.yaml --set runtime.dry_run=false --set runtime.device=cuda
uv run python scripts/train.py --config configs/train/stage_b_frame_manifest_full.yaml --set runtime.dry_run=false --set runtime.device=cuda --set data.manifest_path=data/splits/YOUR_DATASET_train_manifest.json
uv run python scripts/train.py --config configs/train/stage_c_rectified_flow_full.yaml --set runtime.dry_run=false --set runtime.device=cuda --set data.manifest_path=data/splits/YOUR_DATASET_train_manifest.json
uv run python scripts/train.py --config configs/train/stage_d_distill_full.yaml --set runtime.dry_run=false --set runtime.device=cuda --set data.manifest_path=data/splits/YOUR_DATASET_train_manifest.json
uv run python scripts/train.py --config configs/train/stage_e_streaming_full.yaml --set runtime.dry_run=false --set runtime.device=cuda --set data.manifest_path=data/splits/YOUR_DATASET_train_manifest.json
```

Vimeo90K pilot GPU sanity, 2026-05-26:

```bash
uv run python scripts/train.py --config configs/train/stage_a_vimeo90k_mild_real_x4.yaml --output-dir outputs/stage_a_vimeo90k_pilot_gpu_quick --checkpoint-dir outputs/stage_a_vimeo90k_pilot_gpu_quick/checkpoints --set runtime.dry_run=false --set runtime.device=cuda --set data.manifest_path=data/splits/vimeo90k_mild_real_x4_pilot_train_manifest.json --set data.batch_size=1 --set data.frames=7 --set model.hidden_channels=24 --set model.tokenizer.hidden_channels=24 --set model.uncertainty.hidden_channels=24 --set optimizer.max_steps=20 --set optimizer.gradient_accumulation_steps=1 --set checkpoint.save_every_steps=0 --set checkpoint.save_final=true
uv run python scripts/train.py --config configs/train/stage_b_vimeo90k_mild_real_x4.yaml --output-dir outputs/stage_b_vimeo90k_pilot_pretrained_gpu_quick --checkpoint-dir outputs/stage_b_vimeo90k_pilot_pretrained_gpu_quick/checkpoints --set runtime.dry_run=false --set runtime.device=cuda --set data.manifest_path=data/splits/vimeo90k_mild_real_x4_pilot_train_manifest.json --set data.batch_size=1 --set data.frames=7 --set data.height=24 --set data.width=24 --set model.hidden_channels=24 --set model.tokenizer.hidden_channels=24 --set model.uncertainty.hidden_channels=24 --set model.memory.hidden_channels=24 --set model.decoder.hidden_channels=24 --set model.transport.temperature=0.1 --set model.transport.spatial_radius=2 --set model.transport.sinkhorn_iterations=8 --set model.transport.bridge_steps=3 --set model.pretrained.path=outputs/stage_a_vimeo90k_pilot_gpu_quick/checkpoints/final.pt --set optimizer.max_steps=20 --set optimizer.gradient_accumulation_steps=1 --set checkpoint.save_every_steps=0 --set checkpoint.save_final=true --set validation.enabled=false
```

Pilot result: Stage A and Stage B both complete on GPU; Stage B loads 20
tokenizer/uncertainty keys from the Stage A pilot checkpoint with no shape
mismatch. The controlled-motion diagnostic for the Stage B pilot checkpoint is
`top-1 mass=0.0144`, `oracle mass=0.0239`, and `top-1 oracle accuracy=0.0278`.

## 3. Internal Benchmark

After a Stage B or later checkpoint exists, execute selected internal rows:

```bash
uv run python scripts/run_benchmark.py --config configs/benchmark/ccfa_core_benchmark.yaml --method trajflow_stage_b --dataset vid4 --dataset udm10 --degradation synthetic --degradation codec_motion --scale x2 --scale x4 --protocol offline --protocol streaming
```

The runner records missing checkpoints as skipped unless
`--allow-missing-checkpoints` is explicitly passed for smoke testing.

## 4. External Baselines

Export the registry checklist first:

```bash
uv run python scripts/export_baseline_records.py --config configs/baselines/core_vsr_baselines.yaml --output-dir experiments/baselines --name core_vsr_baselines
```

Fill each baseline entry with repo URL, commit, setup command, inference
command, evaluation command, weight source, and final metric JSON path. Then
collect metrics:

```bash
uv run python scripts/export_baseline_metrics.py --config configs/baselines/core_vsr_baselines.yaml --output-dir experiments/baselines --name core_vsr_baseline_metrics
```

## 5. Paper Tables

Export ablation comparisons to Markdown/LaTeX:

```bash
uv run python scripts/export_paper_table.py --input outputs/ablations/stage_b_context_scan_matrix/comparison.json --output-dir docs/paper/tables --name stage_b_context_scan_matrix
```

## 6. Transport Diagnostics

When `ot_sb` underperforms fixed-pixel temporal scan, run the controlled-motion
sanity check and export plan diagnostics before changing the generative stages:

```bash
uv run python scripts/run_ablation.py --config configs/ablation/stage_b_controlled_motion_scan.yaml --output-dir outputs/ablations/stage_b_controlled_motion_scan_gpu --set runtime.dry_run=false --set runtime.device=cuda
uv run python scripts/diagnose_transport.py --config configs/train/stage_b_deterministic.yaml --checkpoint outputs/ablations/stage_b_controlled_motion_scan_gpu/ot_sb/checkpoints/final.pt --output-dir outputs/diagnostics/stage_b_controlled_motion_scan_gpu --name ot_sb --device cpu --set data.name=controlled_motion --set model.memory.scan_policy=ot_sb
```

## 7. Readiness Audit

Before claiming a result in the paper, export the readiness report:

```bash
uv run python scripts/audit_readiness.py --output-dir outputs/readiness --name ccfa_readiness
```
