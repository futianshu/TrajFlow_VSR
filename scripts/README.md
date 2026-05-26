# Scripts

Command-line entrypoints live here. Keep these files thin: parse arguments, load config, then call reusable code from `src/trajflow_vsr/`.

- `scripts/train.py`: staged training and smoke training, with checkpoint resume/output overrides.
- `scripts/run_ablation.py`: expand and run config-variant ablation grids with isolated per-variant artifacts and a summary JSON.
- `scripts/export_paper_table.py`: convert ablation `comparison.json`/`summary.json` artifacts into Markdown and LaTeX ranking/matrix tables.
- `scripts/export_baseline_records.py`: convert baseline registry configs into JSON/CSV/Markdown reproducibility records and checklists.
- `scripts/export_baseline_metrics.py`: collect metric JSON files referenced by a baseline registry into comparison tables.
- `scripts/export_benchmark_plan.py`: expand benchmark protocol configs into JSON/CSV/Markdown run matrices without executing evaluation.
- `scripts/run_benchmark.py`: execute the internal CPU-safe subset of a benchmark matrix and write selected plan, metrics, CSV, Markdown, and JSON summaries.
- `scripts/audit_readiness.py`: export a CCF-A proposal readiness audit for code, configs, data manifests, checkpoints, baselines, official metrics, and paper docs.
- `scripts/export_data_inventory.py`: export a manifest inventory and protocol audit without reading image tensors.
- `scripts/estimate_degradation.py`: estimate offline degradation preprocessing time/storage from an existing manifest without writing LR data.
- `scripts/prepare_data.py`: scan image-sequence folders and write frame-manifest JSON files, with `generic`/`vimeo90k`/`reds`/`vid4`/`udm10` layout presets and optional paired LR/HR targets via `--hr-root`; VideoLQ uses `generic` without `--hr-root`.
- `scripts/degrade_data.py`: synthesize degraded LR frame sequences from HR data with reusable real-world degradation profiles and write paired manifests.
- `scripts/evaluate.py`: offline/streaming evaluation protocols.
- `scripts/diagnose_transport.py`: export OT/SB plan entropy, oracle motion mass, displacement, and reliability diagnostics for a config/checkpoint.
- `scripts/infer_video.py`: synthetic, image-sequence, or video-file inference with PNG frame export.
- `scripts/visualize_uncertainty.py`: export artifact, reliability, and uncertainty maps.
- `scripts/visualize_trajectory.py`: export OT/SB trajectory maps and top-edge graph summaries.
