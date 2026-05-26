# Baseline Tracking

The proposal requires every baseline to keep its command, commit/version,
weight source, and metric record. `configs/baselines/core_vsr_baselines.yaml`
is the first structured registry for that purpose.

Export the current registry without running any baseline:

```bash
uv run python scripts/export_baseline_records.py --config configs/baselines/core_vsr_baselines.yaml --output-dir experiments/baselines --name core_vsr_baselines
```

Collect metric JSON files referenced by the registry:

```bash
uv run python scripts/export_baseline_metrics.py --config configs/baselines/core_vsr_baselines.yaml --output-dir experiments/baselines --name core_vsr_baseline_metrics
```

The exporter writes:

- `core_vsr_baselines.json`
- `core_vsr_baselines.csv`
- `core_vsr_baselines.md`
- `core_vsr_baselines_checklist.md`
- `core_vsr_baselines_export.json`

The metrics exporter writes:

- `core_vsr_baseline_metrics.json`
- `core_vsr_baseline_metrics.csv`
- `core_vsr_baseline_metrics.md`
- `core_vsr_baseline_metrics_export.json`

Fields containing `TODO`, `TBD`, or `FILL_ME` are treated as incomplete.
This keeps placeholder baseline entries visible while preventing them from
being mistaken for reproducible results.

Metric files may use either the project evaluation format:

```json
{"metrics": {"offline": {"psnr": 30.0, "ssim": 0.9, "fps": 12.0}}}
```

or a flat baseline format:

```json
{"psnr": 30.0, "ssim": 0.9, "fps": 12.0}
```
