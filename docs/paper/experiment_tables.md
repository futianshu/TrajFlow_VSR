# Experiment Table Export

`scripts/export_paper_table.py` turns ablation comparison artifacts into
paper-facing Markdown and LaTeX tables.

Inputs can be either:

- `comparison.json`
- `summary.json` containing a `comparison` block

Typical use after an ablation run:

```bash
uv run python scripts/export_paper_table.py --input outputs/ablations/stage_b_context_scan_matrix/comparison.json --output-dir docs/paper/tables --name stage_b_context_scan_matrix
```

For the context length x scan policy grid, variant names use `_x_`, so the
exporter also writes per-metric matrix tables such as:

- `stage_b_context_scan_matrix_psnr_matrix.md`
- `stage_b_context_scan_matrix_psnr_matrix.tex`
- `stage_b_context_scan_matrix_fps_matrix.md`
- `stage_b_context_scan_matrix_fps_matrix.tex`

The ranking table follows the ablation runner's selection score and keeps
quality, temporal stability, efficiency, latency, and model profile fields when
they are present in the comparison rows.
