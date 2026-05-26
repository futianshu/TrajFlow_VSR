# Submission Checklist

This checklist converts the CCF-A proposal into concrete internal gates.

## Code And Reproducibility

- [ ] `uv sync` reproduces the environment on a clean machine.
- [ ] All official training configs load with `scripts/train.py --dry-run`.
- [ ] All evaluation configs load with `scripts/evaluate.py --dry-run`.
- [ ] `scripts/audit_readiness.py` is exported before paper freeze.
- [ ] CPU tests pass with `CUDA_VISIBLE_DEVICES="" uv run pytest`.
- [ ] Final model checkpoints are recorded with config snapshots.

## Data

- [ ] Vimeo90K train/val manifests exist under `data/splits/`.
- [ ] REDS train/val manifests exist under `data/splits/`.
- [ ] Vid4, UDM10, REDS4, and SPMCS-30 evaluation manifests exist.
- [ ] RealVSR paired test manifest exists for real-world degradation evaluation.
- [ ] VideoLQ no-reference manifest exists for real-world qualitative inference.
- [ ] Synthetic, realistic, and codec degradation profiles are fixed.
- [ ] Long-context splits cover 3/7/15/31/63 frames.
- [ ] Streaming protocol splits are frozen before final ablations.

## Baselines

- [ ] BasicVSR++ repo, commit, weights, commands, and metrics are recorded.
- [ ] RealBasicVSR repo, commit, weights, commands, and metrics are recorded.
- [ ] RealViformer repo, commit, weights, commands, and metrics are recorded.
- [ ] VSRM or another Mamba/SSM VSR baseline is recorded.
- [ ] SCST or another diffusion/Mamba restoration baseline is recorded.
- [ ] DiffVSR or another robust diffusion VSR baseline is recorded.

## Metrics

- [ ] PSNR/SSIM are reported on the agreed channel/protocol.
- [ ] LPIPS and DISTS use official packages for final tables.
- [ ] NIQE/MUSIQ/CLIPIQA backend availability is recorded.
- [ ] tOF or warping error protocol is fixed.
- [ ] VMAF/FVD availability is recorded or explicitly excluded.
- [ ] FPS, latency, VRAM, parameters, and MACs are reported.
- [ ] Reliability ECE and selective reconstruction curves are reported.

## Experiments

- [ ] Stage A tokenizer/uncertainty pretraining is complete.
- [ ] Stage B deterministic OT/SB + Koopman-SSM benchmark is complete.
- [ ] Stage C rectified-flow teacher benchmark is complete.
- [ ] Stage D one-step distillation benchmark is complete.
- [ ] Stage E offline/streaming comparison is complete.
- [ ] Full ablation grid is exported to Markdown/LaTeX tables.
- [ ] Quality-speed curves include one-step/two-step/four-step results.

## Paper

- [ ] Method has mathematical definitions for transport, memory, flow, and operator decoding.
- [ ] Main figures show the trajectory transport story clearly.
- [ ] Related work directly contrasts RealViformer, SCST, DiffVSR, and Mamba/SSM VSR.
- [ ] Qualitative figures include uncertainty and posterior samples.
- [ ] Failure cases are documented.
- [ ] Limitations honestly describe compute, hallucination, and real-video generalization risks.
