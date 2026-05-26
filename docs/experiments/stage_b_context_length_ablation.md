# Stage B Context Length Ablation

目标：验证 `CCFA_VSR_NEW_MODEL_PROPOSAL.md` 中第一阶段最小实验的长上下文问题，即 3/7/15/31/63 帧输入在 Stage B deterministic OT/SB + Koopman-SSM 聚合下的质量、时序稳定性和效率变化。

配置入口：

```bash
configs/ablation/stage_b_context_length_grid.yaml
```

当前 variants：

- `frames_3`
- `frames_7`
- `frames_15`
- `frames_31`
- `frames_63`

默认配置使用较小空间尺寸，目的是让长上下文 grid 不会在误运行时占用过大内存。正式实验时可通过 CLI 覆盖 `data.height`、`data.width`、`model.hidden_channels`、`model.transport.sinkhorn_iterations` 和数据集 manifest。

CPU smoke 示例：

```bash
CUDA_VISIBLE_DEVICES="" uv run python scripts/run_ablation.py --config configs/ablation/stage_b_context_length_grid.yaml --variant frames_3 --variant frames_7 --output-dir outputs/ablations/stage_b_context_length_smoke --set runtime.dry_run=false --set runtime.device=cpu --set data.height=5 --set data.width=5 --set model.hidden_channels=8 --set model.transport.sinkhorn_iterations=3 --set model.transport.bridge_steps=2 --set optimizer.max_steps=1
```

输出：

- `summary.json`：每个 context length 的训练 loss、checkpoint、评估结果和 artifacts。
- `comparison.json`：扁平化指标、best-by-metric 和综合 ranking。
- `comparison.csv`：可直接导入表格工具。
- `comparison.md`：可直接复制到实验记录或论文草稿。
