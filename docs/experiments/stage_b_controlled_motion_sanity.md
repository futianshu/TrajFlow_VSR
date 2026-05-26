# Stage B Controlled Motion Sanity

目标：用已知整数平移的视频检查 `ot_sb` 是否真的学到跨帧轨迹，而不是只在普通 synthetic 随机场景中比较重建指标。

## 实现

新增数据模式：

- `data.name=controlled_motion`
- `data.motion.shift_x=2`
- `data.motion.shift_y=0`
- `data.motion.wrap=true`

该模式生成固定纹理图案，并让每一帧相对上一帧在 LR 网格上右移 2 像素。HR target 由 LR clip 双线性上采样得到，因此该实验主要检查 trajectory aggregation 与 transport plan，不引入额外真实退化变量。

新增诊断入口：

```bash
uv run python scripts/diagnose_transport.py --config configs/train/stage_b_deterministic.yaml --checkpoint PATH_TO_FINAL_PT --output-dir outputs/diagnostics/stage_b_controlled_motion_scan_gpu --name ot_sb --device cpu --set data.name=controlled_motion --set data.frames=7 --set data.height=24 --set data.width=24 --set data.motion.shift_x=2 --set data.motion.shift_y=0 --set model.hidden_channels=24 --set model.memory.scan_policy=ot_sb --set model.transport.sinkhorn_iterations=8 --set model.transport.bridge_steps=3
```

诊断指标包括 plan entropy、top-1 mass、same/cross-frame mass、oracle trajectory mass、top-1 oracle accuracy、expected displacement、bridge/source deviation 和 reliability 相关性。

## 2026-05-26 GPU Sanity

运行命令：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_ablation.py --config configs/ablation/stage_b_controlled_motion_scan.yaml --output-dir outputs/ablations/stage_b_controlled_motion_scan_gpu --set runtime.dry_run=false --set runtime.device=cuda --set data.frames=7 --set data.height=24 --set data.width=24 --set model.hidden_channels=24 --set model.transport.sinkhorn_iterations=8 --set model.transport.bridge_steps=3 --set optimizer.max_steps=50 --set checkpoint.save_final=true
```

结果摘要：

| rank | variant | PSNR | SSIM | tOF / temporal delta | FPS | MACs G | selection |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | temporal | 15.4859 | 0.4006 | 0.143797 | 547.94 | 0.4251 | 0.9680 |
| 2 | ot_sb | 15.4559 | 0.3986 | 0.144243 | 523.53 | 0.5107 | 0.6322 |
| 3 | ot_sb_no_koopman | 15.2767 | 0.3966 | 0.143651 | 523.12 | 0.4968 | 0.3067 |
| 4 | bridge_temporal | 15.3489 | 0.3944 | 0.144661 | 548.72 | 0.4251 | 0.2381 |

Transport diagnostics on the final checkpoints:

| variant | entropy norm | top-1 mass | same-frame mass | cross-frame mass | oracle mass | oracle cross-frame mass | top-1 oracle acc. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| temporal | 0.9960 | 0.0070 | 0.2573 | 0.7427 | 0.0145 | 0.0102 | 0.0198 |
| bridge_temporal | 0.9960 | 0.0070 | 0.2576 | 0.7424 | 0.0145 | 0.0102 | 0.0184 |
| ot_sb | 0.9960 | 0.0070 | 0.2572 | 0.7428 | 0.0145 | 0.0102 | 0.0198 |
| ot_sb_no_koopman | 0.9960 | 0.0070 | 0.2603 | 0.7397 | 0.0146 | 0.0102 | 0.0181 |

## 判断

- `ot_sb` 在 controlled motion 上仍未超过 `temporal`，但差距比普通 synthetic scan ablation 小。
- 关键问题不是 true soft trajectory scan 是否接上，而是 transport plan 没学到轨迹：plan entropy normalized 约 0.996，top-1 mass 约 0.007，oracle cross-frame mass 约 0.010。
- 即便目标运动是已知的右移 2 像素/帧，当前 OT/SB plan 仍然接近高熵软平均；这会让 soft expectation 抹平证据，无法给 memory 提供比 fixed-pixel temporal scan 更强的路径。
- 下一步应优先加强 transport plan 学习，而不是继续扩大 scan ablation：加入 oracle/top-k trajectory 对照、motion-supervised transport regularization、低熵/周期一致性约束，或用特征相关性之外的显式位移候选。

## 2026-05-26 Transport-Fix GPU Run

本轮执行了三个修正方向：

- 新增 `model.memory.scan_policy=ot_sb_topk`，对每个目标帧只保留 top-k transport candidates 后做 trajectory expectation。
- 新增 `model.memory.scan_policy=ot_sb_hard`，使用 straight-through top-1 plan 生成 hard trajectory scan。
- 新增 `losses.motion_transport` 和 `losses.transport_entropy`，在 `controlled_motion` 批次上用已知整数位移监督 transport plan，并显式压低 plan entropy。

运行命令：

```bash
uv run python scripts/run_ablation.py --config configs/ablation/stage_b_controlled_motion_scan.yaml --output-dir outputs/ablations/stage_b_controlled_motion_transport_fix_gpu --set runtime.dry_run=false --set runtime.device=cuda --set data.frames=7 --set data.height=24 --set data.width=24 --set model.hidden_channels=24 --set model.transport.sinkhorn_iterations=8 --set model.transport.bridge_steps=3 --set optimizer.max_steps=50 --set checkpoint.save_final=true
```

论文表格：

- `docs/paper/tables/stage_b_controlled_motion_transport_fix_gpu_ranking.md`
- `docs/paper/tables/stage_b_controlled_motion_transport_fix_gpu_ranking.tex`

结果摘要：

| rank | variant | PSNR | SSIM | tOF / temporal delta | FPS | selection |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | temporal | 15.49 | 0.4006 | 0.1438 | 548.0 | 0.9893 |
| 2 | ot_sb | 15.46 | 0.3986 | 0.1442 | 512.1 | 0.8376 |
| 3 | ot_sb_topk | 15.43 | 0.3974 | 0.1441 | 457.2 | 0.7262 |
| 4 | ot_sb_hard | 15.41 | 0.3966 | 0.1443 | 502.4 | 0.7138 |
| 7 | ot_sb_supervised | 15.30 | 0.3860 | 0.1460 | 525.1 | 0.2616 |
| 8 | ot_sb_topk_supervised | 15.28 | 0.3824 | 0.1467 | 463.0 | 0.0627 |
| 9 | ot_sb_hard_supervised | 15.27 | 0.3807 | 0.1470 | 500.0 | 0.0463 |

Transport diagnostics on final checkpoints:

| variant | entropy norm | top-1 mass | oracle mass | oracle cross-frame mass | top-1 oracle acc. |
| --- | --- | --- | --- | --- | --- |
| ot_sb | 0.9960 | 0.0070 | 0.0145 | 0.0102 | 0.0198 |
| ot_sb_topk | 0.9960 | 0.0070 | 0.0145 | 0.0102 | 0.0198 |
| ot_sb_hard | 0.9960 | 0.0070 | 0.0145 | 0.0102 | 0.0181 |
| ot_sb_supervised | 0.9731 | 0.0098 | 0.0249 | 0.0171 | 0.1406 |
| ot_sb_topk_supervised | 0.9669 | 0.0104 | 0.0266 | 0.0183 | 0.1493 |
| ot_sb_hard_supervised | 0.9645 | 0.0107 | 0.0273 | 0.0188 | 0.1533 |

判断：

- top-k/hard trajectory scan 本身没有解决 plan 学习问题；未监督的 `ot_sb_topk` 和 `ot_sb_hard` 仍接近高熵均匀 plan。
- motion-supervised transport 确实让 plan 朝 oracle trajectory 移动：top-1 oracle accuracy 从约 2% 提升到 14-15%，oracle mass 约翻倍。
- 但 plan entropy normalized 仍高于 0.96，top-1 mass 只有约 0.01，说明 transport 只产生弱偏置，还不足以作为强轨迹聚合路径。
- 50 step 小规模训练中，supervised variants 的重建指标下降，说明当前监督权重/transport 参数会与 reconstruction warmup 竞争；不能把 `ot_sb_supervised` 直接作为 Stage B 主配置。
- 下一步更合理的路线是把 motion-supervised transport 作为预训练或 warmup curriculum：先冻结/弱化 decoder 学 transport plan，再逐步打开 reconstruction；同时需要更低温度、更小候选半径或显式 displacement prior，避免 Sinkhorn plan 长期保持高熵。
