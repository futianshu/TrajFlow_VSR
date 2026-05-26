# Stage B Transport Curriculum

目标：先让 OT/SB transport plan 在已知整数平移场景中学到非均匀轨迹，再回到完整 reconstruction 目标。此前 top-k/hard scan 只是在高熵 plan 上做后处理，不能解决 plan 本身接近均匀的问题。

## 实现

训练配置支持：

- `curriculum.enabled=true`
- `curriculum.phases.<name>.start_step`
- `curriculum.phases.<name>.end_step`
- `curriculum.phases.<name>.freeze_components`
- `curriculum.phases.<name>.loss_multipliers`
- `curriculum.phases.<name>.loss_overrides`
- `curriculum.phases.<name>.loss_schedules`

runner 每 step 解析有效 loss 权重，并在 history 中写入：

- `curriculum_phase`
- `curriculum_progress`
- `curriculum_frozen`
- `weight.<loss_name>`

## 当前 ablation

入口：

```bash
uv run python scripts/run_ablation.py --config configs/ablation/stage_b_transport_curriculum_grid.yaml --output-dir outputs/ablations/stage_b_transport_curriculum_grid_gpu --set runtime.dry_run=false --set runtime.device=cuda
```

该 grid 包含：

- `ot_sb_no_curriculum`
- `warmup_motion025_entropy001`
- `warmup_motion050_entropy002`
- `warmup_motion050_entropy005`
- `lowtemp_radius2_warmup`
- `topk_warmup`
- `hard_warmup`

## 诊断

每个候选跑完后，应继续用 `scripts/diagnose_transport.py` 看 plan 形状，而不是只看 PSNR/SSIM：

```bash
uv run python scripts/diagnose_transport.py --config configs/train/stage_b_deterministic.yaml --checkpoint outputs/ablations/stage_b_transport_curriculum_grid_gpu/lowtemp_radius2_warmup/checkpoints/final.pt --output-dir outputs/diagnostics/stage_b_transport_curriculum_grid_gpu --name lowtemp_radius2_warmup --device cpu --set data.name=controlled_motion --set data.frames=7 --set data.height=24 --set data.width=24 --set data.motion.shift_x=2 --set data.motion.shift_y=0 --set model.hidden_channels=24 --set model.transport.temperature=0.1 --set model.transport.spatial_radius=2 --set model.transport.sinkhorn_iterations=8 --set model.transport.bridge_steps=3 --set model.memory.scan_policy=ot_sb
```

优先观察：

- `plan_entropy_normalized`
- `plan_top1_mass`
- `plan_oracle_mass`
- `plan_oracle_cross_frame_mass`
- `plan_top1_oracle_accuracy`

通过标准：先要求 oracle/top-1 诊断明显优于无 curriculum 的 `ot_sb`，再考虑把 warmup 搬到普通 synthetic 或 Vimeo90K mild-real Stage B。

## 2026-05-26 GPU Run

运行命令：

```bash
uv run python scripts/run_ablation.py --config configs/ablation/stage_b_transport_curriculum_grid.yaml --output-dir outputs/ablations/stage_b_transport_curriculum_grid_gpu --set runtime.dry_run=false --set runtime.device=cuda
```

结果文件：

- `outputs/ablations/stage_b_transport_curriculum_grid_gpu/comparison.md`
- `outputs/ablations/stage_b_transport_curriculum_grid_gpu/summary.json`
- `outputs/diagnostics/stage_b_transport_curriculum_grid_gpu/*.json`

Reconstruction / efficiency ranking:

| rank | variant | PSNR | SSIM | tOF / temporal delta | FPS | selection |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | ot_sb_no_curriculum | 15.4558 | 0.3986 | 0.1442 | 512.8 | 0.7781 |
| 2 | warmup_motion050_entropy002 | 4.5045 | -0.0172 | 0.1424 | 526.3 | 0.3070 |
| 3 | warmup_motion025_entropy001 | 4.6349 | -0.0074 | 0.1425 | 522.6 | 0.2994 |
| 4 | warmup_motion050_entropy005 | 4.4410 | -0.0238 | 0.1427 | 525.3 | 0.2688 |
| 5 | lowtemp_radius2_warmup | 4.4846 | -0.0166 | 0.1428 | 525.5 | 0.2584 |
| 6 | hard_warmup | 4.4779 | -0.0205 | 0.1426 | 504.6 | 0.2464 |
| 7 | topk_warmup | 4.4885 | -0.0191 | 0.1424 | 464.5 | 0.1987 |

Transport diagnostics:

| variant | entropy norm | top-1 mass | oracle mass | oracle cross-frame mass | top-1 oracle acc. |
| --- | --- | --- | --- | --- | --- |
| ot_sb_no_curriculum | 0.9960 | 0.0070 | 0.0145 | 0.0102 | 0.0198 |
| warmup_motion025_entropy001 | 0.9958 | 0.0070 | 0.0152 | 0.0105 | 0.0248 |
| warmup_motion050_entropy002 | 0.9901 | 0.0080 | 0.0192 | 0.0132 | 0.0841 |
| warmup_motion050_entropy005 | 0.9901 | 0.0080 | 0.0192 | 0.0132 | 0.0853 |
| lowtemp_radius2_warmup | 0.8750 | 0.0479 | 0.0955 | 0.0519 | 0.4355 |
| topk_warmup | 0.9889 | 0.0082 | 0.0198 | 0.0137 | 0.0918 |
| hard_warmup | 0.9849 | 0.0087 | 0.0217 | 0.0150 | 0.1076 |

判断：

- 全程 50-step warmup 且冻结 decoder/consistency 会严重牺牲 reconstruction：所有 warmup variants 的 PSNR 从 `15.46` 掉到约 `4.44-4.63`。这说明当前 warmup 不能作为最终训练协议，只能作为 transport-plan pretraining/curriculum 诊断。
- 单纯 motion loss + entropy schedule 有弱效果，但不足够：`warmup_motion050_entropy002/005` 把 top-1 oracle accuracy 从 `0.0198` 提到约 `0.084-0.085`，plan entropy norm 仍约 `0.990`。
- `topk_warmup` / `hard_warmup` 在 plan 未充分变尖前仍主要是后处理；top-1 oracle accuracy 只到 `0.092/0.108`。
- `lowtemp_radius2_warmup` 是本轮唯一明显改变 plan 形状的配置：entropy norm 从 `0.9960` 降到 `0.8750`，top-1 mass 从 `0.0070` 升到 `0.0479`，oracle mass 从 `0.0145` 升到 `0.0955`，top-1 oracle accuracy 到 `0.4355`。
- 下一步应把 `temperature=0.1`、`spatial_radius=2` 作为核心，而不是只调 loss 权重；同时把 warmup 改成两阶段：短 transport pretrain 后打开 decoder/consistency reconstruction，避免低 PSNR 崩塌。

## Two-Phase Follow-up

入口：

```bash
uv run python scripts/run_ablation.py --config configs/ablation/stage_b_transport_two_phase_grid.yaml --output-dir outputs/ablations/stage_b_transport_two_phase_grid_gpu --set runtime.dry_run=false --set runtime.device=cuda
```

该 grid 用 100 steps 比较：

- `ot_sb_no_curriculum_100`
- `lowtemp_radius2_no_curriculum_100`
- `two_phase10_lowtemp_radius2`
- `two_phase25_lowtemp_radius2`
- `two_phase25_light_recovery`
- `two_phase25_topk_light_recovery`
- `two_phase25_hard_light_recovery`

设计意图：

- 前 10 或 25 steps 用 `temperature=0.1`、`spatial_radius=2`、motion-supervised transport 和 entropy schedule 学尖锐 plan。
- warmup 结束后解冻 decoder/consistency，并恢复 reconstruction/data-consistency 主目标。
- `light_recovery` variants 在恢复阶段保留很弱的 motion/entropy 正则并线性衰减到 0，检查是否能保住 plan 形状而不继续压低 PSNR。

## 2026-05-26 Two-Phase GPU Run

运行命令：

```bash
uv run python scripts/run_ablation.py --config configs/ablation/stage_b_transport_two_phase_grid.yaml --output-dir outputs/ablations/stage_b_transport_two_phase_grid_gpu --set runtime.dry_run=false --set runtime.device=cuda
```

结果文件：

- `outputs/ablations/stage_b_transport_two_phase_grid_gpu/comparison.md`
- `outputs/ablations/stage_b_transport_two_phase_grid_gpu/summary.json`
- `outputs/diagnostics/stage_b_transport_two_phase_grid_gpu/*.json`

Reconstruction / efficiency ranking:

| rank | variant | PSNR | SSIM | tOF / temporal delta | FPS | selection |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | lowtemp_radius2_no_curriculum_100 | 16.5601 | 0.4914 | 0.1305 | 520.7 | 0.9761 |
| 2 | ot_sb_no_curriculum_100 | 16.5100 | 0.4880 | 0.1312 | 523.3 | 0.8370 |
| 3 | two_phase25_hard_light_recovery | 16.3343 | 0.4825 | 0.1302 | 500.4 | 0.5612 |
| 4 | two_phase25_topk_light_recovery | 16.3459 | 0.4817 | 0.1303 | 462.3 | 0.4951 |
| 5 | two_phase25_light_recovery | 16.3202 | 0.4785 | 0.1311 | 520.5 | 0.4711 |
| 6 | two_phase10_lowtemp_radius2 | 16.4055 | 0.4773 | 0.1332 | 521.1 | 0.4499 |
| 7 | two_phase25_lowtemp_radius2 | 16.2472 | 0.4677 | 0.1338 | 518.3 | 0.0917 |

Transport diagnostics:

| variant | entropy norm | top-1 mass | oracle mass | oracle cross-frame mass | top-1 oracle acc. |
| --- | --- | --- | --- | --- | --- |
| ot_sb_no_curriculum_100 | 0.9966 | 0.0064 | 0.0146 | 0.0102 | 0.0201 |
| lowtemp_radius2_no_curriculum_100 | 0.9966 | 0.0164 | 0.0296 | 0.0170 | 0.0295 |
| two_phase10_lowtemp_radius2 | 0.9966 | 0.0164 | 0.0295 | 0.0171 | 0.0288 |
| two_phase25_lowtemp_radius2 | 0.9956 | 0.0170 | 0.0309 | 0.0176 | 0.0481 |
| two_phase25_light_recovery | 0.9878 | 0.0202 | 0.0381 | 0.0211 | 0.1394 |
| two_phase25_topk_light_recovery | 0.9875 | 0.0203 | 0.0383 | 0.0212 | 0.1429 |
| two_phase25_hard_light_recovery | 0.9857 | 0.0208 | 0.0395 | 0.0219 | 0.1503 |

判断：

- 最强 reconstruction 配置不是 two-phase，而是直接使用 `temperature=0.1` + `spatial_radius=2` 的 `lowtemp_radius2_no_curriculum_100`：PSNR/SSIM/tOF 都优于 100-step 默认 `ot_sb` baseline。
- 低温小半径主要带来更局部、更尖的候选分布，而不是强 oracle 对齐：top-1 mass 从 `0.0064` 到 `0.0164`，oracle mass 从 `0.0146` 到 `0.0296`，但 top-1 oracle accuracy 只有 `0.0295`。
- two-phase warmup 后如果直接回 base loss，oracle 偏置会大幅回落；25-step warmup 比 10-step 稍强，但 reconstruction 稍差。
- `light_recovery` 能保住更多 motion supervision 痕迹：top-1 oracle accuracy 到 `0.139-0.150`，但 PSNR 仍低于无 curriculum 的低温小半径配置。
- hard/top-k 在 light recovery 下主要提升 transport 诊断和 temporal delta，但 FPS 或 PSNR 有轻微代价。

当前建议：

- Stage B 主配置优先改成 `model.transport.temperature=0.1`、`model.transport.spatial_radius=2`，不默认启用 transport curriculum。
- transport curriculum 继续作为诊断/预训练工具保留，但下一轮应尝试更短、更弱的 recovery 正则，或只在前若干 step 单独训练 transport 后丢弃该 checkpoint 作为初始化，而不是在同一 reconstruction run 中长期保留。
