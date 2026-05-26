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
