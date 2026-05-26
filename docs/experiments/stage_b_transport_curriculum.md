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
