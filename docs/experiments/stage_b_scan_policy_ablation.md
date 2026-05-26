# Stage B Scan Policy Ablation

目标：验证 `CCFA_VSR_NEW_MODEL_PROPOSAL.md` 中第一阶段最小实验的核心问题，即 OT/SB soft trajectory scan 相比 raster、Hilbert/Z-order、content-aware 和纯 temporal scan 是否更适合长时序 VSR 聚合。

配置入口：

```bash
configs/ablation/stage_b_scan_policy_grid.yaml
```

当前可比较的 memory 选项：

- `model.memory.scan_policy=raster`
- `model.memory.scan_policy=hilbert`
- `model.memory.scan_policy=content`
- `model.memory.scan_policy=temporal`
- `model.memory.scan_policy=bridge_temporal`
- `model.memory.scan_policy=ot_sb`
- `model.memory.use_koopman=false`

`scripts/run_ablation.py` 会为每个 variant 保存 final checkpoint，并用 `configs/eval/offline.yaml` 做一次同配置评估，最终在 `summary.json` 中同时记录训练 loss、PSNR、SSIM、temporal delta error、latency、FPS 和参数量。它还会输出 `comparison.json`、`comparison.csv` 和 `comparison.md`，按照 `selection.metrics/directions/weights` 生成综合排序，方便直接转成实验表。

CPU smoke 示例：

```bash
CUDA_VISIBLE_DEVICES="" uv run python scripts/run_ablation.py --config configs/ablation/stage_b_scan_policy_grid.yaml --variant raster --variant ot_sb --output-dir outputs/ablations/stage_b_scan_policy_smoke --set runtime.dry_run=false --set runtime.device=cpu --set data.frames=2 --set data.height=6 --set data.width=6 --set model.hidden_channels=8 --set model.transport.sinkhorn_iterations=4 --set model.transport.bridge_steps=3 --set optimizer.max_steps=1 --set checkpoint.save_final=true
```

## 2026-05-26 Phase 1 GPU Smoke

运行命令：

```bash
CUDA_VISIBLE_DEVICES="0" uv run python scripts/run_ablation.py --config configs/ablation/stage_b_scan_policy_grid.yaml --output-dir outputs/ablations/phase1_stage_b_scan_gpu_smoke --set runtime.dry_run=false --set runtime.device=cuda --set data.frames=7 --set data.height=16 --set data.width=16 --set model.hidden_channels=16 --set model.transport.sinkhorn_iterations=6 --set model.transport.bridge_steps=3 --set optimizer.max_steps=10 --set checkpoint.save_final=true
```

运行环境：

- GPU：A100-SXM4-40GB。
- 输入：synthetic，7 frames，16x16 LR，scale x2。
- 模型：hidden_channels=16。
- 训练：每个 variant 10 steps，Stage B deterministic，flow disabled。
- 输出目录：`outputs/ablations/phase1_stage_b_scan_gpu_smoke/`。

结果摘要：

| rank | variant | PSNR | SSIM | tOF / temporal delta | FPS | VRAM GB | selection |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | temporal | 6.9189 | 0.1019 | 0.194008 | 910.53 | 0.3658 | 0.9922 |
| 2 | ot_sb_no_koopman | 6.6542 | 0.0861 | 0.193994 | 912.34 | 0.3658 | 0.3285 |
| 3 | ot_sb | 6.6443 | 0.0853 | 0.193997 | 902.20 | 0.3658 | 0.2970 |
| 4 | content | 6.6757 | 0.0903 | 0.194277 | 14.11 | 0.3658 | 0.1796 |
| 5 | raster | 6.6723 | 0.0900 | 0.194342 | 14.11 | 0.3656 | 0.1328 |
| 6 | hilbert | 6.6725 | 0.0900 | 0.194356 | 14.10 | 0.3658 | 0.1262 |

初步判断：

- 训练、checkpoint、evaluation、comparison export 链路已跑通。
- 该 toy setting 不支持“OT/SB scan 优于 temporal scan”的结论；`temporal` 在综合排名、PSNR、SSIM、LPIPS/DISTS 上更好。
- `ot_sb` 与 `ot_sb_no_koopman` 在 tOF 上略优，但差距极小，不能作为论文证据。
- raster / hilbert / content 的评估 FPS 异常偏低，可能来自 scan sequence layout 或 profiling 计时方式，需要单独排查。
- 下一步应在更真实的退化、更长训练、更长上下文和可视化 crop strip 上验证，而不是直接推进 Stage C 生成分支。

## 排查结论

### 为什么 temporal 暂时强于 ot_sb？

排查时的旧代码中，`ot_sb` 还不是完整的“沿 soft trajectory 重排 token 后做 selective scan”。实际路径是：

- `OTSBTrajectoryBridge` 先根据 transport plan 得到 `transported_grid`。
- 旧实现中 `bridge_grid = 0.5 * (source_grid + transported_grid)`；这会让高 reliability 区域也被 transported evidence 改写。
- `TrajectoryKoopmanSSMMemory(scan_policy=ot_sb)` 再对 `bridge_grid` 按固定像素位置做时间轴 scan。

也就是说，旧版 `ot_sb` 更准确的名字是 **bridge-conditioned temporal scan**，不是严格意义上的 trajectory scan。相比之下，`temporal` 直接对 `source_grid` 做同样的时间轴 scan。在 synthetic toy setting 中，未充分训练的 OT/SB transport 会把局部证据平均到 `bridge_grid` 里，可能稀释低频结构和纹理证据，因此 temporal scan 更强是合理现象。

已做修正：

- `bridge_grid` 改为 reliability-gated bridge，高 reliability 区域保留 `source_grid`，低 reliability 区域才更多使用 `transported_grid`。
- `model.memory.scan_policy=ot_sb` 改为 true soft trajectory scan：对每个 anchor token，使用 `transport_plan` 在每个目标帧内做 soft expectation，构造跨帧 trajectory sequence，再执行 selective scan，并取 anchor 自身时间位置的 state 还原回网格。
- 旧行为保留为 `model.memory.scan_policy=bridge_temporal`，用于比较 source-only temporal、bridge temporal 和 true soft trajectory scan。

后续要证明 OT/SB 的必要性，至少需要继续补上：

- true soft trajectory scan 与 bridge temporal 的配套消融：比较 source-only temporal、reliability-gated bridge temporal、hard top-k trajectory scan、soft expectation trajectory scan。
- 更难协议：遮挡、快速运动、codec artifact、长上下文 15/31/63 帧；在太简单的 synthetic toy 上，temporal scan 本来就很强。

### 为什么 raster / hilbert / content FPS 很低？

这主要来自当前原型的 scan 序列长度差异，而不是 GPU 空闲或 checkpoint 加载问题：

- `temporal` / `bridge_temporal`：sequence shape 约为 `(B*H*W, T, C)`，Python recurrent loop 只循环 `T` 次。
- `ot_sb` true soft trajectory scan：sequence shape 约为 `(B*T*H*W, T, C)`，loop 长度仍为 `T`，但每个 spacetime token 都有自己的 soft trajectory anchor。
- `raster` / `hilbert` / `content`：sequence shape 约为 `(B, T*H*W, C)`，Python recurrent loop 循环 `T*H*W` 次。

在本次 7 帧、16x16 输入下，前者循环 `7` 次，后者循环 `1792` 次，所以 raster/hilbert/content 的 FPS 低是当前 naive Python scan 实现的预期结果。这个结果不能代表优化后的 Mamba/selective-scan kernel 性能。

另一个已确认的度量问题：MAC profiler 原先对 recurrent loop 中复用的 `Linear` 只保留最后一次 hook 结果，没有累加多次调用，因此 raster/hilbert/content 的 MACs 被低估。已修复为累加同一模块的多次调用。

## 2026-05-26 Soft Trajectory Formal GPU Ablation

运行命令：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_ablation.py --config configs/ablation/stage_b_scan_policy_grid.yaml --output-dir outputs/ablations/stage_b_scan_policy_grid_soft_trajectory_formal --set runtime.dry_run=false --set runtime.device=cuda --set data.frames=7 --set data.height=24 --set data.width=24 --set model.hidden_channels=24 --set model.transport.sinkhorn_iterations=8 --set model.transport.bridge_steps=3 --set optimizer.max_steps=50 --set checkpoint.save_final=true
```

运行环境：

- GPU：A100-SXM4-40GB。
- 输入：synthetic，7 frames，24x24 LR，scale x2。
- 模型：hidden_channels=24。
- 训练：每个 variant 50 steps，Stage B deterministic，flow disabled。
- 输出目录：`outputs/ablations/stage_b_scan_policy_grid_soft_trajectory_formal/`。

结果摘要：

| rank | variant | PSNR | SSIM | tOF / temporal delta | FPS | VRAM GB | MACs G | selection |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | temporal | 14.9001 | 0.3308 | 0.188998 | 548.30 | 1.9856 | 0.4251 | 0.9605 |
| 2 | bridge_temporal | 14.8385 | 0.3303 | 0.188942 | 549.04 | 1.9856 | 0.4251 | 0.7936 |
| 3 | ot_sb | 14.8712 | 0.3288 | 0.189280 | 522.66 | 2.0097 | 0.5107 | 0.7079 |
| 4 | ot_sb_no_koopman | 14.7524 | 0.3270 | 0.188571 | 524.14 | 2.0097 | 0.4968 | 0.3705 |
| 5 | raster | 14.7183 | 0.3296 | 0.190480 | 6.24 | 1.9853 | 0.4248 | 0.2258 |
| 6 | content | 14.7399 | 0.3291 | 0.190741 | 6.26 | 1.9856 | 0.4248 | 0.2140 |
| 7 | hilbert | 14.7370 | 0.3288 | 0.190553 | 6.12 | 1.9855 | 0.4248 | 0.1997 |

结论：

- true soft trajectory scan 已经在完整 ablation runner 中跑通，但该 synthetic 7-frame / 24x24 / 50-step setting 仍不支持“OT/SB scan 优于 temporal scan”的性能主张。
- `ot_sb` 相比 `bridge_temporal` 多了真实 soft trajectory anchor，MACs 从约 0.425G 上升到约 0.511G，FPS 从约 549 降到约 523；代价可控，但当前收益不足。
- `ot_sb_no_koopman` 的 tOF 最低，但 PSNR/SSIM 明显落后，不应单独作为支持 OT/SB 的证据。
- raster / Hilbert / content 的 FPS 仍约 6 FPS，主要是 naive Python recurrent loop 的长度为 `T*H*W`，不适合作为效率结论。
- 下一步应把重点转向更难的运动/遮挡/codec 协议，以及改进 transport plan 的监督或正则；在当前 toy synthetic setting 上，固定像素 temporal scan 仍是强基线。
