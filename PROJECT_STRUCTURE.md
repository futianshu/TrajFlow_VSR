# TrajFlow-VSR 工程结构说明

最后更新：2026-05-24

本工程使用 `uv` 管理 Python 版本、虚拟环境和依赖，采用研究代码常用的 `src/` 布局。目录设计围绕 `CCFA_VSR_NEW_MODEL_PROPOSAL.md` 中的主线展开：可靠性建模、OT/SB 轨迹传输、Koopman/SSM 长时序记忆、Rectified Flow 高频后验生成、Spacetime Neural Operator + Wavelet Anti-Aliasing 解码。

## 1. 设计原则

- **代码与实验资产分离**：可复用 Python 代码放在 `src/trajflow_vsr/`；训练脚本放在 `scripts/`；配置放在 `configs/`；实验记录放在 `experiments/`。
- **大文件不进仓库**：数据集、模型权重、推理输出、日志分别放入 `data/`、`checkpoints/`、`outputs/`、`logs/`，并通过 `.gitignore` 默认忽略。
- **按论文模块拆包**：模型代码按 proposal 的关键模块拆分，而不是按临时脚本堆叠，方便后续写论文和做消融。
- **按训练阶段组织配置**：Stage A-E 分别保留独立配置入口，便于复现实验曲线和投稿前 ablation。
- **默认命令走 uv**：后续运行训练、评估、测试时优先使用 `uv run ...`，添加依赖时使用 `uv add ...`。

## 2. 当前目录树

```text
TrajFlow_VSR/
├── .python-version
├── .gitignore
├── pyproject.toml
├── uv.lock
├── README.md
├── main.py
├── CCFA_VSR_NEW_MODEL_PROPOSAL.md
├── PROJECT_STRUCTURE.md
├── assets/
├── checkpoints/
├── configs/
│   ├── ablation/
│   ├── baselines/
│   ├── benchmark/
│   ├── data/
│   ├── eval/
│   ├── infer/
│   ├── model/
│   └── train/
├── data/
│   ├── degradation/
│   ├── external/
│   ├── interim/
│   ├── processed/
│   ├── raw/
│   └── splits/
├── docs/
│   ├── experiments/
│   ├── paper/
│   └── references/
├── experiments/
│   ├── ablations/
│   ├── stage_a_tokenizer/
│   ├── stage_b_deterministic/
│   ├── stage_c_rectified_flow/
│   ├── stage_d_distill/
│   └── stage_e_streaming/
├── logs/
├── outputs/
├── scripts/
├── src/
│   └── trajflow_vsr/
│       ├── baselines/
│       ├── data/
│       ├── evaluation/
│       ├── inference/
│       ├── losses/
│       ├── metrics/
│       ├── models/
│       │   ├── consistency/
│       │   ├── decoder/
│       │   ├── flow/
│       │   ├── memory/
│       │   ├── tokenizer/
│       │   ├── transport/
│       │   └── uncertainty/
│       ├── ops/
│       ├── training/
│       ├── utils/
│       └── visualization/
├── tests/
│   ├── integration/
│   ├── smoke/
│   └── unit/
└── third_party/
```

## 3. 目录职责

### 根目录

- `pyproject.toml`：`uv` 项目入口，后续依赖、测试配置和工具配置都在这里集中维护。
- `uv.lock`：`uv` 生成的依赖锁文件，用于固定可复现实验环境。
- `.python-version`：`uv` 读取的 Python 版本声明。
- `CCFA_VSR_NEW_MODEL_PROPOSAL.md`：模型方案、论文定位、训练路线和投稿验收清单。
- `PROJECT_STRUCTURE.md`：当前文件，用于记录工程结构和目录职责。
- `main.py`：轻量入口占位，后续可改为 CLI 或移交给 `scripts/`。

### `src/trajflow_vsr/`

核心 Python 包，后续所有可复用实现都应优先放在这里。

- `data/`：数据集读取、clip sampling、realistic degradation、streaming split、scale query 构造。
- `models/tokenizer/`：Multi-Scale Evidence Tokenizer。
- `models/uncertainty/`：Degradation-Causal Uncertainty Encoder，输出 artifact、reliability 和 uncertainty。
- `models/transport/`：OT/SB soft trajectory、Sinkhorn、unbalanced OT、bridge builder。
- `models/memory/`：Trajectory Selective State Space、Koopman memory、trajectory scan。
- `models/flow/`：Conditional Rectified Flow residual generator、ODE sampler、distillation student。
- `models/decoder/`：Spacetime Neural Operator、wavelet anti-aliasing、coordinate decoder。
- `models/consistency/`：Reliability-Calibrated Data Consistency Projection。
- `losses/`：`L_ot`、`L_sb`、`L_flow`、`L_koopman`、`L_dc`、`L_temp`、`L_uncertainty` 等。
- `ops/`：可复用底层算子，例如 wavelet transform、Hilbert/Z-order scan、Sinkhorn kernel。
- `training/`：trainer、stage loop、checkpoint、EMA、distributed utilities。
- `evaluation/`：评估协议、benchmark runner、offline/streaming protocol。
- `metrics/`：PSNR/SSIM/LPIPS/DISTS/tOF/VMAF/uncertainty calibration 等指标封装。
- `inference/`：单视频推理、分块推理、online streaming 状态缓存。
- `visualization/`：trajectory、uncertainty、artifact/reliability map、posterior samples 可视化。
- `baselines/`：BasicVSR++、RealBasicVSR、RealViformer、Mamba/Diffusion VSR baseline 适配层。
- `utils/`：配置加载、日志、随机种子、路径、张量工具。

### `configs/`

所有可复现实验配置放在这里。建议后续按以下文件补齐：

- `configs/train/stage_a_tokenizer.yaml`
- `configs/train/stage_a_real_manifest.yaml`
- `configs/train/stage_a_mixed.yaml`
- `configs/train/stage_a_vimeo90k_mild_real_x4.yaml`
- `configs/train/stage_b_deterministic.yaml`
- `configs/train/stage_b_frame_manifest_full.yaml`
- `configs/train/stage_b_vimeo90k_mild_real_x4.yaml`
- `configs/train/stage_c_rectified_flow.yaml`
- `configs/train/stage_c_rectified_flow_full.yaml`
- `configs/train/stage_c_vimeo90k_mild_real_x4.yaml`
- `configs/train/stage_d_distill.yaml`
- `configs/train/stage_d_distill_full.yaml`
- `configs/train/stage_d_vimeo90k_mild_real_x4.yaml`
- `configs/train/stage_e_streaming.yaml`
- `configs/train/stage_e_streaming_full.yaml`
- `configs/train/stage_e_vimeo90k_mild_real_x4.yaml`
- `configs/eval/offline.yaml`
- `configs/eval/streaming.yaml`
- `configs/eval/frame_manifest.yaml`
- `configs/eval/no_reference.yaml`
- `configs/eval/paper_official.yaml`
- `configs/ablation/stage_b_scan_policy_grid.yaml`
- `configs/ablation/stage_b_context_length_grid.yaml`
- `configs/ablation/stage_b_context_scan_matrix.yaml`
- `configs/ablation/stage_b_core_module_grid.yaml`
- `configs/ablation/stage_c_flow_step_grid.yaml`
- `configs/ablation/stage_d_distillation_grid.yaml`
- `configs/ablation/stage_e_protocol_grid.yaml`
- `configs/baselines/core_vsr_baselines.yaml`
- `configs/benchmark/ccfa_core_benchmark.yaml`
- `configs/eval/visualization.yaml`
- `configs/infer/synthetic.yaml`
- `configs/data/frame_manifest.yaml`
- `configs/data/degradation_mild_real.yaml`
- `configs/data/vimeo90k.yaml`
- `configs/data/reds.yaml`
- `configs/data/vid4.yaml`
- `configs/data/udm10.yaml`
- `configs/data/spmcs.yaml`
- `configs/data/realvsr.yaml`
- `configs/data/videolq.yaml`
- `configs/infer/videolq.yaml`
- `configs/ablation/*.yaml`
- `configs/data/*.yaml`
- `configs/model/*.yaml`

### `scripts/`

只放面向命令行的一层薄封装，核心逻辑应调用 `src/trajflow_vsr/`。

推荐入口：

- `scripts/train.py`
- `scripts/run_ablation.py`
- `scripts/export_baseline_records.py`
- `scripts/export_baseline_metrics.py`
- `scripts/export_benchmark_plan.py`
- `scripts/run_benchmark.py`
- `scripts/export_paper_table.py`
- `scripts/audit_readiness.py`
- `scripts/export_data_inventory.py`
- `scripts/estimate_degradation.py`
- `scripts/evaluate.py`
- `scripts/infer_video.py`
- `scripts/prepare_data.py`
- `scripts/degrade_data.py`
- `scripts/visualize_trajectory.py`
- `scripts/visualize_uncertainty.py`

### `data/`

本地数据目录，默认不提交大文件。

- `data/raw/`：原始数据或数据集软链接，例如 Vimeo90K、REDS、Vid4、UDM10、SPMCS、RealVSR、VideoLQ。
- `data/interim/`：中间缓存，例如裁切 clip、预估退化标签。
- `data/processed/`：可直接训练/评估的预处理结果。
- `data/splits/`：固定 train/val/test split。
- `data/degradation/`：退化参数、codec 配置、真实退化 profile。
- `data/external/`：外部 benchmark 元数据或第三方预处理产物。

### `experiments/`

用于记录每个阶段的实验配置快照、指标表、消融表和备注。建议每次关键实验单独建子目录：

```text
experiments/stage_b_deterministic/2026-xx-xx_ot_koopman_v1/
  config.yaml
  metrics.json
  notes.md
```

### `docs/`

论文和项目文档。

- `docs/paper/`：论文大纲、method 草稿、figure 设计、submission checklist。
- `docs/experiments/`：实验计划、ablation 设计、baseline 复现记录。
- `docs/references/`：文献笔记和阅读记录。

### `tests/`

测试目录按风险分层。

- `tests/unit/`：算子、loss、数据变换等局部单元测试。
- `tests/integration/`：小 batch 训练、评估、checkpoint resume。
- `tests/smoke/`：`uv run` 可快速跑通的最小端到端检查。

### `assets/` 与 `third_party/`

- `assets/`：轻量级图表、论文插图草稿、项目静态素材。
- `third_party/`：第三方 baseline 适配、外部代码引用说明或 patch 记录。不要直接混入核心实现。

## 4. uv 使用约定

初始化或同步环境：

```bash
uv sync
```

添加依赖：

```bash
uv add "torch==2.11.0+cu128" "torchvision==0.26.0+cu128" "torchaudio==2.11.0+cu128" --index pytorch=https://download.pytorch.org/whl/cu128
uv add --dev pytest ruff
```

运行脚本：

```bash
uv run python main.py
CUDA_VISIBLE_DEVICES="" uv run python scripts/train.py --config configs/train/stage_a_tokenizer.yaml --set runtime.dry_run=false --set runtime.device=cpu --set optimizer.max_steps=1
CUDA_VISIBLE_DEVICES="" uv run python scripts/train.py --config configs/train/stage_b_deterministic.yaml --set runtime.device=cpu
CUDA_VISIBLE_DEVICES="" uv run python -m unittest discover -s tests
```

当前 `pyproject.toml` 设置为 `package = false`，也就是先把 `uv` 用作 Python 与依赖管理工具。等训练入口和 CLI 稳定后，可以再切换为可安装 package，并添加正式的 `[project.scripts]`。

当前环境已固定为 Python `>=3.13,<3.14` 和 PyTorch `2.11.0+cu128`。当 GPU 正在被其他任务占用时，所有验证命令都应加 `CUDA_VISIBLE_DEVICES=""`，并在训练入口加 `--set runtime.device=cpu`。

锁定依赖：

```bash
uv lock
```

`uv.lock` 生成后建议提交，以保证实验环境可复现。

## 5. 阶段落地顺序

1. **Phase 0**：补齐 `configs/`、`scripts/`、`data/` 的最小可运行入口，建立统一评估协议。
2. **Phase 1**：实现 `tokenizer`、`transport`、`memory`、deterministic decoder 和相关 loss。
3. **Phase 2**：实现 `decoder` 中的 wavelet/operator/anti-aliasing 分支。
4. **Phase 3**：实现 `flow` 中的 conditional rectified flow residual generator。
5. **Phase 4**：加入 consistency distillation 和 one-step student。
6. **Phase 5**：完善 `experiments/`、`docs/paper/` 和 CCF-A 投稿级别消融。

## 6. 当前实现状态

已完成 Phase 0 的第一批代码闭环，并开始落地 Stage A-E、评估、推理和诊断可视化主线：

- `scripts/train.py`：训练入口，支持 `--config`、`--dry-run`、`--set dotted.path=value`、`--resume`、`--output-dir` 和 `--checkpoint-dir`。
- `scripts/run_ablation.py`：消融实验入口，读取 `configs/ablation/*.yaml`，自动展开 variants，为每个 variant 隔离输出目录，可选训练后 checkpoint 评估，并写 `summary.json` 与 `comparison.json/csv/md`。
- `scripts/export_baseline_records.py`：baseline 复现记录导出入口，从 registry 配置生成 JSON/CSV/Markdown 清单和 TODO checklist。
- `scripts/export_baseline_metrics.py`：baseline 指标汇总入口，读取 registry 中的 metrics JSON，生成 JSON/CSV/Markdown 对比表并标记缺失指标。
- `scripts/export_benchmark_plan.py`：benchmark 协议计划导出入口，展开 method × dataset × degradation × scale × protocol 的固定 run matrix，但不执行评估。
- `scripts/run_benchmark.py`：benchmark 执行入口，按 method/dataset/degradation/scale/protocol 过滤固定矩阵，实际运行内部 TrajFlow 评估子集，并写 selected plan、metrics、JSON/CSV/Markdown 汇总；外部 baseline 仍以 registry 指标文件为准。
- `scripts/export_paper_table.py`：论文表格导出入口，从 ablation `comparison.json` 或 `summary.json` 生成 Markdown/LaTeX 排名表和二维矩阵表。
- `scripts/audit_readiness.py`：CCF-A readiness audit 入口，检查核心代码、配置、manifest、checkpoint、baseline registry、official metric backend 和 paper docs。
- `scripts/export_data_inventory.py`：manifest 数据清单和协议审计入口，只读 manifest 元数据，不读取图像 tensor。
- `scripts/estimate_degradation.py`：真实退化预处理耗时/空间估算入口，从现有 manifest 估算 profile 数量、帧数、分片和输出空间。
- `scripts/prepare_data.py`：数据准备入口，扫描图像序列目录并生成 frame-manifest JSON，支持 `generic`、`vimeo90k`、`reds`、`vid4`、`udm10` layout，并可通过 `--hr-root` 写入 paired LR/HR targets。
- `scripts/degrade_data.py`：真实退化数据生成入口，从 HR 图像序列离线合成 LR 序列，支持 `bicubic`、`mild_real`、`strong_real`、`codec_motion` profile，并可直接写 paired manifest。
- `scripts/evaluate.py`：评估入口，支持 offline/streaming protocol、paired frame manifest、checkpoint loading、multi-clip aggregation、`--dry-run` 和 `--set dotted.path=value`。
- `scripts/infer_video.py`：推理入口，支持 synthetic smoke、图像序列/视频文件输入、checkpoint 加载、PNG 帧输出、manifest 和诊断可视化。
- `src/trajflow_vsr/utils/config.py`：零额外依赖的 JSON/TOML/simple-YAML 配置加载器。
- `src/trajflow_vsr/training/runner.py`：训练 runner，支持 dry-run 摘要、Stage A synthetic/degraded/mixed pretraining、Stage B-E PyTorch smoke training、gradient accumulation、gradient clipping、scheduler、periodic validation、best checkpoint、history/manifest 写入、checkpoint resume 和 Stage A pretrained component 加载。
- `src/trajflow_vsr/training/checkpoint.py`：训练 checkpoint 保存/恢复工具，使用 `model_state_dict`、`optimizer_state_dict`、scheduler state、step、summary、history 和 metadata 的标准 payload。
- `src/trajflow_vsr/training/pretrained.py`：Stage A tokenizer/uncertainty checkpoint 部分加载、shape 检查、组件冻结和 trainable parameter 过滤工具。
- `src/trajflow_vsr/experiments/ablation.py`：通用消融 runner，支持训练/评估 base config、variant overrides、matrix axes 笛卡尔展开、per-variant artifact 目录、训练后 evaluation config、指标 ranking 和汇总表导出。
- `src/trajflow_vsr/experiments/paper_tables.py`：论文表格导出工具，读取 ablation comparison rows/ranking，生成 ranking table 和 context x scan 等矩阵表。
- `src/trajflow_vsr/experiments/benchmark.py`：benchmark protocol planner/runner，固定数据集、退化、尺度、协议和方法矩阵，导出 CPU-safe 内部命令、执行内部评估子集，并保留外部 baseline metrics 占位。
- `src/trajflow_vsr/baselines/registry.py`：baseline registry 工具，记录外部 baseline 的 repo、commit、权重来源、运行命令、指标路径和字段完整性。
- `src/trajflow_vsr/baselines/metrics.py`：baseline metrics 工具，收集外部 baseline 指标 JSON，支持项目 evaluation nested 格式和 flat baseline 格式。
- `src/trajflow_vsr/data/manifest.py`：图像序列 manifest 构建、clip 切分、Vimeo90K split file 解析、REDS/Vid4/UDM10/SPMCS/RealVSR layout 适配、paired LR/HR target 绑定、Stage A manifest supervision 构造和 manifest batch 读取工具。
- `src/trajflow_vsr/data/degradation.py`：真实退化 profile、HR-to-LR 张量退化算子、训练帧写入工具和离线 degraded LR dataset 构建工具。
- `src/trajflow_vsr/data/mixed.py`：Stage A synthetic/real manifest 混合采样策略，支持 warmup 和 alternating schedule。
- `configs/train/stage_a_tokenizer.yaml`：Stage A tokenizer/uncertainty 预训练配置，包含退化合成参数、mask ratio、预训练 loss 权重和 checkpoint 配置。
- `configs/train/stage_a_real_manifest.yaml`：Stage A 从 degraded paired manifest 读取真实退化监督的预训练配置模板。
- `configs/train/stage_a_mixed.yaml`：Stage A 合成退化与真实 manifest 交替混合预训练配置模板。
- `configs/train/stage_b_deterministic.yaml`：Stage B deterministic warm-up 配置骨架，包含 checkpoint 和 Stage A pretrained 初始化配置。
- `configs/train/stage_c_rectified_flow.yaml`：Stage C conditional rectified flow residual training smoke 配置骨架，包含 checkpoint 和 Stage A pretrained 初始化/冻结配置。
- `configs/train/stage_c_rectified_flow_full.yaml`：Stage C full-manifest teacher 配置，用于正式 residual posterior/4-step flow 训练。
- `configs/train/stage_d_distill_full.yaml`：Stage D full-manifest distillation 配置，用于从 rectified-flow teacher 蒸馏 one-step student。
- `configs/train/stage_e_streaming_full.yaml`：Stage E full-manifest offline/streaming 联合训练配置，用于 31 帧上下文和 causal protocol。
- `configs/train/stage_d_distill.yaml`：Stage D one-step consistency distillation 配置骨架，包含 checkpoint 和 Stage A pretrained 初始化/冻结配置。
- `configs/train/stage_e_streaming.yaml`：Stage E offline/streaming mixed training 配置骨架，包含 checkpoint 和 Stage A pretrained 初始化/冻结配置。
- `configs/data/frame_manifest.yaml`：通用图像序列 manifest 数据配置模板。
- `configs/data/degradation_mild_real.yaml`：从 HR 序列生成 mild-real LR 数据并写 paired manifest 的配置模板。
- `configs/data/vimeo90k.yaml`：Vimeo90K `sequences/*/*` + split file manifest 配置模板。
- `configs/data/reds.yaml`：REDS 序列目录 manifest 配置模板。
- `configs/data/vid4.yaml`：Vid4 测试序列 manifest 配置模板。
- `configs/data/udm10.yaml`：UDM10 测试序列 manifest 配置模板。
- `configs/data/spmcs.yaml`：SPMCS-30 长上下文测试 manifest 配置模板，默认使用 `BIx4`，并记录 `BDx4` 备用 manifest。
- `configs/data/realvsr.yaml`：RealVSR paired real-world 测试 manifest 配置模板，使用 `LQ_test` 对 `GT_test`。
- `configs/data/videolq.yaml`：VideoLQ no-reference real-world 定性测试 manifest 配置模板，不参与 paired PSNR/SSIM benchmark。
- `configs/eval/offline.yaml`：offline synthetic evaluation protocol，输出质量、时序和 uncertainty calibration 指标。
- `configs/eval/streaming.yaml`：streaming synthetic evaluation protocol，额外检查 causal transport violation。
- `configs/eval/frame_manifest.yaml`：paired frame-manifest benchmark evaluation protocol，支持 checkpoint path 和 clip-count 聚合。
- `configs/eval/no_reference.yaml`：no-reference real-world evaluation protocol，跳过 PSNR/SSIM 等 GT 指标，输出时序、锐度、blockiness、效率和 official no-reference backend status。
- `configs/infer/synthetic.yaml`：synthetic inference smoke 配置，默认 CPU dry-run，输出 PNG frame、manifest 和可选 diagnostics。
- `configs/infer/videolq.yaml`：VideoLQ 图像序列推理配置，默认 CPU dry-run，面向真实视频定性输出和诊断可视化。
- `src/trajflow_vsr/data/synthetic.py`：synthetic video batch 和 Stage A 合成退化 batch，包括 blur、noise、codec/block artifact、motion proxy、exposure 和 reliability/artifact/degradation 监督信号。
- `src/trajflow_vsr/models/tokenizer/stage_a_pretrainer.py`：Stage A 专用预训练模型，串联 Multi-Scale Evidence Tokenizer、Degradation-Causal Uncertainty Encoder、masked LR reconstruction head 和 degradation prediction head。
- `src/trajflow_vsr/models/tokenizer/evidence_tokenizer.py`：Multi-scale evidence tokenizer，融合 low/high band、temporal delta、spacetime coordinates、scale 和 footprint token features。
- `src/trajflow_vsr/losses/stage_a.py`：Stage A masked reconstruction、degradation regression、artifact BCE、reliability calibration loss。
- `src/trajflow_vsr/ops/sinkhorn.py`：Phase 1 OT/SB soft trajectory 所需的 pairwise cost、mass normalization 和 entropic Sinkhorn plan。
- `src/trajflow_vsr/ops/wavelet.py`：Phase 2 decoder 所需的 lightweight low/high-frequency split，用于 wavelet-style frequency consistency。
- `src/trajflow_vsr/models/transport/ot_sb_bridge.py`：reliability-calibrated Sinkhorn transport bridge，输出 `ot_plan`、条件 `transport_plan`、local candidate mask、unmatched/occlusion mass、Schrodinger bridge path、streaming causal mask、transported tokens 和 marginal error。
- `src/trajflow_vsr/models/memory/trajectory_koopman_ssm.py`：Trajectory Koopman-SSM memory，支持 gated selective scan、`ot_sb`、`temporal`、`raster`、`hilbert` 和 `content` scan policy，并可关闭 Koopman prediction head 做消融。
- `src/trajflow_vsr/models/decoder/wavelet_operator_decoder.py`：spacetime operator decoder，支持非整数 scale、coordinate decoding、query footprint、low/high band split 和 anti-aliasing gate。
- `src/trajflow_vsr/losses/reconstruction.py`：Stage B 通用重建、Koopman dynamics 和 reliability-weighted data consistency loss。
- `src/trajflow_vsr/losses/optimal_transport.py`：`L_optimal_transport`，约束 soft trajectory transport 的代价和边缘分布误差。
- `src/trajflow_vsr/losses/schrodinger_bridge.py`：`L_schrodinger_bridge`，约束 bridge path 动能、曲率和扩散日程，为 Stage C residual flow 提供条件路径。
- `src/trajflow_vsr/losses/flow_matching.py`：`L_flow` 和 bridge residual consistency，监督 conditional rectified flow vector field。
- `src/trajflow_vsr/losses/distillation.py`：Stage D consistency distillation 和 teacher-target regularization，支持 one-step student 学习 multi-step teacher。
- `src/trajflow_vsr/losses/streaming.py`：`L_streaming_causality`，约束 streaming mode 下 transport 不向未来帧取证据。
- `src/trajflow_vsr/losses/temporal.py`：`L_temporal`，约束 HR 相邻帧变化与 LR evidence 的时间差分一致。
- `src/trajflow_vsr/losses/wavelet_anti_aliasing.py`：`L_wavelet_frequency` 和 `L_anti_aliasing`，约束跨尺度频带一致性和高频 gate 泄漏。
- `src/trajflow_vsr/losses/trajectory.py`：`L_trajectory`，约束 soft trajectory 的局部性、跨帧非退化匹配和 plan 熵。
- `src/trajflow_vsr/metrics/quality.py`：PSNR、SSIM 近似、temporal delta error、tOF proxy、LPIPS/DISTS proxy、uncertainty-error correlation、reliability ECE 和 selective reconstruction AUC。
- `src/trajflow_vsr/metrics/official.py`：官方指标可选适配层，检测 LPIPS/DISTS/NIQE/VMAF/FVD/MUSIQ/CLIPIQA 后端可用性，并在未安装官方依赖时显式记录 fallback 或 unavailable。
- `src/trajflow_vsr/evaluation/runner.py`：统一评估 runner，复用 synthetic/frame-manifest data、model factory 和 metrics，支持 offline/streaming mode、checkpoint loading、multi-clip aggregation、posterior sample 统计、metric backend status、latency/FPS/profile 统计和 metrics JSON 输出。
- `src/trajflow_vsr/inference/io.py`：基于 imageio/Pillow 的图像序列/视频读取与 PNG frame/video 输出工具。
- `src/trajflow_vsr/inference/runner.py`：统一推理 runner，复用 model factory，支持 checkpoint 加载、offline/streaming mode、posterior sample frame export 和 diagnostics export。
- `src/trajflow_vsr/visualization/export.py`：PNG/PPM/JSON 导出工具，覆盖 uncertainty、artifact/reliability map、trajectory expected target、motion magnitude 和 posterior sample frames。
- `src/trajflow_vsr/visualization/runner.py`：统一可视化 runner，复用 synthetic data、model factory 和 offline/streaming mode，输出 manifest。
- `scripts/visualize_uncertainty.py`：导出 artifact、reliability、motion/texture uncertainty map。
- `scripts/visualize_trajectory.py`：导出 OT/SB soft trajectory map 和 top transport edge graph。
- `src/trajflow_vsr/models/`：按论文模块拆出的 PyTorch 接口骨架，包括 tokenizer、uncertainty、OT/SB bridge、Koopman-SSM memory、rectified flow、wavelet operator decoder 和 data consistency projection。
- `tests/`：标准库 `unittest` 与 `pytest` 测试，当前覆盖配置解析、训练 dry-run 摘要、评估/推理/可视化 dry-run 与 CPU smoke、frame-manifest evaluation、evaluation checkpoint loading、Stage A-E CPU smoke training、Stage A manifest supervision、Stage A mixed sampling、Stage A pretrained loading/freezing、checkpoint save/resume、frame manifest/prepare_data/degrade_data、Stage A 合成退化 batch、真实退化 profile、Sinkhorn/OT bridge、streaming causal transport、rectified flow、consistency distillation、wavelet decoder、metrics、inference IO、visualization export 和 loss 汇总。
- `docs/experiments/stage_b_scan_policy_ablation.md`：Stage B scan policy / w/o Koopman 第一阶段最小验证记录。
- `docs/experiments/stage_b_context_length_ablation.md`：Stage B 3/7/15/31/63 帧长上下文消融记录。
- `docs/experiments/stage_b_context_scan_matrix.md`：Stage B 上下文长度 x 轨迹扫描策略二维消融记录。
- `docs/experiments/baseline_tracking.md`：baseline 命令、版本、权重和指标记录的工程化说明。
- `docs/experiments/benchmark_protocol.md`：CCF-A 固定 benchmark run matrix 和导出流程说明。
- `docs/experiments/critical_gap_closure.md`：proposal 对照后的关键缺口关闭记录和仍需外部资源的边界。
- `docs/paper/experiment_tables.md`：ablation comparison 到论文 Markdown/LaTeX 表格的导出说明。

当前可运行检查：

```bash
uv run python scripts/train.py --config configs/train/stage_a_tokenizer.yaml --dry-run
uv run python scripts/train.py --config configs/train/stage_a_real_manifest.yaml --dry-run
uv run python scripts/train.py --config configs/train/stage_a_mixed.yaml --dry-run
uv run python scripts/train.py --config configs/train/stage_b_deterministic.yaml --dry-run
uv run python scripts/train.py --config configs/train/stage_c_rectified_flow.yaml --dry-run
uv run python scripts/train.py --config configs/train/stage_c_rectified_flow_full.yaml --dry-run
uv run python scripts/train.py --config configs/train/stage_d_distill.yaml --dry-run
uv run python scripts/train.py --config configs/train/stage_d_distill_full.yaml --dry-run
uv run python scripts/train.py --config configs/train/stage_e_streaming.yaml --dry-run
uv run python scripts/train.py --config configs/train/stage_e_streaming_full.yaml --dry-run
uv run python scripts/evaluate.py --config configs/eval/offline.yaml --dry-run
uv run python scripts/evaluate.py --config configs/eval/streaming.yaml --dry-run
uv run python scripts/evaluate.py --config configs/eval/frame_manifest.yaml --dry-run
uv run python scripts/evaluate.py --config configs/eval/paper_official.yaml --dry-run
uv run python scripts/infer_video.py --config configs/infer/synthetic.yaml --dry-run
uv run python scripts/visualize_uncertainty.py --config configs/eval/visualization.yaml --dry-run
uv run python scripts/visualize_trajectory.py --config configs/eval/visualization.yaml --dry-run
uv run python scripts/run_ablation.py --config configs/ablation/stage_b_scan_policy_grid.yaml --dry-run
uv run python scripts/run_ablation.py --config configs/ablation/stage_b_context_scan_matrix.yaml --dry-run
uv run python scripts/run_ablation.py --config configs/ablation/stage_b_core_module_grid.yaml --dry-run
uv run python scripts/run_ablation.py --config configs/ablation/stage_c_flow_step_grid.yaml --dry-run
uv run python scripts/run_ablation.py --config configs/ablation/stage_d_distillation_grid.yaml --dry-run
uv run python scripts/run_ablation.py --config configs/ablation/stage_e_protocol_grid.yaml --dry-run
uv run python scripts/export_paper_table.py --input outputs/ablations/stage_b_context_scan_matrix/comparison.json --output-dir docs/paper/tables --name stage_b_context_scan_matrix
uv run python scripts/export_baseline_records.py --config configs/baselines/core_vsr_baselines.yaml --output-dir experiments/baselines --name core_vsr_baselines
uv run python scripts/export_baseline_metrics.py --config configs/baselines/core_vsr_baselines.yaml --output-dir experiments/baselines --name core_vsr_baseline_metrics
uv run python scripts/export_benchmark_plan.py --config configs/benchmark/ccfa_core_benchmark.yaml --output-dir experiments/benchmark_plans --name ccfa_core_benchmark
uv run python scripts/audit_readiness.py --output-dir outputs/readiness --name ccfa_readiness
CUDA_VISIBLE_DEVICES="" uv run python scripts/run_benchmark.py --config configs/benchmark/ccfa_core_benchmark.yaml --method trajflow_stage_b --dataset vid4 --degradation synthetic --scale x2 --protocol offline --output-dir outputs/benchmark_runs/ccfa_core_synthetic_smoke --allow-missing-checkpoints --set data.name=synthetic --set data.frames=2 --set data.height=6 --set data.width=6 --set model.hidden_channels=8 --set model.transport.sinkhorn_iterations=4 --set model.transport.bridge_steps=3
CUDA_VISIBLE_DEVICES="" uv run python -m unittest discover -s tests
CUDA_VISIBLE_DEVICES="" uv run pytest
uv run ruff check src scripts tests
```

Frame manifest 生成示例：

```bash
uv run python scripts/prepare_data.py --root data/raw/YOUR_SEQUENCE_ROOT --output data/splits/train_manifest.json --dataset custom --split train --clip-length 5 --stride 1
uv run python scripts/degrade_data.py --hr-root data/raw/YOUR_HR_SEQUENCE_ROOT --lr-output-root data/processed/YOUR_DATASET_mild_real_x4 --manifest-output data/splits/YOUR_DATASET_mild_real_x4_train_manifest.json --dataset custom --split train --profile mild_real --scale 4 --clip-length 5 --stride 1
uv run python scripts/prepare_data.py --root data/raw/vimeo90k --output data/splits/vimeo90k_train_manifest.json --dataset vimeo90k --split train --layout vimeo90k --split-file sep_trainlist.txt --clip-length 7
uv run python scripts/prepare_data.py --root data/raw/REDS --output data/splits/reds_train_manifest.json --dataset reds --split train --layout reds --clip-length 15 --stride 5
uv run python scripts/prepare_data.py --root data/raw/VideoLQ/Input --output data/splits/videolq_real_test_manifest.json --dataset videolq --split real_test --layout generic --clip-length 0 --stride 1 --min-frames 1
```

CPU smoke training：

```bash
CUDA_VISIBLE_DEVICES="" uv run python scripts/train.py --config configs/train/stage_a_tokenizer.yaml --set runtime.dry_run=false --set runtime.device=cpu --set data.height=8 --set data.width=8 --set data.frames=2 --set model.hidden_channels=8 --set optimizer.max_steps=1
CUDA_VISIBLE_DEVICES="" uv run python scripts/train.py --config configs/train/stage_b_deterministic.yaml --set runtime.dry_run=false --set runtime.device=cpu --set data.height=6 --set data.width=6 --set data.frames=2 --set model.hidden_channels=8 --set model.transport.sinkhorn_iterations=4 --set model.transport.bridge_steps=3 --set optimizer.max_steps=1
CUDA_VISIBLE_DEVICES="" uv run python scripts/train.py --config configs/train/stage_c_rectified_flow.yaml --set runtime.dry_run=false --set runtime.device=cpu --set data.height=6 --set data.width=6 --set data.frames=2 --set model.hidden_channels=8 --set model.transport.sinkhorn_iterations=4 --set model.transport.bridge_steps=3 --set optimizer.max_steps=1
CUDA_VISIBLE_DEVICES="" uv run python scripts/train.py --config configs/train/stage_d_distill.yaml --set runtime.dry_run=false --set runtime.device=cpu --set data.height=6 --set data.width=6 --set data.frames=2 --set model.hidden_channels=8 --set model.transport.sinkhorn_iterations=4 --set model.transport.bridge_steps=3 --set model.flow.teacher_steps=2 --set optimizer.max_steps=1
CUDA_VISIBLE_DEVICES="" uv run python scripts/train.py --config configs/train/stage_e_streaming.yaml --set runtime.dry_run=false --set runtime.device=cpu --set data.height=6 --set data.width=6 --set data.frames=2 --set model.hidden_channels=8 --set model.transport.sinkhorn_iterations=4 --set model.transport.bridge_steps=3 --set model.flow.teacher_steps=2 --set optimizer.max_steps=2
CUDA_VISIBLE_DEVICES="" uv run python scripts/evaluate.py --config configs/eval/offline.yaml --set runtime.dry_run=false --set runtime.device=cpu --set data.height=6 --set data.width=6 --set data.frames=2 --set model.hidden_channels=8 --set model.transport.sinkhorn_iterations=4 --set model.transport.bridge_steps=3
CUDA_VISIBLE_DEVICES="" uv run python scripts/evaluate.py --config configs/eval/streaming.yaml --set runtime.dry_run=false --set runtime.device=cpu --set data.height=6 --set data.width=6 --set data.frames=2 --set model.hidden_channels=8 --set model.transport.sinkhorn_iterations=4 --set model.transport.bridge_steps=3
CUDA_VISIBLE_DEVICES="" uv run python scripts/evaluate.py --config configs/eval/frame_manifest.yaml --set runtime.dry_run=false --set runtime.device=cpu --set data.manifest_path=data/splits/YOUR_DATASET_test_manifest.json --set evaluation.clip_count=4 --set model.hidden_channels=8 --set model.transport.sinkhorn_iterations=4 --set model.transport.bridge_steps=3
CUDA_VISIBLE_DEVICES="" uv run python scripts/infer_video.py --config configs/infer/synthetic.yaml --set runtime.dry_run=false --set runtime.device=cpu --set data.height=6 --set data.width=6 --set data.frames=2 --set model.hidden_channels=8 --set model.transport.sinkhorn_iterations=4 --set model.transport.bridge_steps=3 --set inference.max_visualization_frames=1
CUDA_VISIBLE_DEVICES="" uv run python scripts/visualize_uncertainty.py --config configs/eval/visualization.yaml --set runtime.dry_run=false --set runtime.device=cpu --set data.height=6 --set data.width=6 --set data.frames=2 --set model.hidden_channels=8 --set model.transport.sinkhorn_iterations=4 --set model.transport.bridge_steps=3 --set visualization.max_frames=1
CUDA_VISIBLE_DEVICES="" uv run python scripts/visualize_trajectory.py --config configs/eval/visualization.yaml --set runtime.dry_run=false --set runtime.device=cpu --set data.height=6 --set data.width=6 --set data.frames=2 --set model.hidden_channels=8 --set model.transport.sinkhorn_iterations=4 --set model.transport.bridge_steps=3 --set visualization.max_frames=1
```
