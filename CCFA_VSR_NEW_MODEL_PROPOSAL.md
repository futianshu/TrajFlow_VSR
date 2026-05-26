# CCF-A 目标新工程方案：TrajFlow-VSR

最后更新：2026-05-26

本文档用于后续单独开启一个全新的 VSR 工程。它不是当前 `ST-VSR` 工程的增量改造，而是面向 CCF-A 主会/期刊投稿的重新立项方案。

## 0. 核心定位

### 0.1 论文题目

**TrajFlow-VSR: Reliability-Calibrated Conditional Transport over Spacetime Trajectories for Arbitrary-Scale Real-World Video Super-Resolution**

中文简称：

**可靠性校准轨迹传输视频超分模型**

### 0.2 一句话主张

把真实世界任意尺度视频超分辨率从“多帧回归重建”重新表述为：

> Real-world arbitrary-scale VSR is a reliability-calibrated conditional transport problem over continuous video trajectories.

即：真实世界 VSR 不是简单地把 LR 多帧映射到 HR 多帧，而是在连续时空轨迹上，根据退化可靠性，完成从 degraded LR evidence 到 clean HR video function 的条件传输。

### 0.3 四根理论支柱

```text
OT / Schrödinger Bridge
  -> 跨帧软对齐、遮挡、多假设对应、degraded-to-clean 分布桥接

SSM / Koopman Dynamics
  -> 长时序证据记忆、动态状态演化、offline/online 双模式

Rectified Flow / Flow Matching
  -> HR 高频 residual 的条件后验生成和低步数采样

Neural Operator + Wavelet Anti-Aliasing
  -> 任意尺度输出、频率一致性、低频保真和高频可控生成
```

模块职责必须保持清楚：

- **OT / Schrödinger Bridge**：决定哪些跨帧证据可以被传输，以及退化视频分布如何桥接到清晰视频分布。
- **SSM / Koopman**：决定证据如何沿时间演化并长期记忆。
- **Rectified Flow**：决定 LR 中不可唯一确定的 HR 高频如何作为条件后验被生成。
- **Neural Operator + 小波抗混叠**：决定输出如何在连续时空和不同尺度下保持频率一致。

### 0.4 旧工程失败复盘给出的硬约束

旧工程 `ST_VSR_Project` 已完成 70 轮训练，并在 Vid4、UDM10、REDS4，以及 H.264/H.265 codec LR 协议下完成系统评估。结论是：旧模型不适合作为论文主结果继续推进，新工程不能围绕旧主线做局部修补。

核心失败不是 epoch 选择错误，也不是可视化挑图问题，而是模型结构和训练策略本身存在生成式先验失控：

- `Ours-e65` 在 NIQE、MUSIQ、CLIPIQA 等部分无参考感知指标上看似改善，但可视化出现明显“融化”、伪纹理和结构错乱；Vid4/calendar 树枝和密集纹理区域尤其典型，细节并非真实恢复，而是不稳定 hallucination。
- `Ours-e45` 比 `Ours-e65` 稳定，但在 PSNR/SSIM、LPIPS/DISTS、NIQE/MUSIQ/CLIPIQA 和可视化质量上仍不足以支撑优于 RealBasicVSR、RealViformer 等参考方法的主张。
- SD3 VAE 语义先验、TSM 浅层物理流、reliability-aware fusion、scale-aware INR 在后期与 LPIPS/GAN 等感知/生成约束同时开启，导致模型逐渐偏向自然但结构错误的高频纹理。

因此 TrajFlow-VSR 的立项约束是：

- 不直接把大生成先验接入 VSR 主干，不让生成式分支影响低频结构。
- 先建立稳定 deterministic VSR backbone；rectified flow/GAN/diffusion 类生成只能作为后期 residual 模块。
- 低频结构必须由数据一致性、重建损失和频率约束保证。
- 高频生成只能在低置信、高不确定区域小幅介入，并由 reliability/uncertainty 显式校准。
- `base_rgb + pred_residual` 形式必须配套 residual 幅度约束、频带约束和 low-frequency projection。
- 必须显式建模 temporal consistency，避免纹理漂移、融化和伪结构。
- 必须加入 wavelet/frequency decomposition，区分低频结构和高频纹理。
- 必须输出 uncertainty/reliability map，并用它控制跨帧聚合、data consistency 权重和高频生成强度。
- 模型选择不能依赖 NIQE/MUSIQ/CLIPIQA 单点改善，必须同时检查 full-reference 指标、no-reference 指标、tOF、可视化和时序 crop strip。

## 1. 与旧工程的边界

旧工程 `ST_VSR_Project` / `ST-VSR` 探索的重点是：

- 3 帧输入。
- SD3 VAE 语义先验。
- TSM/浅层物理流。
- reliability-aware fusion。
- scale-aware INR。
- `x2/x3/x4` 多尺度训练。

新工程应避免围绕这些模块做“小创新”。TrajFlow-VSR 的变化应体现在：

- 从 3 帧局部建模转向长上下文视频轨迹建模。
- 从 optical flow / attention 的经验式对齐转向 OT / Schrödinger Bridge 驱动的概率传输与分布桥接。
- 从 deterministic RGB residual 回归转向条件 posterior transport / rectified flow 生成。
- 从固定局部时序融合转向 trajectory-conditioned selective state space / Koopman memory。
- 从普通 INR 解码转向 spacetime neural operator + wavelet anti-aliasing coordinate decoder。
- 从单一输出转向可控随机采样、可信区间、per-region uncertainty。
- 从离线三帧 VSR 扩展到 offline + online streaming 双模式。

旧 `ST_VSR_Project` / ICONIP 论文可以作为前期探索或内部 baseline。新论文不应沿用“SD3 VAE 语义分支 + TSM 物理分支 + SSFM/SFT + INR”的主叙事。

可复用内容：

- Vimeo90K/REDS/Vid4/UDM10 loader。
- realistic degradation 经验。
- tOF/LPIPS/DISTS/MUSIQ/CLIPIQA 评估工具。
- selection score 的思想，但应升级为 uncertainty-aware multi-objective model selection。

不可作为新论文主贡献：

- SD3 VAE 语义分支。
- TSM 物理分支。
- SSFM/SFT 双流融合。
- 单纯 INR decoder。
- 当前 IPEC2026 已写过的 ST-VSR 主线。

## 2. 科学问题与假设

### 2.1 核心问题

真实世界 VSR 的不确定性来自三个层面：

- **退化不确定性**：blur、noise、codec、motion、downsampling 组合未知。
- **运动不确定性**：遮挡、非刚性形变、快速运动导致对应关系多解。
- **高频不确定性**：LR 中缺失的纹理并非唯一可恢复。

现有 deterministic VSR 常把三者压成一个点估计，输出容易平滑或错误 hallucination。现有 diffusion VSR 虽然能生成高频，但代价是慢、随机闪烁、难以长视频 streaming。

本项目要回答：

> 能否用 OT/SB 建立可靠软对应和分布桥接，用 SSM/Koopman 建模长时序动态，用 rectified flow 生成高频 residual posterior，并用 neural operator + wavelet anti-aliasing 保持跨尺度频率一致，从而同时实现真实感、时序一致性、长上下文和任意尺度输出？

### 2.2 核心假设

- **假设 1**：视频超分的跨帧证据应通过 OT/SB 形成多假设 soft trajectory，而不是依赖单一 optical flow、raster scan 或固定窗口 attention。
- **假设 2**：视频长时序不是普通 token 序列，而是可由 SSM/Koopman memory 描述的动态系统；状态应沿可靠轨迹演化。
- **假设 3**：高频细节应建模为条件分布，而不是确定性残差；rectified flow 比多步 diffusion 更适合低延迟 VSR。
- **假设 4**：任意尺度 VSR 应被表述为 video function operator learning，并显式处理频率带宽和 anti-aliasing，而不是仅靠坐标 MLP 插值。
- **假设 5**：prior strength 必须由 uncertainty/reliability 校准，否则生成先验会在 artifact 区域制造非物理纹理。

## 3. 研究脉络与缺口

### 3.1 真实退化下的可靠性问题

RealViformer 指出真实世界 VSR 中，退化和伪影会被 recurrent/attention 结构传播和放大；空间注意力的 query/key 可能来自真实内容和 artifact 混合信号，导致错误聚合。它进一步观察到 channel attention 相比 spatial attention 对 artifact 更稳健。

启发：

- 新模型不能无条件相信空间相似性。
- 跨帧信息聚合必须携带 reliability / uncertainty。
- 看起来相似的 patch 未必是物理上可信的对应。

### 3.2 生成式视频复原趋势

SCST 将 self-supervised ControlNet 与 spatio-temporal Mamba 结合，用 diffusion prior 处理 real-world VSR。DiffVSR 给出了 robust VSR 的 diffusion recipe。DiTVR 从 diffusion transformer、trajectory-aware attention、wavelet-guided flow-consistent sampler 出发做 zero-shot video restoration。FlashVSR 强调 diffusion-based VSR 必须走向 one-step / streaming / real-time。

启发：

- 生成式 prior 在真实 VSR 中有价值，但多步 diffusion 的时延和时序一致性仍是硬伤。
- few-step / one-step generative restoration 更适合真实系统。
- 高频生成和低频数据一致性需要显式分离。

### 3.3 SSM / Mamba 视频建模趋势

Mamba 的 selective state space 机制提供线性复杂度长序列建模能力。2025 年 VSR 方向已经出现 MambaVSR、TS-Mamba、VSRM、SCST 等工作，分别探索 content-aware scanning、trajectory-aware shifted SSM、robust Mamba-based VSR 和 spatio-temporal continuous Mamba。

启发：

- 长视频 VSR 的核心瓶颈不应继续只靠 transformer attention。
- 普通 raster scan Mamba 对视频空间结构并不天然合理。
- 更有前景的是按运动轨迹或语义连通图重排 token 后再做 selective scan。

### 3.4 Flow Matching / Rectified Flow

Flow Matching 将生成建模表述为学习连续归一化流的向量场，训练时回归固定概率路径上的 vector field。Rectified flow 用直线路径连接 noise 和 data，在高分辨率生成中已经成为重要趋势。Consistency Models 说明 one-step/few-step 生成可以通过一致性蒸馏获得。

启发：

- VSR 可被表述为从噪声高频残差到真实 HR 高频残差的条件传输。
- 学习整段视频 residual 的 transport 比逐帧 diffusion 更适合控制时序一致性。
- 可用 consistency distillation 把多步 rectified flow 压成 1-2 步推理。

### 3.5 Neural Operator 与信号处理

Fourier Neural Operator 的核心思想是学习 function space 到 function space 的映射，而不是固定维度向量之间的映射。对 VSR 来说，这为同一模型跨尺度、跨分辨率、跨时间查询提供更强的理论语言。小波、多速率信号处理和 anti-aliasing 则能让任意尺度输出不只是坐标插值，而是频带可控的连续视频函数重建。

启发：

- arbitrary-scale 不应只被当作 INR 小技巧，而应被写成 LR observation function 到 HR video function 的 operator learning。
- 高频 residual 可由生成模型给出，低频和跨尺度一致性由 neural operator 和 wavelet consistency 约束。

## 4. 模型总览

模型名：`TrajFlow-VSR`

输入：

- LR video clip: `Y_{1:T}`。
- target scale: `s`，支持 `x2/x3/x4/x6` 和非整数 scale。
- query time: offline 模式可输出整段 HR，online 模式只输出当前/过去帧。
- optional degradation hint: codec metadata、estimated CRF、camera motion metadata 等。

输出：

- HR video: `X_hat_{1:T}`。
- uncertainty map: `U_{1:T}`。
- artifact/reliability map: `A_{1:T}`, `R_{1:T}`。
- optional posterior samples: 同一 LR 条件下的多个 plausible HR outputs。

整体结构：

```text
LR video
  -> Multi-scale / wavelet evidence tokenizer
  -> Degradation-causal uncertainty encoder
  -> Reliability-calibrated OT/SB soft trajectory aggregation
  -> Trajectory SSM / Koopman memory
  -> Wavelet-aware neural operator decoder
  -> Optional rectified-flow high-frequency residual generator
  -> Reliability-Calibrated Data Consistency Projection
  -> HR video + uncertainty
```

默认论文主线先以 deterministic path 产生 `X_det`；optional residual generator 只对 `High(X)` 做可靠性校准后的后验采样。

## 5. 关键模块

### 5.1 Multi-Scale Evidence Tokenizer

目标：把 LR 视频拆成多尺度、低频/高频、空间/时间证据 token。

设计：

- 使用 wavelet/Laplacian pyramid 分离低频结构和疑似高频残差。
- 每帧生成三类 token：
  - `structure tokens`：低频结构、轮廓、颜色。
  - `texture tokens`：局部边缘、重复纹理、字母/线条。
  - `artifact tokens`：block artifact、ringing、noise、motion smear。
- 每个 token 带上坐标 `(x, y, t)`、scale footprint `(cell_w, cell_h)`、degradation embedding 和 reliability prior。

### 5.2 Degradation-Causal Uncertainty Encoder

目标：估计退化类型、退化强度、局部 artifact、跨帧不确定性。

输出：

```text
d_global = [blur, noise, codec, motion, scale, severity, exposure, rolling_shutter]
A(x,y,t) = artifact probability
R(x,y,t) = reliability = calibrated confidence
U_motion(x,y,t) = trajectory uncertainty
U_texture(x,y,t) = high-frequency posterior uncertainty
```

理论解释：

- `R` 不只是视觉 mask，而是后续 conditional transport 的 data-confidence coefficient。
- `U_motion` 控制 trajectory 聚合宽度。
- `U_texture` 控制生成采样温度。

建议实现：

- CNN + lightweight ViT/SSM hybrid encoder。
- 自监督任务：synthetic degradation label prediction、masked LR token reconstruction、clean/degraded contrastive alignment。

### 5.3 OT/SB Soft Trajectory & Bridge Builder

目标：不用硬光流做单一对齐，而是建立 token 级多假设轨迹，并把 degraded LR evidence 到 clean HR evidence 的关系写成可靠性校准的分布桥接问题。

输入：

- evidence token embeddings。
- reliability map。
- coarse motion proposal。

输出：

```text
P_{i -> j}^{t -> t+k}
q_t(z | Y_degraded, X_clean)
```

其中：

- `P_{i -> j}^{t -> t+k}` 表示 token `i` 在当前帧与未来/历史帧 token `j` 属于同一物理轨迹的概率。
- `q_t(z | Y_degraded, X_clean)` 表示 degraded distribution 与 clean distribution 之间的中间桥接状态，可用于训练后续 residual transport。

核心机制：

- 局部 patch correlation 给出候选。
- 语义 embedding similarity 给出长期对应。
- reliability 抑制 artifact token 的错误匹配。
- entropic OT / Sinkhorn 约束匹配分布。
- unbalanced OT 允许遮挡、出入画和低可靠区域的 unmatched mass。
- Schrödinger Bridge 将退化视频和清晰视频看成两个边界分布，为 rectified flow residual generator 提供更合理的条件路径。

### 5.4 Trajectory Selective State Space / Koopman Memory

目标：对长视频做线性复杂度建模，支持 31-100 帧上下文，并让 latent state 的时间演化具有动态系统解释。

设计：

- 对每个 soft trajectory 构建 token sequence。
- 沿 trajectory 做 selective state space scan。
- 在 latent observable space 中加入 Koopman-style linear dynamics regularization。
- 对局部空间邻域做 Hilbert / Z-order scan 补充空间连续性。
- 对不同轨迹之间用 sparse cross-trajectory mixer 交换信息。

状态更新：

```text
h_i(t+1) = A_i(Y_t, R_t, U_t) h_i(t) + B_i(Y_t, R_t, U_t) x_i(t)
z_i(t)   = C_i(Y_t, R_t, U_t) h_i(t)
```

可选 Koopman 约束：

```text
phi(z_{t+1}) ≈ K phi(z_t) + E_t
```

其中 `phi` 是 latent observable，`K` 是局部/全局动态算子，`E_t` 表示新观测证据注入。该约束不要求真实视频线性，而是要求在提升后的 observable space 中长期动态更可预测。

### 5.5 Optional Conditional Rectified Flow Residual Generator

目标：生成 HR 高频 residual posterior，而不是只做确定性残差回归。该模块是后期可选模块，不进入低频结构主干，也不作为 Stage B deterministic backbone 的必要条件。

把 HR 分成低频可观测项和高频不可观测项：

```text
X = Base(Y, s) + H
```

其中：

- `Base(Y, s)`：data-consistent low-frequency base。
- `H`：需要生成的 high-frequency residual，只允许承载 wavelet 高频子带。

生成残差必须经过 reliability/uncertainty gate：

```text
G_H = clamp(α (1 - R) + β U_texture, 0, G_max)
X_hat = X_det + G_H * H_flow
Low(H_flow) -> 0
||H_flow|| <= ε(scale, degradation)
```

其中 `X_det` 是 deterministic backbone 输出，`G_H` 控制生成分支介入强度。高 reliability、低 uncertainty 区域中 `G_H` 应接近 0；低 reliability 不等于允许任意 hallucination，而是允许在 residual 幅度、频带和时序一致性约束下小幅补全。

训练 rectified flow：

```text
z_0 ~ N(0, I)
z_1 = Enc_H(H_gt)
z_tau = (1 - tau) z_0 + tau z_1
v_theta(z_tau, tau, C) -> z_1 - z_0
L_flow = || v_theta(z_tau, tau, C) - (z_1 - z_0) ||_2^2
```

条件 `C` 包括 trajectory memory、LR evidence tokens、reliability / uncertainty maps、scale token 和 query cell footprint。

推理模式：

- 4-step ODE：高质量。
- 2-step ODE：默认。
- 1-step distilled：实时/streaming。
- deterministic-only：论文早期主结果和结构保真验证必须保留该模式，作为生成分支是否失控的参照。

### 5.6 Spacetime Neural Operator + Wavelet Anti-Aliasing Decoder

目标：学习从 LR observation function 到 HR video function 的算子，同时显式处理频带、采样 footprint 和 anti-aliasing。该 decoder 首先产生 deterministic `X_det`，再为可选 residual generator 提供高频条件场。

输入：

- latent residual field。
- low-frequency base。
- query coordinates `(x, y, t, cell_w, cell_h)`。
- scale token。

模块：

- Fourier mixing layers：学习全局空间频率响应。
- Wavelet operator layers：在多尺度频带上分别建模低频结构和高频 residual。
- Local coordinate MLP：补充局部高频细节。
- Anti-aliasing head：根据 `cell_w/cell_h` 和目标 scale 抑制非物理高频。
- Temporal phase head：修正非整数帧/非整数 scale 的相位偏移。

输出：

```text
HR(x,y,t) = Base(x,y,t) + Decode_operator(z_H, x,y,t,cell)
```

### 5.7 Reliability-Calibrated Data Consistency Projection

目标：防止生成模型在低频结构上偏离 LR 证据。

设计：

```text
D_s(Low(X_hat)) ≈ Low(Y)
L_dc = || R * (D_s(X_hat) - Y) ||_1
```

高 reliability 区域强制保真；低 reliability 区域降低硬约束，但仍保留低频一致性、残差幅度、wavelet band-limit 和 temporal consistency 约束。`R` 不是“生成开关”，而是 data consistency、transport mass、temporal aggregation 和 residual posterior sampling 的统一校准量。

额外约束：

```text
L_low_drift = || Low(X_hat) - Low(X_det) ||_1
L_res_amp   = || G_H * H_flow ||_1
L_band      = || Low(H_flow) ||_1
```

这三项用于避免旧工程中 `base_rgb + pred_residual` 残差无界扩张导致的结构融化。

## 6. 训练方案

### 6.1 Stage A：退化与 token 预训练

数据：

- Vimeo90K / REDS / YouTube 真实视频片段。
- 合成真实退化。
- 可选无配对真实低清视频。

任务：

- degradation vector prediction。
- artifact/reliability map prediction。
- masked LR token reconstruction。
- clean/degraded contrastive alignment。

目标：先让 tokenizer 和 uncertainty encoder 学会什么是可靠证据。

### 6.2 Stage B：Deterministic warm-up

先不启用 rectified flow，只训练：

- tokenizer。
- OT/SB soft trajectory bridge。
- trajectory SSM / Koopman memory。
- spacetime wavelet neural operator decoder。

禁用项：

- GAN / diffusion / rectified flow 高频生成。
- 大生成先验直接注入。
- 以 NIQE/MUSIQ/CLIPIQA 作为 checkpoint 主选择标准。
- 未经 gate 的 `base_rgb + pred_residual` 全频残差。

损失：

```text
L = L_char + L_freq + L_dc + L_temp + L_ot + L_sb + L_koopman + L_traj
```

目标：建立稳定的低频结构、概率轨迹、长时序动态和任意尺度输出能力。

### 6.3 Stage C：Conditional rectified flow 训练

冻结或半冻结 deterministic backbone，训练 residual flow。

硬约束：

- 低频 base、data consistency projection 和 temporal backbone 默认冻结或低学习率微调。
- flow 只学习 wavelet 高频 residual，不直接预测 RGB 全频图。
- sampling temperature、residual amplitude 和介入区域由 `R/U_texture` 控制。
- Stage C 前期不启用 GAN，也不让 NIQE/MUSIQ/CLIPIQA 参与 checkpoint 选择。
- 若 crop strip 出现纹理漂移、融化、树枝/文字结构错乱，即使 no-reference 指标上升也判为失败。

损失：

```text
L = L_flow + L_recon + L_dc + L_temp + L_uncertainty + L_bridge
```

其中：

- `L_flow`：rectified flow vector field。
- `L_recon`：采样 residual 后的重建损失。
- `L_dc`：低频数据一致性。
- `L_temp`：时序一致性。
- `L_uncertainty`：calibration loss，例如 NLL / ECE-like loss。
- `L_bridge`：让 residual transport 与 OT/SB bridge 的中间状态保持一致。

### 6.4 Stage D：Consistency distillation

把 4-step flow teacher 蒸馏到 1-step student。

目标：

- 保留生成质量。
- 支持实时或近实时 streaming VSR。
- 形成 CCF-A 论文中的效率亮点。

### 6.5 Stage E：Online/offline 联合训练

同一模型支持两种模式：

- `TrajFlow-VSR-O`：offline high-quality，双向上下文，质量优先。
- `TrajFlow-VSR-S`：streaming low-latency，因果上下文，延迟优先。

训练时随机切换：

```text
mode ∈ {offline_bidirectional, online_causal}
```

## 7. 损失函数

建议基础总损失：

```text
L_total =
  λ_rec   L_char
+ λ_ot    L_optimal_transport
+ λ_sb    L_schrodinger_bridge
+ λ_flow  L_rectified_flow
+ λ_kop   L_koopman_dynamics
+ λ_dc    L_data_consistency
+ λ_temp  L_temporal
+ λ_freq  L_wavelet_frequency
+ λ_alias L_anti_aliasing
+ λ_traj  L_trajectory
+ λ_unc   L_uncertainty_calibration
```

关键损失说明：

- `L_optimal_transport`：约束跨帧 token 的 soft matching 分布，支持多假设对应。
- `L_schrodinger_bridge`：约束 degraded-to-clean 分布桥接，为后续 residual flow 提供条件路径。
- `L_rectified_flow`：学习 residual posterior transport。
- `L_koopman_dynamics`：让 latent observable 的长时序演化更稳定。
- `L_data_consistency`：限制低频和 LR 观测一致。
- `L_anti_aliasing`：根据 query cell footprint 抑制非物理高频。
- `L_trajectory`：用 HR/clean embedding 或 cycle consistency 监督 soft trajectory。
- `L_uncertainty_calibration`：让 uncertainty 与真实误差相关，而不是只做漂亮 mask。
- `L_temporal`：可用 RAFT/Farneback/tOF + feature warping + trajectory consistency。

感知/对抗损失只能作为后期可选项，并且只作用于高频 residual：

```text
L_late =
  λ_per_H L_perceptual(High(X_hat), High(X_gt))
+ λ_adv_H L_adversarial(High(X_hat))
```

启用条件：

- Stage B deterministic backbone 已经在 PSNR/SSIM、LPIPS/DISTS、tOF 和可视化上稳定。
- Stage C flow residual 没有低频漂移和时序融化。
- `λ_per_H`、`λ_adv_H` 必须小权重 warm-up，并受 `G_H` gate 限制。
- 禁止用 NIQE/MUSIQ/CLIPIQA 的单点改善覆盖 full-reference 或 crop strip 的结构性失败。

## 8. 数据、对比与指标

### 8.1 训练数据

基础：

- Vimeo90K。
- REDS。
- Vid4/UDM10 只用于评估。

增强：

- 真实 codec degradation：H.264/H.265/AV1 多码率。
- camera pipeline degradation：noise、rolling shutter、exposure flicker。
- motion degradation：camera shake、motion blur、non-rigid deformation。
- online streaming split：长序列裁切为 causal clips。

可选：

- VideoLQ / Real-world LR video，用于无配对自监督。
- YouTube/公开视频，构建真实压缩预训练集。

### 8.2 对比方法

实验比较按优先级维护，避免 benchmark 膨胀到不可控。

P0：主表或核心消融中必须覆盖。

- BasicVSR++：recurrent propagation / alignment fidelity 强基线。
- RealBasicVSR：真实退化、pre-cleaning、artifact propagation 强基线。
- RealViformer：真实退化 attention reliability 强基线。
- EDVR / TDAN：deformable alignment 与 feature-level alignment 对照，用于证明 OT/SB soft trajectory 的必要性。
- MambaVSR / TS-Mamba / VSRM：Mamba/SSM 长上下文直接对照，用于证明 trajectory Koopman-SSM 不是普通 scan 替换。
- SCST：self-supervised ControlNet + spatio-temporal Mamba + diffusion prior，真实世界 VSR 直接竞争方法。
- DiffVSR：robust diffusion VSR recipe，直接对照 residual rectified flow / staged training。
- VideoINR / ST-AVSR：若保留 arbitrary-scale claim，必须至少选择一个连续/任意尺度 VSR 对照。

P1：根据开源权重、算力和篇幅选择进入扩展表。

- BasicVSR / IconVSR：基础 recurrent VSR 对照。
- VRT / RVRT：Transformer long-context video restoration 对照。
- MGLD-VSR / StableVSR / Upscale-A-Video / STAR：diffusion perceptual quality 与 temporal consistency 对照。
- DiTVR：zero-shot diffusion transformer video restoration，对照 trajectory-aware diffusion prior。
- FlashVSR：若强调 one-step / streaming / real-time，必须进入效率表。
- SAVSR / LIIF：任意尺度/连续表示的补充对照。

P2：主要用于 Related Work 和理论动机，不一定跑实验。

- Mamba / Mamba-2 / S4：SSM 与 selective scan 的理论背景。
- Sinkhorn / unbalanced OT / Schrödinger Bridge：probabilistic transport 与 degraded-to-clean bridge 背景。
- Flow Matching / Rectified Flow / Consistency Models：residual posterior transport 与 one-step distillation 背景。
- DeepONet / FNO / WNO / Multiwavelet operator learning：spacetime neural operator 与 wavelet anti-aliasing 背景。
- `ST-VSR`：只能作为内部 baseline 或 prior work，不应作为新模型核心。

### 8.3 指标

保真度：

- PSNR(Y)。
- SSIM(Y)。

感知质量：

- LPIPS。
- DISTS。
- NIQE。
- MUSIQ。
- CLIPIQA。

视频质量：

- tOF / warping error。
- FVD 或 video perceptual score。
- VMAF，适合 codec 场景。

效率：

- FPS。
- latency。
- VRAM。
- MACs。
- one-step / two-step / four-step 质量-速度曲线。

不确定性：

- uncertainty-error correlation。
- reliability ECE。
- selective reconstruction curve：只在高可靠区域计算误差，看校准是否有效。

### 8.4 模型选择与可视化验收

checkpoint 选择必须采用 multi-objective protocol：

- 第一层：PSNR/SSIM、LPIPS/DISTS、tOF/warping error 和低频 data consistency，确保结构与时序不崩。
- 第二层：Vid4、UDM10、REDS4、codec LR 协议下的 crop strip，重点检查树枝、文字、栅栏、重复纹理、脸部边缘和快速运动区域。
- 第三层：NIQE/MUSIQ/CLIPIQA 只作为补充感知信号，不能单独决定最佳 epoch。
- 第四层：uncertainty/reliability calibration，检查高不确定区域是否真的对应高误差或遮挡，而不是学成漂亮 mask。

一票否决项：

- 静帧看似锐利但 crop strip 出现纹理漂移、闪烁、融化。
- 树枝、文字、边缘线条被生成式纹理替换。
- no-reference 指标改善但 PSNR/SSIM、LPIPS/DISTS、tOF 或人工结构判断明显变差。
- reliability 较高区域仍被 flow/GAN/diffusion residual 大幅改写。

## 9. 实验与消融

### 9.1 必要消融

- w/o rectified flow，改 deterministic residual regression。
- w/o OT/SB trajectory bridge，改 optical flow / deformable alignment / attention alignment。
- w/o unbalanced OT，改 balanced OT。
- w/o Koopman dynamics regularization。
- w/o reliability calibration。
- w/o uncertainty loss。
- w/o wavelet anti-aliasing。
- w/o neural operator，改普通 INR / scale-aware MLP。
- w/o data consistency projection。
- w/o consistency distillation。
- one-step vs two-step vs four-step flow。
- offline vs online。
- long context length：3 / 7 / 15 / 31 / 63 frames。

### 9.2 关键证明目标

- OT/SB soft trajectory 比光流、deformable alignment 和普通 attention 更适合真实退化 VSR。
- Koopman-SSM 比普通 Mamba scan 更适合长时序动态建模。
- rectified flow residual 比 deterministic residual 更真实。
- data consistency 能压住 hallucination。
- uncertainty/reliability 不是装饰，而能提升真实退化和时序稳定性。
- wavelet neural operator 带来跨尺度泛化和频率一致性。

### 9.3 第一阶段最小验证

建议第一步不要直接写大模型，而是先做一个小而硬的验证：

**验证题：OT/SB soft trajectory + Koopman-SSM 是否比 raster/content-aware Mamba 更适合 VSR 长时序聚合？**

最小实验：

- 输入 7/15 帧。
- 不上 rectified flow。
- 只比较 raster scan、Hilbert scan、content-aware scan、soft trajectory scan、OT/SB soft trajectory scan。
- 加一组 w/o Koopman regularization。
- 指标：PSNR、SSIM、tOF、runtime、VRAM。

如果这个实验成立，再进入 rectified flow 生成分支。这样项目风险最低，也最容易形成第一张 CCF-A 论文的核心图。

### 9.4 最终预期性能目标

性能目标不应写成“所有指标全面碾压”，而应写成分层达标。TrajFlow-VSR 的最终目标是：deterministic 主干在保真度和时序稳定性上达到或超过真实退化 VSR 强基线，受控 high-frequency residual 分支在不破坏结构的前提下提升 perceptual quality，并在长上下文、codec 退化、任意尺度和 uncertainty calibration 上形成主要优势。

与主要参考方法的预期相对位置：

| 对比对象 | 预期结果 | 关键判断 |
| --- | --- | --- |
| BasicVSR++ / EDVR / TDAN | 在 synthetic degradation 上 PSNR/SSIM 至少持平或小幅领先；在 realistic / codec degradation 上时序稳定性和 artifact robustness 明显更好 | 证明 OT/SB soft trajectory 比固定光流、deformable alignment 或局部 attention 更稳 |
| RealBasicVSR | 在真实退化和 codec LR 上达到同等或更高 PSNR/SSIM，同时降低 tOF/warping error，减少 artifact propagation | 证明 reliability-calibrated transport 能替代单纯 pre-cleaning / recurrent propagation |
| RealViformer | 在 artifact-heavy 区域至少持平，最好在 LPIPS/DISTS、tOF 和可视化结构稳定性上领先 | 证明 reliability 不只是 attention trick，而是 transport、memory、data consistency 的统一控制变量 |
| MambaVSR / TS-Mamba / VSRM | 长上下文 15/31/63 帧下 tOF、temporal crop strip 和 runtime/memory 曲线更优，PSNR/SSIM 不低于普通 scan Mamba | 证明 trajectory-conditioned Koopman-SSM 优于 raster/content scan |
| SCST / DiffVSR / StableVSR / MGLD-VSR | 感知指标接近或部分领先 diffusion VSR，但时序闪烁更少，推理步数、延迟和显存明显更低 | 证明 residual rectified flow 比全图 diffusion 更适合可控 VSR |
| VideoINR / SAVSR / ST-AVSR | 在 x2/x3/x4/x6 和非整数尺度上跨尺度一致性更好，频率 aliasing 更少 | 证明 wavelet neural operator 比普通 INR / scale-aware MLP 更适合任意尺度视频函数重建 |

建议论文内部验收阈值：

- **Stage B deterministic**：相对 RealBasicVSR / RealViformer，PSNR/SSIM 至少接近，最好提升 `+0.1~0.3 dB`；tOF 或 temporal delta error 下降 `5%~15%`；LPIPS/DISTS 不劣化；crop strip 无结构融化。
- **Stage C rectified flow**：相对 Stage B，PSNR 下降不超过 `0.05~0.10 dB`，LPIPS/DISTS 改善 `3%~8%`，NIQE/MUSIQ/CLIPIQA 可改善但不能作为主选择标准。
- **Stage D one-step distillation**：相对 2-step/4-step teacher，感知质量保持 `90%+`，延迟明显下降，streaming 版本保持低 tOF 和低 causal violation。
- **codec / real-world 协议**：H.264/H.265 LR 下不出现 block artifact 放大、树枝/文字/栅栏结构伪造；VMAF、tOF、LPIPS/DISTS 和可视化同时支撑结论。
- **uncertainty calibration**：reliability-error correlation 为正，reliability ECE 低于无校准版本，selective reconstruction curve 能证明高可靠区域确实更准。

最终论文主张应是：

> TrajFlow-VSR is not merely sharper than prior VSR models; it is more reliable under real degradations, more temporally stable over long trajectories, more controllable when using generative residuals, and more consistent across arbitrary scales.

## 10. 预期论文贡献

面向 CCF-A 的贡献可以组织为：

1. **Problem reformulation**：首次将 real-world arbitrary-scale VSR 统一表述为 reliability-calibrated conditional transport over continuous video trajectories。
2. **Transport alignment**：提出 OT/SB soft trajectory bridge，把跨帧对应、遮挡和 degraded-to-clean 分布桥接统一到概率传输框架中。
3. **Dynamic memory**：提出 trajectory-conditioned selective state space / Koopman memory，沿 soft physical trajectories 聚合长时序证据并约束动态演化。
4. **Generative restoration**：提出 residual rectified flow，把 HR 高频恢复建模为可控 posterior generation，并通过 consistency distillation 实现 one-step/streaming。
5. **Operator decoding**：提出 spacetime neural operator + wavelet anti-aliasing decoder，实现跨尺度、跨时间、低 aliasing 的连续视频函数重建。
6. **Evaluation**：构建 long-context、codec-aware、multi-scale、offline/online 双协议，报告质量、时序、效率和 uncertainty calibration。

## 11. 工程目录建议

```text
TrajFlow_VSR/
  README.md
  pyproject.toml
  configs/
    train_stage_a_tokenizer.yaml
    train_stage_b_deterministic.yaml
    train_stage_c_rectified_flow.yaml
    train_stage_d_distill.yaml
    eval_offline.yaml
    eval_streaming.yaml
  data/
    degradation/
    datasets/
  models/
    tokenizer.py
    degradation_uncertainty_encoder.py
    ot_sb_trajectory_bridge.py
    trajectory_koopman_ssm.py
    rectified_flow.py
    spacetime_wavelet_operator_decoder.py
    trajflow_vsr.py
  losses/
    optimal_transport.py
    schrodinger_bridge.py
    flow_matching.py
    koopman_dynamics.py
    data_consistency.py
    wavelet_anti_aliasing.py
    temporal.py
    uncertainty.py
    trajectory.py
  scripts/
    train.py
    eval_protocol.py
    infer_video.py
    visualize_trajectory.py
    visualize_uncertainty.py
  docs/
    paper_outline.md
    experiment_plan.md
    related_work.md
  outputs/
  checkpoints/
```

## 12. 实施路线

### Phase 0：立项与复现基线，1-2 周

- 复现/整理 BasicVSR++、RealBasicVSR、RealViformer、一个 Mamba VSR、一个 diffusion VSR。
- 建立统一评估协议。
- 确认数据集路径、codec degradation pipeline、长序列 loader。

### Phase 1：OT/SB Trajectory + Deterministic Koopman-SSM VSR，3-5 周

- 实现 tokenizer。
- 实现 OT/SB soft trajectory bridge。
- 实现 trajectory selective SSM + Koopman dynamics regularization。
- 先做 deterministic residual decoder。
- 目标：超过或接近 RealViformer/BasicVSR++ 的保真度，同时长序列更稳。

### Phase 2：Spacetime Wavelet Neural Operator Decoder，2-3 周

- 替换普通 decoder。
- 加入 wavelet frequency decomposition 与 anti-aliasing head。
- 支持 `x2/x3/x4/x6` 和非整数 scale。
- 做 scale generalization 实验。

### Phase 3：Rectified Flow Residual Generator，4-6 周

- wavelet residual latent。
- conditional rectified flow。
- 2-step/4-step ODE sampler。
- 质量优先版本。

### Phase 4：Consistency Distillation，2-4 周

- 从 4-step teacher 蒸馏到 1-step student。
- 做质量-速度曲线。
- 推 online streaming 版本。

### Phase 5：大规模实验与论文，4-8 周

- 完整 benchmark。
- 消融。
- 可视化。
- uncertainty calibration。
- 写论文。

总周期：

- 最快 3 个月出 workshop/初稿。
- 5-6 个月冲 CCF-A 主会更稳。

## 13. 风险与版本控制

### 13.1 主要风险

风险 1：方案太大，训练不稳定。

规避：

- 先完成 OT/SB trajectory + deterministic Koopman-SSM。
- 再加 wavelet neural operator。
- 最后加 rectified flow。
- 每阶段都有可发的小成果，但最终论文统一成大故事。

风险 2：rectified flow 带来时序闪烁。

规避：

- 在 residual latent 而非 RGB 全图上建模。
- 条件中加入 trajectory memory。
- 使用 video-level noise，共享时间相关随机变量。
- 加 temporal flow consistency 和 low-frequency projection。

风险 3：OT/SB trajectory field 退化成 noisy optical flow。

规避：

- 不输出单一 flow，而是 reliability-calibrated soft transport graph。
- 加 unmatched/occlusion token。
- reliability 控制匹配温度，unbalanced OT 允许质量缺失。
- 用 cycle consistency、HR feature contrastive loss 和 bridge consistency loss。

风险 4：创新点太分散。

规避：

论文主线只讲一个核心：

> reliability-calibrated conditional transport over trajectories.

其他模块都服务于这条线：

- OT/SB trajectory bridge = 可靠证据传输与 degraded-to-clean 桥接。
- Koopman-SSM = 条件证据聚合与长时序动态。
- rectified flow = posterior transport。
- wavelet neural operator = continuous spacetime output 与频率一致性。
- reliability = transport/data consistency 的校准。

### 13.2 最小可发版本

如果算力或时间不够，保留以下核心：

- OT/SB soft trajectory bridge。
- Trajectory Koopman-SSM。
- Conditional rectified flow residual。
- Reliability-calibrated data consistency。

暂时砍掉：

- online streaming。
- x6/非整数 scale。
- 完整 wavelet neural operator，只保留 scale-aware INR + anti-aliasing head。

最小论文题目：

**Reliability-Calibrated Trajectory Transport for Real-World Video Super-Resolution**

### 13.3 CCF-A 强化版本

如果资源充足，加入以下增强：

- 31-63 帧长上下文。
- online/offline 双模式。
- one-step consistency distilled model。
- uncertainty calibration benchmark。
- 真实 codec 视频测试集。
- 公开视频 demo。
- 部分代码和模型权重开源。

强化论文题目：

**TrajFlow-VSR: Reliability-Calibrated Conditional Transport over Spacetime Trajectories for Arbitrary-Scale Video Super-Resolution**

## 14. CCF-A 投稿验收清单

### 14.1 创新点要求

必须做到：

- 问题定义有提升，不能只说“我们提出一个更好的 VSR 网络”。
- 核心创新不是堆模块，至少有一个能单独成立的机制。
- 理论语言足够新，能解释为什么本方法比普通 attention、普通 Mamba scan、普通 diffusion、普通 INR 更适合 VSR。
- 与已有 ST-VSR/IPEC 工作明显区分。
- 贡献之间有一条主线：`reliability-calibrated conditional transport over spacetime trajectories`。

容易被拒的情况：

- 只比现有方法高 0.1 dB，没有解释为什么。
- 方法看起来像 BasicVSR++/RealViformer/MambaVSR/DiffVSR 的简单组合。
- 创新点太多但每个都证明不深。
- 生成式分支带来视觉纹理，但时序闪烁、低频漂移或 hallucination 控制不足。

### 14.2 工作量要求

最低主会级别工作量：

- 完整实现一个可训练、可复现的新框架，而不是只替换 backbone。
- 至少覆盖 `x2/x3/x4`，最好支持 `x6` 或非整数尺度作为额外亮点。
- 至少完成 synthetic degradation、realistic degradation、codec degradation 三类评估。
- 同时报告保真度、感知质量、时序稳定性、效率和显存。
- 做完整 ablation，证明每个核心模块都有必要。
- 做失败案例分析，说明方法边界。

工程产物要求：

- 统一配置系统，能复现实验。
- 固定数据划分和退化参数。
- 保存可视化脚本，包括 trajectory、uncertainty、artifact/reliability map、posterior samples。
- 为每个 baseline 保留运行命令、commit/version、权重来源和指标记录。

### 14.3 Related Work 组织方式

论文 related work 不能只按年份罗列，应按问题链条组织：

- **Blind / real-world VSR**：真实退化、artifact propagation、attention reliability。
- **Probabilistic transport and alignment**：optimal transport、unbalanced OT、Schrödinger Bridge、soft matching。
- **Long-context video dynamics**：recurrent VSR、transformer VSR、Mamba/SSM VSR、Koopman dynamics。
- **Generative restoration**：diffusion VSR、DiT video restoration、rectified flow / flow matching、posterior sampling。
- **Arbitrary-scale / signal representation**：LIIF/INR 类方法、continuous video representation、neural operator、wavelet / anti-aliasing。
- **Uncertainty and reliability**：calibration、selective prediction、data consistency、hallucination control。

每一组文献都要回答：

- 这个方向解决了什么。
- 它在真实世界 VSR 中还缺什么。
- TrajFlow-VSR 如何针对这个缺口。

### 14.4 投稿前自检

投稿前逐项确认：

- 标题是否一眼看出核心创新，而不是泛泛的 “A Novel Framework”。
- 摘要是否说明问题、方法、理论动机、关键结果和效率。
- Introduction 是否清楚指出现有方法的共同瓶颈。
- Method 是否有足够数学定义，而不是只有模块图。
- Method 是否把 OT/SB、Koopman-SSM、rectified flow、wavelet neural operator 的职责分清。
- Experiments 是否覆盖强 baseline、完整指标、消融和可视化。
- Related Work 是否主动对比 RealViformer、SCST、DiffVSR、MambaVSR/TS-Mamba/VSRM、rectified flow、neural operator。
- Limitations 是否诚实说明算力、推理步数、真实视频泛化或 hallucination 风险。
- 代码和配置是否至少内部可复现。

## 15. 投稿方向

首选：

- CVPR。
- ICCV。
- ECCV。

备选：

- NeurIPS / ICLR，若理论和 generative flow 部分足够强。
- TPAMI / IJCV，若实验和分析规模足够大。

更适合 CVPR/ICCV 的版本：

- 强调视觉结果、benchmark、速度、真实退化。

更适合 NeurIPS/ICLR 的版本：

- 强调 conditional transport、trajectory posterior、uncertainty calibration、operator learning。

## 16. 参考文献与比较矩阵

本节用于维护投稿前必须对齐的参考文献。P0 文献需要尽量进入实验主表或核心消融；P1 文献根据开源权重、算力和篇幅进入扩展表；P2 文献主要支撑 Related Work 和方法动机。

### 16.1 P0：必须实验比较

| 文献 / 方法 | Venue | 为什么必须比较 | 实验角色 | 链接 |
| --- | --- | --- | --- | --- |
| BasicVSR: The Search for Essential Components in Video Super-Resolution and Beyond | CVPR 2021 | VSR propagation / alignment / aggregation / upsampling 的基础拆解 | recurrent VSR 基础对照，通常可被 BasicVSR++ 替代主表 | `https://openaccess.thecvf.com/content/CVPR2021/html/Chan_BasicVSR_The_Search_for_Essential_Components_in_Video_Super-Resolution_and_CVPR_2021_paper.html` |
| BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and Alignment | CVPR 2022 | 强 recurrent fidelity baseline，长期传播与 flow-guided deformable alignment 代表 | 主表 fidelity / runtime 对照 | `https://openaccess.thecvf.com/content/CVPR2022/html/Chan_BasicVSR_Improving_Video_Super-Resolution_With_Enhanced_Propagation_and_Alignment_CVPR_2022_paper.html` |
| EDVR: Video Restoration with Enhanced Deformable Convolutional Networks | CVPRW 2019 | PCD deformable alignment + TSA fusion 是经典 alignment baseline | alignment 消融对照，证明 OT/SB soft trajectory 优势 | `https://openaccess.thecvf.com/content_CVPRW_2019/papers/NTIRE/Wang_EDVR_Video_Restoration_With_Enhanced_Deformable_Convolutional_Networks_CVPRW_2019_paper.pdf` |
| TDAN: Temporally-Deformable Alignment Network for Video Super-Resolution | CVPR 2020 | feature-level deformable alignment 代表，避免显式 optical flow | alignment 消融对照 | `https://openaccess.thecvf.com/content_CVPR_2020/html/Tian_TDAN_Temporally-Deformable_Alignment_Network_for_Video_Super-Resolution_CVPR_2020_paper.html` |
| Investigating Tradeoffs in Real-World Video Super-Resolution / RealBasicVSR | CVPR 2022 | 真实退化 VSR 的 pre-cleaning、artifact suppression、batch-length tradeoff 代表 | real-world VSR 主表对照，VideoLQ 相关 | `https://arxiv.org/abs/2111.12704` |
| RealViformer: Investigating Attention for Real-World Video Super-Resolution | ECCV 2024 | 真实退化下 attention reliability 与 artifact propagation 直接相关 | reliability / artifact robustness 对照 | `https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/4277_ECCV_2024_paper.php` |
| MambaVSR: Content-Aware Scanning State Space Model for Video Super-Resolution | arXiv 2025 | content-aware scan 的 Mamba VSR 代表 | 与 trajectory-conditioned SSM/Koopman memory 对照 | `https://arxiv.org/abs/2506.11768` |
| TS-Mamba: Trajectory-aware Shifted State Space Models for Online Video Super-Resolution | arXiv 2025 | trajectory-aware online Mamba VSR，和本方案 long-context/online claim 最接近 | online / streaming / trajectory scan 对照 | `https://arxiv.org/abs/2508.10453` |
| VSRM: A Robust Mamba-Based Framework for Video Super-Resolution | ICCV 2025 | robust Mamba VSR，含 deformable cross-Mamba alignment 与 frequency loss | Mamba/SSM 主表强对照 | `https://openaccess.thecvf.com/content/ICCV2025/html/Tran_VSRM_A_Robust_Mamba-Based_Framework_for_Video_Super-Resolution_ICCV_2025_paper.html` |
| SCST: Self-supervised ControlNet with Spatio-Temporal Mamba for Real-world Video Super-resolution | CVPR 2025 | self-supervised ControlNet + spatio-temporal Mamba + diffusion prior | real-world perceptual quality / Mamba / diffusion 联合对照 | `https://openaccess.thecvf.com/content/CVPR2025/html/Shi_Self-supervised_ControlNet_with_Spatio-Temporal_Mamba_for_Real-world_Video_Super-resolution_CVPR_2025_paper.html` |
| DiffVSR: Revealing an Effective Recipe for Taming Robust Video Super-Resolution Against Complex Degradations | ICCV 2025 | robust diffusion VSR 的 staged recipe，直接挑战本方案 residual rectified flow | robust degradation / diffusion quality 对照 | `https://openaccess.thecvf.com/content/ICCV2025/html/Li_DiffVSR_Revealing_an_Effective_Recipe_for_Taming_Robust_Video_Super-Resolution_ICCV_2025_paper.html` |
| VideoINR: Learning Video Implicit Neural Representation for Continuous Space-Time Super-Resolution | CVPR 2022 | 连续时空隐式表示，任意空间/时间尺度输出 | arbitrary-scale / continuous video function 对照 | `https://arxiv.org/abs/2206.04647` |
| SAVSR: Arbitrary-Scale Video Super-Resolution via a Learned Scale-Adaptive Network | AAAI 2024 | 明确针对 spatial arbitrary-scale VSR，包括非整数和非对称尺度 | arbitrary-scale 主表或补充表对照 | `https://ojs.aaai.org/index.php/AAAI/article/view/28114` |
| ST-AVSR: Arbitrary-Scale Video Super-Resolution with Structural and Textural Priors | arXiv 2024 | 任意尺度 VSR 的结构/纹理先验强基线 | arbitrary-scale / cross-scale generalization 对照 | `https://arxiv.org/abs/2407.09919` |

### 16.2 P1：扩展实验或补充表

| 文献 / 方法 | Venue | 比较价值 | 实验角色 | 链接 |
| --- | --- | --- | --- | --- |
| VRT: A Video Restoration Transformer | arXiv 2022 | transformer long-range temporal dependency 代表 | long-context transformer 对照 | `https://arxiv.org/abs/2201.12288` |
| RVRT: Recurrent Video Restoration Transformer with Guided Deformable Attention | NeurIPS 2022 / arXiv | recurrent + transformer + deformable attention 的强 restoration baseline | 长上下文质量/效率补充对照 | `https://arxiv.org/abs/2206.02146` |
| MGLD-VSR: Motion-Guided Latent Diffusion for Temporally Consistent Real-world Video Super-resolution | ECCV 2024 / arXiv | motion-guided latent diffusion，强调真实世界 perceptual quality 与 temporal consistency | diffusion perceptual quality 对照 | `https://arxiv.org/abs/2312.00853` |
| StableVSR: Enhancing Perceptual Quality in Video Super-Resolution through Temporally-Consistent Detail Synthesis using Diffusion Models | ECCV 2024 / arXiv | diffusion detail synthesis 与 temporal consistency | perceptual/temporal consistency 对照 | `https://arxiv.org/abs/2311.15908` |
| Upscale-A-Video: Temporal-Consistent Diffusion Model for Real-World Video Super-Resolution | CVPR 2024 | text-guided temporal-consistent diffusion VSR | real-world diffusion visual comparison | `https://github.com/sczhou/Upscale-A-Video` |
| STAR: Spatial-Temporal Augmentation with Text-to-Video Models for Real-World Video Super-Resolution | arXiv 2025 | text-to-video prior 用于 real-world VSR | generative prior / high-frequency detail 对照 | `https://arxiv.org/abs/2501.02976` |
| DiTVR: Zero-Shot Diffusion Transformer for Video Restoration | arXiv 2025 | diffusion transformer + trajectory-aware attention + wavelet-guided consistency | zero-shot / trajectory-aware generative restoration 对照 | `https://arxiv.org/abs/2508.07811` |
| FlashVSR: Towards Real-Time Diffusion-Based Streaming Video Super-Resolution | arXiv 2025 | one-step / streaming / real-time diffusion VSR | streaming efficiency / latency 对照 | `https://arxiv.org/abs/2510.12747` |
| LIIF: Learning Continuous Image Representation with Local Implicit Image Function | CVPR 2021 | continuous image representation 与 arbitrary-scale SR 基础方法 | Related Work + image-level arbitrary-scale baseline | `https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Learning_Continuous_Image_Representation_With_Local_Implicit_Image_Function_CVPR_2021_paper.pdf` |

### 16.3 P2：理论和方法动机

| 文献 | 主题 | 用在论文哪里 | 链接 |
| --- | --- | --- | --- |
| Mamba: Linear-Time Sequence Modeling with Selective State Spaces | selective SSM / long sequence | Long-context video dynamics 背景 | `https://arxiv.org/abs/2312.00752` |
| Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality | Mamba-2 / SSD | 解释 SSM 与 attention 的关系 | `https://arxiv.org/abs/2405.21060` |
| Efficiently Modeling Long Sequences with Structured State Spaces | S4 | SSM 历史背景 | `https://arxiv.org/abs/2111.00396` |
| Leveraging Neural Koopman Operators to Learn Continuous Representations of Dynamical Systems from Scarce Data | Koopman latent dynamics | Koopman memory 和 dynamics regularization 动机 | `https://arxiv.org/abs/2303.06972` |
| Sinkhorn Distances: Lightspeed Computation of Optimal Transportation Distances | entropic OT / Sinkhorn | OT/SB soft trajectory bridge | `https://arxiv.org/abs/1306.0895` |
| Unbalanced Optimal Transport: Dynamic and Kantorovich Formulations | unbalanced OT | occlusion / artifact 下 mass mismatch | `https://angkor.univ-mlv.fr/~vialard/publication/dynamic-to-static/` |
| I2SB: Image-to-Image Schrödinger Bridge | image-to-image SB restoration | degraded-to-clean bridge 背景 | `https://arxiv.org/abs/2302.05872` |
| Implicit Image-to-Image Schrödinger Bridge for Image Restoration | restoration SB extension | SB restoration related work | `https://arxiv.org/abs/2403.06069` |
| Flow Matching for Generative Modeling | continuous flow / vector field regression | residual rectified flow training | `https://arxiv.org/abs/2210.02747` |
| Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow | rectified flow | one/few-step residual transport | `https://arxiv.org/abs/2209.03003` |
| Scaling Rectified Flow Transformers for High-Resolution Image Synthesis | high-resolution rectified flow | high-res generative restoration motivation | `https://arxiv.org/abs/2403.03206` |
| Consistency Models | one-step / few-step distillation | Stage D consistency distillation | `https://arxiv.org/abs/2303.01469` |
| DeepONet: Learning nonlinear operators for identifying differential equations based on the universal approximation theorem of operators | neural operator | spacetime operator decoder 背景 | `https://arxiv.org/abs/1910.03193` |
| Fourier Neural Operator for Parametric Partial Differential Equations | FNO | operator learning 背景 | `https://arxiv.org/abs/2010.08895` |
| Wavelet Neural Operator: a neural operator for parametric partial differential equations | WNO | wavelet neural operator 背景 | `https://arxiv.org/abs/2205.02191` |
| Multiwavelet-based Operator Learning for Differential Equations | multiwavelet operator | multi-scale frequency/operator 背景 | `https://arxiv.org/abs/2109.13459` |
| Making Convolutional Networks Shift-Invariant Again | anti-aliasing / blurpool | anti-aliasing decoder 背景 | `https://arxiv.org/abs/1904.11486` |
| BACON: Band-limited Coordinate Networks for Multiscale Scene Representation | band-limited coordinate network | coordinate representation / aliasing 对照 | `https://arxiv.org/abs/2112.04645` |

### 16.4 维护要求

- P0/P1 中进入实验的每个 baseline，必须在 `configs/baselines/core_vsr_baselines.yaml` 记录 repo URL、commit、权重来源、本地权重路径、inference command、evaluation command 和 metrics path。
- 若无法运行某篇方法，需要在论文补充材料中说明原因，例如权重未公开、license 限制、依赖不可复现、算力不可承受。
- 主表至少覆盖 fidelity、perceptual quality、temporal consistency、runtime/FPS、VRAM 或 peak memory。
- Real-world/no-reference 数据集上不能报告伪造 PSNR/SSIM，应使用 NIQE、MUSIQ、CLIPIQA、DOVER 等无参考指标和人工偏好/可视化。
- CVPR Reviewer Guidelines: `https://cvpr.thecvf.com/Conferences/2026/ReviewerGuidelines`
- NeurIPS Reviewer Guidelines: `https://neurips.cc/Conferences/2025/ReviewerGuidelines`
