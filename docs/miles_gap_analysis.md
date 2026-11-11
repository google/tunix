# Tunix 对齐 Miles 的能力差距与下一步工作

更新日期：2026 年 10 月 29 日。比较对象是下面固定的源码版本，结论用于安排 Tunix 下一阶段研发。

`haoyu-next` 现已加入 sampled-token OPD 的教师 scorer、critique-stage advantage 变换、GSM8K 运行入口与 CPU 正确性测试，见[实现和运行说明](../tunix/dist_rl/examples/on_policy_distillation/README.md)。下表和能力比较仍针对固定的 main 提交；真实 TPU 训练与学习质量验证尚待执行。

| 仓库 | 比较版本 | 本地位置 |
| --- | --- | --- |
| Tunix | `origin/main`，`2176443ad6bb61093e8d133a0d2d19045655db80` | 当前仓库；`haoyu-next` 从这个提交创建 |
| Miles | `main`，`88cf4fc932fca650aebc409ccbb5bc9dd7f59afe` | `.references/miles/` |
| Miles Diffusion | `main`，`221f965fa68e4d865665a48e8478932a231dc4ad` | `.references/miles_diffusion/` |

两个参考仓库是独立的浅克隆，保留各自 Git 元数据；`.references/` 被当前仓库忽略。本文的源码链接固定到上述提交。本次没有运行 GPU 或 TPU 训练，源码和 recipe 的存在不代表已复现其性能或收敛结果。

## 1 最值得追的方向

**建议先补 OPD，再打通图像理解 RL；同时把现有分布式能力整理成有明确支持范围和验收指标的 recipe。** Miles 的优势既有新增训练板块，也有围绕生产训练形成的数据契约、后端适配和验证体系。不能仅按 README 功能名称计数。

优先关注五个增量：

1. **On-policy distillation（OPD）闭环**：学生采样、教师逐 token 评分、逐 token 蒸馏信号与 RL advantage 组合。Tunix 已有数据批次蒸馏，但缺少这条标准工作流。
2. **可复现的多模态 RL**：Miles 有 Qwen3-VL 图像理解的单轮、多轮 GRPO，以及训练后端和 E2E recipe。Tunix 已有 Gemma 视觉／音频、图像 DPO 和媒体采样，主要缺口在 RL 数据与训练链路。
3. **服务化训练和多策略训练**：multi-LoRA/Tinker gateway 与 solver/verifier multi-policy 是不同于单个训练任务的新板块。
4. **采样分布一致性和新的 off-policy 目标**：sampling-support replay、score centering 值得单独研究。Tunix 已有 importance sampling、拒绝采样和 Zero-TIM 内核，新增部分应精确落到数学语义和数据字段。
5. **连续图像／视频 diffusion RL**：属于 Miles Diffusion 独立仓库，和 Tunix 现有离散 token diffusion 合约不是同一类能力，应单独立项。

还有一个值得持续跟踪的变化：Miles 当前不仅有 Megatron 和 FSDP2，还加入了 torchtitan，以及 `torch_tpu` 平台层和 TPU Docker 构建路径。不能再简单把比较写成“GPU 框架对 TPU 框架”；但各平台也不能直接继承所有后端的功能支持。[Miles 平台实现][m-platform]、[TPU 镜像][m-tpu]

## 2 能力总表

这里的“新增”指当前 Tunix 没有对应的标准训练流程；“补齐闭环”指基础组件已存在；“深化”指两边都有该能力，需要比较覆盖范围和质量。

| 板块 | Miles 的具体增量 | Tunix 当前基础 | 判断与优先级 |
| --- | --- | --- | --- |
| OPD | 外部／本地 teacher、sampled-token 和 top-k 信号、按任务路由 teacher、纯蒸馏与 RL 组合 | `DistillationTrainer`、logit/feature/attention 蒸馏、reference KL | **新增工作流，P0** |
| 图像理解 RL | GEO3K 单轮／多轮 GRPO、Qwen3-VL、FSDP/Megatron 数据链路和测试 | Gemma3/4 vision、image processor、VLM DPO、待维护的图像 SFT 脚本、媒体采样 | **补齐 RL 闭环，P0** |
| 音频理解 RL | Inkling audio processor/tower 与媒体训练输入 | Gemma4 audio、audio processor、原生 sampler 输入 | **补齐训练契约及 recipe，P1** |
| 视频理解 RL | processor、Qwen3-VL grid/MRoPE 基础路径 | 尚无标准视频 RL recipe | **后续扩展，P2；不把基础路径算完整支持** |
| 连续 diffusion RL | 独立仓库中的 Flow-GRPO、DiffusionNFT、媒体 rollout 与 reward | 离散 token diffusion batch/scorer 与 weighted CE SFT | **新模型与目标体系，单独评估** |
| Fully async | 更丰富的持续采样、buffer、未消费样本处理和异步评估机制 | `dist_rl` 异步流水、`max_staleness`、dispatch/backpressure | **深化，P1；不是从零新增** |
| TITO | session 代理将 token 连续性带到黑盒 agent harness | `exact_token_continuity`、token IDs 和 logprobs 数据链路 | **扩展外部 harness，P1** |
| MoE routing replay | R3 跨 router/session/backend 传递与 replay | rollout 路由采集、压缩、packing、MaxText replay 及测试 | **深化，P1** |
| 数值一致性 | true-on-policy contract、模型 profile、配套启动与测试 | Zero-TIM Pallas kernels、Qwen3 canonical model、agreement/IS/RS | **从内核走向标准 recipe，P1** |
| Sampling-support replay | 保存 top-k/top-p 后真实 support，训练时在同一 support 归一化 | 当前 batch 未提供对应 support 契约 | **新增正确性机制，P1** |
| Score centering | 利用 rollout 候选概率修正 off-policy score drift | 现有 GRPO/PPO、IS/RS，没有同类目标实现 | **新增算法，P1 研究项** |
| 权重同步 | 多 transport、session handshake、部分接收方故障隔离 | Raiden、版本管理、reshard、幂等与失败协调 | **深化，按实测瓶颈排期** |
| 动态故障恢复 | 重建 cell、恢复当前权重、重新加入 router | worker eviction/retry 和 checkpoint resume，动态 rejoin 尚未完成 | **补齐恢复闭环，P1** |
| 低精度 RL | FP8/MXFP8/NVFP4、INT4 QAT 与 rollout/导出适配 | qwix LoRA/QLoRA、JAX mixed precision | **新增部分闭环；GPU 专用优化不直接移植** |
| Multi-policy | 多个独立 policy 的 trainer、engine、checkpoint 和相互奖励 | agent/env、多模型角色，但无对应通用协同训练入口 | **新增编排板块，P2** |
| Multi-LoRA 服务 | Tinker API、多 adapter slot、租户队列、独立优化器和版本化采样 | 单任务 LoRA/QLoRA | **新增服务板块，P2** |
| Agent 生态 | Harbor、NeMo Gym、OpenEnv、Verifiers 等连接器及 sandbox 抽象 | Agentic RL、DeepSWE、FrozenLake、自定义环境 | **扩展连接器，P1** |
| 评测与运维 | dashboard、训练审计、按 recipe/hardware 分层 CI、历史指标门禁 | perf/Perfetto、指标、CPU/TPU 测试及 nightly regression | **深化产品化验证，P1** |

基础算法也不能简单按数量比：Tunix 已有 PPO、GRPO、DAPO、Dr.GRPO、GSPO-Token、RLOO/GRPO-LOO，以及 DPO/ORPO。Miles 的 GSPO 序列级权重、REINFORCE++、score centering 应逐个核对估计器、归一化和 loss reduction；同名、近似名不代表实现等价，也不代表 Tunix 必须照搬所有目标。[Tunix 算法配置][t-algo-config]、[Miles objective][m-objective]

## 3 OPD 是最适合先落地的新训练板块

### 和 Tunix 现有蒸馏的区别

Tunix 的 `DistillationTrainer` 基于外部提供的数据批次运行 teacher forward，再通过 strategy 训练 student。`LogitStrategy` 已支持 teacher top-k，并计算 `KL(teacher || student)`，还可与标签 CE 混合；feature、attention 蒸馏也已经存在。缺少的是学生自己生成轨迹后的在线教师评分，以及这条信号进入 RL 训练的标准接口。[Tunix 蒸馏 trainer][t-distill]、[LogitStrategy][t-logit]

Miles OPD 的路径是：

```text
prompt → student rollout，保留 token IDs 和旧 student logprobs
       → teacher 对同一条 student trajectory 逐 token 评分
       → teacher_log_probs 或预计算的 opd_reverse_kl
       → 基础 GRPO/PPO 等 advantage
       → 加入 OPD token 信号 → 更新 student → 同步 rollout 权重
```

对 sampled-token 路径，可以用下面的式子理解：

```text
d_t      = log π_old_student(a_t | s_t) − log π_teacher(a_t | s_t)
A_opd,t  = A_task,t − λ_opd × d_t
```

这里 `d_t` 是学生采样 token 上的 reverse-KL 样本信号，单个值可以为负，不是每个位置上的完整非负 KL。Miles 根据 `use_rollout_logprobs` 选择采样时记录的 student scores，或训练更新前重算的 scores；异步时两者的含义尤其需要区分。旧 student scores、teacher scores、最终 advantage 都作为固定训练数据，不对 teacher 求梯度。[OPD loss][m-opd-loss]、[advantage 组合顺序][m-objective]

**reference KL 与 OPD 要保留独立语义和系数。** 前者限制相对 reference policy 的漂移，后者从 teacher 迁移能力；Miles 将 teacher 信号加到 token advantage，不是简单把 reference 换成 teacher。纯 OPD 可以让 task reward 为零，仍由 teacher 差异产生更新。

### Miles 做到哪些工程细节

- **Teacher 评分而非重新生成答案**：SGLang teacher 收到学生的 `input_ids`，以 `max_new_tokens=0` 返回评分，再截取 response span。不能用教师重新生成的文本替代。[rollout teacher scorer][m-opd-rollout]
- **两种部署模式**：外部 SGLang teacher 允许不同架构，但必须有相同 tokenizer/token ID 语义；Megatron 本地 teacher 通过 checkpoint/权重切换复用模型结构。普通不返回指定 token logprobs 的黑盒 API 不能直接使用。
- **Top-k 估计**：可以按 student/teacher top-k、交集、并集等选集合，再按相应概率或 uniform 权重计算信号；和 Tunix teacher-top-k forward-KL 不是同一个目标。
- **多 teacher 路由**：按 sample metadata 选择一个 specialist teacher，支持 default fallback；不是多 teacher 分布 ensemble。[teacher 路由及 top-k 实现][m-opd-rollout]
- **分布式对齐**：teacher scores 随 batch 进入 DP 分发、CP 切片和媒体 token 展开，保持与 response mask 一致。[训练数据转换][m-convert]

此版本有明确限制：top-k 仅支持 SGLang teacher；需要 student 候选的 top-k 策略仍要求 `MILES_USE_LEGACY_ROLLOUT_V1=1`；稀疏逐位置 token-ID 评分需要相应的 patched teacher server，默认未开启。sampling-support replay、score centering 和部分省略 actor forward 的配置不能与 OPD 同开。[OPD 参数校验][m-args]、[score centering 校验][m-score-config]

后端限制也要区分 teacher 部署方式：torchtitan 不提供 Megatron 式本地 teacher 加载；外部 SGLang teacher 沿共享数据和 loss 路径接入，仍需单独端到端验证。不能从拒绝本地 teacher 的断言推成整个后端禁止 OPD。[torchtitan actor][m-titan]、[teacher 角色创建条件][m-train-group]、[共享训练流程][m-torch-native]

### 建议 Tunix 的最小实现

先做 **同步、纯文本、同 tokenizer、sampled-token、固定外部 teacher**，支持 pure OPD 和 GRPO+OPD。实现集中在 `dist_rl`；具体顺序和首个 recipe 见[实施路线](miles_implementation_roadmap.md)。

| 改动位置 | 具体工作 |
| --- | --- |
| 新的异步 teacher scorer | 输入原始 token IDs、response/assistant mask；输出同长度 teacher logprobs；初始化校验 tokenizer 兼容性，明确 timeout/retry 和失败行为 |
| `tunix/dist_rl/orchestrator/rl_program.py` 的 `critique_stage` | base payload 产生后、ScoredGroup 入队前 await teacher 评分，再用纯函数把 OPD 注入已有 token advantages；不在同步 adapter 里请求网络 |
| `tunix/dist_rl/orchestrator/algorithm_adapter.py` | 保持基础 estimator 和同步 payload 组装；共用 token advantage transform，避免把 OPD 做成互斥 estimator |
| 现有 `RLTrainerPayload.advantages` 与 assembler | 首版直接传已组合的 token advantage；teacher 不必作为新字段贯穿所有 trainer/packing 路径，旧 learner 和 TrainExample 不必同步改动 |
| 配置与 recipe | 固定外部 teacher、独立 OPD 系数、全词表采样、G=2、padded batch；先跑同步 pure OPD，再加 task reward、packing 和 async |

纯 OPD 还需检查两处现有 GRPO 语义：现有 GSM8K 入口拒绝 G≤1，`GRPOAdapter` 也会将有效 trajectory 少于两个的整组 mask 清零，所以最早 E2E 保留 G=2。以后支持单样本 OPD 时，再把 task-baseline 的组大小条件与 teacher 信号分开。Miles 在 task advantage 之后添加 OPD，group reward 标准化不会先吞掉 teacher 信号，但可选后置 advantage normalization 会改变组合尺度。[Tunix adapter][t-adapter]、[Miles objective][m-objective]

验收至少包括：相同 task-reward 配置下，`λ=0` 与关闭 OPD 一致；pure OPD 的 task reward 与 λ 同时为零时无有效更新；同一 teacher/student 数值路径下信号接近零；小分布例子验证梯度方向；prompt/padding/tool observation 梯度为零；reward=0 且 teacher 不同时仍产生更新；teacher 参数不变；长度、tokenizer 或返回字段不兼容时立即失败。再用同一 student、teacher 和 prompt set 比较数据集蒸馏、reference KL、pure OPD、GRPO+OPD，记录任务表现、输出长度、teacher 开销和 masked OPD 信号。

## 4 Multimodal 的重点是打通 RL 数据链路

### Tunix 已经有哪些多模态能力

Tunix 不是纯文本框架：Gemma3 有视觉输入，Gemma4 有视觉和音频输入；`image_processor.py`、`audio_processor.py` 已处理媒体和 placeholder。`SamplingRequest` 已包含 `images`、`audios`，`dist_rl/rollout/vanilla_sampler.py` 的原生采样路径会调用 Gemma4 processor；其他 sampler 不能自动继承这一能力。图像 DPO 也有 trainer 数据类型和完整 notebook。[Gemma4 模型][t-gemma4]、[采样请求][t-sampler]、[VLM DPO 示例][t-vldpo]

另外还有 [Gemma3 图像 SFT 脚本](../examples/sft/vlm_training.py)，使用 LaTeX_OCR 数据集、`TrainingInput.images` 和 LoRA。但它硬编码 GCS checkpoint/tokenizer，并仍调用当前模型不存在的 `get_positions_and_attention_mask`，应列为待维护参考示例，不能当作已验证的一键 recipe。[Gemma4 sampling notebook](../tunix/models/gemma4/sampling_example.ipynb) 则实际包含图像和音频推理调用；这些示例都不等同于标准多模态 GRPO/OPD 闭环。

**缺口在标准 RL 训练链路。** 通用 RL `TrainExample` 没有媒体字段；普通 GRPO 的 generate 和 actor/reference scoring 没有将媒体作为同一批数据完整传递。`compute_per_token_logps` 虽有可选 images，不等于上游 rollout、trajectory、packing、loss 已自动连通；audio 也不能由 images 参数代替。任意 metadata 容器同样不能替代训练侧可消费的媒体契约。[通用 RL batch][t-rl-common]、[GRPO learner][t-grpo]

还存在需要先验证的 attention 语义：vanilla rollout 调用模型的 `get_attention_mask`，Gemma3 用它处理视觉 token 块；通用 RL `process_ids` 当前构造普通 causal mask。只透传 images 仍不能保证 rollout 和 trainer 看到同一分布，首条 VLM recipe 应先统一模型输入／attention，再进入学习实验。[rollout mask][t-vlm-sampler-mask]、[RL mask][t-vlm-rl-mask]

### Miles 多出的完整链路

```text
结构化消息和媒体 → 模型 processor → prompt IDs + 训练媒体 tensors
                → SGLang image_data rollout，legacy 路径另有 audio_data
                → Sample 保留媒体、response、logprobs
                → ragged tensor 分发 → backend forward → response-only RL loss
```

Miles 将原始媒体和 `multimodal_train_inputs` 分开保存，后者随 batch 到 FSDP/Megatron forward。Qwen3-VL 还要处理 image/video grids、MRoPE、媒体 token 展开和 packed segment 位置。真正值得学习的是 **同一份媒体在 rollout、actor、reference 和 loss 中保持相同含义与位置**，以及处理器输出、mask、packing、CP 的共同契约。[processor][m-processor]、[媒体 batch 展开][m-mm-data]、[Qwen3-VL 模型适配][m-qwen-vl]

GEO3K 的单轮和多轮 recipe，以及真实 Qwen3-VL FSDP E2E 测试，为图像理解 RL 提供了比“model.forward 接受图片”更完整的参考。[GEO3K 示例][m-geo3k]、[VLM E2E][m-vlm-test]

### 支持范围不能泛化

| 方向 | 此版本可确认的范围 | 需要保留的边界 |
| --- | --- | --- |
| 图像理解 | 图像输入、文本回答的 RL recipe 和多轮示例 | 不意味着所有模型、packing、CP 和异步模式任意组合 |
| 音频理解 | Inkling audio processor/tower、legacy audio payload、训练媒体位置实现 | 默认 v2 单轮 payload 尚未发送 audio，不能当作默认可用闭环，也未达到与图像相同程度的 recipe 验证 |
| 视频理解 | 通用 processor、Qwen3-VL video grid 和 MRoPE 路径 | 已读默认 rollout 只明确发送 image/audio；Inkling processor 明确拒绝视频，不能宣称通用视频 RL 闭环 |
| Encoder 更新 | Inkling 默认冻结 vision/audio towers | `train=True` 尚未接完整 tower weight-sync/checkpoint，属于实验路径 |
| 并行与异步 | 已实现媒体 token 展开及部分附属张量重切片 | Inkling 媒体展开不支持 CP>1；Kimi 媒体展开后 policy-version span 的 remap 仍有 TODO；不能据此泛化多模态 OPD 覆盖 |

上述限制可在 [Inkling processor][m-inkling-processor]、[tower 实现][m-inkling-towers] 和 [multimodal batch][m-mm-data] 中直接核对。图像／视频**生成**则属于后文的独立 diffusion 体系。

### 建议 Tunix 的实施顺序

1. **默认 Gemma3 图像输入、单轮 GRPO、冻结 vision tower**。沿 dist_rl 接线；先验证模型加载、processor、attention mask 和同权重评分，再定义媒体 PyTree 并贯穿采样、trajectory、payload、assembler 和 loss。Gemma4 的动态 patch/count 与静态 PyTree metadata 另验。
2. **固定图片数和 patch bucket 的 recipe**。先避免每步 shape 改变触发 JAX 重编译，再扩展无图／多图混合 batch、截断和 packing。
3. **多轮截图或工具回传图片**。维护每轮 media-to-token 对齐，验证 tool observation mask，然后扩展 Gemma4 audio。
4. **视频及媒体生成单独扩展**。不因 processor 有视频参数就宣布支持视频训练，也不把输入理解和输出生成混为一项。

验收要同时覆盖媒体数量与 placeholder 校验、不同图片尺寸／顺序、response logprobs 对齐、prompt/media/tool token 的零训练 mask、冻结 tower 不变、LM/LoRA 确实更新、packing 后位置不变，以及固定 bucket 下稳定的编译次数。任务评测可从 GEO3K 或固定视觉问答子集起步。

## 5 已有系统能力应追覆盖范围和验证

### 异步训练与 token 连续性

当前主线已经有 `tunix/dist_rl/`，不能再以旧的 experimental distributed 目录作为比较基线。已有异步采样／训练流水、`max_staleness`、dispatch window、背压及权重版本信息。下一步应核查多轮轨迹跨版本、过期样本丢弃、GRPO group 完整性、评测快照隔离和 checkpoint 恢复后的消费语义。[Tunix orchestrator][t-orchestrator]

Tunix 的 `exact_token_continuity` 已处理框架内部的 token 连续性。Miles TITO/session 的增量主要是让外部黑盒 harness 仍使用原始 token IDs/logprobs，降低工具链重新 tokenize 的风险。优先补一个真实外部 harness 的 session adapter，再检查多轮 template、tool observation 和截断位置；不需要为追 TITO 名字另写一套相同 token 容器。

### MoE replay 与数值一致性

Tunix 已采集 rollout expert routing，并通过 `routed_experts`、packing、压缩编码送到 trainer，MaxText 路径也已有测试。Miles R3 应对标 replay 的模型／后端覆盖、长轨迹传输开销、session 路径、prompt/response 对齐和缺失 routing 的失败行为，不能列成 Tunix 完全缺失的模块。[Tunix replay 测试][t-replay-test]

Tunix 的 `zero_tim_kernel` 已有固定归约顺序、canonical attention、log-softmax、可微 Pallas forward 和 Qwen3 canonical model。其支持范围受模型、TP 配置和 kernel contract 限制，README 也明确 MoE 路径仍需跟进；这些内核和 benchmark 不等于任意模型的默认训练路径都已零误差。[Tunix Zero-TIM][t-zero-tim]

Miles 值得借鉴的是把 true-on-policy 组织成 **模型 profile → kernel/parallel contract → 启动参数校验 → E2E 指标**，当前契约也有模型范围。Tunix 下一步应将已有内核接入一个标准 Qwen3 recipe，明确支持的模型、mesh、dtype、temperature，再由同版本权重的 sampler/engine rescoring/trainer 对照验证，逐步扩大覆盖。[Miles true-on-policy contract][m-true-on-policy]

### 权重同步和故障恢复

Tunix 已有 Raiden 同步、coordinator、健康监控与 worker 生命周期。应比较的是传输耗时、暂停范围、版本原子切换、失败回滚、是否支持局部 worker 恢复，而非把 Miles P2P 当成 Tunix 的第一个同步方案。[Tunix 权重同步][t-weight-sync]

Miles 将权重生产、transport 和 engine pause/flush/announce/resume handshake 分开，适合借鉴接口边界；具体 GPU RDMA/NCCL/CUDA IPC 不能直接复制为 JAX/TPU 实现。恢复能力也必须按 rollout/trainer、单 worker/整个 cell、checkpoint/in-memory state 分开验收；进程能重启不等于训练状态无损恢复。[Miles 同步 session][m-weight-session]、[reconcile 设计][m-reconcile]

**动态重新加入是明确缺口。** Tunix 当前 worker 容错默认关闭；启用后能逐出失败 worker 并重试任务，但尚未完成自动动态 rejoin，短暂错误也可能永久缩小 rollout pool。Miles 已有健康检查、自动 suspend/resume、实例 incarnation reconciliation，以及新 cell 接收当前权重后才注册 router 的流程。建议将“重建 → 新 incarnation → 全量初始化当前权重 → 安全 rejoin”列为 P1；验收应包括传输中 receiver 退出、短暂网络错误、旧进程迟到回复和幸存者继续服务。[Tunix 容错限制][t-ft]、[Miles 恢复控制器][m-ft-controller]、[新 cell 就绪顺序][m-server-cell]

### 低精度和 agent 生态

Tunix 已用 qwix 支持 LoRA/QLoRA。Miles 真正值得补充的是低精度训练、rollout 量化和导出之间的一致闭环。MXFP8/NVFP4 有 Blackwell 硬件边界；INT4 QAT 聚焦 routed expert 的 GroupedLinear，不能推成整个优化器、激活和梯度都变成 INT4。TPU 路线应先定义量化前后 logprob/gradient 的误差预算和评测，再选择合适 kernel。[Tunix qwix 接入][t-qwix]、[Miles 低精度][m-low-precision]、[INT4 QAT][m-int4]

Agentic RL 两边都有。Miles 已有 Harbor、HUD、NeMo Gym、OpenEnv、Verifiers 等参考集成和 sandbox provider 抽象，其中许多连接器位于 `examples/experimental/`，成熟度应按连接器和 recipe 分别确认。Tunix 可以先从 DeepSWE/FrozenLake 已有接口中提炼 adapter，选一个外部环境跑通，验证 cancellation、timeout、sandbox cleanup 和 reward attribution。[环境集成][m-environments]

当前 TITO session 路径尚不能承载 image/video，VLM 需走 `/generate`，不能默认继承纯文本 harness 的全部路径。[Agentic rollout 的媒体限制][m-agentic-rollout]

## 6 另外几个真正的新板块

### Sampling-support replay

即使权重和 forward 完全相同，top-k/top-p 过滤后的采样概率也不同于全词表 softmax 概率。Miles 保存每个输出 token 实际使用的 support，并让训练端在同一 support 上归一化当前 logits：

```text
qθ(a | S) = exp(zθ(a) / T) / Σ[j∈S] exp(zθ(j) / T)
```

这和 replay MoE expert routes 是两件事，也不能靠一般 IS 权重或 Zero-TIM kernel 自动解决。当前 Tunix 的 RL batch 没有这种 support 记录，应先审计各 sampler 返回的是原始、temperature 后还是过滤后 logprobs，再决定是否补该机制。[Miles support 契约][m-sampling-mask]、[使用和限制][m-support-doc]

Miles 目前要求有界正 top-k 来承载有限 top-p 的 support，且暂不允许与 reference KL、OPD 同开，因为 loss 还缺少独立的 full-vocabulary actor score。Tunix 设计时最好一开始就区分 **behavior sampling score** 和 **full-policy score**，避免所有下游目标共用一个含义不明的 logprob 字段。

### Score centering

Miles 另有 score-centering loss，利用生成时记录的候选概率修正 off-policy score drift；可选 TIS/MIS。它是独立的 REINFORCE 风格目标，不是把 PPO clip 参数调整一下。top-k 近似对词表尾部建模，不应声称任意尾部分布下都精确抵消梯度偏移。[loss 实现][m-score-loss]

当前实现主要支持 GRPO advantage；对 OPD、true-on-policy、multi-LoRA、speculative decoding、multimodal expansion 等组合有显式限制。Tunix 可作为 P1 算法研究项，先用小词表完整分布的独立梯度 oracle 验证，再评估候选记录的通信和内存成本；例如候选 IDs 与 fp32 logprobs 的原始数组约为 `8 × response_length × K` 字节。[配置校验][m-score-config]

### Multi-policy 和 Multi-LoRA 服务

这两项要分别规划：

| 维度 | Multi-policy | Multi-LoRA/Tinker |
| --- | --- | --- |
| 训练对象 | 多个独立 policy，如 solver 与 verifier | 一个 frozen base 上多个独立 adapter |
| 主要需求 | 每个 policy 的 trainer/engine、数据路由、奖励交互、独立版本和协调存档 | 客户端 API、adapter slots、跨客户 batching、独立 optimizer、版本化采样 |
| Miles 入口 | `train_multi_policy.py`，solver/verifier GSM8K 示例 | `serve_tinker.py`，`miles/tinker/`，multi-LoRA gateway |
| 当前约束示例 | fully async、非 colocated、不支持 critic 等 | Megatron 路线、非 colocated、PP=1，gateway 的 futures/leases 不跨重启持久化 |
| Tunix 建议 | 有 self-play/协同训练任务再立项 | 有共享训练资源／服务客户需求再立项 |

单个 RL 任务里有 actor/reference/critic，或者用一个冻结 teacher，不等于 multi-policy 协同训练；单 adapter LoRA 也不等于多客户训练服务。Tunix 无须为功能表齐全立即承担两套新的运行时语义。[Multi-policy 编排][m-multi-policy]、[校验][m-multi-policy-config]、[Tinker 服务][m-tinker]、[multi-LoRA 范围][m-multi-lora]

## 7 图像和视频 Diffusion 应单独立项

Miles Diffusion 是独立的 `radixark/miles_diffusion`，已一并下载。它围绕连续 latent/DiT 图像视频模型，把 sglang-diffusion rollout、去噪轨迹、训练 batch、视觉 reward、Flow-GRPO/DiffusionNFT/SFT 和 FSDP2 接起来。[Miles Diffusion 源码][md-repo]

| 目标 | 源码中的实际训练信号 | 当前边界 |
| --- | --- | --- |
| Flow-GRPO | 保存 denoising step 的 latent/next latent、sigma、旧 logprob 和 advantage，重算 SDE transition logprob 后做 PPO ratio/clip | 依赖连续去噪轨迹及其概率，不能直接复用文本 token GRPO 数据 |
| DiffusionNFT | 从 clean x0 加噪，用 new/old policy 构造正负分支的 x0 重建误差，再按 advantage 组合 | 当前单 DiT component、visual stream |
| Diffusion SFT | 按 latent stream 采样时间与噪声，以 `noise − x0` 为 target 做 velocity MSE | 通用目标可选 visual/audio，内建 encoder 只分派 Wan2.2 和 H3 |

这些目标共用训练接口，但需要不同 batch 和 scorer。[Flow-GRPO 实现][md-flow]、[NFT 实现][md-nft]、[SFT 实现][md-sft]

音视频支持也需要按分支拆开：H3 T2VA rollout 可以生成音视频，但现有 Flow-GRPO 只训练 video branch，不训练 audio；Ref2VA 有 SFT 路径而无 RL 路径，现成 T2VA LoRA SFT recipe 默认也只监督 visual。不能从“输出含音频”推断“音频 RL 已完成”。[H3 模型配置][md-h3]

Tunix 当前 `DiffusionTokenBatch` 要求 `[batch, length]` target token IDs、对齐 logits 和 loss weights，`tunix/sft/diffusion.py` 已实现不做 AR shift 的 weighted token CE。这是离散 token diffusion 的通用接口／目标层，不能直接视为连续图像／视频 diffusion RL 的基础实现。[Tunix diffusion 合约][t-diffusion]、[SFT loss][t-diffusion-sft]

若决定追图像／视频生成，需要另行设计：模型和 VAE adapter、noise schedule、连续轨迹/action 与 step logprob 契约、视觉 reward 执行服务、训练／采样数值对齐、媒体存储与评测、LoRA 导出，以及训练后端。Flow-GRPO、DiffusionNFT 和 SFT 对轨迹／评分的需要也应分别定义，不宜硬塞进现有 AR `completion_ids`。

建议先选一个图像模型和一个 reward 跑通同步小规模 recipe，再做视频。Miles Diffusion 的 recipe 验证等级有明确含义，值得借鉴：

| 等级 | 项目定义 |
| --- | --- |
| FG | 跑过完整训练曲线；canonical recipe 至少每晚运行确定性 E2E；所有注册 metric 与 committed standard 精确一致 |
| PG | 跑过完整训练曲线；每晚确定性 E2E 覆盖文档指定的缩小单机 proxy，原始拓扑不享有同等覆盖 |
| V | 跑过完整训练曲线，但尚无满足 FG/PG 的确定性 E2E |
| NV | 没有完整训练曲线；smoke/debug 短跑不算验证 |

例如 SD3.5 Flow-GRPO+OCR 和 H3 video-only Flow-GRPO 标为 FG，Wan2.2 多机 full-finetune recipe 标为 PG，H3 T2VA LoRA SFT 标为 V，Wan2.2 LoRA SFT 标为 NV。等级只适用于指定脚本和拓扑，不能推广到整个模型族或所有 loss/并行/LoRA 组合。[验证等级定义与清单][md-verification]

## 8 下一阶段建议拆成的工作包

详细执行顺序、现成脚本、代码落点和每一步的完成门槛统一放在[实施路线](miles_implementation_roadmap.md)。按单人推进的默认顺序是：**native 文本基线 → OPD 闭环与质量对照 → Gemma3 单图输入和评分一致性 → 图像 RL → packing／多轮 → 有界异步与恢复**。OPD 与 VLM 可在公共基线后部分并行，不需要先重写 trainer 或完成全部系统优化。

每条 recipe 至少统一记录四类指标：

- **学习质量**：固定评测集表现、输出长度、有效样本／有效 token 比例；OPD 另记 masked student-teacher 信号。
- **训练正确性**：同版本 train/rollout logprob 差异、policy version lag、IS/clip 比例、mask/packing 对齐、NaN/Inf。
- **吞吐与成本**：rollout、teacher scoring、batch assembly、train、weight sync、eval 分项耗时；编译与 warm-up 单列。
- **稳定性**：失败／重试／丢弃样本数量、恢复耗时、恢复后权重版本和数据消费一致性。

Tunix 已有 perf/Perfetto 和 nightly regression，可以在这些基础上增加 recipe 的数值与质量门禁。Miles 的历史指标 gate 以同一 test/backend/hardware/metric 的可信历史作为基线，能发现单次 smoke test 不容易发现的慢性退化；这比只扩大模型清单更值得学习。[Miles 历史指标门禁][m-ci-gate]

## 9 源码阅读顺序与后续更新

最快建立整体认识的顺序：

1. Miles `train.py`、`train_async.py`、`miles/rollout/inference_rollout/`：训练与采样生命周期。
2. `miles/utils/types.py`、`miles/ray/rollout/train_data_conversion.py`：sample 到训练 batch 的契约。
3. `miles/backends/training_utils/loss/objective.py`、`loss/hub/opd.py`、`miles/rollout/on_policy_distillation.py`：OPD 和其他目标怎样组合。
4. `examples/geo3k_vlm/`、`miles/utils/processing_utils.py`、`data/multimodal.py`：图像 RL 的实际数据路径。
5. `miles/backends/training_utils/weight_update/`、`miles/ray/`、`miles/true_on_policy/`：系统契约与验证。
6. 最后按需求看 `train_multi_policy.py`、`miles/tinker/` 和独立 `miles_diffusion`，避免一开始把项目范围铺得过大。

后续跟进时，在 Tunix 仓库根目录执行：

```bash
git -C .references/miles fetch origin main
git -C .references/miles diff HEAD..origin/main -- miles/ examples/ tests/ docs/
git -C .references/miles_diffusion fetch origin main
git -C .references/miles_diffusion diff HEAD..origin/main -- miles/ examples/ scripts/ tests/
```

先审阅变化，再决定是否更新参考 checkout 与本文的固定提交。优先跟踪 OPD、multimodal、torchtitan/TPU、session、true-on-policy 和测试中的变化；功能更新应同时检查参数校验、recipe 与 E2E 覆盖。

官方背景材料：[Miles 仓库](https://github.com/radixark/miles)、[OPD 设计与初步实验](https://www.lmsys.org/blog/2026-07-18-opd-support-in-miles)、[Miles v0.1 技术报告](https://arxiv.org/abs/2609.08368)。排期以本文固定源码的实现与限制为准。

[t-algo-config]: https://github.com/google/tunix/blob/2176443ad6bb61093e8d133a0d2d19045655db80/tunix/rl/algorithm_config.py
[t-distill]: https://github.com/google/tunix/blob/2176443ad6bb61093e8d133a0d2d19045655db80/tunix/distillation/distillation_trainer.py#L108
[t-logit]: https://github.com/google/tunix/blob/2176443ad6bb61093e8d133a0d2d19045655db80/tunix/distillation/strategies/logit.py#L116
[t-rl-common]: https://github.com/google/tunix/blob/2176443ad6bb61093e8d133a0d2d19045655db80/tunix/rl/common.py#L103
[t-grpo]: https://github.com/google/tunix/blob/2176443ad6bb61093e8d133a0d2d19045655db80/tunix/rl/grpo/grpo_learner.py#L190
[t-gemma4]: https://github.com/google/tunix/blob/2176443ad6bb61093e8d133a0d2d19045655db80/tunix/models/gemma4/model.py#L523
[t-sampler]: https://github.com/google/tunix/blob/2176443ad6bb61093e8d133a0d2d19045655db80/tunix/dist_rl/rollout/sampler.py#L68
[t-vldpo]: https://github.com/google/tunix/blob/2176443ad6bb61093e8d133a0d2d19045655db80/examples/vl_dpo_gemma3.ipynb
[t-orchestrator]: https://github.com/google/tunix/blob/2176443ad6bb61093e8d133a0d2d19045655db80/tunix/dist_rl/orchestrator/orchestrator.py
[t-replay-test]: https://github.com/google/tunix/blob/2176443ad6bb61093e8d133a0d2d19045655db80/tests/rl/router_replay_maxtext_test.py
[t-zero-tim]: https://github.com/google/tunix/blob/2176443ad6bb61093e8d133a0d2d19045655db80/tunix/zero_tim_kernel/README.md
[t-weight-sync]: https://github.com/google/tunix/tree/2176443ad6bb61093e8d133a0d2d19045655db80/tunix/dist_rl/weight_sync
[t-qwix]: https://github.com/google/tunix/blob/2176443ad6bb61093e8d133a0d2d19045655db80/tunix/cli/utils/model.py#L31
[t-diffusion]: https://github.com/google/tunix/blob/2176443ad6bb61093e8d133a0d2d19045655db80/tunix/diffusion/README.md
[t-diffusion-sft]: https://github.com/google/tunix/blob/2176443ad6bb61093e8d133a0d2d19045655db80/tunix/sft/diffusion.py#L30
[t-adapter]: https://github.com/google/tunix/blob/2176443ad6bb61093e8d133a0d2d19045655db80/tunix/dist_rl/orchestrator/algorithm_adapter.py#L285
[t-ft]: https://github.com/google/tunix/blob/2176443ad6bb61093e8d133a0d2d19045655db80/tunix/dist_rl/common/datatypes.py#L220
[t-vlm-sampler-mask]: https://github.com/google/tunix/blob/2176443ad6bb61093e8d133a0d2d19045655db80/tunix/dist_rl/rollout/vanilla_sampler.py#L691
[t-vlm-rl-mask]: https://github.com/google/tunix/blob/2176443ad6bb61093e8d133a0d2d19045655db80/tunix/rl/common.py#L313
[m-platform]: https://github.com/radixark/miles/blob/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/miles/platforms/tpu.py
[m-tpu]: https://github.com/radixark/miles/blob/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/docker/Dockerfile.tpu
[m-objective]: https://github.com/radixark/miles/blob/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/miles/backends/training_utils/loss/objective.py#L31
[m-opd-loss]: https://github.com/radixark/miles/blob/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/miles/backends/training_utils/loss/hub/opd.py
[m-opd-rollout]: https://github.com/radixark/miles/blob/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/miles/rollout/on_policy_distillation.py
[m-convert]: https://github.com/radixark/miles/blob/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/miles/ray/rollout/train_data_conversion.py#L206
[m-args]: https://github.com/radixark/miles/blob/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/miles/utils/arguments.py#L1046
[m-titan]: https://github.com/radixark/miles/blob/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/miles/backends/torchtitan_utils/actor.py#L64
[m-train-group]: https://github.com/radixark/miles/blob/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/miles/ray/train/group.py#L81
[m-torch-native]: https://github.com/radixark/miles/blob/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/miles/backends/training_utils/torch_native/actor.py#L198
[m-processor]: https://github.com/radixark/miles/blob/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/miles/utils/processing_utils.py#L113
[m-mm-data]: https://github.com/radixark/miles/blob/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/miles/backends/training_utils/data/multimodal.py
[m-qwen-vl]: https://github.com/radixark/miles/blob/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/miles_plugins/models/qwen3_vl.py#L289
[m-geo3k]: https://github.com/radixark/miles/tree/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/examples/geo3k_vlm
[m-vlm-test]: https://github.com/radixark/miles/blob/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/tests/e2e/fsdp/test_qwen3_vl_4B_fsdp.py#L13
[m-inkling-processor]: https://github.com/radixark/miles/blob/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/miles_plugins/models/inkling/mm_processor.py#L240
[m-inkling-towers]: https://github.com/radixark/miles/blob/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/miles_plugins/models/inkling/mm_towers.py#L77
[m-true-on-policy]: https://github.com/radixark/miles/tree/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/miles/true_on_policy
[m-weight-session]: https://github.com/radixark/miles/blob/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/miles/backends/training_utils/weight_update/session.py
[m-reconcile]: https://github.com/radixark/miles/blob/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/docs/developer/reconcile-loop.md
[m-ft-controller]: https://github.com/radixark/miles/blob/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/miles/utils/ft_utils/mini_ft_controller.py#L169
[m-server-cell]: https://github.com/radixark/miles/blob/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/miles/ray/rollout/server_cell.py#L233
[m-low-precision]: https://github.com/radixark/miles/blob/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/docs/advanced/low-precision.md
[m-int4]: https://github.com/radixark/miles/blob/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/docs/advanced/int4-qat.md
[m-environments]: https://github.com/radixark/miles/blob/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/docs/user-guide/environments.md
[m-agentic-rollout]: https://github.com/radixark/miles/blob/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/docs/user-guide/agentic-rollout.md
[m-sampling-mask]: https://github.com/radixark/miles/blob/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/miles/utils/sampling_mask.py
[m-support-doc]: https://github.com/radixark/miles/blob/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/docs/advanced/sampling-support-replay.md
[m-score-loss]: https://github.com/radixark/miles/blob/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/miles/backends/training_utils/loss/hub/score_centering/estimator.py#L66
[m-score-config]: https://github.com/radixark/miles/blob/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/miles/utils/score_centering.py#L10
[m-multi-policy]: https://github.com/radixark/miles/blob/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/train_multi_policy.py
[m-multi-policy-config]: https://github.com/radixark/miles/blob/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/miles/utils/multi_policy/utils.py#L113
[m-tinker]: https://github.com/radixark/miles/blob/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/miles/tinker/core/service.py#L38
[m-multi-lora]: https://github.com/radixark/miles/blob/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/docs/advanced/multi-lora.md
[m-ci-gate]: https://github.com/radixark/miles/blob/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/docs/developer/ci/03-metric-history-gate.md
[md-repo]: https://github.com/radixark/miles_diffusion/tree/221f965fa68e4d865665a48e8478932a231dc4ad
[md-flow]: https://github.com/radixark/miles_diffusion/blob/221f965fa68e4d865665a48e8478932a231dc4ad/miles/backends/fsdp_utils/loss_hub/flow_grpo.py#L108
[md-nft]: https://github.com/radixark/miles_diffusion/blob/221f965fa68e4d865665a48e8478932a231dc4ad/miles/backends/fsdp_utils/loss_hub/nft.py#L23
[md-sft]: https://github.com/radixark/miles_diffusion/blob/221f965fa68e4d865665a48e8478932a231dc4ad/miles/backends/fsdp_utils/loss_hub/sft.py#L37
[md-h3]: https://github.com/radixark/miles_diffusion/blob/221f965fa68e4d865665a48e8478932a231dc4ad/miles/backends/fsdp_utils/configs/h3.py#L1
[md-verification]: https://github.com/radixark/miles_diffusion/blob/221f965fa68e4d865665a48e8478932a231dc4ad/docs/user-guide/recipe-verification.md
