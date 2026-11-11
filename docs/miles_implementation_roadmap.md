# Tunix 追赶 Miles 的实施顺序

日期：2026 年 10 月 29 日。源码基线与能力证据见[差距分析](miles_gap_analysis.md)。以下是实施计划，尚未在加速器上完成这些训练验收。

当前进度：`dist_rl` 的固定外部 teacher、sampled-token pure OPD／GRPO+OPD、异步 critique 接线与 CPU 测试已实现，运行方式见[OPD 示例](../tunix/dist_rl/examples/on_policy_distillation/README.md)。下一步是在实际 teacher endpoint 与 TPU 上执行顺序 0 至 2 的运行和学习验收；top-k、多 teacher、G=1 与多模态 OPD 仍属后续扩展。全局方向、具体缺口和启动条件见[研发地图](tunix_dev_map.md)。

**默认路线是：固定文本基线 → 文本 OPD 闭环与学习实验 → 图像 RL 闭环 → packing 和多轮 → 有界异步及恢复 → 按瓶颈扩展。** 首版以 `tunix/dist_rl` 为唯一编排入口，复用现有 trainer、loss 和 token advantage。OPD 与图像 RL 没有算法上的先后依赖；按单人推进、先交付新增训练能力的目标，OPD 的改动面更小，适合先做。

第一阶段只追两个可交付结果：**一个有质量对照的文本 OPD recipe，一个图像输入的单轮 RL recipe**。教师评分资源不可用时，可以先交付图像 RL；不必等本地 teacher runtime 或多后端支持全部完成。

## 1 按这个顺序实施

| 顺序 | 具体交付 | 为什么在这里做 | 进入下一步的条件 |
| --- | --- | --- | --- |
| 0 | 固定 native Tunix 的单机文本 GRPO 基线 | 先排除采样、token、权重版本和评分语义错误 | 两轮真实权重同步闭环可用；能重放同一轨迹核对 token/logprob/mask |
| 1 | TeacherScorer 和同步 sampled-token pure OPD | 复用文本链路，只增加教师评分与 token advantage 变换 | 零 task reward 仍能有效更新；同 task-reward 配置下 λ=0 与关闭 OPD 一致；teacher 固定 |
| 2 | Pure OPD 与 GRPO+OPD 的学习对照 | 确认新增能力有实际价值，再扩大实现范围 | 任务表现、输出长度、teacher 成本和有效学习信号可解释 |
| 3 | Gemma3 单图模型输入与评分一致性 | 图像 RL 最容易隐藏的错误在 mask、placeholder 与模型配置 | 同一图片、token、权重下，rollout/trainer 使用相同媒体与 attention 语义 |
| 4 | 单轮图像理解 RL recipe | 接通真实新增板块，并控制 shape 和状态复杂度 | 冻结 vision、语言参数确实更新、图像对输出有影响，固定评测可复现 |
| 5 | Packing 和多轮工具任务 | 在已正确的 recipe 上扩大场景；优先复用现有 DeepSWE | packed/unpacked 对照成立，tool mask、截断和 exact tokens 不变 |
| 6 | 有界异步、恢复与性能优化 | 基于学习结果和分项耗时，决定系统投入 | version lag、样本消费、恢复后身份／权重一致，吞吐收益可测量 |

顺序 0 至 2 形成第一个独立里程碑；顺序 3 至 4 形成第二个。多人协作时，可在共同基线后并行做 VLM 的加载、媒体契约与 attention 小原型；teacher scorer 接入和 VLM 输入适配由不同人负责，避免同时修改同一编排文件。

## 2 第一阶段先交付文本 OPD

### 第一个基线用现成 native smoke

入口是 `tunix/dist_rl/examples/recipes/ci_smoke_gsm8k_qwen3_0p6b.sh`，它已有 Qwen3-0.6B、native Tunix trainer、inprocess vLLM 和 Raiden 的单机布局。[基线脚本][baseline]

首轮保持该脚本的 batch、dtype、mesh 和 checkpoint 配置，只将下面的 OPD 约束明确记录并校验：

| 设置 | 首版选择 | 原因 |
| --- | --- | --- |
| Student | Qwen3-0.6B | 复用已有小模型入口，控制训练成本 |
| Rollout / trainer | inprocess vLLM / native Tunix | 已有参考链路；暂不同时加入 MaxText 评分路径 |
| 权重同步 | Raiden，至少两轮 | 验证更新后继续采样；关闭 sync 的 smoke 不能证明训练闭环 |
| 每 prompt 采样数 | 2 | 当前入口拒绝 G≤1，adapter 也会清空不足两个有效样本的整组 mask |
| `USE_ROLLOUT_LOGPS` | 显式 true | `StandardRLProgram` 会按此配置覆盖 `return_logprobs` |
| Temperature / top-p / top-k | 1 / 1 / disabled | 使用全词表随机采样，先避开 support replay 的额外契约 |
| Reference beta | 0 | 与 OPD teacher 系数分开，首轮减少干扰项 |
| Staleness / in-flight updates | 0 / false | 保留受控策略版本；这不等于跨引擎数值严格一致 |
| Batch 形式 | 显式长度的 padded batch | Packing 不是首个 OPD 或 VLM 闭环的前置条件 |
| 每批更新次数和额外修正 | 一次更新，关闭 sampler IS/RS 与额外 entropy 项 | 先把 OPD 自身的梯度和信号解释清楚，再逐项恢复 |

这里 CLI 的 `top_k=-1` 会在 GSM8K 入口转成请求里的 `0`；不能把不同后端的禁用哨兵值直接混用。基线 shell launcher 原先未转发 temperature/top-p/top-k 三个参数，`haoyu-next` 已补齐接线和回归检查。[入口参数与组大小检查][gsm8k-entry]

先重放少量轨迹检查原始 token IDs、completion 起点、padding、EOS、temperature 与实际服务权重版本。随机挑题的 GRPO reward 可能全相同，不能把一次零 loss 当作损坏，也不能把进程退出成功当作发生了有效学习；至少用一个有非零 advantage 的受控样本确认更新方向。

### OPD 拆成三个可 review 的改动

**OPD A：独立的异步 teacher scorer 与评分探针。** 输入未 packing 的原始 prompt/completion IDs，返回逐 completion token 的 fp32 teacher logprobs，以及 teacher checkpoint/revision 和评分分布语义。首版固定一个外部 teacher，做 tokenizer/token-ID 兼容性检查、timeout、错误传播和有界请求；缺字段、长度错位时不能回填零分。先用记录好的轨迹验证 scorer，不启动训练更新。[Miles 评分参考][m-opd]

**OPD B：在 critique 阶段注入 token advantage。** 当前 `critique_stage` 先算 scalar reward，再调用同步 `create_trainer_payloads`。建议在 base payload 产生后、`ScoredGroup` 入队前 `await teacher_scorer`，随后用纯函数更新已有 `payload.advantages`：[实际接入点][critique]

```text
原始 task reward → base payloads
                  → await teacher scorer
                  → A'_t = A_task,t − λ × (rollout_logp_t − teacher_logp_t)
                  → ScoredGroup → 现有 pad/pack → 现有 policy loss
```

adapter 保持同步数学函数，不在里面做 HTTP、加载模型或循环重试。首版用完 teacher vector 后只保留聚合指标和必要的评分身份；**不必先给 TrainExample、RLTrainerPayload、所有 assembler 和 loss 增加一套 teacher 字段**。现有逐 token advantages 已能随 batch 传递。以后需要逐 token 审计、重新计算其他目标或分布式评分缓存时，再扩展专门 side-channel。

pure OPD 将参与训练的 task reward 置零，但保留原始任务得分作为评测指标；teacher 差异在基础 advantage 之后注入。首版保留 G=2。G=1 的纯 OPD 优化应单独修改入口和有效组语义，不能通过全局删除 GRPO 的有效性检查实现。[adapter 的有效组与 payload][adapter]

**OPD C：真实 recipe 和质量对照。** 先做两轮闭环，再使用固定训练／评测子集比较原 student、pure OPD、GRPO、GRPO+OPD。相同权重的 teacher/student 可验证机制；能力迁移实验需要有目标行为或更强任务表现的固定 teacher。先验证 tokenizer 兼容，再选择 teacher checkpoint，不以模型名称相同替代检查。

第一个里程碑的验收：

- 相同 task-reward 配置下，λ=0 与关闭 OPD 的更新一致；pure OPD 同时将 task reward 和 λ 置零时应无有效更新。toy 分布验证梯度符号，旧 student scores 固定。
- 零 task reward、teacher 与 student 不同的有效样本产生非零更新；teacher 参数不变。
- padding、prompt、失败 trajectory 不参与 loss；有效 token 数不被重复计数。
- 第二轮使用同步后的学生权重；checkpoint 恢复后 teacher 身份与 OPD 配置一致。
- 任务指标、输出长度、masked OPD 信号、teacher 时延和有效 token 数一起记录；不以单条 loss 曲线下降宣称学习成功。

首版外部 scorer 避免占用 reference 角色。现有 `InferenceWorker` 明确限制 `model_role="reference"`，且接口只接受 padded payload；未来本地 teacher provider 需要独立身份和配置，不是换 checkpoint 就自动完成。[现有 frozen scorer 边界][inference]

## 3 第二阶段做图像 RL 时先验证输入语义

### 默认先选 Gemma3 和固定单图

首选 **Gemma3-4B-IT、原生 vanilla sampler、固定单图、padded batch、冻结 vision tower**。依据是现有 VLM DPO 已演示 `text_only=False` 的图像加载和训练路径，Gemma3 的普通 batch-major 图片张量与固定 image soft-token 数比 Gemma4 的可变 patch/count 契约更适合首轮隔离问题。[现有图像 DPO][vl-dpo]

Gemma4 E2B 的模型规模有吸引力，但 `PreprocessedVisionInput.soft_token_counts` 是 PyTree static metadata，模型还会依其值循环；不同图片 bucket、DP 切片和编译缓存要一起处理。如果设备资源迫使优先选 Gemma4，就把这些列为首轮必要实现，不因模型更小而省略。[Gemma4 媒体结构][gemma4-input]

### VLM 先拆两个改动

**VLM A：统一模型 factory、可训练参数和输入语义。** 两侧明确加载 `text_only=False`，配置 image processor，使一张图片确实进入模型。首轮关闭 flash attention 等优化，使用可检查的 dense mask；先对冻结的同 checkpoint 跑图像敏感性和 response-logprob 对照，再做 optimizer 更新。

这里有两个必须提前解决的实际问题：

1. **Attention mask 不同。** vanilla prefill 会调用 `model.get_attention_mask`；Gemma3 据此构造视觉 token 的特殊 attention。当前 RL `process_ids` 构造普通 causal mask，单纯把 images 透传到 forward 仍会产生不同语义。应抽出或复用模型相关输入／mask 构建，保持文本回退；现有 DPO 已显式使用该接口。[rollout mask][sampler-mask]、[RL mask][rl-mask]、[DPO mask][dpo-mask]
2. **冻结与 LoRA 不是直接开 flag。** 正确性演示优先采用显式 freeze filter 的 LM 全参更新，保留普通模型图与全树同步，先确认设备内存可容纳。optimizer、autodiff、初始及重建的 gradient accumulator 必须共用同一 trainable filter；只改 optimizer 的 `wrt` 会让梯度树不一致。首版明确列出并冻结 vision encoder 与媒体 projection/norm 路径。如果必须使用 LoRA，需按 Gemma 语言参数路径配置：当前 factory 的默认正则面向 Qwen；vanilla sampler 也不会自动构建相同 LoRA graph。因此训练与采样需共用模型／adapter factory，首版先验证全树同步，不先做 adapter-only transport。两条路线都要验证冻结参数不变、语言参数改变和参数树一致。[当前 optimizer 与 accumulator][trainable-filter]

**VLM B：接通媒体数据到单轮 RL。** 给媒体使用明确 PyTree/DTO，串通 dataset/request、agent/collector、trajectory、RLTrainerPayload、padded assembler、actor/reference scoring 与 loss。特别要覆盖 `LogprobsRequest`、TrainerWorker 的 microbatch 切片和 InferenceWorker 重建请求三处，避免评分时丢媒体。图片不能只放 metadata：训练 payload 预处理会清理 metadata，且 JIT 的输入／sharding 需要真实字段。[payload 处理][prepare-payload]

首版只支持一张固定 bucket 图片、文本回答、冻结 vision；先做少量有可检查答案的视觉题，再做 GEO3K 或固定视觉问答子集。媒体 packing 启动即拒绝；超过 prompt budget、可能截掉部分 image placeholder 块的输入也先拒绝。全局 batch 不需要立刻支持无图／多图混合、视频、audio 和可训练 vision encoder。

第二个里程碑的验收：换图片会改变目标模型的分布；图片/placeholder 数量不符时报错；rollout 和 trainer 的图片、位置、attention mask 一致；只对 assistant answer 训练；vision 权重不变、语言参数更新；跨一次权重同步还能继续采样；固定 bucket 连续多步不反复重编译。

## 4 两个闭环之后再扩大场景

**默认先验证文本 packing，再扩大多轮，最后增加 policy lag。** Packing 不是多轮的技术前置；若下一项就是现有 padded DeepSWE OPD，可以直接复用多轮路径。对 OPD，teacher 仍在 packing 之前评分，现有 token advantages 随 packing 传递；对 VLM，必须另做 media-to-segment 映射，不能因文本 packing 已通过就默认支持，媒体 packing 也不应阻塞纯文本 DeepSWE。

多轮优先复用现有 `tunix/dist_rl/examples/deepswe_dist/` 或更小的工具环境。检查 assistant/tool mask、exact token continuity、截断、取消和 sandbox cleanup。已有 DeepSWE 能表达任务时，先在该路径验证 OPD，不把 Harbor/OpenEnv 接入变成前置重写。

异步从 `max_staleness=0` 向有限 lag 扩展，始终保留实际采样的 student score 和 policy version。OPD 在 critique 已固定，后续 sampler/trainer agreement 的重算不能悄悄改变它的含义。分别统计 teacher 等待、queue 等待、样本过期、group 不完整和有效训练 token，才能判断异步是否抵消了新增成本。

长程、多 worker 或可抢占集群实验开始前，安排动态恢复：失败检测 → evict → 重建新实例 → 初始化当前权重 → 安全重新加入。Tunix 当前 eviction 后尚不支持动态 rejoin；短程功能验证可以 checkpoint restart，长程任务不能把永久缩池当成恢复成功。[当前容错边界][fault-tolerance]

## 5 哪些事情先后取决于实际瓶颈

| 工作 | 提前的触发条件 | 默认排期 |
| --- | --- | --- |
| Zero-TIM recipe | 同权重跨引擎的伪 OPD 信号已接近目标 teacher 的真实差异；或 IS/clip 异常主要由数值路径引起 | 基线先测量；无需等所有模型 bitwise 相同才写 OPD |
| G=1 pure OPD | G=2 rollout 已占主要成本，且不需要 group task baseline | 文本闭环之后；修改专用校验，不放宽普通 GRPO |
| Teacher batching/cache | profiler 显示 teacher 请求／评分主导步时 | OPD 学习对照之后；缓存键包含 teacher revision、token IDs 和评分配置 |
| 动态 rejoin | 长程 DeepSWE、多 rollout worker、抢占实例，或已出现持续缩池 | 可与功能主线并行；规模化之前完成 |
| Sampling-support replay | 任务必须使用 top-k/top-p filtering | 单独实现 behavior score/full-policy score；不阻塞全词表采样的 OPD 首版 |
| 外部 harness/session | 必须使用 Harbor/OpenEnv 等 benchmark，或维护多套 harness 已有重复成本 | 先接一个具体 harness，不全面重写 agentic 层 |
| Top-k OPD／多 teacher | sampled-token 方差／监督密度或多任务能力成为明确瓶颈 | 基础 OPD 之后，不因 Miles 有此 flag 就提前 |
| 音频理解 | 有具体音频任务和 Gemma4 资源 | 图像输入契约稳定后扩展；VLM+OPD 是两条链路各自通过后的组合项 |
| 低精度、score centering | 已有质量基线，且内存、吞吐或 off-policy 稳定性成为瓶颈 | 后续独立评估；不与首版 OPD 同时引入 |
| Multi-policy／multi-LoRA 服务 | 有 solver/verifier 协同训练或共享训练服务的实际需求 | 独立项目，不作为新增训练板块的前置 |
| 连续图像／视频 diffusion RL | 已确定生成模型方向、JAX 模型／采样后端和视觉 reward 资源 | 独立项目；不能沿 AR token payload 少量加字段完成 |

TeacherScorer 和 critique-stage OPD transform 现已接入。下一项工作是用真实固定 teacher 验证评分协议，再在目标 TPU 上跑原 native baseline 与两步 OPD recipe，确认第二轮使用更新权重。通过后开展固定评测集的学习对照，再决定投入哪些系统能力。

[baseline]: https://github.com/google/tunix/blob/2176443ad6bb61093e8d133a0d2d19045655db80/tunix/dist_rl/examples/recipes/ci_smoke_gsm8k_qwen3_0p6b.sh
[gsm8k-entry]: https://github.com/google/tunix/blob/2176443ad6bb61093e8d133a0d2d19045655db80/tunix/dist_rl/examples/math_gsm8k_dist/run_gsm8k_dist_grpo.py#L405
[critique]: https://github.com/google/tunix/blob/2176443ad6bb61093e8d133a0d2d19045655db80/tunix/dist_rl/orchestrator/rl_program.py#L762
[adapter]: https://github.com/google/tunix/blob/2176443ad6bb61093e8d133a0d2d19045655db80/tunix/dist_rl/orchestrator/algorithm_adapter.py#L285
[inference]: https://github.com/google/tunix/blob/2176443ad6bb61093e8d133a0d2d19045655db80/tunix/dist_rl/worker/inference_worker.py#L144
[m-opd]: https://github.com/radixark/miles/blob/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/miles/rollout/on_policy_distillation.py#L355
[vl-dpo]: https://github.com/google/tunix/blob/2176443ad6bb61093e8d133a0d2d19045655db80/examples/vl_dpo_gemma3.ipynb
[gemma4-input]: https://github.com/google/tunix/blob/2176443ad6bb61093e8d133a0d2d19045655db80/tunix/models/gemma4/config.py#L34
[sampler-mask]: https://github.com/google/tunix/blob/2176443ad6bb61093e8d133a0d2d19045655db80/tunix/dist_rl/rollout/vanilla_sampler.py#L691
[rl-mask]: https://github.com/google/tunix/blob/2176443ad6bb61093e8d133a0d2d19045655db80/tunix/rl/common.py#L313
[dpo-mask]: https://github.com/google/tunix/blob/2176443ad6bb61093e8d133a0d2d19045655db80/tunix/sft/dpo/dpo_trainer.py#L379
[prepare-payload]: https://github.com/google/tunix/blob/2176443ad6bb61093e8d133a0d2d19045655db80/tunix/dist_rl/train/peft_trainer_v2.py#L1095
[trainable-filter]: https://github.com/google/tunix/blob/2176443ad6bb61093e8d133a0d2d19045655db80/tunix/dist_rl/train/peft_trainer_v2.py#L457
[fault-tolerance]: https://github.com/google/tunix/blob/2176443ad6bb61093e8d133a0d2d19045655db80/tunix/dist_rl/common/datatypes.py#L220
