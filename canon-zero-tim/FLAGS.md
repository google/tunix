# FLAGS.md — CANON_* 注册表

> 政策:**建 flag 自由但必须登记(带日落条件);删 flag 有序按日落执行。**
> 焊死数值类 flag = 删代码路径 = 程序变更,走与开启同级认证门(verify+ALIGN+canary)。
> 生命周期档位:试验 → 已认证 → 默认开 → 焊死(开关可删)→ 退役/否决。
> 普查基点 a94d6c0c(285 个可设置 env flag,与 ebba4850 普查零漂移);普查后续现役附录
> 当前 368 个;本表分层登记,D 层按前缀组、语义欠账标"待考古"。
> 全量机器清单:落地 CL 时由 `grep -rhoE` 生成为附录,条目数必须 == 普查数(排除项列明)。

## A 层 · 数值语义类(动它 = 动程序身份;焊死走认证门)

| Flag | 语义 | 默认 | 生命周期 | 日落条件 |
|---|---|---|---|---|
| CANON_FIXED_AR | R1:TP 归约换固定序 ppermute 树 | off(canonical lane 开) | 已认证(1host+DP16/DP8/256) | 转正焊死:全负载默认开满一周期后无条件化 |
| CANON_FIXED_AR_EMBED | R3 补漏:vocab 分片 embedding gather 固定序 | off | 已认证 | 同上,与 FIXED_AR 同批 |
| CANON_RPA_VJP2(+VJP2_MAX_SEQS) | R4:cache-aware 认证反向 | off | 已认证(fp64 oracle+20/20+21/21) | 转正焊死;MAX_SEQS>1 需归约序审计先行 |
| CANON_LOGPROB_M / CANON_PROMPT_PROCESSED_LOGPROBS | R5:三臂共享 logprob callable,M=256 | off | 已认证(G9 4/4+负控) | 转正焊死 |
| CANON_MM_ALGO / CANON_MM_ALGO_PRESET | P19/P38:非 Pallas einsum 的 dot-algorithm 判别器;P38 仅允许 fixed `BF16_BF16_F32` 单变量臂 | off / BF16_BF16_F32 | **否决区**:旧 M16/M2048 e2e 无效;P38s22 三轮 A-B 仍红 | 可删,判决记录永存 |
| CANON_P38_FIXED_LM_HEAD | P38.2x/2y/2y1/2y2:registered Qwen3 output heads 的 M8/16/32/64/128/256 request buckets 均 pad 到 M256；通常 learner M4096 映射为 16xM256，只有 Qwen3-8B/TP8 FrozenLake 另登记 learner M2048→8xM256；untied `JaxLmHead` 与 tied `JaxEmbed.decode` 共用 Pallas body 和 endpoint-scoped receipts。几何为 1.7B K2048/TP4 tied、8B K4096/TP4 或 TP8 untied（TP4 N37984→38144；TP8 N18992→19200）、4B K2560/TP8 tied、32B K5120/TP8 untied（N18992→19200）；tiles 均 BM128/BN256/BK256 | off；仅显式 renderer opt-in | 历史 8B/TP4 backward-no-commit 与多模型 pinned-image construction 证据保留；首个 FrozenLake DP8×TP8 full attempt `f45g` 在 pre-backward C-forward 暴露 M2048 漏登记，当前 source/host 26/26 已修复并含旧 M4096 冒充的负控，post-fix pinned image 与 DP8×TP8 target 均未跑；one-host DP4×TP1 代理仍因未登记 TP1 fixed-head 几何而明确保持 off | 每个 model/TP/endpoint 独立 target 绿后逐项转正；任一红仅退役对应 registry entry |
| CANON_PROMPT_DIRECT_LOGPROBS / ABSOLUTE_TARGET_IDS | R5 同族实现细节开关 | off | 已认证 | 随 R5 同批焊死 |
| CANON_VLLM_ENABLE_PREFIX_CACHING | Phase3 APC:仅改变 A rollout 的 vLLM prefix-cache 读取路径；B rescore 继续固定 `reset_prefix_cache=True` 全量重算 | off；缺省/空/0 均关，仅 1 开 | 试验；Qwen3-8B DP1×TP4 G-A/G-B/G-C/G-D、脏页阴性与匹配性能/XProf 已绿，G-E/64 卡未验 | G-E 与用户主跑的 multi-turn canary 全绿后逐项转正；单轮负载不强制 |
| CANON_PALLAS_{CANONICAL_VJP,ALL_PROJ,ALL_RMSNORM,MPAD,SWIGLU,SWIGLU_MPAD} | canonical Pallas 内核族选通 | off | 已认证 | 转正焊死(P22.XI 部分已无条件) |
| CANON_P28_SEGMENTED_TRAIN | 分段 fixed-M 训练前向 | off | 已认证 | 转正焊死 |
| CANON_P59_RANK_PARALLEL_BACKWARD | 每个 trajectory group 的 DP rank-local VJP 从 host 逐-rank 串行改为一次手动 `shard_map`;TP1 保留 DP-manual/unit-TP carrier，TP>1 在同一物理设备上改用 engine `data/model` 二轴词汇并令 DP+TP 均 manual，使 inner engine shard_map 复用已绑定 TP collective；processed-logprob VJP 产出的 full logical-vocab cotangent 在 head VJP 入口显式约束为 `P(data,model)`，随后 fixed-head 只消费 TP-local vocab；projection 与 attention 的 P59-local 边界均只由精确的双 manual-axis context 选择，RPA 不再二次扩展已经 TP-local 的 GQA K/V；replicated-input TP hidden cotangent以 FP32、升序 rank、逐项 operand barrier 累加后只在边界 cast 一次；leading-DP 暂存后仍走原 fixed reduction，group 顺序不变 | off；仅显式 V1/P58.7 high-performance full profile 开 | **gradient-correctness KEEP / DP4 PERF KEEP / Attempt-3 repair one-host mechanism PASS / exact-image+target pending**:ordinary-JAX FP64 oracle relL2 `3.91e-16`，真实 Qwen 梯度 relL2 `1.582%` 过冻结梯度门，DP4 reverse 3.605x；串行与并行 AdamW 首步 delta relL2 `9.976%` 是已披露 trajectory difference。Attempt 3 的 GSM8K `g64m` 与 P45 `f45m` 均在 step-0 strict pre-alignment 逐字节全同且 0 FAIL，随后分别在 TP4/TP8 证明 attention 入口重复扩展已 local 的 KV；追加 patch 25 只在精确 P59 manual DP×TP context 跳过该扩展并强校验 local Q/K/V/cache。M15 `m15m` 在更早的独立 token-contract 门停止，不构成 P59 数值判决。host V1 21/21、P57 144/144、P59 34/34、APC 31/31、flags 366/366 通过；真实 v5p `DP2xTP2` RPA forward+VJP2、wrong-cache negative 与普通 `DP1xTP4` GQA control 通过且零 optimizer commit。installed-attention DP2×TP4/TP8 pinned-image 正负控与真实 DP16×TP4/DP8×TP8 optimizer commit 仍未验证 | Phase4 三个 full target 与 P58.7 full 归档后按 workload 转正；任一 real ALIGN FAIL 立即退役；全局默认仍 off |
| CANON_P59_DP4_SERIAL_MESH_BRIDGE / CANON_P61_BACKWARD_NUMERICAL_DIR | P59/P61 DP4 代理与 full-tree 数值载具，不是生产 recipe | off/空 | 载具完成；历史 serial/update 差异永久保留 | P59/P61 证据交付后退役载具，生产 profile 禁止开启 |
| CANON_OPT_STATE_RESIDENT / CANON_P30_OPT_STATE_OFFLOAD | 优化器驻留/卸载 | resident=生产默认 | 默认开 | resident 焊死后 OFFLOAD 降级为逃生开关保留 |
| CANON_KV_UNIFIED | U 臂读路径统一实验 | off | **否决区**:生产红(43→9 仍 0.28),非修复 | 可删,判决记录永存 |
| MIN_TOKEN_BUCKET / max_num_batched_tokens(非 CANON 但同级) | R2:全局/每 rank 桶契约 | 钉死 256 族 | 已认证 | 永不自由化;新几何走契约注册 |

## B 层 · perf/仪器类(observational;发布协议 = 每负载 =verify 首步绿 → =1)

| Flag | 语义 | 生命周期 | 日落 |
|---|---|---|---|
| CANON_PERF_LOG | [PERF] 分段计时(=0 静音) | 默认开 | 长期保留(观测契约) |
| CANON_BATCHED_EVIDENCE | 证据取回批量化(-6s/步) | GSM8K 默认;FL/DS 待 verify | 全负载转正后焊死 |
| CANON_P28_BATCHED_REPORT(=1/=verify) | report 窗合并+remap jit 化(FL -14.5%) | GSM8K 默认;DP16 待验 | 同上 |
| CANON_P28_BATCHED_REVERSE(=1/=verify) | P52 反向脚手架合并(-13.3%) | 一宿主认证;DP16 等 grouped 移植 | 同上 |
| CANON_P28_LAYER_SCAN | =verify 恒等仪器/=verify_rev THIRDPROG 演示 | **=1 否决(净负 -5%)** | 仪器保留;=1 进否决区 |
| CANON_CONTINUE_DECODE(=K) | 设备内 `lax.while_loop` 连续 decode，摊薄逐 token host 往返；async scheduling 必须关 | off；V1/P58.7 profiles 固定 K=8；一宿主 r20c/r20y KEEP；DeepSWE reader 已接线但 target 未跑 | Phase4 三个 full recipe 与 P58.7 各自 target 绿后按 workload 转正；新 K/尾桶重认证 |
| CANON_FIXED_AR_GATHER | fixed TP reduction 的三轮 ppermute 传输换成一次 all-gather，本地仍按相同 rank 顺序相加 | off；一宿主 r11 KEEP；P58.7 target 未跑 | DP16/DP8 full strict 绿后按 workload 转正 |
| CANON_PALLAS_GATHERED_LOGPROBS | Pallas scorer 片上直接产 selected logprob/top1/rank，避免全词表 logprob 物化；Phase4 新增 DP8/DP16 row-sharding 与每-rank M256 padding | off；一宿主 r10 KEEP；DP port IMPLEMENTED/TARGET NOT RUN，含 P58.7 Qwen3-4B | Phase4 三个 full 与 P58.7 target 的 exact gate/XProf 绿后转正；任一形状红回到 materialized scorer |
| CANON_LOGPROB_STEP_FUSION | decode logprob 的 slice/pad/gather/slice 胶水收进一个 jitted program，值链不变 | off；一宿主 r15 KEEP；P58.7 target 未跑 | Phase4 三个 full 与 P58.7 target 绿后按 workload 转正 |
| CANON_FUSED_TREE_OPS / CANON_PALLAS_NORM_MATMUL / CANON_PALLAS_INPUT_FUSION | P56 默认-off 候选实现；V1.1 不启用：P59 已取代主要 host-glue 靶，norm/input fusion 未进入最终 serving 配方 | off；保留历史 KEEP/边际/未转正事实；V1 profile 明确为 0 | V1 full 完成后按 P56 判决裁撤或另立新证据重开 |
| CANON_SAMPLE_SPLIT_FUSION / CANON_ENGINE_LOGPROB_READBACK / CANON_ANCHOR_OVERLAP / CANON_GSM8K_VANILLA | P56 中性、被取代或仅对标/载具开关，不属于 V1 默认配方 | off；不进入三个 full recipe | 战役归档后退役；禁止借 V1 profile 开启 |
| CANON_P3_APC_BOUNDARY_REPORT | Phase3 G-A 固定 token deep-prefix 边界报告路径;有值才运行 cache-hit prefill vs B full-reset 的前向探针 | 试验,缺省空/off | P3.1 结束后退役;证据保留 |
| CANON_XPROF_DIR/_SKIP_STEPS/_STEPS | XProf+perfetto 捕获(一次出双产物) | 仪器 | 长期保留 |
| CANON_PERF_TRACE_DIR / CANON_PERF_TRACE_EXPORT_STEP | 官方 tunix.perf v2 语义时间线导出目录与零起点单步窗口；V1 只序列化 warmed step 2，避免 full train 每步写盘，空目录仍为 NoopTracer 零开销 | 仪器；V1 full 固定 step=2 | 长期保留(官方 Metrics 契约) |
| CANON_XPROF_PYTHON_TRACER/_HOST_TRACER | tracer 档位;**python=0 是 device plane 的前提**(开着它训练捕获退化为 host-only) | 仪器;载具默认 python=0 | 长期保留 |
| CANON_XPROF_TPU_TRACE_MODE | update 窗 TPU trace 密度；V1 full 固定 `TRACE_COMPUTE` 以降低超 2GB drop 风险 | 仪器；仅 `phase=update` 接受 | 三个 target XProf 无 drop 后决定是否保留默认 |
| CANON_XPROF_LABELS | 为 rollout model/logits/sample 与 trainer fwd/bwd/report JIT 写语义名称，不改数值 | 仪器；V1 full 开；P56 r21/r22 已认证 | XProf 原生提供等价稳定命名后退役 |
| CANON_P59_XPROF_BACKWARD_DIR / CANON_P59_DP4_TAIL8 / CANON_P60_DETERMINISTIC_AB | P59/P60 DP4 专用 profile、tail 与跨臂载具 | off | 历史载具完成，不进入 V1 full | 证据交付后退役 |
| CANON_V1_GSM8K_XPROF_ARM | `native|zero-hp` 的 one-host GSM8K matched-work/XProf 观测 selector；固定 DP4×TP1、3 commits、update 1→2 capture，打印 profiled batch token/advantage hashes；Native 必须 vanilla stock trainer，Zero-HP 必须 strict V1/P59 bundle | 空/off；仅两条薄 wrapper 设置 | pair XProf 归档后退役；不得进入 full recipe |
| CANON_XPROF_PHASE | 捕获窗模式:step=整步(device 缓冲 ~283 万事件/核,decode ~25s 填满,实为 engine 前 25s 织物)/ update=G6 update 入口→步完成(rollout 不入镜,缓冲装下完整 backward)/ diagnostic=冻结权重 precheck 的一个完整 A-rollout/B-full-rescore/C-old-forward round | 仪器;载具旋钮 P51_XPROF_PHASE;Phase3 profile 固定 diagnostic skip=1 steps=1 | 长期保留 |
| CANON_UPDATE_REPORT / CANON_PRE_ALIGN_REPORT / CANON_ALIGN_REPORT | 对齐/更新报告选通 | 默认开(监控契约) | 长期保留;A−B 哨兵不可撤(用户裁决 2026-08-15) |
| JAX_COMPILATION_CACHE_DIR(非 CANON) | 持久编译缓存(-72s/重启) | 一宿主认证；Phase4 三个 full manifest 已锁定本地目录与 GCS root，restore/save 回执 host 绿；**Pathways target hit 未验** | 三个 full target 记录 hit/miss 与 JIT 后决定是否推广 |

## C 层 · P38 诊断家族(~30 个 CANON_P38_*)

| 组 | 代表 | 日落条件(一条覆盖全家) |
|---|---|---|
| capture/journal/ledger/capsule/GCS/replay | SERVING_CAPTURE_*、REQUEST_JOURNAL、INCIDENT_LEDGER、MISMATCH_CAPSULE、GCS_PREFIX、DURABILITY_PROFILE、FROZENLAKE_REPLAY、PRECHECK_ONLY… | **carrier 结案(strict 复验全绿)→ 全家整体退役**;判决类结论(U 臂、slot bug、co-batch、shape-1)迁 FOOTGUNS 后删 |
| Phase3 APC dirty-page negative | CANON_P3_APC_DIRTY_PAGE:布尔诊断旗标;缺省/空/0 不污染,仅 boundary dirty mode 的 writer 向直接 JAX/vLLM reader 透传 1;污染 A 确定会复用的 layer-0 单个真实 KV page,B 仍 full reset | G-D `p3gd1` 已命中;为 G-E 复验保留默认-off 载具,Phase3 最终结案 CL 退役 |

## D 层 · 发射/基建管道(~230,按前缀组;逐条语义允许"待考古")

| 前缀组 | 用途 | 处置 |
|---|---|---|
| CANON_RUN_* / STATE / PKG / PROFILE* / SHIM_ROOT / MODE | 发射管道(渲染/安装/运行合同) | phase2 三层 profile 落地时逐条核对归位 |
| CANON_GCS_CACHE_BUCKET | JAX persistent-cache 的 GCS root；Phase4 三个 full 固定到 P33 cache root，按 resolved profile 分 namespace，restore/save 失败只生成显式性能回执，绝不替代或放宽 Zero-TIM 数值门 | 基建性能合同；三个 target 均完成可审计 cache hit/miss 与 JIT 记账后决定默认范围 |
| CANON_V1_HP_FULL | workload-level execution identity；仅 Phase4 三个 renderer 与 P58.7 Zero-full renderer 设 1，并由各 workload profile 派生完整 serving/trainer/XProf bundle | 试验、默认 0；Phase4 GSM8K/P45/M15 与 P58 Zero/1000 是四种闭集；四个 full 归档并逐 workload 转正后退役此 campaign selector |
| CANON_WANDB_* | 观测账号面(用户所有,凭据纪律) | 保留,不动 |
| CANON_QWEN3_*(8 个几何) | 模型几何契约 | 保留;属 workload profile 层 |
| CANON_P3x/P4x_*_ADMITTED / NO_COMMIT / RUN_STAGE / 工作负载选通 | 各任务 admission 门 | 任务结案随任务退役(C 层同规) |
| CANON_P46_CENSUS_FIRST_PASS | P46 reward-only full campaign 的 breadth-first 调度：每个尚无 durable attempt 的 identity 只跑一次，invalid 留证后延后；不进入采样 fingerprint，也不放宽 strict finalizer | 试验、默认关；P46 完成一次 exact 1851 x N16 strict campaign 后退役 |
| CANON_P46_FROZEN_V6_IMPORT_ID | 显式选择已封存的 v6 resume snapshot；只允许在新 resume tag 内迁移原始轨迹与 sampler provenance，并逐条留迁移来源 | 迁移期、默认空；旧 campaign 全部升级至当前 harness 后退役 |
| CANON_P57_TIM_ARM / RUN_KIND / INFERENCE_REGIME / EXPECTED_UPDATES / STOP_AFTER_STEP / EVALUATION / EVAL_* / WORKLOAD_CANDIDATE / DATA_SPLIT / CALIBRATION_* | P57 FrozenLake TIM 因果实验身份与耐久产物；`TIM_ARM=mismatch` 为 native/no-IS，`is` 为相同 native 数值程序加 token TIS，`zero` 为完整 zero-TIM/no-IS；两 paired workload 为原始 P45/300（candidate/split 均空）与 materialized M15-main/300；主训练强制 `CANON_P33_ENABLE_EVAL=1,CANON_P31_ENABLE_EVAL=1`，在同一 JobSet 生成 `0,50,...,300` rollout-only held-out 曲线；native arms 必须 `INFERENCE_REGIME=stock-fast`，zero 禁止该 override；P57.1 calibration/selection 仍只接受 mismatch、M15/200 且无 in-process eval；active 300-step primary 的 `STOP_AFTER_STEP` 必须等于 300，历史 selection 才允许 horizon 内 50-step 边界；`EVALUATION=1` 仅保留为 step-0/final recovery audit，必须绑定显式 checkpoint provenance，不是主曲线 | 试验、默认空/关；P57 完成并归档最终因果报告后整体退役 |
| CANON_FROZENLAKE_CKPT_INTERVAL / CANON_FROZENLAKE_CKPT_MAX_TO_KEEP | FrozenLake 保存频率与滚动保留数；active P57 P45/300 与 M15-main/300 的 train/eval 三臂固定 `300/1`，只保存最终 actor+optimizer；legacy P45、calibration、M15-selection/200 保持 `10/1`。profile、resolved-env、Python parser、renderer 四层必须一致 | 基建合同；P57 primary 完成后再决定是否将 final-only 推广到通用 P45 |
| CANON_FROZENLAKE_CKPT_MILESTONE_INTERVAL | FrozenLake 额外证据 checkpoint 保留间隔；P57 active 300-step arms 强制 `0`，只保留 `LatestN(1)`，因为七点主曲线在训练内生成；历史 `50` 仅用于已归档的旧 isolated-eval 设计，不能重新用于 active P57 | 试验、默认 `0`；旧 milestone evidence 清理完成后退役正值路径 |
| CANON_P58_DEEPSWE_TIM / TIM_ADMITTED / TIM_ARM / EXPECTED_UPDATES / DEBUG_DIR / NATIVE_STOCK_PROMPT_OBSERVER / ONEHOST_XPROF_ARM / one-host provenance 族 | P58 Qwen3-4B-Instruct native-vs-zero 因果训练身份；固定 128-chip synchronous disaggregated、B8xG16、16K、compact filter、TPU optimizer 与完整 trajectory journal；`TIM_ARM=native|zero` 选择 numerical runtime。Native 的 sampler recipe 另由既有 `CANON_P34_DISABLE_SAMPLER_IS:CANON_P34_DISABLE_TIS` 闭集选择：`1:1`=raw，`0:0`=token TIS(threshold 2.0)，混合 tuple 拒绝；Zero/Zero-HP 必须 `1:1`。Native-IS 是 mitigation arm，不改变原生 numerical program，且 group filter 仍关；Native 保留完整 stock serving/trainer program，所有 shape-valid finite A/B/T_old/T_current mismatch 只观测，Zero 全边界 exact；`NATIVE_STOCK_PROMPT_OBSERVER=1` 只为 native arm 的 rollout 后 B 观察值提供 processed prompt logprobs，不进入采样、trainer、loss、反向或 optimizer，且与 canonical `PROMPT_PROCESSED_LOGPROBS` 互斥；`ONEHOST_XPROF_ARM=native|zero-hp` 仅准入 DP1xTP4、两次相同输入、固定 `[-1,1]` 诊断 cotangent、零 optimizer commit 的 update-profile 载具，缺省空，不能认证 DP8xTP8/P59/4B-TP8 fixed-head 或生产轨迹；`EXPECT_HOSTNAME/MODEL_SNAPSHOT/R2EGYM_COMMIT/TASK_IMAGE_ID/RUNNER_SHA256/SOURCE_DIFF_SHA256` 是该载具的字符串/路径 provenance receipts，缺省空且不改变数值 | 试验、默认关；one-host matched XProf package 归档后先退役 one-host selector/provenance 族，P58 production 族在完整 campaign 归档后整体退役 |
| CANON_P59_GCS_PREFIX / CANON_P59_INNER_RUN_CMD / CANON_P59_KIND / CANON_P59_REQUIRE_XPROF | P59 单次载具的证据目的地、冻结内层命令、臂身份与 XProf 完整性要求 | 试验；仅 P59 renderer/one-host wrapper 设置 | P59 证据载具归档后整体退役 |
| CANON_ALIGN*/EXPECT_*/DP_SIZE/TP_SIZE/TRAJECTORIES 族 | 对齐门与拓扑断言 | 监控契约,长期保留 |

## MARKERS(日志 marker 契约,非开关;~60 个)

关键项:`[CANON_ALIGN_PRE]`(四边界判决行)、`[CANON_ALIGN_PRE_JSON/EVIDENCE]`、
`[PERF]`、`[CANON_P38] DIAGNOSTIC_COVERAGE_CONTRACT / PRECHECK_COMPLETE`、
`[CANON_P38_SERVING_CAPTURE*]`、PATHTRACE 族(固定树行数 =2×层+1)。
Marker 是观测契约:改名/删除 = 破坏 postflight 与历史可比性,按合同类文档对待。

## 否决与退役区(只增不删)

| 项 | 判决 | 出处 |
|---|---|---|
| 优化3(rescore-B 降频) | 用户永久否决 | 2026-08-12 |
| CANON_P28_LAYER_SCAN=1 | 净负 -5%,判死 | P49 消融 |
| CANON_KV_UNIFIED 作为修复 | 生产红,非修复 | p38u1 |
| C3 延迟写回 | 净零 +4.5±4.1s(PCIe 争用),机制留备胎默认关 | P49 |
| truncated-cache backward | 丢 97.5% Wv,结构错 | R4 裁决 |

## Appendix — machine-generated full inventory (basis a94d6c0c, count must equal census)

```
CANON_ALIGNMENT_EXPECTED_RED
CANON_ALIGNMENT_GATE
CANON_ALIGNMENT_GATE_ONLY
CANON_ALIGNMENT_TRAIN
CANON_ALIGNMENT_UPDATE_CANARY
CANON_ALIGN
CANON_ALIGN_PRE
CANON_ALIGN_REPORT
CANON_ANCHOR_OVERLAP
CANON_BATCHED_EVIDENCE
CANON_CANONICAL_DEPTHS
CANON_CHECKPOINT_CONTRACT_JSON
CANON_CLIENT_IMAGE
CANON_CLUSTER
CANON_CONTINUE_DECODE
CANON_CUT
CANON_DEEPSWE_ALIGNMENT_WARN_ONLY
CANON_DEEPSWE_CLEANUP_TIMEOUT_SECS
CANON_DEEPSWE_ONEHOST_DEBUG_DIR
CANON_DEEPSWE_ONEHOST_NO_COMMIT
CANON_DEEPSWE_ONEHOST_REPORT
CANON_DEEPSWE_ONEHOST_ROLLOUT_ONLY
CANON_DEEPSWE_ONEHOST_SMOKE
CANON_DEEPSWE_ONEHOST_STAGE
CANON_DEEPSWE_ONEHOST_TASK_IMAGE
CANON_DEEPSWE_PER_TURN_TIMEOUT_SECS
CANON_DEEPSWE_REWARD_TIMEOUT_SECS
CANON_DEEPSWE_ROLLOUT_BATCH_TIMEOUT_SECS
CANON_DEEPSWE_STEP_TIMEOUT_SECS
CANON_DEEPSWE_TRAJECTORY_TIMEOUT_SECS
CANON_DP_PROBE_LOCAL_SAMPLES
CANON_DP_SIZE
CANON_ENGINE_DP_SIZE
CANON_ENGINE_LOGPROB_READBACK
CANON_ENGINE_MODULE_C
CANON_ENV
CANON_EXPECTED_SLICE_DEVICES
CANON_EXPECT_COMMIT
CANON_EXPECT_JAX_VERSION
CANON_EXPECT_MODEL_MESH_IDS
CANON_EXPECT_PATHWAYS_RELEASE
CANON_EXPECT_TRAIN_MESH_IDS
CANON_EXPECT_VISIBLE_DEVICES
CANON_FIXED_AR
CANON_FIXED_AR_EMBED
CANON_FIXED_AR_GATHER
CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY
CANON_FROZENLAKE_C0
CANON_FROZENLAKE_CKPT_INTERVAL
CANON_FROZENLAKE_CKPT_MAX_TO_KEEP
CANON_FROZENLAKE_CKPT_MILESTONE_INTERVAL
CANON_FROZENLAKE_CKPT_MODE
CANON_FROZENLAKE_CKPT_ROOT
CANON_FROZENLAKE_CKPT_TAG
CANON_FROZENLAKE_GRAD_PROBE
CANON_FROZENLAKE_L3
CANON_FROZENLAKE_P27
CANON_FROZENLAKE_RELEASE_GRAD_PROBE
CANON_GCS_CACHE_BUCKET
CANON_GLOBAL_PROMPTS
CANON_GLOBAL_TRAJECTORIES
CANON_FUSED_TREE_OPS
CANON_GSM8K_ACTIVE
CANON_GSM8K_AB_REPORT_ONLY
CANON_GSM8K_ALIGNMENT_WARN_ONLY
CANON_GSM8K_GRAD_PROBE
CANON_GSM8K_L3
CANON_GSM8K_TRAIN
CANON_GSM8K_UPDATE_CANARY
CANON_GSM8K_VANILLA
CANON_IN_CONTAINER
CANON_KV_PACKING
CANON_KV_UNIFIED
CANON_L3_A3_DIAG
CANON_LOCAL_PROMPTS
CANON_LOCAL_TRAJECTORIES
CANON_LOGPROB_M
CANON_LOGPROB_STEP_FUSION
CANON_MAX_BATCHED
CANON_MESH_SHAPE
CANON_MINREPRO_N
CANON_MM_ALGO
CANON_MM_ALGO_PRESET
CANON_MODE
CANON_MODEL_DIR_NAME
CANON_NUM_GENERATIONS
CANON_N_LAYERS
CANON_OPT_STATE_RESIDENT
CANON_OUT
CANON_OUT_BYTES
CANON_P28_BATCHED_REPORT
CANON_P28_BATCHED_REVERSE
CANON_P28_G4_BLOCK_CAP_SECONDS
CANON_P28_G4_EXTENSION
CANON_P28_G5C_ONLY
CANON_P28_G5_FIRST_CAP_SECONDS
CANON_P28_G5_ONLY
CANON_P28_G5_REPEAT_CAP_SECONDS
CANON_P28_G5_TOTAL_CAP_SECONDS
CANON_P28_G6_UPDATE
CANON_P28_LAYER_SCAN
CANON_P28_SEGMENTED_FORWARD
CANON_P28_SEGMENTED_PULLBACK
CANON_P28_SEGMENTED_TRAIN
CANON_P28_SEGMENTED_VJP
CANON_P29_FULL_TRAIN
CANON_P30_DONATE_MODEL
CANON_P30_FUSED_PAIR_ACCUMULATION
CANON_P30_OPT_STATE_OFFLOAD
CANON_P30_POST_COMMIT_GC
CANON_P30_RELEASE_CAPTURED_STATE
CANON_P30_RESHARD_ACCUMULATOR
CANON_P30_REUSE_SEGMENTED_ENGINE
CANON_P30_SHARDING_PROFILE
CANON_P30_SPARSE_GRAD_ASSEMBLY
CANON_P31_METRICS
CANON_P31_CONVERGENCE
CANON_P31_ENABLE_EVAL
CANON_P31_MONOTONIC_METRICS
CANON_P32_CHECKPOINT_DIR
CANON_P32_DP16_SEGMENTED
CANON_P32_DP_ADMISSION
CANON_P32_DP_REDUCTION_ADMITTED
CANON_P32_MODEL_INIT_ONLY
CANON_P32_MODEL_STATE_KIND
CANON_P32_OPTIMIZER_MEMORY_KIND
CANON_P32_RC
CANON_P32_RC_STAGE
CANON_P32_TRAIN_ADMITTED
CANON_P32_WORKLOAD
CANON_P33_DISABLE_EVAL
CANON_P33_DP
CANON_P33_DP4
CANON_P33_ENABLE_EVAL
CANON_P33_EVAL
CANON_P33_NO_COMMIT
CANON_P33_RUN_STAGE
CANON_P33_SHARED_MESH
CANON_P33_SHORT_ALIGNMENT
CANON_P33_WANDB
CANON_P33_WORKLOAD_LAUNCH_ADMITTED
CANON_P34_ABCPROD
CANON_P34_CLEAN_ROWS
CANON_P34_DATASET_NAME
CANON_P34_DATASET_REVISION
CANON_P34_DATASET_ROWS
CANON_P34_DATASET_SPLIT
CANON_P34_DEEPSWE
CANON_P34_DISABLE_SAMPLER_IS
CANON_P34_DISABLE_TIS
CANON_P34_MAX_BATCHED_TOKENS
CANON_P34_MAX_NUM_SEQS
CANON_P34_NO_COMMIT
CANON_P34_PREFIX_CACHE
CANON_P34_RUN_STAGE
CANON_P34_STRICT_CLI
CANON_P34_TOPOLOGY_ADMITTED
CANON_P34_TP8_ADMITTED
CANON_P34_TRAJECTORY_ADMITTED
CANON_P34_TRAJECTORY_CAPTURE
CANON_P34_UPDATE_ADMITTED
CANON_P34_WEIGHT_REPORT
CANON_P35_ARM
CANON_P35_CLASSIFICATION
CANON_P35_ENVELOPE
CANON_P35_ENVELOPE_REPORT
CANON_P35_EXACT_REPLAY
CANON_P35_EXACT_REPLAY_CLASSIFICATION
CANON_P35_EXACT_REPLAY_REPORT
CANON_P35_METADATA_DIR
CANON_P35_PRE_REPLAY_REPORT
CANON_P35_REPLAY_STAGE_CLASSIFICATION
CANON_P35_REPLAY_STAGE_PROBE
CANON_P35_REPLAY_STAGE_REPORT
CANON_P38_AVAL_REPORT
CANON_P38
CANON_P38_CONTROLLED_EXIT
CANON_P38_DIAGNOSTIC_ROUNDS
CANON_P38_DIAGNOSTIC_ROUND_FILE
CANON_P38_DURABILITY_PROFILE
CANON_P38_EXPECTED_POLICY_VERSION
CANON_P38_FROZENLAKE_REPLAY
CANON_P38_FIXED_LM_HEAD
CANON_P38_GCS_PREFIX
CANON_P38_INCIDENT_LEDGER
CANON_P38_INCIDENT_MAX_BYTES
CANON_P38_INCIDENT_MAX_PREFIX
CANON_P38_INCIDENT_MIN_PREFIX
CANON_P38_KV_OBSERVER_CLASSIFICATION
CANON_P38_KV_OBSERVER_DIR
CANON_P38_KV_OBSERVER_MAX_BYTES
CANON_P38_KV_OBSERVER_MAX_CANDIDATES
CANON_P38_KV_OBSERVER_MAX_PAGES
CANON_P38_KV_OBSERVER_MAX_READ_BYTES
CANON_P38_LIVE_COLLECT_ACK_FILE
CANON_P38_LIVE_COLLECT_REQUEST_FILE
CANON_P38_LIVE_COMPLETE_ACK_FILE
CANON_P38_LIVE_COMPLETE_REQUEST_FILE
CANON_P38_LIVE_INCLUDE_OBSERVER
CANON_P38_LIVE_SNAPSHOT_INTERVAL_SECONDS
CANON_P38_LIVE_SNAPSHOT_STOP_FILE
CANON_P38_LIVE_SNAPSHOT_WORKER_LOG
CANON_P38_MIN_ACTION_KV
CANON_P38_MISMATCH_CAPSULE
CANON_P38_MISMATCH_CAPSULE_MAX_ROWS
CANON_P38_ONEHOST_REHEARSAL
CANON_P38_PRECHECK_ONLY
CANON_P38_REQUEST_JOURNAL
CANON_P38_ROUND_SEAL_ACK_DIR
CANON_P38_ROUND_SEAL_REQUEST_DIR
CANON_P38_SEAM_CLASSIFICATION
CANON_P38_SEAM_LAYER
CANON_P38_SEAM_MAX_BYTES
CANON_P38_SEAM_MAX_POSITION
CANON_P38_SEAM_MIN_POSITION
CANON_P38_SEAM_OBSERVER
CANON_P38_SEAM_OBSERVER_DIR
CANON_P38_SERVING_CAPTURE_ARCHIVE
CANON_P38_SERVING_CAPTURE_CLASSIFICATION
CANON_P38_SERVING_CAPTURE_DIR
CANON_P38_SERVING_CAPTURE_EXPECTED_PATH
CANON_P38_SERVING_CAPTURE_EXPECTED_RECORDS
CANON_P38_SERVING_CAPTURE_FREE_SPACE_MULTIPLIER
CANON_P38_SERVING_CAPTURE_MAX_CALLS
CANON_P38_SERVING_CAPTURE_MIN_PREFIX
CANON_P38_SERVING_CAPTURE_PREFIX_BOUNDS
CANON_P38_TAIL_MAX_BYTES
CANON_P38_TAIL_OBSERVER
CANON_P38_TERMINAL_CLASSIFICATION
CANON_P38_TERMINAL_DISCRIMINATOR
CANON_P38_TERMINAL_MAX_BYTES
CANON_P39_64CHIP_PILOT
CANON_P39_PILOT_ADMITTED
CANON_P3_APC_BOUNDARY_REPORT
CANON_P3_APC_DIRTY_PAGE
CANON_P41_OPTIMIZER_BENCH
CANON_P43_DEBUG_ADMITTED
CANON_P43_DEEPSWE_DEBUG
CANON_P43_ROLLOUT_ONLY
CANON_P44_DEEPSWE_PARITY
CANON_P44_PARITY_ADMITTED
CANON_P44_ROLLOUT_ONLY
CANON_P44_TOPOLOGY
CANON_P45_HOST_GC_INTERVAL
CANON_P45_HOST_MEMORY_TELEMETRY
CANON_P46_DEEPSWE_TRAIN
CANON_P46_EVALUATION
CANON_P46_EVALUATION_MODE
CANON_P46_CENSUS_FIRST_PASS
CANON_P46_FROZEN_V6_IMPORT_ID
CANON_P46_LOGICAL_SHARD_INDEX
CANON_P46_ONEHOST_PROBE
CANON_P46_PARITY_CANARY
CANON_P46_PHYSICAL_SHARD_INDEX
CANON_P46_RESUME_TAG
CANON_P46_TOPOLOGY
CANON_P57_DATA_SPLIT
CANON_P57_INFERENCE_REGIME
CANON_P57_CALIBRATION_MODE
CANON_P57_CALIBRATION_OUTPUT
CANON_P57_CALIBRATION_RECIPES
CANON_P57_EVALUATION
CANON_P57_EVAL_CHECKPOINT_STEP
CANON_P57_EVAL_OUTPUT
CANON_P57_EXPECTED_UPDATES
CANON_P57_RUN_KIND
CANON_P57_STOP_AFTER_STEP
CANON_P57_TIM_ARM
CANON_P57_WORKLOAD_CANDIDATE
CANON_P58_DEBUG_DIR
CANON_P58_DEEPSWE_TIM
CANON_P58_EXPECTED_UPDATES
CANON_P58_EXPECT_HOSTNAME
CANON_P58_MODEL_SNAPSHOT
CANON_P58_NATIVE_STOCK_PROMPT_OBSERVER
CANON_P58_ONEHOST_XPROF_ARM
CANON_P58_R2EGYM_COMMIT
CANON_P58_RUNNER_SHA256
CANON_P58_SOURCE_DIFF_SHA256
CANON_P58_TASK_IMAGE_ID
CANON_P58_TIM_ADMITTED
CANON_P58_TIM_ARM
CANON_P59_DP4_SERIAL_MESH_BRIDGE
CANON_P59_DP4_TAIL8
CANON_P59_GCS_PREFIX
CANON_P59_INNER_RUN_CMD
CANON_P59_KIND
CANON_P59_RANK_PARALLEL_BACKWARD
CANON_P59_REQUIRE_XPROF
CANON_P59_XPROF_BACKWARD_DIR
CANON_P60_DETERMINISTIC_AB
CANON_P61_BACKWARD_NUMERICAL_DIR
CANON_PALLAS_ALL_PROJ
CANON_PALLAS_ALL_RMSNORM
CANON_PALLAS_CANONICAL_VJP
CANON_PALLAS_LOGSOFTMAX
CANON_PALLAS_GATHERED_LOGPROBS
CANON_PALLAS_INPUT_FUSION
CANON_PALLAS_MATERIALIZE
CANON_PALLAS_MATMUL
CANON_PALLAS_MPAD
CANON_PALLAS_NORM_MATMUL
CANON_PALLAS_SWIGLU
CANON_PALLAS_SWIGLU_MPAD
CANON_PERF_LOG
CANON_PERF_TRACE_DIR
CANON_PERF_TRACE_EXPORT_STEP
CANON_PKG
CANON_POD_NAME
CANON_POSTRPA_M
CANON_PRE_ALIGN_GATE
CANON_PRE_ALIGN_REPORT
CANON_PROFILE
CANON_PROFILE_FILE
CANON_PROMPT_PROCESSED_LOGPROBS
CANON_QWEN3_HEAD_DIM
CANON_QWEN3_HIDDEN_SIZE
CANON_QWEN3_INTERMEDIATE_SIZE
CANON_QWEN3_NUM_ATTENTION_HEADS
CANON_QWEN3_NUM_KV_HEADS
CANON_QWEN3_TP_SIZE
CANON_R2EGYM_COMMIT
CANON_R2EGYM_INSTALL
CANON_REQUIRE_PATHWAYS
CANON_REQUIRE_TRAIN_MESH_PIN
CANON_RPA_D
CANON_RPA_M
CANON_RPA_P
CANON_RPA_VJP2
CANON_RUN_CMD
CANON_RUN_CWD
CANON_RUN_ID
CANON_RUN_LOG
CANON_RUN_P38_AVAL
CANON_RUN_T2_DP
CANON_SAMPLE_SPLIT_FUSION
CANON_SHIM_ROOT
CANON_SITES
CANON_SOURCE_BRANCH
CANON_STATE
CANON_STATE_REPORT
CANON_T1_LOG
CANON_TAIL
CANON_TARGET_M
CANON_TOTAL_DEVICES
CANON_TPU_INFERENCE_PATH
CANON_TP_SIZE
CANON_TP_WIDTHS
CANON_TRAIN_DP_SHARDING
CANON_UPDATE_REPORT
CANON_V1_GSM8K_XPROF_ARM
CANON_V1_HP_FULL
CANON_VJP2_MAX_SEQS
CANON_VLLM_ENABLE_PREFIX_CACHING
CANON_WANDB_GROUP
CANON_WANDB_ONLINE_REQUIRED
CANON_WANDB_PROJECT
CANON_WANDB_RUN_NAME
CANON_WAYCOUNT_DEPTHS
CANON_WAYCOUNT_WIDTHS
CANON_XPROF_DIR
CANON_XPROF_HOST_TRACER
CANON_XPROF_LABELS
CANON_XPROF_PHASE
CANON_XPROF_PYTHON_TRACER
CANON_XPROF_SKIP_STEPS
CANON_XPROF_STEPS
CANON_XPROF_TPU_TRACE_MODE
```

Count: 368 settable names (appendix inventory above; exclusions: none).
