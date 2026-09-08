# FLAGS.md — CANON_* 注册表

> 政策:**建 flag 自由但必须登记(带日落条件);删 flag 有序按日落执行。**
> 焊死数值类 flag = 删代码路径 = 程序变更,走与开启同级认证门(verify+ALIGN+canary)。
> 生命周期档位:试验 → 已认证 → 默认开 → 焊死(开关可删)→ 退役/否决。
> 普查基点 a94d6c0c(285 个可设置 env flag,与 ebba4850 普查零漂移);普查后续现役附录
> 当前 423 个;本表分层登记,D 层按前缀组、语义欠账标"待考古"。
> 全量机器清单:落地 CL 时由 `grep -rhoE` 生成为附录,条目数必须 == 普查数(排除项列明)。

## A 层 · 数值语义类(动它 = 动程序身份;焊死走认证门)

| Flag | 语义 | 默认 | 生命周期 | 日落条件 |
|---|---|---|---|---|
| CANON_P57_TITO_ONEHOST_NEUTRALITY | T9d-3 observer-neutrality selector。`off|on` 两臂都复用既有 Qwen3-8B FrozenLake Perf-v2 DP1×TP4、三次真实 backward/AdamW、strict A=B=C 载具并开启 generic exact TiTO；`on` 唯一增量是 `record-full` host 证据，`off` 禁止该 debug selector。配对判决要求两臂逐 update gradient norm、alignment hashes、抽样模型/optimizer/accumulator fingerprints 与 12 条 strict alignment 数值完全相同，并同时命中 r7 三个 gradient-norm 与 forward implementation 锚点 | 默认 absent；只准 P45、DP1×TP4、3 updates、APC-off、no-eval/no-checkpoint、strict alignment 的本地 one-host gate；production profile、M15、DP8×TP8、其他 horizon、空值均 fatal。它不开放 actor snapshot 或 GCS 写入 | host/CPU construction in progress；one-host target 未跑，不得据此声明 observer neutrality | matched off/on target 绿且证据登记后退役该 selector；若旧锚点不符先判 RED/INCONCLUSIVE，不得静默重钉 |
| CANON_FIXED_AR | R1:TP 归约换固定序 ppermute 树 | off(canonical lane 开) | 已认证(1host+DP16/DP8/256) | 转正焊死:全负载默认开满一周期后无条件化 |
| CANON_FIXED_AR_EMBED | R3 补漏:vocab 分片 embedding gather 固定序 | off | 已认证 | 同上,与 FIXED_AR 同批 |
| CANON_RPA_VJP2(+VJP2_MAX_SEQS) | R4:cache-aware 认证反向 | off | 已认证(fp64 oracle+20/20+21/21) | 转正焊死;MAX_SEQS>1 需归约序审计先行 |
| CANON_LOGPROB_M / CANON_PROMPT_PROCESSED_LOGPROBS | R5:三臂共享 logprob callable,M=256 | off | 已认证(G9 4/4+负控) | 转正焊死 |
| CANON_MM_ALGO / CANON_MM_ALGO_PRESET | P19/P38:非 Pallas einsum 的 dot-algorithm 判别器;P38 仅允许 fixed `BF16_BF16_F32` 单变量臂 | off / BF16_BF16_F32 | **否决区**:旧 M16/M2048 e2e 无效;P38s22 三轮 A-B 仍红 | 可删,判决记录永存 |
| CANON_P38_FIXED_LM_HEAD | P38.2x/2y/2y1/2y2:registered Qwen3 output heads 的 M8/16/32/64/128/256 request buckets 均 pad 到 M256；通常 learner M4096 映射为 16xM256，只有 Qwen3-8B/TP8 FrozenLake 另登记 learner M2048→8xM256；untied `JaxLmHead` 与 tied `JaxEmbed.decode` 共用 Pallas body 和 endpoint-scoped receipts。几何为 1.7B K2048/TP4 tied、8B K4096/TP4 或 TP8 untied（TP4 N37984→38144；TP8 N18992→19200）、4B K2560/TP8 tied、32B K5120/TP8 untied（N18992→19200）；tiles 均 BM128/BN256/BK256 | off；仅显式 renderer opt-in | 历史 8B/TP4 backward-no-commit 与多模型 pinned-image construction 证据保留；首个 FrozenLake DP8×TP8 full attempt `f45g` 在 pre-backward C-forward 暴露 M2048 漏登记，当前 source/host 26/26 已修复并含旧 M4096 冒充的负控，post-fix pinned image 与 DP8×TP8 target 均未跑；one-host DP4×TP1 代理仍因未登记 TP1 fixed-head 几何而明确保持 off | 每个 model/TP/endpoint 独立 target 绿后逐项转正；任一红仅退役对应 registry entry |
| CANON_PROMPT_DIRECT_LOGPROBS / ABSOLUTE_TARGET_IDS | R5 同族实现细节开关 | off | 已认证 | 随 R5 同批焊死 |
| CANON_VLLM_ENABLE_PREFIX_CACHING | Phase3 APC:仅改变 A rollout 的 vLLM prefix-cache 读取路径；B rescore 继续固定 `reset_prefix_cache=True` 全量重算 | off；缺省/空/0 均关，仅 1 开；三个 production full recipes 当前统一 off | 试验；Qwen3-8B DP1×TP4 G-A/G-B/G-C/G-D、脏页阴性与匹配性能/XProf 已绿；M15 DP8×TP8 `m15i` G-E 在 A−B 红 1389 bytes/760 elements、B−C exact，故 target 修复前禁止 production APC | fresh target carrier 完成复现/首红定位/最小修复，随后对应 workload G-E 与脏页负控全绿后逐项转正；认证不可跨 workload 继承 |
| CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY | FrozenLake有限差异的可观测训练策略。历史 Native/IS full 继续保留 broad warning；优化 Zero 仅精确 P45（candidate/split 均 absent）或 M15/main v1-hp、DP8×TP8、300-update、no-eval/no-checkpoint concept run 可开，且只将有限 `S_decode_vs_S_prefill` 与由其直接派生的 `w`/`wr`/clip/TIS 差异降为 warning。`S_prefill_vs_T_old`、`T_old_vs_T_current`、`r`、任意 nonfinite、梯度、副本与 optimizer transaction 始终 fatal | off；renderer 对精确 P45/M15 Zero concept arms 写 1 | 临时收敛曲线逃生阀；host/exact-image admission 已覆盖两条 identity，DP8×TP8 target 未跑。开启后只能声明 `convergence-only / alignment-degraded`，不得声明 Zero-TIM | 分别修复 P45/M15 carrier 并完成 strict 300-update target 后恢复 0；失败与 warning 剂量证据永久保留 |
| CANON_DEEPSWE_ALIGNMENT_WARN_ONLY | DeepSWE有限差异的可观测训练策略。历史 P58 Native full 保留 broad serving/trainer warning；优化 Zero 只允许精确 Qwen3-4B-Instruct-2507、P58 Zero-HP/full/1,000-update、DP8×TP8 双角色、128 trajectories 的 production profile 开启，且只将有限 `S_decode_vs_S_prefill` 与由它直接派生的 `w`/`wr`/clip/TIS 差异降为 warning。`S_prefill_vs_T_old`、`T_old_vs_T_current`、`r`、任意 nonfinite、shape、gradient、replica 与 optimizer transaction 始终 fatal；precheck、checked-VMA、seam、one-host、ordinary Zero 都必须为 0/strict | off；renderer 仅对 Native 或精确 P58 Zero-HP production full 写 1 | P58.32 host policy/profile/classifier、P34 static、409/409 registry 与完整 digest-pinned image gates 通过；DP8×TP8 target 尚未跑。Zero-HP warning lane 只能声明 `convergence-only / alignment-degraded`，不得声明 Zero-TIM | 修复 DeepSWE decode/prefill carrier并完成同配置 strict target 后恢复 0；保留 warning 剂量和失败证据 |
| CANON_DEEPSWE_SYSTEM_OPTIMIZATION_ARM | v2 Phase 2 DeepSWE 严格 control/treatment 数值准入 selector。只准精确 P44 Qwen3-4B、three-update、DP4×TP8 或 DP8×TP8 strict profile；两臂共同启用 fixed head、rank-parallel checked-VMA、P59-only VMA scope、first-update gate 与 host-receipt bundle。`control` 保持 KEEP_TAPE/REDUCE_ONCE 真缺省，`treatment` 只增加 `KEEP_TAPE=stream` 与 `DP_REDUCE_ONCE=1`，因此明确暴露唯一会改变梯度归约顺序的刀 | absent/off；空值不构成 arm，renderer 仅在显式 `control|treatment` 时写入，既有六份 P44 render 逐字节语义不变 | 试验；renderer/profile/Python contract、负控、adjacent 与 pinned exact-image host 门已绿；DP4×TP8 与 DP8×TP8 strict target 两臂均未跑，不能从其他 workload/几何继承锚或性能结论 | P44 两几何完成 A=B=C、G4/D4/G5/G6、G1、逐臂 commit-norm/HBM/update 秒后，按 target 判决将 treatment 纳入 DeepSWE 默认包或否决并删除 selector；失败证据永久保留 |
| CANON_M15_TOKEN_CONTINUITY | 历史 M15 later-turn token continuity selector。`verify` 仅观察 serving 实际 prompt IDs；`exact` 为 later turns 直接提交 initial tail + sampled assistant IDs + nonterminal environment IDs，关闭 chat-template 重应用并逐 token 回验。签名身份保留三类：M15 APC debug DP8×TP8 layer/backward-no-commit、one-host DP1×TP4 verify/exact、以及旧 P67 M15/main full exact；malformed/missing/negative arrays hard-fail | 全局 absent/off。新 P67 renderer 不再写此 key；兼容 CLI `--m15-tito-exact` 映射到新的 P57 selector。P45 永远禁止此历史 key；与 `CANON_P57_TOKEN_CONTINUITY` 同时存在 fatal | 历史 exact-default `3fc7ef8b` 已撤出 production default。Legacy r7 APC-off 17/17 equal；matched exact r8 17/17 equal，三轮 strict 且与 r7 prompt/trajectory hashes 全同；E0v/E0w one-host APC-off/on exact pair 同样全零。这些均不继承到 DP8×TP8 | 历史 APC debug/one-host 证据与 classifier 退役后移除旧 selector 的可达写入；证据文本永久保留 |
| CANON_P57_TOKEN_CONTINUITY | FrozenLake P45/M15 later-turn exact token-in/token-out selector。合法值仅 `exact`：首轮保持 chat 输入；后续轮次直接提交 initial prompt tail + sampled assistant IDs + nonterminal environment IDs，并把 serving 实际消费的 prompt IDs 与整数 ledger 逐 token 回验。receipt 必须带 `workload=p45|m15`、fresh trajectory ID、request ID、policy step、group 与 sequence row；full classifier 对全部预期 row 做 identity join，并把未进入 later-turn 比较的单轮 trajectory 诚实记为 `UNEXERCISED`。selector 单独使用时任一 token 不等保持 fatal；仅与注册的 `record-full` 调试策略组合时，same-request-ID token red 可留证后让同一未改 row 继续训练，missing/duplicate/swapped/foreign identity 与 malformed token array 仍 fatal。只准精确 P57 Zero v1-hp、DP8×TP8、300-update、no-eval/no-checkpoint full 身份；不改变采样、reward、loss、backward 或 optimizer | 数值旗标，默认 absent/off；renderer 闭集 `legacy|p45-exact|m15-exact|both-exact` 按 workload 派生。空值、`0`、`verify`、Native/IS/eval/GSM8K/DeepSWE、错误 profile/topology/horizon 均 fatal。与历史 `CANON_M15_TOKEN_CONTINUITY` 同时存在 fatal | M15 shared implementation 有 DP1×TP4 r8 17/17 exact-equal、三轮 strict 证据；P45 与 P45/M15 DP8×TP8 target 均未跑，不能继承 M15 one-host 认证。T9c host 216/216、V1 102/102、flags 421/421 与完整 pinned-image construction 绿 | 两个显式 DP8×TP8 exact treatment分别完成 full horizon、完整 row/join/classifier/GCS 证据后，决定是否按 workload 默认化；token red 的 record-full arm 永远只算 `NON_ZERO_TIM_DATA_COLLECTION`；在此之前始终 explicit-only |
| CANON_P57_TOKEN_CONTINUITY_DEBUG | P45/M15 exact TiTO 调试策略。`first-diff` 只准 production full：每个 trajectory engine 至多写一份分段 capsule，随后保持 fatal。`collect-64` 只准独立 DP8×TP8 rollout-only 载具：差异结束并 mask 当前 trajectory，永不进入训练。`record-full` 只准显式 P45/M15 v1-hp 300-update full：每一个 same-ID token-diff event（submit-vs-engine echo 或 later-turn actual-vs-ledger，含 update 0、同轨迹重复事件和第 65 件以后）都写一份带全量 actual/expected token、segment、request/trajectory/policy/event join 的 mode-0600 capsule；不设 record-full 事件上限，同一普通 row 不 mask、不 drop、不 retry、不 reweight，继续原 GRPO。missing/duplicate/swapped/foreign request ID、漏件/重号/篡改 capsule 或写盘失败仍 fatal。single-turn 记 `UNEXERCISED`，不冒充 equal | 诊断旗标，默认 absent/off；三值都要求 generic exact selector。`collect-64` 要求专用 no-commit/no-eval/no-checkpoint/APC-off/P59-off profile；`record-full` 要求生产 DP8×TP8、300 updates、no-eval/no-checkpoint 和既有有限 A-B warning carrier。Native/IS/legacy/eval/GSM8K/DeepSWE、空值和 partial tuple 均 fatal。record-full 证据量随 diff 数和上下文长度增长，不能作为性能证据；任一 token red 或有限 A-B red 只能声明 `NON_ZERO_TIM_DATA_COLLECTION` | `first-diff`、`collect-64` 与 T9d `record-full` 已有 host/pinned-image construction；T9e all-event host 234/234、V1 102/102、flags 422/422 与完整 pinned-image construction 通过，one-host observer-neutrality、real GCS 与 DP8×TP8 target 均未跑 | P45/M15 drift 根因关闭且 capsule 归档后退役可达写入；历史 marker/证据永久保留 |
| CANON_P57_TITO_ROLLOUT_ONLY / CANON_P57_TITO_RUNNER_WITNESS_DIR / CANON_P57_TITO_GCS_{PREFIX,INTERVAL_SECONDS,READY,STOP_FILE,FINALIZE_FILE,FINAL_ACK,HEARTBEAT,WORKER_LOG} | T9b/T9c 数据载具合同：提交 token hash 与 `RequestOutput.prompt_token_ids` echo 按 request ID 绑定；rollout-only 另用 TPU runner 的 host-owned `input_batch.token_ids_cpu` 作第三方 witness，full-record 不声称逐请求 runner witness。可逆 token 只写 `$CANON_STATE` 下 mode-0600 capsule。后台 worker 先以非敏感 probe 完成 GCS no-clobber 上传、下载验 SHA 并写 READY；随后低 CPU/I/O 优先级运行，只对新完成的 immutable files 建 delta tar，瞬时传输失败有界退避并发布 heartbeat；正常完成再用所有 delta receipts 与完整本地 inventory 生成 final manifest | 默认 absent；只由 P45/M15 exact `collect-64` 专用 profile或 `record-full` v1-hp profile派生。prefix 按 JobSet/attempt 隔离；raw caller values、证据改写/消失/重复上传、坏 delta、final 缺项、readback SHA 差异均 fatal。live 瞬时失败不改变训练 row，但 final 无法证明完整时 `evidence_verdict=FAIL`。异常退出只保证最近成功 delta，最后同步窗口内新文件可能丢失 | fake-remote incremental/retry/tamper/heartbeat 与 T9b/T9c immutable-image construction 已绿；real GCS、one-host neutrality、DP8×TP8 full record 均未跑 | target 数据集归档并完成离线复刻后退役整组诊断路径；raw evidence 不进 Git |
| CANON_PALLAS_{CANONICAL_VJP,ALL_PROJ,ALL_RMSNORM,MPAD,SWIGLU,SWIGLU_MPAD} | canonical Pallas 内核族选通 | off | 已认证 | 转正焊死(P22.XI 部分已无条件) |
| CANON_P28_SEGMENTED_TRAIN | 分段 fixed-M 训练前向；默认 production clipping 继续使用 stock `optax.clip_by_global_norm`。Attempt-7 P62 no-commit 载具额外打印 element-finiteness、naive/max-scaled L2、DP/TP reduction 与 accumulator receipt；G5b 已证明 16/16 groups 与最终 accumulator 全 finite，旧 `norm=inf` 是 FP32 sum-of-squares overflow | off | 历史 segmented 路径已认证；P62 DP16×TP4 G5b target 仅认证 finite backward 与零 commit，未认证 optimizer transaction | 默认路径不变；仅精确 P63 full profiles 可启用 hybrid clip，首次真实 commit 与完整 horizon 仍是 target gate |
| CANON_P59_RANK_PARALLEL_BACKWARD | 每个 trajectory group 的 DP rank-local VJP 从 host 逐-rank 串行改为一次手动 `shard_map`;TP1 保留 DP-manual/unit-TP carrier，TP>1 在同一物理设备上改用 engine `data/model` 二轴词汇并令 DP+TP 均 manual，使 inner engine shard_map 复用已绑定 TP collective；processed-logprob VJP 产出的 full logical-vocab cotangent 在 head VJP 入口显式约束为 `P(data,model)`，随后 fixed-head 只消费 TP-local vocab；projection 与 attention 的 P59-local 边界均只由精确的双 manual-axis context 选择，RPA 不再二次扩展已经 TP-local 的 GQA K/V；replicated-input TP hidden cotangent以 FP32、升序 rank、逐项 operand barrier 累加后只在边界 cast 一次；leading-DP 暂存后仍走原 fixed reduction，group 顺序不变 | off；仅显式 V1/P58.7 high-performance full profile 开 | **gradient-correctness KEEP / DP4 PERF KEEP / Attempt-3 repair one-host mechanism PASS / exact-image+target pending**:ordinary-JAX FP64 oracle relL2 `3.91e-16`，真实 Qwen 梯度 relL2 `1.582%` 过冻结梯度门，DP4 reverse 3.605x；串行与并行 AdamW 首步 delta relL2 `9.976%` 是已披露 trajectory difference。Attempt 3 的 GSM8K `g64m` 与 P45 `f45m` 均在 step-0 strict pre-alignment 逐字节全同且 0 FAIL，随后分别在 TP4/TP8 证明 attention 入口重复扩展已 local 的 KV；追加 patch 25 只在精确 P59 manual DP×TP context 跳过该扩展并强校验 local Q/K/V/cache。M15 `m15m` 在更早的独立 token-contract 门停止，不构成 P59 数值判决。host V1 21/21、P57 144/144、P59 34/34、APC 31/31、flags 366/366 通过；真实 v5p `DP2xTP2` RPA forward+VJP2、wrong-cache negative 与普通 `DP1xTP4` GQA control 通过且零 optimizer commit。installed-attention DP2×TP4/TP8 pinned-image 正负控与真实 DP16×TP4/DP8×TP8 optimizer commit 仍未验证 | Phase4 三个 full target 与 P58.7 full 归档后按 workload 转正；任一 real ALIGN FAIL 立即退役；全局默认仍 off |
| CANON_P59_CHECKED_VMA | Production selector for the P66 checked-VMA repair. Registered Phase4 full contexts and the exact P58 Qwen3-4B Zero-HP full profile may set it; `00_env.sh` validates the closed workload/stage/arm geometry and derives the historical P66 implementation spelling as an internal compatibility alias. It changes only P59 backward VMA ownership and does not alter serving forward Zero-TIM or reduction order | off; registered full profiles only; P58 requires Zero/full/1,000 updates, DP8×TP8 and the complete HP bundle. P58.32 production may use narrow finite A-B warning admission, but diagnostics remain strict | P66 G1 causal PASS and G1.5 six-endpoint same-point ordinary-JAX oracle PASS; P58 construction validation in progress and target optimizer/convergence not run | each registered full horizon plus target receipts green, then fold into the final P59 production identity or retire per workload on any numerical red |
| CANON_P59_DP4_SERIAL_MESH_BRIDGE / CANON_P61_BACKWARD_NUMERICAL_DIR | P59/P61 full-tree 数值载具，不是生产 recipe。历史主臂为精确 DP4×TP1 one-update；v2 Phase 0 另准入精确 DP2×TP2 three-update deterministic rank-parallel checked-VMA full-train，只写 update 0 的完整 model-before/gradient/model-after 树，供 reduce-once off/on 同输入 G4 对照，update 1/2 不覆盖证据 | off/空；DP2 扩展只能由 one-host runner 的 outer-only `V2_P0_CAPTURE_FULL_TREE=1` 进入，默认 bundle 不变 | DP4 载具完成；DP2 扩展 host admission/manifest/comparator CPU 门绿，one-host G4 尚未运行；历史 serial/update 差异永久保留 | v2 Phase 0 G4 证据交付后收回 DP2 临时准入；生产 profile 永远禁止开启；P59/P61 其余证据交付后退役载具 |
| CANON_P66_BACKWARD_ARM / CANON_P66_BACKWARD_CAPTURE_DIR | P66 backward 诊断总开关：保留 DP4×TP1 `ordinary|segmented` 整树载具，并增加 one-host 完整 28 层 DP1×TP4 `tp4-serial|tp4-p59-old|tp4-p59|tp4-gather-off` 因果臂；G1.5 `tp4-vma-oracle` 在同一 checked-VMA candidate 之后旁路调用 ordinary serial pullback，比较 head/norm/layer27/14/0/embed 的参数、activation 与 cache cotangent，serial 结果绝不回灌；全部零 optimizer，TP4 臂只反传 group0，`grad_norm>1e6` 直接红停 | 空/off；仅 P66 wrapper 设置，生产 profile 禁止开启 | DP4×TP1 已跑；TP4 G1 verdict `H1_VMA_SUPPORTED`，P/R exact、U `1.5402e21` expected-red；final-source G1.5 host 16/16、pinned `2x37/37`、one-host 17/17 与六 endpoint/observer-neutrality 全 PASS；target 未跑 | G2 target 依赖闭包完成后退役，失败及分类证据永久保留 |
| CANON_P66_P59_CHECK_VMA | P66 结构/修复诊断：把 P59 外层 `shard_map` 从历史 `check_vma=False` 切到官方 VMA 复制一致性检查；checked 路径把 DP-local 参数显式标成 varying，并让 VMA 转置拥有 TP replicated-input `psum`，避免 fixed-head/projection 再手工归约一次 | 0/off；仅 P66 focused probe/`tp4-p59*|tp4-vma-oracle` 设置 | G0 pinned-image DP2×TP4/TP8 真实 shim全绿；完整 28 层 G1 one-host 修复臂 finite、距 serial norm `0.1112%`，固定 gather 被排除；G1.5 六 endpoint 最差 rel-L2 `0.5257%` 且 observer-neutrality exact；target 未跑 | P59 TP 复制/归约语义在 target capsule 门通过后，再决定替换生产默认或退役 |
| CANON_P67_P66_VMA_P59_ONLY | P67 serving 程序同一性修复：当 P66 checked-VMA 进程级 alias 开启时，只允许精确 P59 outer manual `data/model` pullback 消费 pcast/Pallas out-shape/RPA out-shape/embed invariant 登记；ordinary serving decode/prefill 保持历史图。它不关闭 P59 backward 修复，不改变数学值或 fixed TP reduction order，也不自行放宽 alignment gate | 0/off；`CANON_V1_FL_TP8_AB_ARM=serving-scope` 诊断，精确 P45-readiness/M15-main DP8×TP8 strict-zero 300-update FrozenLake V1 full profile，或精确 P58 Qwen3-4B Zero/full DP8×TP8 1,000-update HP profile可设 1；GSM8K、P58 Native/IS、非 HP Zero、Qwen3-32B 与其他 profile 禁止。P58.32 的 A-B warning 是独立、精确限定的 policy | host/exact-image gates通过；FrozenLake Wave 5 real P45 DP8×TP8 serving-scope 为48,594 action tokens、depth 2,472、A−B/B−C strict `0/0`、zero backward/commit，P45 serving recovery已验证；M15 serving、FrozenLake full backward/AdamW/perf/convergence未验证。P58 profile/environment/Python contract与完整 pinned-image gate通过，marker含 `vma_p59_only=1`；P58 target尚未重跑 | FrozenLake P45/M15各自300-update full horizon与P58 fresh target必须独立通过首commit backward-health/first-update与full-horizon gates；P58 warning target须保持 B-C/current exact 并记录 A-B 剂量，strict carrier 修复后再恢复 strict A=B=C |
| CANON_V1_HP_FIRST_UPDATE_GATE | Exact registered full-run numerical admission. On train step 0, observes the complete accumulator before AdamW and requires the workload-specific denominator/microsteps, all-finite, nonzero, and stable-L2 in `(0,1e6]`; after AdamW it requires finite/coherent optimizer evidence before outer weight sync/checkpoint. P58 Qwen3-4B uses 16 rank-major gradient groups and denominator 16 despite eight outer prompt chunks. The bound is a regression sentinel, not a clip value | off; registered Phase4 full contexts plus exact P58 Zero-HP full, always requiring `CANON_P59_CHECKED_VMA=1` | Phase4 pre-registered in V1.P4.9; P58.11 construction validation in progress; target first commits pending | retain as a long-term first-commit safety gate or retire per workload only after target full horizons establish a tighter envelope |
| CANON_V1_FL_TP8_AB_ARM | FrozenLake DP8×TP8 serving A/B first-red 双臂 selector：`p66-off` 整体关闭 checked-VMA 作定罪臂；`serving-scope` 保留 P59 checked-VMA backward、仅将共享 serving kernel 圈回历史程序作候选修复臂。两臂固定完整 32-prompt/256-trajectory P45 或 M15/main 几何，单轮 pre-backward 受控退出，zero backward/zero optimizer commit | 空/off；仅 `qwen3-8b-dp8-tp8-frozenlake-v1-ab-debug.env` 可设两个登记值 | host resolved-env、Attempt-9 red/recovery/B−C-negative classifier、双 P45 64-chip renderer、fixed-image focused/full gates 通过；DP8×TP8 target 未跑，任何 arm 的 Kubernetes Complete 不替代 classification JSON | 双臂给出可归因判决并由后续 TP8 trainer-forward/backward oracle 收口后退役；失败证据永久保留 |
| CANON_P78_SEGMENTED_ACTOR_LOGPS | P78 old-policy trainer C scorer 数值 selector：只在精确 `frozenlake-p45-onehost-dp4-tp1` 上，把 `RLCluster` 的 standalone `common.compute_per_token_logps` 外层 giant-jit 改派到 adapter 已有 P32 host-segmented engine；保持 global M=1024、local M=256 与原 `_sequence_group` 算术，不降低训练 reverse 的 local M。每个模块程序内部才切片/转置/cast该模块所需 trainer leaf，映射参数不作为 JAX 输出逃逸，因此 C 仍独立来自 trainer state而不物化完整399-leaf engine state。序列长度由 learner 已持有的 NumPy masks 计算后传入，device mask 独立复算并以 finite guard fail-closed；不做任何 device-to-host 取回。缺省/空/0 精确直调历史 common scorer；其他值、缺 host lengths、其他 workload/DP/TP 均 fatal | off；one-host runner 只对 P45 DP4×TP1 measurement 默认写 1，P45 DP2×TP2、DP1×TP4 与 M15 全部写 0 | 试验；初版 CPU/fixed-image 门绿但 R11 在80.5 GiB入场后因完整映射保活于首个25 MiB cache处仅余897 KiB而 `INCONCLUSIVE`。后续 deferred-map scratch 通过逐位 parity、`jax.transfer_guard_device_to_host('disallow')`、零 mapped-leaf output及缺映射负控；修订后仓内门与真实 strict A=B=C/梯度/HBM/XProf待跑 | fresh R12 exact one-host 同时通过 strict Zero-TIM、geometry gradient anchor/HBM 与 actor-score XProf 后才可进入 P45 DP4默认包；其余 workload/geometry必须逐一独立准入，任一数值红即否决并保留证据 |
| CANON_OPT_STATE_RESIDENT / CANON_P30_OPT_STATE_OFFLOAD | 优化器驻留/卸载 | resident=生产默认 | 默认开 | resident 焊死后 OFFLOAD 降级为逃生开关保留 |
| CANON_KV_UNIFIED | U 臂读路径统一实验 | off | **否决区**:生产红(43→9 仍 0.28),非修复 | 可删,判决记录永存 |
| MIN_TOKEN_BUCKET / max_num_batched_tokens(非 CANON 但同级) | R2:全局/每 rank 桶契约 | 钉死 256 族 | 已认证 | 永不自由化;新几何走契约注册 |

### CANON_P59_RANK_PARALLEL_BACKWARD DP1×TP4 精确补充（2026-09-06）

- v2 FrozenLake one-host 的 `frozenlake-p45-onehost-dp1-tp4` 与
  `frozenlake-m15-onehost-dp1-tp4` 是仅有的非 P66、unit-DP P59 admission。
  两者的 default stream tape 依赖 P59，因此必须同时保持
  `CANON_P66_P59_CHECK_VMA=1`；VMA off 立即 fatal。其他 DP1 workload、错误
  TP 或前缀相似名称仍拒绝。P66 的四个既有 mapped 诊断臂语义不变。
- 此补充当前只有固定镜像 CPU 的 DP1×TP4 两层真实 mapped-VJP 逐位、
  `transfer_guard('disallow')` 与负控证据；完整 Qwen3-8B one-host
  reverse/gradient/HBM 尚未通过，不得外推为 target 或 GKE 认证。
- `CANON_P67_P66_VMA_P59_ONLY=1` 的 P59-context 判定承认这个 size-one
  manual data axis，仍要求 `data/model` 两轴都是 Manual、model size > 1、
  rank-parallel 与 checked-VMA 同时开启。这样 Pallas out-shape 在外层
  checked `shard_map` 中始终取得非空 `manual_axis_type`；普通 serving
  不在该 abstract mesh 内，继续返回 `None`，不改变其程序。
- Projection 的 `_p59_local_tp_context()` 同样只负责核对执行上下文，不复制
  workload 白名单：size-one manual data axis 仅在 rank-parallel 已开、
  `data/model` 双轴都是 Manual、model size > 1，且 context 与 live engine
  mesh 的 data/model size 精确相等时成立。具体 P45/M15 准入仍只由 adapter
  上述白名单和 checked-VMA 门决定；ordinary serving、错误 mesh 与 TP1 不变。
- Attention 的 `_p59_local_attention_context()` 使用同一职责边界：不读取
  P66/GSM8K 诊断 selector，只在 rank-parallel、精确双 Manual 轴、positive
  data、model size > 1 与 live engine data/model size 精确相等时识别已经
  DP/TP-local 的 Q/K/V/cache。它不改变 RPA 算术或 local-KV shape 检查；
  ordinary serving、flag-off 与 topology mismatch 继续不选或 fail-closed。

## B 层 · perf/仪器类(observational;发布协议 = 每负载 =verify 首步绿 → =1)

| Flag | 语义 | 生命周期 | 日落 |
|---|---|---|---|
| CANON_PERF_LOG | [PERF] 分段计时(=0 静音) | 默认开 | 长期保留(观测契约) |
| CANON_BATCHED_EVIDENCE | 证据取回批量化(-6s/步) | GSM8K 默认;FL/DS 待 verify;P68 补 grouped mirror(DP16 反向收据),配对 verify 待跑 | 全负载转正后焊死 |
| CANON_DP_COLLECTIVE_REDUCE | P69 刀2:DP 梯度归约 collective 选择器,只作用于 `FixedDPRankGradientReducer` 的 reduce transaction。缺省/空/0=历史 fixed ppermute 树(reduce+broadcast、operand barrier 逐对相加)原函数对象直调,程序逐字节不变;1=A 案:sub-32bit 浮点 leaf 升 FP32 后每 leaf 单次原生 `jax.lax.psum`(既有 shard_map check_vma=False 内),cast 回原 dtype;tree=B 案:all_gather 后每 rank 以 registered `fixed_dp_sum` 固定二叉树本地相加(与历史树同配对同序,host 上与历史逐位同值);其他值 fatal。前向零接触,`compare_local` 看门狗、staged 写路径与 receipt schema 全不变(`reduction_rounds` 仍报历史 ppermute 轮数) | off | 试验;scratch host 门(pinned image CPU):FP64 包络、×2 双计负控、flag-off bitwise 冻结指纹、receipt 不变已绿;one-host DP4 E3 已测(psum 归约段 −62%、warm −9%,commit 范数与 legacy 逐位同;tree 结构性逐位同);dp4 one-host profile 条件默认 1(`:-1`,可覆盖);dp16/target 未跑 | P69 Lane1 前向指纹 + Lane2 全套 + Lane3 material 判词后按 workload 转正;任一红退役,判决记录保留 |
| CANON_DP_COMPARE_MODE | P70.4 刀1:DP reduce 后 replica 看门狗选择器,只作用于 `FixedDPRankGradientReducer` 的 replica compare。缺省/空/0/full=历史全量逐元素 ppermute 比对(整棵 reduced 树过邻居,程序与 receipt 逐字节不变);fingerprint-hybrid=每 reducer 生命周期(生产=每 update)前 `HYBRID_FULL_COMPARE_GROUPS`(=2)组保留全量比对且同组跑指纹程序作自检(指纹与全量判决不一致即红停),其余组只 ppermute 每 leaf 双独立 uint32 校验和(rot-add + rot-xor 两混合器,位精确 bitcast,2×N_leaf 标量)并在 mismatch 时报 rank/leaf/path;其他值 fatal。检出弱化:同内容不同位置的补偿性篡改需同时碰撞两个代数独立混合器(NOTES 碰撞论证);−0.0/+0.0 分歧从漏放变为检出(更严),同位 NaN 分歧交给有限位门(顺序与历史一致) | off | 试验;scratch host 门(pinned image CPU):kill-test 单比特翻转必响并指认 leaf、补偿双元素 swap 骗过 naive sum 但双校验和必响、flag-off 冻结 jaxpr/receipt 逐字节同、p69 冻结指纹回归绿;one-host/target 未跑 | P70.4 GATE(kill-test 双项+one-host 配对 walls/范数锚逐位/strict 绿/程序清单 diff)后按 workload 转正;任一红退役,判决记录保留 |
| CANON_DP_DISTINCT_SCHEDULE | P70.4 刀2:per-rank distinct-fingerprint 签名的计算降频。缺省/空/0/every-group=历史每组每 rank 全量 `_gradient_signature`+sha256(receipt 逐字节不变);first-group-warmup=每 update 首组 + 进程前 `DISTINCT_FINGERPRINT_WARMUP_UPDATES`(=3)个 update 的所有组照旧计算,其余组跳过签名(receipt 指纹置 `skipped:receipt-schedule` 并加 `rank_local_fingerprint_mode=skipped`,distinctness 检查在 skipped 组不判);接线正确性属程序级性质:调度/staging/归约程序不随组变,首组+暖机组的检出对 wiring 类故障延迟有界(≤1 update);其他值 fatal。与 deterministic_repeat 互斥(adapter 显式红停) | off | 试验;scratch host 门:调度正确性 kill-test(首组/暖机/skip 序列断言)、flag-off 逐字节同、p69 回归绿;one-host/target 未跑 | 同 CANON_DP_COMPARE_MODE 的 P70.4 GATE;任一红退役,判决记录保留 |
| CANON_DP_FINITE_FETCH | P70.4 刀3:isfinite 位取回的同步点。缺省/空/0/sync=历史逐组同步 device_get+立即 raise(程序与 receipt 逐字节不变);batched-commit=有限位仍逐组在设备端计算(staged+reduced 两段),host 取回合并为 commit 点前单次 int32 向量 `jax.device_get`(P68 批量收据通道),`drain_deferred_finite_receipts()` 在任何梯度进 optimizer commit 前校验全部收据,violation 在 commit 门 raise(带 group/stage/rank/leaf/path);fail-closed 语义不变,只移动 host 同步点(检出延迟 ≤1 update,仍先于 commit);receipt `post_reduction_all_finite=deferred-commit` 字符串逐 receipt 传播,严禁在 drain 前宣称 finite;其他值 fatal。与 deterministic_repeat 互斥 | off | 试验;scratch host 门:非有限注入 kill-test(commit 前必拦、commit callback 零调用)、flag-off 逐字节同、p68/p69 回归绿;one-host/target 未跑 | 同 CANON_DP_COMPARE_MODE 的 P70.4 GATE;任一红退役,判决记录保留 |
| CANON_DP_REDUCE_ONCE | K2(tasks/v1_perf_arch phase2):每个 update 只做一次固定序 DP 归约。缺省/空/`0`=off(每组一次 reduce-and-broadcast,照旧);`1`=每组把 rank-local staged 梯度表在自己的 DP 分片上逐叶累加(`zt_tr_dp_staged_accum`,无集合通信),update 末尾对累加表做一次 `finalize_staged`(同一归约程序、同一固定树、同一 replica/有限性收据);每组收据改为 staged 表的逐 rank 签名/有限位/精确非零计数(单程序,末尾一次取回);sink 一次调用代表全部组(trainer 节拍与累加分母同步前进 G)。求和顺序有意改变(先跨组后跨 rank)⇒ 梯度比特变、锚重钉;logprob 契约不变。要求 RANK_PARALLEL=1、无 P66 diagnostic arm / numeric debug / deterministic_repeat,否则 fail-closed;其他值 fatal。v2 Phase 0 起 treatment reducer 自身的 shard_map 强制 `check_vma=True`；fixed-tree 仍按原顺序 reduce 到 rank 0，再把 FP32 位解释为 int32，以非源 rank `int32.min` sentinel 做一次 `pmax` 精确发布，VMA 因而证明 DP invariant 且不增加浮点算术；flag-off 仍选择历史 unchecked reducer。census `--dp-reduce-once 1` 要求每组 staged_accumulate、update 级 fixed_dp_reduce/gradient_accumulate 各 1 | off;CPU 门:reduce-once 梯度逐字节 == fixed_dp_sum(Σ_g staged)·scale,两次跑逐字节同,与逐组流 rtol 1e-5,sink 一次(index 0, microbatches=G),诊断模式拒;真归约器 2×2 CPU 网格两 update 通过;r5 捕获负控响。v2 Phase 0 host 门另证 DP2×TP2/DP4×TP1 checked reducer 与 legacy 输出(含 `-0.0`)逐位同、`jax.transfer_guard('disallow')` 下执行、旧 reducer 开 VMA 必红；D4 census 为 DP2 `ppermute[data]=1+pmax[data]=1`、DP4 `2+1`，flag-off 仍为旧 DP2 `ppermute[data]=2`。**一主机 dp2-tp2(2026-09-02 r4,commit 3090f17c + census 补丁)**:三 update 全 commit、transactions=1、replicas exact、strict 绿、hierarchy(stream+reduce-once)绿、P74 gap 绿;新锚 `1.6838101148605347 / 3.3025834560394287 / 1.8203867673873901`(update 1 第 7 位有效数字变,系有意重结合);warm **19.07→16.38s(−14.1%)**,HBM 39.79→42.84 GiB(staged 累加器 3.44 GB 常驻);raw /mnt/disks/tunix-data/gsm8k-onehost-xprof/v1_zero-hp_dp2tp2-k2_reduce_once_20260902_r4。**Phase 0 checked-reducer full-tree pair(b4d9fde,r4)**:control/candidate 同一 update-0 WORK 与 310 个真实叶；G4 全叶为 `REASSOC_NOISE`，最差 cosine `0.9999999999999968`、rel-L2 `7.7658e-08`；G5 为 control 96 组 distinct + candidate 3 次 staged 2/2；G6 为两臂各 31 行 outer VMA + candidate 3 次 checked reducer。XProf trace 恰在百万事件上限截断，comparator 仅在 `trace_event_count>=1,000,000` 且 classification 唯一原因为 `trace_census_rc=1` 时放行；semantic/hierarchy census 与 99/99 alignment 均绿。P61 捕获会把 update-0 warmup LR 从 0 换成常数 `2e-7`，故这对 run 只证明 G4/G5/G6，**不作锚或性能接收发**。raw `/mnt/disks/tunix-data/gsm8k-onehost-xprof/v1_zero-hp_dp2tp2-v2p0g4{ctl,cand}_20260903_r4`，receipt `/tmp/v2_default_phase0/dp2_reduce_once_admission_r4.json` | 可与 stream 一同进入 target 优化包(需用户批);DP16×TP4 上每 update 省 15 次 8 轮归约,收益应更大;r1-r3 三次 CODE_REJECT 均为 PartitionSpec 拼法相等性(已改为等价校验) |
| CANON_P75_REPORT_ADJOINT_BUCKETS | P75 P45容量候选：只在精确`frozenlake-p45-onehost-dp2-tp2` rank-parallel report-adjoint中，把399-leaf trainer映射按source依赖与每芯片输出字节静态分桶；同一source的所有engine target留在同一桶并继续通过原JAX VJP，桶自身`shard_map check_vma=True`。每桶完成后释放其cotangent输入，避免8.19 GB engine梯度与16.38 GB trainer输出整棵并存。缺省/空/0走原monolithic函数与调用面逐字节不变；1为实验；其他值或邻近workload/geometry fatal。首版每桶host block以证明容量上界，故性能列预先不合格，后续若容量绿须另刀改成device dependency | off；r18 compiler/HBM diagnosis已证明monolithic输出16,382,088,768 B、alias 1,232,896 B、temp 0 B；实现与CPU数值准入进行中，one-host treatment未跑 | r0b通过bitwise full-tree、G4/D4/G5/G6、strict A=B=C、P66行数、HBM<=90%后才可继续无host-block版本；任一数值/VMA红立即否决并保留证据；仅在P45及后续每个workload/geometry独立性能认证后扩大范围 |
| CANON_P76_CHUNK_DEPENDENCY_TICKET | P76 P45容量候选：精确`frozenlake-p45-onehost-dp2-tp2` rank-parallel reverse在每个非末尾chunk的whole-tree start/add之后，以`shard_map(check_vma=True)`逐设备读取每个已完成accumulator leaf的一个标量位型；`x XOR optimization_barrier(x)`逐项产精确0，OR后只做一次标量TP psum证明TP复制，再把0 XOR进下一chunk首个模型操作数。由此建立真实device依赖，阻止已消费8.19GB chunk pack与下一chunk VJP异步重叠；无host wait/D2H，starter逐位不变。缺省/空/0不构建、不调用、旧路径逐字节不变；1为实验；其他值或邻近workload fatal | off；`r23@c088d6db` correctness全绿但HBM `90.8175%`，与shape-matched r19只差1.8MB allocator噪声；device执行依赖不能推迟PJRT dispatch-time output allocation，容量列REJECT | 保留default-off失败证据，不进入任何默认包；不得强化或重复P76，后续仅在同一CL清理失败实验时退役 |
| CANON_P77_CHUNK_BACKPRESSURE | P77 P45容量候选：精确`frozenlake-p45-onehost-dp2-tp2` rank-parallel reverse在每chunk原有pullback全部派发后、whole-tree start/add之前等待`chunk_pack` device-ready；add与consumed-pack delete后再等待`grad_pack` device-ready，随后才允许下一chunk或report adjoint派发。它不读取/物化数组、无D2H/H2D、不改算术、sharding、VMA或对象；缺省/空/0不等待，1为实验，其他值/邻近workload/与P76并开均fatal。每group打印两类各`num_chunks`等待数，classifier按landed chunk向量精确核对 | off；`r24@d51a35b8`的仅post-add版本correctness全绿但HBM `90.8162%`，容量REJECT。`r25@47f72ee4`两边界版本strict/DP/finite/P66/state全绿且HBM `85.2835%`，容量PASS；新rollout尚无梯度锚，timing因P75/P77 host block不合格。zero_tim=pass@r25；correctness=免测@numerics-admission-dispatch；perf=ineligible@capacity-first | R26 judge把r25只读重判为`MEASUREMENT_ONLY`，不是certify PASS。下一步先建立可重复输入与geometry-specific anchor，再用同一P75/P77容量机制做matched性能；每个新geometry/recipe独立准入 |
| CANON_V2_TRAINING_CAPSULE_MODE / CANON_V2_TRAINING_CAPSULE / CANON_V2_TRAINING_CAPSULE_SHA256 / CANON_V2_MODEL_BINDING_SHA256 | V2 one-host exact-input载具：`capture`只在strict pre-alignment PASS后原子保存完整tensorized `TrainExample`与A/B/T观测；`replay`必须逐文件/逐数组SHA、capture identity及live model fingerprint全命中，且明确绕过environment/rollout/rescore。当前只准精确FrozenLake Qwen3-8B DP2×TP2 no-commit P45/M15；与P64 namespace及P62/P64 numeric debug互斥 | 全部默认unset；capture/replay均为认证载具，不进production profile。capture保留fresh Zero-TIM收据但性能不合格；replay只作同输入gradient/perf，不单独声称fresh Zero-TIM | R27须先过P64 legacy不变、V2全部字段round-trip、one-byte/hash/model/identity/producer-bypass负控与exact-image；真机capture→fresh replay后才可登记anchor。每个新geometry/recipe另行扩identity并准入 |
| CANON_P32_LENGTH_SORT | tasks/v1_long_context phase3:分组 update 在分组前按行的真实长度(prompt_mask + completion_valid)降序稳定排序并按组轮发到各 DP rank,使同组各行长度接近(组的 chunk 数由最长行决定;P45/M15 日志估计可省 35–45% 的 chunk pass)。整棵 TrainExample 一个 gather 程序取同一排列(out_shardings 钉原 sharding),流式余切、末尾 eager 自检、收据行号一致;每行 logprob 不依赖同组邻居(paged 夹具测试钉住),只有组间梯度求和顺序改变 ⇒ **锚重钉**,非零比特。缺省/空/`0`=到达顺序、逐位不变;`1`=开;其他值拒绝。CPU 门 97 passed;一台机 dp2-tp2 / dp4-tp1 / dp2-tp2-long 三几何上**锚均逐位不变**(无需重钉);长几何(0.8k–2.1k 行)实测 chunk pass 69→59、update 12.31→10.62 s(−13.7%),收益与省掉的 chunk pass 同比例(tasks/v1_long_context_onehost/phase3.md)。 | off(候选,建议进 P45/M15 bundle) |
| CANON_P32_CHUNK_BATCH | tasks/v2_dispatch phase5:分组 forward 把 k 个连续 256 宽 chunk 的程序(chunk 输入、embed、层——`fwd_block` 或逐层——、norm、head、rows glue)用 Python 循环 trace 成**一个**直线图程序 `zt_tr_fwd_chunks`(cache 与 glue 缓冲在图内 carry,每 chunk 的 tape 作输出;chunk_start 作操作数,每个 k 一次编译);组内 chunk 数不足 k、余数、以及 fwd_block 程序未建时的首 chunk 走原逐 chunk 路径;reverse 零接触。**缺省/空 = 批 2**(tasks/v2_dispatch Phase 16 清一版,2026-09-07;选了 `CANON_P28_LAYER_SCAN` 档时缺省回逐 chunk,二者互斥);`0`/`1` = 逐 chunk;`2..64` = 批;其他 fatal。CPU 门:玩具 2-chunk 组 k=2 与 k=1 logps/entropy/caches/tape/反向梯度逐字节(`tests/rl/test_p32_chunk_batch.py`)。 | off(候选);**dp2-tp2 一台机认证(2026-09-06)**:k=2 逐层 `…v2disp_p5_dp2_20260906_r1` 派发 9,006 → 6,734、warm 10.57 → 9.38 s、HBM +0.3%;k=2 + `CANON_P71_SCAN=fwd_block` `…v2disp_p5b_dp2_20260906_r2` 6,734、9.60 s、HBM +0.3%;两发锚逐位、A=B=C 96/96、普查 GREEN。首发 r1 因批程序捕获引擎叶(1f77ddf4 修为操作数)在 update 3 撞对齐门,已修;**dp2-tp2-long 认证(2026-09-06 21:09 UTC,tasks/v2_dispatch phase7 R4)**:k=2 + `fwd_block` `…dp2tp2long-v2disp_p7r4_long_sel_20260906_r1` vs 同 base control `…_long_ctl_…`(lane 40ae9f83,选择器关):派发 7,179 → 4,789、锚逐位、A=B=C 全 PASS、HBM +0.4%、warm 9.1 → 8.86 s;8B 一台机 CAPACITY_REJECT(phase7.md 算术表),清一版按 1.7B 双几何证据进默认 | **默认 = 2**(Phase 16 清一版;显式 `1` 回逐 chunk;dp2-tp2 与 long 的 fresh 认证见 Phase 16 账本行) |
| CANON_ONEHOST_JAX_CACHE_DIR | 一台机 xprof-pair 载具的可选持久编译缓存目录(宿主机路径,须在 /mnt/disks/tunix-data 挂载下);设了就把它作为容器内 `JAX_COMPILATION_CACHE_DIR` 并带上集群 JobSet 相同的 `JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0` / `JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=all`。只影响发射时的冷编译(step 1 ~300 s),普查只看被捕获的 update(compile 事件本就为 0)。缺省=不开,配方不变。 | off(载具工具) |
| CANON_P32_LONG_PROMPT_EXAMPLES | 一台机长上下文载具的确定性数据构造器；值必须为 `lo-hi`，只在 GSM8K synthetic worked-example 前缀中控制每行追加的长度，不进入 production renderer/profile，也不改变模型或优化器定义 | off；仅 `dp2-tp2-long` / `dp2-tp2-long8k` 载具设置 | 长上下文几何锚与性能矩阵归档后保留在载具层；若删除载具则同 CL 退役 |
| CANON_P28_BATCHED_REPORT(=1/=verify) | report 窗合并+remap jit 化(FL -14.5%) | GSM8K 默认;DP16 待验 | 同上 |
| CANON_P28_BATCHED_REVERSE(=1/=verify) | P52 反向脚手架合并(-13.3%) | 一宿主认证;DP16 等 grouped 移植 | 同上 |
| CANON_P28_LAYER_SCAN | =verify 恒等仪器/=verify_rev THIRDPROG 演示 | **=1 否决(净负 -5%)** | 仪器保留;=1 进否决区 |
| CANON_P71_SCAN | P71 scan-over-layers 阶梯选择器,仅 grouped reverse(`_p32_reverse_group`)消费,enum 解析:**缺省/空 = `fwd_block`**(tasks/v2_dispatch Phase 16 清一版,2026-09-07;选了 `CANON_P28_LAYER_SCAN` 档时缺省回逐层,二者互斥);`0`/`off` = 历史逐层路径(原函数对象直调,程序逐字节不变);`fwd`=E1:每 chunk 的反向 forward tape 由单个 `zt_tr_fwd_scan` scan 程序重建(scan body 复用 fwd_layer 同一组成与 P50 `_ensure_layer_scan` stacked 参数布局,不新增第二份参数栈;scan 体内 `jax.named_scope` 保留 p71_params_merge/p71_layer_fwd/p71_fwd_tape_scan 分段层级供 xplane 寻址),serial 分支 pullback 走既有 `bwd_layer_block_tape` 栈内静态切片程序,rank-parallel 分支保留原 mapped 逐层 pullback(hidden tape 一次 jitted unstack,cache tape 即 replay 自身逐层对象,mapped 程序与操作数值不变);`bwd`=E2′(含 fwd,**展开块而非 scan**):rank-parallel 分支的逐层 mapped pullback 改为每 chunk ceil(L/B) 个展开块程序(`zt_tr_dp_parallel_bwd_block_NN` 家族,B=`_P71_BWD_BLOCK_LAYERS`=7 模块常量,降落伞 7→4→2、B=1 即退回逐层,任意层数按整除-余数切连续 span;块体=Python 循环 tracing 展开的直线图,**无 lax.scan/fori_loop**,scan 的 loop 重结合入口不存在——E2 scan NUMERICAL_REJECT 的 norm-scale 重分块前科即源于该入口),块程序由 `_p59_parallel_map` 同一构建路径产生(mesh 选择/manual axes/check-vma pcast/localize 上下文全复用,`rank_local_arg_indices=(1,2,4,5)` 与逐层 map 相同;层堆叠 tape 操作数走 axis-1 data 分区),块内自顶向下逐层调用逐层 map 所 trace 的同一 pullback 函数对象(check-vma 档同规则选 raw 闭包),`jax.named_scope` 保留 p71_bwd_layer_NN 逐层 HLO 层级(P60-2G:逐层身份在块内仍可寻址,优于 scan);tape 双源:生产档=E1 stacked tape 程序内静态 `index_in_dim` 切片(bwd_layer_block_tape 同款,免 hidden unstack 程序),fwd-off 源=逐层操作数(测试/解耦保留);dcache/dhidden 块内 in-graph 链接、块间按逐层同界传递;块输出=逐层 staged 梯度 tuple 与逐层 cache 余切 tuple(下游 P70.1 累加/组装与 cache 余切合约零改动);leaf 操作数取 P70.3 prepared pack 切片;要求 CANON_P59_RANK_PARALLEL_BACKWARD=1(serial 分支 fatal)且 TP1(非 unit model 轴 fatal);`full`=E3 预留档,fatal;其他值 fatal;与 CANON_P66_BACKWARD_ARM、CANON_P28_LAYER_SCAN 互斥 fail-closed。前向 pass 零接触(scan 全在反向 tape 侧);梯度验收按 `tasks/v1_hp_zero_tim/phases/v1-p71-scan-fusion.md` 预注册两级(一级 bitwise 锚,二级降档五项全绿自动接受制;E2′ 展开无循环重结合,一级锚有实在机会,P50 r3/r4 类 norm-scale 归约由真机锚定谳)**`fwd_block`(tasks/v2_dispatch Phase 3,2026-09-06)**:grouped FORWARD 把一个 chunk 的 L 层用 Python 循环 trace 成一个直线图程序 `zt_tr_fwd_block`(每层参数与 cache 独立操作数,不 stack 参数/缓存,不 scan),输出每层 new cache、每层 hidden_in(tape)与 hidden_out;reverse 零接触;与 `CANON_P28_LAYER_SCAN` 互斥。dp2-tp2 一台机认证:锚逐位、A=B=C 96/96、派发 9,006 → 7,278、warm 10.57 → 9.95 s、HBM +0.2%,普查 GREEN(`…v2disp_p3_dp2_20260906_r1`)。`bwd` 档在 TP2 上 NUMERICAL_REJECT(`…v2disp_p4_dp2_20260906_r1`,锚不逐位),保持 TP1-only。 | off;E1 已落地(5017c279,真机一级锚逐位命中);E2′(bwd)在 TP2 **NUMERICAL_REJECT**(tasks/v2_dispatch phase4,`…v2disp_p4_dp2_20260906_r1` 锚不逐位,封存);`fwd_block` **dp2-tp2 认证**(phase3,`…v2disp_p3_dp2_20260906_r1` 9,006 → 7,278、锚逐位、A=B=C 96/96)+ **dp2-tp2-long 认证**(phase7 R4,与 `CANON_P32_CHUNK_BATCH=2` 同发 `…dp2tp2long-v2disp_p7r4_long_sel_20260906_r1` vs `…_ctl_…`:7,179 → 4,789、锚逐位、HBM +0.4%);阶梯 off→fwd→bwd→full 逐级包含,回滚按段 | E1/E2/E3 各段两基准硬门+梯度锚(或二级五项)+显存 +5% 红线绿后按段转正;任一段 NUMERICAL_REJECT 封存该段草案并保持 default off |
| CANON_CONTINUE_DECODE(=K) | 设备内 `lax.while_loop` 连续 decode，摊薄逐 token host 往返；async scheduling 必须关 | off；V1/P58.7 profiles 固定 K=8；M15 target-debug 也保留 K=8 以复刻生产程序，其 replay envelope 从首个 registered continue-decode call 记录 mixed chronology，四个 tensor strata 仍仅由 standard path 产生；一宿主 r20c/r20y KEEP；DeepSWE reader 已接线但 target 未跑 | Phase4 三个 full recipe 与 P58.7 各自 target 绿后按 workload 转正；新 K/尾桶重认证 |
| CANON_FIXED_AR_GATHER | fixed TP reduction 的三轮 ppermute 传输换成一次 all-gather，本地仍按相同 rank 顺序相加 | off；一宿主 r11 KEEP；P58.7 target 未跑 | DP16/DP8 full strict 绿后按 workload 转正 |
| CANON_PALLAS_GATHERED_LOGPROBS | Pallas scorer 片上直接产 selected logprob/top1/rank，避免全词表 logprob 物化；Phase4 新增 DP8/DP16 row-sharding 与每-rank M256 padding | off；一宿主 r10 KEEP；DP port IMPLEMENTED/TARGET NOT RUN，含 P58.7 Qwen3-4B | Phase4 三个 full 与 P58.7 target 的 exact gate/XProf 绿后转正；任一形状红回到 materialized scorer |
| CANON_LOGPROB_STEP_FUSION | decode logprob 的 slice/pad/gather/slice 胶水收进一个 jitted program，值链不变 | off；一宿主 r15 KEEP；P58.7 target 未跑 | Phase4 三个 full 与 P58.7 target 绿后按 workload 转正 |
| CANON_FUSED_TREE_OPS / CANON_PALLAS_NORM_MATMUL / CANON_PALLAS_INPUT_FUSION | P56 默认-off 候选实现；V1.1 不启用：P59 已取代主要 host-glue 靶，norm/input fusion 未进入最终 serving 配方 | off；保留历史 KEEP/边际/未转正事实；V1 profile 明确为 0 | V1 full 完成后按 P56 判决裁撤或另立新证据重开 |
| CANON_SAMPLE_SPLIT_FUSION / CANON_ENGINE_LOGPROB_READBACK / CANON_ANCHOR_OVERLAP / CANON_GSM8K_VANILLA | P56 中性、被取代或仅对标/载具开关，不属于 V1 默认配方 | off；不进入三个 full recipe | 战役归档后退役；禁止借 V1 profile 开启 |
| CANON_P3_APC_BOUNDARY_REPORT | Phase3 G-A 固定 token deep-prefix 边界报告路径;有值才运行 cache-hit prefill vs B full-reset 的前向探针 | 试验,缺省空/off | P3.1 结束后退役;证据保留 |

### v2 Phase 4 lifecycle decision ledger (2026-09-04)

| Family | Classification | Decision boundary |
|---|---|---|
| `CANON_P32_KEEP_TAPE` | default candidate; stream is the only production value | Do not delete legacy values until P45 and DeepSWE target anchors pass and the test-only reference replaces the existing batch comparison with equal discriminating power. |
| `CANON_DP_REDUCE_ONCE` | default for admitted 1.7B/8B recipes; retain `0` as a kill-switch | DeepSWE admission and 32B HBM decide additional workload scope. The per-group path remains for at least one release and for geometries that cannot hold the staged table. |
| `CANON_P32_LENGTH_SORT` | experiment, default off | P45 DP8xTP8 control/treatment decides defaultization. Until then only the explicit P45 renderer option may set `1`; GSM8K and M15 full manifests keep it absent. |
| `CANON_P71_SCAN` | geometry-scoped selector; `fwd` admitted, other values not default | Keep the selector and its topology refusals. Do not weld or broaden it without TP>1 target evidence. |
| `CANON_ONEHOST_JAX_CACHE_DIR`, `CANON_P32_LONG_PROMPT_EXAMPLES` | carrier tools | Never enter production bundles. Retain only while the one-host compilation/long-context carriers remain active. |
| `CANON_FUSED_TREE_OPS` and unselected P56 fusion arms | retirement candidates | Remove only after target baseline anchors are frozen; each code deletion keeps or replaces its negative/control gate under R1. |
| P38/P61/P62/P64/P66 families(P61 扩展 2026-09-07:`CANON_P61_BACKWARD_NUMERICAL_DIR` 下另写 `example`/`logps`/`algo_config.json`;`CANON_P61_STOCK_GRADIENT=1` 再写 stock 反向的 `stock_gradient`/`stock_logps` 并跳过 `model_after`——只在捕获目录下有意义,tasks/v2_dispatch Phase 13/13b)| diagnostics | Consolidation may rename entrypoints only in a separate CL; no diagnostic flag becomes a production default. Runtime markers and historical evidence names remain stable. |
| CANON_XPROF_DIR/_SKIP_STEPS/_STEPS | XProf+perfetto 捕获(一次出双产物)；需要 Pathways XProf 的 V1 载具，其 DIR 固定为按 JobSet/attempt 隔离的 `gs://.../p33/<job>/attempt-<n>/xprof-update`，结束后硬门回收到本地证据目录；P58.37 起 1,000-update DeepSWE production full 明确全部 absent，XProf 只留在独立 one-host/P59 诊断载具 | 仪器；P58 production full off/absent | 长期保留 |
| CANON_PERF_TRACE_DIR / CANON_PERF_TRACE_EXPORT_STEP | 官方 tunix.perf v2 语义时间线导出目录与零起点单步窗口；有 profile 的 V1 载具只序列化 warmed step 2，避免 full train 每步写盘；P58.37 起 DeepSWE production full 两者 absent，不创建 Perfetto exporter | 仪器；P58 production full off/absent | 长期保留(官方 Metrics 契约) |
| CANON_XPROF_PYTHON_TRACER/_HOST_TRACER | tracer 档位;**python=0 是 device plane 的前提**(开着它训练捕获退化为 host-only) | 仪器;载具默认 python=0 | 长期保留 |
| CANON_XPROF_TPU_TRACE_MODE | update 窗 TPU trace 密度；V1 full 固定 `TRACE_COMPUTE`；P60-2G 签名 one-host UI carrier 固定低密度 `TRACE_ONLY_XLA`，并以 full XPlane 的 8/8 module/tail gate 和独立 trace-JSON gate 双重防 drop；该 carrier 另固定 XProf 逻辑字节 soft 1.2e9 / hard 1.5e9，超 hard 保留原件但 arm RED | 仪器；仅 `phase=update` 接受；budget 是 task runner 固定合同，不新增可调 flag | 三个 target XProf 与 P60-2G fresh carrier 均无 drop 后决定是否保留默认 |
| CANON_XPROF_LABELS | 为 rollout model/logits/sample 与 trainer fwd/bwd/report JIT 写语义名称；P60-2B/2E 历史合同是一个 whole-update `train` 加 accumulator/optimizer metadata；P60-2G 对签名 Zero-HP arm 将 16 个真实 reverse/reduce/accumulate transaction 映射为 Native API `train_(update*16+micro)`，末个真实 train 跨到 optimizer 结束，无 synthetic terminal step，不改数值、JIT、同步或 Perfetto 词汇；P58.37 production full 必须 absent，避免观察器改变/阻断训练程序 | 仪器；独立 profile 载具开，P58 production full 关；P56 r21/r22 已认证；P60-2F 历史 clean source `5549b5b6` 仍为其原合同 TARGET PASS，但新 UI 判据为 Native-like FAIL；P60-2G local implementation，TARGET NOT RUN | XProf 原生提供等价稳定命名后退役 |
| CANON_P59_XPROF_BACKWARD_DIR / CANON_P59_DP4_TAIL8 / CANON_P60_DETERMINISTIC_AB | P59/P60 DP4 专用 profile、tail 与跨臂载具 | off | 历史载具完成，不进入 V1 full | 证据交付后退役 |
| CANON_V1_GSM8K_XPROF_ARM | `native|zero-hp` 的 one-host GSM8K matched-work/XProf 观测 selector；固定 DP4×TP1、3 commits、warm update 2→3 capture，打印 profiled batch token/advantage hashes；Native 必须 vanilla stock trainer，Zero-HP 必须 strict V1/P59 bundle；P60-2G 只重跑 Zero-HP | 空/off；仅两条薄 wrapper 设置；P60-2G local，TARGET NOT RUN | pair XProf 归档后退役；不得进入 full recipe |
| CANON_P58_ONEHOST_XPROF_ARM / CANON_P58_ONEHOST_SEAM_PROBE | P58 Qwen3-4B one-host mutation-free 诊断载具。`XPROF_ARM=native\|zero-hp` 固定 DP1×TP4/backward-no-commit；`SEAM_PROBE=1` 只允许 Zero-HP arm，并把同一载具扩展为单个已签名 Pillow task、G2、8K response、16 turns、serial scheduler。它保留真实 rollout/durable trajectory/strict decode-vs-prefill gate；有限 RED 与 exact 都只描述该 TP4 carrier，绝不认证 DP8×TP8/TP8 | 空/0；仅 tracked thin wrapper 设置；生产 P58 selector 必须 0 | TP4 首差分类和必要的 DP8×TP8 exact-geometry follow-up 结案后整体退役；不得进入 full recipe |
| CANON_P58_Q4_TP4_ZERO_ADMISSION / CANON_P58_Q4_TP4_SEAM_DIAGNOSTIC | P58.20/21 Qwen3-4B-Instruct-2507 direct-v5p DP1×TP4 full-overlay Zero admission 及其唯一 `standard-decode` 因果控制。主 selector 仅接受 `0\|1`；诊断仅接受空/`standard-decode`，且只替换 continue-decode，模型、采样、fixed head、prefix-cache-off 与 strict A=B=C 不变 | 试验、默认 `0`/空；只由 P58 one-host tracked wrapper 设置，production P58/P59 必须拒绝 | TP4 admission 和后续 TP8 promotion 归档后退役 |
| CANON_P58_Q4_TP4_CONTINUE_KV_DIAGNOSTIC / CANON_P58_Q4_TP4_CONTINUE_KV_MIN_PREFIX / CANON_P58_Q4_TP4_CONTINUE_KV_MAX_PREFIX | P58.22 continue-decode KV discriminator 及其不可调窗口 `2280/3072`；只允许 baseline continue=8、strict precheck-only、bounded integer fingerprint，禁止进入 backward/optimizer | 诊断、默认 `0`/absent；tracked wrapper 原子派生 | 一次因果分类与修复复验归档后整体退役 |
| CANON_P58_Q4_TP4_SHORT_BACKWARD / CANON_P58_Q4_TP4_CARRIER_SCREEN | P58.22 one-host 短宽度 backward-no-commit 与 rollout-only carrier 筛选。short arm 保留真实数据、strict gate、TPU-resident optimizer/zero commit；screen 必须依赖 short 且不得进入 trainer | 诊断、默认 `0`；仅 P58 tracked wrappers | P58.23 optimized replay 完成并由真实 TP8 promotion 取代后退役 |
| CANON_P32_KEEP_TAPE | 分组前向相保留 tape 供反向相消费(v1_forward_dedup S2a):缺省/空/`0`=off(反向相照旧重算 replay 与 tape scan);`1`=前向相 `_p32_forward_group(keep_tape=True)` 在逐层循环里保留每 chunk 每层输入 (cache, hidden) 与 pre_norm hidden,反向闭包直接复用 `forwards[index]`,`_p32_reverse_group` 跳过 embed/replay(F2)/tape scan(F3)——消费的是前向相已算出的同一批张量,同程序少跑一次,构造性逐位;消费后置空释放 HBM(峰值=未反向组的 tape);`stream`(S2b)=同一 tape 但前向 g+1 先于反向 g 发出、消费即释放(活 tape ≤2 组),第 g 组 loss 余切取自整批 loss 程序对已知 logps 的 vjp 切片,update 末尾整批 pullback 逐组逐字节复核余切与 scale,不等即 raise(与 deterministic_repeat、backward numeric debug 互斥)。要求 CANON_P59_RANK_PARALLEL_BACKWARD=1、非 P66 臂、逐层前向(CANON_P28_LAYER_SCAN 未设),否则 fail-closed;其他值 fatal。census 以 `--p32-keep-tape 1|stream` 要求 fwd tape scan 程序缺席,`stream` 另要求逐组 group_loss_pullback、无 forward_groups 父级与两 tape 窗口顺序 | off;CPU 门:reverse 消费 tape 的梯度/余切逐字节==重算版(legacy 与 scan 两 tape 模式),flag-off digest 不变;`1` one-host DP2×TP2 r3/r4:锚逐位、fwd_layer 减半、fwd_scan=0、warm 29.19→18.95s,HBM +17.3%(批保留下限);`stream` CPU 门 51 passed:调度序列/活 tape=2/余切·梯度·logps 逐字节==`1`,行间耦合 loss 负控被拒,jit 余切程序跨 update 只 trace 一次;`stream` 真机(2026-09-02,commit 5454caed):DP2×TP2 r5 锚逐位、fwd_layer 3584→1792、fwd_scan 0、hierarchy stream 档绿无编译、HBM 41.06→39.79 GiB、warm 29.19→19.07s(−34.7%,vs `1` +0.6%);DP4×TP1 stream 锚逐位、fwd_layer 2688→896、HBM 59.26→59.49 GiB(+0.4%)、warm 16.68→11.55s(−30.8%);raw:/mnt/disks/tunix-data/gsm8k-onehost-xprof/v1_zero-hp_{dp2tp2-fwddedup_s2b_20260902_r5,dp4tp1-fwddedup_s2b_20260902_r1,dp4tp1-fwddedup_s2b_base_20260902_r1} | `stream` 可作 P45/M15 target 的默认档(M15 两 tape 峰值 11-19 GB/芯片,见 tasks/v1_forward_dedup/phase5.md D1;S2c 单份终态 cache 可再降);dp1-tp4 载具待 P4B lane 落地后一发;`1` 仅作诊断档保留 |
| CANON_P58_Q4_TP4_TRAJECTORY_REPLAY | P58.23 单一事实源：启用受哈希约束的两个真实 task group、每组 reward `[1,0]` 的 B2×G2 replay；全局 `batch_size=2`，prompt/response=`2048/512`、K=2560，严格禁止 B1。它原子要求 P28 segmented forward/train + P30 sparse/reuse/release/reshard + hardware-certified `CANON_P71_SCAN=fwd`；DP1 明确保持 P59 rank-parallel off。只跑 re-score/alignment/repeat-exact backward-no-commit，不调用 sandbox/decode，不声称 fresh rollout 或 TP8 | 诊断、默认 `0`；仅 tracked P58.23 wrapper 设置 | bounded one-host compile/backward 结论与 TP8 follow-up 归档后退役 |
| CANON_P58_REPLAY_JOURNAL / CANON_P58_REPLAY_JOURNAL_SHA256 | P58.23 replay source 的绝对路径与 SHA-256 receipt；只能与 trajectory-replay selector 成对出现，指向 deterministic B2×G2 merged journal，任何缺失/漂移 fail-closed | 默认 absent；由 wrapper 从已签名本地 artifact 派生 | 与 P58.23 selector 同时退役；不得进入 production profile |
| CANON_P58_CHECKED_VMA_DIAGNOSTIC | P58.18 exact-geometry matched-control selector；合法值仅 `off\|on`。只准入 Qwen3-4B Zero-HP/full 的 128-chip disaggregated DP8×TP8+DP8×TP8 Step-0 carrier。`off` 原子派生 checked-VMA/P66 alias/P67 scoping=`0/0/0`；`on` 派生 `1/1/1`。两者都把 first-update gate/P63 clip 固定为 `0/0`，保留 fixed-head/continue-decode/Fixed-AR/serving HP、完整 trajectory + pre-alignment，并在 backward/optimizer 前受控退出。生产 selector 缺省 absent 时仍是完整 `1/1/1/1/1`，不受诊断影响 | 缺省 absent；仅 P58.18 ON-A/OFF/ON-B 三臂诊断可设；不得与 normal Zero-HP、native recipe 或 subordinate 手工 override 混用 | 三臂因果裁决与 fresh checked-VMA-on strict Step-0 修复复验归档后退役；不得成为 full-training 默认值 |
| CANON_P58_SEAM_LOCALIZATION | P58.19 exact-geometry coarse seam selector；合法值仅 `coarse`。只准入 Qwen3-4B Zero-HP/full 的 128-chip disaggregated DP8×TP8+DP8×TP8 frozen Step-0 carrier；单一 selector 派生 P38 layer seam + terminal-tail、位置窗 `[1686,4096)`、三轮 per-round seal/classify/ACK 与 `p58-seam-v1` durability。每轮 seam/tail 各自使用独立 4 GiB byte budget，轮次只能单调 `0→1→2`，record index 不复用；这是 p58s19d 在累计 1 GiB 自停后的诊断容量修复，不改变 tensor 内容。该窗口覆盖 p58z07/P58.18 已观测的 2,513/3,438/3,715/3,880/4,032 first-red prefixes，同时避免 p58s19b 的 `[3072,4608)` 零记录载具失败。保留 production `CANON_CONTINUE_DECODE=8`：`standard` 是唯一 tensor-strata 来源；`continue_decode` 只允许保留 scheduler chronology，跳过 incident/tensor payload并回显 `CANON_P58_CONTINUE_DECODE_OBSERVER_BYPASS ... tensor_capture=0`。任何非 P58 profile 或未知 program path 仍 fail-closed。保留 production checked-VMA/P67/first-update/clip tuple，backward 与 optimizer commit 均不可达 | 缺省 absent；仅 P58.19 三轮粗定位可设；与 checked-VMA diagnostic、native、普通 Zero-HP full、M15 wide 及 subordinate 手工 override 互斥 | 三轮给出可重复 coarse first-red signature 并完成后续 fine localization 后退役；任何无 join、B-C red、非重复 boundary 或 observer-neutrality red 均只得 INCONCLUSIVE/FAIL，不得转为训练开关 |
| CANON_XPROF_PHASE | 捕获窗模式:step=整步(device 缓冲 ~283 万事件/核,decode ~25s 填满,实为 engine 前 25s 织物)/ update=G6 update 入口→步完成(rollout 不入镜,缓冲装下完整 backward)/ diagnostic=冻结权重 precheck 的一个完整 A-rollout/B-full-rescore/C-old-forward round | 仪器;载具旋钮 P51_XPROF_PHASE;Phase3 profile 固定 diagnostic skip=1 steps=1 | 长期保留 |
| CANON_UPDATE_REPORT / CANON_PRE_ALIGN_REPORT / CANON_ALIGN_REPORT | 对齐/更新报告选通 | 默认开(监控契约) | 长期保留;A−B 哨兵不可撤(用户裁决 2026-08-15) |
| JAX_COMPILATION_CACHE_DIR(非 CANON) | 持久编译缓存(-72s/重启) | 一宿主认证；Phase4 三个 full manifest 已锁定本地目录与 GCS root，restore/save 回执 host 绿；**Pathways target hit 未验** | 三个 full target 记录 hit/miss 与 JIT 后决定是否推广 |

### Phase 0 reduce-once 接收补据（2026-09-04）

- 标准、无 P61 的 `stream + CANON_DP_REDUCE_ONCE=1` 接收发在本地 tip `60142108` 上通过：
  DP2×TP2 三锚逐位为 `1.6838101148605347 / 3.3025834560394287 / 1.8203867673873901`，
  96 条 update alignment + 3 条 pre-alignment 全绿；DP4×TP1 三锚逐位为
  `1.4907878637313843 / 2.2041752338409424 / 2.6263937950134277`，48 + 3 条 alignment 全绿。
  两臂均为 3 commits、replica exact、semantic/hierarchy/module census GREEN；DP2 P74 gap
  mean/max=`0.062512/0.064294 ms`、victim overlap=0。通用 trace JSON 分别在 1,000,447 / 1,000,429
  events 截断，classifier 唯一 reason 为已用低事件负控钉住的 `trace_census_rc=1`，不覆盖独立普查。
  Raw：`/mnt/disks/tunix-data/gsm8k-onehost-xprof/v1_zero-hp_dp2tp2-v2p0accept-dp2_20260904_r1`、
  `/mnt/disks/tunix-data/gsm8k-onehost-xprof/v1_zero-hp_v2p0accept-dp4_20260904_r1`。

### Phase 1 reduce-once 默认包状态（2026-09-04）

- 本地未发布的 default-bundle CL 把 `CANON_DP_REDUCE_ONCE=1` 注入 GSM8K、FrozenLake P45/M15；
  DeepSWE 继续缺席直到 Phase 2 完成 4B 几何、梯度与 `rank_major_rows` 认证。两个 renderer、full-run
  classifier、相邻 HANDOFF/RUNBOOK 与 exact manifest golden 使用同一 truth table；
  `CANON_DP_COLLECTIVE_REDUCE` 仍必须缺席。
- Offline gate：expanded CPU `313 passed, 3 skipped, 132 subtests`；固定镜像最终收据
  `V1_HP_EXACT_IMAGE_PASS ... frozenlake_system_optimization=1 ... manifests=3`；三份 unpublished render
  均含 reduce-once。**Target 仍未验证**：P45 DP8×TP8 的 A=B=C、G1、HBM 与 update timing 在独立发射
  通过前不得写成已认证或性能收益。

## C 层 · P38 诊断家族(~30 个 CANON_P38_*)

| 组 | 代表 | 日落条件(一条覆盖全家) |
|---|---|---|
| capture/journal/ledger/capsule/GCS/replay | SERVING_CAPTURE_*、REQUEST_JOURNAL、INCIDENT_LEDGER、MISMATCH_CAPSULE、GCS_PREFIX、DURABILITY_PROFILE、FROZENLAKE_REPLAY、PRECHECK_ONLY… | **carrier 结案(strict 复验全绿)→ 全家整体退役**;判决类结论(U 臂、slot bug、co-batch、shape-1)迁 FOOTGUNS 后删 |
| precheck-only parser | `CANON_P38_PRECHECK_ONLY`:诊断布尔、非数值 treatment；缺省/空/`0`=关，精确 `1`=通过 strict pre-backward 后受控停止，其他值 fatal。P58 Zero-HP/full production 应缺省 absent，其 finite-A-B warning 身份只接受三种 off 表示并拒绝诊断 `1` | 与 P38 diagnostic carrier 共同退役；在此之前 shell/profile/Python/postflight 必须共享同一闭集真值表 |
| Phase3 APC dirty-page negative | CANON_P3_APC_DIRTY_PAGE:布尔诊断旗标;缺省/空/0 不污染,仅 boundary dirty mode 的 writer 向直接 JAX/vLLM reader 透传 1;污染 A 确定会复用的 layer-0 单个真实 KV page,B 仍 full reset | G-D `p3gd1` 已命中;为 G-E 复验保留默认-off 载具,Phase3 最终结案 CL 退役 |
| M15 target APC carrier | CANON_APC_M15_TARGET_DEBUG=`off\|on`:只准入 DP8×TP8、M15/main、zero-commit 的 bounded target reproducer；observer=`none` 精确单轮，`m15-wide-v1` 的 layer/full observer 与 `m15-e0-kv-v1` 的 `kv3` observer 精确三轮且每轮权重冻结、先 seal/readback 再推进。精确 `kv3` profile 还固定 round-0 的 32 条 prompt identity 并在 round 1/2 重放该 inventory，三轮各自重新执行 rollout/request/cache chronology；其他 profile 继续推进 dataset。历史 `kv` 一轮路径保留，仅用于旧 Attempt-18 回收。`off` 是 APC-off control，`on` 是 production cache-read treatment；两臂保留生产 `CANON_CONTINUE_DECODE=8` 和签名的 `sampler_is=None` rollout-logprob recipe，不允许 TIS weights；四个 tensor capture 仍限定 standard path；A 强制 `prompt_logprobs=None/logprobs=1/skip_reading_prefix_cache=False`，B 强制 `reset_prefix_cache=True` 且 cache tokens 全零 | fresh target red 冻结并完成首红定位、修复后 G-E 全零与脏页负控再次通过后退役；不得进入 full production profile |
| M15 E0 targeted live-KV discriminator | `CANON_P38_KV_OBSERVER_LAYER`、`CANON_P38_KV_OBSERVER_TARGET_PREFIX_SHA256`、`CANON_P38_KV_OBSERVER_TARGET_PREFIX_TOKENS`:三者只能成组设置；仅 D3e 已登记的 M15/main DP8×TP8 frozen off/on carrier 可设为 layer 0、1226-token canonical-action 前缀及其 self-hashed identity。`kv` 保留历史一轮合同；新 `kv3` 固定三轮，每轮重新取得 8 个合法 request alias、独立 128 MiB byte budget，record index 全局单调且不复用。每个 alias 只读取 77 个逻辑页（静态上限 96），随后用同轮 replay-ledger future-prefix 证据唯一绑定 source request；prefix L 的 snapshot 可以 join logical-prefix L 的 next-token red，仍拒绝任何 future `>L` red。旧 P38 observer 缺省仍读全部层且页上限 32。输出是整数 aggregate/fixed-sample diagnostic fingerprint，不是完整 KV bytes；任何 alias 缺失、跨轮配对、非唯一 future binding、无 red join、B-C red 或 observer-neutrality red 均 fail closed | 默认 absent/off；Attempt 19 因 equality-boundary join 与跨轮 dataset advance 两个 carrier 缺陷 INCONCLUSIVE；E0t repair host/fake-GCS 已绿，post-repair pinned exact-image 与 DP8×TP8 target 未跑；成功区分 live stored-KV red/equal 后退役，不能进入 production profile 或放宽 B full reset |
| M15 replay envelope | CANON_APC_M15_REPLAY_LEDGER=`<capture-dir>/m15_replay_envelope.jsonl`:仅与 `CANON_APC_M15_TARGET_DEBUG=off\|on` 成对启用；逐 serving call 保存 host 侧 dispatch/request/position/page 几何与 token-history SHA，不读取 device tensor；A 的冻结 carrier 必须机械证明 standard+continue-decode，B 必须只走 full-reset standard；路径直接位于 P38 capture 目录。无专用 durability 的旧 carrier 仍保留 incident/replay 单文件界；`m15-wide-v1` 与 `m15-e0-kv-v1` 明确跳过重复且会在多轮中饱和的 incident ledger，replay envelope 继续作为 chronology authority | M15 target red 的完整 producer unit、serving chronology 与 first-red join 已冻结并完成 deterministic replay 后退役；不得单独开启 |
| M15 E0 KV three-round durability | CANON_P38_DURABILITY_PROFILE=`m15-e0-kv-v1`:仅允许精确 M15 target `off\|on` + targeted layer-0 KV observer，且 `CANON_P38_DIAGNOSTIC_ROUNDS=3`、seam/tail/terminal observer 全空。该 profile 对 round-0 batch 的 `p57_index/seed/map_sha256` 生成 self-hash，要求 1 个 frozen marker、2 个 requeue marker 与唯一共同 SHA，禁止跨轮 dataset advance；每轮仍重新执行 rollout/request/cache chronology。每轮严格执行 16 条 KV record（8A+8B）隔离 staging → self-hashed classifier-input archive upload/readback → classifier PASS → round archive/upload/readback → `ROUND_COMPLETE` → learner ACK；round 0/1 成功后即使 round 2 或 root collect 失败也不得丢失。三轮完成后才生成 arm aggregate；只读回传必须优先读取 per-round 小收据，不以 `COLLECTED/COMPLETE` 存在为回收前提 | 试验、仅 M15 E0 target；Attempt 19 证明旧实现推进 dataset 后 round 1 无目标记录；E0t host fake-GCS/aggregate 已绿，post-repair pinned exact-image/target 未跑；首红机制裁决完成后整体退役，失败轮与 partial salvage 永久保留 |
| M15 wide durability | CANON_P38_DURABILITY_PROFILE=`m15-wide-v1`:只允许精确 M15 target `off\|on` + `layer\|full` observer；精确三轮，每轮独立 shard root 与 observer byte budget，record 序号不复用；每 30 秒最多复制 32 对完整 JSON/NPZ、逻辑 payload 最多 256 MiB，生成确定性 archive/SHA，远端下载复核后才写 `SHARD_COMPLETE.json`；round classifier 只读本轮 sealed shard 隔离副本并过滤 cumulative replay ledger；终态顺序固定为 shard complete → round complete → COLLECTED → postflight COMPLETE；根终态丢失时可用 checked-in 小回传脚本恢复已 seal 轮次；每次持久化还核对执行 checkout HEAD 与 rendered full SHA | 试验、仅 M15 wide target；首红定位和最小修复完成后整体退役，失败 shard 永久保留 |
| M15 wide seam bundle | CANON_APC_M15_SEAM_BUNDLE=`<state>/m15_wide_seam_bundle.tar`:仅在 M15 target 的 `layer\|full` seam observer 上启用；round worker 从 sealed shard union 中运行 classifier，再从 A/B 精确 join 选出的原始 seam/tail records、alignment、capsule、replay ledger 生成 deterministic SHA bundle。`m15-wide-v1` 会把该紧凑 bundle、分类和输入收据上传到既有任务 GCS evidence prefix；它不上传整个 live capture 目录 | 首红定位完成并把最小修复通过 G-E 后退役；新的 target 发射仍需用户单独批准 |
| Attempt-7 backward first-red | CANON_P62_BACKWARD_NUMERIC_DEBUG:布尔诊断旗标；仅严格 GSM8K DP16xTP4、P59 fixed-head、`backward-no-commit` 载具可开，打印 loss/VJP/DP-reduce/scale/accumulator 紧凑数值 receipt，累加器最终丢弃且 optimizer commit 必须为 0；它不启用 stable clipping | 默认 off；首红根因定位并由独立修复通过 target 后退役，失败证据永久保留 |
| Registered full-recipe overflow-safe clip | CANON_P63_OVERFLOW_SAFE_CLIP:数值布尔旗标；缺省/0 关闭、仅 1 开，空值或其他值 fatal。仅注册的 Phase4 committed full contexts 与精确 P58 Qwen3-4B Zero-HP full recipe 可开；stock norm finite 时逐位返回原 Optax transform，只有独立 `all_finite` 且 stock norm overflow 时才选 max-scaled L2；真实 NaN/Inf 永不 fallback。GSM8K 与 P58 max norm 1，P45/M15 max norm 100。P58.32 narrow A-B warning 不放宽此 gate | 默认 off；Phase4 host 372/372 与完整 pinned-image `p63_clip=1` 已绿；P58 construction validation in progress；所有 target optimizer commit/full horizon 均按 workload 独立认证，Phase4 证据不自动转移到 P58 |
| P45 rank-1 first-red | CANON_P64_P45_NUMERIC_DEBUG 与 `CANON_P64_TRAINING_CAPSULE_{MODE,GCS_URI,SHA256}`、`CANON_P64_TRAINING_CAPSULE`、`CANON_P64_MODEL_BINDING_SHA256`:仅严格原始 P45 DP8xTP8、APC-off、P59 fixed-head、`backward-no-commit` 载具可开。capture 在 strict pre-alignment 后原子保存完整 tensorized train batch，并在 backward 前绑定 live model sample；replay 逐数组/文件/模型指纹验真，跳过 environment/rollout/B-rescore，只执行完整 trainer forward 与 group-0 backward，随后丢弃 accumulator。Replay 明示 `certification=0`，不能冒充新 Zero-TIM 认证；首次 NaN/Inf 立即停，绝不 clamp/cast/commit | 默认 off；定位 Attempt-7 P45 rank1 的首个 finite→non-finite 边界并完成根因修复后整体退役，所有 capsule、失败证据与 GCS 路径永久保留 |

## D 层 · 发射/基建管道(~230,按前缀组;逐条语义允许"待考古")

T9g registers `CANON_P57_TRAIN_GEOMETRY`: absent retains the historical
DP8xTP8/B32xG8 full recipe; the only present value is `dp4-tp8-b128`.
It selects a separate P45/M15 Zero-HP full profile (32 chips, B16xG8,
global M1024/local M256, 32 gradient groups, 300 updates), never a diagnostic,
Native/IS or GSM8K override. This changes global batch/data budget; it is not
a speed-only same-trajectory claim. APC/eval/checkpoint remain off and all
existing numerical/backward gates remain in force. Default off; construction
and target certification tracked in T9g. Retire only after a separately approved
recipe migration; old evidence remains attached to DP8xTP8/B256.

The enum is workload/infrastructure, not a numerical-algorithm switch.
The P67 wrapper and P57 renderer write it only for the explicit small
option; profiles and `00_env.sh` validate raw/resolved values, the learner
Python process validates `examples.frozenlake.training_geometry`, and
`90_run.sh` forwards the same geometry to both postflight classifiers.
Empty, `0`, unknown and explicitly present legacy values reject; the
legacy renderer emits no key. P59/P63/P67/first-update, fixed-head and TiTO
full identities may use the new geometry **only as a complete T9g tuple**;
diagnostic identities remain DP8xTP8. Qwen3-8B/TP8 adds caller-global M1024
to the head admission table without changing local/kernel M256 or TP
reduction math. T9g host (P57 248/V1 104/APC 12/flags 423), eight-case real
runtime admission, bounded DP4xTP8 CPU reducer and complete pinned-image
gates pass; see `tasks/multiturn-tito-cross-workload/evidence/t9g-host-20260907/receipt.json`.
These are fresh construction claims, not inherited DP4xTP8 target
certification. Both workload arms must declare their global
batch/data budget when compared with historical Native/IS curves.

| 前缀组 | 用途 | 处置 |
|---|---|---|
| CANON_RUN_* / STATE / PKG / PROFILE* / SHIM_ROOT / MODE | 发射管道(渲染/安装/运行合同) | phase2 三层 profile 落地时逐条核对归位 |
| CANON_GCS_CACHE_BUCKET | JAX persistent-cache 的 GCS root；Phase4 三个 full 固定到 P33 cache root，按 resolved profile 分 namespace，restore/save 失败只生成显式性能回执，绝不替代或放宽 Zero-TIM 数值门 | 基建性能合同；三个 target 均完成可审计 cache hit/miss 与 JIT 记账后决定默认范围 |
| CANON_P29_LOG_DIR | P29/P41/P48/P57 one-host trainer metrics 的 host 路径覆盖；缺省/空均回落到 recipe 的 `TB_LOG_DIR`，只由 JAX client 的 `MetricsLoggerOptions.log_dir` 读取，不进入 logits、loss、backward 或 optimizer | 观测/基建路径，默认空；相关 one-host 载具退役且日志目的地由非 CANON workload contract 接管后退役 |
| CANON_V1_HP_FULL | workload-level execution identity；仅 Phase4 三个 renderer 与 P58.7 Zero-full renderer 设 1，并由各 workload profile 派生完整 serving/trainer/XProf bundle | 试验、默认 0；Phase4 GSM8K/P45/M15 与 P58 Zero/1000 是四种闭集；四个 full 归档并逐 workload 转正后退役此 campaign selector |
| CANON_WANDB_* | 观测账号面(用户所有,凭据纪律) | 保留,不动 |
| CANON_QWEN3_*(8 个几何) | 模型几何契约 | 保留;属 workload profile 层 |
| CANON_P3x/P4x_*_ADMITTED / NO_COMMIT / RUN_STAGE / 工作负载选通 | 各任务 admission 门 | 任务结案随任务退役(C 层同规) |
| CANON_P46_CENSUS_FIRST_PASS | P46 reward-only full campaign 的 breadth-first 调度：每个尚无 durable attempt 的 identity 只跑一次，invalid 留证后延后；不进入采样 fingerprint，也不放宽 strict finalizer | 试验、默认关；P46 完成一次 exact 1851 x N16 strict campaign 后退役 |
| CANON_P46_FROZEN_V6_IMPORT_ID | 显式选择已封存的 v6 resume snapshot；只允许在新 resume tag 内迁移原始轨迹与 sampler provenance，并逐条留迁移来源 | 迁移期、默认空；旧 campaign 全部升级至当前 harness 后退役 |
| CANON_P57_TIM_ARM / RUN_KIND / INFERENCE_REGIME / EXPECTED_UPDATES / STOP_AFTER_STEP / EVALUATION / EVAL_* / WORKLOAD_CANDIDATE / DATA_SPLIT / CALIBRATION_* | P57 FrozenLake TIM 因果实验身份与耐久产物；`TIM_ARM=mismatch` 为 native/no-IS，`is` 为相同 native 数值程序加 token TIS，`zero` 为完整 zero-TIM/no-IS，且 active P45/M15 300-update zero reference 强制使用已登记 `frozenlake-v1-hp` 优化载具（P59/P66/P67/first-update gate），基础 `frozenlake-tim` profile 不再允许 zero train；三臂继续写入同一 W&B project `zero-tim-p57-frozenlake-tim`，仅以 group/name 区分，fast Zero group 加 `-noeval` 后缀；两 paired workload 为原始 P45/300（candidate/split 均空）与 materialized M15-main/300；native/is 主训练强制 `CANON_P33_ENABLE_EVAL=1,CANON_P31_ENABLE_EVAL=1` 并生成 `0,50,...,300` rollout-only held-out 曲线，active optimized Zero 强制 `CANON_P33_ENABLE_EVAL=0,CANON_P33_DISABLE_EVAL=1,CANON_P31_ENABLE_EVAL=0`、`--eval_every_n_steps=0` 和 checkpoint disabled，不产生 in-process eval 或 checkpoint receipt；native arms 必须 `INFERENCE_REGIME=stock-fast`，zero 禁止该 override；P57.1 calibration/selection 仍只接受 mismatch、M15/200 且无 in-process eval；active 300-step primary 的 `STOP_AFTER_STEP` 必须等于 300，历史 selection 才允许 horizon 内 50-step 边界；checkpoint-free Zero 是效率优先的概念验证，崩溃后不能 resume，也不形成 held-out/final-checkpoint 科学证据 | 试验、默认空/关；P57 完成并归档最终因果报告后整体退役 |
| CANON_FROZENLAKE_CKPT_INTERVAL / CANON_FROZENLAKE_CKPT_MAX_TO_KEEP | FrozenLake 保存频率与滚动保留数；active optimized Zero P45/300 与 M15-main/300 必须 mode=`disabled` 且 root/tag/interval/retention/milestone 全空；Native/IS 以及相应 eval carrier 仍固定 `300/1`，legacy P45、calibration、M15-selection/200 保持 `10/1`。checkpoint-free 身份只允许 exact `frozenlake-v1-hp` Zero/no-eval full，profile、resolved-env、Python parser、renderer 四层必须一致 | 基建合同；效率概念验证结束后再决定是否恢复 Zero 最终 checkpoint |
| CANON_FROZENLAKE_CKPT_MILESTONE_INTERVAL | FrozenLake 额外证据 checkpoint 保留间隔；optimized Zero fast train 必须为空（整个 checkpoint 合同 disabled），Native/IS active 300-step arms 为 `0` 并只保留 `LatestN(1)`；历史 `50` 仅用于已归档的旧 isolated-eval 设计，不能重新用于 active Zero | 试验、默认 `0`；旧 milestone evidence 清理完成后退役正值路径 |
| CANON_P58_DEEPSWE_TIM / TIM_ADMITTED / TIM_ARM / EXPECTED_UPDATES / DEBUG_DIR / NATIVE_STOCK_PROMPT_OBSERVER / ONEHOST_XPROF_ARM / one-host provenance 族 | P58 Qwen3-4B-Instruct native-vs-zero 因果训练身份；固定 128-chip synchronous disaggregated、B8xG16、16K、compact filter、TPU optimizer 与完整 trajectory journal；`TIM_ARM=native|zero` 选择 numerical runtime。Native 的 sampler recipe 另由既有 `CANON_P34_DISABLE_SAMPLER_IS:CANON_P34_DISABLE_TIS` 闭集选择：`1:1`=raw，`0:0`=token TIS(threshold 2.0)，混合 tuple 拒绝；Zero/Zero-HP 必须 `1:1`。Native-IS 是 mitigation arm，不改变原生 numerical program，且 group filter 仍关；Native 保留完整 stock serving/trainer program，所有 shape-valid finite A/B/T_old/T_current mismatch 只观测，Zero 全边界 exact；`NATIVE_STOCK_PROMPT_OBSERVER=1` 只为 native arm 的 rollout 后 B 观察值提供 processed prompt logprobs，不进入采样、trainer、loss、反向或 optimizer，且与 canonical `PROMPT_PROCESSED_LOGPROBS` 互斥；production full 自 P58.37 起强制无 XProf/Perfetto，保持完整高性能 backward/optimizer bundle；`ONEHOST_XPROF_ARM=native|zero-hp` 仅准入 DP1xTP4、两次相同输入、固定 `[-1,1]` 诊断 cotangent、零 optimizer commit 的 update-profile 载具，缺省空，不能认证 DP8xTP8/P59/4B-TP8 fixed-head 或生产轨迹；`EXPECT_HOSTNAME/MODEL_SNAPSHOT/R2EGYM_COMMIT/TASK_IMAGE_ID/RUNNER_SHA256/SOURCE_DIFF_SHA256` 是该载具的字符串/路径 provenance receipts，缺省空且不改变数值 | 试验、默认关；one-host matched XProf package 归档后先退役 one-host selector/provenance 族，P58 production 族在完整 campaign 归档后整体退役 |
| CANON_P59_GCS_PREFIX / CANON_P59_INNER_RUN_CMD / CANON_P59_KIND / CANON_P59_REQUIRE_XPROF | P59 单次载具的证据目的地、冻结内层命令、臂身份与 XProf 完整性要求 | 试验；仅 P59 renderer/one-host wrapper 设置 | P59 证据载具归档后整体退役 |
| CANON_ALIGN*/EXPECT_*/DP_SIZE/TP_SIZE/TRAJECTORIES 族 | 对齐门与拓扑断言 | 监控契约,长期保留 |

### T9f record-full empty-response boundary

`CANON_P57_TOKEN_CONTINUITY_DEBUG=record-full` records request IDs for
completed responses, not every submitted request. A terminal row with no
completed model call/trajectory step/completion token/action token may have an
empty request list only with an explicit `canon.p57-tito-empty-response.v1`
receipt. Row-map, A/B/C sidecar and final classifier validate the same receipt;
sidecar masks must also contain no valid completion or action for that row.
These rows count as unexercised, never as successful token comparisons. A
returned response with missing/duplicate/foreign identity still fails. There
is no new flag, no default change and no change to training rows or deadlines.
T9f target validation is pending; see the owning task's phase and handoff.

## MARKERS(日志 marker 契约,非开关;~60 个)

关键项:`[CANON_ALIGN_PRE]`(四边界判决行)、`[CANON_ALIGN_PRE_JSON/EVIDENCE]`、
`[PERF]`、`[CANON_P38] DIAGNOSTIC_COVERAGE_CONTRACT / PRECHECK_COMPLETE`、
`[CANON_P38_SERVING_CAPTURE*]`、`[CANON_P38_INCIDENT_LEDGER_BYPASS]`、
`[CANON_P58_CONTINUE_DECODE_OBSERVER_BYPASS]`、
`[CANON_P58_CONTINUE_KV_CANDIDATE]`、`[CANON_P58_CONTINUE_KV_CLEAN_EMPTY]`、
`[CANON_P58_CONTINUE_KV_CLEAN_PREFIX]`、`[CANON_P58_CONTINUE_KV_CLEAN_READY]`、
`[CANON_APC_M15_B_CONTRACT]`、
`[CANON_P57_TOKEN_CONTINUITY_DEBUG]`、
`[CANON_P57_TOKEN_CONTINUITY_DEBUG_JSON]`、
`[CANON_P57_TOKEN_CONTINUITY_DEBUG_CAPSULE]`、
`[CANON_P57_TOKEN_CONTINUITY_SUMMARY]`、
`[CANON_P57_TITO_HOST_WITNESS]`、
`[CANON_P57_TITO_DIAGNOSTIC]`、
`[P57.TITO.EMPTY_RESPONSE]`、
`[CANON_P38_DURABLE_COLLECTION]`、`[CANON_P38_SEAM_CLASSIFICATION_JSON]`、
PATHTRACE 族(固定树行数 =2×层+1)。
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
CANON_APC_M15_TARGET_DEBUG
CANON_APC_M15_REPLAY_LEDGER
CANON_APC_M15_SEAM_BUNDLE
CANON_BATCHED_EVIDENCE
CANON_DP_COLLECTIVE_REDUCE
CANON_DP_COMPARE_MODE
CANON_DP_DISTINCT_SCHEDULE
CANON_DP_FINITE_FETCH
CANON_DP_REDUCE_ONCE
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
CANON_DEEPSWE_SYSTEM_OPTIMIZATION_ARM
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
CANON_M15_TOKEN_CONTINUITY
CANON_MAX_BATCHED
CANON_MESH_SHAPE
CANON_MINREPRO_N
CANON_MM_ALGO
CANON_MM_ALGO_PRESET
CANON_MODE
CANON_MODEL_DIR_NAME
CANON_NUM_GENERATIONS
CANON_N_LAYERS
CANON_ONEHOST_JAX_CACHE_DIR
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
CANON_P29_LOG_DIR
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
CANON_P32_KEEP_TAPE
CANON_P32_CHUNK_BATCH
CANON_P32_LENGTH_SORT
CANON_P32_LONG_PROMPT_EXAMPLES
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
CANON_P38_KV_OBSERVER_LAYER
CANON_P38_KV_OBSERVER_MAX_BYTES
CANON_P38_KV_OBSERVER_MAX_CANDIDATES
CANON_P38_KV_OBSERVER_MAX_PAGES
CANON_P38_KV_OBSERVER_MAX_READ_BYTES
CANON_P38_KV_OBSERVER_TARGET_PREFIX_SHA256
CANON_P38_KV_OBSERVER_TARGET_PREFIX_TOKENS
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
CANON_P57_TITO_GCS_FINAL_ACK
CANON_P57_TITO_GCS_FINALIZE_FILE
CANON_P57_TITO_GCS_HEARTBEAT
CANON_P57_TITO_GCS_INTERVAL_SECONDS
CANON_P57_TITO_GCS_PREFIX
CANON_P57_TITO_GCS_READY
CANON_P57_TITO_GCS_STOP_FILE
CANON_P57_TITO_GCS_WORKER_LOG
CANON_P57_TITO_ONEHOST_NEUTRALITY
CANON_P57_TITO_ROLLOUT_ONLY
CANON_P57_TITO_RUNNER_WITNESS_DIR
CANON_P57_TOKEN_CONTINUITY
CANON_P57_TOKEN_CONTINUITY_DEBUG
CANON_P57_TRAIN_GEOMETRY
CANON_P57_WORKLOAD_CANDIDATE
CANON_P58_CHECKED_VMA_DIAGNOSTIC
CANON_P58_DEBUG_DIR
CANON_P58_DEEPSWE_TIM
CANON_P58_EXPECTED_UPDATES
CANON_P58_EXPECT_HOSTNAME
CANON_P58_MODEL_SNAPSHOT
CANON_P58_NATIVE_STOCK_PROMPT_OBSERVER
CANON_P58_ONEHOST_SEAM_PROBE
CANON_P58_ONEHOST_XPROF_ARM
CANON_P58_Q4_TP4_CARRIER_SCREEN
CANON_P58_Q4_TP4_CONTINUE_KV_DIAGNOSTIC
CANON_P58_Q4_TP4_CONTINUE_KV_MAX_PREFIX
CANON_P58_Q4_TP4_CONTINUE_KV_MIN_PREFIX
CANON_P58_Q4_TP4_SEAM_DIAGNOSTIC
CANON_P58_Q4_TP4_SHORT_BACKWARD
CANON_P58_Q4_TP4_TRAJECTORY_REPLAY
CANON_P58_Q4_TP4_ZERO_ADMISSION
CANON_P58_R2EGYM_COMMIT
CANON_P58_REPLAY_JOURNAL
CANON_P58_REPLAY_JOURNAL_SHA256
CANON_P58_RUNNER_SHA256
CANON_P58_SEAM_LOCALIZATION
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
CANON_P59_CHECKED_VMA
CANON_P59_REQUIRE_XPROF
CANON_P59_XPROF_BACKWARD_DIR
CANON_P60_DETERMINISTIC_AB
CANON_P61_BACKWARD_NUMERICAL_DIR
CANON_P61_STOCK_GRADIENT
CANON_P62_BACKWARD_NUMERIC_DEBUG
CANON_P63_OVERFLOW_SAFE_CLIP
CANON_P64_MODEL_BINDING_SHA256
CANON_P64_P45_NUMERIC_DEBUG
CANON_P64_TRAINING_CAPSULE
CANON_P64_TRAINING_CAPSULE_GCS_URI
CANON_P64_TRAINING_CAPSULE_MODE
CANON_P64_TRAINING_CAPSULE_SHA256
CANON_P66_BACKWARD_ARM
CANON_P66_BACKWARD_CAPTURE_DIR
CANON_P66_P59_CHECK_VMA
CANON_P67_P66_VMA_P59_ONLY
CANON_P71_SCAN
CANON_P75_REPORT_ADJOINT_BUCKETS
CANON_P76_CHUNK_DEPENDENCY_TICKET
CANON_P77_CHUNK_BACKPRESSURE
CANON_P78_SEGMENTED_ACTOR_LOGPS
CANON_V1_HP_FIRST_UPDATE_GATE
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
CANON_V2_MODEL_BINDING_SHA256
CANON_V2_TRAINING_CAPSULE
CANON_V2_TRAINING_CAPSULE_MODE
CANON_V2_TRAINING_CAPSULE_SHA256
CANON_V1_GSM8K_XPROF_ARM
CANON_V1_FL_TP8_AB_ARM
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

Count: 439 settable names (appendix inventory above; exclusions: none).


## 无 flag 的行为变更(tasks/v1_long_context,2026-09-02/03;均零比特,双几何双门通过)
- **S2c(CL 9e9d50fb)**:分组 tape 不再保留每 chunk 一份整长 KV 快照,只留每组最终 cache;反向按"最终 cache + 位置掩码"重建每个 chunk 的入口 cache(逐字节同一数组:kernel 只写自己的槽、从不读 ≥ chunk_start 的槽)。tape 的 cache 部分从 O(num_chunks × max_model_len) 降到 O(max_model_len)。
- **累加器拼写对齐(CL 89ec5d79)**:首次构建 scaled-step 程序前把全 None 拼写的复制叶子改为 `P()`、单设备 denom 放到 mesh,消除 update 2 的第二次编译(一台机 grad_accumulate 6.9 s → 0.04 s)。
- **胶水定宽(CL 0671cf23)**:packed rows / flat logps / flat 余切按 ceil(max_model_len/bucket)·bucket 定宽,chunk_start 作操作数;warm-up 不再随长度分布重编译。

## 无 flag 的行为变更(tasks/v2_dispatch Phase 1,2026-09-06;dp2-tp2 与 dp2-tp2-long 双门通过,锚与 A=B=C 逐位)
- **chunk 输入一个程序(CL 3c24933f)**:`_p32_group_chunk_inputs` 只剩融合程序 `_fused_chunk_metadata`(chunk_start 作操作数,输出钉 `_input_sharding`),原 eager 臂(每次 61 个小程序、每 chunk pass 122 次)退役为测试 oracle(`tests/rl/test_p32_chunk_inputs.py`);不复用 P56 捆绑旗 `CANON_FUSED_TREE_OPS`。
- **glue 折进 rows 程序(CL 0e10324a)**:`zt_tr_fwd_logprob` 内含 astype/rows/reshape/glue 写入,`zt_tr_bwd_logprob` 内含余切切片/pullback/astype;每组一个余切散布程序;glue 缓冲 sharding 由 `_input_sharding` 派生(CL 6593a22c,引擎 mesh 轴名是 `dp/tp`);P74 gap 普查的窗口 seed 相应改为 `zt_tr_bwd_logprob`(等效门)。
- **组规格一个程序(CL a2bd88da)**、**对齐/余切检查每 update 一次同步(CL 556248aa)**、**stream 余切每组一个程序(CL 0b755719)**:同 base control(640836c6 + 拼写修法)对比,dp2-tp2 每 update 派发 23,146 → 9,006(−61%),warm update 13.08 → 10.57 s;long 19,297 → 7,371,11.44 → 9.28 s;HBM 不变;commit 范数三值与注册锚逐位,alignment hash 96/96、24/24 相同;两发同字节(r4 = r5)。
  收据:`/mnt/disks/tunix-data/gsm8k-onehost-xprof/v1_zero-hp_dp2tp2-v2disp_p1_dp2_20260906_r{4,5}`、`…v2disp_p13c_dp2_20260906_r1`、`…dp2tp2long-v2disp_p1_long_20260906_r1`;control `…dp2tp2-v2disp_basefix_dp2_20260906_r3`、`…dp2tp2long-v2disp_basefix_long_20260906_r1`。dp4-tp1 `Not verified because base 640836c6 cannot run dp4-tp1(融合累加 TP1 VMA,sibling lane)`。
- **融合累加签名守卫的拼写规范化(CL 53a3cb68、bc46edfc)**:base 回归修法——commit 后复制/部分分片叶的 spec 拼写变化(`P(None,)`→`P()`、`P(None,'tp',None)`→`P(None,'tp')`)不再被当作布局变化;真实 shape/dtype/布局变化仍抛错。无刀 base 在 update 2 必红(`…v2disp_base_dp2_20260906_r1`)。
- **普查等效门**:模块普查把 `jit__precomputed_gradient_adopt_scaled_step` 计为 scaled step(CL 605d961b);派发计数普查 `census_dispatch_count.py`(CL c1411572)按家族给每 update 派发数,是本 lane 每把刀的第三判据。

## 无 flag 的行为变更(tasks/v2_dispatch Phase 8,2026-09-06;dp2-tp2 一发认证,锚与 A=B=C 逐位)
归因捕获(`…v2disp_p7r3_pytrace_dp2_20260906_r2`,`attribute_pytrace_launches.py`)把认证栈每 update 的 6,734 次派发按最内层 Python 帧定位后,六把纯搬运/索引/cast/元数据的刀,全部无 flag:
- **learner sidecar 行在 host 切片(CL 394d6547)**:`_host_microbatch_views` 每 update 把 sidecar 与 per_token_logps 各叶 D2H 一次,microbatch 循环用 numpy 索引(原每叶设备 gather 8 次 × 每组,1,760 次/update);`check_batch` 本就哈希 host 字节,行逐字节相同。
- **spec 数组 commit 到引擎行 sharding(CL 83c9f98f)+ grouped_inputs 一个拆分程序(CL 49259e83)**:`group_pack`/`_p32_split_groups_fn` 的输出钉 `_input_sharding` 派生的 rows/scalar sharding,消费者不再每次调用 `shard_args` 切片(普查里的 `jit__multi_slice` 655 → 131)。
- **权重同步 cast 一个程序(CL 72a85216)**:`_p32_cast_leaves`,每叶 out_sharding 钉原 sharding(310 → 1);峰值 HBM +1.6 GiB(输出与 fp32 源同时存活),在 +5% 门内。
- **mapped pullback 叶每 update 重贴一次(CL e100c9c4 + 修法 569c978d)**:`prepare_block_pullback_group` 在 mapped 程序建好后按 engine_leaves 元组身份缓存对齐后的叶(bootstrap 组保持 trainer 拼写——否则构建器把 P59 mesh 当 trainer mesh,`…v2disp_p8_dp2_20260906_r2` 的两 mesh 红);host `shard_args` 60,096/update 预计 → ≈19k,待归因捕获量。
- **stream loss 输出与 scale 一个程序(CL cd90f033)**:`_p32_stream_loss_fn`,scale(mask 计数 + 一次除法)精确,loss 值只作报告;eager batch pullback oracle 不动。
  收据:`…dp2tp2-v2disp_p8_dp2_20260906_r3`(lane 569c978d)vs control `…v2disp_p5b_dp2_20260906_r2`:派发 **6,734 → 3,680(−45%)**,warm update **9.60 → 7.55 s(−21%)**,HBM 38.26 → 39.87 GiB(+4.2%),锚三值逐位,A=B=C 96/96,普查 GREEN(classifier 只余惯例 trace 红)。r1 被 codex TiTO 链的 launcher kill(exit 137,docker events),r2 为 K5 bootstrap 顺序 bug,目录均保留。

## 无 flag 的行为变更(tasks/v2_dispatch Phase 9,2026-09-07;dp2-tp2 一发认证,锚与 A=B=C 逐位)
- **mapped pullback 程序钉操作数拼写(CL 2b298a91)**:TP>1 时 `_p59_parallel_map` 给程序钉 in_shardings(样本操作数的拼写)与 out_shardings(译成 trainer 拼写的 out_specs),invoke 不再每次 device_put 贴入/贴回(每层 22 次、每 update 40k 次 host 拷贝);Phase 8 的叶预重贴随之删除。两 mesh CPU 单测(`tests/rl/test_p59_pinned_map_shardings.py`)逐字节、0 次重贴。
- **forward 尾部 gather/mask 一个程序 + chunk-start 标量按值缓存(CL cd151a1f)**:`_p32_group_rows_fn`、`_p32_start_scalar`(416 → 32/update)。
- **每组 replay 相等检查一个程序(CL cb3fcc20)**:`_p32_pair_equal`(equal+all 一个 jit,每组一次;数组随流式 tape 释放,不能推迟到 update 末尾)。
  收据:`…dp2tp2-v2disp_p9_dp2_20260907_r1`(lane 2b298a91)vs control `…v2disp_p8_dp2_20260906_r3`:派发 **3,680 → 3,200**,warm update **7.55 → 5.48 s(−27%)**,HBM 39.87 → 39.48 GiB,锚三值逐位,A=B=C 96/96,普查 GREEN。归因捕获 `…v2disp_p9_pytrace_dp2_20260907_r1`(同码 + tracer):host `shard_args` 来源 `align` 40,692 → 64/update,全部 host 拷贝 62,285 → ≈1.3k。累计(dp2-tp2 同 base 链):23,146 → 3,200 次/update,13.08 → 5.48 s。
## 反向侧 fp64 仲裁(tasks/v2_dispatch Phase 13,2026-09-07;诊断,不改数值路径)
封存的反向嵌套类(p7 跨层、p10 跨 chunk)与认证路径谁离 fp64 更近。载具:P61 full-tree 捕获(`V2_P0_CAPTURE_FULL_TREE=1`,认证 recipe 本就 check-VMA 开)加 `example`/`logps`/`algo_config.json`(CL dd51843e,只在 `CANON_P61_BACKWARD_NUMERICAL_DIR` 下生效);参照 `canon-zero-tim/tests/p61_backward/fp64_reference.py`(进程内 `jnp.float32` → float64 重绑、参数树按 manifest 灌入并可 `--params-from` 同哈希顶替、zero-TIM 点求值 old := 自身 logps、`--also-dtype bfloat16` 同进程精度孪生)。捕获发:A = lane 82ce7a7e + 捕获(`…v2disp_p13_cap_lane_20260907_r1`,update-1 锚 1.6838101148605347 逐位)、B = p10 e635e7c5 + 捕获(`…v2disp_p13_cap_p10_20260907_r1`,update-1 锚 1.6850484609603882 = p10 r1 逐位);捕获载具在 update 1 之后确定性偏离(与 Phase 0 捕获同值),只比 update 1。证据:`tasks/v1-gsm8k-onehost-xprof-pair/evidence/p13_fp64_arbitration_20260907/`。
| 比较(update-1 梯度,310 叶) | rel_l2 | one_minus_cos | 范数比误差 |
|---|---|---|---|
| A(认证)↔ fp64 | 0.2122 | 2.18e-2 | 6.5% |
| B(p10)↔ fp64 | 0.2124 | 2.19e-2 | 6.5% |
| A ↔ B | 1.30e-2 | 8.4e-5 | 7.4e-4 |
| bf16-CPU 孪生 ↔ fp64 | 0.1265 | 7.9e-3 | 1.1% |
| A ↔ bf16-CPU 孪生 | 0.260 | 3.3e-2 | 7.6% |
结论:参照语义成立(孪生 0.13、logps 与引擎只差 bf16 噪声);B 相对真值与 A 差 0.1%,A↔B 的 1.3% 远小于两者到真值的 21%;§5.1 的 1e-5 相对范数线量的是舍入序列的可复现性而非精度。**重钉规则**(用户 2026-09-07 批准,正本 `tasks/v2_dispatch/GOAL.md`):fp64 oracle 等距(整体 ×1.05、叶类 ×1.05、叶组 ×1.10、new↔A ≤ 2×两者到真值)+ 护栏(new↔A 整体 one_minus_cos ≤ 1e-3、叶组范数比 ±2%)+ 新程序自身锚两发同字节 + 其余门不动;只管梯度侧。B 按校准判据通过,按原预注册字面在 4/114 个 128 元素 layernorm 小组的 one_minus_cos 比超 5–9%(两种读法并存,校准由用户裁定)。**未决发现**:A 的梯度范数比 fp64 系统性小 6.5% 且随层深衰减(layer 27 ≈ 0.99 → 底层 0.86–0.94),bf16-CPU 孪生 +1~6% 无衰减 ⇒ 引擎反向余切逐层系统性衰减,Phase 13b(stock TPU 反向捕获)分类是 TPU bf16 共性还是 canonical 特有;canonical 特有则先修再默认落地。

### 13b/13c:分类与机制搜索(2026-09-07;诊断,不改数值路径)
同批、同参数、同 zero-TIM 评测点再加两方:stock TPU 反向 C(`CANON_P61_STOCK_GRADIENT=1`,学习器在 canonical update 前用 stock nnx 模型的 XLA autodiff 算同一 loss 的梯度;`…dp2tp2-v2disp_p13b_stock_dp2_20260907_r1`)与 f32 累加器试验 T(分支 `local/v2-dispatch-p13c-f32acc` @ 34948e9f:chunk 间叶梯度累加与 tied 合并改 f32,forward 不动;`…dp2tp2-v2disp_p13c_f32acc_dp2_20260907_r1`,A=B=C 3/3 update 0 差异字节)。
| 比较(update-1 梯度,310 叶) | rel_l2 | one_minus_cos | 范数比 |
|---|---|---|---|
| A(认证)↔ fp64 | 0.2122 | 2.18e-2 | 0.9347(逐层 0.99 → 0.86 → 0.92) |
| C(stock TPU)↔ fp64 | 0.1453 | 1.06e-2 | 0.9944(逐层平) |
| T(f32 累加器)↔ fp64 | 0.2122 | 2.18e-2 | 0.9347 |
| A ↔ T | 2.19e-4 | 2.4e-8 | 1 − 1e-6 |
分类:**canonical 特有**(C 无衰减)。forward 数值点无偏:A、C 的 logps 相对 fp64 有符号均值 +6.0e-4 / −5.6e-4(mean|d| 0.0106 / 0.0108,16,233 个 action token)。机制搜索(fp64 参照 + 单点 bf16 仿真,`evidence/p13_fp64_arbitration_20260907/mechanism_probe.py`,每探针 A↔probe 的 rel_l2 / 范数比):softmax 反向 bf16 0.2122/0.9347、注意力核心反向 bf16 0.2119/0.9348、分块 + bf16 K/V cache 0.1944/0.9589、同分块 cache 不舍入 0.2122/0.9327、只舍 forward 的 K/V 0.1945/0.9590(⇒ 分块探针的 −2.5% 全是 forward 舍入的仿真伪象,反向 dcache 载体无贡献)、跨 chunk 停梯度 1.46/1.78(该路径占梯度一半,A 完整保有)、每层残差余切 bf16 0.1951/0.9594(= 分块探针)、每行每 chunk bf16 叶梯度累加 0.2122/0.9347(= 基线);TPU 试验 T 同上。结构 trace(seam、head pullback、dcache 路由、累加顺序与缩放)无丢项、双缩放或零化。**结论**:可仿真与可试验的反向载体全部排除,canonical 的梯度是引擎 bf16 forward 函数在其(A=B=C 钉住的)数值点上的精确 VJP,与 stock 的差别源于 forward 函数本身;未找到可修之处 ⇒ GOAL 的"修好并重钉后再进 15/16 默认落地"前置未满足,交用户裁决(`tasks/v2_dispatch/phase13.md` §13b/13c 定谳)。收据:`metrics_stock_tpu_backward.json`、`metrics_f32acc_trial.json`、`probe_*.json`、`forward_bias_logps.txt`。

## 默认位与 fresh 认证(tasks/v2_dispatch Phase 16,2026-09-07;v2_default 顺序 4 清一版)
默认位 CL b16f2f51:`CANON_P71_SCAN` 缺省/空 → `fwd_block`、`CANON_P32_CHUNK_BATCH` 缺省/空 → 2(选 `CANON_P28_LAYER_SCAN` 档时二者缺省回旧路径,互斥;显式 `0`/`off`、`0`/`1` 回旧路径);模块普查空值同映射;launcher 不变(空值即默认)。收口:本 lane 无可删死路径(逐层 pullback 循环 = chunk 程序的 bootstrap 与 oracle;逐 chunk/逐层 forward = 显式旧路径),封存分支 p4/p7/p10 与诊断分支 p14diag、p13c-f32acc 保留不并入;更广的 stream 收口属 v2_default §6。
fresh 认证(EXTRA_ENV 只有 `CANON_DP_REDUCE_ONCE=1`,tape `stream`,launch worktree @ 6f62d236):
| 发 | 派发/update | 锚(judge mode reduce-once+chunk) | A=B=C / 普查 | HBM 峰 | warm reverse |
|---|---|---|---|---|---|
| `…dp2tp2-v2disp_p16_dp2_20260907_r1` | 736 | 逐位 [1.6838672161102295, 3.302103281021118, 1.8242988586425781] | 96/96 · GREEN | 39.21 GiB | 4.963 s |
| `…dp2tp2-v2disp_p16_dp2_20260907_r2` | 736 | 逐位(= r1) | 96/96 · GREEN | 39.21 GiB | 4.980 s |
| `…dp2tp2long-v2disp_p16_long_20260907_r1` | 526 | 逐位 [3.650982141494751, 2.332655191421509, 4.374945640563965] | 24/24 · GREEN | 40.22 GiB | 6.788 s |
| `…dp2tp2long-v2disp_p16_long_20260907_r2` | 526 | 逐位(= r1,两发同字节;N_action 逐 update 与 P15 long 相同) | 24/24 · GREEN | 40.22 GiB | 6.877 s |
classifier 四发只余惯例 `trace_census_rc=1`。**全链(dp2-tp2 同 base,派发/update · warm reverse)**:23,146 → Phase 3–5 6,734 → Phase 8 3,680(6.74 s)→ Phase 9 3,200(4.67 s)→ Phase 11 3,040(4.64 s)→ Phase 14 K14c 3,040(5.33 s,long HBM 43.08 → 39.96)→ **Phase 15/16 736(4.96 s)**;long:7,179 → 4,789 → 3,071 → 3,007 → **526**(7.20 → 6.79 s,HBM 40.2)。几何与容量:8B 一台机 **CAPACITY_REJECT**(phase7.md 算术表);**dp4-tp1 Not verified because 本 lane 六个 phase 只在 dp2-tp2 与 dp2-tp2-long 认证,dp4-tp1 未发**(judge 的 dp4-tp1 锚仍是旧路径的);dp1-tp4 载具未落地,Not verified。梯度质量基线:用户 2026-09-07 裁决接受认证路径 A 为 fp64 基线(A↔fp64 0.2122 / 范数 −6.5%,canonical 特有、机制未定位,FLAGS.md §13b/13c),Phase 15 的新程序与 A 等距(0.2128)。缺点:Phase 14 的两次 chunk 边界等待留下的 wall 代价没有全部收回——dp2 warm reverse Phase 11 4.64 s → K14c 5.33 s → Phase 15/16 4.96 s(净 +0.32 s),long 6.55 s → 7.20 s → 6.79 s(净 +0.24 s),换来 long 的峰值 HBM 从 43.08 回到 40.2 GiB;默认位落地与 push 待用户批准。

## 无 flag 的行为变更(tasks/v2_dispatch Phase 15,2026-09-07;dp2-tp2 三发 + long 一发,锚按 fp64 规则重钉)
反向 chunk 程序(Phase 7 的类在认证 lane 上重做,CL c61a7497):kept tape 上的 rank-parallel 反向每个 chunk 的整段反向体——chunk inputs、entry-cache 重建、norm/head 重算、rows pullback、头余切分区、head/norm/28 层/embed 的映射 pullback——trace 成**一个**程序 `zt_tr_bwd_chunk`,在图内按原样调用已建好的映射程序(shard_map 原样嵌套、不重做块体),叶/tape/cache/余切全作操作数,输出 sharding 钉到 bootstrap chunk 记录的拼写,每个原程序边界一个 `optimization_barrier`;组内首个反向 chunk 是第二个编译变体(dcache 零在图内造);进程第一组的第一个 chunk 走逐程序路径建映射程序并记录 sharding(bootstrap),之后每 chunk 一个程序。无 flag:准入 = rank-parallel ∧ kept tape ∧ 非 P71 block ∧ 非 P66 arm ∧ 无 HBM sink ∧ 无 P76 ticket(Phase 14 的两次 chunk 边界等待照常,两条路径共用 `finish_chunk`);`chunk_program=False` 为测试 oracle。普查等效门:模块普查 `--p32-reverse-chunk`(launcher 从源码 grep)按 `bwd_chunk` 家族计数、折入的程序不得单独出现;P74 gap 普查改计 chunk 程序;classifier 读 receipt 的 `reverse_chunk` 变体(CL 2df3b398)。CPU 门:`tests/rl/test_p32_reverse_chunk.py` 独立进程 `--xla_cpu_max_isa=AVX`(XLA:CPU 的 FMA 收缩在独立程序与 chunk 程序里不同)逐字节 = 逐程序 oracle,每 chunk 1 次 `bwd_chunk`;邻居门 54 + 17。
收据(dp2-tp2,配方 `stream` + `CANON_DP_REDUCE_ONCE=1 CANON_P32_CHUNK_BATCH=2 CANON_P71_SCAN=fwd_block`,control = K14c 认证发 3,040 / 38.27 GiB / warm reverse 5.327 s):
| 发 | 派发/update | 锚 | HBM | warm reverse |
|---|---|---|---|---|
| measure `…v2disp_p15_dp2_20260907_r1` | **736**(每 pass 11.5:bwd_chunk 1、grad_add 1、group_glue 1、glue_eager 3.5、other 5) | 1.6838672161102295 / 3.302103281021118 / 1.8242988586425781(相对 A +3.4e-5 / −1.5e-4 / +2.1e-3) | 39.21 GiB(+2.5%) | 4.939 s |
| 同字节复跑 `…_r2` | 736 | **= r1 逐位** | 39.21 | 4.958 s |
| 捕获 `…v2disp_p15_cap_dp2_20260907_r1` | — | update-1 = r1 | 39.21 | 4.934 s |
A=B=C 96/96 三发、hierarchy/modules/P74 gap 全 GREEN。**fp64 重钉规则**(GOAL.md,A 为参照;评测点 zero-TIM,`--params-from` 同哈希树):new↔fp64 rel_l2 **0.2128**(A 0.2122,比 1.003)、one_minus_cos **2.195e-2**(A 2.181e-2,比 1.006)、范数比误差 6.525e-2(A 6.528e-2);new↔A rel_l2 1.255e-2、one_minus_cos 7.87e-5、范数比误差 3.4e-5;叶类比 ≤ 1.0035、叶组 rel_l2 比 ≤ 1.036、one_minus_cos 比 ≤ 1.085;组范数比 [0.9886, 1.0112] ⇒ 规则①②③④全过,锚重钉入 judge(mode `reduce-once+chunk`,runs r1/r2)。累计(dp2-tp2 同 base 链):派发 **23,146 → 736(−96.8%)**、warm reverse(K14c 起)5.327 → 4.94 s;lane 从 13.08 s/update 起算的 warm 见 Phase 16 fresh 认证。
**long 一发** `…dp2tp2long-v2disp_p15_long_20260907_r1`(p8_launch @ 2df3b398):派发 **526/update**(每 pass 7.74;control K14c 3,007;预注册 ≈520,440–600 ✅);锚 3.650982141494751 / 2.332655191421509 / 4.374945640563965(update 1/2 相对 A −6.2e-5 / −2.4e-4;update 3 的 +61% 是 rollout 换批:N_action 10,367 vs 10,896,P74 计 68 窗 vs 69——策略微差让采样 token 翻转,长序列后续全变);A=B=C 3/3;普查 GREEN、classifier 只余 `trace_census_rc=1`;HBM **40.97 GiB**(K14c 39.96,+2.5%,≤ 41.96);warm reverse **6.808 s**(K14c 7.202,−5.5%)。long 的两发同字节由 Phase 16 的 long 两发承担(同代码);judge 已钉 `reduce-once+chunk` 两几何的锚(run ids 见 judge 注释)。
缺点:锚不再与 A 逐位(+3.4e-5 量级,fp64 等距证明为同类噪声);每 chunk 一个大程序的编译时间(cold update)与两个变体各一次编译;证据 `evidence/p15_reverse_chunk_20260907/`(fp64 四方 slim JSON、`repin_check.py`、两发计数 JSON)。

## 无 flag 的行为变更(tasks/v2_dispatch Phase 14,2026-09-07;long 一发认证 + dp2 回归,锚与 A=B=C 逐位)
Phase 8/9 的 host 胶水刀之后 dp2-tp2-long 的峰值 HBM 从 39.39 升到 43.08 GiB(+9.4%),dp2-tp2 +3.2%。三发二分(`phase12.md`)把它归到 K5 家族(`…v2disp_p12_long_bisect_{72a85216,bcdefa3d,4eb128c1}_r1` = 39.39 / 43.10 / 43.10),但 K7 删掉 K5 的 relabel cache 后峰值不降;lane tip 素发 `…dp2tp2long-v2disp_p14_long_tip_20260907_r1`(锚逐位、3,007 派发、warm reverse 6.545 s)仍 43.08,而每个 update 的常驻剖面(before / after_accumulation / after_commit 的 in_use)与 K5 前**完全相同** ⇒ 差的 +3.6 GB 是**纯瞬态**:去掉每次 mapped 调用的 22–29 个 host device_put 后 host 快了 ~27%,整 chunk 的层 pullback 排在设备前面,PJRT 在派发时就分配它们模型量级的输出;时间线诊断发(diag 分支 cb374d81,每 chunk 三个同步点)把 update 1 的峰从 41.24 压到 38.96 印证了这一点;CPU 两 mesh 夹具上 K7 钉拼写的程序 HLO 无 copy(`evidence/p14_long_hbm_20260907/hlo_pinned_map.py`)。
刀(无 flag、只等待、不读值):
- **K14(CL c12363c5)**:GSM8K 载体(`CANON_P32_WORKLOAD` 非 frozenlake、P76 ticket 关)的 rank-parallel 反向在每个 chunk 固定 P77 的两次 readiness 等待(whole-tree start/add 之前等 chunk_pack,add 与 consumed-pack 释放之后等 grad_pack);P77 选择器、frozenlake-only 准入与 `[P77.CHUNK_BACKPRESSURE]` 收据保持只随 flag。门 `tests/rl/test_p32_chunk_backpressure_default.py`(16 设备真反向:每组 2×num_chunks 次等待、frozenlake 0 次、梯度逐字节)。
- **K14b(CL 63d349b1,已 revert 20a56154)**:post-add 等待前预派发下一 chunk 的 inputs/entry-cache 程序——两发 wall 与 K14 相同,无收益。
- **K14c(CL c6a5f476)**:`_p77_wait_for_chunk_completion` 只等 `jax.tree.leaves(completed)[0]`(chunk_pack 首叶 = 最后派发的 embed pullback 输出;grad_pack 首叶 = add 输出;设备按派发顺序执行,一个就绪即整 chunk 就绪),省去 310 叶逐叶的 PJRT 往返。
收据(锚三值逐位、A=B=C 逐位、普查 GREEN、派发计数 = control):
| 发 | HBM 峰 GiB | warm reverse s | 派发/update |
|---|---|---|---|
| tip `…v2disp_p14_long_tip_20260907_r1` / P11 dp2 | 43.08 / 39.51 | 6.545 / 4.642 | 3,007 / 3,040 |
| K14 `…v2disp_p14_k14_{long,dp2}_20260907_r1` | 39.96 / 38.27 | 7.560 / 5.644 | 3,007 / 3,040 |
| K14b `…v2disp_p14_k14b_{long,dp2}_20260907_r1` | 39.96 / 38.27 | 7.639 / 5.662 | 3,007 / 3,040 |
| **K14c** `…v2disp_p14_k14c_{long,dp2}_20260907_r1` | **39.96 / 38.27**(long ≤ 41.36 ✅) | **7.202 / 5.327** | 3,007 / 3,040 |
缺点:每个 chunk 两次排空把此前藏在设备队列后的 host 工作暴露出来,warm reverse long +0.66 s(+10%)、dp2 +0.69 s(+15%)/update(idle_per_pass 12.9 → 18.9 ms;CPU 上签名 1.2 ms、release 0.3 ms,余下是排空后首个程序的派发延迟与 add 本身)。follow-up:有界深度(允许一个 chunk 在途)或只留 pre-add 等待,需再各一发验证 HBM。诊断分支 `local/v2-dispatch-p14diag`(时间线)不并入;计数 JSON 与脚本在 `evidence/p14_long_hbm_20260907/`。

## 无 flag 的行为变更(tasks/v2_dispatch Phase 11,2026-09-07;dp2-tp2 一发认证,锚与 A=B=C 逐位)
Phase 9 归因捕获剩下的 3,200 次/update 里可折的 host 胶水、零值与放置,三把刀全部无 flag、零数值风险:
- **每组零值程序合一(CL c0a4fa1b)**:`_p32_forward_zeros` 一个程序产出 fresh caches 与两条 glue 零(out_shardings = 各自生产者钉的 `_cache_sharding` / `_p32_glue_sharding`;CPU 夹具无几何时回退原两程序);反向首个 chunk 的 entry 重建 `_p32_entry_caches(zero_carry=True)` 顺带产出 dcache 零,`_p32_zero_trees` 不再每组调用。96 → 32 次。
- **host 行长度进每组 spec(CL e0d7b1ba)**:update 开头对两条 mask 各 D2H 一次(紧跟已有的 action-mask 子集同步)求行和,按 split 程序同一 rank-major 分组后作 `host_prompt_length/host_completion_length` 传入,每组的 `group_lengths` 程序与 `device_get` 同步消失;pack 程序仍产 `lengths_match`,update 末一次同步校验(不匹配 fatal)。−32 次、−32 次同步。
- **train example 一次 commit 到引擎 mesh(CL 7ac9338b)**:`_p32_commit_example` 把 loss 读的 9 个叶 replicated 放到引擎 mesh(新对象,调用者的 example 不动),`_p32_group_index_scalar` 给 stream_step 一个按值缓存的 committed 组索引;stream/oracle/split 每组的 `jit__multi_slice` 与 host 拷贝消失。
  收据:`…dp2tp2-v2disp_p11_dp2_20260907_r1`(lane 7ac9338b)vs control `…v2disp_p9_dp2_20260907_r1`:派发 **3,200 → 3,040**(`jit__multi_slice` 67 → 2、`jit__lambda` 96 → 32、`jit_group_lengths` 32 → 0、`jit_fwd_glue_zeros` 32 → 0、`jit_forward_zeros` 32),锚三值逐位,A=B=C 96/96,普查全 GREEN(classifier 只余惯例 trace 红),HBM 39.48 → 39.51 GiB(+0.1%),warm 5.48 → 5.44 s。反向侧嵌套两类(Phase 7、Phase 10)封存后,dp2-tp2 每 pass 47.5 次里 28 次是必要的 mapped pullback,剩余为每 chunk 一次的命名程序与每组的累加/检查。

