# FLAGS.md — CANON_* 注册表

> 政策:**建 flag 自由但必须登记(带日落条件);删 flag 有序按日落执行。**
> 焊死数值类 flag = 删代码路径 = 程序变更,走与开启同级认证门(verify+ALIGN+canary)。
> 生命周期档位:试验 → 已认证 → 默认开 → 焊死(开关可删)→ 退役/否决。
> 普查基点 a94d6c0c(285 个可设置 env flag,与 ebba4850 普查零漂移);普查后续现役附录
> 当前 409 个;本表分层登记,D 层按前缀组、语义欠账标"待考古"。
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
| CANON_VLLM_ENABLE_PREFIX_CACHING | Phase3 APC:仅改变 A rollout 的 vLLM prefix-cache 读取路径；B rescore 继续固定 `reset_prefix_cache=True` 全量重算 | off；缺省/空/0 均关，仅 1 开；三个 production full recipes 当前统一 off | 试验；Qwen3-8B DP1×TP4 G-A/G-B/G-C/G-D、脏页阴性与匹配性能/XProf 已绿；M15 DP8×TP8 `m15i` G-E 在 A−B 红 1389 bytes/760 elements、B−C exact，故 target 修复前禁止 production APC | fresh target carrier 完成复现/首红定位/最小修复，随后对应 workload G-E 与脏页负控全绿后逐项转正；认证不可跨 workload 继承 |
| CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY | FrozenLake有限差异的可观测训练策略。历史 Native/IS full 继续保留 broad warning；优化 Zero 仅精确 P45（candidate/split 均 absent）或 M15/main v1-hp、DP8×TP8、300-update、no-eval/no-checkpoint concept run 可开，且只将有限 `S_decode_vs_S_prefill` 与由其直接派生的 `w`/`wr`/clip/TIS 差异降为 warning。`S_prefill_vs_T_old`、`T_old_vs_T_current`、`r`、任意 nonfinite、梯度、副本与 optimizer transaction 始终 fatal | off；renderer 对精确 P45/M15 Zero concept arms 写 1 | 临时收敛曲线逃生阀；host/exact-image admission 已覆盖两条 identity，DP8×TP8 target 未跑。开启后只能声明 `convergence-only / alignment-degraded`，不得声明 Zero-TIM | 分别修复 P45/M15 carrier 并完成 strict 300-update target 后恢复 0；失败与 warning 剂量证据永久保留 |
| CANON_M15_TOKEN_CONTINUITY | M15 later-turn token continuity selector。`verify` 仅观察 serving 实际 prompt IDs；`exact` 为 later turns 直接提交 initial tail + sampled assistant IDs + nonterminal environment IDs，关闭 chat-template 重应用并逐 token 回验。exact 任一不等立即 fatal。签名身份有三类：M15 APC debug DP8×TP8 off/on 的 layer/backward-no-commit 载具只准 exact；one-host DP1×TP4 允许 APC-off legacy verify、APC-off/on exact；P67 full 只准精确 M15/main Zero v1-hp DP8×TP8、300-update、no-eval/no-checkpoint exact。malformed/missing/negative arrays hard-fail | 全局 absent/off。P67 默认不写；只有显式 `--m15-tito-exact` 才写 exact。P45 永远 absent。APC debug 与 one-host 必须满足各自完整身份，禁止与 target 混用。空值、`0`、full verify、P45/GSM8K/Native/IS/eval、其他拓扑均 fatal | 历史 exact-default `3fc7ef8b` 已撤出 production default。Legacy r7 APC-off 17/17 equal；matched exact r8 17/17 equal，三轮 strict 且与 r7 prompt/trajectory hashes 全同。远端 E0v/E0w one-host APC-off/on exact pair 同样全零并记录 APC-on hits；这些均不继承到 DP8×TP8。P67 full exact 与 APC debug target 均未跑 | 默认 full 必须证明 selector absent且零 receipt；显式 exact M15 full 必须逐 prompt equal、env receipt 恰一。APC debug/one-host 按各自 classifier 判决；DP8×TP8 首次启用仍是独立 target gate |
| CANON_PALLAS_{CANONICAL_VJP,ALL_PROJ,ALL_RMSNORM,MPAD,SWIGLU,SWIGLU_MPAD} | canonical Pallas 内核族选通 | off | 已认证 | 转正焊死(P22.XI 部分已无条件) |
| CANON_P28_SEGMENTED_TRAIN | 分段 fixed-M 训练前向；默认 production clipping 继续使用 stock `optax.clip_by_global_norm`。Attempt-7 P62 no-commit 载具额外打印 element-finiteness、naive/max-scaled L2、DP/TP reduction 与 accumulator receipt；G5b 已证明 16/16 groups 与最终 accumulator 全 finite，旧 `norm=inf` 是 FP32 sum-of-squares overflow | off | 历史 segmented 路径已认证；P62 DP16×TP4 G5b target 仅认证 finite backward 与零 commit，未认证 optimizer transaction | 默认路径不变；仅精确 P63 full profiles 可启用 hybrid clip，首次真实 commit 与完整 horizon 仍是 target gate |
| CANON_P59_RANK_PARALLEL_BACKWARD | 每个 trajectory group 的 DP rank-local VJP 从 host 逐-rank 串行改为一次手动 `shard_map`;TP1 保留 DP-manual/unit-TP carrier，TP>1 在同一物理设备上改用 engine `data/model` 二轴词汇并令 DP+TP 均 manual，使 inner engine shard_map 复用已绑定 TP collective；processed-logprob VJP 产出的 full logical-vocab cotangent 在 head VJP 入口显式约束为 `P(data,model)`，随后 fixed-head 只消费 TP-local vocab；projection 与 attention 的 P59-local 边界均只由精确的双 manual-axis context 选择，RPA 不再二次扩展已经 TP-local 的 GQA K/V；replicated-input TP hidden cotangent以 FP32、升序 rank、逐项 operand barrier 累加后只在边界 cast 一次；leading-DP 暂存后仍走原 fixed reduction，group 顺序不变 | off；仅显式 V1/P58.7 high-performance full profile 开 | **gradient-correctness KEEP / DP4 PERF KEEP / Attempt-3 repair one-host mechanism PASS / exact-image+target pending**:ordinary-JAX FP64 oracle relL2 `3.91e-16`，真实 Qwen 梯度 relL2 `1.582%` 过冻结梯度门，DP4 reverse 3.605x；串行与并行 AdamW 首步 delta relL2 `9.976%` 是已披露 trajectory difference。Attempt 3 的 GSM8K `g64m` 与 P45 `f45m` 均在 step-0 strict pre-alignment 逐字节全同且 0 FAIL，随后分别在 TP4/TP8 证明 attention 入口重复扩展已 local 的 KV；追加 patch 25 只在精确 P59 manual DP×TP context 跳过该扩展并强校验 local Q/K/V/cache。M15 `m15m` 在更早的独立 token-contract 门停止，不构成 P59 数值判决。host V1 21/21、P57 144/144、P59 34/34、APC 31/31、flags 366/366 通过；真实 v5p `DP2xTP2` RPA forward+VJP2、wrong-cache negative 与普通 `DP1xTP4` GQA control 通过且零 optimizer commit。installed-attention DP2×TP4/TP8 pinned-image 正负控与真实 DP16×TP4/DP8×TP8 optimizer commit 仍未验证 | Phase4 三个 full target 与 P58.7 full 归档后按 workload 转正；任一 real ALIGN FAIL 立即退役；全局默认仍 off |
| CANON_P59_CHECKED_VMA | Production selector for the P66 checked-VMA repair. Registered Phase4 full contexts and the exact P58 Qwen3-4B strict Zero-HP full profile may set it; `00_env.sh` validates the closed workload/stage/arm geometry and derives the historical P66 implementation spelling as an internal compatibility alias. It changes only P59 backward VMA ownership and does not alter serving forward Zero-TIM or reduction order | off; registered full profiles only; P58 requires Zero/full/1,000 updates, DP8×TP8, strict alignment, and the complete HP bundle | P66 G1 causal PASS and G1.5 six-endpoint same-point ordinary-JAX oracle PASS; P58.11 construction validation in progress and target optimizer/convergence not run | each registered full horizon plus target receipts green, then fold into the final P59 production identity or retire per workload on any numerical red |
| CANON_P59_DP4_SERIAL_MESH_BRIDGE / CANON_P61_BACKWARD_NUMERICAL_DIR | P59/P61 DP4 代理与 full-tree 数值载具，不是生产 recipe | off/空 | 载具完成；历史 serial/update 差异永久保留 | P59/P61 证据交付后退役载具，生产 profile 禁止开启 |
| CANON_P66_BACKWARD_ARM / CANON_P66_BACKWARD_CAPTURE_DIR | P66 backward 诊断总开关：保留 DP4×TP1 `ordinary|segmented` 整树载具，并增加 one-host 完整 28 层 DP1×TP4 `tp4-serial|tp4-p59-old|tp4-p59|tp4-gather-off` 因果臂；G1.5 `tp4-vma-oracle` 在同一 checked-VMA candidate 之后旁路调用 ordinary serial pullback，比较 head/norm/layer27/14/0/embed 的参数、activation 与 cache cotangent，serial 结果绝不回灌；全部零 optimizer，TP4 臂只反传 group0，`grad_norm>1e6` 直接红停 | 空/off；仅 P66 wrapper 设置，生产 profile 禁止开启 | DP4×TP1 已跑；TP4 G1 verdict `H1_VMA_SUPPORTED`，P/R exact、U `1.5402e21` expected-red；final-source G1.5 host 16/16、pinned `2x37/37`、one-host 17/17 与六 endpoint/observer-neutrality 全 PASS；target 未跑 | G2 target 依赖闭包完成后退役，失败及分类证据永久保留 |
| CANON_P66_P59_CHECK_VMA | P66 结构/修复诊断：把 P59 外层 `shard_map` 从历史 `check_vma=False` 切到官方 VMA 复制一致性检查；checked 路径把 DP-local 参数显式标成 varying，并让 VMA 转置拥有 TP replicated-input `psum`，避免 fixed-head/projection 再手工归约一次 | 0/off；仅 P66 focused probe/`tp4-p59*|tp4-vma-oracle` 设置 | G0 pinned-image DP2×TP4/TP8 真实 shim全绿；完整 28 层 G1 one-host 修复臂 finite、距 serial norm `0.1112%`，固定 gather 被排除；G1.5 六 endpoint 最差 rel-L2 `0.5257%` 且 observer-neutrality exact；target 未跑 | P59 TP 复制/归约语义在 target capsule 门通过后，再决定替换生产默认或退役 |
| CANON_P67_P66_VMA_P59_ONLY | P67 serving 程序同一性修复：当 P66 checked-VMA 进程级 alias 开启时，只允许精确 P59 outer manual `data/model` pullback 消费 pcast/Pallas out-shape/RPA out-shape/embed invariant 登记；ordinary serving decode/prefill 保持历史图。它不关闭 P59 backward 修复，不改变数学值或 fixed TP reduction order，也不放宽 alignment gate | 0/off；`CANON_V1_FL_TP8_AB_ARM=serving-scope` 诊断，精确 P45-readiness/M15-main DP8×TP8 strict-zero 300-update FrozenLake V1 full profile，或精确 P58 Qwen3-4B Zero/full DP8×TP8 1,000-update HP profile可设 1；GSM8K、P58 Native/IS、非 HP Zero、Qwen3-32B 与其他 profile 禁止 | host/exact-image gates通过；FrozenLake Wave 5 real P45 DP8×TP8 serving-scope 为48,594 action tokens、depth 2,472、A−B/B−C strict `0/0`、zero backward/commit，P45 serving recovery已验证；M15 serving、FrozenLake full backward/AdamW/perf/convergence未验证。P58 profile/environment/Python contract与完整 pinned-image gate通过，marker含 `vma_p59_only=1`；P58 target尚未重跑 | FrozenLake P45/M15各自300-update full horizon与P58 fresh target必须独立通过 strict A=B=C、首commit backward-health/first-update与full-horizon gates；任一target drift均否决对应 workload admission并保留证据 |
| CANON_V1_HP_FIRST_UPDATE_GATE | Exact registered full-run numerical admission. On train step 0, observes the complete accumulator before AdamW and requires the workload-specific denominator/microsteps, all-finite, nonzero, and stable-L2 in `(0,1e6]`; after AdamW it requires finite/coherent optimizer evidence before outer weight sync/checkpoint. P58 Qwen3-4B uses 16 rank-major gradient groups and denominator 16 despite eight outer prompt chunks. The bound is a regression sentinel, not a clip value | off; registered Phase4 full contexts plus exact P58 Zero-HP full, always requiring `CANON_P59_CHECKED_VMA=1` | Phase4 pre-registered in V1.P4.9; P58.11 construction validation in progress; target first commits pending | retain as a long-term first-commit safety gate or retire per workload only after target full horizons establish a tighter envelope |
| CANON_V1_FL_TP8_AB_ARM | FrozenLake DP8×TP8 serving A/B first-red 双臂 selector：`p66-off` 整体关闭 checked-VMA 作定罪臂；`serving-scope` 保留 P59 checked-VMA backward、仅将共享 serving kernel 圈回历史程序作候选修复臂。两臂固定完整 32-prompt/256-trajectory P45 或 M15/main 几何，单轮 pre-backward 受控退出，zero backward/zero optimizer commit | 空/off；仅 `qwen3-8b-dp8-tp8-frozenlake-v1-ab-debug.env` 可设两个登记值 | host resolved-env、Attempt-9 red/recovery/B−C-negative classifier、双 P45 64-chip renderer、fixed-image focused/full gates 通过；DP8×TP8 target 未跑，任何 arm 的 Kubernetes Complete 不替代 classification JSON | 双臂给出可归因判决并由后续 TP8 trainer-forward/backward oracle 收口后退役；失败证据永久保留 |
| CANON_OPT_STATE_RESIDENT / CANON_P30_OPT_STATE_OFFLOAD | 优化器驻留/卸载 | resident=生产默认 | 默认开 | resident 焊死后 OFFLOAD 降级为逃生开关保留 |
| CANON_KV_UNIFIED | U 臂读路径统一实验 | off | **否决区**:生产红(43→9 仍 0.28),非修复 | 可删,判决记录永存 |
| MIN_TOKEN_BUCKET / max_num_batched_tokens(非 CANON 但同级) | R2:全局/每 rank 桶契约 | 钉死 256 族 | 已认证 | 永不自由化;新几何走契约注册 |

## B 层 · perf/仪器类(observational;发布协议 = 每负载 =verify 首步绿 → =1)

| Flag | 语义 | 生命周期 | 日落 |
|---|---|---|---|
| CANON_PERF_LOG | [PERF] 分段计时(=0 静音) | 默认开 | 长期保留(观测契约) |
| CANON_BATCHED_EVIDENCE | 证据取回批量化(-6s/步) | GSM8K 默认;FL/DS 待 verify;P68 补 grouped mirror(DP16 反向收据),配对 verify 待跑 | 全负载转正后焊死 |
| CANON_DP_COLLECTIVE_REDUCE | P69 刀2:DP 梯度归约 collective 选择器,只作用于 `FixedDPRankGradientReducer` 的 reduce transaction。缺省/空/0=历史 fixed ppermute 树(reduce+broadcast、operand barrier 逐对相加)原函数对象直调,程序逐字节不变;1=A 案:sub-32bit 浮点 leaf 升 FP32 后每 leaf 单次原生 `jax.lax.psum`(既有 shard_map check_vma=False 内),cast 回原 dtype;tree=B 案:all_gather 后每 rank 以 registered `fixed_dp_sum` 固定二叉树本地相加(与历史树同配对同序,host 上与历史逐位同值);其他值 fatal。前向零接触,`compare_local` 看门狗、staged 写路径与 receipt schema 全不变(`reduction_rounds` 仍报历史 ppermute 轮数) | off | 试验;scratch host 门(pinned image CPU):FP64 包络、×2 双计负控、flag-off bitwise 冻结指纹、receipt 不变已绿;one-host DP4 E3 已测(psum 归约段 −62%、warm −9%,commit 范数与 legacy 逐位同;tree 结构性逐位同);dp4 one-host profile 条件默认 1(`:-1`,可覆盖);dp16/target 未跑 | P69 Lane1 前向指纹 + Lane2 全套 + Lane3 material 判词后按 workload 转正;任一红退役,判决记录保留 |
| CANON_DP_COMPARE_MODE | P70.4 刀1:DP reduce 后 replica 看门狗选择器,只作用于 `FixedDPRankGradientReducer` 的 replica compare。缺省/空/0/full=历史全量逐元素 ppermute 比对(整棵 reduced 树过邻居,程序与 receipt 逐字节不变);fingerprint-hybrid=每 reducer 生命周期(生产=每 update)前 `HYBRID_FULL_COMPARE_GROUPS`(=2)组保留全量比对且同组跑指纹程序作自检(指纹与全量判决不一致即红停),其余组只 ppermute 每 leaf 双独立 uint32 校验和(rot-add + rot-xor 两混合器,位精确 bitcast,2×N_leaf 标量)并在 mismatch 时报 rank/leaf/path;其他值 fatal。检出弱化:同内容不同位置的补偿性篡改需同时碰撞两个代数独立混合器(NOTES 碰撞论证);−0.0/+0.0 分歧从漏放变为检出(更严),同位 NaN 分歧交给有限位门(顺序与历史一致) | off | 试验;scratch host 门(pinned image CPU):kill-test 单比特翻转必响并指认 leaf、补偿双元素 swap 骗过 naive sum 但双校验和必响、flag-off 冻结 jaxpr/receipt 逐字节同、p69 冻结指纹回归绿;one-host/target 未跑 | P70.4 GATE(kill-test 双项+one-host 配对 walls/范数锚逐位/strict 绿/程序清单 diff)后按 workload 转正;任一红退役,判决记录保留 |
| CANON_DP_DISTINCT_SCHEDULE | P70.4 刀2:per-rank distinct-fingerprint 签名的计算降频。缺省/空/0/every-group=历史每组每 rank 全量 `_gradient_signature`+sha256(receipt 逐字节不变);first-group-warmup=每 update 首组 + 进程前 `DISTINCT_FINGERPRINT_WARMUP_UPDATES`(=3)个 update 的所有组照旧计算,其余组跳过签名(receipt 指纹置 `skipped:receipt-schedule` 并加 `rank_local_fingerprint_mode=skipped`,distinctness 检查在 skipped 组不判);接线正确性属程序级性质:调度/staging/归约程序不随组变,首组+暖机组的检出对 wiring 类故障延迟有界(≤1 update);其他值 fatal。与 deterministic_repeat 互斥(adapter 显式红停) | off | 试验;scratch host 门:调度正确性 kill-test(首组/暖机/skip 序列断言)、flag-off 逐字节同、p69 回归绿;one-host/target 未跑 | 同 CANON_DP_COMPARE_MODE 的 P70.4 GATE;任一红退役,判决记录保留 |
| CANON_DP_FINITE_FETCH | P70.4 刀3:isfinite 位取回的同步点。缺省/空/0/sync=历史逐组同步 device_get+立即 raise(程序与 receipt 逐字节不变);batched-commit=有限位仍逐组在设备端计算(staged+reduced 两段),host 取回合并为 commit 点前单次 int32 向量 `jax.device_get`(P68 批量收据通道),`drain_deferred_finite_receipts()` 在任何梯度进 optimizer commit 前校验全部收据,violation 在 commit 门 raise(带 group/stage/rank/leaf/path);fail-closed 语义不变,只移动 host 同步点(检出延迟 ≤1 update,仍先于 commit);receipt `post_reduction_all_finite=deferred-commit` 字符串逐 receipt 传播,严禁在 drain 前宣称 finite;其他值 fatal。与 deterministic_repeat 互斥 | off | 试验;scratch host 门:非有限注入 kill-test(commit 前必拦、commit callback 零调用)、flag-off 逐字节同、p68/p69 回归绿;one-host/target 未跑 | 同 CANON_DP_COMPARE_MODE 的 P70.4 GATE;任一红退役,判决记录保留 |
| CANON_P28_BATCHED_REPORT(=1/=verify) | report 窗合并+remap jit 化(FL -14.5%) | GSM8K 默认;DP16 待验 | 同上 |
| CANON_P28_BATCHED_REVERSE(=1/=verify) | P52 反向脚手架合并(-13.3%) | 一宿主认证;DP16 等 grouped 移植 | 同上 |
| CANON_P28_LAYER_SCAN | =verify 恒等仪器/=verify_rev THIRDPROG 演示 | **=1 否决(净负 -5%)** | 仪器保留;=1 进否决区 |
| CANON_P71_SCAN | P71 scan-over-layers 阶梯选择器,仅 grouped reverse(`_p32_reverse_group`)消费,enum 解析:缺省/空/`0`/`off` 同义=历史逐层路径(原函数对象直调,程序逐字节不变);`fwd`=E1:每 chunk 的反向 forward tape 由单个 `zt_tr_fwd_scan` scan 程序重建(scan body 复用 fwd_layer 同一组成与 P50 `_ensure_layer_scan` stacked 参数布局,不新增第二份参数栈;scan 体内 `jax.named_scope` 保留 p71_params_merge/p71_layer_fwd/p71_fwd_tape_scan 分段层级供 xplane 寻址),serial 分支 pullback 走既有 `bwd_layer_block_tape` 栈内静态切片程序,rank-parallel 分支保留原 mapped 逐层 pullback(hidden tape 一次 jitted unstack,cache tape 即 replay 自身逐层对象,mapped 程序与操作数值不变);`bwd`=E2′(含 fwd,**展开块而非 scan**):rank-parallel 分支的逐层 mapped pullback 改为每 chunk ceil(L/B) 个展开块程序(`zt_tr_dp_parallel_bwd_block_NN` 家族,B=`_P71_BWD_BLOCK_LAYERS`=7 模块常量,降落伞 7→4→2、B=1 即退回逐层,任意层数按整除-余数切连续 span;块体=Python 循环 tracing 展开的直线图,**无 lax.scan/fori_loop**,scan 的 loop 重结合入口不存在——E2 scan NUMERICAL_REJECT 的 norm-scale 重分块前科即源于该入口),块程序由 `_p59_parallel_map` 同一构建路径产生(mesh 选择/manual axes/check-vma pcast/localize 上下文全复用,`rank_local_arg_indices=(1,2,4,5)` 与逐层 map 相同;层堆叠 tape 操作数走 axis-1 data 分区),块内自顶向下逐层调用逐层 map 所 trace 的同一 pullback 函数对象(check-vma 档同规则选 raw 闭包),`jax.named_scope` 保留 p71_bwd_layer_NN 逐层 HLO 层级(P60-2G:逐层身份在块内仍可寻址,优于 scan);tape 双源:生产档=E1 stacked tape 程序内静态 `index_in_dim` 切片(bwd_layer_block_tape 同款,免 hidden unstack 程序),fwd-off 源=逐层操作数(测试/解耦保留);dcache/dhidden 块内 in-graph 链接、块间按逐层同界传递;块输出=逐层 staged 梯度 tuple 与逐层 cache 余切 tuple(下游 P70.1 累加/组装与 cache 余切合约零改动);leaf 操作数取 P70.3 prepared pack 切片;要求 CANON_P59_RANK_PARALLEL_BACKWARD=1(serial 分支 fatal)且 TP1(非 unit model 轴 fatal);`full`=E3 预留档,fatal;其他值 fatal;与 CANON_P66_BACKWARD_ARM、CANON_P28_LAYER_SCAN 互斥 fail-closed。前向 pass 零接触(scan 全在反向 tape 侧);梯度验收按 `tasks/v1_hp_zero_tim/phases/v1-p71-scan-fusion.md` 预注册两级(一级 bitwise 锚,二级降档五项全绿自动接受制;E2′ 展开无循环重结合,一级锚有实在机会,P50 r3/r4 类 norm-scale 归约由真机锚定谳) | off;E1 已落地(5017c279,真机一级锚逐位命中);E2′(bwd)展开块草案,host CPU 门先行、真机 gate 未跑;阶梯 off→fwd→bwd→full 逐级包含,回滚按段 | E1/E2/E3 各段两基准硬门+梯度锚(或二级五项)+显存 +5% 红线绿后按段转正;任一段 NUMERICAL_REJECT 封存该段草案并保持 default off |
| CANON_CONTINUE_DECODE(=K) | 设备内 `lax.while_loop` 连续 decode，摊薄逐 token host 往返；async scheduling 必须关 | off；V1/P58.7 profiles 固定 K=8；M15 target-debug 也保留 K=8 以复刻生产程序，其 replay envelope 从首个 registered continue-decode call 记录 mixed chronology，四个 tensor strata 仍仅由 standard path 产生；一宿主 r20c/r20y KEEP；DeepSWE reader 已接线但 target 未跑 | Phase4 三个 full recipe 与 P58.7 各自 target 绿后按 workload 转正；新 K/尾桶重认证 |
| CANON_FIXED_AR_GATHER | fixed TP reduction 的三轮 ppermute 传输换成一次 all-gather，本地仍按相同 rank 顺序相加 | off；一宿主 r11 KEEP；P58.7 target 未跑 | DP16/DP8 full strict 绿后按 workload 转正 |
| CANON_PALLAS_GATHERED_LOGPROBS | Pallas scorer 片上直接产 selected logprob/top1/rank，避免全词表 logprob 物化；Phase4 新增 DP8/DP16 row-sharding 与每-rank M256 padding | off；一宿主 r10 KEEP；DP port IMPLEMENTED/TARGET NOT RUN，含 P58.7 Qwen3-4B | Phase4 三个 full 与 P58.7 target 的 exact gate/XProf 绿后转正；任一形状红回到 materialized scorer |
| CANON_LOGPROB_STEP_FUSION | decode logprob 的 slice/pad/gather/slice 胶水收进一个 jitted program，值链不变 | off；一宿主 r15 KEEP；P58.7 target 未跑 | Phase4 三个 full 与 P58.7 target 绿后按 workload 转正 |
| CANON_FUSED_TREE_OPS / CANON_PALLAS_NORM_MATMUL / CANON_PALLAS_INPUT_FUSION | P56 默认-off 候选实现；V1.1 不启用：P59 已取代主要 host-glue 靶，norm/input fusion 未进入最终 serving 配方 | off；保留历史 KEEP/边际/未转正事实；V1 profile 明确为 0 | V1 full 完成后按 P56 判决裁撤或另立新证据重开 |
| CANON_SAMPLE_SPLIT_FUSION / CANON_ENGINE_LOGPROB_READBACK / CANON_ANCHOR_OVERLAP / CANON_GSM8K_VANILLA | P56 中性、被取代或仅对标/载具开关，不属于 V1 默认配方 | off；不进入三个 full recipe | 战役归档后退役；禁止借 V1 profile 开启 |
| CANON_P3_APC_BOUNDARY_REPORT | Phase3 G-A 固定 token deep-prefix 边界报告路径;有值才运行 cache-hit prefill vs B full-reset 的前向探针 | 试验,缺省空/off | P3.1 结束后退役;证据保留 |
| CANON_XPROF_DIR/_SKIP_STEPS/_STEPS | XProf+perfetto 捕获(一次出双产物)；Pathways V1 full 的 DIR 固定为按 JobSet/attempt 隔离的 `gs://.../p33/<job>/attempt-<n>/xprof-update`，结束后硬门回收到本地证据目录 | 仪器 | 长期保留 |
| CANON_PERF_TRACE_DIR / CANON_PERF_TRACE_EXPORT_STEP | 官方 tunix.perf v2 语义时间线导出目录与零起点单步窗口；V1 只序列化 warmed step 2，避免 full train 每步写盘，空目录仍为 NoopTracer 零开销 | 仪器；V1 full 固定 step=2 | 长期保留(官方 Metrics 契约) |
| CANON_XPROF_PYTHON_TRACER/_HOST_TRACER | tracer 档位;**python=0 是 device plane 的前提**(开着它训练捕获退化为 host-only) | 仪器;载具默认 python=0 | 长期保留 |
| CANON_XPROF_TPU_TRACE_MODE | update 窗 TPU trace 密度；V1 full 固定 `TRACE_COMPUTE`；P60-2G 签名 one-host UI carrier 固定低密度 `TRACE_ONLY_XLA`，并以 full XPlane 的 8/8 module/tail gate 和独立 trace-JSON gate 双重防 drop；该 carrier 另固定 XProf 逻辑字节 soft 1.2e9 / hard 1.5e9，超 hard 保留原件但 arm RED | 仪器；仅 `phase=update` 接受；budget 是 task runner 固定合同，不新增可调 flag | 三个 target XProf 与 P60-2G fresh carrier 均无 drop 后决定是否保留默认 |
| CANON_XPROF_LABELS | 为 rollout model/logits/sample 与 trainer fwd/bwd/report JIT 写语义名称；P60-2B/2E 历史合同是一个 whole-update `train` 加 accumulator/optimizer metadata；P60-2G 对签名 Zero-HP arm 将 16 个真实 reverse/reduce/accumulate transaction 映射为 Native API `train_(update*16+micro)`，末个真实 train 跨到 optimizer 结束，无 synthetic terminal step，不改数值、JIT、同步或 Perfetto 词汇 | 仪器；V1 full 开；P56 r21/r22 已认证；P60-2F 历史 clean source `5549b5b6` 仍为其原合同 TARGET PASS，但新 UI 判据为 Native-like FAIL；P60-2G local implementation，TARGET NOT RUN | XProf 原生提供等价稳定命名后退役 |
| CANON_P59_XPROF_BACKWARD_DIR / CANON_P59_DP4_TAIL8 / CANON_P60_DETERMINISTIC_AB | P59/P60 DP4 专用 profile、tail 与跨臂载具 | off | 历史载具完成，不进入 V1 full | 证据交付后退役 |
| CANON_V1_GSM8K_XPROF_ARM | `native|zero-hp` 的 one-host GSM8K matched-work/XProf 观测 selector；固定 DP4×TP1、3 commits、warm update 2→3 capture，打印 profiled batch token/advantage hashes；Native 必须 vanilla stock trainer，Zero-HP 必须 strict V1/P59 bundle；P60-2G 只重跑 Zero-HP | 空/off；仅两条薄 wrapper 设置；P60-2G local，TARGET NOT RUN | pair XProf 归档后退役；不得进入 full recipe |
| CANON_P58_ONEHOST_XPROF_ARM / CANON_P58_ONEHOST_SEAM_PROBE | P58 Qwen3-4B one-host mutation-free 诊断载具。`XPROF_ARM=native\|zero-hp` 固定 DP1×TP4/backward-no-commit；`SEAM_PROBE=1` 只允许 Zero-HP arm，并把同一载具扩展为单个已签名 Pillow task、G2、8K response、16 turns、serial scheduler。它保留真实 rollout/durable trajectory/strict decode-vs-prefill gate；有限 RED 与 exact 都只描述该 TP4 carrier，绝不认证 DP8×TP8/TP8 | 空/0；仅 tracked thin wrapper 设置；生产 P58 selector 必须 0 | TP4 首差分类和必要的 DP8×TP8 exact-geometry follow-up 结案后整体退役；不得进入 full recipe |
| CANON_P58_Q4_TP4_ZERO_ADMISSION / CANON_P58_Q4_TP4_SEAM_DIAGNOSTIC | P58.20/21 Qwen3-4B-Instruct-2507 direct-v5p DP1×TP4 full-overlay Zero admission 及其唯一 `standard-decode` 因果控制。主 selector 仅接受 `0\|1`；诊断仅接受空/`standard-decode`，且只替换 continue-decode，模型、采样、fixed head、prefix-cache-off 与 strict A=B=C 不变 | 试验、默认 `0`/空；只由 P58 one-host tracked wrapper 设置，production P58/P59 必须拒绝 | TP4 admission 和后续 TP8 promotion 归档后退役 |
| CANON_P58_Q4_TP4_CONTINUE_KV_DIAGNOSTIC / CANON_P58_Q4_TP4_CONTINUE_KV_MIN_PREFIX / CANON_P58_Q4_TP4_CONTINUE_KV_MAX_PREFIX | P58.22 continue-decode KV discriminator 及其不可调窗口 `2280/3072`；只允许 baseline continue=8、strict precheck-only、bounded integer fingerprint，禁止进入 backward/optimizer | 诊断、默认 `0`/absent；tracked wrapper 原子派生 | 一次因果分类与修复复验归档后整体退役 |
| CANON_P58_Q4_TP4_SHORT_BACKWARD / CANON_P58_Q4_TP4_CARRIER_SCREEN | P58.22 one-host 短宽度 backward-no-commit 与 rollout-only carrier 筛选。short arm 保留真实数据、strict gate、TPU-resident optimizer/zero commit；screen 必须依赖 short 且不得进入 trainer | 诊断、默认 `0`；仅 P58 tracked wrappers | P58.23 optimized replay 完成并由真实 TP8 promotion 取代后退役 |
| CANON_P58_Q4_TP4_TRAJECTORY_REPLAY | P58.23 单一事实源：启用受哈希约束的两个真实 task group、每组 reward `[1,0]` 的 B2×G2 replay；全局 `batch_size=2`，prompt/response=`2048/512`、K=2560，严格禁止 B1。它原子要求 P28 segmented forward/train + P30 sparse/reuse/release/reshard + hardware-certified `CANON_P71_SCAN=fwd`；DP1 明确保持 P59 rank-parallel off。只跑 re-score/alignment/repeat-exact backward-no-commit，不调用 sandbox/decode，不声称 fresh rollout 或 TP8 | 诊断、默认 `0`；仅 tracked P58.23 wrapper 设置 | bounded one-host compile/backward 结论与 TP8 follow-up 归档后退役 |
| CANON_P58_REPLAY_JOURNAL / CANON_P58_REPLAY_JOURNAL_SHA256 | P58.23 replay source 的绝对路径与 SHA-256 receipt；只能与 trajectory-replay selector 成对出现，指向 deterministic B2×G2 merged journal，任何缺失/漂移 fail-closed | 默认 absent；由 wrapper 从已签名本地 artifact 派生 | 与 P58.23 selector 同时退役；不得进入 production profile |
| CANON_P58_CHECKED_VMA_DIAGNOSTIC | P58.18 exact-geometry matched-control selector；合法值仅 `off\|on`。只准入 Qwen3-4B Zero-HP/full 的 128-chip disaggregated DP8×TP8+DP8×TP8 Step-0 carrier。`off` 原子派生 checked-VMA/P66 alias/P67 scoping=`0/0/0`；`on` 派生 `1/1/1`。两者都把 first-update gate/P63 clip 固定为 `0/0`，保留 fixed-head/continue-decode/Fixed-AR/serving HP、完整 trajectory + pre-alignment，并在 backward/optimizer 前受控退出。生产 selector 缺省 absent 时仍是完整 `1/1/1/1/1`，不受诊断影响 | 缺省 absent；仅 P58.18 ON-A/OFF/ON-B 三臂诊断可设；不得与 normal Zero-HP、native recipe 或 subordinate 手工 override 混用 | 三臂因果裁决与 fresh checked-VMA-on strict Step-0 修复复验归档后退役；不得成为 full-training 默认值 |
| CANON_P58_SEAM_LOCALIZATION | P58.19 exact-geometry coarse seam selector；合法值仅 `coarse`。只准入 Qwen3-4B Zero-HP/full 的 128-chip disaggregated DP8×TP8+DP8×TP8 frozen Step-0 carrier；单一 selector 派生 P38 layer seam + terminal-tail、位置窗 `[1686,4096)`、三轮 per-round seal/classify/ACK 与 `p58-seam-v1` durability。每轮 seam/tail 各自使用独立 4 GiB byte budget，轮次只能单调 `0→1→2`，record index 不复用；这是 p58s19d 在累计 1 GiB 自停后的诊断容量修复，不改变 tensor 内容。该窗口覆盖 p58z07/P58.18 已观测的 2,513/3,438/3,715/3,880/4,032 first-red prefixes，同时避免 p58s19b 的 `[3072,4608)` 零记录载具失败。保留 production `CANON_CONTINUE_DECODE=8`：`standard` 是唯一 tensor-strata 来源；`continue_decode` 只允许保留 scheduler chronology，跳过 incident/tensor payload并回显 `CANON_P58_CONTINUE_DECODE_OBSERVER_BYPASS ... tensor_capture=0`。任何非 P58 profile 或未知 program path 仍 fail-closed。保留 production checked-VMA/P67/first-update/clip tuple，backward 与 optimizer commit 均不可达 | 缺省 absent；仅 P58.19 三轮粗定位可设；与 checked-VMA diagnostic、native、普通 Zero-HP full、M15 wide 及 subordinate 手工 override 互斥 | 三轮给出可重复 coarse first-red signature 并完成后续 fine localization 后退役；任何无 join、B-C red、非重复 boundary 或 observer-neutrality red 均只得 INCONCLUSIVE/FAIL，不得转为训练开关 |
| CANON_XPROF_PHASE | 捕获窗模式:step=整步(device 缓冲 ~283 万事件/核,decode ~25s 填满,实为 engine 前 25s 织物)/ update=G6 update 入口→步完成(rollout 不入镜,缓冲装下完整 backward)/ diagnostic=冻结权重 precheck 的一个完整 A-rollout/B-full-rescore/C-old-forward round | 仪器;载具旋钮 P51_XPROF_PHASE;Phase3 profile 固定 diagnostic skip=1 steps=1 | 长期保留 |
| CANON_UPDATE_REPORT / CANON_PRE_ALIGN_REPORT / CANON_ALIGN_REPORT | 对齐/更新报告选通 | 默认开(监控契约) | 长期保留;A−B 哨兵不可撤(用户裁决 2026-08-15) |
| JAX_COMPILATION_CACHE_DIR(非 CANON) | 持久编译缓存(-72s/重启) | 一宿主认证；Phase4 三个 full manifest 已锁定本地目录与 GCS root，restore/save 回执 host 绿；**Pathways target hit 未验** | 三个 full target 记录 hit/miss 与 JIT 后决定是否推广 |

## C 层 · P38 诊断家族(~30 个 CANON_P38_*)

| 组 | 代表 | 日落条件(一条覆盖全家) |
|---|---|---|
| capture/journal/ledger/capsule/GCS/replay | SERVING_CAPTURE_*、REQUEST_JOURNAL、INCIDENT_LEDGER、MISMATCH_CAPSULE、GCS_PREFIX、DURABILITY_PROFILE、FROZENLAKE_REPLAY、PRECHECK_ONLY… | **carrier 结案(strict 复验全绿)→ 全家整体退役**;判决类结论(U 臂、slot bug、co-batch、shape-1)迁 FOOTGUNS 后删 |
| Phase3 APC dirty-page negative | CANON_P3_APC_DIRTY_PAGE:布尔诊断旗标;缺省/空/0 不污染,仅 boundary dirty mode 的 writer 向直接 JAX/vLLM reader 透传 1;污染 A 确定会复用的 layer-0 单个真实 KV page,B 仍 full reset | G-D `p3gd1` 已命中;为 G-E 复验保留默认-off 载具,Phase3 最终结案 CL 退役 |
| M15 target APC carrier | CANON_APC_M15_TARGET_DEBUG=`off\|on`:只准入 DP8×TP8、M15/main、zero-commit 的 bounded target reproducer；observer=`none` 精确单轮，`m15-wide-v1` 的 layer/full observer 与 `m15-e0-kv-v1` 的 `kv3` observer 精确三轮且每轮权重冻结、先 seal/readback 再推进。精确 `kv3` profile 还固定 round-0 的 32 条 prompt identity 并在 round 1/2 重放该 inventory，三轮各自重新执行 rollout/request/cache chronology；其他 profile 继续推进 dataset。历史 `kv` 一轮路径保留，仅用于旧 Attempt-18 回收。`off` 是 APC-off control，`on` 是 production cache-read treatment；两臂保留生产 `CANON_CONTINUE_DECODE=8` 和签名的 `sampler_is=None` rollout-logprob recipe，不允许 TIS weights；四个 tensor capture 仍限定 standard path；A 强制 `prompt_logprobs=None/logprobs=1/skip_reading_prefix_cache=False`，B 强制 `reset_prefix_cache=True` 且 cache tokens 全零 | fresh target red 冻结并完成首红定位、修复后 G-E 全零与脏页负控再次通过后退役；不得进入 full production profile |
| M15 E0 targeted live-KV discriminator | `CANON_P38_KV_OBSERVER_LAYER`、`CANON_P38_KV_OBSERVER_TARGET_PREFIX_SHA256`、`CANON_P38_KV_OBSERVER_TARGET_PREFIX_TOKENS`:三者只能成组设置；仅 D3e 已登记的 M15/main DP8×TP8 frozen off/on carrier 可设为 layer 0、1226-token canonical-action 前缀及其 self-hashed identity。`kv` 保留历史一轮合同；新 `kv3` 固定三轮，每轮重新取得 8 个合法 request alias、独立 128 MiB byte budget，record index 全局单调且不复用。每个 alias 只读取 77 个逻辑页（静态上限 96），随后用同轮 replay-ledger future-prefix 证据唯一绑定 source request；prefix L 的 snapshot 可以 join logical-prefix L 的 next-token red，仍拒绝任何 future `>L` red。旧 P38 observer 缺省仍读全部层且页上限 32。输出是整数 aggregate/fixed-sample diagnostic fingerprint，不是完整 KV bytes；任何 alias 缺失、跨轮配对、非唯一 future binding、无 red join、B-C red 或 observer-neutrality red 均 fail closed | 默认 absent/off；Attempt 19 因 equality-boundary join 与跨轮 dataset advance 两个 carrier 缺陷 INCONCLUSIVE；E0t repair host/fake-GCS 已绿，post-repair pinned exact-image 与 DP8×TP8 target 未跑；成功区分 live stored-KV red/equal 后退役，不能进入 production profile 或放宽 B full reset |
| M15 replay envelope | CANON_APC_M15_REPLAY_LEDGER=`<capture-dir>/m15_replay_envelope.jsonl`:仅与 `CANON_APC_M15_TARGET_DEBUG=off\|on` 成对启用；逐 serving call 保存 host 侧 dispatch/request/position/page 几何与 token-history SHA，不读取 device tensor；A 的冻结 carrier 必须机械证明 standard+continue-decode，B 必须只走 full-reset standard；路径直接位于 P38 capture 目录。无专用 durability 的旧 carrier 仍保留 incident/replay 单文件界；`m15-wide-v1` 与 `m15-e0-kv-v1` 明确跳过重复且会在多轮中饱和的 incident ledger，replay envelope 继续作为 chronology authority | M15 target red 的完整 producer unit、serving chronology 与 first-red join 已冻结并完成 deterministic replay 后退役；不得单独开启 |
| M15 E0 KV three-round durability | CANON_P38_DURABILITY_PROFILE=`m15-e0-kv-v1`:仅允许精确 M15 target `off\|on` + targeted layer-0 KV observer，且 `CANON_P38_DIAGNOSTIC_ROUNDS=3`、seam/tail/terminal observer 全空。该 profile 对 round-0 batch 的 `p57_index/seed/map_sha256` 生成 self-hash，要求 1 个 frozen marker、2 个 requeue marker 与唯一共同 SHA，禁止跨轮 dataset advance；每轮仍重新执行 rollout/request/cache chronology。每轮严格执行 16 条 KV record（8A+8B）隔离 staging → self-hashed classifier-input archive upload/readback → classifier PASS → round archive/upload/readback → `ROUND_COMPLETE` → learner ACK；round 0/1 成功后即使 round 2 或 root collect 失败也不得丢失。三轮完成后才生成 arm aggregate；只读回传必须优先读取 per-round 小收据，不以 `COLLECTED/COMPLETE` 存在为回收前提 | 试验、仅 M15 E0 target；Attempt 19 证明旧实现推进 dataset 后 round 1 无目标记录；E0t host fake-GCS/aggregate 已绿，post-repair pinned exact-image/target 未跑；首红机制裁决完成后整体退役，失败轮与 partial salvage 永久保留 |
| M15 wide durability | CANON_P38_DURABILITY_PROFILE=`m15-wide-v1`:只允许精确 M15 target `off\|on` + `layer\|full` observer；精确三轮，每轮独立 shard root 与 observer byte budget，record 序号不复用；每 30 秒最多复制 32 对完整 JSON/NPZ、逻辑 payload 最多 256 MiB，生成确定性 archive/SHA，远端下载复核后才写 `SHARD_COMPLETE.json`；round classifier 只读本轮 sealed shard 隔离副本并过滤 cumulative replay ledger；终态顺序固定为 shard complete → round complete → COLLECTED → postflight COMPLETE；根终态丢失时可用 checked-in 小回传脚本恢复已 seal 轮次；每次持久化还核对执行 checkout HEAD 与 rendered full SHA | 试验、仅 M15 wide target；首红定位和最小修复完成后整体退役，失败 shard 永久保留 |
| M15 wide seam bundle | CANON_APC_M15_SEAM_BUNDLE=`<state>/m15_wide_seam_bundle.tar`:仅在 M15 target 的 `layer\|full` seam observer 上启用；round worker 从 sealed shard union 中运行 classifier，再从 A/B 精确 join 选出的原始 seam/tail records、alignment、capsule、replay ledger 生成 deterministic SHA bundle。`m15-wide-v1` 会把该紧凑 bundle、分类和输入收据上传到既有任务 GCS evidence prefix；它不上传整个 live capture 目录 | 首红定位完成并把最小修复通过 G-E 后退役；新的 target 发射仍需用户单独批准 |
| Attempt-7 backward first-red | CANON_P62_BACKWARD_NUMERIC_DEBUG:布尔诊断旗标；仅严格 GSM8K DP16xTP4、P59 fixed-head、`backward-no-commit` 载具可开，打印 loss/VJP/DP-reduce/scale/accumulator 紧凑数值 receipt，累加器最终丢弃且 optimizer commit 必须为 0；它不启用 stable clipping | 默认 off；首红根因定位并由独立修复通过 target 后退役，失败证据永久保留 |
| Registered full-recipe overflow-safe clip | CANON_P63_OVERFLOW_SAFE_CLIP:数值布尔旗标；缺省/0 关闭、仅 1 开，空值或其他值 fatal。仅注册的 Phase4 committed full contexts 与精确 P58 Qwen3-4B strict Zero-HP full recipe 可开；stock norm finite 时逐位返回原 Optax transform，只有独立 `all_finite` 且 stock norm overflow 时才选 max-scaled L2；真实 NaN/Inf 永不 fallback。GSM8K 与 P58 max norm 1，P45/M15 max norm 100 | 默认 off；Phase4 host 372/372 与完整 pinned-image `p63_clip=1` 已绿；P58.11 construction validation in progress；所有 target optimizer commit/full horizon 均按 workload 独立认证，Phase4 证据不自动转移到 P58 |
| P45 rank-1 first-red | CANON_P64_P45_NUMERIC_DEBUG 与 `CANON_P64_TRAINING_CAPSULE_{MODE,GCS_URI,SHA256}`、`CANON_P64_TRAINING_CAPSULE`、`CANON_P64_MODEL_BINDING_SHA256`:仅严格原始 P45 DP8xTP8、APC-off、P59 fixed-head、`backward-no-commit` 载具可开。capture 在 strict pre-alignment 后原子保存完整 tensorized train batch，并在 backward 前绑定 live model sample；replay 逐数组/文件/模型指纹验真，跳过 environment/rollout/B-rescore，只执行完整 trainer forward 与 group-0 backward，随后丢弃 accumulator。Replay 明示 `certification=0`，不能冒充新 Zero-TIM 认证；首次 NaN/Inf 立即停，绝不 clamp/cast/commit | 默认 off；定位 Attempt-7 P45 rank1 的首个 finite→non-finite 边界并完成根因修复后整体退役，所有 capsule、失败证据与 GCS 路径永久保留 |

## D 层 · 发射/基建管道(~230,按前缀组;逐条语义允许"待考古")

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
| CANON_P58_DEEPSWE_TIM / TIM_ADMITTED / TIM_ARM / EXPECTED_UPDATES / DEBUG_DIR / NATIVE_STOCK_PROMPT_OBSERVER / ONEHOST_XPROF_ARM / one-host provenance 族 | P58 Qwen3-4B-Instruct native-vs-zero 因果训练身份；固定 128-chip synchronous disaggregated、B8xG16、16K、compact filter、TPU optimizer 与完整 trajectory journal；`TIM_ARM=native|zero` 选择 numerical runtime。Native 的 sampler recipe 另由既有 `CANON_P34_DISABLE_SAMPLER_IS:CANON_P34_DISABLE_TIS` 闭集选择：`1:1`=raw，`0:0`=token TIS(threshold 2.0)，混合 tuple 拒绝；Zero/Zero-HP 必须 `1:1`。Native-IS 是 mitigation arm，不改变原生 numerical program，且 group filter 仍关；Native 保留完整 stock serving/trainer program，所有 shape-valid finite A/B/T_old/T_current mismatch 只观测，Zero 全边界 exact；`NATIVE_STOCK_PROMPT_OBSERVER=1` 只为 native arm 的 rollout 后 B 观察值提供 processed prompt logprobs，不进入采样、trainer、loss、反向或 optimizer，且与 canonical `PROMPT_PROCESSED_LOGPROBS` 互斥；`ONEHOST_XPROF_ARM=native|zero-hp` 仅准入 DP1xTP4、两次相同输入、固定 `[-1,1]` 诊断 cotangent、零 optimizer commit 的 update-profile 载具，缺省空，不能认证 DP8xTP8/P59/4B-TP8 fixed-head 或生产轨迹；`EXPECT_HOSTNAME/MODEL_SNAPSHOT/R2EGYM_COMMIT/TASK_IMAGE_ID/RUNNER_SHA256/SOURCE_DIFF_SHA256` 是该载具的字符串/路径 provenance receipts，缺省空且不改变数值 | 试验、默认关；one-host matched XProf package 归档后先退役 one-host selector/provenance 族，P58 production 族在完整 campaign 归档后整体退役 |
| CANON_P59_GCS_PREFIX / CANON_P59_INNER_RUN_CMD / CANON_P59_KIND / CANON_P59_REQUIRE_XPROF | P59 单次载具的证据目的地、冻结内层命令、臂身份与 XProf 完整性要求 | 试验；仅 P59 renderer/one-host wrapper 设置 | P59 证据载具归档后整体退役 |
| CANON_ALIGN*/EXPECT_*/DP_SIZE/TP_SIZE/TRAJECTORIES 族 | 对齐门与拓扑断言 | 监控契约,长期保留 |

## MARKERS(日志 marker 契约,非开关;~60 个)

关键项:`[CANON_ALIGN_PRE]`(四边界判决行)、`[CANON_ALIGN_PRE_JSON/EVIDENCE]`、
`[PERF]`、`[CANON_P38] DIAGNOSTIC_COVERAGE_CONTRACT / PRECHECK_COMPLETE`、
`[CANON_P38_SERVING_CAPTURE*]`、`[CANON_P38_INCIDENT_LEDGER_BYPASS]`、
`[CANON_P58_CONTINUE_DECODE_OBSERVER_BYPASS]`、
`[CANON_P58_CONTINUE_KV_CANDIDATE]`、`[CANON_P58_CONTINUE_KV_CLEAN_EMPTY]`、
`[CANON_P58_CONTINUE_KV_CLEAN_PREFIX]`、`[CANON_P58_CONTINUE_KV_CLEAN_READY]`、
`[CANON_APC_M15_B_CONTRACT]`、
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

Count: 409 settable names (appendix inventory above; exclusions: none).
