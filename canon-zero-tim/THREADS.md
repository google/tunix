# THREADS.md — 六线程看板

> 看板类文档,行级写权:线程执行者更新**自己线程的行**(及自己 run 的 EVIDENCE 行);
> 评估者拥有板面结构、跨线仲裁与措辞降格权。只写"现在",历史看各线程 log.md。
> 更新:2026-08-17 @ af362e8a base + local P38.2s reducer implementation

| # | 线程 | 状态 | 下一个门 | 等谁 | 任务目录 |
|---|---|---|---|---|---|
| 1 | **zero-tim-carrier** | P38s18r2 Round 0 为 analysis-grade：B-C 逐位 0，A-B 45 bytes / 32 elements、max_abs 0.101；仅 1/32 红点在 256 边界。远端 3,896-object 回执完整，但 direct classifier 因重复 seam key fail-closed。P38.2s alias-aware seam+tail reducer/auditor/wrapper 已在隔离 worktree 实现并本地全绿，尚未发布 | **审阅并发布 P38.2s；随后 GCS 端只运行固定 wrapper，回传小型 audited bundle；不重跑 TPU** | 等用户审阅/批准 commit+push，再由 GCS agent 执行 | tasks/p38-pathways-decode-prefill-carrier |
| 2 | **perf** | 一宿主收官(warm 199s);flags 已 push(20a67129);2.6× 解码税=契约价 | **DP16 一发**(E 窗验证+税率表+裁 jit 整并) | 等卡 + 用户渲染 | tasks/p48-onehost-perf(分支)+ 外层 p48-p52 |
| 3 | **frozenlake-train** | p45r8（DP8xTP8 resident 路径，无评测模式 `--eval_every_n_steps=0`，`fl-prod-noeval-001`）已成功在 64 TPU 上线运行，JIT 编译通过，Step 0 反向传播（40ms/microstep）与 DP8 规约正常进行中 | **持续推进 Step 1~450 迭代，Step 10 自动持久化新 GCS 检查点** | 训练中 | tasks/p45-frozenlake-dp8-tp8-resident |
| 4 | **deepswe-eval** | p46e12808 修复了 Kueue 工作负载冲突注解与 cpu-np 节点池亲和性，Kueue 准入通过（`Admitted: True`），绑定 `--resume-tag p46e12806` 严格从 Wave 27 续跑（复用 6460+ 轨迹） | **等待集群 128 TPU 拓扑释放后自动调度点火** | 排队中 | tasks/p46-deepswe-eval-training-profiles |
| 5 | **deepswe-train** | 依赖 #4 干净数据;Q4 parity 任务在册(p44) | Q4 3-step debug 配置定稿 | 排 #2#3 绿后 | tasks/p44-deepswe-qwen4b-parity |
| 6 | **delivery-docs** | ✅ 2026-08-16 design doc 大修完成:§4.5 按 11-run 新证据重写(1498 地板/turn-0/s17 翻案/两嫌疑人)、8/9 缺陷已修(剩 aval-split 机制段,缺源材料);zero-tim-clean 闲置;2aa25558 cherry-pick 悬挂 | 分支迁移决策 + 收编余量 + one-pager 择期同步新指纹 | 等用户批 | 外层 tasks/canon_zero_tim_package 等 |

**登记不追的开口**:cross-bucket L1(§12#2)、step0 采样非确定(E0 后再议,P38 力量倍增器)、
d2h 2-3GB/s vs h2d 117GB/s 40× 之谜、生产 h2d 21s。

**维护线**(不占线程号,不阻塞热线):tasks/canon_system_redesign/(本看板即其 phase0 产物)。
