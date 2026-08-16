# THREADS.md — 六线程看板

> 看板类文档,行级写权:线程执行者更新**自己线程的行**(及自己 run 的 EVIDENCE 行);
> 评估者拥有板面结构、跨线仲裁与措辞降格权。只写"现在",历史看各线程 log.md。
> 更新:2026-08-16 @ e0c1aef7 base + local P38.2q rc4 sealing

| # | 线程 | 状态 | 下一个门 | 等谁 | 任务目录 |
|---|---|---|---|---|---|
| 1 | **zero-tim-carrier** | s18l v2 selector 报告22个快照无合规2-round source;该结果目前只有人工摘要,rc=4耐久 inventory bundle 修复已本地完成 | **发布修复并零TPU重跑selector**,审计/封存s18l后才注册下一次seam+tail终局采集;不得直接晋级tail | 等本CL评审 + GCP agent | tasks/p38-pathways-decode-prefill-carrier |
| 2 | **perf** | 一宿主收官(warm 199s);flags 已 push(20a67129);2.6× 解码税=契约价 | **DP16 一发**(E 窗验证+税率表+裁 jit 整并) | 等卡 + 用户渲染 | tasks/p48-onehost-perf(分支)+ 外层 p48-p52 |
| 3 | **frozenlake-train** | warning-only lane 可跑;p45r6 step-0 checkpoint 契约失败已归档,修复已落(a94d6c0c "Fix P45 checkpointed G6 training") | checkpointed G6 重跑;perf flags verify-first 搭常规 run;P47b 押 E0 判决后 | P45 线重跑窗口 | tasks/p45-frozenlake-dp8-tp8-resident 等 |
| 4 | **deepswe-eval** | reward-only 已 landed;256 卡有 subshard pass;128-chip profiles 已加 | 64 卡 L3 双臂 → 晋升默认;shard 并行未做 | 用户排 L3 | tasks/p46-deepswe-256chip-reward-only-eval |
| 5 | **deepswe-train** | 依赖 #4 干净数据;Q4 parity 任务在册(p44) | Q4 3-step debug 配置定稿 | 排 #2#3 绿后 | tasks/p44-deepswe-qwen4b-parity |
| 6 | **delivery-docs** | ✅ 2026-08-16 design doc 大修完成:§4.5 按 11-run 新证据重写(1498 地板/turn-0/s17 翻案/两嫌疑人)、8/9 缺陷已修(剩 aval-split 机制段,缺源材料);zero-tim-clean 闲置;2aa25558 cherry-pick 悬挂 | 分支迁移决策 + 收编余量 + one-pager 择期同步新指纹 | 等用户批 | 外层 tasks/canon_zero_tim_package 等 |

**登记不追的开口**:cross-bucket L1(§12#2)、step0 采样非确定(E0 后再议,P38 力量倍增器)、
d2h 2-3GB/s vs h2d 117GB/s 40× 之谜、生产 h2d 21s。

**维护线**(不占线程号,不阻塞热线):tasks/canon_system_redesign/(本看板即其 phase0 产物)。
