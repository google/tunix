# THREADS.md — 六线程看板

> 看板类文档,行级写权:线程执行者更新**自己线程的行**(及自己 run 的 EVIDENCE 行);
> 评估者拥有板面结构、跨线仲裁与措辞降格权。只写"现在",历史看各线程 log.md。
> 更新:2026-08-17 @ 10fe951f base + local P45.3c no-eval/checkpoint admission

| # | 线程 | 状态 | 下一个门 | 等谁 | 任务目录 |
|---|---|---|---|---|---|
| 1 | **zero-tim-carrier** | P38s18r 在 round-0 durability seal 超时，整发 `INCONCLUSIVE_DURABILITY_SEAL_TIMEOUT`；round-0 A-B/B-C 只作 analysis-grade。远端 step fallback 被审出跨轮/fail-open 风险，本地 strict round-scope 修复与 CPU/fake-GCS 门已绿，尚未 commit/push | **用户审修复 → 发布明确 SHA → 新 run-id P38s18r2 跑满 3 轮至 Exit 42** | 等用户批准本地 CL；禁止复用 p38s18r | tasks/p38-pathways-decode-prefill-carrier |
| 2 | **perf** | 一宿主收官(warm 199s);flags 已 push(20a67129);2.6× 解码税=契约价 | **DP16 一发**(E 窗验证+税率表+裁 jit 整并) | 等卡 + 用户渲染 | tasks/p48-onehost-perf(分支)+ 外层 p48-p52 |
| 3 | **frozenlake-train** | p45r7 在 DP8xTP8 resident 路径训练到 step 10/11 后，streaming eval 的 canonical prefill rescore 等待 driver-wide idle prefix-cache reset 300s 超时；旧 checkpoint 的 GCS 对象未由现有证据验证，且其 source/cadence 合同与新 no-eval 源不兼容。P45.3c 本地固定镜像门和真实 one-host v5p model/Adam/metadata roundtrip 已绿，TARGET NOT RUN | **发布 P45.3c SHA → 新 tag/no-eval FULL 跑过 step 10/11并回传 GCS listing → 相同 source/tag 显式 resume 到下一提交步** | 等本 CL 发布后由操作端发 64 卡 Attempt-0 | tasks/p45-frozenlake-dp8-tp8-resident |
| 4 | **deepswe-eval** | reward-only 已 landed;256 卡有 subshard pass;128-chip profiles 已加 | 64 卡 L3 双臂 → 晋升默认;shard 并行未做 | 用户排 L3 | tasks/p46-deepswe-256chip-reward-only-eval |
| 5 | **deepswe-train** | 依赖 #4 干净数据;Q4 parity 任务在册(p44) | Q4 3-step debug 配置定稿 | 排 #2#3 绿后 | tasks/p44-deepswe-qwen4b-parity |
| 6 | **delivery-docs** | ✅ 2026-08-16 design doc 大修完成:§4.5 按 11-run 新证据重写(1498 地板/turn-0/s17 翻案/两嫌疑人)、8/9 缺陷已修(剩 aval-split 机制段,缺源材料);zero-tim-clean 闲置;2aa25558 cherry-pick 悬挂 | 分支迁移决策 + 收编余量 + one-pager 择期同步新指纹 | 等用户批 | 外层 tasks/canon_zero_tim_package 等 |

**登记不追的开口**:cross-bucket L1(§12#2)、step0 采样非确定(E0 后再议,P38 力量倍增器)、
d2h 2-3GB/s vs h2d 117GB/s 40× 之谜、生产 h2d 21s。

**维护线**(不占线程号,不阻塞热线):tasks/canon_system_redesign/(本看板即其 phase0 产物)。
