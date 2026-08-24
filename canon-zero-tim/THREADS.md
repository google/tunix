# THREADS.md — 六线程看板

> 看板类文档,行级写权:线程执行者更新**自己线程的行**(及自己 run 的 EVIDENCE 行);
> 评估者拥有板面结构、跨线仲裁与措辞降格权。只写"现在",历史看各线程 log.md。
> 更新:2026-08-18 @ c04013bd;perf 行 2026-08-19(P54/P55:官方栈 + 训练段 span + update 捕获窗)

| # | 线程 | 状态 | 下一个门 | 等谁 | 任务目录 |
|---|---|---|---|---|---|
| 1 | **zero-tim-carrier** | P38s23r3 forward exact、P38.2h actual-model backward-no-commit PASS；1.7B tied endpoint repair 已发布；P38.2y2 又注册 4B tied K2560/TP8 与 32B untied K5120/TP8，focused/pinned-image CPU gates 绿，TP8 target 未跑 | **先审查 P38.2y2 本地 diff；获批发布后，4B/32B 各跑独立 bounded target，不能互相继承认证** | 等用户批准 commit/push，再等 TP8 target | tasks/p38-pathways-decode-prefill-carrier |
| 2 | **perf** | P55:查明整步 xplane 实为 engine 前 ~25s(device 缓冲 ~283 万事件/核填满即丢,backward 从未入镜),缩窗 land-and-revert 否决;`CANON_XPROF_PHASE=update` 在 G6 入口起窗→backward 完整(p55c census: block_pullback×1758/adjoint×17,decode 零);p55a/a2/b/c 全 3/3 步 51/51 全零;训练段语义 span 重写完成(每步一条扁平官方 peft_train,与 weight_sync 同落位,p55d 认证;首版画糊 land-and-revert e8d4caaf 入 census 负控) | **DP16 一发**(读 p32_vag_reverse 的 adjoint= 验 E;P52 grouped 移植定量) | 等卡 + 用户渲染;P55 包待批推 | tasks/p48-onehost-perf(state.md)+ 外层 p48-p55 |
| 3 | **frozenlake-train** | P45 native `n45j` 已完成首个 optimizer commit，随后 evaluation receipt 把 sync 前的 `global_steps=0` 误当完成行而假红；run 为 `INCONCLUSIVE`。本地修复改读已提交的 `actor train_steps=1` 并硬验 deferred cluster counter；active P57 checkpoint 改为 final-only 300/latest1，legacy 仍 10/latest1；P57 136/136、V1 12/12 绿，target 未重跑 | **审查并获批发布计数生命周期+checkpoint cadence 修复；从同一新 40-char SHA fresh 重跑 P45/M15 × native-no-IS/token-IS 四组，先越过 step-0 receipt/weight sync/policy step 1，最终只接受 step-300 checkpoint** | 等用户批准 commit/push；随后四个 launch 另行批准。不得 resume/reuse earlier attempts | tasks/p57-frozenlake-tim-causal-study |
| 4 | **deepswe-eval** | p46e12808 修复了 Kueue 工作负载冲突注解与 cpu-np 节点池亲和性，Kueue 准入通过（`Admitted: True`），绑定 `--resume-tag p46e12806` 严格从 Wave 27 续跑（复用 6460+ 轨迹） | **等待集群 128 TPU 拓扑释放后自动调度点火** | 排队中 | tasks/p46-deepswe-eval-training-profiles |
| 5 | **deepswe-train** | V1 Attempt 3 修复栈 host 全绿；P58/V1 pinned-image 均 exit 0，真实 installed-attention TP4/TP8、`p59_rpa=2`、`m15_token=1` 与三 manifest 全绿，raw log/receipt 已 SHA 封存；三个 post-fix target 仍 0 optimizer commit | **提交证据 CL，fetch 后正常 push/readback；另获 launch 批准后从同一 SHA 同时起 GSM8K/P45/M15 三个 full JobSet，各自独立首 commit 门** | 等发布收口；P58.6 direct pair、P58.7 full 和三组 V1 full 均保持各自授权边界 | tasks/v1-phase4-three-full-recipes |
| 6 | **delivery-docs** | ✅ 2026-08-16 design doc 大修完成:§4.5 按 11-run 新证据重写(1498 地板/turn-0/s17 翻案/两嫌疑人)、8/9 缺陷已修(剩 aval-split 机制段,缺源材料);zero-tim-clean 闲置;2aa25558 cherry-pick 悬挂 | 分支迁移决策 + 收编余量 + one-pager 择期同步新指纹 | 等用户批 | 外层 tasks/canon_zero_tim_package 等 |

**登记不追的开口**:cross-bucket L1(§12#2)、step0 采样非确定(E0 后再议,P38 力量倍增器)、
d2h 2-3GB/s vs h2d 117GB/s 40× 之谜、生产 h2d 21s。

**维护线**(不占线程号,不阻塞热线):tasks/canon_system_redesign/(本看板即其 phase0 产物)。
