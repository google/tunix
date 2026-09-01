# THREADS.md — 六线程看板

> 看板类文档,行级写权:线程执行者更新**自己线程的行**(及自己 run 的 EVIDENCE 行);
> 评估者拥有板面结构、跨线仲裁与措辞降格权。只写"现在",历史看各线程 log.md。
> 更新:2026-08-26 @ 23e0bddf + P60-2G fe94345c;perf 行 2026-08-26

| # | 线程 | 状态 | 下一个门 | 等谁 | 任务目录 |
|---|---|---|---|---|---|
| 1 | **zero-tim-carrier** | P38s23r3 forward exact、P38.2h actual-model backward-no-commit PASS；1.7B tied endpoint repair 已发布；P38.2y2 又注册 4B tied K2560/TP8 与 32B untied K5120/TP8，focused/pinned-image CPU gates 绿，TP8 target 未跑 | **先审查 P38.2y2 本地 diff；获批发布后，4B/32B 各跑独立 bounded target，不能互相继承认证** | 等用户批准 commit/push，再等 TP8 target | tasks/p38-pathways-decode-prefill-carrier |
| 2 | **perf** | P60-2G runtime commit `fe94345c84c2181ed99997c6768f78d913b2da94` 已无冲突 rebase 到 `23e0bddf`，clean host/static/exact-image 全绿：warm update-2 的 16 个真实 transaction 映射为 `train_32..47`，末个 train 拥有 optimizer；full-XPlane compile=0、streaming UI tail、固定 XProf soft 1.2e9/hard 1.5e9 及 raw-artifact SHA 门均机械化。P60 13/13、P59 37/37、V1/P64 67/67、flags 378/378；pinned 容器 TPU=0，TARGET NOT RUN | **发布后 remote 必须 checkout exact `fe94345c…` clean SHA，只跑 fresh Zero-HP one-host，不重跑 Native；验 3/3、51/51、train 32..47、8/8 tail、decode/compile=0、size≤1.5GB、SHA ledger** | 本次只允许普通 fast-forward 发布；remote v5p launch 仍需单独批准 | tasks/v1-gsm8k-onehost-xprof-pair |
| 3 | **frozenlake-train** | P57.1c 已把 Wave 15 根因修为 export→queue、active commit 非破坏性拒绝；one-host `r7` 真实完成 3/3 AdamW、12/12 strict 零差并跨过 Step-1 旧崩点，beta-0 semantic Perfetto PASS。P57 172/172、P45 exact-image、V1 90/90、flags 395/395 绿；稳态两步 36.93/35.98s，reverse 18.45/17.73s | **核验已获批的两 CL 发布栈；fresh P45/M15 full render/launch 仍需另批** | G4 PASS；source CL `ec9884e9`；G5 full target 未跑 | tasks/p57-frozenlake-tim-causal-study |
| 4 | **deepswe-eval** | p46e12808 修复了 Kueue 工作负载冲突注解与 cpu-np 节点池亲和性，Kueue 准入通过（`Admitted: True`），绑定 `--resume-tag p46e12806` 严格从 Wave 27 续跑（复用 6460+ 轨迹） | **等待集群 128 TPU 拓扑释放后自动调度点火** | 排队中 | tasks/p46-deepswe-eval-training-profiles |
| 5 | **frozenlake-zero-full** | M15 Step 61 的单-rank prompt-only construction stop 已由 P4.19 source CL `813bb7c5` 修复：共享 action-mask subset 后按 zero-loss/zero-gradient 保留该行，全批 zero-action 仍 fatal；P57 184/184、V1 93/93、完整 P58/V1 pinned-image 绿 | **发布隔离的 source+ledger CL 并精确 readback；随后另开 default-off chunk-bucket phase，M15 DP8×TP8 target 仍需单独发射批准** | source 已本地 commit；remote readback/target未跑；当前性能 gap 仅分析、未改代码 | tasks/v1-phase4-three-full-recipes |
| 6 | **delivery-docs** | ✅ 2026-08-16 design doc 大修完成:§4.5 按 11-run 新证据重写(1498 地板/turn-0/s17 翻案/两嫌疑人)、8/9 缺陷已修(剩 aval-split 机制段,缺源材料);zero-tim-clean 闲置;2aa25558 cherry-pick 悬挂 | 分支迁移决策 + 收编余量 + one-pager 择期同步新指纹 | 等用户批 | 外层 tasks/canon_zero_tim_package 等 |

**登记不追的开口**:cross-bucket L1(§12#2)、step0 采样非确定(E0 后再议,P38 力量倍增器)、
d2h 2-3GB/s vs h2d 117GB/s 40× 之谜、生产 h2d 21s。

**维护线**(不占线程号,不阻塞热线):tasks/canon_system_redesign/(本看板即其 phase0 产物)。
