# Plan — canon-zero-tim 可分发包

Status: **P0–P5 complete; P32 admission extension active (2026-08-06)**
Owner: Claude(plan.md 归 Claude,已获用户批准)

## 1. 为什么做

zero-TIM(`A = B = C` 逐比特)这套改动目前的存在形态是**不可分发、不可移植**的:

| 现状 | 后果 |
|---|---|
| 补丁是上游文件的**整文件副本**(如 `linear_patched.py` = 406 行副本 + ~60 行新增) | 读者无法分辨哪些是本项目的改动 |
| tunix 训练侧改动只以 **dirty worktree** 存在(`sequence_packing/tunix` @ `yuxzhang/p22-align-integration`,14 个文件) | 无任何 diff artifact;一次误操作即丢失 |
| 补丁内**硬编码绝对路径** `/mnt/disks/tunix-data/claude_work`(`attn_iface_patched.py:508,521`) | 换环境静默失效 ⇒ import 失败 ⇒ 走 stock ⇒ **绿灯是假的** |
| `claude_work/` 下 138 个 `.py` + 89 个 `.sh` 平铺,修复与诊断探针混在同一批文件 | 读者分不清解法与探针 |
| 两个活地雷:`run_p19x6.sh:23` 硬编码旧的 prefill-only `CANON_RPA_VJP=1`;同行 `P18_SKIP_ENV_CHECK` 默认 `1`(守卫默认关) | 任何人拿到仓库按默认跑 = 走已知有 bug 的反向 + 关掉配置断言 |

**最终目标(用户 2026-08-05 明确)**:在**大集群 multi-host** 上测试 **deepswe** 与 **frozenlake**。
因此本包的定位不只是"能跑测试",而是 **"搬到新集群时用来判定 zero-TIM 可不可达、还需要重钉什么"的准入工具**。

## 2. 成功定义

1. 一个 **`main` 切出的干净分支** `yuxzhang/canon-zero-tim`,含自足目录 `canon-zero-tim/`。
2. 所有改动以 **真 diff(patch)** 呈现,不是整文件副本;`stock + patch` 与现有副本**逐字节相同**。
3. **T0(纯 CPU 秒级)** 与 **T1(≥4 芯 TPU,无模型无镜像,秒级)** 两层测试,任何人拿到即可跑。
4. **T2/T3** 以 **配方 + 预期输出行 + artifact SHA-256 清单** 呈现(不搬数据)。
5. **`CLUSTER_ADMISSION.md`** —— 新集群/multi-host 准入清单,含可执行探针。
6. **round-trip 验证**:只用包里的东西,在本 VM 复现**已签字的数字**,逐字节相同。

## 3. 已锁定的决策(2026-08-05,用户确认)

| # | 决策 | 理由 |
|---|---|---|
| D1 | 分支名 **`yuxzhang/canon-zero-tim`**;基点 **`3a00d951`**(= `yuxzhang/p22-align-integration` 的 tip) | 避开 `prod`(项目纪律不允许声称生产就绪)。**基点由 main 改为 `3a00d951`**:P0.F3 实测 18 个 zero-TIM 文件中 10 个在 main 与 `9fa7e251` 之间已分叉,且 `peft_trainer.py` 的分叉正是 P26 依赖的 fp32 梯度累加 ⇒ **锚在 main 语义上错误**,不只是会冲突。分支内容 = 可运行产物;`patches/tunix/07` = zero-TIM delta 的文档(base `9fa7e251`,可从 `origin/yuxzhang/fix_accum_fp32` 到达) |
| D2 | T2/T3 **只配方 + SHA 清单**,不搬 raw.log | `goal.md` 禁止未估算容量就落盘大 artifact;清单足以让读者核对自己跑出的结果 |
| D3 | 两个活地雷 **只在包内修正 + 文档标注**,不动冻结 runner | 冻结 runner 承载已签字 release 证据,改了会让旧 artifact 无法回溯 |
| D4 | 诊断探针(`CANON_CUT/BARRIER/BARRIER_ALL/TAIL/MM_ALGO/ATTN_DUMP/FORCE_MIXED`)**不进包**,仅附录列名 | 修复与探针必须可区分 |
| D5 | **P6b(P26 G1a 实跑,~1h + 需 W&B key)单独再批** | 用户 2026-08-05 明确:"真正测试跑的时候再单独批" |
| D6 | 用 `git worktree add`,**不用** `git checkout -b` | `sequence_packing/tunix` 有 14 个 dirty 文件,**它们就是产出 P26 release 证据的那份代码**;checkout 会波及它 |

## 4. Phase 划分

| Phase | 内容 | GATE | 文件 |
|---|---|---|---|
| **P0** | 取证冻结(不改任何东西) | diff `git apply --check` 干净 **且** 各 `*_patched.py` SHA-256 与 P26 G3 raw.log 记录一致 | `phase0.md` |
| **P1** | 真 diff 化(6 个 tpu_inference patch) | **`stock + patch` 产物与现有 `*_patched.py` 逐字节相同**(SHA-256 相等) | `phase1.md` |
| **P2** | 可移植性修复(去硬编码路径、分离 fix/diagnostic、包内修地雷) | 非 `/mnt/disks` 路径下 import 成功 + PATHTRACE 照打;重生成的 patch 仍过 P1 gate,差异**逐行列出** | `phase2.md` |
| **P3** | T0 层(CPU 秒级) | 干净 CPU 跑出:值 `0` · grad `≤5e-16` · FD `1.1e-08` · 多序列 `~5e-17` | `phase3.md` |
| **P4** | T1 层 + **新集群准入探针** ★ | 本机 4 芯复现:1D-4dev **DIFFERS** · 2×2 单轴 **SAME** · F4 令其归零。8-way / 多 slice 臂标 `UNVERIFIED — 待目标集群` | `phase4.md` |
| **P5** | 配方 / 证据 / 文档 | **脚本**核对 EVIDENCE 每条 artifact 在磁盘存在且 SHA 匹配;README 的 T0/T1 命令真跑过并贴输出 | `phase5.md` |
| **P6a** | round-trip 便宜层 | G1/G2 probe gate 逐字节符合记录值 | `phase6.md` |
| **P6b** | round-trip 完整层 **(等单独批准)** | P26 G1a:三边界 0 字节 · 梯度 `0.2502315640449524` 完全一致 | `phase6.md` |

估时 ≈ 5–6 小时,其中 P6b 是唯一长跑。

## 5. 交付结构(持续扩展;精确清单以 Git tree 为准)

```
canon-zero-tim/
├── README.md ANCHORS.md EVIDENCE.md KNOWN_FOOTGUNS.md CLUSTER_ADMISSION.md
├── install.sh                    双模式(--from-image / --from-path)+ MANIFEST 校验
├── verify_evidence.sh            证据存在性与 SHA 的脚本化核对
├── MANIFEST.sha256 (29)          安装产物期望 SHA
├── STOCK_MANIFEST.sha256 (6)     patch 锚点的上游文件 SHA
├── patches/tpu_inference/        6 个真 diff,共 1252 行
├── patches/tunix/07-*.patch      训练侧 delta(文档用途,base 9fa7e251)
├── src/engine_shims/             25 个 .py + models/{qwen1p7b,qwen8b}/
├── tests/t0_cpu/                 run.sh + negative_control.sh
├── tests/t1_tpu/                 4 个 minrepro + 4 个准入探针 + run.sh
├── cluster/                      entrypoint + staged steps/profiles + jobset-64chip.yaml + README
├── recipes/                      T2/T3 配方与预期输出
├── evidence/artifacts.sha256     8 条 release artifact
└── docs/                         plan/design/phase0-5

## 6. 停止规则

- **P1 逐字节 gate 红 ⇒ 停**。说明 diff 拆错或副本与 stock 不同源,继续做下去全是沙上建塔。
- **P6a 任一数字对不上 ⇒ 停**。包漏了东西,回 P1/P2 找。
- **P6b 红 ⇒ 停**,且**必须区分**「包的问题」还是「环境漂移」—— 用同一 VM 上未打包的原路径跑一次做对照。
- 任何 phase 的 gate 无输出 = **没测 ≠ 通过**(`lessons.md` 坑 #24)。

## 7. 边界

**P0–P6 原始工作不做**
- 不动任何冻结 runner(只在包内提供修正 wrapper)
- 不改生产默认 / 模型数学 / loss / GRPO 语义 / reward / 权重 / tokenizer
- 不在当时 commit/push(P0 建分支 ≠ commit;当时等用户批准)
- 不删除任何历史证据

**非目标**
- 不让 T2/T3 变成"任何人都能跑"(需镜像 + 模型 + 4 芯 v5p)
- 不推进 #11(跨桶机制)、#14(FrozenLake)、#15(DeepSWE)—— 包是**工具**,不是新结论
- 不做性能优化(含 F4 的 recursive-doubling 改写)—— 只把成本模型写进 `CLUSTER_ADMISSION.md` 供 tp8 决策

**回滚**
产出 = 新 worktree 目录 + 新分支。回滚 = `git worktree remove` + `git branch -D`。
现有 13/15 证据链、dirty worktree、镜像、模型、生产默认**零影响**。

## 8. 进度

- [x] **P0 取证冻结 — ✅ GATE PASS(2026-08-05)**。分支 `yuxzhang/canon-zero-tim` @ `3a00d951`
      与 P26 G3 **25/25 逐字节同一**;patch 07(11644 行/64 文件)`apply --check -R` 干净;
      6 个 stock 原件已抽出;运行时闭包 = 28 文件。详见 `phase0.md`(含 F1–F6 六项 Finding)
- [x] **P1 真 diff 化 — ✅ GATE PASS(2026-08-05)**。6 个 patch 共 **1252 行**,
      `stock + patch` 逐字节等于链底 patched 文件 **6/6**;`06-tpu-runner.patch` 产出的
      `b8b1e118ca3c` 正是 G3 密码学记录的 SHA。详见 `phase1.md`
- [x] **P2 可移植性修复 — ✅ GATE PASS(2026-08-05)**。9 处硬编码 `BASE_PATH` 全部改为
      `canon_shim_root.resolve()`,**每文件恰好改一行**;包内零 `/mnt/disks`;py_compile 26/26;
      两套模型特化显式建模;`install.sh` + `MANIFEST.sha256`(29 条)。详见 `phase2.md`
- [x] **P3 T0 层 — ✅ GATE PASS(2026-08-05)**。`tests/t0_cpu/run.sh` 实跑 exit=0,
      7 项测量与 SKILL G5 记载一致(值 `0.000e+00` / grad `5.039e-16` / FD `1.106e-08` /
      msq `~5e-17`);负控 4/4 全部被拒。详见 `phase3.md`
- [~] **P4 T1 层 + 准入探针 — 实现完成,TPU gate 阻塞**。4 个 minrepro 搬入并参数化 `N`;
      新写 4 个准入探针;两个解析型探针已出数(dp=64 需 `MIN_TOKEN_BUCKET=16384`;
      F4 在 n=8 的通信代价 4×)。TPU 实测等用户 P31 训练结束。详见 `phase4.md`
- [x] **P5 配方 / 证据 / 文档 — ✅ GATE PASS(2026-08-05)**。`verify_evidence.sh` 8/8;
      cluster 管线 `probe-only`/`install-only` 真实镜像内实跑 exit=0,镜像 6/6 SAME、
      ROPE 正确判 `not_needed`、overlay 字节+活体 import 双通道全绿。详见 `phase5.md`
- [ ] P6a round-trip 便宜层
- [ ] P6b round-trip 完整层(等单独批准)
- [~] **P7 DP16×TP4 admission** — CPU probe + negative control PASS; 64-chip Pathways NOT RUN.
      The P32 profile is fail-closed and does not admit training. See `phase7.md`.

P7 authorization supersedes only the historical no-push boundary above: on 2026-08-06 the user
explicitly approved committing and pushing P31+P32 to `yuxzhang/canon-zero-tim`. It does not
authorize a PR, main mutation, production-default change or cloud execution by this agent.

## 9. 交付的 CL(分支 `yuxzhang/canon-zero-tim`,基点 `3a00d951`)

按 §17 的 CL 规范切分,每个自洽、可单独 checkout。正文均按 §17.2 四项写,**缺点栏未略过**。

```
  7101b4a5  Record the canonical engine changes as diffs against a pinned upstream
  53c0448b  Make the canonical shim chain resolve its members relative to itself
  7748dbeb  Add a CPU gate for the differentiable attention contract, with a negative control
  53198034  Add topology admission probes for reduction width, device order and bucket arithmetic
  370d00c3  Move the cluster launch sequence out of YAML into reviewable scripts
```

未 push。push 前须处理 `.git/config` 里的明文 PAT。
