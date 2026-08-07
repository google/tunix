# Phase 4 — T1 层 + 新集群准入探针

Status: **generic P1 Attempt 0 complete/dirty; P1b blocked by Mosaic v15/server<=13;
TP2/4/8 scan and P1a compatibility gate implemented locally**
Date: 2026-08-07

---

## Finding

- 4 个历史 minrepro 中,`p19_minrepro_f4.py` 把 `N = 4` 写死(v5p-8 探针主机),
  `p19_minrepro_mesh2d.py` / `topo.py` 用 `devs[:4]`。目标拓扑若非 4 芯则用不了。
- On the directly attached four-device host, reduction width was a dominant carrier. The first
  Pathways attempt showed that this conclusion does not transfer by assumption: its replicated
  H3 arm was also dirty, so reduction was not necessary for that observation.
- 宿主无 TPU-enabled jaxlib,T1 必须在镜像内跑 ⇒ runner 需双模式(host 自包装 / 容器内直跑)。

## Execution

- [x] **P4.1** 4 个 minrepro 搬入 `tests/t1_tpu/`
- [x] **P4.2** `p19_minrepro_f4.py` 的 `N` 参数化(`CANON_MINREPRO_N`,默认 `min(4, n_devices)`)
      —— 单锚点替换,`diff` 显示恰好一处
- [x] **P4.3** 新写 4 个准入探针:
      `probe_waycount.py`(枚举归约宽度 × 深度 × {stock AR, F4 树},比 jit vs value_and_grad primal)
      `probe_mesh_order.py`(device order + slice 结构 + `CANON_EXPECT_MODEL_MESH_IDS` 断言)
      `probe_bucket_contract.py`(dp 下 `MIN_TOKEN_BUCKET` 反推,优先用引擎自带函数)
      `probe_f4_cost.py`(通信/显存解析模型 + recursive-doubling 对照)
- [x] **P4.4** `tests/t1_tpu/run.sh` 双模式 + fail-closed(每个探针须产出下限条数的测量行)
- [ ] **P4.5** GATE: directly attached 4-chip run remains external to this rewrite
- [x] **P4.6** Replace device-prefix probes with full-slice `(replica,tp)` P1 arms; add
      replicated/stock/F4 paired inputs, physical group attestation, magnitude metrics, strict
      measurement counts, and unified-runner fail-stop semantics
- [ ] **P4.7** GATE: fresh 64-chip Pathways rerun of `(32,2)`, `(16,4)` and diagnostic `(8,8)`
- [x] **P4.8** Add a hard P1b gate that calls the live P22.XK Qwen MLP operator chain at the
      installed model dimensions and differentiates weights on the full `(replica, model)` mesh
- [x] **P4.9** Move optional T2 into the unified Python client after P1b; make Step 75 a pure
      persisted-marker validator so it cannot create a second IFRT proxy client
- [x] **P4.9a** Add P1a: report client JAX/JAXLIB/PathwaysUtils versions and compile the exact
      promoted RMSNorm through Mosaic before P1b
- [ ] **P4.10** GATE: fresh 64-chip P1b depths `1,2,4,8`, then same-session DP16×TP4 T2

## GATE 状态

| 判据 | 结果 |
|---|---|
| `py_compile` 全部探针 | ✅ |
| `probe_bucket_contract.py` (dp=64, target M=256) | ✅ 输出 `SET MIN_TOKEN_BUCKET=16384`,并给出"照抄 256 ⇒ 每 replica 只有 4"的警告 |
| `probe_f4_cost.py` | ✅ ratio n=4 → `2.00`,n=8 → `4.00`;recursive-doubling n=8 = 7.50 MiB vs 17.50 MiB |
| `run.sh` 在 XLA_FLAGS 缺 `--xla_allow_excess_precision=false` 时拒绝 | ✅(写死在 run.sh 与 probe_waycount 双重检查) |
| **本机 4 芯完整 T1** | ⛔ **阻塞** |

**阻塞原因(基础设施,非数值)**:TPU 被用户的
`p31_frozenlake_0805_p31_signal10_r1`(Qwen3-8B FrozenLake, 150 batches)占用;
`open(/dev/vfio/0): Device or resource busy`。**未干预该进程。**
按坑 #21 口径记为 compile/资源阻塞,不得记为数值 FAIL。

首跑日志保留:`scratchpad/t1_run1.log`(run.sh 正确输出 `T1 FAIL -- a probe did not run`,
fail-closed 行为得到实证)。

## 副产物 Finding:两个 profile 的开关集

从用户 P31 的实跑命令与 `run_p26_gsm8k_train.sh` 双向提取,**两者共享同一套引擎开关**,
差异只在模型几何与任务专属开关。此前文档只列了其中一部分(缺 `CANON_PALLAS_*` 六项、
`CANON_PALLAS_LOGSOFTMAX`、`CANON_ENGINE_MODULE_C`、`FL_SHARED_MESH`、
`CANON_QWEN3_*` 六项几何)。已落为 `cluster/profiles/`:
`_canonical_engine.env`(共享) + `qwen3-1p7b.env` / `qwen3-8b.env`(几何)。

`CANON_EXPECT_MODEL_MESH_IDS=0,2,1,3` 在用户生产里**已在使用**,印证坑 #34。

## Result log

**2026-08-05**

已验证 by:`py_compile`;两个解析型探针宿主实跑并给出与手算一致的结果;
`run.sh` 的 fail-closed 分支在真实失败(TPU busy)下正确触发并报 FAIL。
未验证:way-count / mesh order / 四个 minrepro 的 TPU 实测 —— 等设备空闲。

下一步:用户 P31 结束后重跑 `tests/t1_tpu/run.sh`,判据见 `plan.md` P4 行。

**2026-08-07 — 64-chip Attempt 0 reconciliation**

- Connection, overlay, Pathways initialization, and visible 64-device inventory succeeded.
- P1 printed four TP2 rows, then failed before every TP4 row because `devices[:4]` requested an
  invalid cross-host subslice. The P1 verdict is INCONCLUSIVE; the partial byte counts do not
  establish whether F4 works on Pathways.
- P2-P4 and all historical H rows ran after the P1 runtime exception in the same session. They
  are retained as triage observations but tainted for release use.
- The rewrite constructs full-slice `(32,2)` and production `(16,4)` meshes directly. Every
  numerical point shares identical arrays across replicated, stock-AR, and F4 arms. The new
  runner stops at the first error and names all suppressed probes.
- CPU gates: 19 unit tests cover topology shape, duplicate/missing-device rejection, metric and
  row-count negative controls, Pathways bootstrap, probe ordering, fail-stop behavior, and the
  T2 full-slice mesh contract. The T2 positive gate and rank-fault negative control also pass.
- Hardware status: NOT RUN after the rewrite. A fresh GKE Attempt 0 is the next gate.

**2026-08-07 — full-slice artifact reconciliation and P1b design**

- The rewritten generic P1 completed all 12 rows.  Every replicated, stock-AR and F4 row is
  dirty.  This proves a generic Pathways forward-vs-gradient-program carrier and that F4 alone
  does not close that handwritten graph.
- Source review found that P1 still used local `_rms`, `jnp.einsum` and a handwritten MLP.  It
  never called the promoted production Qwen operators, so its dirty F4 rows cannot reject the
  canonical P22.XK path.
- P1b now imports the live overlay, attests the terminal P22.XK functions, and executes the exact
  RMSNorm, gate/up projections, SwiGLU, down projection and F4 reduction at model dimensions.
  Any byte drift, dead gradient, missing row or inactive promotion is a hard red.
- The old Step 75 launched a second Python process.  The archived log shows a second IFRT proxy
  client connection and then ends with zero T2 markers.  Optional T2 now runs immediately after
  P1b in the existing unified client; Step 75 only validates the persisted artifact.
- Local status before GKE: 22 T1 unit tests and 5 T2 unit tests pass; Python/Bash syntax and
  `git diff --check` pass.  This is implementation evidence only; P1b/T2 hardware status remains
  NOT RUN.

**2026-08-07 — Mosaic compatibility and TP8 diagnostic extension**

- Attempt 0 reached P1b but produced no numerical row. The client emitted stable Mosaic v15 and
  the Pathways service rejected versions above 13; T2 was correctly tainted and skipped.
- P1a now logs JAX/JAXLIB/PathwaysUtils versions and compiles the exact promoted RMSNorm on a
  minimal full-slice input before generic P1 and P1b. It normalizes a version mismatch to one
  fail-closed marker instead of dumping hundreds of MLIR lines.
- Generic P1 now registers widths `2,4,8`; on 64 devices and depths `8,15`, completeness requires
  18 rows. TP8 is diagnostic only; P1b/T2 remain on the explicit TP4 model contract.
- The pinned client image reports JAX/JAXLIB `0.10.2`. Official registry lookup resolved the
  matching `20260730-jax_0.10.2` server/proxy tags; both JobSets now use their immutable digests.
- Local gates: T1 `26/26`, T2/validator `5/5`, profile widths positive and `2,4` negative,
  Python/Bash/diff/English-output checks PASS.
- Hardware status: TP8 generic P1, P1a, P1b and T2 are NOT RUN after this extension.
