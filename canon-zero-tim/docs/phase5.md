# Phase 5 — 文档 / 配方 / cluster 接线

Status: **✅ 闭环(2026-08-05)**

---

## Finding

- 用户 cluster 是 **Pathways on GKE**(JobSet:head 含 proxy+resource_manager,
  worker 为 Indexed Job,每 pod 4 芯),代码交付方式是容器内
  `git fetch <branch> && git reset --hard FETCH_HEAD` —— **分支即 CD**。
- yaml 内联执行块共 10 项操作(时间戳 / git safe.directory / 三步同步 / 模型 HF 自动下载 /
  **ROPE 热补丁** / 缩进修复 / `sleep 60` / 10 项运行时 env / **secret 去空白** / 启动+传出码)。
- **ROPE 热补丁是版本相关的**:锚点镜像的 stock `qwen3.py` 已从
  `tpu_inference.layers.jax.rope_interface` 上游 import `get_rope_theta/get_rope_scaling`
  (`:35-37, :66-67`),该补丁的两处 `code.replace` 在此镜像上是**静默 no-op**,
  而注入的函数定义仍会被 prepend(重复定义)。⇒ 必须条件化 + 未知版本 fail-loud。
- k8s 无 bind mount ⇒ `-v file:target:ro` 必须改为**复制覆盖**。
- `install.sh` 原先只支持 `docker create/cp`,容器内无 docker ⇒ 需 `--from-path`。

## Execution

- [x] **P5.1** `install.sh` 双模式(`--from-image` / `--from-path` / `--model`),
      MANIFEST 不匹配 **fail**(不再只警告)
- [x] **P5.2** `cluster/profiles/`:`_canonical_engine.env` + 两个模型 profile
- [x] **P5.3** `cluster/entrypoint.sh` + 10 个 step,四种模式
      (`probe-only` / `install-only` / `gate-only` / `run`)
- [x] **P5.4** `cluster/jobset-64chip.yaml`(v5p 4x4x4 = 64 芯 / 16 pods × 4),
      容器 command 缩为 7 行 + `exec entrypoint.sh`
- [x] **P5.5** `cluster/README.md`(操作手册四块:怎么跑 / 期望看到 / **什么算红** / 报什么回来)
- [x] **P5.6** 顶层文档:`README.md` `ANCHORS.md` `EVIDENCE.md` `KNOWN_FOOTGUNS.md`
      `CLUSTER_ADMISSION.md` `recipes/README.md`
- [x] **P5.7** `docs/` 收录 plan/design/phase0-5(供另一侧 agent track)
- [x] **P5.8** `verify_evidence.sh` + `evidence/artifacts.sha256`

## GATE 结果

**① 证据核对(脚本,非人眼)**

```
$ ./verify_evidence.sh
  OK x8   --- ok=8 changed=0 gone=0
  EVIDENCE VERIFIED
```

**② cluster 管线实跑 —— 不碰 TPU**

`probe-only`(无 `--privileged`):
```
[env]   resolved configuration written to /tmp/canon-state/env.sh (48 lines)
[sync]  正确拒绝(worktree 的 .git 是指针文件) → 带 override 后继续
[probe] SAME x6   SUMMARY same=6 drift=0 missing=0   image matches the patch anchor exactly
[rope]  new form already present (upstream) -- not patching   ROPE_FIX=not_needed
exit=0
```

`install-only`(`JAX_PLATFORMS=cpu`):
```
[verify] A. byte identity of overlay targets -> OK x6
[verify] B. live import of the promoted chain
           OK tpu_inference.layers.jax.linear.P22XK_MATMUL_ACTIVE=True
           OK tpu_inference.layers.jax.linear.P22XK_LINEAR_BASE
           OK tpu_inference.layers.jax.embed._CANON_F4E_ANNOUNCED
           OK tpu_inference.models.jax.qwen3.P22XK_RMSNORM_ACTIVE=True
           OK tpu_inference.models.jax.qwen2.P22XK_SWIGLU_ACTIVE=True
[verify] OVERLAY VERIFIED
exit=0
```

**这一条同时端到端坐实了 P2**:安装目录是 `/tmp/canon-state/canon`,完全在 `/mnt/disks` 之外,
4 层 shim 链靠 `canon_shim_root.resolve()` 全部解析成功并加载。

**⇒ P5 GATE PASS**

## Result log

**2026-08-05 · P5 闭环**

已验证 by:`verify_evidence.sh` 实跑 8/8;`probe-only` 与 `install-only` 两个模式在真实镜像内
实跑 exit=0,且 `20_probe_image` 给出 6/6 SAME、`25_rope_fix` 给出 `not_needed`
(而非盲打)、`50_verify_overlay` 的字节比对与活体 import 双通道全绿;
`bash -n` 全部脚本;entrypoint 引用的 10 个 step 文件均存在。

未验证:`gate-only` 与 `run` 模式(需 TPU,同 P4 阻塞);真实 GKE 上的 apply(需集群访问)。

产物:`canon-zero-tim/` 共 77 个文件。
