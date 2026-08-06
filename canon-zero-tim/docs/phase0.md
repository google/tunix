# Phase 0 — 取证冻结与 Finding

Status: **✅ 闭环(2026-08-05)**
Scope: 只读审计 + 建隔离 worktree + 抽 stock 原件。**用户工作区零改动,无 commit,无 push。**

---

## Finding(已核实,全部带证据)

### F1 · release 实跑挂载的引擎侧文件 ≠ 先前会话检视的那批

`run_p26_gsm8k_train.sh:173-184` 的 `docker run -v` 实际挂载(`SP=/usr/local/lib/python3.12/site-packages/tpu_inference`):

| 容器内路径 | 宿主源文件 | 行数 |
|---|---|---|
| `layers/common/attention_interface.py` | `claude_work/tpu_inference_src/attn_iface_patched.py` | 783 |
| `layers/jax/linear.py` | **`claude_work/linear_p22xk.py`** | **88** |
| `layers/jax/embed.py` | `claude_work/tpu_inference_src/embed_patched.py` | 121 |
| `runner/tpu_runner.py` | **`tasks/.../patches/tpu_runner_p21_l30.py`** | **3428** |
| `models/jax/qwen3.py` | **`claude_work/qwen3_p22xk.py`** | **59** |
| `models/jax/qwen2.py` | **`claude_work/qwen2_p22xk.py`** | **67** |

实测差异:`linear_patched.py` vs `linear_p22xk.py` = **492 行**;
`tpu_runner.py` vs `tpu_runner_p21_l30.py` = **81 行**。

新发现的开关:`CANON_PALLAS_CANONICAL_VJP` · `CANON_PROMPT_ABSOLUTE_TARGET_IDS` ·
`CANON_ENGINE_MODULE_C` · `CANON_PALLAS_LOGSOFTMAX` · `FL_SHARED_MESH=1,4`

### F2 · 三条 shim 链,运行时闭包 = 28 个文件

```
linear:  linear_p22xk(88) → linear_p22xi(62) → linear_p22xf(185) → linear_patched(406)=stock+F4
qwen3:   qwen3_p22xk(59)  → qwen3_p22xh(139) → SRC/qwen3.py(531)=stock+89
qwen2:   qwen2_p22xk(67)  → qwen2_p22xj(43)  → qwen2_p22xg(74) → qwen2_patched(494)=stock+53
attn:    attn_iface_patched(783) → rpa_diff_chunked(274), rpa_diff(156)
embed:   embed_patched(121)                    ← 无依赖
runner:  tpu_runner_p21_l30(3428)              ← 无依赖
```

运行时支撑模块(11 个):`p22_pallas_{matmul,rmsnorm,swiglu}` ·
`p22x{f,g,h,i,j,k}_contract` · `p22xi_padded_matmul` · `p22xj_padded_swiglu` · `p22xk_vjp_ops`

- **每层 shim 都有硬编码 `/mnt/disks/tunix-data/claude_work/...` 的 `BASE_PATH`**
  (`linear_p22xk.py:12` · `linear_p22xi.py:11` · `linear_p22xf.py:11` ·
  `qwen3_p22xk.py:11` · `qwen2_p22xk.py:12`)
- `claude_work/` 里 48 个 `p22x*.py`,只有 **11 个是运行时**,其余是 `_hlo_classify` /
  `_standalone` / `_verify_raw` / `_gate0` / `_promo_*` 诊断脚本
- **模型特化**:`p22xf_contract.py` 与 `qwen3_p22xh.py` **每模型一份**
  (`qwen1p7b_p22xk/` 与 `qwen8b_p22xk/`),靠 `PYTHONPATH="$Q17:$Q8:$REPO:$W"` 顺序遮蔽,
  外加 `-v "$Q17/qwen3_p22xh.py":"$W/qwen3_p22xh.py":ro` 覆盖(因为 BASE_PATH 是绝对路径,绕过 PYTHONPATH)

### F3 · tunix 锚点 = `9fa7e251`,且**可从远程到达**

```
main..HEAD                   = 109 commits
git diff main                = 257 files / 51546 insertions   ← 含 102 个提交的【依赖】,非噪声
zero-TIM delta (9fa7e251..3a00d951) = 64 files / 11644 行 patch
git branch -r --contains 9fa7e251 → origin/yuxzhang/fix_accum_fp32   ✅ 已在 GitHub
```

18 个 zero-TIM 文件中 **10 个**在 `main` 与 `9fa7e251` 之间已不同
(`peft_trainer.py` 346+/40−、`common.py` 143+/29−、`algo_core.py` 100+/49− …)。
其中 `peft_trainer.py` 的差异来自 `15214c78 fix(sft): accumulate gradients in float32` —— 
**P26 合同的「一次 fp32 accumulated commit」建立在其上** ⇒ 锚在 main **语义上错误**,不只是会冲突。

### F4 · ★ P26 G3 是在**已提交状态**下跑的 —— 无需抢救

```
25 个金标文件(G3 raw.log 的 sha256sum 表):
  当前工作区:  MATCH=13  DRIFT=12
  DRIFT 的 12 个:  HEAD(3a00d951) 内容 == G3 金标   12/12  ✅
⇒ G3 的 release-exact 源码 == 提交 3a00d951 == p22-align-integration 的 tip
⇒ 用户的 14 个 dirty 文件是 G3【之后】的新工作,与本包无关
```

原计划的「提交 1 = 快照 dirty 文件」**取消**:证据锚已经是一个 commit。

### F5 · 引擎侧证据链有缺口(需在 EVIDENCE.md 写明)

G3 raw.log 只对 **1/28** 个引擎侧闭包文件记录了 sha256(`tpu_runner_p21_l30.py`
= `b8b1e118ca3cb353`,已核对一致)。其余 27 个**未被记录**。

替代证据:28 个闭包文件 mtime **全部 ≤ 2026-08-03 22:51**,而 G3 冻结 runner mtime =
**2026-08-04 17:48**,raw.log 完成 = **2026-08-04 20:29** ⇒ 全部早于 run ≥19 小时,无一被改动。
**这是 mtime 级证据,不是密码学级** —— 必须在 EVIDENCE.md 如实标注。

**后续建议(非本任务范围)**:P26 runner 的 `sha256sum` 块应补上全部 6 个引擎挂载点及其闭包。

### F6 · 🔴 安全:仓库 remote URL 内嵌明文 GitHub PAT

`sequence_packing/tunix/.git/config` 的 origin URL 含 `ghp_…` token(对 `google/tunix` 有 push 权限)。
**未写入任何文件**。已当面报告,建议 rotate。打包时 `.git/config` 绝不可进包。
现有 secret scan 覆盖 W&B run tree,**`.git/config` 是盲区**。

---

## Execution

- [x] **P0.1** `git worktree add /mnt/disks/tunix-data/canon_zero_tim_wt -b yuxzhang/canon-zero-tim 3a00d951`
      (从 tip 切,**不用** checkout;用户工作区零接触)
- [x] **P0.2** 生成 patch 07 = `git diff 9fa7e251 3a00d951`,11644 行 / 64 文件
- [x] **P0.3** 从镜像 `418dc632edd8` 抽出 6 个 stock 原件
- [x] **P0.4** 记录闭包 SHA 清单(引擎 31 条 + stock 6 条)

**GATE 结果**

| # | 判据 | 结果 |
|---|---|---|
| ① | `git apply --check -R` patch 07 | ✅ 干净 |
| ② | 新 worktree 25 个 tunix 文件 SHA vs G3 金标 | ✅ **MATCH=25 / DRIFT=0** |
| ②b | `tpu_runner_p21_l30.py` SHA vs G3 金标 | ✅ `b8b1e118ca3cb353` |
| ②c | 其余 27 个闭包文件 | ⚠️ G3 未记录 SHA;mtime 全部早于 run ≥19h(见 F5) |
| ③ | stock 原件抽取(6 个) | ✅ 667/65/284/2857/442/441 行 |
| ④ | 用户工作区未被打扰 | ✅ `HEAD=3a00d951 branch=p22-align-integration dirty=14`,前后一致 |

**⇒ P0 GATE PASS**

---

## 抽出的 stock 原件(锚点 = 镜像 `tunix_frozenlake_image:vllm-tpu0.25.0` / `418dc632edd8`)

| 文件 | 行数 | sha256(前16) |
|---|---|---|
| `layers/common/attention_interface.py` | 667 | `52c6d50347e3a155` |
| `layers/jax/embed.py` | 65 | `dfbef8f419254e37` |
| `layers/jax/linear.py` | 284 | `2d3731c50a587225` |
| `models/jax/qwen2.py` | 441 | `0abc320480ec6a80` |
| `models/jax/qwen3.py` | 442 | `6a120c5820ca8d41` |
| `runner/tpu_runner.py` | 2857 | `71dc7559a3183f73` |

---

## Result log

**2026-08-05 · P0 闭环**

已验证 by:
- `run_p26_gsm8k_train.sh:173-184` 挂载列表(读)
- `sha256sum` 实测 25 个 tunix 文件 vs G3 raw.log 金标表 → 新 worktree 25/25
- `git show HEAD:<file> | sha256sum` → 12 个 drift 文件 12/12 可从 HEAD 恢复
- `git worktree add` 后前后对比用户工作区 HEAD/branch/dirty count → 完全一致
- `git apply --check -R` patch 07 → 干净
- `docker create` + `docker cp` 抽出 6 个 stock 原件 → 行数合理
- 闭包追踪脚本(scratchpad `closure.py`)沿 `BASE_PATH` + 本地 import 求传递闭包 → 28 文件
- `stat -c %y` 28 个闭包文件 mtime 全部 ≤ 2026-08-03 22:51 < G3 run 2026-08-04 17:48

未验证:
- 27 个引擎侧闭包文件与 G3 的**密码学**同一性(G3 未记录其 SHA;仅有 mtime 级证据 —— 见 F5)
- patch 07 能否 apply 到 `main`(**已证不该做**:10/18 文件在 main 与 9fa7e251 间已分叉,
  且 `peft_trainer.py` 的分叉是 P26 的语义依赖)

产物(scratchpad,尚未入包):
`07-training-hooks.patch` · `engine_closure_sha.txt`(31 条) · `stock_sha.txt`(6 条) ·
`g3_golden_sha.txt`(25 条) · `stock/`(6 个 stock 原件)

对计划的修正:
- 取消原「提交 1 = 快照 dirty 文件」(F4:证据锚已是 commit `3a00d951`)
- patch 清单以 F1 表为准;`src/engine_shims/` 装 F2 的 28 文件闭包 + 两套模型特化
- patch 07 定位为**文档**,非组装步骤(分支 checkout 即可用)
- EVIDENCE.md 必须写明 F5 的证据缺口
