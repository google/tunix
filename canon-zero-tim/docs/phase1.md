# Phase 1 — 真 diff 化

Status: **✅ 闭环(2026-08-05)**
Scope: 把引擎侧的「整文件副本」还原成相对 stock 的真 patch。锚点 = 镜像 `418dc632edd8`。

---

## Finding

P0.F2 已证:6 个挂载点里有 3 个是 shim 链的入口(`linear_p22xk` / `qwen3_p22xk` /
`qwen2_p22xk`),但**每条链的链底仍是「stock + 改动」的整文件副本**:

| 挂载点 | stock 行数 | 链底 patched 文件 | 行数 |
|---|---|---|---|
| `layers/common/attention_interface.py` | 667 | `attn_iface_patched.py` | 783 |
| `layers/jax/embed.py` | 65 | `embed_patched.py` | 121 |
| `layers/jax/linear.py` | 284 | `linear_patched.py`(链底) | 406 |
| `models/jax/qwen3.py` | 442 | `SRC/qwen3.py`(链底) | 531 |
| `models/jax/qwen2.py` | 441 | `qwen2_patched.py`(链底) | 494 |
| `runner/tpu_runner.py` | 2857 | `tpu_runner_p21_l30.py` | 3428 |

⇒ **一文件一 patch**,共 6 个。shim 层不属于 patch(它们是新增模块,归 P2)。

## Execution

- [x] **P1.1** `diff -u stock patched` 生成 6 个 patch,落 `patches/tpu_inference/`
- [x] **P1.2** GATE:把每个 patch 打回 stock 的干净副本,与链底 patched 文件比 SHA-256

## GATE 结果

```
=== stock + patch 是否逐字节等于 patched ===
  ✅ 01-attention-interface.patch   171 行   → 8f8cec35f296
  ✅ 02-embed.patch                  72 行   → 4a0540323c75
  ✅ 03-linear.patch                152 行   → 9eba5881f7d1
  ✅ 04-qwen3.patch                 122 行   → fd9b2fc584c3
  ✅ 05-qwen2.patch                  79 行   → 6e533ceb30c5
  ✅ 06-tpu-runner.patch            656 行   → b8b1e118ca3c
  PASS=6  FAIL=0
```

**⇒ P1 GATE PASS** —— 6 个 patch 无损、无遗漏、无夹带。

**额外证据**:`06-tpu-runner.patch` 产出的 `b8b1e118ca3c…` **正是 P26 G3 raw.log 里
密码学记录的那个 SHA**(`b8b1e118ca3cb353`)⇒ patch 链复现了已认证的 release 文件本身,
不只是"一个看起来一样的文件"。

## 结果的意义

引擎侧的全部改动 = **1252 行 patch**(171+72+152+122+79+656)。
此前它以 **5588 行整文件副本**(783+121+406+531+494+3428 减去对应 stock)的形式存在,
读者无法分辨哪些是本项目的改动。现在可以逐行读。

## Result log

**2026-08-05 · P1 闭环**

已验证 by:`diff -u` 生成 + `patch` 回放到 stock 干净副本 + `sha256sum` 比对链底文件,6/6 相同。
未验证:patch 在**其他版本**的 tpu_inference 上是否 apply 干净(锚点明确写死为镜像
`tunix_frozenlake_image:vllm-tpu0.25.0` / `418dc632edd8`,换镜像须重新生成)。

产物:`canon-zero-tim/patches/tpu_inference/0{1..6}-*.patch`
