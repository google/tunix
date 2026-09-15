# P58 DeepSWE Qwen3-4B 128-chip — `jit_compute_per_token_logps` 编译产物超过 Pathways 2 GiB 上限

JobSet: `canon-p58-128s-zsoptt-three-timbd06`
日期: 2026-09-15 (08:44Z 启动 → 10:24Z 崩溃，存活 1h40m)
Source commit: `fdc67a38f6c50d6da1e3542ac6e739e00ff46c96`
Profile: `cluster/profiles/qwen3-4b-dp8-tp8-deepswe-tim-systemopt.env`
Arm: `CANON_P58_TIM_ARM=zero` / `CANON_DEEPSWE_SYSTEM_OPTIMIZATION_ARM=treatment`
Stage: `three-update` (`CANON_P58_EXPECTED_UPDATES=3`)

> [!IMPORTANT]
> 本报告的每一条结论都附了产生它的**原始日志行**或**文件:行号**。
> 推断出来的结论显式标注为【推断】。取不到的标 UNKNOWN。

---

## 0. TL;DR

`update 0` 的 old-policy logprob 前向，其单个 SPMD 程序的编译产物达
**2,715,131,245 bytes**，超过 Pathways `CompilationResponseProto` 的
**2,147,418,111 bytes**（2³¹）上限。

体积驱动量是 `canonical_qwen3_adapter.py:14432` 的
**`for chunk_index in range(num_chunks)` —— 一个被完整 inline 进 HLO 的 python 循环**，
`num_chunks = (max_prompt 4096 + max_response 16384) / local_M 256 = 80`，
每个 chunk 内含完整 36 层 transformer forward。

```
2,715,131,245 / 80 chunks ≈ 33.9 MB / chunk
要降到上限以下需 ≥ 21% 降幅  →  num_chunks ≤ 63
```

**这是单个 SPMD 程序，体积与设备数无关。降 DP / 换拓扑不会让它变小。**

---

## 1. 崩溃原文

```
W0915 10:24:22.713850 1351 rpc_helper.cc:436] compile: ABORTED:
  Compilation service has crashed or shut down while compilation was pending
  for computation with key: 511c18287d024b9c,
  computation name: jit_compute_per_token_logps,
  original error: DATA_LOSS: learning_pathways.CompilationResponseProto
  (2715131245 bytes) exceeds maximum supported size 2147418111
    Server pipe /compilation_service id=5761549897086367622
=== Source Location Trace: ===
third_party/pathways/data_parallel/pipe.cc:307

[PERF] step=0 stage=trainer_old seconds=3698.687 rows=128
```

调用栈（`crash_traceback.log` 全文）：

```
examples/deepswe/train_deepswe_nb.py:2153   agentic_grpo_learner.train(...)
tunix/rl/agentic/agentic_rl_learner.py:4229 _batch_to_train_example
tunix/rl/agentic/agentic_rl_learner.py:3488 _process_results
tunix/rl/agentic/agentic_grpo_learner.py:1070 rl_cluster.get_actor_per_token_logps
tunix/rl/rl_cluster.py:1458                 common.compute_per_token_logps
```

崩溃前**耗时 3698.687 秒（≈62 分钟）编译**才报出超限 —— 任何验证性发射都要预留这个成本。

---

## 2. 根因链（逐级可查）

### 2.1 jit 名字来源

`tunix/rl/common.py:318-330` 用 `@functools.partial(jax.jit, static_argnames=(...))`
装饰 `compute_per_token_logps` ⇒ 崩溃日志里的 `jit_compute_per_token_logps`。

### 2.2 走的是 canonical adapter，不是原生 NNX

`common.py:390-410`：`canonical_actor=True` 且 `canonical_forward.enabled()` 时委派给 adapter。

`canonical_forward.py:48-49`：`enabled()` 读 `CANON_ENGINE_MODULE_C == "1"`。

profile 链：
```
qwen3-4b-dp8-tp8-deepswe-tim.env:13  source qwen3-32b-dp16-tp8-deepswe.env
qwen3-32b-dp16-tp8-deepswe.env       source _canonical_engine.env
_canonical_engine.env:47             export CANON_ENGINE_MODULE_C=1
```
`qwen3-4b-dp8-tp8-deepswe-tim.env:109` 的 `CANON_ENGINE_MODULE_C=0` 只在 **`native)`** 分支里
（`zero)` 分支从 :130 开始，不覆盖）⇒ **zero 臂实际值为 1**。

### 2.3 80× 展开点

`tunix/rl/canonical_qwen3_adapter.py:14363-14369`
```python
full = jnp.concatenate((prompt, completion), axis=1)
num_chunks = (full.shape[1] + self._sequence_bucket - 1) // self._sequence_bucket
padded_width = num_chunks * self._sequence_bucket
```
`tunix/rl/canonical_qwen3_adapter.py:14425-14432`
```python
print("[PATHTRACE] CANON_ADAPTER_DP_FIXED_M_CHUNKS ...")
for chunk_index in range(num_chunks):          # ← python 循环，全展开进 HLO
```

运行时实测打印（`run.log:17041`）：
```
[PATHTRACE] CANON_ADAPTER_DP_FIXED_M_CHUNKS data=8 static_width=20480
            chunks=80 global_M=2048 local_M=256
```

### 2.4 `local_M=256` / `global_M=2048` 从哪来

`canonical_qwen3_adapter.py:2320-2358`（`_canonical_topology_contract`）
```python
local_m  = int(os.environ.get("CANON_LOGPROB_M", "0"))
global_m = int(os.environ.get("MIN_TOKEN_BUCKET", "0"))
target_m = int(os.environ.get("CANON_TARGET_M", "0"))
...
if local_m != 256 or target_m != local_m:
    raise FunctionalMappingError(
        "P32 training requires CANON_LOGPROB_M=CANON_TARGET_M=256; ...")
if global_m != data_size * local_m:
    raise FunctionalMappingError(
        "P32 training requires MIN_TOKEN_BUCKET=dp*CANON_LOGPROB_M; ...")
```
`_sequence_bucket = global_M / dp = 2048 / 8 = 256`（`:6721-6727`）。

---

## 3. 一个容易误判的陷阱（请先读，否则会得出相反结论）

`trainer_old` 窗口（`run.log:17038`–`18272`）内的 PATHTRACE 计数**看起来**像是
「层栈只 trace 了一次」：

```
CANON_PALLAS_SWIGLU        36     ← 恰好 = Qwen3-4B 的层数
CANON_PALLAS_ALL_PROJ     252     ← 252/36 = 7 = q,k,v,o,gate,up,down
CANON_P38_FIXED_LM_HEAD     1     ← 只有 1 次
```

**这个读数不能用来推断展开倍数。** `canon-zero-tim/src/engine_shims/p38_fixed_lm_head.py`
里那个 print（:524-532）**没有任何 once/dedup/lru_cache 守卫**，却在 80 个 chunk 都要调
head 的情况下只出现 1 次。唯一自洽的解释是：

> `_trainer_compute_logits_fn` 与层栈都是**预构建的缓存程序**，
> trace 只发生一次（所以只打印一次），但在 python 循环里被 **inline 80 份进 HLO**。

⇒ **PATHTRACE 计数 ≠ HLO 展开倍数。**

---

## 4. 已被逐条证伪的修复路径（不要重复走）

| 路径 | 证伪依据 |
|---|---|
| 降 `compute_logps_micro_batch_size` 8→4 | 外层是 `jax.lax.map`（`canonical_qwen3_adapter.py:15019`），body 只 trace 一次 ⇒ HLO 体积与 batch 无关。且撞 `train_deepswe_nb.py:793` 签名断言 + `render_p58_deepswe_tim.py:1302` 渲染断言 |
| 设 `compute_logps_chunk_size=2048` | adapter `:14927-14930` 对 `chunk_size != 0` **直接 raise**（"canonical engine adapter owns its fixed-M chunking"）。FrozenLake `train_frozenlake.py:740` 能用是因为它走原生 NNX 路径 |
| `CANON_P78_SEGMENTED_ACTOR_LOGPS=1` | `:14689-14699` 硬钉 `frozenlake-p45-onehost-dp4-tp1` + DP4 + TP1。DeepSWE 是 DP8×TP8，开了会**更早** raise。（注：host lengths 前置条件是满足的，见 `agentic_grpo_learner.py:995-1001,1078-1079,1102-1103`；唯一障碍就是这道白名单）|
| 抬高 `local_M` 256→512（`MIN_TOKEN_BUCKET`→4096）| `deepswe_contract.py:303` 要求 `max_num_batched_tokens_per_dp == 256`（焊死）；`:309` 要求 `max_num_batched_tokens_per_dp * dp_size == global_m` ⇒ `global_m` 被强制 `= 256 × dp_size`；`:232` 要求 `dp_size * local_m == global_m` ⇒ **`local_m` 被强制 = 256**。`MIN_TOKEN_BUCKET` 同时是 vLLM rollout 调度器的全局 token 容量，抬它会与 `--max_num_batched_tokens=256` 冲突（`FLAGS.md:540` 标注「钉死 256 族；永不自由化」）|
| 降到 64split (DP4×TP8) | 见 §3：这是**单个 SPMD 程序**，`num_chunks` 在 DP4 下**仍是 80**（`local_M` 恒为 256）。体积与设备数无关。<br>⚠️ **bd04 不能用作反证**：它是 `--batch_size=4 --num_generations=4` 的 P44 小规模 parity 探针（16 条轨迹），`grep -c jit_compute_per_token_logps = 0`，**根本没走到这个编译** |

---

## 5. 仍然成立的候选修复

| 方案 | 机制 | 预计产物 | 主要代价 |
|---|---|---|---|
| **B 缩短序列宽度** | `num_chunks = width/256`，线性 | width 12288 → 48 chunks → ≈1.63 GB<br>width 15360 → 60 chunks → ≈2.04 GB | 零代码风险，但**改变实验语义**（截断轨迹）；要同步改 `render_p34_jobset.py:343` / `render_p58_deepswe_tim.py:1304` / `train_deepswe_nb.py:798` 三处断言 |
| **C′ trainer 侧 chunk 宽度 ×2** | 只把 logprob 前向的 chunk 从 256 改成 512 token，**不动** `MIN_TOKEN_BUCKET` / rollout 调度契约 | 40 chunks → ≈1.36 GB | 改 adapter 核心前向；要放宽 `:14490` 的 `logits.shape == (self._bucket, vocab)` 契约。semantic_M 会变 4096，**P38 注册表已收录**（`p38_fixed_lm_head.py:19 QWEN4B_TP8_LEARNER_M = (2048, 4096)`），不需动 registry |
| **C 把 `:14432` 的 python for 改成 `lax.scan`/`fori_loop`** | 结构性修复 | 降约 80× | 工作量最大，改核心前向，需逐位 parity 认证 |

> [!NOTE]
> 上表「预计产物」均基于 **HLO 体积 ∝ num_chunks 且截距为 0** 的线性外推。
> 截距未实测 ⇒ **【推断】**。B 方案取 60 chunks 时余量只有 5%，而一次验证要 62 分钟编译。

---

## 6. 本次运行已经证明成立的事（不要重跑验证）

bd06 之前的 bd05 死于另一个完全不同的原因（R2E-Gym 的 `pods/exec` 跨 namespace 403），
**已修复并提交在 `fdc67a38`**。bd06 证明了该修复生效，且以下全部为运行时实测 PASS：

```
[P34.DEVICE_INVENTORY] PASS devices=128 hosts=32 devices_per_host=4
                       rollout_hosts=16 trainer_hosts=16
[P34.TOPOLOGY]         PASS rollout_devices=64 trainer_devices=64
Memory statistics | total_hbm_limit_gb=6080.0GiB | total_hbm_limit_cap_gb=3648.0GiB
                  | total_hbm_used_gb=59.96GiB | total_hbm_avail_gb=3588.04GiB
[P58.LOGPS_BATCH] configured_prompts=8 generations=16
                  execution_trajectories=128 observed_trajectories=128
```

- 128 个 R2E 沙箱全部 Running（`sandbox-cpu-pool`，trellis namespace）
- `pods/exec` 403：**0 次**（bd05 时每条 rollout 都有）
- `[PERF] rollout_generate`：**1,864 次**（bd05 为 0）
- `RepoEnv created`：128（全部）
- run.log 行数 18,326（bd05 仅 2,543）
- 真实 OOM / RESOURCE_EXHAUSTED：**0 次**

⇒ **放置、沙箱、命名空间、rollout 全链路已通。唯一阻塞就是这个编译上限。**

---

## 7. 复现与取证方法

- 日志唯一可靠来源是 PVC 上的 `CANON_RUN_LOG`：
  `/mnt/disks/linchai_data/deepswe_zero_tim/<jobset-name>/run.log`
  （head pod 随 JobSet 失败即销毁，worker 所在 TPU 节点被 NAP 回收，`kubectl logs` 立即失效）
- PVC `haoyugao-cpu-np-pvc` 在 **trellis** namespace
- trellis ns 开了 Kueue pod integration ⇒ 捞日志的裸 Pod **必须**带
  `kueue.x-k8s.io/queue-name: multislice-queue`，否则被 gate 住

## 8. 本目录内容

| 文件 | 说明 |
|---|---|
| `run.log` | bd06 完整 run.log，18,326 行 |
| `crash_traceback.log` | `run.log:18260-18326`，编译错误 + 完整 python traceback |
| `ERROR_REPORT.md` | 本文件 |
| `SHA256SUMS` | 校验和 |
