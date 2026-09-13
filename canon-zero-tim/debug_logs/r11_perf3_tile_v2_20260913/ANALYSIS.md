# r11 perf3 / tile-v2 — 64-chip measurement and rollout bottleneck analysis

Date: 2026-09-13
Source commit under test: `3f95bdc762b93b59ad2c8fb2f7a332ea5b3548d4`
JobSet under test: `canon-p57-fl-zero-m15-r11-3f95bdc7`
Baseline JobSet: `canon-p57-fl-zero-m15-r10-06a0fdb9` (commit `06a0fdb9`)
Topology: dp8 x tp8 = 64 v5p chips (4x4x4), profile `qwen3-8b-dp8-tp8-frozenlake-v1-hp`
Workload: FrozenLake m15 (15-turn), zero arm, 300 updates, checkpoint disabled, eval disabled

Authoritative timing metric: WandB `perf/train/global_step_time`
(entity `yuxzhang-google`, project `zero-tim-p57-frozenlake-tim`).

---

## 1. Headline result

Step-index-aligned comparison. Both runs use the same data split, the same
`seed=42`, and byte-identical `CANON_RUN_CMD` apart from the commit SHA.

See `step_comparison.md` for the generated table and
`raw/r11_step_times.json` / `raw/r10_baseline_step_times.json` for the source data.

| step | r11 s | r10 s | gain | r11 solve | r10 solve | note |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 1927.8 | 2979.0 | 35.3% | 0.176 | 0.223 | compile |
| 1 | 539.1 | 1284.7 | 58.0% | 0.227 | 0.195 | |
| 2 | 512.5 | 1398.0 | 63.3% | 0.176 | 0.215 | |
| 3 | 523.7 | 1439.1 | 63.6% | 0.266 | 0.207 | |
| 4 | 555.6 | 1582.0 | 64.9% | 0.262 | 0.227 | |
| 5 | 617.3 | 1574.4 | 60.8% | 0.223 | 0.242 | |
| 6 | 573.1 | 1708.9 | 66.5% | 0.375 | 0.316 | |
| 7 | 575.5 | 1525.1 | 62.3% | 0.293 | 0.281 | |
| 8 | 634.0 | 1657.9 | 61.8% | 0.230 | 0.293 | |
| 9 | 588.2 | 1848.6 | 68.2% | 0.297 | 0.293 | |
| 10 | 766.6 | 2079.8 | 63.1% | 0.324 | 0.352 | **audit ON** |
| 11 | 640.6 | 2302.9 | 72.2% | 0.230 | 0.281 | |
| 12 | 594.5 | 2146.1 | 72.3% | 0.262 | 0.297 | |

Steady state (step >= 1): mean **64.7%**, min 58.0%, max 72.3%, n=12.

Solve ratio is not degraded. Step 0 differs (0.176 vs 0.223), so the two runs are
not bitwise reproductions of each other. That is expected: this wave includes
`CANON_DP_REDUCE_ONCE` and `CANON_P32_LENGTH_SORT`, which change reduction order,
and vLLM continuous batching is itself nondeterministic in request ordering.

### Baseline correction (important)

An earlier reading of this experiment used a single baseline of **2843.5 s**,
which is r10 at step 147. That is wrong. r10's own step time grows from
**1284.7 s at step 1 to ~2870 s after step 100** as completion length grows.

r10 full curve: `median=2869.5s  mean=2749.2s  min=1284.7 (step 1)  max=2979.0 (step 0)`;
steady (step>=5) median `2872.0s`.

Any comparison must be step-index aligned. Numbers derived from the flat 2843.5 s
baseline (rollout "4.81x", token throughput "2.20x", per-call latency "2.33x")
are systematically inflated and must not be quoted.

---

## 2. Tile v2 gate: confirmed firing on the tp8 contract

`f4717eea` records that on dp8-tp8 the old tile policy never fired, because the
Qwen3-8B tp8 contract passes `(128,128,128)` while the policy only recognised
`(128,256,256)`. `src/engine_shims/p22_pallas_matmul.py:40` now has
`CONTRACT_TILES = frozenset({(BM, BN, BK), (128, 128, 128)})`.

r11 is the first 64-chip run where the knife is not a no-op. Receipts in
`raw/pathtrace_tile_receipts.log` (16 unique matmul shapes, 17 rmsnorm shapes):

| use | M | K | N | bm | bn | bk |
|---|---:|---:|---:|---:|---:|---:|
| gate/up (trainer) | 2048 | 4096 | 1536 | 256 | 512 | 128 |
| down (trainer) | 2048 | 1536 | 4096 | 256 | 1024 | 128 |
| q | 2048 | 4096 | 512 | 256 | 512 | 128 |
| o | 2048 | 512 | 4096 | 256 | 1024 | 128 |
| k/v | 2048 | 4096 | 128 | 256 | 128 | 128 |
| LM head | 256 | 4096 | 19200 | 256 | 256 | 256 |
| decode gate/up | 128 | 4096 | 1536 | 128 | 512 | 128 |
| decode down | 128 | 1536 | 4096 | 128 | 1024 | 128 |

- `bk=128` is preserved on every shape: the tp8 k-block contract is honoured exactly.
- `bm`/`bn` widen to the micro-benchmark's fast configurations.
- `N=128` k/v correctly falls back to the caller's `bn`.
- rmsnorm takes `bm=256` wherever `rows % 256 == 0`, and keeps `bm=8` below that envelope.

Other knives in this wave, confirmed active:

- `CANON_ALIGNMENT_AUDIT_EVERY=10`: `[CANON_ALIGN] audit=skip step=1 every=10 rescore_b=0 trainer_old=0`.
  First exercise of `67c38ebf`'s sidecar presence check at 64 chips. `[CANON_ALIGN_PRE] step=1 verdict=PASS N_action=121483 bounds=[]`.
- `CANON_P32_LENGTH_SORT=1`: `n_real` is strictly monotone decreasing across the 32
  forward groups (`7112 -> 5994 -> 5344 -> 5183 -> 4373 -> ... -> 1365`).
- `[ENGINE_STEP]` / `[ENGINE_DRIVER]` from `3f95bdc7` emit correctly at 64 chips.

---

## 3. Stage decomposition

### step 1 (first compile-free step), total 539.1 s

Boundaries taken from `kubectl logs --timestamps`; the parts sum to the reported total.

| stage | window | s | share |
|---|---|---:|---:|
| rollout | 03:02:52 -> 03:10:10 | 438 | 81.2% |
| audit (`rescore_b` + `trainer_old`) | skipped | 0 | 0% |
| p32 reverse, 32 groups | 03:10:11 -> 03:11:44 | 93 | 17.2% |
| `weight_sync` | | 8.7 | 1.6% |

`weight_sync` sub-items: `engine 6.44s / gc 2.18s / anchor_d2h 0.09s`.
r10 step-147 reference: `p32_vag_reverse 476.4s`, `rescore_b 230.9s`, `weight_sync 14.2s`.

### step 0 compile cost

| stage | s |
|---|---:|
| rollout (02:31:09 -> 02:38:30) | 441 |
| `trainer_old` | 349.0 (compile-inflated) |
| `rescore_b` | 77.3 |
| p32, 32 groups (02:47:54 -> 02:59:27) | 693 |
| `weight_sync` | 7.9 |

p32 group cadence shows the compile clearly: group1->2 **421 s**, 2->3 **166 s**,
3->4 31 s, then steady **2-5 s/group**; groups 4->30 (26 groups) total only **71 s**.

### Attribution, resolved at step 10

r10 has no `CANON_ALIGNMENT_AUDIT_EVERY`, so it audits on every step while r11
audits 1 step in 10. Separating the two effects needs a compile-free audit step,
which is step 10 (`raw/step10_audit_evidence.log`):

```
[PERF] step=10 stage=trainer_old seconds=  2.985 rows=256
[PERF] step=10 stage=rescore_b   seconds= 71.005 rows=256
[PERF] step=10 stage=weight_sync seconds=  9.797
[step 10] train_solve=0.324 ... time=766.6s
```

`trainer_old` in steady state is **2.985 s** against **349.0 s** at step 0, so
**99.1% of the step-0 figure was compile**. Using step 0 would have overstated the
audit cost by 5.8x. The true steady-state audit cost is `2.985 + 71.005 = 74.0 s`,
not 426.3 s.

| | s |
|---|---:|
| r11 non-audit steps (8, 9) | 634.0, 588.2 -> mean **611.1** |
| r11 audit step (10) | **766.6** |
| net cost of auditing | **+155.5** (74.0 in the two `[PERF]` stages, the remaining ~81 in `CANON_ALIGN_PRE` host sidecar attach and related work) |
| r10 same range (8, 9, 10) | 1657.9, 1848.6, 2079.8 -> mean **1862.1** |

Averaged over one 10-step cycle:

```
r11 as configured        = (9 * 611.1 + 766.6) / 10 = 626.7 s
r11 if it audited always =                            766.6 s   (same policy as r10)
r10                      =                           1862.1 s
```

| source of gain | percentage points |
|---|---:|
| **tile v2 + backward knives** (`DP_REDUCE_ONCE`, `P32_LENGTH_SORT`) | **58.8** |
| audit skipping (`AUDIT_EVERY=10`) | 7.5 |
| total | 66.3 |

**About 89% of the gain (58.8 of 66.3 points) comes from tile v2 and the backward
knives themselves.** Even if the audit were restored to r10's every-step policy,
r11 would still be 58.8% faster. Step 10 against step 10 directly:
766.6 s vs 2079.8 s, 63.1%.

For reference, `rescore_b` in steady state is 71.0 s while r10 at step 147 reports
230.9 s. Those are different step indices and the ratio must not be quoted.

---

## 4. Rollout is now the dominant cost, and the instrumentation cannot attribute it

rollout is 81.2% of a step. The `[ENGINE_STEP]` / `[ENGINE_DRIVER]` receipts from
`3f95bdc7` were used to look inside it. **The conclusion below is a negative
result about the instrumentation, and it invalidates a first reading of the data.**

### 4.1 What the numbers look like

663 `std`-path samples, bucketed by request count (`raw/engine_step_lines.log`):

| bucket | n | sample_ms | exec_ms | sched_tok |
|---|---:|---:|---:|---:|
| reqs <= 1 | 7 | 154.53 | 6.09 | 256 |
| reqs 2-4 | 21 | 154.89 | 6.14 | 443 |
| reqs 5-16 | 54 | 153.91 | 6.25 | 519 |
| reqs 17-64 | 303 | 156.12 | 6.94 | 2048 |
| reqs > 64 | 278 | 157.05 | 7.64 | 2048 |

`sample_ms` varies by 1.6% while load varies 64x. Global min 148.21, max 162.68.
24 `cd`-path samples have `sample_ms = 0.00` exactly (min 0, max 0).

Window decomposition (`raw/engine_step_summary.log`):

```
[ENGINE_STEP_SUMMARY] n=8500 window_s=94.36 steps=500 reqs_mean=59.2 fill=1.000
    exec_ms_sum=4142  gap_ms_sum=12058  sample_ms_sum=78051  sync_ms_sum=0
    gen_tok=25614 tok_per_s=271.5
```

`4142 + 12058 + 78051 = 94251 ms` against `window_s = 94360 ms`: the three parts
are a complete, non-overlapping, serial decomposition of wall time (0.1% residual).

### 4.2 Why the obvious reading is WRONG

The obvious reading is "the forward pass costs 6-7 ms and a fixed 156 ms of host
sampling overhead dominates". **That reading is incorrect.** Reading the
instrumentation in `patches/tpu_inference/39-tpu-runner-perf3-engine-step-log.patch`:

```python
def execute_model(...):
    _perf3_rec = _engine_step_begin(self, scheduler_output)
    with jax.set_mesh(self.mesh), jax.profiler.TraceAnnotation(...):
        output = self._execute_model(scheduler_output, intermediate_tensors)
    _perf3_rec["exec_ms"] = (time.perf_counter() - _perf3_rec["t_begin"]) * 1e3
    if _perf3_rec["path"] in ("cd", "empty"):
        # continue-decode already synced on device_get; nothing left to time.
        _engine_step_emit(_perf3_rec)

def sample_tokens(self, grammar_output):
    self._perf3_t_sample = time.perf_counter()
```

JAX dispatch is asynchronous. `_execute_model` returns once the ops are enqueued;
it does not wait for the device. Therefore:

- **`exec_ms` measures Python dispatch time, not device time.**
- The first blocking synchronisation is the token readback inside `sample_tokens`,
  so **`sample_ms` absorbs the device execution time of the forward pass.**

The patch author's own comment is the direct evidence: continue-decode "already
synced on device_get; nothing left to time" — implying the `std` path has *not*
synced at that point. The data agrees exactly:

| path | where the sync happens | exec_ms | sample_ms |
|---|---|---:|---:|
| `cd` | inside `_execute_model` | 97.03 | 0.00 |
| `std` | inside `sample_tokens` | 7.18 | 156.30 |

These are two views of one phenomenon, not two different costs.

Likewise, `sample_ms` being flat across load is **not** evidence of fixed host
overhead. Every `[ENGINE_STEP]` line carries `pad_tok=2048 pad_reqs=8 pad_per_rank=256`:
the program is padded to a constant 2048-token shape, so constant device time is
the expected consequence of padding.

`sync_ms = 0` on every line, and emission happens in `_engine_step_sample_done`
rather than in `get_output()`, which indicates `sample_tokens` is running
synchronously rather than through the async-output path.

### 4.3 What is actually established

| statement | status |
|---|---|
| One `std` engine step costs **183.4 ms** wall (gap 19.9 + blocking 163.5) | established, n=663 |
| Device forward + sampling + readback are **inseparable** in these receipts | established by construction |
| Forward is only 6-7 ms | **RETRACTED** — that is dispatch time |
| The 156 ms is fixed host overhead | **RETRACTED** — padding explains the flatness |
| `cd` path is "14.6x cheaper" | **RETRACTED** — it is a different sync point, not a different cost |

### 4.4 The efficiency problem is real regardless

The padded program is 2048 tokens and `fill` is 0.851-1.000, so the program is
genuinely full — this is not padding waste.

```
FLOPs  = 2 * 8e9 params * 2048 tokens          = 3.3e13
compute= 64 chips * ~459 TFLOP/s (v5p bf16)    = 2.9e16 FLOP/s
ideal  = 1.1 ms        measured = 163.5 ms     -> ~150x
MFU    ~ 0.7%
```

This estimate depends only on "a 2048-token program took 163.5 ms", which is a
direct measurement. Rollout therefore has large headroom, but **these receipts
cannot say where the time goes.**

Also recorded: `sched_tok / gen_tok` is **40.0** and **52.3** in the two summary
windows, and `prefill_reqs > 0` on **663/663** `std` steps. This is the expected
signature of a 15-turn rollout re-prefilling conversation history every turn.

---

## 5. Constraints on the obvious remedies

### Prefix caching: blocked

`cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-v1-hp.env:28,46` set
`CANON_VLLM_ENABLE_PREFIX_CACHING=0`, and line 196 carries it as a certified
validation case rather than a plain default.

> **Prefix caching currently has a bug and must not be enabled.** (owner guidance,
> 2026-09-13). Do not re-propose it as a rollout optimisation until that bug is fixed.

### Scheduler geometry: pinned by contract

`FLAGS.md:62` records `MIN_TOKEN_BUCKET / max_num_batched_tokens` as
"pinned to the 256 family, certified, never to be liberalised; new geometry goes
through contract registration".

`.claude/skills/manage-canon-zero-tim-branch/references/shape-contracts.md:83-125`
adds that `max_num_batched_tokens` and `max_num_seqs` are **per DP rank** and are
multiplied by `dp_size` by the runner. With dp8, `--vllm_max_num_seqs=32` means a
global concurrency of 256, not 32. Passing the wrong pair historically produced
token buckets `[4096,8192,16384,32768,65536]` and five backbone precompiles.

An earlier suggestion to raise `max_num_seqs` in a follow-up run was therefore
wrong on both counts: the scheduler is not the binding constraint
(`fill=1.000`, `idle_polls=0`, `reqs_mean=59.2`), and the value is contract-pinned.

### Sampling-path patches that exist but are not in the profile chain

| patch | env switch | present in profile chain? |
|---|---|---|
| `22-tpu-runner-p56-logprob-step-fusion` | `CANON_LOGPROB_STEP_FUSION` | yes, `=1` (v1-hp.env:16) |
| — | `CANON_PALLAS_GATHERED_LOGPROBS` | yes, `=1` (v1-hp.env:15) |
| `21-tpu-runner-p56-logprob-readback` | `CANON_ENGINE_LOGPROB_READBACK` | **not found** |
| `23-tpu-runner-p56-sample-split-fusion` | `CANON_SAMPLE_SPLIT_FUSION` | **not found** |

Both unset patches claim bitwise equivalence in their own comments (21 collapses
three sequential `device_get` calls into one; 23 fuses rng-split, f32 cast and
sample into a single jit program).

> **UNVERIFIED.** This is a grep over the profile files only. The live pod
> environment was not read: `kubectl get pod -o json` hung for 37 minutes without
> returning. Confirm with `kubectl exec ... printenv` before acting.

---

## 6. Next step: xprof is required

The host-side receipts cannot separate device forward from readback, because
asynchronous dispatch collapses them into one blocking call. Attribution of the
163.5 ms requires a device profile.

Available material:

- `patches/tpu_inference/24-tpu-runner-p56-xprof-labels.patch` already labels this region
- `jax.profiler.TraceAnnotation(f"execute_model: {reqs} reqs, {toks} toks")` is already in the code path
- `_agents/skills/tpu-profiler/` and `_agents/skills/xprof-analyzer/`

A short capture against the running r11 would be side-channel and requires no
restart or configuration change.

---

## 7. Operational notes

| item | note |
|---|---|
| head container name | `jax-tpu`. `pathways-proxy` and `pathways-rm` are `initContainers` with `restartPolicy: Always` (native sidecars), so `kubectl get pods` shows `0/3` and any resource comparison must include `initContainers`. |
| tile receipts | `[PATHTRACE]` receipts print once per shape at compile time, very early. `--tail=6000` misses them and yields a false negative. |
| log rotation | by 04:38Z, `--tail=400000` returned only 9454 lines for a run started 02:18Z. Early-run evidence must be captured while it is still retrievable; WandB `scan_history` is the durable source for per-step timing. |
| `[PATHTRACE]` volume | after entering the trainer phase, `[PATHTRACE]` is ~89% of log lines (21375 of 36569). Monitoring windows need >= 40000 lines. |
| `ReachedMaxRestarts` | a Pathways JobSet that finishes normally still shows `Failed`. Judge completion by head log `Reached max_steps`, WandB `_step == N` / state `finished`, and `replicatedJobsStatus` head `failed=0`. r11 at 300 will also show `Failed`. |
| renderer defaults | `render_p57_frozenlake_tim.py` at HEAD is not deployable as-is on bodaborg-v5p-nap. Five post-render patches were required: `priorityClassName very-high -> medium` (class does not exist), `canon-cpu-pool -> cpu-np` (pool does not exist), `multislice-queue -> default`, add `CANON_P57_TOKEN_CONTINUITY` to match r10, and `pathways-proxy` 32cpu/200Gi + `pathways-rm` 8cpu/32Gi -> 8/16Gi and 4/16Gi (296Gi total exceeded the 239.9Gi node allocatable and left the head pod unschedulable). No script in `kubectl_scripts/` performs these patches. |
| single-variable proof | a full structural diff of the rendered r11 manifest against r10's live object left exactly one meaningful difference, `pathways-worker.nodeSelector`, which Kueue injects from `resourceflavor/tpu-v5p-flavor`. |

---

## 8. Files in this directory

| file | contents |
|---|---|
| `step_comparison.md` | generated step-index-aligned comparison table |
| `raw/r11_step_times.json` | r11 `perf/train/global_step_time` + solve, from WandB |
| `raw/r10_baseline_step_times.json` | r10 same, 149 steps |
| `raw/engine_step_lines.log` | 687 `[ENGINE_STEP]` receipts (663 std, 24 cd) |
| `raw/engine_step_summary.log` | 9 `[ENGINE_STEP_SUMMARY]` windows |
| `raw/engine_driver.log` | 88 `[ENGINE_DRIVER]` windows |
| `raw/pathtrace_tile_receipts.log` | 33 unique tile receipts (16 matmul, 17 rmsnorm) |
| `raw/stage_and_step_timings.log` | `[PERF]`, `[step N]`, audit, p32 group, weight-sync, rollout-metric lines |
| `raw/step10_audit_evidence.log` | the compile-free step-10 audit stages used for the attribution in section 3 |
| `raw/r11_step_lines.log` | `[step N]` lines still retrievable after rotation |
