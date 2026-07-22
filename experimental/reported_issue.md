# DeepSWE Experiment Launch Issues & Observations (2026-07-22)

## 1. Missing Checkpoint & Resumption Logic (train_deepswe_nb.py)
**Severity**: High
**Symptom**: The main training wrapper silently fails to save or resume checkpoints, running completely statelessly from scratch every time it starts.
**Root Cause**: 
Although the script successfully reads the external `--ckpt_dir` and initializes `ocp.CheckpointManagerOptions()` options on line 672, they are explicitly ignored when instantiating the training engine. 
At line 763 onwards, `RLTrainingConfig` has its arguments hardcoded to `None`:
```python
    training_config=rl_cluster_lib.RLTrainingConfig(
        # ...
        checkpoint_root_directory=None,       # Should be CKPT_DIR
        checkpointing_options=None,           # Should be checkpointing_options
    )
```
**Consequence**: If the cluster goes down due to OOM or is preempted, all TPU states/metrics are lost, and progress resets.

---

## 2. Token Generation Throughput vs Sandbox Timeout Imbalance
**Severity**: Medium-High
**Symptom**: Trajectories will inevitably hit a hard killed timeout before reaching `--max_turns=50`.
**Setup & Hyperparameters**:
- Model: `Qwen3-32B`
- Topology: `64 TPUs (FSDP=8, TP=8)`
- `MAX_CONCURRENCY=128` (Parallel sandboxes)
- Python Episode Timeout: `10800` (3 hours)
- R2E-Gym Sandbox Timeout: `activeDeadlineSeconds: 3600` (1 hour) (injected via sed patch)

**Bottleneck Analysis**:
- Based on `loggers.py:271` metrics at `T+30m`, the system handles the 128 sequences by bottlenecking at the generation token boundary.
- **Total Generation Throughput**: `~214.3 tokens/s` (across `126 reqs` currently generating in the KV Cache).
- **Speed per Agent**: `214.3 / 126 = ~1.7 tokens/sec` per agent episode.
- **The Theoretical Ceiling**: Given the `3600` second physical Sandbox timeout, $3600 \times 1.7 = \sim 6,120$ tokens. 
- **The Conflict**: It is physically impossible for the agent to finish an extended `<50 turns>` loop because the physical lifespan of the container will expire as soon as the total tokens exceed ~6000 (which is vastly under the `--max_response_length=32768` budget). The process is heavily token-generation bound, meaning the CPU Sandbox spends most of its 1 hour simply waiting for the TPU rollout engine to trickle tokens to it.

**Suggested Fixes for Next Run**:
- Increase `activeDeadlineSeconds` patch in `relaunch_deepswe_256_mlperf.sh` to `10800` (3 hours) to align with Python `episode_timeout_secs`.
- Decrease `MAX_CONCURRENCY` to `64` to increase average tokens/sec/instance (speed up each agent's individual trajectory response times).
- Fix the `RLTrainingConfig` instantiation.

---

# Solution Analysis & Fix Plan (2026-07-22, engineer review)

Both issues re-verified against the code at this branch (8f6bf366). Issue 1 is
a confirmed 2-line bug plus a config trap. Issue 2's arithmetic is right, but
the throughput number itself is ~10-40x below the hardware roofline, which
points at a deeper root cause than concurrency; a zero-code diagnostic below
decides between the two candidate causes. Exact fixes + verification gates
follow so the executing agent can act directly.

## Issue 1 fix — checkpoint wiring (CONFIRMED bug, 2 lines + 1 launch flag)

**Verified**: `examples/deepswe/train_deepswe_nb.py` reads `--ckpt_dir` into
`CKPT_DIR` (:462, default `/tmp/cp/deepswe_ckpt/01` from :106) and builds
`checkpointing_options = ocp.CheckpointManagerOptions(...)` (:672), but the
`RLTrainingConfig` hardcodes both to None (:772-773).

**Fix 1a — code** (train_deepswe_nb.py:772-773):

```python
# BEFORE
        checkpoint_root_directory=None,
        checkpointing_options=None,
# AFTER
        checkpoint_root_directory=CKPT_DIR,
        checkpointing_options=checkpointing_options,
```

**Fix 1b — launch config** (`experimental/deepswe-256-mlperf.yaml`, the
train_deepswe_nb.py command line, :185): the default `--ckpt_dir=/tmp/cp/...`
is POD-LOCAL disk — a preemption kills the checkpoint together with the pod,
making resumption pointless. Add an explicit GCS path:

```
--ckpt_dir=gs://yuxzhang-tunix-models/deepswe_ckpt/${RUN_TAG or date}
```

**Gates**:
- After `SAVE_INTERVAL_STEPS`, `gsutil ls <ckpt_dir>` shows orbax step dirs.
- Kill + relaunch the jobset: training resumes from the saved step (check
  `iter_steps`/`global_steps` in logs continue rather than reset to 0).
- NOTE for the agent: verify the tunix trainer auto-restores from
  `checkpoint_root_directory` when it is non-empty (peft_trainer /
  CheckpointManager restore path). If restore is NOT automatic, wire it and
  report — saving alone does not give resumption.

## Issue 2 analysis — why 214 tok/s is 10-40x too low (mesh roofline math)

**Config confirmed** (train_deepswe_nb.py:724-725): rollout = 64 chips as
**8 data-parallel vLLM engines x tp=8** (`data_parallel_size = fsdp = 8`),
`max_num_seqs=128` and `max_num_batched_tokens=8192` are PER-ENGINE.
`enable_prefix_caching=True` is passed (:731). The "214 tokens/s, 126 reqs"
line is vLLM's standard logger: **generation throughput counts DECODE tokens
only**; prefill is reported separately as "Avg prompt throughput".

**Roofline** (v5p: ~2.77 TB/s HBM/chip, ~459 bf16 TFLOP/s/chip):
- Qwen3-32B bf16 weights = 64 GB; tp=8 -> 8 GB/chip. A decode iteration is
  memory-bound: weights read ≈ 8/2770 ≈ 3 ms; KV read (GQA ≈ 256 KB/token/seq,
  ~6k ctx ≈ 1.5 GB/seq x ~16 seqs/engine) ≈ +1 ms; with overhead call it
  10-20 ms/iteration.
- Per engine: ~16 running seqs x 1 token / 15 ms ≈ 1000 tok/s -> **~8000
  tok/s aggregate expected**. Even with a pessimistic 50 ms/iteration of
  pathways-proxy overhead: ~2500 tok/s.
- **Observed 214 tok/s. Back-solve: 214 / 8 engines / 16 seqs ≈ 600 ms per
  iteration.** Decode physically cannot be that slow — each iteration is
  being consumed by something else.

**Hypothesis A (primary): prefill starvation — APC not actually effective.**
Multi-turn agentic resubmits the full growing context every turn. If
tpu_inference silently ignores/doesn't support `enable_prefix_caching`, every
turn re-prefills ~6k tokens. One 8192-token chunked-prefill iteration for 32B
tp=8 ≈ 2*32e9*8192 / (8*459e12) ≈ 143 ms compute + overhead ≈ 200-300 ms —
the same order as the back-solved 600 ms. Engine iterations become almost all
prefill; decode (the reported "generation throughput") starves at ~214.

**Hypothesis B (secondary): fixed per-iteration overhead** (pathways-proxy
RPC / host sampling round-trip ~hundreds of ms). Comparable gsm8k runs on the
same proxy architecture can serve as a reference iteration rate.

**Decisive diagnostic (ZERO code change, do this first)**: read the SAME
vLLM log lines already collected:
- "Avg prompt throughput" is thousands of tok/s  ->  **A confirmed** (engine
  is busy prefilling).
- "Avg prompt throughput" ≈ 0                    ->  **B** (iterations are
  slow without doing prefill work).
Also grep engine startup logs for whether prefix caching was actually
enabled/supported by the tpu_inference backend.

**GKE side note**: sandboxes are k8s pods pinned to the SAME
`deepswe-cpu-pool` as the pathways head (`R2E_K8S_NODE_SELECTOR`, yaml :149).
At the reported snapshot Running≈126 means agents were in-engine (not waiting
on env), so env execution is NOT the current bottleneck — but monitor pool
CPU saturation (`kubectl top nodes`) since env steps add per-turn latency
outside the engine.

## Issue 2 fixes (ordered)

| # | Action | When |
|---|---|---|
| 2a | `activeDeadlineSeconds` 3600 -> 10800 in the sed at yaml :175 (align with `episode_timeout_secs=10800`): `sed -i 's/"restartPolicy": "Never",/"restartPolicy": "Never", "activeDeadlineSeconds": 10800,/g' ...` | unconditional, now |
| 2b | Run the decisive diagnostic above (prompt-throughput line) | now, zero-code |
| 2c | If A: confirm/enable APC in tpu_inference; raise `--max_num_batched_tokens` 8192 -> 16384 (or 32768; HBM util 0.8 has headroom) so prefill drains faster and decode starves less | after 2b |
| 2d | If B: xprof the rollout engine step; find the fixed ~600 ms; compare against a gsm8k run's iteration rate on the same proxy architecture | after 2b |
| 2e | `MAX_CONCURRENCY` 128 -> 64: halves prefill pressure / per-iteration batch either way, and (with 2a) lets single trajectories finish within the sandbox deadline. Note: aggregate throughput stays roughly the same if compute-bound — this buys per-trajectory completion, not faster training | optional, after 2c/2d |

**Gates**:
- 2a: no sandbox killed at 3600 s; trajectories exceed ~6k tokens / reach more turns.
- 2c/2d: "Avg generation throughput" rises to >=1000 tok/s aggregate (target:
  >=10x current per-agent rate); per-agent ≈ generation/Running >= 15 tok/s.
- Overall: a full batch of trajectories completes without deadline kills, and
  step time drops proportionally.

## Diagnostic Result for Issue 2 (2026-07-22, runtime verification)

The required zero-code diagnostic (Action 2b) was captured successfully from the live vLLM training log at `T+30m`:
```log
INFO 07-22 02:21:06 [loggers.py:271] Engine 000: Avg prompt throughput: 31994.5 tokens/s, Avg generation throughput: 214.3 tokens/s, Running: 126 reqs, Waiting: 0 reqs, GPU KV cache usage: 7.6%, Prefix cache hit rate: 0.0%
```

**Conclusion:**
- **Hypothesis A is conclusively confirmed.** 
- The prompt throughput is astronomically high (~32k tok/s) while generation is starved, and crucially, the Prefix Cache hit rate is **0.0%**.
- APC (Automatic Prefix Caching) is completely failing to hit or fundamentally unsupported by this `tpu_inference` build during this multi-turn trajectory run.
- The engine spends almost all its iteration cycles repeatedly churning the identical history context through new prefill passes, suffocating decode rate.

**Over to the reviewing agent:** Given the confirmed 0.0% APC hit rate, how do we correctly wire/enable APC in this JAX setup, or should we execute fallback Option 2c (batch tuning) + 2a (relaxing sandbox limits) directly?

---

## 3. Current Run State & Workarounds (Reverted Baseline 2026-07-22)

To forcefully keep the deepswe pipeline running and collect actionable trajectory data despite the 0.0% APC starvation issue and initialization crashes, we have deployed the following "Reverted Baseline" in our latest cluster launch:

**1. Scaled Down Concurrency (Relieving APC Starvation)**
To ensure each episode receives enough decoding throughput to finish before the physical sandbox dies, we halved the concurrent agents:
- `MAX_CONCURRENT=64`
- `--max_concurrency=64`
- `--rollout_vllm_max_num_seqs=64`

**2. Relaxed Sandbox Timeout Limits (Fix 2a)**
We extended the hard death-limit of the Kubernetes Pods from 1 hour to 3 hours, officially aligning it with the Python default:
- `activeDeadlineSeconds` patched to `10800` via sed.
- `TIMEOUT=10800` exported.

**3. Fixed Cold Pool Initialization Race Condition (Node Cache Issue)**
Since the CPU head pod now runs on an autoscaled cold node (`deepswe-cpu-pool`), fetching its 20GB docker image takes exactly 5-6 minutes. Since the TPU workers sit on warm slices and boot instantly, they were suffering DNS timeouts and aggressively self-immolating, causing K8s to garbage-collect the whole JobSet.
- **Fix:** Increased `backoffLimit` under `pathways-worker` from `0` to `100`. The workers now passively wait up to 100 restarts for the Head to compile without deleting the fleet.

**4. Isolated Checkpoint & Log Routing**
To prevent new retries from corrupting our previous crash logs, the hardcoded `run1` directories were replaced with dynamic Bash expansion:
- `ckpt_dir` is now `gs://.../.../checkpoint/${RUN_TAG}/`
- Current `RUN_TAG` is explicitly tracked as `"qwen3_64_retry_3"`.
- We temporarily bypassed fixing the "None" parameter in internal code (Issue 1) to prioritize getting this diagnostic run off the ground. 

*(This baseline configuration successfully mitigates the immediate crash symptoms and affords us the maximum runway to investigate the underlying JAX 0% Prefix Cache logic when we return to this file).*

---

## 4. DeepSWE Training Data Source Alignment
**Severity**: Medium
**Symptom**: The current training or evaluation pipeline might be blindly iterating over standard or noisy R2E-Gym datasets rather than the vetted gold subset.
**Observation**: The actual verified, high-quality trajectory data that we uniquely want to use for DeepSWE is situated inside `task_report_good_qwen3_128_retry_20260713_090141.jsonl`.
**Action Required (For Delegated Team Member)**:
- Modify the data ingestion script (e.g., `train_deepswe.py` or the whitelist parser).
- Guarantee that the pipeline **strictly filters its training set** using ONLY the instances defined within `task_report_good_qwen3_128_retry_20260713_090141.jsonl`.
- *Status*: Acknowledged and logged for delegation (2026-07-22). No code changes have been implemented yet per instructions.


## 5. Performance Breakthrough: APC Validated Under 256-Chip Scale (2026-07-22)
**Status**: Exceptional / Active Monitoring
**Context**: After fixing the `FileNotFoundError` by routing `--gold_whitelist` to the downloaded `task_report_...jsonl`, JAX successfully compiled the 32B model graph across the 64 pods.
**Hardware Topology Config**:
- **Total Capacity**: **256 TPU v5p chips** (64 host nodes x 4 chips) + 1 Master CPU Head.
- **Disaggregated Rendering**: `--rollout_split_fraction=0.5`.
- **vLLM Inference (Rollout)**: 128 chips (`tp=8, fsdp=8`, meaning 2 Mesh Replicas of 64 chips each).
- **JAX Training**: 128 chips (`tp=8, fsdp=8`, meaning 2 Mesh Replicas of 64 chips each).
- **Workload**: `MAX_CONCURRENCY=128`.

**Live Throughput Stats (at T+45m)**:
- **Avg Prompt Throughput (Prefill)**: Sustained **~4000 to 7000+ tokens/sec**.
- **Avg Generation Throughput (Decode)**: Sustained **~450 to 650 tokens/sec**.
- **Prefix Cache Hit Rate (APC)**: Warmed up perfectly, scaling from 0% -> 53% -> 89% -> stabilizing at **~93.0%**.
- **GPU KV Cache Usage**: Sitting at a shockingly low **~3.5%** despite handling `~120 running` requests simultaneously.

**Timeout Risk Assessment & Resolution**:
- The agent decode step is effectively pacing at **~5.4 tokens/second per sandbox**.
- A standard 50-turn trajectory requires thousands of tokens.
- We confirmed the Kubernetes `activeDeadlineSeconds` sandbox isolation timeout was cleanly sed-patched to `10800` (3 hours) inside the `deepswe-256-mlperf.yaml` launch logic.
- **Verdict**: Trajectories will easily finish before reaching the ceiling. Timeout risk is mitigated.

**Future Optimization (Scaling Headroom)**: 
The user observation is correct: things can be pushed much harder. Because KV Cache usage is currently `< 5%` and APC hit rate is continuously `> 90%` while pushing 650 decodes/s on 128 concurrency, the 128-chip Rollout mesh has massive untouched capacity. We can confidently double or quadruple `max_concurrency` (e.g. `256` or `512`) in future runs to dramatically speed up total episodic rollouts.
**Raw Engine Trace Metrics Extract**:
```log
INFO 07-22 17:50:31 [loggers.py:271] Engine 000: Avg prompt throughput: 360.2 tokens/s, Avg generation throughput: 0.2 tokens/s, Running: 2 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
INFO 07-22 17:50:56 [loggers.py:271] Engine 000: Avg prompt throughput: 99.0 tokens/s, Avg generation throughput: 0.9 tokens/s, Running: 5 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 53.6%
INFO 07-22 17:51:07 [loggers.py:271] Engine 000: Avg prompt throughput: 670.0 tokens/s, Avg generation throughput: 9.6 tokens/s, Running: 35 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.1%, Prefix cache hit rate: 86.6%
INFO 07-22 17:51:51 [loggers.py:271] Engine 000: Avg prompt throughput: 312.5 tokens/s, Avg generation throughput: 116.9 tokens/s, Running: 64 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.1%, Prefix cache hit rate: 89.6%
INFO 07-22 17:52:37 [loggers.py:271] Engine 000: Avg prompt throughput: 2219.1 tokens/s, Avg generation throughput: 136.7 tokens/s, Running: 113 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.2%, Prefix cache hit rate: 91.7%
INFO 07-22 17:53:31 [loggers.py:271] Engine 000: Avg prompt throughput: 3732.0 tokens/s, Avg generation throughput: 649.1 tokens/s, Running: 118 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.4%, Prefix cache hit rate: 89.7%
INFO 07-22 17:53:41 [loggers.py:271] Engine 000: Avg prompt throughput: 7004.2 tokens/s, Avg generation throughput: 390.6 tokens/s, Running: 123 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.8%, Prefix cache hit rate: 87.7%
INFO 07-22 17:55:09 [loggers.py:271] Engine 000: Avg prompt throughput: 4353.9 tokens/s, Avg generation throughput: 593.9 tokens/s, Running: 117 reqs, Waiting: 0 reqs, GPU KV cache usage: 3.0%, Prefix cache hit rate: 90.7%
INFO 07-22 17:55:50 [loggers.py:271] Engine 000: Avg prompt throughput: 4009.9 tokens/s, Avg generation throughput: 571.1 tokens/s, Running: 116 reqs, Waiting: 0 reqs, GPU KV cache usage: 3.7%, Prefix cache hit rate: 93.0%
```

**Retracted Recommendation**: The extreme headroom available on the hardware was noted. However, scaling `max_concurrency` higher than 128 (e.g. to 256 or 512) **is not recommended for DeepSWE** because generation velocity per trajectory would plummet below the ~5.4 tokens/second baseline, causing trajectories to exceed the maximum 3-hour Kubernetes lifespan. Completing convergence metrics absolutely demands environments finish properly without truncation timeouts. The 128 boundary strikes the mathematically perfect alignment between APC hits and max duration safety.

## 6. Throughput Drop & CPU Sandbox Bottleneck (At T+2h)
**Status**: Expected RL Rollout Behavior (Long Tail Bottleneck)
**Observation**: Around 2 hours into the run, the overall vLLM generation throughput appeared to plummet from `~600 tokens/s` down to `~0.1 tokens/s`.
**Cause Analysis**:
This is **not** a TPU inference failure, nor has the system transitioned into the PPO training update phase yet.
As 85% of the trajectories (191 / 224) have successfully completed, only the most difficult ~30 tasks remain. These surviving agents often write faulty validation scripts (e.g., infinite loops in `pytest` or `execute_bash`) that trigger the R2E sandboxes' native 5-minute (`300` seconds) hard timeout per step.
Because the CPU sandboxes are taking exactly `300.05s` to return execution results to the LLM, the TPU engines spend 5 minutes severely starved of requests (`Running: 0 reqs, Waiting: 0 reqs`), instantly process the next prompt in `<1s`, and then wait another 5 minutes. The mathematical aggregate throughput drops because the overall GPU timeline is almost completely empty.

**Verification Log**:
```log
INFO:root:[SWEEnv group=14 pair=4] env.step done in 300.05s done=False reward=0 obs_chars=130
INFO:root:[DeepSWE Debug] model_call start group=14 pair=4 prompt_len=44 max_generation_steps=28199
INFO 07-22 18:41:28 [loggers.py:271] Engine 000: Avg prompt throughput: 11.9 tokens/s, Avg generation throughput: 0.1 tokens/s, Running: 1 reqs, Waiting: 0 reqs
INFO:root:[DeepSWE Debug] model_call done group=14 pair=4 in 0.64s outputs=1
INFO:absl:[step_idx=21, pair_index=4, group_id=14] prompt token alignment OK len=6360 local_context_len=8665
INFO:root:[SWEEnv group=14 pair=4] env.step start step=21 function=execute_bash
INFO 07-22 18:41:30 [loggers.py:271] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 9.7 tokens/s, Running: 0 reqs, Waiting: 0 reqs
```
**Conclusion**: This definitively confirms the system is correctly running but stalled waiting for the CPU sandboxes to time out broken agent execution loops. The rollout phase will naturally conclude once these last environments hit their 50-turn cap or 3-hour limit, transitioning safely into the PPO update.
