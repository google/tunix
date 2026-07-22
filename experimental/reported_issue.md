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
