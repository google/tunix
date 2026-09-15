# Qwen3.5-35B-A3B distributed GRPO on TPU v5p — reproduction report (v7)

Supersedes `qwen35_report_v6.md`, which stays as the forensic log: v6 records
how the two blocking bugs were found, including the wrong turns. **This file is
the one to follow if you just want to run it.**

Base image `gcr.io/cloud-tpu-multipod-dev/yixuannwang_google_com-runner:yixuann-e2e-0912head-v8`
(Yixuan's e2e image) plus an **eight-file** overlay. Everything measured on
cluster `bodaborg-v5p-nap`, project `cloud-tpu-shared-capacity`, region
`europe-west4`, namespace `trellis`.

> **The branch has since been rebased onto `tunix` `main`.** Two of the files
> that were overlaid at measurement time are no longer needed (upstream fixed
> rollout routing itself) and two defaults changed underneath the repro script.
> Nothing in the measured numbers is invalidated, but the image has to be
> rebuilt before the next run. See §1.3 for the exact deltas.

---

## 0. Status

Two independent bugs were pinning `reward_mean` at exactly `0.0000`. Both are
fixed in the overlay:

| # | Where | Effect | Fix |
|---|---|---|---|
| 1 | `k8s_launcher.sh` set `--prefuse_moe_weights=true` on **both** ends | trainer's MoE `wi` reached the rollout permuted; the model generated digit soup | trainer must be `false`, rollout `true` — they mean different layouts (§2) |
| 2 | `rl_program.py:_extract_reward` read `traj["reward"]` | the rollout emits `"trajectory_reward"`; every trajectory scored 0.0, zeroing the **GRPO advantages**, not just the metric | read the key the rollout actually emits (§3) |

Bug 1 alone made the model useless. Bug 2 alone made training a no-op even with
a perfect model. They masked each other: fixing 1 gave
`24/24 reward=1.00 format_ok=True answer_ok=True` on the rollout pods while the
orchestrator still printed `reward_mean: 0.0000`.

With both fixed, the run trains:

```
Train step 0 - loss: 0.0000 - reward_mean: 0.9375 - perplexity: 1.0000 - step_time: 697.59s
Train step 1 - loss: 0.0345 - reward_mean: 0.9062 - perplexity: 1.0351 - step_time: 402.26s
Train step 2 - loss: 0.0109 - reward_mean: 0.9406 - perplexity: 1.0110 - step_time: 447.13s
```

A third problem, non-blocking but silent: **W&B was never actually logging.** See
§9 — it needed `WANDB_ENTITY`, and the failure surfaced as a single INFO line.

**The 100-step run completed.** 100/100 steps in **11.75 h**, no preemption, no
restarts, no recompiles after step 0.

| | |
|---|---|
| Mean step time | **423.0 s** |
| Mean `reward_mean` | 0.9147 |
| `reward_mean`, first 10 / last 10 steps | 0.9419 / 0.9403 |
| Steps with loss exactly 0.0000 | 40 / 100 |
| W&B | `wandb.ai/google-trellis/qwen35-35b-a3b-grpo/runs/9sau7d0x` |

**Performance verdict: the 100-steps-in-2-hours target is not reachable on this
topology** — it came in at 11.75 h against a 2 h goal. ~56% of every step is
weight sync. See §5 for the breakdown and what would have to change.

**The run did not visibly learn**, and §3 explains why: GSM8K is at this model's
ceiling, so 40 of 100 steps produced literally zero gradient. The pipeline is
correct; the task is too easy to demonstrate it.

---

## 1. Reproducing it

```bash
export KUBECONFIG=~/.kube/config.cloud-tpu-shared-capacity.europe-west4.bodaborg-v5p-nap

cd ~/git/tunix
gcloud auth configure-docker gcr.io -q      # once
./build_qwen35_overlay.sh                   # builds + pushes qwen35-repro-v12

cd tunix/experimental/examples/math_gsm8k_dist
MAX_STEPS=2 RUN_TAG=smoke ./run_qwen35_repro.sh start
./run_qwen35_repro.sh stop                  # tears down all 10 jobsets
```

Two environment traps, both already handled by `run_qwen35_repro.sh` but fatal
if you bypass it:

1. **`$USER` is `igorts_google_com` on these VMs and underscores are illegal in
   k8s object names.** The launcher sanitizes with `cut -d_ -f1`. Bypass it and
   every `kubectl apply` is rejected.
2. **`KUBECONFIG` is not the default path.** Without the export above `kubectl`
   dials `localhost:8080`.

`DRY_RUN=true ./run_qwen35_repro.sh start` prints every manifest and applies
nothing. Use it to check flags — in particular that the trainer gets
`--prefuse_moe_weights=false` and the eight rollouts get `true`:

```bash
DRY_RUN=true ./run_qwen35_repro.sh start \
  | grep -oE 'run_(trainer|rollout)_node|prefuse_moe_weights=[a-z]+' | sort | uniq -c
#   1 prefuse_moe_weights=false
#   8 prefuse_moe_weights=true
#   8 run_rollout_node
#   1 run_trainer_node
```

### 1.1 Hyperparameters — the complete set

Every value below is a default in `run_qwen35_repro.sh`; the 100-step run used
them unmodified. Anything marked ⚠ is exported but **not actually plumbed
through** — see the note at the end of this section.

**Model and data**

| | |
|---|---|
| Model | `Qwen/Qwen3.5-35B-A3B` (MaxText `qwen3.5-35b-a3b`) |
| Architecture | 40 layers (10 full-attn + 30 Gated-DeltaNet), cycle 4, 10 scanned blocks, 256 experts, `base_emb_dim=2048`, `padded_moe_mlp_dim=512`, vocab 248320 |
| Base checkpoint | `gs://hengtaoguo-maxtext-logs/checkpoints/qwen3.5-35b-a3b/scanned/2026-06-11-10-27/0/items` |
| Dataset | GSM8K train (`tfds`), reward mode `env` |
| Precision | `bfloat16` weights and compute |
| Fine-tune type | **full-parameter** (`USE_LORA` is not plumbed through in this image — do not set it and expect LoRA) |

**Batch shape**

| Knob | Value | Note |
|---|---|---|
| `MAX_STEPS` | 100 | |
| `BATCH_SIZE` | 16 | prompts per step |
| `NUM_GENERATIONS` | 16 | GRPO group size |
| → rollouts per step | **256** | `BATCH_SIZE × NUM_GENERATIONS` |
| `MINI_BATCH_SIZE` | 256 | = all of them ⇒ **one optimizer update per step** |
| `TRAIN_MICRO_BATCH_SIZE` | 8 | must equal `TRAINER_MESH_FSDP` so MaxText sees `per_device_batch_size=1` |
| `MAX_PROMPT_LENGTH` | 512 | |
| `MAX_RESPONSE_LENGTH` | 1024 | |
| → max sequence | 1536 tokens | |
| `MAX_SEQ_TOKEN_PER_TPU` | 4096 | packing budget per row; ~7.6 trajectories/row, 5 microbatches/step |
| `USE_ROLLOUT_LOGPS` | true | `old_logprobs` come from vLLM, not recomputed by the trainer |
| `EOS_TOKENS` | unset for the measured run; now `<\|im_end\|>` | see §1.3 — changed after the rebase |

**GRPO / optimizer**

| Knob | Value | Reaches the job? |
|---|---|---|
| `LEARNING_RATE` | `2.0e-7` | yes — `--learning_rate` on the trainer |
| `ADAM_B1` | 0.9 | yes |
| `ADAM_B2` | 0.99 | yes |
| `WEIGHT_DECAY` | 0.01 | yes |
| `MAX_GRAD_NORM` | 1.0 | yes |
| `ADAM_EPS` | `1.0e-8` | yes |
| `OPT_CHAIN_TYPE` | `clip_by_global_norm` | yes |
| `SCHEDULE_TYPE` | `""` → constant `2.0e-7` | yes; see §1.3 — upstream's default is now a cosine schedule and had to be pinned off |
| `BETA` (KL coeff) | 0 | ⚠ **never passed** — argparse default 0.0 happens to match |
| `EPSILON` (clip) | 0.2 | ⚠ **never passed** — argparse default 0.2 happens to match |

With `BETA=0` there is no reference model and no KL term, so the loss is pure
clipped policy gradient.

> ⚠ **`BETA` and `EPSILON` are exported by both `run_qwen35_repro.sh` and
> `k8s_launcher.sh` but are never interpolated into any command line.** This run
> was unaffected only by luck: the argparse defaults in
> `run_gsm8k_dist_grpo.py` (0.0 and 0.2) are identical to the exported values.
> Setting `BETA=0.04` to match the Qwen3 GSM8K recipe would silently do nothing.
> Confirm what was actually used from the orchestrator log, which prints it:
> ```
> beta=0.0000, epsilon=0.20, reward_mode=env
> ```
> See §8.

### 1.2 What step time actually depends on

Worth stating plainly, because the intuition is misleading: **the dominant term
in step time is independent of batch size and sequence length.**

| Component | ~Share | Scales with |
|---|---|---|
| Weight sync | 235 s (56%) | model size and **topology** only — not batch, not seqlen |
| Rollout generation | ~150 s | batch × generations × response length |
| Gradient computation | a few s | batch × seqlen (and it is negligible — see §5) |

So halving `BATCH_SIZE` or `MAX_RESPONSE_LENGTH` would cut at most the ~150 s
rollout term and leave the 235 s sync untouched — roughly 422 s/step → 350 s/step
at best, while halving the data per update. This is why §5.3 concludes the
2-hour target needs a change to *weight sync*, not to the batch shape.

### 1.3 What changed when the branch was rebased onto `tunix` `main`

The 100-step run was measured against a branch based on Yixuan's lineage. The
branch has since been rebased onto `tunix` `main` (`f5f595a8`) and the changes
re-split into one commit per fix. **Read this section before rerunning.** None
of it invalidates a measured number, but the image is stale.

**Rebuild the image and smoke-test before the next 100-step run.** The overlaid
files now come from upstream's lineage rather than the image's, so they are
newer relative to everything else in `/app/tunix`. `MAX_STEPS=2 RUN_TAG=smoke
./run_qwen35_repro.sh start` is enough to catch a signature mismatch; it costs
~25 minutes.

Dropped as obsolete — upstream does each of these now:

| What we carried | Upstream |
|---|---|
| round-robin rollout routing (`distributed_rl_engine.py`, `run_gsm8k_dist_grpo.py`) | `7b8f7779` routes on `req.traj_id` |
| `TRAIN_MICRO_BATCH_SIZE` plumbing | `f5f595a8` |
| orchestrator-side `checkpoint_save_interval_steps=0` handling | `run_trainer_node.py:573` returns a read-only `CheckpointManagerOptions` at 0 |
| the `DISABLE_CHECKPOINTING` monkeypatch in `maxtext_utils` | never needed; the image already supported restore-without-save |

Two *defaults* differ from what the run measured, and `run_qwen35_repro.sh` now
pins both explicitly:

| Knob | Measured run | Now | Why |
|---|---|---|---|
| `EOS_TOKENS` | unset (tokenizer EOS) | `<|im_end|>` | upstream's new default, and the right one: Qwen3 chat closes a turn with `<\|im_end\|>`, so without it every generation runs to `MAX_RESPONSE_LENGTH` and trips the overlong filter described in §3. Expect *fewer* dropped trajectories than the measured run had. |
| `SCHEDULE_TYPE` | constant 2e-7 | pinned to `""` (constant 2e-7) | upstream now defaults to `warmup_cosine_decay_schedule` over `LR_DECAY_STEPS=500`. A 100-step run would never leave the warmup ramp, so the effective LR would be far below 2e-7 and the steps here would not be comparable. Use the cosine schedule for real training runs, not for this repro. |

### Checking a live run

The `[gsm8k] completion ...` lines with per-trajectory rewards are emitted on
the **rollout** pods, not the orchestrator. That cost a lot of confusion:

```bash
# per-trajectory grading (rollout pods)
kubectl logs -n trellis -l jobset.sigs.k8s.io/jobset-name=igorts-roll-0 --tail=-1 \
  | grep -oE 'reward=[0-9.]+ format_ok=\w+ answer_ok=\w+' | sort | uniq -c

# aggregated step metrics (orchestrator)
kubectl logs -n trellis -l jobset.sigs.k8s.io/jobset-name=igorts-orch --tail=-1 \
  | grep -E 'Train step|Weight sync finished'
```

Pods are reaped once a JobSet finishes, so after the fact use Cloud Logging:

```bash
gcloud logging read \
  'resource.type="k8s_container" resource.labels.namespace_name="trellis"
   labels."k8s-pod/jobset_sigs_k8s_io/jobset-name"="igorts-orch"' \
  --project=cloud-tpu-shared-capacity --limit=3000 \
  --format='value(textPayload)' --order=asc
```

Add a `timestamp>"..."` clause: the entry limit otherwise truncates you into the
*previous* run of a reused jobset name.

---

## 2. Bug 1 — `prefuse_moe_weights` must **differ** between trainer and rollout

This is the important finding and it is counter-intuitive enough that the
pre-existing comment in `k8s_launcher.sh` asserted the opposite ("has to match
on both sides"). The same flag name selects two *different* physical layouts
depending on which attention implementation reads the tensor.

| end | attention | who reads `wi` | layout it needs |
|---|---|---|---|
| trainer | `dot_product` (default) | `moe.py:3692` — `n = wi.shape[-1]//2; w0 = wi[..., :n]; w1 = wi[..., n:]` | **global** `[gate \| up]` |
| rollout | `vllm_rpa` (`configs/inference/vllm.yml:15`) | `moe.py:3690` → `fused_moe_func` → tokamax `gmm_v2` (`rhs_up_ref = rhs[..., out_size_n:]`, on the **local shard**) | **per-shard** `[g_s0 \| u_s0 \| g_s1 \| u_s1]` |

The conversion between them is done by the converter — but only if the source
tree still carries the unfused `wi_0`/`wi_1`:

```python
# maxtext/src/maxtext/integration/vllm/convert_utils.py:149
if not src_key or src_key[-1] != "wi_0":
    continue          # already fused -> the per-shard interleave is SKIPPED
```

So with `prefuse_moe_weights=true` on the trainer:

1. `model_creation_utils.py:221 _fuse_moe_weights` builds `wi` at checkpoint
   load. It takes `n_shards` from the trainer's own sharding of `wi`'s last
   axis, which under FSDP (`P(None, None, 'fsdp', None)`) is unsharded, so
   `n_shards=1` and the concat is **global**.
2. The converter sees a fused source and skips the interleave entirely (the
   `"Fusing MoE %s: wi_0=%s, wi_1=%s -> %s on axis %d"` line at
   `convert_utils.py:175` never appears — that absence is the tell).
3. At rollout TP=2 the global `[gate|up]` is split down the middle by the mesh:
   **shard 0 receives all gate, shard 1 all up.**

Step 3 is a pure permutation. Raiden's `checksums()` returns per-tensor float32
**abs-sums**, which are permutation-invariant, so all 633 tensors verified green
while the model was thoroughly broken. That is why this survived so long.

The fix splits the knob in two (`k8s_launcher.sh`, and mirrored in
`run_qwen35_repro.sh`):

```bash
export PREFUSE_MOE_WEIGHTS=${PREFUSE_MOE_WEIGHTS:-true}                  # rollout
export TRAINER_PREFUSE_MOE_WEIGHTS=${TRAINER_PREFUSE_MOE_WEIGHTS:-false} # trainer
```

Confirmation on the 40-chip run: trainer logs `wi_0 shape=(256, 10, 2048, 512)`
(unfused; fused would be `(256, 10, 2048, 1024)`), 633 sync variables, and every
logged completion graded `reward=1.00 format_ok=True answer_ok=True` with
correct arithmetic, well-formed `<reasoning>`/`<answer>` tags and a clean
`<|im_end|>` stop — e.g. `extracted='13' gold='13' chars=824`.

### How to check this yourself in one line

Sanity-check the trainer's shape rather than trusting the flag:

```bash
kubectl logs -n trellis -l jobset.sigs.k8s.io/jobset-name=igorts-train --tail=-1 \
  | grep -oE 'wi_0 shape=\([0-9, ]+\)' | head -1
# want (256, 10, 2048, 512) -- 1024 means the trainer fused and you have bug 1
```

---

## 3. Bug 2 — rewards never reached GRPO

With bug 1 fixed the rollout was grading 1.00 across the board and the
orchestrator still reported:

```
[Orchestrator] Train step 0 - loss: 0.0000 - reward_mean: 0.0000 -
               advantage_mean: 0.0000 - perplexity: 1.0000 - step_time: 704.34s
```

`TrajectoryCollectEngine.collect()` in **Token** mode — the mode the distributed
rollout path uses — returns the episode reward under `"trajectory_reward"`:

```python
# tunix/rl/agentic/trajectory/trajectory_collect_engine.py:337
"trajectory_reward": self.agent.trajectory.reward,
```

`"reward"` is the *per-step* key from **Steps** mode, and that is the only key
`_extract_reward` looked for. Every trajectory therefore returned `0.0`.

This is not merely a broken metric. `_extract_reward` feeds two call sites:

- `rl_program.py:811` → `step_rewards` → the logged `reward_mean`;
- `rl_program.py:352` → `rewards` → `self.algo.create_trainer_payloads(group, rewards=rewards)` → `compute_advantages`.

GRPO's advantage is `(r − mean(r)) / std(r)`. All-zero rewards give all-zero
advantages, and with `BETA=0` the loss is then exactly 0. **The run was a
perfect no-op regardless of how good the completions were.**

The non-distributed path was never affected —
`agentic_grpo_learner.py:595` already reads `item.traj.get("trajectory_reward")`.
Only the orchestrator's copy was wrong.

Fix (overlay file 9, `tunix/experimental/orchestrator/rl_program.py`):

```python
if isinstance(traj, dict):
    if "reward" in traj:
        return float(traj["reward"] or 0.0)
    return float(traj.get("trajectory_reward") or 0.0)
```

The `"reward"` branch is kept and checked first because `rl_program.py:398`
writes the resolved reward back as `traj_dict["reward"] = reward_val` before the
payload moves downstream, so re-extraction from an already-processed item must
keep seeing it.

### Reading the metrics after the fix

`advantage_mean: 0.0000` is **correct and expected**, not a leftover symptom.
GRPO centres advantages within each group — `(r − mean(r)) / std(r)` — so their
mean over a batch is zero by construction. Do not use it as a health signal.

`loss: 0.0000` at **step 0 only** is also expected. At the first inner epoch the
importance ratio is 1, so with `BETA=0` the scalar GRPO loss is exactly zero
while the *gradient* is not. From step 1 on the loss is small but non-zero
(0.0345, 0.0109) because `USE_ROLLOUT_LOGPS=true` takes `old_logprobs` from
vLLM while the trainer recomputes them in MaxText; the two disagree slightly, so
the ratio drifts off 1. Perplexity tracks it (1.0351, 1.0110).

The signal to watch is **`reward_mean`**, and the thing to watch out for on this
task is the *ceiling*: Qwen3.5-35B-A3B already scores ~0.94 on GSM8K, so most
16-generation groups come back all-correct, `std(r) = 0`, and the advantage for
that whole group is zero. A large fraction of each batch contributes no gradient
at all. If the point is to demonstrate learning rather than to reproduce a
pipeline, GSM8K is too easy for this model and a harder dataset would show far
more movement.

The completed 100-step run bears this out. `reward_mean` averaged **0.9147** and
went essentially nowhere — 0.9419 over the first ten steps, 0.9403 over the last
ten. Two things confirm the ceiling rather than a broken gradient path:

- **`reward_mean` lands on exact sixteenths.** With 16 generations per prompt,
  that only happens when groups are internally uniform — the model solves a
  prompt in all 16 attempts or in none.
- **40 of the 100 steps had `loss` exactly 0.0000.** With `BETA=0` there is no KL
  term, so a group with `std(r) = 0` yields an advantage of exactly zero and the
  step is a literal no-op. 41% of the run applied no gradient.

This is a *task* result, not a pipeline defect: the same metrics read 0.0000
before the §3 fix and read real per-trajectory grades after it.

---

## 4. The overlay — eight files, and why each one is there

Build with `./build_qwen35_overlay.sh`; the Dockerfile asserts every patch is
present so a moved upstream path fails the *build*, not the job.

Keep the overlay this small. The base image's `/app/tunix` is a different
lineage from `github.com/google/tunix` HEAD; copying the whole tree in silently
downgrades Yixuan's fixes.

Five files carry a fix:

| # | File | Why |
|---|---|---|
| 1 | `maxtext .../training_engine/maxtext_engine.py` | MaxText PR #5219: clear `payload.metadata` before `_gen_model_input_fn`. `RLTrainerPayload.metadata` is a *static* PyTree field, so its microbatch-varying keys change the treedef and recompile the whole 35B graph **every microbatch**. |
| 2 | `examples/common/run_rollout_node.py` | pass `data_parallel_size` to the `vllm` sampler's `AsyncEngineArgs` |
| 3 | `utils/maxtext_utils.py` | let `max_seq_token_per_tpu` raise `max_target_length` |
| 4 | `examples/common/run_trainer_node.py` | same; see §6 |
| 5 | `experimental/orchestrator/rl_program.py` | **bug 2** (§3) |

Three are diagnostics, all off or nearly off by default — none changes what the
run computes, and all three exist because the questions they answer cannot be
answered from outside the container:

| # | File | Why |
|---|---|---|
| 6 | `rl/agentic/trajectory/trajectory_collect_engine.py` | log raw generations (`TUNIX_LOG_GENERATIONS=N`, default 0) *before* the overlong filter drops them |
| 7 | `examples/math_gsm8k_dist/gsm8k.py` | log the first few *scored* completions per rollout worker at INFO (`GSM8K_LOG_COMPLETIONS`, default 3), with the reward breakdown |
| 8 | `experimental/rollout/collector.py` | log the raw sampler response at its source (`TUNIX_LOG_ROLLOUT_TEXT=1`, which `--debug` sets on the rollout node) |

Note on #1: this file also carries the local Pathways checkpoint-drain commit
`639cbe7be`. That code is unreachable while checkpoint saving is off, which it
is here (`CHECKPOINT_SAVE_INTERVAL_STEPS=0`).

Two files that *were* overlaid for the measured run are gone —
`orchestrator/distributed_rl_engine.py` and
`examples/math_gsm8k_dist/run_gsm8k_dist_grpo.py`, which together stopped
`prefix_hash` from pinning all N generations of a prompt onto one rollout
worker. Upstream `7b8f7779` now routes on `req.traj_id`, which spreads them the
same way. See §1.3.

---

## 5. Performance: where a step actually goes

Measured on a clean 2-step run — 8-chip trainer, 8 × 4-chip rollout slices,
batch 16 × 16 generations = 256 rollouts/step, packing on.

| Phase | Step 0 | Step 1 (steady state) |
|---|---|---|
| Weight sync | 253.59 s + 236.83 s | **235.35 s** |
| Rollout + train | ~460 s (incl. first-step compile) | **~166 s** |
| **Total `step_time`** | **704.34 s** | **401.30 s** |

Step 0 is compile-dominated; step 1 is the number to plan with. **Weight sync is
59% of a steady-state step.**

Over the full 100-step run the mean step came out slightly higher, at **423.0 s**
(11.75 h total), because both the step and its sync drift upward as the run
progresses — see §5.1.2.

| | steps 1–10 | steps 90–99 |
|---|---|---|
| mean `step_time` | 406.4 s | **441.8 s** (+8.7%) |
| mean weight sync | 240.1 s | **291.0 s** (+21%) |

### 5.0 There is no recompilation after step 0 — verify it this way

Overlay file 1 (MaxText PR #5219) exists to stop a per-microbatch recompile of
the 35B graph. Confirm it held rather than assuming it: every trainer graph
compile falls inside step 0.

```bash
for k in first_kernel accum_kernel; do
  echo "--- jit_$k"
  kubectl logs -n trellis -l jobset.sigs.k8s.io/jobset-name=igorts-train --tail=-1 \
    | grep "jit_$k" | grep -oE '^I[0-9]+ [0-9:]+' | sort | sed -n '1p;$p'
done
```

Measured on the 100-step run (step 0 ended 04:28:23):

| module | first compile | last compile |
|---|---|---|
| `jit_first_kernel` | 04:20:57 | 04:21:35 |
| `jit_accum_kernel` | 04:23:30 | 04:24:09 |

Nothing after 04:24, and step times 1–3 are 402.93 / 393.58 / 398.10 s — flat to
within 2%. A recompile would show as a multi-minute outlier step.

Sequence packing collapsed a step from 32 microbatches to 5 and bought about
4 seconds. That is the clearest possible evidence that **gradient computation was
never the bottleneck** — keep packing on (it is strictly better and free), but
do not expect it to move the total.

### 5.1 Inside the 235 s weight sync

From the orchestrator timestamps of `wsync-v2-r2`:

| Sub-phase | Duration |
|---|---|
| request → sender-coordinator start (param gather, registration) | 13.3 s |
| **schedule generation** | **45.5 s** |
| actual transfer (633 variables, 180,063,636 blocks, 1 → 8 destinations) | 176.5 s |

**The 45.5 s of schedule generation is paid on every single step, and it does
not have to be.** `RaidenController` has a plan cache (`enable_plan_cache=True`
by default) keyed purely on topology — `(src_units, dst_units, dst_mem_type,
skip_d2h, parallelism, group_size, skip_tiling, controller addresses)` — none of
which changes between our steps. It never hits, because:

```python
# tpu_sync/rpc/raiden_controller.py:1503, end of register_work_unit()
self._plan_cache.clear()
```

and tunix's `weight_sync_coordinator.py:1095` re-registers **every** source and
destination work unit on **every** sync round. The cache is invalidated by
construction, so `"reusing cached schedule"` never appears in any of our logs —
only `"generated schedule"`, once per step.

Whether the re-registration is actually necessary depends on whether the
trainer's parameter device buffers keep stable addresses across an optimizer
step (with donation they plausibly do). **Reporting, not fixing** — it needs
that question answered first, and it is ~11% of step time, not the 59%.

### 5.1.1 `wait_for_all()` is *not* why the sync is slow

The weight-sync path does drain every in-flight TPU computation before it reads
the parameters, and it is reasonable to suspect that barrier of hiding the cost:

```python
# /app/maxtext/src/maxtext/training_engine/maxtext_engine.py:2167
# 1. Drain all in-flight TPU computations to ensure weights are fully updated
self._throttler.wait_for_all()
# 2. Extract clean trainable parameters
params_state = self._get_trainable_params_state()
if self._use_weight_converter:
  converted_state = self._weight_converter.convert(params_state)
```

Note that this is in **MaxText**, not tunix — grepping the tunix checkout finds
only `sft/peft_trainer.py:1104` and `experimental/train/peft_trainer_v2.py:1434`,
both at end-of-train-loop, neither on this path. (Only eight files are overlaid,
so for anything else the container's copy is the one that runs; grep the image,
not the checkout.)

It is not the problem. Decomposing step 97's 297.73 s sync by orchestrator
timestamp:

| Sub-phase | Wall clock | Duration |
|---|---|---|
| sync start → `transfer` call — **contains `wait_for_all`**, param extraction, weight conversion, work-unit registration | 15:41:52 → 15:42:04 | **12 s** |
| schedule generation | 15:42:04 → 15:43:52 | **108 s** |
| transfer | 15:43:52 → 15:46:50 | **178 s** |

The drain shares a 12 s bucket with three other operations, so it is **at most
4%** of the sync and in practice well under that. The trainer has nothing
outstanding at that point anyway — the optimizer step it is draining is the same
work whose result the sync is about to read, so the cost is the step, not the
barrier.

### 5.1.2 The sync gets ~21% slower over 100 steps, and it is all schedule generation

Weight sync is not stationary. Sampling `"Weight sync finished in %.2f seconds"`
across all 101 syncs of the 100-step run:

| | | | |
|---|---|---|---|
| sync #1 | 254.4 s | sync #51 | 265.2 s |
| sync #11 | 241.5 s | sync #61 | 280.9 s |
| sync #21 | 243.7 s | sync #71 | 271.8 s |
| sync #31 | 255.4 s | sync #81 | 279.7 s |
| sync #41 | 264.9 s | sync #91 | 278.4 s |
| | | sync #101 | 305.2 s |

**First 10 mean 240.1 s → last 10 mean 291.0 s (+21%).** Split by sub-phase, the
regression is entirely in one place:

| Sub-phase | early | late |
|---|---|---|
| prep (`wait_for_all` + extract + convert + register) | ~13 s | ~12 s |
| **schedule generation** | **45 s** | **108 s** (+140%) |
| transfer | ~177 s | ~178 s |

Transfer is flat — the data volume and the fabric are unchanged, as they should
be. Schedule generation more than doubles. That is a per-round cost growing with
the number of rounds already completed, which is the signature of an accumulating
data structure, and there is a plain candidate:

```python
# tpu_sync/rpc/raiden_controller.py:2051, 2073, 2939
self._active_transfers[req_id] = plan
```

`_active_transfers` is written in three places and **never popped anywhere** —
contrast the sibling bookkeeping in the same file, which is cleaned up properly:

```python
# tpu_sync/rpc/raiden_controller.py:1450-1451
self._active_tasks.pop(req_id, None)
self._task_units.pop(req_id, None)
```

So every completed transfer's plan is retained for the life of the controller.
With ~180M blocks per plan these are not small objects; by sync #101 there are
101 of them live. Whether the growth cost is the dict scan itself, allocator
pressure, or GC time walking a heap that keeps growing was not isolated.

This compounds with the plan-cache invalidation in §5.1: the cache would make
schedule generation nearly free, and its absence is what leaves this growing cost
exposed on every step. **Reported, not fixed** — it is in Raiden, outside both
checkouts, and needs an owner who can confirm the plan is genuinely dead after
the transfer completes.

### 5.2 Scaling the trainer makes it *worse* (measured)

The intuition that more trainer hosts give Raiden's D2H more parallel egress is
wrong:

| Trainer | Chips | Hosts | FSDP | Weight sync |
|---|---|---|---|---|
| `tpuv5p:2x2x2` | 8 | 2 | 8 | **233–258 s** |
| `tpuv5p:2x2x4` | 16 | 4 | 16 | **345 s** |

Doubling the trainer made the dominant cost ~48% worse: more FSDP shards means
more, smaller per-shard transfers, and per-shard overhead swamps the extra
egress. 180M blocks over ~70 GB is ~390 bytes per block — this transfer is
overhead-bound, not bandwidth-bound, which is exactly why adding shards hurts.

8 chips is also the memory floor: full-parameter Adam on 35B needs ~560 GB of
params + grads + fp32 moments + master weights against 95 GB × 8 = 760 GB.
**8 trainer chips is both the minimum that fits and the fastest for weight sync.
Leave `TRAINER_TPU_SLICE` alone — trainer chip count is not the lever.**

### 5.3 The 100-steps-in-2-hours target

2 hours over 100 steps is a 72 s/step budget. Weight sync alone is 235 s. Even
with rollout and training at literally zero, 100 steps costs **6.5 hours**.

The run itself landed at **11.75 h** (423.0 s/step mean) — between the 401 s
steady-state step of the 2-step run and the 442 s of its own last decile, which
is exactly the §5.1.2 drift.

The target is not reachable by tuning batch shapes, by scaling the trainer, or
by packing. It requires making weight sync cheaper. In descending order of
expected payoff:

- **Overlap sync with rollout (`max_staleness > 0`).** Currently
  `max_staleness=0`, so generation waits for the sync to land. One step of
  staleness lets rollout run against the previous policy while the sync
  proceeds, hiding the ~166 s of rollout behind the ~235 s of sync: ~6.5 h
  rather than ~11 h. This changes the algorithm from strictly on-policy GRPO to
  slightly off-policy — a common and legitimate choice in async RL, but a
  **semantic** change to the thing being reproduced, not a tuning knob. Flagged
  for a human decision, deliberately not done.
- **Fix the plan-cache invalidation (§5.1) and the `_active_transfers` leak
  (§5.1.2).** Together these are the whole schedule-generation phase — 45 s/step
  at the start and 108 s/step by the end, ~19% of the run's wall clock. No
  semantic change, and it also removes the upward drift. Worth ~2 h of the 11.75.
- **Fewer rollout destinations.** The transfer is 1 → 8. Rollout-side memory is
  slack, so fewer, busier replicas may cost little generation throughput while
  cutting fan-out. Untested.
- **Larger Raiden blocks.** ~390 bytes/block is very small; if `group_size` or
  the tiling can be coarsened, the 176 s transfer is where it would show.
  Untested.

---

## 6. Sequence packing

`MAX_SEQ_TOKEN_PER_TPU=4096` with `MAX_PROMPT_LENGTH=512`,
`MAX_RESPONSE_LENGTH=1024`.

Left at its floor, packing is a no-op *by construction*:
`validate_packing_budget` demands `budget >= max_prompt + max_response`, and
stock `maxtext_utils` pins `max_target_length` to that same sum — so a packed
row holds exactly one maximal trajectory. Measured that way: ~1.14 trajectories
per row. Overlay files 6 and 7 let `max_seq_token_per_tpu` raise
`max_target_length` instead.

At 4096 a step packs 42/67/60/28 trajectories per microbatch (~7.6 per row),
collapsing 32 microbatches to 5. Worth ~4 s of a 400 s step (see §5) — keep it,
but it is not where the time is.

`TRAIN_MICRO_BATCH_SIZE` must equal `trainer_fsdp × trainer_dp` so MaxText gets
`per_device_batch_size = 1`; the launcher derives it from `TRAINER_MESH_FSDP`.

---

## 7. Topology, and why the architecture is left alone

| Role | Slice | Chips | Mesh |
|---|---|---|---|
| Trainer | `tpuv5p:2x2x2` (Pathways) | 8 | FSDP=8 |
| Rollout | 8 × `tpuv5p:2x2x1` | 32 | TP=2 × DP=2 each |

**Rollout TP must stay at 2.** At TP=4, `maxtext_utils` replicates
`base_num_kv_heads` 2 → 4 (which MaxText then rejects outright, since the model
yml also sets it) and `gmm_v2` pads the MoE MLP dim 512 → 1024 because 512/4 is
not a multiple of 2×128. The MoE experts are ~32B of this 35B model, so that
padding roughly doubles what has to be held. At TP=2, 512/2 = 256 is already
aligned and `num_kv_heads` is untouched.

No architectural config is modified. The goal is to run this model, not a
different one.

### `inprocess_vllm`, not `vllm`

`SAMPLER=inprocess_vllm` is required, not a preference. On the plain `vllm` path
the Raiden destination registry is never populated with MaxText-named variables,
so weight sync dies in preflight before step 0 with all 633 source variables
unmatched (`source variable 'decoder.decoder_norm.scale' has no destination
counterpart`). `inprocess_vllm` logs `Using local registration for destination
metadata` and matches all 633. Same image, same mesh, everything else identical.

---

## 8. Open issues — reported, not fixed

- **§5.1 Raiden plan cache is invalidated every round.** ~45 s/step. Needs the
  buffer-address-stability question answered first.
- **§5.1.2 `_active_transfers` is never popped**, and weight sync degrades 21%
  over 100 steps as a result (schedule generation 45 s → 108 s; transfer flat).
  `raiden_controller.py` writes `self._active_transfers[req_id] = plan` at lines
  2051, 2073 and 2939 and removes it nowhere, while the sibling `_active_tasks` /
  `_task_units` are cleaned up at 1450-1451. Needs a Raiden owner to confirm the
  plan is dead once the transfer completes.
- **Clipped trajectories score a hard 0.0.** A trajectory that spends its whole
  response budget is marked `MAX_CONTEXT_LIMIT_REACHED`;
  `trajectory_collect_engine.collect()` then skips `_append_final_reward`
  entirely, so it is never graded — it does not merely lose the format points,
  it scores zero. It is also invisible to the example's own completion logging,
  which hangs off `env.step`. Overlay file 8 adds `TUNIX_LOG_GENERATIONS=N` to
  see them. With `MAX_RESPONSE_LENGTH=1024` this is no longer pinning the mean
  to zero, but it still silently mislabels long correct answers as failures.
- **`_rule_table_for` makes `WeightConverter` unreachable for qwen3.5.**
  `maxtext_vllm_rollout.py:56` returns a real rule table only for
  `qwen3-0.6b`; everything else gets `_NO_RULE_TABLE`, so the `WeightConverter`
  returns at lines 103/111 are dead for our model and control always falls
  through to `Qwen35MaxTextToVLLMConverter` at line 116. Harmless today (that
  converter is the right one) but the `use_weight_converter` flag is not doing
  what its name says.
- **`sampler.update_params()` is a no-op in `validate_converter`.** The 4-chip
  harness generates from vLLM's `load_format:'dummy'` weights, so it cannot
  validate a conversion at all. Three different sync paths produced
  byte-identical output with a 0.37 s / +0.00 GiB "sync". Do not trust it as a
  ground-truth harness until this is fixed — see v6 §6.0.1.
- **`resolve_prefuse_moe_weights` is dead code**, and **`kv_tp_size` /
  `moe_mlp_tp_size` never reach the MaxText config** (v6 §6.4).
- **`BETA` and `EPSILON` are exported but never plumbed through** (§1.1).
  `k8s_launcher.sh:60-61` exports them; no command line ever interpolates them.
  The GRPO KL coefficient and the PPO clip range are therefore always the
  argparse defaults. Inert for this run (defaults match), but a silent trap for
  the next person who tries `BETA=0.04`. The fix is one line in each of the two
  `extra_flags` blocks — held back only because the standing instruction is to
  report before changing anything.
- **Silent metric-backend failures.** `sft/metrics_logger.py:199` catches bare
  `Exception` from backend construction and logs it at INFO. A misconfigured
  W&B costs you a whole multi-hour run's telemetry with no warning (§9). This
  should be at least a WARNING, and arguably fatal when a key was explicitly
  supplied.
- **GSM8K is at the reward ceiling for this model** (~0.94), so most GRPO groups
  have zero advantage variance and contribute no gradient. Fine for validating
  the pipeline, weak for demonstrating learning (§3).

---

## 9. Metrics: W&B needs `WANDB_ENTITY`, and fails silently without it

W&B is enabled from `WANDB_API_KEY` in `~/.bashrc`. `.bashrc` returns early for
non-interactive shells, so the launcher reads the export line directly rather
than sourcing:

```bash
eval "$(grep -m1 '^export WANDB_API_KEY=' "${HOME}/.bashrc" || true)"
```

**That alone is not enough, and the failure is nearly invisible.** metrax's
backend calls `wandb.init(project=..., name=..., anonymous="allow")` with no
entity (`metrax/logging/wandb_backend.py:58`), and wandb refuses to start a run
when the key's viewer has no `defaultEntity`. tunix catches the exception and
downgrades it to one INFO line (`sft/metrics_logger.py:199`):

```
[Orchestrator] WandbBackend skipped: entity not specified, and viewer has no default entity
```

The run then proceeds happily with W&B off. An earlier 100-step attempt got all
the way to step 2 before I noticed.

This key's viewer has `defaultEntity: null` and belongs to one team, so:

```bash
export WANDB_ENTITY=${WANDB_ENTITY:-google-trellis}
```

`wandb.init()` reads `WANDB_ENTITY` from the environment, so this is a host-side
fix in `k8s_launcher.sh` and needs **no image rebuild**. To find the right value
for a different key, without ever printing the key:

```bash
curl -s -u "api:${WANDB_API_KEY}" https://api.wandb.ai/graphql \
  -H 'Content-Type: application/json' \
  -d '{"query":"{viewer{username defaultEntity{name} teams{edges{node{name}}}}}"}'
```

Confirm it worked — absence of this line means W&B is off:

```
wandb: 🚀 View run at https://wandb.ai/google-trellis/qwen35-35b-a3b-grpo/runs/<id>
```

Project `qwen35-35b-a3b-grpo`, run name
`${USER}-${RUN_TAG}-b${BATCH_SIZE}g${NUM_GENERATIONS}-s${MAX_STEPS}`,
`FLUSH_METRICS_EVERY_N_STEPS=1`.

Note for whoever inherits this: the key in `~/.bashrc` authenticates as wandb
user `mazumdera`, not `igorts`. Runs land under the `google-trellis` team, which
is the intended destination, but the run author will look wrong.

Everything else host-side needs no image change: the launcher, the YAML
generator, and the Kueue `WorkloadPriorityClass`. **Set the priority class.**
Without it the workload sits at priority 0 and anything else in the
`tpu-shared-cohort` evicts it — an earlier 100-step attempt died at step 43 that
way. `PRIORITY_CLASS=medium` (= 500) is the default here.

Checkpoint saving is off (`CHECKPOINT_SAVE_INTERVAL_STEPS=0`): the orchestrator
stops issuing save requests while `maxtext_utils` keeps
`enable_checkpointing=True` (still needed to *restore* the base weights) and
pushes `checkpoint_period` out to 1e9. Saving is being fixed separately.
