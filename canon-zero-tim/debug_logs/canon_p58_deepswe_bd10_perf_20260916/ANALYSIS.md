# bd10 — P58 Qwen3-4B DeepSWE zero-TIM, 128 chips: performance receipt and crash post-mortem

Run identity
- JobSet `canon-p58-128s-zsoptt-full-timbd10`, namespace `trellis`, cluster `bodaborg-v5p-nap` (europe-west4).
- Source commit pinned in the container: `30e3b015dc2fba61f97a2c33f32431c2c67f1d0e`.
- Image `europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/yuxzhang-repo/tunix_frozenlake_image@sha256:c9f9fd34054216bc67ba386f71e8d58658676f4a878e5980087c59db0b2d7d16`.
- Profile `cluster/profiles/qwen3-4b-dp8-tp8-deepswe-tim-systemopt.env`, DP8 x TP8, `CANON_P58_TIM_ARM=zero`,
  `CANON_DEEPSWE_SYSTEM_OPTIMIZATION_ARM=treatment`, `CANON_DEEPSWE_ALIGNMENT_WARN_ONLY=0`.
- Container start 2026-09-16T07:07:04Z. First rollout 07:18:01Z. Process died 10:46:45Z.
- Six rollout batches completed (steps 0 through 5). Five optimizer commits landed (updates 0 through 4).
  The step 5 backward pass raised and the trainer exited.

Everything below is parsed from `run.log`, which was copied off the PVC at
`/mnt/disks/linchai_data/deepswe_zero_tim/canon-p58-128s-zsoptt-full-timbd10/run.log` after the crash.
The pod's stdout had already rotated, so `kubectl logs` no longer covers steps 1 through 3; the PVC copy is
the only complete record.

## 1. Headline

The optimisation stack under test landed and is worth **-22.8% wall clock per training step** against the
bd07 baseline. The run then died for an unrelated reason: the sandbox pool degraded over three hours until a
single batch lost 100 of 128 trajectories to environment timeouts, which starved four of the eight data
parallel ranks and tripped a fail-closed contract in the segmented backward.

| | bd07 baseline | bd10 | delta |
| --- | ---: | ---: | ---: |
| steady step wall (s) | 2388 | **1843.0** | **-545.0 (-22.8%)** |
| rollout (s) | 2135 | 1713.6 | -421.4 (-19.7%) |
| post-rollout tail (s) | 253 | 129.4 | -123.6 (-48.9%) |
| `segmented_value_and_grad` (s) | 118.0 | 103.9 | -14.1 (-11.9%) |
| `optimizer_transaction` (s) | 1.1 | 1.01 | -0.09 |
| engine step p50 (ms) | 129.21 | 104.0 | -25.2 (-19.5%) |
| projected 1000 steps | 27.6 days | **21.3 days** | -6.3 days |

bd10 steady numbers are the mean over steps 1 through 4. Step 0 is excluded because it carries first-trace
compilation; step 5 is excluded because it is the failed batch.

## 2. Per-step wall clock

Step wall is the difference between consecutive `[P58.36.BATCH] DEADLINE_START batch_started_unix=` stamps,
so it includes rollout, log-prob recomputation, alignment, backward, optimizer and weight sync.

| step | step wall (s) | rollout (s) | tail (s) | `segmented_value_and_grad` (s) | `optimizer_transaction` (s) | `weight_sync_engine` (s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2922.4 | 1794.9 | 1127.5 | 726.767 | 53.608 | 6.339 |
| 1 | 1820.9 | 1685.9 | 135.0 | 108.774 | 1.098 | 8.244 |
| 2 | 1559.5 | 1440.1 | 119.4 | 95.783 | 0.952 | 6.495 |
| 3 | 1719.2 | 1591.0 | 128.2 | 101.868 | 0.986 | 8.340 |
| 4 | 2272.4 | 2137.4 | 135.0 | 109.137 | 0.989 | 6.838 |
| 5 | (crashed) | 2221.7 | - | - | - | - |

Step 0 costs 2922.4 s against bd07's 3376 s, so even the compiling step is 453.6 s cheaper. Its tail is
dominated by the one-time trace: `segmented_value_and_grad` at 726.767 s collapses to a 95.8 to 109.1 s band
from step 1 onward, and `optimizer_transaction` drops from 53.608 s to roughly 1.0 s once the Adam program is
cached.

## 3. Where the -22.8% comes from

### 3.1 Kernel work: patch 40 delivered exactly what it predicted

`[ENGINE_DRIVER]` reports a sustained `step_ms_p50` of 103.9 to 105.8 ms for the whole run, against 129.21 ms
on bd07. That is -19.5%, and it matches the 105.12 ms that the offline field-by-field accounting predicted
from the patch 40 receipt. The gain sits in `sample_ms` (84 to 92 ms here, 111.13 ms on bd07) while the host
side is flat, which is the signature of device-side work getting cheaper rather than dispatch getting
cheaper. `busy_pct` stays at 99 to 100 with `idle_polls=0`, so the engine is never starved.

### 3.2 Audit downsampling: confirmed by receipt, not by inference

`CANON_ALIGNMENT_AUDIT_EVERY=10` is honoured. Every step from 1 onward emits

```
[CANON_ALIGN] audit=skip step=N every=10 rescore_b=0 trainer_old=0
```

and neither a `rescore_b` nor a `trainer_old` stage appears anywhere in steps 1 through 5. On bd07 those two
stages cost 80.4 s and 121.7 s on every step. The measured tail drop is 253 s to 129.4 s, which is -123.6 s,
consistent with removing that pair net of the log-prob and alignment work that still runs each step.

Strictness is not weakened by the downsampling: `[CANON_ALIGN_PRE] step=N verdict=PASS` is emitted on every
step, `[P34.WEIGHTS] EXACT step=N leaves=398 elements=4022468096 devices=64` confirms an exact weight sync,
and `CANON_DEEPSWE_ALIGNMENT_WARN_ONLY=0` is set, so a violation would still have failed the run.

### 3.3 First update is numerically clean

```json
{"dp":8,"tp":8,"update":0,"effective_learning_rate":9.999999974752427e-07,
 "gradient_finite":true,"optimizer_transaction_valid":true,
 "parameter_changed_elements":3662568721,"parameter_delta_finite":true,
 "train_steps_before":0,"train_steps_after":1,"workload":"p58-qwen4b-tim-128"}
```

3662568721 of 4022468096 parameters changed, which is 91.1%. `naive_norm` and `stable_norm` agree to six
significant figures at 0.015574, so the accumulator is not losing precision. The largest per-leaf gradient is
5.06e-4 on `layers[6].mlp.down_proj.kernel`.

### 3.4 What did not land

`CANON_FIXED_AR_GATHER` and `CANON_FIXED_AR_SCATTER` are unset on this arm, so the `all_to_all` tensor
parallel reduction from commit `65392207` is not exercised here. Enabling it needs a profile-level admission
change and is tracked separately. Four of the five candidate optimisations are active in this run.

## 4. Training quality

| step | solve ratio | nonzero advantage ratio | solved | incomplete | env timeouts | status histogram |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0.0312 | 0.125 | 4 | 6 | 0 | SUCCEEDED 122, MAX_CONTEXT_LIMIT_REACHED 3, MODEL_TIMEOUT 3 |
| 1 | 0.0078 | 0.125 | 1 | 8 | 0 | SUCCEEDED 120, MODEL_TIMEOUT 7, MAX_CONTEXT_LIMIT_REACHED 1 |
| 2 | 0.0234 | 0.117 | 3 | 3 | 0 | SUCCEEDED 125, MODEL_TIMEOUT 3 |
| 3 | 0.0312 | 0.375 | 4 | 4 | 0 | SUCCEEDED 124, MAX_CONTEXT_LIMIT_REACHED 3, MODEL_TIMEOUT 1 |
| 4 | 0.0234 | 0.227 | 3 | 10 | 0 | SUCCEEDED 118, MODEL_TIMEOUT 6, MAX_CONTEXT_LIMIT_REACHED 4 |
| 5 | 0.0000 | 0.000 | 0 | 101 | 100 | SUCCEEDED 27, ENV_TIMEOUT 100, MAX_CONTEXT_LIMIT_REACHED 1 |

Five steps is far too short to say anything about learning. The solve ratio moves between 0.0078 and 0.0312
with no trend, which is what an untrained Qwen3-4B-Instruct on R2E-Gym looks like.

The number worth attention is `nonzero_advantage_ratio`. With `CANON_GLOBAL_PROMPTS=8` and
`CANON_NUM_GENERATIONS=16`, RLOO's within-group baseline gives exactly zero advantage to any group that is
uniformly solved or uniformly failed. In steps 0 through 4 only one to three of the eight groups were mixed,
so 12.5% to 37.5% of the sampled trajectories carried gradient. This is a recipe property, not a defect in
the code under test, but it caps sample efficiency and is the obvious lever if the run is to be extended.

## 5. Crash post-mortem

### 5.1 The exception

```
File "/app/tunix/rl/agentic/agentic_rl_learner.py", line 4468, in train
  segmented_result = self._run_p28_g6_update(
File "/app/tunix/rl/agentic/agentic_rl_learner.py", line 1809, in _run_p28_g6_update
  result = adapter.segmented_dp_grpo_value_and_grad(
File "/app/tunix/rl/canonical_qwen3_adapter.py", line 12072, in segmented_dp_grpo_value_and_grad
  specs = tuple(
File "/app/tunix/rl/canonical_qwen3_adapter.py", line 10168, in _p32_group_spec
  raise FunctionalMappingError(
tunix.rl.canonical_qwen3_adapter.FunctionalMappingError: P32 grouped reverse requires a nonempty prompt and
at least two real tokens on every rank:
  n=[4641, 4557, 4475, 4149, 0, 0, 0, 0]
  prompt=[1837, 1803, 1851, 1822, 0, 0, 0, 0]
  completion=[2804, 2754, 2624, 2327, 0, 0, 0, 0]
```

Four of the eight data parallel ranks received zero tokens. The contract refuses to build a group spec for an
empty rank and fails closed rather than emitting a silently wrong gradient. The guard behaved correctly.

### 5.2 Why the ranks were empty

Step 5 lost 100 of 128 trajectories to `ENV_TIMEOUT`:

```json
{"env_timeout_trajectories": 100, "env_timeout_trajectory_ratio": 0.78125,
 "compact_filtered_trajectories": 101, "compact_filtered_trajectory_ratio": 0.7890625,
 "incomplete_trajectories": 101, "solved_trajectories": 0}
```

That left 27 usable rows. `[P32.LENGTH_SORT] enabled=1 rows=128 dp=8 groups=16` sorts rows by length before
sharding, which concentrates all the real content into the low ranks and leaves the high ranks holding only
filtered-out rows. With 27 rows spread across 16 groups the packing filled four ranks and emptied four.

### 5.3 The sandbox pool degraded for three hours before it broke

`sandbox_start` p50, per step, in seconds:

| step | 0 | 1 | 2 | 3 | 4 | 5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sandbox_start p50 | 20.2 | 315.8 | 150.4 | 315.8 | 848.6 | 1229.6 |

Step 0 started sandboxes in 20 seconds. By step 5 the median sandbox took 20 minutes to come up, a factor of
61. `CANON_DEEPSWE_SANDBOX_TIMEOUT_S` is 3300 s and the per-turn deadline is 300 s, so once startup latency
climbs into the high hundreds of seconds the trajectories begin losing their budget to scheduling rather than
to work, and the batch collapses.

This is a monotone degradation, not a spike, and it was visible three steps before the crash. It is the
leading indicator to alarm on.

Root cause of the degradation is not established from this log alone. Candidates, none of them confirmed:
- RepoEnv pods on `sandbox-cpu-pool` not being reclaimed as fast as they are created, so the pool saturates.
- Contention on the shared `sandbox-cpu-pool` node pool from workloads outside this JobSet.
- Image pull or disk pressure on the sandbox nodes accumulating over the run.

Deciding between these needs node-level and namespace-level data from the window 07:18Z to 10:47Z, which this
bundle does not contain.

### 5.4 Cost of the failure mode

The trainer exited at 10:46:45Z. The JobSet has no restart policy for this path, and the pods stayed up with
the Python process gone, so 33 pods and 128 v5p chips sat idle from 10:46:45Z until the run was noticed. There
is no checkpoint: `--ckpt_dir=none` is set, so the five completed updates are not recoverable.

Two independent gaps made this expensive. First, nothing in the workload notices that the trainer died while
the pods stay Running, so neither JobSet status nor pod phase reflects the failure. Second, the wandb probe
that was being used for monitoring never received a single history row for this run (see section 6), so the
external monitor could not see progress either.

## 6. The wandb metric path produced nothing

Run `yuxzhang-google/tunix/52kdikup` was created at 07:13:58Z and reported `state=running` throughout, but
`scan_history` returns zero rows and the summary has zero keys, including after the process died. Only one run
was created; the second init was folded into the first:

```
07:13:58  setting up run 52kdikup
07:18:00  wandb.init() called while a run is active and reinit is set to 'default', so returning the previous run.
07:18:00  [CANON_P34_WANDB] ONLINE_RUN_PASS project=zero-tim-deepswe-4b-native-zero group=qwen3-4b-p58-full name=canon-p58-128s-zsoptt-full-timbd10
```

So the run object exists and is correctly associated with the JobSet, but no metric ever reached the server.
The head process is CPU-saturated orchestrating 128 sandboxes, and the wandb writer is asynchronous, which is
the most plausible explanation, but that is a hypothesis: the local run directory at
`/app/wandb/run-20260916_071358-52kdikup` was not recovered before the pod was reclaimed, so the backlog was
never inspected.

The practical consequence stands regardless of cause: for this workload wandb is not a trustworthy liveness or
progress signal, and `run.log` on the PVC is.

## 7. Contents of this directory

| file | what it is |
| --- | --- |
| `run.log.gz` | full trainer stdout, 07:07:04Z to 10:46:45Z, copied from the PVC |
| `per_step_metrics.csv` | one row per step: wall, rollout, tail, quality, stage percentiles |
| `batch_metrics_step{0..5}.json` | raw `[P58.BATCH_METRICS_JSON]` payloads |
| `trajectory_log_slim.csv.gz` | per-trajectory metrics, 768 rows; `conversation_text`, `conversation_tokens`, `conversation_masks` and `old_logprobs` removed because the raw file is 122 MB |
| `env.sh` | the container environment as resolved at startup |
| `updates.jsonl.gz` | optimizer update receipts |
| `alignment.jsonl.gz`, `pre_alignment.jsonl` | zero-TIM alignment evidence |
| `weight_attestation.jsonl` | per-step weight sync attestations |
| `metrics/events.out.tfevents.*` | TensorBoard event file |

## 8. Follow-ups

1. The sandbox pool degradation is the blocker. Relaunching without understanding it will reach the same
   state in roughly three hours.
2. `_p32_group_spec` fails closed on an empty rank, which is right, but a batch that loses 78% of its
   trajectories should be dropped or retried before it reaches the backward, rather than taking the trainer
   down. A skip-batch path guarded by a floor on usable rows would turn a fatal into a logged skip.
3. Alarm on `sandbox_start` p50 crossing a threshold. It moved 20.2 to 315.8 to 848.6 to 1229.6 s over four
   steps and was unambiguous well before the failure.
4. `--ckpt_dir=none` means any failure discards all completed updates. For runs intended to go past a handful
   of steps this should change.
5. Nothing detects trainer death while the pods stay Running. A liveness probe on the training process, or a
   JobSet `successPolicy`, would release the chips instead of idling them.
6. Recover or disable the wandb path. As configured it consumed a run slot and produced no data.
7. `nonzero_advantage_ratio` between 0.117 and 0.375 means most sampled trajectories carry no gradient.
   Worth revisiting `CANON_GLOBAL_PROMPTS=8` and the prompt difficulty mix before committing to a long run.
