# Garbage rollout tokens confined to two rollout replicas (mk-7x-0925y)

**Run:** `mk-7x-0925y`, Qwen3.5-397B-A17B DeepSWE on TPU v7x (`bodaborg-tpu7x-gsc`, `priority-dev`), 2026-09-25 ~17:05–18:17 UTC.
Trainer `tpu7x:4x4x8`; 16 rollout replicas, each `tpu7x:2x2x2` (EP=16, TP=1, DP attention).
Raiden wheel `tpu_sync_jax-0.0.1.dev987922633`, Pathways images `raiden_987686911`.

**Summary:** about 1% of all generated tokens in the first batch were token id 0 (`!`), inserted mid-word and corrupting tool calls.
All of it came from **2 of the 16 rollout replicas, `roll-6` and `roll-9`**. Every trajectory they served is corrupted;
the other 14 replicas, running the same code, weights and config, produced clean output. The same two replicas ran
3–5x slower and stalled repeatedly. Weight conversion looks unlikely as the cause. A follow-up run on different nodes
(`mk-7x-0925z`) is clean so far.

## Symptom

From the trajectory store `gs://mohitkhatwani-trellis-v7x-uc1/trajectories/mk-7x-0925y/store/run_1790356051_d5260ee3/`
(256 trajectories = 16 tasks × 16 generations, 7,667 agent turns, 1.25M generated tokens):

- **12,872 tokens (1.03%) are token id 0 (`!`)**, found in 31% of agent turns. (255/256 trajectories contain at least one token 0, but that count includes genuine `!` in text; see the per-replica table for the real split.)
- They are almost always isolated single tokens (12,630 singles, 97 pairs, 16 triples) inside otherwise coherent text:
  - `str_replace_editor {'command': 'view', 'path!': '/testbed/tornado/ioloop.py', 'view_range': '[!>[598, 625]]'}`
  - `execute_bash {'command': 'ldd /testbed! _libs/window/aggregations.cpython-38-x86_64-linux-gnu.so ...`
  - garbled tool names: `execute!_!ash`, `str!replace_editor`, `<function=execute_bash!>`, `find /!testbe!d/Tests/images`
- **970 of the token-0 positions have a `null` logprob, and no other token in the dataset has a null logprob.**
  The remaining token-0s have logprob ≈ 0 (probability ≈ 1). This fits NaN logits (sampler falls back to index 0,
  NaN is written as null) plus finite but degenerate logits dominated by token 0: one failure seen two ways.
- Onset: no NaN in agent turns 0–2; it appears from about turn 3 and only in trajectories finishing after ~18:01 UTC,
  i.e. the affected replicas got worse as the run went on.
- Training impact: these tokens carry `assistant_mask=1`, and null logprobs would poison the importance ratios.

## Evidence: only roll-6 and roll-9 are affected

Rollout pods do not log trajectory IDs. Each trajectory was mapped to its replica by matching the rollout pod's
`Terminal step has N assistant tokens ...` log line (same N, within 30 s) to the trajectory's last agent turn and
the write time of its `metadata.json` in the store. 240/256 trajectories matched uniquely.

A trajectory counts as **bad** if it has any token 0 with a null logprob (no false positives: nothing else has one).

| replica | trajectories | bad | null-logprob token 0 | token 0 % of generated |
|---|---|---|---|---|
| **roll-6** | 20 | **20/20** | **698** | **10.72%** |
| **roll-9** | 19 | **19/19** | **258** | **1.30%** |
| roll-0 | 17 | 0 | 0 | 0.23% |
| roll-1 | 12 | 0 | 0 | 0.22% |
| roll-2 | 9 | 0 | 0 | 0.20% |
| roll-3 | 17 | 0 | 0 | 0.21% |
| roll-4 | 18 | 0 | 0 | 0.26% |
| roll-5 | 10 | 0 | 0 | 0.20% |
| roll-7 | 17 | 0 | 0 | 0.22% |
| roll-8 | 17 | 0 | 0 | 0.21% |
| roll-10 | 17 | 0 | 0 | 0.23% |
| roll-11 | 14 | 0 | 0 | 0.19% |
| roll-12 | 17 | 0 | 0 | 0.22% |
| roll-13 | 12 | 0 | 0 | 0.24% |
| roll-14 | 14 | 0 | 0 | 0.19% |
| roll-15 | 10 | 0 | 0 | 0.24% |

The 0.19–0.26% token 0 on healthy replicas is genuine `!` in text (e.g. "I see!").

### Same task, bad replica vs good replica

| trajectory | replica | generated tokens | token 0 | null-logprob token 0 |
|---|---|---|---|---|
| `traj_deepswe_0_g13` | roll-6 | 2,984 | 2.45% | 38 |
| `traj_deepswe_0_g10` | roll-0 | 6,276 | 0.41% | 0 |
| `traj_deepswe_10_g6` | roll-9 | 1,993 | 1.25% | 12 |
| `traj_deepswe_10_g0` | roll-3 | 1,997 | 0.10% | 0 |

Worst trajectories (all from the bad replicas): `traj_deepswe_14_g4` (4,751 token-0), `traj_deepswe_11_g3` (1,235), `traj_deepswe_6_g5` (901).

### The bad replicas were also slow and stalling

vLLM engine stats lines (`... generation throughput: X tokens/s ...`), host 0 of each replica, 17:50–18:02 UTC:

| replica | stats lines | mean generation throughput | lines at 0 tok/s |
|---|---|---|---|
| **roll-6** | 31 | **27.1 tok/s** | **15** |
| **roll-9** | 41 | **56.8 tok/s** | **14** |
| roll-3 | 57 | 148.4 tok/s | 9 |
| roll-0 | 51 | 150.2 tok/s | 8 |

- Stalls of 1–4 minutes at 0 tok/s with ~20 requests running, e.g. roll-6 17:52:07 → 17:56:07, roll-9 18:01:04 → 18:02:54.
- roll-6 and roll-9 finished last (18:09–18:11 UTC vs ~18:01 for the rest).
- Concurrency alone doesn't explain it: both peaked at 20 running requests, but roll-12 peaked at 22 and roll-10
  sat at ≥18 for long stretches, and both are clean.

### What the logs do *not* show

- Weight-sync destination checksums (`post_weight_sync`) are identical on all 16 replicas
  (e.g. decoder_norm 2744.49, A_log 178.56, conv1d 592.67).
- No missing/unexpected/skipped parameter, shape or dtype mismatch warnings.
- No HBM, ICI, NaN, restart or crash messages on roll-6/roll-9.
- The only error lines there are `Container stop/delete error: ForbiddenException()`, which also appear on healthy replicas.

## Nodes behind the bad replicas

From the `compute.googleapis.com/resource_name` label on the rollout pods' Cloud Logging entries:

| replica | host 0 | host 1 |
|---|---|---|
| roll-6 | `gke-tpu-fd763e16-pxxs` | `gke-tpu-fd763e16-2ct8` |
| roll-9 | `gke-tpu-04442ddf-wvx7` | `gke-tpu-04442ddf-112d` |

## Hypotheses (ranked)

1. **roll-6 and roll-9 are degraded (hardware, HBM, or per-replica runtime state).** Identical code, weights and config
   on 16 replicas, yet only 2 fail, and those 2 are also the slow, stalling ones. This is the strongest signal.
2. **State corruption in the align-mode mamba prefix cache, spreading within a replica.** This is the first run on the
   MaxText rollout path with `mamba_cache_mode=align` (enabled by maxtext commit 2fd1b5fad). tpu-inference does not
   act on vLLM's `new_block_ids_to_zero` / `kv_cache_block_copies`, so mamba blocks are not zeroed or copied on TPU,
   and out-of-range mamba slot indices are clipped (aliasing another request's state). NaN is ~5x more likely on the
   first token after a prefill and more common after short prefills, which fits reading bad saved state; once a
   replica's pool holds a NaN block, prefix-cache hits would spread it to every later request on that replica. The
   logs can't separate this from #1: the stalls could be the trigger rather than a symptom.
3. **Weight conversion error (unlikely).** Checksums are identical, the same conversion works for 35B, turns 0–2
   (~45k tokens) are NaN-free, and 14/16 replicas with the same synced weights are clean. A static conversion bug would
   hit every replica.

## Follow-up run: mk-7x-0925z (different nodes)

Same image, wheel and rollout config (trainer batch settings changed for an unrelated trainer OOM). None of the four
nodes above are used by its rollout replicas.

As of 19:39 UTC (first batch: 128 trajectories, 3,803 agent turns, 630k generated tokens):
**token 0 = 0.21%, null-logprob token 0 = 0**, which matches the healthy-replica baseline.

If it stays clean through later turns, that points to #1 (hardware on the four nodes above). If corruption appears
on other replicas, that points to #2.

## Next steps (cheapest first)

1. Keep watching mk-7x-0925z trajectories for null-logprob token 0, especially in later turns.
2. If 0925z stays clean: report `gke-tpu-fd763e16-{pxxs,2ct8}` and `gke-tpu-04442ddf-{wvx7,112d}` for hardware checks.
3. Log the replica id with each request/trajectory, and record node names for every rollout pod at launch, so the
   mapping doesn't depend on timing.
4. Add a per-step NaN-logits counter in the rollout sampler, logged next to the engine stats line.
5. If corruption reproduces on other hardware: rerun with prefix caching off (`enable_prefix_caching=False`), then
   with `mamba_cache_mode=none`, one change at a time.
6. Replay a bad prompt (e.g. `traj_deepswe_14_g4` step 11) on a healthy replica with prefix caching on and off.

## Log queries

Cloud Logging, project `cloud-tpu-shared-capacity`. Base filter:

```
resource.type="k8s_container"
resource.labels.cluster_name="bodaborg-tpu7x-gsc"
resource.labels.namespace_name="priority-dev"
timestamp>="2026-09-25T17:00:00Z" timestamp<="2026-09-25T18:20:00Z"
```

- Trajectory completions per replica: `resource.labels.pod_name:"mk-7x-0925y-roll-" textPayload:"Terminal step has"`
- Engine throughput: `resource.labels.pod_name:"mk-7x-0925y-roll-6-proc-0-0" textPayload:"generation throughput"` (swap in roll-9 / roll-3 / roll-0)
- Weight-sync checksums: `resource.labels.pod_name:"mk-7x-0925y-roll-" textPayload:"post_weight_sync"`
- Node of a pod: read `labels."compute.googleapis.com/resource_name"` on any entry for that pod.
