# Qwen3.5-35B-A3B distributed GRPO on TPU v5p — reproduction report (v8)

Companion to igorts' `qwen35_report_v7.md`, not a replacement. v7 measured a
wide topology (8 rollout slices, 256 rollouts/step) and landed at 11.75 h for
100 steps. This report measures a **narrow** one — a single 4-chip rollout
replica, 32 rollouts/step — and lands at **2 h 15 m**. Both are correct runs of
the same pipeline; §3.4 explains why the difference is almost entirely Raiden
fan-out and not anything that made training better.

Everything below was measured on cluster `bodaborg-v5p-nap`, project
`cloud-tpu-shared-capacity`, region `europe-west4`, namespace `default`.

Base image `gcr.io/cloud-tpu-multipod-dev/yixuannwang_google_com-runner:yixuann-e2e-0912head-v8`
plus a **six-patch** overlay, built by `docker/maz-q35/build_push.sh`. The
overlay patches files in place rather than copying them in: the base image's
`/app/tunix` is a different lineage from `github.com/google/tunix` HEAD, and
copying whole files over it silently reverts Yixuan's fixes.

---

## 0. Status

**The 100-step run completed.** `maz-q35-3`: 100/100 steps, exit code 0, no
preemption, no restarts, no recompilation after step 2.

| | |
|---|---|
| Wall clock, container start → exit | **2 h 14 m 46 s** |
| Wall clock, step 0 start → finish | 2 h 04 m 30 s |
| Mean `step_time`, all 100 steps | **75.42 s** |
| Mean `step_time`, steps 3–99 | **72.12 s** (p50 71.66, p90 74.63, max 90.87) |
| Mean weight sync, 101 syncs | **60.21 s** — 80% of a steady-state step |
| Mean `reward_mean` | 0.7342 (min 0.0938, max 1.0000) |
| `reward_mean`, first 10 / last 10 steps | 0.6691 / 0.6859 |
| Steps with `loss` exactly 0.0000 | 13 / 100 |
| W&B | [`google-trellis/trellis-gsm8k/runs/v8t407pk`](https://wandb.ai/google-trellis/trellis-gsm8k/runs/v8t407pk) |

Three things this run establishes that v7 could not:

1. **`SAMPLER=vllm` works.** v7 §7 reports that the plain `vllm` path dies in
   preflight with all 633 source variables unmatched, and concludes
   `inprocess_vllm` is "required, not a preference". tunix PR 2229 fixes it.
   All 101 syncs of this run matched 633/633 over a separate vLLM server
   process. See §2.
2. **The trajectory-reward bug is fixed upstream and the fix is in this image.**
   `reward_mean` takes 40 distinct values across the run rather than being
   identically 0.0000. See §2, patch `tunix-0003`.
3. **Sequence packing actually packs.** `MAX_SEQ_TOKEN_PER_TPU=4096` collapses
   all 32 trajectories of a step into a single microbatch, every step. Stock
   `maxtext_utils` makes packing a no-op by construction; see §4.

**Two things are still broken, both in checkpointing.** This run has
`CHECKPOINT_SAVE_INTERVAL_STEPS=0` for that reason. The preceding run
`maz-q35-2` had it at 10 and was OOM-killed writing the step-10 checkpoint, and
its restart then hit a GCS IAM failure. Both are described in §6 with the
evidence, and neither is fixed here.

---

## 1. Reproducing it

### 1.0 Prerequisites

```bash
gcloud auth login                       # as your @google.com identity
gcloud auth application-default login   # same identity; the kubectl plugin reads this
gcloud auth configure-docker gcr.io -q
```

You need a `kubectl` context for `bodaborg-v5p-nap` that authenticates **as
you**, not as the VM's default compute service account.
`gcloud container clusters get-credentials` writes a context backed by
`gke-gcloud-auth-plugin`, which authenticates as gcloud's *active* account — on
a GCE launch VM that is `<project-number>-compute@developer.gserviceaccount.com`,
which cannot create JobSets here. Mint from Application Default Credentials
instead:

```bash
cat > ~/.local/bin/kubectl-adc-cred <<'EOF'
#!/bin/bash
set -euo pipefail
token="$(gcloud auth application-default print-access-token)"
expiry="$(date -u -d '+5 minutes' +%Y-%m-%dT%H:%M:%SZ)"
printf '{"apiVersion":"client.authentication.k8s.io/v1beta1","kind":"ExecCredential","status":{"token":"%s","expirationTimestamp":"%s"}}\n' \
  "${token}" "${expiry}"
EOF
chmod +x ~/.local/bin/kubectl-adc-cred

gcloud container clusters get-credentials bodaborg-v5p-nap \
  --region europe-west4 --project cloud-tpu-shared-capacity
kubectl config set-credentials my-adc \
  --exec-api-version=client.authentication.k8s.io/v1beta1 \
  --exec-command="$HOME/.local/bin/kubectl-adc-cred"
kubectl config set-context bodaborg-adc \
  --cluster=gke_cloud-tpu-shared-capacity_europe-west4_bodaborg-v5p-nap \
  --user=my-adc --namespace=default
kubectl config use-context bodaborg-adc

kubectl auth whoami -o jsonpath='{.status.userInfo.username}'   # want your @google.com
kubectl auth can-i create jobsets.jobset.x-k8s.io -n default    # want "yes"
```

`submit.sh` re-checks both of these and refuses to launch otherwise, because a
wrong identity fails at `kubectl apply` after the image has already been built.

### 1.1 Build and submit

```bash
cd ~/git/tunix
bash docker/maz-q35/build_push.sh q35-0915-v2
```

The build asserts every patch landed — first with `grep` against the file on
disk inside the layer, then by importing each patched module in the finished
image and reading the symbol back *through the import system*, so a stale copy
earlier on `sys.path` fails the build rather than the job. It prints
`ok: all six overlay patches are live in the image` and then the digest.

Pin the digest it prints into `docker/maz-q35/submit.sh` (`TUNIX_IMAGE`). Do not
use the tag: three JobSets are applied in sequence, a tag can be repointed
between them, and weight sync is meaningless unless orchestrator, trainer and
rollout run identical code. The digest this report measured is:

```
gcr.io/cloud-tpu-multipod-dev/mazumdera-runner@sha256:3a8cab3879e655ade728ff3841911b1b2b6b2efa648d1e06b7cae42f6757a4dc
```

Then:

```bash
export WANDB_API_KEY=<your key>          # deliberately not stored in the repo
DRY_RUN=true bash docker/maz-q35/submit.sh 3 100   # renders the 3 manifests, applies nothing
bash docker/maz-q35/submit.sh 3 100                # submits maz-q35-3-{orch,train,roll}
bash docker/maz-q35/submit.sh 3 100 stop           # tears all three down
```

The first argument is the run number and the second is `MAX_STEPS`. The run
number becomes `$USER`, from which the launcher derives every JobSet name, so it
must be unique per run — a reused name silently attaches to the old JobSet's
Cloud Logging history and makes the logs unreadable.

`submit.sh` is the single file that carries every deviation from the launcher's
defaults, and it carries the reasoning for each one in comments. Read it before
changing anything; several of the values are load-bearing in non-obvious ways
(§5).

### 1.2 Hyperparameters — the complete set

**Model and data**

| | |
|---|---|
| Model | `Qwen/Qwen3.5-35B-A3B` (MaxText `qwen3.5-35b-a3b`) |
| Base checkpoint | `gs://hengtaoguo-maxtext-logs/checkpoints/qwen3.5-35b-a3b/scanned/2026-06-11-10-27/0/items` |
| Dataset | GSM8K train (`tfds`), reward mode `env` |
| Precision | `bfloat16` weights and compute |
| Fine-tune type | full-parameter |

**Batch shape**

| Knob | Value | Note |
|---|---|---|
| `MAX_STEPS` | 100 | |
| `BATCH_SIZE` | 4 | prompts per step |
| `NUM_GENERATIONS` | 8 | GRPO group size |
| → rollouts per step | **32** | one eighth of v7's 256; see §3.4 before copying this |
| `MINI_BATCH_SIZE` | 32 | = all of them ⇒ one optimizer update per step |
| `TRAIN_MICRO_BATCH_SIZE` | 8 | must equal `TRAINER_MESH_FSDP` so MaxText sees `per_device_batch_size=1.0` |
| `MAX_PROMPT_LENGTH` | 512 | |
| `MAX_RESPONSE_LENGTH` | 512 | half of v7's 1024; see §7 |
| `MAX_SEQ_TOKEN_PER_TPU` | 4096 | packing budget per row → 4 trajectories/row, 1 microbatch/step |
| `USE_ROLLOUT_LOGPS` | true | `old_logprobs` come from vLLM, not recomputed by the trainer |

**GRPO / optimizer** — all launcher defaults, unmodified:

| Knob | Value |
|---|---|
| `LEARNING_RATE` | `2.0e-7` |
| `ADAM_B1` / `ADAM_B2` / `ADAM_EPS` | 0.9 / 0.99 / `1.0e-8` |
| `WEIGHT_DECAY` | 0.01 |
| `MAX_GRAD_NORM` | 1.0 |
| `BETA` (KL coefficient) | 0 — no reference model, loss is pure clipped policy gradient |
| `EPSILON` (clip) | 0.2 |

`BETA` and `EPSILON` are exported by `k8s_launcher.sh` but never interpolated
into any command line (v7 §1.1 found this). They were inert here only because
the argparse defaults match. Confirm from the orchestrator log, which prints
what was actually used:

```
beta=0.0000, epsilon=0.20, reward_mode=env
```

### 1.3 Checking a live run

```bash
K="kubectl --context bodaborg-adc"

$K get jobset | grep maz-q35
$K get pods   | grep maz-q35

# aggregated step metrics
$K logs <orch-pod> -c main --tail=-1 | grep -E 'Train step|Weight sync finished'

# trainer's actual MaxText config -- verify the knobs arrived, do not trust the export
$K logs <train-pod> -c main --tail=-1 \
  | grep -oE 'max_target_length=[0-9]+|per_device_batch_size=[0-9.]+|prefuse_moe_weights=[A-Za-z]+'
```

Pods are reaped once a JobSet finishes, so after the fact use Cloud Logging.
Always add a `timestamp>` clause — the entry limit otherwise truncates you into
a *previous* run of a reused JobSet name:

```bash
gcloud logging read \
  'resource.type="k8s_container" resource.labels.container_name="main"
   labels."k8s-pod/jobset_sigs_k8s_io/jobset-name"="maz-q35-3-orch"
   timestamp>"2026-09-15T23:00:00Z"' \
  --project=cloud-tpu-shared-capacity --limit=5000 \
  --format='value(textPayload)' --order=asc
```

Node auto-provisioning works on this cluster and is worth waiting out. `maz-q35-3`
was submitted with zero free v5p nodes; `TriggeredScaleUp ... 0->2` provisioned
the 2x2x2 in about a minute. Pods sitting `Pending` for the first 60–90 s is
normal, not a capacity failure.

---

## 2. The overlay — six patches

`docker/maz-q35/Dockerfile` applies each as a `patch -p1 --fuzz=0` with
`--forward` off. An already-applied or inexactly-matching patch therefore fails
the **build**. That is deliberate: it means the base image differs from the one
this overlay was written against, and a run from such an image would not match
this document.

| # | Source | What it fixes |
|---|---|---|
| `maxtext-0001` | MaxText [PR 5219](https://github.com/AI-Hypercomputer/maxtext/pull/5219) | `RLTrainerPayload.metadata` is a *static* PyTree field. Its microbatch-varying keys change the treedef, which recompiles the whole 35B graph **every microbatch**. Clears it before `_gen_model_input_fn`. |
| `maxtext-0002` | MaxText [PR 5234](https://github.com/AI-Hypercomputer/maxtext/pull/5234) | Orbax staged the entire 35B device-to-host at once during a save. Adds `checkpoint_storage_device_host_concurrent_gb` (default 8) to bound in-flight staged bytes. |
| `tunix-0001` | tunix [PR 2229](https://github.com/google/tunix/pull/2229) | Makes `SAMPLER=vllm` work. `tpu_inference` publishes destination variable names bracketed (`['base']['decoder']...`) while the Raiden source side uses dotted keys, so preflight failed with all 633 source variables unmatched. Canonicalises them, and passes `data_parallel_size` through to `AsyncEngineArgs`. |
| `tunix-0002` | tunix [PR 2228](https://github.com/google/tunix/pull/2228) | The two parts of that PR the base image lacks: the `CKPT_D2H_CONCURRENT_GB` override and the lost-device-set exit. Most of 2228 is already in the base image — see §6.1. |
| `tunix-0003` | tunix `bf13cd2c` | The rollout emits `traj["trajectory_reward"]`; `rl_program._extract_reward` read `traj["reward"]`. **Every GRPO advantage in the run was computed from a reward of 0.0.** |
| `tunix-0004` | local, no upstream equivalent | Lets `max_seq_token_per_tpu` raise `max_target_length`. Without it packing is a no-op by construction — see §4. |

`tunix-0003` is upstream `bf13cd2c` with one hunk dropped: it targets a
trajectory-logging block added after this base image was cut. Note that
upstream's `_extract_reward` is now **strict** — it raises `KeyError` rather
than defaulting to 0.0 — which is the right shape, because the silent 0.0
default is exactly what hid this bug for weeks. v7 §3 kept a permissive
`"reward"`-first branch; upstream did not, and the build asserts the strict
behaviour:

```python
assert rl_program._extract_reward({"trajectory_reward": 1.5}) == 1.5
try:
  rl_program._extract_reward({"reward": 1.5})
except KeyError:
  pass
else:
  raise AssertionError("_extract_reward still accepts the pre-bf13cd2c key")
```

### 2.1 Rebasing onto tunix `main` does not work with this base image

Worth recording, because it looks like the obvious move. `git rebase --onto
origin/main` produces one trivial conflict and two hard blockers:

- **Host and image must move together.** `main`'s `k8s_launcher.sh` passes
  `--eos_tokens`, `--trajectory_log_dir` and `--optimizer_*` flags that the
  image's Python rejects, and the image's launcher passes `--max_grad_norm`,
  which `main` dropped.
- **`main` requires a newer `tpu-inference` than the image pins.**
  `vllm_sampler_v2.py:417` calls `_call_worker_method("raiden_h2d", uuid=uuid)`;
  the image's `tpu_inference/worker/tpu_worker.py:847` is `def raiden_h2d(self)`
  with no `uuid` parameter.

`main`'s tunix is 1765 insertions / 900 deletions across 35 files ahead of the
image tree. Take individual upstream commits as patches, as `tunix-0003` does.

---

## 3. Performance: where a step actually goes

### 3.0 There is no recompilation after step 2 — the evidence

`maxtext-0001` exists to stop a per-microbatch recompile. It held:

| Step | `step_time` |
|---|---|
| 0 | 343.85 s — compile-dominated |
| 1 | 111.80 s |
| 2 | 90.28 s |
| 3–99 | **72.12 s mean**, p50 71.66, p90 74.63 |

Over steps 3–99 there is exactly one step above 85 s (step 5, 90.87 s) and no
upward drift: the last ten steps mean 73.37 s against a 72.12 s overall mean. A
recompile of this graph costs minutes, not seconds, and would be unmissable in
that distribution. The trainer logs exactly one compile event, at 23:38:40,
before step 0.

This is the check to run on any rerun — the shape of the distribution, not the
presence or absence of a log line:

```bash
kubectl --context bodaborg-adc logs <orch-pod> -c main --tail=-1 \
  | grep -oE 'step_time: [0-9.]+s'
# want: one large step 0, then flat. Any later multi-minute outlier is a recompile.
```

### 3.1 Weight sync is 80% of a steady-state step

| Phase | Steady state | Share |
|---|---|---|
| **Weight sync** | **60.2 s** | **80%** |
| Rollout generation + gradient + optimizer | ~12 s | 20% |
| **Total `step_time`** | **72.1 s** | |

Decomposed from the orchestrator's own timestamps:

| Sub-phase | sync #2 | sync #50 | sync #101 |
|---|---|---|---|
| prep (`wait_for_all`, param extract, weight convert, work-unit registration) | 13.3 s | 13.0 s | 13.1 s |
| schedule generation | 7.1 s | 9.3 s | 9.9 s |
| transfer (633 variables, 180,063,636 blocks, 1 → 1 destination) | 38.1 s | 37.6 s | 37.7 s |
| **total** | **58.5 s** | **59.9 s** | **60.7 s** |

Sync #1 is 83.8 s; its prep is 34.0 s rather than 13 s because it also does
first-time registration.

### 3.2 The Raiden regressions from v7 are present but small here

v7 §5.1 and §5.1.2 report two Raiden problems. Both reproduce, and both are
order-of-magnitude smaller at fan-out 1:

- **The plan cache never hits.** `raiden_controller.py:1503` clears
  `self._plan_cache` at the end of every `register_work_unit()`, and tunix
  re-registers every work unit on every sync round, so the cache is invalidated
  by construction. Confirmed: **101 `generated schedule`, 0 `reusing cached
  schedule`.** Costs 7–10 s/step here versus v7's 45–108 s/step.
- **Schedule generation drifts upward** — 7.1 s → 9.9 s, **+39%** over 100
  steps, while transfer stays flat at ~38 s. Same signature and same relative
  magnitude as v7's 45 s → 108 s (+140%), consistent with its diagnosis: the
  `self._active_transfers[req_id] = plan` entries written at
  `raiden_controller.py:2051/2073/2939` are never popped, unlike the sibling
  `_active_tasks` / `_task_units` cleaned up at 1450-1451.

Total cost of the drift over this run is ~3 s/step by the end, ~0.1 h of the
2.25 h. It is worth fixing in Raiden — it is ~4% here and ~18% at v7's fan-out —
but it is not the first thing to address on this topology. **Reported, not fixed:** it
is in Raiden, outside both checkouts, and needs an owner who can confirm the
plan is genuinely dead once the transfer completes.

### 3.3 Where the remaining time is, on this topology

The 37.7 s transfer is now the single largest component of the run — 52% of a
step, 1.05 h of the 2.25 h. 180M blocks over ~70 GB is ~390 bytes per block, so
this is overhead-bound, not bandwidth-bound. Coarsening `group_size` or the
tiling is the untested lever with the most headroom.

After that, `max_staleness > 0` would hide the ~12 s of rollout behind the 60 s
of sync. At this topology that is worth at most 16%, against 60% at v7's shape —
and it changes strictly on-policy GRPO into slightly off-policy GRPO, which is a
semantic change to what is being reproduced. Not done; flagged for a human.

Scaling the trainer is not a lever. v7 §5.2 measured `2x2x4`/FSDP=16 at 345 s of
weight sync against `2x2x2`/FSDP=8 at 233–258 s: more FSDP shards means more,
smaller per-shard transfers, and per-shard overhead swamps the extra egress.
8 chips is also the memory floor — full-parameter Adam on 35B needs ~560 GB of
params + grads + fp32 moments + master weights against 95 GB × 8 = 760 GB.

### 3.4 Why this is 5.6× faster than v7, and why that is not an improvement

The honest comparison:

| | v7 | maz-q35-3 |
|---|---|---|
| Rollout | 8 × `2x2x1` = 32 chips | **1 × `2x2x1` = 4 chips** |
| Sampler | `inprocess_vllm` | `vllm` |
| Rollouts per step | 16 prompts × 16 gens = **256** | 4 × 8 = **32** |
| `MAX_RESPONSE_LENGTH` | 1024 | 512 |
| Raiden fan-out | 1 → 8 | **1 → 1** |
| Weight sync | 235 s (56% of step) | **60.2 s (80% of step)** |
| — of which transfer | 176 s | 37.7 s |
| — of which schedule gen | 45 → 108 s | 7.1 → 9.9 s |
| Mean step | 423.0 s | **75.4 s** |
| 100 steps | 11.75 h | **2.25 h** |
| Mean `reward_mean` | 0.9147 | 0.7342 |
| Steps with zero loss | 40 / 100 | **13 / 100** |

**The speedup is fan-out, not efficiency.** Per trajectory this run is *slower*:
75.4 s / 32 rollouts = 2.36 s per rollout, against v7's 423.0 / 256 = 1.65 s.
The wall clock improved because each step does one eighth of the work, and the
dominant per-step cost — weight sync — shrank with the number of destinations
rather than with the amount of data.

So "100 steps in ~2 hours" is reachable, but at a batch size of 32. Whether 100
such steps are worth more or less than 12 of v7's is a training question, not an
infrastructure one. What the two runs jointly establish is that **the cost model
is fan-out × model size, essentially independent of batch shape** — which is
also why §3.3 puts transfer coarsening ahead of every batch-shape knob.

---

## 4. Sequence packing

`MAX_SEQ_TOKEN_PER_TPU=4096` against `MAX_PROMPT_LENGTH=512` +
`MAX_RESPONSE_LENGTH=512`.

Left at its floor, packing is a no-op **by construction**:
`batch_assembly.validate_packing_budget` requires
`budget >= max_prompt + max_response`, and stock `maxtext_utils` pins
`max_target_length` to exactly that sum. The smallest legal budget therefore
holds exactly one maximal trajectory. `tunix-0004` lets `max_seq_token_per_tpu`
raise `max_target_length` instead:

```python
# tunix/utils/maxtext_utils.py
max_target_length = max(
    max_prompt_length + max_response_length, max_seq_token_per_tpu
)
```

That value also has to *reach* the trainer. The launcher passed
`MAX_SEQ_TOKEN_PER_TPU` to the orchestrator only; `tunix-0004` adds
`--max_seq_token_per_tpu` to `run_trainer_node.py` and the launcher's trainer
command. If the orchestrator packs to one budget and the trainer's
`max_target_length` is another, the rows do not fit.

Confirm from both ends:

```
[Orchestrator] Packed 32 trajectories into microbatch: [...]      # x100, every step
[TrainerNode]  ... max_target_length=4096 ...
```

All 32 trajectories of a step land in **one** microbatch of 8 rows × 4096 tokens
— 4 trajectories per row, one row per FSDP shard. That `Packed 32` line appears
exactly 100 times, once per step, and `train/orchestrator/num_microbatches` is 1
in W&B.

`max_segments_per_packed_row` is deliberately left unset. A lower cap would
produce more than 8 rows, which the mesh cannot place.

`TRAIN_MICRO_BATCH_SIZE` must equal `TRAINER_MESH_FSDP × TRAINER_MESH_DP` so
MaxText sees `per_device_batch_size = 1.0`. Verified in the trainer's argv.

---

## 5. Topology, and the three settings that look arbitrary

| Role | Slice | Chips | Mesh |
|---|---|---|---|
| Trainer | `tpuv5p:2x2x2` (Pathways) | 8 | FSDP=8, TP=1 |
| Rollout | 1 × `tpuv5p:2x2x1` | 4 | TP=2 × DP=2 |

**Rollout uses TP, not FSDP.** FSDP on the rollout all-gathers the weights on
every decode step; TP leaves them sharded and splits the matmuls instead.

**Rollout TP must be 2, not 4.** At TP=4, `maxtext_utils` replicates
`base_num_kv_heads` from 2 to 4 — which MaxText then rejects outright, because
the `qwen3.5-35b-a3b` yml sets it too — and `gmm_v2` pads the MoE MLP dim from
512 to 1024, because 512/4 is not a multiple of 2×128. The experts are ~32B of
this 35B model, so that padding roughly doubles what the rollout must hold. At
TP=2, 512/2 = 256 is already aligned and `num_kv_heads` is untouched. The
remaining 2 chips go to data parallelism, which PR 2229 plumbs through to the
vLLM engine as `data_parallel_size`.

**`PREFUSE_MOE_WEIGHTS=true` applies to the rollout only.** This is v7's bug 1, and
of everything in `submit.sh` it is the setting most likely to produce a run that
looks healthy and is wrong. The same flag name
selects two *different* physical layouts: `[gate|up]` fusion is a global
concatenation on the trainer and a per-shard interleave on the rollout. Fusing
on both sides produces a permuted MoE weight — and Raiden's `checksums()`
returns permutation-invariant per-tensor abs-sums, so **weight-sync verification
still passes while the model emits unrelated digits instead of an answer.** The launcher passes
`--prefuse_moe_weights` in `start_rollout` and nowhere else, and
`run_trainer_node` defaults it to `False`, so the default arrangement is
correct — but verify it from the shape rather than the flag:

```bash
kubectl --context bodaborg-adc logs <train-pod> -c main --tail=-1 \
  | grep -oE 'wi_0 shape=\([0-9, ]+\)' | head -1
# want (256, 10, 2048, 512). A trailing 1024 means the trainer fused too.
```

`maz-q35-3` logs `wi_0 shape=(256, 10, 2048, 512)` on the trainer and
`prefuse_moe_weights=True` on the rollout, which is the correct pairing.

**Trainer pod memory.** The trainer's `proc` pod runs three containers on one
`n2d-standard-64` (~245 GiB allocatable): `pathways-rm` at 16G, `pathways-proxy`,
and the trainer's own container. A container with only `limits` gets
`requests = limits`, so the three must sum under the node. PR 2228's own default
proxy limit of 190G does not (190 + 16 + 60 > 245). This run uses:

```bash
export PATHWAYS_PROXY_MEMORY_LIMIT=120G     # PR 2228 default is 190G
export USER_CONTAINER_MEMORY=60G            # PR 2228 default
export USER_CONTAINER_MEMORY_LIMIT=70G      # PR 2228 default
export PATHWAYS_WORKER_MEMORY=165G          # PR 2228 default
```

120G still holds the ~70 GB Raiden stages device-to-host. It is **not** implicated
in the §6 OOM, which killed the user container, not the proxy.

---

## 6. Checkpointing is still broken — two separate failures

Run `maz-q35-2` was this exact configuration with
`CHECKPOINT_SAVE_INTERVAL_STEPS=10`. Steps 0–8 ran cleanly at ~70 s with rewards
matching `maz-q35-3` step for step (0.8125, 0.6875, 0.969, 0.719, 0.094, …). It
died writing the first checkpoint.

### 6.1 The trainer's user container is OOM-killed by the save

```
Saving checkpoint at step 10
Scheduling D2H of 106 prioritized jax.Array
... 36 seconds later ...
Memory cgroup out of memory: Killed process 22693 (python) ... anon-rss:68032400kB
```

68,032,400 kB is **64.9 GiB** against the 70G container limit. `SIGKILL`, so
there is no traceback — this is only visible in the node's kernel log and in the
pod's termination reason, not in the application log.

**Yes, PR 2228's fix was in effect, and yes it still crashed.** Specifically:

| PR 2228 component | In effect for `maz-q35-2`? |
|---|---|
| `_MeshBoundTrainer._drain_inflight_checkpoint()`, called from `prepare_weight_sync()` | **Yes** — already in the base image, verified in `run_trainer_node.py:389,416` |
| `_clear_resumed_mid_step`, `_suppress_final_checkpoint`, `_final_checkpoint_would_duplicate`, `_last_saved_train_step` | **Yes** — all in the base image |
| `CKPT_D2H_CONCURRENT_GB` override | **Yes** — via `tunix-0002`; trainer argv shows `checkpoint_storage_device_host_concurrent_gb=8` |
| Lost-device-set exit (`_is_unrecoverable_runtime_error`) | Yes — via `tunix-0002` |
| Memory defaults 60G / 70G / 165G | **Yes** — used verbatim |

PR 2228's drain targets a *concurrency* failure: its own docstring says two
simultaneous host copies of a 35B model (2 × 64.6 GiB) OOM-killed the **Pathways
proxy**. That is not what happened here. The kill was the **user container's**
Python at 64.9 GiB, 36 s into the save, before any weight sync had started —
i.e. **the checkpoint save alone exceeds 70G**, with no overlap for the drain to
prevent. PR 2228 is not wrong; it addresses a different failure, and the 8 GiB
D2H bound from MaxText PR 5234 evidently bounds only the *prioritized*-array
staging concurrency, not the process's total host footprint.

Two things follow, neither attempted here:

- Raising `USER_CONTAINER_MEMORY_LIMIT` above 70G means lowering
  `PATHWAYS_PROXY_MEMORY_LIMIT` below 120G or moving to a larger CPU machine
  than `n2d-standard-64`. The node budget is the real constraint.
- Someone should measure what the save's 64.9 GiB is actually made of. A 35B
  bf16 model is ~70 GB, so a full unbounded host copy is the obvious candidate,
  which would mean the `concurrent_gb` bound is not reaching the path that
  matters.

### 6.2 Restarting after a successful save hits a GCS IAM failure

Once `checkpoints/10/` exists, every subsequent container start dies before
training:

```
google.api_core.exceptions.Forbidden: 403 GET
  .../b/mazumdera-bucket-cloud-tpu-multipod-dev:
  390987599272-compute@developer.gserviceaccount.com
  does not have storage.buckets.get access
```

Path: Orbax `CheckpointManager.__init__` → `_load_checkpoint_infos` → `find_all`
→ `_glob_step_paths` → `is_hierarchical_namespace_enabled` → `client.get_bucket()`.

Object reads and writes work fine — the pods' service account has object-level
access. `get_bucket` is a *bucket-level* metadata call, and it is only reached
once at least one step directory exists, which is why a fresh run never sees it.

Fix is a one-line grant to the pods' service account on whatever bucket
`MAXTEXT_OUTPUT_DIR` points at:

```bash
gcloud storage buckets add-iam-policy-binding gs://<your-bucket> \
  --member=serviceAccount:<project-number>-compute@developer.gserviceaccount.com \
  --role=roles/storage.legacyBucketReader
```

Not applied here, because §6.1 makes checkpointing unusable regardless.

---

## 7. Open issues — reported, not fixed

- **§6.1 The checkpoint save OOM-kills the trainer at 70G** even with PR 2228
  and MaxText PR 5234 both in effect. This is the blocker for checkpointing.
- **§6.2 `storage.buckets.get` is missing** for the pods' service account, so any
  run that *does* save cannot then restart.
- **§3.2 The Raiden plan cache is invalidated every round** and
  `_active_transfers` is never popped. 7 → 10 s/step here; 45 → 108 s/step at
  v7's fan-out. Needs a Raiden owner.
- **`MAX_RESPONSE_LENGTH=512` probably costs reward.** This run's mean
  `reward_mean` is 0.7342 against v7's 0.9147 at 1024. v7 §8 documents that a
  trajectory which exhausts its response budget is marked
  `MAX_CONTEXT_LIMIT_REACHED`, and `trajectory_collect_engine.collect()` then
  skips `_append_final_reward` entirely — so it is never graded and scores a hard
  **0.0**, rather than merely losing format points. Halving the budget should
  produce more such trajectories. The 0.0938 minimum is consistent with this.
  Not isolated: this image does not carry v7's diagnostic overlay
  (`TUNIX_LOG_GENERATIONS`), which is what would show the clipped generations
  before the filter drops them. Worth a 5-step A/B at 512 vs 1024.
- **The run did not visibly learn.** `reward_mean` went 0.6691 (first ten steps)
  → 0.6859 (last ten). At `LEARNING_RATE=2e-7`, batch 32, 100 steps, that is the
  expected result — it is 3200 trajectories of signal against a 35B model. 13 of
  100 steps had `loss` exactly 0.0000, which with `BETA=0` means a GRPO group
  with `std(r) = 0` and therefore zero advantage. That fraction is much better
  than v7's 40/100 (its model was at the GSM8K reward ceiling), but this run is
  still far too short to demonstrate learning. It demonstrates a working
  pipeline.
- **`BETA` and `EPSILON` are exported but never plumbed through** (§1.2). Inert
  here because the defaults match, but `BETA=0.04` would silently do nothing.
  One line in each of two `extra_flags` blocks.
- **`VERIFY_WEIGHTS=true` was left on for all 100 steps.** It did not show up in
  the step time, but its checksums are permutation-invariant abs-sums and
  therefore cannot catch the §5 `prefuse_moe_weights` class of bug. It is not the
  safety net it appears to be.

---

## 8. Metrics: W&B needs `WANDB_ENTITY` and fails silently without it

metrax calls `wandb.init(project=..., name=..., anonymous="allow")` with no
entity (`metrax/logging/wandb_backend.py:58`), and wandb refuses to start a run
when the key's viewer has no `defaultEntity`. tunix catches the exception and
downgrades it to a single INFO line (`sft/metrics_logger.py:199`):

```
[Orchestrator] WandbBackend skipped: entity not specified, and viewer has no default entity
```

The run then proceeds with W&B off. So:

```bash
export WANDB_ENTITY=${WANDB_ENTITY:-google-trellis}
```

This is host-side and needs no image rebuild. To find the right value for a
different key without printing the key:

```bash
curl -s -u "api:${WANDB_API_KEY}" https://api.wandb.ai/graphql \
  -H 'Content-Type: application/json' \
  -d '{"query":"{viewer{username defaultEntity{name} teams{edges{node{name}}}}}"}'
```

Confirm it worked — the absence of this line means W&B is off:

```
wandb: 🚀 View run at https://wandb.ai/google-trellis/trellis-gsm8k/runs/v8t407pk
```

Project `trellis-gsm8k`, run name `maz-q35-<n>-<steps>steps`,
`FLUSH_METRICS_EVERY_N_STEPS=1`.

**Set the priority class.** `PRIORITY_CLASS_NAME=medium` (500) is what the rest
of this cluster uses. Without it the workload sits at priority 0 and anything
else in the cohort evicts it; v7 lost a 100-step attempt at step 43 that way.
Kueue reads the same value for admission, because no JobSet here carries a
`kueue.x-k8s.io/priority-class` label.

**Keep `DEBUG=0`.** `DEBUG=1` turns on httpx wire-level logging, which floods the
orchestrator log and makes the step-time and weight-sync lines unfindable over a
100-step run.
