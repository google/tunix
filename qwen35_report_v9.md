# Qwen3.5-35B-A3B distributed GRPO on TPU v5p — run 6 failure report (v9)

Companion to `qwen35_report_v8.md`, which measured a completed 100-step run on a
**narrow** topology (one 4-chip rollout replica, 32 rollouts/step, 2 h 15 m).
This report covers `maz-q35-6`, the first attempt to move that pipeline onto
v7's **wide** topology — 8 rollout replicas, 32 chips, 256 rollouts/step,
1024-token responses.

**It failed before step 0, in the initial weight sync.** No training step ran,
so there are no step times, no rewards and no W&B run to report. What follows is
the failure analysis and the two things the attempt did establish.

Everything below was measured on cluster `bodaborg-v5p-nap`, project
`cloud-tpu-shared-capacity`, region `europe-west4`, namespace `default`, on
2026-09-16.

---

## 0. Status

**`maz-q35-6` never completed round 0 of weight sync.** All 12 pods came up
clean, all 8 rollout destinations bound their weights and entered
`pre_weight_sync`, and then nothing moved for eleven minutes until the
orchestrator raised on a socket deadline inside the `tpu_sync` wheel.

| | |
|---|---|
| Steps completed | **0 of 100** |
| Failure point | weight sync round 0, `policy_version=0`, before step 0 |
| Time from `pre_weight_sync` to raise | **650 s** |
| Orchestrator's own accounting | `Weight sync finished in 684.18 seconds` |
| Destinations that received any data | **0 of 8** |
| Proximate error | `TimeoutError: timed out` from `tpu_sync/rpc/raiden_controller.py:669` |
| Comparable baseline (`maz-q35-5`, 1 destination, default K) | sync in **86.44 s**, reached step 0 |
| W&B | none — the run never emitted a metric |

The single variable this run introduced over `maz-q35-5` that touches the
transfer path is `RAIDEN_BROADCAST_K=1`. §2.3 shows the traceback proves that
variable took effect and selected the tree-broadcast code path.

**What is not yet known** is whether the hang is specific to `fanout_k=1` or
affects `_execute_slice_broadcast` at any K. `maz-q35-7` is running the same
configuration with `RAIDEN_BROADCAST_K=4` to separate those; see §6.

---

## 1. What changed from v8

### 1.1 Configuration

v8 measured `maz-q35-3`. This run is the same pipeline on v7's topology. The
deltas, all in `docker/maz-q35/submit.sh`:

| | v8 / `maz-q35-3` | `maz-q35-6` |
|---|---|---|
| Rollout | 1 × `tpuv5p:2x2x1` (4 chips) | **8 × `tpuv5p:2x2x1` (32 chips)** |
| Rollout JobSets | `maz-q35-3-roll` | **`maz-q35-6-roll-0` … `-roll-7`** |
| Raiden fan-out | 1 → 1 | **1 → 8** |
| `RAIDEN_BROADCAST_K` | unset (default 64) | **1** |
| `BATCH_SIZE` × `NUM_GENERATIONS` | 4 × 8 = 32 | **16 × 16 = 256** |
| `MINI_BATCH_SIZE` | 32 | **256** |
| `MAX_RESPONSE_LENGTH` | 512 | **1024** |
| Micro-batches per step | 4 | **≥16** |
| `DEBUG` | 0 | **1** |
| Raiden wheel | `tpu_raiden_jax 0.0.1.dev20260908164139` | **`tpu_sync_jax 0.0.1.dev20260914193202`** |
| Pathways server | `unsanitized_server:raiden_20260904` | **`unsanitized_server:raiden_20260914_fix`** |
| Pathways proxy | `unsanitized_proxy_server:raiden_20260904` | **`unsanitized_proxy_server:raiden_20260914`** |

Unchanged and matching v7: trainer `tpuv5p:2x2x2` FSDP=8, rollout mesh
dp=2 × tp=2, `MAX_PROMPT_LENGTH=512`, `MAX_SEQ_TOKEN_PER_TPU=4096`,
`TRAIN_MICRO_BATCH_SIZE=8`, `LEARNING_RATE=2.0e-7`, `BETA=0`, `EPSILON=0.2`,
`CHECKPOINT_SAVE_INTERVAL_STEPS=0`.

Two deliberate departures from v7, both carried over from v8: `SAMPLER=vllm`
rather than `inprocess_vllm` (v7 §7 reports the plain path cannot weight-sync;
tunix PR 2229 fixes that, and v8 §2 verifies it over 101 syncs), and
`PRIORITY_CLASS_NAME=medium`.

`TRAIN_MICRO_BATCH_SIZE` stays at 8 because it must equal `TRAINER_MESH_FSDP` —
one packed row per FSDP shard per micro-batch. At 512+1024 worst case against a
4096-token row, 2 sequences share a row, so 256 sequences need at least 128 rows
and therefore at least 16 micro-batches per step rather than v8's 4.

### 1.2 The image

```
gcr.io/cloud-tpu-multipod-dev/mazumdera-runner@sha256:37d4070d501a54c8a167ba2287bbe9626726e65224a6019750430b5d2ff9aaf5
```

Same six-patch overlay as v8 (`docker/maz-q35/Dockerfile`), plus one wheel
replacement. The Raiden weight-sync engine is moved from the base image's
2026-09-08 build to the 2026-09-14 one so that it matches the `raiden_20260914`
Pathways pair.

The distribution was renamed across that boundary — `tpu_raiden_jax` →
`tpu_sync_jax` — while keeping the same top-level `tpu_sync` package. pip cannot
know the two are the same thing, so installing over the top would leave the old
build's files wherever the new wheel does not happen to overwrite them, plus two
`dist-info` directories claiming the same paths. The Dockerfile uninstalls by
the old name first:

```dockerfile
ARG TPU_SYNC_WHEEL=tpu_sync_jax-0.0.1.dev20260914193202-cp312-cp312-manylinux_2_31_x86_64.whl
RUN --mount=type=bind,source=${TPU_SYNC_WHEEL},target=/tmp/${TPU_SYNC_WHEEL} \
    pip uninstall -y tpu_raiden_jax \
 && pip install --no-deps --force-reinstall /tmp/${TPU_SYNC_WHEEL}
```

`--mount=type=bind` rather than `COPY`, so the wheel's 109 MB do not become an
image layer. The wheel is not in the branch — `.gitignore` excludes `*.whl` — so
`build_push.sh` fetches it from
`gs://cloud-tpu-inference-test-datenglin/` into the build context first.

`--no-deps` is deliberate: the wheel pins `jax==0.11.0`, `jaxlib==0.11.0` and
`libtpu==0.0.47`. The first two already match. The image has `libtpu 0.0.44`,
and libtpu is also the rollout's vLLM runtime, so moving it is not a change this
overlay should make implicitly. §4 reports that 0.0.44 works in practice.

`build_push.sh` asserts the swap in the finished image:

```python
assert _md.version("tpu_sync_jax") == "0.0.1.dev20260914193202"
try:
  _md.version("tpu_raiden_jax")
except _md.PackageNotFoundError:
  pass
else:
  raise AssertionError("the superseded tpu_raiden_jax distribution is still installed")
from tpu_sync.frameworks.jax import weight_synchronizer_ffi  # noqa: F401
```

---

## 2. The failure

### 2.1 Timeline

```
18:38     12 pods created; node auto-provisioning adds ~10 TPU hosts
18:49:14  rollouts begin "Eagerly warming up Raiden weight sync..."
18:50:54  roll-7 registers with the discovery server
18:51:27  roll-0 registers (last of the eight)
18:53:10  all 8 rollouts bind:
            "skipped 150 non-weight leaves (KV cache) when binding"
18:53:44  all 8 rollouts:
            "Executing pre_weight_sync (policy_version=0, free_kv_cache=True)"
18:53:44  trainer's last log line, then silence
  ...     no destination logs anything for the next 11 minutes
19:04:34  "round 0: source staging deliberately NOT released;
           a timed-out transfer may still be reading it"
19:04:34  "Weight sync finished in 684.18 seconds"
19:04:34  FATAL: StandardRLProgram execution failed
```

The 684.18 s the orchestrator reports is measured from the start of the round at
~18:53:10; the 650 s from `pre_weight_sync` to the raise is the part spent
inside the transfer itself.

Every destination stopped at exactly the same place. None of them logged
receiving a single block.

### 2.2 The traceback

```
File "/app/tunix/experimental/weight_sync/raiden_handler.py", line 407, in transfer
  loop.run_until_complete(future.wait())
File ".../tpu_sync/rpc/raiden_controller.py", line 1015, in wait
  await self._transfer_task
File ".../tpu_sync/rpc/raiden_controller.py", line 3301, in _execute_transfer
  await asyncio.gather(*push_tasks)
File ".../tpu_sync/rpc/raiden_controller.py", line 2755, in _execute_slice_broadcast
  raise fut.exception()
File ".../tpu_sync/rpc/raiden_controller.py", line 2731, in _run_single_transfer
  await self.worker_rpc_client.start_transfer(s_node, plan)
File ".../tpu_sync/rpc/raiden_controller.py", line 735, in start_transfer
  await asyncio.gather(...)
File ".../tpu_sync/rpc/raiden_controller.py", line 740, in _send_and_verify
  resp_bytes = await self._send_rpc(addr, payload)
File ".../tpu_sync/rpc/raiden_controller.py", line 669, in _send_rpc
  return await loop.run_in_executor(...)
TimeoutError: timed out
```

tunix wraps that twice on the way out — `raiden_handler.py:414` raises
`TransferOutcomeUnknownError: transfer wsync-v0-r0: the controller future or
final status could not be observed`, and
`weight_sync_coordinator.py:1207` raises:

```
WeightSyncError: round 0 (req_id wsync-v0-r0, uuid 1): transfer timed out;
destinations NOT aborted and source staging NOT released because the transfer
may still be running; final state unknown_transfer_state
```

### 2.3 What `RAIDEN_BROADCAST_K` does, and that it took effect

`tpu_sync` reads the variable once, at controller construction
(`raiden_controller.py:1413`):

```python
self.broadcast_k = (
    broadcast_k if broadcast_k is not None
    else int(os.environ.get("RAIDEN_BROADCAST_K", "64"))
)
```

It is then used in two places. The threshold, at `:2301`:

```python
is_tree_broadcast = (
    len(unique_dst_units) > 1 and len(unique_dst_units) > self.broadcast_k
)
```

and the fan-out, at `:3292`, passed into `_execute_slice_broadcast(...,
fanout_k=self.broadcast_k, ...)`.

At the default 64, 8 destinations do not exceed K and the group takes the direct
branch — the source pushes to all 8 itself. Below 8 it takes
`_execute_slice_broadcast`, which is a work-stealing loop rather than a fixed
tree: `available_sources` starts as `[src_unit]` (`:2594`), each source may have
at most `fanout_k` pushes in flight (`:2610`, `active_pushes[s] < fanout_k`), and
every destination that completes is promoted into `available_sources` (`:2758`).
So `fanout_k=1` is not a relay chain of depth 8 — it is one concurrent push per
node, which reaches 8 destinations in roughly 3 rounds.

**The `_execute_slice_broadcast` frame in §2.2 is the proof the variable took
effect.** Had K stayed at 64, the stack would have gone through the direct
branch instead.

**The hang is in the first hop.** No destination received anything, so this is
not "the tree was slower than the deadline allowed"; the very first
`start_transfer` never returned.

### 2.4 The 600 s deadline is not settable from outside the image

`raiden_controller.py:664`:

```python
async def _send_rpc(self, addr: str, payload: bytes, timeout: float = 600.0) -> bytes:
```

`start_transfer` (`:735`) calls `_send_and_verify` (`:740`), which calls
`_send_rpc` without passing a timeout, so the 600 s default governs. Nothing in
tunix, the launcher or the environment can change it. The coordinator's own
per-phase deadline is much larger and never fired —
`weight_sync_coordinator.PhaseTimeouts.transfer` is 1800.0 s.

Raising this deadline would not obviously help, since the symptom is a hang
rather than slow progress, but it is worth recording that it cannot currently be
tried without patching the wheel.

### 2.5 The wedge afterwards

When the outcome is unknown the coordinator deliberately declines to clean up:
source staging stays pinned and destinations are not aborted, because a
timed-out transfer may still be reading. That is the right call in isolation,
but the JobSet-level recovery does not match it:

- The **orchestrator** JobSet restarted (`RESTARTS=1`, new pod at 19:14:53).
- The **trainer and all 8 rollout** pods did **not** restart — 32 minutes old,
  0 restarts, still holding pinned staging and still registered to the dead
  orchestrator.
- The new orchestrator sat at `Waiting for workers to register via discovery
  service...` indefinitely.

The run had to be torn down by hand. A failed weight sync therefore costs the
full 40 chips until someone notices.

TODO(tunix): decide whether the JobSet failure policy should restart all three
roles together, or whether workers should re-register with a restarted
orchestrator.

---

## 3. What run 6 does and does not establish

**Establishes:**

1. The wide topology schedules. 8 rollout JobSets plus the Pathways trainer —
   10 TPU hosts, 40 v5p chips — were admitted at `medium` priority and all 12
   pods reached `Running`.
2. All 8 destinations reach `pre_weight_sync` correctly under `SAMPLER=vllm`.
   Binding works at fan-out 8; each replica logs
   `skipped 150 non-weight leaves (KV cache) when binding`.
3. `RAIDEN_BROADCAST_K` is settable per-role and reaches the right process
   (§5).

**Does not establish, because no step ran:**

- Any step time, reward, or packing behaviour at 256 rollouts/step.
- Whether `MAX_RESPONSE_LENGTH=1024` recovers the reward that v8 §7 suspects
  512 was costing.
- Whether the checkpoint-save OOM of v8 §6.1 behaves differently at this
  topology. `CHECKPOINT_SAVE_INTERVAL_STEPS=0` here regardless.

---

## 4. The 2026-09-14 Raiden wheel and Pathways pair do work

`maz-q35-5` ran the same image, the same `tpu_sync_jax 0.0.1.dev20260914193202`
wheel and the same `raiden_20260914` Pathways server and proxy, on v8's narrow
topology — one rollout replica, default `RAIDEN_BROADCAST_K`:

```
18:35:51  Weight sync finished in 86.44 seconds.
18:35:51  Weight synchronization complete (policy_version=0).
18:37:31  >>> Step 0 starting | Policy Version: 0
18:37:57  Executing train_step on actor worker
```

It was stopped deliberately at step 0 to free the trainer slice for run 6, so
there is no steady-state number from it. But it settles the risk flagged in
§1.2: **the wheel's `.so` files load and transfer correctly against the image's
`libtpu 0.0.44` despite the wheel pinning `libtpu==0.0.47`.** No load failure,
no wrong answer at the verification checksum.

86.44 s at 1 destination is the number to compare any fan-out-8 result against,
and it is also a direct like-for-like against the old wheel: on the 2026-09-08
wheel and the `raiden_20260904` Pathways pair, v8 §3.1 measured **83.8 s for
sync #1** and 60.2 s steady-state, on the same topology and the same fan-out.
86.44 s against 83.8 s is a 3% difference on a single first sync that includes
first-time work-unit registration. **The 2026-09-14 wheel is not a
regression at fan-out 1.**

---

## 5. Setting an environment variable per role

`RAIDEN_BROADCAST_K` has to reach the **orchestrator** container and only that
one. tunix constructs exactly one `RaidenController` in the whole job, and it is
there:

```
orchestrator.py:341  create_default_handler(mode=self._weight_sync_mode)
  weight_sync_coordinator.py:167  raiden_handler.RaidenHandler(...)
    raiden_handler.py:474          _RaidenTransport(...)
      raiden_handler.py:149        raiden_controller.RaidenController(port=port, ...)
```

`broadcast_k` is not passed at that call site, so the environment wins. The
trainer and rollout containers never construct a controller and never read the
variable.

`k8s_launcher.sh` only had `TRAINER_EXTRA_ENV`, so two more were added alongside
it:

```bash
export TRAINER_EXTRA_ENV=${TRAINER_EXTRA_ENV:-}
export ORCHESTRATOR_EXTRA_ENV=${ORCHESTRATOR_EXTRA_ENV:-}
export ROLLOUT_EXTRA_ENV=${ROLLOUT_EXTRA_ENV:-}
```

each prepended to its own role's `--worker_startup_command`. `submit.sh` then
sets:

```bash
export ORCHESTRATOR_EXTRA_ENV="RAIDEN_BROADCAST_K=1"
```

This is host-side only and needs no image rebuild. Confirm it landed by
rendering the manifests and reading the orchestrator command:

```bash
DRY_RUN=true bash docker/maz-q35/submit.sh <n> <steps> > /tmp/maz-q35-<n>.yaml
grep -o 'RAIDEN_BROADCAST_K=[0-9]*' /tmp/maz-q35-<n>.yaml
```

It must appear exactly once — in the `-orch` JobSet's command, immediately
before `python -m tunix.experimental.distributed.runtime.main`.

---

## 6. Reproducing, and the next experiment

Build and submit exactly as v8 §1.1, with the run number bumped:

```bash
cd ~/git/tunix
bash docker/maz-q35/build_push.sh q35-0916-v3     # fetches the wheel, builds, asserts, pushes
export WANDB_API_KEY=<your key>                   # deliberately not stored in the repo
DRY_RUN=true bash docker/maz-q35/submit.sh 6 100  # renders all 10 manifests, applies nothing
bash docker/maz-q35/submit.sh 6 100               # submits orch + train + roll-0..7
```

Tearing down needs all ten names — `submit.sh <n> <steps> stop` derives them
from `ROLLOUT_REPLICAS`, so it must be run with the same value the run used:

```bash
kubectl delete jobset maz-q35-6-orch maz-q35-6-train \
  maz-q35-6-roll-{0,1,2,3,4,5,6,7} -n default
```

**The next experiment is `maz-q35-7`: identical, with
`RAIDEN_BROADCAST_K=4`.** 8 destinations still exceed K, so it stays on the
`_execute_slice_broadcast` path, but each node may have four pushes in flight
instead of one. Two outcomes, both informative:

- **Syncs and trains** — `fanout_k=1` was too serial for the 600 s per-hop
  deadline, and the tree-broadcast path itself is sound.
- **Hangs the same way, first hop, ~650 s** — the tree-broadcast path does not
  work at this topology regardless of K, and the next test is dropping
  `ORCHESTRATOR_EXTRA_ENV` entirely to see whether the direct 8-way push works.

---

## 7. Open issues — reported, not fixed

- **§2 Weight sync hangs on the first hop of `_execute_slice_broadcast` with
  `RAIDEN_BROADCAST_K=1` and 8 destinations.** This is the blocker for the wide
  topology. Needs a Raiden owner. Not yet isolated to `fanout_k=1` versus the
  tree path in general; `maz-q35-7` is the discriminating run.
- **§2.4 `_send_rpc`'s 600 s deadline is hardcoded** and `start_transfer` does
  not override it, so per-hop transfer time cannot be extended without patching
  the wheel.
- **§2.5 A failed weight sync wedges the job rather than failing it.** The
  orchestrator restarts, the workers do not, and the restarted orchestrator
  waits for registrations that never come — holding 40 chips until someone
  intervenes.
- **Everything in v8 §7 still stands.** In particular §6.1, the checkpoint save
  OOM-killing the trainer, which has since been reproduced a second time: at the
  raised 120G limit `maz-q35-4` was killed at `anon-rss:116105140kB` = 110.7 GiB,
  59 s into the step-10 save, after `maz-q35-2` was killed at 64.9 GiB against
  70G, 36 s in. Both had `CKPT_D2H_CONCURRENT_GB=8` in force and neither had a
  weight sync running. Each died at roughly 93% of whatever ceiling it was given,
  so the staging is unbounded and raising the limit is not a fix.
- **v8 §6.2's GCS IAM failure is fixed.** `roles/storage.legacyBucketReader` was
  granted to `390987599272-compute@developer.gserviceaccount.com` on
  `gs://mazumdera-bucket-cloud-tpu-multipod-dev`. Any teammate pointing
  `MAXTEXT_OUTPUT_DIR` at a different bucket needs the same grant; the command is
  in v8 §6.2.
- **`DEBUG=1` is on in this configuration**, at the user's request, to log
  sampled trajectories. It also turns on httpx wire-level logging, which floods
  every log with `_cygrpc`, `response_closed` and raw HTTP header lines. Read
  step times with `grep` rather than by scrolling. v8 §8's advice to keep it at 0
  applies whenever the trajectories are not wanted.
