# `TRAINER_STEP_TIMING`: trainer-side timeline of one optimizer step

The orchestrator log records when it *sent* each `fwd_bwd` (`Executing
train_step ...`) and when the step's weight sync started, but not when the
trainer actually ran anything: `fwd_bwd` and `update` RPCs return at JAX
dispatch, before the device work completes, and nothing is logged on return.
Tools that read the orchestrator log (e.g. the Trellis RL perf explorer) could
only infer trainer activity from the spacing of later lines.

This log line closes that gap. The trainer worker keeps a `StepTimer`
(`tunix/experimental/worker/trainer_worker.py`) and the orchestrator fetches
its snapshot once per step, after the weight-sync source has staged (so the
update is provably complete), and logs it on **one line**:

```
TRAINER_STEP_TIMING step=<N> policy_version=<v> rtt_s=<r> timing=<json>
```

| field | meaning |
|---|---|
| `step` | orchestrator step index (the `>>> ... step N starting` / `COMMIT step=N` step) |
| `policy_version` | the orchestrator's policy version after this step's update |
| `rtt_s` | round trip of the `get_step_timing` RPC, measured by the orchestrator |
| `timing` | compact JSON, schema below |

## `timing` schema (version 1)

```json
{
  "v": 1,
  "train_step": 12,
  "mb": [
    {"i": 0, "dispatch_begin_ago_s": 71.412, "dispatch_end_ago_s": 70.980, "ready_ago_s": 56.301, "via": "fwd_bwd_staged"},
    {"i": 1, "dispatch_begin_ago_s": 56.250, "dispatch_end_ago_s": 55.903, "ready_ago_s": 41.077, "via": "fwd_bwd"}
  ],
  "update": {"dispatch_begin_ago_s": 40.900, "dispatch_end_ago_s": 40.611, "ready_ago_s": 27.455, "via": "apply_deferred_update"},
  "prepare": {"recv_ago_s": 6.120, "done_ago_s": 0.834}
}
```

* `train_step` – the trainer's own optimizer-step counter returned by `update`.
* `mb[i]` – the i-th microbatch of this step, in the order it was dispatched,
  whichever call ran it: a `fwd_bwd` RPC, or `fwd_bwd_staged` for a
  microbatch received ahead by `stage_fwd_bwd` (the pipelined step boundary).
  `dispatch_begin` / `dispatch_end` bracket the worker's handler (which returns
  at dispatch); `ready` is when the microbatch's outputs became ready on
  device, i.e. when its forward/backward actually finished. A `stage_fwd_bwd`
  itself is not an entry: it only receives the data, and its microbatch is
  recorded when `fwd_bwd_staged` runs it, in the step it belongs to.
* `update` – the same three instants for the step's optimizer update, run by
  the `update` RPC, inside the last microbatch's call (`fwd_bwd` /
  `fwd_bwd_staged` with `apply_optimizer=True`), or by
  `apply_deferred_update` after a `defer_update=True` last microbatch; `ready`
  is when the optimizer step finished on device.
* `via` (on each `mb[i]` and on `update`) – the call that ran it: `fwd_bwd`,
  `fwd_bwd_staged`, `update` or `apply_deferred_update` (a fused update is
  recorded under the microbatch call that ran it). Added to schema version 1
  without a version bump; records from earlier builds lack it.
* `prepare` – when this round's `prepare_weight_sync` was received and
  finished on the worker. It reads the updated weights, so `recv` is never
  before `update.ready`. `null` when the step took no weight sync.
* Every `*_ago_s` is **seconds before the worker handled `get_step_timing`**,
  measured on the trainer host's monotonic clock. `ready_ago_s` is `null` when
  the readiness was not (yet) observed, e.g. a trainer backend that does not
  publish readiness tokens.

### Anchoring on the orchestrator clock

No cross-host clock agreement is assumed. For a line logged at `t_line`:

```
t_event ≈ t_line − rtt_s / 2 − <field>_ago_s
```

The error is bounded by the RPC's one-way latency asymmetry (milliseconds).

### Companion lines

The engine also logs the RPC returns, which were previously silent:

```
train_step fwd_bwd RPC returned on actor worker in 14.812s (request_id=train_req_ab12cd34)
train_step update RPC returned on actor worker in 0.412s (train_step=12)
```

These are orchestrator-clock instants of the *dispatch* acknowledgements and
complement the device-side instants above.

### When the record is short

The worker counts the microbatches it dispatched in the step (and whether the
update ran) apart from the timer. When `get_step_timing` would return fewer
`mb` entries than microbatches ran, or no `update` after one ran, the trainer
log carries a warning naming the call(s) whose work is missing, e.g.
`TRAINER_STEP_TIMING: the step record lists 0 of the 8 micro-batches this
worker ran; not recorded, by entry point: fwd_bwd x7, fwd_bwd_staged x1.`
<!-- TODO: verify against a real trainer log line once a run hits it -->
The orchestrator still logs the line as received and, when `mb` is empty or
`update` is `null`, follows it with a warning
(`TRAINER_STEP_TIMING step=<N> is incomplete (...)`) that points to the
trainer log, so an empty record is never read as "no microbatches ran".

Builds that carried the first cherry-pick of this change onto the
deferred-update / staged-boundary branch re-created the timer on every
worker call, so their records hold only `prepare` (`"mb": []`,
`"update": null`, `"train_step": null`) on every path.

## How readiness is observed without blocking

After dispatching, the worker spawns a daemon thread that calls
`jax.block_until_ready` on a token array the trainer already produced (the
microbatch loss for `fwd_bwd`, the gradient norm for `update`; all outputs of
one jitted executable become ready together) and stamps the time. The serving
thread never waits. A trainer backend opts in by exposing
`last_fwd_bwd_token` / `last_update_token` attributes; `PeftTrainer` v2 does.
Backends without them still report the dispatch instants.

## Reading it

* Per-microbatch device time: `mb[i].ready` − max(`mb[i].dispatch_end`,
  `mb[i-1].ready`) (the device runs microbatches back to back).
* Trainer idle inside the step: gaps between `mb[i].ready` and
  `mb[i+1].dispatch_begin` (waiting for the orchestrator to send the next one).
* Update cost: `update.ready` − `update.dispatch_begin` (includes draining the
  last microbatch).
* Wait on the previous weight-sync round (background sync): `prepare.recv` −
  `update.ready`, when the orchestrator was ready to sync but the coordinator
  was still running the previous round.
