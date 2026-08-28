# P57.1c — Perf v2 step-boundary isolation

- Status: complete — G4 pass; full target not run
- Source CL: `ec9884e9` (runtime repair, direct gates, and bounded carrier)
- Incident source: `799a0bd1ed5ecfd7a2f6e42eeaced82886fec76c`
- Incident evidence:
  `evidence/f45w15_timeline_tracer_incident/`

## Finding

Wave 15 P45 completed update 0 with strict A/B/C alignment and a finite AdamW
commit, then failed during Step 1 rollout.  The decisive chronology is:

1. the learner queued the next prompt batch;
2. the producer opened a Step 1 `rollout` Perf v2 host span;
3. the main thread exported Step 0 metrics;
4. `Timeline.commit_step()` purged that still-open span and cleared its stack;
5. the rollout context later exited and `stop_span()` raised
   `no more spans to end`.

The raw log explicitly records `Purging uncompleted span 'rollout'` before the
underflow on the same host timeline.  This is an observational step-boundary
bug, not evidence of a Zero-TIM, backward, loss, gradient, optimizer, TPU
worker, or Pathways failure.  The incident report's broader explanation in
terms of arbitrary cross-thread stack corruption is superseded by this exact
export-versus-next-rollout chronology.

## Objective

Keep Perf v2 and the one-step semantic Perfetto artifact without allowing an
observer to overlap two logical training steps or terminate training.  Preserve
all numerical programs, rollout inputs, update order, and Zero-TIM gates.

## Pre-registered design

### Application boundary

The next dataset batch may be loaded under the completed step's data-loading
span, but it must not be put on the producer queue until the completed step's
Perf v2 timeline has committed.  Only the short commit/export operation is on
the critical path.  Host-memory cleanup and ordinary metric buffering remain
eligible to overlap with the newly started rollout.

Required ordering:

~~~text
weight sync -> load next batch -> perf_v2 export/commit -> queue next batch
~~~

Forbidden ordering:

~~~text
queue next batch -> producer opens rollout -> perf_v2 export/commit
~~~

### Tracer defense

`Timeline.commit_step()` must not destructively purge an active host span and
leave its context manager with an invalid stack.  A step commit attempted with
active host spans must fail before mutation with a diagnostic that names the
timeline and active span count.  The caller may defer the observational export;
it must not silently accept a corrupted trace.  Do not catch and discard the
existing `stop_span()` underflow.

### Performance boundary

Do not serialize host GC ahead of the next rollout.  The repair may delay the
queue only by the Perf v2 commit/export itself.  A later snapshot/writer split
is optional and requires separate measurement; it is not part of this
correctness repair.

## Gates

### G1 — deterministic incident reproduction

A host test opens a rollout span in a worker, attempts a step commit while it
is active, and proves the old destructive behavior/underflow signature.  The
repaired behavior must reject or defer before mutation, then allow the span to
close and a later commit to succeed.

### G2 — learner ordering contract

A focused source/behavior contract proves the completed-step Perf v2 export
precedes `_put_prompts_to_queue` on every next-batch branch, including the
weight-sync path used by P45/M15.  A reversed-order mutation must fail.

### G3 — host regression suites

Run the focused tracer tests, the focused P57 ordering/contract tests, the full
P57 CPU suite, flag audit, Python syntax, and `git diff --check`.  Record exact
counts and terminal markers.

### G4 — one-host target admission (requires separate TPU approval)

Run at least two FrozenLake optimizer commits with production-like concurrent
trajectory collection.  Require strict alignment PASS with zero failures,
finite nonzero updates, no `Purging uncompleted span`, no span underflow, and a
readable target-step semantic Perfetto artifact.  Construction gates do not
promote this claim.

### G5 — full target

Fresh P45 and M15 full identities may launch only after their separate user
approval.  The first two steps must satisfy the G4 receipts before either run
is treated as having exercised the repair.  Full campaign acceptance remains
the owning P57 contract.

## Rollback

The application-order change is isolated to observer-enabled step finalization;
when Perf v2 is absent it preserves the Noop path.  If it fails a gate, revert
this phase's code hunks only.  As an emergency operational fallback, leaving
`CANON_PERF_TRACE_DIR` empty restores the existing `NoopTracer`, but that
removes the semantic Perfetto artifact and requires an explicit profile and
postflight decision; it is not silently substituted in this phase.

## Result log

- 2026-08-28: phase opened from sealed Wave 15 evidence.
- G1 PASS: the new learner-order contract failed 1/2 against the old ordering
  with `Perf v2 export must precede every next-step queue`; after the repair it
  passes 2/2.  The sealed target log remains the real incident reproduction.
- G2 PASS: all three full-train next-batch branches now resolve the batch first,
  perform one common `perf_v2.export()`, publish the producer input, and only
  then enter P45 host-memory cleanup.  A reversed-order negative is rejected.
- G3 PASS: P57 CPU 172/172; pinned-image timeline 17/17 and tracer 34/34;
  complete P45 exact-image terminal markers
  `P57_PERF_V2_STEP_BOUNDARY_PASS` and `P45_EXACT_IMAGE_CPU_PASS`; V1 Phase4
  90/90; `FLAG_AUDIT_PASS` 395/395; Python syntax and `git diff --check` pass.
- G4 PASS on approved one-host v5p with pinned image
  `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`.
  Attempts `r1` through `r6` are preserved and remain non-admitting: `r1`
  found the TPU occupied, `r2` rejected a wrong concurrency, `r3`/`r4`
  exposed the local mesh/sharding carrier gap, `r5` completed a real reverse
  before exposing an unbound legacy workload identity, and `r6` was rejected
  during the r5 container-exit race before a TPU program started.  Fresh `r7`
  at
  `/mnt/disks/tunix-data/logp_probe_1host/p57_perf_v2_p57c_g4_cb38cf67_r7`
  completed 3/3 AdamW transactions and 12/12 strict alignment rows with zero
  differing bytes at every A/B/C boundary.  All three updates had finite,
  nonzero commit gradients and changed about 6.94 billion parameter elements.
  Step 1 rollout crossed the exact Wave 15 failure boundary with no purge,
  active-span rejection, or `stop_span` underflow.
- G4 performance: cold Step 0 was 419.70s, including 122.593s reverse and
  58.499s optimizer transaction.  Steady Steps 1/2 were 36.93s/35.98s
  (mean 36.455s); reverse was 18.449s/17.726s (mean 18.088s), and optimizer
  was 0.624s/0.420s (mean 0.522s).  The following weight-sync operations were
  35.646s/39.398s, so this small one-host carrier identifies weight sync as a
  separate cycle-time cost; it is not an at-scale throughput estimate.
- G4 semantic adjudication: the original `r7` classifier correctly remains
  preserved as RED because its carrier incorrectly required
  `reference_inference`.  FrozenLake is explicitly beta zero and does not run
  that operation.  The repaired classifier requires the event to be absent
  under a signed `disabled` contract while still requiring data loading,
  rollout, advantage, PEFT train, and weight sync.  Add-only artifacts
  `semantic.beta0.json` and `classification.beta0.json` pass; their SHA-256
  values are `02c51f1d7a8abc8a01bf27feb99372a0b2ce4a779f892d50a48c15ec522d26f1`
  and `a1544f4d2f1094a71924cf63c753ff272fe3f3d0fb497d73e07ab01541e734b2`.
  Runtime learner/tracer inputs were unchanged, so no target rerun was needed
  for this classifier-only correction.
- G5 NOT RUN: no fresh P45/M15 target manifest or JobSet was rendered or
  launched.
