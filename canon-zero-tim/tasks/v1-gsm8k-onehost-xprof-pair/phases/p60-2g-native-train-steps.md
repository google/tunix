# P60-2G — Native-like train microsteps and warm UI capture

## Status

Local and pinned exact-image PASS on the latest operator tip. Target is not
run; no target launch is authorized for this phase.

## Trigger and baseline

The historical P60-2F Zero-HP target is numerically complete and its full
XPlane passes the eight-plane backward and optimizer-tail gates, but it does
not satisfy the newly requested XProf navigation contract:

- Native's full XPlane contains 17 host `train` events numbered 16..32. The
  16 real accumulation iterations are 16..31; 32 is a terminal iterator
  probe. The last real iteration owns the optimizer update.
- The P60-2F Zero-HP full XPlane contains one `train(step_num=1)` spanning the
  entire 62.66-second captured update. Its first reverse/reduce transaction is
  dominated by first-use compiler work, including a roughly 20-second
  `jit_reduce_local` CompileAndLoad event.
- The exported Zero-HP trace JSON contains about one million events and stops
  before `loss_pullback`, all reverse transactions, and `optimizer_commit`,
  even though those spans are present in the full XPlane. The historical
  full-XPlane claim remains valid; the UI-readability claim does not.
- Update 0 changes model/optimizer sharding at commit. Update 1 therefore
  still pays first-use compilation for the stable post-commit identity;
  update 2 is the first warm capture candidate.

Under the P60-2G criterion, the historical result is
`NUMERICAL/FULL-XPLANE PASS / NATIVE-LIKE UI FAIL / PERFORMANCE INCONCLUSIVE`.
Profile-wall time and the input-mismatched Native/Zero pair remain invalid for
a causal performance ratio.

## Contract

When and only when the signed `zero-hp` XProf arm and
`CANON_XPROF_LABELS=1` are active:

1. Keep one `zero_tim_update(update_step=2)` parent around the real update.
2. Keep forward and loss spans truthful to the Zero-HP schedule; do not
   pretend they occur inside Native's per-example train loop.
3. Wrap each real reverse/reduce/accumulate transaction in one Native API
   `StepTraceAnnotation("train", step_num=update_step * 16 + micro_step)`.
   For captured update 2 this is exactly `train_32..train_47`.
4. Keep the final `train_47` open through the real optimizer commit. Do not
   emit a synthetic terminal `train_48`.
5. Remove the crossing aggregate `reverse_groups` annotation from this arm;
   each `train_N` directly owns its matching `reverse_group` transaction.
6. Capture update 2 to 3 with `skip_steps=2`, `steps=1`, three total updates,
   and low-density `TRACE_ONLY_XLA`. No JIT, shard map, reduction, numerical
   operation, synchronization, or semantic Perfetto vocabulary may change.
7. Keep every raw XProf regular file. Record logical byte sizes in an
   immutable receipt, warn above 1.2 GB, and fail the arm above 1.5 GB. Do not
   truncate or delete an oversized capture. Directly include raw XProf and the
   semantic Perfetto in the final SHA ledger.

All other workloads retain their current annotation behavior. Labels absent,
empty, or `0` remain an exact no-op.

## Mechanical gates

- Source/API positives and fail-closed lifecycle negatives prove 16 sequential
  transactions, exact step numbers 32..47, and optimizer ownership by the
  last train span.
- The full-XPlane gate requires all hierarchy spans on one `/host:CPU`
  `python3` track, 8/8 non-empty device Steps rows, all five backward
  families, scaled-step exactly 16 and commit exactly one per plane, and
  decode absent.
- The full-XPlane warm gate rejects `backend_compile_and_load`,
  `PJRT_Client_Compile`, or `TpuCompiler::Compile` inside the captured update.
- A separate streaming trace-JSON gate requires all 16 visible `train` spans,
  all 16 visible reverse transactions, and the optimizer tail contained by
  `train_47` and the update. Full-XPlane completeness cannot substitute for
  this UI carrier gate.
- A separate size census writes `xprof_size_receipt.json`, recomputes the
  current regular-file set in the arm classifier, accepts `PASS` at or below
  1.2 GB and `WARN` only through 1.5 GB, and returns RED above the exact
  `1,500,000,000`-byte hard maximum. Raw XProf tampering after manifest
  creation is a SHA-ledger negative.
- The classifier requires capture `2->3`, profiled work step 2, all runtime
  and alignment gates, hierarchy/module/trace censuses, and the immutable
  evidence ledger.

## Exit boundary

Local completion requires P60, P59, aggregate V1/P64, complete flag audit,
branch preflight, syntax, `git diff --check`, synchronization-token audit, and
the pinned exact-image ladder. These gates do not certify TPU behavior.

Target PASS requires a separately authorized fresh clean committed-tree
Zero-HP one-host run satisfying 3/3 updates, 51/51 alignment, train steps
32..47, complete same-track metadata and optimizer containment, 8/8 backward
planes with scaled-step x16 plus commit x1, decode absent, zero captured
compiler events, complete UI trace JSON, XProf logical bytes at or below
1,500,000,000 with a matching receipt, and a valid SHA ledger that directly
covers raw XProf and semantic Perfetto. No Native rerun is required unless a
new matched performance comparison is explicitly authorized.

## Local result

- Focused P60 suite: 13/13 PASS; document set 15/15 PASS. The added controls
  cover size PASS, soft WARN, hard RED, stale receipt, and post-manifest raw
  XProf tamper.
- P59 suite: 37/37 PASS. Aggregate V1/P64 suite: 67/67 PASS.
- Complete flag audit: declared/actual/unique 378/378/378,
  `FLAG_AUDIT_PASS`. Branch preflight, Python/shell syntax, secret scan,
  no-new-sync token audit, and `git diff --check` PASS.
- Pinned image
  `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`
  completed the full aggregate ladder and the focused P60 probe on CPU with
  zero TPU chips. The exact receipt is:

```text
P60_XPROF_ANNOTATION_API_PASS train_steps=32..47 micro_steps=0..15 last_accumulate=15 optimizer_update=2 optimizer_owned_by_last=1 compiler_events=0 trace_events=166 metadata=integer host_plane=/host:CPU host_line=python3 xplane=1 trace=1
```

- The fixed size census replayed the immutable P60-2F target as
  `V1_GSM8K_XPROF_SIZE_CENSUS_GREEN status=PASS
  xprof_bytes=802001091 ... files=2 xplanes=1 traces=1`, matching its
  768,320,714-byte XPlane and 33,680,377-byte trace JSON. This is a tool replay,
  not a P60-2G target run.

- Immutable P60-2F trace JSON negative: 1,000,448 events, `train=1`,
  `forward_group=16`, and loss/reverse/accumulator/optimizer all zero; the new
  UI census returns RED.
- Immutable P60-2F full-XPlane negative preserves all 16 old transactions and
  8/8 non-empty Steps rows, then returns RED for train cardinality/step range,
  old metadata, and exactly three contained events for each compiler family:
  `backend_compile_and_load`, `PJRT_Client_Compile`, and
  `TpuCompiler::Compile`.

Verdict: `LOCAL/EXACT-IMAGE PASS / TARGET NOT RUN`. These gates prove
instrumentation structure and fail-closed evidence handling, not fresh v5p
timing or target trace retention.
