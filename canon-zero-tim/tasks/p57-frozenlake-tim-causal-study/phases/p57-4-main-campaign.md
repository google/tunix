# P57.4 — Paired multi-seed main campaign

## Purpose

Measure capability, stability, and cost under the frozen causal contract.

## Design

- Arms: zero TIM and finite TIM.
- Paired seeds: 42, 43, and 44; identical seed/order/checkpoint pairing across
  arms.
- Primary horizon: 200 updates.
- Checkpoints: every 10 updates.
- Isolated held-out evaluations: updates 0, 20, 50, 100, 150, and 200.
- Evaluation contract: immutable held-out maps, deterministic greedy decoding
  (`temperature=0`), eight identical-policy generations per map (the minimum
  global row count divisible by the DP8 trainer-rescore axis), fixed map order,
  and no prefix state shared with training. Capability statistics are map-level;
  duplicate deterministic generations are retained as a coverage check, not
  treated as independent samples.
- Each expensive arm launch requires explicit user approval.

## Continuation rule

A 450-update extension is permitted only if, before inspecting the arm gap, at
least one arm improves by more than one percentage point from update 150 to
200 and no invalidating gate fires. The rule uses learning slope, not the sign
or magnitude of the treatment effect. Extension still requires user approval
and applies symmetrically to both arms and every seed.

## Run receipts

Every run must persist:

- source/image/model/recipe digests and intent diff;
- train transaction, checkpoint, and isolated-evaluation receipts;
- per-step A-B/B-C dose summaries and zero exactness;
- solve/reward, effective/mixed groups, context/turn/completion lengths,
  truncation and invalid actions;
- importance ratios, clipping, gradient/update norms, nonfinite counters;
- wall time, sampled tokens, TPU topology/HBM, and failure/restart history.

Cluster recovery may restart a run, but scientific pairing must resume from the
same signed checkpoint and data cursor. A restart from initialization is a new
attempt, not a continuation of the original seed.

## Exit gate

All six primary runs complete through update 200 with valid paired receipts and
the registered isolated evaluations, or the phase records an explicit invalid
or inconclusive terminal state.

## Claim boundary

Do not inspect partial results to stop the apparently losing arm. Equal budget
and the fixed horizon are part of the causal contract.
