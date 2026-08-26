# Phase C — prepare an executable replay prefix

- Status: active

## Objective

Turn the immutable Attempt-6 APC-on carrier into a mechanically verified replay
input plan that a bucket-capable execution agent can produce with one command.
This phase does not execute model inference and must not be reported as a
reproduction, localization, or repair.

## Four distinct coordinates

The tooling must keep these facts separate:

1. `canonical_first_mismatch` is the first red `(source_row,
   completion_position)` in the saved A/B arrays.  Attempt 6 reports row 201,
   position 0, logical prefix 1066.
2. `earliest_red_request` is the earliest serving request in wall-clock call
   chronology whose producer row is red.  In Attempt 6 this is row 245 at call
   164; it is not the canonical array-order mismatch.
3. `canonical_first_mismatch_request` is the request that contains item 1.  In
   Attempt 6, row 201 first enters at call 187; the bounded interval through
   its first possible returned token ends at call 188.
4. `first_fully_captured_incident` is the earliest exact incident selected by
   the capture bands.  Attempt 6 selected row 245, call 565,
   `num_computed_tokens=1248`; this state is already downstream of the true
   position-0 onset.

No document or script may call item 4 the actual onset.  A replay intended to
explain causality must prime the original cache chronology from call 1 through
the canonical mismatch request interval rather than jumping directly to call
565.

## Deliverables

1. `analyze_m15_replay_carrier.py`:
   - verifies the producer arrays, serving envelope, target classification,
     replay contract, first captured incident, and upstream GCS audit receipt
     under one source SHA and immutable Attempt-0 URI;
   - recomputes A-B and B-C byte counts directly from the saved arrays;
   - joins every request token-history SHA to exact producer-token prefixes;
   - emits all four coordinates and a compact call-prefix plan;
   - fails closed on any missing request, chronology gap, count drift, B-C red,
     wrong serving path, or source mismatch.
2. `run_m15_replay_gcs_prepare.sh`:
   - accepts one immutable Attempt-0 GCS URI;
   - downloads and verifies the root manifest and terminal markers;
   - safely extracts the serving capture;
   - reruns the existing GCS audit locally;
   - runs the new analyzer and uploads a versioned, immutable small result with
     its manifest uploaded last.
3. CPU positive and negative tests for exact onset classification, count drift,
   and B-C contamination.
4. Runbook and Handoff commands that require the remote agent to return the
   generated receipt verbatim rather than manually summarize it.

## Exit gate

```text
M15_REPLAY_ANALYSIS_TEST_PASS
M15_REPLAY_GCS_PREPARE_SYNTAX_PASS
M15_REPLAY_INPUT_PLAN_READY_NOT_EXECUTED
```

The last marker is produced only from real Attempt-6 GCS data.  Local synthetic
tests may prove the analyzer contract but cannot mint that target marker.

## Next-phase decision table

| Replay outcome | Interpretation | Next action |
|---|---|---|
| production A and B bytes reproduced | deterministic carrier | observer-neutral tensor walk |
| B reproduced but A is exact with B | target envelope missing | add one topology/scheduler variable |
| B or saved inputs do not reproduce | harness invalid | repair harness; no mechanism claim |
| observer changes endpoint bytes | observer invalid | redesign observer |
