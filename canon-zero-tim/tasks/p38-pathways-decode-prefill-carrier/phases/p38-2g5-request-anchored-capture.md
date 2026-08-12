# P38.2g5: request-anchored serving capture

- Status: published at `02e8c05d`; P38s6 target run pending.

## Problem

P38s5 completed rollout work but produced no serving capture, mismatch capsule,
terminal precheck marker, classifier, archive, or outer postflight. Its log
contains no evidence that the patched runtime hook was imported or called.
The prior explanation that FrozenLake prompts were about 200 tokens is not
supported by the captured trajectory data, and the explanation that
FrozenLake bypassed `GRPOLearner` contradicts the recipe and call site.

Two implementation mismatches made the run non-diagnostic:

1. the trigger selected a stratum from packed device `input_positions`, while
   the durable record and mismatch join were request-anchored on scheduler
   `num_computed_tokens`; and
2. a known A-B red was passed to `check_pre_backward(fail_closed=True)`, so it
   raised before the precheck-only stop marker that outer postflight required.

## Change

- Select candidate strata from one-token scheduled requests and their host
  `num_computed_tokens` values.
- Continue to verify packed input positions, active masks, sequence lengths,
  selectors, block tables, and query starts after a candidate is selected.
- Emit one import-time `CANON_P38_SERVING_CAPTURE_INIT` marker and at most 32
  `CANON_P38_SERVING_CAPTURE_OBSERVE` lines, deduplicated by 256-token prefix
  band. A miss now reports hook call count, request count, and observed range.
- Require outer postflight to see exactly one init marker and at least one
  observation marker.
- In precheck-only mode, persist a finite A-B result with exact B-C and stop
  before backward. Empty actions, invalid shapes, non-finite values, or B-C
  drift remain fatal. Normal training remains fail-closed.

## Local exit gate

- Patch reconstructs against the pinned TPU inference image.
- Qwen3-1.7B and Qwen3-8B overlays each match all 29 manifest entries and pass
  16/16 exact-image tests.
- Frozen-image CPU gate passes 81 workload tests, 31 alignment tests, and all
  adjacent suites.
- Renderer 5/5 and P38 shell postflight pass.
- `git diff --check`, Python compilation, and shell syntax pass.

Installed runner SHA-256:
`72c4307859c32de4e7080823bbe0693fb04c21a67ab82a3cfe829bb6c39ed18c`.

## Target exit gate

P38s6 is stock-only and Attempt 0. It must emit exactly one init marker, at
least one observation, four pre/post pairs, finite A-B red, exact B-C, one
precheck-complete stop, a real run-specific mismatch capsule, a serving
classification PASS with at least one exact token-history join, a serving
archive, and outer acceptance with backward and optimizer commits both zero.

If no stratum is captured, return the observation lines and classify the run
as inconclusive. Do not lower thresholds or relaunch automatically; the
observed scheduler prefix range determines the next single-variable change.

## Claim ceiling and rollback

This repairs diagnostic reachability and observability only. It does not
repair A-B, prove RoPE/page/cache causality, or admit training. Leave all
`CANON_P38_SERVING_CAPTURE_*` variables and `CANON_P38_PRECHECK_ONLY` unset to
restore stock runtime behavior.
