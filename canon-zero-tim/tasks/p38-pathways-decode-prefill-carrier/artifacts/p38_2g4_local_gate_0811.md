# P38.2g4 D0 local gate evidence

- Date: 2026-08-11 UTC
- Worktree HEAD: `faa2529754f2600260988001b237ef161175ecb0`
- Branch: `codex/p39-deepswe-production-contract-0810`
- Verdict: `PASS` for local construction only

## Implemented contract

- Capture exactly one concrete scheduled request from each prefix interval:
  `[1536,1792)`, `[1792,2048)`, `[2048,2304)`, and `[2304,2560)`.
- Preserve the anchor request ID and prefix, token history, request/DP/slot/
  attention/page mapping, source commit, and callable identities.
- Require four unique strata, a five-times free-space margin, and at least one
  unambiguous exact token-history join to the run-specific mismatch capsule.
- Keep the diagnostic default-off and preserve stock precision, attention,
  prefix-cache, sampling, loss, backward, and optimizer behavior.

## Verification

- Classifier negative controls: `25/25 PASS`.
- Renderer controls: `5/5 PASS`.
- Shell postflight: `PASS`.
- Python compilation and shell syntax: `PASS`.
- Patch dry-run, application to the pinned stock runner, and `py_compile`:
  `PASS`.
- Reconstructed runner SHA-256:
  `fe81622996a1c73bbd17187ee603e6a191165202da40d07b5e428fe41b5db516`.
- Exact-image Qwen3-1.7B: all 29 manifest entries match; `14/14 PASS`.
- Exact-image Qwen3-8B: all 29 manifest entries match; `14/14 PASS`.
- Complete frozen-image P33 CPU gate: 78 workload tests, 29 alignment tests,
  and all adjacent suites `PASS`.
- `git diff --check`: `PASS`.
- Executable/docstring Han-character scan over the P38 implementation:
  `PASS`.

## Claim ceiling

The local Docker validation had no `/dev/vfio`; no numerical TPU execution or
Pathways serving capture occurred. No target mismatch was reproduced, no
operator or cache hypothesis was selected, and no backward or optimizer code
ran. No commit, push, cloud job, or external resource action was performed.

## Next gate

After source review, explicit publication approval, and separate resource
approval, run one stock-only Attempt-0 FrozenLake diagnostic. D1 must return
all four strata, the exact mismatch join, complete outer log, serving archive,
run-specific capsule, classifier JSON, and final PATHTRACE. Do not run U or a
repair arm.

## Rollback

Leave `CANON_P38_SERVING_CAPTURE_DIR` and the prefix-strata variables unset, or
discard the uncommitted P38.2g4 diff. Runtime behavior remains stock.
