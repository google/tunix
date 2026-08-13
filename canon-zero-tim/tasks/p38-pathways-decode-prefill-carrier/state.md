# State

- Status: active.
- Objective: localize and remove the Pathways serving decode-versus-prefill
  carrier without weakening the strict zero-TIM release contract.
- Definition of done: one source-pinned flag-on run reports exact
  `S_decode_vs_S_prefill`, exact `S_prefill_vs_T_old`, and exact
  `T_old_vs_T_current` before a strict full workload is admitted.
- Active phase: P38.2j, P38s12a accounting, row-231 E0-lite, and true
  concurrency-32 P38s12b.
- Task directory:
  `canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/`.

## Latest target facts

- The evidence published under the `p38s12b` label actually used concurrency
  256. Its core numerical/capture evidence is internally consistent and is
  accounted as P38s12a analysis-level evidence. `rc=137`, an incomplete outer
  bundle, an eight-row capsule cap that omitted row 255, and a stale self-hash
  prevent formal target admission.

- P38s11 is the first terminal full-coverage stock capture. It covered 32
  prompts / 256 trajectories, reproduced 27 differing A-B elements among
  48,449 actions with maximum absolute difference about 0.1044, kept B-C
  exact, emitted no capture error, stopped before backward, and returned its
  real run-specific capsule/archive.
- The carrier again begins deep: logical KV 1686--1977, turns 3--4, with red
  rows concentrated in the final producer units. P38s10's first-four-prompt
  exact result was an under-depth subset and is not repair evidence.
- Exact offline token-prefix/SHA joins from the P38s11 archive map capsule rows
  199 and 206 to six serving records across turns and DP ranks. The full table
  is in `artifacts/p38_2i_p38s11_offline_join_0813.md`.
- Those global snapshots did not observe the red rows at their mismatch times.
  They establish provenance and show that exact joins are feasible, but they
  do not establish page ownership, stale KV, RoPE, residual/cast, or another
  numerical cause.
- Unified KV is a production negative: it remained materially red and must not
  be rerun as a repair candidate.

## Current local implementation

- Row 231 E0-lite is complete. REF reproduced production B/T-old exactly, but
  mask-derived R0/R1 missed production A at 470 / 566 action values. Verdict:
  `E0_LITE_ENVELOPE_NOT_REPRODUCED`. Strict E0 and the first-divergence seam
  walk remain blocked on missing live-serving state.
- Target P38 diagnostics terminate with explicit exit 42 after the durable
  pre-alignment record and terminal marker. Outer postflight accepts only that
  exit and still rejects missing evidence or a shallow workload.
- The capsule cap is 16, and every report records host-derived action-depth
  geometry. P38 postflight requires logical KV at least 1686.
- The evidence sealer requires the complete Kubernetes/Pathways package,
  excludes `SHA256SUMS` from itself, and immediately validates every digest.
- The renderer admits concurrency 256 or 32 explicitly. The intent-diff gate
  compares same-source manifests and permits only the concurrency argument and
  matching attestation-label change.

- The classifier restores production block tables serialized as a flat array
  and accepts multiple unique row joins in one snapshot while rejecting an
  ambiguous request-to-row mapping.
- The bounded mismatch capsule records prompt-group/generation identity and
  now permits 16 selected red rows, covering the nine-row observed population.
- P38 prefix bands are now `[1536,1664)`, `[1664,1792)`, `[1792,1920)`, and
  `[1920,2048)`, all reached by the known carrier domain.
- Patch 13 adds a default-off host-only request journal. It records token
  history/SHA, request/DP/slot, physical page map, co-batch membership, and
  explicitly observational page generations once per request/band. It never
  fetches a device buffer. Records from the same scheduler call share one
  append/fsync.
- Renderer `--stock-only` emits only the known-red arm. The legacy default
  paired render remains available for regression tests.
- The postflight requires a nonempty journal and requires every capsule row to
  have a journal join. The journal is archived with the serving records.

## Local gates at this checkpoint

- Row-231 one-host Qwen3-8B DP1xTP4 E0-lite completed with repeat-exact arms,
  a detected one-bit negative control, exact 399-leaf weight attestation, no
  backward, and zero optimizer commits.
- Actual same-source concurrency-256 versus concurrency-32 manifest intent
  diff: PASS; no change outside `--max_concurrency` and its label.
- Complete pinned-image P33 CPU/adjacent gate: PASS (81 workload tests, 34
  alignment tests, 15 adjacent tests, all focused P38 tests, and terminal
  marker `[P33.WORKLOAD] CPU_GATE PASS`).
- Exact-image Qwen3-1.7B and Qwen3-8B overlays: 23/23 each; all 29 manifest
  entries match; terminal marker `P33_EXACT_IMAGE_PASS`.

- Classifier: 30 tests PASS.
- Renderer: 8 tests PASS.
- Outer serving postflight: PASS, including red/U/error/coverage controls and
  a marker-present but journal-file-missing negative control.
- Patch 13 applies to both pinned Qwen3-1.7B and Qwen3-8B overlays; each passes
  23 exact-image tests and all 29 manifest entries. Installed runner SHA-256 is
  `3a219b251020894ade2002e480aa8b3fef90ea62a70794116b143bad89b36b17`;
  the installed runner compiles.
- Complete pinned-image P33 CPU/adjacent gate: PASS, including the new journal
  negative control and terminal marker `[P33.WORKLOAD] CPU_GATE PASS`.
- Shell syntax, Python compilation, executable-source ASCII scan,
  credential-pattern scan, ordinary-source whitespace scan, and patch
  application: PASS. Patch 13 necessarily retains unified-diff blank-context
  prefix spaces and passes exact-image manifest identity.
- Detailed local evidence is in `artifacts/p38_2i_local_gate_0813.md`.
- No new target cluster run occurred. The true concurrency-32 P38s12b remains
  NOT RUN.

## Next action

1. After user review, publication, and separate cluster approval, execute one
   Attempt-0 **stock-only** true P38s12b at concurrency 32 using the exact
   runbook. Render a same-source concurrency-256 baseline for intent-diff only;
   never apply it.
2. Require exact B-C, full 256-trajectory coverage, logical KV at least 1686,
   controlled exit 42, journal joins, the complete infrastructure bundle, and
   SHA seal PASS. Any missing item is inconclusive.
3. If A-B is exact, repeat one depth-sufficient concurrency-32 arm. If A-B is
   red, conclude only that concurrency 32 is insufficient; neither result
   identifies an operator.
4. Do not repeat E0-lite or start the seam walk from R0/R1. Construct strict E0
   only after an exact live-serving reconstruction becomes available.

## Claim ceiling and blockers

- Observation generations are not allocator generations. They cannot prove an
  unobserved free/reuse event or equal KV contents.
- Full device KV content hashing is intentionally absent from P38s12a because
  it can perturb the program. Add it only for an exactly joined red request and
  only with observer-neutrality evidence.
- Exact E0 remains the hard gate before the RoPE/RPA/residual/logits seam walk
  or any repair.
- P38 capture is diagnostic-only and must not be injected into P45 committed
  training. GSM8K/DeepSWE warning-only campaigns are separate workstreams and
  do not promote P38.

## Rollback

Leave `CANON_P38_SERVING_CAPTURE_DIR`, `CANON_P38_REQUEST_JOURNAL`,
`CANON_P38_PRECHECK_ONLY`, and `CANON_KV_UNIFIED` unset. The diagnostic is
default-off and does not change training, evaluation, prefix cache, precision,
optimizer placement, or canonical kernels.

- Updated: 2026-08-13 UTC; P38.2j local gates and E0-lite complete, true
  P38s12b target NOT RUN.
