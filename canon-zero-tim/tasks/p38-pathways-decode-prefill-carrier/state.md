# State

- Status: active.
- Objective: localize and remove the Pathways serving decode-versus-prefill
  carrier without weakening the strict zero-TIM release contract.
- Definition of done: one source-pinned flag-on run reports exact
  `S_decode_vs_S_prefill`, exact `S_prefill_vs_T_old`, and exact
  `T_old_vs_T_current` before a strict full workload is admitted.
- Active phase: P38.2l incident-durable terminal capture is complete.
  P38s13a is the next target action and must use a clean source-pinned render;
  no target run has occurred from this implementation.
- Task directory:
  `canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/`.

## Latest target facts

- P38s12f is a valid Attempt-0 concurrency-32 numerical diagnostic from
  source `b4391703`. It reached logical KV 1972 and measured A-B red at 11 /
  46,390 elements (`33` bytes, `max_abs=0.16271209716796875`) while B-C stayed
  exact. This falsifies concurrency 32 as a repair. Different trajectories
  prevent treating its lower mismatch density as a causal speed/accuracy
  improvement.
- P38s12f did not return the replay payload. Its committed `head.full.log`
  ends before the pre-align timestamp, JobSet termination reports a worker
  failure, and neither the mismatch capsule nor serving archive is present.
  The run answers the concurrency question but cannot construct strict E0.

- P38s12e is not a new run. Its SHA-verified directory contains only repeated
  P38s12d/source-`bdc96818` output: five copies of a 199-line geometry-failure
  log and 360 copies of a 113-line stale-evidence log. `pre-alignment.jsonl`
  is empty and `serving-classification.json` concatenates five JSON objects.
  It has no rollout, capture, alignment, depth, or numerical verdict. Details
  are in `artifacts/p38_2j_p38s12e_evidence_audit_0813.md`.

- P38s12d is infrastructure/configuration-inconclusive. Its rendered command
  correctly selected concurrency 32, but source `bdc96818` still required 256
  in the FrozenLake recipe and failed before rollout with
  `P32 FrozenLake geometry mismatch: {'max_concurrency': 32}`. It produced no
  carrier evidence. The local repair narrowly admits 32 only for the complete
  stock P38 backward-no-commit capture envelope; production/full/eval and all
  other diagnostics retain 256. Details are in
  `artifacts/p38_2j_p38s12d_geometry_fix_0813.md`.

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

- P38 renderings now pin a unique attempt-0 evidence prefix under
  `gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/`.
- A write/read `PREFLIGHT.json` occurs before the workload. Core artifacts and
  `SHA256SUMS` are uploaded before `COLLECTED.json`; `COMPLETE.json` is written
  only after the existing P38 postflight is accepted. Upload failures are
  fail-closed.
- No P38 PVC is mounted. GCS is the sole durable evidence transport for this
  phase; adding a second storage system remains out of scope.
- A live worker now uploads immutable, SHA-sealed host evidence every 30
  seconds when log/journal/ledger/round/report/capsule state changes, and once
  more after the workload stops. `LIVE.json` is written last. A live snapshot
  is crash evidence only; `COLLECTED` and postflight-only `COMPLETE` retain
  their prior meanings.

- Row 231 E0-lite is complete. REF reproduced production B/T-old exactly, but
  mask-derived R0/R1 missed production A at 470 / 566 action values. Verdict:
  `E0_LITE_ENVELOPE_NOT_REPRODUCED`. Strict E0 and the first-divergence seam
  walk remain blocked on missing live-serving state.
- Target P38 diagnostics terminate with explicit exit 42 after the durable
  pre-alignment record and terminal marker. Outer postflight accepts only that
  exit and still rejects missing evidence or a shallow workload.
- The target capsule cap is 256, and every red round is immutable. The stable
  capsule path aliases the most recent red round, so a later exact round cannot
  erase earlier evidence. Every report records host-derived action-depth
  geometry, and P38 postflight still requires logical KV at least 1686.
- The diagnostic performs three frozen-weight rollout/alignment rounds. Each
  nonterminal round queues new prompts and skips backward/optimizer; the final
  round exits through controlled code 42. The target therefore samples 768
  trajectories without changing weights.
- The evidence sealer requires the complete Kubernetes/Pathways package,
  excludes `SHA256SUMS` from itself, and immediately validates every digest.
- The renderer admits concurrency 256 or 32 explicitly. The intent-diff gate
  compares same-source manifests and permits only the concurrency argument and
  matching attestation-label change.

- The classifier restores production block tables serialized as a flat array
  and accepts multiple unique row joins in one snapshot while rejecting an
  ambiguous request-to-row mapping.
- The bounded mismatch capsule records prompt-group/generation identity and
  now permits all 256 target rows.
- P38 prefix bands are now `[1536,1664)`, `[1664,1792)`, `[1792,1920)`, and
  `[1920,2048)`, all reached by the known carrier domain.
- Patch 13 adds a default-off host-only request journal. It records token
  history/SHA, request/DP/slot, physical page map, co-batch membership, and
  explicitly observational page generations once per request/band. It never
  fetches a device buffer. Records from the same scheduler call share one
  append/fsync.
- Renderer `--stock-only` emits only the known-red arm. The legacy default
  paired render remains available for regression tests.
- Patch 14 adds a bounded host-only exact-call incident ledger. It records all
  scheduled requests in `[1400,3072)`, keyed by diagnostic round, call,
  request/token-history identity, logical position, physical pages, observed
  generations, and exact co-batch. The classifier requires every selected red
  row to join this ledger; coarse journal records retain provenance only.
- The postflight requires nonempty journal and incident ledger, three round
  markers, the final published round, full coverage, controlled exit, and
  successful transport/persistence before acceptance.

## Local gates at this checkpoint

- Real Qwen3-8B DP1xTP4 three-round rehearsal: capture-on and capture-off both
  PASS with no backward/commit. Per-round token, action-mask, S_decode,
  S_prefill, T_old, geometry, boundary, and verdict fields are identical. The
  on arm produced 729 incident records / 2,118,899 bytes. Local KV reached
  1577, so this is a neutrality/reachability gate, not a carrier result.
- Complete pinned-image P33 CPU/adjacent gate: PASS after synchronizing the
  target 256-row capsule contract in the dedicated and shared validators.
- Exact-image Qwen3-1.7B/Qwen3-8B overlays: 23/23 each, all 29 manifest entries
  match, terminal `P33_EXACT_IMAGE_PASS`.
- Detailed P38.2l evidence is in
  `artifacts/p38_2l_onehost_rehearsal_0814.md`.

- Row-231 one-host Qwen3-8B DP1xTP4 E0-lite completed with repeat-exact arms,
  a detected one-bit negative control, exact 399-leaf weight attestation, no
  backward, and zero optimizer commits.
- Actual same-source concurrency-256 versus concurrency-32 manifest intent
  diff: PASS; no change outside `--max_concurrency` and its label.
- Complete pinned-image P33 CPU/adjacent gate: PASS (85 workload tests, 37
  alignment tests, all adjacent/focused P38 tests, and terminal
  marker `[P33.WORKLOAD] CPU_GATE PASS`).
- Exact-image Qwen3-1.7B and Qwen3-8B overlays: 23/23 each; all 29 manifest
  entries match; terminal marker `P33_EXACT_IMAGE_PASS`.

- Classifier: 34 tests PASS.
- Renderer: 9 tests PASS.
- Outer serving postflight: PASS, including red/U/error/coverage controls and
  missing-journal and missing-incident negative controls.
- Patch 14 applies to both pinned Qwen3-1.7B and Qwen3-8B overlays; each passes
  23 exact-image tests and all 29 manifest entries. Installed runner SHA-256 is
  `f6ea2ad526a2924b16e85ba804e52f4dc628194712cafc5d02569b04ed2421c4`;
  the installed runner compiles.
- Complete pinned-image P33 CPU/adjacent gate: PASS, including the new journal
  negative control and terminal marker `[P33.WORKLOAD] CPU_GATE PASS`.
- Shell syntax, Python compilation, executable-source ASCII scan,
  credential-pattern scan, ordinary-source whitespace scan, and patch
  application: PASS. Patch 14 necessarily retains unified-diff blank-context
  prefix spaces and passes exact-image manifest identity.
- Historical P38.2i evidence remains in
  `artifacts/p38_2i_local_gate_0813.md`.
- Local fake-GCS persistence and P38 postflight suites pass. The complete
  pinned-image CPU gate and exact-image Qwen3-1.7B/Qwen3-8B overlay gate also
  pass. P38.2k is published at `246eeb87`.

## Next action

1. Render from the exact clean source containing P38.2l. Do not mutate the
   instrumentation except to fix a failing gate.
2. Execute one Attempt-0 stock P38s13a at concurrency 256. Retrieve live and
   final GCS artifacts and use its round-scoped capsule/incident ledger for
   strict E0. Do not repeat concurrency 32, E0-lite, or KV-unified.

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

- Updated: 2026-08-14 UTC; P38.2k is published at `246eeb87`; P38.2l is
  complete, and P38s13a remains NOT RUN.
