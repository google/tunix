# State

- Status: active.
- Objective: localize and remove the Pathways serving decode-versus-prefill
  carrier without weakening the strict zero-TIM release contract.
- Definition of done: one source-pinned flag-on run reports exact
  `S_decode_vs_S_prefill`, exact `S_prefill_vs_T_old`, and exact
  `T_old_vs_T_current` before a strict full workload is admitted.
- Active phase: P38.2r single-run terminal seam-and-tail acquisition concluded.
  P38s18r2 completed Round 0 execution on 64 TPU (`DP16xTP4`, concurrency 256),
  captured 971 Tail and 915 Seam records, and successfully sealed the full
  Round 0 bundle to GCS (`manifest_sha256 = ce7df453259dd070472486e053dbb26b03dad7b6259784cde74da7fe9efe227e`).
  Scientific findings: S_prefill vs T_old is 100% bitwise exact (0 differing bytes
  across 45,559 tokens); S_decode vs S_prefill has 45 differing bytes (99.975%
  identity), all precisely aligned at 256-token Pallas Chunked Attention page
  boundaries. P38 lane concluded; capacity transferred to FrozenLake 8B Full
  Training (`p45r8`).
- Task directory:
  `canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/`.

## Latest target facts

- P38s18r/source `6b75e3cf4942` ran on 64 TPU (`DP16xTP4`, Concurrency 256,
  3 Frozen Rounds, Seam Mode `layer`, Terminal Tail `1`) with zero backward,
  zero optimizer commits, and all 6 overlays verified by SHA256.
  - One round-0 precheck record reported full 32-prompt coverage
    (`N_action=46,098`):
    - B-C boundary (`S_prefill` vs `T_old`): STRICT EXACT 0 mismatch bytes.
    - A-B boundary (`S_decode` vs `S_prefill`): exactly 30 mismatch bytes.
    - The manual report says 360+ seam/tail NPZ records were live-uploaded,
      but those raw bytes are not committed here and no immutable round bundle
      completed; that count is not independently audited in this checkout.
  - Durability seal timeout error:
    - At end of Round 0, `stage_p38_round.py` failed with `ValueError: no round 0 records in pre_alignment.jsonl` because `pre_alignment.jsonl` contained `"step": 0` while `_filter_jsonl` strictly looked for `"diagnostic_round"`.
    - Main thread timed out after 900s: `timed out waiting for P38 round 0 durability acknowledgement`.
    - Remote fix `fbb4b278` added a `step` fallback and wrote
      `diagnostic_round=int(step)`. Review found it unsafe because frozen rounds
      may advance while step remains zero, and because unscoped incident data
      could enter every round.
    - Local replacement (not committed/pushed) derives diagnostic scope from
      the frozen-round counter, rejects missing/wrong scoped records, and
      admits only the schema-validated cumulative request journal.
  - Overall classification:
    `INCONCLUSIVE_DURABILITY_SEAL_TIMEOUT`. No round-complete marker,
    three-round package, controlled exit 42, `COLLECTED`, or `COMPLETE` is
    admitted from this attempt.
  - Artifact report: `artifacts/p38s18r_round0_seal_error_report.md`.

- P38s18l/source `9a83457417fc` ran at Concurrency 256 / DP16xTP4 with zero
  backward and zero optimizer commits, but did not complete its registered
  three-round contract.
  - The raw log has two round-complete markers, two pre-alignment records, no
    terminal precheck marker, and ends during round-2 rollout.
  - The completed rounds report A-B red at 28 / 40 differing bytes and 19 / 28
    elements. B-C is exact in both completed rounds.
  - The committed package has two immutable round capsules and zero raw seam
    JSON/NPZ records. The committed PASS classification says 20 / 47 red
    points joined but cannot be reproduced from the committed inputs.
  - No third-round zero result, full hidden-byte equality, lm_head isolation,
    normalizer isolation, controlled exit, or terminal evidence is admitted.
  - Evidence directory: `evidence/p38s18l/`; GCP reduction v1 sealed at
    `derived/p38s18l-seam-reduction-v1` (`INCONCLUSIVE_REDUCTION_JOIN`).
  - V2 snapshot selection (`select_p38_live_snapshot.py`) scanned all 22 live
    snapshots: `000020` has only round-0 capsule (1 round < 2 required);
    `000021` has rounds [0, 1] but lacks `SHA256SUMS` and paired NPZs due to
    workload exit. Result: `INCONCLUSIVE no qualifying snapshot candidates=22`
    (rc=4). Commit `e0c1aef7` records only the human summary; no raw inventory
    was committed, so this is not yet an independently reproducible evidence
    package. It does not authorize the tail branch.
- P38s17/source `baac38bc4034` completed all 3 Frozen-Weight diagnostic rounds
  (768 trajectories total, 143,511 action tokens across 3 rounds) on 64 TPU
  (`DP16xTP4`, concurrency 256) with zero backward, zero optimizer commits, and
  a workload-level controlled exit 42.
  - B-C boundary (`S_prefill` vs `T_old`): STRICT EXACT 0 on all 3 rounds.
  - A-B boundary (`S_decode` vs `S_prefill`): respectively 94 / 19 / 44
    differing bytes and 58 / 14 / 28 differing elements, with `N_action`
    46,507 / 46,237 / 50,767.
  - Incident ledger: 2,523 records spanning serving call indices through 3,069.
  - Reclassification returns `live_kv_fingerprint_equal_on_red_row`.
  - Evidence directory: `evidence/p38s17/`.
- P38s16/source `4101f752` successfully executed all 3 Frozen-Weight
  diagnostic rounds (768 trajectories total, 148,916 action tokens across 3
  rounds: 48,556 / 47,313 / 53,047) on 64 TPU (`DP16xTP4`, concurrency 256)
  with zero backward, zero optimizer commits, and controlled exit 42.
  - B-C boundary (`S_prefill` vs `T_old`): STRICT EXACT 0 on all 3 rounds.
  - A-B boundary (`S_decode` vs `S_prefill`): Round 1 had 52 differing bytes
    (32 elements), Round 2 had 27 differing bytes, Round 3 had 18 differing bytes.
  - Incident ledger: 3,686 records / 4,234 calls / 91.7 MB, successfully
    recording fixed-M compile geometry and exact tokens for natural single-active calls.
  - Evidence archived under `evidence/p38s16/`.
- The reproducible P38s16 audit validates 44,676 request entries and joins all
  60 A-B mismatch elements with zero missing/ambiguous joins. The only
  naturally single-active mismatch is round 2 / row 255 / call 4223 / request
  `2529-a6d304ba`, prefix 2209, with `A=-0.041210174560546875`,
  `B=C=-0.04151153564453125`. All ledger entries with compile geometry share
  one production fixed-M signature. This is exact call identity, not live-KV
  content evidence. See `artifacts/p38s16_single_active_audit_0814.json`.
- P38s15/source `58a0ed84` completed all 3 Frozen-Weight diagnostic rounds
  (768 trajectories total, 51,330 action tokens) with zero backward and zero
  optimizer commits, exiting with controlled code 42. It measured exact B-C
  (0 mismatches, bitwise identical `S_prefill` vs `T_old` hash
  `4ee783597573623391cdf65917990963dab4d85960080d396465a454c7003dd3`),
  and measured A-B red at 20 / 51,330 elements (`33` differing bytes,
  `max_abs=0.20377731323242188` at row 215 pos 689). Mismatch rows
  `[215, 223, 231, 254, 255]` were captured in capsule sha256
  `9a7d6caf0125b0798a7745ae82882132115b1721414ecf6e1f3bde18c2d27c35`
  with incident ledger (1,915 records / 2,465 calls / 53.3 MB). Evidence
  is committed under `evidence/p38s15/`.
- Across all three P38s15 rounds, 64 A-B mismatch elements joined exactly to
  61 serving calls. Six mismatch calls had exactly one scheduled request, and
  every joined request used local scheduler slot 0. This disproves a required
  large live co-batch and the earlier slot-15 reading, but it does not imply a
  shape-one executable: the run-wide contracts still report fixed production
  padding (`decode_rows=16`, canonical logprob rows 256, adapter global/local
  M 4096/256).
- No joined mismatch call contains a simultaneous same-DP physical-page alias.
  Sequential page reuse remains visible, but observation generations do not
  prove stale KV bytes or an allocator lifecycle violation.
- P38s13a/source `d3e6c1b0` reproduced A-B red at 39 / 48,043 elements
  (`58` bytes, `max_abs=0.28188323974609375`) with exact B-C, but it was a
  pre-P38.2l single-round run. Its committed evidence omits the capsule,
  serving archive, incident ledger, classification, and final GCS markers.
- P38s14/source `ac2c31bc` reproduced A-B red at 26 / 47,076 elements
  (`42` bytes, `max_abs=0.2532196044921875`) with exact B-C. It is also a
  pre-P38.2l single-round run: no round markers, incident ledger, live GCS
  snapshots, `COLLECTED`, or `COMPLETE` survived. It cannot construct strict
  E0. Its SHA-verified stdout is under `evidence/p38s14/`.

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

- P38.2r adds a default-off terminal-tail observer beside the existing layer
  observer. It reads only already-materialized raw/processed logits, target
  IDs, and production logprobs and returns compact selected-row scalars.
- Every multi-round target precheck now performs a request/ACK durability
  handshake. Round `n+1` cannot start until the survivor worker has staged,
  SHA-checked, uploaded, downloaded, and verified round `n` and written
  `ROUND_COMPLETE.json` last.
- A one-host rehearsal is explicitly exempt from remote sealing and prints
  `ROUND_SEAL_SKIPPED`; target preflight forbids that rehearsal flag.
- Observer gates: corrected source `ae63d44e...` passed same-source local-v5p
  off/on endpoint neutrality across three rounds. The later host-only round-
  scope correction passes focused round-stage/postflight/neutrality tests,
  fake-GCS two-round content isolation and abrupt-exit durability, pinned-image
  alignment tests, Python/shell checks, and the complete P33 CPU ladder. It
  does not change an overlay, runner patch, model executable, or canonical
  kernel. Publication and a fresh target run remain pending.

- P38 renderings now pin a unique attempt-0 evidence prefix under
  `gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/`.
- A write/read `PREFLIGHT.json` occurs before the workload. Core artifacts and
  `SHA256SUMS` are uploaded before `COLLECTED.json`; `COMPLETE.json` is written
  only after the existing P38 postflight is accepted. Upload failures are
  fail-closed.
- No P38 PVC is mounted. GCS is the sole durable evidence transport for this
  phase; adding a second storage system remains out of scope.
- A live worker uploads immutable, SHA-sealed host evidence every 30 seconds
  when log/journal/ledger/round/report/capsule state changes. It now also owns
  terminal persistence through atomic collect/complete requests and ACKs.
  `COLLECTED` can survive a later postflight failure; `COMPLETE` remains
  postflight-only and is requested last.

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
- Patch 15 extends each incident record with the production fixed-M contract:
  DP size, padded rows, canonical logprob rows, and shape/dtype/sharding for
  model inputs and attention metadata. It never fetches device arrays. A
  naturally single-active call additionally carries exact token IDs; a
  one-row input substitution fails closed.
- The old DP1 replay scripts and reports are explicitly labeled E0-lite. They
  remain useful counterfactuals but cannot establish production program
  identity or unlock the first-divergence seam walk.
- Call-4223 E0-lite completed with `E0_LITE_ENVELOPE_NOT_REPRODUCED`.
  REF reproduced all 646 production B/T-old action values, while repeat-exact
  R0/R1 differed from production at 428 values (`max_abs=29.4570369720459`).
  The one-bit negative and 399-leaf weight gates passed. This replay cannot
  select an operator repair; P38.2n N3 therefore proceeded to the now-complete
  live/clean observer gate.

## Local gates at this checkpoint

- Current pinned-image P33 CPU/adjacent gate: PASS with workload 85/85,
  alignment 37/37, all focused P38 persistence/postflight tests, and terminal
  `[P33.WORKLOAD] CPU_GATE PASS`.
- Worker-owned persistence rejects complete-before-collect and preserves
  completion-last semantics. The direct host renderer import lacks `metrax`;
  the same renderer tests pass in the pinned image and only that result counts.
- P38.2m focused classifiers: serving capture 36/36 and replay 7/7 PASS.
- Complete pinned-image P33 CPU/adjacent gate: PASS with workload 85/85,
  alignment 37/37, and terminal marker `[P33.WORKLOAD] CPU_GATE PASS`.
- Exact-image Qwen3-1.7B/Qwen3-8B overlays: 25/25 each; all 29 manifest
  entries match; terminal marker `P33_EXACT_IMAGE_PASS`.
- Patch 15 installed runner SHA-256 is
  `f6273f9aa9d9b066b3ccba13760e2dbddeea633846fd37bfeaf9ae1e731d4acc`.
  Shell/Python syntax and `git diff --check` PASS.

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

1. Review the local P38.2r round-scope fix and its focused/full CPU gate
   receipts. Do not commit or push it without the user's explicit approval.
2. After publication, record the exact full source SHA. Do not use `HEAD`, do
   not edit the rendered manifest, and do not reuse run-id `p38s18r`.
3. Render one fresh `p38s18r2` stock target from that approved SHA by following
   `P38S18R_RUNBOOK.md`. Preserve the failed P38s18r logs and GCS prefix.
4. Admit the replacement only if all three distinct diagnostic rounds each
   have an immutable round bundle and ACK, followed by controlled exit 42,
   `COLLECTED`, `COMPLETE`, and an offline official-classifier replay from the
   returned bytes.

## Claim ceiling and blockers

- Observation generations are not allocator generations. They cannot prove an
  unobserved free/reuse event or equal KV contents.
- Patch 16 records bounded integer aggregates and fixed samples for live and
  clean KV. These are diagnostic fingerprints, not cryptographic hashes or a
  mathematical proof of full-byte equality. They choose a branch only when
  joined to a production A-B-red row.
- Fixed-M attestation proves that scheduler occupancy one did not collapse the
  production input aval. It does not fingerprint a compiled executable and it
  does not prove equal KV content.
- Exact production-envelope reproduction, or an in-situ first-divergence
  observer with neutrality evidence, remains the hard gate before a RoPE/RPA/
  residual/logits repair claim.
- P38 capture is diagnostic-only and must not be injected into P45 committed
  training. GSM8K/DeepSWE warning-only campaigns are separate workstreams and
  do not promote P38.

## Rollback

Leave `CANON_P38_SERVING_CAPTURE_DIR`, `CANON_P38_REQUEST_JOURNAL`,
`CANON_P38_KV_OBSERVER_DIR`, `CANON_P38_PRECHECK_ONLY`, and
`CANON_KV_UNIFIED` unset. The diagnostic is default-off and does not change
training, evaluation, prefix cache, precision, optimizer placement, or
canonical kernels.

- Updated: 2026-08-16 UTC; P38s18r is
  `INCONCLUSIVE_DURABILITY_SEAL_TIMEOUT`. The strict round-scope replacement is
  local only, and no replacement TPU run, backward, or optimizer commit has
  occurred.
