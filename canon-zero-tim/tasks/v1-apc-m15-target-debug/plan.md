# Plan

## Outcome

Localize the real M15 DP8xTP8 APC-on serving mismatch to the first exact tensor interval and source boundary before changing numerical code. B remains an independent full recomputation with `reset_prefix_cache=True`; all production recipes remain APC-off. No TPU, pinned-image, Kubernetes, commit, or push action is implicit in this plan.

The immutable numerical contract is:

```text
A = APC-on rollout decode
B = serving prefill rescore of A's exact action IDs with reset_prefix_cache=True
C = trainer old-policy forward

A - B = 0 bytes
B - C = 0 bytes
```

## Phases

| Phase | Deliverable | Exit gate | Status |
|---|---|---|---|
| A | `M15_FIRST_RED_INPUT_CONTRACT` with mismatch distribution, identity hashes, and artifact-completeness matrix | every field is traced to immutable `file:line` or explicitly marked missing | complete |
| B | fresh captured-red replay carrier and APC-off/APC-on decision table | full 256-row producer, mixed standard/continue A chronology, standard-only full-reset B, exact request joins, GCS durability, zero backward/commit | complete: Attempt 6 off=`CONTROL_GREEN`, on=`FRESH_TARGET_RED_FROZEN` |
| C | executable replay-prefix input plan followed by a single-variable reproduction ladder | saved bytes and chronology revalidate; control A-B=0; treatment either reproduces red or is `ONEHOST_NOT_REPRODUCED` | complete: r10-r13c one-host ladder exact through full chronology; scale/topology remains |
| D | observer-neutral coarse-to-fine first-red walk on known-red DP8xTP8 | `FIRST_RED_LOCALIZED` names last exact and first red tensor, shape ledger, request/token/cache coordinate, and `file:line` | d33 five-file subset reports `k_post_rope -> rpa_output`, but official per-round classifiers/operator receipts are missing; not complete |
| D2 | M15-only incremental observer shards, sealed-input classifier, terminal ordering, and source self-verification | forced death preserves a verified shard; unsealed/tampered/source-mismatched inputs fail closed | host and exact-image pass; Attempt-12 runtime used this path, but its committed return omitted the remote binding audit |
| D3 | three frozen-weight full Layer-0 rounds per arm plus one self-hashed numerical/operator return | each round seals before the next; root-incomplete, partial-round, missing JobSet terminal, and missing raw-log receipt remain mechanically distinguishable | 162/162 completion receipts and manifests checked, 100% round 0 (`D33_FLAT_SHARDS_ROUND0_ONLY`); tar payloads were not independently re-hashed and seal/ACK repair is required before rerun |
| D3a | harden the first seal/ACK transition with stage receipts and a fail-fast failure channel | host three-round ACK positive control and forced-persistence failure negative control pass; numerical source remains unchanged | local PASS (137/137 task tests, P38 persistence PASS, flag audit 394/394); exact-image and target not run |
| D3b | bind every cumulative replay-envelope row to its live diagnostic round | installed-source AST gate rejects missing or hard-coded round; assembler selects round 0/1/2 independently | local PASS (139/139 task tests, patch applies to registered runner, P38 persistence PASS); exact-image and target not run |
| D3c | distinguish same-prefix serving requests and checkpoint classifier inputs before analysis | candidate-set classifier never conflates requests or fabricates a single interval; classifier inputs survive an analysis failure | cluster exercised by Attempt 17: request-aware classification and checkpoint durability PASS; treatment Round 0 preserved a mixed candidate set, not a localization |
| D3d | bind Attempt-17 source rows to serving requests from future token-prefix continuity without another rollout | immutable bundle and committed receipt verify; one request is selected only beyond the latest explicit elimination horizon, otherwise the candidate set is preserved | complete for request identity: read-only GCS/CPU return binds source row 217 uniquely through prefix 1300; global mixed signatures preserve the candidate-set verdict |
| D3e | separate the canonical completion-position-zero decision scope from later red-action diagnostics | decision-scope mixed/exact candidates still fail closed; global signatures and all unobserved red points remain explicit; immutable Attempt-17 bundle is reclassified without target execution | complete: committed return is `FIRST_RED_LOCALIZED` at Layer 0 `k_post_rope -> rpa_output`; evidence remains partial-round/analysis-grade and does not authorize a repair |
| E0 | distinguish stored Layer-0 live-KV content from page selection/read/RPA execution context at the uniquely bound 1226-token prefix | all eight prefix aliases are captured; future-prefix proof selects exactly one request; control and B-C stay exact; compact evidence self-verifies | target execution is reported at `ff33dcd2`, but the committed compact return is incomplete/non-official and the E0 verdict is not admitted |
| E0r | recover and fail-closed admit the official Attempt-18 compact return | exact four-file inventory; SHA256 manifest; full eight-candidate binding; source/COMPLETE/B-reset/zero-cached-token receipts; official and intake markers | local implementation active; host intake 9/9 PASS; GCS recovery and exact-image NOT RUN |
| D4 | Attempt-13 two-arm registered-root inventory and offline semantic review | both listings succeed; exact 77/70 shard triples verify; physical shard counts and immutable classifier counts remain separate; seven-file inventory self-verifies | transport complete; no-live confirmed; count drift -29/+101 preserved; official replay impossible |
| E | minimal localized repair, default off or experiment-bound | reproducer flips red to zero; APC-off and B are unchanged; adjacent and dirty-page negatives fire | pending |
| F | certification ladder | host -> exact-image -> one-host clean/repeat/dirty -> matched profile -> separately approved DP8xTP8 G-E | pending |

## Phase-C decision table

| Observation | Interpretation | Next action | Forbidden claim |
|---|---|---|---|
| replay tokens/order differ from `m15i` | comparison invalid | repair capture/join | APC mechanism conclusion |
| APC-off control is red | carrier or shared serving contract is invalid | preserve the concurrently run treatment but do not interpret it as APC-specific | prefix-cache root cause |
| APC-on one-host is exact | production envelope not reproduced | add exactly one missing M15 variable | bug disappeared |
| APC-on clean run is red | deterministic carrier available | observer-neutral Phase D | RoPE/page cause before localization |
| observer changes endpoint bytes | instrumentation invalid | redesign observer | first-red conclusion |
| first red is cache read with stored bytes exact | transient mapping/read interval | test one mapping/order degree of freedom | stale stored-page claim |
| target repair is exact | target forward candidate admitted | run required negative/repeat gates | production default-on before all gates |

## Decisions

- Confirmed: one-host Qwen3-8B DP1xTP4 G-A through G-D is green; no APC numerical repair was made there.
- Confirmed: M15 `m15i` on DP8xTP8 is a strict A-B red and B-C exact at step 0; production APC is off.
- Confirmed: first mismatch occurs at logical prefix 1226, so the older 1686-depth heuristic is not a valid lower bound for this case.
- Confirmed: all 760 mismatches belong to prompt-major group 24; rows 192, 193, 194, 196, 197, 198, and 199 are red while row 195/generation 3 is clean.
- Confirmed: only 6/760 red coordinates are exactly on a 256-token boundary; the first red is at offset 202. A simple block-boundary trigger is not supported.
- Confirmed: the archived `m15i` bundle contains hashes and all mismatch coordinates but no reversible full arrays or serving chronology. Historical `m15i` cannot be called an exact replay input.
- Confirmed: the receipt source SHA conflicts with the runtime sync receipt. The executed source is `71d889a32f4668353c758d5c00df88299e6c0d35`; the receipt's `7a2a456c...` is retained as a provenance defect, not used for replay.
- Hypothesis: M15 chronology, cache residency/churn, or TP8/multihost placement supplies a condition absent from the one-host carrier.
- Decision: do not change RoPE, attention, KV values, lm-head, loss, backward, or optimizer until Phase D passes.
- Decision: XProf/Perfetto work is deferred until a numerical red is reproduced; profiles cannot decide equality.
- Decision: reuse the checked-in P38 mismatch capsule, request journal, incident ledger, and fail-closed join classifier. Do not add a second cache observer until these existing host-only fields prove insufficient.
- Decision: because `m15i` raw replay inputs are absent, the next strict replay source must be a fresh captured target red. A new stochastic run may be compared structurally with `m15i`, but it is not the same historical trajectory.
- Decision: one fresh real rollout is required to create the carrier. Once
  frozen, Phase C reuses the saved token streams and serving chronology, so it
  does not need to solve FrozenLake or resample actions on every debug run.
- Decision: large producer/envelope payloads remain in the registered GCS
  attempt. The GCS-side audit uploads only small, self-hashed receipts under a
  versioned derived prefix.
- Confirmed: Attempt 0 did not reach an APC numerical boundary. Its command
  selected `m15/main` on the CLI but omitted the matching signed environment
  fields, so the FrozenLake entrypoint rejected the split identity before
  learner construction. The repaired carrier transports and checks both sides
  of that identity; this is a launcher-contract repair, not an APC repair.
- Confirmed: Attempt 1 also did not reach an APC numerical boundary. It passed
  overlay and GCS preflight, then the entrypoint treated every P38 precheck as
  the legacy DP16 carrier and rejected the signed M15 target's
  `mini_batch_size=32` and `sampler_is=none`. A second hardcoded legacy check
  would also have rejected `frozenlake-dp8-tp8` and the one-unit DP8 geometry.
- Decision: admission now has two explicit contracts. Legacy P38 remains
  `frozenlake`, DP16, 8 x 4 prompts, token IS; the experiment-scoped
  `CANON_APC_M15_TARGET_DEBUG=off|on` path is exactly
  `frozenlake-dp8-tp8`, DP8, 1 x 32 prompts, no IS. Cross-use and partial
  geometries fail closed. This changes no APC or model arithmetic.
- Confirmed: Attempt 2 preserved the production `CANON_CONTINUE_DECODE=8`
  program and reached all four standard tensor-capture strata before the
  drain/tail switched to `_execute_continue_decode`. Removing the flag would
  change the executable being investigated and is therefore rejected.
- Confirmed: the same run saturated the 256 MiB incident ledger at call 326
  (`268,192,266` bytes) and later reached about 1,894 calls. The M15-only
  signed bound is raised to 2 GiB; ordinary P38 remains renderer-limited to
  128 MiB.
- Decision corrected by Attempt 3: tensor capture remains standard-only, but
  an M15 target may enter `continue_decode` before four standard strata exist.
  The observer therefore admits the registered path from its first call only
  into the dedicated full replay envelope. Generic tensor capture, request
  journal, and incident ledger remain standard-only. Full replay packaging
  requires A to attest both program paths and B to remain full-reset standard;
  unknown paths and non-M15 use still fail closed.
- Confirmed: Attempt 4 proves patch 28 reached the end of the real APC-on
  rollout: 2,560 requests completed with 92.5% prefix-cache hits. The run then
  stopped before A/B/C because the generic alignment admission omitted the
  already signed M15 `sampler_is=none` recipe. It is not a numerical verdict.
- Decision: no-IS admission is restricted to the exact M15 target identity:
  APC off/on selector, APC debug profile, M15/main, DP8xTP8, precheck-only,
  controlled exit, backward-no-commit, and zero commit. It requires rollout
  logprobs to be present, token-IS weights to be absent, and emits one
  fail-closed sampler receipt. Unsigned FrozenLake remains token-IS-only.
- Decision: Attempt 4 lacks a matched fresh control, so it cannot satisfy the
  paired decision. The newly rendered off/on arms may execute concurrently
  from one source SHA; only their interpretation is ordered. Classify off
  first, and use on for an APC-specific claim only after `CONTROL_GREEN`.
- Confirmed: Attempt 6 satisfies that paired decision. The off arm is
  `CONTROL_GREEN`; the on arm is `FRESH_TARGET_RED_FROZEN` with A-B=1,770
  bytes / 748 elements and B-C=0. No additional FrozenLake rollout is needed
  to prepare the next replay input.
- Decision: preserve four coordinates rather than collapsing them: the
  canonical first mismatch (row 201, completion position 0), the earliest
  request belonging to any red row (row 245, call 164), the request containing
  the canonical mismatch (row 201, call 187; its first output interval ends at
  call 188), and the later first fully captured tensor incident (row 245, call
  565). Call 565 is useful for tensor evidence but is not the mismatch onset.
- Confirmed: Attempt 11/d17 did not produce a first-red classification. Its
  wide observer accumulated roughly 2,100 records per arm, but the separate
  legacy incident ledger exceeded 2 GiB before A/B/C evidence and compact
  output could be sealed. The committed receipt/report do not substitute for
  classifier JSON or GCS terminal markers.
- Decision: inspect Attempt 9's registered GCS roots before changing code or
  launching again. Its receipt claims a completed one-round wide observation;
  the checked-in salvage script returns any real machine classifier and bundle
  verification without returning token-bearing payloads.
- Superseded decision: the pre-Attempt-12 one-round rule was correct while the
  classifier, shard root, replay copy and observer byte budget were all
  single-round. Attempt 12 exposed a distinct terminal-return risk. Phase D3
  therefore admits exactly three rounds only after isolating per-round shards,
  filtering cumulative ledgers, binding every receipt to its round, resetting
  only the per-round byte budget, and adding a small GCS recovery audit.
- Confirmed: the committed Attempt-12 return verifies only four derived files.
  It omits the three remote terminal markers, root manifest, compact-bundle
  binding, raw-log identity, and canonical on-arm classifier fields needed to
  reproduce the coarse verdict. This is an analysis-grade return, not proof
  that the runtime failed to upload those objects.
- Decision: before rendering the Layer-0 full observer, a bucket-capable
  executor must run the checked-in wide-seam GCS audit against the Attempt-12
  receipt. Only `LAYER_SELECTED` with both arms evidence-bound and no source
  conflict admits the next target run. `INCOMPLETE` or `SOURCE_MISMATCH` means
  evidence recovery, not a numerical fix and not another rollout.
- Corrected by the real Attempt-13 inventory: d32 is a single-round
  flat-shard generation, not a three-round `wide/rounds` generation. The old
  wrapper's `NO_DURABLE_ROUND` was a path/schema false negative. Phase D4 now
  verifies all `wide/shards` objects, selects a matching `live` snapshot, and
  replays exactly round 0 with the official classifier.
- Decision: d32 can establish at most one historical localization round. It
  cannot satisfy the three-round repeat gate or authorize a numerical repair.
  d33 is deferred until the flat replay is reviewed, unless that replay proves
  the required live input is genuinely absent.
- Correction: the 2026-08-28 executor stopped after a non-zero control
  `live/` listing. It did not audit treatment or run the official classifier.
  Non-zero is not equivalent to absence. The new read-only inventory queries
  both arms independently, validates only object geometry and small completion
  receipts, and emits a self-hashed `D32_INVENTORY.json`. Only a successful
  no-live review (currently `D32_LIVE_ABSENT_WITH_COUNT_DRIFT`) makes d33
  preparation eligible for separate review;
  `D32_LIVE_PRESENT_REPLAY_SHOULD_CONTINUE` returns to the flat replay; any
  query failure leaves both paths blocked.
- Correction after the sealed return: both recursive queries did succeed and
  prove that the roots contain no replayable `live/` objects. The shard
  completion totals are 2,445/2,188, while the immutable receipt's classifier
  seam totals are 2,474/2,087. These are now separate named metrics; neither is
  rewritten to match the other. d33 preparation is eligible because historical
  replay is physically impossible, not because d32 passed an official
  classifier. Launch and numerical repair remain unauthorized.
- Confirmed: the submitted d33 manifest is internally valid but covers only
  four derived files. It omits the six per-round official classifiers and all
  operator receipts required by D3. Therefore the reported
  `k_post_rope -> rpa_output` interval is analysis-grade and does not yet prove
  a particular RPA sub-operation or authorize Phase E.
- Decision: recover d33 in place before any relaunch. The submitted receipt is
  used only to locate the immutable source, JobSets, and object roots; all
  numerical claims must be regenerated from the official per-round artifacts.
- Correction after reviewing the recovered machine return: all seven listed
  payloads verify, but `MULTIROUND_SUMMARY.status=NO_DURABLE_ROUND`, both arms
  have zero sealed rounds, both JobSet queries failed, and both raw-log
  receipts are reported absent. No official per-round classifier was returned.
- Decision: Phase E remains closed. Before another target launch, replace the
  boolean remote probes with a receipt-bound read-only inventory that
  distinguishes permission/transient/query failure from not-found, directly
  stats `run.log`, and extracts only round-handshake/durability markers. More
  rollout steps are not a substitute for per-round seal durability.
- Confirmed by Attempt 17: D3c no longer conflates same-prefix concurrent
  requests. Three control rounds sealed exact; treatment Round 0 sealed with
  A-B=207 bytes / 95 elements and B-C=0, then returned a mixed
  `FIRST_RED_CANDIDATE_SET`. Treatment Round 1 failed assembly and Round 2 is
  absent, so the paired run is analysis-grade partial evidence.
- Decision: do not spend another 64-chip run before testing whether the sealed
  treatment bundle already contains enough future request history to bind
  source row 217. Absence never disambiguates a request; every eliminated
  candidate needs an explicit conflicting future prefix, and the selected
  proof must reach the latest elimination horizon.
- Decision: a read-only GCS/CPU reclassification and a future target launch
  are separate approvals. A preserved candidate set justifies observational
  provenance work, not a numerical repair. A unique offline localization must
  still be reviewed for last exact, first red, shape, coordinates, and source
  anchors before Phase E opens.
- Confirmed by the verified D3d return: source row 217 / completion position 0
  uniquely binds to A request `79-b8334848`; selected future-prefix proof 1300
  exceeds the required elimination horizon 1227. The former request-identity
  ambiguity is closed for this anchor.
- Confirmed: D3d still reports two global signatures across seven joinable red
  points, Layer-0 `rpa_output` and `final_norm`, with 88 red points explicitly
  unobserved. The completion-position-zero decision anchor itself is uniquely
  red at Layer-0 `rpa_output` with no exact-through alternative.
- Decision: Phase D3e makes completion-position-zero the declared classifier
  decision scope when `require_first_action=True`, while preserving all later
  signatures under separate `all_join_*` fields. Mixed or exact candidates
  within the decision scope remain fail closed. This is an analysis-accounting
  change, not a numerical repair.
- Confirmed by the committed D3e return: completion position zero is
  `FIRST_RED_LOCALIZED` at Layer 0 `k_post_rope -> rpa_output`, shape
  `[2048,1,15,8]`, source row 217 / position 1225 / A call 83. A-B is 207
  bytes / 95 elements and B-C is zero. The return remains analysis-grade
  partial because treatment rounds 1/2 and root completion are absent.
- Decision: the next discriminator is E0, not a speculative numerical repair.
  It captures all eight A aliases sharing the 1226-token prefix, restricts the
  live-KV fingerprint to Layer 0 and 77 valid pages, and uses later replay
  history to require one explicit future-prefix binding. Fingerprint equality
  is not a collision-free complete-byte proof.
- Decision: prepare-only render, official pinned exact-image, target launch,
  and compact GCS return are four distinct gates. The current local E0 tree
  authorizes none of the external gates. After publication, the other agent
  first runs the prepare wrapper on a clean exact-SHA worktree; pinned-image
  and DP8xTP8 launch each require separate user approval.
- Decision: E0 is observation only. `LIVE_KV_FINGERPRINT_DIFFERS` directs the
  next discussion toward cache production/storage/page ownership;
  `LIVE_KV_FINGERPRINT_EQUAL` directs it toward page-table/read/RPA execution
  context. Neither verdict authorizes changing RoPE, RPA, attention, KV values,
  production defaults, or the B full-reset judge.
