# P38 target handoff

Scope: this handoff covers only the GSM8K/FrozenLake P38 workstream. For the
parallel Qwen3-32B DeepSWE workstream, read
`../p39-deepswe-production/HANDOFF.md`. P38 evidence cannot promote P39, and
P39 evidence cannot promote P38.

## CURRENT: P38.2h fixed-lm-head backward-no-commit

P38s23r3 closed the forward candidate boundary: three 64-TPU FrozenLake
rounds measured 146,042 action tokens with exact A-B and B-C. It did not run
backward. The next and only P38 target is one actual-model DP16xTP4
backward-no-commit transaction under `P38H_BACKWARD_RUNBOOK.md`.

The first real-v5p M4096 VJP gate exposed a true backward defect: automatic
transpose of the outer 16xM256 map produced 11,950 different shared-weight
gradient elements (`max_abs=2.0`) against an explicit-order oracle while
`dHidden` remained exact. The local repair keeps the forward program and
accumulates the 16 completed M256 `dWeight` contributions with a loop-carried
ascending `lax.scan`. The rerun is fully exact for `dHidden`, `dWeight`, and
repeat determinism with finite/nonzero gradients and a live negative control.
Receipt: `artifacts/p38_2h_fixed_lm_head_vjp_onehost_0819.md`.

The checked-in target must be reviewed and published before launch. The
operator then runs only `P38H_BACKWARD_RUNBOOK.md` and returns its compact
SHA-sealed directory. Do not relaunch P38s23r3, use the historical P38
precheck renderer, or enable warning-only/full training yet.

## HISTORY: P38s23r3 64-TPU Three-Round Zero-Error Exact Pass (`P38S23R3_FORWARD_EXACT_PASS`)

P38s23r3 ran on 64 TPU (`canon-p38-fl-stock-p38s23r3-7c852e76`, `DP16xTP4`, concurrency 256, source `7c852e7660d165d2b4731f4e37ffa016f58db428`) under the `round-alignment-v1` lightweight durability profile.

### Admitted P38s23r3 Facts:
1. **Mechanical Classification Verdict**: **`P38S23R3_FORWARD_EXACT_PASS`** 🟢
2. **Bitwise Zero-Error Across All 3 Rounds ($146,042$ Action Tokens)**:
   - **Round 0**: $N_{\text{action}} = 47,230$, $S_{\text{decode}}$ vs $S_{\text{prefill}}$ = 0 differing bytes (`max_abs = 0.0`), $S_{\text{prefill}}$ vs $T_{\text{old}}$ = 0 differing bytes (`max_abs = 0.0`), Pearson $r = 1.00000$ 🟢. Sealed & verified (`b2fc7a41...`).
   - **Round 1**: $N_{\text{action}} = 47,998$, $S_{\text{decode}}$ vs $S_{\text{prefill}}$ = 0 differing bytes (`max_abs = 0.0`), $S_{\text{prefill}}$ vs $T_{\text{old}}$ = 0 differing bytes (`max_abs = 0.0`), Pearson $r = 1.00000$ 🟢. Sealed & verified (`92ce69be...`).
   - **Round 2**: $N_{\text{action}} = 50,814$, $S_{\text{decode}}$ vs $S_{\text{prefill}}$ = 0 differing bytes (`max_abs = 0.0`), $S_{\text{prefill}}$ vs $T_{\text{old}}$ = 0 differing bytes (`max_abs = 0.0`), Pearson $r = 1.00000$ 🟢. Sealed & verified (`fe34043a...`).
   - **Cumulative**: **$146,042$ action tokens, 0 byte discrepancy ($A = B = C$) across all tokens**.
3. **P38 Fixed-LM-Head PATHTRACE Compilation**:
   - All 7 receipts verified (`semantic_M = 16, 32, 64, 128, 256` padded to `fixed_M = 256` with `chunks = 1`, and `semantic_M = 4096` tiled into 16 `fixed_M = 256` tiles with `chunks = 16`).
   - Serving decode and learner prefill shared the identical `[256, 4096] @ [4096, 38144]` fixed Pallas tile without secondary compilation or ValueError.
4. **Frozen Diagnostic Contract & Controlled Exit**:
   - `backward = 0, optimizer_commits = 0` across all 3 rounds.
   - `[CANON_P38] CONTROLLED_EXIT code=42 backward=0 optimizer_commits=0` executed cleanly.
5. **Durability & Sealed Evidence Bundle**:
   - All 3 round archives (`ROUND_ARCHIVE.tar`, `ROUND_COMPLETE.json`, `SHA256SUMS`) sealed in GCS.
   - Evidence bundle verified at `evidence/p38s23r3/`.
   - Comprehensive receipt: `artifacts/p38s23r3_forward_exact_pass_report.md`.
   - Mechanical classifier output: `evidence/p38s23r3/verdict.json`.

## HISTORY: P38s23r2 64-TPU first round exact; durability timeout

P38s23r2/source `6814774eef70aa0c67610eab9f355d964d420378` (*Map learner lm-head through fixed Pallas chunks*) ran on 64 TPU (`DP16xTP4`, concurrency 256, `CANON_P38_FIXED_LM_HEAD=1`).

### Admitted P38s23r2 facts:
1. **Compilation & Tile Sharing**:
   - All 7 PATHTRACE receipts present (`M=8, 16, 32, 64, 128, 256` chunks=1, and `M=4096` chunks=16).
   - Serving decode and learner prefill shared the identical `[256, 4096] @ [4096, 38144]` fixed Pallas tile without secondary compilation or ValueError.
2. **Bitwise Zero-Error Exactness (Round 1)**:
   - `N_action = 49,177`
   - `S_decode_vs_S_prefill`: **0 differing bytes, 0 differing elements, `max_abs=0.0`** (100% bitwise exact)
   - `S_prefill_vs_T_old`: **0 differing bytes, 0 differing elements, `max_abs=0.0`** (100% bitwise exact)
   - `sampler-trainer`: `logp_diff=(0.00000, 0.00000)`, `prob_diff=(0.00000, 0.00000)`, `pearson=1.00000`
   - `[CANON_P38] PRECHECK_ROUND_COMPLETE round=1/3 step=0 N_action=49177 verdict=PASS a_b_differing_bytes=0 backward=0 optimizer_commits=0`
3. **Run Termination Status**:
   - After Round 1 PASS, `_seal_p38_diagnostic_round(round_index=0)` timed out after 900 seconds waiting for GCS durability worker ACK (`AlignmentGateError`).
   - Run verdict: `INCONCLUSIVE_DURABILITY_SEAL_TIMEOUT` with `ANALYSIS_GRADE_ROUND1_EXACT_PASS`.
   - Evidence bundle sealed at `evidence/p38s23r2/`. Receipt: `artifacts/p38s23r2_round0_seal_timeout_report.md`.

## HISTORY: P38s23r1 learner-shape contract failure

### Admitted P38s22 facts

1. B-C is 100% bitwise exact (0 differing bytes, max_abs=0.0) in all 3 completed rounds.
2. A-B remains sparse red in all three rounds:
   - Round 0: 48 elements / 82 bytes across 45,865 actions; max_abs 0.263157;
   - Round 1: 10 elements / 14 bytes across 43,982 actions; max_abs 0.0160103;
   - Round 2: 8 elements / 15 bytes across 53,617 actions; max_abs 0.289223.
3. The `BF16_BF16_F32` dot algorithm preset does not close the A-B decode vs prefill carrier.
4. Per the decision table, `CANON_MM_ALGO` is rejected as a causal repair. Do not tune more generic precision flags.
5. The independent-round salvage audit passed. The current scientific step is
   the default-off fixed-tile Pallas `lm_head` described in
   `phases/p38-2x-fixed-tile-pallas-lm-head.md`.

Receipt:
`artifacts/p38s22_analysis_0818.md`.

Do not cite `evidence/p38s22/p38_terminal.classification.json` as P38s22
evidence unless a future mechanical receipt returns the missing raw terminal
inputs and provenance. Under the current no-observer contract it is
unadmitted; P38s21 remains the admitted lm-head interval localization.

## HISTORY: P38s21 partial 2-of-3 diagnostic

P38s21/source `f7f9fee6256f1f31f99c2794c4f346e9196b010c` ran on 64 TPU
(`DP16xTP4`, concurrency 256). It sealed rounds 0 and 1, then Round 2 exceeded
the 4-GiB local terminal-evidence bound. There is no third round, controlled
exit, root `COLLECTED`, or root `COMPLETE`. The run-level status is
`ANALYSIS_GRADE_PARTIAL_2_OF_3`, not `CLASSIFICATION_COMPLETE`.

### Admitted P38s21 facts

1. B-C is byte-exact in both completed rounds.
2. A-B remains sparse red:
   - Round 0: 47 elements / 76 bytes across 45,276 actions;
   - Round 1: 7 elements / 9 bytes across 44,695 actions.
3. All 54 selected red points join. Their captured complete final-hidden rows
   are byte-exact, and the first measured red interval is `lm_head_logits`.
4. The exact-hidden statement applies to those 54 selected points. The bundle
   does not prove every token or every internal backbone checkpoint.
5. `lm_head_logits` is interval localization, not a root mechanism. In the
   lm-head equation `TD,DV->TV`, hidden K=4096 is reduced; vocabulary V=151,936
   is an output axis. Summation over vocabulary is not the operation measured.

Receipt:
`artifacts/p38s21_analysis_0818.md`.

## HISTORY: P38s20 bounded-object transport timeout

## HISTORY: P38s18r2 target-aware tail reduction

P38s18r2/source `10fe951f0186...` ran on 64 TPU (`DP16xTP4`, concurrency 256,
three frozen rounds, seam mode `layer`, terminal tail enabled). Round 0 reached
the numerical precheck, but the learner timed out after 900 seconds waiting for
the durability ACK. The worker took about 57 minutes to serially upload and
download/verify 3,776 small objects and wrote the ACK only after the learner had
exited. Rounds 1 and 2 never started.

### Admitted facts

- **S_prefill vs T_old (B vs C)**: **0 differing bytes** across 45,559 action
  tokens (100% bitwise exact identity between training forward pass and vLLM
  prefill rescore on 64 TPU).
- **S_decode vs S_prefill (A vs B)**: **45 differing bytes / 32 differing
  elements** across 45,559 tokens, with
  `max_abs=0.10101699829101562`.
- The committed pre-alignment input rejects the old hand-written boundary
  claim: only **1 of 32** mismatch elements has
  `logical_kv_prefix_length % 256 == 0`. No Pallas-boundary root cause is
  admitted.
- **GCS Round 0 seal (returned manifest receipt)**: the worker
  reported `ROUND_COMPLETE` for
  `gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/canon-p38-fl-stock-p38s18r2-10fe951f/attempt-0/rounds/000000`
  with `manifest_sha256 = ce7df453259dd070472486e053dbb26b03dad7b6259784cde74da7fe9efe227e`.
  Commit `a514c3bf` returns a 3,896-object listing and 3,894-entry source
  manifest whose filename sets close exactly after accounting for the two
  sealing files. It reports 972 paired seam and 972 paired tail records.
- **Direct remote classification failed closed**: the official classifier
  returned rc 1 at `duplicate seam token-prefix record`, before producing
  `classification.json`. The returned verdict is
  `INCONCLUSIVE_REMOTE_CLASSIFICATION`. This is an analysis-workflow mismatch,
  not evidence that the source files are missing or corrupt.
- **The committed v2 compact bundle is complete but scientifically
  inconclusive**: 371/371 SHA entries and its independent audit pass. It joins
  64/64 seam keys and 63/64 tail keys. The sole tail conflict shares one
  source-prefix SHA but mixes target 54852 (required by the capsule; two
  byte-identical aliases) with unrelated target 13598. Tail identity was
  under-specified; source bytes are not contradictory.
- **Exploratory target-aware reclassification, not yet admitted evidence**:
  all 32 red points join. The first measured difference is
  `raw_log_normalizer` for 26 points and `raw_target_logit` for 6 points; all
  recorded layer/final-norm fingerprints remain equal. This must be reproduced
  by the immutable v3 wrapper before it enters the evidence ledger.
- **Real one-host construction control**: identical `[256,151936]` float32
  logits passed through two outer TPU programs and the same canonical
  log-softmax with 0/38,895,616 different elements; a one-bit negative reported
  exactly one. Therefore a same-input canonical-reducer construction bug is
  not sufficient to explain production.
- **Overall verdict**: `INCONCLUSIVE_DURABILITY_SEAL_TIMEOUT`. There is no
  controlled exit 42, root `COLLECTED`/`COMPLETE`, or three-round target
  verdict. The Round 0 numbers are analysis-grade.

### Operational constraint

Raw seam/tail NPZ files stay in GCS. Do not ask the local evaluator to list or
download the bucket, and do not manually summarize thousands of records. Do
not rerun the direct whole-directory classifier or immutable v2 reducer. A
GCS-authorized agent must use the target-aware seam-plus-tail reduction
registered in `phases/p38-2t-target-aware-tail-join.md`, then return the
new compact byte-preserving bundle defined by
`P38S18R2_ALIAS_REDUCTION_RUNBOOK.md`.

The target-aware amendment is complete: reducer, independent auditor, new
immutable v3 contract, one-command GCS wrapper, and focused/fake-GCS positive
and negative controls. Publication was explicitly approved on 2026-08-17.
Remote execution waits only for a clean checkout containing this CL.

### Exact next steps for the incoming agent

1. **Do not launch TPU and do not overwrite any source or v1 derived object.**
2. Confirm the clean checkout contains the published target-aware Stage A CL
   and read `P38S18R2_ALIAS_REDUCTION_RUNBOOK.md`.
3. Run the exact Stage B wrapper command once. Do not hand-build arguments;
   use `scripts/p38s18r2_round0_target_join_contract.json`.
4. Require 32 red points, 64/64 seam keys, 64/64 tail keys, no same-target
   payload conflicts, mandatory target-aware tail join, and standalone auditor
   PASS. Wrong-target candidates remain preserved as provenance.
5. Return the wrapper-created
   `evidence/p38s18r2/seam-tail-target-aware-v3/` directory, which contains
   every raw candidate for the required keys, capsule, classifier output, and
   audit JSON. Raw unrelated records remain in GCS.
6. Stop and report. Do not commit or push the returned evidence until the user
   explicitly approves that separate action.

## HISTORY: P38s18l/P38.2q no-source inventory

P38s18l/source `9a83457417fc` ran at DP16xTP4/concurrency 256 with zero
backward and optimizer commits, but the committed package is not a complete
three-round run:

- the raw log contains exactly two `PRECHECK_ROUND_COMPLETE` markers and two
  pre-alignment records;
- it ends during the third rollout and contains no terminal
  `PRECHECK_COMPLETE STOP_BEFORE_BACKWARD`;
- immutable capsules exist only for rounds 0 and 1;
- those two completed rounds reproduce A-B red at 19 and 28 elements while
  B-C remains exact;
- no raw `p38_seam_*.json/.npz` is committed, so the official classifier
  cannot reproduce the committed PASS JSON; and
- `LIVE.json`, `COLLECTED.json`, `COMPLETE.json`, `PACKAGING.txt`, and
  `verdict.json` are absent from the committed directory.

The existing hand-authored result says 20 of 47 red points have equal layer
fingerprints. Treat that only as a candidate tail direction. It does not prove
all red points, full hidden bytes, lm_head, or the normalizer.

The first GCP reduction (`v1`) is immutable analysis evidence, not an admitted
classification:

- it verified 2,441 source files and scanned 1,217 seam records;
- it selected snapshot `000020`, which contains only capsule round 0 and 19
  red points rather than the two completed rounds / 47 red points;
- 37 of 38 round-0 A/B keys were unique under the old join;
- one A key hit both records 319 and 398, so the reducer correctly returned
  `INCONCLUSIVE_REDUCTION_JOIN`; and
- no official classification was produced. The phrase “confined to the tail
  normalizer” is withdrawn; the tail has not been subdivided yet.

The first v2 selector execution inventoried 22 live snapshots and returned
rc=4: `000020` contains only round 0, while `000021` contains capsule names for
rounds 0 and 1 but lacks `LIVE.json`, `SHA256SUMS`, and paired seam NPZs. No
snapshot satisfies the two-round source contract. Commit `e0c1aef7` recorded
that result only as prose; it did not return `SNAPSHOT_SELECTION.json` or the
raw object listing, so the inventory is not yet mechanical evidence.

### Next operator step: no TPU launch

Run the amended GCP-side wrapper described in:

`P38S18L_GCP_REDUCTION_RUNBOOK.md`

The v2 wrapper inventories every snapshot and automatically requires at least
immutable rounds 0 and 1 before downloading. It records every candidate row,
admits duplicate records only when their position/token/checkpoint metadata and
all layer/final fingerprints are identical, preserves conflicts fail-closed,
and uploads a compact self-contained package. When no snapshot qualifies, it
uploads a selection-only package with the raw object listing, selector
JSON/stdout/stderr, verdict, and SHA inventory before exiting 4. A separate
bundle auditor verifies every SHA and either re-runs the selector or the
official classifier from only those returned files. It never modifies source
objects or fabricates round 2 or terminal markers.

The remote agent must return and prepare an append-only evidence CL containing
the entire uploaded bundle, not just audit metadata. For the expected no-source
outcome, this means `OBJECT_LISTING.txt`, selector JSON/stdout/stderr, verdict,
packaging note, and SHA manifest, with standalone auditor PASS. If a source is
unexpectedly admitted, only a bundle with all 47 red points / 94 arm keys joined
may select a next diagnostic:

- observed hidden/final fingerprints exact -> build one bounded tail observer;
- any hidden checkpoint red -> withdraw the tail claim and localize the first
  measured hidden seam;
- missing/ambiguous join -> keep P38s18l partial and do not promote it.

The no-source outcome retires P38s18l and **does not** authorize a tail-only
probe. The separately registered successor is
`phases/p38-2r-terminal-seam-tail-acquisition.md`: one production-shape run
captures hidden seams and bounded-tail checkpoints together and seals every
round before continuing. It remains unlaunched and requires its one-host
observer-neutrality gate first.

## HISTORY: completed P38s17, P38s16, P38s15, P38s14, P38s13a, P38s12f

### P38s17: valid live/clean KV fingerprints are equal

P38s17/source `baac38bc4034` completed all three Frozen-Weight diagnostic
rounds on 64 TPU. Recomputed Live-KV Observer classification confirmed
`live_kv_fingerprint_equal_on_red_row`.
Evidence directory: `tasks/p38-pathways-decode-prefill-carrier/evidence/p38s17/`.

### Next operator step after review and publication approval

1. Use a new production-shape stock acquisition to obtain a naturally
   single-active mismatch record with patch 15's fixed-M geometry and exact
   token history. Do not change concurrency, DP/TP, canonical M, or padding.
2. Reject any record whose input aval collapses to one row. This is a different
   executable, even if its scheduler occupancy is one.
3. Treat the new record as a discriminator, not a root-cause result. Choose
   exactly one decisive observer next: neutral live-KV content comparison, or
   (if KV is exact) the first-divergence seam walk from q_norm through the
   normalizer.
4. Do not rerun concurrency 32, KV-unified, or the DP1 E0-lite arm.

Historical status at P38.2m: no target launch was then authorized. P38.2n has
since completed the live/clean observer gate; use only the CURRENT P38.2o
section above. The P38s17 launch block in the runbook is historical.

## HISTORY: completed P38s15, P38s14, P38s13a, P38s12f

This section is retained for provenance only. No new target command is admitted
until P38.2o's local evidence and observer-neutrality gates pass.

### P38s12d is void; do not rerun its source

P38s12d used concurrency 32 but source `bdc96818` retained the recipe's old
hard-coded concurrency-256 assertion. It failed before rollout with
`P32 FrozenLake geometry mismatch: {'max_concurrency': 32}`. Missing capture
files and later stale-evidence messages are downstream symptoms. It has no
numerical verdict.

The repaired source admits concurrency 32 only for the exact stock P38
backward-no-commit capture envelope. Full training, evaluation, DP8xTP8, and
KV-unified paths still require concurrency 256. Fetch a commit containing
`validate_frozenlake_max_concurrency`; never edit/reuse the P38s12d YAML.

### P38s12e is duplicated P38s12d evidence, not a target run

The committed P38s12e directory is transport-complete but semantically wrong.
All 365 `[sync] HEAD` records are source `bdc96818`; all JobSet/state paths say
P38s12d. The file is exactly five copies of one 199-line pod log plus 360
copies of one 113-line pod log. It contains five geometry failures followed by
360 stale-`run.log` failures, an empty pre-alignment file, no capture, and five
concatenated classification JSON objects. It cannot answer the concurrency
question and must not be cited as a new experiment.

Use a completely new run id, `p38s12f`. The updated runbook rejects an old
source, an existing JobSet, duplicate source/command markers, old run ids,
empty/multi-object JSON, and append-collected `head.full.log` evidence before
sealing.

### Account the returned bundle correctly

Evidence commit `23bb2a3c` is named `p38s12b`, but its command used
`--max_concurrency=256`. Account it as **P38s12a analysis-level evidence**:
core numerical/capture artifacts are internally reproducible, A-B is red and
B-C exact, but `rc=137`, missing infrastructure artifacts, one omitted ninth
red row, and a stale `SHA256SUMS` self-entry prevent formal run admission.
Do not relaunch this arm and do not cite it as concurrency 32.

### Completed local action

Source row 231 was run from the returned capsule on the authorized one-host
v5p. Do not rerun it. The command is retained for provenance:

```bash
bash \
  canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/run_p38_frozenlake_replay.sh \
  /path/to/p38s12a-mismatch-capsule.npz \
  p38s12a_row231_e0lite 231
```

The preregistered result landed in the second branch:

- `E0_LITE_ENVELOPE_NOT_REPRODUCED`: REF reconstructed B/T-old exactly across
  566 action values, but R0/R1 each missed production A at 470 values. R0/R1
  and repeats were internally exact, and the negative control fired.

Therefore stop interpreting mask-derived operator counterfactuals. Do not
start the RoPE/RPA/residual first-divergence walk from this replay. Full values
and SHA identities are in
`artifacts/p38_2j_row231_e0lite_0813.md`.

### Target next action after publication/resource approval

Run one **stock-only concurrency-32** P38s12f. Before apply, render a same-source
concurrency-256 baseline and require `check_p38_intent_diff.py` PASS. Apply
only the concurrency-32 YAML. The new source also requires:

- controlled target exit 42; `137` remains infrastructure-inconclusive;
- capsule capacity 16;
- host-derived logical-KV depth at least 1686;
- the complete Kubernetes/Pathways bundle, including both rendered intents
  and the intent-diff report; and
- `seal_p38_evidence.sh` PASS, with no self-hash in `SHA256SUMS`.

If concurrency 32 is exact and depth-sufficient, repeat it once. A red result
only proves that small concurrency is insufficient; repeated exact only makes
concurrency/churn a necessary trigger. Neither identifies an operator.

P48 remains separate and waits for DP16 capacity. Do not borrow its evidence,
resources, or admission claims for P38.

---

## HISTORY: P38s12a stock request-journal capture

The section below is retained only as historical operator evidence. Do not run
it now.

### Goal and non-goals

P38s11 already reproduced the full-coverage stock carrier. P38s12a keeps that
known-red environment unchanged at 32 prompts x 8 generations, engine DP16,
and concurrency 256. It changes only evidence selection:

- up to eight red rows instead of two;
- four reachable request-prefix bands:
  `1536,1664,1792,1920,2048`;
- a host-only per-request journal containing exact token history, request/DP/
  slot, physical blocks, co-batch membership, and observational page
  generations;
- exact journal joins for every selected red row.

It does not fetch/hash device KV content, run backward, commit an optimizer,
enable prefix cache, change precision/kernels, or rerun unified KV. Do not add
any of those manually.

### 1. Fetch one immutable source and render stock only

Run these commands sequentially from an existing clean `google/tunix` clone:

```bash
set -euo pipefail
git fetch origin yuxzhang/canon-zero-tim
SOURCE_COMMIT="$(git rev-parse FETCH_HEAD)"
RUN_ID="p38s12a"
WORKTREE="/tmp/canon-zero-tim-$RUN_ID"
OUT="/tmp/p38-serving-$RUN_ID"
EVIDENCE="/tmp/p38-return-$RUN_ID"
test ! -e "$WORKTREE"
test ! -e "$OUT"
test ! -e "$EVIDENCE"
git worktree add --detach "$WORKTREE" "$SOURCE_COMMIT"
cd "$WORKTREE"
test "$(git rev-parse HEAD)" = "$SOURCE_COMMIT"
test -z "$(git status --porcelain)"
rg -q '_p38_request_journal' \
  canon-zero-tim/patches/tpu_inference/13-tpu-runner-p38-request-journal.patch
test -f \
  canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/phases/p38-2i-request-journal-concurrency-discriminator.md
rg -q 'CANON_P38_MISMATCH_CAPSULE_MAX_ROWS.*8' \
  canon-zero-tim/cluster/render_p38_serving_jobsets.py
mkdir -p "$EVIDENCE"
printf '%s\n' "$SOURCE_COMMIT" > "$EVIDENCE/source_commit.txt"

python3 canon-zero-tim/cluster/render_p38_serving_jobsets.py \
  --source-commit "$SOURCE_COMMIT" \
  --run-id "$RUN_ID" \
  --output-dir "$OUT" \
  --stock-only | tee "$EVIDENCE/render.txt"

STOCK="$OUT/jobset-p38-serving-stock.yaml"
test -f "$STOCK"
test ! -e "$OUT/jobset-p38-serving-unified.yaml"
cp "$STOCK" "$EVIDENCE/rendered-stock.yaml"
kubectl apply --dry-run=server -f "$STOCK" | \
  tee "$EVIDENCE/dry-run-stock.txt"
```

The rendered stock YAML must contain these literal contracts:

```text
CANON_KV_UNIFIED=0
CANON_P38_PRECHECK_ONLY=1
CANON_P38_MISMATCH_CAPSULE_MAX_ROWS=8
CANON_P38_SERVING_CAPTURE_MAX_CALLS=4
CANON_P38_SERVING_CAPTURE_MIN_PREFIX=1536
CANON_P38_SERVING_CAPTURE_PREFIX_BOUNDS=1536,1664,1792,1920,2048
CANON_P38_SERVING_CAPTURE_EXPECTED_RECORDS=4
CANON_P38_SERVING_CAPTURE_EXPECTED_PATH=standard
CANON_P38_REQUEST_JOURNAL=<capture-dir>/p38_request_journal.jsonl
CANON_RUN_CMD contains --batch_size=32
CANON_RUN_CMD contains --mini_batch_size=4
CANON_RUN_CMD contains --num_generations=8
CANON_RUN_CMD contains --mesh_dp=16
maxRestarts: 0
```

If any value differs, stop. Do not edit the YAML.

### 2. Apply stock only and preserve a byte-zero terminal log

```bash
set -euo pipefail
kubectl apply -f "$STOCK" | tee "$EVIDENCE/apply.txt"

JOBSET="canon-p38-fl-stock-${RUN_ID}-${SOURCE_COMMIT:0:8}"
HEAD_JOB="${JOBSET}-pathways-head-0"
POD=""
for unused in $(seq 1 180); do
  POD="$(kubectl get pods -n default -l "job-name=$HEAD_JOB" \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
  [ -n "$POD" ] && break
  sleep 10
done
test -n "$POD"
printf '%s\n' "$JOBSET" > "$EVIDENCE/jobset-name.txt"
printf '%s\n' "$POD" > "$EVIDENCE/head-pod-name.txt"

set +e
kubectl logs -n default -f "$POD" -c jax-tpu | \
  tee "$EVIDENCE/head.follow.log"
follow_rc="${PIPESTATUS[0]}"
set -e
printf '%s\n' "$follow_rc" > "$EVIDENCE/log-follow-rc.txt"
```

The end of `kubectl logs -f` is not a terminal verdict. Wait for the exact
JobSet/pod to become `Completed` or `Failed`; do not delete it. Then fetch all
evidence from byte zero:

```bash
set -euo pipefail
kubectl get jobset -n default "$JOBSET" -o yaml > \
  "$EVIDENCE/jobset.final.yaml"
kubectl get pod -n default "$POD" -o yaml > \
  "$EVIDENCE/head-pod.final.yaml"
kubectl describe pod -n default "$POD" > \
  "$EVIDENCE/head-pod.describe.txt"
kubectl logs -n default "$POD" -c jax-tpu > \
  "$EVIDENCE/head.full.log"
kubectl logs -n default "$POD" -c pathways-proxy > \
  "$EVIDENCE/pathways-proxy.log" 2>&1 || true
kubectl logs -n default "$POD" -c pathways-rm > \
  "$EVIDENCE/pathways-rm.log" 2>&1 || true
kubectl logs -n default "$POD" -c jax-tpu --previous > \
  "$EVIDENCE/head.previous.log" 2>&1 || true
kubectl get events -n default \
  --field-selector "involvedObject.name=$POD" \
  --sort-by=.lastTimestamp > "$EVIDENCE/head-pod.events.txt"
```

Do not use `--tail`, timestamps, UI excerpts, or pasted terminal fragments as
the canonical log. If the pod disappears, return the partial infrastructure
package and classify it `INCONCLUSIVE`; do not relaunch automatically.

### 3. Admission checklist

The unedited `head.full.log` must contain all of the following:

1. Attempt 0 and the exact source SHA;
2. exactly one standard-path capture INIT and positive OBSERVE count;
3. exactly four pre/post capture pairs, one per new prefix band;
4. one full 32-prompt / 256-trajectory coverage PASS;
5. finite A-B red and exact B-C;
6. zero capture errors;
7. one terminal `PRECHECK_COMPLETE STOP_BEFORE_BACKWARD`;
8. positive `[CANON_P38_REQUEST_JOURNAL]` markers;
9. classifier PASS with
   `request_journal_joined_source_rows == selected_rows`;
10. mismatch capsule, classification JSON, and serving archive base64;
11. outer acceptance with backward=0 and optimizer_commits=0; and
12. final PATHTRACE with `p38_kv_unified=0`, journal>0, and coverage=1.

Missing row joins are `INCONCLUSIVE_CAPTURE_SELECTION`. A different A-B count
is not by itself a failure because rollout trajectories are stochastic. B-C
red, invalid/nonfinite data, missing coverage, capture errors, or any backward
or optimizer commit are hard failures.

### 4. Extract and return the exact bundle

```bash
set -euo pipefail
python3 \
  canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/extract_p38_capsule.py \
  --log "$EVIDENCE/head.full.log" \
  --output "$EVIDENCE/p38s12a-mismatch-capsule.npz"
python3 \
  canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/extract_p38_serving_archive.py \
  --log "$EVIDENCE/head.full.log" \
  --output "$EVIDENCE/p38s12a-serving-capture.tar"
sed -n 's/^\[CANON_PRE_ALIGN_ARTIFACT_JSON\] //p' \
  "$EVIDENCE/head.full.log" > "$EVIDENCE/pre-alignment.jsonl"
sed -n 's/^\[CANON_P38_SERVING_CLASSIFICATION_JSON\] //p' \
  "$EVIDENCE/head.full.log" > "$EVIDENCE/serving-classification.json"
test -s "$EVIDENCE/pre-alignment.jsonl"
test -s "$EVIDENCE/serving-classification.json"
tar -tf "$EVIDENCE/p38s12a-serving-capture.tar" | \
  grep -q './p38_request_journal.jsonl'
sha256sum "$EVIDENCE"/* | tee "$EVIDENCE/SHA256SUMS"
```

Return the entire `$EVIDENCE` directory. Do not return screenshots or only the
last 200 lines. Do not launch P38s12b yet. After this bundle is reviewed, the
next gate is exact whole-vector E0 replay; concurrency 32 is a separate later
arm with a KV>=1686 sufficiency check.

## NEXT AFTER REVIEW/PUBLICATION: run one P38s11 full-coverage stock capture

Do not launch from the current remote until the local P38.2g9 changes are
reviewed and published. The source used for P38s11 must contain
`p38-2g9-full-coverage-key-capture.md`, patch 12, and the full-coverage consumer
marker. Do not relaunch the P38s10 source.

P38s10 reached a real terminal precheck and measured A-B=B-C=0, but only for
the first four prompts (`32` trajectories, `N_action=2731`, solve ratio 1.0).
P38s1/P38s2 measured all `256` trajectories and placed sparse red rows mostly
outside that subset. P38s10 is therefore an exact subset PASS, not evidence
that the carrier was repaired. It also emitted three typed-PRNG-key capture
errors and returned no admitted serving archive.

This is the first attempt whose operator contract treats one terminal,
byte-zero log as a prerequisite instead of inferring a verdict from a pasted
excerpt.
Do not call the previously committed P38s8 excerpt a complete log:

- `p38_p38s5_head_full.raw.log` and `p38_p38s6_head_full.raw.log` in
  `42139ffa` are byte-for-byte copies of the already audited s5/s6 logs. They
  add no new evidence.
- `p38_p38s8_frozenlake_stock.raw.log` has 1,437 lines and 173,137 bytes, starts
  inside a device-memory report, and ends during the first canonical model
  compilation. It has one standard-path INIT marker but no byte-zero
  source/Attempt-0 preamble, OBSERVE, capture, alignment, child exit,
  classifier, archive, or outer postflight. Its verdict is
  `INCONCLUSIVE_PARTIAL_EXCERPT`.
- The old explanation that s8 merely missed `min_prefix=1536` is withdrawn.
  The runner emits `CANON_P38_SERVING_CAPTURE_OBSERVE` before applying the
  prefix-stratum filter. A terminal full log with INIT but no OBSERVE would
  indicate that the standard hook was not reached; the partial s8 excerpt
  cannot decide that question.

This section supersedes every older P38 serving-capture launch command below.
The committed `p38_p38s4_frozenlake_stock.raw.log` is not a complete run log:
it contains exactly 200 tail lines, starts inside layer 30, and ends after the
final RMSNorm without a workload exit, serving capture, classifier, archive,
or final postflight. It is `INCONCLUSIVE` and must not be used to select a
repair.

### Why this remains a strict diagnostic instead of a P45 sidecar

A local RoPE decode-shape versus prefill-shape comparison is a useful cheap
screen and may run in parallel, but it cannot replace this capture. A nonzero
operator result would show that one RoPE aval pair can drift; it would not
prove that RoPE carries the production A-B boundary. An exact result would not
exclude wrong production positions or a different outer fusion envelope. E0
whole-vector reproduction remains mandatory before selecting a repair.

Do not inject the current P38 environment into P45 full training. The published
P38 contract deliberately requires `backward-no-commit`,
`CANON_P38_PRECHECK_ONLY=1`, exactly four records, and a classifier/archive
postflight. P45 is a warning-only committed-training profile. Combining them
without a separately reviewed shadow-capture design would either fail the
environment contract or let capture/postflight failure interrupt production.
P38s11 is therefore the next diagnostic after publication; a nonblocking P45 shadow
capture is a future code change, not an operator YAML edit.

P38s6 is also inconclusive. It initialized the patched module but emitted zero
observations because the hook existed only in `_execute_continue_decode`, while
FrozenLake uses standard `_execute_model` with `enable_continue_decode=False`.
Its log also ends without alignment, terminal precheck, classifier, archive,
or outer postflight. Lowering the prefix threshold cannot repair an unreachable
hook. P38s7 then reached the real standard hook, but the 32-group diagnostic
consumer accepted a five-group partial tail and passed 40 trajectories to the
DP16 adapter. P38s11 keeps each producer unit at 4 x 8 = 32 trajectories so
every unit is DP16-divisible, but the consumer waits for all eight units before
alignment. The measured batch is therefore all 32 prompts / 256 trajectories;
a partial tail is rejected. The operator's only job is to run one fresh
**stock-only** standard-path P38 diagnostic and return the complete evidence
bundle. Do not launch unified
KV, FrozenLake full training, GSM8K, backward, or an optimizer commit. Do not
force-enable continue-decode because that changes the program being diagnosed.

### 1. Fetch one immutable source and render

Run every command block in this section sequentially in the same Bash shell
from an existing clone of `google/tunix`. Use a new run ID if `p38s11` already
exists. Do not reuse or overwrite an earlier output directory.

```bash
set -euo pipefail
git fetch origin yuxzhang/canon-zero-tim
SOURCE_COMMIT="$(git rev-parse FETCH_HEAD)"
git merge-base --is-ancestor \
  4a2cb8cd2bff2e1e9f5f82a6d2e0575d166759bd "$SOURCE_COMMIT"

RUN_ID="p38s11"
WORKTREE="/tmp/canon-zero-tim-$RUN_ID"
OUT="/tmp/p38-serving-$RUN_ID"
EVIDENCE="/tmp/p38-return-$RUN_ID"
test ! -e "$WORKTREE"
test ! -e "$OUT"
test ! -e "$EVIDENCE"
git worktree add --detach "$WORKTREE" "$SOURCE_COMMIT"
cd "$WORKTREE"
test "$(git rev-parse HEAD)" = "$SOURCE_COMMIT"
test -z "$(git status --porcelain)"
rg -q 'program_path="standard"' \
  canon-zero-tim/patches/tpu_inference/10-tpu-runner-p38-standard-capture.patch
rg -q '_p38_capture_leaf' \
  canon-zero-tim/patches/tpu_inference/12-tpu-runner-p38-prng-key-capture.patch
rg -q 'CANON_P38_SERVING_CAPTURE_EXPECTED_PATH' \
  canon-zero-tim/cluster/render_p38_serving_jobsets.py
rg -q '_DIAGNOSTIC_UNITS = 8' \
  canon-zero-tim/cluster/render_p38_serving_jobsets.py
rg -q 'DIAGNOSTIC_COVERAGE_CONTRACT' \
  tunix/rl/agentic/agentic_rl_learner.py
rg -q 'stop_after_diagnostic_precheck' tunix/rl/alignment.py
mkdir -p "$EVIDENCE"
printf '%s\n' "$SOURCE_COMMIT" > "$EVIDENCE/source_commit.txt"

python3 canon-zero-tim/cluster/render_p38_serving_jobsets.py \
  --source-commit "$SOURCE_COMMIT" \
  --run-id "$RUN_ID" \
  --output-dir "$OUT" | tee "$EVIDENCE/render.txt"

STOCK="$OUT/jobset-p38-serving-stock.yaml"
UNIFIED="$OUT/jobset-p38-serving-unified.yaml"
cp "$STOCK" "$EVIDENCE/rendered-stock.yaml"
kubectl apply --dry-run=server -f "$STOCK" | \
  tee "$EVIDENCE/dry-run-stock.txt"
kubectl apply --dry-run=server -f "$UNIFIED" | \
  tee "$EVIDENCE/dry-run-unified.txt"
```

The renderer emits both YAML files, but the operator must apply **only**
`$STOCK`. Before applying, verify that the stock YAML contains all of these
literal values:

```text
CANON_KV_UNIFIED=0
CANON_P38_PRECHECK_ONLY=1
CANON_P38_SERVING_CAPTURE_MAX_CALLS=4
CANON_P38_SERVING_CAPTURE_MIN_PREFIX=1536
CANON_P38_SERVING_CAPTURE_PREFIX_BOUNDS=1536,1792,2048,2304,2560
CANON_P38_SERVING_CAPTURE_EXPECTED_RECORDS=4
CANON_P38_SERVING_CAPTURE_FREE_SPACE_MULTIPLIER=5
CANON_P38_SERVING_CAPTURE_EXPECTED_PATH=standard
CANON_RUN_CMD contains --batch_size=32
CANON_RUN_CMD contains --mini_batch_size=4
CANON_RUN_CMD contains --num_generations=8
CANON_RUN_CMD contains --mesh_dp=16
maxRestarts: 0
```

### 2. Apply stock only and start full-log collection immediately

```bash
set -euo pipefail
kubectl apply -f "$STOCK" | tee "$EVIDENCE/apply.txt"

JOBSET="canon-p38-fl-stock-${RUN_ID}-${SOURCE_COMMIT:0:8}"
HEAD_JOB="${JOBSET}-pathways-head-0"
POD=""
for unused in $(seq 1 180); do
  POD="$(kubectl get pods -n default -l "job-name=$HEAD_JOB" \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
  [ -n "$POD" ] && break
  sleep 10
done
test -n "$POD"
printf '%s\n' "$JOBSET" > "$EVIDENCE/jobset-name.txt"
printf '%s\n' "$POD" > "$EVIDENCE/head-pod-name.txt"

set +e
kubectl logs -n default -f "$POD" -c jax-tpu | \
  tee "$EVIDENCE/head.follow.log"
follow_rc="${PIPESTATUS[0]}"
set -e
printf '%s\n' "$follow_rc" > "$EVIDENCE/log-follow-rc.txt"
```

Do not use `--tail`, `tail`, `grep`, or a UI copy buffer for the evidence log.
Do not add `--timestamps` to the canonical raw log: the capsule and serving
archive extractors require their `[CANON_...]` markers at column zero. A
separate timestamped diagnostic copy is allowed, but it cannot replace
`head.full.log`.

The `kubectl logs -f` command ending does not by itself prove that the JobSet
is terminal. Check the exact JobSet and pod until they are `Completed` or
`Failed`. Do not delete either object. Once terminal, fetch the complete log
again from byte zero:

```bash
set -euo pipefail
kubectl get jobset -n default "$JOBSET" -o yaml > \
  "$EVIDENCE/jobset.final.yaml"
kubectl get pod -n default "$POD" -o yaml > \
  "$EVIDENCE/head-pod.final.yaml"
kubectl describe pod -n default "$POD" > \
  "$EVIDENCE/head-pod.describe.txt"
kubectl logs -n default "$POD" -c jax-tpu > \
  "$EVIDENCE/head.full.log"
kubectl logs -n default "$POD" -c pathways-proxy > \
  "$EVIDENCE/pathways-proxy.log" 2>&1 || true
kubectl logs -n default "$POD" -c pathways-rm > \
  "$EVIDENCE/pathways-rm.log" 2>&1 || true
kubectl logs -n default "$POD" -c jax-tpu --previous > \
  "$EVIDENCE/head.previous.log" 2>&1 || true
kubectl get events -n default \
  --field-selector "involvedObject.name=$POD" \
  --sort-by=.lastTimestamp > "$EVIDENCE/head-pod.events.txt"
```

If the pod disappears before the terminal fetch, preserve `head.follow.log`
and all JobSet/events/proxy evidence and classify the run as infrastructure
`INCONCLUSIVE`. Never replace a missing full log with its final 200 lines.

### 3. Validate and recover the durable artifacts

The unedited `head.full.log` must contain all of the following:

1. exactly one `JOBSET_ATTEMPT 0 (first attempt)`;
2. `[sync] HEAD=$SOURCE_COMMIT` and a clean sync verdict;
3. exactly one `[CANON_P38_SERVING_CAPTURE_INIT]` carrying
   `expected_path=standard` and at least one
   `[CANON_P38_SERVING_CAPTURE_OBSERVE]` record carrying
   `"program_path": "standard"`;
4. four `pre` and four `post` `[CANON_P38_SERVING_CAPTURE]` records, one pair
   for each registered prefix stratum;
5. finite `S_decode_vs_S_prefill` (red or exact) and exact
   `S_prefill_vs_T_old`;
6. exactly one `[CANON_P38] DIAGNOSTIC_COVERAGE_CONTRACT` carrying
   `prompt_groups=32`, `units=8`, `trajectories=256`, and
   `partial_tail=reject verdict=PASS`;
7. zero `[CANON_P38_SERVING_CAPTURE_ERROR]` records;
8. exactly one `[CANON_P38] PRECHECK_COMPLETE STOP_BEFORE_BACKWARD` whose
   verdict and byte count agree with item 5;
9. child `[run] exit=1` followed by
   `[CANON_PRE_ALIGN_ARTIFACT]` and `[CANON_P38_CAPSULE_ARTIFACT]`;
10. official `[CANON_P38_SERVING_CLASSIFICATION]` with JSON verdict `PASS` and
   at least one exact request/token-history join;
11. `[CANON_P38_SERVING_ARCHIVE]` and every base64 payload line;
12. `[run] P38 serving expected precheck exit=1 accepted; backward=0
   optimizer_commits=0`; and
13. final `[run] PATHTRACE` with `p38_kv_unified=0`,
    `p38_capture_init=1`, positive `p38_capture_observe`,
    `p38_capture_error=0`, and `p38_coverage=1`.

Classify an incomplete run from the terminal `head.full.log` only:

| Terminal evidence | Verdict | Next action |
|---|---|---|
| INIT=1, OBSERVE=0 | `INCONCLUSIVE_STANDARD_HOOK_NOT_REACHED` | inspect standard `_execute_model` wiring; do not lower prefix bounds |
| OBSERVE>0 and every observed maximum is below 1536 | `INCONCLUSIVE_PREFIX_RANGE_MISS` | choose one new bounded range from the recorded request prefixes |
| OBSERVE crosses a registered stratum but no pre/post pair exists | `INCONCLUSIVE_SELECTION_OR_MAPPING` | fix request/packed-row selection; do not change the workload |
| Four pre/post pairs exist but classifier/archive is missing | `INCONCLUSIVE_POSTFLIGHT` | preserve the records and fix artifact transport only |
| All thirteen items exist | `CAPTURE_ADMITTED` | extract artifacts, then start exact E0 replay |

If no stratum is captured, return the complete failure package including every
`CANON_P38_SERVING_CAPTURE_OBSERVE` line. Do not lower the bounds or relaunch
automatically: the observed request-level prefix range is the evidence needed
for the next single-variable revision.

Then recover the binaries from the unedited, non-timestamped complete log:

```bash
set -euo pipefail
python3 \
  canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/extract_p38_capsule.py \
  --log "$EVIDENCE/head.full.log" \
  --output "$EVIDENCE/p38s11-mismatch-capsule.npz"
python3 \
  canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/extract_p38_serving_archive.py \
  --log "$EVIDENCE/head.full.log" \
  --output "$EVIDENCE/p38s11-serving-capture.tar"
sed -n 's/^\[CANON_PRE_ALIGN_ARTIFACT_JSON\] //p' \
  "$EVIDENCE/head.full.log" > "$EVIDENCE/pre-alignment.jsonl"
sed -n 's/^\[CANON_P38_SERVING_CLASSIFICATION_JSON\] //p' \
  "$EVIDENCE/head.full.log" > "$EVIDENCE/serving-classification.json"
test -s "$EVIDENCE/pre-alignment.jsonl"
test -s "$EVIDENCE/serving-classification.json"
sha256sum \
  "$EVIDENCE/head.full.log" \
  "$EVIDENCE/p38s11-mismatch-capsule.npz" \
  "$EVIDENCE/p38s11-serving-capture.tar" \
  "$EVIDENCE/pre-alignment.jsonl" \
  "$EVIDENCE/serving-classification.json" | \
  tee "$EVIDENCE/SHA256SUMS"
```

If any required marker or extraction is missing, stop and return the failure
package. Do not automatically relaunch and do not interpret partial numerical
values.

### 4. Return this exact bundle

Return the entire `$EVIDENCE` directory, not screenshots or pasted tails. At a
minimum it must contain:

```text
source_commit.txt
render.txt
rendered-stock.yaml
dry-run-stock.txt
dry-run-unified.txt
apply.txt
jobset-name.txt
head-pod-name.txt
head.full.log
head.follow.log
jobset.final.yaml
head-pod.final.yaml
head-pod.describe.txt
head-pod.events.txt
pathways-proxy.log
pathways-rm.log
head.previous.log
p38s11-mismatch-capsule.npz
p38s11-serving-capture.tar
pre-alignment.jsonl
serving-classification.json
SHA256SUMS
```

The required ancestor is only a baseline evidence anchor. The two `rg` checks
above are mandatory proof that the checked-out source also contains P38.2g6.
The operator must report in plain text: source SHA, run
ID, JobSet name, pod name, final JobSet condition, pod exit reason/code,
restart count, and
whether all thirteen acceptance items above were present. The operator must not
claim PASS from an A/B number alone.

## Purpose

P38.2 separates two observed flag-on `S_decode_vs_S_prefill` signatures. GSM8K is a tail-aval
candidate; FrozenLake contains a `0.10390` maximum difference and requires upstream/multi-turn
localization. The original strict probes are pre-backward diagnostics. The
later P38.2d amendment separately admits GSM8K full training under an all-
alignment warning-only policy; FrozenLake remains strict and no-commit. A
P38.2d GSM8K run is `convergence-only`, not a zero-TIM completion claim.

## Proven locally

- The alignment test suite passes 28/28 in `tunix_frozenlake_image:vllm-tpu0.25.0`.
- The complete P33 CPU gate passes, including a deliberately failed workload whose pre-alignment
  JSON and SHA survive in stdout.
- The existing hard gate is unchanged: any nonzero pre-backward boundary still exits nonzero.
- A signed GSM8K DP1xTP4 direct-attached run observed 11,340 action tokens with
  `S_decode_vs_S_prefill=0/45360 bytes` and
  `S_prefill_vs_T_old=0/45360 bytes`; the classifier verdict is
  `LOCAL_NOT_REPRODUCED`.
- A production-shape canonical-tail control compared 38,895,616 f32 elements
  across two outer JIT programs with zero differences and detected an injected
  one-bit negative control.
- A model-free DP1xTP4 aval probe ran the live sampling transform at M16/M256
  and the live canonical scorer at M256/M256. Its transform HLO digests were
  different but all five numerical comparisons were exact. This is
  `MODEL_FREE_NOT_REPRODUCED`, not a target fix.
- A synthetic multi-turn mismatch now records turn index, action-run offset,
  completion and sequence chunk coordinates, logical KV prefix length, and
  distance to the next M256 boundary. The complete CPU gate passed.
- A real zero-LR Adam commit with 16 active gradient microbatches advances
  optimizer state, keeps all parameter elements unchanged, and passes the new
  schedule-aware transaction gate. A positive constant-LR control reports
  nonzero post-rounding parameter changes.
- A blocking pre-backward mismatch can persist at most two replay rows to a
  hash-attested NPZ. The failed-run wrapper emits base64 to stdout, and
  `scripts/extract_p38_capsule.py` rejects corrupt transport or array hashes.

## Not proven

- P38d5 ran on 64 target chips, but the new schedule-aware commit evidence and
  mismatch capsule were implemented afterward and remain untested there.
- The FrozenLake A-B carrier has not been localized to a specific operator,
  page layout, or attention tile. Its observed onset is a coordinate, not a
  causal repair.
- GSM8K P38d5 measured `T_old_vs_T_current=0` across all 16 microbatches.
  FrozenLake still stops pre-backward, so its actual-model third-program
  boundary remains unmeasured.
- The new schedule-aware commit evidence and mismatch capsule have not run on
  DP16xTP4 target hardware.

The one-host result is a construction gate, not evidence that r35 was repaired.
Its immutable local artifacts are:

- `/mnt/disks/tunix-data/logp_probe_1host/p38_onehost_0810_r2.result.json`
- `/mnt/disks/tunix-data/logp_probe_1host/p38_onehost_0810_r2/pre_alignment.jsonl`
- `/mnt/disks/tunix-data/logp_probe_1host/p38_tail_0810_r1.result.json`
- `/mnt/disks/tunix-data/logp_probe_1host/p38_aval_0810_r1.result.json`

## Source and render the model-free target probe

Run only after the P38 patch has been reviewed, committed, and pushed with explicit approval.
Use a clean `yuxzhang/canon-zero-tim` worktree and replace `p38a0` if that run id already exists.

```bash
test "$(git branch --show-current)" = yuxzhang/canon-zero-tim
test -z "$(git status --porcelain)"
git pull --ff-only origin yuxzhang/canon-zero-tim

SOURCE_COMMIT="$(git rev-parse HEAD)"
RUN_ID="p38a0"
TARGET="/tmp/jobset-p38-aval-$RUN_ID.yaml"
python3 canon-zero-tim/cluster/render_p38_aval_jobset.py \
  --source-commit "$SOURCE_COMMIT" \
  --run-id "$RUN_ID" \
  --output "$TARGET"
kubectl apply --dry-run=server -f "$TARGET"
```

The dry run must pass before resource allocation. Do not apply the rendered directory and do not
queue `gsm8k-full` or `frozenlake-full` in this phase.

## Stage 1 target run: model-free aval discriminator

The external operator may apply exactly one manifest after confirming resource
approval. It uses no model, workload, backward, optimizer, checkpoint, or W&B:

```bash
kubectl apply -f "$TARGET"
```

Require Attempt 0, zero restarts, the source commit printed by the renderer, and the proxy
`XLA_FLAGS=--xla_allow_excess_precision=false` contract. A failed numerical gate is an expected
diagnostic outcome; do not restart it automatically.

Return the complete head-pod stdout plus the durable
`CANON_P38_AVAL_REPORT`. The report must contain five completed comparisons,
the registered DP16xTP4 shape table (transform M16/M4096, score M256/M4096),
sharding specs, HLO digests, and a one-element negative control. Missing fields
make the run inconclusive. A fully exact model-free result does not prove the
production boundary; it advances to Stage 2.

## Stage 2 target runs: both production boundaries

After Stage 1 is classified, render the existing no-commit production probes
from the same source commit. Do not substitute one workload for the other:

```bash
RUN_ID="p38prod0"
OUT="/tmp/p38-jobsets-$RUN_ID"
python3 canon-zero-tim/cluster/render_p33_jobsets.py \
  --source-commit "$SOURCE_COMMIT" \
  --run-id "$RUN_ID" \
  --output-dir "$OUT"

GSM="$OUT/jobset-p33-gsm8k-alignment-short.yaml"
FL="$OUT/jobset-p33-frozenlake-alignment-short.yaml"
kubectl apply --dry-run=server -f "$GSM"
kubectl apply --dry-run=server -f "$FL"
```

With separate resource approval, apply only `GSM` and `FL`. Both stop before
backward and optimizer commit. GSM8K tests the low-amplitude tail candidate.
FrozenLake independently tests the `0.10390` multi-turn signature and must
emit turn, action-run, M256 chunk, and logical-KV coordinates for every
reported mismatch.

## Evidence to return

Archive the complete head-pod stdout and report its SHA. The raw log must contain:

- `[CANON_ALIGN_PRE_JSON]` with both boundary records;
- `[CANON_ALIGN_PRE_EVIDENCE]` with the report SHA;
- on failure, `[CANON_PRE_ALIGN_ARTIFACT]` and every
  `[CANON_PRE_ALIGN_ARTIFACT_JSON]` row;
- the exact source commit, Attempt 0 marker, proxy XLA environment, mesh order, local canonical
  row count, `N_action`, and workload exit code.

For every mismatch, preserve coordinate, token id, exact A/B bits, XOR, byte offsets, ULP
distance, and absolute delta. The report is inconclusive if a target line is missing; absence is
not equality.

## Pre-registered verdict

- A-B nonzero and B-C zero: P38.2b reproduces the GSM8K serving carrier; classify the transform,
  score, and implied-normalizer fields before selecting a repair. FrozenLake is still required.
- A-B zero and B-C zero: GSM8K did not reproduce the sparse r35 carrier. This is not proof of a
  fix; P38.2c FrozenLake remains independently required.
- B-C nonzero, an invalid shape, missing evidence, source drift, a retry, or an infrastructure
  disconnect: the numerical result is not admitted.

FrozenLake evidence must additionally identify the turn index, assistant-run offset, canonical
chunk index, logical KV prefix length, and whether the mismatch is adjacent to a turn or M256
boundary. A tail-only repair is not admitted for a `0.10390` upstream signature.

No tolerance, report-only committing mode, old-logprob recomputation, precision change, or
optimizer commit is authorized by this handoff.

The paragraph above records the original strict handoff. It is superseded only
by the 2026-08-11 amendment in
`phases/p38-2d-gsm8k-bounded-ab-campaign.md`: committed GSM8K full may set
`CANON_GSM8K_ALIGNMENT_WARN_ONLY=1`. All finite numerical alignment mismatch,
including B/C, old/current, ratio exactness, and clip/TIS observations, remains
visible but cannot terminate that campaign. The claim ceiling is
`convergence-only`. FrozenLake, invalid shapes, NaN/Inf, gradients, DP
reduction, replica equality, optimizer integrity, and runtime failures remain
hard. Old-logprob recomputation and precision changes remain forbidden.

## P38.2d operator handoff

After pulling the source commit, render the P33 queue with a fresh run id. The
renderer must show `CANON_GSM8K_ALIGNMENT_WARN_ONLY=1` only in `gsm8k-full`,
`CANON_GSM8K_AB_REPORT_ONLY=0`, and both flags `0` in every other YAML. Apply
only the FrozenLake backward-no-commit and GSM8K full manifests. Do not apply
FrozenLake full.

The GSM8K classifier may exit successfully as
`PASS_WITH_ALIGNMENT_WARNINGS`. This means the requested convergence run
completed under a warning-only alignment policy, not that A=B=C was proven.
Archive the raw log and all alignment/update JSONL before deleting either
JobSet.

The refreshed FrozenLake backward-no-commit manifest must contain
`CANON_P38_MISMATCH_CAPSULE_MAX_ROWS=2` and a run-isolated
`CANON_P38_MISMATCH_CAPSULE` path. On the expected hard red, archive all
`[CANON_P38_CAPSULE_ARTIFACT]` and `[CANON_P38_CAPSULE_B64]` lines. Recover the
file without editing the raw log:

```bash
python3 canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/extract_p38_capsule.py \
  --log /path/to/frozenlake-head.raw.log \
  --output /path/to/p38-frozenlake-capsule.npz
```

Do not rerun FrozenLake before the capsule has passed transport and embedded
array SHA checks. Prefix cache remains disabled. The next action is the
single-row prefix sweep specified in
`phases/p38-2f-frozenlake-threshold-capsule.md`, not a full training launch.

## Current FrozenLake execution order

The published source is `e9cfe298`. The first refreshed
`frozenlake-backward-no-commit` run is expected to stop at the pre-backward A-B
gate. Treat it as P38.2f capsule capture; do not claim that backward ran merely
because of the manifest name. Do not add KV-unified or another numerical arm
to this first target run.

After capsule recovery, follow
`phases/p38-2g-frozenlake-causal-replay.md`: R0 stock multi-turn reproduction,
R1 same-depth single-turn control, R2 MIXED-only two-pass cache write/read, and
R3 all-distribution two-pass. Each arm requires an independent cache seed.
Only a selected candidate with exact forward boundaries may advance to a new
target backward-no-commit run. That later run must prove gradient and DP
reducer health while committing neither parameters nor optimizer state.

GSM8K full may run in parallel on independent resources. Its schedule-aware
transaction result is independent evidence and cannot promote FrozenLake.

## P38.2g implementation ready in the current worktree

The current uncommitted worktree on top of `e9cfe298` contains the locally
admitted R0/R1 replay implementation. It has not been published and has not
run on a real Qwen3-8B TPU model. Do not tell the target operator to pull it
until the user explicitly approves commit and push.

After P38.2f produces a capsule, recover it with
`scripts/extract_p38_capsule.py`, copy the verified NPZ to the authorized
DP1xTP4 host, and run:

```bash
scripts/run_p38_frozenlake_replay.sh \
  /absolute/path/to/recovered-p38-capsule.npz <unique-label>
```

The runner verifies the capsule before model initialization, loads Qwen3-8B,
executes R0/R1/reference twice with fresh caches, runs a one-bit negative
control, requires bitwise-equal actor/engine leaves, writes a bounded report,
and exits before the agentic learner/backward with zero optimizer commits.
R0 is `mask-derived-v1`; exact serving scheduler metadata was not captured.

Expected classification is one of:

- `MULTITURN_SCHEDULE_CARRIER_CANDIDATE`: R0 is red and R1 is exact against
  the fixed-chunk reference;
- `LOCAL_CARRIER_NOT_REPRODUCED`: R0 is exact locally;
- `LOCAL_CARRIER_NOT_ISOLATED`: R0 and R1 are both red.

All three are measurement outcomes, not production repair admission. R2/R3
must remain absent until a verified target capsule makes R0/R1 interpretable.
Local proof and exact commands are in `artifacts/p38_2g_local_gate.md`.

Real Qwen3-8B synthetic controls have now exercised the runner. At prompt
lengths 256 and 1788, R0 and R1 were bitwise exact with each other, while both
were red against REF at all eight scored actions. The shallow maximum was
larger than the deep maximum. This is `LOCAL_CARRIER_NOT_ISOLATED`, not a
production reproduction and not evidence for a KV-1791 threshold. See
`artifacts/p38_2g_onehost_synthetic_0811.md`.

Do not implement or interpret R2/R3 from the synthetic result. The target
operator must still produce a verified P38.2f capsule. If target R0/R1 show the
same broad split, add an exact serving-envelope control before changing KV
update behavior.
# 2026-08-11 target-row update: local serving envelope rejected

The verified P38e1 source row 191 has now run on real Qwen3-8B DP1xTP4. Do not
implement or interpret local R2/R3 from the current mask-derived schedule.

- R0 equals R1 bitwise at raw target, processed target, normalizer, and logprob.
- R0/R1 differ from REF at 395 of 517 action logprobs.
- REF logprob SHA exactly equals captured `S_prefill`/`T_old`.
- R0/R1 do not equal captured `S_decode`.
- Measurement integrity passed; classification is
  `LOCAL_CARRIER_NOT_ISOLATED`; no production repair is admitted.
- Evidence: `artifacts/p38_2g_onehost_target_row191_0811.md`.

The next implementation belongs in the actual serving envelope. The existing
P18/P35 capture in `patches/tpu_inference/06-tpu-runner.patch` and
`07-tpu-runner-p35-metadata.patch` records metadata only when
`input_batch.num_prompt_logprobs` is non-empty, which captures rescore B but not
ordinary decode A. Add a separate default-off P38 serving-metadata capture that
also executes for decode and records, per scheduler call:

1. monotonically increasing call ordinal and the live request/slot IDs needed
   to join a capsule row back to its serving request;
2. input IDs and positions;
3. attention input positions, block tables, sequence lengths, query starts,
   and request distribution;
4. exact logical-to-physical page IDs used by the selected requests;
5. cache shape, dtype, sharding, page size, and configured D/P/M block tuples;
6. whether the call is decode, prefill, or mixed and the effective
   `update_kv_cache` value.

The capture must be bounded, attempt-unique, fail on overwrite, print completed
record counts, and include a negative control proving that missing decode
records reject classification. The mismatch capsule must also preserve the
row-to-request-ID mapping; row order alone is not an admissible join key.

The pinned-source audit is recorded in
`phases/p38-2g2-pathways-serving-envelope.md`. Production A can execute inside
`runner/decode_loop.py::continue_decode`, so capture limited to ordinary
`model_fn` or prompt-logprob calls is invalid. The v3 public API also cannot
construct a clean write-only arm: `update_kv_cache=False` both skips the write
and forces all-cache reads. Do not label any v2-writer experiment as a
single-variable `W` arm.

After an exact serving record exists, run stock first. Only if that record
reproduces captured `S_decode` may a separate source-pinned diagnostic enable
the combined historical `U` arm: stock RPA writes the cache, its output is
discarded, and a second RPA call with `update_kv_cache=False` supplies the
attention output. This can establish causality for the combined mechanism but
cannot distinguish fused-write effects from read-source effects.

Keep prefix cache disabled, backward disabled, optimizer commits zero, and the
precision/fixed-M/fixed-reduction configuration unchanged. Rollback is leaving
the new capture and counterfactual environment variables unset.

## P38.2g2 historical local handoff (superseded below)

This section records the pre-target state. Do not execute its U instruction;
the 2026-08-11 evidence correction below is the current operator contract.

Commit `763b60b1` introduced the implementation described above. The admission
hardening is published at `bbc1d329` on `yuxzhang/canon-zero-tim`:

- patch 09 captures the actual donated-cache `continue_decode` call, including
  request IDs, full current token histories, physical page IDs, scheduler and
  attention metadata, sampling leaves, the physical/logical selector, and
  bounded post-dispatch outputs;
- patch 08 adds only the combined two-pass `U` arm. It cannot distinguish the
  fused writer from the read-source change and must never be named `W`;
- `render_p38_serving_jobsets.py` renders separate stock and U Attempt-0
  manifests; and
- `90_run.sh` classifies and emits a SHA-verified tar as base64 so evidence
  survives pod deletion. `extract_p38_serving_archive.py` recovers it. Both
  manifests force `CANON_P38_PRECHECK_ONLY=1`, so an exact U arm stops before
  backward rather than falling through the misleadingly named workload stage.

The four admission gaps from the post-publication review are closed locally:
only scheduled requests are selected without compacting physical slots;
request/DP/slot/global/attention/selector/page mappings are explicit and
internally validated; stock must join one durable mismatch by request/token
history; and postflight requires zero U PATHTRACE hits for stock plus a
positive hit for U. Pinned-image install is 29/29 for both model overlays,
installed runtime tests pass 13/13 for each overlay, serving classifier 18/18,
renderer 5/5, archive transport 4/4, shell stock/U controls PASS, and the
complete P33 CPU suite passes 67 workload plus 28 alignment tests and all
adjacent negative controls. At the time this historical section was written,
no Pathways/TPU target result existed; see the correction below for p38s1 and
p38u1.

Fetch `yuxzhang/canon-zero-tim`, verify `git rev-parse HEAD` is `bbc1d329`, and
follow the exact stock-first commands in
`phases/p38-2g2-pathways-serving-envelope.md`. Dry-run both manifests but apply
only stock; never apply the output directory. The historical instruction was
to defer U until stock joined the mismatch capsule. U has since run and
remained red; do not rerun it.

The next pending phase is
`phases/p38-2g3-page-topology-discriminator.md`. It treats fragmented physical
pages and padding-boundary leakage as hypotheses, not established facts. The
first stock record must join the actual A-B mismatch by request/token history.
Only then may real, relocated, contiguous, or padding-sanitized page-table arms
be interpreted, and every topology arm must prove logical page-content
equivalence at each registered write event. E0 must reproduce the entire
captured action vector, not only known mismatch coordinates. The padding arm
has a dedicated poison control: the same padding-only sentinel table is run
with zero and finite poison contents after proving that no valid row references
the sentinel. If E0 is not exact, stop all counterfactuals and use the bounded
recapture-state list in the phase document.

## Time-sensitive GSM8K convergence exception

The next GSM8K full campaign is a separate, explicitly degraded track. The
locally gated implementation uses `CANON_GSM8K_ALIGNMENT_WARN_ONLY=1` exactly
as registered in `phases/p38-2d-gsm8k-bounded-ab-campaign.md`. With that flag,
all alignment reds become durable warnings and cannot throw
`AlignmentGateError`; there is no A-B/B-C/T-current mismatch threshold. The
run remains fatal on invalid/nonfinite numerics, reducer/replica failure,
optimizer transaction failure, infrastructure errors, and ordinary runtime
exceptions.

Only committed GSM8K full may set the flag. Its terminal result is
`PASS_WITH_ALIGNMENT_WARNINGS`, `claim_level=convergence-only`. It is not
zero-TIM evidence and does not relax FrozenLake, DeepSWE, or P38.2g2/P38.2g3.
Commit, push, and target launch remain pending explicit approval.

## 2026-08-11 evidence correction: stock/U values exist, serving capture does not

The branch now contains the complete available head logs and alignment-layer
reports for `p38s1` and `p38u1`. They are useful numerical evidence, but
neither run completed the P38.2g2 serving-capture postflight. Do not treat
either run as a completed serving-envelope capture.

Verified observations:

- `p38s1` stock ran on Attempt 0 with `CANON_KV_UNIFIED=0`. It measured
  `S_decode_vs_S_prefill` red at 43/46,417 action elements (68 differing
  bytes, `max_abs=0.2780647277832031`) while `S_prefill_vs_T_old` was exact.
- `p38u1` executed the `KV_UNIFIED_two_pass` path and remained red at
  9/46,589 action elements (16 differing bytes,
  `max_abs=0.27657318115234375`) while `S_prefill_vs_T_old` remained exact.
  Therefore combined U is not a sufficient repair for the production
  boundary.
- Stock and U sampled different trajectories and have different action counts.
  The change from 43 to 9 elements is not a controlled paired reduction and
  must not be used to claim that U improved the carrier, changed writer
  timing, or proved a page-lifecycle cause.
- Both head logs end at the child `AlignmentGateError`. They do not contain
  the outer `[run] exit=...`, `[CANON_PRE_ALIGN_ARTIFACT]`, official
  `[CANON_P38_SERVING_CLASSIFICATION]`,
  `[CANON_P38_SERVING_ARCHIVE]`, or final `[run] PATHTRACE` records. The
  checked-in `p38-serving-capture-stock` and
  `p38-serving-capture-unified` JSON files are alignment summaries, not the
  official `classify_p38_serving_capture.py` output.
- The generic checked-in `debug_logs/p38_frozenlake_mismatch_capsule.npz` has
  SHA-256 `dae4e75d3b4689f2607047edd74ea1e48ffaf97a853cec74a204caafc3dc626b`.
  It is byte-for-byte the older P38e1 capsule, not the `p38s1` capsule logged
  as `2dffb993...` or the `p38u1` capsule logged as `245a0c9b...`. It contains
  no serving block table or page-ownership history.

Consequently:

- the earlier `artifacts/p38_2g2_pathways_stock_capture_0811.md` PASS claim is
  withdrawn; it proves the stock alignment red but not a complete serving
  capture;
- P38.2g2 remains `INCONCLUSIVE` at the serving-envelope layer;
- scheduler/page ownership, stale block tables, partial writes, padding
  leakage, and other serving-state hypotheses remain unproven; and
- do not run U again. Its only registered question has already been answered:
  the combined operation remained materially red.

### Remote operator: run one fresh stock-only capture

This older single-record instruction is superseded by P38.2g4 below. Do not
launch from `b7b20e2` or use `CANON_P38_SERVING_CAPTURE_MIN_PREFIX=1788` after
the P38.2g4 implementation is published.

Use exact source `b7b20e261433977bc57bd83452fd6ac1c4680cdd` (or a later reviewed
documentation-only descendant that leaves the renderer/runtime unchanged).
Use a fresh run ID and output directory. Render both manifests only because
the renderer emits them as a pair; dry-run both, but apply **stock only**:

```bash
git fetch origin yuxzhang/canon-zero-tim
git checkout --detach b7b20e261433977bc57bd83452fd6ac1c4680cdd
test "$(git rev-parse HEAD)" = "b7b20e261433977bc57bd83452fd6ac1c4680cdd"

SOURCE_COMMIT="$(git rev-parse HEAD)"
RUN_ID="p38s2-serving"
OUT="/tmp/p38-serving-$RUN_ID"
python3 canon-zero-tim/cluster/render_p38_serving_jobsets.py \
  --source-commit "$SOURCE_COMMIT" \
  --run-id "$RUN_ID" \
  --output-dir "$OUT"

kubectl apply --dry-run=server -f "$OUT/jobset-p38-serving-stock.yaml"
kubectl apply --dry-run=server -f "$OUT/jobset-p38-serving-unified.yaml"
kubectl apply -f "$OUT/jobset-p38-serving-stock.yaml"
```

Capture the complete head-container log only after the JobSet is terminal. Do
not stop log collection at the child traceback. The stock result is admitted
only if the complete log contains all of the following:

1. `JOBSET_ATTEMPT 0 (first attempt)`;
2. the known finite hard `S_decode_vs_S_prefill` red and exact
   `S_prefill_vs_T_old`;
3. `[run] exit=1` from the child numerical gate;
4. `[CANON_PRE_ALIGN_ARTIFACT]` and
   `[CANON_P38_CAPSULE_ARTIFACT]` plus all capsule base64 lines;
5. one official `[CANON_P38_SERVING_CLASSIFICATION]` whose JSON verdict is
   `PASS` and whose request/token-history join is exact;
6. `[CANON_P38_SERVING_ARCHIVE]` plus all serving-archive base64 lines; and
7. a final `[run] PATHTRACE ... p38_kv_unified=0`.

Stock is expected to exit nonzero because its known A/B red remains hard. It
must not print the U hit, a successful backward, or an optimizer commit. The
outer wrapper is still required to run its classifier/archive/PATHTRACE logic
after the child failure. If any item above is absent, classify the attempt as
`INCONCLUSIVE`; do not infer anything from the numerical values alone.

Recover both durable binary artifacts from the unedited complete log:

```bash
python3 canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/extract_p38_capsule.py \
  --log /path/to/p38s2-serving-head.raw.log \
  --output /path/to/p38s2-serving-capsule.npz
python3 canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/extract_p38_serving_archive.py \
  --log /path/to/p38s2-serving-head.raw.log \
  --output /path/to/p38s2-serving-capture.tar
sha256sum /path/to/p38s2-serving-capsule.npz \
  /path/to/p38s2-serving-capture.tar
```

Return the unedited complete head log, both recovered binaries, their SHA-256
values, and the official serving-classification JSON. Do not substitute a
pretty-printed alignment JSON or the generic `dae4e75d...` NPZ. P38.2g3 E0 is
the next phase only after these artifacts pass transport, exact join, and
whole-vector reproduction. E1-E5 and all repair claims remain blocked until
E0 is exact.

## P38.2g4 current handoff: source published; resource launch still separate

The current active plan is
`phases/p38-2g4-decode-envelope-seam.md`. It does not claim that RoPE, cache
pages, or scheduler ownership is the root cause. It first replaces the brittle
single capture threshold with four bounded intervals:

```text
[1536,1792) [1792,2048) [2048,2304) [2304,2560)
```

These are continue-decode capture filters only. They do not change prompt
length, response length, attention tiles, prefix cache, or model geometry.
Capture at most one call per interval, require the complete four-record set,
and require at least one unambiguous token-history join to the run-specific
mismatch capsule. Missing/duplicate intervals, zero joins, source identity
drift, an incomplete outer log, or missing archive transport are
`INCONCLUSIVE`.

D0 is locally complete: classifier 25/25, renderer 5/5, shell postflight,
exact-image Qwen3-1.7B and Qwen3-8B 14/14 each with all 29 manifest entries,
and the full frozen-image P33 CPU gate all pass. The installed runner SHA-256
is `fe81622996a1c73bbd17187ee603e6a191165202da40d07b5e428fe41b5db516`.
Docker had no TPU device, so this proves construction and image compatibility,
not target behavior. The implementation is published at
`b89435ca7d64faa65c00b5a85152f71fdfc60167`. The external operator may fetch
and verify that source, but must not apply a target manifest until the user
separately approves the 64-chip resource use.

After publication and resource approval, run stock only on Attempt 0 with
prefix cache disabled, backward disabled, and zero optimizer commits. Do not
rerun combined U. Recover the complete head log, run-specific mismatch capsule,
serving tar, classifier JSON, and final PATHTRACE. Before launch, estimate the
archive size and require at least five times that size as free space.

The next numerical gate is E0: the joined request's entire A vector and entire
B vector must reproduce the production vectors bitwise, while source A-B stays
red. Only then may the same record enter the ordered seam probe (layer input,
Q/K/V projection, Q/K norm, post-RoPE, RPA, output/residual, MLP, layer output,
final norm/logit/normalizer) and the independent page/cache E1-E4 branch.
Observer instrumentation is void unless observer-off and observer-on endpoints
remain bitwise equal.
