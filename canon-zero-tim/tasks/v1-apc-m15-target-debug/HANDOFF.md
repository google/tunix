# M15 APC target-debug handoff

## START HERE — Attempt 15 (d34) executed Round 0 with exact numerical PASS; assemble failed on missing diagnostic_round in replay envelope

### Incident Summary (Attempt 15 / d34)
Matched pair on 64 TPU each (`canon-v1-apc-m15-off-d34-57d9ab8e` and `canon-v1-apc-m15-on-d34-57d9ab8e`):
- **Round 0 Prefill Rescore Alignment (Verified PASS)**:
  - **APC-Off**: `Prefix cache hit rate: 0.0%`, `N_action: 120,889`, `S_decode_vs_S_prefill differing_bytes: 0`, `Pre-alignment verdict: PASS`. 85 shards staged and uploaded.
  - **APC-On**: `Prefix cache hit rate: 93.2%`, `N_action: 130,468`, `S_decode_vs_S_prefill differing_bytes: 0`, `Pre-alignment verdict: PASS`. 72 shards staged and uploaded.
- **Fatal Failure Point**:
  At round 0 completion, `_seal_p38_diagnostic_round(round_index=0)` triggered the background round sealer. `assemble_m15_wide_round.py` failed with:
  ```text
  [M15.WIDE.ROUND] RED replay round is invalid at line 1
  tunix.rl.alignment.AlignmentGateError: P38 round-seal worker failed before acknowledgement: round=0 stage=assemble exit_code=2
  ```
- **Root Cause**: `patches/tpu_inference/26-tpu-runner-m15-replay-envelope.patch` omits `"diagnostic_round"` in the JSON envelope record. `assemble_m15_wide_round.py` expects `record["diagnostic_round"] == round_index`, evaluating to `-1` and asserting out.
- **Sealed Incident Package**: `evidence/v1_apc_m15_attempt15_d34_20260828/` (`INCIDENT_REPORT.md`, `m15_off_d34_attempt15_tail.log`, `m15_on_d34_attempt15_tail.log`, `p38_live_worker_off.log`, `p38_live_worker_on.log`, `m15_replay_envelope_head.jsonl`, `SHA256SUMS`).

### Action Plan for Attempt 16 (d35)
1. Add `"diagnostic_round": int(_p38_seam_round())` (or `diagnostic_round`) to `patches/tpu_inference/26-tpu-runner-m15-replay-envelope.patch`.
2. Run test suites (`test_target_carrier.py`, `test_m15_wide_durability.py`, `test_classify_m15_apc_wide_seam.py`).
3. Re-render, dry-run, and launch Attempt 16 (`d35`) matched pair on 64 TPU.

## d33 flat-shard content audit verified Round 0 only; fix first seal/ACK before rerun

Attempt 14 (`d33`) now has three immutable small returns:

- `evidence/v1_apc_m15_attempt14_d33_operator_return_20260828/` records the
  original `NO_DURABLE_ROUND_OPERATOR_RECEIPTS_INCOMPLETE` result;
- `evidence/v1_apc_m15_attempt14_d33_inventory_return_20260828/` resolves the
  query ambiguity and proves both recursive listings succeeded (265 off / 223 on objects);
- `evidence/v1_apc_m15_attempt14_d33_flat_shard_audit_20260828/` completes the
  receipt-bound flat-shard content audit for all 162 shards.

The flat-shard content audit verified:

| Arm | Shards | Record pairs | Payload bytes | Diagnostic rounds | Receipt/manifest audit |
|---|---:|---:|---:|---|---|
| APC off | 88 (`000000..000087`) | 2,780 | 1,792,189,157 | Round 0: 88 (100%) | 88/88 completion + manifest OK |
| APC on | 74 (`000000..000073`) | 2,302 | 472,614,342 | Round 0: 74 (100%) | 74/74 completion + manifest OK |

Every listed shard directory contains `SHARD_COMPLETE.json`, `SHA256SUMS`, and
`SHARD_ARCHIVE.tar`.  The small audit independently binds each completion
receipt to its manifest and confirms that the producer receipt carries a
well-formed archive digest.  It does **not** download or independently re-hash
the archive payload; the old wording that archive contents/digests were
independently verified is withdrawn.  The round metadata itself is decisive:
**100% of receipts belong to diagnostic round 0**. Neither arm crossed the
first round 0 seal to emit round 1 or round 2.

The machine decision is:

```text
AUDIT_M15_ATTEMPT14_D33_FLAT_SHARDS decision=D33_FLAT_SHARDS_ROUND0_ONLY rounds=[0] off_shards=88 on_shards=74
```

The strict status is:

```text
FLAT_SHARD_AUDIT_PASS / D33_FLAT_SHARDS_ROUND0_ONLY /
ROUND0_RECEIPTS_AND_MANIFESTS_VERIFIED /
ARCHIVE_PAYLOAD_NOT_INDEPENDENTLY_REHASHED / ROUND1_2_NOT_REACHED /
OFFICIAL_CLASSIFIER_MISSING / FIRST_RED_NOT_LOCALIZED /
PHASE_E_CLOSED / NUMERICAL_REPAIR_NOT_AUTHORIZED
```

### Current work order — review locally green seal/ACK hardening before publication

Per the decision table:
- `D33_FLAT_SHARDS_ROUND0_ONLY`: "valid content exists only for round 0 -> inspect/fix the first seal/ACK path before any rerun".
- `phases/phase-d3-seal-ack-hardening.md` owns the additive repair: stage
  receipts, an atomic `round-N.failure.json`, learner fail-fast handling, a
  three-round positive control, a forced-persistence negative control, and a
  stage-aware small-return audit.
- Host gates are green: 137/137 M15 tests, the P38 persistence suite, 394/394
  flag audit, syntax/compile checks, and `git diff --check` all pass. The fake
  GCS end-to-end return distinguishes explicit failure from interrupted
  progress without accepting either as numerical evidence.
- This is not target admission. Exact-image, commit/push, and a fresh matched
  pair each require separate user approval.
- Phase E remains closed; production APC stays off; B remains an independent full-reset computation.

## Historical — offline-review d32, then render d33; do not launch implicitly

The seven-file Attempt-13 (`d32`) inventory is transport-complete and proves
that both recursive listings succeeded and both registered roots contain no
`live/` or `wide/rounds/` objects. It also exposes an unresolved count drift:

| Arm | Physical shard completion `record_pairs` | Immutable receipt/classifier `seam_records` | Delta |
|---|---:|---:|---:|
| APC off | 2,445 | 2,474 | -29 |
| APC on | 2,188 | 2,087 | +101 |

These field names are not assumed to be the same metric. The checked-in
inventory was produced by changing the expected values after observing GCS, so
its old generic `PASS` is accepted only as an object-transport fact. It is not
an official classifier replay and does not authorize an RPA repair.

`prepare_m15_multiround_pair.sh` now reruns a checked-in offline validator
before rendering. The validator verifies every member of the immutable
seven-file return, re-derives both object geometries and count deltas, and emits
`D32_LIVE_ABSENT_WITH_COUNT_DRIFT`. The renderer embeds that review in its
self-hashed run contract. A reviewer may prepare d33; only the user may approve
the two 64-TPU launches.

### Remote executor command for d33 review and preparation

```bash
cd /home/yuxuan/code_rl_repro/sequence_packing/tunix
git fetch origin yuxzhang/canon-zero-tim
git pull --ff-only origin yuxzhang/canon-zero-tim
SOURCE_SHA="$(git rev-parse HEAD)"
RUN_ID=<fresh-1-to-16-char-lowercase-dns-label>
OUT=/tmp/v1-apc-m15-${RUN_ID}
test ! -e "$OUT"

bash canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/prepare_m15_multiround_pair.sh \
  "$SOURCE_SHA" "$RUN_ID" "$OUT" full 0
(cd "$OUT" && sha256sum -c SHA256SUMS)
```

The renderer must produce exactly:
- `jobset-v1-apc-m15-off-full.yaml`
- `jobset-v1-apc-m15-on-full.yaml`
- `D32_REVIEW.json` with `live_absence_status=CONFIRMED`,
  `count_contract_status=DRIFT`, `d33_preparation_eligible=true`, and both
  authorization booleans false;
- `RUN_CONTRACT.json` with `diagnostic_rounds=3`, `observer=full`,
  `seam_layer=0`, zero backward/commit, and the exact D32 review SHA;
- `SHA256SUMS` covering the two YAMLs, review, and run contract.

After separate launch approval, both standalone `kubectl apply` commands may
be issued concurrently. Preparation is not launch authority.

---

## Historical — Attempt 13 (`d32`) object inventory (transport-complete)

Attempt 13 (`d32`) was a **single diagnostic round** produced by the older
flat-shard runtime. The registered roots contain 77 contiguous control shards
whose completion receipts sum to 2,445 record pairs and 70 contiguous treatment
shards whose receipts sum to 2,188. The historical classifier receipts instead
report 2,474 and 2,087 seam records; that difference remains explicit.

The self-hashed read-only inventory completed with:
```text
M15_ATTEMPT13_REVIEW_PASS decision=D32_LIVE_ABSENT_WITH_COUNT_DRIFT count_contract_status=DRIFT d33_preparation_eligible=1 d33_launch_authorized=0 numerical_repair_authorized=0
```
Evidence is sealed in `evidence/v1_apc_m15_attempt13_d32_inventory_20260828/`.
Because no `live/` directory exists, historical flat replay is unviable. d33 is
the next evidence-producing experiment after separate review and launch
approval; it is not a numerical repair.

### Mechanical interpretation reference

| Decision | Meaning | Next action |
|---|---|---|
| `D32_LIVE_ABSENT_WITH_COUNT_DRIFT` | both recursive queries succeeded and neither root listed a `live/` object, while shard and classifier counts disagree | d33 preparation is eligible; preserve the drift and require separate launch approval |
| `D32_LIVE_PRESENT_REPLAY_SHOULD_CONTINUE` | the registered roots contain at least one `live/` object | run the existing flat replay only after inspecting this return; do not launch d33 first |
| `D32_INVENTORY_AUDIT_RED` or non-zero exit | a query, identity, shard geometry, completion receipt, or count failed | fix only the read-only inventory path; absence is unproven and d33 remains blocked |

d33 is one matched APC-off/APC-on pair, each containing three evaluation-only
rounds with frozen weights, zero backward, and zero optimizer commits.  The
full observer is pinned to Layer 0 because Attempt 12 placed the analysis-grade
coarse interval between Layer-0 input and output.  A previously rendered
`/tmp` directory is not a durable source artifact and is not launch authority;
render again from the reviewed, published full SHA when that experiment is
approved.

This is not “run longer and hope the final upload works.”  At the end of each
round the learner blocks until the live worker has:

```text
sealed bounded shards -> uploaded them -> downloaded and verified them
-> classified the sealed union -> written WIDE_ROUND_COMPLETE -> ACKed the learner
```

Only then may the next evaluation begin.  Therefore a death after round 1
cannot erase round 0.  `COLLECTED.json` and `COMPLETE.json` are still required
for a full signed run, but the new small-return script can recover every sealed
round even when that final root close is missing.

After the reviewed source is separately committed and pushed, the remote
executor must fetch it and use exactly:

```bash
cd /home/yuxuan/code_rl_repro/sequence_packing/tunix
git fetch origin yuxzhang/canon-zero-tim
git pull --ff-only origin yuxzhang/canon-zero-tim
SOURCE_SHA="$(git rev-parse HEAD)"
RUN_ID=<fresh-1-to-16-char-lowercase-dns-label>
OUT=/tmp/v1-apc-m15-${RUN_ID}
test ! -e "$OUT"

bash canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/prepare_m15_multiround_pair.sh \
  "$SOURCE_SHA" "$RUN_ID" "$OUT" full 0
(cd "$OUT" && sha256sum -c SHA256SUMS)
```

The renderer must produce exactly:

- `jobset-v1-apc-m15-off-full.yaml`;
- `jobset-v1-apc-m15-on-full.yaml`;
- `D32_REVIEW.json` preserving the count drift and both false authorization
  fields;
- `RUN_CONTRACT.json` with `diagnostic_rounds=3`, `observer=full`,
  `seam_layer=0`, zero backward/commit, and the D32 review SHA.

After separate launch approval, both standalone `kubectl apply` commands may
be issued concurrently.  Do not pipeline either command and do not reuse a run
label.  When both JobSets terminate, the same executor must have read-only
Kubernetes access plus bucket access and run exactly one return wrapper:

```bash
RETURN=/tmp/v1-apc-m15-${RUN_ID}-small-return
test ! -e "$RETURN"
bash canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/run_m15_multiround_operator_return.sh \
  "$OUT" "$RETURN" /mnt/disks/tunix-data default
(cd "$RETURN" && sha256sum -c SHA256SUMS)
```

Do not separately run `run_m15_multiround_gcs_return.sh`; the operator wrapper
calls it internally.  It performs only `kubectl get` and GCS reads.  It does not
delete a JobSet, mutate GCS, download `run.log`, or return a token-bearing tar.
It reads the root manifest and object metadata to bind each remote `run.log` by
sanitized identity, SHA-256, and byte size.

Return the complete small `$RETURN` directory unchanged, the wrapper's final
`[M15.OPERATOR.RETURN] COMPLETE ...` line, and independent
`sha256sum -c SHA256SUMS` output.  Do not manually transcribe statuses or JSON.
The directory itself must contain:

- `MULTIROUND_SUMMARY.json`;
- `off.round-000000..000002.classification.json` for sealed off rounds;
- `on.round-000000..000002.classification.json` for sealed on rounds;
- `off/on.round-XXXXXX.stage-<ordinal>-<stage>-<status>.json` for every small
  stage receipt found remotely; these contain no token payload;
- `JOBSET_STATUS.json`, containing the sanitized terminal condition for both
  exact JobSet names;
- `RAW_LOG_RECEIPTS.json`, containing each immutable `run.log` identity,
  SHA-256, and byte size without the log payload or GCS root;
- `OPERATOR_RETURN_SUMMARY.json` and `OPERATOR_PACKAGING.txt`;
- `PACKAGING.txt` and one final `SHA256SUMS` covering every returned file.

Interpretation is mechanical:

- `COMPLETE`: all six rounds sealed and both roots terminal;
- `ROUNDS_RECOVERED_ROOT_INCOMPLETE`: all six classifiers survived, but the
  overall run is analysis-grade because root finalization died;
- `PARTIAL_ROUNDS_RECOVERED`: at least one round survived; use it, but do not
  call the paired target run complete;
- `ROUND_STAGE_FAILURE_IDENTIFIED`: no round sealed, but a remote FAIL receipt
  names the exact publisher stage and exit code; repair that stage first;
- `ROUND_STAGE_PROGRESS_ONLY`: no round sealed, but ordered stage receipts show
  the last completed or active stage; inspect the terminal worker log before
  relaunch;
- `NO_DURABLE_ROUND`: neither a sealed round nor any stage receipt exists;
- any off-arm red, B-C red, source/round/hash mismatch: hard stop.

`OPERATOR_RETURN_SUMMARY.json.status` equals the numerical core status only
when both JobSets are terminal (`Completed` or controlled-exit `Failed`) and
both raw-log receipts are present.  Otherwise it appends
`_OPERATOR_RECEIPTS_INCOMPLETE` while preserving any sealed numerical rounds.
The operator status never upgrades the numerical status in
`MULTIROUND_SUMMARY.json`.

The script deliberately queries `wide/rounds/000000..000002`; it does not rely
on the root aliases that an early exit may omit.  This is the required answer
to the previous “run finished but wanted data did not return” failure mode.

## Historical — Attempt 12 audit before the Layer-0 full observer

Attempt 12 (`d20-395c0e0d`) is currently **analysis-grade**, not signed target
evidence.  The checked-in five-file return is internally intact (`4/4` entries
listed by its `SHA256SUMS` verify), and its summaries report:

```text
off: A-B=0 bytes, B-C=0 bytes
on:  A-B=477 bytes / 227 elements, B-C=0 bytes
coarse interval: Layer 0 layer_input fingerprint exact -> layer_output fingerprint red
```

That package does **not** bind those summaries to the remote bounded shards.
It omits the remote `PREFLIGHT.json`, `COLLECTED.json`, `COMPLETE.json`, root
`SHA256SUMS`, compact-bundle verification, raw-log identity, and Kubernetes
terminal receipt.  The returned on-arm classification is also a minimized
copy: it omits the canonical classifier's `anchors`,
`first_difference_signatures`, `mixed_first_difference_signatures`,
`replay_ledger_receipts`, and `expected_layer` fields.  Therefore its current
gate is only `COARSE_FIRST_RED_INTERVAL`; it is not the final
`FIRST_RED_LOCALIZED` gate and it does not authorize a numerical repair.

### Why the previous return did not contain the complete evidence chain

Do not describe this as proof that the runtime failed to upload its data.  The
large seam JSON/NPZ payloads were intentionally left in GCS because the two
arms report approximately 6.6 GiB and 5.5 GiB of observer data.  The missing
step was the **post-run return audit**: the executor committed a manually
minimized receipt plus two classifier summaries and hashed only those four
small files.  It did not run the checked-in wide-seam GCS audit and return the
audit package that proves the classifiers are bound to the remote root or the
compact bundle.

The old Handoff made this mistake easier: its top `START HERE` section still
described Phase D2 publication, a later generic section referred to the older
replay audit, and the Attempt-12 entry jumped directly to the next launch.
There was no fail-closed Attempt-12 return checklist adjacent to that entry.
This section supersedes those stale operational instructions.  It does not
assert that the remote terminal objects are present or absent; the audit below
answers that question mechanically.

### Bucket-capable executor: perform this read-only audit now

Do not launch TPU/Kubernetes and do not retain, return, or commit the
token-bearing bundle.  The checked-in audit may fetch that compact tar into a
temporary directory solely to verify its internal manifest, then deletes the
scratch directory.  Use a clean checkout containing the published Attempt-12
receipt and run:

```bash
cd /home/yuxuan/code_rl_repro/sequence_packing/tunix
test -z "$(git status --porcelain)"
git fetch origin yuxzhang/canon-zero-tim
git pull --ff-only origin yuxzhang/canon-zero-tim

RECEIPT=canon-zero-tim/tasks/v1-apc-m15-target-debug/evidence/v1_apc_m15_attempt12_paired_d20_20260827/receipt.json
RETURN=/tmp/v1-apc-m15-attempt12-d20-gcs-audit
test -f "$RECEIPT"
test ! -e "$RETURN"

bash canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/run_m15_wide_seam_gcs_salvage.sh \
  "$RECEIPT" "$RETURN" /mnt/disks/tunix-data
(cd "$RETURN" && sha256sum -c SHA256SUMS)
```

Despite the historical `salvage` name, this wrapper accepts the two roots from
the supplied receipt and is the correct read-only verifier for Attempt 12.  It
checks both classifier aliases, all three terminal markers, the root manifest,
the compact bundle's internal manifest, and source identity.  The
token-bearing tar is verified in scratch space and excluded from the return.

### Exact return contract

Return the complete small `$RETURN` directory without editing or reformatting
any JSON.  It must contain:

1. `SALVAGE_SUMMARY.json`;
2. `PACKAGING.txt`;
3. `SHA256SUMS`;
4. `off.classification.json` and `on.classification.json` when present.

Also return, without copying token arrays:

5. the terminal line printed by the wrapper, including `status`, summary SHA,
   manifest SHA, and return path;
6. independent `sha256sum -c SHA256SUMS` output;
7. the exact full source SHA, both JobSet names, Attempt number, and the
   Kubernetes terminal status for each arm;
8. for each arm, either the immutable raw-log object identity plus SHA/size or
   a self-hashed text excerpt containing every line with `CANON_ALIGN_PRE`,
   `P3_APC_CONFIG`, `Prefix cache hit rate`, `CONTROLLED_EXIT`, `FATAL`, or
   `Traceback`.

The audit summary must mechanically report the presence/hash/source fields for
`PREFLIGHT.json`, `COLLECTED.json`, and `COMPLETE.json`, whether a root
manifest exists, whether the classifier is manifest-bound, and whether the
compact bundle's internal manifest passes.  A prose statement or a new
four-file summary is not an acceptable substitute.

Acceptance is exactly:

```text
status=LAYER_SELECTED
next_action=render full observer only at layer 0
off classifier=M15_OBSERVER_CONTROL_EXACT
on classifier=M15_LAYER_FIRST_RED_LOCALIZED
source conflicts=[]
both arms evidence_bound=true
```

`INCOMPLETE`, `SOURCE_MISMATCH`, missing terminal markers, missing manifest
binding, a failed hash, off-arm red, or B-C red is a hard stop.  Preserve the
return and repair/recover only the missing evidence; do not launch `d21` and do
not change model numerics.

Only after this audit passes may a separately approved paired Layer-0 full
observer run be rendered from the exact Attempt-12 source.  That run must use
all 15 checkpoints listed in the Phase-D document, rerun both APC-off and
APC-on arms, and reach `M15_INTERNAL_FIRST_RED_LOCALIZED` /
`FIRST_RED_LOCALIZED` before Phase E may propose a repair.  More diagnostic
rounds are not a substitute for the full Layer-0 checkpoint walk.

## Background — Phase D2 durability contract

The published source implements an evidence-transport repair, not an APC
numerical fix. The intended runtime contract is:

```text
observer JSON+NPZ complete
  -> bounded shard (<=32 pairs, <=256 MiB)
  -> upload archive+SHA
  -> remote read-back verify
  -> SHARD_COMPLETE
  -> classifier reads sealed shard union only
  -> WIDE_ROUND_COMPLETE
  -> COLLECTED
  -> postflight COMPLETE
```

`m15-wide-v1` also bypasses the redundant legacy incident ledger. The replay
envelope, request journal, seam/tail pairs, pre-alignment record and capsule
remain authoritative. No RoPE, attention, KV, LM-head, loss, backward,
optimizer, B-arm reset, or production APC behavior changes.

The Phase D2 source has passed host and pinned exact-image gates. The following
host gates must be rerun after any further edit:

```bash
cd /mnt/disks/tunix-data/worktrees/m15_wide_observer_0826

python3 -m unittest discover \
  -s canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts \
  -p 'test_*.py'
bash canon-zero-tim/tests/p38_serving/test_gcs_persistence.sh
python3 canon-zero-tim/.claude/skills/manage-canon-flags/scripts/audit_flag_registry.py \
  --repo . --changed-base origin/yuxzhang/canon-zero-tim
python3 canon-zero-tim/tests/manage_canon_flags/test_audit_flag_registry.py
bash -n canon-zero-tim/cluster/steps/00_env.sh \
  canon-zero-tim/cluster/steps/90_run.sh \
  canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/persist_p38_gcs.sh \
  canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/p38_live_snapshot_worker.sh
git diff --check
```

The fake-GCS persistence test is the required forced-death gate. It must report
`m15_shards=bounded-survive-abrupt-exit`, and both `COLLECTED.json` and
`COMPLETE.json` must be absent in that simulated interrupted run. The source
mismatch negative must report `source_mismatch=rejected`.

Pinned image `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`
terminated with `V1_HP_EXACT_IMAGE_PASS ... apc_m15_carrier=66
m15_durability=1 ...`. The image gate uses a read-only Git common-directory
mount so the existing live `git rev-parse HEAD` source check runs inside the
container; no receipt-only provenance shortcut was admitted.

Claim ceiling now:

```text
ANALYSIS_GRADE_COARSE_LAYER_0 / REMOTE_EVIDENCE_BINDING_PENDING /
NUMERICAL_FIX_NOT_AUTHORIZED
```

See [Phase D2](phases/phase-d2-durable-wide-shards.md).

## Historical — Attempt-9 read-only GCS inventory (complete)

The first salvage pass is complete and self-verifies 2/2 files under
`evidence/v1_apc_m15_attempt9_gcs_salvage_20260827/`. It established two
separate defects:

1. both registered Attempt-0 roots contain `PREFLIGHT.json`, but the six
   expected post-preflight objects (`COLLECTED.json`, `COMPLETE.json`, root
   `SHA256SUMS`, both classifier aliases, and the compact bundle) are absent;
2. both runtime preflight markers name source
   `3f159250c4781b3faafde238f768457a0478446b`, while the later prose receipt
   names the nonexistent full SHA
   `3f159250917fa9ee6062fbe7554f67644fcffec9`.

Therefore the receipt's claimed `0/1329` byte verdict and 2,313 tensor records
are not signed or reproducible evidence. Do not infer a layer from them. The
salvage wrapper checked seven exact object names; it did **not** enumerate
other objects that might survive under those roots. The later full inventory
did enumerate them and found only `PREFLIGHT.json` in each arm, so Attempt 9
is irrecoverable. The command below is retained only as historical provenance.

From a bucket-capable checkout of the latest published operator branch, run
exactly. This command downloads no object payload and mutates no GCS state:

```bash
cd /home/yuxuan/code_rl_repro/sequence_packing/tunix
git fetch origin yuxzhang/canon-zero-tim
BASE_SHA="$(git rev-parse origin/yuxzhang/canon-zero-tim)"
WORKTREE=/mnt/disks/tunix-data/worktrees/m15_attempt9_full_inventory_20260827
test ! -e "$WORKTREE"
git worktree add --detach "$WORKTREE" "$BASE_SHA"
cd "$WORKTREE"

RECEIPT=canon-zero-tim/tasks/v1-apc-m15-target-debug/evidence/v1_apc_m15_attempt9_paired_d15_20260826/receipt.json
RETURN=canon-zero-tim/tasks/v1-apc-m15-target-debug/evidence/v1_apc_m15_attempt9_gcs_full_inventory_20260827
test -f "$RECEIPT"
test ! -e "$RETURN"
command -v gcloud >/dev/null

python3 - "$RECEIPT" "$RETURN" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys

receipt_path = Path(sys.argv[1])
output = Path(sys.argv[2])
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
output.mkdir(parents=True)
records = {}
for arm, key in (("off", "control_arm_off"), ("on", "treatment_arm_on")):
  root = receipt[key]["gcs_source_uri"].rstrip("/")
  completed = subprocess.run(
      ["gcloud", "storage", "ls", "--recursive", root + "/**"],
      check=False,
      capture_output=True,
      text=True,
  )
  if completed.returncode:
    raise SystemExit(
        f"GCS inventory failed for {arm}: rc={completed.returncode} "
        f"stderr={completed.stderr[-500:]}"
    )
  prefix = root + "/"
  relative = []
  for raw in completed.stdout.splitlines():
    value = raw.strip()
    if not value or value.endswith(":"):
      continue
    if not value.startswith(prefix):
      raise SystemExit(f"unexpected inventory entry outside {arm} root")
    relative.append(value[len(prefix):])
  records[arm] = sorted(set(relative))

summary = {
    "schema": "m15-apc-attempt9-full-object-inventory-v1",
    "receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
    "receipt_source_commit": receipt["source_commit"],
    "runtime_source_commit": "3f159250c4781b3faafde238f768457a0478446b",
    "source_identity_matches": False,
    "payloads_downloaded": False,
    "remote_state_mutated": False,
    "objects": records,
    "object_counts": {arm: len(values) for arm, values in records.items()},
}
inventory = output / "OBJECT_INVENTORY.json"
inventory.write_text(
    json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8"
)
packaging = output / "PACKAGING.txt"
packaging.write_text(
    "M15 Attempt-9 full object-name inventory\n"
    "payloads_downloaded=0\n"
    "remote_state_mutated=0\n"
    "source_identity_matches=0\n",
    encoding="utf-8",
)
manifest = output / "SHA256SUMS"
manifest.write_text(
    "".join(
        f"{hashlib.sha256((output / name).read_bytes()).hexdigest()}  {name}\n"
        for name in ("OBJECT_INVENTORY.json", "PACKAGING.txt")
    ),
    encoding="ascii",
)
print(json.dumps(summary["object_counts"], sort_keys=True))
PY

(cd "$RETURN" && sha256sum -c SHA256SUMS)
python3 - "$RETURN/OBJECT_INVENTORY.json" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
print(json.dumps(value["objects"], sort_keys=True, indent=2))
PY
```

Return exactly these three small files, the two printed object counts, the
independent `sha256sum -c` output, and `git status --short`:

```text
OBJECT_INVENTORY.json
PACKAGING.txt
SHA256SUMS
```

Do not return full bucket roots, credentials, environment dumps, raw logs,
NPZs, compact bundles, or token contents. Do not download anything yet. Do not
commit or push unless the user separately authorizes that exact evidence-only
action.

Interpretation is mechanical:

| Full inventory | What the execution agent does |
|---|---|
| Any object other than `PREFLIGHT.json` exists | stop and return the inventory; the analysis owner prepares a narrowly scoped downloader/classifier for those exact names |
| Each arm contains only `PREFLIGHT.json` | stop and return the inventory; classify Attempt 9 as irrecoverable from its registered GCS roots |
| Listing fails or contains an out-of-root entry | hard stop; return stderr and do not retry with broader permissions |

Regardless of inventory outcome, the later receipt's source SHA remains
invalid and cannot authenticate the run. Attempt 11/d17 also remains
inconclusive: it collected roughly 2,100 observer records per arm in the pod,
but the legacy incident ledger exceeded 2 GiB before classifier/bundle
persistence. No current result selects a layer or authorizes a numerical fix.

## Historical decision — do not rerun before Phase D2 certification

Before any new target run, implement and certify all four durability changes:

1. wide mode must bypass the redundant legacy P38 incident ledger rather than
   raising its byte bound;
2. bounded observer shards must upload incrementally while the worker is
   alive;
3. the classifier must run from persisted shards and write `COLLECTED`, then a
   self-hashed manifest, then `COMPLETE` from the surviving worker;
4. runtime source identity must come from the executing checkout and agree
   with the rendered source SHA.

Rehearse forced failure after one shard and require that the shard, source
marker, and an `INCONCLUSIVE` terminal receipt survive. Keep one diagnostic
round only. After host/exact-image packaging gates pass, a new DP8xTP8 off/on
pair still requires separate user approval.

The one-host ladder is exhausted: real scheduler publication, 32-request
composition, `continue_decode=8`, and full M15 chronology all stayed exact on
DP1xTP4. The root remains a scale/topology seam. Do not run another one-host
replay and do not guess a RoPE/page repair.

This source prepares a known-red target localization run. It changes no model
arithmetic and keeps production APC off. The first target run attaches one
identical observer to an APC-off control and APC-on treatment:

- all 36 layer input/output fingerprints;
- final norm and terminal tail;
- positions 960..4096, covering the historical 1226 and Attempt-6 prompt
  boundaries;
- exact request/call/token-prefix/page receipts;
- automatic M15-aware classification and compact selected-record bundle.

After the user explicitly approves commit/push and the exact source SHA is
available, render only with:

```bash
cd /home/yuxuan/code_rl_repro/sequence_packing/tunix
SOURCE_SHA=<40-character-published-sha>
RUN_ID=<fresh-unique-label>
OUT=/tmp/v1-apc-m15-wide-${RUN_ID}
test ! -e "$OUT"
python3 canon-zero-tim/cluster/render_v1_apc_m15_target_debug.py \
  --source-commit "$SOURCE_SHA" \
  --run-id "$RUN_ID" \
  --observer layer \
  --output-dir "$OUT"
sha256sum "$OUT"/*.yaml
```

Expected manifests:

```text
jobset-v1-apc-m15-off-layer.yaml
jobset-v1-apc-m15-on-layer.yaml
```

If both allocations are available, the user may submit the two standalone
commands without waiting between them. Do not append pipes, `tee`, `&&`, or a
monitor:

```bash
kubectl apply -f "$OUT/jobset-v1-apc-m15-off-layer.yaml"
```

```bash
kubectl apply -f "$OUT/jobset-v1-apc-m15-on-layer.yaml"
```

Interpret control first. The off arm must remain A-B=0 and B-C=0. The on arm
must keep B-C=0; if red, it must emit one of:

```text
M15_LAYER_FIRST_RED_LOCALIZED
M15_HIDDEN_EXACT_TAIL_FIRST_RED_LOCALIZED
```

If the first result selects layer `L`, do not guess or edit YAML. A separately
approved follow-up is rendered by:

```bash
python3 canon-zero-tim/cluster/render_v1_apc_m15_target_debug.py \
  --source-commit "$SOURCE_SHA" \
  --run-id "${RUN_ID}-full-l${L}" \
  --observer full \
  --seam-layer "$L" \
  --output-dir "/tmp/v1-apc-m15-${RUN_ID}-full-l${L}"
```

Return exactly:

1. full source SHA, both JobSet names, attempts, Kubernetes terminal states,
   and both GCS Attempt-0 URIs;
2. the complete `CANON_ALIGN_PRE` line for each arm;
3. both `p38_seam.classification.json` files and their SHA-256;
4. the `CANON_APC_M15_SEAM_BUNDLE` path/size/SHA marker for each arm;
5. on red: `classification`, `gate`, `selected_layer`, `last_exact_boundary`,
   `first_red_boundary`, `coverage`, and `source_interval` from the JSON;
6. any nonzero return code plus complete stderr/raw-log tail.

The compact bundle contains real token/capsule material. Under the dedicated
`m15-wide-v1` contract it is uploaded only to the task's already authorized
P38 evidence prefix, after classification from sealed shards. Do not copy it
to any other location or return it through chat/Git.

Current claim ceiling is `WIDE OBSERVER READY / TARGET NOT RUN / ROOT CAUSE
NOT LOCALIZED`. See [Phase D](phases/phase-d-wide-target-observer.md).

## Historical Phase C replay input and Attempt-6 evidence

Attempt 6 paired execution (`d12-9f91d930`, source commit `9f91d93001dd5b44659f062626eb93fc65e6fcb4`) ran on 64 TPUs (DP8xTP8) for both control and treatment arms, persisted complete raw payloads to GCS Attempt-0 roots, and successfully passed the GCS replay audit `run_m15_replay_gcs_audit.sh`:

- **Control Arm (`canon-v1-apc-m15-off-d12-9f91d930`)**:
  - Rollout: 2,560 requests completed, 0.0% prefix cache hit rate.
  - JAX Pre-alignment: `[CANON_ALIGN_PRE] step=0 verdict=PASS N_action=117415 bounds=[('S_decode_vs_S_prefill', 0), ('S_prefill_vs_T_old', 0)]` ($A-B=0, B-C=0$).
  - GCS Audit Verdict: `CONTROL_GREEN` (`receipt_sha256=c9550f73...`, `manifest_sha256=b91cd34c...`).
  - Terminal: Controlled exit code 42, zero backward, zero optimizer commits.
- **Treatment Arm (`canon-v1-apc-m15-on-d12-9f91d930`)**:
  - Rollout: 2,560 requests completed, **92.9%** prefix cache hit rate.
  - JAX Pre-alignment: `[CANON_ALIGN_PRE] step=0 verdict=FAIL N_action=119565 bounds=[('S_decode_vs_S_prefill', 1770), ('S_prefill_vs_T_old', 0)]` (**Captured exact mismatch of 1,770 bytes / 748 elements**).
  - Canonical first mismatch: row 201, completion position 0, logical prefix 1066; its request starts at call 187 and the bounded interval ends at call 188.
  - Earliest request belonging to any red row: row 245 at call 164.
  - First fully captured tensor incident: row 245, request `400-bc7daec5`, serving call 565, DP rank 0, slot 29, `num_computed_tokens=1248`, 296 exact joins. This is not the onset.
  - Mismatch Capsule: 15,148 bytes (`sha256:9e79a18d...`).
  - Producer Unit: 762 KB, 256 rows (`m15_producer_unit.npz`).
  - Replay Envelope: 103.7 MB, 3,027 calls (`m15_replay_envelope.jsonl`).
  - GCS Audit Verdict: `FRESH_TARGET_RED_FROZEN` (`receipt_sha256=557801a3...`, `manifest_sha256=93f56a0a...`).
  - Terminal: Controlled exit code 42, zero backward, zero optimizer commits.

### Phase C Execution Summary (Replay Input Plan Prepared)

`run_m15_replay_gcs_prepare.sh` was executed on `canon-v1-apc-m15-on-d12-9f91d930/attempt-0` and terminated with:
```text
[M15.APC.REPLAY.PREPARE] COMPLETE status=M15_REPLAY_INPUT_PLAN_READY_NOT_EXECUTED analysis_sha256=a3c381f8d5e8143ac266a96fb082679e86d85a96eb749255696aaebe649ceff0 manifest_sha256=ed0c67413e51acd639e79dbb95df8698ed8e0386ea606dfcfcb0b1a4fb3e2355 prefix_sha256=b8c00fc704cdd698318a2088c70b9593737a996da8eda1e55d98986d5a8f30a7 prefix_bytes=3938394 red_rows=201,245 replay_prefix_end_call=188 destination=gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/canon-v1-apc-m15-on-d12-9f91d930/attempt-0/derived/m15-replay-input-plan-v1
```

Evidence sealed in `evidence/v1_apc_m15_attempt6_paired_d12_20260825/` and `evidence/v1_apc_m15_replay_input_plan_d12_20260826/`.

Current claim ceiling:
```text
M15_REPLAY_INPUT_PLAN_READY_NOT_EXECUTED
```

Next phase: **Phase D (Deterministic replay and tensor-level tap)**.

## Historical Phase-D replay proposal — superseded by the target observer

The replay plan is prepared (`replay-prefix-plan.jsonl`, 188 calls, 3.94 MB).
The next action is to execute the deterministic replay harness using the saved 188 prefix calls to:
1. Prime the cache from Call 1 through Call 188.
2. Verify reproduction of the exact 1,770 byte A-B mismatch on Row 201/245.
3. Tap the attention / RoPE / block-table layers to pinpoint the exact numerical root cause.

## Scope and current ceiling

This task is the independent APC/prefix-cache numerical lane. It does not
change production APC defaults and it does not authorize a TPU launch,
commit, or push.

Current immutable facts:

- one-host Qwen3-8B DP1xTP4 Phase3 G-A through G-D is exact;
- M15 `m15i` on DP8xTP8 is A-B red by 1389 bytes / 760 elements and B-C exact;
- the first historical red is row 192, completion position 0, logical prefix
  1226;
- the historical archive has hashes but not reversible tokens/request order/
  cache lineage, so `m15i` itself is not an exact replay carrier;
- no numerical source has been repaired and all production full recipes remain
  APC-off.
- Attempt 0 (`canon-v1-apc-m15-off-d3-eb58954f`) is `INCONCLUSIVE`: it never
  reached alignment or created a serving capture. Its command selected
  `m15/main` on the CLI but omitted `CANON_P57_WORKLOAD_CANDIDATE` and
  `CANON_P57_DATA_SPLIT`, which the FrozenLake entrypoint requires to match.
- The bootstrap repair carries exact `m15/main` identity through the renderer,
  profile, and Step-00 resolver and preserves the package-safe module
  entrypoint. It changes no APC, model, alignment, backward, or optimizer math.
- Attempt 1 (`canon-v1-apc-m15-off-d4-283cb67e`) is also prelearner-only: it
  passed all overlays and GCS preflight, then legacy P38 DP16 assertions
  rejected the M15 carrier's `(mini_batch_size, sampler_is)=(32, none)` and
  would next have rejected its DP8 workload/unit identity. No A/B/C verdict or
  replay payload was produced.
- The bounded follow-up keeps legacy P38 exactly at `frozenlake`, DP16,
  8 x 4-prompt producer units and token IS, while only
  `CANON_APC_M15_TARGET_DEBUG=off|on` admits `frozenlake-dp8-tp8`, DP8,
  1 x 32-prompt unit and no IS. Cross-mode and partial geometry negatives are
  executable host tests. It changes no numerical code.
- Attempt 2 (`canon-v1-apc-m15-off-d7-41a2043c`) finally reached the real
  DP8xTP8 rollout and completed more than 1,800 serving calls plus all four
  standard capture strata. It did not reach A/B/C classification: the
  incident ledger saturated at call 326 (268,192,266 bytes) and the drain tail
  later entered the production `continue_decode` path, which the old
  single-path observer rejected.
- Removing `CANON_CONTINUE_DECODE=8` is explicitly rejected because `m15i`
  used it. Attempt 3 proved patch 27's remaining assumption was also wrong:
  APC-on can enter `continue_decode` before four standard tensor strata have
  been captured. Append-only patch 28 admits that registered M15 program path
  from its first call and writes only the dedicated host replay envelope;
  standard tensor capture and generic request/incident evidence stay
  standard-only. The M15-only incident/replay ceiling remains 2 GiB. A frozen
  red must attest A=`standard+continue_decode` and B=`standard`; unknown paths
  and any non-M15 use remain fatal, while a B-side continue path is rejected
  by packaging.
- Attempt 4 (`canon-v1-apc-m15-on-d10-618eb775`) proves patch 28 reached the
  end of the real rollout: 2,560 requests completed, prefix-cache hit rate was
  92.5%, and solve ratio was 0.203. It then failed before A/B/C because the
  generic alignment gate did not admit this carrier's signed
  `sampler_is=None` recipe. Its two committed files pass `SHA256SUMS`; they
  prove the fatal admission boundary but are not a complete replay package.
- Attempt 5 paired run (`d11-a909fda1`, source `a909fda1`) produced hash-valid
  Git snapshots for both arms.  The snapshots show 0.0% cache hits off and
  approximately 89.4%--97.5% on, but contain no alignment, sampler,
  controlled-exit, classification, or GCS-terminal markers.  The accompanying
  receipt is an unverified summary until the GCS-side audit returns.

Claim ceiling: `FULL_REPLAY_CARRIER_FROZEN_REPLAY_NOT_RUN`.

The exact remote procedure is in [RUNBOOK.md](RUNBOOK.md). The execution agent
must run those commands rather than constructing a new carrier by hand.

## Prepared bounded carrier

The new renderer creates a matched pair from one committed source tree:

- `off`: APC-off shared-serving control;
- `on`: production-congruent cache-read treatment.

Both use the exact M15 main geometry: DP8xTP8, 32 prompts, 8 generations,
256 trajectories, concurrency 256, `vllm_max_num_seqs=32`, batched tokens 256,
15 turns, prompt 4096, response 8192, temperature 0.7, seed 42. Both stop after
one strict pre-alignment round with zero backward and zero optimizer commit.
Both deliberately use `--sampler_is=none`: A supplies rollout logprobs as the
old-policy source and no token-IS correction weights may exist.

The only intended cross-arm values are
`CANON_APC_M15_TARGET_DEBUG=off|on` and derived
`CANON_VLLM_ENABLE_PREFIX_CACHING=0|1`; a structural test rejects any other
document difference after arm-path normalization.

A must attest:

```text
prompt_logprobs=None
logprobs=1
skip_reading_prefix_cache=False
```

B must attest `reset_prefix_cache=True` and zero cached tokens. The classifier
rejects any B-C byte difference, optimizer marker, wrong source, wrong command,
missing capsule/journal/incident join, or M15 classifier failure hidden behind
the expected controlled exit code 42.

For every arm, postflight/GCS collection preserves:

- `m15_producer_unit.npz`: all 256 final token/logprob rows;
- `m15_replay_envelope.jsonl`: every serving call's exact host-side dispatch,
  request, DP slot, prefix hash/position and physical page table.

For a fresh red, postflight additionally creates
`p38_serving_capture/m15_first_red_replay/` with:

- `first_red_capsule.npz`: the earliest exact incident row's complete prompt,
  completion, masks, A/B/C, policy version, and sampling values;
- `first_red_contract.json`: request/call/position, DP slot, physical pages,
  page generations, and co-batch request IDs;
- `SHA256SUMS`.

It then creates `m15_full_replay_carrier/`. Its request-row join proves that
every scheduled token history comes from the saved 256-row producer and that
the first-red request/call/pages match the incident ledger. The full carrier
records scheduler dispatch order but has not yet forced that order through a
replay harness, so it remains input evidence rather than a mechanism verdict.

## CPU validation already run

From the independent worktree:

```bash
cd /home/yuxuan/code_rl_repro/worktrees/m15_eval_fix_0825
python3 canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/test_analyze_m15i_evidence.py -v
python3 canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/test_classify_m15_apc_target_run.py -v
python3 canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/test_package_first_red_replay.py -v
python3 canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/test_package_full_replay_carrier.py -v
python3 canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/test_target_carrier.py -v
python3 canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/test_resolved_env.py -v
python3 canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/test_classify_p38_serving_capture.py -v
python3 canon-zero-tim/tests/p3_prefix_cache/test_contract.py -v
python3 canon-zero-tim/.claude/skills/manage-canon-flags/scripts/audit_flag_registry.py \
  --repo . --changed-base ff913a84
bash -n canon-zero-tim/cluster/steps/00_env.sh canon-zero-tim/cluster/steps/90_run.sh
git diff --check
```

All task-specific tests and the flag audit pass. After the Attempt-4 repair,
the task carrier is 46/46, P57 is 146/146, V1 CPU is 67/67, and flags are
378/378. The pinned image
`sha256:418dc632...e53a` was then run on the final runtime/test tree. The first
attempt exposed an image-only test PATH defect (`python3` lives under
`/usr/local/bin`); after the test inherited the active interpreter directory,
the full rerun terminated with `V1_HP_EXACT_IMAGE_PASS ...
apc_m15_carrier=46 ... manifests=3`. This is exact-image admission, not a
one-host numerical replay or DP8xTP8 target result.

The worktree was initially created at reference `687b2bd6...`. The operator
branch later advanced through `ff913a84` to `9f79cc56`; the intervening raw-log,
P58 seed, and P64 shared-entrypoint changes were reviewed, then the release
commit was rebased without conflict before the final aggregate gate.

## Historical — Attempt-4 next-launch instructions (superseded)

This section records how the earlier paired carrier was admitted.  It is not
the current operation.  Follow the `START HERE` Attempt-12 GCS audit instead.

Do not relaunch source `eb58954f...`; its missing signed identity is
deterministically invalid. Patch 28 has passed the targeted and aggregate
exact-image gates on the current tree. First publish the observer repair, then
verify that the committed tree is identical to the admitted tree and render
only from that new full SHA. One paired-launch approval covers both target
arms. Use a unique label and a new output directory:

```bash
cd /home/yuxuan/code_rl_repro/sequence_packing/tunix
SOURCE_SHA=<full-committed-sha>
RUN_ID=<new-unique-label>
OUT=/tmp/v1-apc-m15-${RUN_ID}
python3 canon-zero-tim/cluster/render_v1_apc_m15_target_debug.py \
  --source-commit "$SOURCE_SHA" \
  --run-id "$RUN_ID" \
  --output-dir "$OUT"
sha256sum "$OUT"/*.yaml
```

Do not edit the rendered YAML. Do not render from a dirty or abbreviated SHA.

If any runtime or test file changes before publication, rerun the dependency-
complete pinned-image gate against the immutable production image:

```bash
cd /home/yuxuan/code_rl_repro/sequence_packing/tunix
bash canon-zero-tim/tests/v1_phase4/run_exact_image.sh \
  sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a
```

The post-fix terminal must include `apc_m15_carrier=46`. The current
Attempt-4 repair tree produced that terminal with exit 0. This remains a CPU/image
admission gate, not a DP8xTP8 numerical result.

The installed-runner test is not a string-only predicate check: with zero
captured records and zero strata it executes `_p38_serving_begin`, requires the
M15 replay ledger call, and requires generic incident capture to remain absent.

Before applying the YAML, the rendered environment must contain all four
members of one workload identity:

```text
CANON_P57_WORKLOAD_CANDIDATE=m15
CANON_P57_DATA_SPLIT=main
--p57_workload_candidate=m15
--p57_data_split=main
--sampler_is=none
```

The checked-in renderer and real Step-00 resolver now enforce this and reject
wrong identity or file-path-entrypoint negatives before TPU work begins.
It must also contain `CANON_CONTINUE_DECODE=8`,
`CANON_P38_SERVING_CAPTURE_EXPECTED_PATH=standard`, and
`CANON_P38_INCIDENT_MAX_BYTES=2147483648`. This combination is intentional:
the tensor records stay single-path while the replay envelope attests the
mixed production tail.

Do not relaunch Attempt-4 source `618eb775...`: it deterministically lacks the
new sampler admission. Attempt 4 has no matched fresh APC-off arm, so it cannot
substitute for either member of the new pair below.

## Historical — paired-launch contract used by Attempt 12

After one explicit paired-launch approval, issue both standalone commands
immediately. Do not append a pipe, `tee`, `&&`, or a monitor to either command:

```bash
kubectl apply -f "$OUT/jobset-v1-apc-m15-off.yaml"
```
```bash
kubectl apply -f "$OUT/jobset-v1-apc-m15-on.yaml"
```

Do not wait for off to finish before submitting on. The arms may execute and
fail concurrently; they still use distinct JobSets, logs, and JobSet-derived
GCS roots. A failure in one arm must not cancel or delete the other arm.

Interpretation remains control-first even though execution is concurrent.
First classify off and require `CONTROL_GREEN`, B-C zero, plus
`PREFLIGHT.json`, `COLLECTED.json`, and `COMPLETE.json`. If off is red or
inconclusive, preserve and report the on package, but do not use on to make an
APC-specific causal claim.

After a green control, the treatment has two admissible outcomes:

- `FRESH_TARGET_RED_FROZEN`: proceed to Phase C/D using the bundled first-red
  carrier; do not infer RoPE/page/cache mechanism yet;
- `TARGET_NOT_REPRODUCED`: this one bounded target observation was exact at
  representative depth/cache occupancy; it does not prove APC fixed.

Any `INCONCLUSIVE`, B-C red, missing join, missing GCS terminal marker, or
unexpected optimizer/backward evidence is a hard stop.

## Historical — generic replay return contract

Large GCS evidence remains durable and must not be added wholesale to Git. On
the machine that can read the bucket, run the checked-in GCS audit:

```bash
bash canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/run_m15_replay_gcs_audit.sh \
  gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/<jobset>/attempt-0
```

Return these small outputs plus their SHA values:

1. `PREFLIGHT.json`, `COLLECTED.json`, `COMPLETE.json`, and remote
   `SHA256SUMS`;
2. `serving-classification.json`;
3. the derived audit URI and `RETURN_RECEIPT.json`;
4. from the derived audit:
   - `m15-classification.json`;
   - on a red, `first-red-contract.json`, `replay-contract.json`, and
     `request-row-joins.jsonl`;
5. the raw log, or if it is too large, an immutable raw-log URI/SHA plus every
   line containing the alignment-pre marker, M15 APC marker prefix,
   `P3_APC_CONFIG`,
   `Prefix cache hit rate`, `CONTROLLED_EXIT`, and `FATAL`;
6. the exact source SHA, JobSet name, attempt number, GCS prefix, and Kubernetes
   terminal status.

The audit script verifies the remote root manifest and both nested replay
manifests before uploading its own `SHA256SUMS` last. Do not call a hash-valid
subset a complete upload; the three terminal markers and required
classifications remain separate completeness gates.

## Rollback

The new selector is default-off, so publication does not change production
behavior. Reverting this bounded carrier must remove the
renderer/profile/classifier/marker additions as one concern. Production
recipes remain APC-off throughout.


## Attempt 7 M15 Target Debug Runs (d13-663cb547)

Attempt 7 dual-arm execution (`d13-663cb547`, source commit `663cb5474490173cfaf33fce3a323d95e5fc2ee1`) was launched on dual 64-TPU allocations:
- **Control Arm (`canon-v1-apc-m15-off-d13-663cb547`)**: Successfully uploaded `PREFLIGHT.json`, terminated during startup.
- **Treatment Arm (`canon-v1-apc-m15-on-d13-663cb547`)**: Successfully uploaded `PREFLIGHT.json`, terminated during startup.
- Retained evidence: `evidence/v1_apc_m15_attempt7_d13_20260826/`.

## Attempt 8 M15 Target Debug Runs (d14-3820b168 Phase D Wide Layer Observer)

Attempt 8 dual-arm execution (`d14-3820b168`, source commit `3820b168e37080ea9c4e2e2832810a950a7c493f`) ran on dual 64-TPU allocations (DP8xTP8) with all 36-layer observers attached:
- **Control Arm (`canon-v1-apc-m15-off-d14-3820b168`)**:
  - Rollout: 256 trajectories completed, 0.0% prefix cache hit rate, solve rate 15.2%.
  - Pre-alignment: `verdict=PASS`, 0 differing bytes on A-B and B-C.
  - Collected >2,420 wide observer records across all 36 layers.
  - Terminal: Controlled exit code 42, zero backward, zero optimizer commits.
- **Treatment Arm (`canon-v1-apc-m15-on-d14-3820b168`)**:
  - Rollout: 256 trajectories completed, **93.1%** prefix cache hit rate, solve rate 20.7%.
  - Pre-alignment: `[CANON_ALIGN_PRE] step=0 verdict=FAIL N_action=122951 bounds=[('S_decode_vs_S_prefill', 1191), ('S_prefill_vs_T_old', 0)]` (Reproduced 1,191 diff bytes between $S_{\text{decode}}$ and $S_{\text{prefill}}$).
  - Evidence: `evidence_sha256=740a34978c4519a0cd696aa6dc283ad111dcdb8f0bf8cbbe02a4c62722426854`.
  - Collected >2,112 wide observer records across all 36 layers.
  - Terminal: Controlled exit code 42, zero backward, zero optimizer commits.
- Retained evidence: `evidence/v1_apc_m15_attempt8_paired_d14_20260826/`.

## Attempt 9 M15 Target Debug Runs (d15-3f159250 Phase D Wide Layer Observer)

The historical receipt under
`evidence/v1_apc_m15_attempt9_paired_d15_20260826/` claimed a completed paired
run, APC-off `0/0`, APC-on A-B red by 1,329 bytes, and 2,313 tensor records.
That claim is **superseded as unsigned prose**:

- its full source SHA does not exist in the repository;
- both real GCS preflight markers instead identify the valid commit
  `3f159250c4781b3faafde238f768457a0478446b`;
- the expected-object GCS audit found no `COLLECTED`, `COMPLETE`, root manifest,
  classifier, or compact bundle in either arm.

Only bucket writability at startup and the runtime-marker source identity are
currently verified. The historical numerical values must not select a layer,
close a gate, or justify a repair. See
`evidence/v1_apc_m15_attempt9_gcs_salvage_20260827/`; the full object-name
inventory at the top of this handoff is the only admitted next operation.

## Attempt 12 M15 Target Debug Runs (d20-395c0e0d Phase D Wide Layer Observer)

Attempt 12 paired dual-arm execution (`d20-395c0e0d`, source commit `395c0e0de8626c96e85457b997efddd2dd2dec48`) ran on dual 64-TPU allocations (DP8xTP8) with all 36-layer observers attached:
- **Control Arm (`canon-v1-apc-m15-off-d20-395c0e0d`)**:
  - Rollout: 256 trajectories completed, **0.0%** prefix cache hit rate, solve rate **18.4%** (47/256).
  - Pre-alignment: `[CANON_ALIGN_PRE] step=0 verdict=PASS N_action=118186 bounds=[('S_decode_vs_S_prefill', 0), ('S_prefill_vs_T_old', 0)]` ($A-B=0, B-C=0$).
  - Classification: `M15_OBSERVER_CONTROL_EXACT`, `gate=OBSERVER_REACHED_EXACT_ENDPOINT`.
  - Seam records: 2,474 pairs across all 36 layers verified bitwise exact.
  - Terminal: Controlled exit code 42, zero backward, zero optimizer commits.
- **Treatment Arm (`canon-v1-apc-m15-on-d20-395c0e0d`)**:
  - Rollout: 256 trajectories completed, **92.5%** prefix cache hit rate, solve rate **22.7%** (58/256).
  - Pre-alignment: `[CANON_ALIGN_PRE] step=0 verdict=FAIL N_action=115908 bounds=[('S_decode_vs_S_prefill', 477), ('S_prefill_vs_T_old', 0)]` ($B-C=0$ exact, captured 477 differing bytes across 227 elements).
  - Layer Fingerprint Comparison:
    - Layer 0 `layer_input`: 100% bitwise exact between uncached prefill writer (Gen 0) and cached readers (Gen 1..7).
    - Layer 0 `layer_output`: First red boundary identified (`first diff=(0, 'layer_output')`).
    - Cached readers (Gen 1 vs Gen 2 vs ... vs Gen 7): 100% bitwise identical to each other (`total differing = 0`).
  - Classification: `M15_LAYER_FIRST_RED_LOCALIZED`, `gate=COARSE_FIRST_RED_INTERVAL`, `selected_layer=0`.
  - Terminal: Controlled exit code 42, zero backward, zero optimizer commits.
- Retained evidence: `evidence/v1_apc_m15_attempt12_paired_d20_20260827/`.

### Follow-up action after the `START HERE` audit passes

Do not run the command below from the currently checked-in four-member summary
alone.  The read-only Attempt-12 GCS audit at the top of this Handoff must first
return `LAYER_SELECTED`, both arms evidence-bound, and no source conflict.
Only then, with separate user approval for the paired target launch, render:
Render and launch the Layer 0 full 15-checkpoint observer:
```bash
python3 canon-zero-tim/cluster/render_v1_apc_m15_target_debug.py \
  --source-commit "395c0e0de8626c96e85457b997efddd2dd2dec48" \
  --run-id "d21-full-l0" \
  --observer full \
  --seam-layer 0 \
  --output-dir "/tmp/v1-apc-m15-d21-full-l0"
```

## Attempt 13 M15 Target Debug Runs (d32-7d30f382 Phase D Layer-0 Full Observer)

The five checked-in files report that Attempt 13 (`d32-7d30f382`, source
commit `7d30f3827480e6f9d5ae972f55ca4d16f07de6df`) executed a paired dual-arm
DP8xTP8 Layer-0 full-observer run.  The following values are retained as the
submitted summary, not as a replayed official classification:

- **Control Arm (`canon-v1-apc-m15-off-d32-7d30f382`)**:
  - Rollout: 256 trajectories completed, **0.0%** prefix cache hit rate, solve rate **16.0%** (41/256).
  - Pre-alignment: `[CANON_ALIGN_PRE] step=0 verdict=PASS N_action=112544 bounds=[('S_decode_vs_S_prefill', 0), ('S_prefill_vs_T_old', 0)]` ($A-B=0, B-C=0$).
  - Classification: `M15_OBSERVER_CONTROL_EXACT`, `gate=OBSERVER_REACHED_EXACT_ENDPOINT`.
  - Seam records: 2,474 pairs across Layer 0 verified bitwise exact.
  - Terminal: Controlled exit code 42, zero backward, zero optimizer commits.

- **Treatment Arm (`canon-v1-apc-m15-on-d32-7d30f382`)**:
  - Rollout: 256 trajectories completed, **92.7%** prefix cache hit rate, solve rate **19.9%** (51/256).
  - Pre-alignment: `[CANON_ALIGN_PRE] step=0 verdict=FAIL N_action=115396 bounds=[('S_decode_vs_S_prefill', 239), ('S_prefill_vs_T_old', 0)]` ($B-C=0$ exact, captured 239 differing bytes).
  - Submitted fingerprint checkpoint summary:
    - `[0] layer_input`: 🟢 EXACT MATCH ($\Delta = 0.0$)
    - `[1] input_norm`: 🟢 EXACT MATCH ($\Delta = 0.0$)
    - `[2] q_proj`: 🟢 EXACT MATCH ($\Delta = 0.0$)
    - `[3] k_proj`: 🟢 EXACT MATCH ($\Delta = 0.0$)
    - `[4] v_proj`: 🟢 EXACT MATCH ($\Delta = 0.0$)
    - `[5] q_norm`: 🟢 EXACT MATCH ($\Delta = 0.0$)
    - `[6] k_norm`: 🟢 EXACT MATCH ($\Delta = 0.0$)
    - `[7] q_post_rope`: 🟢 EXACT MATCH ($\Delta = 0.0$)
    - `[8] k_post_rope`: 🟢 EXACT MATCH ($\Delta = 0.0$)
    - **`[9] rpa_output`**: red fingerprint; the reported
      `7.1857e8` is an integer-fingerprint delta, not an activation `max_abs`.
    - `[10..14] o_proj, residual, post_norm, mlp, layer_output`: 🔴 RED (downstream propagation).
  - Classification: `M15_INTERNAL_FIRST_RED_LOCALIZED`, `gate=INTERNAL_FIRST_RED_LOCALIZED`, `selected_layer=0`.
  - Terminal: Controlled exit code 42, zero backward, zero optimizer commits.
- Retained evidence: `evidence/v1_apc_m15_attempt13_paired_d32_20260828/`.

### Evaluator correction — Attempt 13 is not classifier-replayable yet

- The local subset's four hashed payloads are intact, but the complete
  three-round and terminal evidence chain is absent.
- Both submitted classifiers omit `diagnostic_round`; the on-arm classifier
  also omits the official anchor/signature/replay-ledger fields.
- The classifier in the claimed runtime source fails while resolving the
  reported `rpa_output` source anchor.  The local correction points to the
  observer patch, but has not yet been run against the private compact bundle;
  the submitted `source_interval` therefore remains unverified.
- Fingerprint equality is not full-tensor byte equality.  The available subset
  supports only an RPA/attention-call **interval hypothesis**, not a proven
  block-table or cached-KV-read defect.
- Claim ceiling:
  `ATTEMPT13_SUBSET_HASH_VALID / OFFICIAL_CLASSIFIER_NOT_REPLAYABLE /
  RPA_ATTENTION_CALL_INTERVAL_HYPOTHESIS`.
- Next action: follow the top `START HERE` section and recover d32's six
  per-round classifiers from GCS.  Do not launch again and do not enter Phase E
  until that return is independently hashed and the official classifier can be
  replayed.
