# P38s22 offsite evidence-audit runbook

This is the only current P38s22 operator card. It is a zero-TPU, read-only
audit of the immutable Attempt-0 objects already stored in GCS. Do not relaunch
P38s22, run a classifier by hand, edit a receipt, or reconstruct a missing
number from prose. Read
`phases/p38-2w1-p38s22-offsite-evidence-audit.md` first.

## Purpose

The returned Git bundle contains enough endpoint evidence to reject
`DotAlgorithmPreset.BF16_BF16_F32`, but its newly returned durability receipts
are not admissible yet:

- each claimed `ROUND_ARCHIVE.tar` SHA equals the corresponding capsule NPZ
  SHA even though the checked-in transport always creates a tar containing
  `SHA256SUMS` plus its logical members;
- the hand-written receipt copied Round-0/1 action counts from P38s21; and
- a 66-point terminal classification was returned even though P38s22
  explicitly disabled the terminal observer and no raw terminal JSON/NPZ
  inputs were returned.

This audit reads the source objects beside GCS, verifies them mechanically,
and returns only a small sealed receipt. The remote operator performs no
analysis and writes no numerical conclusion.

## Precondition

Use one clean checkout of the exact user-approved published SHA containing the
offsite audit. The machine needs read access to the existing evidence bucket,
`python3`, NumPy, and either `gcloud` or `gsutil`. It needs no TPU, Kubernetes,
Pathways, model weights, `.env` edit, or writable GCS destination.

```bash
set -euo pipefail
SOURCE_COMMIT="<USER_APPROVED_FULL_SHA_CONTAINING_OFFSITE_AUDIT>"
AUDIT_WORKTREE="/tmp/p38s22-offsite-audit-${SOURCE_COMMIT:0:12}"
AUDIT_BRANCH="local/p38s22-offsite-audit-${SOURCE_COMMIT:0:12}"
git fetch origin yuxzhang/canon-zero-tim
test "$(git rev-parse FETCH_HEAD)" = "$SOURCE_COMMIT"
test ! -e "$AUDIT_WORKTREE"
git worktree add -b "$AUDIT_BRANCH" "$AUDIT_WORKTREE" "$SOURCE_COMMIT"
cd "$AUDIT_WORKTREE"
test "$(git rev-parse HEAD)" = "$SOURCE_COMMIT"
test -z "$(git status --short)"
```

Never substitute `HEAD`, another run ID, another Attempt, another bucket, or a
hand-written URI. The immutable source, expected source SHA, all three capsule
SHAs, and all expected endpoint numbers live in
`scripts/p38s22_offsite_audit_contract.json`.

## The one command

Run exactly this block from the outer checkout/worktree root that contains the
`canon-zero-tim/` directory:

```bash
set -euo pipefail
TASK="canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier"
RETURN="$TASK/evidence/p38s22/offsite-audit-v1"
test ! -e "$RETURN"

set +e
bash "$TASK/scripts/run_p38s22_offsite_audit.sh" \
  "$TASK/scripts/p38s22_offsite_audit_contract.json" \
  /tmp \
  "$RETURN"
audit_rc=$?
set -e
test "$audit_rc" -eq 0 -o "$audit_rc" -eq 4

(cd "$RETURN" && sha256sum -c SHA256SUMS)
python3 -m json.tool "$RETURN/verdict.json"
python3 -m json.tool "$RETURN/AUDIT.json"
git status --short
printf 'P38S22_OFFSITE_AUDIT_RC=%s\n' "$audit_rc"
```

`rc=0` means the immutable source and all receipts passed and the generic
lm-head algorithm preset was mechanically rejected. `rc=4` is also a valid
operator return: it means the script sealed an `INCONCLUSIVE` receipt naming
the exact missing, corrupt, or contradictory source object. The operator must
not repair that result, rerun TPU work, or change the contract.

## What the wrapper checks

The wrapper, not the operator:

1. downloads root `PREFLIGHT`, `COLLECTED`, `COMPLETE`, and the root manifest;
2. downloads and verifies every root-manifest member;
3. downloads each of the three round archives, manifests, and completion
   markers;
4. verifies the actual tar SHA, marker SHA, manifest SHA, deterministic tar
   structure, and every archived member SHA;
5. requires the six base round-stage members, including the real capsule,
   request journal, incident ledger, and round inventory; then verifies the
   JSONL line counts, schemas, round scopes, and staged/root pre-alignment
   identity rather than trusting inventory counters alone;
6. checks the sealed and root copies of every round capsule are identical;
7. recomputes A-B and B-C element/byte counts and `max_abs` from each capsule,
   and checks all registered action counts against the root pre-alignment
   records;
8. requires the algorithm PATHTRACE, three frozen-round markers, controlled
   exit 42, zero backward, and zero optimizer commits;
9. confirms that the terminal observer did not run and marks any terminal
   classification without raw terminal inputs unadmitted; and
10. emits a self-excluding `SHA256SUMS` over one small return directory.

The script never uploads or overwrites a GCS object. Large archives and NPZs
remain in the temporary directory and are deleted after the sealed receipt is
written.

## Exact return contract

For `rc=0`, return the complete generated directory, not screenshots or a
prose summary:

```text
evidence/p38s22/offsite-audit-v1/
  AUDIT.json
  SUMMARY.txt
  verdict.json
  receipts/
    PREFLIGHT.json
    COLLECTED.json
    COMPLETE.json
    ROOT_SHA256SUMS
    ROUND_COMPLETE.round-000000.json
    ROUND_COMPLETE.round-000001.json
    ROUND_COMPLETE.round-000002.json
    ROUND_SHA256SUMS.round-000000
    ROUND_SHA256SUMS.round-000001
    ROUND_SHA256SUMS.round-000002
  SHA256SUMS
```

For `rc=4`, return the complete generated directory exactly as written. It
contains every receipt that was fetched successfully; `AUDIT.json` names the
missing, corrupt, or contradictory object. Do not create a placeholder for a
missing receipt.

Also return the terminal `RETURN_READY` line, the printed
`P38S22_OFFSITE_AUDIT_RC`, and `git status --short`. Do not commit or push the
generated directory until the user separately approves that action.

## Acceptance and next action

Only `status=PASS` plus
`verdict=GENERIC_LM_HEAD_ALGORITHM_PRESET_REJECTED` closes the P38s22
durability debt. It does not create new terminal localization; the admitted
lm-head interval remains inherited from P38s21.

After PASS, do not run P38s22 again. The next scientific phase is the dedicated
fixed-tile Pallas lm-head: first a real-weight one-host M16/M256 bitwise and
negative-control gate, then one P38s23 64-TPU three-round frozen diagnostic.

## Prompt for a background-free remote agent

> Work only on the P38s22 zero-TPU offsite audit. Read
> `canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/P38S22_OFFSITE_AUDIT_RUNBOOK.md`
> completely. Use a clean checkout at the exact full SHA supplied by the user.
> Run its “The one command” block exactly, without editing the contract or
> adding environment overrides. You are an operator only: do not interpret
> numbers, run a classifier manually, launch TPU/Kubernetes work, mutate GCS,
> fabricate a receipt, or repair an `rc=4`. Return the entire generated
> `offsite-audit-v1` directory, the `RETURN_READY` line, the final rc line, and
> `git status --short`. Stop before commit or push until the user explicitly
> approves that separate action.
