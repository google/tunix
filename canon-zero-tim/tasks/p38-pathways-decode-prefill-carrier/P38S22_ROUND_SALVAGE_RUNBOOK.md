# P38s22 independent-round salvage runbook

This is the only current P38s22 operator card. It performs one zero-TPU,
read-only audit of the three independently sealed diagnostic rounds already in
durable storage. It does not repair or fabricate the absent run-level root
postflight.

Read `phases/p38-2w2-p38s22-round-seal-salvage.md` first. Do not relaunch
P38s22 and do not run a classifier or copy a numerical field by hand.

## Why this replaces the root-first audit

Offsite audit v1 returned a valid sealed `rc=4` receipt. It proved the root
`SHA256SUMS`, `COLLECTED.json`, and `COMPLETE.json` were unavailable, but it
stopped before checking the three round tar objects. The three round completion
markers and manifests survived and identify distinct deterministic archives.

Round seals were designed to survive loss of the final root postflight. This
audit therefore verifies each round archive first and keeps root completeness
as an independent, explicitly false claim. A round-level PASS cannot be cited
as root postflight, terminal localization, backward, or optimizer evidence.

## Precondition

Use a clean named worktree at the exact user-approved full SHA containing this
runbook and its scripts. The machine needs read access to the existing evidence
bucket, `python3`, NumPy, and either `gcloud` or `gsutil`. It needs no TPU,
Kubernetes, Pathways, model weights, `.env` edit, or writable bucket.

```bash
set -euo pipefail
SOURCE_COMMIT="<USER_APPROVED_FULL_SHA_CONTAINING_ROUND_SALVAGE>"
SALVAGE_WORKTREE="/tmp/p38s22-round-salvage-${SOURCE_COMMIT:0:12}"
SALVAGE_BRANCH="local/p38s22-round-salvage-${SOURCE_COMMIT:0:12}"
git fetch origin yuxzhang/canon-zero-tim
test "$(git rev-parse FETCH_HEAD)" = "$SOURCE_COMMIT"
test ! -e "$SALVAGE_WORKTREE"
git worktree add -b "$SALVAGE_BRANCH" "$SALVAGE_WORKTREE" "$SOURCE_COMMIT"
cd "$SALVAGE_WORKTREE"
test "$(git rev-parse HEAD)" = "$SOURCE_COMMIT"
test -z "$(git status --short)"
```

## The one command

Run from the outer worktree root containing `canon-zero-tim/`:

```bash
set -euo pipefail
TASK="canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier"
RETURN="$TASK/evidence/p38s22/round-salvage-v1"
test ! -e "$RETURN"

set +e
bash "$TASK/scripts/run_p38s22_round_salvage.sh" \
  "$TASK/scripts/p38s22_round_salvage_contract.json" \
  /tmp \
  "$RETURN"
salvage_rc=$?
set -e
test "$salvage_rc" -eq 0 -o "$salvage_rc" -eq 4

(cd "$RETURN" && sha256sum -c SHA256SUMS)
python3 -m json.tool "$RETURN/verdict.json"
python3 -m json.tool "$RETURN/AUDIT.json"
git status --short
printf 'P38S22_ROUND_SALVAGE_RC=%s\n' "$salvage_rc"
```

The wrapper owns every source identity, expected SHA, calculation, and verdict.
The operator must not add flags, edit the contract, substitute a URI, or repair
an `rc=4` result.

## Mechanical checks

For each of rounds 0, 1, and 2, the wrapper verifies:

1. acquisition status, byte size, and SHA without recording a source URI;
2. source commit, Attempt, diagnostic round, transport, and completion status;
3. archive, manifest, capsule, and logical-file-count SHAs against both marker
   and immutable contract;
4. deterministic tar structure and every member listed by the archived
   manifest;
5. stage inventory, one scoped pre-alignment record, cumulative request-journal
   schema/count, and scoped incident-ledger schema/count;
6. complete JSON/NPZ observer pairing, required KV evidence, and absence of
   seam/tail/terminal records;
7. algorithm PATHTRACE, frozen-round/no-update markers, and absence of terminal
   observer execution; and
8. A-B/B-C elements, bytes, `max_abs`, and action counts recomputed from the
   sealed capsule rather than copied from prose.

The acquisition ledger records missing optional root receipts as
`missing_or_unreadable`. It contains logical labels only, never remote URIs.

## Exact return contract

Return the complete generated directory:

```text
evidence/p38s22/round-salvage-v1/
  ACQUISITION.jsonl
  AUDIT.json
  SUMMARY.txt
  verdict.json
  receipts/
    PREFLIGHT.json
    ROUND_COMPLETE.round-000000.json
    ROUND_COMPLETE.round-000001.json
    ROUND_COMPLETE.round-000002.json
    ROUND_SHA256SUMS.round-000000
    ROUND_SHA256SUMS.round-000001
    ROUND_SHA256SUMS.round-000002
    # Optional only if available: COLLECTED.json, COMPLETE.json,
    # ROOT_SHA256SUMS
  SHA256SUMS
```

Also return the `RETURN_READY` line, `P38S22_ROUND_SALVAGE_RC`, and
`git status --short`. Do not commit or push until the user separately approves.

## Decision table

| Result | Decision |
|---|---|
| `PASS` + `ROUND_SEALED_GENERIC_LM_HEAD_ALGORITHM_PRESET_REJECTED`, three rounds, totals 143,464 / 66 / 111 / B-C 0 | Admit only the three-round forward discriminator; move to dedicated fixed-tile Pallas lm-head one-host gate |
| `PASS` with `root_postflight.receipts_present=false` | Expected; root/run-level postflight remains unadmitted and must be written explicitly |
| `rc=4` naming a missing/corrupt round tar, manifest, capsule, or scoped member | Preserve `INCONCLUSIVE`; do not infer round durability and do not relaunch TPU automatically |
| Any terminal observer record | Fail closed; P38s22 was a no-terminal-observer arm |

## Prompt for a background-free remote agent

> Work only on the zero-TPU P38s22 independent-round salvage. Read
> `canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/P38S22_ROUND_SALVAGE_RUNBOOK.md`
> completely and execute its precondition and “The one command” blocks from
> the exact full SHA supplied by the user. You are an operator only: do not
> interpret numbers, edit the contract, enter a source URI, run a classifier
> manually, launch TPU/Kubernetes, mutate durable storage, fabricate missing
> root receipts, or repair rc=4. Return the complete generated
> `round-salvage-v1` directory plus the requested terminal lines. Stop before
> commit or push until the user explicitly approves that separate action.
