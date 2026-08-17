# P38s18r2 alias-aware seam-and-tail reduction runbook

This is the current operator card for P38s18r2. Read
`phases/p38-2s-round0-alias-aware-seam-tail-reduction.md` first. The task is a
zero-TPU analysis of the already sealed Round 0; do not launch or relaunch a
JobSet.

## Why the old command is retired

Commit `a514c3bf` returned a valid failure receipt from the direct classifier:

```text
classifier.rc=1
SeamError: duplicate seam token-prefix record
verdict=INCONCLUSIVE_REMOTE_CLASSIFICATION
```

The raw source is not missing. Its object listing contains 972 seam JSON/NPZ
pairs and 972 tail JSON/NPZ pairs, and its 3,894-entry manifest closes exactly
against the 3,896-object listing after adding the two sealing files. The raw
observer records overlap; a whole-directory classifier requires uniqueness
and therefore stops before joining the 32 red points.

Do not rerun the same direct classifier. Do not pick the first or last
duplicate. Do not hand-write a classification.

## Stage A — review and publish the completed offline reducer

The implementation is complete in the review worktree. It adds:

- `reduce_p38_seam_tail_evidence.py`, the immutable-round seam-plus-tail
  reducer;
- `audit_p38_seam_tail_reduction.py`, an independent auditor that rescans every
  candidate, recomputes both alias maps, and reruns the official classifier;
- `p38s18r2_round0_contract.json`, the only admitted production contract;
- `run_reduce_p38s18r2_round0_on_gcp.sh`, the one-command GCS wrapper; and
- `test_reduce_p38_seam_tail_evidence.py`, including fake-GCS end-to-end and
  32-red-point/64-key fixtures.

Before publication, review the diff and require:

1. all registered positive, alias, conflict, missing-record, tamper, empty-URI,
   direct-classifier-negative, 32-red-point, and fake-GCS gates to pass;
2. the existing seam classifier/reducer/wrapper and tail tests to stay green;
3. Python compilation, shell syntax, JSON parsing, credential scan, and
   `git diff --check` to pass; and
4. the review handoff to provide:
   - diff summary;
   - exact test commands and terminal output;
   - proposed commit message;
   - revert command; and
   - confirmation that no TPU, source GCS mutation, commit, or push occurred.

Stop for user approval. Do not commit or push automatically.

The existing reducer is not sufficient unchanged: it reduces hidden seam
records only and invokes the classifier without `require_tail=True`.

## Stage B — execute one immutable remote reduction

Only after Stage A is reviewed and published, use its exact clean full SHA on
a machine with access to `yuxzhang-tunix-models`. Do not reconstruct the
command by hand. From the clean published checkout run exactly:

```bash
TASK="canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier"
RETURN="$TASK/evidence/p38s18r2/seam-tail-reduction-v2"

test -z "$(git status --short)"
bash "$TASK/scripts/run_reduce_p38s18r2_round0_on_gcp.sh" \
  "$TASK/scripts/p38s18r2_round0_contract.json" \
  /tmp \
  "$RETURN"
```

The wrapper reads every source/destination URI, SHA, count, mode, and byte
ceiling from the checked-in contract. No `.env` or manual override is part of
this execution.

Fixed source and destination:

```bash
ROUND_URI="gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/canon-p38-fl-stock-p38s18r2-10fe951f/attempt-0/rounds/000000"
DEST="gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/canon-p38-fl-stock-p38s18r2-10fe951f/attempt-0/derived/p38s18r2-round0-seam-tail-reduction-v2"
```

Fail before downloading if `$DEST/files/SHA256SUMS` already exists. Never
overwrite v1, v2, or source objects.

The wrapper performs, in order:

1. list the complete source prefix;
2. download to a new `mktemp` directory;
3. verify `ROUND_COMPLETE.json`, `ROUND_INVENTORY.json`, source manifest SHA,
   exact manifest inventory, and every file SHA;
4. require actual counts `seam_records=972`, `tail_records=972`, object listing
   3,896, source manifest 3,894, and capsule red points 32;
5. run alias-aware seam-plus-tail reduction for all 64 A/B keys;
6. run the official classifier with `--mode layer --require-tail` over only
   the byte-preserved selected records when and only when no key is missing or
   conflicting;
7. write the compact bundle defined by P38.2s and a self-excluding
   `SHA256SUMS`;
8. run the standalone auditor from that compact bundle alone;
9. upload the immutable bundle only after the local audit passes;
   `files/SHA256SUMS` is uploaded last and is the remote completion marker;
10. copy the same audited compact result to `$RETURN` for an evidence CL; and
11. print one terminal line containing verdict, red points, matched seam/tail
    keys, aliases, conflicts, classifier rc, manifest SHA, audit SHA,
    destination, and analysis source commit.

The wrapper must populate `source_gcs_uri`; an empty value is a hard packaging
error. Preserve `classifier.stdout`, `classifier.stderr`, and `classifier.rc`
for both successful and failed classification.

## Required return to the reviewing agent

The wrapper writes `$RETURN/files/`, `$RETURN/bundle-audit.json`, and its SHA.
Return all of the following; a prose summary alone is insufficient:

1. the terminal COMPLETE line;
2. the complete compact `files/` directory, including every raw candidate used
   to prove a unique result, equivalent alias, or conflict, plus the capsule;
3. the standalone bundle-audit JSON and its SHA-256;
4. exact contents of `verdict.json`, `REDUCTION_MANIFEST.json`,
   `AMBIGUITY_AUDIT.json`, `classification.json` when present,
   `classifier.stderr`, and `SHA256SUMS`;
5. the immutable source and destination GCS URIs;
6. the full analysis source commit and classifier SHA; and
7. `git status --short` from the clean execution checkout.

After the command, rerun the auditor from the returned directory:

```bash
python3 "$TASK/scripts/audit_p38_seam_tail_reduction.py" \
  --bundle-dir "$RETURN/files" \
  --output /tmp/p38s18r2-return-audit.json
cmp /tmp/p38s18r2-return-audit.json "$RETURN/bundle-audit.json"
git status --short
```

Prepare an append-only evidence CL containing `$RETURN` only after both audits
pass. Stop before commit or push unless the user explicitly approves each
action.

## Acceptance

A usable scientific classification requires 32/32 joined red points, 64/64
matched seam keys, 64/64 matched tail keys, no payload conflicts, mandatory
tail join, and standalone auditor PASS. The scientific verdict still reads
`INCONCLUSIVE_PARTIAL_RUN`, because the original three-round run stopped after
Round 0.

Missing keys, seam conflicts, tail conflicts, classifier failure, or auditor
failure stay `INCONCLUSIVE`; they do not authorize a new TPU run until the
returned compact candidates have been reviewed.

Reducer rc 4 or 5 after a printed COMPLETE line is an intentional fail-closed
scientific receipt, not permission to rerun or overwrite the destination.

## Prompt for a background-free remote agent

> Work only on P38s18r2 offline evidence. First read
> `canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/HANDOFF.md`, then
> `canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/phases/p38-2s-round0-alias-aware-seam-tail-reduction.md`, then
> `canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/P38S18R2_ALIAS_REDUCTION_RUNBOOK.md`
> completely. Do not launch TPU work and do not rerun the old direct
> whole-directory classifier. After Stage A is approved and published, do not
> rewrite its reducer/auditor/wrapper. Verify a clean checkout at the approved
> full SHA, run the exact Stage B command with the checked-in contract and
> return directory, and return the entire generated compact result plus the
> terminal COMPLETE line and `git status --short`—not a prose summary. Never
> overwrite source/v1/v2 objects, fabricate missing rounds, manually choose a
> duplicate, or call this one-round result signed evidence. Stop before commit
> or push until the user explicitly approves that separate act.
