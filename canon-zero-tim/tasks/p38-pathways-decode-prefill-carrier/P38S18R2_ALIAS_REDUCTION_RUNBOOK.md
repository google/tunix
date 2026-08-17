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

## Stage A — implement and review the offline reducer

1. Start from a clean detached worktree at the current
   `origin/yuxzhang/canon-zero-tim` tip.
2. Read the P38.2s phase completely.
3. Extend the existing reduction and audit machinery with
   `--require-tail`, including independent alias/conflict accounting for seam
   and tail payloads.
4. Add all focused positive, alias, conflict, missing-record, tamper, empty-URI,
   and direct-classifier negative controls registered by P38.2s.
5. Run the focused gates and provide:
   - diff summary;
   - exact test commands and terminal output;
   - proposed commit message;
   - revert command; and
   - confirmation that no TPU, source GCS mutation, commit, or push occurred.
6. Stop for user approval. Do not commit or push automatically.

The existing reducer is not sufficient unchanged: it reduces hidden seam
records only and invokes the classifier without `require_tail=True`.

## Stage B — execute one immutable remote reduction

Only after Stage A is reviewed and published, use its exact clean full SHA on
a machine with access to `yuxzhang-tunix-models`.

Fixed source and destination:

```bash
ROUND_URI="gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/canon-p38-fl-stock-p38s18r2-10fe951f/attempt-0/rounds/000000"
DEST="gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/canon-p38-fl-stock-p38s18r2-10fe951f/attempt-0/derived/p38s18r2-round0-seam-tail-reduction-v2"
```

Fail before downloading if `$DEST/files/SHA256SUMS` already exists. Never
overwrite v1, v2, or source objects.

The remote wrapper must perform, in order:

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
9. upload the immutable bundle only after the local audit passes; and
10. print one terminal line containing verdict, red points, matched seam/tail
    keys, aliases, conflicts, classifier rc, manifest SHA, audit SHA,
    destination, and analysis source commit.

The wrapper must populate `source_gcs_uri`; an empty value is a hard packaging
error. Preserve `classifier.stdout`, `classifier.stderr`, and `classifier.rc`
for both successful and failed classification.

## Required return to the reviewing agent

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

Prepare an append-only evidence CL only after the local audit passes. Stop
before commit or push unless the user explicitly approves each action.

## Acceptance

A usable scientific classification requires 32/32 joined red points, 64/64
matched seam keys, 64/64 matched tail keys, no payload conflicts, mandatory
tail join, and standalone auditor PASS. The scientific verdict still reads
`INCONCLUSIVE_PARTIAL_RUN`, because the original three-round run stopped after
Round 0.

Missing keys, seam conflicts, tail conflicts, classifier failure, or auditor
failure stay `INCONCLUSIVE`; they do not authorize a new TPU run until the
returned compact candidates have been reviewed.

## Prompt for a background-free remote agent

> Work only on P38s18r2 offline evidence. First read
> `canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/HANDOFF.md`, then
> `canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/phases/p38-2s-round0-alias-aware-seam-tail-reduction.md`, then
> `canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/P38S18R2_ALIAS_REDUCTION_RUNBOOK.md`
> completely. Do not launch TPU work and do not rerun the old direct
> whole-directory classifier. Stage A is to implement and test alias-aware
> seam-plus-tail reduction with `require_tail=True`; stop before commit/push
> for user review. After that implementation is separately approved and
> published, Stage B runs once against the fixed immutable Round 0 URI and a
> new v2 destination. Verify all source SHA/inventory contracts, resolve only
> byte-identical duplicates as aliases, preserve conflicts fail-closed, run
> the official classifier and standalone auditor, and return the entire small
> reduced bundle plus audit—not a prose summary. Never overwrite source/v1/v2
> objects, never fabricate missing rounds, and never call this one-round result
> signed evidence.
