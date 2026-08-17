# P38.2u — One-pass terminal discriminator

Status: implementation and local gates complete. No 64-TPU launch is
authorized until the user has reviewed, committed, and published the source.

## Entering evidence

The admitted P38s18r2 target-aware v3 bundle is complete as analysis evidence:
374/374 SHA entries and both audits pass, all 32 A-B-red actions join A and B,
and B-C remains exact. The first measured terminal difference is
`raw_log_normalizer` for 26 actions and `raw_target_logit` for 6. All recorded
36-layer and final-norm fingerprints are equal. The run completed only round 0
of three, so its scientific verdict remains `INCONCLUSIVE_PARTIAL_RUN`.

Those fingerprints do not prove byte equality of the final hidden row or the
full vocabulary logits. The open question is therefore:

```text
exact selected final hidden row
  -> lm_head raw logits
  -> fixed vocabulary-block max/exp-sum
  -> existing tail observer
  -> production target logprob
```

## Deliverable

Add one default-off observer to the already registered stock P38 envelope. For
every selected A/B row it records:

1. the exact selected post-final-norm hidden row bytes;
2. six uint32 signatures per 256-logit block for both raw and processed logits;
3. raw/processed block maxima, stable block exp sums, row maxima, and observer
   log normalizers; and
4. the existing six tail values, including the production logprob.

The observer never fetches full logits to host, never changes model, sampling,
or scoring callables, and is bounded by `CANON_P38_TERMINAL_MAX_BYTES`.
Records are target-aware and self-SHA-checked.

Floating reduction must run through one shared fixed-four-row executable.
Shape-dependent gather executes separately and only materializes four rows on
device. This boundary is mandatory: the first v5p prototype fused gather and
reduction, compiled different A/B executables from different source-logit
shapes, and falsely reported 148 reduction differences while the production
A-B endpoint and raw-logit signatures were exact.

## Local gate

Before a target launch, all of the following must pass:

- both pinned Qwen overlays install with every manifest hash exact and all
  runner tests green;
- CPU positive/negative controls detect a one-bit raw-logit mutation and
  reject variable observer row geometry;
- classifier fixtures cover every registered terminal branch and reject a
  missing, ambiguous, or conflicting target-aware red-point join;
- one real Qwen3-8B DP1xTP4 run completes three frozen rounds with zero
  backward and zero optimizer commits;
- local production A-B remains exact, every captured A row has a B join, and
  the corrected classifier returns `terminal_rows_exact`; and
- the observer-off/on production alignment endpoints are bitwise neutral.

The one-host run is a construction and neutrality gate. It cannot reproduce or
repair the 64-chip carrier.

The final real-v5p receipt is
`../artifacts/p38_2u_terminal_discriminator_onehost_0817.md`. It records
33/33 manifest files, 34/34 runner tests per overlay, three exact frozen
rounds, 155 exact A/B terminal joins, a real-TPU one-bit negative, and exact
observer-off/on production endpoints. Two earlier observer constructions were
rejected locally before publication because they created shape-dependent
false differences.

## Target run

Exactly one fresh stock diagnostic is admitted after publication:

```text
DP16xTP4
concurrency=256
prefix_cache=off
seam_mode=layer
terminal_tail=1
terminal_discriminator=1
frozen_rounds=3
backward=0
optimizer_commits=0
```

The terminal classifier must receive every immutable mismatch capsule and
must join every capsule red action to exactly one A and one B terminal row. A
classification over unrelated clean rows is rejected even if it says exact.
The existing round seal, live GCS snapshots, SHA inventories, controlled exit,
`COLLECTED`, and `COMPLETE` contracts remain mandatory.

## Decision table

| First differing stage on a capsule red action | Decision |
|---|---|
| `pre_lm_head_hidden` | Reopen the upstream seam; fingerprint equality hid a selected-row byte difference |
| `lm_head_logits` | Localize lm_head projection/program envelope; reducer is downstream |
| `vocab_block_reduction` | Raw signatures/maxima are equal; localize the raw block exp-sum/merge program |
| `logits_processing` | Raw rows are exact but processed-logit signatures differ; localize sampling/logits transforms |
| `processed_vocab_block_reduction` | Processed signatures/maxima are equal; localize their exp-sum/merge program |
| `production_tail_only` | All independent checkpoints are exact; localize production gather/subtract wiring |
| multiple stages | Preserve the mixed carrier; never force one root cause |
| missing/ambiguous red join or observer drift | `INCONCLUSIVE`; repair instrumentation before interpreting values |

This run localizes the first divergent terminal subprogram. It does not
guarantee that the selected numerical repair will be completed in the same
launch.

## Claim ceiling

Exact hidden-row records are exact only for selected rows. Raw/processed logit
equality is represented by multiword block fingerprints plus exact maxima, not
by full logit bytes. Legacy tail-observer intermediate drift is reported but
does not outrank the shared fixed-four-row observer; one-host proved the old
shape-dependent tail observer can differ while its production endpoint is
exact. A block-reduction classification is diagnostic until the selected
repair passes a separate strict 64-chip A=B=C validation. No result from this
phase alone closes zero-TIM or admits full training as bitwise exact.

## Rollback and sunset

The feature is default-off. Remove the renderer flag or set
`CANON_P38_TERMINAL_DISCRIMINATOR=0` to restore the pre-P38.2u path. Retire the
observer, classifier, and its three env flags after the terminal carrier is
repaired and strict GSM8K/FrozenLake validation is green.
