# P58.19 — three-round coarse decode/prefill seam localization

## Deliverable

Prepare one default-off exact-geometry P58 diagnostic that keeps the signed
Qwen3-4B-Instruct-2507 Zero-HP carrier and performs three sequential
frozen-weight rollout/alignment rounds inside one 128-chip JobSet.  A bounded
coarse observer records every Transformer block input/output, final norm, and
the measured terminal logprob path for standard-path A/B rows.  Every round is
classified and durably sealed before the learner may queue the next round.

This is a localization experiment, not a new numerical treatment.  It must not
change sampling, precision, fixed-M geometry, checked-VMA, loss, backward, or
optimizer state.  It does not reuse the P58.18 ON/OFF selector.

## Bound source and evidence

- Worktree base when the phase was opened:
  `7fed8307a6bdf9f5887593b83dcd5dc83051b1f0`.
- P58.18 evidence:
  `evidence/p58aba01_checked_vma_aba_wave/`.
- P58.18 classifier decision: `CHECKED_VMA_NOT_SUFFICIENT` with exact B-C and
  finite A-B RED in ON-A, OFF, and ON-B.
- First recorded A-B mismatches in the three exact-geometry arms occur at
  logical KV prefixes 3,438, 3,880, and 4,032 and at action-run starts after an
  environment token.  This justifies a bounded initial coarse observation
  band; absence of a join remains INCONCLUSIVE rather than evidence of repair.

## Shape and program ledger

| Boundary | Contract |
|---|---|
| TPU allocation | one 4x4x8 slice, 128 chips |
| rollout role | DP8 x TP8, 64 chips |
| trainer role | DP8 x TP8, 64 disjoint chips |
| prompts / generations | B8 x G16 = 128 trajectories per round |
| sampling | temperature 1.0, top-p 1.0, top-k 0, engine seed 42 |
| response / turns | 16,384 tokens / 50 turns |
| scheduler | concurrency 128, per-DP max-num-seqs 16 |
| fixed-head caller-global M | 2,048 |
| shard-local / canonical-kernel M | 256 / 256 |
| optimizer | resident but unreachable; zero commits |

## Single selector and derived observer contract

Register one workload selector, `CANON_P58_SEAM_LOCALIZATION=coarse`.  The
renderer derives every subordinate P38 observer, precheck, round, durability,
and output-path field.  Hand-setting a partial tuple fails closed.  Selector
absence leaves production P58 Zero-HP unchanged, and every non-P58 workload
must reject the selector.

The initial observation window is `[3072, 4608)`.  Coarse layer fingerprints
are diagnostic, non-cryptographic summaries; equality is not promoted to
full-tensor byte equality.  The independent endpoint gate remains the source
of truth for A-B/B-C.

## Phases and gates

### L0 — ledger reconciliation

- Promote the sealed P58.18 target result into `state.md` and `plan.md`.
- Record that checked-VMA is not sufficient; do not claim it is unrelated or
  that a P67 repair is disproved.

Exit gate: current state and plan name P58.19 as the only active phase.

### L1 — observer construction and neutrality

- Add the single P58 selector and end-to-end flag trace.
- Reuse the existing layer/full observer implementation without changing its
  numerical checkpoints.
- Run host truth-table, renderer/profile, classifier, and shared durability
  construction tests.  The legacy DP1xTP4 one-host seam carrier deliberately
  excludes the Qwen3 TP8 model observer and cannot be used as neutrality
  evidence for this selector.

Exit gate: host truth table and neighboring-workload negatives pass; the exact
TP8 observer remains target-gated.  Real v5p execution remains separately
authorized and must not be inferred from the old DP1xTP4 carrier.

### L2 — exact-geometry three-round carrier

- Render exactly one 128-chip Zero/full JobSet with three sequential precheck
  rounds, strict finite A-B plus exact B-C admission, controlled exit 42, and
  no VJP/backward/optimizer activity.
- Each round produces a mismatch capsule, coarse seam/tail records, one
  round-local classification, a deterministic archive, SHA manifest, remote
  read-back verification, `ROUND_COMPLETE`, and only then a round ACK.
- Avoid periodic multi-GiB observer snapshots; the round seal is the durability
  boundary.

Exit gate: renderer/profile/real `00_env.sh` contracts and synthetic
three-round sealing/classification tests pass.

### L3 — target localization

Launch remains separately approval-gated.  A valid target package requires
three round completions, three P58 classifications, exactly three precheck
markers, one controlled exit, exact B-C in every round, finite positive A-B
evidence, and zero backward/optimizer commits.

Decision table:

| Observation | Decision |
|---|---|
| same first-red layer/checkpoint in all three rounds | promote a three-round 15-checkpoint fine scan of that layer |
| different checkpoints but one common coarse interval | preserve as analysis-grade and refine only that interval |
| backbone exact and terminal path first red | route to the P38 LM-head/log-normalizer discriminator |
| layer-0 input already red | inspect embedding/position/KV handoff before Transformer blocks |
| B-C red, observer endpoint drift, missing round, or no join | INCONCLUSIVE; repair the observer/durability prerequisite only |

## Claim ceiling

Construction and one-host results do not localize the production seam.  A
three-round exact-geometry result localizes only the observed standard-path
interval; continue-decode rows remain explicitly unobserved, and fingerprint
equality is not full-tensor equality.  No result from this phase certifies
backward, optimizer correctness, full training, convergence, or general
Zero-TIM readiness.
