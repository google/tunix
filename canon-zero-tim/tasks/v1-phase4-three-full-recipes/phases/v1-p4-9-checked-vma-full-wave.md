# V1.P4.9 — Checked-VMA three-full launch wave

Status: active; runtime and host/image construction admission complete.
Publication, final rendering, and target launch remain unrun. No target launch
is authorized by this phase file.

## Objective

Promote the P66 checked-VMA P59 backward repair into exactly the three Phase4
high-performance full recipes, then prepare one simultaneous launch wave:

- GSM8K Qwen3-1.7B DP16xTP4, 200 committed updates;
- FrozenLake P45 Qwen3-8B DP8xTP8, 300 committed updates; and
- FrozenLake M15/main Qwen3-8B DP8xTP8, 300 committed updates.

The three jobs are independent. One recipe's red stops only that recipe and
does not cancel another healthy full run.

## Motivation and current evidence

Attempt 7 proved that all three recipes could pass strict forward Zero-TIM,
but the old P59 TP>1 backward produced `1e21`-scale GSM8K gradients and
non-finite FrozenLake gradients before the first commit. P66 G1 localized the
cause to erased varying-manual-axis/replication ownership under the old nested
`check_vma=False` composition. P66 G1.5 then compared the repaired candidate
to ordinary JAX at the same model/input/cache/cotangent across six full-Qwen
endpoints; worst relative-L2 was `0.0052568`, all registered caps passed, and
the old unsafe arm remained an expected red.

This is strong one-host source-freeze evidence, not DP16xTP4 or DP8xTP8 target
certification. The three full jobs are the first target-topology optimizer and
convergence certification for the repaired path.

## Frozen contract

- Strict Zero-TIM is unchanged. Every expected `CANON_ALIGN_PRE` and
  `CANON_ALIGN` record must pass; any real FAIL kills that recipe.
- B-arm rescore remains an independent full recomputation. APC remains off in
  all three recipes.
- The repaired P59 path is default off and admitted only by the exact three
  `CANON_V1_HP_FULL=1` profiles. Partial bundles and neighboring profiles fail
  closed.
- P63 remains only an overflow-safe global-norm implementation. It may not
  turn a non-finite or unexplained huge gradient into an admitted update.
- A run label, output directory, JobSet name, XProf path, and evidence root are
  single-use. Failed evidence is retained.
- All three JobSets may be applied in one wave after user launch approval.
  There is no cross-recipe first-commit dependency.
- This preparation turn does not launch, commit, or push.

## G-A — Production flag and profile admission

- Add one descriptive default-off production flag for checked VMA; retain the
  P66 diagnostic spelling only for its immutable one-host carriers.
- Require the production flag in the exact GSM8K/P45/M15 full profiles,
  rendered environments, resolved-profile checks, flag registry, and final
  classifier.
- Emit a checked-VMA runtime receipt from the real P59 backward path. The
  classifier must reject the old unchecked path, a missing receipt, a wrong
  topology, or a partial bundle.

## G-B — First-update fail-closed admission

Before the first AdamW invocation, observe the complete accumulated gradient
without changing it and require:

- the registered microstep count and accumulator denominator;
- every element finite;
- at least one nonzero element;
- finite stable-L2 norm greater than zero and no greater than `1e6`; and
- strict alignment and per-microstep finite/activity checks already green.

The `1e6` threshold is a regression sentinel, not a clipping threshold. The
historical one-host maximum was `26.2`; the old P59 red was `1e21`-`1e22`.
Crossing the sentinel aborts before AdamW.

After AdamW, require the existing candidate evidence to prove finite gradient,
finite parameter delta, coherent learning-rate behavior, a valid step
transition, unchanged reference/accumulator contract, and a material update
when the learning rate permits it. Only after this function returns may the
outer learner synchronize weights or checkpoint. Emit one signed first-update
receipt. Missing either the precommit or admitted-commit receipt is fatal.

Later updates retain the existing per-step strict alignment, finite-gradient,
P63, P59 reduction, optimizer, and full-horizon gates.

## G-C — Carrier and classifier negatives

Focused host tests must cover:

- all three exact profiles and topologies;
- missing/zero/non-finite/over-threshold first accumulator;
- wrong denominator or microstep count;
- old unchecked P59, wrong profile, and partial checked-VMA bundle;
- missing, duplicate, or post-red receipts; and
- valid first-update plus complete-horizon classification.

The renderer must still produce exactly three distinct immutable manifests and
must assert the checked-VMA flag plus all existing JAX-cache, XProf, Perfetto,
APC-off, evaluation, checkpoint, and strict-alignment contracts.

## G-D — Admission and launch handoff

Run focused tests, V1/P57/P59/P66/APC suites, flag audit, syntax, manifest,
and diff hygiene. Then run the complete immutable-image gate on the exact
runtime tree. Host or image construction green does not certify target
behavior.

After user separately approves commit/push, render three fresh manifests from
the exact published 40-character SHA. The user, not this preparation turn,
will launch all three. The first target checkpoint for each is its own signed
first-update receipt; the terminal checkpoint is the full 200/300-update
postflight. Performance/XProf analysis starts only after the user reports the
launched run IDs.

## Decision table

| Observation | Verdict | Action |
|---|---|---|
| Any strict alignment FAIL | hard Zero-TIM red | stop that recipe; preserve evidence |
| Precommit non-finite or stable norm > `1e6` | backward regression | stop before AdamW; do not invoke P63 as an excuse |
| First optimizer evidence invalid | optimizer admission red | stop before weight sync/checkpoint |
| First update admitted, later update red | training red | stop at that update; preserve completed evidence |
| One recipe red while another is green | independent result | keep healthy jobs running |
| Three complete horizons and postflights PASS | target KEEP | proceed to matched XProf/performance analysis |

## Claim ceiling

Until target completion: `P66 SAME-POINT ORACLE PASS / HOST-IMAGE ADMISSION
PASS / PUBLICATION AND TARGET NOT RUN`. Do not claim serial
trajectory identity. The admitted claim is ordinary-JAX gradient correctness
within the registered oracle envelope plus strict Zero-TIM forward identity.

## Result log

### 2026-08-26T00:48:00Z — Production promotion and first-update gate

- Added default-off `CANON_P59_CHECKED_VMA` and
  `CANON_V1_HP_FIRST_UPDATE_GATE` to exactly the three full profiles and
  rendered environments. `00_env.sh` maps the descriptive production flag to
  the P66 internal compatibility spelling so the source-frozen adapter/shim
  implementation remains unchanged.
- The real learner emits one checked-VMA topology receipt per update. Before
  the first AdamW call, it validates and emits the full-accumulator finite,
  activity, stable-L2, microstep, and denominator receipt. After the existing
  optimizer transaction validates, it emits a `0 -> 1` finite/material update
  receipt before outer synchronization/checkpoint can proceed.
- Added pure positive/negative gates plus full-log classifier negatives for
  wrong topology/profile/chunks, missing/duplicate receipts, non-finite,
  zero, over-threshold, and incoherent optimizer evidence.
- Verified by V1 74/74, P57 146/146, P59 37/37, P66 16/16, APC 31/31, and
  flag audit 383/383. No target behavior is verified by these host tests.

### 2026-08-26T00:50:00Z — Fixed-image construction admission

- The complete immutable-image gate on
  `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`
  exited zero. Its unique terminal includes
  `p59_checked_vma_real_shim=4`, `first_update_gate=4`, and `manifests=3`.
- This verifies the installed TP4/TP8 shim carrier and the rendered contract.
  The raw output was not durably saved, so this is an execution-transcript
  receipt rather than a signed raw artifact.
- Not verified because no target was launched: real DP16xTP4/DP8xTP8 first
  optimizer commit, convergence, GCS XProf restoration, and performance.

### 2026-08-26T00:52:00Z — Render-only handoff

- Added `scripts/prepare_checked_vma_three_full_wave.sh`. It requires a clean
  worktree whose HEAD equals the approved SHA, distinct fresh IDs, and an
  absent output root. It renders three manifests and prints—but never
  executes—the apply commands.
- Updated HANDOFF/RUNBOOK/state so the selected wave is GSM8K, P45, and M15
  together; P64 is no longer in the active launch matrix.
- No final manifest was rendered because the current tree is dirty and
  uncommitted. No commit, push, JobSet, TPU workload, or optimizer target
  transaction occurred.

### 2026-08-26T00:59:00Z — Latest-tip rebase admission

- Rebased the four local CLs onto operator tip
  `cb5b4df38410852033291c35083bf15cac6c7652`, then fast-forward rebased once
  more onto evidence-only tip
  `75e97a1db4a4bb328fa174f75869f039defc4b98`. Conflict resolution retained
  the upstream train-step XProf hierarchy, P64 64-TPU first-red evidence, and
  the expanded M15 APC fixed-image suite.
- Post-rebase host gates pass V1 74/74, P57 146/146, P59 37/37, P66 16/16,
  P61 6/6, APC 31/31, and flags 383/383.
- The complete immutable-image gate exited zero with one terminal containing
  `p59_checked_vma_real_shim=4`, `first_update_gate=4`,
  `apc_m15_carrier=46`, and `manifests=3`. The output was not durably saved,
  so this remains an execution-transcript receipt rather than a signed raw
  artifact.
- No target was launched and no final manifest was rendered. Publication and
  exact remote read-back remain pending.
