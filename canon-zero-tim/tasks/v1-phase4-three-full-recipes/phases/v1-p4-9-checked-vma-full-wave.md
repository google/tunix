# V1.P4.9 — Checked-VMA three-full launch wave

Status: active; Attempt 8 launched from published source `c2833eea` and the
strict gate stopped both FrozenLake recipes before backward. P45 reported
`396/0` differing bytes and M15 reported `20/0` at the A-B/B-C boundaries.
A local serving-scope repair is host/image green and passes bounded one-host
TP4 ring and gather strict carriers, but has not restored the failed TP8
executable. No relaunch is authorized by this phase file.

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
PASS / ATTEMPT 8 FROZENLAKE STRICT RED / LOCAL REPAIR TARGET NOT RUN`. Do not claim serial
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

### 2026-08-26T01:04:00Z — Approved source published and read back

- Pushed the four-CL stack by ordinary fast-forward and read back
  `refs/heads/yuxzhang/canon-zero-tim` exactly at
  `ff33ea1a38d1d75c2409ccf480c57e9ff0151075`.
- Freeze that SHA as the approved source for all three final manifests. This
  publication-ledger follow-up is documentation-only and does not change the
  already tested runtime.
- No manifest, JobSet, TPU workload, or target optimizer transaction occurred.

### 2026-08-26T01:58:00Z — Attempt 8 P45/M15 strict reds and local repair

- Pulled signed Attempt 8 evidence through `e43a0fe2`. At source
  `c2833eea5a41438e454ac7e81e599d41fd739d87`, both FrozenLake DP8xTP8
  jobs stopped at the step-0 pre-backward gate with APC disabled. P45 has
  `N_action=44470`, A-B `396` differing bytes, and B-C `0`; M15 has
  `N_action=123381`, A-B `20` differing bytes over `15` elements, max-abs
  `0.007526397705078125`, and B-C `0`. M15's first mismatch is row 199,
  position 5364, logical KV prefix 6544, turn 14. Verified by
  `evidence/v1_hp_three_full_attempt8_20260826/receipt.json`, the two raw error
  logs, the two pre-alignment JSONL files, and the passing `SHA256SUMS` ledger.
- The shared signature is A decode versus B full prefill only; B full prefill
  remains byte-identical to T-old. Both runs are APC-off and fail before any
  backward or optimizer work. This classifies one P66 forward-scope
  regression, not an M15-only APC or backward defect. GSM8K's TP4 pass does
  not certify the distinct TP8 long-context serving executable.
- Located the scope leak in `src/engine_shims/linear_p22xf.py`: the P66
  completed-sum `pmean` was controlled by a process-wide flag and therefore
  entered ordinary serving o_proj/down_proj graphs. The unpublished repair
  additionally requires the live P59 manual data/model context, so the
  checked-VMA annotation remains in trainer-local backward and ordinary
  serving returns the historical fixed-order sum unchanged.
- Verified by V1 74/74, P59 37/37, P66 16/16, syntax, manifest, and diff
  hygiene. The immutable-image installed-shim TP4/TP8 gate passes and its new
  real contract-parallel serving negative requires flag-off/on byte identity
  while making any serving `pmean` fatal. Not verified because no repaired
  TPU carrier has run: real DP8xTP8 P45/M15 `0/0`, optimizer correctness,
  convergence, and performance. The user subsequently authorized this repair
  publication batch; runtime CL `41f50d23` is local and remote readback is
  pending.

### 2026-08-26T02:19:00Z — Ring/gather one-host scope carriers

- Expanded the installed-shim serving negative to execute both the ring
  fixed-tree and production `CANON_FIXED_AR_GATHER=1` branches with the P66
  process flag off/on. TP4 and TP8 both preserve the historical output
  exactly, and any ordinary-serving `pmean` is fatal. The immutable image
  `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`
  passes with `ordinary_contract_p66_global_negative=2` for each topology and
  terminal `manifests=2x37/37`. The raw image stream was not durably saved;
  this is a reproducible admission receipt, not a signed raw artifact.
- One-host Qwen3-8B DP1xTP4 ring carrier
  `p66scopefix_20260826t0204z` completed three strict rounds with actions
  `409,565,897`; both boundaries are `0/0` bytes in every round, backward and
  optimizer commits are zero. Raw log SHA256 is
  `9022cad0bcfa81595bd99f0847da1423012626b6157b4919da8218d66fcf3d04`;
  pre-alignment SHA256 is
  `6ec6cf96f337939f12805506b525de0bb34f6d4504cbc79944a491ca03953903`.
- Fresh gather carrier `p66scopegather_20260826t0212z` executed 216
  `gather-ordered-sum` PATHTRACEs and completed the same three strict action
  counts with both boundaries `0/0`, zero backward, and zero commits. Raw log
  SHA256 is
  `601d0ffc3d6c0aabd517765f0c1352e92bdb75f59be1a3801a5ff713ffae9839`;
  pre-alignment SHA256 is
  `955ae87b9892ec3ac12a8245da3d970242242cf185ba628fb96a9b994d624306`.
- Verified by real one-host v5p TP4 execution. Not verified because this host
  exposes only four JAX TPU devices: the failed DP8xTP8 P45/M15 executable,
  long prefixes beyond this carrier's envelope, backward, optimizer, and
  convergence. Claim ceiling is `TP4 MECHANISM PASS / TP8 TARGET NOT RUN`.
- At measurement time no commit, push, optimizer update, or full relaunch had
  occurred. Runtime CL `41f50d23` was then created under explicit user
  approval; publication readback and both requested FrozenLake full launches
  remain pending.
