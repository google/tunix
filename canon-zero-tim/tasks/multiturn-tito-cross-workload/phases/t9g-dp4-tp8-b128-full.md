# T9g — explicit 32-chip / 128-trajectory FrozenLake full option

## Authority and scope (2026-09-07)

At implementation start the user approved local changes only. The later
2026-09-07 `commit push` instruction explicitly approves T9g source and
ledger publication to `yuxzhang/canon-zero-tim`, not target render/launch,
image publication or remote storage. Source: clean published `c07ea8fa10cbfc55957503b9831ed863f8c36814`,
branch `local/p57-tito-pair-0902`. Reuse the existing P45/M15 full renderer.
The old 64-chip option and user-owned scheduling settings remain unchanged.

## Preregistered identity and shape ledger

| Quantity | Existing default | New explicit option |
|---|---|---|
| selector | `dp8-tp8-b256` | `dp4-tp8-b128` |
| v5p topology / devices | `4x4x4` / 64 | `2x4x4` / 32 |
| mesh | DP8 x TP8 | DP4 x TP8 |
| prompts / generations / trajectories | 32 / 8 / 256 | 16 / 8 / 128 |
| per-rank prompts / trajectories | 4 / 32 | 4 / 32 |
| caller-global M | 2048 | 1024 |
| local / kernel M | 256 / 256 | 256 / 256 |
| trainer microbatch / gradient groups | 8 / 32 | 4 / 32 |
| per-rank serving sequence / token capacity | 32 / 256 | 32 / 256 |
| max concurrency | 256 | 128 |
| policy updates / total trajectories | 300 / 76800 | 300 / 38400 |

P45 retains five turns and 4096+2048 tokens; M15/main retains fifteen turns
and 4096+8192 tokens. Dataset hashes, seeds, RLOO group size eight, optimizer,
learning rate, clipping, sampler, token treatment, record-full policy,
independent B rescore and numerical/backward gates remain unchanged. APC,
evaluation and ordinary checkpoints remain off. This is a new statistical
recipe, not a same-trajectory speed-only change. A lower chip count does not
promise a faster step or repair the cause of the prior model timeout.

## Implementation phases and gates

1. Register a default-absent closed geometry selector and shared geometry
   contract. Derive profile, renderer, worker count and command values from
   the same tuple. Gate: old/new positive cases and invalid/mixed identities.
2. Wire the real shell and Python admission chain, fixed-head global M,
   first-update denominator, TiTO rows and full classifiers. Preserve the
   complete optimized backward path. Gate: renderer -> resolved environment
   -> runtime/classifier, with wrong batch/DP/M/profile/chunks negatives.
3. Run focused and complete host gates and a pinned-image construction gate;
   update HANDOFF with commands and output obligations. Target certification
   remains NOT RUN; no 64-chip evidence transfers to the new geometry.

## Rollback

Omit the new option to keep the historical 64-chip render. Review/revert only
this phase's explicit diff after checking ownership; do not reset the tree,
rewrite the base scheduling YAML or delete evidence. Source-only rollback is
`git revert ae2e4884d7df20cdf7cda6dd4a910ea61280e443` after separate approval
and an overlap review; preserve the evidence/handoff follow-up as history.

## Result log

- Preregistration: verified clean source and published readback at the SHA
  above before edits. Only local implementation was authorized.
- Implemented the closed selector and shape ledger in
  `examples/frozenlake/training_geometry.py:7` and
  `canon-zero-tim/cluster/steps/p57_training_geometry.sh:4`.
  Verified by real renderer -> raw environment -> `00_env.sh` -> resolved
  environment -> real Python admission, not by rewriting a DP4 identity to
  pretend it is DP8. Malformed/empty/zero/foreign selector, mixed profile,
  wrong raw batch/DP/TP/M, duplicate or wrong CLI values all reject.
- `canon-zero-tim/cluster/profiles/qwen3-8b-dp4-tp8-frozenlake-v1-hp.env:1`
  reuses the existing full bundle. `tunix/rl/dp_workloads.py:596` registers
  the exact separate workload. P59, checked-VMA, P67 scope, first-update gate,
  fixed reducer, clip mathematics, RLOO group size and token-record policy
  are not replaced. `src/engine_shims/p38_fixed_lm_head.py` only adds the
  caller-global M1024 admission for 8B/TP8; its updated manifest hash is
  `60f7173fc44236e6ae1d9400080ffde00f769ccf86484ad77283cf07d7e0b564`.
- Verified new full postflight by
  `tests/v1_phase4/test_full_classifier.py:311` and head receipt positives
  plus wrong global/local M, chunks, reduction marker, profile and geometry
  negatives. `tasks/multiturn-tito-cross-workload/scripts/classify_tito_full_record.py:588`
  now expects DP4 and 128 rows for the selected option, including single
  writer, Orbax probe, sidecar and actor-snapshot metadata; wrong DP8 evidence
  is rejected. The journal/capsule stream is not capped or filtered.
  Additional DP4 controls execute the real Orbax-probe and actor-snapshot
  consumers with fake managers, check DP4 metadata and wrong-restore/DP8
  negatives, and retain pre-update actor-only semantics. These are not
  real-storage probes.
- Verified scheduling preservation by
  `tests/v1_phase4/test_p67_frozenlake_two_full_renderer.py:85`: both legacy
  and Bodaborg targets, each with legacy or record-full transport. Worker pod
  content is identical after the topology selector/annotation substitution;
  count becomes eight. The base YAML has zero diff; no new nodepool pin.
- Host PASS: P57 248/248, V1 104/104, APC 12/12, flags 423/423; all ten changed
  shell/profile files pass individual `bash -n`, changed Python files compile,
  and source/doc `git diff --check` passes. Fixed-image focused runtime PASS:
  eight old/new x P45/M15 x legacy/record-full cases plus a real
  `FixedDPRankGradientReducer` staged-table check on 32 forced CPU devices
  (DP4xTP8); exact expected sum, finite values and replica agreement.
- Failed construction checks retained: the new head receipt negative initially
  omitted a required test argument; a new wrapper fixture used a trailing
  hyphen in its synthetic state name (correctly refused by existing
  JobSet-identity checks). Corrected the fixtures without relaxing admission.
  Full pinned-image attempt 1 then caught a stale exact-image assertion of
  the old 8B/TP8 learner-M list; updated only that list and added installed
  M1024 acceptance plus neighboring-model/topology rejection. The successful
  full rerun is recorded below. Two manual source-directory manifest invocations used the
  installed-output manifest against an uninstalled tree and reported missing
  generated files; installed manifests must be checked by `install.sh`.
- Unverified because no target launch is authorized: DP4xTP8 actual forward
  alignment, full-model backward/optimizer, TPU memory, complete 300-update
  training, throughput/JIT/XProf, real GCS/Orbax, and matched observer
  neutrality. Existing DP8 evidence does not certify these. No claim that
  reduced concurrency fixes the cause of generation timeout.

## Performance decision ledger

| Candidate | Correctness construction | Target wall / PERF / XProf | Decision and limit |
|---|---|---|---|
| DP4xTP8/B128 full | Host, focused pinned CPU and complete image PASS | NOT RUN / no timings | implemented, not performance-certified; local per-rank work unchanged, data budget halves |

There is no KEEP/speedup claim from construction tests. Report sampling,
prefill/decode/B-rescore, trainer/reverse/optimizer, compile and evidence-I/O
costs separately after a separately approved target; use the same tokens and
profiling window for causal performance comparisons.

### Final local admission — 2026-09-07

- Complete pinned CPU image rerun exits 0. Raw
  `evidence/t9g-host-20260907/exact-image-r2.log:532` admits 8B/TP8 M1024;
  :728 records installed TP4/TP8 shim/manifest PASS; :738 records the new
  eight-case runtime plus DP4xTP8 reducer PASS; :1458 is the terminal
  `V1_HP_EXACT_IMAGE_PASS`. Full raw log SHA256:
  `091707da9a2fd7476ff08eefe6322e04cc511310b150901eebd2841dba5277a3`.
- Final P57 248/248, V1 104/104, APC 12/12 and flags 423/423 PASS;
  fake-manager DP4 actor/probe checks and wrong-geometry evidence negatives
  are included. All 41 audited code/gate blobs are unchanged after testing.
  The 14 raw logs and receipt verify 15/15 against the local SHA manifest.
  Failed attempts are retained, not normalized or deleted.
- Receipt SHA256: `5df76e984f354109e50cc9ff5be9b1e584ab09017c4ce39baa0716d676fbf8fb`.
  Final disposition: LOCAL IMPLEMENTATION / HOST + PINNED CPU CONSTRUCTION
  PASS / TARGET NOT RUN. No commit/push, launch or remote write. Await the
  user's fresh release approval; the baseline commit does not contain T9g.
- Final whitespace audit: tracked source/docs and all six new source/phase
  files have no whitespace diagnostics. The immutable terminal raw logs
  contain trailing whitespace at `exact-image.log` lines 342, 363; `exact-image-r2.log` lines 342, 363.
  Preserve those original bytes and hashes; record this raw-artifact-only
  formatting exception during commit review rather than rewriting evidence.

### Authorized source freeze — 2026-09-07

- User explicitly approved T9g commit/push. Source CL `ae2e4884d7df20cdf7cda6dd4a910ea61280e443`
  has tree `6d364a5e1deadba87abf8281cac33e55d7208b4e`; the evidence/handoff
  follows separately. All 41 code/gate blobs match the pinned-image-tested
  candidate; the only additional core file is FLAGS registration.
- Publication reruns: P57 248/248, V1 104/104, APC 12/12 and flags 423/423;
  syntax, source staged whitespace and secret-pattern checks PASS.
  Verified by `evidence/t9g-release-20260907/receipt.json` and raw logs.
  The earlier local-admission no-commit statement describes that historical
  gate, not the current authorization.
- Frozen baseline still equals fetched remote `c07ea8fa`. Delivery requires
  a final re-fetch and full-SHA readback. Target remains unverified because
  no TPU/Kubernetes run is authorized; do not promote host/image claims.
