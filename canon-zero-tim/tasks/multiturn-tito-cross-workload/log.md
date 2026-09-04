# Log

## 2026-08-30T09:34:49Z — T0: cross-workload TiTO investigation opened

- Type: decision
- Fact: current source is `0b90ff75ef7581c4230c0253df67779d06066792` on `local/m15-apc-attempt17-review-0829`, with a clean preflight at task start.
- Fact: DeepSWE has real DP1xTP4 exact-token continuity evidence; M15 has construction evidence only and remains non-TiTO in production.
- Action: bound a new task directory and preregistered the offline oracle, isolated DP1xTP8 carrier, legacy observer, exact replay, DeepSWE regression, performance, and target-decision phases.
- Command: omitted; task initialization only.
- Result: active. The v5p lane was occupied by `m15_e0v_e0w4_on`, so no container or TPU mutation was performed.
- Files/artifacts: `state.md`; `plan.md`; `phases/t0-real-tokenizer-oracle.md`
- Rollback: remove only this uncommitted task scaffold if the user cancels before evidence exists; never remove later run evidence.
- Next: implement the T0 real-tokenizer oracle and negative controls.

## 2026-08-30T09:49:00Z — T0: real-tokenizer oracle complete

- Type: experiment
- Fact: the existing `m15_e0v_tito_e0w4_pair` is not a TiTO on/off pair. Both arms force exact TiTO and differ only at APC; its completed off arm is a useful DP1xTP4 exact-transport/strict-alignment control but cannot classify retokenization.
- Action: added a read-only tokenizer oracle that compares full-chat re-render/re-tokenization with the exact initial-prompt plus sampled/environment token ledger. Added value, length, malformed-vector, and one-token poison controls.
- Commands: focused unittest; real Qwen3-4B DeepSWE trajectory audit; Qwen3-8B thinking-off FrozenLake user-turn fixture audit.
- Result: tests 3/3 PASS. DeepSWE is equal at turn 1 and different at turns 2-11, with first mismatch token 2242. The FrozenLake fixture is equal at both later turns. Both poison negatives trip at the injected coordinate.
- Files/artifacts: `scripts/audit_tito_transcript.py`; `../../tests/multiturn_tito/test_audit_tito_transcript.py`; `evidence/t0-tokenizer-audit-summary.json`.
- Limitation: FrozenLake used a realistic synthetic transcript because no persisted M15 trajectory with message/token boundaries is available. It does not establish that live M15 is equal.
- Next: run one real M15 APC-off `verify` observer on DP1xTP4 after the current TPU carrier exits; exact override and backward/optimizer remain disabled for that arm.

## 2026-08-30T09:47:16Z — T1 r1: carrier rejected before TPU program

- Type: experiment / expected immutable failure evidence
- Label: `tito-vfy-r1-0830`
- Action: launched the registered APC-off, three-round DP1xTP4 verify carrier after the lane became idle.
- Result: `INCONCLUSIVE`, Docker exit 1 after 20 seconds. JAX enumerated four TPU devices, but `train_frozenlake_qwen3.py` rejected the materialized M15 workload because its pre-existing profile allowlist did not yet recognize the new one-host observer identity. No rollout, alignment, backward, or optimizer program ran.
- Evidence: `/mnt/disks/tunix-data/logp_probe_1host/m15_tito_verify_tito-vfy-r1-0830`; its three-file failure manifest verifies.
- Repair: the workload admission now calls the same fail-closed token-continuity selector and admits only its exact `verify` one-host result. It does not add a profile name, admit exact mode, or change production behavior.
- Verification: AST parse, 12/12 focused tests, flag audit 409/409, shell syntax, and diff check PASS.
- Next: rerun with a fresh label; never reuse or delete r1.

## 2026-08-30T09:50:00Z — External exact control reconciled

- Type: evidence reconciliation
- Fact: the independently running E0w4 pair completed after T0. Both arms used exact TiTO; the only experimental variable was APC off/on.
- Result: each arm completed three rounds and 17 exact-equal token receipts. All six A-B and B-C byte counts are zero; backward and optimizer commits are zero. The pair manifest verifies.
- Boundary: source commit/diff differ from this task, topology is DP1xTP4, and exact input override makes it incapable of classifying the legacy rendered-text seam. It is an analysis-grade exact control only.
- Evidence: `evidence/external-e0w4-exact-control.json`; external root `/mnt/disks/tunix-data/logp_probe_1host/m15_e0v_tito_e0w4_pair`.

## 2026-08-30T09:51:30Z — DeepSWE first-red boundary localized

- Type: analysis
- Fact: the persistent token-2242 red lies 171 tokens inside the second assistant sample, before the next environment message.
- Mechanism: sampled transport represented the model-emitted `command=view` fragment with two tokens; re-encoding the same visible text merged it into one token. All later full-chat prompts inherit that one-token displacement.
- Interpretation: DeepSWE TiTO prevents a genuine tokenizer non-involution in assistant text. The initial user/tool role boundary is not the first cause in this trajectory.
- Boundary: this mechanism does not prove M15 encounters such a fragment; the live M15 verify arm decides that separately.

## 2026-08-30T09:52:00Z — T1 r2/r3: lane refusal and root-filesystem prelaunch failure

- Type: carrier incidents
- r2: label `tito-vfy-r2-0830` was rejected before output creation because unrelated container `p4c_adjacent_baseline_20260830_r1` acquired the pinned-image lane. It was not interrupted and r2 is not reused.
- r3: label `tito-vfy-r3-0830` created a source-bound directory, then `install.sh` failed while extracting the image with `no space left on device`. No TPU container launched. Root evidence remains at `/mnt/disks/tunix-data/logp_probe_1host/m15_tito_verify_tito-vfy-r3-0830`.
- Audit: `/tmp` shares a 97 GiB root filesystem with only 488 MiB free. The failed install's own `mktemp` directory was already removed; no other task's temporary files were deleted.
- Repair: bind `TMPDIR` for canonical extraction to the fresh evidence root on `/mnt/disks/tunix-data`. This changes only carrier scratch placement, not installed bytes or runtime flags.
- Next: syntax/focused gates, then fresh r4 after lane check.

## 2026-08-30T10:01:00Z — T1 r4: numerical result green, required B receipt absent

- Type: experiment / incomplete admission evidence
- Label: `tito-vfy-r4-0830`
- Action: ran the registered APC-off, three-round DP1xTP4 legacy `verify`
  carrier in the pinned image at source `0b90ff75ef7581c4230c0253df67779d06066792`
  plus the recorded dirty diff.
- Numerical result: all 17 later-turn continuity receipts are
  `TOKEN_STREAM_EQUAL`; the observed prompt-token lengths and hashes also
  match the independent exact-TiTO control. The three strict rounds cover
  515, 766, and 748 action tokens; every A-B and B-C byte count is zero and
  all three alignment verdicts are PASS. Backward and optimizer commit counts
  are zero.
- Admission result: `INCOMPLETE`, not T1 PASS. The raw log contains three vLLM
  prefix-cache reset messages and source provenance establishes zero cached
  tokens, but the preregistered explicit
  `[CANON_APC_M15_B_CONTRACT] reset_prefix_cache=True all_num_cached_tokens_zero=True`
  receipt was absent. The classifier correctly failed only
  `B_full_reset.count`; the gate was not weakened.
- Evidence: `/mnt/disks/tunix-data/logp_probe_1host/m15_tito_verify_tito-vfy-r4-0830`;
  elapsed 491 seconds, controlled exit 42; immutable raw evidence retained.
- Repair: extend the existing fail-closed B-contract observation to the exact
  one-host M15 `verify` identity. It checks both full reset and all-zero cached
  token counts before emitting the receipt; production profiles and numerical
  programs are unchanged. Correct-worktree rollout tests pass 31/31, task
  tests 6/6, P57 182/182, flag audit 409/409, shell syntax and diff hygiene
  pass.
- Next: fresh r5 only after the unrelated active v5p container exits; never
  reuse or delete r1-r4.

## 2026-08-30T10:04:00Z — T1 r5/r6: prelaunch lane refusals

- Type: carrier scheduling incidents
- Labels: `tito-vfy-r5-0830`, `tito-vfy-r6-0830`
- Result: both invocations exited 4 at the runner's pinned-image busy-lane
  preflight. An external sequence launched successive short-lived containers
  between each idle observation and our launch attempt. No task container,
  TPU program, output directory, or numerical evidence was created.
- Discipline: neither label will be reused; the external containers were not
  interrupted. The next attempt requires a fresh label after the sequence is
  stably idle.

## 2026-08-30T10:18:00Z — T2 r7: formal live M15 observer PASS

- Type: target experiment / decision
- Label: `tito-vfy-r7-0830`
- Action: after the unrelated pinned-image sequence released the lane, ran the
  repaired one-host observer without a launch pipeline. The pinned image is
  `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`;
  source is `0b90ff75ef7581c4230c0253df67779d06066792` plus diff SHA
  `01f7b7338cb62113df7cafa5fbc61fd0d6061bfa2b3e001e441665b72e4b41af`.
- Result: `LEGACY_TOKEN_EQUAL`. All 17 live later-turn receipts are equal and
  none differ. Three rounds contain 515, 766, and 748 action tokens; A-B and
  B-C are zero bytes in every round. The required B-full-reset/all-cached-zero
  marker appears exactly three times. Controlled exit is 42 after 489 seconds;
  backward and optimizer commit counts are zero.
- Evidence: external immutable root
  `/mnt/disks/tunix-data/logp_probe_1host/m15_tito_verify_tito-vfy-r7-0830`;
  all seven manifest entries verify. Durable in-tree summary:
  `evidence/m15-onehost-r7-summary.json`.
- Decision: the preregistered equal branch fires. Production M15 stays
  non-TiTO; exact replay and paired TiTO performance work are not warranted by
  this evidence. DeepSWE remains TiTO because its real trajectory independently
  has 10/11 later-turn drift.
- Boundary: this is DP1xTP4 transport/alignment evidence only. It neither
  certifies DP8xTP8 nor authorizes an M15 production selector.

## 2026-08-30T10:20:00Z — adjacency and host closeout

- Host gates: task tests 6/6, rollout canonical 31/31, P57 182/182, and flag
  registry 409/409 pass; shell syntax and diff hygiene pass.
- DeepSWE adjacency: rollout canonical includes the P58 Native/Zero isolation
  cases and passes. Two focused DeepSWE continuation tests executed and showed
  two pass dots under CPU-only pytest, but that legacy test process retained a
  background thread and did not cleanly exit, so it is not registered as a
  suite PASS. The first full-file attempt was likewise stopped after host TPU
  metadata polling made no progress. No token-continuity helper or DeepSWE
  branch changed in this task; the only shared-rollout change is guarded by
  the exact M15 one-host verify identity.
- Delivery: no commit, push, image publication, Kubernetes mutation, or
  production profile change.

## 2026-08-30T10:24:00Z — T3 reopened by user requirement

- Type: decision / preregistration
- User requirement: prove that M15 exact TiTO itself has no problem; legacy r7
  equality is not sufficient.
- Action: reopened T3/T6 and registered a matched APC-off DP1xTP4 exact arm.
  It must prove every serving-consumed later prompt equals the integer token
  ledger, three strict A/B/C rounds, three independent B-reset receipts, and
  zero backward/commit. A cross-arm comparator must also match ordered prompt
  SHA/length and per-round trajectory hashes to legacy r7.
- Boundary: this does not silently enable the production M15 flag and cannot
  be promoted to DP8xTP8 certification.

## 2026-08-30T19:35:19Z — T3 r8: exact TiTO and cross-arm gates PASS

- Type: one-host target experiment / decision.
- Label: `tito-exact-r8-0830`.
- Action: ran the matched M15/main APC-off DP1xTP4 exact-token carrier in the
  pinned image at source `0b90ff75ef7581c4230c0253df67779d06066792`
  plus diff SHA
  `e788a1c8571ef335b8851c45d310b96a855d76249d42bf7fc2ddb38450a75a64`.
- Result: `EXACT_TOKEN_CONTINUITY_ALIGNMENT_PASS`. All 17 later-turn exact
  receipts are equal. Three independent B-full-reset receipts are present.
  Rounds contain 515, 766, and 748 action tokens; every A-B and B-C byte count
  is zero. Backward and optimizer commits are zero. Controlled exit is 42
  after 499 seconds.
- Cross-arm result: r7 verify versus r8 exact is `MATCH` for all 17 ordered
  prompt receipt lengths/hashes and all three round token/action-mask hashes.
  Legacy r7 took 489 seconds and exact r8 took 499 seconds; the 10-second
  difference is one bounded total-wall observation, not a component-level
  performance result.
- Evidence: immutable external root
  `/mnt/disks/tunix-data/logp_probe_1host/m15_tito_exact_tito-exact-r8-0830`;
  its seven-entry manifest verifies. Raw SHA is
  `c3f026734255dc06a4cfca2d82f0769e62ffdafeca2926b68331c45c93d4dd64`.
  Durable summaries are `evidence/m15-onehost-r8-exact-summary.json` and
  `evidence/m15-onehost-r7-r8-cross-arm.json`.
- Decision: M15 exact TiTO is healthy at one-host scope. Production remains
  selector-absent because DP8xTP8 was not run; no commit, push, image publish,
  Kubernetes launch, or production-default change was performed.

## 2026-08-30T19:48:18Z — exact M15 retained as a default-off full option

- User decision: expose the already one-host-green exact carrier as an
  explicit M15 full renderer option while keeping the default selector absent.
- Scope: only exact M15/main Zero v1-hp DP8xTP8 300-update full may select it;
  P45 and every neighboring workload remain fatal negatives.
- Claim: the first selected DP8xTP8 exact run is still a target certification.
  This registration does not upgrade the r8 one-host result and does not
  authorize an M15 launch in the current action.

## 2026-09-02T09:07:57Z — T7: paired P45/M15 exact treatment opened

- Type: experiment / decision.
- Fact: a clean worktree at latest fetched source
  `6842edae88b5692c7d4c6ae4ecadfc9e2bf1e411` passes package preflight.
- Action: ran the existing P67 two-full renderer suite before editing.
- Command: `python3 -m unittest canon-zero-tim/tests/v1_phase4/test_p67_frozenlake_two_full_renderer.py`
- Result: 1/6 passed and 5/6 errored before manifest validation. Every error is
  the same duplicate-write guard over the seven optimization keys now emitted
  by the base P57 renderer. This is a latest-tip integration regression, not a
  TiTO numerical result.
- Files/artifacts: `phases/t7-p45-m15-exact-treatment.md`.
- Rollback: keep the base renderer as the single owner; the P67 wrapper change
  can be reverted independently from generic token continuity.
- Next: remove only the duplicate outer write and prove the legacy renderer
  tests green before expanding the selector.

## 2026-09-02T09:17:03Z — T7/T8: renderer integration and generic contract host-green

- Type: code change / experiment.
- Fact: P45 and M15 share the trajectory token ledger; their full recipes
  differ in turn horizon and materialized dataset identity, not in the exact
  reconstruction algorithm.
- Action: made the base P57 renderer the sole FrozenLake performance-bundle
  writer; added the closed `legacy|p45-exact|m15-exact|both-exact` renderer
  selector; added generic full-only environment/Python admission, workload-
  labelled exact receipts, classifier completeness, and negative controls.
  Historical M15 debug/one-host evidence keeps its old selector and marker.
- Commands: `bash canon-zero-tim/tests/p57_frozenlake_tim/run_cpu.sh`;
  `bash canon-zero-tim/tests/v1_phase4/run_cpu.sh`; focused TiTO/renderer/
  classifier unittest; flag registry audit; syntax and diff checks.
- Result: P57 189/189, V1 97/97, focused 50/50, flags 410/410, and all syntax/
  diff checks pass. Four renderer modes resolve through real `00_env.sh`;
  default legacy leaves both selectors absent. No target runtime ran.
- Files/artifacts: `phases/t7-p45-m15-exact-treatment.md`; source/tests and
  Phase4 handoff/runbook in this worktree.
- Rollback: revert generic selector/runtime/classifier edits while preserving
  the independent duplicate-injection repair; or revert both concerns before
  publication. No committed SHA exists yet.
- Next: run the complete pinned-image gate and stop on its first red.

## 2026-09-02T09:25:38Z — T9: complete host admission PASS

- Type: experiment / review.
- Action: added raw-input drift detection so a conflicting caller-supplied
  topology or stage cannot be overwritten and laundered by the profile;
  omitted fields remain correctly derived by that locked profile. Added a
  whole-JobSet structural A/B test for `legacy` versus `both-exact` and kept
  the historical classifier report separate from the new generic receipts.
- Result: P57 189/189, V1 99/99, flags 410/410, shell/Python syntax, and diff
  hygiene pass. The structural comparison removes the one generic selector
  from each treatment manifest and obtains the complete legacy JobSet exactly.
  All registered negative selector/profile/topology/receipt controls are red.
- Blocker: the complete pinned-image gate has not started because active
  container `v1_gsm8k_xprof_zero_hp_dp4tp1-fwddedup_s2b_20260902_r1` is still
  consuming the shared host for an unrelated XProf capture. It was observed
  read-only and not interrupted. No TPU/Kubernetes target ran.

## 2026-09-02 — T9a first-diff diagnostics preregistered

- Type: user decision / preregistration.
- Requirement: make a mismatching multi-turn trajectory and its token ledger
  directly debuggable through an explicit flag.
- Decision: default-off `first-diff` diagnostics will emit complete integer
  token evidence in bounded JSON chunks, not free-form conversation text. It
  is scoped to selected generic exact P45/M15 full arms and leaves the existing
  immediate mismatch fatal unchanged.
- Files/artifacts: `phases/t9a-first-diff-diagnostics.md`.

## 2026-09-02T09:55:36Z — T9a host admission PASS

- Type: code change / experiment.
- Action: added the default-off generic P45/M15 `first-diff` selector, a
  one-shot runtime dump, atomic JSON capsule persistence, raw-log extractor,
  renderer/classifier contracts, and positive/negative tests. Each record is
  capsule-ID tagged so mixed worker output can be selected deterministically.
- Result: focused diagnostics 13/13, P57 191/191, V1 101/101, flags 411/411,
  and shell/Python syntax plus diff hygiene pass. Local and raw-log capsule
  round trips are exact; corrupt, incomplete, equal-stream, unscoped, and
  successful-run-with-dump controls fail.
- Boundary: the capsule reproduces the token-continuity input pair without a
  new stochastic rollout. It does not by itself freeze model weights, KV
  scheduling, rewards, backward, or optimizer state. Pinned image and
  DP8xTP8 target remain unverified.
- Blocker: an unrelated three-update XProf capture still owns the shared host
  lane. It was observed read-only and not interrupted.

## 2026-09-02T10:16:52Z — T9/T9a initial pinned-image PASS, later superseded

- Type: immutable-image construction gate / evidence closeout.
- Action: after the unrelated XProf container exited naturally, ran the
  complete V1 exact-image gate from the beginning against local image
  `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`.
- First result: one documentation-route test rejected the new top HANDOFF
  section because its heading no longer contained the historical `P74`
  routing token. No runtime or numerical test was red. Restored `P74` in that
  heading, reran its focused 4/4 control, then reran the complete image gate
  from the beginning.
- Final result: exit 0 with terminal `V1_HP_EXACT_IMAGE_PASS`; the terminal
  includes `frozenlake_tito_impl=2`, `frozenlake_tito_selector=closed`,
  `frozenlake_tito_debug=1`, `frozenlake_tito_default=legacy`, and
  `manifests=3`. The image directly exercised generic exact P45 and M15,
  first-diff capsule persistence/extraction, interleaved-log selection,
  learner routing, and the inherited P59/APC regression gates.
- Claim boundary: host plus pinned image certifies construction only. No TPU,
  Kubernetes, optimizer, or DP8xTP8 target program ran. Commit, push, render,
  and launch remain separately approval-gated.
- Supersession: subsequent review hardened capsule metadata validation and
  full-run per-trajectory summary completeness. This earlier green therefore
  remains historical evidence but is not the final admission artifact.

## 2026-09-02T10:35:03Z — post-review pinned-image RED preserved

- Type: immutable-image construction gate / failed evidence.
- Action: reran the complete V1 exact-image gate after metadata, extractor
  permission, one-shot, and per-trajectory summary hardening.
- Result: RED. The new `trajectory_id` field had been inserted between
  `workload=...` and the historical `mode=exact` text, so existing consumers
  correctly rejected the changed stable receipt prefix. This was a receipt
  compatibility regression, not a token equality or numerical failure.
- Repair: restored `[CANON_P57_TOKEN_CONTINUITY] workload=... mode=exact` as
  the stable prefix and retained `trajectory_id` later in the same receipt.
  Focused host controls passed before rerunning the full gate from the start.
- Raw local log: `/tmp/p57_tito_pair_pinned_20260902T101652Z.log`, SHA256
  `76eb406a92bd295c989da03b7b438260f33ef39772768c2309147a2a5b3bf2cc`.
  It is preserved locally and has not been copied to durable GCS.

## 2026-09-02T10:42:52Z — T9/T9a hardened complete pinned-image PASS

- Type: immutable-image construction gate / final local admission evidence.
- Action: reran the complete V1 exact-image gate from the beginning after the
  stable-prefix repair against local image
  `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`.
- Result: exit 0 with terminal `V1_HP_EXACT_IMAGE_PASS`. The terminal includes
  `frozenlake_tito_impl=2`, `frozenlake_tito_selector=closed`,
  `frozenlake_tito_summary=1`, `frozenlake_tito_debug=1`,
  `frozenlake_tito_capsule_integrity=1`,
  `frozenlake_tito_default=legacy`, and `manifests=3`.
- Raw local log: `/tmp/p57_tito_pair_pinned_20260902_r2.log`, SHA256
  `9bc28afb41ac0a7049eb66a2c65aa47abb912bdcf42ed4620603c961700446a3`.
  This is a local admission artifact; raw output has not been copied to
  durable GCS. The checked-in evidence receipt records its identity without
  embedding the sensitive or voluminous raw log.
- Claim boundary: construction PASS only. P45/M15 DP8xTP8 targets, optimizer
  commits, and the shared-runtime DeepSWE DP1xTP4 adjacency carrier did not
  run. No commit, push, manifest render, Kubernetes, or TPU launch occurred.

## 2026-09-02T10:45:01Z — final dirty-tree host audit PASS

- Type: release-candidate local audit.
- Result: P57 191/191, V1 101/101, flag audit 411/411 plus its 2/2 test,
  Python compilation, shell syntax, evidence JSON parsing, and
  `git diff --check` all pass after evidence closeout.
- Scope audit: 22 tracked files are modified and four files are new. No YAML
  file is modified; autoscaling, exclusive topology, node selectors, and
  launch configuration remain outside this diff. The tree is intentionally
  dirty and uncommitted pending a separate user approval.

## 2026-09-02T19:21:02Z — T9b multi-diff and engine-witness phase opened

- Type: user decision / preregistration.
- Requirement: complete TiTO validation and data extraction so one diagnostic
  run can preserve multiple independent token-diff trajectories and return
  them durably.
- Confirmed source facts: current equal/different receipts compare the integer
  ledger against `SamplerOutput.padded_prompt_tokens`, which is rebuilt from
  the submit-side IDs after generation and is not a TPU-runner echo.
  `RequestOutput.prompt_token_ids` is available but unread; it is an engine
  API echo, not by itself proof of runner consumption. P38's request journal
  reads the deeper `runner.input_batch.token_ids_cpu`, but its existing
  serving-capture identity and output volume are unsuitable for production.
  The collect engine already masks selected timeout statuses, contradicting a
  claim that status has no consumer, but using that path in ordinary GRPO
  would alter effective rows and can create all-zero metric edge cases.
- Decision: production exact remains first-diff fatal. New `collect-64` is
  admitted only on a dedicated rollout-only/no-backward/no-commit diagnostic;
  a mismatch ends that trajectory after evidence capture and then collection
  continues with another trajectory. No bad trajectory reaches loss,
  advantage, backward, or optimizer code.
- Decision: certify three distinct layers—submitted ledger, RequestOutput
  echo, and request-ID-matched runner input length/SHA. Persist capsules and
  journal shards with atomic local writes, no-clobber protected-GCS uploads,
  readback hashes, and a final manifest. Worker-log chunks remain the recovery
  fallback, not the only durable channel.
- Scope: reuse this task directory. Keep every concern hunk-separable in the
  current uncommitted worktree; no commit, push, manifest render, Kubernetes,
  TPU, or remote mutation is authorized by this decision.
- Files/artifacts: `phases/t9b-engine-witness-and-multidiff-collection.md`.

## 2026-09-02T20:05:00Z — T9b-0 construction oracle PASS with bounded claim

- Type: code change / host gate.
- Action: introduced one typed, ordered continuation-prompt segment builder
  and routed both exact reconstruction and first-diff capsule metadata through
  it. Added an independent B/C helper matching the trainer rescore shape
  `unpadded_prompt + conversation[:completed_offset]`.
- Result: 15/15 focused tests pass. A three-turn FrozenLake fixture proves
  equality at every later turn; the first prompt uses the sampler-submitted
  padded IDs plus explicit `prompt_length`, a one-token conversation poison is
  detected at the injected position, and a missing nonterminal environment
  segment fails closed. Python compilation and diff hygiene pass.
- Evidence boundary: the preserved r7 run says trajectory logging was
  disabled and contains receipt hashes but no raw trajectory token arrays.
  Therefore this is a production-shape construction proof, not a claimed r7
  token replay. The first approved data-collection target must instantiate the
  same oracle on a real captured trajectory.
- Next: T9b-1 request-ID-joined submit/engine-echo/runner-input witness.

## 2026-09-02T20:12:57Z — T9b-1 through T9b-3 host construction PASS

- Type: code change / host and installed-overlay gates.
- Engine witness: preserved submitted and `RequestOutput.prompt_token_ids`
  length/SHA evidence and added an explicit submit-future/result request-ID
  check. Overlay patch 38 observes only A-path prompt rows from
  `runner.input_batch.token_ids_cpu`; it excludes B/rescore and persists
  mode-0600 request-ID/length/SHA records. The installed-overlay execution
  test proves the actual patched helper runs and catches missing/duplicate/cap
  defects.
- Collection: added the closed `collect-64` diagnostic value. It is legal only
  for the dedicated P45/M15 DP8xTP8 exact-token rollout-only profile, reserves
  at most 64 process-wide slots, ends a different trajectory after atomic
  capsule capture, and never reaches backward, optimizer, checkpoint, or
  training-step mutation. The classifier distinguishes mechanical `PASS` from
  the scientific `token_verdict=EQUAL|DIFFERENT`.
- Return path: before workload entry, the GCS worker must upload, download, and
  SHA-verify a non-sensitive no-clobber probe and issue an exact READY ACK.
  Snapshots validate their exact regular-file member set, mode-0600 tar modes,
  sizes, and SHA256 values. Finalization is retry-idempotent after a final
  manifest upload.
- Correction to the 19:21 preregistration: collect-mode stdout now contains no
  reversible token chunks. Worker logs are not a recovery fallback. Abrupt pod
  loss preserves only complete files included in the last successful periodic
  GCS poll; a newly renamed file can be lost inside the 30-second interval.
  Also, asynchronous runner capture order need not match sampler submission
  order; the proof is a request-ID join plus unique contiguous runner record
  indices.
- Host result: classifier 6/6, GCS 4/4, diagnostic renderer 3/3, P57 209/209,
  V1 101/101, flag audit 420/420, Python compilation, shell syntax, and
  `git diff --check` pass. P33 focused exact-image overlay execution also
  passes with terminal `P33_EXACT_IMAGE_PASS ...
  p57_tito_runner_execution=1 ... overlays=2`.
- Boundary: the complete V1 immutable-image gate is still running at this log
  point. No one-host/DP8xTP8 target, real GCS upload, durable manifest render,
  Kubernetes launch, commit, or push occurred.

## 2026-09-02T20:18:00Z — complete immutable-image gate PASS

- Type: immutable-image construction gate.
- Command: `bash tests/v1_phase4/run_exact_image.sh
  tunix_frozenlake_image:vllm-tpu0.25.0`.
- Image identity:
  `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`.
- Result: exit 0 with terminal `V1_HP_EXACT_IMAGE_PASS`; the terminal includes
  `frozenlake_tito_engine_witness=1`, `frozenlake_tito_collect64=1`,
  `frozenlake_tito_gcs=1`, `frozenlake_tito_default=legacy`, and
  `manifests=3`. The selected pinned tests also pass the submitted-future vs
  `RequestOutput.request_id` negative and the exact installed runner-witness
  overlay execution gate.
- Evidence boundary: this invocation's terminal output was observed directly
  but was not redirected to a durable raw-log file, so no raw-log SHA is
  claimed. This is immutable-image construction evidence, not TPU-runner,
  one-host observer-neutrality, real-GCS, or DP8xTP8 target evidence.
- No manifest render, Kubernetes/TPU launch, commit, push, or real GCS write
  occurred.

## 2026-09-02T21:00:00Z — T9c full-record phase opened

- Type: user correction / preregistration.
- Finding: T9b's `collect-64` carrier is internally consistent but stops after
  one initial-policy rollout pass. It cannot collect token/numerical evidence
  across the requested 300-update P45/M15 training curves. It also labels
  single-turn trajectories equal without a later-turn comparison and reports
  zero backward/update/checkpoint values as carrier literals rather than
  runtime measurements.
- User decision: add a separate `record-full` policy. A token difference is
  captured and the exact same trajectory continues ordinary training—no mask,
  drop, retry, replacement, or reweighting. Such a run is explicitly
  `NON_ZERO_TIM_DATA_COLLECTION`, not strict Zero-TIM. Missing, duplicate,
  swapped, or foreign request identity remains fatal.
- Scope: add stable trajectory/request/step/row joins, truthful coverage and
  runtime counters, incremental GCS deltas with retry/health receipts, and
  explicit P45/M15 full record renderers. Preserve legacy default, exact
  first-diff fatal behavior, rollout-only `collect-64`, and all base YAML
  topology/resource settings.
- Boundary: phase recorded before implementation. No TPU/Kubernetes launch,
  real GCS mutation, commit, or push is authorized.

## 2026-09-02T21:37:00Z — T9c host and immutable-image construction PASS

- Type: implementation / host and fixed-image gates.
- Implemented `record-full` as a separate, default-absent full-training
  policy. A same-ID token difference writes at most one bounded capsule per
  trajectory and the original row continues through reward, GRPO, backward,
  and optimizer unchanged. Identity corruption and non-whitelisted numerical
  failures remain fatal. Single-turn trajectories are counted unexercised.
- Added trajectory/request/policy-step/group/sequence-row joins, measured
  backward/microbatch/commit/alignment/checkpoint accounting, and a terminal
  classifier with separate execution, token, Zero-TIM, and evidence verdicts.
  Any token/alignment red or incomplete evidence forbids a Zero-TIM PASS.
- Replaced periodic whole-tree uploads with immutable delta snapshots,
  readback hashes, bounded retry/backoff, a heartbeat, final union proof, and
  trap-safe finalization. This is fake-remote certified only; real GCS is not.
- Host result: P57 216/216, V1 102/102, focused full-record 3/3, collection
  6/6, GCS 6/6, renderer 12/12, flag audit 421/421, Python/shell syntax, and
  `git diff --check` pass.
- The first complete fixed-image attempt caught a stale latch interaction:
  T9c reset the capsule latch per trajectory, but legacy `first-diff` requires
  one emission per process. Separate process-lifetime and per-trajectory
  latches restored both contracts. The full fixed-image gate was then rerun.
- Final fixed-image result: exit 0 on
  `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`
  with terminal `V1_HP_EXACT_IMAGE_PASS` including
  `frozenlake_tito_record_full=1 frozenlake_tito_gcs=1
  frozenlake_tito_default=legacy`.
- Evidence boundary: fixed-image output was observed directly and was not
  redirected to a durable raw log, so no raw-log SHA is claimed. No one-host,
  real-GCS, DP8xTP8 target, durable render, commit, push, or launch occurred.

## 2026-09-02T21:43:07Z — T9c operator handoff synchronized

- Type: documentation and release-contract audit; no runtime edit.
- `tasks/multiturn-tito-cross-workload/HANDOFF.md` is the authoritative T9c
  operator handoff. It records the exact `both-exact` plus `record-full` render
  command, generated paths, separately approval-gated unpiped launch commands,
  runtime and terminal receipts, four-way verdict interpretation, and complete
  return package.
- The active START HERE section in the production three-full `HANDOFF.md` and
  `RUNBOOK.md` now points to the same carrier. The obsolete shorthand
  `--token-continuity-debug` is explicitly identified as `first-diff`, not the
  T9c full-record mode.
- Post-documentation checks: `git diff --check` passes; flag audit remains
  421/421; focused full-record classifier is 3/3; paired renderer is 12/12.
- Boundary: no durable manifest was rendered and no commit, push, TPU/
  Kubernetes launch, or real GCS mutation occurred.

## 2026-09-02T22:20:00Z — T9d replay-capture phase opened

- Type: user-approved implementation scope / preregistration.
- Finding: T9c preserves full token arrays only for token-continuity
  differences and summary coordinates for numerical reds. It cannot replay a
  later A-B red without the full row arrays and the actor weights at that
  policy version. Its row-map, pre/post alignment, and update JSONL journals
  are also final-only; adding mutable files to the immutable live glob would
  trigger the existing changed-content rejection.
- User decision: store one structured host-only A/B/C sidecar for every update,
  not a redundant text CSV. Preserve at most the first-any and first-`>=1.0`
  nat red policy versions as actor-only pre-update evidence snapshots. These
  are non-resumable and do not relax the ordinary checkpoint-disabled contract.
- Scope: first implement immutable complete-line journal deltas and strict
  classifier poison controls, then all-update sidecars, bounded snapshot
  request/consumer wiring, GCS/final classification, observer-neutrality, host
  suites, and the complete pinned-image gate.
- Boundary: recorded before T9d runtime edits. No commit, push, durable render,
  TPU/Kubernetes launch, or remote mutation is authorized.

## 2026-09-02T22:45:00Z — T9d host and immutable-image construction PASS

- Type: implementation / poison gates / fixed-image gate.
- Added one atomic mode-0600 no-pickle NPZ per completed alignment update. It
  persists the host-owned prompt/completion IDs and masks, action mask,
  S-decode, S-prefill, T-old, policy version, sampling values, and stable
  trajectory/request/group/pair/row joins. Metadata binds every shape, dtype,
  array SHA, source commit, image identity, workload, DP8xTP8, and the
  pre-alignment record that issued the receipt.
- Added immutable complete-line byte-range chunks for full-row-map,
  pre-alignment, post-alignment, and update journals. Finalization requires the
  ordered chunks to reconstruct all four sources byte-for-byte. Live polling
  reuses uploaded path/size/SHA identities so accumulated multi-GB sidecars
  are not re-hashed every 30 seconds; terminal finalization still re-hashes
  every local evidence file and prior tar member.
- Added a bounded snapshot handshake. The alignment producer reserves only
  first-any and first-`>=1.0`-nat finite A-B policy steps. The trainer thread
  consumes the immutable request at the equal pre-update train step before
  backward/optimizer mutation and synchronously saves full actor state with
  no optimizer. Source SHA, image, workload, DP/TP, policy step, trigger,
  model inventory, and bounded fingerprints are recorded. Save failure leaves
  training rows unchanged and fails the terminal evidence classifier.
- Strengthened the classifier so a nominal PASS carrying any blocking,
  warning, or reported red list is poison. Sidecar, request, receipt, trigger,
  identity, permission, inventory, and join tampering are fatal.
- Verification: P57 225/225, V1 102/102, flag audit 421/421, Python and shell
  syntax, and `git diff --check` pass. Complete pinned-image gate exits 0 on
  `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`
  with terminal `V1_HP_EXACT_IMAGE_PASS` including
  `frozenlake_tito_record_full=1 frozenlake_tito_gcs=1
  frozenlake_tito_default=legacy`.
- Evidence boundary: the fixed-image output was observed directly without a
  durable raw-log file. Snapshot storage and GCS are fake transports in host
  gates. One-host observer-neutrality, real GCS/Orbax, abrupt-exit recovery,
  production-shape volume, DP8xTP8 target, commit, push, durable render, and
  Kubernetes launch remain unverified/unauthorized.

## 2026-09-02T23:00:00Z — T9d host repeat PASS; one-host carrier gap made explicit

- Type: verification repeat / carrier audit; no runtime implementation change.
- Repeated gates: P57 225/225, V1 102/102, flag audit 421/421, and
  `git diff --check` all pass.
- Carrier finding: `run_m15_onehost_verify.sh` is intentionally pre-alignment
  only (`backward=0`, `optimizer_commits=0`), while P64 replay is fixed to
  P45 DP8xTP8 and backward-no-commit. Neither can prove T9d's required
  gradient/update observer neutrality, and they must not be relabelled.
- Next implementation gate: a dedicated default-off DP1xTP4 one-update pair
  with equal seven input hashes, A/B/C, gradient and post-update
  fingerprints, request/row joins, module inventory, and separately measured
  capture I/O. Cross-arm input drift is `INCONCLUSIVE_INPUT_MISMATCH`.
- Boundary: no source commit, push, render, TPU/Kubernetes launch, or remote
  storage mutation occurred.

## 2026-09-03T01:02:12Z — T9d-3 carrier and pinned-image construction PASS

- Type: implementation / host admission / fixed-image gate.
- Reused the existing Perf-v2 FrozenLake DP1xTP4 three-update trainer carrier
  and added a closed exact-TiTO witness-off/witness-on identity. The pair
  runner waits for a continuously idle 120-second local container window and
  never interrupts another workload. Its judge requires equal seven-hash
  inputs, historical r7 gradient norms, post-update state fingerprints,
  strict alignment rows, canonical implementation identity, and equal
  semantic event censuses. Input drift is
  `INCONCLUSIVE_INPUT_MISMATCH`, never PASS.
- Production `record-full` now emits an immutable O_EXCL single-controller
  receipt, runs a distinct Tunix CheckpointManager save/restore probe against
  the actor-snapshot destination before rollout, and fails update 0 before
  backward if any token-continuity row differs. Red-policy actor snapshots are
  bounded to first-any and first-`>=1`/`>=8`/`>=32` categories, at most four
  non-resumable actor-only snapshots.
- Final host verification: P57 232/232, V1 102/102, APC
  31/31, flag audit 422/422, and focused one-host carrier/judge tests 5/5.
- The complete pinned-image gate twice exposed stale `record-full` test
  callers that lacked the newly mandatory source/image or DP/TP identity.
  Only the test fixtures were repaired; runtime admission was not relaxed.
  The final complete rerun exits zero on
  `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`
  with terminal `V1_HP_EXACT_IMAGE_PASS` and all 37 overlay files matching.
- Evidence boundary: fixed-image output was observed directly and not saved as
  a durable raw log, so no raw-log SHA is claimed. A direct train-venv test
  timed out after 120 seconds without a verdict; the same installed-code path
  passed in the pinned image and the timeout is not counted as certification.
  The matched v5p pair, real Orbax/GCS transport, abrupt-exit recovery,
  production-volume DP8xTP8, DeepSWE DP1xTP4 adjacency, commit, push, durable
  render, and Kubernetes launch remain unverified or unauthorized.

## 2026-09-04T08:21:00Z — five-CL release closeout PASS

- Type: user-approved local commit/push closeout; no TPU/Kubernetes launch.
- Runtime CLs:
  `c5d5ddd9c25c8ef00fb8bdfeac1a5e404601f510` exact FrozenLake TiTO
  runtime;
  `067cf3bf7f67bd976a361b514f245d71df829d71` installed runner witness;
  `dcde8a9105e2e7cd82748b7c2ffac6c0d81eb05a` replay-complete evidence;
  and `ba533dd7d8888c83d4c2ee50472a9346ccd3741c` closed carriers.
  The fifth CL contains this ledger, the peer handoff, and the durable gate
  bundle.
- Verification: P57 232/232, V1 102/102, APC 31/31, flags 422/422,
  Python/shell syntax, secret-pattern scan, and `git diff --check` pass.
  The complete fixed-image log ends in `V1_HP_EXACT_IMAGE_PASS`. After
  normalizing two whitespace-only context lines in patch 38, the focused
  installed-overlay rerun ends in `P33_EXACT_IMAGE_PASS`, proving both
  Qwen overlays still match all 37 manifest entries and the runner witness
  executes.
- Evidence: immutable local logs, per-file SHA256 values, the pinned image
  identity, runtime commit chain, and claim ceiling are recorded in
  `evidence/release_closeout_20260904_r1/receipt.json`.
- Claim ceiling: verified by CPU host and digest-pinned installed-image
  construction. The matched DP1xTP4 observer-neutrality pair, real GCS/Orbax,
  abrupt-exit recovery, DeepSWE DP1xTP4 adjacency, and P45/M15 DP8xTP8 remain
  unverified because they were not run.
- Next operator order: read back the pushed fifth SHA; use a clean checkout;
  obtain separate direct-TPU approval; run
  `run_tito_onehost_neutrality_pair.sh`; return both run roots,
  pair classification, and SHA ledgers; wait for review; only then seek
  separate render/launch approval for the P45 and M15 full pair.

## 2026-09-04T10:02:05Z — T9e all-event token-difference stream construction PASS

- Type: implementation / host admission / fixed-image gate; no commit, push,
  render, TPU/Kubernetes launch, or remote-state mutation.
- User decision implemented: every structurally valid `record-full` token
  difference, including update 0, repeated differences in one trajectory, and
  events after ordinal 64, is recorded and the unchanged trajectory continues
  through the full training path.  `collect-64` is deliberately unchanged.
- Evidence contract: every difference reserves one contiguous process-wide
  ordinal and emits one immutable mode-0600 capsule containing the complete
  actual/expected token streams plus request, trajectory, policy-step, turn,
  group/pair, and segment-ledger identity.  The terminal classifier and final
  GCS inventory require a one-to-one event/capsule mapping.  Missing,
  duplicate, foreign, malformed, tampered, or unwritten evidence is fatal.
- Verification: P57 234/234, V1 102/102, APC 12/12, flags 422/422, Python and
  shell syntax, and `git diff --check` pass.  The complete pinned-image gate
  exits zero on image
  `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`
  with terminal `V1_HP_EXACT_IMAGE_PASS`, including record-full, capsule
  integrity, engine witness, and GCS durability receipts.
- Honest boundary: these capsules fully reproduce token-transport and ledger
  differences; they do not create a full actor checkpoint for every event.
  Numerical A-B replay continues to use the separately bounded actor snapshot
  categories.  One-host neutrality, real GCS/Orbax behavior, abrupt-exit
  recovery, DeepSWE adjacency, and DP8xTP8 execution remain unrun.

## 2026-09-04T10:14:32Z — T9e pre-push rebase and focused recertification PASS

- Type: integration / post-rebase verification; user-approved commit/push,
  with no render, TPU/Kubernetes launch, or remote evidence mutation.
- The published branch moved from `a10c061a` to `90fd0e55` through two P67
  cluster-renderer commits. Their changed files do not overlap T9e, and the
  local T9e CL rebased without conflict.
- Post-rebase verification: P57 234/234, V1 102/102, APC 12/12, flag audit
  422/422, Python/diff hygiene, and the complete digest-pinned image gate
  pass. The terminal is `V1_HP_EXACT_IMAGE_PASS` with the record-full,
  capsule-integrity, engine-witness, and GCS-durability receipts present.
- Target boundary is unchanged: matched one-host neutrality, real GCS/Orbax,
  abrupt-exit recovery, DeepSWE adjacency, and DP8xTP8 remain unrun.
