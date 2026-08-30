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
