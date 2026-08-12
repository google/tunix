# State

- Status: active
- Objective: Localize and remove the 64-chip Pathways decode-versus-prefill bitwise carrier while
  keeping every committing training gate fail-closed.
- Definition of done: One source-pinned flag-on run reports `S_decode_vs_S_prefill=0`,
  `S_prefill_vs_T_old=0`, and `T_old_vs_T_current=0` before a full workload is allowed to commit.
- Task directory: `canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/`
- Directory state: tracked. P38.2g4 D0 is published at
  `b89435ca7d64faa65c00b5a85152f71fdfc60167` on
  `origin/yuxzhang/canon-zero-tim`. The complete P38s5 operator
  run-and-return contract is published at
  `d5a0ac30bdc1ecdd4bf3c5948baf8e54c48502b5` on the same branch. Main was not
  checked out or modified. P38.2g5 is published at
  `02e8c05d45d9423a05ea96ce066b0f7009a511e2` on the same branch.
- Current phases: two independent tracks are deliberately active. P38.2d has
  added and locally gated the user-requested GSM8K-full warning-only alignment
  override so the time-sensitive convergence campaign cannot be stopped by an
  alignment gate.
  P38.2g2 remains the strict zero-TIM root-cause track. Its local admission
  hardening is complete, green, and published. Stock and U target numerical
  runs now exist, but both logs stop at the child alignment traceback before
  the serving classifier/archive and final postflight. P38.2g2 therefore
  remains inconclusive at the serving-envelope layer. P38.2g3 is pending on a
  complete stock-only capture and exact E0 reproduction. P38.2g4 is active and
  hardens that capture before another expensive target run: four bounded
  prefix strata replace the brittle single `min_prefix=1788` trigger, callable
  identity is attested, and first-divergence localization is preregistered
  without naming RoPE or page state as the cause. P38.2g5 is now the active
  diagnostic-reachability phase after P38s5 produced no hook evidence.
- Latest local gate: P38.2g5 is complete locally. The trigger now selects
  request-level scheduler prefixes, retains packed positions as an attestation,
  and emits bounded init/observation evidence even on misses. A finite A-B red
  with exact B-C now persists its capsule and exits before backward instead of
  raising before the required stop marker. Qwen3-1.7B and Qwen3-8B overlays
  each pass 16/16 exact-image tests with all 29 manifest entries matching; the
  complete frozen-image CPU gate passes 81 workload tests, 31 alignment tests,
  and all adjacent suites. The installed runner SHA-256 is
  `72c4307859c32de4e7080823bbe0693fb04c21a67ab82a3cfe829bb6c39ed18c`.
- Prior local gate: P38.2g4 D0 completed. The real
  `continue_decode` capture now excludes
  live-but-unscheduled requests without compacting their physical scheduler
  slots, selects one concrete scheduled request per prefix stratum, emits and
  validates the anchor request/prefix plus request/DP/slot/global/attention/
  selector/page mappings, and requires an exact request/token-history join to
  the durable mismatch capsule for stock. Source commit and callable identity
  must remain stable across all four records. Classifier controls pass 25/25,
  renderer controls 5/5, shell postflight passes, exact-image Qwen3-1.7B and
  Qwen3-8B overlays pass 14/14 each with all 29 manifest entries matching, and
  the complete frozen-image P33 CPU gate passed (78 workload tests, 29
  alignment tests, all adjacent suites). The installed runner SHA-256 is
  `fe81622996a1c73bbd17187ee603e6a191165202da40d07b5e428fe41b5db516`.
- Last verified fact: P38e1 source row 191 completed on real Qwen3-8B DP1xTP4
  with exact weights, deterministic repeats, an effective negative control,
  no backward, and zero optimizer commits. R0 and R1 were exact at every
  measured stage, while both differed from REF at 395 of 517 action logprobs.
  REF exactly reproduced the captured `S_prefill`/`T_old` logprob SHA, but
  local R0/R1 reproduced neither the captured decode values nor their sparse
  boundary. Classification is `LOCAL_CARRIER_NOT_ISOLATED`; the mask-derived
  serving envelope is not admissible for R2/R3. The P38e1 target capsule and
  pre-alignment JSON are
  present in the repository and pass file SHA, schema, embedded-array SHA, and
  schedule-coverage checks. They preserve source rows 191 and 199. Across the
  source batch, `S_prefill_vs_T_old` is exact while
  `S_decode_vs_S_prefill` differs in 15 of 49,002 action elements with maximum
  absolute difference 0.10391998291015625. The mismatches lie at logical KV
  prefixes 1850 through 2462, turns 3 through 4, and sequence chunks 7 through
  9; none immediately follows an environment token. The capsule does not
  preserve the original serving block tables or per-call scheduler metadata,
  so its R0 schedule remains a mask-derived causal counterfactual. Earlier
  real-Qwen3-8B DP1xTP4 synthetic controls at prompt
  lengths 256 and 1788 completed with exact actor/engine weights across
  8,190,735,360 elements, deterministic repeats, effective one-bit negative
  controls, no backward, and zero optimizer commits. R0 and R1 were bitwise
  equal at every observed stage, but both were red against REF at every scored
  action at both depths; the shallow difference was larger. This admits the
  measurement path but does not reproduce the production KV-1791 onset. The
  complete exact-image P33 CPU gate is also green (67 workload tests, 28
  alignment tests, all adjacent suites and negative controls). P38.2g adds 11
  schedule/capsule tests, 5 classifier controls, and one four-device fake-model
  adapter integration test; all pass. Qwen3-1.7B and Qwen3-8B overlay gates
  each pass 10/10. The verified target capsule has passed CPU admission but has
  not yet executed on the one-host TPU.
- Local limitation: the legacy GSM8K L3 runner has a pre-existing two-versus-
  eight trajectory contract mismatch, so it was not used to manufacture a
  hardware PASS. Real-model compile time and peak HBM for the new scalar commit
  evidence remain NOT RUN.
- Historical correction: Phase 13 executed `CANON_KV_UNIFIED` and observed no
  numerical change; it was a clean negative in its four-chip/short-context
  domain, not a successful repair that was later dropped. The default-off U
  operation has now also executed in the 64-chip production domain and
  remained materially red; it is not a sufficient repair and need not be
  rerun.
- Latest target evidence: stock `p38s1` reproduced the strict A-B red at
  43/46,417 action elements (`max_abs=0.2780647277832031`) with B-C exact.
  U `p38u1` executed `KV_UNIFIED_two_pass` and remained red at 9/46,589
  elements (`max_abs=0.27657318115234375`) with B-C exact. U is therefore not
  a sufficient repair. Because these are different sampled trajectories, the
  43-to-9 count change is not a controlled improvement measurement and does
  not prove a writer, timing, or page-lifecycle mechanism.
- Evidence correction: neither target head log reached outer postflight. Both
  lack the official serving-classification PASS, serving archive, complete
  capsule transport, and final PATHTRACE. The generic committed capsule has
  SHA `dae4e75d...` and is an exact duplicate of the older P38e1 capsule, not
  the logged p38s1 (`2dffb993...`) or p38u1 (`245a0c9b...`) artifact. The
  earlier stock-capture PASS wording is withdrawn.
- P38s4 evidence correction: commit `819207bd` contains only a 200-line tail
  beginning inside layer 30 and ending after final RMSNorm. It has no
  source/Attempt-0 preamble, workload exit, serving-capture records, alignment
  artifact, classifier, archive, or final postflight. P38s4 is therefore
  `INCONCLUSIVE_TAIL_ONLY`, not a completed D1 attempt and not evidence of a
  code failure.
- P38s5 evidence correction: the 6,069-line log starts at byte zero but ends
  after final-norm trace without a child exit, alignment record, terminal
  precheck, classifier, archive, or outer postflight. It contains no init,
  observation, or capture marker, so it cannot establish whether the hook was
  imported, called, or merely missed its ranges. The archived claim that
  FrozenLake prompts were about 200 tokens and that the recipe bypassed
  `GRPOLearner` is withdrawn as unsupported by the current code and evidence.
- Next action for the strict track: after P38.2g5 is published and separate
  resource approval is granted, execute the P38s6
  stock-only Attempt-0 run-and-return protocol at the top of `HANDOFF.md`.
  Preserve the complete non-timestamped terminal head log through outer
  postflight and return the exact evidence directory, including the real
  run-specific capsule and serving tar. Do not rerun U. P38.2g3 E0 remains
  blocked until the official stock capture passes an exact request/token-
  history join and whole-vector reproduction.
  The independent GSM8K convergence campaign remains governed by its
  published warning-only policy.
- Blockers: the decisive serving block-table/page-state archive is missing.
  Current evidence does not prove scheduler ownership, stale page-table,
  partial-write, padding-leak, or physical-topology causality. The external
  64-chip operator is required for one complete stock-only capture. The pinned
  v3 API cannot express a clean write-only `W` arm.
- Key artifacts: `../../debug_logs/p33_r35_gsm8k_full.raw.log`,
  `../../debug_logs/p33_r35_frozenlake_full.raw.log`, `plan.md`,
  `phases/p38-1-evidence-hardening.md`, `HANDOFF.md`
- Key local artifacts: `/mnt/disks/tunix-data/logp_probe_1host/p38_onehost_0810_r2.result.json`,
  `/mnt/disks/tunix-data/logp_probe_1host/p38_tail_0810_r1.result.json`,
  `/mnt/disks/tunix-data/logp_probe_1host/p38_aval_0810_r1.result.json`,
  `../../debug_logs/p38_p38e1_frozenlake_mismatch_capsule.npz`,
  `../../debug_logs/p38_p38e1_frozenlake_pre_alignment.jsonl`,
  `../../debug_logs/p38_p38s1_frozenlake_stock.raw.log`,
  `../../debug_logs/p38_p38s1_frozenlake_pre_alignment.jsonl`,
  `../../debug_logs/p38_p38u1_frozenlake_unified.raw.log`,
  `../../debug_logs/p38_p38u1_frozenlake_pre_alignment.jsonl`,
  `../../debug_logs/p38_p38s4_frozenlake_stock.raw.log`,
  `artifacts/p38_2g_local_gate.md`,
  `artifacts/p38_2g_onehost_synthetic_0811.md`,
  `artifacts/p38_2g_onehost_target_row191_0811.md`,
  `artifacts/p38_2g2_local_gate_0811.md`,
  `artifacts/p38_2g4_local_gate_0811.md`,
  `phases/p38-2g2-pathways-serving-envelope.md`,
  `phases/p38-2g3-page-topology-discriminator.md`
- Current local gates: frozen-image P33 CPU gate PASS (78 workload tests, 29
  alignment tests, adjacent regressions and negative controls), exact-image
  Qwen3-1.7B/Qwen3-8B overlay gates 14/14 each PASS, serving classifier 25/25,
  renderer 5/5, and postflight stock/U
  PATHTRACE negative controls PASS.
- Updated: 2026-08-12 UTC after auditing P38s5 and locally completing the
  request-anchored P38.2g5 diagnostic repair
- Rollback: leave `CANON_P38_FROZENLAKE_REPLAY`,
  `CANON_P38_SERVING_CAPTURE_DIR`, and `CANON_KV_UNIFIED` unset. The published
  mechanisms are default-off; loss, precision, prefix cache, stock attention,
  and optimizer behavior remain unchanged. Leave
  `CANON_GSM8K_ALIGNMENT_WARN_ONLY=0` to restore the strict GSM8K alignment
  gate.
