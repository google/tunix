# State

- Status: active
- Objective: Localize and remove the 64-chip Pathways decode-versus-prefill bitwise carrier while
  keeping every committing training gate fail-closed.
- Definition of done: One source-pinned flag-on run reports `S_decode_vs_S_prefill=0`,
  `S_prefill_vs_T_old=0`, and `T_old_vs_T_current=0` before a full workload is allowed to commit.
- Task directory: `canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/`
- Directory state: publication state is current through local base `05db7fd8`;
  the remote branch has advanced independently. The current dirty worktree
  contains uncommitted P38.2g replay work, the P38.2g2 serving-capture/U
  implementation, and a separately approved GSM8K retry-policy change. No
  commit or push occurred in this phase.
- Current phase: P38.2g2 source-pinned Pathways serving capture is the sole
  active phase. P38.2e GSM8K full is ready for an independent parallel target
  run but does not advance the FrozenLake causal ladder. The generated GSM8K
  full JobSet now has the user-approved operational retry budget of three;
  checkpointing remains disabled and every retry starts from update 0. All
  other P33 entries retain zero restarts.
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
  complete exact-image P33 CPU gate is also green (67 workload tests, 26
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
  domain, not a successful repair that was later dropped. It may be retested
  only as a default-off causal arm in the new domain.
- Next action: review and explicitly authorize publication of the locally
  gated P38.2g2 code. The target operator then renders the dedicated stock/U
  manifests, dry-runs both, and applies stock only. Stock must emit exactly
  one complete serving record, a verified stdout archive, the known hard A/B
  red, Attempt 0, and zero commits before U is applied. The pinned v3 API
  cannot express a clean write-only `W` arm, so do not report one.
- Blockers: the capsule lacks original serving block tables, page allocation,
  and per-call request distributions. A mask-derived one-host result cannot by
  itself promote a production repair. The target schedule-aware update verdict
  also still requires the external 64-chip operator.
- Key artifacts: `../../debug_logs/p33_r35_gsm8k_full.raw.log`,
  `../../debug_logs/p33_r35_frozenlake_full.raw.log`, `plan.md`,
  `phases/p38-1-evidence-hardening.md`, `HANDOFF.md`
- Key local artifacts: `/mnt/disks/tunix-data/logp_probe_1host/p38_onehost_0810_r2.result.json`,
  `/mnt/disks/tunix-data/logp_probe_1host/p38_tail_0810_r1.result.json`,
  `/mnt/disks/tunix-data/logp_probe_1host/p38_aval_0810_r1.result.json`,
  `../../debug_logs/p38_p38e1_frozenlake_mismatch_capsule.npz`,
  `../../debug_logs/p38_p38e1_frozenlake_pre_alignment.jsonl`,
  `artifacts/p38_2g_local_gate.md`,
  `artifacts/p38_2g_onehost_synthetic_0811.md`,
  `artifacts/p38_2g_onehost_target_row191_0811.md`,
  `artifacts/p38_2g2_local_gate_0811.md`,
  `phases/p38-2g2-pathways-serving-envelope.md`
- Updated: 2026-08-11 05:42 UTC
- Rollback: leave `CANON_P38_FROZENLAKE_REPLAY`,
  `CANON_P38_SERVING_CAPTURE_DIR`, and `CANON_KV_UNIFIED` unset and discard
  the uncommitted P38 work to return to the local base. The loss, precision,
  prefix cache, production default attention branch, and optimizer schedule
  are unchanged.
