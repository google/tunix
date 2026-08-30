# State

- Status: active; `E0W_ONEHOST_TITO_CARRIER_PASS`.
  Attempt 20 (`run_id=k02`) already executed. Its control is independently
  A-B/B-C exact for rounds 0/1/2 but root-terminal incomplete; treatment has
  zero completed rounds. E0u subsequently retrieved and hash-verified the
  treatment round-0 classifier input, but the archived classifier failed
  closed with no token-exact red join. The committed E0u incident is
  `evidence/v1_apc_m15_attempt20_e0u_r0_recovery_20260830/`; its
  `SHA256SUMS` SHA256 is
  `827b4038d269870d5b72e4f432b9680c89d79923d8bb2952163daca0e60ea093`.
  It reports a 1226-token observer/capsule divergence beginning at token 913.
  The admitted status remains `classification=NONE / INCONCLUSIVE`, with no
  B-reset runtime receipt, no target PASS, and no numerical repair authority.
- Attempt 19 used target source
  `d93d2729c5f036506fe754b929d42b142177a9b7`. Its incident bundle is
  `evidence/m15_e0_kv3_attempt19_incident/`; the `SHA256SUMS` file has SHA256
  `bc824561d39ed4e0bb5df65f56baff68e86ac64b8694a073f13a40bf31ba1636`.
- Treatment round 0 is a real serving-only red: A-B 366 bytes / 160 elements,
  B-C zero, 92.8% prefix-cache hits, first mismatch `[131,0]`, logical KV
  prefix 1226, comparison geometry `[256,8192]`. It preserved classifier input
  but failed classification before `ROUND_COMPLETE`; rounds 1/2 are absent.
- Control round 0 completed exact. Control round 1 was also A-B/B-C exact but
  emitted zero targeted KV records; round 2 is absent. Attempt 19 is therefore
  `INCONCLUSIVE_CARRIER_FAILURE`, not a three-round mechanism verdict.
- Root cause 1: the classifier required a red logical prefix to be strictly
  inside the KV snapshot. The first red is the next action scored from the
  prefix-1226 snapshot, so equality is causally valid. The repair admits
  equality in source binding/red join, reports it separately as
  `next_token_boundary_mismatch_positions`, and still rejects future `>L`
  reds.
- Root cause 2: the three diagnostic rounds advanced the FrozenLake dataset
  although the KV target prefix was statically bound to the D3e round-0 prompt.
  The repair freezes and hashes the exact 32-prompt round-0 identity inventory
  and requeues it for rounds 1/2 only under the exact `m15-e0-kv-v1` profile.
  Every round still reruns rollout requests, calls, cache chronology, A, B,
  and C. Neighboring profiles still advance their dataset.
- Runtime admission now requires one
  `E0_KV3_PROMPT_BATCH_FROZEN` marker, two
  `E0_KV3_PROMPT_BATCH_REQUEUED` markers, and one common prompt-batch SHA.
  Missing, duplicate, or drifting markers fail closed in `90_run.sh`.
- The immutable per-round order remains:
  `16 KV records -> classifier-input self-hash/upload/readback -> classifier
  PASS -> round archive/upload/readback -> ROUND_COMPLETE -> learner ACK`.
  B remains `reset_prefix_cache=True` with all cached-token counts zero.
- Last admitted tensor localization remains D3e Layer 0
  `k_post_rope -> rpa_output`, shape `[2048,1,15,8]`, source row 217 / source
  position 1225 / A call 83. Exact TiTO changes later-turn model input, so E0v
  does not inherit that numerical interval or the historical 1226-token
  prefix into a fresh target. No RoPE, RPA, attention, KV value, LM-head,
  loss, backward, optimizer, production profile, or production APC default
  changed.
- E0v implementation preserves already verified input facts on classifier
  failure, writes a bounded token-prefix/LCP audit with hashed request IDs,
  and returns eight A/B KV comparisons explicitly as unbound. The preserved-
  scratch wrapper is local-only. The fresh prepare wrapper renders exact TiTO
  in both APC arms only for `observer=layer`, `m15-wide-v1`, three rounds;
  historical `kv/kv3` rejects TiTO and no old target prefix is rendered.
- Validation: canonical E0v aggregate HOST PASS — task discovery 210/210,
  E0u failure recovery 8/8, TiTO postflight 5/5, token continuity 6/6, V1 CPU
  92/92, P3 prefix cache 31/31, fake-GCS persistence/forced failures, flag
  registry 409/409, Python/Bash syntax, static runner manifest `dae6dfa8...`,
  and `git diff --check`. Terminal:
  `M15_E0V_HOST_PASS task_discovery=210 return=1 round0_recovery=8
  tito_postflight=5 token_continuity=6 v1_cpu=92 p3_prefix_cache=31
  persistence=1 flags=409 manifest=dae6dfa8 syntax=1 diff_check=1
  exact_image=0 target_rerun=0 gcs=0 kubernetes=0 tpu=0`.
  Raw log: `/tmp/m15-e0v-host-gate-20260830-r1.log`, SHA256
  `75f7263daea0768ef74dc7c27cbabeae35c438045a4d0ec1d7be7928ef697e69`.
- Prepared scripts:
  `run_m15_attempt20_on_round0_preserved_scratch_audit.sh` is the only current
  Attempt-20 operation and performs local CPU/disk reads only;
  `prepare_m15_e0v_tito_layer_pair.sh` prepares but never launches the fresh
  program. Historical Attempt-20 GCS recovery and every `kv/kv3` prepare or
  launch tool must not be rerun.
- E0w adds a bounded DP1xTP4 exact-TiTO one-host pair before any fresh target.
  It uses the existing one-host rehearsal identity without a target selector
  or profile, runs APC-off before APC-on, and binds both arms to one
  source/diff/image. Each arm requires three strict A/B/C rounds, exact TiTO
  and B full-reset/all-cached-token-zero receipts in every round, positive
  APC-on cache hits, zero backward, and zero optimizer commits. Treatment
  exact and treatment red are separate complete one-host outcomes; neither is
  a DP8xTP8 PASS or `FIRST_RED_LOCALIZED`.
- Validation: canonical E0w aggregate HOST PASS — task discovery 225/225,
  E0u recovery 8/8, TiTO postflight 7/7, one-host arm 5/5, pair 5/5, runner
  contract 3/3, token continuity 7/7, V1 CPU 92/92, P3 31/31, persistence,
  flags 409/409, syntax, manifest, and diff hygiene. Terminal:
  `M15_E0W_HOST_PASS task_discovery=225 return=1 round0_recovery=8
  tito_postflight=7 onehost_arm=5 onehost_pair=5 onehost_runner=3
  token_continuity=7 v1_cpu=92 p3_prefix_cache=31 persistence=1 flags=409
  manifest=dae6dfa8 syntax=1 diff_check=1 exact_image=0 onehost_v5p=0
  target_rerun=0 gcs=0 kubernetes=0 tpu=0`. Latest post-ledger raw log:
  `/tmp/m15-e0w-host-gate-20260830-r3.log`, SHA256
  `4a159cbed02337b4c878de582e06f319d991e1609f04da2b9de3f8d3c8af9762`.
- Pinned exact-image PASS on image
  `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`.
  Latest complete raw log is
  `/tmp/m15-e0w-exact-image-20260830-r4.log`, SHA256
  `65a1c193887601db845aada30532bddce1b6b69b5d79c266b1357f4d0414c105`,
  1,194 lines / 246,198 bytes. It exits zero with
  `V1_HP_EXACT_IMAGE_PASS`, E0w TiTO/arm/pair/runner `7/5/5/3`, durability
  `1`, round provenance `1`, and `manifests=3`.
- One-host attempts `e0w1`, `e0w2`, and `e0w3` are preserved immutable
  carrier failures: P57 CLI/env mismatch, missing narrow profile-less M15
  one-host admission, and `fsdp,tp` versus required replicated `dp,tp` mesh,
  respectively. Fixes affect one-host identity/admission only. Attempt
  `e0w4` then completed `ONEHOST_PAIR_EXACT`: APC-off and APC-on A-B/B-C are
  `[0,0,0]/[0,0,0]`, APC-on reaches 91.5% prefix-cache hits, TiTO receipt
  counts are `[4,7,6]` per arm, and B full-reset/all-cached-token-zero receipt
  counts are `[1,1,1]` per arm. Root manifest SHA256 is
  `b52fe75af56f5c66c2ba352d25163a18ab854c823b17c71d91a555fc02155589`.
  Backward and optimizer commits remain zero.
- Publication admission: the E0v/E0w delta was rebased directly onto fetched
  remote tip `89ef0ad567d5abe33074a53c6655a6b8bc80cf6e`. That tip differs from the
  source-tested predecessor `29c923dc...` only by nine added incident evidence
  files; no executable, profile, test, M15 handoff, or image runner changed.
  Post-rebase canonical host
  raw log
  `/home/yuxuan/code_rl_repro/m15-e0w-host-gate-postrebase-r1.log`, SHA256
  `fcf24a07c0ab7f6199fa0555ac583968568f58e9ed51bd59144e94eaf8140f05`,
  ends `M15_E0W_HOST_PASS`. Post-rebase complete pinned-image raw log
  `/home/yuxuan/code_rl_repro/m15-e0w-exact-image-postrebase-r1.log`, SHA256
  `7f4b2dc4703ce4713b5ed2a6802f279481f27a7cd60eec301678a3b114984b49`,
  exits zero with `m15_tito_impl=1 m15_tito_default=off`, E0w `7/5/5/3`,
  durability/provenance `1/1`, and `manifests=3`. The authoritative source is
  the exact full remote-read branch SHA containing this state, never the base
  or a transient local candidate. The pre-rebase `e0w4` one-host result remains
  source-bound and is not inherited by the rebased source.
- Current gate: on the bucket-capable machine, fetch the exact published SHA,
  create a clean `local/*` worktree, and run prepare-only with a fresh label.
  Any fresh DP8xTP8 apply/launch is another separate approval and may only be
  the matched three-round E0v debug pair. Rebased-source TPU, GCS,
  Kubernetes, DP8xTP8 target, and preserved-scratch execution remain NOT RUN.
- Infrastructure warning: `/tmp` shares a root filesystem reporting 100% use
  with about 488 MiB available after the gates. No evidence was deleted.
- Claim ceiling:
  `ATTEMPT20_CONTROL_EXACT_3_OF_3_ANALYSIS_GRADE /
  ATTEMPT20_TREATMENT_ZERO_COMPLETED_ROUNDS /
  ATTEMPT20_ROUND_EVIDENCE_PARTIAL /
  E0U_GCS_RECOVERY_CLASSIFIER_JOIN_FAILED /
  E0V_FAILURE_AUDIT_IMPLEMENTED /
  E0V_EXACT_TITO_LAYER_CARRIER_IMPLEMENTED /
  E0V_CANONICAL_HOST_PASS /
  E0W_ONEHOST_CARRIER_IMPLEMENTED /
  E0W_CANONICAL_HOST_PASS /
  E0W_PINNED_EXACT_IMAGE_PASS /
  E0W_ONEHOST_V5P_PAIR_EXACT /
  E0V_PRESERVED_SCRATCH_AUDIT_NOT_RUN /
  ATTEMPT20_TREATMENT_CLASSIFICATION_NONE /
  ATTEMPT20_B_RESET_RUNTIME_RECEIPT_UNAVAILABLE /
  NO_DP8TP8_3_OF_3 /
  NO_TARGET_PASS /
  TARGET_RERUN_NO /
  HISTORICAL_FIRST_RED_NOT_INHERITED /
  ATTEMPT19_REAL_APC_RED /
  ATTEMPT19_INCONCLUSIVE_CARRIER_FAILURE /
  PHASE_E_CLOSED /
  APC_NUMERICAL_FIX_NOT_IMPLEMENTED`.
- Updated: 2026-08-30.
