# State

- Status: active;
  `E0U_ATTEMPT20_ON_ROUND0_OFFLINE_RECOVERY_HOST_PASS`.
  Attempt 20 (`run_id=k02`) already executed and its compact read-only salvage
  return is archived at
  `evidence/v1_apc_m15_attempt20_e0_kv3_salvage_return_20260830/`.
  Control achieved `CONTROL_EXACT_3_OF_3`: rounds 0/1/2 have A-B=0 and B-C=0,
  but root terminal state is absent. Treatment has zero completed rounds.
  Round 0 has a classifier-input-receipt presence signal but no returned
  `ROUND_INPUT`, classification, or completion; rounds 1/2 have no checkpoint
  receipt. Root runtime proof of B full reset/all cached-token counts zero is
  also unavailable. The returned status is `ROUND_EVIDENCE_PARTIAL`, not
  target PASS.
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
  position 1225 / A call 83. The preceding E0t change touches classifier accounting,
  diagnostic prompt scheduling, postflight admission, tests, and operator
  wrappers only. No RoPE, RPA, attention, KV value, LM-head, loss, backward,
  optimizer, production profile, or production APC default changed.
- Validation: canonical E0u aggregate HOST PASS — task discovery 199/199,
  Attempt-19 return regression, E0u recovery 6/6, V1 CPU 91/91, P3
  prefix-cache 31/31, fake-GCS persistence and forced failures, flag registry
  409/409, Python/Bash syntax, current static runner manifest `dae6dfa8...`,
  and `git diff --check`. Terminal:
  `M15_E0U_HOST_PASS task_discovery=199 return=1 round0_recovery=6 v1_cpu=91
  p3_prefix_cache=31 persistence=1 flags=409 manifest=dae6dfa8 syntax=1
  diff_check=1 exact_image=0 target_rerun=0 gcs=0 kubernetes=0 tpu=0`.
  Raw log: `/tmp/m15-e0u-host-gate-20260830-r4.log`, SHA256
  `738d8df5a7ca9adef35735375e319c242f225b1404e14ef818fb72ef5ba4c4bf`.
- Prepared scripts:
  `run_m15_e0_kv3_host_gate.sh`,
  `prepare_m15_attempt20_e0_kv3_pair.sh`, and
  `run_m15_attempt20_e0_kv3_return_recovery.sh` are historical Attempt-20
  launch/return tools and must not be rerun now. The only current executor
  entrypoint is
  `run_m15_attempt20_on_round0_offline_recovery.sh`; it is read-only GCS plus
  local CPU and must run only on the bucket-capable machine after separate
  approval.
- Current gate: review this local diff. Official pinned exact-image,
  commit/push, the read-only GCS recovery, and any future DP8xTP8 launch are
  separate user approvals. Do not launch a target before attempting the
  already durable round-0 recovery.
- Claim ceiling:
  `ATTEMPT20_CONTROL_EXACT_3_OF_3_ANALYSIS_GRADE /
  ATTEMPT20_TREATMENT_ZERO_COMPLETED_ROUNDS /
  ATTEMPT20_ROUND_EVIDENCE_PARTIAL /
  E0U_RECOVERY_IMPLEMENTED /
  E0U_CANONICAL_HOST_PASS /
  E0U_EXACT_IMAGE_NOT_RUN /
  E0U_GCS_RECOVERY_NOT_RUN /
  TREATMENT_CLASSIFICATION_NONE /
  B_RESET_RUNTIME_RECEIPT_UNAVAILABLE /
  NO_3_OF_3 /
  NO_TARGET_PASS /
  TARGET_RERUN_NO /
  ATTEMPT19_REAL_APC_RED /
  ATTEMPT19_INCONCLUSIVE_CARRIER_FAILURE /
  PHASE_E_CLOSED /
  APC_NUMERICAL_FIX_NOT_IMPLEMENTED`.
- Updated: 2026-08-30.
