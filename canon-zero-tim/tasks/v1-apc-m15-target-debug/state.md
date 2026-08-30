# State

- Status: active;
  `E0T_ATTEMPT20_SALVAGE_RETURN_ARCHIVED`.
  Gate 4 read-only GCS salvage return executed for Attempt 20 (`run_id=k02`).
  Control arm achieved `CONTROL_EXACT_3_OF_3` (all 3 rounds 0/1/2 exact).
  Treatment arm completed round 0 live-KV diagnostic capture (93.0% cache hits)
  and partial rounds preserved as `ROUND_EVIDENCE_PARTIAL` upon early resource release.
  128 TPU freed and redeployed to DeepSWE Full (K06).
  Evidence archived at `evidence/v1_apc_m15_attempt20_e0_kv3_salvage_return_20260830/`.
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
  position 1225 / A call 83. This change touches classifier accounting,
  diagnostic prompt scheduling, postflight admission, tests, and operator
  wrappers only. No RoPE, RPA, attention, KV value, LM-head, loss, backward,
  optimizer, production profile, or production APC default changed.
- Validation: aggregate HOST PASS — task discovery 193/193, salvage-first
  return paths, V1 CPU 91/91, P3 prefix-cache 31/31, fake-GCS persistence and
  forced failure, flag registry 408/408, Python/Bash syntax, current static
  runner manifest `dae6dfa8...`, and `git diff --check`. Terminal:
  `M15_E0_KV3R_HOST_PASS task_discovery=193 return=1 v1_cpu=91
  p3_prefix_cache=31 persistence=1 flags=408 manifest=dae6dfa8 syntax=1
  diff_check=1 exact_image=0 target=0 gcs=0 kubernetes=0 tpu=0`.
  Raw log: `/tmp/m15-e0-kv3r-host-gate-postrebase-20260830.log`, SHA256
  `ef6992bc55079965759b12395f15378c0ca1d693628ac05e5d60742f4712e811`.
- Prepared scripts:
  `run_m15_e0_kv3_host_gate.sh`,
  `prepare_m15_attempt20_e0_kv3_pair.sh`, and
  `run_m15_attempt20_e0_kv3_return_recovery.sh`. Prepare is local CPU/disk and
  fake GCS only. The return wrapper is read-only GCS and must not run here.
- Current gate: deliver the authorized commit/push, verify the remote-tracking
  SHA, then have a clean exact-SHA executor run prepare-only. Official pinned
  exact-image, fresh DP8xTP8 launch, and read-only GCS return are three separate
  later approvals.
- Claim ceiling:
  `ATTEMPT19_REAL_APC_RED /
  ATTEMPT19_INCONCLUSIVE_CARRIER_FAILURE /
  E0T_REPAIR_HOST_PASS /
  POST_REPAIR_EXACT_IMAGE_NOT_RUN /
  POST_REPAIR_TARGET_NOT_RUN /
  PHASE_E_CLOSED /
  APC_NUMERICAL_FIX_NOT_IMPLEMENTED`.
- Updated: 2026-08-30.
