# State

- Status: active; `E0_KV3_DURABILITY_IMPLEMENTED_HOST_PASS`. The implementation
  was built from base `a951656e90ee91d5d7781d625377831dfd6c255d`; the user
  explicitly authorized its commit/push delivery. The published source is the
  full commit containing this file and must be resolved with `git rev-parse
  HEAD`, not inferred from the implementation base.
- Why this phase exists: Attempt 18 was deliberately rendered as a one-round
  E0 discriminator; it did not accidentally stop before round 3. Its returned
  package later failed provenance admission, and a one-round/root-dependent
  design cannot establish repeat stability or guarantee that useful evidence
  survives a later round/root failure. The historical `observer=kv` one-round
  route remains unchanged for read-only Attempt-18 recovery.
- New execution identity: `observer=kv3`, durability profile
  `m15-e0-kv-v1`, exactly three frozen-weight DP8×TP8 M15/main rounds per
  arm. Production profiles remain APC-off and do not select this identity.
- Per-round order is fail closed:
  `16 KV records (8A+8B) staged -> classifier-input self-hash/archive ->
  upload/readback -> classifier PASS -> round archive/upload/readback ->
  ROUND_COMPLETE -> learner ACK`. `ROUND_COMPLETE` hashes the round input,
  classifier result, and classifier-input receipt. The full run log is
  collected once at the root instead of being copied into every round.
  Record indices remain globally monotonic; only the per-round candidate set
  and byte budget reset. Cross-round A/B pairing is rejected.
- Durability boundary: round 0/1 evidence is independently recoverable even
  if round 2 or root collection fails. The read-only return downloads the
  small per-round completion/classifier/checkpoint receipts before attempting
  `COLLECTED.json` or `COMPLETE.json`; missing root terminal state is reported
  as `ROUNDS_RECOVERED_ROOT_INCOMPLETE`, not as missing evidence and not as a
  target PASS.
- Incident ledger: the three-round E0 profile bypasses the redundant bounded
  incident ledger, which otherwise saturates across the M15 chronology. The
  replay envelope plus sealed KV rounds are the replacement. Legacy and
  non-E0 profiles retain their old behavior.
- Aggregate decisions require all three rounds. APC-off must be
  `CONTROL_EXACT_3_OF_3`. APC-on may be
  `TARGET_NON_REPRODUCTION_3_OF_3`,
  `LIVE_KV_FINGERPRINT_EQUAL_3_OF_3`, or
  `LIVE_KV_FINGERPRINT_DIFFERS_3_OF_3`; mixed outcomes fail closed. Every
  round still requires B-C exact and B full reset with zero cached tokens.
- Last admitted numerical boundary remains D3e, Layer 0
  `k_post_rope -> rpa_output`, shape `[2048,1,15,8]`, source row 217 / source
  position 1225 / A call 83. No RoPE, RPA, attention, KV value, LM-head,
  loss, backward, optimizer, or production-default repair is present.
- Attempt-18 official-return status remains rejected:
  `ATTEMPT18_E0_RETURN_PROVENANCE_FAIL`. Its reported control/treatment
  values do not become official evidence through this implementation.
- Validation: aggregate HOST PASS — task discovery 193/193, salvage-first
  return partial/full paths, V1 CPU 91/91, P3 prefix-cache 31/31, fake-GCS
  persistence including three sealed/classified/readback rounds and a forced
  round-2 failure preserving rounds 0/1, flag registry 398/398, Python/Bash
  syntax, static manifest binding, and `git diff --check`. Patch 36 also
  applied to the registered pre-Patch-36 runner and reproduced manifest SHA
  `15fddce5eb5157494cc01639a50e677e5d7ce775b883ff5c7d29f6a854317f67`.
  Raw host log: `/tmp/m15-e0-kv3-host-gate-final-20260830.log`, SHA256
  `cccc0bdce2dd01d5dd84f1fdc61f31ba4be7570ed692fccedb43387e839cf12d`.
  Official pinned exact-image and DP8×TP8 target are NOT RUN.
- Prepared local-only scripts:
  `run_m15_e0_kv3_host_gate.sh`,
  `prepare_m15_attempt19_e0_kv3_pair.sh`,
  `run_m15_attempt19_e0_kv3_gcs_return.sh`, and
  `run_m15_attempt19_e0_kv3_return_recovery.sh`. Running the prepare script is
  local CPU/disk only. Pinned Docker, GCS reads, and Kubernetes/TPU launch are
  separate approval gates.
- Current gate: clean exact-SHA prepare after publication. The other agent must
  use a clean worktree at the full delivered SHA, run prepare-only, obtain
  separate pinned exact-image approval, then obtain a separate target-launch
  approval. GCS return is a further read-only approval.
- Claim ceiling:
  `FIRST_RED_LOCALIZED_FROM_D3E /
  E0_KV3_DURABILITY_HOST_PASS /
  EXACT_IMAGE_NOT_RUN /
  TARGET_NOT_RUN /
  PHASE_E_CLOSED /
  APC_NUMERICAL_FIX_NOT_IMPLEMENTED /
  NUMERICAL_REPAIR_NOT_AUTHORIZED`.
- Updated: 2026-08-30.
