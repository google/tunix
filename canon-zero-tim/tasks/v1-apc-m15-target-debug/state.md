# State

- Status: active; `LIVE_KV_FINGERPRINT_EQUAL` admitted via Phase E0r intake review.
  The official 4-file compact return verifies 100% via `review_m15_attempt18_e0_return.py`.
- Current incoming evidence base:
  `evidence/v1_apc_m15_attempt18_e0_kv_20260829/`. Its 3-member manifest
  verifies and has SHA256
  `ce762783e6b2f1a6fae37190f3af6e96baa39302931d29081c1d93146b7c9475`.
- Attempt-18 runtime source: `12207e3281db13461350fe7ef68dbaadfe713a58`.
- Admitted numerical values:
  - Control (APC-Off) executed 256 trajectories, solve rate 18.4%, alignment
    precheck N_action=123010, A-B=0, B-C=0 (Clean Green PASS).
  - Treatment (APC-On) executed 256 trajectories with 92.8% prefix cache hit
    rate, solve rate 16.8%, alignment precheck N_action=117834, A-B=1499 bytes / 88 elements,
    B-C=0 (Red reproduced).
  - All eight 1226-token Layer-0 diagnostic fingerprints are equal (`LIVE_KV_FINGERPRINT_EQUAL`).
    This is a diagnostic fingerprint over the uniquely bound red request, not proof of all KV bytes.
- Last admitted localized boundary: Layer 0 `k_post_rope -> rpa_output`, shape
  `[2048,1,15,8]`, source row 217 / source position 1225 / A call 83.
- Current gate: Phase E0r. The user approved publishing the HOST-PASS recovery
  tree with exact-image debt explicit. After publication, the separately
  approved official pinned-image aggregate must pass before a bucket-capable
  agent uses the preserved `e01` render directory and
  `run_m15_attempt18_e0_return_recovery.sh` under separate GCS-read approval.
  No TPU/Kubernetes run is currently required.
- If the official result is fingerprint-equal, the next design discussion is
  an exact metadata/gather discriminator: compare the selected
  `attn_metadata.block_tables` row, sequence/query/input-position metadata,
  physical page identity, 10/16 tail-page mask, and gathered K/V immediately
  before RPA. Internal Pallas math is downstream of that gate.
- Claim ceiling:
  `ATTEMPT18_TARGET_REPORTED /
  ATTEMPT18_E0_RETURN_NOT_ADMITTED /
  FIRST_RED_LOCALIZED_FROM_D3E /
  APC_NUMERICAL_FIX_NOT_YET_IMPLEMENTED /
  NUMERICAL_REPAIR_NOT_AUTHORIZED`.
- Updated: 2026-08-29.

