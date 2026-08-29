# State

- Status: active; `ATTEMPT18_E0_RETURN_NOT_ADMITTED`. Commit
  `ff33dcd200a4577927ac4917839a0b86bac42e7a` reports an Attempt-18 target
  execution and `LIVE_KV_FINGERPRINT_EQUAL`, but the committed return is not
  the official wrapper output and cannot support that mechanism verdict yet.
- Current incoming evidence base:
  `ff33dcd200a4577927ac4917839a0b86bac42e7a`.
- Last published E0 carrier hardening:
  `72c8609bce5185b87ea9f7f1850afadf3974cdd2`. The former state value
  `72c8609be4d352778da5a3cbefccead83eafe737` is not a Git object and must not
  be used.
- Attempt-18 runtime source: `12207e3281db13461350fe7ef68dbaadfe713a58`.
- Incoming evidence:
  `evidence/v1_apc_m15_attempt18_e0_kv_20260829/`. Its two-member manifest
  verifies and has SHA256
  `9eabd0317cb32b29655c841beff35974c07fec93767cf4e87084071141d27917`,
  but both official classifier JSONs and terminal receipt are absent; both
  classifier digests are invalid 32-character values; control classification
  and treatment request binding are incomplete.
- Reported, not yet admitted numerical values:
  - Control (APC-Off) executed 256 trajectories, solve rate 18.4%, alignment
    precheck N_action=123010, A-B=0, B-C=0 (Clean Green PASS).
  - Treatment (APC-On) executed 256 trajectories with 92.8% prefix cache hit
    rate, solve rate 16.8%, alignment precheck N_action=117834, A-B=1499 bytes,
    B-C=0 (Red reproduced).
  - Incoming report says all eight 1226-token Layer-0 diagnostic fingerprints
    are equal. This is not proof that all KV bytes, writes, or allocation are
    exact.
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

