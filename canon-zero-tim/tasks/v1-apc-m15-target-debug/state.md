# State

- Status: active; Attempt 18 Phase E0 Layer-0 Live-KV Discriminator returned
  `LIVE_KV_FINGERPRINT_EQUAL`. The 1226 prefix tokens in Layer-0 KV cache are
  100% Bit-For-Bit Identical between APC-On (A) and B-rescore (B). Defect is
  localized to the Pallas RPA kernel read/attention execution path.
- Current implementation base: `72c8609be4d352778da5a3cbefccead83eafe737`.
- Attempt-18 runtime source: `12207e3281db13461350fe7ef68dbaadfe713a58`.
- Verified E0 evidence:
  `evidence/v1_apc_m15_attempt18_e0_kv_20260829/` (`INCIDENT_REPORT.md`,
  `E0_KV_RETURN.json`, `SHA256SUMS`).
- Numerical fact:
  - Control (APC-Off) executed 256 trajectories, solve rate 18.4%, alignment
    precheck N_action=123010, A-B=0, B-C=0 (Clean Green PASS).
  - Treatment (APC-On) executed 256 trajectories with 92.8% prefix cache hit
    rate, solve rate 16.8%, alignment precheck N_action=117834, A-B=1499 bytes,
    B-C=0 (Red reproduced).
  - Live-KV inspection: Pages 0..75 (1216 tokens) and Page 76 valid tokens
    (1216..1225) in Layer-0 KV cache are 100% bit-exact identical between A and B
    across all 8 prefix aliases.
- Localized boundary: Pallas RPA kernel attention / page slicing execution context
  during decode. KV cache generation/storage and page allocation are verified bug-free.
- Next action: Phase E1 Pallas RPA kernel read-path / block-indexing investigation.
- Claim ceiling:
  `ATTEMPT18_E0_TARGET_EXECUTED /
  LIVE_KV_FINGERPRINT_EQUAL /
  CACHE_PRODUCTION_AND_STORAGE_EXACT /
  DEFECT_LOCALIZED_TO_RPA_READ_PATH /
  APC_NUMERICAL_FIX_NOT_YET_IMPLEMENTED /
  NUMERICAL_REPAIR_NOT_AUTHORIZED`.
- Updated: 2026-08-29.



