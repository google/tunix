# State

- Status: active; `OFFICIAL_RETURN_PROVENANCE_FAIL`. Published commit
  `971bb2281417ecb6e33cfa6bb68a422f7fd24f00` contains a locally
  manifest-valid four-file Attempt-18 E0 package, but its classifier source
  identity and payload provenance cannot come from the pinned runtime source.
  `LIVE_KV_FINGERPRINT_EQUAL` is not admitted.
- Rejected package:
  `evidence/v1_apc_m15_attempt18_e0_kv_20260829/`; `SHA256SUMS` SHA256
  `ce762783e6b2f1a6fae37190f3af6e96baa39302931d29081c1d93146b7c9475`.
  It is an immutable rejected snapshot and must not be overwritten.
- Rejection audit:
  `evidence/v1_apc_m15_attempt18_e0_return_rejection_20260829/`;
  `REJECTION_REPORT.json` SHA256
  `92b704d5e6cb9ed0dd90e6d2b8648ee7980d7643218bb176d146fc40b1e5b9fa`.
  The overwritten/deleted ff33dcd2 inputs are preserved byte-for-byte under
  `evidence/v1_apc_m15_attempt18_e0_incoming_rejected_ff33dcd2_20260829/`.
- Attempt-18 runtime source:
  `12207e3281db13461350fe7ef68dbaadfe713a58`. Its official classifier is
  `classify_p38_kv_observer.py`, SHA256
  `99cc7d9c50777a9be182e2edd33a3cdca3daabaa396c019e4925e0ac531049f6`.
  The rejected package instead names a different classifier, collapses
  unrelated record/manifest digests, truncates runtime fields, uses absolute
  temporary paths, and lacks a preserved raw terminal receipt.
- Reported but unadmitted target values: control APC-off `N_action=123010`,
  A-B=0, B-C=0; treatment APC-on `N_action=117834`, A-B=1499 bytes / 88
  elements, B-C=0, 92.8% cache hits. These values do not admit a mechanism
  verdict.
- Last admitted numerical boundary: D3e, Layer 0
  `k_post_rope -> rpa_output`, shape `[2048,1,15,8]`, source row 217 /
  source position 1225 / A call 83. The 1707700e NumPy online-softmax probe is
  a toy hypothesis example, not target causality or a repair.
- Implementation: provenance admission only. The reviewer pins the exact
  runtime/classifier identity, complete runtime-emitted comparison/red-join
  fields, distinct record/arm provenance, basename-only paths, exact claim
  ceiling, and mandatory CLI raw log. A/B/C, APC behavior, numerical code,
  production flags, backward, and optimizer are unchanged.
- Validation: HOST PASS — task discovery 187/187; return intake 14/14; E0
  admission 9/9; V1 CPU 91/91; P3 prefix-cache 31/31; P38 persistence; flag
  audit 398/398; Python/Bash syntax and `git diff --check`. Raw log:
  `/tmp/m15-e0r-provenance-hardening-971bb228-retry2-20260829.log`, SHA256
  `f11ab8b9bf137f7f7ca39a801fe06b6da6298b7b558fe817ea2f503f7f74a4e4`.
  Official pinned exact-image, real GCS recovery, TPU, and Kubernetes are NOT
  RUN.
- Current gate: after separately approved commit/push, a clean exact-SHA
  worktree must pass the separately approved official pinned exact-image
  aggregate (`m15_e0=30`). Then a bucket-capable agent may, under a separate
  explicit GCS-read approval, run the checked-in recovery wrapper against the
  preserved `e01` render and a fresh local output path. No TPU target rerun is
  currently requested.
- Claim ceiling:
  `ATTEMPT18_E0_RETURN_PROVENANCE_FAIL /
  TARGET_RESULT_NOT_ADMITTED /
  FIRST_RED_LOCALIZED_FROM_D3E /
  PHASE_E_CLOSED /
  APC_NUMERICAL_FIX_NOT_IMPLEMENTED /
  NUMERICAL_REPAIR_NOT_AUTHORIZED`.
- Updated: 2026-08-29.
