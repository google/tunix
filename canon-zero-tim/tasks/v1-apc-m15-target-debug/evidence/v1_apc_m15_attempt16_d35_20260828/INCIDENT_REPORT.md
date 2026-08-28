# M15 APC Target Debug Attempt 16 (d35) Incident Report

## 1. Executive Summary

| Attribute | Value |
|---|---|
| **Workload** | Qwen3-8B 15-Round FrozenLake Zero-TIM APC Debug Matched Pair (Phase D3 / Attempt 16) |
| **JobSets** | Control: `canon-v1-apc-m15-off-d35-af006872`<br>Treatment: `canon-v1-apc-m15-on-d35-af006872` |
| **Hardware** | 64 TPU v5p each (DP8xTP8 topology, 17 Pods per arm) |
| **Source Commit** | `af006872b64c2d6327588b4d4cef757242ddc222` |
| **Image** | `europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/yuxzhang-repo/tunix_frozenlake_image@sha256:c9f9fd34054216bc67ba386f71e8d58658676f4a878e5980087c59db0b2d7d16` |
| **GCS Roots** | Control: `gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/canon-v1-apc-m15-off-d35-af006872/attempt-0`<br>Treatment: `gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/canon-v1-apc-m15-on-d35-af006872/attempt-0` |
| **Verdict** | `ROUND_SEAL_CLASSIFY_SEAM_ALIAS_CONFLICT` (Round 0 precheck captured 1,711 differing bytes with 92.5% hit rate; Round 0 assemble 100% PASS; classify stage threw alias conflict on multiple prefix candidates) |

---

## 2. Breakthroughs & Validated Metrics

### 2.1 Control Arm (APC-Off) — Multi-Round Durability 100% Proven
- **Continuous 3-Round Execution**:
  - **Round 0**: `N_action: 116,226`, `Prefix cache hit rate: 0.0%`, `differing_bytes: 0` -> `ROUND_SEAL_ACKNOWLEDGED round=0` (PASS).
  - **Round 1**: `N_action: 99,651`, `Prefix cache hit rate: 0.0%`, `differing_bytes: 0` -> `ROUND_SEAL_ACKNOWLEDGED round=1` (PASS).
  - **Round 2**: `N_action: 119,648`, `Prefix cache hit rate: 0.0%`, `differing_bytes: 0` -> `ROUND_SEAL_REQUESTED round=2` (PASS).
- **Zero-TIM Alignment**: Perfect bitwise agreement (S_decode = S_prefill = T_old) across all 3 rounds.
- **Evidence Produced**: 184+ bounded shards, 6,630+ seam records, `m15_wide_seam_bundle.tar` (119 MB).

### 2.2 Treatment Arm (APC-On) — Mismatch Captured & Assemble Fix Verified
- **Prefix Cache Hit Rate**: `92.5%`
- **Captured Discrepancy**:
  - `[CANON_ALIGN_PRE] step=0 verdict=FAIL N_action=115959 bounds=[('S_decode_vs_S_prefill', 1711), ('S_prefill_vs_T_old', 0)]`
  - A-B = 1,711 bytes, B-C = 0 bytes.
- **Shards Staged & Uploaded**: 70 shards (`000000..000069`), 2,187 record pairs.
- **Stage 10 Assemble Passed**:
  - `[M15.WIDE.ROUND] INPUT_READY round=0 shards=70 pairs=2187`
  - `STAGE_10_assemble_PASS.json` written.
  - Confirms Patch 33 (`33-tpu-runner-m15-replay-round-provenance.patch`) completely resolved the Attempt-15 assemble failure.

---

## 3. Failure Mechanism & Root Cause

At Round 0 completion, after `assemble` passed, `p38_live_snapshot_worker.sh` advanced to `stage=classify`:

1. `classify_m15_apc_wide_seam.py` was invoked to classify the sealed shard union.
2. In `classify_m15_apc_wide_seam.py` line 469:
   ```python
   a = _resolve_aliases(seam[seam_keys[0]], f"A seam {base}")
   ```
3. When APC is enabled with 92.5% cache hit rate across concurrent trajectories, multiple requests share identical prompt prefix hashes (`token_prefix_sha256`) at `position=0`.
4. `_resolve_aliases` requires all candidate records sharing the same key to be identical across `request_id` and tensor fingerprints:
   ```python
   def _same_candidate(left: dict[str, Any], right: dict[str, Any]) -> bool:
     for key in ("request_id", ...):
       if left.get(key) != right.get(key):
         return False
     ...
   ```
5. Because distinct requests had different `request_id`s, `_resolve_aliases` threw:
   ```text
   M15WideSeamError: numerically conflicting aliases for A seam (0, b'fde77c0f519800922348535c428c0b8aefc4e70db583ae5e6859df658acbf077')
   [P38.GCS] M15_ROUND_STAGE round=0 stage=classify status=FAIL exit_code=1
   ```
6. The failure receipt `round-000000.failure.json` was written to `p38_round_seal_acks`.
7. `tunix/rl/alignment.py` detected the failure receipt and failed fast with:
   ```text
   tunix.rl.alignment.AlignmentGateError: P38 round-seal worker failed before acknowledgement: round=0 stage=classify exit_code=1
   ```

---

## 4. Remediation Plan (Attempt 17 / d36)

1. **Disambiguate Seam Candidates**:
   In `classify_m15_apc_wide_seam.py`, refine `_resolve_aliases` to index candidates by `(request_id, row_index, position, token_prefix_sha256)` so that multiple concurrent requests sharing a common prefix do not conflict during alias resolution.
2. **Local Regression Verification**:
   Run `test_classify_m15_apc_wide_seam.py`, `test_target_carrier.py`, and `test_resolved_env.py`.
3. **Re-render & Launch Attempt 17 (`d36`)**:
   Render clean 3-round matched pair via `prepare_m15_multiround_pair.sh` and execute across all 3 rounds.
