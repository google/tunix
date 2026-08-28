# M15 APC Target Debug Attempt 15 (d34) Incident Report

## 1. Executive Summary

| Attribute | Value |
|---|---|
| **Workload** | Qwen3-8B 15-Round FrozenLake Zero-TIM APC Debug Matched Pair (Phase D3) |
| **JobSets** | Control: `canon-v1-apc-m15-off-d34-57d9ab8e`<br>Treatment: `canon-v1-apc-m15-on-d34-57d9ab8e` |
| **Hardware** | 64 TPU v5p each (DP8xTP8 topology, 17 Pods per arm) |
| **Source Commit** | `57d9ab8e9282da41d06e3e57140e791e847c23bc` |
| **Image** | `europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/yuxzhang-repo/tunix_frozenlake_image@sha256:c9f9fd34054216bc67ba386f71e8d58658676f4a878e5980087c59db0b2d7d16` |
| **GCS Roots** | Control: `gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/canon-v1-apc-m15-off-d34-57d9ab8e/attempt-0`<br>Treatment: `gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/canon-v1-apc-m15-on-d34-57d9ab8e/attempt-0` |
| **Verdict** | `ROUND_SEAL_ASSEMBLE_REPLAY_ROUND_DRIFT (Round 0 prefill rescore 100% PASS, assemble exit 2)` |

---

## 2. Breakthroughs & Validated Metrics

### 2.1 Complete Round 0 Execution & Alignment Pass
Both arms executed complete Round 0 long-horizon multi-turn rollouts, forward/backward Pallas hot paths, and prefill rescoring with exact numerical alignment:

- **Control (APC-Off)**:
  - `Prefix cache hit rate`: `0.0%`
  - `N_action`: `120,889`
  - `S_decode_vs_S_prefill`: `differing_bytes=0`, `element_fraction=0.0`
  - `Pre-alignment Verdict`: `PASS`
  - `Shards Staged & Uploaded`: 85 shards (`000000..000084`) verified with remote checksums.

- **Treatment (APC-On)**:
  - `Prefix cache hit rate`: `93.2%`
  - `N_action`: `130,468`
  - `S_decode_vs_S_prefill`: `differing_bytes=0`, `element_fraction=0.0`
  - `Pre-alignment Verdict`: `PASS`
  - `Shards Staged & Uploaded`: 72 shards (`000000..000071`) verified with remote checksums.

---

## 3. Failure Mechanism & Root Cause

At the completion of Round 0 precheck, the learner issued `_seal_p38_diagnostic_round(round_index=0)` and awaited background ACK.

1. `p38_live_snapshot_worker.sh` invoked `assemble_m15_wide_round.py` to compile sealed shards and metadata.
2. `assemble_m15_wide_round.py` lines 53-58 read `m15_replay_envelope.jsonl`:
   ```python
   record_round = int(record.get("diagnostic_round", -1))
   _require(0 <= record_round < 8, f"replay round is invalid at line {line_number}")
   ```
3. In `patches/tpu_inference/26-tpu-runner-m15-replay-envelope.patch`, the serialized dictionary for `m15-apc-serving-envelope-v1` records omitted the `"diagnostic_round"` key.
4. `record.get("diagnostic_round", -1)` evaluated to `-1`, triggering:
   ```text
   [M15.WIDE.ROUND] RED replay round is invalid at line 1
   [P38.GCS] M15_ROUND_STAGE round=0 stage=assemble status=FAIL exit_code=2
   ```
5. `tunix/rl/alignment.py` detected the failure receipt and failed fast with:
   ```text
   tunix.rl.alignment.AlignmentGateError: P38 round-seal worker failed before acknowledgement: round=0 stage=assemble exit_code=2
   ```

---

## 4. Remediation Plan (Attempt 16 / d35)

1. **Patch Update**:
   In `patches/tpu_inference/26-tpu-runner-m15-replay-envelope.patch`, add `"diagnostic_round": int(_p38_seam_round())` (or `diagnostic_round`) to the envelope record dictionary.
2. **Unit Test & Regression Verification**:
   Run `test_target_carrier.py`, `test_m15_wide_durability.py`, and `test_classify_m15_apc_wide_seam.py`.
3. **Re-render & Launch Attempt 16 (`d35`)**:
   Launch matched pair on 64 TPU to proceed across all 3 rounds.
