# M15 E0 KV3 Attempt 19 Incident Report

**Date**: 2026-08-30
**Workloads**:
- Control (APC-Off): `canon-v1-apc-m15-off-k01-d93d2729` (64 TPU v5p)
- Treatment (APC-On): `canon-v1-apc-m15-on-k01-d93d2729` (64 TPU v5p)
**Commit Base**: `d93d2729c5f036506fe754b929d42b142177a9b7`

---

## 1. Executive Summary

Attempt 19 was launched as the first three-round durability carrier (`observer=kv3`, `CANON_P38_DURABILITY_PROFILE=m15-e0-kv-v1`).
Both jobs were successfully scheduled and admitted on 64 TPU v5p each, but **both failed prematurely without completing the required 3 rounds**:

1. **Treatment Arm (APC-On)**: Failed at **Round 0 Sealing Stage** (`02:43 UTC`).
   - Rollout completed with 92.8% prefix cache hit rate and $A-B = 366$ differing bytes ($B-C = 0$).
   - Raw classifier input archive (102.1 MB) was uploaded to GCS.
   - Live worker sealer failed during `classify_p38_kv_observer.py` with `ObserverError: no paired observer candidate joined a red capsule row`.
   - Never entered Round 1 or Round 2.

2. **Control Arm (APC-Off)**: Failed at **Round 1 Staging Stage** (`02:58 UTC`).
   - Round 0 completed successfully, passed classifier, and sealed `ROUND_COMPLETE.json` to GCS.
   - Round 1 rollout completed with $A-B = 0$ ($B-C = 0$), but generated **0 KV observer records** because the dataset advanced to Step 1 whose prompt histories did not match the single static `_P38_KV_OBSERVER_TARGET_PREFIX_SHA256`.
   - `stage_m15_e0_kv_round.py` raised `E0RoundError: round 1 requires 16 KV records`.
   - Never entered Round 2.

---

## 2. Root Cause Analysis

### Issue A: Control Arm — Static Target Prefix Filter in Multi-Step Dataset (`stage_m15_e0_kv_round.py`)
- **Mechanism**:
  In `tpu_runner_p21_l30.py` (Patch 35/36):
  ```python
  if (_p38_token_history_sha256(token_ids) != _P38_KV_OBSERVER_TARGET_PREFIX_SHA256):
      continue
  ```
  `_P38_KV_OBSERVER_TARGET_PREFIX_SHA256` is configured to a single prompt prefix hash from the D3e diagnostic seed.
  In FrozenLake M15, the training loop advances data per step (`step=0`, `step=1`, `step=2`).
  - In `step=0` (Round 0), the batch contained the targeted prompt, emitting 8 A + 8 B records.
  - In `step=1` (Round 1), the batch advanced to new trajectory prompts whose token prefix hashes differed from the static target. Zero records were captured in Round 1.
  - When `stage_m15_e0_kv_round.py` asserted `len(selected) == 16` for Round 1, it raised `E0RoundError: round 1 requires 16 KV records`.

### Issue B: Treatment Arm — Strict Red-Candidate Join Assertion (`classify_p38_kv_observer.py`)
- **Mechanism**:
  In `classify_p38_kv_observer.py:475`:
  ```python
  def _join_red_candidates(candidates, red_rows, ...):
      ...
      if not joined:
          raise ObserverError("no paired observer candidate joined a red capsule row")
  ```
  In Round 0 of Attempt 19, APC-On produced $A-B = 366$ differing bytes.
  However, the 8 sampled/observed aliases in that rollout did not coincide with the specific sequence rows that suffered the 366-byte divergence (or diverged at later positions beyond prefix token 1226).
  The hard assertion in `_join_red_candidates` failed closed, causing an `AlignmentGateError` before the round could be sealed and ACKed.

---

## 3. Evidence Status

- **Control (Off)**: GCS retains complete Round 0 evidence (`rounds/000000/`); Round 1 and Round 2 are missing.
- **Treatment (On)**: GCS retains complete Round 0 classifier input archive (`rounds/000000/classifier-input/CLASSIFIER_INPUT_ARCHIVE.tar`, 102.1 MB); Round 0 classification and Round 1/2 are missing.
- Root 4-file bundle is absent on both arms due to the 3-round fail-closed contract.
