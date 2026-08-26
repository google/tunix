# P58.14 — DeepSWE 4B 128-TPU Step-0 Device Sharding Mismatch

Status: Step-0 Rollout & 36-layer Pallas VJP PASS; Step-0 trainer per-token logprob calculation blocked by 128-TPU device mesh mismatch.

## Overview

DeepSWE Zero-HP Full training (`canon-p58-ds4b-zero-hp-full-p58z03`) was launched on 128 TPU chips (2 slices of 64 chips: 16 DP x 8 TP) with Qwen3-4B-Instruct from commit `8eb65480d3705d96ab282799ad5a6c1901596248`.

## Findings & Evidence

1. **Step-0 Rollout & Fixed LM Head**:
   - Completed 128 trajectory generation rollouts in one wave.
   - P38 Fixed LM Head successfully admitted `semantic_M=2048` for Qwen3-4B TP8 (`(2560, 8)`).
   - Completed all 36 decoder layers of Pallas SwiGLU, RMSNorm, and projection VJP passes.

2. **Step-0 Trainer Logprob JIT Device Collision**:
   - In `get_actor_per_token_logps` -> `compute_per_token_logps`:
     ```text
     ValueError: Received incompatible devices for jitted computation. Got argument state['embedder']['input_embedding'].value of compute_per_token_logps with shape float32[151936,2560] and with device ids [2, 3, 18, 19, 34, 35, 50, 51, 66, 67, 82, 83, 98, 99, 114, 115, ...] on platform TPU and sharding_constraint inside jit with device ids [0, 4, 8, 12, 1, 5, 9, 13, 16, 20, 24, 28, ...] on platform TPU at /app/tunix/rl/canonical_qwen3_adapter.py:483:13 (_safe_sharding_constraint)
     ```
   - Root cause: On 128 TPUs across 2 physical slices, the actor trainer state mesh device ordering differed from the JIT compilation sharding constraint device slice ordering.

## Archived Evidence
- Stored in `tasks/p58-deepswe-native-zero-comparison/evidence/p58z03_device_sharding_error/`:
  - `run.log`
  - `p38_fixed_lm_head_receipts.json`
  - `SHA256SUMS`
