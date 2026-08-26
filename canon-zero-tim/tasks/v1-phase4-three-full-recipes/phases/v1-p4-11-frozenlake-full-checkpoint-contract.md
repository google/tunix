# V1.P4.11 — FrozenLake Full-Run Step-0 Checkpoint Contract & Zero-TIM Pass

Status: Step-0 Zero-TIM Pre-Alignment PASS (0/0 differing bytes across P45 and M15); Step-0 update blocked by checkpoint interval configuration contract.

## Overview

Attempt 10 full training runs for FrozenLake P45 (`canon-p57-fl-zero-f45w01-8eb65480`) and M15 (`canon-p57-fl-zero-m15-mw01-8eb65480`) were launched from commit `8eb65480d3705d96ab282799ad5a6c1901596248` on 64 TPUs (DP8xTP8) each, alongside the active GSM8K full run (`canon-v1hp-gsm8k-g11-c2833eea`).

## Findings & Evidence

### 1. Step-0 Zero-TIM Pre-Alignment (PASS 🟢)
- **FrozenLake P45 (`canon-p57-fl-zero-f45w01-8eb65480`)**:
  - $N_{\\text{action}} = 48,753$ action tokens.
  - S_decode vs S_prefill differing bytes: 0 (0.0%).
  - S_prefill vs T_old differing bytes: 0 (0.0%).
  - Masked SHA-256: `ab02e5670435c3b4ee0b1d2db57dfe19cbcdb9c6a64e1bbd0198248981dedd20` byte-exact across all 3 terms.
  - Verdict: `PASS`.
- **FrozenLake M15 (`canon-p57-fl-zero-m15-mw01-8eb65480`)**:
  - $N_{\\text{action}} = 122,162$ action tokens.
  - S_decode vs S_prefill differing bytes: 0 (0.0%).
  - S_prefill vs T_old differing bytes: 0 (0.0%).
  - Masked SHA-256: `75f26cce92df6cef101e188acf3565caa16354ae52b8f95c8c7635c4ddd2d9c9` byte-exact across all 3 terms.
  - Verdict: `PASS`.

### 2. Backward Pass Execution (PASS 🟢)
- Both P45 and M15 completed all 36 decoder layers of P59 rank-parallel backward VJPs under `CANON_P67_P66_VMA_P59_ONLY=1`.

### 3. Step-0 Update Gate / Checkpoint Contract (FAIL 🔴)
- When `_run_p28_g6_update` called `actor_trainer.accumulate_precomputed_scaled_gradient_microbatch(...)`, `_validate_precomputed_gradient_contract()` in `tunix/sft/peft_trainer.py:945` threw:
  ```text
  ValueError: P28 G6 canary requires checkpointing disabled unless the committed P45 checkpoint contract is admitted
  ```
- Root cause: `_p45_precomputed_checkpointing_admitted()` asserts `CANON_FROZENLAKE_CKPT_INTERVAL == "10"` and `CANON_P32_WORKLOAD == "frozenlake-dp8-tp8"`, whereas `qwen3-8b-dp8-tp8-frozenlake-v1-hp.env` passed `CANON_FROZENLAKE_CKPT_INTERVAL="300"` and lacked `CANON_P32_WORKLOAD`.

## Archived Evidence
- Stored in `tasks/v1-phase4-three-full-recipes/evidence/v1_hp_three_full_attempt10_20260826/`:
  - `p45_full_w01_error.log`
  - `p45_full_w01_pre_alignment.jsonl`
  - `m15_full_mw01_error.log`
  - `m15_full_mw01_pre_alignment.jsonl`
  - `receipt.json`
  - `SHA256SUMS`
