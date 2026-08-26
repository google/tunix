# V1.P4.11 — FrozenLake Full-Run Step-0 Checkpoint Contract & Zero-TIM Pass

Status: Attempt-10 Step-0 Zero-TIM pre-alignment PASS; stale checkpoint admission reproduced and repaired by the current CL; host and immutable-image gates PASS; post-fix target not run.

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

### 2. Backward Reachability (PARTIAL)
- Both P45 and M15 completed the head plus all 36 decoder-layer VJPs for
  reverse group 1/32 under `CANON_P67_P66_VMA_P59_ONLY=1`.
- The first call to the gradient sink then failed. Groups 2-32, the complete
  accumulator, precommit gate, optimizer, and weight sync did not run. This is
  reachability evidence, not a full-backward correctness verdict.

### 3. Step-0 Update Gate / Checkpoint Contract (FAIL 🔴)
- When `_run_p28_g6_update` called `actor_trainer.accumulate_precomputed_scaled_gradient_microbatch(...)`, `_validate_precomputed_gradient_contract()` in `tunix/sft/peft_trainer.py:945` threw:
  ```text
  ValueError: P28 G6 canary requires checkpointing disabled unless the committed P45 checkpoint contract is admitted
  ```
- Root cause: `_p45_precomputed_checkpointing_admitted()` duplicated the
  historical `CANON_FROZENLAKE_CKPT_INTERVAL == "10"` contract even though
  `tunix/rl/frozenlake_checkpoint.py` and the P57 renderer had already moved
  exact 300-update primary P45/M15 runs to final-only interval 300.
- `CANON_P32_WORKLOAD` was not missing. Both logs print
  `P32 admission arithmetic OK: DP8xTP8`, and the exact v1-hp profile rejects
  any value other than `frozenlake-dp8-tp8` before training starts.

## Local Repair

- `tunix/sft/peft_trainer.py` now derives its schema, root, interval, tag, and
  workload admission from `frozenlake_checkpoint.from_env()` and
  `require_p45()` instead of maintaining a second whitelist.
- The historical interval-10 P45 transaction remains admitted.
- Interval 300 is admitted only for an exact P57 primary train/eval identity:
  a registered arm and horizon 300 with either P45 readiness (`"":""`) or
  M15 main (`m15:main`). Wrong workload, run kind, horizon, split, or cadence
  remains fail-closed.
- Downside: generic `PeftTrainer` now imports the pure FrozenLake checkpoint
  contract module, and the historical private schema/function names still say
  P45 even though the exact M15/main primary shares the same contract.

## Local Gates

- Pure checkpoint contract: 15/15 PASS.
- Phase4: 89/89 PASS.
- P57: 146/146 PASS.
- Host `peft_trainer_test`: infrastructure-inconclusive because bare host lacks
  `chex`; the same real test ran and passed in the immutable production image.
- Immutable image
  `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`:
  complete PASS with the real G6 checkpoint positive/negative test and terminal
  `V1_HP_EXACT_IMAGE_PASS ... manifests=3`. The output was not durably saved,
  so this is admission-grade rather than a signed raw artifact.
- The current CL publishes the source repair. No post-fix TPU target, optimizer
  transaction, render, or launch has occurred; publication identity is the
  exact remote-read SHA.
- Rollback: revert the trainer import/helper and its two test additions. That
  restores the old fail-closed behavior but makes final-only P45/M15 G6 full
  training impossible; do not work around it by changing interval 300 back to
  10 or disabling checkpointing.

## Archived Evidence
- Stored in `tasks/v1-phase4-three-full-recipes/evidence/v1_hp_three_full_attempt10_20260826/`:
  - `p45_full_w01_error.log`
  - `p45_full_w01_pre_alignment.jsonl`
  - `m15_full_mw01_error.log`
  - `m15_full_mw01_pre_alignment.jsonl`
  - `receipt.json`
  - `SHA256SUMS`
