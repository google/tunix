# DeepSWE K23 Gradient Accumulation Contract Mismatch Incident Report

## 1. Incident Overview
- **Run Identifier**: `canon-p58-ds4b-zero-hp-full-k23`
- **Hardware Topology**: 128 TPU v5p (32 Hosts), Disaggregated 64 TPU Rollout (`dp=8, tp=8`) + 64 TPU Trainer (`dp=8, tp=8`)
- **Step Reached**: Step 0 Rollout (128 trajectories, 8 solved, solve rate 6.25%, 47 nonzero advantage samples), Pre-alignment Gate (393,135 tokens, 0B diff, 100% PASS), VJP Backward Pullback (Layers 35 -> 0 PASS), Trainer Mesh DP8 Reverse Reduce Group 1/16 PASS (`replicas_exact=1`).
- **Termination Reason**: `ValueError: segmented update accumulation changed: 8 != 16` raised at `_validate_precomputed_gradient_contract` in `tunix/sft/peft_trainer.py:L924`.

## 2. Root Cause Analysis
1. In DeepSWE (P34), 16 reverse groups are streamed into `peft_trainer.accumulate_precomputed_scaled_gradient_microbatch`.
2. `_precomputed_expected_microbatches(os.environ)` resolves `expected_steps = 16`.
3. In `examples/deepswe/train_deepswe_nb.py`:
   - `train_trajectory_micro_batch_size` was assigned `p34.local_trajectories` (`16`).
   - In `tunix/rl/rl_cluster.py:L184`, `gradient_accumulation_steps = trajectory_mini_batch_size (128) // train_trajectory_micro_batch_size (16) = 8`.
   - `actor_trainer.config.gradient_accumulation_steps` was thus set to `8`.
4. When streaming the first microbatch, `peft_trainer._validate_precomputed_gradient_contract()` checked `self.config.gradient_accumulation_steps == expected_steps` (`8 != 16`), triggering the mismatch exception.

## 3. Resolution
- In `examples/deepswe/train_deepswe_nb.py`, set:
  `train_trajectory_micro_batch_size = p34.global_trajectories // p34.local_trajectories` (`128 // 16 = 8` global trajectories per microstep, which equals 1 trajectory per DP rank on DP=8).
- This produces `gradient_accumulation_steps = 128 // 8 = 16`, satisfying the `16 == 16` contract.
