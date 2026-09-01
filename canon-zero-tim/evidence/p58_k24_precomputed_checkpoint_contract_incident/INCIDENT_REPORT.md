# DeepSWE K24 Precomputed Gradient Checkpoint Contract Incident Report

## 1. Incident Overview
- **Run Identifier**: `canon-p58-ds4b-zero-hp-full-k24`
- **Hardware Topology**: 128 TPU v5p (32 Hosts), Disaggregated 64 TPU Rollout (`dp=8, tp=8`) + 64 TPU Trainer (`dp=8, tp=8`)
- **Step Reached**: 
  - Step 0 Rollout 100% complete (128 trajectories, 6 solved, solve rate 4.69%, 46 nonzero advantage samples).
  - Zero-TIM Pre-alignment Gate (388,328 action tokens, 0B diff, `logp_diff=(0,0)`, `pearson=1.000`, 100% PASS).
  - Weight Attestation (64 Devices / 398 leaf nodes, 0 diff, `verdict=PASS`).
  - 16 Forward Groups (128 trajectories, `forward_group_done group=1..16` PASS).
  - Pallas VJP Backward Pullback (Layers 35 -> 0 and Embed Layer PASS).
  - Trainer Mesh DP8 Reverse Reduce Group 1/16 (`reverse_group_done group=1/16`, `replicas_exact=1` PASS).
- **Termination Reason**: `ValueError: P28 G6 canary requires checkpointing disabled unless the committed P45 checkpoint contract is admitted` raised at `_validate_precomputed_gradient_contract` in `tunix/sft/peft_trainer.py:L935`.

## 2. Root Cause Analysis
1. In `tunix/sft/peft_trainer.py:L928-L938`, the P28 G6 precomputed gradient accumulation contract mandates that checkpointing is disabled (`checkpoint_root_directory is None`) unless an explicit precomputed checkpoint contract is admitted (currently only `_P45_PRECOMPUTED_CHECKPOINT_CONTRACT` for FrozenLake P45).
2. In `canon-zero-tim/cluster/render_p58_deepswe_tim.py`, `_command` calls `render_p34_jobset._command("three-update", run_root=run_root, whitelist=whitelist)`.
3. `render_p34_jobset._command` appends `f"--ckpt_dir={run_root}/checkpoints"`, which sets `training_config.checkpoint_root_directory = CKPT_DIR` in `examples/deepswe/train_deepswe_nb.py:L1849`.
4. Because DeepSWE has no admitted `_P45_PRECOMPUTED_CHECKPOINT_CONTRACT` schema, `peft_trainer._validate_precomputed_gradient_contract` rejects the active `checkpoint_root_directory`.

## 3. Resolution
- In `canon-zero-tim/cluster/render_p58_deepswe_tim.py`, replace `--ckpt_dir={run_root}/checkpoints` with `--ckpt_dir=none` in `_command` (or override `CKPT_DIR = None` when running segmented DeepSWE training), disabling checkpointing in compliance with the DeepSWE `checkpoint=off` training contract.
