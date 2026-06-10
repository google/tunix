import os
import numpy as np
from transformers import AutoTokenizer

def main():
  dump_path = "/mnt/disks/linchai-data/last_turn_debug.npz/last_turn_debug.npz"
  if not os.path.exists(dump_path):
    print(f"Error: Dump file {dump_path} does not exist. Run the training script first on this machine!")
    return

  data = np.load(dump_path)
  prompt_ids = data["prompt_ids"]
  completion_ids = data["completion_ids"]
  rollout_logps = data["rollout_logps"]
  trainer_logps = data["trainer_logps"]
  masks = data["masks"]
  pad_value = int(data["pad_value"])

  print("=== Dump Info ===")
  print(f"prompt_ids shape: {prompt_ids.shape}")
  print(f"completion_ids shape: {completion_ids.shape}")
  print(f"rollout_logps shape: {rollout_logps.shape}")
  print(f"trainer_logps shape: {trainer_logps.shape}")
  print(f"masks shape: {masks.shape}")
  print(f"pad_value: {pad_value}")

  # Load tokenizer
  try:
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")
  except Exception as e:
    print(f"Warning: Failed to load tokenizer: {e}. Decoding will show raw IDs.")
    tokenizer = None

  # Look at the first item in the batch
  b_idx = 0
  p_ids = prompt_ids[b_idx]
  c_ids = completion_ids[b_idx]
  r_logps = rollout_logps[b_idx]
  t_logps = trainer_logps[b_idx]
  m = masks[b_idx]

  # Find active tokens (excluding padding)
  # JAX prompt is left-padded, completion is right-padded
  p_active_start = np.where(p_ids != pad_value)[0][0] if np.any(p_ids != pad_value) else 0
  c_active_end = np.where(c_ids != pad_value)[0][-1] + 1 if np.any(c_ids != pad_value) else len(c_ids)

  active_prompt = p_ids[p_active_start:]
  active_completion = c_ids[:c_active_end]
  full_active_seq = np.concatenate([active_prompt, active_completion], axis=0)

  print("\n=== Active Token Counts ===")
  print(f"Prompt active start: {p_active_start} (unpadded length: {len(active_prompt)})")
  print(f"Completion active end: {c_active_end} (unpadded length: {len(active_completion)})")
  print(f"Full active sequence length: {len(full_active_seq)}")

  # Filter masks to active indices
  active_mask = m[p_active_start : len(p_ids) + c_active_end]
  active_rollout_logps = r_logps[p_active_start : len(p_ids) + c_active_end]
  active_trainer_logps = t_logps[p_active_start : len(p_ids) + c_active_end]

  print(f"active_mask shape: {active_mask.shape}")
  print(f"active_rollout_logps shape: {active_rollout_logps.shape}")
  print(f"active_trainer_logps shape: {active_trainer_logps.shape}")

  # Mask stats
  num_active_compared = int(np.sum(active_mask > 0))
  print(f"Number of compared (non-pad, active) tokens: {num_active_compared}")

  # Overall diffs on active compared tokens
  active_diff = np.abs(active_rollout_logps - active_trainer_logps)
  diff_mean = np.sum(active_diff * (active_mask > 0)) / max(num_active_compared, 1)
  diff_max = np.max(np.where(active_mask > 0, active_diff, 0.0))
  print(f"Filtered Mean Logp Diff: {diff_mean:.6f}")
  print(f"Filtered Max Logp Diff: {diff_max:.6f}")

  # Print first 60 tokens comparison
  print("\n=== Token-by-Token Comparison (First 60 Active Positions) ===")
  print(f"{'Idx':<4} | {'TokenID':<8} | {'TokenDecoded':<20} | {'RolloutLogp':<11} | {'TrainerLogp':<11} | {'Diff':<10} | {'Compared':<8}")
  print("-" * 85)
  for idx in range(min(60, len(full_active_seq))):
    tok_id = int(full_active_seq[idx])
    tok_str = tokenizer.decode([tok_id]) if tokenizer else str(tok_id)
    r_val = float(active_rollout_logps[idx])
    t_val = float(active_trainer_logps[idx])
    d_val = r_val - t_val
    comp_str = "YES" if active_mask[idx] > 0 else "NO"
    print(f"{idx:03d}  | {tok_id:<8} | {tok_str!r:<20} | {r_val:<11.4f} | {t_val:<11.4f} | {d_val:<+10.4f} | {comp_str}")

  # Run Shift Search
  print("\n=== Shift Analysis ===")
  # Find best shift s of trainer logprobs relative to rollout logprobs
  # to minimize absolute difference on compared positions
  best_s = 0
  min_mean_diff = float("inf")
  best_diffs = []
  
  for s in range(-5, 6):
    diffs = []
    for idx in range(len(full_active_seq)):
      if active_mask[idx] > 0:
        t_idx = idx + s
        if 0 <= t_idx < len(active_trainer_logps):
          diffs.append(abs(active_rollout_logps[idx] - active_trainer_logps[t_idx]))
        else:
          diffs.append(999.0) # boundary penalty
    
    if diffs:
      mean_diff = np.mean(diffs)
      print(f"Shift s={s:2d} | Mean active diff: {mean_diff:.6f}")
      if mean_diff < min_mean_diff:
        min_mean_diff = mean_diff
        best_s = s
        best_diffs = diffs

  print(f"\nBest Shift Offset: s = {best_s} (Mean diff = {min_mean_diff:.6f})")

if __name__ == "__main__":
  main()
