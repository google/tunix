"""Batch-boundary evidence from the real learner, never a loss substitution."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

from examples.frozenlake.gemma4_tim import artifacts, recipe


def compare(left, right, mask) -> dict:
  left, right, mask = map(np.asarray, (left, right, mask))
  if left.shape != right.shape or mask.shape != left.shape or left.dtype != right.dtype:
    raise recipe.RecipeError("logprob shape or dtype mismatch")
  if mask.dtype != np.bool_ or left.dtype != np.float32:
    raise recipe.RecipeError("expected bool mask and float32 logprobs")
  a, b = np.ascontiguousarray(left[mask]), np.ascontiguousarray(right[mask])
  if not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
    raise recipe.RecipeError("empty or nonfinite action logprobs")
  words = a.view(np.uint32) != b.view(np.uint32)
  return {"actions": int(a.size), "differing_elements": int(words.sum()),
          "differing_bytes": int((a.view(np.uint8) != b.view(np.uint8)).sum()),
          "max_abs": float(np.max(np.abs(a.astype(np.float64) - b))),
          "mean_abs": float(np.mean(np.abs(a.astype(np.float64) - b)))}


class BatchObserver:
  """One fetch per completed learner group; no per-token device reads.

  This first collector preserves A/C, mask/token/weight-version provenance,
  and the exact loss input selection. B and D are explicitly NOT_OBSERVED
  until their independent production collectors are connected.
  """

  def __init__(self, root: Path, contract: dict):
    recipe.validate_resolved(contract)
    self.root = root
    self.contract = contract
    self.records = 0
    self.rows_by_step = {}
    self.seen = set()
    self.wall_seconds = 0.0
    self.lock = threading.Lock()

  def __call__(self, *, batch, s_decode, t_old, completion_lengths, trajectory_ids, mode, step):
    started = time.perf_counter()
    import jax  # No JAX initialization in recipe-only processes.
    if s_decode is None or t_old is None:
      raise recipe.RecipeError("real learner omitted rollout or trainer logps")
    arm = self.contract["arm"]
    expected_old = t_old if arm == "tis" else s_decode
    if batch.old_per_token_logps is not expected_old:
      raise recipe.RecipeError("actual learner old-logps source violates arm")
    if (batch.sampler_is_weights is not None) != (arm == "tis"):
      raise recipe.RecipeError("actual learner TIS ownership violates arm")
    arrays = jax.device_get({
        "prompt_ids": batch.prompt_ids, "prompt_mask": batch.prompt_mask,
        "completion_ids": batch.completion_ids,
        "completion_valid_mask": batch.completion_valid_mask,
        "action_mask": batch.completion_mask, "s_decode": s_decode,
        "t_old": t_old, "loss_old": batch.old_per_token_logps,
        "policy_version": batch.policy_version, "advantages": batch.advantages,
        **({"tis_weights": batch.sampler_is_weights} if arm == "tis" else {}),
    })
    arrays = {k: np.asarray(v) for k, v in arrays.items()}
    arrays["completion_lengths"] = np.asarray(completion_lengths, dtype=np.int32)
    arrays["trajectory_ids"] = np.asarray(trajectory_ids, dtype=np.int64)
    mode_value = getattr(mode, "value", str(mode))
    mask = arrays["action_mask"].astype(bool)
    valid = arrays["completion_valid_mask"].astype(bool)
    if np.any(mask & ~valid):
      raise recipe.RecipeError("action mask is not a subset of valid context")
    arrays["action_mask"] = mask
    comparison = compare(arrays["s_decode"], arrays["t_old"], mask)
    if (arrays["trajectory_ids"].shape != (mask.shape[0], 2)
        or any(not 0 <= pair < 8 for _, pair in trajectory_ids)
        or (mode_value == "train" and any(not int(step) * 64 <= group < (int(step) + 1) * 64
                                          for group, _ in trajectory_ids))):
      raise recipe.RecipeError("trajectory identity is outside the default workload")
    identities = {(mode_value, int(step), int(group), int(pair)) for group, pair in trajectory_ids}
    if len(identities) != mask.shape[0]:
      raise recipe.RecipeError("duplicate trajectory in a learner group")
    if np.any(arrays["policy_version"] != int(step)):
      raise recipe.RecipeError("captured batch has a stale or mixed policy version")
    if arm == "zero":
      # A/C alone cannot admit Zero, even if identical. This is deliberately
      # removed only when independent B and real value-and-grad D are wired.
      raise recipe.RecipeError("Zero-TIM requires independent B and D collectors")
    with self.lock:
      if identities & self.seen:
        raise recipe.RecipeError("duplicate trajectory across learner groups")
      root = self.root / f"batch-{self.records:06d}"
      root.mkdir(mode=0o700)
      payload = root / "arrays.npz"
      with payload.open("xb") as stream:
        payload.chmod(0o600)
        np.savez_compressed(stream, **arrays)
      receipt = {
          "schema": "gemma4-e2b-batch-v1", "arm": arm, "step": int(step),
          "rows": int(mask.shape[0]),
          "mode": mode_value,
          "contract_sha256": self.contract["contract_sha256"],
          "arrays_sha256": artifacts.file_sha(payload),
          "A_C": comparison, "A_B": "NOT_OBSERVED", "C_D": "NOT_OBSERVED",
          "old_logps": self.contract["old_logps"],
          "correction": self.contract["correction"],
          "claim": "stock-mismatch-observation-not-zero-tim",
      }
      artifacts.write_json(root / "receipt.json", receipt)
      self.records += 1
      self.seen.update(identities)
      if receipt["mode"] == "train":
        self.rows_by_step[int(step)] = self.rows_by_step.get(int(step), 0) + int(mask.shape[0])
    print("GEMMA4_E2B_BATCH " + recipe.json_bytes(receipt).decode().strip(), flush=True)
    with self.lock:
      self.wall_seconds += time.perf_counter() - started
