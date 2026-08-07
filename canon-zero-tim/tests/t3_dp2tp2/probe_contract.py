#!/usr/bin/env python3
"""Exercise the static DP2xTP2 training contract and its negatives."""

from __future__ import annotations

import json

import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P

from tunix.rl import dp_training


def _must_reject(label, fn):
  try:
    fn()
  except ValueError as exc:
    return {"label": label, "rejected": True, "error": str(exc)}
  raise AssertionError(f"negative control did not reject: {label}")


def main():
  contract = dp_training.DPTrainingContract(
      dp_size=2,
      tp_size=2,
      global_prompts=4,
      num_generations=8,
      local_trajectories=16,
  )
  contract.validate()
  groups = np.repeat(np.arange(4), 8)
  contract.validate_prompt_groups(groups)
  rank_indices = contract.rank_indices()
  inventory = dp_training.validate_dp_replicated_partition_specs(
      {"left": P(None, "tp"), "right": P("tp", None)}, label="params"
  )
  fixed = dp_training.fixed_dp2_sum(
      {"g": jnp.asarray([1.0e8, 1.0], jnp.float32)},
      {"g": jnp.asarray([-1.0e8, 2.0], jnp.float32)},
  )

  split_groups = groups.copy()
  split_groups[15], split_groups[16] = split_groups[16], split_groups[15]
  negatives = [
      _must_reject(
          "wrong-local-count",
          lambda: dp_training.DPTrainingContract(
              dp_size=2,
              tp_size=2,
              global_prompts=4,
              num_generations=8,
              local_trajectories=15,
          ).validate(),
      ),
      _must_reject(
          "split-prompt-group",
          lambda: contract.validate_prompt_groups(split_groups),
      ),
      _must_reject(
          "dp-sharded-parameter",
          lambda: dp_training.validate_dp_replicated_partition_specs(
              {"bad": P("dp", "tp")}, label="params"
          ),
      ),
  ]
  report = {
      "status": "pass",
      "topology": {"dp": 2, "tp": 2, "devices": 4},
      "batch": {
          "global_prompts": contract.global_prompts,
          "num_generations": contract.num_generations,
          "global_trajectories": contract.global_trajectories,
          "local_prompts": contract.local_prompts,
          "local_trajectories": contract.local_trajectories,
          "rank_counts": [int(rows.size) for rows in rank_indices],
      },
      "inventory": inventory,
      "fixed_sum": np.asarray(fixed["g"]).tolist(),
      "negative_count": len(negatives),
      "negatives": negatives,
  }
  print(f"P32.D0.CONTRACT {json.dumps(report, sort_keys=True)}", flush=True)


if __name__ == "__main__":
  main()
