#!/usr/bin/env python3
"""Bounded DP16xTP4 fixed replay for ordinary-vs-P59 gradient scale."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tunix.rl import dp_training
from tunix.sft import utils as sft_utils


_DP = 16
_TP = 4
_GROUPS = 16
_GLOBAL_TRAJECTORIES = 256
_INPUT = 8
_OUTPUT = 16
_SEED = 42
_LOSS_SCALE = np.float32(1.0 / _GLOBAL_TRAJECTORIES)
_STREAMED_MULTIPLIER = np.float32(_GROUPS / _GLOBAL_TRAJECTORIES)


def _hash(value: Any) -> str:
  array = np.ascontiguousarray(np.asarray(value))
  digest = hashlib.sha256()
  digest.update(str(array.dtype).encode("ascii"))
  digest.update(json.dumps(array.shape).encode("ascii"))
  digest.update(array.tobytes())
  return digest.hexdigest()


def _frozen_replay() -> dict[str, np.ndarray]:
  """Returns one deterministic capsule consumed by both backward arms."""
  rng = np.random.default_rng(_SEED)
  tokens = rng.integers(
      0, 4096, size=(_DP, _GROUPS, 4), dtype=np.int32
  )
  action_mask = (
      np.arange(_DP * _GROUPS * 4).reshape(_DP, _GROUPS, 4) % 5 != 0
  )
  advantages = np.linspace(
      -1.25, 1.75, _GLOBAL_TRAJECTORIES, dtype=np.float32
  ).reshape(_DP, _GROUPS)
  token_float = tokens.astype(np.float32)
  values = np.stack(
      tuple(
          (
              token_float[..., index % 4]
              * np.float32((index + 1) / 4096.0)
              + advantages * np.float32((index + 3) / 17.0)
          )
          for index in range(_INPUT)
      ),
      axis=-1,
  ).astype(np.float32)
  valid_fraction = action_mask.mean(axis=-1, dtype=np.float32)
  cotangent = np.stack(
      tuple(
          advantages * np.float32((index + 1) / 23.0)
          + valid_fraction * np.float32((index % 5 + 1) / 29.0)
          + token_float[..., index % 4] * np.float32(1.0 / 65536.0)
          for index in range(_OUTPUT)
      ),
      axis=-1,
  ).astype(np.float32)
  weight = (
      np.arange(_INPUT * _OUTPUT, dtype=np.float32).reshape(_INPUT, _OUTPUT)
      / np.float32(127.0)
  )
  return {
      "tokens": tokens,
      "action_mask": action_mask,
      "advantages": advantages,
      "values": values,
      "cotangent": cotangent,
      "weight": weight,
  }


def _put(value, mesh, spec):
  return jax.device_put(
      value,
      jax.sharding.NamedSharding(
          mesh, jax.sharding.PartitionSpec(*spec)
      ),
  )


def _fixed_tp_sum(value):
  gathered = jax.lax.all_gather(value, "model", axis=0, tiled=False)
  total = jnp.zeros_like(value, dtype=jnp.float32)
  for rank in range(_TP):
    total = jax.lax.optimization_barrier(total) + jax.lax.optimization_barrier(
        gathered[rank].astype(jnp.float32)
    )
  return total.astype(value.dtype)


def _relative_l2(actual, expected) -> float:
  actual64 = np.asarray(actual, dtype=np.float64)
  expected64 = np.asarray(expected, dtype=np.float64)
  return float(
      np.linalg.norm(actual64 - expected64)
      / max(np.linalg.norm(expected64), np.finfo(np.float64).tiny)
  )


def _cosine(actual, expected) -> float:
  actual64 = np.asarray(actual, dtype=np.float64).reshape(-1)
  expected64 = np.asarray(expected, dtype=np.float64).reshape(-1)
  return float(
      np.dot(actual64, expected64)
      / max(
          np.linalg.norm(actual64) * np.linalg.norm(expected64),
          np.finfo(np.float64).tiny,
      )
  )


def run() -> dict[str, Any]:
  devices = np.asarray(jax.devices())
  if devices.size != _DP * _TP:
    raise RuntimeError(
        "GSM fixed replay requires exactly 64 devices for DP16xTP4: "
        f"{devices.size}"
    )
  mesh = jax.sharding.Mesh(devices.reshape(_DP, _TP), ("data", "model"))
  replay = _frozen_replay()
  replay_hashes = {name: _hash(value) for name, value in replay.items()}
  capsule_hash = hashlib.sha256(
      json.dumps(replay_hashes, sort_keys=True).encode("ascii")
  ).hexdigest()

  weight = _put(replay["weight"], mesh, (None, "model"))
  values_all = _put(
      replay["values"].reshape(_GLOBAL_TRAJECTORIES, _INPUT),
      mesh,
      ("data", None),
  )
  cotangent_all = _put(
      replay["cotangent"].reshape(_GLOBAL_TRAJECTORIES, _OUTPUT),
      mesh,
      ("data", "model"),
  )

  def forward(weight_arg, values_arg):
    return jnp.einsum("bi,io->bo", values_arg, weight_arg)

  _, ordinary_pullback = jax.vjp(forward, weight, values_all)
  ordinary_weight, ordinary_values = ordinary_pullback(cotangent_all)
  ordinary_final = ordinary_weight * jnp.asarray(
      _LOSS_SCALE, ordinary_weight.dtype
  )

  def local_pullback(local_weight, local_values, local_cotangent):
    _, pullback = jax.vjp(forward, local_weight, local_values)
    dweight, dvalues_partial = pullback(local_cotangent)
    dvalues = _fixed_tp_sum(dvalues_partial)
    return jnp.expand_dims(dweight.astype(jnp.float32), 0), dvalues

  parallel_pullback = jax.jit(jax.shard_map(
      local_pullback,
      mesh=mesh,
      in_specs=(
          jax.sharding.PartitionSpec(None, "model"),
          jax.sharding.PartitionSpec("data", None),
          jax.sharding.PartitionSpec("data", "model"),
      ),
      out_specs=(
          jax.sharding.PartitionSpec("data", None, "model"),
          jax.sharding.PartitionSpec("data", None),
      ),
      axis_names=frozenset(("data", "model")),
      check_vma=False,
  ))

  reduced_groups = []
  rank_hashes = []
  for group in range(_GROUPS):
    values = _put(replay["values"][:, group, :], mesh, ("data", None))
    cotangent = _put(
        replay["cotangent"][:, group, :], mesh, ("data", "model")
    )
    staged, parallel_values = parallel_pullback(weight, values, cotangent)
    jax.block_until_ready((staged, parallel_values))
    expected_values = np.einsum(
        "bo,io->bi",
        replay["cotangent"][:, group, :].astype(np.float64),
        replay["weight"].astype(np.float64),
    ).astype(np.float32)
    np.testing.assert_allclose(
        np.asarray(parallel_values), expected_values, rtol=2e-6, atol=2e-6
    )
    staged_host = np.asarray(staged)
    group_rank_hashes = tuple(_hash(staged_host[rank]) for rank in range(_DP))
    if len(set(group_rank_hashes)) != _DP:
      raise AssertionError(
          f"rank-local gradient ownership duplicated at group {group}"
      )
    rank_hashes.append(group_rank_hashes)
    rank_receipt = sft_utils.tree_numeric_receipt(
        {"weight": staged}, ranked=True
    )
    if not rank_receipt["all_finite"] or rank_receipt["rank_count"] != _DP:
      raise AssertionError(
          f"rank-local gradient receipt failed at group {group}: "
          f"{rank_receipt}"
      )
    reducer = dp_training.FixedDPRankGradientReducer(
        {"weight": ordinary_weight},
        dp_size=_DP,
        dp_axis="data",
        require_distinct_fingerprints=False,
    )
    reduced, report = reducer.finalize_staged({"weight": staged})
    if (
        report["rank_contributions"] != _DP
        or not report["post_reduction_all_finite"]
        or not report["post_reduction_replicas_exact"]
    ):
      raise AssertionError(f"fixed DP reduction changed: {report}")
    reduced_groups.append(reduced["weight"])

  parallel_final = sum(
      gradient * jnp.asarray(_STREAMED_MULTIPLIER, gradient.dtype)
      for gradient in reduced_groups
  ) / jnp.asarray(_GROUPS, jnp.float32)
  jax.block_until_ready((ordinary_final, parallel_final, ordinary_values))

  oracle = np.einsum(
      "bi,bo->io",
      replay["values"].reshape(_GLOBAL_TRAJECTORIES, _INPUT).astype(np.float64),
      replay["cotangent"].reshape(
          _GLOBAL_TRAJECTORIES, _OUTPUT
      ).astype(np.float64),
  ) / np.float64(_GLOBAL_TRAJECTORIES)
  ordinary_host = np.asarray(ordinary_final)
  parallel_host = np.asarray(parallel_final)
  ordinary_rel_l2 = _relative_l2(ordinary_host, oracle)
  parallel_rel_l2 = _relative_l2(parallel_host, oracle)
  arm_rel_l2 = _relative_l2(parallel_host, ordinary_host)
  ordinary_cosine = _cosine(ordinary_host, oracle)
  parallel_cosine = _cosine(parallel_host, oracle)
  if (
      max(ordinary_rel_l2, parallel_rel_l2, arm_rel_l2) > 3e-6
      or min(ordinary_cosine, parallel_cosine) < 0.999999
  ):
    raise AssertionError(
        "ordinary/P59 fixed replay disagrees with FP64 oracle: "
        f"ordinary={ordinary_rel_l2} parallel={parallel_rel_l2} "
        f"arms={arm_rel_l2}"
    )
  wrong_denominator = parallel_host * np.float32(_GROUPS)
  duplicate_dp_sum = parallel_host * np.float32(_DP)
  wrong_denominator_rel_l2 = _relative_l2(wrong_denominator, oracle)
  duplicate_dp_rel_l2 = _relative_l2(duplicate_dp_sum, oracle)
  if wrong_denominator_rel_l2 < 10.0 or duplicate_dp_rel_l2 < 10.0:
    raise AssertionError("scale/ownership negatives did not separate")

  first_trajectory = {
      "tokens": replay["tokens"][0, 0].tolist(),
      "action_mask": replay["action_mask"][0, 0].astype(int).tolist(),
      "advantage": float(replay["advantages"][0, 0]),
  }
  return {
      "schema": "canon-v1-gsm-fixed-replay-scale-v1",
      "verdict": "PASS",
      "topology": "DP16xTP4",
      "seed": _SEED,
      "global_trajectories": _GLOBAL_TRAJECTORIES,
      "local_trajectories": _GROUPS,
      "groups": _GROUPS,
      "loss_denominator": float(_GLOBAL_TRAJECTORIES),
      "loss_scale": float(_LOSS_SCALE),
      "streamed_multiplier": float(_STREAMED_MULTIPLIER),
      "accumulator_denominator": float(_GROUPS),
      "capsule_sha256": capsule_hash,
      "checkpoint_sha256": replay_hashes["weight"],
      "tokens_sha256": replay_hashes["tokens"],
      "action_mask_sha256": replay_hashes["action_mask"],
      "advantages_sha256": replay_hashes["advantages"],
      "cotangent_sha256": replay_hashes["cotangent"],
      "first_trajectory": first_trajectory,
      "rank_partial_unique_per_group": [
          len(set(group_hashes)) for group_hashes in rank_hashes
      ],
      "ordinary_gradient_sha256": _hash(ordinary_host),
      "p59_gradient_sha256": _hash(parallel_host),
      "ordinary_p59_byte_exact": bool(
          np.array_equal(ordinary_host, parallel_host)
      ),
      "ordinary_rel_l2_fp64": ordinary_rel_l2,
      "p59_rel_l2_fp64": parallel_rel_l2,
      "ordinary_p59_rel_l2": arm_rel_l2,
      "ordinary_cosine_fp64": ordinary_cosine,
      "p59_cosine_fp64": parallel_cosine,
      "wrong_denominator_rel_l2": wrong_denominator_rel_l2,
      "duplicate_dp_sum_rel_l2": duplicate_dp_rel_l2,
      "optimizer_commits": 0,
      "claim_ceiling": "bounded_projection_topology_and_scale_only",
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--receipt", type=Path)
  args = parser.parse_args()
  result = run()
  rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
  if args.receipt:
    if args.receipt.exists():
      raise FileExistsError(f"refusing to overwrite receipt: {args.receipt}")
    args.receipt.write_text(rendered, encoding="utf-8")
  print("[V1.GSM.FIXED_REPLAY] " + json.dumps(result, sort_keys=True))
  print(
      "V1_GSM_FIXED_REPLAY_PASS topology=DP16xTP4 groups=16 "
      "rank_ownership=16/16 fp64=ordinary,p59 negatives=denominator,dp_sum "
      "optimizer_commits=0"
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
