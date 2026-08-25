#!/usr/bin/env python3
"""Real-v5p DP2xTP2 matched backward and P59 scaling carrier."""

from __future__ import annotations

import json
import os

import jax
import jax.numpy as jnp
import numpy as np

from tunix.rl import dp_training
from tunix.sft import utils as sft_utils


_DP = 2
_TP = 2
_GROUPS = 2
_INPUT = 8
_OUTPUT = 12
_GLOBAL_TRAJECTORIES = _DP * _GROUPS
_LOSS_SCALE = np.float32(1.0 / _GLOBAL_TRAJECTORIES)
_STREAMED_MULTIPLIER = np.float32(_LOSS_SCALE * _GROUPS)
_CONSTRUCTION_ENV = "P62_ONEHOST_CONSTRUCTION_ONLY"


def _put(value, mesh, spec):
  return jax.device_put(
      value,
      jax.sharding.NamedSharding(
          mesh, jax.sharding.PartitionSpec(*spec)
      ),
  )


def _fixed_tp_sum(value):
  gathered = jax.lax.all_gather(
      value, "model", axis=0, tiled=False
  )
  total = jnp.zeros_like(value, dtype=jnp.float32)
  for rank in range(_TP):
    total = jax.lax.optimization_barrier(total) + (
        jax.lax.optimization_barrier(gathered[rank].astype(jnp.float32))
    )
  return total.astype(value.dtype)


def _relative_l2(actual, expected):
  actual = np.asarray(actual, dtype=np.float64)
  expected = np.asarray(expected, dtype=np.float64)
  return float(
      np.linalg.norm(actual - expected)
      / max(np.linalg.norm(expected), np.finfo(np.float64).tiny)
  )


def _cosine(actual, expected):
  actual = np.asarray(actual, dtype=np.float64).reshape(-1)
  expected = np.asarray(expected, dtype=np.float64).reshape(-1)
  return float(
      np.dot(actual, expected)
      / max(
          np.linalg.norm(actual) * np.linalg.norm(expected),
          np.finfo(np.float64).tiny,
      )
  )


def _select_devices():
  construction_value = os.environ.get(_CONSTRUCTION_ENV, "")
  if construction_value not in ("", "0", "1"):
    raise ValueError(f"{_CONSTRUCTION_ENV} must be unset/0/1")
  construction_only = construction_value == "1"
  devices = np.asarray(
      jax.devices() if construction_only else jax.devices("tpu")
  )
  if devices.size != 4:
    raise RuntimeError(
        f"P62 one-host gate requires exactly four devices: {devices}"
    )
  if not construction_only and {
      str(device.device_kind) for device in devices.flat
  } != {"TPU v5"}:
    raise RuntimeError("P62 one-host gate requires four v5p devices")
  return devices, construction_only


def main() -> None:
  devices, construction_only = _select_devices()
  mesh = jax.sharding.Mesh(devices.reshape(_DP, _TP), ("data", "model"))

  weight_host = (
      np.arange(_INPUT * _OUTPUT, dtype=np.float32).reshape(_INPUT, _OUTPUT)
      / np.float32(97.0)
  )
  weight = _put(weight_host, mesh, (None, "model"))
  group_values = []
  group_cotangents = []
  for group in range(_GROUPS):
    group_values.append(_put(
        (
            np.arange(_DP * _INPUT, dtype=np.float32).reshape(
                _DP, 1, _INPUT
            )
            + np.float32(group * 3)
        )
        / np.float32(29.0),
        mesh,
        ("data", None, None),
    ))
    group_cotangents.append(_put(
        (
            np.arange(_DP * _OUTPUT, dtype=np.float32).reshape(
                _DP, 1, _OUTPUT
            )
            + np.float32(group + 1)
        )
        / np.float32(31.0),
        mesh,
        ("data", None, "model"),
    ))

  def forward(weight_arg, values_arg):
    return jnp.einsum("brd,do->bro", values_arg, weight_arg)

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
          jax.sharding.PartitionSpec("data", None, None),
          jax.sharding.PartitionSpec("data", None, "model"),
      ),
      out_specs=(
          jax.sharding.PartitionSpec("data", None, "model"),
          jax.sharding.PartitionSpec("data", None, None),
      ),
      axis_names=frozenset(("data", "model")),
      check_vma=False,
  ))

  reduced_groups = []
  serial_groups = []
  for group, (values, cotangent) in enumerate(
      zip(group_values, group_cotangents, strict=True)
  ):
    _, global_pullback = jax.vjp(forward, weight, values)
    serial_weight, serial_values = global_pullback(cotangent)
    staged, parallel_values = parallel_pullback(
        weight, values, cotangent
    )
    jax.block_until_ready((staged, parallel_values))
    np.testing.assert_array_equal(
        np.asarray(parallel_values), np.asarray(serial_values)
    )

    rank_receipt = sft_utils.tree_numeric_receipt(
        {"weight": staged}, ranked=True
    )
    if not rank_receipt["all_finite"]:
      raise AssertionError(f"P62 staged gradient is non-finite: {rank_receipt}")
    reducer = dp_training.FixedDPRankGradientReducer(
        {"weight": serial_weight},
        dp_size=_DP,
        dp_axis="data",
        require_distinct_fingerprints=False,
    )
    reduced, report = reducer.finalize_staged({"weight": staged})
    reduced_weight = reduced["weight"]
    np.testing.assert_array_equal(
        np.asarray(reduced_weight), np.asarray(serial_weight)
    )
    if (
        report["rank_contributions"] != _DP
        or not report["post_reduction_all_finite"]
        or not report["post_reduction_replicas_exact"]
    ):
      raise AssertionError(f"P62 reducer evidence changed: {report}")
    reduced_groups.append(reduced_weight)
    serial_groups.append(serial_weight)
    if group in (0, _GROUPS - 1):
      print(
          "[P62.ONEHOST] "
          + json.dumps({
              "schema": "canon-p62-onehost-group-v1",
              "group": group,
              "rank_local": rank_receipt,
              "reduction_rounds": report["reduction_rounds"],
              "replicas_exact": report["post_reduction_replicas_exact"],
          }, sort_keys=True, separators=(",", ":")),
          flush=True,
      )

  parallel_final = sum(
      gradient * jnp.asarray(_STREAMED_MULTIPLIER, gradient.dtype)
      for gradient in reduced_groups
  ) / jnp.asarray(_GROUPS, jnp.float32)
  serial_final = sum(serial_groups) * jnp.asarray(_LOSS_SCALE, jnp.float32)
  jax.block_until_ready((parallel_final, serial_final))
  np.testing.assert_array_equal(
      np.asarray(parallel_final), np.asarray(serial_final)
  )

  oracle = np.zeros_like(weight_host, dtype=np.float64)
  for values, cotangent in zip(
      group_values, group_cotangents, strict=True
  ):
    oracle += np.einsum(
        "brd,bro->do",
        np.asarray(values, dtype=np.float64),
        np.asarray(cotangent, dtype=np.float64),
    )
  oracle /= np.float64(_GLOBAL_TRAJECTORIES)
  final_host = np.asarray(parallel_final, dtype=np.float32)
  rel_l2 = _relative_l2(final_host, oracle)
  cosine = _cosine(final_host, oracle)
  if not np.isfinite(rel_l2) or rel_l2 > 2.0e-7 or cosine < 0.9999999:
    raise AssertionError(
        f"P62 ordinary/parallel FP64 oracle mismatch: rel_l2={rel_l2} "
        f"cosine={cosine}"
    )

  wrong_multiplier = sum(reduced_groups) / np.float32(_GROUPS)
  wrong_dp_sum = parallel_final * np.float32(_DP)
  wrong_multiplier_rel_l2 = _relative_l2(wrong_multiplier, oracle)
  wrong_dp_rel_l2 = _relative_l2(wrong_dp_sum, oracle)
  if wrong_multiplier_rel_l2 < 0.9 or wrong_dp_rel_l2 < 0.9:
    raise AssertionError(
        "P62 scaling negatives did not separate: "
        f"multiplier={wrong_multiplier_rel_l2} dp={wrong_dp_rel_l2}"
    )

  final_receipt = sft_utils.tree_numeric_receipt(
      {"weight": parallel_final}
  )
  if not final_receipt["all_finite"] or not final_receipt["any_nonzero"]:
    raise AssertionError(f"P62 final gradient invalid: {final_receipt}")
  terminal = (
      "P62_NUMERIC_ONEHOST_CONSTRUCTION_PASS"
      if construction_only
      else "P62_NUMERIC_ONEHOST_V5P_PASS"
  )
  print(
      terminal + " "
      "topology=DP2xTP2 groups=2 global_trajectories=4 "
      "loss_scale=0.25 streamed_multiplier=0.5 accumulator_denom=2 "
      f"rel_l2_fp64={rel_l2:.9g} cosine_fp64={cosine:.9g} "
      f"wrong_multiplier_rel_l2={wrong_multiplier_rel_l2:.9g} "
      f"wrong_dp_sum_rel_l2={wrong_dp_rel_l2:.9g} "
      "fixed_tp_input_reduction=1 fixed_dp_reduction=1 "
      "optimizer_commits=0",
      flush=True,
  )


if __name__ == "__main__":
  main()
