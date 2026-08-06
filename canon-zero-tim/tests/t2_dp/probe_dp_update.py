#!/usr/bin/env python3
"""Characterize DP gradient reduction and one AdamW update.

This probe deliberately separates three contracts that are easy to conflate:

* repeatability with a fixed sample-to-rank mapping;
* sensitivity to the physical device order used by the mesh;
* sensitivity to regrouping the same global samples across DP ranks.

Only the first contract is a release requirement for the initial P32 recipe.
The other two are measurements that decide whether a canonical DP reduction or
canonical per-example accumulation is needed before promotion.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import sys
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "t1_tpu")))
try:
  from pathways_bootstrap import initialize_pathways
  initialize_pathways()
except Exception:
  pass

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P


def _csv_ints(value: str) -> tuple[int, ...]:
  return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _sha(value: Any) -> str:
  array = np.ascontiguousarray(np.asarray(jax.device_get(value)))
  return hashlib.sha256(array.view(np.uint8)).hexdigest()


def _exact(left: Any, right: Any) -> bool:
  return bool(np.array_equal(
      np.asarray(jax.device_get(left)), np.asarray(jax.device_get(right))
  ))


def _rows_exact(value: Any) -> bool:
  rows = np.asarray(jax.device_get(value))
  return all(np.array_equal(rows[0], row) for row in rows[1:])


def _make_mesh(devices: np.ndarray, dp: int, tp: int) -> Mesh:
  return Mesh(devices.reshape(dp, tp), ("dp", "tp"))


def _topology_mesh(devices: list[jax.Device], dp: int, tp: int) -> Mesh:
  try:
    arranged = jax._src.mesh_utils.create_device_mesh(  # pylint: disable=protected-access
        (dp, tp), devices
    )
  except Exception as exc:  # topology support varies by backend
    print(
        "[P32.DP] topology-aware mesh unavailable; using logical reshape: "
        f"{type(exc).__name__}: {exc}",
        flush=True,
    )
    arranged = np.asarray(devices, dtype=object).reshape(dp, tp)
  return _make_mesh(np.asarray(arranged, dtype=object), dp, tp)


def _build_reducer(mesh: Mesh, dp: int, global_samples: int):
  out_replicated = P(None, "tp")
  out_by_dp = P("dp", None, "tp")

  @functools.partial(
      jax.shard_map,
      mesh=mesh,
      in_specs=(P("dp", None), P(None, "tp")),
      out_specs=(
          out_replicated,
          out_replicated,
          out_by_dp,
          out_by_dp,
          out_by_dp,
      ),
      check_vma=False,
  )
  def reduce_gradients(x, weight):
    def local_loss(candidate):
      logits = x @ candidate
      # Each shard contributes its local sum. The subsequent DP sum therefore
      # equals the global mean over exactly ``global_samples`` rows.
      return jnp.sum(jnp.square(logits)) / jnp.asarray(
          global_samples, jnp.float32
      )

    local_gradient = jax.grad(local_loss)(weight)
    stock = jax.lax.psum(local_gradient, "dp")

    # all_gather transports values without summing. Adding the logical source
    # ranks one at a time, with barriers between additions, is the small-probe
    # canonical reference. It is not proposed as the production DP16 reducer:
    # materialising every model-sized gradient would be prohibitively costly.
    gathered = jax.lax.all_gather(
        local_gradient, "dp", axis=0, tiled=False
    )
    fixed = gathered[0]
    for source_rank in range(1, dp):
      fixed = (
          jax.lax.optimization_barrier(fixed) + gathered[source_rank]
      )

    rank = jax.lax.axis_index("dp").astype(stock.dtype)
    fault = stock + rank * jnp.asarray(2.0**-10, stock.dtype)
    return stock, fixed, stock[None, ...], fixed[None, ...], fault[None, ...]

  return jax.jit(reduce_gradients)


def _build_auto_gradient(mesh: Mesh, global_samples: int):
  x_sharding = jax.sharding.NamedSharding(mesh, P("dp", None))
  weight_sharding = jax.sharding.NamedSharding(mesh, P(None, "tp"))

  def gradient(x, weight):
    return jax.grad(
        lambda candidate: (
            jnp.sum(jnp.square(x @ candidate))
            / jnp.asarray(global_samples, jnp.float32)
        )
    )(weight)

  return jax.jit(
      gradient,
      in_shardings=(x_sharding, weight_sharding),
      out_shardings=weight_sharding,
  )


def _adamw_step(weight, gradient):
  b1 = jnp.asarray(0.9, jnp.float32)
  b2 = jnp.asarray(0.95, jnp.float32)
  learning_rate = jnp.asarray(1.0e-6, jnp.float32)
  epsilon = jnp.asarray(1.0e-8, jnp.float32)
  moment = (1.0 - b1) * gradient
  variance = (1.0 - b2) * jnp.square(gradient)
  update = moment / (jnp.sqrt(variance) + epsilon)
  next_weight = weight - learning_rate * update.astype(weight.dtype)
  return next_weight, moment, variance


def _make_inputs(dp: int, tp: int, local_samples: int):
  global_samples = dp * local_samples
  features = 16
  outputs = 8 * tp
  rng = np.random.default_rng(20260806)
  x = rng.standard_normal((global_samples, features), dtype=np.float32)
  # Mixed magnitudes make reassociation visible without changing dtype.
  scales = np.exp2(
      ((np.arange(global_samples, dtype=np.int32) % 17) - 8).astype(np.float32)
  )
  x *= scales[:, None]
  weight = rng.standard_normal((features, outputs), dtype=np.float32) / 8.0
  return x, weight


def _regroup_rows(x: np.ndarray, dp: int, local_samples: int) -> np.ndarray:
  # Same rows, same within-group order, different sample-to-rank assignment.
  grouped = x.reshape(dp, local_samples, x.shape[1])
  return grouped.transpose(1, 0, 2).reshape(x.shape)


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--dp", type=int, default=int(os.getenv("CANON_DP_SIZE", "4")))
  parser.add_argument("--tp", type=int, default=int(os.getenv("CANON_TP_SIZE", "1")))
  parser.add_argument(
      "--local-samples",
      type=int,
      default=int(os.getenv("CANON_DP_PROBE_LOCAL_SAMPLES", "16")),
  )
  parser.add_argument(
      "--inject-rank-fault",
      action="store_true",
      help="negative control: substitute a rank-dependent result for stock",
  )
  args = parser.parse_args()

  if args.dp <= 1 or args.tp <= 0 or args.local_samples <= 0:
    raise ValueError("dp must be >1; tp and local-samples must be positive")
  required = args.dp * args.tp
  devices = list(jax.devices())
  if len(devices) != required:
    raise RuntimeError(
        f"P32 DP probe requires exactly dp*tp={required} devices, got {len(devices)}"
    )

  mesh = _topology_mesh(devices, args.dp, args.tp)
  mesh_ids = tuple(int(device.id) for device in mesh.devices.flat)
  expected_ids = _csv_ints(os.getenv("CANON_EXPECT_TRAIN_MESH_IDS", ""))
  if expected_ids and mesh_ids != expected_ids:
    raise RuntimeError(
        "CANON_EXPECT_TRAIN_MESH_IDS mismatch: "
        f"expected={expected_ids} actual={mesh_ids}"
    )
  alt_mesh = _make_mesh(
      np.asarray(mesh.devices, dtype=object)[::-1, :], args.dp, args.tp
  )

  x_np, weight_np = _make_inputs(args.dp, args.tp, args.local_samples)
  regrouped_np = _regroup_rows(x_np, args.dp, args.local_samples)
  global_samples = int(x_np.shape[0])

  reducer = _build_reducer(mesh, args.dp, global_samples)
  reducer_alt = _build_reducer(alt_mesh, args.dp, global_samples)
  auto_gradient = _build_auto_gradient(mesh, global_samples)
  auto_gradient_alt = _build_auto_gradient(alt_mesh, global_samples)

  first = reducer(jnp.asarray(x_np), jnp.asarray(weight_np))
  second = reducer(jnp.asarray(x_np), jnp.asarray(weight_np))
  alt = reducer_alt(jnp.asarray(x_np), jnp.asarray(weight_np))
  regrouped = reducer(jnp.asarray(regrouped_np), jnp.asarray(weight_np))
  auto_first = auto_gradient(jnp.asarray(x_np), jnp.asarray(weight_np))
  auto_second = auto_gradient(jnp.asarray(x_np), jnp.asarray(weight_np))
  auto_alt = auto_gradient_alt(jnp.asarray(x_np), jnp.asarray(weight_np))
  auto_regrouped = auto_gradient(
      jnp.asarray(regrouped_np), jnp.asarray(weight_np)
  )

  stock, fixed, stock_rows, fixed_rows, fault_rows = first
  if args.inject_rank_fault:
    stock_rows = fault_rows

  next_weight, moment, variance = jax.jit(_adamw_step)(
      jnp.asarray(weight_np), auto_first
  )
  # Materialize all asynchronous work before reporting a verdict.
  jax.block_until_ready((
      first,
      second,
      alt,
      regrouped,
      auto_first,
      auto_second,
      auto_alt,
      auto_regrouped,
      next_weight,
      moment,
      variance,
  ))

  checks = {
      "stock_repeat_exact": _exact(stock, second[0]),
      "fixed_repeat_exact": _exact(fixed, second[1]),
      "auto_repeat_exact": _exact(auto_first, auto_second),
      "stock_replicas_exact": _rows_exact(stock_rows),
      "fixed_replicas_exact": _rows_exact(fixed_rows),
      "fault_rejected": not _rows_exact(fault_rows),
      "fixed_mesh_order_exact": _exact(fixed, alt[1]),
  }
  observations = {
      "stock_vs_fixed_exact": _exact(stock, fixed),
      "auto_vs_stock_exact": _exact(auto_first, stock),
      "stock_mesh_order_exact": _exact(stock, alt[0]),
      "auto_mesh_order_exact": _exact(auto_first, auto_alt),
      "stock_regroup_exact": _exact(stock, regrouped[0]),
      "fixed_regroup_exact": _exact(fixed, regrouped[1]),
      "auto_regroup_exact": _exact(auto_first, auto_regrouped),
  }

  # The logical outputs are replicated over DP. Materialising them only once
  # is insufficient evidence, so reducer() separately exports one row per DP
  # rank above. The update hashes prove the actual optimizer arithmetic was
  # executed; the replicated-gradient gate is what establishes replica sync.
  update = {
      "gradient_sha256": _sha(auto_first),
      "parameter_sha256": _sha(next_weight),
      "moment_sha256": _sha(moment),
      "variance_sha256": _sha(variance),
  }
  required_checks = all(checks.values())
  decision = (
      "FIXED_TOPOLOGY_STOCK_ADMISSIBLE"
      if required_checks
      else "NOT_ADMITTED"
  )
  if required_checks and not observations["auto_mesh_order_exact"]:
    decision = "FIXED_TOPOLOGY_ONLY_DEVICE_ORDER_SENSITIVE"
  if required_checks and not observations["auto_regroup_exact"]:
    decision += "+BATCH_PLACEMENT_SENSITIVE"

  summary = {
      "dp": args.dp,
      "tp": args.tp,
      "local_samples": args.local_samples,
      "global_samples": global_samples,
      "mesh_ids": mesh_ids,
      "mapping_sha256": hashlib.sha256(
          np.arange(global_samples, dtype=np.int32).tobytes()
      ).hexdigest(),
      "checks": checks,
      "observations": observations,
      "update": update,
      "decision": decision,
  }
  print(
      f"[P32.DP] CONFIG dp={args.dp} tp={args.tp} "
      f"local_samples={args.local_samples} global_samples={global_samples}",
      flush=True,
  )
  print(f"[P32.DP] MESH ids={mesh_ids}", flush=True)
  print(f"[P32.DP] CHECKS {json.dumps(checks, sort_keys=True)}", flush=True)
  print(
      f"[P32.DP] OBSERVATIONS {json.dumps(observations, sort_keys=True)}",
      flush=True,
  )
  print(f"[P32.DP] UPDATE {json.dumps(update, sort_keys=True)}", flush=True)
  print(f"[P32.DP] DECISION {decision}", flush=True)
  print(f"[P32.DP] JSON {json.dumps(summary, sort_keys=True)}", flush=True)
  print(
      f"[P32.DP] VERDICT {'PASS' if required_checks else 'FAIL'}",
      flush=True,
  )
  return 0 if required_checks else 1


if __name__ == "__main__":
  sys.exit(main())
