# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pure numerical comparator for the P66 same-point VJP diagnostic."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np


CAPS = {
    "rel_l2": 4.0e-2,
    "one_minus_cos": 3.2e-4,
    "norm_ratio_error": 4.0e-2,
    "sign_mismatch_rate": 2.0e-2,
}


class OracleContractError(ValueError):
  """Raised when the paired VJP trees do not share one exact contract."""


def unstage_unit_rank(tree, *, endpoint: str):
  """Removes only the proven-unit diagnostic DP staging dimension."""

  def unstage(value):
    if value.ndim < 1 or int(value.shape[0]) != 1:
      raise OracleContractError(
          f"P66 oracle {endpoint} rank staging changed: {value.shape}"
      )
    return jnp.squeeze(value, axis=0)

  return jax.tree.map(unstage, tree)


def compare(reference, candidate, *, endpoint: str, emit=True):
  """Compares one ordinary and checked-VMA pullback at the same inputs."""
  reference_leaves = tuple(jax.tree.leaves(reference))
  candidate_leaves = tuple(jax.tree.leaves(candidate))
  if not reference_leaves or len(reference_leaves) != len(candidate_leaves):
    raise OracleContractError(
        f"P66 oracle {endpoint} tree size changed: "
        f"{len(reference_leaves)} != {len(candidate_leaves)}"
    )
  for index, (expected, actual) in enumerate(
      zip(reference_leaves, candidate_leaves, strict=True)
  ):
    if expected.shape != actual.shape or expected.dtype != actual.dtype:
      raise OracleContractError(
          f"P66 oracle {endpoint} leaf {index} contract changed: "
          f"{expected.shape}/{expected.dtype} != "
          f"{actual.shape}/{actual.dtype}"
      )

  @jax.jit
  def device_metrics(expected, actual):

    def maximum(values):
      result = jnp.asarray(0.0, jnp.float32)
      for value in values:
        result = jnp.maximum(
            result, jnp.max(jnp.abs(value.astype(jnp.float32)))
        )
      return result

    expected_max = maximum(expected)
    actual_max = maximum(actual)
    differences = tuple(
        got.astype(jnp.float32) - want.astype(jnp.float32)
        for want, got in zip(expected, actual, strict=True)
    )
    difference_max = maximum(differences)
    expected_scale = jnp.where(expected_max > 0, expected_max, 1.0)
    actual_scale = jnp.where(actual_max > 0, actual_max, 1.0)
    difference_scale = jnp.where(difference_max > 0, difference_max, 1.0)
    expected_sumsq = jnp.asarray(0.0, jnp.float32)
    actual_sumsq = jnp.asarray(0.0, jnp.float32)
    difference_sumsq = jnp.asarray(0.0, jnp.float32)
    scaled_dot = jnp.asarray(0.0, jnp.float32)
    sign_mismatch = jnp.asarray(0, jnp.int32)
    reference_nonzero = jnp.asarray(0, jnp.int32)
    elements = jnp.asarray(0, jnp.int32)
    exact = jnp.asarray(True, jnp.bool_)
    finite = jnp.asarray(True, jnp.bool_)
    live_reference = []
    dead_candidate = []
    leaf_max_reference = []
    leaf_max_difference = []
    for want, got, difference in zip(
        expected, actual, differences, strict=True
    ):
      want32 = want.astype(jnp.float32)
      got32 = got.astype(jnp.float32)
      expected_sumsq += jnp.sum(jnp.square(want32 / expected_scale))
      actual_sumsq += jnp.sum(jnp.square(got32 / actual_scale))
      difference_sumsq += jnp.sum(
          jnp.square(difference / difference_scale)
      )
      scaled_dot += jnp.sum(
          (want32 / expected_scale) * (got32 / actual_scale)
      )
      nonzero = want32 != 0
      reference_nonzero += jnp.count_nonzero(nonzero)
      sign_mismatch += jnp.count_nonzero(
          nonzero & (jnp.sign(want32) != jnp.sign(got32))
      )
      elements += want.size
      exact &= jnp.array_equal(want, got)
      finite &= jnp.all(jnp.isfinite(want32))
      finite &= jnp.all(jnp.isfinite(got32))
      want_nonzero = jnp.any(nonzero)
      live_reference.append(want_nonzero)
      dead_candidate.append(want_nonzero & ~jnp.any(got32 != 0))
      leaf_max_reference.append(jnp.max(jnp.abs(want32)))
      leaf_max_difference.append(jnp.max(jnp.abs(difference)))
    return {
        "expected_max": expected_max,
        "actual_max": actual_max,
        "difference_max": difference_max,
        "expected_sumsq": expected_sumsq,
        "actual_sumsq": actual_sumsq,
        "difference_sumsq": difference_sumsq,
        "scaled_dot": scaled_dot,
        "sign_mismatch": sign_mismatch,
        "reference_nonzero": reference_nonzero,
        "elements": elements,
        "exact": exact,
        "finite": finite,
        "live_reference": jnp.stack(live_reference),
        "dead_candidate": jnp.stack(dead_candidate),
        "leaf_max_reference": jnp.stack(leaf_max_reference),
        "leaf_max_difference": jnp.stack(leaf_max_difference),
    }

  host = jax.device_get(device_metrics(reference_leaves, candidate_leaves))
  expected_norm = float(host["expected_max"]) * float(
      np.sqrt(np.float64(host["expected_sumsq"]))
  )
  actual_norm = float(host["actual_max"]) * float(
      np.sqrt(np.float64(host["actual_sumsq"]))
  )
  difference_norm = float(host["difference_max"]) * float(
      np.sqrt(np.float64(host["difference_sumsq"]))
  )
  if expected_norm == 0.0:
    rel_l2 = 0.0 if actual_norm == 0.0 else float("inf")
    one_minus_cos = 0.0 if actual_norm == 0.0 else float("inf")
    norm_ratio_error = 0.0 if actual_norm == 0.0 else float("inf")
  elif actual_norm == 0.0:
    rel_l2 = one_minus_cos = norm_ratio_error = 1.0
  else:
    rel_l2 = difference_norm / expected_norm
    denominator = float(
        np.sqrt(
            np.float64(host["expected_sumsq"])
            * np.float64(host["actual_sumsq"])
        )
    )
    cosine = float(host["scaled_dot"]) / denominator
    cosine = max(-1.0, min(1.0, cosine))
    one_minus_cos = max(0.0, 1.0 - cosine)
    norm_ratio_error = abs(actual_norm / expected_norm - 1.0)
  reference_nonzero = int(host["reference_nonzero"])
  sign_mismatch_rate = int(host["sign_mismatch"]) / max(
      1, reference_nonzero
  )
  leaf_scales = np.asarray(host["leaf_max_difference"], np.float64) / np.maximum(
      np.asarray(host["leaf_max_reference"], np.float64),
      np.finfo(np.float64).tiny,
  )
  worst_leaf = int(np.argmax(leaf_scales))
  metrics = {
      "rel_l2": rel_l2,
      "one_minus_cos": one_minus_cos,
      "norm_ratio_error": norm_ratio_error,
      "sign_mismatch_rate": sign_mismatch_rate,
  }
  passed = (
      bool(host["finite"])
      and not bool(np.any(host["dead_candidate"]))
      and all(
          np.isfinite(metrics[name]) and metrics[name] <= limit
          for name, limit in CAPS.items()
      )
  )
  record = {
      "schema": "canon-p66-same-point-vjp-oracle-v1",
      "endpoint": endpoint,
      "verdict": "PASS" if passed else "FAIL",
      "leaf_count": len(reference_leaves),
      "elements": int(host["elements"]),
      "finite": bool(host["finite"]),
      "array_exact": bool(host["exact"]),
      "live_reference_leaves": int(np.count_nonzero(host["live_reference"])),
      "dead_candidate_leaves": int(np.count_nonzero(host["dead_candidate"])),
      "reference_nonzero_elements": reference_nonzero,
      "sign_mismatch_elements": int(host["sign_mismatch"]),
      "reference_norm": expected_norm,
      "candidate_norm": actual_norm,
      "difference_norm": difference_norm,
      "worst_leaf_index": worst_leaf,
      "worst_leaf_scaled_max_error": float(leaf_scales[worst_leaf]),
      "metrics": metrics,
      "caps": dict(CAPS),
  }
  if emit:
    print(
        "[P66.ORACLE.ENDPOINT] "
        + json.dumps(record, sort_keys=True, separators=(",", ":")),
        flush=True,
    )
  return record


def negative_control():
  """Proves the same comparator rejects a normal-value perturbation."""
  reference = (jnp.asarray([1.0, -2.0, 3.0], jnp.float32),)
  candidate = (jnp.asarray([1.0, -1.0, 3.0], jnp.float32),)
  record = compare(
      reference, candidate, endpoint="negative_control", emit=False
  )
  if record["verdict"] != "FAIL":
    raise OracleContractError(
        f"P66 oracle negative control did not fire: {record}"
    )
  print(
      "[P66.ORACLE.NEGATIVE] detected=1 perturbation=normal_value ",
      f"rel_l2={record['metrics']['rel_l2']}",
      flush=True,
  )
  return True
