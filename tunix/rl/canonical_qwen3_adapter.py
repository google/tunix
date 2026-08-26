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

"""Functional building blocks for the canonical Qwen3 engine adapter.

The rollout weight-sync utility mutates the destination engine state.  That is
correct for serving, but it cannot be the differentiable path used by the
trainer loss.  This module applies the same mapping transforms without writing
the target state and returns leaves in the target engine state's flat order.

The pure weight-map helpers implement the A1 contract.  The live adapter adds
the separately admitted model/cache/metadata contract and reuses the engine's
exact processed-logprob call boundary; neither layer mutates serving state.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import hashlib
import importlib
import json
import os
import re
import threading
import time
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from tunix.generate import utils as generate_utils
from tunix.rl import canonical_logsoftmax
from tunix.rl import dp_training
from tunix.rl import dp_workloads
from tunix.rl import deepswe_contract
from tunix.rl import gsm8k_xprof
from tunix.rl import p64_training_capsule
from tunix.rl import p66_vjp_oracle
from tunix.rl import p38_frozenlake_replay
from tunix.sft import utils as sft_utils


class FunctionalMappingError(ValueError):
  """Raised when a trainer-to-engine weight map is not a bijection."""


def _p62_numeric_debug_enabled() -> bool:
  """Parses the exact default-off Attempt-7 numerical observer."""
  value = os.environ.get("CANON_P62_BACKWARD_NUMERIC_DEBUG", "")
  if value not in ("", "0", "1"):
    raise FunctionalMappingError(
        "CANON_P62_BACKWARD_NUMERIC_DEBUG must be unset/0/1, "
        f"got {value!r}"
    )
  return value == "1"


def _p64_numeric_debug_enabled() -> bool:
  """Parses the exact default-off P45 first-red observer."""
  value = os.environ.get("CANON_P64_P45_NUMERIC_DEBUG", "")
  if value not in ("", "0", "1"):
    raise FunctionalMappingError(
        "CANON_P64_P45_NUMERIC_DEBUG must be unset/0/1, "
        f"got {value!r}"
    )
  return value == "1"


def _backward_numeric_debug_mode() -> str:
  p62 = _p62_numeric_debug_enabled()
  p64 = _p64_numeric_debug_enabled()
  if p62 and p64:
    raise FunctionalMappingError("P62 and P64 numerical observers conflict")
  return "p62" if p62 else "p64" if p64 else ""


def _numeric_debug_identity(mode: str) -> tuple[str, str]:
  if mode == "p62":
    return "P62", "canon-p62"
  if mode == "p64":
    return "P64", "canon-p64"
  raise FunctionalMappingError(f"unknown numerical debug mode: {mode!r}")


def _p62_emit_tree_receipt(
    *,
    stage: str,
    group: int,
    group_count: int,
    tree: Any,
    ranked: bool = False,
    force: bool = False,
    mode: str = "p62",
) -> dict[str, Any]:
  """Emits one compact first-red receipt and fails on non-finite data."""
  marker, schema_prefix = _numeric_debug_identity(mode)
  receipt = sft_utils.tree_numeric_receipt(tree, ranked=ranked)
  should_emit = (
      force
      or group in (0, group_count - 1)
      or not receipt["all_finite"]
      or not receipt["naive_norm_finite"]
  )
  record = {
      "schema": f"{schema_prefix}-tree-numeric-v1",
      "stage": stage,
      "group": group,
      "groups": group_count,
      **receipt,
  }
  if should_emit:
    print(
        f"[{marker}.NUMERIC] "
        + json.dumps(record, sort_keys=True, separators=(",", ":")),
        flush=True,
    )
  if not receipt["all_finite"]:
    raise FunctionalMappingError(
        f"{marker} first non-finite numerical boundary: "
        f"stage={stage} group={group} "
        f"first={receipt['first_nonfinite']} "
        f"rank={receipt.get('first_nonfinite_rank')}"
    )
  return record


def _p62_emit_loss_receipt(
    *, loss_output: Any, contract: Any, mode: str = "p62"
) -> dict[str, Any]:
  """Validates the frozen loss denominator and compact GRPO metrics."""
  marker, schema_prefix = _numeric_debug_identity(mode)

  def metric_value(value):
    compute = getattr(value, "compute", None)
    return compute() if callable(compute) else value

  metrics = {
      "advantage_abs_mean": metric_value(
          loss_output.aux_metrics["advantage/abs_mean"]
      ),
      "advantage_max": metric_value(
          loss_output.aux_metrics["advantage/max"]
      ),
      "advantage_min": metric_value(
          loss_output.aux_metrics["advantage/min"]
      ),
      "effective_rows": metric_value(
          loss_output.aux_metrics["loss/effective_rows"]
      ),
      "is_ratio_max": metric_value(
          loss_output.aux_metrics["is_ratio/max"]
      ),
      "is_ratio_min": metric_value(
          loss_output.aux_metrics["is_ratio/min"]
      ),
      "loss": loss_output.primary_loss.compute(),
      "loss_denominator": loss_output.primary_loss.denominator,
      "loss_scale": loss_output.primary_loss.compute_scale(),
      "valid_tokens": metric_value(
          loss_output.aux_metrics["loss/valid_tokens"]
      ),
  }
  host = {
      name: float(np.asarray(value))
      for name, value in jax.device_get(metrics).items()
  }
  expected_denominator = float(contract.global_trajectories)
  expected_scale = float(np.float32(1.0 / expected_denominator))
  record = {
      "schema": f"{schema_prefix}-loss-scale-v1",
      "stage": "loss_scale",
      "dp": int(contract.dp_size),
      "tp": int(contract.tp_size),
      "global_trajectories": int(contract.global_trajectories),
      "local_trajectories": int(contract.local_trajectories),
      "gradient_groups": int(contract.local_trajectories),
      "global_M": int(contract.dp_size * 256),
      "local_M": 256,
      "expected_accumulator_denominator": int(contract.local_trajectories),
      "expected_streamed_multiplier": float(np.float32(
          expected_scale * contract.local_trajectories
      )),
      **host,
  }
  print(
      f"[{marker}.NUMERIC] "
      + json.dumps(record, sort_keys=True, separators=(",", ":")),
      flush=True,
  )
  finite = all(np.isfinite(value) for value in host.values())
  if (
      not finite
      or host["loss_denominator"] != expected_denominator
      or host["loss_scale"] != expected_scale
      or host["effective_rows"] <= 0.0
      or host["valid_tokens"] <= 0.0
  ):
    raise FunctionalMappingError(
        f"{marker} loss-scale contract failed: {record}"
    )
  return record


def _p66_tp4_arm() -> str:
  """Returns the exact default-off one-host TP4 discriminator arm."""
  value = os.environ.get("CANON_P66_BACKWARD_ARM", "")
  return value if value in (
      "tp4-serial",
      "tp4-p59-old",
      "tp4-p59",
      "tp4-gather-off",
      "tp4-vma-oracle",
  ) else ""


def _p66_emit_layerwise_profile(segmented, engine_gradients, *, arm: str):
  """Emits one full-depth max-abs fingerprint for a P66 group-0 VJP."""
  leaves = tuple(engine_gradients)
  groups = (
      ("embed", tuple(segmented._embed_full_indices)),  # pylint: disable=protected-access
      *tuple(
          (f"layer_{index}", tuple(indices))
          for index, indices in enumerate(
              segmented._local_layer_full_indices  # pylint: disable=protected-access
          )
      ),
      ("norm", tuple(segmented._norm_full_indices)),  # pylint: disable=protected-access
      ("head", tuple(segmented._head_full_indices)),  # pylint: disable=protected-access
  )
  if any(not indices for _, indices in groups):
    raise FunctionalMappingError("P66 layerwise profile has an empty group")

  def profile(values):
    maxima = []
    for _, indices in groups:
      maximum = jnp.max(jnp.abs(values[indices[0]].astype(jnp.float32)))
      for index in indices[1:]:
        maximum = jnp.maximum(
            maximum,
            jnp.max(jnp.abs(values[index].astype(jnp.float32))),
        )
      maxima.append(maximum)
    return jnp.stack(maxima)

  maxima = np.asarray(jax.device_get(jax.jit(profile)(leaves)), np.float32)
  record = {
      "schema": "canon-p66-full-depth-profile-v1",
      "arm": arm,
      "stage": "engine_vjp_group0",
      "components": {
          label: float(maximum)
          for (label, _), maximum in zip(groups, maxima, strict=True)
      },
  }
  print(
      "[P66.TP4.PROFILE] "
      + json.dumps(record, sort_keys=True, separators=(",", ":")),
      flush=True,
  )
  return record


def _p66_emit_row_cotangent_profile(
    profiles, *, arm: str, host_n_real, sequence_bucket: int
):
  """Separates real/padding residual scale and cotangent in a P66 replay."""
  host_profiles = jax.device_get(tuple(profiles))
  records = []

  def distribution(values):
    values = np.asarray(values, np.float32).reshape(-1)
    finite = values[np.isfinite(values)]
    if not finite.size:
      return {"min": None, "p01": None, "p50": None, "max": None}
    return {
        "min": float(np.min(finite)),
        "p01": float(np.percentile(finite, 1.0)),
        "p50": float(np.percentile(finite, 50.0)),
        "max": float(np.max(finite)),
    }

  for chunk_index, layer_index, hidden_rms, dhidden_max in host_profiles:
    hidden_rms = np.asarray(hidden_rms, np.float32)
    dhidden_max = np.asarray(dhidden_max, np.float32)
    if hidden_rms.shape != dhidden_max.shape or hidden_rms.shape != (
        len(host_n_real), sequence_bucket
    ):
      raise FunctionalMappingError(
          "P66 row profile shape mismatch: "
          f"hidden={hidden_rms.shape} dhidden={dhidden_max.shape}"
      )
    chunk_start = int(chunk_index) * sequence_bucket
    q_len = np.clip(
        np.asarray(host_n_real, np.int32) - chunk_start,
        0,
        sequence_bucket,
    )
    row = np.arange(sequence_bucket, dtype=np.int32)[None, :]
    real_mask = row < q_len[:, None]
    padding_mask = ~real_mask
    real_dhidden = dhidden_max[real_mask]
    padding_dhidden = dhidden_max[padding_mask]
    padding_nonzero = int(np.count_nonzero(padding_dhidden != 0.0))
    record = {
        "schema": "canon-p66-row-cotangent-v1",
        "arm": arm,
        "chunk": int(chunk_index),
        "layer": int(layer_index),
        "real_rows": int(np.count_nonzero(real_mask)),
        "padding_rows": int(np.count_nonzero(padding_mask)),
        "padding_token_id": 0,
        "real_hidden_rms": distribution(hidden_rms[real_mask]),
        "padding_hidden_rms": distribution(hidden_rms[padding_mask]),
        "real_dhidden_max": distribution(real_dhidden),
        "padding_dhidden_max": distribution(padding_dhidden),
        "real_dhidden_nonzero_rows": int(
            np.count_nonzero(real_dhidden != 0.0)
        ),
        "padding_dhidden_nonzero_rows": padding_nonzero,
        "padding_dhidden_nonfinite_rows": int(
            np.count_nonzero(~np.isfinite(padding_dhidden))
        ),
        "padding_hidden_below_0p05_rows": int(
            np.count_nonzero(hidden_rms[padding_mask] < 0.05)
        ),
    }
    records.append(record)
    print(
        "[P66.TP4.ROWS] "
        + json.dumps(record, sort_keys=True, separators=(",", ":")),
        flush=True,
    )

  nonzero = [
      record
      for record in records
      if record["padding_dhidden_nonzero_rows"] > 0
  ]
  padding_hidden_minima = [
      record["padding_hidden_rms"]["min"]
      for record in records
      if record["padding_hidden_rms"]["min"] is not None
  ]
  summary = {
      "schema": "canon-p66-row-cotangent-summary-v1",
      "arm": arm,
      "records": len(records),
      "chunks": len({record["chunk"] for record in records}),
      "layers": sorted({record["layer"] for record in records}),
      "padding_row_layer_nonzero": int(sum(
          record["padding_dhidden_nonzero_rows"] for record in records
      )),
      "padding_row_layer_nonfinite": int(sum(
          record["padding_dhidden_nonfinite_rows"] for record in records
      )),
      "padding_hidden_rms_min": (
          float(min(padding_hidden_minima)) if padding_hidden_minima else None
      ),
      "first_nonzero_padding_cotangent": (
          {
              "chunk": nonzero[0]["chunk"],
              "layer": nonzero[0]["layer"],
              "rows": nonzero[0]["padding_dhidden_nonzero_rows"],
              "max_abs": nonzero[0]["padding_dhidden_max"]["max"],
          }
          if nonzero
          else None
      ),
  }
  print(
      "[P66.TP4.ROWS.SUMMARY] "
      + json.dumps(summary, sort_keys=True, separators=(",", ":")),
      flush=True,
  )
  return summary


def _xprof_jit(fun, *, module_name: str, scope_name: str, **jit_kwargs):
  """Adds compact module/op labels only for an explicit XProf capture.

  The disabled route is exactly ``jax.jit(fun, ...)``.  The enabled route
  changes only JAX source metadata: ``module_name`` names the executable on
  XProf's XLA Modules line, while ``scope_name`` prefixes the HLO operation
  source stack used by Trace Viewer and HLO Op Profile.
  """
  try:
    enabled = gsm8k_xprof.labels_enabled()
  except ValueError as exc:
    raise FunctionalMappingError(str(exc)) from exc
  if not enabled:
    return jax.jit(fun, **jit_kwargs)
  if not re.fullmatch(r"[a-z0-9_]+", module_name):
    raise FunctionalMappingError(
        f"invalid XProf module label {module_name!r}"
    )
  if not re.fullmatch(r"[a-z0-9_./-]+", scope_name):
    raise FunctionalMappingError(
        f"invalid XProf operation scope {scope_name!r}"
    )
  labeled = jax.named_call(fun, name=scope_name)
  labeled.__name__ = module_name
  labeled.__qualname__ = module_name
  return jax.jit(labeled, **jit_kwargs)


def _p59_xprof_backward_directory(
    *, workload_name: str, dp_size: int, tp_size: int, rank_parallel: bool
) -> str:
  """Validates the profile-only one-group P59 backward capture."""
  directory = os.environ.get("CANON_P59_XPROF_BACKWARD_DIR", "")
  if not directory:
    return ""
  if os.environ.get("CANON_XPROF_DIR", ""):
    raise FunctionalMappingError(
        "P59 narrow backward XProf cannot nest inside CANON_XPROF_DIR"
    )
  if (
      workload_name != "gsm8k-p59-dp4-tp1"
      or (dp_size, tp_size) != (4, 1)
      or not rank_parallel
  ):
    raise FunctionalMappingError(
        "P59 narrow backward XProf requires the exact DP4xTP1 candidate"
    )
  if os.environ.get("CANON_XPROF_LABELS", "") != "1":
    raise FunctionalMappingError(
        "P59 narrow backward XProf requires CANON_XPROF_LABELS=1"
    )
  if (
      os.environ.get("CANON_XPROF_HOST_TRACER", "1") != "1"
      or os.environ.get("CANON_XPROF_PYTHON_TRACER", "0") != "0"
  ):
    raise FunctionalMappingError(
        "P59 narrow backward XProf requires host tracer 1 and Python tracer 0"
    )
  return directory


@functools.partial(jax.jit, static_argnums=2)
def fused_micro_scale(tree, scale, count):
  """Whole-tree microbatch gradient scaling in one dispatch.

  Keeps the eager path's exact per-leaf expression -- (value * scale) *
  asarray(float(count), value.dtype), two ordered multiplies and a cast --
  as one program instead of ~three tiny launches per leaf. The pinned
  --xla_allow_excess_precision=false forbids float reassociation, so the
  compiled result keeps the eager rounding order; the 51/51 alignment
  gate is the judge regardless. CANON_FUSED_TREE_OPS gates the call.
  """
  return jax.tree.map(
      lambda value: value * scale * jnp.asarray(float(count), value.dtype),
      tree,
  )


class P35ReplayStageProbeComplete(RuntimeError):
  """Stops the default-off P35.3c probe without a numerical verdict."""


def _safe_sharding_constraint(value, sharding):
  if sharding is None or value is None:
    return value
  if hasattr(value, "sharding") and value.sharding == sharding:
    return value
  try:
    return jax.reshard(value, sharding)
  except Exception:
    try:
      return jax.lax.with_sharding_constraint(value, sharding)
    except Exception:
      return value


def _manual_axis_partition_spec(
    value, axis_name: str, manual_axes: frozenset[str] | None = None
):
  """Keeps admitted manual mesh axes and leaves every other axis automatic."""
  sharding = getattr(value, "sharding", None)
  if not isinstance(sharding, jax.sharding.NamedSharding):
    return jax.sharding.PartitionSpec()
  admitted = frozenset((axis_name,)) if manual_axes is None else manual_axes
  if axis_name not in admitted:
    raise FunctionalMappingError(
        f"manual axis set omits required data axis {axis_name!r}"
    )

  def keep_axis(entry):
    names = () if entry is None else (
        (entry,) if isinstance(entry, str) else tuple(entry)
    )
    kept = tuple(name for name in names if name in admitted)
    if not kept:
      return None
    return kept[0] if len(kept) == 1 else kept

  return jax.sharding.PartitionSpec(
      *(keep_axis(entry) for entry in tuple(sharding.spec))
  )


def _manual_axis_specs(
    tree, axis_name: str, manual_axes: frozenset[str] | None = None
):
  return jax.tree.map(
      lambda value: _manual_axis_partition_spec(
          value, axis_name, manual_axes
      ),
      tree,
  )


def _rank_staged_specs(
    tree, axis_name: str, manual_axes: frozenset[str] | None = None
):
  """Prepends the staged-DP row while retaining admitted TP placement."""

  def staged_spec(value):
    sharding = getattr(value, "sharding", None)
    if not isinstance(sharding, jax.sharding.NamedSharding):
      return jax.sharding.PartitionSpec(axis_name)
    data_axis, _ = _p59_mesh_roles(
        sharding.mesh, "P59 staged rank gradient"
    )
    if data_axis != axis_name:
      raise FunctionalMappingError(
          "P59 staged rank gradient data axis changed: "
          f"{data_axis!r} != {axis_name!r}"
      )
    original_spec = _manual_axis_partition_spec(
        value, axis_name, manual_axes
    )
    if data_axis in _p59_partition_axes(original_spec):
      raise FunctionalMappingError(
          "P59 staged rank gradient requires DP-replicated parameters, got "
          f"{original_spec}"
      )
    return jax.sharding.PartitionSpec(axis_name, *tuple(original_spec))

  return jax.tree.map(staged_spec, tree)


def _rank_local_leading_specs(
    tree,
    axis_name: str,
    axis_size: int,
    label: str,
    manual_axes: frozenset[str] | None = None,
):
  """Partitions semantic per-rank rows even if the input arrived replicated."""
  axis_size = int(axis_size)

  def spec(value):
    if getattr(value, "ndim", 0) < 1:
      return jax.sharding.PartitionSpec()
    if int(value.shape[0]) % axis_size:
      raise FunctionalMappingError(
          f"{label} leading rows are not divisible by {axis_name}: "
          f"shape={value.shape} size={axis_size}"
      )
    retained = list(
        _manual_axis_partition_spec(value, axis_name, manual_axes)
    )
    retained.extend(None for _ in range(value.ndim - len(retained)))
    first = retained[0] if retained else None
    first_axes = () if first is None else (
        (first,) if isinstance(first, str) else tuple(first)
    )
    retained[0] = (
        axis_name
        if not first_axes
        else tuple(dict.fromkeys((axis_name, *first_axes)))
    )
    return jax.sharding.PartitionSpec(*retained)

  return jax.tree.map(spec, tree)


def _named_sharding_mesh(tree, axis_name: str | None, label: str):
  """Returns the single NamedSharding mesh carried by an array tree."""
  meshes = []
  for value in jax.tree.leaves(tree):
    sharding = getattr(value, "sharding", None)
    if not isinstance(sharding, jax.sharding.NamedSharding):
      continue
    if axis_name is not None and axis_name not in sharding.mesh.axis_names:
      raise FunctionalMappingError(
          f"{label} sharding mesh omits axis {axis_name!r}"
      )
    if not any(sharding.mesh == mesh for mesh in meshes):
      meshes.append(sharding.mesh)
  if len(meshes) != 1:
    raise FunctionalMappingError(
        f"{label} requires exactly one NamedSharding mesh, got {len(meshes)}"
    )
  return meshes[0]


def _p59_replicated_data_mesh(tree, label: str):
  """Returns one registered replicated-DP mesh and its actual data axis."""
  mesh = _named_sharding_mesh(tree, None, label)
  axes = tuple(mesh.axis_names)
  if axes == ("data", "model"):
    return mesh, "data"
  if axes == ("dp", "tp"):
    return mesh, "dp"
  raise FunctionalMappingError(
      f"{label} requires replicated DP mesh data/model or dp/tp, got {axes}"
  )


def _p59_mesh_roles(mesh, label: str):
  """Returns the data/model roles for one admitted trainer or engine mesh."""
  axes = tuple(mesh.axis_names)
  if axes == ("dp", "tp"):
    return "dp", "tp"
  if axes == ("data", "model"):
    return "data", "model"
  engine_axes = (
      "data",
      "attn_dp",
      "attn_dp_expert",
      "expert",
      "model",
      "dcp",
  )
  if axes == engine_axes:
    non_unit_aux = {
        axis: int(mesh.shape[axis])
        for axis in ("attn_dp", "attn_dp_expert", "expert", "dcp")
        if int(mesh.shape[axis]) != 1
    }
    if non_unit_aux:
      raise FunctionalMappingError(
          f"{label} engine mesh has non-unit auxiliary axes: {non_unit_aux}"
      )
    return "data", "model"
  raise FunctionalMappingError(
      f"{label} has unsupported mesh axes {axes}"
  )


def _p59_engine_data_model_mesh(mesh, label: str):
  """Returns the exact engine devices as a two-axis data/model mesh."""
  data_axis, model_axis = _p59_mesh_roles(mesh, label)
  data_size = int(mesh.shape[data_axis])
  model_size = int(mesh.shape[model_axis])
  devices = np.asarray(mesh.devices).reshape(data_size, model_size)
  return jax.sharding.Mesh(devices, ("data", "model"))


def _p59_manual_rank_axes(mesh, data_axis: str, label: str):
  """Makes unit TP manual when an explicit TP1 spec must be retained."""
  actual_data_axis, model_axis = _p59_mesh_roles(mesh, label)
  if actual_data_axis != data_axis:
    raise FunctionalMappingError(
        f"{label} data axis changed: {actual_data_axis!r} != {data_axis!r}"
    )
  manual_axes = {data_axis}
  if int(mesh.shape[model_axis]) == 1:
    manual_axes.add(model_axis)
  return manual_axes


def _p59_restore_physically_equal_staged_specs(
    trainer_state, staged_gradient, data_axis: str
):
  """Restores staged metadata only after exact physical equivalence."""
  if jax.tree.structure(trainer_state) != jax.tree.structure(staged_gradient):
    raise FunctionalMappingError(
        "P59 staged-spec restoration tree differs from trainer state"
    )

  def restore(state_value, staged_value):
    state_sharding = getattr(state_value, "sharding", None)
    staged_sharding = getattr(staged_value, "sharding", None)
    if not isinstance(state_sharding, jax.sharding.NamedSharding):
      raise FunctionalMappingError(
          "P59 staged-spec restoration requires NamedSharding parameters"
      )
    if not isinstance(staged_sharding, jax.sharding.NamedSharding):
      raise FunctionalMappingError(
          "P59 staged-spec restoration requires NamedSharding gradients"
      )
    mesh = state_sharding.mesh
    actual_data_axis, model_axis = _p59_mesh_roles(
        mesh, "P59 staged-spec restoration"
    )
    if actual_data_axis != data_axis:
      raise FunctionalMappingError(
          "P59 staged-spec restoration data axis changed: "
          f"{actual_data_axis!r} != {data_axis!r}"
      )
    expected_shape = (int(mesh.shape[data_axis]),) + tuple(state_value.shape)
    if (
        staged_value.shape != expected_shape
        or staged_value.dtype != jnp.float32
    ):
      raise FunctionalMappingError(
          "P59 staged-spec restoration shape/dtype changed: "
          f"{staged_value.shape}/{staged_value.dtype} != "
          f"{expected_shape}/{jnp.float32}"
      )
    expected_sharding = jax.sharding.NamedSharding(
        mesh,
        jax.sharding.PartitionSpec(
            data_axis, *tuple(state_sharding.spec)
        ),
    )
    if staged_sharding == expected_sharding:
      return staged_value
    if staged_sharding.mesh != mesh:
      raise FunctionalMappingError(
          "P59 staged-spec restoration is not a same-mesh difference"
      )
    actual_spec = tuple(staged_sharding.spec)
    if not actual_spec or actual_spec[0] != data_axis:
      raise FunctionalMappingError(
          "P59 staged-spec restoration requires leading-DP placement, got "
          f"{staged_sharding.spec}"
      )
    if _p59_partition_axes(staged_sharding.spec) - {data_axis, model_axis}:
      raise FunctionalMappingError(
          "P59 staged-spec restoration found an unexpected mesh axis: "
          f"{staged_sharding.spec}"
      )
    if (
        staged_sharding.devices_indices_map(staged_value.shape)
        != expected_sharding.devices_indices_map(staged_value.shape)
    ):
      raise FunctionalMappingError(
          "P59 staged-spec restoration placements are not physically equal"
      )
    return jax.device_put(staged_value, expected_sharding)

  return jax.tree.map(restore, trainer_state, staged_gradient)


def _p59_align_to_mesh(tree, target_mesh, label: str):
  """Relabels compatible engine shardings onto one trainer DP/TP mesh."""
  target_data, target_model = _p59_mesh_roles(target_mesh, label)
  target_devices = tuple(target_mesh.devices.flat)

  def align(value):
    sharding = getattr(value, "sharding", None)
    if not isinstance(sharding, jax.sharding.NamedSharding):
      return value
    source_mesh = sharding.mesh
    if source_mesh == target_mesh:
      return value
    source_data, source_model = _p59_mesh_roles(source_mesh, label)
    if tuple(source_mesh.devices.flat) != target_devices:
      raise FunctionalMappingError(
          f"{label} trainer and engine device orders differ"
      )
    if (
        int(source_mesh.shape[source_data])
        != int(target_mesh.shape[target_data])
        or int(source_mesh.shape[source_model])
        != int(target_mesh.shape[target_model])
    ):
      raise FunctionalMappingError(
          f"{label} trainer and engine DP/TP dimensions differ"
      )

    def translate(entry):
      if entry is None:
        return None
      names = (entry,) if isinstance(entry, str) else tuple(entry)
      translated = []
      for axis in names:
        if axis == source_data:
          mapped = target_data
        elif axis == source_model:
          mapped = target_model
        elif int(source_mesh.shape[axis]) == 1:
          continue
        else:
          raise FunctionalMappingError(
              f"{label} cannot translate non-unit mesh axis {axis!r}"
          )
        if mapped not in translated:
          translated.append(mapped)
      if not translated:
        return None
      return translated[0] if len(translated) == 1 else tuple(translated)

    translated_spec = jax.sharding.PartitionSpec(
        *(translate(entry) for entry in tuple(sharding.spec))
    )
    return jax.device_put(
        value, jax.sharding.NamedSharding(target_mesh, translated_spec)
    )

  return jax.tree.map(align, tree)


def _p59_partition_head_cotangent(dlogits, trainer_mesh, label: str):
  """Restores the TP-local vocabulary boundary before the P59 head VJP."""
  if getattr(dlogits, "ndim", None) != 2:
    raise FunctionalMappingError(
        f"{label} requires rank-2 logits cotangent, got "
        f"{getattr(dlogits, 'shape', None)}"
    )
  data_axis, model_axis = _p59_mesh_roles(trainer_mesh, label)
  data_size = int(trainer_mesh.shape[data_axis])
  model_size = int(trainer_mesh.shape[model_axis])
  if int(dlogits.shape[0]) % data_size:
    raise FunctionalMappingError(
        f"{label} rows are not divisible by {data_axis}: "
        f"shape={dlogits.shape} size={data_size}"
    )
  if int(dlogits.shape[1]) % model_size:
    raise FunctionalMappingError(
        f"{label} vocabulary is not divisible by {model_axis}: "
        f"shape={dlogits.shape} size={model_size}"
    )
  aligned = _p59_align_to_mesh(dlogits, trainer_mesh, label)
  target = jax.sharding.NamedSharding(
      trainer_mesh,
      jax.sharding.PartitionSpec(data_axis, model_axis),
  )
  return jax.device_put(aligned, target)


def _p59_align_serial_gradient_to_trainer_state(
    trainer_state, gradient, label: str
):
  """Relabels a TP1 serial report gradient onto exact trainer shardings.

  The DP4 proxy's trainer uses ``dp/tp`` while its engine uses the equivalent
  six-axis ``data/.../model`` mesh.  The serial mapping adjoint can therefore
  return a trainer-shaped tree whose arrays still carry the engine vocabulary.
  This bridge permits metadata-only relabeling when every physical placement
  is identical; it rejects data-sharded gradients and any non-unit TP repair.
  """
  if jax.tree.structure(trainer_state) != jax.tree.structure(gradient):
    raise FunctionalMappingError(
        f"{label} gradient tree differs from trainer state"
    )
  trainer_mesh, data_axis = _p59_replicated_data_mesh(
      trainer_state, label
  )
  aligned = _p59_align_to_mesh(gradient, trainer_mesh, label)
  actual_data_axis, model_axis = _p59_mesh_roles(trainer_mesh, label)
  if actual_data_axis != data_axis:
    raise FunctionalMappingError(
        f"{label} trainer data axis changed: "
        f"{actual_data_axis!r} != {data_axis!r}"
    )

  def restore(state_value, gradient_value):
    state_sharding = getattr(state_value, "sharding", None)
    gradient_sharding = getattr(gradient_value, "sharding", None)
    if not isinstance(state_sharding, jax.sharding.NamedSharding):
      raise FunctionalMappingError(
          f"{label} trainer state requires NamedSharding leaves"
      )
    if not isinstance(gradient_sharding, jax.sharding.NamedSharding):
      raise FunctionalMappingError(
          f"{label} gradient requires NamedSharding leaves"
      )
    if (
        state_sharding.mesh != trainer_mesh
        or gradient_sharding.mesh != trainer_mesh
    ):
      raise FunctionalMappingError(
          f"{label} relabeled leaves do not share the trainer mesh"
      )
    if (
        gradient_value.shape != state_value.shape
        or gradient_value.dtype != jnp.float32
    ):
      raise FunctionalMappingError(
          f"{label} shape/dtype changed: "
          f"{gradient_value.shape}/{gradient_value.dtype} != "
          f"{state_value.shape}/{jnp.float32}"
      )
    expected_sharding = jax.sharding.NamedSharding(
        trainer_mesh, state_sharding.spec
    )
    if gradient_sharding == expected_sharding:
      return gradient_value
    if int(trainer_mesh.shape[model_axis]) != 1:
      raise FunctionalMappingError(f"{label} sharding repair is TP1-only")
    if (
        _p59_partition_axes(gradient_sharding.spec) - {model_axis}
        or _p59_partition_axes(expected_sharding.spec) - {model_axis}
    ):
      raise FunctionalMappingError(
          f"{label} requires DP-replicated TP-only parameter placement"
      )
    if (
        gradient_sharding.devices_indices_map(gradient_value.shape)
        != expected_sharding.devices_indices_map(gradient_value.shape)
    ):
      raise FunctionalMappingError(
          f"{label} placements are not physically identical"
      )
    return jax.device_put(gradient_value, expected_sharding)

  return jax.tree.map(restore, trainer_state, aligned), data_axis


_P59_NESTED_SHARD_MAP_LOCK = threading.RLock()
_P59_ENGINE_MESH_AXES = (
    "data",
    "attn_dp",
    "attn_dp_expert",
    "expert",
    "model",
    "dcp",
)


def _p59_partition_axes(specs):
  """Returns every named mesh axis referenced by a PartitionSpec tree."""
  axes = set()

  def visit(value):
    if isinstance(value, jax.sharding.PartitionSpec):
      for entry in value:
        if entry is None:
          continue
        if isinstance(entry, str):
          axes.add(entry)
        else:
          axes.update(entry)
      return
    if isinstance(value, Mapping):
      for item in value.values():
        visit(item)
      return
    if isinstance(value, (tuple, list)):
      for item in value:
        visit(item)

  visit(specs)
  return frozenset(axes)


def _p59_translate_partition_specs(
    specs, source_axis: str, target_axis: str | None
):
  """Relabels or consumes one admitted axis in a PartitionSpec tree."""
  if isinstance(specs, jax.sharding.PartitionSpec):

    def translate(entry):
      if entry is None:
        return None
      if isinstance(entry, str):
        return target_axis if entry == source_axis else entry
      translated = [
          target_axis if axis == source_axis else axis for axis in entry
      ]
      translated = [axis for axis in translated if axis is not None]
      if not translated:
        return None
      return translated[0] if len(translated) == 1 else tuple(translated)

    return jax.sharding.PartitionSpec(*(translate(entry) for entry in specs))
  if isinstance(specs, Mapping):
    return type(specs)(
        (
            key,
            _p59_translate_partition_specs(value, source_axis, target_axis),
        )
        for key, value in specs.items()
    )
  if isinstance(specs, tuple):
    return tuple(
        _p59_translate_partition_specs(value, source_axis, target_axis)
        for value in specs
    )
  if isinstance(specs, list):
    return [
        _p59_translate_partition_specs(value, source_axis, target_axis)
        for value in specs
    ]
  return specs


@contextlib.contextmanager
def _p59_localize_engine_shard_maps(target_mesh, label: str):
  """Runs compatible nested engine shard maps inside the outer DP map.

  P56 engine kernels use a six-axis shard_map even when every axis except
  data is unit sized. P59 already maps the surrounding pullback manually over
  data, so nesting that concrete engine mesh below the trainer dp/tp
  AbstractMesh is illegal. For the four-chip TP1 proxy only, rebuild the inner
  map on the current trainer AbstractMesh and relabel its unit model specs. A
  size-one vmap binds the engine's model axis so its fixed collective body
  retains the exact TP1 all-gather/ppermute semantics, while the retained
  shard_map remains the explicit partitioning boundary required by Mosaic.

  The inner engine data spec is consumed because the outer P59 map already
  applied that exact partition. Proven-unit auxiliary specs are also consumed
  and their named axes are bound at size one. At TP>1 the outer P59 map uses a
  two-axis ``data/model`` view of the exact engine topology and makes both axes
  manual. The nested map therefore consumes both partitions while its body
  reuses those already-bound names for fixed-order TP collectives. Unknown
  axes, non-unit auxiliaries, topology changes, and any other axis-type
  transition remain fail-closed.
  """
  shard_map_module = importlib.import_module("jax.experimental.shard_map")
  original_experimental_shard_map = shard_map_module.shard_map
  original_jax_shard_map = jax.shard_map
  target_data, target_model = _p59_mesh_roles(target_mesh, label)
  target_axes = tuple(target_mesh.axis_names)

  def localize_shard_map(original, modern, fun, kwargs):

    context_mesh = jax.sharding.get_abstract_mesh()
    if tuple(context_mesh.axis_names) != target_axes:
      return original(fun, **kwargs)
    axis_types = dict(zip(context_mesh.axis_names, context_mesh.axis_types))
    target_tp = int(target_mesh.shape[target_model])
    if (
        axis_types.get(target_data) is not jax.sharding.AxisType.Manual
        or axis_types.get(target_model) is not jax.sharding.AxisType.Manual
    ):
      return original(fun, **kwargs)

    inner_mesh = kwargs.get("mesh")
    if not isinstance(inner_mesh, jax.sharding.Mesh):
      raise FunctionalMappingError(
          f"{label} nested shard_map requires a concrete engine mesh"
      )
    if tuple(inner_mesh.axis_names) != _P59_ENGINE_MESH_AXES:
      raise FunctionalMappingError(
          f"{label} nested shard_map has unsupported axes "
          f"{tuple(inner_mesh.axis_names)}"
      )
    inner_data, inner_model = _p59_mesh_roles(inner_mesh, label)
    if int(inner_mesh.shape[inner_model]) != target_tp:
      raise FunctionalMappingError(
          f"{label} nested engine shard_map TP changed: "
          f"{int(inner_mesh.shape[inner_model])} != {target_tp}"
      )
    if (
        int(target_mesh.shape[target_data])
        != int(inner_mesh.shape[inner_data])
        or tuple(target_mesh.devices.flat) != tuple(inner_mesh.devices.flat)
    ):
      raise FunctionalMappingError(
          f"{label} nested engine shard_map device topology changed"
      )
    referenced_axes = _p59_partition_axes(
        (kwargs.get("in_specs"), kwargs.get("out_specs"))
    )
    unsupported_axes = referenced_axes - set(_P59_ENGINE_MESH_AXES)
    if unsupported_axes:
      raise FunctionalMappingError(
          f"{label} nested engine shard_map uses unsupported axes: "
          f"axes={sorted(unsupported_axes)}"
      )

    def bind_unit_axis(inner_fun, axis_name):

      def bound_fun(*args):
        bound = jax.vmap(
            inner_fun,
            in_axes=tuple(None for _ in args),
            out_axes=0,
            axis_name=axis_name,
            axis_size=1,
        )(*args)
        return jax.tree.map(lambda value: value[0], bound)

      return bound_fun

    localized_fun = fun
    for unit_axis in _P59_ENGINE_MESH_AXES:
      if unit_axis not in (inner_data, inner_model):
        localized_fun = bind_unit_axis(localized_fun, unit_axis)
    if target_tp == 1:
      localized_fun = bind_unit_axis(localized_fun, inner_model)

    if (
        target_tp > 1
        and os.environ.get("CANON_P66_P59_CHECK_VMA", "0") == "1"
    ):
      # The surrounding P59 shard_map has already consumed the physical data
      # and model partitions. Re-entering shard_map on that same AbstractMesh
      # requires the nested specs to be erased to P(None); that erasure also
      # lies about values which vary over the outer manual axes. With
      # check_vma=False the lie was silent and its transpose produced the P66
      # exploding-gradient regression. With check_vma=True it fails at the
      # first endpoint whose cotangent remains V:data.
      #
      # Execute the already-local engine body directly instead. The outer map
      # is the required manual/Mosaic boundary, its model axis remains bound
      # for the engine collectives, and the size-one auxiliary axes above are
      # still bound explicitly. No dimension is repartitioned here and VMA
      # types now flow through the real local primitives into their VJPs.
      print(
          f"[P66.VMA] nested_engine_body_reuses_outer_map label={label} "
          f"tp={target_tp}",
          flush=True,
      )
      return localized_fun

    localized_kwargs = dict(kwargs)
    localized_kwargs["mesh"] = context_mesh
    for specs_name in ("in_specs", "out_specs"):
      translated_specs = kwargs.get(specs_name)
      for engine_axis in _P59_ENGINE_MESH_AXES:
        translated_specs = _p59_translate_partition_specs(
            translated_specs,
            engine_axis,
            target_model if engine_axis == inner_model else None,
        )
      localized_kwargs[specs_name] = translated_specs
    if target_tp > 1:
      # Data and model were already partitioned by the outer two-axis engine
      # view. Keep the inner shard_map as the Mosaic/Pallas boundary, but do
      # not divide either dimension a second time. Both legacy and modern
      # calls route through the modern primitive so the exact current
      # AbstractMesh, rather than the original six-axis concrete mesh, is
      # retained.
      for specs_name in ("in_specs", "out_specs"):
        translated_specs = localized_kwargs[specs_name]
        translated_specs = _p59_translate_partition_specs(
            translated_specs, target_data, None
        )
        translated_specs = _p59_translate_partition_specs(
            translated_specs, target_model, None
        )
        localized_kwargs[specs_name] = translated_specs
      check_vma = localized_kwargs.pop(
          "check_vma", localized_kwargs.pop("check_rep", True)
      )
      localized_kwargs["axis_names"] = {target_data, target_model}
      localized_kwargs["check_vma"] = check_vma
      return original_jax_shard_map(localized_fun, **localized_kwargs)
    if modern:
      localized_kwargs["axis_names"] = {target_data, target_model}
    else:
      # Installed projection/norm shims use the deprecated experimental API.
      if "check_vma" in localized_kwargs:
        localized_kwargs["check_rep"] = localized_kwargs.pop("check_vma")
      localized_kwargs.pop("axis_names", None)
    return original(localized_fun, **localized_kwargs)

  def localized_experimental_shard_map(fun=None, /, **kwargs):
    if fun is None:
      return lambda wrapped: localized_experimental_shard_map(
          wrapped, **kwargs
      )
    return localize_shard_map(
        original_experimental_shard_map, False, fun, kwargs
    )

  def localized_jax_shard_map(fun=None, /, **kwargs):
    if fun is None:
      return lambda wrapped: localized_jax_shard_map(wrapped, **kwargs)
    return localize_shard_map(original_jax_shard_map, True, fun, kwargs)

  with _P59_NESTED_SHARD_MAP_LOCK:
    if (
        shard_map_module.shard_map is not original_experimental_shard_map
        or jax.shard_map is not original_jax_shard_map
    ):
      raise FunctionalMappingError(
          f"{label} nested shard_map hook is already active"
      )
    shard_map_module.shard_map = localized_experimental_shard_map
    jax.shard_map = localized_jax_shard_map
    try:
      yield
    finally:
      if (
          shard_map_module.shard_map is not localized_experimental_shard_map
          or jax.shard_map is not localized_jax_shard_map
      ):
        raise FunctionalMappingError(
            f"{label} nested shard_map hook changed during tracing"
        )
      shard_map_module.shard_map = original_experimental_shard_map
      jax.shard_map = original_jax_shard_map


def _segmented_loss_geometry(environ) -> tuple[int, tuple[int, int]]:
  """Returns the fail-closed batch geometry for segmented GRPO loss."""
  p41_optimizer_bench = environ.get("CANON_P41_OPTIMIZER_BENCH", "") == "1"
  if p41_optimizer_bench:
    if (
        environ.get("CANON_GSM8K_L3", "") != "1"
        or environ.get("CANON_GSM8K_UPDATE_CANARY", "") != "1"
        or environ.get("CANON_P33_WORKLOAD_LAUNCH_ADMITTED", "") == "1"
        or environ.get("CANON_P34_DEEPSWE", "") == "1"
    ):
      raise FunctionalMappingError(
          "P41 segmented loss requires the bounded GSM8K L3 update canary"
      )
    return 2, (256, 64)
  if (
      environ.get("CANON_GSM8K_TRAIN", "") == "1"
      and environ.get("CANON_P33_WORKLOAD_LAUNCH_ADMITTED", "") != "1"
      and environ.get("CANON_P34_DEEPSWE", "") != "1"
  ):
    # One-host real-geometry GSM8K training: 4 prompts x 8 generations per
    # update at prompt/response 1024/1024 (admitted 2026-08-15 for P51).
    return 32, (1024, 1024)
  if environ.get("CANON_P31_CONVERGENCE", "") == "1":
    return 32, (4096, 2048)
  return 8, (2048, 64)


_P35_REPLAY_STAGE_PROBE_ENV = "CANON_P35_REPLAY_STAGE_PROBE"
_P35_REPLAY_STAGE_REPORT_ENV = "CANON_P35_REPLAY_STAGE_REPORT"
_P35_REPLAY_STAGE_NAMES = (
    "model",
    "logits",
    "sample",
    "logprobs",
    "target_gathers",
    "record_outputs",
)


def _p35_replay_stage_probe_enabled() -> bool:
  value = os.environ.get(_P35_REPLAY_STAGE_PROBE_ENV, "0")
  if value not in ("0", "1"):
    raise FunctionalMappingError(
        f"{_P35_REPLAY_STAGE_PROBE_ENV} must be exactly 0 or 1"
    )
  return value == "1"


def _p35_stage_shape(value: Any) -> list[list[int]]:
  """Returns static leaf shapes without fetching device values."""
  return [
      [int(dimension) for dimension in leaf.shape]
      for leaf in jax.tree.leaves(value)
      if hasattr(leaf, "shape")
  ]


def _p35_wait_for_stage(
    value: Any,
    *,
    replay_label: str,
    record_index: int,
    record_count: int,
    stage: str,
) -> None:
  """Makes one async boundary observable for the default-off stage probe."""
  if not _p35_replay_stage_probe_enabled():
    return
  if stage not in _P35_REPLAY_STAGE_NAMES:
    raise FunctionalMappingError(f"unknown P35.3c replay stage: {stage}")
  ordinal = _P35_REPLAY_STAGE_NAMES.index(stage) + 1
  report_path = os.environ.get(_P35_REPLAY_STAGE_REPORT_ENV, "")
  if not report_path:
    raise FunctionalMappingError(
        f"{_P35_REPLAY_STAGE_PROBE_ENV}=1 requires "
        f"{_P35_REPLAY_STAGE_REPORT_ENV}"
    )
  if ordinal == 1:
    with open(report_path, "x", encoding="utf-8") as stream:
      stream.flush()
      os.fsync(stream.fileno())
  elif not os.path.exists(report_path):
    raise FunctionalMappingError(
        "P35.3c stage report disappeared after the first stage"
    )
  print(
      "[CANON_P35.3C] STAGE_BEGIN "
      f"replay={replay_label} record={record_index + 1}/{record_count} "
      f"stage={stage} ordinal={ordinal}/{len(_P35_REPLAY_STAGE_NAMES)}",
      flush=True,
  )
  jax.block_until_ready(value)
  event = {
      "schema_version": 1,
      "event": "ready",
      "replay": replay_label,
      "record_index": record_index + 1,
      "record_count": record_count,
      "stage": stage,
      "ordinal": ordinal,
      "stage_count": len(_P35_REPLAY_STAGE_NAMES),
      "leaf_shapes": _p35_stage_shape(value),
  }
  with open(report_path, "a", encoding="utf-8") as stream:
    stream.write(json.dumps(event, sort_keys=True) + "\n")
    stream.flush()
    os.fsync(stream.fileno())
  print(
      "[CANON_P35.3C] STAGE_READY "
      f"replay={replay_label} record={record_index + 1}/{record_count} "
      f"stage={stage} ordinal={ordinal}/{len(_P35_REPLAY_STAGE_NAMES)}",
      flush=True,
  )


@jax.jit
def _exact_leaf_bits_equal(left: Any, right: Any) -> jax.Array:
  """Reduces one same-shaped leaf to an exact bytewise device boolean."""
  left_bits = jax.lax.bitcast_convert_type(left, jnp.uint8)
  right_bits = jax.lax.bitcast_convert_type(right, jnp.uint8)
  return jnp.all(left_bits == right_bits)


def _normalize_exact_compare_memory(
    left: Any, right: Any
) -> tuple[Any, Any]:
  """Places mixed host/device inputs in one existing device sharding."""
  left_sharding = getattr(left, "sharding", None)
  right_sharding = getattr(right, "sharding", None)
  left_memory = getattr(left_sharding, "memory_kind", None)
  right_memory = getattr(right_sharding, "memory_kind", None)
  if (
      left_memory is None
      or right_memory is None
      or left_memory == right_memory
  ):
    return left, right

  if left_memory == "device":
    device_sharding = left_sharding
  elif right_memory == "device":
    device_sharding = right_sharding
  else:
    raise FunctionalMappingError(
        "exact bitwise equality requires a device operand when explicit "
        "memory spaces differ; got "
        f"{left_memory!r} and {right_memory!r}"
    )

  if left_memory != "device":
    left = jax.device_put(left, device_sharding)
  if right_memory != "device":
    right = jax.device_put(right, device_sharding)
  return left, right


def _bitwise_arrays_equal(left: Any, right: Any) -> bool:
  """Returns exact device-side equality, including NaN payloads and signed zero."""
  left_value = getattr(left, "value", left)
  right_value = getattr(right, "value", right)
  if (
      tuple(left_value.shape) != tuple(right_value.shape)
      or left_value.dtype != right_value.dtype
  ):
    return False
  if jnp.dtype(left_value.dtype).itemsize not in (1, 2, 4, 8):
    raise FunctionalMappingError(
        f"exact bitwise equality does not support dtype {left_value.dtype}"
    )
  left_value, right_value = _normalize_exact_compare_memory(
      left_value, right_value
  )
  return bool(
      np.asarray(jax.device_get(_exact_leaf_bits_equal(left_value, right_value)))
  )


def _bitwise_difference_summary(left: Any, right: Any) -> dict[str, Any]:
  """Returns a small exact-bit comparison without copying full tensors."""
  left_value = getattr(left, "value", left)
  right_value = getattr(right, "value", right)
  if (
      tuple(left_value.shape) != tuple(right_value.shape)
      or left_value.dtype != right_value.dtype
  ):
    return {
        "valid": False,
        "shape_left": tuple(left_value.shape),
        "shape_right": tuple(right_value.shape),
        "dtype_left": str(left_value.dtype),
        "dtype_right": str(right_value.dtype),
    }
  itemsize = jnp.dtype(left_value.dtype).itemsize
  bit_dtype = {
      1: jnp.uint8,
      2: jnp.uint16,
      4: jnp.uint32,
      8: jnp.uint64,
  }.get(itemsize)
  if bit_dtype is None:
    raise FunctionalMappingError(
        f"P35.3 cannot compare dtype {left_value.dtype} bitwise"
    )
  left_value, right_value = _normalize_exact_compare_memory(
      left_value, right_value
  )
  left_bits = jax.lax.bitcast_convert_type(left_value, bit_dtype)
  right_bits = jax.lax.bitcast_convert_type(right_value, bit_dtype)
  differing = int(
      np.asarray(jax.device_get(jnp.count_nonzero(left_bits != right_bits)))
  )
  total = int(left_value.size)
  return {
      "valid": True,
      "shape": tuple(int(value) for value in left_value.shape),
      "dtype": str(left_value.dtype),
      "differing_elements": differing,
      "total_elements": total,
      "exact": differing == 0,
  }


def _host_difference_summary(left: Any, right: Any) -> dict[str, Any]:
  """Returns exact and numeric diagnostics for one bounded host-side vector."""
  left_value = np.ascontiguousarray(np.asarray(left))
  right_value = np.ascontiguousarray(np.asarray(right))
  if left_value.shape != right_value.shape or left_value.dtype != right_value.dtype:
    return {
        "valid": False,
        "shape_left": list(left_value.shape),
        "shape_right": list(right_value.shape),
        "dtype_left": str(left_value.dtype),
        "dtype_right": str(right_value.dtype),
    }
  byte_difference = left_value.view(np.uint8) != right_value.view(np.uint8)
  element_difference = byte_difference.reshape(
      left_value.size, left_value.dtype.itemsize
  ).any(axis=1).reshape(left_value.shape)
  coordinates = np.argwhere(element_difference)
  first = None
  if coordinates.size:
    coordinate = tuple(int(value) for value in coordinates[0])
    first = {
        "index": list(coordinate),
        "left": float(left_value[coordinate]),
        "right": float(right_value[coordinate]),
    }
  if left_value.size:
    max_abs = float(
        np.max(np.abs(left_value.astype(np.float64) - right_value.astype(np.float64)))
    )
  else:
    max_abs = 0.0
  return {
      "valid": True,
      "shape": list(left_value.shape),
      "dtype": str(left_value.dtype),
      "differing_bytes": int(byte_difference.sum()),
      "total_bytes": int(byte_difference.size),
      "differing_elements": int(element_difference.sum()),
      "total_elements": int(element_difference.size),
      "exact": not bool(element_difference.any()),
      "max_abs": max_abs,
      "first_mismatch": first,
      "left_sha256": hashlib.sha256(left_value.tobytes()).hexdigest(),
      "right_sha256": hashlib.sha256(right_value.tobytes()).hexdigest(),
  }


def _canonical_topology_contract() -> tuple[int, int, int, int]:
  """Returns the admitted data, tensor, local-M, and global-M contract."""
  training_admitted = os.environ.get("CANON_P32_TRAIN_ADMITTED", "0")
  if training_admitted not in ("0", "1"):
    raise FunctionalMappingError(
        "CANON_P32_TRAIN_ADMITTED must be exactly 0 or 1"
    )
  if training_admitted == "0":
    data_size, tp_size, local_m, global_m = 1, 0, 256, 256
  else:
    try:
      data_size = int(os.environ.get("CANON_DP_SIZE", "0"))
      tp_size = int(os.environ.get("CANON_TP_SIZE", "0"))
      local_m = int(os.environ.get("CANON_LOGPROB_M", "0"))
      global_m = int(os.environ.get("MIN_TOKEN_BUCKET", "0"))
      target_m = int(os.environ.get("CANON_TARGET_M", "0"))
    except ValueError as exc:
      raise FunctionalMappingError(
          "P32 topology values must be integers"
      ) from exc
    p34 = os.environ.get("CANON_P34_DEEPSWE", "") == "1"
    workload = (
        deepswe_contract.active_workload(os.environ)
        if p34
        else dp_workloads.active_workload(os.environ)
    )
    if workload is None:
      raise FunctionalMappingError(
          "canonical training requires an active workload contract"
      )
    expected_tp = workload.tp_size
    expected_dp = workload.dp_size
    if (data_size, tp_size) != (expected_dp, expected_tp):
      raise FunctionalMappingError(
          f"canonical training admits exactly DP{expected_dp}xTP{expected_tp}; got "
          f"DP{data_size}xTP{tp_size}"
      )
    if local_m != 256 or target_m != local_m:
      raise FunctionalMappingError(
          "P32 training requires CANON_LOGPROB_M=CANON_TARGET_M=256; "
          f"got {local_m}/{target_m}"
      )
    if global_m != data_size * local_m:
      raise FunctionalMappingError(
          "P32 training requires MIN_TOKEN_BUCKET=dp*CANON_LOGPROB_M; "
          f"got {global_m} != {data_size}*{local_m}"
      )
  return data_size, tp_size, local_m, global_m


def _canonical_logprob_bucket() -> int:
  """Returns the fixed global M admitted by the selected topology."""
  data_size, _, local_m, global_m = _canonical_topology_contract()
  if data_size == 1:
    raw_logprob = os.environ.get("CANON_LOGPROB_M", "0")
    raw_token = os.environ.get("MIN_TOKEN_BUCKET", "")
    if raw_logprob != str(local_m) or raw_token != str(global_m):
      raise FunctionalMappingError(
          "canonical adapter requires "
          "CANON_LOGPROB_M=MIN_TOKEN_BUCKET=256"
      )
  return global_m


def _canonical_logprob_row_spec(mesh) -> jax.sharding.PartitionSpec:
  """Returns the topology-specific global-row sharding for log-softmax."""
  axis_names = tuple(mesh.axis_names)
  if "data" in axis_names:
    return jax.sharding.PartitionSpec("data", None)
  return jax.sharding.PartitionSpec(None, None)


_ISSUE_ANATOMY = {"prep": 0.0, "call": 0.0, "n": 0}


@functools.partial(jax.jit, static_argnums=(3, 4, 5, 6))
def _fused_p28_chunk_inputs(
    n_real,
    packed_ids,
    next_ids,
    chunk_start,
    bucket,
    max_num_reqs,
    blocks_per_req,
):
  """One-dispatch build of a P28 single-request chunk's engine inputs.

  The eager body issues ~8 tiny programs per build (arange/where, three
  zeros-with-.at sets, two slices, an asarray). chunk_start stays a
  static python int, so the slices keep their exact static semantics and
  the trace count is bounded by the chunk count (<= a handful at the
  fixed bucket). Every output value is unchanged integer index math.
  P52's per-(sequence, chunk) cache already bounds how often this runs;
  the fusion trims the residual builds to one dispatch each.
  """
  rows = jnp.arange(bucket, dtype=jnp.int32)
  q_len = jnp.minimum(bucket, n_real - chunk_start)
  kv_len = jnp.minimum(n_real, chunk_start + bucket)
  positions = jnp.where(rows < q_len, chunk_start + rows, 0)
  query_start = jnp.zeros((max_num_reqs + 1,), jnp.int32).at[1:].set(q_len)
  seq_lens = jnp.zeros((max_num_reqs,), jnp.int32).at[0].set(kv_len)
  block_tables = jnp.zeros(
      (max_num_reqs, blocks_per_req), jnp.int32
  ).at[0].set(jnp.arange(blocks_per_req, dtype=jnp.int32))
  request_distribution = jnp.asarray((0, 0, 1), jnp.int32)
  ids = packed_ids[chunk_start : chunk_start + bucket]
  targets = next_ids[chunk_start : chunk_start + bucket]
  return (
      ids,
      targets,
      positions,
      block_tables.reshape(-1),
      seq_lens,
      query_start,
      request_distribution,
  )


@functools.partial(jax.jit, static_argnums=(4, 5, 6, 7))
def _fused_chunk_metadata(
    n_real,
    packed_ids,
    next_ids,
    chunk_start,
    sequence_bucket,
    data_size,
    max_num_reqs,
    blocks_per_req,
):
  """One-dispatch build of a chunk's engine-call inputs.

  Eagerly, _p32_group_chunk_inputs plus the metadata-arrays helper issue
  ~15 tiny programs per chunk (arange/clip/where/minimum, two .at sets,
  slices, reshapes) -- measured as the add/convert scalar-glue swarm in
  the p56r3 update window. This computes the same integer index math in
  one program; the python-int slices become dynamic slices whose starts
  are chunk_index * bucket and therefore in bounds by construction, so
  every output value is unchanged. chunk_start travels as an array to
  keep a single trace across chunks.
  """
  rows = jnp.arange(sequence_bucket, dtype=jnp.int32)
  q_len = jnp.clip(n_real - chunk_start, 0, sequence_bucket)
  kv_len = jnp.where(
      q_len > 0,
      jnp.minimum(n_real, chunk_start + sequence_bucket),
      0,
  )
  chunk_ids = jax.lax.dynamic_slice_in_dim(
      packed_ids, chunk_start, sequence_bucket, axis=1
  )
  chunk_targets = jax.lax.dynamic_slice_in_dim(
      next_ids, chunk_start, sequence_bucket, axis=1
  )
  positions = jnp.where(
      rows[None, :] < q_len[:, None], chunk_start + rows[None, :], 0
  )
  local_max_num_reqs = max_num_reqs // data_size
  active = q_len > 0
  block_tables = jnp.zeros(
      (data_size, local_max_num_reqs, blocks_per_req), jnp.int32
  ).at[:, 0, :].set(
      jnp.broadcast_to(
          jnp.arange(blocks_per_req, dtype=jnp.int32),
          (data_size, blocks_per_req),
      )
  )
  query_start = jnp.where(
      jnp.arange(local_max_num_reqs + 1)[None, :] == 0,
      0,
      q_len[:, None],
  ).astype(jnp.int32)
  seq_lens = jnp.zeros(
      (data_size, local_max_num_reqs), jnp.int32
  ).at[:, 0].set(jnp.where(active, kv_len, 0))
  request_distribution = jnp.stack(
      (
          jnp.zeros_like(q_len),
          jnp.zeros_like(q_len),
          active.astype(jnp.int32),
      ),
      axis=1,
  )
  return (
      chunk_ids.reshape(-1),
      chunk_targets.reshape(-1),
      positions.reshape(-1),
      block_tables.reshape(-1),
      seq_lens.reshape(-1),
      query_start.reshape(-1),
      request_distribution.reshape(-1),
  )


def _canonical_dp_attention_metadata_arrays(
    *,
    data_size,
    max_num_reqs,
    blocks_per_req,
    q_len,
    kv_len,
):
  """Builds rank-major RPA metadata with one request per data rank."""
  data_size = int(data_size)
  max_num_reqs = int(max_num_reqs)
  blocks_per_req = int(blocks_per_req)
  if data_size < 1 or max_num_reqs % data_size:
    raise FunctionalMappingError(
        "RPA metadata requires max_num_reqs divisible by data size"
    )
  if q_len.shape != (data_size,) or kv_len.shape != (data_size,):
    raise FunctionalMappingError(
        "RPA metadata lengths must contain one scalar per data rank"
    )
  local_max_num_reqs = max_num_reqs // data_size
  active = q_len > 0
  block_tables = jnp.zeros(
      (data_size, local_max_num_reqs, blocks_per_req), jnp.int32
  ).at[:, 0, :].set(
      jnp.broadcast_to(
          jnp.arange(blocks_per_req, dtype=jnp.int32),
          (data_size, blocks_per_req),
      )
  )
  query_start = jnp.where(
      jnp.arange(local_max_num_reqs + 1)[None, :] == 0,
      0,
      q_len[:, None],
  ).astype(jnp.int32)
  seq_lens = jnp.zeros(
      (data_size, local_max_num_reqs), jnp.int32
  ).at[:, 0].set(jnp.where(active, kv_len, 0))
  request_distribution = jnp.stack(
      (
          jnp.zeros_like(q_len),
          jnp.zeros_like(q_len),
          active.astype(jnp.int32),
      ),
      axis=1,
  )
  return (
      block_tables.reshape(-1),
      seq_lens.reshape(-1),
      query_start.reshape(-1),
      request_distribution.reshape(-1),
  )


def _make_canonical_compute_and_gather(gather_logprobs, mesh):
  """Builds the one shared rollout/trainer logprob function object."""

  data_size, _, local_m, global_m = _canonical_topology_contract()
  # These are the exact request paddings precompiled by the pinned vLLM TPU
  # runner. Under engine DP, the scorer receives the caller-global row count
  # while shard_map sees one data-rank slice. Every admitted short slice is
  # row-independently zero-padded to the canonical M256 kernel and sliced back.
  request_rows = (8, 16, 32, 64, 128, 256)
  admitted_global_rows = tuple(sorted({
      local_m,
      global_m,
      *(rows for rows in request_rows if rows % data_size == 0),
  }))
  admitted_local_rows = tuple(
      sorted({rows // data_size for rows in admitted_global_rows})
  )

  def local_log_softmax(logits):
    if data_size > 1:
      rows = int(logits.shape[0])
      if rows in admitted_local_rows and rows != local_m:
        logits = jnp.pad(
            logits,
            ((0, local_m - rows), (0, 0)),
            constant_values=jnp.float32(0),
        )
        return canonical_logsoftmax.log_softmax(logits)[:rows]
      if rows != local_m:
        raise FunctionalMappingError(
            "canonical log-softmax per-rank row count changed: "
            f"{rows} not in {admitted_local_rows}"
        )
    return canonical_logsoftmax.log_softmax(logits)

  row_spec = _canonical_logprob_row_spec(mesh)

  try:
    mapped_log_softmax = jax.shard_map(
        local_log_softmax,
        mesh=mesh,
        in_specs=row_spec,
        out_specs=row_spec,
        check_vma=False,
    )
  except TypeError:
    mapped_log_softmax = jax.shard_map(
        local_log_softmax,
        mesh=mesh,
        in_specs=row_spec,
        out_specs=row_spec,
        check_rep=False,
    )

  gathered_mode = (
      os.environ.get("CANON_PALLAS_GATHERED_LOGPROBS", "") == "1"
  )
  if gathered_mode:
    # P56.4.3: skip the [rows, vocab] stage-3 materialize and the stock
    # gather; stages 1+2 are shared verbatim and every comparison runs
    # on the same x - normalizer values, so the emitted logprob, top-1,
    # and rank are bit-identical (CPU interpret gate + 51/51 judge).
    from tpu_inference.layers.jax.sample.sampling import LogprobsTensors  # pylint: disable=g-import-not-at-top

    continue_decode = os.environ.get("CANON_CONTINUE_DECODE", "")
    print(
        "[P56.GATHERED_LOGPROBS] installed "
        f"data={data_size} local_m={local_m} "
        f"continue_decode={continue_decode or '0'}",
        flush=True,
    )

    def local_gathered(logits, tokens):
      rows = int(logits.shape[0])
      if int(tokens.shape[0]) != rows:
        raise FunctionalMappingError(
            "canonical gathered-logprobs logits/token rows differ: "
            f"{rows} vs {tokens.shape[0]}"
        )
      if data_size > 1:
        if rows in admitted_local_rows and rows != local_m:
          padded_logits = jnp.pad(
              logits,
              ((0, local_m - rows), (0, 0)),
              constant_values=jnp.float32(0),
          )
          padded_tokens = jnp.pad(
              tokens,
              ((0, local_m - rows),),
              constant_values=jnp.int32(0),
          )
          output = canonical_logsoftmax.gathered_logprobs(
              padded_logits, padded_tokens
          )
          return tuple(item[:rows] for item in output)
        if rows != local_m:
          raise FunctionalMappingError(
              "canonical gathered-logprobs per-rank row count changed: "
              f"{rows} not in {admitted_local_rows}"
          )
      if continue_decode and data_size == 1:
        return canonical_logsoftmax.continue_decode_gathered_logprobs(
            logits, tokens
        )
      return canonical_logsoftmax.gathered_logprobs(logits, tokens)

    vector_spec = (
        jax.sharding.PartitionSpec("data")
        if "data" in tuple(mesh.axis_names)
        else jax.sharding.PartitionSpec(None)
    )
    gathered_out_specs = (
        vector_spec, vector_spec, vector_spec, vector_spec
    )
    try:
      mapped_gathered = jax.shard_map(
          local_gathered,
          mesh=mesh,
          in_specs=(row_spec, vector_spec),
          out_specs=gathered_out_specs,
          check_vma=False,
      )
    except TypeError:
      mapped_gathered = jax.shard_map(
          local_gathered,
          mesh=mesh,
          in_specs=(row_spec, vector_spec),
          out_specs=gathered_out_specs,
          check_rep=False,
      )

    def compute_and_gather_fused(logits, next_tokens, max_logprobs):
      if int(max_logprobs) != 1:
        raise FunctionalMappingError(
            "CANON_PALLAS_GATHERED_LOGPROBS=1 requires the max_logprobs=1 "
            f"rollout contract, got {max_logprobs}"
        )
      token_logprob, top_value, top_index, token_ranks = mapped_gathered(
          logits, next_tokens
      )
      token_ids = jnp.int32(next_tokens)[:, None]
      indices = jnp.concatenate((token_ids, top_index[:, None]), axis=1)
      values = jnp.concatenate(
          (token_logprob[:, None], top_value[:, None]), axis=1
      )
      return LogprobsTensors(jnp.int32(indices), values, token_ranks)

    return _xprof_jit(
        compute_and_gather_fused,
        module_name="zt_ro_logprob_gather",
        scope_name="zt/ro/logprob/gather",
        static_argnames=("max_logprobs",),
    )

  def compute_and_gather(logits, next_tokens, max_logprobs):
    rows = int(logits.shape[0])
    if data_size > 1 and rows not in admitted_global_rows:
      raise FunctionalMappingError(
          "canonical log-softmax global row count changed: "
          f"{rows} not in {admitted_global_rows}"
      )
    logprobs = mapped_log_softmax(logits)
    return gather_logprobs(logprobs, next_tokens, max_logprobs)

  return jax.jit(compute_and_gather, static_argnames=("max_logprobs",))


def _install_shared_logprob_pipeline(
    runner,
    *,
    stock_compute_and_gather,
    gather_logprobs,
    runner_module=None,
    sampling_module=None,
):
  """Installs one default-off canonical scorer at both live lookup sites."""
  if os.environ.get(canonical_logsoftmax.ENV, "") != "1":
    return getattr(
        runner, "_canonical_compute_and_gather_logprobs", stock_compute_and_gather
    )
  if runner_module is None:
    runner_module = importlib.import_module(type(runner).__module__)
  if sampling_module is None:
    sampling_module = importlib.import_module(
        "tpu_inference.layers.jax.sample.sampling"
    )

  canonical = getattr(
      runner_module, "_canonical_logsoftmax_compute_and_gather", None
  )
  runner_stock = getattr(
      runner_module,
      "_canonical_stock_compute_and_gather_logprobs",
      stock_compute_and_gather,
  )
  current_runner = getattr(
      runner_module, "compute_and_gather_logprobs", None
  )
  if current_runner not in (runner_stock, canonical):
    raise FunctionalMappingError(
        "refusing to overwrite an unknown runner logprob implementation"
    )
  current_sampling = getattr(
      sampling_module, "compute_and_gather_logprobs", None
  )
  if current_sampling not in (stock_compute_and_gather, canonical):
    raise FunctionalMappingError(
        "refusing to overwrite an unknown sampling logprob implementation"
    )
  if canonical is None:
    canonical = _make_canonical_compute_and_gather(gather_logprobs, runner.mesh)
    runner_module._canonical_logsoftmax_compute_and_gather = canonical
    runner_module._canonical_stock_compute_and_gather_logprobs = runner_stock

  runner_module.compute_and_gather_logprobs = canonical
  sampling_module.compute_and_gather_logprobs = canonical
  runner._canonical_compute_and_gather_logprobs = canonical
  if not (
      runner_module.compute_and_gather_logprobs
      is sampling_module.compute_and_gather_logprobs
      is runner._canonical_compute_and_gather_logprobs
  ):
    raise FunctionalMappingError("shared canonical logprob identity check failed")
  print(
      "[CANON_ADAPTER] shared canonical logprob pipeline installed "
      "runner_sampling_adapter_same_object=True stages=partial,combine,normalize",
      flush=True,
  )
  return canonical


def _make_processed_target_logprob_vjp(compute_and_gather, max_logprobs):
  """Keeps the exact engine primal while supplying the analytic logp VJP."""

  def exact_value(logits, token_ids):
    return compute_and_gather(
        logits, token_ids, max_logprobs
    ).logprobs[:, 0]

  @jax.custom_vjp
  def target_logprobs(logits, token_ids):
    return exact_value(logits, token_ids)

  def forward(logits, token_ids):
    return exact_value(logits, token_ids), (logits, token_ids)

  def backward(residual, cotangent):
    print(
        "[PATHTRACE] CANON_PROCESSED_LOGPROB_VJP backward",
        flush=True,
    )
    logits, token_ids = residual
    probabilities = jax.nn.softmax(logits, axis=-1)
    selected = jax.nn.one_hot(
        token_ids, logits.shape[-1], dtype=logits.dtype
    )
    d_logits = (selected - probabilities) * cotangent[:, None]
    return d_logits, None

  target_logprobs.defvjp(forward, backward)
  return target_logprobs


@dataclasses.dataclass(frozen=True)
class FunctionalEngineLeaves:
  """Mapped engine leaves and their stable target paths."""

  paths: tuple[str, ...]
  leaves: tuple[jax.Array, ...]
  source_to_target: tuple[tuple[str, str], ...]


@dataclasses.dataclass(frozen=True)
class MappingManifestEntry:
  """One shape-only source-to-target mapping attestation."""

  source_path: str
  target_path: str
  source_shape: tuple[int, ...]
  source_dtype: str
  target_shape: tuple[int, ...]
  target_dtype: str
  mapped_shape: tuple[int, ...]
  mapped_dtype: str


@dataclasses.dataclass(frozen=True)
class MappingManifest:
  """A materialization-free inventory of the real mapping contract."""

  entries: tuple[MappingManifestEntry, ...]
  target_paths: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class LiveEngineContract:
  """JSON-safe attestation for a live in-process engine runner."""

  implementation_id: str
  mapping_entries: int
  target_path_sha256: str
  state_leaves: int
  mesh_shape: tuple[tuple[str, int], ...]
  kv_caches: int
  model_fn: str
  compute_logits_fn: str


@dataclasses.dataclass(frozen=True)
class SegmentedForwardContract:
  """Static attestation for the P28 host-segmented engine forward."""

  implementation_id: str
  state_leaves: int
  start_layer: int
  end_layer: int
  block_depth: int


@dataclasses.dataclass(frozen=True)
class SegmentedBlockVjpContract:
  """Static attestation for one independently differentiated real layer."""

  layer_index: int
  local_state_leaves: int
  local_state_bytes: int
  block_depth: int


class _P28SegmentedEngineForward:
  """Host-orchestrated Qwen3 forward with one JIT per real decoder layer.

  This is deliberately not a pytree and must never be enclosed by another
  JAX transform.  The host is the composition boundary: each compiled layer
  receives the complete engine state as runtime leaves, but only the selected
  layer is reachable from that executable.  G3 decides whether these extra JIT
  boundaries preserve the canonical whole-model value bitwise.
  """

  def __init__(self, runner):
    if os.environ.get("CANON_P28_SEGMENTED_FORWARD", "") != "1":
      raise FunctionalMappingError(
          "P28 segmented forward requires CANON_P28_SEGMENTED_FORWARD=1"
      )
    if not bool(runner.is_first_rank) or not bool(runner.is_last_rank):
      raise FunctionalMappingError(
          "P28 segmented forward currently admits no pipeline parallelism"
      )
    if not hasattr(runner, "model") or not hasattr(runner, "state"):
      raise FunctionalMappingError(
          "live runner does not expose the NNX model/state reconstruction seam"
      )
    if not isinstance(getattr(runner, "mesh", None), jax.sharding.Mesh):
      raise FunctionalMappingError(
          "P28 segmented forward requires the concrete live engine mesh"
      )
    self._engine_mesh = runner.mesh
    self._engine_data_size = int(self._engine_mesh.shape.get("data", 1))
    self._engine_tp_size = int(self._engine_mesh.shape.get("model", 1))

    from flax import nnx  # pylint: disable=g-import-not-at-top

    graphdef, live_state = nnx.split(runner.model)
    live_treedef = jax.tree_util.tree_structure(live_state)
    runner_treedef = jax.tree_util.tree_structure(runner.state)
    if live_treedef != runner_treedef:
      raise FunctionalMappingError(
          "runner.model and runner.state have different NNX state trees"
      )
    live_leaves = tuple(jax.tree_util.tree_leaves(live_state))
    runner_leaves = tuple(jax.tree_util.tree_leaves(runner.state))
    if len(live_leaves) != len(runner_leaves):
      raise FunctionalMappingError(
          "runner.model and runner.state have different leaf counts"
      )
    for index, (live_leaf, runner_leaf) in enumerate(
        zip(live_leaves, runner_leaves)
    ):
      if (
          tuple(live_leaf.shape) != tuple(runner_leaf.shape)
          or live_leaf.dtype != runner_leaf.dtype
      ):
        raise FunctionalMappingError(
            "runner.model/state leaf contract differs at index "
            f"{index}: {live_leaf.shape}/{live_leaf.dtype} != "
            f"{runner_leaf.shape}/{runner_leaf.dtype}"
        )

    try:
      backbone = runner.model.model
      layers = tuple(backbone.layers)
      start_layer = int(backbone.start_layer)
      end_layer = int(backbone.end_layer)
      embed_tokens = backbone.embed_tokens
      final_norm = backbone.norm
    except (AttributeError, TypeError, ValueError) as exc:
      raise FunctionalMappingError(
          "live model is not the admitted Qwen3 NNX layer-stack structure"
      ) from exc
    if start_layer != 0 or end_layer != len(layers):
      raise FunctionalMappingError(
          "P28 requires the complete local Qwen3 layer range: "
          f"start={start_layer} end={end_layer} layers={len(layers)}"
      )
    if len(runner.kv_caches) != len(layers):
      raise FunctionalMappingError(
          "P28 layer/cache cardinality differs: "
          f"layers={len(layers)} caches={len(runner.kv_caches)}"
      )

    def merge(state_leaves):
      state = jax.tree_util.tree_unflatten(live_treedef, state_leaves)
      return nnx.merge(graphdef, state)

    def embed(state_leaves, input_ids):
      model = merge(state_leaves)
      return model.model.embed_tokens(input_ids)

    def norm(state_leaves, hidden):
      model = merge(state_leaves)
      return model.model.norm(hidden)

    full_leaf_id_to_index = {id(leaf): index for index, leaf in enumerate(live_leaves)}
    if len(full_leaf_id_to_index) != len(live_leaves):
      raise FunctionalMappingError(
          "P28 full engine state contains aliased leaf objects; explicit "
          "local-to-full gradient assembly is ambiguous"
      )

    def split_local(module, label):
      local_graphdef, local_state = nnx.split(module)
      local_treedef = jax.tree_util.tree_structure(local_state)
      local_leaves = tuple(jax.tree_util.tree_leaves(local_state))
      full_indices = []
      for local_index, leaf in enumerate(local_leaves):
        full_index = full_leaf_id_to_index.get(id(leaf))
        if full_index is None:
          raise FunctionalMappingError(
              f"P28 {label} local leaf {local_index} is absent from the "
              "full engine state"
          )
        full_indices.append(full_index)
      if len(set(full_indices)) != len(full_indices):
        raise FunctionalMappingError(
            f"P28 {label} local state maps to duplicate full-state leaves"
        )
      return (
          local_graphdef,
          local_treedef,
          local_leaves,
          tuple(full_indices),
      )

    def merge_local(graphdef, treedef, leaves):
      return nnx.merge(
          graphdef, jax.tree_util.tree_unflatten(treedef, leaves)
      )

    endpoint_contract = None
    embed_local_fn = None
    embed_pullback_fn = None
    embed_pullback_vma_fn = None
    norm_local_fn = None
    norm_pullback_fn = None
    norm_pullback_vma_fn = None
    head_local_fn = None
    head_pullback_fn = None
    head_pullback_vma_fn = None
    embed_full_indices = ()
    norm_full_indices = ()
    head_full_indices = ()
    embed_local_leaves = ()
    norm_local_leaves = ()
    head_local_leaves = ()
    tied_word_embeddings = False
    if os.environ.get("CANON_P28_SEGMENTED_TRAIN", "") == "1":
      model_config = getattr(runner, "model_config", None)
      hf_config = getattr(model_config, "hf_config", None)
      if hf_config is None:
        vllm_model_config = getattr(
            getattr(runner, "vllm_config", None), "model_config", None
        )
        hf_config = getattr(vllm_model_config, "hf_config", None)
      tied_word_embeddings = bool(
          getattr(hf_config, "tie_word_embeddings", False)
      )
      if not tied_word_embeddings and not hasattr(runner.model, "lm_head"):
        raise FunctionalMappingError(
            "P28 G5c untied model requires an explicit lm_head"
        )
      (
          embed_graphdef,
          embed_treedef,
          embed_local_leaves,
          embed_full_indices,
      ) = split_local(embed_tokens, "embed")
      (
          norm_graphdef,
          norm_treedef,
          norm_local_leaves,
          norm_full_indices,
      ) = split_local(final_norm, "final norm")
      if tied_word_embeddings:
        if not callable(getattr(embed_tokens, "decode", None)):
          raise FunctionalMappingError(
              "P28 G5c tied embeddings require embed_tokens.decode"
          )
        head_graphdef = embed_graphdef
        head_treedef = embed_treedef
        head_local_leaves = embed_local_leaves
        head_full_indices = embed_full_indices
      else:
        (
            head_graphdef,
            head_treedef,
            head_local_leaves,
            head_full_indices,
        ) = split_local(runner.model.lm_head, "lm head")
      if not embed_local_leaves or not norm_local_leaves or not head_local_leaves:
        raise FunctionalMappingError(
            "P28 G5c embed/norm/lm-head must each expose parameter leaves"
        )

      def fwd_embed(leaves, input_ids):
        module = merge_local(embed_graphdef, embed_treedef, leaves)
        return module(input_ids)

      def bwd_embed(leaves, input_ids, dhidden):
        _, pullback = jax.vjp(fwd_embed, leaves, input_ids)
        dleaves, _ = pullback(dhidden)
        return dleaves

      def fwd_norm(leaves, hidden):
        module = merge_local(norm_graphdef, norm_treedef, leaves)
        return module(hidden)

      def bwd_norm(leaves, hidden, dnormalized):
        _, pullback = jax.vjp(fwd_norm, leaves, hidden)
        return pullback(dnormalized)

      def fwd_lm_head(leaves, hidden):
        module = merge_local(head_graphdef, head_treedef, leaves)
        if tied_word_embeddings:
          return module.decode(hidden)
        return module(hidden)

      def bwd_lm_head(leaves, hidden, dlogits):
        _, pullback = jax.vjp(fwd_lm_head, leaves, hidden)
        return pullback(dlogits)

      embed_pullback_vma_fn = bwd_embed
      norm_pullback_vma_fn = bwd_norm
      head_pullback_vma_fn = bwd_lm_head

      embed_local_fn = _xprof_jit(
          fwd_embed,
          module_name="zt_tr_fwd_embed",
          scope_name="zt/tr/embed/fwd",
      )
      embed_pullback_fn = _xprof_jit(
          bwd_embed,
          module_name="zt_tr_bwd_embed",
          scope_name="zt/tr/embed/bwd",
      )
      norm_local_fn = _xprof_jit(
          fwd_norm,
          module_name="zt_tr_fwd_norm",
          scope_name="zt/tr/final_norm/fwd",
      )
      norm_pullback_fn = _xprof_jit(
          bwd_norm,
          module_name="zt_tr_bwd_norm",
          scope_name="zt/tr/final_norm/bwd",
      )
      head_local_fn = _xprof_jit(
          fwd_lm_head,
          module_name="zt_tr_fwd_head",
          scope_name="zt/tr/lm_head/fwd",
      )
      head_pullback_fn = _xprof_jit(
          bwd_lm_head,
          module_name="zt_tr_bwd_head",
          scope_name="zt/tr/lm_head/bwd",
      )
      endpoint_contract = {
          "embed": embed_full_indices,
          "norm": norm_full_indices,
          "head": head_full_indices,
      }
      if tied_word_embeddings:
        print(
            "[P28.G5C] TIED_EMBEDDING_HEAD on "
            f"shared_leaves={len(embed_full_indices)}",
            flush=True,
        )

    layer_fns = []
    local_layer_fns = []
    local_layer_defs = []
    local_layer_vjp_fns = []
    local_layer_pullback_fns = []
    local_layer_pullback_vma_fns = []
    local_layer_pullback_tape_fns = []
    local_layer_leaves = []
    local_layer_contracts = []
    for layer_index in range(start_layer, end_layer):

      def run_layer(
          state_leaves, cache, hidden, attention_metadata, *, _index=layer_index
      ):
        model = merge(state_leaves)
        return model.model.layers[_index](cache, hidden, attention_metadata)

      layer_fns.append(
          _xprof_jit(
              run_layer,
              module_name="zt_tr_ref_fwd_layer",
              scope_name="zt/tr/layer/reference",
          )
      )

      layer_graphdef, layer_state = nnx.split(layers[layer_index])
      layer_treedef = jax.tree_util.tree_structure(layer_state)
      layer_leaves = tuple(jax.tree_util.tree_leaves(layer_state))

      def fwd_layer(
          leaves, cache, hidden, attention_metadata,
          *, _graphdef=layer_graphdef, _treedef=layer_treedef
      ):
        state = jax.tree_util.tree_unflatten(_treedef, leaves)
        layer = nnx.merge(_graphdef, state)
        return layer(cache, hidden, attention_metadata)

      def block_objective(
          leaves, cache, hidden, attention_metadata,
          *, _run=fwd_layer
      ):
        next_cache, next_hidden = _run(
            leaves, cache, hidden, attention_metadata
        )
        row_seed = (
            (jnp.arange(next_hidden.shape[0], dtype=jnp.float32) % 17) + 1
        ) / 17.0
        loss = jnp.sum(
            next_hidden.astype(jnp.float32) * row_seed[:, None]
        ) / jnp.asarray(next_hidden.size, jnp.float32)
        return loss, (next_cache, next_hidden)

      def bwd_layer_block(
          leaves,
          cache,
          hidden,
          attention_metadata,
          dnext_cache,
          dnext_hidden,
          *,
          _run=fwd_layer,
      ):
        def primal(p, c, h):
          return _run(p, c, h, attention_metadata)

        _, pullback = jax.vjp(primal, leaves, cache, hidden)
        return pullback((dnext_cache, dnext_hidden))

      def bwd_layer_block_tape(
          leaves,
          stacked_caches,
          stacked_hidden,
          attention_metadata,
          dnext_cache,
          dnext_hidden,
          *,
          _run=fwd_layer,
          _index=layer_index,
      ):
        # scan_fwd reverse: this layer's (cache, hidden) tape entries are
        # static-index slices of the scanned stack, taken INSIDE the
        # pullback program so no standalone unstack dispatch exists.  The
        # slice is exact; the vjp body below is bwd_layer_block's,
        # unchanged, on the same values.
        cache = jax.tree.map(
            lambda x: jax.lax.index_in_dim(x, _index, 0, keepdims=False),
            stacked_caches,
        )
        hidden = jax.lax.index_in_dim(
            stacked_hidden, _index, 0, keepdims=False
        )

        def primal(p, c, h):
          return _run(p, c, h, attention_metadata)

        _, pullback = jax.vjp(primal, leaves, cache, hidden)
        return pullback((dnext_cache, dnext_hidden))

      local_layer_fns.append(
          _xprof_jit(
              fwd_layer,
              module_name="zt_tr_fwd_layer",
              scope_name="zt/tr/layer/fwd",
          )
      )
      local_layer_vjp_fns.append(
          _xprof_jit(
              jax.value_and_grad(
                  block_objective, argnums=(0, 1, 2), has_aux=True
              ),
              module_name="zt_tr_probe_vjp_layer",
              scope_name="zt/tr/layer/probe_vjp",
          )
      )
      local_layer_pullback_fns.append(
          _xprof_jit(
              bwd_layer_block,
              module_name="zt_tr_bwd_layer",
              scope_name="zt/tr/layer/bwd",
          )
      )
      local_layer_pullback_vma_fns.append(bwd_layer_block)
      local_layer_pullback_tape_fns.append(
          _xprof_jit(
              bwd_layer_block_tape,
              module_name="zt_tr_bwd_tape_layer",
              scope_name="zt/tr/layer/bwd_tape",
          )
      )
      local_layer_leaves.append(layer_leaves)
      local_layer_defs.append((layer_graphdef, layer_treedef))
      local_layer_contracts.append(
          SegmentedBlockVjpContract(
              layer_index=layer_index,
              local_state_leaves=len(layer_leaves),
              local_state_bytes=sum(
                  int(leaf.size * leaf.dtype.itemsize) for leaf in layer_leaves
              ),
              block_depth=1,
          )
      )

    local_layer_full_indices = []
    for layer_index, layer in enumerate(layers):
      _, _, _, full_indices = split_local(layer, f"layer {layer_index}")
      local_layer_full_indices.append(full_indices)

    if endpoint_contract is not None:
      if tied_word_embeddings and head_full_indices != embed_full_indices:
        raise FunctionalMappingError(
            "P28 G5c tied head must map exactly to embedding state leaves"
        )
      endpoint_groups = [
          ("embed", embed_full_indices),
          ("norm", norm_full_indices),
      ]
      if not tied_word_embeddings:
        endpoint_groups.append(("head", head_full_indices))
      covered = set()
      for label, full_indices in endpoint_groups:
        if covered.intersection(full_indices):
          raise FunctionalMappingError(
              f"P28 G5c {label} parameter group overlaps another endpoint"
          )
        covered.update(full_indices)
      for full_indices in local_layer_full_indices:
        if covered.intersection(full_indices):
          raise FunctionalMappingError(
              "P28 G5c local parameter groups overlap in full engine state"
          )
        covered.update(full_indices)
      expected = set(range(len(live_leaves)))
      if covered != expected:
        raise FunctionalMappingError(
            "P28 G5c local parameter groups do not cover the full engine "
            f"state: missing={sorted(expected - covered)} "
            f"extra={sorted(covered - expected)}"
        )

    self._embed_fn = jax.jit(embed)
    self._layer_fns = tuple(layer_fns)
    self._local_layer_fns = tuple(local_layer_fns)
    self._local_layer_defs = tuple(local_layer_defs)
    self._layer_scan_fn = None
    self._layer_tape_scan_fn = None
    self._layer_rev_scan_fn = None
    self._layer_unstack_fn = None
    self._layer_acc_fn = None
    self._layer_scan_stack = None
    self._local_layer_vjp_fns = tuple(local_layer_vjp_fns)
    self._local_layer_pullback_fns = tuple(local_layer_pullback_fns)
    self._local_layer_pullback_vma_fns = tuple(
        local_layer_pullback_vma_fns
    )
    self._local_layer_pullback_tape_fns = tuple(local_layer_pullback_tape_fns)
    self._local_layer_leaves = tuple(local_layer_leaves)
    self._local_layer_full_indices = tuple(local_layer_full_indices)
    self._local_layer_contracts = tuple(local_layer_contracts)
    self._norm_fn = jax.jit(norm)
    self._embed_local_fn = embed_local_fn
    self._embed_pullback_fn = embed_pullback_fn
    self._embed_pullback_vma_fn = embed_pullback_vma_fn
    self._norm_local_fn = norm_local_fn
    self._norm_pullback_fn = norm_pullback_fn
    self._norm_pullback_vma_fn = norm_pullback_vma_fn
    self._head_local_fn = head_local_fn
    self._head_pullback_fn = head_pullback_fn
    self._head_pullback_vma_fn = head_pullback_vma_fn
    self._embed_local_leaves = tuple(embed_local_leaves)
    self._norm_local_leaves = tuple(norm_local_leaves)
    self._head_local_leaves = tuple(head_local_leaves)
    self._embed_full_indices = tuple(embed_full_indices)
    self._norm_full_indices = tuple(norm_full_indices)
    self._head_full_indices = tuple(head_full_indices)
    self._tied_word_embeddings = tied_word_embeddings
    self._endpoint_contract = endpoint_contract
    self._full_state_leaves = tuple(live_leaves)
    self._num_state_leaves = len(live_leaves)
    self._captured_state_released = False
    self._p30_sparse_grad_assembly = (
        os.environ.get("CANON_P30_SPARSE_GRAD_ASSEMBLY", "") == "1"
    )
    if self._p30_sparse_grad_assembly:
      print(
          "[P30.G2] SPARSE_GRAD_ASSEMBLY on scalar_zero_canonicalization=1",
          flush=True,
      )
    self.contract = SegmentedForwardContract(
        implementation_id=(
            f"{type(runner.model).__module__}."
            f"{type(runner.model).__qualname__}:p28-segmented-layer1"
        ),
        state_leaves=len(live_leaves),
        start_layer=start_layer,
        end_layer=end_layer,
        block_depth=1,
    )

  @staticmethod
  def _reject_outer_transform(*trees):
    if any(
        isinstance(leaf, jax.core.Tracer)
        for tree in trees
        for leaf in jax.tree_util.tree_leaves(tree)
    ):
      raise FunctionalMappingError(
          "P28 segmented forward is a host boundary and must not be wrapped "
          "in jax.jit/value_and_grad"
      )

  @staticmethod
  def _state_spec(leaf):
    """Keeps an array's compile contract without retaining its live buffer."""
    sharding = getattr(leaf, "sharding", None)
    if sharding is None:
      return jax.ShapeDtypeStruct(leaf.shape, leaf.dtype)
    return jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=sharding)

  def release_captured_state(self):
    """Drops construction-time weights after explicit-state call paths exist."""
    if self._captured_state_released:
      raise FunctionalMappingError(
          "P30 segmented construction state was already released"
      )
    released_bytes = sum(
        int(leaf.size * leaf.dtype.itemsize)
        for leaf in self._full_state_leaves
    )
    self._full_state_leaves = tuple(
        self._state_spec(leaf) for leaf in self._full_state_leaves
    )
    self._local_layer_leaves = tuple(
        tuple(self._state_spec(leaf) for leaf in leaves)
        for leaves in self._local_layer_leaves
    )
    self._embed_local_leaves = tuple(
        self._state_spec(leaf) for leaf in self._embed_local_leaves
    )
    self._norm_local_leaves = tuple(
        self._state_spec(leaf) for leaf in self._norm_local_leaves
    )
    self._head_local_leaves = tuple(
        self._state_spec(leaf) for leaf in self._head_local_leaves
    )
    self._captured_state_released = True
    return released_bytes

  def run(
      self,
      state_leaves,
      caches,
      input_ids,
      attention_metadata,
      *,
      inputs_embeds=None,
  ):
    """Runs embed -> real layers -> final norm without an outer JIT."""
    self._reject_outer_transform(
        state_leaves,
        caches,
        input_ids,
        attention_metadata,
        inputs_embeds,
    )
    state_leaves = tuple(state_leaves)
    if len(state_leaves) != self._num_state_leaves:
      raise FunctionalMappingError(
          "P28 segmented state leaf count changed: "
          f"{len(state_leaves)} != {self._num_state_leaves}"
      )
    if len(caches) != len(self._layer_fns):
      raise FunctionalMappingError(
          "P28 segmented cache count changed: "
          f"{len(caches)} != {len(self._layer_fns)}"
      )
    hidden = (
        self._embed_fn(state_leaves, input_ids)
        if inputs_embeds is None
        else inputs_embeds
    )
    next_caches = []
    for layer_fn, cache in zip(self._layer_fns, caches):
      cache, hidden = layer_fn(
          state_leaves, cache, hidden, attention_metadata
      )
      next_caches.append(cache)
    hidden = self._norm_fn(state_leaves, hidden)
    return next_caches, hidden

  def run_block_vjp(
      self, layer_index, state_leaves, cache, hidden, attention_metadata
  ):
    """Runs one isolated real-layer primal and VJP without an outer transform."""
    self._reject_outer_transform(
        state_leaves, cache, hidden, attention_metadata
    )
    layer_index = int(layer_index)
    if layer_index < 0 or layer_index >= len(self._layer_fns):
      raise FunctionalMappingError(
          f"P28 block layer index out of range: {layer_index}"
      )
    if self._captured_state_released:
      raise FunctionalMappingError(
          "P30 run_block_vjp cannot use captured state after it was released"
      )
    reference = self._layer_fns[layer_index](
        tuple(state_leaves), cache, hidden, attention_metadata
    )
    local_leaves = self._local_layer_leaves[layer_index]
    isolated = self._local_layer_fns[layer_index](
        local_leaves, cache, hidden, attention_metadata
    )
    (loss_aux, gradients) = self._local_layer_vjp_fns[layer_index](
        local_leaves, cache, hidden, attention_metadata
    )
    loss, vjp_output = loss_aux
    return {
        "contract": self._local_layer_contracts[layer_index],
        "reference": reference,
        "isolated": isolated,
        "loss": loss,
        "vjp_output": vjp_output,
        "gradients": gradients,
    }

  def run_block_forward(
      self, layer_index, state_leaves, cache, hidden, attention_metadata
  ):
    """Returns full-state and isolated-state primals for one layer."""
    self._reject_outer_transform(
        state_leaves, cache, hidden, attention_metadata
    )
    layer_index = int(layer_index)
    reference = self._layer_fns[layer_index](
        tuple(state_leaves), cache, hidden, attention_metadata
    )
    state_leaves = tuple(state_leaves)
    local_leaves = tuple(
        state_leaves[index]
        for index in self._local_layer_full_indices[layer_index]
    )
    isolated = self._local_layer_fns[layer_index](
        local_leaves,
        cache,
        hidden,
        attention_metadata,
    )
    return reference, isolated

  def run_layer_forward(
      self, layer_index, state_leaves, cache, hidden, attention_metadata
  ):
    """Runs one isolated layer with explicit current engine-state leaves."""
    self._reject_outer_transform(
        state_leaves, cache, hidden, attention_metadata
    )
    layer_index = int(layer_index)
    if layer_index < 0 or layer_index >= len(self._local_layer_fns):
      raise FunctionalMappingError(
          f"P28 layer index out of range: {layer_index}"
      )
    state_leaves = tuple(state_leaves)
    if len(state_leaves) != self._num_state_leaves:
      raise FunctionalMappingError(
          "P28 layer state leaf count changed: "
          f"{len(state_leaves)} != {self._num_state_leaves}"
      )
    local_leaves = tuple(
        state_leaves[index]
        for index in self._local_layer_full_indices[layer_index]
    )
    return self._local_layer_fns[layer_index](
        local_leaves, cache, hidden, attention_metadata
    )

  def layer_scan_mode(self):
    """Returns the CANON_P28_LAYER_SCAN mode.

    '' (off) | '1' (tape/forward scan + loop pullbacks; byte-preserving) |
    'verify' (loop authoritative, bitwise-compare the pieces mode 1 uses) |
    'verify_rev' (verify plus the full reverse scan comparison -- expected
    RED: r3/r4 proved the scanned reverse reorders one norm-scale gradient
    reduction; kept as the reproducible THIRDPROG demonstration).
    """
    mode = os.environ.get("CANON_P28_LAYER_SCAN", "")
    if mode not in ("", "0", "1", "verify", "verify_rev"):
      raise FunctionalMappingError(
          "CANON_P28_LAYER_SCAN must be unset/0/1/verify/verify_rev, "
          f"got {mode!r}"
      )
    return "" if mode == "0" else mode

  def _ensure_layer_scan(self, engine_leaves):
    """Builds the shared scan body once and the leaf stack per leaves object.

    The scan body reuses fwd_layer's exact composition (unflatten ->
    nnx.merge -> layer(...)) with layer 0's graphdef/treedef; a non-uniform
    stack fails closed rather than silently falling back.
    """
    if self._layer_scan_fn is None:
      from flax import nnx  # pylint: disable=g-import-not-at-top

      graphdef0, treedef0 = self._local_layer_defs[0]

      def normalized_graphdef_repr(graphdef):
        # Layer graphdefs are allowed to differ ONLY by object identity of
        # per-layer init closures (init-time-only statics) and by layer-index
        # naming; erase both, keep every other Static value verbatim.
        text = repr(graphdef)
        text = re.sub(r" at 0x[0-9a-fA-F]+", " at 0xX", text)
        return re.sub(r"layers\.\d+", "layers.N", text)

      norm0 = None
      for index, (graphdef, treedef) in enumerate(
          self._local_layer_defs[1:], 1
      ):
        if treedef != treedef0:
          raise FunctionalMappingError(
              "P50 layer scan requires a uniform layer stack; layer "
              f"{index} treedef differs from layer 0"
          )
        if graphdef == graphdef0:
          continue
        if norm0 is None:
          norm0 = normalized_graphdef_repr(graphdef0)
        norm = normalized_graphdef_repr(graphdef)
        if norm != norm0:
          cut = next(
              (k for k in range(min(len(norm), len(norm0)))
               if norm[k] != norm0[k]),
              min(len(norm), len(norm0)),
          )
          raise FunctionalMappingError(
              "P50 layer scan requires a uniform layer stack; layer "
              f"{index} graphdef differs from layer 0 beyond closure "
              f"identity/naming at char {cut}: "
              f"...{norm0[max(0, cut - 120):cut + 120]!r} vs "
              f"...{norm[max(0, cut - 120):cut + 120]!r}"
          )
      if norm0 is not None:
        print(
            "[P50] layer graphdefs differ only by init-closure identity/"
            "layer naming; scan merges every layer with layer 0's graphdef "
            "(verify mode is the byte-level judge)",
            flush=True,
        )

      def scan_layers(stacked_leaves, stacked_caches, hidden, metadata):
        def body(h, xs):
          leaves, cache = xs
          state = jax.tree_util.tree_unflatten(treedef0, tuple(leaves))
          layer = nnx.merge(graphdef0, state)
          new_cache, new_h = layer(cache, h, metadata)
          return new_h, new_cache

        hidden_out, new_caches = jax.lax.scan(
            body, hidden, (list(stacked_leaves), stacked_caches)
        )
        return new_caches, hidden_out

      self._layer_scan_fn = jax.jit(scan_layers)

      def tape_scan_layers(stacked_leaves, stacked_caches, hidden, metadata):
        def body(h, xs):
          leaves, cache = xs
          state = jax.tree_util.tree_unflatten(treedef0, tuple(leaves))
          layer = nnx.merge(graphdef0, state)
          new_cache, new_h = layer(cache, h, metadata)
          return new_h, (h, new_cache)

        hidden_out, (hidden_ins, new_caches) = jax.lax.scan(
            body, hidden, (list(stacked_leaves), stacked_caches)
        )
        # new_caches is returned (not dropped inside the jit) so the scanned
        # tape keeps the loop path's materialization obligations.
        return hidden_ins, new_caches, hidden_out

      self._layer_tape_scan_fn = jax.jit(tape_scan_layers)

      def rev_scan_layers(
          stacked_leaves,
          stacked_cache_ins,
          stacked_hidden_ins,
          metadata,
          stacked_dcaches,
          dhidden,
      ):
        def body(dh, xs):
          leaves, cache_in, hidden_in, dcache = xs

          def primal(p, c, h):
            state = jax.tree_util.tree_unflatten(treedef0, tuple(p))
            layer = nnx.merge(graphdef0, state)
            return layer(c, h, metadata)

          _, pullback = jax.vjp(primal, tuple(leaves), cache_in, hidden_in)
          dleaves, dcache_in, dh_prev = pullback((dcache, dh))
          return dh_prev, (dleaves, dcache_in)

        dh_out, (stacked_dleaves, stacked_dcache_ins) = jax.lax.scan(
            body,
            dhidden,
            (
                list(stacked_leaves),
                stacked_cache_ins,
                stacked_hidden_ins,
                stacked_dcaches,
            ),
            reverse=True,
        )
        return stacked_dleaves, stacked_dcache_ins, dh_out

      self._layer_rev_scan_fn = jax.jit(rev_scan_layers)

      layer_total = len(self._local_layer_defs)

      def unstack_hiddens(stacked):
        return tuple(stacked[index] for index in range(layer_total))

      self._layer_unstack_fn = jax.jit(unstack_hiddens)

      def accumulate_grads(acc, delta):
        return jax.tree.map(lambda a, b: a + b, acc, delta)

      self._layer_acc_fn = jax.jit(accumulate_grads)
    if (
        self._layer_scan_stack is None
        or self._layer_scan_stack[0] is not engine_leaves
    ):
      per_layer = [
          tuple(engine_leaves[index] for index in indices)
          for indices in self._local_layer_full_indices
      ]
      width = len(per_layer[0])
      if any(len(leaves) != width for leaves in per_layer[1:]):
        raise FunctionalMappingError(
            "P50 layer scan requires equal leaf counts per layer"
        )
      stacked = tuple(
          jnp.stack([leaves[k] for leaves in per_layer])
          for k in range(width)
      )
      if self._layer_scan_stack is None:
        print(
            "[P50] stacked leaf shardings: "
            + "; ".join(
                f"{tuple(x.shape)}:{x.sharding}"
                for x in (stacked[0], per_layer[0][0])
            ),
            flush=True,
        )
      self._layer_scan_stack = (engine_leaves, stacked)
    return self._layer_scan_stack[1]

  def run_layers_scan(self, engine_leaves, caches, hidden, metadata):
    """Runs the whole layer stack as one scanned program."""
    stacked_leaves = self._ensure_layer_scan(engine_leaves)
    stacked_caches = jax.tree.map(lambda *xs: jnp.stack(xs), *caches)
    new_stacked, hidden_out = self._layer_scan_fn(
        stacked_leaves, stacked_caches, hidden, metadata
    )
    layer_count = len(self._local_layer_fns)
    new_caches = tuple(
        jax.tree.map(lambda x, _i=i: x[_i], new_stacked)
        for i in range(layer_count)
    )
    return new_caches, hidden_out

  def run_layers_tape_scan(self, engine_leaves, caches, hidden, metadata):
    """Rebuilds the layer tape as one scanned program."""
    stacked_leaves = self._ensure_layer_scan(engine_leaves)
    stacked_cache_ins = jax.tree.map(lambda *xs: jnp.stack(xs), *caches)
    hidden_ins, new_caches, hidden_out = self._layer_tape_scan_fn(
        stacked_leaves, stacked_cache_ins, hidden, metadata
    )
    del new_caches
    return stacked_cache_ins, hidden_ins, hidden_out

  def run_layers_rev_scan(
      self,
      engine_leaves,
      stacked_cache_ins,
      stacked_hidden_ins,
      metadata,
      stacked_dcaches,
      dhidden,
  ):
    """Applies the whole layer pullback stack as one reverse scan."""
    stacked_leaves = self._ensure_layer_scan(engine_leaves)
    return self._layer_rev_scan_fn(
        stacked_leaves,
        stacked_cache_ins,
        stacked_hidden_ins,
        metadata,
        stacked_dcaches,
        dhidden,
    )

  def unstack_hidden_ins(self, engine_leaves, stacked_hidden_ins):
    """Splits the scanned tape hiddens into per-layer arrays in one call."""
    self._ensure_layer_scan(engine_leaves)
    return self._layer_unstack_fn(stacked_hidden_ins)

  def accumulate_layer_grads(self, engine_leaves, layer_grads, chunk_grads):
    """Adds one chunk's per-layer gradients in a single elementwise call."""
    self._ensure_layer_scan(engine_leaves)
    return self._layer_acc_fn(layer_grads, chunk_grads)

  def zero_gradient_pack(self, final_caches):
    """Returns the cached zero accumulators (values are literal zeros)."""
    if getattr(self, "_p52_zero_pack", None) is None:
      tree_zeros = lambda tree: jax.tree.map(jnp.zeros_like, tree)
      self._p52_zero_pack = (
          tuple(tree_zeros(leaves) for leaves in self._local_layer_leaves),
          tree_zeros(self._embed_local_leaves),
          tree_zeros(self._norm_local_leaves),
          tree_zeros(self._head_local_leaves),
          tuple(tree_zeros(cache) for cache in final_caches),
      )
    return self._p52_zero_pack

  def run_block_pullback(
      self,
      layer_index,
      cache,
      hidden,
      attention_metadata,
      dnext_cache,
      dnext_hidden,
      *,
      state_leaves=None,
  ):
    """Applies one real layer pullback to caller-provided output cotangents."""
    # Issue-anatomy timers: R2-R4 showed the reverse issue segment pinned
    # at ~14s while every dispatch-count lever left it unmoved, so the
    # suspect is this call itself -- python prep (validation walk, the
    # per-call leaf tuple) versus the jitted dispatch. The [PERF]
    # vag_reverse line reports both sums.
    _anatomy_t0 = time.perf_counter()
    self._reject_outer_transform(
        cache,
        hidden,
        attention_metadata,
        dnext_cache,
        dnext_hidden,
    )
    layer_index = int(layer_index)
    if layer_index < 0 or layer_index >= len(self._local_layer_pullback_fns):
      raise FunctionalMappingError(
          f"P28 pullback layer index out of range: {layer_index}"
      )
    if state_leaves is None and self._captured_state_released:
      raise FunctionalMappingError(
          "P30 pullback requires explicit current state after captured state "
          "was released"
      )
    local_leaves = self._local_layer_leaves[layer_index]
    if state_leaves is not None:
      state_leaves = tuple(state_leaves)
      if len(state_leaves) != self._num_state_leaves:
        raise FunctionalMappingError(
            "P28 pullback state leaf count changed: "
            f"{len(state_leaves)} != {self._num_state_leaves}"
        )
      local_leaves = tuple(
          state_leaves[index]
          for index in self._local_layer_full_indices[layer_index]
      )
    _anatomy_t1 = time.perf_counter()
    result = self._local_layer_pullback_fns[layer_index](
        local_leaves,
        cache,
        hidden,
        attention_metadata,
        dnext_cache,
        dnext_hidden,
    )
    _anatomy_t2 = time.perf_counter()
    _ISSUE_ANATOMY["prep"] += _anatomy_t1 - _anatomy_t0
    _ISSUE_ANATOMY["call"] += _anatomy_t2 - _anatomy_t1
    _ISSUE_ANATOMY["n"] += 1
    return result

  def run_block_pullback_tape(
      self,
      layer_index,
      stacked_caches,
      stacked_hidden,
      attention_metadata,
      dnext_cache,
      dnext_hidden,
      *,
      state_leaves=None,
  ):
    """Applies one layer pullback that slices its tape inputs in-program."""
    _anatomy_t0 = time.perf_counter()
    self._reject_outer_transform(
        stacked_caches,
        stacked_hidden,
        attention_metadata,
        dnext_cache,
        dnext_hidden,
    )
    layer_index = int(layer_index)
    if layer_index < 0 or layer_index >= len(
        self._local_layer_pullback_tape_fns
    ):
      raise FunctionalMappingError(
          f"P28 tape pullback layer index out of range: {layer_index}"
      )
    if state_leaves is None and self._captured_state_released:
      raise FunctionalMappingError(
          "P30 pullback requires explicit current state after captured state "
          "was released"
      )
    local_leaves = self._local_layer_leaves[layer_index]
    if state_leaves is not None:
      state_leaves = tuple(state_leaves)
      if len(state_leaves) != self._num_state_leaves:
        raise FunctionalMappingError(
            "P28 pullback state leaf count changed: "
            f"{len(state_leaves)} != {self._num_state_leaves}"
        )
      local_leaves = tuple(
          state_leaves[index]
          for index in self._local_layer_full_indices[layer_index]
      )
    _anatomy_t1 = time.perf_counter()
    result = self._local_layer_pullback_tape_fns[layer_index](
        local_leaves,
        stacked_caches,
        stacked_hidden,
        attention_metadata,
        dnext_cache,
        dnext_hidden,
    )
    _anatomy_t2 = time.perf_counter()
    _ISSUE_ANATOMY["prep"] += _anatomy_t1 - _anatomy_t0
    _ISSUE_ANATOMY["call"] += _anatomy_t2 - _anatomy_t1
    _ISSUE_ANATOMY["n"] += 1
    return result

  def _require_full_loss_endpoints(self):
    if self._endpoint_contract is None:
      raise FunctionalMappingError(
          "P28 full-loss endpoints require CANON_P28_SEGMENTED_TRAIN=1 "
          "before the segmented engine is built"
      )

  def _endpoint_leaves(self, state_leaves, indices, captured, label):
    if state_leaves is None:
      if self._captured_state_released:
        raise FunctionalMappingError(
            f"P30 {label} requires explicit current state after captured "
            "state was released"
        )
      return captured
    state_leaves = tuple(state_leaves)
    if len(state_leaves) != self._num_state_leaves:
      raise FunctionalMappingError(
          f"P28 G5c {label} state leaf count changed: "
          f"{len(state_leaves)} != {self._num_state_leaves}"
      )
    return tuple(state_leaves[index] for index in indices)

  def run_embed_forward(self, input_ids, *, state_leaves=None):
    """Runs the isolated embedding endpoint used by the G5c host schedule."""
    self._reject_outer_transform(input_ids)
    self._require_full_loss_endpoints()
    leaves = self._endpoint_leaves(
        state_leaves,
        self._embed_full_indices,
        self._embed_local_leaves,
        "embed",
    )
    return self._embed_local_fn(leaves, input_ids)

  def _p59_parallel_map(
      self,
      local_fn,
      args,
      out_specs_factory,
      *,
      rank_local_arg_indices,
      module_name,
      scope_name,
  ):
    """Maps one pullback manually over DP while non-unit TP stays automatic."""
    trainer_mesh, _ = _p59_replicated_data_mesh(
        args[0], module_name
    )
    _, trainer_model_axis = _p59_mesh_roles(trainer_mesh, module_name)
    # A TP>1 engine body contains nested shard_maps and named fixed-order TP
    # collectives. Its surrounding P59 map therefore uses the exact concrete
    # engine mesh vocabulary, then relabels the result back to the trainer
    # mesh after the compiled boundary. TP1 keeps the already-certified
    # trainer-mesh carrier.
    mesh = (
        _p59_engine_data_model_mesh(self._engine_mesh, module_name)
        if int(trainer_mesh.shape[trainer_model_axis]) > 1
        else trainer_mesh
    )
    data_axis, _ = _p59_mesh_roles(mesh, module_name)
    _, model_axis = _p59_mesh_roles(mesh, module_name)
    aligned_args = tuple(
        _p59_align_to_mesh(value, mesh, module_name) for value in args
    )
    p66_unit_data = (
        int(mesh.shape[data_axis]) == 1
        and int(mesh.shape[model_axis]) == 4
        and _p66_tp4_arm()
        in (
            "tp4-p59-old",
            "tp4-p59",
            "tp4-gather-off",
            "tp4-vma-oracle",
        )
    )
    if int(mesh.shape[data_axis]) <= 1 and not p66_unit_data:
      raise FunctionalMappingError(
          f"{module_name} requires a multi-rank data mesh"
      )
    # TP1 retains the certified data-plus-unit-model carrier. At TP>1 the
    # reduced engine mesh makes both real axes manual so nested engine kernels
    # reuse, rather than replace or repartition, their TP collective name.
    manual_axes = (
        frozenset(mesh.axis_names)
        if int(mesh.shape[_p59_mesh_roles(mesh, module_name)[1]]) > 1
        else frozenset(_p59_manual_rank_axes(
            mesh, data_axis, module_name
        ))
    )
    p66_check_vma_value = os.environ.get(
        "CANON_P66_P59_CHECK_VMA", "0"
    )
    if p66_check_vma_value not in ("0", "1"):
      raise FunctionalMappingError(
          "CANON_P66_P59_CHECK_VMA must be exactly 0 or 1, got "
          f"{p66_check_vma_value!r}"
      )
    p66_check_vma = p66_check_vma_value == "1"
    if p66_check_vma:
      print(
          f"[P66.VMA] outer_check_enabled module={module_name} "
          f"manual_axes={sorted(manual_axes)}",
          flush=True,
      )
    mapped_local_fn = local_fn
    if p66_check_vma:

      def vma_local_fn(*local_args):
        # Parameter/state arguments are physically replicated over DP, but
        # P59 intentionally computes and stages one *local* parameter
        # cotangent per DP rank. Tell the inner VJP that those values may vary
        # over the manual data axis so reverse mode does not insert a DP psum
        # before the leading-rank staging boundary. This pcast is a runtime
        # no-op; TP placement and rank-local data arguments remain unchanged.
        def mark_data_varying(leaf):
          manual_axis_type = jax.typeof(leaf).mat
          if data_axis in manual_axis_type.varying:
            return leaf
          if (
              data_axis in manual_axis_type.unreduced
              or data_axis in manual_axis_type.reduced
          ):
            raise FunctionalMappingError(
                f"{module_name} cannot relabel {data_axis!r} from "
                f"{manual_axis_type} to varying"
            )
          return jax.lax.pcast(leaf, data_axis, to="varying")

        localized_args = tuple(
            value
            if index in rank_local_arg_indices
            else jax.tree.map(
                mark_data_varying,
                value,
            )
            for index, value in enumerate(local_args)
        )
        return local_fn(*localized_args)

      mapped_local_fn = vma_local_fn
    mapped = jax.shard_map(
        mapped_local_fn,
        mesh=mesh,
        in_specs=tuple(
            _rank_local_leading_specs(
                value,
                data_axis,
                int(mesh.shape[data_axis]),
                module_name,
                manual_axes,
            )
            if index in rank_local_arg_indices
            else _manual_axis_specs(value, data_axis, manual_axes)
            for index, value in enumerate(aligned_args)
        ),
        out_specs=out_specs_factory(
            data_axis, int(mesh.shape[data_axis]), aligned_args, manual_axes
        ),
        axis_names=manual_axes,
        check_vma=p66_check_vma,
    )
    compiled = _xprof_jit(
        mapped, module_name=module_name, scope_name=scope_name
    )

    def invoke(*runtime_args):
      aligned_runtime_args = tuple(
          _p59_align_to_mesh(value, mesh, module_name)
          for value in runtime_args
      )
      with _p59_localize_engine_shard_maps(mesh, module_name):
        result = compiled(*aligned_runtime_args)
      if mesh != trainer_mesh:
        result = _p59_align_to_mesh(
            result, trainer_mesh, f"{module_name} output"
        )
      return result

    return invoke

  @staticmethod
  def _p59_stage_rank_gradient(tree):
    return jax.tree.map(lambda value: jnp.expand_dims(value, 0), tree)

  def run_embed_pullback(self, input_ids, dhidden, *, state_leaves=None):
    """Returns local embedding parameter cotangents."""
    self._reject_outer_transform(input_ids, dhidden)
    self._require_full_loss_endpoints()
    leaves = self._endpoint_leaves(
        state_leaves,
        self._embed_full_indices,
        self._embed_local_leaves,
        "embed",
    )
    return self._embed_pullback_fn(leaves, input_ids, dhidden)

  def run_embed_pullback_rank_parallel(
      self, input_ids, dhidden, *, state_leaves=None
  ):
    """Returns one physically local embedding gradient row per DP rank."""
    self._reject_outer_transform(input_ids, dhidden)
    self._require_full_loss_endpoints()
    leaves = self._endpoint_leaves(
        state_leaves,
        self._embed_full_indices,
        self._embed_local_leaves,
        "embed",
    )
    if getattr(self, "_p59_embed_pullback_fn", None) is None:
      pullback_fn = (
          getattr(
              self, "_embed_pullback_vma_fn", self._embed_pullback_fn
          )
          if os.environ.get("CANON_P66_P59_CHECK_VMA", "0") == "1"
          else self._embed_pullback_fn
      )

      def local_pullback(local_leaves, local_ids, local_dhidden):
        gradients = pullback_fn(
            local_leaves, local_ids, local_dhidden
        )
        return self._p59_stage_rank_gradient(gradients)

      self._p59_embed_pullback_fn = self._p59_parallel_map(
          local_pullback,
          (leaves, input_ids, dhidden),
          lambda data_axis, axis_size, aligned, manual_axes: _rank_staged_specs(
              aligned[0], data_axis, manual_axes
          ),
          rank_local_arg_indices=(1, 2),
          module_name="zt_tr_dp_parallel_bwd_embed",
          scope_name="zt/tr/dp_parallel/embed/bwd",
      )
    return self._p59_embed_pullback_fn(leaves, input_ids, dhidden)

  def run_norm_forward(self, hidden, *, state_leaves=None):
    """Runs the isolated final-norm endpoint used by the G5c host schedule."""
    self._reject_outer_transform(hidden)
    self._require_full_loss_endpoints()
    leaves = self._endpoint_leaves(
        state_leaves,
        self._norm_full_indices,
        self._norm_local_leaves,
        "final norm",
    )
    return self._norm_local_fn(leaves, hidden)

  def run_norm_pullback(self, hidden, dnormalized, *, state_leaves=None):
    """Returns final-norm parameter and hidden cotangents."""
    self._reject_outer_transform(hidden, dnormalized)
    self._require_full_loss_endpoints()
    leaves = self._endpoint_leaves(
        state_leaves,
        self._norm_full_indices,
        self._norm_local_leaves,
        "final norm",
    )
    return self._norm_pullback_fn(leaves, hidden, dnormalized)

  def run_norm_pullback_rank_parallel(
      self, hidden, dnormalized, *, state_leaves=None
  ):
    """Returns staged norm gradients and the ordinary sharded hidden VJP."""
    self._reject_outer_transform(hidden, dnormalized)
    self._require_full_loss_endpoints()
    leaves = self._endpoint_leaves(
        state_leaves,
        self._norm_full_indices,
        self._norm_local_leaves,
        "final norm",
    )
    if getattr(self, "_p59_norm_pullback_fn", None) is None:
      pullback_fn = (
          getattr(self, "_norm_pullback_vma_fn", self._norm_pullback_fn)
          if os.environ.get("CANON_P66_P59_CHECK_VMA", "0") == "1"
          else self._norm_pullback_fn
      )

      def local_pullback(local_leaves, local_hidden, local_dnormalized):
        gradients, dhidden = pullback_fn(
            local_leaves, local_hidden, local_dnormalized
        )
        return self._p59_stage_rank_gradient(gradients), dhidden

      self._p59_norm_pullback_fn = self._p59_parallel_map(
          local_pullback,
          (leaves, hidden, dnormalized),
          lambda data_axis, axis_size, aligned, manual_axes: (
              _rank_staged_specs(aligned[0], data_axis, manual_axes),
              _rank_local_leading_specs(
                  aligned[1], data_axis, axis_size, "P59 norm output",
                  manual_axes,
              ),
          ),
          rank_local_arg_indices=(1, 2),
          module_name="zt_tr_dp_parallel_bwd_norm",
          scope_name="zt/tr/dp_parallel/final_norm/bwd",
      )
    return self._p59_norm_pullback_fn(leaves, hidden, dnormalized)

  def run_head_forward(self, hidden, *, state_leaves=None):
    """Runs the isolated tied or untied output endpoint used by G5c."""
    self._reject_outer_transform(hidden)
    self._require_full_loss_endpoints()
    leaves = self._endpoint_leaves(
        state_leaves,
        self._head_full_indices,
        self._head_local_leaves,
        "lm head",
    )
    return self._head_local_fn(leaves, hidden)

  def run_head_pullback(self, hidden, dlogits, *, state_leaves=None):
    """Returns lm-head parameter and normalized-hidden cotangents."""
    self._reject_outer_transform(hidden, dlogits)
    self._require_full_loss_endpoints()
    leaves = self._endpoint_leaves(
        state_leaves,
        self._head_full_indices,
        self._head_local_leaves,
        "lm head",
    )
    return self._head_pullback_fn(leaves, hidden, dlogits)

  def run_head_pullback_rank_parallel(
      self, hidden, dlogits, *, state_leaves=None
  ):
    """Returns staged head gradients and the ordinary sharded hidden VJP."""
    self._reject_outer_transform(hidden, dlogits)
    self._require_full_loss_endpoints()
    leaves = self._endpoint_leaves(
        state_leaves,
        self._head_full_indices,
        self._head_local_leaves,
        "lm head",
    )
    trainer_mesh, _ = _p59_replicated_data_mesh(
        leaves, "P59 head cotangent"
    )
    dlogits = _p59_partition_head_cotangent(
        dlogits, trainer_mesh, "P59 head cotangent"
    )
    if getattr(self, "_p59_head_pullback_fn", None) is None:
      data_axis, model_axis = _p59_mesh_roles(
          trainer_mesh, "P59 head cotangent"
      )
      local_rows = int(dlogits.shape[0]) // int(
          trainer_mesh.shape[data_axis]
      )
      local_vocab = int(dlogits.shape[1]) // int(
          trainer_mesh.shape[model_axis]
      )
      print(
          f"[P59.DP{int(trainer_mesh.shape[data_axis])}] "
          "head_cotangent_partition_ready "
          f"global_shape={tuple(map(int, dlogits.shape))} "
          f"local_shape=({local_rows},{local_vocab}) "
          f"placement={data_axis},{model_axis}",
          flush=True,
      )
      pullback_fn = (
          getattr(self, "_head_pullback_vma_fn", self._head_pullback_fn)
          if os.environ.get("CANON_P66_P59_CHECK_VMA", "0") == "1"
          else self._head_pullback_fn
      )

      def local_pullback(local_leaves, local_hidden, local_dlogits):
        gradients, dhidden = pullback_fn(
            local_leaves, local_hidden, local_dlogits
        )
        return self._p59_stage_rank_gradient(gradients), dhidden

      self._p59_head_pullback_fn = self._p59_parallel_map(
          local_pullback,
          (leaves, hidden, dlogits),
          lambda data_axis, axis_size, aligned, manual_axes: (
              _rank_staged_specs(aligned[0], data_axis, manual_axes),
              _rank_local_leading_specs(
                  aligned[1], data_axis, axis_size, "P59 head output",
                  manual_axes,
              ),
          ),
          rank_local_arg_indices=(1, 2),
          module_name="zt_tr_dp_parallel_bwd_head",
          scope_name="zt/tr/dp_parallel/lm_head/bwd",
      )
    return self._p59_head_pullback_fn(leaves, hidden, dlogits)

  def run_block_pullback_rank_parallel(
      self,
      layer_index,
      cache,
      hidden,
      attention_metadata,
      dnext_cache,
      dnext_hidden,
      *,
      state_leaves=None,
  ):
    """Returns staged layer gradients and ordinary sharded input VJPs."""
    self._reject_outer_transform(
        cache,
        hidden,
        attention_metadata,
        dnext_cache,
        dnext_hidden,
    )
    layer_index = int(layer_index)
    if layer_index < 0 or layer_index >= len(self._local_layer_pullback_fns):
      raise FunctionalMappingError(
          f"P59 pullback layer index out of range: {layer_index}"
      )
    local_leaves = self._local_layer_leaves[layer_index]
    if state_leaves is not None:
      state_leaves = tuple(state_leaves)
      if len(state_leaves) != self._num_state_leaves:
        raise FunctionalMappingError(
            "P59 pullback state leaf count changed: "
            f"{len(state_leaves)} != {self._num_state_leaves}"
        )
      local_leaves = tuple(
          state_leaves[index]
          for index in self._local_layer_full_indices[layer_index]
      )
    functions = getattr(self, "_p59_layer_pullback_fns", None)
    if functions is None:
      functions = [None] * len(self._local_layer_pullback_fns)
      self._p59_layer_pullback_fns = functions
    if functions[layer_index] is None:
      pullback_fn = (
          getattr(
              self,
              "_local_layer_pullback_vma_fns",
              self._local_layer_pullback_fns,
          )[layer_index]
          if os.environ.get("CANON_P66_P59_CHECK_VMA", "0") == "1"
          else self._local_layer_pullback_fns[layer_index]
      )

      def local_pullback(
          leaves,
          local_cache,
          local_hidden,
          local_metadata,
          local_dcache,
          local_dhidden,
      ):
        gradients, dcache, dhidden = pullback_fn(
            leaves,
            local_cache,
            local_hidden,
            local_metadata,
            local_dcache,
            local_dhidden,
        )
        return self._p59_stage_rank_gradient(gradients), dcache, dhidden

      functions[layer_index] = self._p59_parallel_map(
          local_pullback,
          (
              local_leaves,
              cache,
              hidden,
              attention_metadata,
              dnext_cache,
              dnext_hidden,
          ),
          lambda data_axis, axis_size, aligned, manual_axes: (
              _rank_staged_specs(aligned[0], data_axis, manual_axes),
              _rank_local_leading_specs(
                  aligned[1], data_axis, axis_size, "P59 cache output",
                  manual_axes,
              ),
              _rank_local_leading_specs(
                  aligned[2], data_axis, axis_size, "P59 layer output",
                  manual_axes,
              ),
          ),
          rank_local_arg_indices=(1, 2, 4, 5),
          module_name=f"zt_tr_dp_parallel_bwd_layer_{layer_index:02d}",
          scope_name=f"zt/tr/dp_parallel/layer/{layer_index:02d}/bwd",
      )
    return functions[layer_index](
        local_leaves,
        cache,
        hidden,
        attention_metadata,
        dnext_cache,
        dnext_hidden,
    )

  def assemble_full_state_gradient(
      self, *, embed, layers, norm, head, rank_axis_size=None
  ):
    """Assembles ordinary or rank-staged cotangents in full-state order."""
    self._reject_outer_transform(embed, layers, norm, head)
    self._require_full_loss_endpoints()
    if rank_axis_size is not None:
      rank_axis_size = int(rank_axis_size)
      p66_unit_rank = (
          rank_axis_size == 1
          and self._engine_data_size == 1
          and self._engine_tp_size == 4
          and _p66_tp4_arm()
          in (
              "tp4-p59-old",
              "tp4-p59",
              "tp4-gather-off",
              "tp4-vma-oracle",
          )
      )
      if rank_axis_size <= 1 and not p66_unit_rank:
        raise FunctionalMappingError(
            f"P59 staged rank axis must exceed one, got {rank_axis_size}"
        )
      if not self._p30_sparse_grad_assembly:
        raise FunctionalMappingError(
            "P59 rank-parallel assembly requires "
            "CANON_P30_SPARSE_GRAD_ASSEMBLY=1"
        )
    if len(layers) != len(self._local_layer_full_indices):
      raise FunctionalMappingError(
          "P28 G5c layer-gradient count changed: "
          f"{len(layers)} != {len(self._local_layer_full_indices)}"
      )
    if self._p30_sparse_grad_assembly:
      # The endpoint/layer construction already proves exact disjoint cover.
      # Avoid keeping a second full zero tree alive while the local VJP trees
      # are assembled.  Scalar +0 deliberately preserves the legacy signed-
      # zero canonicalization and therefore the optimizer/update bit contract.
      full = [None] * len(self._full_state_leaves)
    else:
      full = [jnp.zeros_like(leaf) for leaf in self._full_state_leaves]

    def add(indices, values, label):
      values = tuple(jax.tree_util.tree_leaves(values))
      if len(indices) != len(values):
        raise FunctionalMappingError(
            f"P28 G5c {label} cotangent count changed: "
            f"{len(values)} != {len(indices)}"
        )
      for index, value in zip(indices, values, strict=True):
        target = self._full_state_leaves[index]
        target_shape = (
            target.shape
            if rank_axis_size is None
            else (rank_axis_size,) + target.shape
        )
        if value.shape != target_shape:
          raise FunctionalMappingError(
              f"P28 G5c {label} cotangent shape changed at full leaf "
              f"{index}: {value.shape} != {target_shape}"
          )
        if self._p30_sparse_grad_assembly:
          if full[index] is not None:
            raise FunctionalMappingError(
                f"P30 G2 duplicate cotangent at full leaf {index}"
            )
          cast = value.astype(target.dtype)
          full[index] = jnp.asarray(0, target.dtype) + cast
        else:
          full[index] = full[index] + value.astype(full[index].dtype)

    if self._tied_word_embeddings:
      embed_values = tuple(jax.tree_util.tree_leaves(embed))
      head_values = tuple(jax.tree_util.tree_leaves(head))
      if (
          len(embed_values) != len(self._embed_full_indices)
          or len(head_values) != len(self._embed_full_indices)
      ):
        raise FunctionalMappingError(
            "P28 G5c tied embed/head cotangent count changed"
        )
      tied_values = []
      for index, embed_value, head_value in zip(
          self._embed_full_indices,
          embed_values,
          head_values,
          strict=True,
      ):
        target = self._full_state_leaves[index]
        target_shape = (
            target.shape
            if rank_axis_size is None
            else (rank_axis_size,) + target.shape
        )
        if (
            embed_value.shape != target_shape
            or head_value.shape != target_shape
        ):
          raise FunctionalMappingError(
              "P28 G5c tied embed/head cotangent shape changed at full leaf "
              f"{index}: {embed_value.shape}/{head_value.shape} != "
              f"{target_shape}"
          )
        tied_values.append(
            embed_value.astype(target.dtype) + head_value.astype(target.dtype)
        )
      add(self._embed_full_indices, tuple(tied_values), "tied embed/head")
    else:
      add(self._embed_full_indices, embed, "embed")
    for layer_index, (indices, values) in enumerate(
        zip(self._local_layer_full_indices, layers, strict=True)
    ):
      add(indices, values, f"layer {layer_index}")
    add(self._norm_full_indices, norm, "norm")
    if not self._tied_word_embeddings:
      add(self._head_full_indices, head, "head")
    if self._p30_sparse_grad_assembly:
      missing = [index for index, value in enumerate(full) if value is None]
      if missing:
        raise FunctionalMappingError(
            f"P30 G2 missing full-state cotangents: {missing}"
        )
    return tuple(full)


def build_p28_segmented_engine_forward(runner):
  """Builds the default-off P28 forward-only depth-segmentation probe."""
  return _P28SegmentedEngineForward(runner)


def _weight_attestation_mesh_shape(runner: Any) -> tuple[tuple[str, int], ...]:
  """Returns the public logical mesh while validating the live engine mesh."""
  engine_shape = {
      str(name): int(size) for name, size in runner.mesh.shape.items()
  }
  if os.environ.get("CANON_P34_DEEPSWE", "") != "1":
    return tuple(engine_shape.items())

  workload = deepswe_contract.active_workload(os.environ)
  if (
      engine_shape.get("data") != workload.dp_size
      or engine_shape.get("model") != workload.tp_size
  ):
    raise FunctionalMappingError(
        "live engine mesh differs from the signed DeepSWE logical mesh: "
        f"engine={engine_shape} expected_dp={workload.dp_size} "
        f"expected_tp={workload.tp_size}"
    )
  unexpected_nontrivial = {
      name: size
      for name, size in engine_shape.items()
      if name not in ("data", "model") and size != 1
  }
  if unexpected_nontrivial:
    raise FunctionalMappingError(
        "live engine has an unregistered nontrivial mesh axis: "
        f"{unexpected_nontrivial}"
    )
  return (("dp", workload.dp_size), ("tp", workload.tp_size))


def attest_exact_live_engine_weights(
    *,
    sampler: Any | None,
    trainer_state: Any,
    runner: Any | None = None,
    engine_state_contract: Any | None = None,
    key_mappings: Mapping[str, tuple[str, tuple[str | None, ...]]] | None = None,
    transpose_keys: Mapping[str, tuple[int, ...]] | None = None,
    key_mapping_hook_fns: Mapping[str, Any] | None = None,
    tp_size: int | None = None,
) -> dict[str, Any]:
  """Bitwise-compares a trainer anchor with an unmodified live engine.

  This is an observer only. It neither constructs/registers the canonical
  forward adapter nor replaces any serving function. The optional explicit
  arguments let the registered adapter reuse the exact same comparison.
  """
  if runner is None:
    if sampler is None:
      raise FunctionalMappingError(
          "exact live-engine weight attestation requires a sampler or runner"
      )
    try:
      runner = sampler._model_runner  # pylint: disable=protected-access
    except (AttributeError, RuntimeError) as exc:
      raise FunctionalMappingError("rollout has no live model runner") from exc
  required = ("state", "state_leaves", "mesh", "model_config")
  missing = [name for name in required if not hasattr(runner, name)]
  if missing:
    raise FunctionalMappingError(
        f"live engine is missing weight-attestation attributes: {missing}"
    )
  if engine_state_contract is None:
    engine_state_contract = runner.state
  if key_mappings is None:
    key_mappings = (
        getattr(sampler, "to_hf_key_mappings", None) or {}
        if sampler is not None
        else {}
    )
  if transpose_keys is None and sampler is not None:
    transpose_keys = getattr(sampler, "to_hf_transpose_keys", None)
  if key_mapping_hook_fns is None and sampler is not None:
    key_mapping_hook_fns = getattr(sampler, "to_hf_hook_fns", None)
  if tp_size is None:
    tp_size = int(
        getattr(sampler, "args", {}).get("tensor_parallel_size", 1)
        if sampler is not None
        else 1
    )

  model_config = runner.model_config
  mapped = map_trainer_state_to_engine_leaves(
      trainer_state=trainer_state,
      engine_state_contract=engine_state_contract,
      key_mappings=key_mappings,
      transpose_keys=transpose_keys,
      key_mapping_hook_fns=key_mapping_hook_fns,
      num_kv_heads=model_config.get_total_num_kv_heads(),
      head_dim=model_config.get_head_size(),
      tp_size=tp_size,
  )
  mapped_leaves = tuple(mapped.leaves)
  live_leaves = tuple(runner.state_leaves)
  if len(mapped_leaves) != len(live_leaves):
    raise FunctionalMappingError(
        "mapped/live engine leaf counts differ: "
        f"{len(mapped_leaves)} != {len(live_leaves)}"
    )

  mismatches = []
  total_elements = 0
  normalized_memory_leaves = 0
  memory_kind_pairs: dict[str, int] = {}
  for index, (mapped_leaf, live_leaf) in enumerate(
      zip(mapped_leaves, live_leaves)
  ):
    mapped_value = getattr(mapped_leaf, "value", mapped_leaf)
    live_value = getattr(live_leaf, "value", live_leaf)
    mapped_memory = getattr(
        getattr(mapped_value, "sharding", None), "memory_kind", None
    )
    live_memory = getattr(
        getattr(live_value, "sharding", None), "memory_kind", None
    )
    memory_pair = (
        f"{mapped_memory or 'unspecified'}->"
        f"{live_memory or 'unspecified'}"
    )
    memory_kind_pairs[memory_pair] = memory_kind_pairs.get(memory_pair, 0) + 1
    if mapped_memory != live_memory and "device" in (
        mapped_memory,
        live_memory,
    ):
      normalized_memory_leaves += 1
    exact = _bitwise_arrays_equal(mapped_value, live_value)
    if not exact:
      mismatches.append(index)
    if tuple(mapped_value.shape) == tuple(live_value.shape):
      total_elements += int(mapped_value.size)

  mesh_devices = tuple(int(device.id) for device in runner.mesh.devices.flat)
  return {
      "equal": not mismatches,
      "mapped_leaves": len(mapped_leaves),
      "live_leaves": len(live_leaves),
      "total_elements": total_elements,
      "mismatch_indices": tuple(mismatches),
      "normalized_memory_leaves": normalized_memory_leaves,
      "memory_kind_pairs": memory_kind_pairs,
      "mesh_shape": _weight_attestation_mesh_shape(runner),
      "mesh_device_ids": mesh_devices,
  }


class Qwen3EngineForwardAdapter:
  """Differentiable fixed-M Qwen3 forward backed by the live engine module."""

  is_engine_module = True
  supports_value_and_grad = True

  def __init__(
      self,
      *,
      sampler: Any,
      sampling_kwargs: Mapping[str, Any] | None = None,
  ):
    try:
      runner = sampler._model_runner  # pylint: disable=protected-access
    except (AttributeError, RuntimeError) as exc:
      raise FunctionalMappingError("rollout has no live model runner") from exc
    if os.environ.get("CANON_RPA_VJP2", "") != "1":
      raise FunctionalMappingError("canonical adapter requires CANON_RPA_VJP2=1")
    if os.environ.get("CANON_VJP2_MAX_SEQS", "") != "1":
      raise FunctionalMappingError(
          "canonical adapter executes one sequence per model_fn call; "
          "CANON_VJP2_MAX_SEQS must be explicitly 1"
      )
    bucket = _canonical_logprob_bucket()
    admitted_data_size, admitted_tp_size, local_m, _ = (
        _canonical_topology_contract()
    )
    sampling_kwargs = dict(sampling_kwargs or {})
    top_k = sampling_kwargs.get("top_k", 0)
    top_p = sampling_kwargs.get("top_p", 1.0)
    if top_k not in (None, 0, -1) or top_p not in (None, 1.0):
      raise FunctionalMappingError(
          "canonical adapter currently admits only neutral top-k/top-p; "
          f"got top_k={top_k!r}, top_p={top_p!r}"
      )
    required = (
        "state",
        "model_fn",
        "compute_logits_fn",
        "mesh",
        "kv_caches",
        "layer_name_to_kvcache_index",
        "is_first_rank",
        "is_last_rank",
        "vllm_config",
        "max_num_reqs",
        "block_size",
        "model_config",
    )
    missing = [name for name in required if not hasattr(runner, name)]
    if missing:
      raise FunctionalMappingError(
          f"live runner is missing adapter attributes: {missing}"
      )
    if not runner.kv_caches:
      raise FunctionalMappingError("live runner exposes no paged kv caches")
    cache0 = runner.kv_caches[0]
    if cache0.ndim != 5 or int(cache0.shape[1]) != int(runner.block_size):
      raise FunctionalMappingError(
          f"unexpected KV-cache contract: {cache0.shape} block={runner.block_size}"
      )

    from tpu_inference.layers.common.attention_metadata import (  # pylint: disable=g-import-not-at-top
        AttentionMetadata,
    )
    from vllm.forward_context import set_forward_context  # pylint: disable=g-import-not-at-top
    from tpu_inference.layers.jax.sample.sampling import (  # pylint: disable=g-import-not-at-top
        compute_and_gather_logprobs,
        gather_logprobs,
        sample,
    )
    from tpu_inference.layers.jax.sample.sampling_metadata import (  # pylint: disable=g-import-not-at-top
        TPUSupportedSamplingMetadata,
    )

    self._runner = runner
    self._engine_state_contract = runner.state
    self._key_mappings = getattr(sampler, "to_hf_key_mappings", None) or {}
    self._transpose_keys = getattr(sampler, "to_hf_transpose_keys", None)
    self._hook_fns = getattr(sampler, "to_hf_hook_fns", None)
    self._tp_size = int(
        getattr(sampler, "args", {}).get("tensor_parallel_size", 1)
    )
    if "data" not in runner.mesh.axis_names:
      raise FunctionalMappingError(
          "canonical adapter requires an explicit 'data' mesh axis"
      )
    self._dp_axis = "data"
    self._data_size = int(runner.mesh.shape.get("data", 1))
    mesh_tp_size = int(runner.mesh.shape.get("model", 1))
    if self._data_size != admitted_data_size:
      raise FunctionalMappingError(
          "engine data mesh does not match the admitted topology: "
          f"{self._data_size} != {admitted_data_size}"
      )
    if self._tp_size != mesh_tp_size:
      raise FunctionalMappingError(
          "sampler and engine mesh TP sizes differ: "
          f"{self._tp_size} != {mesh_tp_size}"
      )
    if admitted_tp_size and self._tp_size != admitted_tp_size:
      raise FunctionalMappingError(
          "engine TP contract does not match the admitted topology: "
          f"sampler={self._tp_size} mesh={mesh_tp_size} "
          f"expected={admitted_tp_size}"
      )
    self.implementation_id = (
        f"{type(runner).__module__}.{type(runner).__qualname__}:"
        f"qwen3-canonical-dp{self._data_size}-tp{self._tp_size}-"
        f"m{local_m}-vjp2"
    )
    if bucket % self._data_size:
      raise FunctionalMappingError(
          f"global M {bucket} is not divisible by data size {self._data_size}"
      )
    self._bucket = bucket
    self._sequence_bucket = bucket // self._data_size
    if self._sequence_bucket != local_m:
      raise FunctionalMappingError(
          "per-rank engine bucket changed: "
          f"{self._sequence_bucket} != {local_m}"
      )
    self._max_model_len = int(runner.model_config.max_model_len)
    runner_vocab_size = getattr(runner, "vocab_size", None)
    if runner_vocab_size is None:
      runner_vocab_size = runner.model_config.get_vocab_size()
    self._vocab_size = int(runner_vocab_size)
    self._max_num_reqs = int(runner.max_num_reqs)
    if self._max_num_reqs % self._data_size:
      raise FunctionalMappingError(
          "max_num_reqs must be divisible by the engine data size: "
          f"{self._max_num_reqs} vs {self._data_size}"
      )
    self._local_max_num_reqs = self._max_num_reqs // self._data_size
    self._block_size = int(runner.block_size)
    self._blocks_per_req = (
        int(runner.model_config.max_model_len) + self._block_size - 1
    ) // self._block_size
    self._cache_shape = (
        self._data_size * self._blocks_per_req,
    ) + tuple(cache0.shape[1:])
    self._cache_dtype = cache0.dtype
    self._cache_sharding = cache0.sharding
    valid_mesh_axes = set(runner.mesh.axis_names)
    data_axes = tuple(
        ax
        for ax in ("data", "attn_dp", "attn_dp_expert")
        if ax in valid_mesh_axes
    )
    if len(data_axes) == 1:
      data_spec = data_axes[0]
    elif len(data_axes) > 1:
      data_spec = data_axes
    else:
      data_spec = None
    self._input_sharding = jax.sharding.NamedSharding(
        runner.mesh,
        jax.sharding.PartitionSpec(
            data_spec,
        ),
    )
    self._metadata_cls = getattr(
        runner, "_canonical_attention_metadata_cls", AttentionMetadata
    )
    self._set_forward_context = getattr(
        runner, "_canonical_set_forward_context", set_forward_context
    )
    self._sample = getattr(runner, "_canonical_sample", sample)
    self._sampling_metadata_cls = getattr(
        runner,
        "_canonical_sampling_metadata_cls",
        TPUSupportedSamplingMetadata,
    )
    self._compute_and_gather_logprobs = _install_shared_logprob_pipeline(
        runner,
        stock_compute_and_gather=compute_and_gather_logprobs,
        gather_logprobs=gather_logprobs,
    )
    self._max_logprobs = int(runner.model_config.max_logprobs)
    self._processed_target_logprobs = _make_processed_target_logprob_vjp(
        self._compute_and_gather_logprobs, self._max_logprobs
    )

    g5c_shared_logsoftmax = os.environ.get(
        "CANON_P28_G5C_SHARED_LOGSOFTMAX", "1"
    )
    if g5c_shared_logsoftmax not in ("0", "1"):
      raise FunctionalMappingError(
          "CANON_P28_G5C_SHARED_LOGSOFTMAX must be exactly 0 or 1"
      )
    self._p28_g5c_shared_logsoftmax = g5c_shared_logsoftmax == "1"
    stock_target_logprobs = _make_processed_target_logprob_vjp(
        compute_and_gather_logprobs, self._max_logprobs
    )
    p28_target_logprobs = (
        self._processed_target_logprobs
        if self._p28_g5c_shared_logsoftmax
        else stock_target_logprobs
    )

    def fwd_logprob_rows(logits, target_ids, temperature):
      sampling_metadata = self._sampling_metadata_cls(
          temperature=jnp.full(
              (self._bucket,), temperature, dtype=jnp.float32
          ),
          top_k=jnp.full((self._bucket,), -1, dtype=jnp.int32),
          top_p=jnp.ones((self._bucket,), dtype=jnp.float32),
          do_sampling=True,
          logprobs=True,
      )
      _, processed_logits = self._sample(
          jax.random.PRNGKey(0),
          self._runner.mesh,
          logits.astype(jnp.float32),
          sampling_metadata,
      )
      target_logprobs = p28_target_logprobs(processed_logits, target_ids)
      normalized = jax.nn.log_softmax(processed_logits, axis=-1)
      probabilities = jnp.exp(normalized)
      entropy = -jnp.sum(
          jnp.where(probabilities > 0, probabilities * normalized, 0.0),
          axis=-1,
      )
      return target_logprobs, entropy

    def bwd_logprob_rows(
        logits,
        target_ids,
        temperature,
        dtarget_logprobs,
        dentropy,
    ):
      def primal(values):
        return fwd_logprob_rows(values, target_ids, temperature)

      _, pullback = jax.vjp(primal, logits)
      return pullback((dtarget_logprobs, dentropy))[0]

    self._p28_processed_rows_fn = _xprof_jit(
        fwd_logprob_rows,
        module_name="zt_tr_fwd_logprob",
        scope_name="zt/tr/logprob/fwd",
    )
    self._p28_processed_rows_pullback_fn = _xprof_jit(
        bwd_logprob_rows,
        module_name="zt_tr_bwd_logprob",
        scope_name="zt/tr/logprob/bwd",
    )
    print(
        "[CANON_ADAPTER] processed-logprob custom VJP installed "
        f"m={self._bucket} max_logprobs={self._max_logprobs}",
        flush=True,
    )
    self._static_kv_indices = tuple(
        runner.layer_name_to_kvcache_index.items()
    )

  def attest_exact_live_weights(self, trainer_state) -> dict[str, Any]:
    """Bitwise-compares mapped trainer leaves with the live serving state.

    This diagnostic never substitutes a checksum for equality. Each comparison
    reduces on device to one boolean; model leaves are not copied to the host.
    """
    return attest_exact_live_engine_weights(
        sampler=None,
        trainer_state=trainer_state,
        runner=self._runner,
        engine_state_contract=self._engine_state_contract,
        key_mappings=self._key_mappings,
        transpose_keys=self._transpose_keys,
        key_mapping_hook_fns=self._hook_fns,
        tp_size=self._tp_size,
    )

  def p35_envelope_contract_attestation(self) -> dict[str, Any]:
    """Returns runtime facts needed to admit the adapter C arm."""
    return {
        "data_size": self._data_size,
        "tp_size": self._tp_size,
        "local_m": self._sequence_bucket,
        "global_m": self._bucket,
        "rank_strided_groups": True,
        "fresh_cache_per_group": True,
        "block_tables_rank_local_contiguous": True,
        "mesh_shape": tuple(
            (str(name), int(size))
            for name, size in self._runner.mesh.shape.items()
        ),
        "mesh_device_ids": tuple(
            int(device.id) for device in self._runner.mesh.devices.flat
        ),
    }

  def _p35_run_captured_records(
      self,
      engine_leaves,
      records,
      *,
      replay_label,
      prompt_tokens,
      completion_tokens,
      prompt_mask,
      completion_mask,
      temperature,
      score_mask=None,
  ):
    """Replays captured B tensors through the canonical model entry."""
    records = tuple(records)
    logical_logits_shape = (self._bucket, self._vocab_size)
    logical_logits_bytes = (
        self._bucket * self._vocab_size * np.dtype(np.float32).itemsize
    )
    stage_probe = _p35_replay_stage_probe_enabled()
    if stage_probe and replay_label != "R0_live_first":
      raise FunctionalMappingError(
          "P35.3c stage probe admits only the first live replay arm"
      )
    print(
        "[CANON_P35.3] CAPTURED_REPLAY_BEGIN "
        f"replay={replay_label} records={len(records)} "
        f"logical_logits_shape={logical_logits_shape} "
        f"logical_logits_bytes={logical_logits_bytes} "
        "tail=original_program_serialized",
        flush=True,
    )
    prompts = np.asarray(prompt_tokens)
    completions = np.asarray(completion_tokens)
    prompt_valid = np.asarray(prompt_mask, dtype=np.bool_)
    completion_valid = np.asarray(completion_mask, dtype=np.bool_)
    score_valid = (
        completion_valid
        if score_mask is None
        else np.asarray(score_mask, dtype=np.bool_)
    )
    if (
        prompts.ndim != 2
        or completions.ndim != 2
        or prompts.shape != prompt_valid.shape
        or completions.shape != completion_valid.shape
        or completions.shape != score_valid.shape
        or prompts.shape[0] != self._data_size
        or completions.shape[0] != self._data_size
    ):
      raise FunctionalMappingError(
        "P35.3 captured replay requires one prompt/completion per data rank"
      )
    if np.any(score_valid & ~completion_valid):
      raise FunctionalMappingError(
          "P35.3 score mask includes an invalid completion token"
      )
    prompt_lengths = prompt_valid.sum(axis=1, dtype=np.int64)
    completion_lengths = completion_valid.sum(axis=1, dtype=np.int64)
    sequences = []
    for rank in range(self._data_size):
      sequences.append(
          np.concatenate(
              (
                  prompts[rank][prompt_valid[rank]],
                  completions[rank][completion_valid[rank]],
              )
          )
      )

    sampling_metadata = self._sampling_metadata_cls(
        temperature=self._engine_array(
            jnp.full((self._bucket,), temperature, jnp.float32)
        ),
        top_k=self._engine_array(jnp.full((self._bucket,), -1, jnp.int32)),
        top_p=self._engine_array(jnp.ones((self._bucket,), jnp.float32)),
        do_sampling=True,
        logprobs=True,
    )
    caches = self._fresh_caches()
    completion_width = completions.shape[1]
    replay_logps = jnp.zeros(
        (self._data_size, completion_width), jnp.float32
    )
    raw_targets = jnp.zeros_like(replay_logps)
    processed_targets = jnp.zeros_like(replay_logps)
    log_normalizers = jnp.zeros_like(replay_logps)
    hidden_rows = None
    observed = np.zeros_like(completion_valid)

    for record_index, record in enumerate(records):
      print(
          "[CANON_P35.3] RECORD_BEGIN "
          f"replay={replay_label} record={record_index + 1}/{len(records)} "
          f"logical_logits_shape={logical_logits_shape} "
          f"logical_logits_bytes={logical_logits_bytes}",
          flush=True,
      )
      arrays = record.get("arrays", {})
      required = {
          "input_ids",
          "input_positions",
          "md_input_positions",
          "md_block_tables",
          "md_seq_lens",
          "md_query_start_loc",
          "md_request_distribution",
      }
      if not required.issubset(arrays):
        raise FunctionalMappingError(
            f"P35.3 B record {record_index} lacks captured tensors: "
            f"{sorted(required - set(arrays))}"
        )
      input_ids_host = np.asarray(arrays["input_ids"]).reshape(-1)
      positions_host = np.asarray(arrays["input_positions"]).reshape(-1)
      metadata_positions_host = np.asarray(
          arrays["md_input_positions"]
      ).reshape(-1)
      if (
          input_ids_host.size != self._bucket
          or positions_host.size != self._bucket
          or not np.array_equal(positions_host, metadata_positions_host)
      ):
        raise FunctionalMappingError(
            f"P35.3 B record {record_index} lost the global-M input contract"
        )
      meta = record.get("meta", {})
      padded_num_reqs = int(meta.get("md_padded_num_reqs", 0) or 0)
      if padded_num_reqs <= 0 or padded_num_reqs % self._data_size:
        raise FunctionalMappingError(
            f"P35.3 B record {record_index} has invalid padded requests"
        )
      local_slots = padded_num_reqs // self._data_size
      query_start_host = np.asarray(
          arrays["md_query_start_loc"]
      ).reshape(self._data_size, local_slots + 1)
      seq_lens_host = np.asarray(arrays["md_seq_lens"]).reshape(
          self._data_size, local_slots
      )
      block_tables_host = np.asarray(arrays["md_block_tables"]).reshape(-1)
      if block_tables_host.size % padded_num_reqs:
        raise FunctionalMappingError(
            f"P35.3 B record {record_index} has malformed block tables"
        )
      blocks_per_request = block_tables_host.size // padded_num_reqs
      blocks_by_rank = block_tables_host.reshape(
          self._data_size, local_slots, blocks_per_request
      )
      input_by_rank = input_ids_host.reshape(
          self._data_size, self._sequence_bucket
      )
      positions_by_rank = positions_host.reshape(
          self._data_size, self._sequence_bucket
      )
      target_ids = np.zeros((self._bucket,), np.int32)
      scatter_rows = []
      scatter_slots = []
      flat_predictor_rows = []
      for rank, sequence in enumerate(sequences):
        q_len = int(query_start_host[rank, 1] - query_start_host[rank, 0])
        if q_len < 0 or q_len > self._sequence_bucket:
          raise FunctionalMappingError(
              f"P35.3 B record {record_index} rank {rank} has invalid q_len"
          )
        if q_len:
          active_seq_len = int(seq_lens_host[rank, 0])
          blocks_needed = (
              active_seq_len + self._block_size - 1
          ) // self._block_size
          active_pages = blocks_by_rank[rank, 0, :blocks_needed]
          if (
              active_pages.size
              and (
                  int(active_pages.min()) < 0
                  or int(active_pages.max()) >= self._cache_shape[0]
              )
          ):
            raise FunctionalMappingError(
                "P35.3 captured active page ids do not fit the fresh replay "
                f"cache: record={record_index} rank={rank} "
                f"pages={active_pages.tolist()} cache_blocks={self._cache_shape[0]}"
            )
        for local_row in range(q_len):
          position = int(positions_by_rank[rank, local_row])
          if position < 0 or position >= len(sequence):
            raise FunctionalMappingError(
                f"P35.3 B record {record_index} rank {rank} position out of range"
            )
          if int(input_by_rank[rank, local_row]) != int(sequence[position]):
            raise FunctionalMappingError(
                f"P35.3 B record {record_index} rank {rank} token mismatch"
            )
          target_position = position + 1
          if target_position >= len(sequence):
            continue
          flat_row = rank * self._sequence_bucket + local_row
          target_ids[flat_row] = int(sequence[target_position])
          completion_slot = target_position - int(prompt_lengths[rank])
          if (
              0 <= completion_slot < int(completion_lengths[rank])
              and score_valid[rank, completion_slot]
          ):
            if observed[rank, completion_slot]:
              raise FunctionalMappingError(
                  "P35.3 captured records duplicate one action predictor"
              )
            observed[rank, completion_slot] = True
            scatter_rows.append(rank)
            scatter_slots.append(completion_slot)
            flat_predictor_rows.append(flat_row)

      input_ids = self._engine_array(jnp.asarray(input_ids_host, jnp.int32))
      positions = self._engine_array(jnp.asarray(positions_host, jnp.int32))
      metadata = self._metadata_cls(
          input_positions=self._engine_array(
              jnp.asarray(metadata_positions_host, jnp.int32)
          ),
          block_tables=self._engine_array(
              jnp.asarray(block_tables_host, jnp.int32)
          ),
          seq_lens=self._engine_array(
              jnp.asarray(np.asarray(arrays["md_seq_lens"]), jnp.int32)
          ),
          query_start_loc=self._engine_array(
              jnp.asarray(np.asarray(arrays["md_query_start_loc"]), jnp.int32)
          ),
          request_distribution=self._engine_array(
              jnp.asarray(
                  np.asarray(arrays["md_request_distribution"]), jnp.int32
              )
          ),
      )
      metadata.padded_num_reqs = padded_num_reqs
      with self._set_forward_context(None, self._runner.vllm_config):
        caches, hidden, _, _ = self._runner.model_fn(
            engine_leaves,
            caches,
            input_ids,
            metadata,
            None,
            positions,
            self._static_kv_indices,
            None,
            None,
            bool(self._runner.is_first_rank),
            bool(self._runner.is_last_rank),
        )
      _p35_wait_for_stage(
          (caches, hidden),
          replay_label=replay_label,
          record_index=record_index,
          record_count=len(records),
          stage="model",
      )
      logits = self._runner.compute_logits_fn(
          engine_leaves, hidden, None
      ).astype(jnp.float32)
      _p35_wait_for_stage(
          logits,
          replay_label=replay_label,
          record_index=record_index,
          record_count=len(records),
          stage="logits",
      )
      _, processed_logits = self._sample(
          jax.random.PRNGKey(0), self._runner.mesh, logits, sampling_metadata
      )
      _p35_wait_for_stage(
          processed_logits,
          replay_label=replay_label,
          record_index=record_index,
          record_count=len(records),
          stage="sample",
      )
      target_ids_device = self._engine_array(jnp.asarray(target_ids, jnp.int32))
      all_logps = self._processed_target_logprobs(
          processed_logits, target_ids_device
      )
      _p35_wait_for_stage(
          all_logps,
          replay_label=replay_label,
          record_index=record_index,
          record_count=len(records),
          stage="logprobs",
      )
      raw_target_all = jnp.take_along_axis(
          logits, target_ids_device[:, None], axis=-1
      )[:, 0]
      processed_target_all = jnp.take_along_axis(
          processed_logits, target_ids_device[:, None], axis=-1
      )[:, 0]
      _p35_wait_for_stage(
          (raw_target_all, processed_target_all),
          replay_label=replay_label,
          record_index=record_index,
          record_count=len(records),
          stage="target_gathers",
      )
      if scatter_rows:
        ranks = jnp.asarray(scatter_rows, jnp.int32)
        slots = jnp.asarray(scatter_slots, jnp.int32)
        predictors = jnp.asarray(flat_predictor_rows, jnp.int32)
        selected_logps = all_logps[predictors]
        selected_raw = raw_target_all[predictors]
        selected_processed = processed_target_all[predictors]
        replay_logps = replay_logps.at[ranks, slots].set(selected_logps)
        raw_targets = raw_targets.at[ranks, slots].set(selected_raw)
        processed_targets = processed_targets.at[ranks, slots].set(
            selected_processed
        )
        log_normalizers = log_normalizers.at[ranks, slots].set(
            selected_processed - selected_logps
        )
        if hidden_rows is None:
          hidden_rows = jnp.zeros(
              (self._data_size, completion_width, hidden.shape[-1]),
              hidden.dtype,
          )
        hidden_rows = hidden_rows.at[ranks, slots].set(hidden[predictors])

      _p35_wait_for_stage(
          (
              replay_logps,
              raw_targets,
              processed_targets,
              log_normalizers,
              hidden_rows if hidden_rows is not None else (),
          ),
          replay_label=replay_label,
          record_index=record_index,
          record_count=len(records),
          stage="record_outputs",
      )

      jax.block_until_ready((
          caches,
          replay_logps,
          raw_targets,
          processed_targets,
          log_normalizers,
          hidden_rows if hidden_rows is not None else (),
          all_logps,
          raw_target_all,
          processed_target_all,
      ))
      del (
          all_logps,
          logits,
          processed_logits,
          processed_target_all,
          raw_target_all,
          target_ids_device,
      )
      print(
          "[CANON_P35.3] RECORD_COMPLETE "
          f"replay={replay_label} record={record_index + 1}/{len(records)}",
          flush=True,
      )
      if stage_probe:
        print(
            "[CANON_P35.3C] STAGE_PROBE_COMPLETE "
            f"replay={replay_label} record={record_index + 1}/{len(records)} "
            "last_stage=record_outputs NO_NUMERICAL_VERDICT",
            flush=True,
        )
        raise P35ReplayStageProbeComplete(
            "P35.3c stopped after the first record without a numerical verdict"
        )

    print(
        "[CANON_P35.3] CAPTURED_REPLAY_COMPLETE "
        f"replay={replay_label} records={len(records)}",
        flush=True,
    )

    if not np.array_equal(observed, score_valid):
      missing = np.argwhere(score_valid & ~observed)
      extra = np.argwhere(observed & ~score_valid)
      raise FunctionalMappingError(
          "P35.3 captured records do not cover the selected action mask: "
          f"missing={missing[:8].tolist()} extra={extra[:8].tolist()}"
      )
    if hidden_rows is None:
      raise FunctionalMappingError("P35.3 captured replay produced no actions")
    mask = jnp.asarray(score_valid)
    return replay_logps, {
        "final_hidden": jnp.where(mask[..., None], hidden_rows, 0),
        "raw_targets": jnp.where(mask, raw_targets, 0),
        "processed_targets": jnp.where(mask, processed_targets, 0),
        "implied_log_normalizers": jnp.where(mask, log_normalizers, 0),
        "logps": jnp.where(mask, replay_logps, 0),
    }

  def run_p38_frozenlake_causal_replay(
      self,
      *,
      capsule_path,
      row_index=0,
      temperature=0.7,
  ) -> dict[str, Any]:
    """Runs the default-off DP1 R0/R1 FrozenLake causal discriminator."""
    if os.environ.get("CANON_P38_FROZENLAKE_REPLAY", "") != "1":
      raise FunctionalMappingError(
          "P38 FrozenLake replay requires CANON_P38_FROZENLAKE_REPLAY=1"
      )
    if self._data_size != 1 or self._tp_size != 4:
      raise FunctionalMappingError(
          "P38 FrozenLake replay requires the admitted DP1 x TP4 host: "
          f"data={self._data_size} model={self._tp_size}"
      )
    if self._bucket != 256 or self._sequence_bucket != 256:
      raise FunctionalMappingError(
          "P38 FrozenLake replay requires global/local M=256: "
          f"{self._bucket}/{self._sequence_bucket}"
      )
    capsule = p38_frozenlake_replay.load_verified_capsule(capsule_path)
    if row_index < 0 or row_index >= len(capsule.rows):
      raise FunctionalMappingError(
          f"P38 capsule row index is out of range: {row_index}"
      )
    row = capsule.rows[row_index]
    expected_policy_version = int(
        os.environ.get("CANON_P38_EXPECTED_POLICY_VERSION", "0")
    )
    policy_versions = np.asarray(row.policy_version).reshape(-1)
    if (
        policy_versions.size == 0
        or not np.all(policy_versions == expected_policy_version)
    ):
      raise FunctionalMappingError(
          "P38 capsule policy version does not match the local base checkpoint: "
          f"observed={policy_versions.tolist()} "
          f"expected={expected_policy_version}"
      )

    r0_schedule = p38_frozenlake_replay.build_r0_mask_derived_schedule(
        row, local_m=self._sequence_bucket
    )
    r1_schedule = p38_frozenlake_replay.build_r1_continuous_decode_schedule(
        row, local_m=self._sequence_bucket
    )
    reference_schedule = (
        p38_frozenlake_replay.build_fixed_chunk_reference_schedule(
            row, local_m=self._sequence_bucket
        )
    )

    def records(schedule):
      return p38_frozenlake_replay.build_engine_records(
          schedule,
          max_num_reqs=self._max_num_reqs,
          blocks_per_request=self._blocks_per_req,
          cache_blocks=self._cache_shape[0],
      )

    prompt = jnp.asarray(row.prompt_ids[None, :], jnp.int32)
    completion = jnp.asarray(row.completion_ids[None, :], jnp.int32)
    prompt_valid = jnp.ones_like(prompt, dtype=jnp.bool_)
    completion_valid = jnp.ones_like(completion, dtype=jnp.bool_)
    score_mask = jnp.asarray(row.action_mask[None, :], jnp.bool_)
    engine_leaves = tuple(self._runner.state_leaves)

    def execute(schedule, label):
      return self._p35_run_captured_records(
          engine_leaves,
          records(schedule),
          replay_label=label,
          prompt_tokens=prompt,
          completion_tokens=completion,
          prompt_mask=prompt_valid,
          completion_mask=completion_valid,
          temperature=temperature,
          score_mask=score_mask,
      )

    r0_first = execute(r0_schedule, "P38_R0_first")
    r0_repeat = execute(r0_schedule, "P38_R0_repeat")
    r1_first = execute(r1_schedule, "P38_R1_first")
    r1_repeat = execute(r1_schedule, "P38_R1_repeat")
    reference_first = execute(reference_schedule, "P38_REF_first")
    reference_repeat = execute(reference_schedule, "P38_REF_repeat")
    jax.block_until_ready((
        r0_first,
        r0_repeat,
        r1_first,
        r1_repeat,
        reference_first,
        reference_repeat,
    ))

    action_indices = np.flatnonzero(row.action_mask)

    def selected(value):
      array = np.asarray(jax.device_get(value))
      if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
      if array.shape[0] != row.completion_length:
        raise FunctionalMappingError(
            "P38 replay stage is not completion aligned: "
            f"{array.shape}/{row.completion_length}"
        )
      return np.ascontiguousarray(array[action_indices])

    stage_names = (
        "raw_targets",
        "processed_targets",
        "implied_log_normalizers",
        "logps",
    )
    arm_values = {
        "R0": {name: selected(r0_first[1][name]) for name in stage_names},
        "R1": {name: selected(r1_first[1][name]) for name in stage_names},
        "REF": {
            name: selected(reference_first[1][name]) for name in stage_names
        },
    }
    repeat_values = {
        "R0": {name: selected(r0_repeat[1][name]) for name in stage_names},
        "R1": {name: selected(r1_repeat[1][name]) for name in stage_names},
        "REF": {
            name: selected(reference_repeat[1][name]) for name in stage_names
        },
    }
    repeat_comparisons = {
        arm: {
            name: _host_difference_summary(values[name], repeat_values[arm][name])
            for name in stage_names
        }
        for arm, values in arm_values.items()
    }
    if not all(
        comparison["exact"]
        for arm in repeat_comparisons.values()
        for comparison in arm.values()
    ):
      raise FunctionalMappingError("P38 replay is not bitwise deterministic")

    negative = arm_values["R0"]["logps"].copy()
    negative.view(np.uint8).reshape(-1)[0] ^= np.uint8(1)
    negative_control = _host_difference_summary(
        arm_values["R0"]["logps"], negative
    )
    if negative_control["exact"]:
      raise FunctionalMappingError("P38 one-bit negative control was not detected")

    comparisons = {}
    for left, right in (("R0", "R1"), ("R0", "REF"), ("R1", "REF")):
      comparisons[f"{left}_vs_{right}"] = {
          name: _host_difference_summary(
              arm_values[left][name], arm_values[right][name]
          )
          for name in stage_names
      }
    r0_reference_exact = comparisons["R0_vs_REF"]["logps"]["exact"]
    r1_reference_exact = comparisons["R1_vs_REF"]["logps"]["exact"]
    if not r0_reference_exact and r1_reference_exact:
      classification = "MULTITURN_SCHEDULE_CARRIER_CANDIDATE"
    elif r0_reference_exact:
      classification = "LOCAL_CARRIER_NOT_REPRODUCED"
    else:
      classification = "LOCAL_CARRIER_NOT_ISOLATED"

    captured = {
        "S_decode_vs_S_prefill": _host_difference_summary(
            row.s_decode[action_indices], row.s_prefill[action_indices]
        ),
        "S_prefill_vs_T_old": _host_difference_summary(
            row.s_prefill[action_indices], row.t_old[action_indices]
        ),
    }
    captured_values = {
        "S_decode": np.asarray(row.s_decode[action_indices]),
        "S_prefill": np.asarray(row.s_prefill[action_indices]),
        "T_old": np.asarray(row.t_old[action_indices]),
    }
    replay_vs_captured = {
        arm: {
            captured_name: _host_difference_summary(
                values["logps"], captured_value
            )
            for captured_name, captured_value in captured_values.items()
        }
        for arm, values in arm_values.items()
    }
    prerequisites = (
        captured["S_decode_vs_S_prefill"]["exact"] is False
        and captured["S_prefill_vs_T_old"]["exact"] is True
        and replay_vs_captured["REF"]["S_prefill"]["exact"] is True
    )
    if not prerequisites:
      e0_lite_classification = "E0_LITE_PREREQUISITE_FAILED"
    elif replay_vs_captured["R0"]["S_decode"]["exact"] is True:
      e0_lite_classification = "E0_LITE_REPRODUCED"
    else:
      e0_lite_classification = "E0_LITE_ENVELOPE_NOT_REPRODUCED"
    report = {
        "schema": "p38-frozenlake-causal-replay-v1",
        "measurement_status": "COMPLETE",
        "classification": classification,
        "claim_ceiling": (
            "R0 is derived from action and validity masks; the capsule does "
            "not contain the original serving scheduler call metadata."
        ),
        "capsule": {
            "path": str(capsule.path),
            "sha256": capsule.sha256,
            "source_row": row.source_row,
            "row_index": row_index,
            "policy_version": expected_policy_version,
        },
        "geometry": {
            "data": self._data_size,
            "model": self._tp_size,
            "local_m": self._sequence_bucket,
            "global_m": self._bucket,
            "prompt_length": row.prompt_length,
            "completion_length": row.completion_length,
            "action_tokens": int(action_indices.size),
            "prefix_cache": False,
            "runtime_kv_cache": True,
        },
        "schedules": [
            r0_schedule.as_dict(),
            r1_schedule.as_dict(),
            reference_schedule.as_dict(),
        ],
        "captured_boundaries": captured,
        "replay_vs_captured": replay_vs_captured,
        "e0_lite_classification": e0_lite_classification,
        "e0_lite_claim_ceiling": (
            "R0 remains mask-derived and does not contain the exact live "
            "scheduler/cache state; only E0_LITE_REPRODUCED may promote the "
            "row to strict E0 construction."
        ),
        "comparisons": comparisons,
        "repeat_comparisons": repeat_comparisons,
        "negative_control": negative_control,
        "no_backward": True,
        "no_optimizer": True,
    }
    print(
      "[CANON_P38_REPLAY] "
      f"classification={classification} row={row.source_row} "
      f"e0_lite={e0_lite_classification} "
      f"R0_vs_REF={comparisons['R0_vs_REF']['logps']['differing_elements']} "
        f"R1_vs_REF={comparisons['R1_vs_REF']['logps']['differing_elements']}",
        flush=True,
    )
    return report

  def p35_exact_input_replay(
      self,
      trainer_state,
      records,
      *,
      full_prompt_tokens,
      full_completion_tokens,
      full_prompt_mask,
      full_completion_mask,
      selected_row_indices,
      pad_id,
      eos_id,
      temperature,
  ) -> dict[str, Any]:
    """Separates captured inputs, weight placement, and adapter envelope."""
    if os.environ.get("CANON_P35_EXACT_REPLAY", "") != "1":
      raise FunctionalMappingError(
          "P35.3 exact replay requires CANON_P35_EXACT_REPLAY=1"
      )
    model_config = self._runner.model_config
    mapped = map_trainer_state_to_engine_leaves(
        trainer_state=trainer_state,
        engine_state_contract=self._engine_state_contract,
        key_mappings=self._key_mappings,
        transpose_keys=self._transpose_keys,
        key_mapping_hook_fns=self._hook_fns,
        num_kv_heads=model_config.get_total_num_kv_heads(),
        head_dim=model_config.get_head_size(),
        tp_size=self._tp_size,
    )

    prompts = jnp.asarray(full_prompt_tokens)
    completions = jnp.asarray(full_completion_tokens)
    prompt_mask = jnp.asarray(full_prompt_mask, dtype=jnp.bool_)
    completion_mask = jnp.asarray(full_completion_mask, dtype=jnp.bool_)
    rows = np.asarray(selected_row_indices, dtype=np.int64)
    if (
        prompts.ndim != 2
        or completions.ndim != 2
        or prompts.shape != prompt_mask.shape
        or completions.shape != completion_mask.shape
        or prompts.shape[0] != completions.shape[0]
        or rows.shape != (self._data_size,)
        or np.unique(rows).size != rows.size
        or np.any(rows < 0)
        or np.any(rows >= prompts.shape[0])
    ):
      raise FunctionalMappingError(
          "P35.3 full-batch replay inputs or selected rows are malformed"
      )
    selected_prompts = prompts[rows]
    selected_completions = completions[rows]
    selected_prompt_mask = prompt_mask[rows]
    selected_completion_mask = completion_mask[rows]

    def execute_captured(leaves, replay_label):
      return self._p35_run_captured_records(
          leaves,
          records,
          replay_label=replay_label,
          prompt_tokens=selected_prompts,
          completion_tokens=selected_completions,
          prompt_mask=selected_prompt_mask,
          completion_mask=selected_completion_mask,
          temperature=temperature,
      )

    def execute_adapter_direct():
      return self._sequence_group(
          tuple(mapped.leaves),
          selected_prompts,
          selected_completions,
          selected_prompt_mask,
          selected_completion_mask,
          pad_id,
          temperature,
      )[0]

    live_first = execute_captured(
        tuple(self._runner.state_leaves), "R0_live_first"
    )
    live_second = execute_captured(
        tuple(self._runner.state_leaves), "R0_live_repeat"
    )
    mapped_first = execute_captured(tuple(mapped.leaves), "R1_mapped_first")
    mapped_second = execute_captured(
        tuple(mapped.leaves), "R1_mapped_repeat"
    )
    adapter_direct_first = execute_adapter_direct()
    adapter_direct_second = execute_adapter_direct()
    adapter_envelope = self.compute_per_token_logps(
        graphdef=None,
        state=trainer_state,
        prompt_tokens=prompts,
        completion_tokens=completions,
        pad_id=pad_id,
        eos_id=eos_id,
        stop_gradient=True,
        temperature=temperature,
        prompt_mask=prompt_mask,
        completion_mask=completion_mask,
    )[rows]
    stages = tuple(live_first[1])

    def memory_kind_counts(leaves):
      counts: dict[str, int] = {}
      for leaf in leaves:
        kind = str(
            getattr(getattr(leaf, "sharding", None), "memory_kind", None)
        )
        counts[kind] = counts.get(kind, 0) + 1
      return counts

    return {
        "r0_live_logps": live_first[0],
        "r1_mapped_logps": mapped_first[0],
        "r2_adapter_direct_logps": adapter_direct_first,
        "r3_adapter_envelope_logps": adapter_envelope,
        "stage_comparisons": {
            "R0_live_vs_R1_mapped": {
                stage: _bitwise_difference_summary(
                    live_first[1][stage], mapped_first[1][stage]
                )
                for stage in stages
            }
        },
        "repeat_comparisons": {
            "R0_live_repeat": {
                stage: _bitwise_difference_summary(
                    live_first[1][stage], live_second[1][stage]
                )
                for stage in stages
            },
            "R1_mapped_repeat": {
                stage: _bitwise_difference_summary(
                    mapped_first[1][stage], mapped_second[1][stage]
                )
                for stage in stages
            },
            "R2_adapter_direct_repeat": {
                "logps": _bitwise_difference_summary(
                    adapter_direct_first, adapter_direct_second
                )
            },
        },
        "metadata": {
            "records": len(records),
            "stages": stages,
            "live_memory_kind_counts": memory_kind_counts(
                self._runner.state_leaves
            ),
            "mapped_memory_kind_counts": memory_kind_counts(mapped.leaves),
        },
    }

  def _engine_array(self, value):
    return _safe_sharding_constraint(value, self._input_sharding)

  def _fresh_caches(self):
    return [
        _safe_sharding_constraint(
            jnp.zeros(self._cache_shape, self._cache_dtype),
            self._cache_sharding,
        )
        for _ in self._runner.kv_caches
    ]

  def _group_batch_rows(self, value):
    """Groups a global batch into one independent row per data rank."""
    if value.shape[0] % self._data_size:
      raise FunctionalMappingError(
          "global batch must be divisible by the engine data size: "
          f"{value.shape[0]} vs {self._data_size}"
      )
    local_batch = value.shape[0] // self._data_size
    reshaped = value.reshape(
        (self._data_size, local_batch) + value.shape[1:]
    )
    return jnp.swapaxes(reshaped, 0, 1)

  def _ungroup_batch_rows(self, value):
    """Restores data-rank-major global batch row order."""
    if value.ndim < 2 or value.shape[1] != self._data_size:
      raise FunctionalMappingError(
          "grouped batch does not match the engine data size: "
          f"{value.shape} vs {self._data_size}"
      )
    transposed = jnp.swapaxes(value, 0, 1)
    return transposed.reshape(
        (transposed.shape[0] * transposed.shape[1],) + value.shape[2:]
    )

  def map_engine_cotangents_to_trainer_state(
      self, trainer_state, engine_cotangents
  ):
    """Applies only the pure trainer->engine mapping adjoint on the host."""
    _P28SegmentedEngineForward._reject_outer_transform(  # pylint: disable=protected-access
        trainer_state, engine_cotangents
    )
    if os.environ.get("CANON_P28_SEGMENTED_TRAIN", "") != "1":
      raise FunctionalMappingError(
          "P28 mapping adjoint requires CANON_P28_SEGMENTED_TRAIN=1"
      )
    model_config = self._runner.model_config

    def mapping(state):
      return map_trainer_state_to_engine_leaves(
          trainer_state=state,
          engine_state_contract=self._engine_state_contract,
          key_mappings=self._key_mappings,
          transpose_keys=self._transpose_keys,
          key_mapping_hook_fns=self._hook_fns,
          num_kv_heads=model_config.get_total_num_kv_heads(),
          head_dim=model_config.get_head_size(),
          tp_size=self._tp_size,
      ).leaves

    mapped, pullback = jax.vjp(mapping, trainer_state)
    engine_cotangents = tuple(engine_cotangents)
    if len(mapped) != len(engine_cotangents):
      raise FunctionalMappingError(
          "P28 mapping-adjoint cotangent count changed: "
          f"{len(engine_cotangents)} != {len(mapped)}"
      )
    for index, (value, cotangent) in enumerate(
        zip(mapped, engine_cotangents, strict=True)
    ):
      if value.shape != cotangent.shape:
        raise FunctionalMappingError(
            "P28 mapping-adjoint cotangent shape changed at leaf "
            f"{index}: {cotangent.shape} != {value.shape}"
        )
    return pullback(engine_cotangents)[0]

  def _batched_report_adjoint(self, trainer_state, engine_cotangents):
    """One-dispatch mapping adjoint + f32 cast (data movement + cast only).

    The eager path re-traces jax.vjp over the whole trainer->engine mapping
    per trajectory; this compiles the identical composition once. The
    per-call shape contract stays host-side and fail-closed.
    """
    _P28SegmentedEngineForward._reject_outer_transform(  # pylint: disable=protected-access
        trainer_state, engine_cotangents
    )
    if os.environ.get("CANON_P28_SEGMENTED_TRAIN", "") != "1":
      raise FunctionalMappingError(
          "P28 mapping adjoint requires CANON_P28_SEGMENTED_TRAIN=1"
      )
    engine_cotangents = tuple(engine_cotangents)
    if getattr(self, "_p50_adjoint_fn", None) is None:
      model_config = self._runner.model_config

      def mapping(state):
        return map_trainer_state_to_engine_leaves(
            trainer_state=state,
            engine_state_contract=self._engine_state_contract,
            key_mappings=self._key_mappings,
            transpose_keys=self._transpose_keys,
            key_mapping_hook_fns=self._hook_fns,
            num_kv_heads=model_config.get_total_num_kv_heads(),
            head_dim=model_config.get_head_size(),
            tp_size=self._tp_size,
        ).leaves

      def bwd_adjoint_link(state, cotangents):
        _, pullback = jax.vjp(mapping, state)
        gradient = pullback(tuple(cotangents))[0]
        return jax.tree.map(
            lambda value: value.astype(jnp.float32), gradient
        )

      self._p50_adjoint_shapes = tuple(
          value.shape for value in jax.eval_shape(mapping, trainer_state)
      )
      self._p50_adjoint_fn = _xprof_jit(
          bwd_adjoint_link,
          module_name="zt_tr_bwd_adjoint",
          scope_name="zt/tr/report/adjoint",
      )
    expected = self._p50_adjoint_shapes
    if len(expected) != len(engine_cotangents):
      raise FunctionalMappingError(
          "P28 mapping-adjoint cotangent count changed: "
          f"{len(engine_cotangents)} != {len(expected)}"
      )
    for index, (shape, cotangent) in enumerate(
        zip(expected, engine_cotangents, strict=True)
    ):
      if shape != cotangent.shape:
        raise FunctionalMappingError(
            "P28 mapping-adjoint cotangent shape changed at leaf "
            f"{index}: {cotangent.shape} != {shape}"
        )
    return self._p50_adjoint_fn(trainer_state, engine_cotangents)

  def _p59_rank_parallel_report_adjoint(
      self, trainer_state, staged_engine_cotangents
  ):
    """Maps every DP-local engine-gradient row to trainer state in parallel."""
    _P28SegmentedEngineForward._reject_outer_transform(  # pylint: disable=protected-access
        trainer_state, staged_engine_cotangents
    )
    staged_engine_cotangents = tuple(staged_engine_cotangents)
    mesh, data_axis = _p59_replicated_data_mesh(
        (trainer_state, staged_engine_cotangents),
        "P59 report adjoint",
    )
    cached_axis = getattr(self, "_p59_report_dp_axis", None)
    if cached_axis is not None and cached_axis != data_axis:
      raise FunctionalMappingError(
          "P59 report adjoint data axis changed: "
          f"{cached_axis!r} != {data_axis!r}"
      )
    if getattr(self, "_p59_report_adjoint_fn", None) is None:
      model_config = self._runner.model_config

      def mapping(state):
        return map_trainer_state_to_engine_leaves(
            trainer_state=state,
            engine_state_contract=self._engine_state_contract,
            key_mappings=self._key_mappings,
            transpose_keys=self._transpose_keys,
            key_mapping_hook_fns=self._hook_fns,
            num_kv_heads=model_config.get_total_num_kv_heads(),
            head_dim=model_config.get_head_size(),
            tp_size=self._tp_size,
        ).leaves

      expected = tuple(
          value.shape for value in jax.eval_shape(mapping, trainer_state)
      )
      if len(expected) != len(staged_engine_cotangents):
        raise FunctionalMappingError(
            "P59 mapping-adjoint cotangent count changed: "
            f"{len(staged_engine_cotangents)} != {len(expected)}"
        )

      def local_adjoint(state, staged_cotangents):
        cotangents = jax.tree.map(
            lambda value: jnp.squeeze(value, axis=0), staged_cotangents
        )
        _, pullback = jax.vjp(mapping, state)
        gradient = pullback(tuple(cotangents))[0]
        return jax.tree.map(
            lambda value: jnp.expand_dims(
                value.astype(jnp.float32), axis=0
            ),
            gradient,
        )

      mapped = jax.shard_map(
          local_adjoint,
          mesh=mesh,
          in_specs=(
              _manual_axis_specs(trainer_state, data_axis),
              _manual_axis_specs(
                  staged_engine_cotangents, data_axis
              ),
          ),
          out_specs=_rank_staged_specs(trainer_state, data_axis),
          axis_names=_p59_manual_rank_axes(
              mesh, data_axis, "P59 report adjoint"
          ),
          check_vma=False,
      )
      self._p59_report_adjoint_shapes = expected
      self._p59_report_dp_axis = data_axis
      self._p59_report_adjoint_fn = _xprof_jit(
          mapped,
          module_name="zt_tr_dp_parallel_bwd_adjoint",
          scope_name="zt/tr/dp_parallel/report/adjoint",
          donate_argnums=(1,),
      )
    expected = self._p59_report_adjoint_shapes
    for index, (shape, cotangent) in enumerate(
        zip(expected, staged_engine_cotangents, strict=True)
    ):
      staged_shape = (self._data_size,) + tuple(shape)
      if cotangent.shape != staged_shape:
        raise FunctionalMappingError(
            "P59 mapping-adjoint staged cotangent shape changed at leaf "
            f"{index}: {cotangent.shape} != {staged_shape}"
        )
    staged_trainer_gradient = self._p59_report_adjoint_fn(
        trainer_state, staged_engine_cotangents
    )
    return _p59_restore_physically_equal_staged_specs(
        trainer_state, staged_trainer_gradient, data_axis
    )

  def _p59_reducer_template(self, trainer_state, staged_gradient):
    """Builds a buffer-free reducer template and checks staged shardings."""
    if jax.tree.structure(trainer_state) != jax.tree.structure(staged_gradient):
      raise FunctionalMappingError(
          "P59 staged trainer-gradient tree differs from trainer state"
      )
    _, data_axis = _p59_replicated_data_mesh(
        (trainer_state, staged_gradient), "P59 reducer template"
    )
    if data_axis != getattr(self, "_p59_report_dp_axis", None):
      raise FunctionalMappingError(
          "P59 reducer and report-adjoint data axes differ"
      )

    def template(state_value, staged_value):
      state_sharding = getattr(state_value, "sharding", None)
      staged_sharding = getattr(staged_value, "sharding", None)
      if not isinstance(state_sharding, jax.sharding.NamedSharding):
        raise FunctionalMappingError(
            "P59 trainer state requires NamedSharding leaves"
        )
      expected_shape = (self._data_size,) + tuple(state_value.shape)
      expected_spec = jax.sharding.PartitionSpec(
          data_axis, *tuple(state_sharding.spec)
      )
      expected_sharding = jax.sharding.NamedSharding(
          state_sharding.mesh, expected_spec
      )
      if (
          staged_value.shape != expected_shape
          or staged_value.dtype != jnp.float32
          or staged_sharding != expected_sharding
      ):
        raise FunctionalMappingError(
            "P59 staged trainer-gradient placement changed: "
            f"{staged_value.shape}/{staged_value.dtype}/{staged_sharding} != "
            f"{expected_shape}/{jnp.float32}/{expected_sharding}"
        )
      return jax.ShapeDtypeStruct(
          state_value.shape,
          jnp.float32,
          sharding=jax.sharding.NamedSharding(
              state_sharding.mesh, state_sharding.spec
          ),
      )

    return jax.tree.map(template, trainer_state, staged_gradient)

  def _batched_report_add(self, total_tree, delta_tree):
    """One-dispatch elementwise tree add (no reduction freedom)."""
    if getattr(self, "_p50_acc_fn", None) is None:
      # Named so profiles show bwd_report_acc instead of jit__lambda
      # (1.5ms per call at the certified geometry -- worth a name).
      def bwd_report_acc(total, delta):
        return jax.tree.map(lambda a, b: a + b, total, delta)

      self._p50_acc_fn = _xprof_jit(
          bwd_report_acc,
          module_name="zt_tr_bwd_report_acc",
          scope_name="zt/tr/report/accumulate",
      )
    return self._p50_acc_fn(total_tree, delta_tree)

  def _batched_report_evidence(
      self, engine_groups, trainer_leaves, engine_gradients, cache_leaves
  ):
    """One-dispatch evidence predicates (exact bool/int reductions)."""
    if getattr(self, "_p50_evidence_fn", None) is None:
      group_index_items = tuple(
          (label, tuple(indices))
          for label, indices in engine_groups.items()
      )

      def stacked_finite(leaves):
        if not leaves:
          return jnp.ones((0,), jnp.bool_)
        return jnp.stack([jnp.all(jnp.isfinite(value)) for value in leaves])

      def stacked_nonzero(leaves):
        if not leaves:
          return jnp.zeros((0,), jnp.int32)
        return jnp.stack([jnp.count_nonzero(value) for value in leaves])

      def evidence(trainer_lv, engine_lv, cache_lv):
        groups = {
            label: (
                stacked_finite(tuple(engine_lv[i] for i in indices)),
                stacked_nonzero(tuple(engine_lv[i] for i in indices)),
            )
            for label, indices in group_index_items
        }
        return {
            "trainer": (
                stacked_finite(trainer_lv), stacked_nonzero(trainer_lv)
            ),
            "groups": groups,
            "cache": (stacked_finite(cache_lv), stacked_nonzero(cache_lv)),
        }

      self._p50_evidence_fn = _xprof_jit(
          evidence,
          module_name="zt_tr_evidence",
          scope_name="zt/tr/report/evidence",
      )
    return self._p50_evidence_fn(
        tuple(trainer_leaves), tuple(engine_gradients), tuple(cache_leaves)
    )

  def _p28_sequence_spec(
      self,
      prompt,
      completion,
      prompt_valid,
      completion_valid,
      temperature,
  ):
    """Builds one host-visible fixed-M sequence schedule."""
    _P28SegmentedEngineForward._reject_outer_transform(  # pylint: disable=protected-access
        prompt, completion, prompt_valid, completion_valid
    )
    prompt = jnp.asarray(prompt)
    completion = jnp.asarray(completion)
    prompt_valid = jnp.asarray(prompt_valid, dtype=jnp.bool_)
    completion_valid = jnp.asarray(completion_valid, dtype=jnp.bool_)
    if prompt.ndim != 1 or completion.ndim != 1:
      raise FunctionalMappingError("P28 G5c sequence rows must be rank 1")
    if prompt.shape != prompt_valid.shape:
      raise FunctionalMappingError("P28 G5c prompt mask shape changed")
    if completion.shape != completion_valid.shape:
      raise FunctionalMappingError("P28 G5c completion mask shape changed")
    full = jnp.concatenate((prompt, completion), axis=0)
    valid = jnp.concatenate((prompt_valid, completion_valid), axis=0)
    n_real = int(np.asarray(jax.device_get(jnp.sum(valid, dtype=jnp.int32))))
    prompt_length = int(
        np.asarray(jax.device_get(jnp.sum(prompt_valid, dtype=jnp.int32)))
    )
    completion_length = int(
        np.asarray(jax.device_get(jnp.sum(completion_valid, dtype=jnp.int32)))
    )
    if n_real < 2 or prompt_length < 1 or completion_length < 1:
      raise FunctionalMappingError(
          "P28 G5c requires nonempty prompt/completion and at least two tokens"
      )
    if n_real > self._max_model_len:
      raise FunctionalMappingError(
          f"P28 G5c sequence length {n_real} exceeds {self._max_model_len}"
      )
    num_chunks = (n_real + self._bucket - 1) // self._bucket
    padded_width = num_chunks * self._bucket
    order = jnp.nonzero(valid, size=padded_width, fill_value=0)[0]
    packed_active = jnp.arange(padded_width, dtype=jnp.int32) < n_real
    packed_ids = jnp.where(
        packed_active, full[order], jnp.asarray(0, full.dtype)
    )
    next_ids = jnp.concatenate(
        (packed_ids[1:], jnp.zeros((1,), packed_ids.dtype)), axis=0
    )
    completion_ordinal = (
        jnp.cumsum(completion_valid, dtype=jnp.int32) - 1
    )
    source_rows = jnp.clip(
        prompt_length + completion_ordinal - 1, 0, padded_width - 1
    )
    return {
        "packed_ids": packed_ids,
        "next_ids": next_ids,
        "source_rows": source_rows,
        "completion_valid": completion_valid,
        "n_real": n_real,
        "num_chunks": num_chunks,
        "temperature": jnp.asarray(temperature, jnp.float32),
    }

  def _p28_chunk_inputs(self, spec, chunk_index):
    """Constructs one real engine metadata/input tuple at fixed M."""
    chunk_index = int(chunk_index)
    if os.environ.get("CANON_P28_BATCHED_REVERSE", "") in (
        "1", "verify", "scan_fwd"
    ):
      cache = spec.get("_p52_chunk_inputs")
      if cache is None:
        cache = {}
        spec["_p52_chunk_inputs"] = cache
      cached = cache.get(chunk_index)
      if cached is not None:
        return cached
    chunk_start = chunk_index * self._bucket
    q_len = min(self._bucket, spec["n_real"] - chunk_start)
    if q_len <= 0:
      raise FunctionalMappingError("P28 G5c attempted an empty chunk")
    if os.environ.get("CANON_FUSED_TREE_OPS", "") == "1":
      (
          ids,
          targets,
          positions,
          block_tables_flat,
          seq_lens,
          query_start,
          request_distribution,
      ) = _fused_p28_chunk_inputs(
          jnp.asarray(spec["n_real"], jnp.int32),
          spec["packed_ids"],
          spec["next_ids"],
          chunk_start,
          self._bucket,
          int(self._max_num_reqs),
          int(self._blocks_per_req),
      )
      metadata = self._metadata_cls(
          input_positions=self._engine_array(positions),
          block_tables=self._engine_array(block_tables_flat),
          seq_lens=self._engine_array(seq_lens),
          query_start_loc=self._engine_array(query_start),
          request_distribution=self._engine_array(request_distribution),
      )
      metadata.padded_num_reqs = self._max_num_reqs
      result = (
          self._engine_array(ids),
          self._engine_array(targets),
          metadata,
      )
      if os.environ.get("CANON_P28_BATCHED_REVERSE", "") in (
          "1", "verify", "scan_fwd"
      ):
        spec["_p52_chunk_inputs"][chunk_index] = result
      return result
    kv_len = min(spec["n_real"], chunk_start + self._bucket)
    rows = jnp.arange(self._bucket, dtype=jnp.int32)
    positions = jnp.where(rows < q_len, chunk_start + rows, 0)
    query_start = jnp.zeros((self._max_num_reqs + 1,), jnp.int32)
    query_start = query_start.at[1:].set(q_len)
    seq_lens = jnp.zeros((self._max_num_reqs,), jnp.int32)
    seq_lens = seq_lens.at[0].set(kv_len)
    block_tables = jnp.zeros(
        (self._max_num_reqs, self._blocks_per_req), jnp.int32
    ).at[0].set(jnp.arange(self._blocks_per_req, dtype=jnp.int32))
    metadata = self._metadata_cls(
        input_positions=self._engine_array(positions),
        block_tables=self._engine_array(block_tables.reshape(-1)),
        seq_lens=self._engine_array(seq_lens),
        query_start_loc=self._engine_array(query_start),
        request_distribution=self._engine_array(
            jnp.asarray((0, 0, 1), jnp.int32)
        ),
    )
    metadata.padded_num_reqs = self._max_num_reqs
    result = (
        self._engine_array(
            spec["packed_ids"][chunk_start : chunk_start + self._bucket]
        ),
        self._engine_array(
            spec["next_ids"][chunk_start : chunk_start + self._bucket]
        ),
        metadata,
    )
    if os.environ.get("CANON_P28_BATCHED_REVERSE", "") in (
        "1", "verify", "scan_fwd"
    ):
      spec["_p52_chunk_inputs"][chunk_index] = result
    return result

  @staticmethod
  def _p50_scan_verify(loop_hidden, scan_hidden, loop_caches, scan_caches,
                       chunk_index):
    """Bitwise gate between the per-layer loop and the scanned program."""
    hidden_same = bool(np.asarray(jnp.array_equal(loop_hidden, scan_hidden)))
    cache_same = all(
        bool(np.asarray(jnp.array_equal(a, b)))
        for a, b in zip(
            jax.tree.leaves(loop_caches), jax.tree.leaves(scan_caches)
        )
    )
    if not (hidden_same and cache_same):
      raise FunctionalMappingError(
          "P50 layer-scan verify mismatch at chunk "
          f"{chunk_index}: hidden_same={hidden_same} cache_same={cache_same}"
      )

  @staticmethod
  def _p50_rev_verify(label, chunk_index, loop_tree, scan_tree):
    """Bitwise gate between the loop reverse and the scanned reverse.

    Returns None when every leaf matches; otherwise a diagnostic string
    naming each mismatching leaf (index, shape/dtype, differing-element
    count, max abs difference, and which stacked rows differ when the
    leading axis looks like the layer axis).
    """
    loop_leaves = jax.tree.leaves(loop_tree)
    scan_leaves = jax.tree.leaves(scan_tree)
    if len(loop_leaves) != len(scan_leaves):
      raise FunctionalMappingError(
          "P50 reverse-scan verify leaf-count mismatch at chunk "
          f"{chunk_index}: {label} {len(loop_leaves)} != {len(scan_leaves)}"
      )
    details = []
    for leaf_index, (a, b) in enumerate(zip(loop_leaves, scan_leaves)):
      if bool(np.asarray(jnp.array_equal(a, b))):
        continue
      differing = int(np.asarray(jnp.sum(a != b)))
      max_abs = float(
          np.asarray(
              jnp.max(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32)))
          )
      )
      row_note = ""
      if a.ndim >= 1 and a.shape[0] <= 64:
        row_equal = np.asarray(
            jnp.all(
                (a != b).reshape(a.shape[0], -1) == False,  # noqa: E712
                axis=1,
            )
        )
        bad_rows = [int(i) for i in np.nonzero(~row_equal)[0]]
        row_note = f" rows={bad_rows}"
      details.append(
          f"leaf[{leaf_index}] shape={tuple(a.shape)} dtype={a.dtype} "
          f"n_diff={differing}/{a.size} max_abs_diff={max_abs:.3e}{row_note}"
      )
    if not details:
      return None
    detail_text = "; ".join(details)
    print(
        f"[P50DIAG] chunk={chunk_index} {label}: {detail_text}",
        flush=True,
    )
    return f"{label}: {detail_text}"

  def _p28_forward_sequence(
      self, segmented, engine_leaves, spec, *, keep_cache_inputs
  ):
    """Runs one sequence and optionally retains only inter-chunk cache states."""
    caches = tuple(self._fresh_caches())
    cache_inputs = []
    chunk_logps = []
    chunk_entropies = []
    counts = {"embed_forward": 0, "layer_forward": 0,
              "norm_forward": 0, "head_forward": 0,
              "processed_forward": 0}
    with self._set_forward_context(None, self._runner.vllm_config):
      for chunk_index in range(spec["num_chunks"]):
        input_ids, target_ids, metadata = self._p28_chunk_inputs(
            spec, chunk_index
        )
        if keep_cache_inputs:
          cache_inputs.append(caches)
        hidden = segmented.run_embed_forward(
            input_ids, state_leaves=engine_leaves
        )
        counts["embed_forward"] += 1
        layer_scan_mode = segmented.layer_scan_mode()
        # scan_fwd (P56.4.11): the forward chunk also rides the
        # byte-preserving layer scan (one program instead of one per
        # layer); the P50 verify machinery certified this exact branch.
        scan_fwd_forward = (
            os.environ.get("CANON_P28_BATCHED_REVERSE", "") == "scan_fwd"
        )
        scan_caches = scan_hidden = None
        if layer_scan_mode or scan_fwd_forward:
          scan_caches, scan_hidden = segmented.run_layers_scan(
              engine_leaves, caches, hidden, metadata
          )
        if layer_scan_mode == "1" or scan_fwd_forward:
          caches = scan_caches
          hidden = scan_hidden
          counts["layer_forward"] += len(caches)
        else:
          next_caches = []
          for layer_index, cache in enumerate(caches):
            cache, hidden = segmented.run_layer_forward(
                layer_index,
                engine_leaves,
                cache,
                hidden,
                metadata,
            )
            next_caches.append(cache)
            counts["layer_forward"] += 1
          caches = tuple(next_caches)
          if layer_scan_mode in ("verify", "verify_rev"):
            self._p50_scan_verify(
                hidden, scan_hidden, caches, scan_caches, chunk_index
            )
        normalized = segmented.run_norm_forward(
            hidden, state_leaves=engine_leaves
        )
        counts["norm_forward"] += 1
        raw_logits = segmented.run_head_forward(
            normalized, state_leaves=engine_leaves
        )
        logits = raw_logits.astype(jnp.float32)
        counts["head_forward"] += 1
        target_logps, entropy = self._p28_processed_rows_fn(
            logits, target_ids, spec["temperature"]
        )
        counts["processed_forward"] += 1
        chunk_logps.append(target_logps)
        chunk_entropies.append(entropy)

    flat_logps = jnp.concatenate(chunk_logps, axis=0)
    flat_entropies = jnp.concatenate(chunk_entropies, axis=0)
    completion_valid = spec["completion_valid"]
    logps = jnp.where(
        completion_valid,
        jnp.take(flat_logps, spec["source_rows"], axis=0),
        jnp.zeros(completion_valid.shape, jnp.float32),
    )
    entropy = jnp.where(
        completion_valid,
        jnp.take(flat_entropies, spec["source_rows"], axis=0),
        jnp.zeros(completion_valid.shape, jnp.float32),
    )
    return {
        "logps": logps,
        "entropy": entropy,
        "cache_inputs": tuple(cache_inputs),
        "final_caches": caches,
        "counts": counts,
    }

  def _p28_reverse_sequence(
      self, segmented, engine_leaves, spec, dlogps, dentropy
  ):
    """Reverses one fixed-M sequence across chunks and real layer boundaries."""
    replay = self._p28_forward_sequence(
        segmented, engine_leaves, spec, keep_cache_inputs=True
    )
    padded_width = spec["num_chunks"] * self._bucket
    completion_valid = spec["completion_valid"]
    flat_dlogps = jnp.zeros((padded_width,), jnp.float32).at[
        spec["source_rows"]
    ].add(jnp.where(completion_valid, dlogps, 0.0))
    flat_dentropy = jnp.zeros((padded_width,), jnp.float32).at[
        spec["source_rows"]
    ].add(jnp.where(completion_valid, dentropy, 0.0))

    def tree_zeros(tree):
      return jax.tree.map(jnp.zeros_like, tree)

    def tree_add(left, right):
      return jax.tree.map(lambda a, b: a + b, left, right)

    # Un-jitted, these dispatch one tiny program per leaf: the gradient
    # accumulation of a ~310-leaf state walks head/norm/embed plus 28
    # layers x 16 chunks and shows up in a profile as tens of thousands
    # of jit_add launches per update, all host dispatch overhead. Jitting
    # the whole-tree op keeps every leaf's elementwise a + b exactly as
    # it was (no cross-leaf math exists to reassociate), so the committed
    # gradient stays bitwise identical; the 51/51 alignment gate is the
    # judge, and the flag keeps the certified recipe untouched until it
    # rules.
    if os.environ.get("CANON_FUSED_TREE_OPS", "") == "1":
      tree_zeros = jax.jit(tree_zeros)
      tree_add = jax.jit(tree_add)
    reverse_mode = os.environ.get("CANON_P28_BATCHED_REVERSE", "")
    if reverse_mode not in ("", "0", "1", "verify", "scan_fwd"):
      raise FunctionalMappingError(
          "CANON_P28_BATCHED_REVERSE must be unset/0/1/verify/scan_fwd, "
          f"got {reverse_mode!r}"
      )
    # scan_fwd: batched accumulation plus the forward tape rebuilt by the
    # byte-preserving layer tape scan (one program instead of one per
    # layer), with each layer pullback slicing its (cache, hidden) inputs
    # from the stacked tape INSIDE its own program.  The P49 layer-scan
    # ablation lost to the standalone unstack step; slicing inside the
    # pullback removes that step instead of paying it.  The pullback
    # arithmetic is the same per-layer program body on the same values;
    # the 51/51 gate judges the compiled result.
    batched_reverse = reverse_mode in ("1", "scan_fwd")
    scan_fwd_reverse = reverse_mode == "scan_fwd"
    reverse_verify = reverse_mode == "verify"
    if batched_reverse:
      zero_layers, zero_embed, zero_norm, zero_head, zero_caches = (
          segmented.zero_gradient_pack(replay["final_caches"])
      )
      layer_grads = list(zero_layers)
      embed_grad = zero_embed
      norm_grad = zero_norm
      head_grad = zero_head
      dcache_carry = zero_caches
    else:
      layer_grads = [
          tree_zeros(leaves) for leaves in segmented._local_layer_leaves  # pylint: disable=protected-access
      ]
      embed_grad = tree_zeros(segmented._embed_local_leaves)  # pylint: disable=protected-access
      norm_grad = tree_zeros(segmented._norm_local_leaves)  # pylint: disable=protected-access
      head_grad = tree_zeros(segmented._head_local_leaves)  # pylint: disable=protected-access
      dcache_carry = tuple(
          tree_zeros(cache) for cache in replay["final_caches"]
      )
    layer_scan_mode = segmented.layer_scan_mode()
    if (batched_reverse or reverse_verify) and layer_scan_mode:
      raise FunctionalMappingError(
          "CANON_P28_BATCHED_REVERSE requires CANON_P28_LAYER_SCAN unset"
      )
    layer_count = len(segmented._local_layer_leaves)  # pylint: disable=protected-access
    counts = dict(replay["counts"])
    counts.update({
        "embed_pullback": 0,
        "layer_pullback": 0,
        "norm_pullback": 0,
        "head_pullback": 0,
        "processed_pullback": 0,
    })

    with self._set_forward_context(None, self._runner.vllm_config):
      for chunk_index in reversed(range(spec["num_chunks"])):
        input_ids, target_ids, metadata = self._p28_chunk_inputs(
            spec, chunk_index
        )
        caches = replay["cache_inputs"][chunk_index]
        hidden = segmented.run_embed_forward(
            input_ids, state_leaves=engine_leaves
        )
        counts["embed_forward"] += 1
        scan_tape = None
        if layer_scan_mode:
          scan_tape = segmented.run_layers_tape_scan(
              engine_leaves, caches, hidden, metadata
          )
        if scan_fwd_reverse:
          scan_stacked_caches, scan_stacked_hidden, hidden = (
              segmented.run_layers_tape_scan(
                  engine_leaves, caches, hidden, metadata
              )
          )
          layer_tape = [None] * layer_count
          counts["layer_forward"] += layer_count
        elif layer_scan_mode == "1":
          stacked_cache_ins, stacked_hidden_ins, hidden = scan_tape
          hidden_ins = segmented.unstack_hidden_ins(
              engine_leaves, stacked_hidden_ins
          )
          layer_tape = list(zip(caches, hidden_ins))
          counts["layer_forward"] += layer_count
        else:
          layer_tape = []
          for layer_index, cache in enumerate(caches):
            layer_tape.append((cache, hidden))
            _, hidden = segmented.run_layer_forward(
                layer_index,
                engine_leaves,
                cache,
                hidden,
                metadata,
            )
            counts["layer_forward"] += 1
          if layer_scan_mode in ("verify", "verify_rev"):
            stacked_cache_ins, stacked_hidden_ins, scan_hidden_out = scan_tape
            tape_faults = [
                fault
                for fault in (
                    self._p50_rev_verify(
                        "tape hidden_ins",
                        chunk_index,
                        jnp.stack([entry[1] for entry in layer_tape]),
                        stacked_hidden_ins,
                    ),
                    self._p50_rev_verify(
                        "tape hidden_out", chunk_index, hidden, scan_hidden_out
                    ),
                )
                if fault is not None
            ]
            if tape_faults:
              raise FunctionalMappingError(
                  "P50 tape-scan verify mismatch at chunk "
                  f"{chunk_index}: {'; '.join(tape_faults)}"
              )
        pre_norm = hidden
        normalized = segmented.run_norm_forward(
            pre_norm, state_leaves=engine_leaves
        )
        counts["norm_forward"] += 1
        raw_logits = segmented.run_head_forward(
            normalized, state_leaves=engine_leaves
        )
        logits = raw_logits.astype(jnp.float32)
        counts["head_forward"] += 1
        start = chunk_index * self._bucket
        dchunk_logps = flat_dlogps[start : start + self._bucket]
        dchunk_entropy = flat_dentropy[start : start + self._bucket]
        dlogits = self._p28_processed_rows_pullback_fn(
            logits,
            target_ids,
            spec["temperature"],
            dchunk_logps,
            dchunk_entropy,
        )
        # This is the transpose of the explicit bf16 -> fp32 cast above.
        # JAX VJPs require a cotangent with the differentiated output's dtype.
        dlogits = dlogits.astype(raw_logits.dtype)
        counts["processed_pullback"] += 1
        local_head_grad, dnormalized = segmented.run_head_pullback(
            normalized, dlogits, state_leaves=engine_leaves
        )
        counts["head_pullback"] += 1
        if batched_reverse or reverse_verify:
          acc_snapshot = (
              tuple(layer_grads), embed_grad, norm_grad, head_grad
          )
        if not batched_reverse:
          head_grad = tree_add(head_grad, local_head_grad)
        local_norm_grad, dhidden = segmented.run_norm_pullback(
            pre_norm, dnormalized, state_leaves=engine_leaves
        )
        counts["norm_pullback"] += 1
        if not batched_reverse:
          norm_grad = tree_add(norm_grad, local_norm_grad)

        if layer_scan_mode == "1":
          chunk_grads = []
          previous_cache_carry = [None] * layer_count
          for layer_index in reversed(range(layer_count)):
            cache_in, hidden_in = layer_tape[layer_index]
            local_grad, dcache, dhidden = segmented.run_block_pullback(
                layer_index,
                cache_in,
                hidden_in,
                metadata,
                dcache_carry[layer_index],
                dhidden,
                state_leaves=engine_leaves,
            )
            chunk_grads.append(local_grad)
            previous_cache_carry[layer_index] = dcache
            counts["layer_pullback"] += 1
          dcache_carry = tuple(previous_cache_carry)
          layer_grads = list(
              segmented.accumulate_layer_grads(
                  engine_leaves,
                  layer_grads,
                  list(reversed(chunk_grads)),
              )
          )
        else:
          verify_dcaches_in = verify_dhidden_in = None
          verify_local_grads = None
          if layer_scan_mode == "verify_rev":
            verify_dcaches_in = jax.tree.map(
                lambda *xs: jnp.stack(xs), *dcache_carry
            )
            verify_dhidden_in = dhidden
            verify_local_grads = []
          chunk_layer_grads = (
              [] if batched_reverse or reverse_verify else None
          )
          previous_cache_carry = [None] * len(layer_tape)
          for layer_index in reversed(range(len(layer_tape))):
            if scan_fwd_reverse:
              local_grad, dcache, dhidden = segmented.run_block_pullback_tape(
                  layer_index,
                  scan_stacked_caches,
                  scan_stacked_hidden,
                  metadata,
                  dcache_carry[layer_index],
                  dhidden,
                  state_leaves=engine_leaves,
              )
            else:
              cache_in, hidden_in = layer_tape[layer_index]
              local_grad, dcache, dhidden = segmented.run_block_pullback(
                  layer_index,
                  cache_in,
                  hidden_in,
                  metadata,
                  dcache_carry[layer_index],
                  dhidden,
                  state_leaves=engine_leaves,
              )
            if chunk_layer_grads is not None:
              chunk_layer_grads.append(local_grad)
            if not batched_reverse:
              layer_grads[layer_index] = tree_add(
                  layer_grads[layer_index], local_grad
              )
            if verify_local_grads is not None:
              verify_local_grads.append(local_grad)
            previous_cache_carry[layer_index] = dcache
            counts["layer_pullback"] += 1
          dcache_carry = tuple(previous_cache_carry)
          if layer_scan_mode == "verify_rev":
            # The loop above walked layers high->low; restack in layer order.
            verify_local_grads = list(reversed(verify_local_grads))
            scan_dleaves, scan_dcache_ins, scan_dh = (
                segmented.run_layers_rev_scan(
                    engine_leaves,
                    stacked_cache_ins,
                    stacked_hidden_ins,
                    metadata,
                    verify_dcaches_in,
                    verify_dhidden_in,
                )
            )
            rev_faults = [
                fault
                for fault in (
                    self._p50_rev_verify(
                        "rev dleaves",
                        chunk_index,
                        jax.tree.map(
                            lambda *xs: jnp.stack(xs), *verify_local_grads
                        ),
                        scan_dleaves,
                    ),
                    self._p50_rev_verify(
                        "rev dcache",
                        chunk_index,
                        jax.tree.map(lambda *xs: jnp.stack(xs), *dcache_carry),
                        scan_dcache_ins,
                    ),
                    self._p50_rev_verify(
                        "rev dhidden", chunk_index, dhidden, scan_dh
                    ),
                )
                if fault is not None
            ]
            if rev_faults:
              raise FunctionalMappingError(
                  "P50 reverse-scan verify mismatch at chunk "
                  f"{chunk_index}: {'; '.join(f.split(':')[0] for f in rev_faults)}"
              )
        local_embed_grad = segmented.run_embed_pullback(
            input_ids, dhidden, state_leaves=engine_leaves
        )
        if not batched_reverse:
          embed_grad = tree_add(embed_grad, local_embed_grad)
        counts["embed_pullback"] += 1
        if batched_reverse or reverse_verify:
          delta_pack = (
              tuple(reversed(chunk_layer_grads)),
              local_embed_grad,
              local_norm_grad,
              local_head_grad,
          )
          batched_acc = self._batched_report_add(acc_snapshot, delta_pack)
          if batched_reverse:
            layer_tuple, embed_grad, norm_grad, head_grad = batched_acc
            layer_grads = list(layer_tuple)
          else:
            fault = self._p50_rev_verify(
                "reverse accumulate",
                chunk_index,
                (tuple(layer_grads), embed_grad, norm_grad, head_grad),
                batched_acc,
            )
            if fault is not None:
              raise FunctionalMappingError(
                  f"P52 batched-reverse verify mismatch: {fault}"
              )

    return {
        "engine_gradients": segmented.assemble_full_state_gradient(
            embed=embed_grad,
            layers=tuple(layer_grads),
            norm=norm_grad,
            head=head_grad,
        ),
        "initial_cache_cotangents": dcache_carry,
        "counts": counts,
        "replay_logps": replay["logps"],
        "replay_entropy": replay["entropy"],
    }

  def _p32_group_spec(
      self,
      prompt,
      completion,
      prompt_valid,
      completion_valid,
      temperature,
  ):
    """Builds one fixed-M schedule with one sequence per DP rank."""
    _P28SegmentedEngineForward._reject_outer_transform(  # pylint: disable=protected-access
        prompt, completion, prompt_valid, completion_valid
    )
    p59_four_chip_proxy = (
        self._data_size == 4
        and os.environ.get("CANON_P32_WORKLOAD", "")
        == "gsm8k-p59-dp4-tp1"
    )
    p66_tp4_proxy = (
        self._data_size == 1
        and self._tp_size == 4
        and os.environ.get("CANON_P32_WORKLOAD", "")
        == "gsm8k-p66-dp1-tp4"
        and bool(_p66_tp4_arm())
    )
    if (
        self._data_size not in (8, 16)
        and not p59_four_chip_proxy
        and not p66_tp4_proxy
    ):
      raise FunctionalMappingError(
          "P32 grouped reverse requires data size 8 or 16, the exact "
          "P59 four-chip proxy, or the exact P66 DP1xTP4 proxy; got "
          f"{self._data_size}"
      )
    prompt = jnp.asarray(prompt)
    completion = jnp.asarray(completion)
    prompt_valid = jnp.asarray(prompt_valid, dtype=jnp.bool_)
    completion_valid = jnp.asarray(completion_valid, dtype=jnp.bool_)
    expected_prefix = (self._data_size,)
    values = {
        "prompt": prompt,
        "completion": completion,
        "prompt_valid": prompt_valid,
        "completion_valid": completion_valid,
    }
    for label, value in values.items():
      if value.ndim != 2 or value.shape[:1] != expected_prefix:
        raise FunctionalMappingError(
            f"P32 {label} group must have one row per DP rank: "
            f"shape={value.shape} data={self._data_size}"
        )
    if prompt.shape != prompt_valid.shape:
      raise FunctionalMappingError("P32 grouped prompt mask shape changed")
    if completion.shape != completion_valid.shape:
      raise FunctionalMappingError("P32 grouped completion mask shape changed")

    full = jnp.concatenate((prompt, completion), axis=1)
    valid = jnp.concatenate((prompt_valid, completion_valid), axis=1)
    n_real = jnp.sum(valid, axis=1, dtype=jnp.int32)
    prompt_length = jnp.sum(prompt_valid, axis=1, dtype=jnp.int32)
    completion_length = jnp.sum(
        completion_valid, axis=1, dtype=jnp.int32
    )
    host_n_real = np.asarray(jax.device_get(n_real), dtype=np.int32)
    host_prompt_length = np.asarray(
        jax.device_get(prompt_length), dtype=np.int32
    )
    host_completion_length = np.asarray(
        jax.device_get(completion_length), dtype=np.int32
    )
    if np.any(host_n_real < 2) or np.any(host_prompt_length < 1) or np.any(
        host_completion_length < 1
    ):
      raise FunctionalMappingError(
          "P32 grouped reverse requires nonempty prompt/completion on every "
          f"rank: n={host_n_real.tolist()} "
          f"prompt={host_prompt_length.tolist()} "
          f"completion={host_completion_length.tolist()}"
      )
    if np.any(host_n_real > self._max_model_len):
      raise FunctionalMappingError(
          "P32 grouped sequence exceeds the model limit: "
          f"n={host_n_real.tolist()} max={self._max_model_len}"
      )
    num_chunks = int(
        (int(host_n_real.max()) + self._sequence_bucket - 1)
        // self._sequence_bucket
    )
    padded_width = num_chunks * self._sequence_bucket

    def pack_row(full_row, valid_row, count):
      order = jnp.nonzero(valid_row, size=padded_width, fill_value=0)[0]
      active = jnp.arange(padded_width, dtype=jnp.int32) < count
      return jnp.where(
          active, full_row[order], jnp.asarray(0, full_row.dtype)
      )

    packed_ids = jax.vmap(pack_row)(full, valid, n_real)
    next_ids = jnp.concatenate(
        (
            packed_ids[:, 1:],
            jnp.zeros((self._data_size, 1), packed_ids.dtype),
        ),
        axis=1,
    )
    completion_ordinal = (
        jnp.cumsum(completion_valid, axis=1, dtype=jnp.int32) - 1
    )
    source_rows = jnp.clip(
        prompt_length[:, None] + completion_ordinal - 1,
        0,
        padded_width - 1,
    )
    return {
        "packed_ids": packed_ids,
        "next_ids": next_ids,
        "source_rows": source_rows,
        "completion_valid": completion_valid,
        "n_real": n_real,
        "host_n_real": tuple(int(value) for value in host_n_real),
        "num_chunks": num_chunks,
        "temperature": jnp.asarray(temperature, jnp.float32),
    }

  def _p32_group_chunk_inputs(self, spec, chunk_index):
    """Constructs one global-M engine call from data-rank-local sequences."""
    chunk_index = int(chunk_index)
    chunk_start = chunk_index * self._sequence_bucket
    if os.environ.get("CANON_FUSED_TREE_OPS", "") == "1":
      # Same guards the eager helper enforces per call; config is pinned
      # but a fused path must not be the one that skips the checks.
      if self._data_size < 1 or self._max_num_reqs % self._data_size:
        raise FunctionalMappingError(
            "RPA metadata requires max_num_reqs divisible by data size"
        )
      if spec["n_real"].shape != (self._data_size,):
        raise FunctionalMappingError(
            "RPA metadata lengths must contain one scalar per data rank"
        )
      (
          ids_flat,
          targets_flat,
          positions_flat,
          block_tables,
          seq_lens,
          query_start,
          request_distribution,
      ) = _fused_chunk_metadata(
          spec["n_real"],
          spec["packed_ids"],
          spec["next_ids"],
          jnp.asarray(chunk_start, jnp.int32),
          self._sequence_bucket,
          int(self._data_size),
          int(self._max_num_reqs),
          int(self._blocks_per_req),
      )
      metadata = self._metadata_cls(
          input_positions=self._engine_array(positions_flat),
          block_tables=self._engine_array(block_tables),
          seq_lens=self._engine_array(seq_lens),
          query_start_loc=self._engine_array(query_start),
          request_distribution=self._engine_array(request_distribution),
      )
      metadata.padded_num_reqs = self._max_num_reqs
      return (
          self._engine_array(ids_flat),
          self._engine_array(targets_flat),
          metadata,
      )
    rows = jnp.arange(self._sequence_bucket, dtype=jnp.int32)
    q_len = jnp.clip(
        spec["n_real"] - chunk_start, 0, self._sequence_bucket
    )
    kv_len = jnp.where(
        q_len > 0,
        jnp.minimum(spec["n_real"], chunk_start + self._sequence_bucket),
        0,
    )
    chunk_ids_group = spec["packed_ids"][
        :, chunk_start : chunk_start + self._sequence_bucket
    ]
    chunk_targets_group = spec["next_ids"][
        :, chunk_start : chunk_start + self._sequence_bucket
    ]
    positions_group = jnp.where(
        rows[None, :] < q_len[:, None], chunk_start + rows[None, :], 0
    )
    (
        block_tables,
        seq_lens,
        query_start,
        request_distribution,
    ) = _canonical_dp_attention_metadata_arrays(
        data_size=self._data_size,
        max_num_reqs=self._max_num_reqs,
        blocks_per_req=self._blocks_per_req,
        q_len=q_len,
        kv_len=kv_len,
    )
    metadata = self._metadata_cls(
        input_positions=self._engine_array(positions_group.reshape(-1)),
        block_tables=self._engine_array(block_tables),
        seq_lens=self._engine_array(seq_lens),
        query_start_loc=self._engine_array(query_start),
        request_distribution=self._engine_array(request_distribution),
    )
    metadata.padded_num_reqs = self._max_num_reqs
    return (
        self._engine_array(chunk_ids_group.reshape(-1)),
        self._engine_array(chunk_targets_group.reshape(-1)),
        metadata,
    )

  def _p32_forward_group(
      self, segmented, engine_leaves, spec, *, keep_cache_inputs
  ):
    """Runs independent DP-rank sequences through segmented Qwen."""
    caches = tuple(self._fresh_caches())
    cache_inputs = []
    chunk_logps = []
    chunk_entropies = []
    counts = {
        "embed_forward": 0,
        "layer_forward": 0,
        "norm_forward": 0,
        "head_forward": 0,
        "processed_forward": 0,
    }
    with self._set_forward_context(None, self._runner.vllm_config):
      for chunk_index in range(spec["num_chunks"]):
        input_ids, target_ids, metadata = self._p32_group_chunk_inputs(
            spec, chunk_index
        )
        if keep_cache_inputs:
          cache_inputs.append(caches)
        hidden = segmented.run_embed_forward(
            input_ids, state_leaves=engine_leaves
        )
        counts["embed_forward"] += 1
        layer_scan_mode = segmented.layer_scan_mode()
        scan_caches = scan_hidden = None
        if layer_scan_mode:
          scan_caches, scan_hidden = segmented.run_layers_scan(
              engine_leaves, caches, hidden, metadata
          )
        if layer_scan_mode == "1":
          caches = scan_caches
          hidden = scan_hidden
          counts["layer_forward"] += len(caches)
        else:
          next_caches = []
          for layer_index, cache in enumerate(caches):
            cache, hidden = segmented.run_layer_forward(
                layer_index,
                engine_leaves,
                cache,
                hidden,
                metadata,
            )
            next_caches.append(cache)
            counts["layer_forward"] += 1
          caches = tuple(next_caches)
          if layer_scan_mode in ("verify", "verify_rev"):
            self._p50_scan_verify(
                hidden, scan_hidden, caches, scan_caches, chunk_index
            )
        normalized = segmented.run_norm_forward(
            hidden, state_leaves=engine_leaves
        )
        counts["norm_forward"] += 1
        raw_logits = segmented.run_head_forward(
            normalized, state_leaves=engine_leaves
        )
        logits = raw_logits.astype(jnp.float32)
        counts["head_forward"] += 1
        target_logps, entropy = self._p28_processed_rows_fn(
            logits, target_ids, spec["temperature"]
        )
        counts["processed_forward"] += 1
        chunk_logps.append(
            target_logps.reshape(self._data_size, self._sequence_bucket)
        )
        chunk_entropies.append(
            entropy.reshape(self._data_size, self._sequence_bucket)
        )

    flat_logps = jnp.concatenate(chunk_logps, axis=1)
    flat_entropies = jnp.concatenate(chunk_entropies, axis=1)
    completion_valid = spec["completion_valid"]
    logps = jnp.where(
        completion_valid,
        jnp.take_along_axis(flat_logps, spec["source_rows"], axis=1),
        jnp.zeros(completion_valid.shape, jnp.float32),
    )
    entropy = jnp.where(
        completion_valid,
        jnp.take_along_axis(flat_entropies, spec["source_rows"], axis=1),
        jnp.zeros(completion_valid.shape, jnp.float32),
    )
    return {
        "logps": logps,
        "entropy": entropy,
        "cache_inputs": tuple(cache_inputs),
        "final_caches": caches if keep_cache_inputs else (),
        "counts": counts,
    }

  def _p32_reverse_group(
      self, segmented, engine_leaves, spec, dlogps, dentropy, replay=None
  ):
    """Reverses one group of rank-local sequences by layer and chunk."""
    parallel_value = os.environ.get("CANON_P59_RANK_PARALLEL_BACKWARD", "")
    if parallel_value not in ("", "0", "1"):
      raise FunctionalMappingError(
          "CANON_P59_RANK_PARALLEL_BACKWARD must be unset/0/1, "
          f"got {parallel_value!r}"
      )
    rank_parallel = parallel_value == "1"
    p66_arm = _p66_tp4_arm()
    p66_oracle = p66_arm == "tp4-vma-oracle"
    p66_unit_data = (
        self._data_size == 1
        and self._tp_size == 4
        and p66_arm
        in (
            "tp4-p59-old",
            "tp4-p59",
            "tp4-gather-off",
            "tp4-vma-oracle",
        )
    )
    if rank_parallel and self._data_size <= 1 and not p66_unit_data:
      raise FunctionalMappingError(
          "P59 rank-parallel backward requires more than one DP rank"
      )
    if replay is None:
      replay = self._p32_forward_group(
          segmented, engine_leaves, spec, keep_cache_inputs=True
      )
    padded_width = spec["num_chunks"] * self._sequence_bucket
    completion_valid = spec["completion_valid"]
    rank_rows = jnp.arange(self._data_size, dtype=jnp.int32)[:, None]
    flat_dlogps = jnp.zeros(
        (self._data_size, padded_width), jnp.float32
    ).at[rank_rows, spec["source_rows"]].add(
        jnp.where(completion_valid, dlogps, 0.0)
    )
    flat_dentropy = jnp.zeros(
        (self._data_size, padded_width), jnp.float32
    ).at[rank_rows, spec["source_rows"]].add(
        jnp.where(completion_valid, dentropy, 0.0)
    )

    def tree_zeros(tree):
      return jax.tree.map(jnp.zeros_like, tree)

    def tree_add(left, right):
      return jax.tree.map(lambda a, b: a + b, left, right)

    def tree_start(right):
      return jax.tree.map(
          lambda value: jnp.asarray(0, value.dtype) + value, right
      )

    # Un-jitted, these dispatch one tiny program per leaf: the gradient
    # accumulation of a ~310-leaf state walks head/norm/embed plus 28
    # layers x 16 chunks and shows up in a profile as tens of thousands
    # of jit_add launches per update, all host dispatch overhead. Jitting
    # the whole-tree op keeps every leaf's elementwise a + b exactly as
    # it was (no cross-leaf math exists to reassociate), so the committed
    # gradient stays bitwise identical; the 51/51 alignment gate is the
    # judge, and the flag keeps the certified recipe untouched until it
    # rules.
    if os.environ.get("CANON_FUSED_TREE_OPS", "") == "1":
      tree_zeros = jax.jit(tree_zeros)
      tree_add = jax.jit(tree_add)
      tree_start = jax.jit(tree_start)
    if rank_parallel:
      layer_grads = [
          None
          for _ in segmented._local_layer_leaves  # pylint: disable=protected-access
      ]
      embed_grad = norm_grad = head_grad = None
    else:
      layer_grads = [
          tree_zeros(leaves)
          for leaves in segmented._local_layer_leaves  # pylint: disable=protected-access
      ]
      embed_grad = tree_zeros(segmented._embed_local_leaves)  # pylint: disable=protected-access
      norm_grad = tree_zeros(segmented._norm_local_leaves)  # pylint: disable=protected-access
      head_grad = tree_zeros(segmented._head_local_leaves)  # pylint: disable=protected-access
    dcache_carry = tuple(
        tree_zeros(cache) for cache in replay["final_caches"]
    )
    counts = dict(replay["counts"])
    counts.update({
        "embed_pullback": 0,
        "layer_pullback": 0,
        "norm_pullback": 0,
        "head_pullback": 0,
        "processed_pullback": 0,
    })
    p66_row_profiles = []
    p66_oracle_records = []
    if p66_arm:

      @jax.jit
      def p66_row_profile(hidden_value, dhidden_value):
        hidden_rows = hidden_value.reshape(
            self._data_size, self._sequence_bucket, -1
        ).astype(jnp.float32)
        dhidden_rows = dhidden_value.reshape(
            self._data_size, self._sequence_bucket, -1
        ).astype(jnp.float32)
        return (
            jnp.sqrt(jnp.mean(jnp.square(hidden_rows), axis=-1)),
          jnp.max(jnp.abs(dhidden_rows), axis=-1),
        )
    if p66_oracle:
      p66_vjp_oracle.negative_control()

    with self._set_forward_context(None, self._runner.vllm_config):
      for chunk_index in reversed(range(spec["num_chunks"])):
        input_ids, target_ids, metadata = self._p32_group_chunk_inputs(
            spec, chunk_index
        )
        caches = replay["cache_inputs"][chunk_index]
        hidden = segmented.run_embed_forward(
            input_ids, state_leaves=engine_leaves
        )
        counts["embed_forward"] += 1
        layer_tape = []
        for layer_index, cache in enumerate(caches):
          layer_tape.append((cache, hidden))
          _, hidden = segmented.run_layer_forward(
              layer_index,
              engine_leaves,
              cache,
              hidden,
              metadata,
          )
          counts["layer_forward"] += 1
        pre_norm = hidden
        normalized = segmented.run_norm_forward(
            pre_norm, state_leaves=engine_leaves
        )
        counts["norm_forward"] += 1
        raw_logits = segmented.run_head_forward(
            normalized, state_leaves=engine_leaves
        )
        logits = raw_logits.astype(jnp.float32)
        counts["head_forward"] += 1
        start = chunk_index * self._sequence_bucket
        dchunk_logps = flat_dlogps[
            :, start : start + self._sequence_bucket
        ].reshape(-1)
        dchunk_entropy = flat_dentropy[
            :, start : start + self._sequence_bucket
        ].reshape(-1)
        dlogits = self._p28_processed_rows_pullback_fn(
            logits,
            target_ids,
            spec["temperature"],
            dchunk_logps,
            dchunk_entropy,
        ).astype(raw_logits.dtype)
        counts["processed_pullback"] += 1
        if rank_parallel:
          local_head_grad, dnormalized = (
              segmented.run_head_pullback_rank_parallel(
                  normalized, dlogits, state_leaves=engine_leaves
              )
          )
          if p66_oracle and chunk_index == 0:
            serial_head_grad, serial_dnormalized = (
                segmented.run_head_pullback(
                    normalized, dlogits, state_leaves=engine_leaves
                )
            )
            p66_oracle_records.append(p66_vjp_oracle.compare(
                (serial_head_grad, serial_dnormalized),
                (
                    p66_vjp_oracle.unstage_unit_rank(
                        local_head_grad, endpoint="head"
                    ),
                    dnormalized,
                ),
                endpoint="head",
            ))
        else:
          local_head_grad, dnormalized = segmented.run_head_pullback(
              normalized, dlogits, state_leaves=engine_leaves
          )
        counts["head_pullback"] += 1
        head_grad = (
            tree_start(local_head_grad)
            if head_grad is None
            else tree_add(head_grad, local_head_grad)
        )
        if rank_parallel:
          local_norm_grad, dhidden = (
              segmented.run_norm_pullback_rank_parallel(
                  pre_norm, dnormalized, state_leaves=engine_leaves
              )
          )
          if p66_oracle and chunk_index == 0:
            serial_norm_grad, serial_dhidden = (
                segmented.run_norm_pullback(
                    pre_norm, dnormalized, state_leaves=engine_leaves
                )
            )
            p66_oracle_records.append(p66_vjp_oracle.compare(
                (serial_norm_grad, serial_dhidden),
                (
                    p66_vjp_oracle.unstage_unit_rank(
                        local_norm_grad, endpoint="norm"
                    ),
                    dhidden,
                ),
                endpoint="norm",
            ))
        else:
          local_norm_grad, dhidden = segmented.run_norm_pullback(
              pre_norm, dnormalized, state_leaves=engine_leaves
          )
        counts["norm_pullback"] += 1
        norm_grad = (
            tree_start(local_norm_grad)
            if norm_grad is None
            else tree_add(norm_grad, local_norm_grad)
        )

        previous_cache_carry = [None] * len(layer_tape)
        for layer_index in reversed(range(len(layer_tape))):
          cache_in, hidden_in = layer_tape[layer_index]
          incoming_dcache = dcache_carry[layer_index]
          incoming_dhidden = dhidden
          if rank_parallel:
            local_grad, dcache, dhidden = (
                segmented.run_block_pullback_rank_parallel(
                    layer_index,
                    cache_in,
                    hidden_in,
                    metadata,
                    incoming_dcache,
                    incoming_dhidden,
                    state_leaves=engine_leaves,
                )
            )
            if (
                p66_oracle
                and chunk_index == 0
                and layer_index in (27, 14, 0)
            ):
              serial_grad, serial_dcache, serial_dhidden = (
                  segmented.run_block_pullback(
                      layer_index,
                      cache_in,
                      hidden_in,
                      metadata,
                      incoming_dcache,
                      incoming_dhidden,
                      state_leaves=engine_leaves,
                  )
              )
              p66_oracle_records.append(p66_vjp_oracle.compare(
                  (serial_grad, serial_dcache, serial_dhidden),
                  (
                      p66_vjp_oracle.unstage_unit_rank(
                          local_grad, endpoint=f"layer_{layer_index}"
                      ),
                      dcache,
                      dhidden,
                  ),
                  endpoint=f"layer_{layer_index}",
              ))
          else:
            local_grad, dcache, dhidden = segmented.run_block_pullback(
                layer_index,
                cache_in,
                hidden_in,
                metadata,
                incoming_dcache,
                incoming_dhidden,
                state_leaves=engine_leaves,
            )
          if p66_arm:
            hidden_rms, dhidden_max = p66_row_profile(hidden_in, dhidden)
            p66_row_profiles.append(
                (chunk_index, layer_index, hidden_rms, dhidden_max)
            )
          layer_grads[layer_index] = (
              tree_start(local_grad)
              if layer_grads[layer_index] is None
              else tree_add(layer_grads[layer_index], local_grad)
          )
          previous_cache_carry[layer_index] = dcache
          counts["layer_pullback"] += 1
        dcache_carry = tuple(previous_cache_carry)
        if rank_parallel:
          local_embed_grad = segmented.run_embed_pullback_rank_parallel(
              input_ids, dhidden, state_leaves=engine_leaves
          )
          if p66_oracle and chunk_index == 0:
            serial_embed_grad = segmented.run_embed_pullback(
                input_ids, dhidden, state_leaves=engine_leaves
            )
            p66_oracle_records.append(p66_vjp_oracle.compare(
                serial_embed_grad,
                p66_vjp_oracle.unstage_unit_rank(
                    local_embed_grad, endpoint="embed"
                ),
                endpoint="embed",
            ))
        else:
          local_embed_grad = segmented.run_embed_pullback(
              input_ids, dhidden, state_leaves=engine_leaves
          )
        embed_grad = (
            tree_start(local_embed_grad)
            if embed_grad is None
            else tree_add(embed_grad, local_embed_grad)
        )
        counts["embed_pullback"] += 1

    if any(
        value is None
        for value in (embed_grad, norm_grad, head_grad, *layer_grads)
    ):
      raise FunctionalMappingError("P59 reverse emitted an empty gradient pack")
    p66_row_summary = None
    if p66_arm:
      p66_row_summary = _p66_emit_row_cotangent_profile(
          p66_row_profiles,
          arm=p66_arm,
          host_n_real=spec["host_n_real"],
          sequence_bucket=self._sequence_bucket,
      )
    p66_oracle_summary = None
    if p66_oracle:
      expected_endpoints = {
          "head", "norm", "layer_27", "layer_14", "layer_0", "embed"
      }
      observed_endpoints = [record["endpoint"] for record in p66_oracle_records]
      p66_oracle_summary = {
          "schema": "canon-p66-same-point-vjp-oracle-summary-v1",
          "arm": p66_arm,
          "negative_control_detected": True,
          "expected_endpoints": sorted(expected_endpoints),
          "observed_endpoints": observed_endpoints,
          "records": p66_oracle_records,
          "verdict": (
              "PASS"
              if set(observed_endpoints) == expected_endpoints
              and len(observed_endpoints) == len(expected_endpoints)
              and all(record["verdict"] == "PASS" for record in p66_oracle_records)
              else "FAIL"
          ),
      }
      print(
          "[P66.ORACLE.SUMMARY] "
          + json.dumps(
              p66_oracle_summary, sort_keys=True, separators=(",", ":")
          ),
          flush=True,
      )
      if p66_oracle_summary["verdict"] != "PASS":
        raise FunctionalMappingError(
            f"P66 same-point VJP oracle failed: {p66_oracle_summary}"
        )
    return {
        "engine_gradients": segmented.assemble_full_state_gradient(
            embed=embed_grad,
            layers=tuple(layer_grads),
            norm=norm_grad,
            head=head_grad,
            rank_axis_size=self._data_size if rank_parallel else None,
        ),
        "initial_cache_cotangents": dcache_carry,
        "counts": counts,
        "replay_logps": replay["logps"],
        "replay_entropy": replay["entropy"],
        "p66_row_cotangent_summary": p66_row_summary,
        "p66_vjp_oracle": p66_oracle_summary,
    }

  def segmented_dp_grpo_value_and_grad(
      self,
      *,
      trainer_state,
      train_example,
      algo_config,
      pad_id,
      eos_id,
      gradient_microbatch_sink=None,
      deterministic_repeat=False,
      xprof_train_schedule=None,
  ):
    """Runs rank-local DP reverse and one fixed reduction per group.

    Each rank-major group contains one trajectory from every DP rank. The
    cotangent is isolated rank by rank, then staged on a physically partitioned
    leading DP axis. One fixed reduce-and-broadcast transaction produces an
    exactly replicated gradient before it is streamed into the donated
    optimizer accumulator. Without a sink, materialized accumulation is kept
    only for bounded tests.
    """
    del eos_id
    _P28SegmentedEngineForward._reject_outer_transform(  # pylint: disable=protected-access
        trainer_state, train_example
    )
    if os.environ.get("CANON_P32_DP16_SEGMENTED", "") != "1":
      raise FunctionalMappingError(
          "P32 DP16 reverse requires CANON_P32_DP16_SEGMENTED=1"
      )
    p34 = os.environ.get("CANON_P34_DEEPSWE", "") == "1"
    workload = (
        deepswe_contract.active_workload(os.environ)
        if p34
        else dp_workloads.active_workload(os.environ)
    )
    if workload is None:
      raise FunctionalMappingError(
          "grouped reverse requires an active workload contract"
      )
    expected_tp = workload.tp_size
    expected_dp = workload.dp_size
    if (
        os.environ.get("CANON_P32_TRAIN_ADMITTED", "") != "1"
        or self._data_size != expected_dp
        or self._tp_size != expected_tp
    ):
      raise FunctionalMappingError(
          "grouped reverse requires the admitted "
          f"DP{expected_dp}xTP{expected_tp} contract"
      )
    if os.environ.get("CANON_P28_SEGMENTED_TRAIN", "") != "1":
      raise FunctionalMappingError(
          "P32 grouped reverse requires CANON_P28_SEGMENTED_TRAIN=1"
      )
    if (
        os.environ.get("CANON_P32_DP_REDUCTION_ADMITTED", "") != "1"
        or os.environ.get("CANON_P33_WORKLOAD_LAUNCH_ADMITTED", "") != "1"
    ):
      raise FunctionalMappingError(
          "P33 rank-local reverse requires admitted reduction and workload "
          "launch gates"
      )
    if p34:
      deepswe_contract.validate_environment(os.environ)
      contract = workload
      reverse_groups = contract.rank_major_rows()
    else:
      dp_workloads.validate_environment(
          workload, require_reduction_admission=True
      )
      contract = workload.training_contract()
      reverse_groups = contract.rank_major_reverse_groups()
    trainer_dp_axis = self._dp_axis
    if not p34:
      _, trainer_dp_axis = _p59_replicated_data_mesh(
          trainer_state, "P32 grouped trainer state"
      )
    if getattr(train_example, "segment_ids", None) is not None:
      raise FunctionalMappingError("P32 D3b0 admits unpacked trajectories only")

    prompts = jnp.asarray(train_example.prompt_ids)
    completions = jnp.asarray(train_example.completion_ids)
    prompt_masks = jnp.asarray(train_example.prompt_mask, dtype=jnp.bool_)
    completion_masks = jnp.asarray(
        train_example.completion_mask, dtype=jnp.bool_
    )
    completion_valid_value = getattr(
        train_example, "completion_valid_mask", None
    )
    completion_valid_masks = (
        completion_masks
        if completion_valid_value is None
        else jnp.asarray(completion_valid_value, dtype=jnp.bool_)
    )
    if prompts.shape != prompt_masks.shape:
      raise FunctionalMappingError("P32 D3b0 prompt batch/mask mismatch")
    if completions.shape != completion_masks.shape:
      raise FunctionalMappingError("P32 D3b0 completion batch/mask mismatch")
    if completions.shape != completion_valid_masks.shape:
      raise FunctionalMappingError(
          "P32 D3b0 completion batch/valid-mask mismatch"
      )
    if bool(np.asarray(jax.device_get(jnp.any(
        completion_masks & ~completion_valid_masks
    )))):
      raise FunctionalMappingError(
          "P32 D3b0 action mask is not a subset of completion validity"
      )
    if prompts.shape[0] != completions.shape[0]:
      raise FunctionalMappingError("P32 D3b0 prompt/completion batch mismatch")
    if int(prompts.shape[0]) != contract.global_trajectories:
      raise FunctionalMappingError(
          "P32 grouped reverse requires the frozen global trajectory count: "
          f"{prompts.shape[0]} != {contract.global_trajectories}"
      )
    try:
      expected_widths = dp_workloads.expected_token_widths(workload)
    except ValueError as exc:
      raise FunctionalMappingError(str(exc)) from exc
    if (int(prompts.shape[1]), int(completions.shape[1])) != expected_widths:
      raise FunctionalMappingError(
          f"canonical {getattr(workload, 'name', 'deepswe')} token contract changed: "
          f"{prompts.shape[1]}/{completions.shape[1]} != "
          f"{expected_widths[0]}/{expected_widths[1]}"
      )

    grouped_inputs = jax.tree.map(
        self._group_batch_rows,
        (
            prompts,
            completions,
            prompt_masks,
            completion_valid_masks,
        ),
    )
    expected_group_shape = (
        contract.local_trajectories,
        contract.dp_size,
    )
    if grouped_inputs[0].shape[:2] != expected_group_shape:
      raise FunctionalMappingError(
          "P32 rank-major grouping changed: "
          f"{grouped_inputs[0].shape[:2]} != {expected_group_shape}"
      )

    model_config = self._runner.model_config
    mapped = map_trainer_state_to_engine_leaves(
        trainer_state=trainer_state,
        engine_state_contract=self._engine_state_contract,
        key_mappings=self._key_mappings,
        transpose_keys=self._transpose_keys,
        key_mapping_hook_fns=self._hook_fns,
        num_kv_heads=model_config.get_total_num_kv_heads(),
        head_dim=model_config.get_head_size(),
        tp_size=self._tp_size,
    )
    engine_leaves = tuple(mapped.leaves)
    segmented = getattr(self, "_p32_d3b_segmented_engine", None)
    if segmented is None:
      segmented = build_p28_segmented_engine_forward(self._runner)
      self._p32_d3b_segmented_engine = segmented
      phase = "P34" if p34 else "P33"
      print(
          f"[{phase}.DP{contract.dp_size}] segmented_engine_ready "
          f"data={contract.dp_size} tp={self._tp_size} "
          f"groups={contract.local_trajectories} local_M=256 "
          f"global_M={self._bucket}",
          flush=True,
      )

    specs = tuple(
        self._p32_group_spec(
            grouped_inputs[0][index],
            grouped_inputs[1][index],
            grouped_inputs[2][index],
            grouped_inputs[3][index],
            algo_config.temperature,
        )
        for index in range(contract.local_trajectories)
    )
    report_mode = os.environ.get("CANON_P28_BATCHED_REPORT", "")
    if report_mode not in ("", "0", "1", "verify"):
      raise FunctionalMappingError(
          "CANON_P28_BATCHED_REPORT must be unset/0/1/verify, "
          f"got {report_mode!r}"
      )
    batched_report = report_mode == "1"
    report_verify = report_mode == "verify"
    rank_parallel_value = os.environ.get(
        "CANON_P59_RANK_PARALLEL_BACKWARD", ""
    )
    if rank_parallel_value not in ("", "0", "1"):
      raise FunctionalMappingError(
          "CANON_P59_RANK_PARALLEL_BACKWARD must be unset/0/1, "
          f"got {rank_parallel_value!r}"
      )
    rank_parallel_backward = rank_parallel_value == "1"
    p66_tp4_arm = _p66_tp4_arm()
    numeric_debug_mode = _backward_numeric_debug_mode()
    numeric_debug = bool(numeric_debug_mode)
    p64_capsule_mode = (
        p64_training_capsule.mode()
        if numeric_debug_mode == "p64"
        else ""
    )
    if numeric_debug:
      common_numeric_contract = (
          not p34
          and contract.global_trajectories == 256
          and rank_parallel_backward
          and os.environ.get("CANON_P33_RUN_STAGE", "")
          == "backward-no-commit"
          and os.environ.get("CANON_P33_NO_COMMIT", "") == "1"
          and os.environ.get("CANON_P38_FIXED_LM_HEAD", "") == "1"
          and os.environ.get("CANON_V1_HP_FULL", "0") == "0"
      )
      if numeric_debug_mode == "p62":
        numeric_contract = (
            common_numeric_contract
            and workload.name == "gsm8k"
            and (contract.dp_size, contract.tp_size) == (16, 4)
            and contract.local_trajectories == 16
            and self._bucket == 4096
            and os.environ.get(
                "CANON_GSM8K_ALIGNMENT_WARN_ONLY", "0"
            ) == "0"
        )
        admission = (
            "workload=gsm8k dp=16 tp=4 global_trajectories=256 "
            "local_trajectories=16 global_M=4096 local_M=256 "
            "optimizer_commits=0"
        )
      else:
        numeric_contract = (
            common_numeric_contract
            and workload.name == "frozenlake-dp8-tp8"
            and (contract.dp_size, contract.tp_size) == (8, 8)
            and contract.local_trajectories == 32
            and self._bucket == 2048
            and os.environ.get(
                "CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY", "0"
            ) == "0"
            and p64_capsule_mode in ("capture", "replay")
        )
        admission = (
            "workload=frozenlake-dp8-tp8 dp=8 tp=8 "
            "global_trajectories=256 local_trajectories=32 "
            "global_M=2048 local_M=256 optimizer_commits=0"
        )
      if not numeric_contract:
        raise FunctionalMappingError(
            f"{numeric_debug_mode.upper()} numerical debug requires its "
            "exact strict target geometry, P59 fixed-head "
            "backward-no-commit, and must not impersonate V1 full"
        )
      marker, _ = _numeric_debug_identity(numeric_debug_mode)
      print(
          f"[{marker}.NUMERIC] admission {admission}",
          flush=True,
      )
    if p66_tp4_arm:
      expected_rank_parallel = p66_tp4_arm != "tp4-serial"
      expected_gather = p66_tp4_arm != "tp4-gather-off"
      expected_vma = p66_tp4_arm in (
          "tp4-p59", "tp4-gather-off", "tp4-vma-oracle"
      )
      p66_contract = (
          not p34
          and workload.name == "gsm8k-p66-dp1-tp4"
          and (contract.dp_size, contract.tp_size) == (1, 4)
          and contract.global_trajectories == 16
          and contract.local_trajectories == 16
          and self._bucket == 256
          and rank_parallel_backward == expected_rank_parallel
          and os.environ.get("CANON_FIXED_AR_GATHER", "0")
          == ("1" if expected_gather else "0")
          and os.environ.get("CANON_P66_P59_CHECK_VMA", "0")
          == ("1" if expected_vma else "0")
          and os.environ.get("CANON_P33_RUN_STAGE", "")
          == "backward-no-commit"
          and os.environ.get("CANON_P33_NO_COMMIT", "") == "1"
          and os.environ.get("CANON_P38_FIXED_LM_HEAD", "") == "1"
          and os.environ.get("CANON_P63_OVERFLOW_SAFE_CLIP", "0") == "0"
          and not numeric_debug
      )
      if not p66_contract:
        raise FunctionalMappingError(
            "P66 TP4 discriminator requires exact DP1xTP4 group-0 "
            "backward-no-commit geometry"
        )
      print(
          f"[P66.TP4] admission arm={p66_tp4_arm} topology=DP1xTP4 "
          "global_trajectories=16 local_M=256 global_M=256 "
          "reverse_groups=1/16 optimizer_commits=0",
          flush=True,
      )
    p59_xprof_directory = _p59_xprof_backward_directory(
        workload_name=workload.name,
        dp_size=contract.dp_size,
        tp_size=contract.tp_size,
        rank_parallel=rank_parallel_backward,
    )
    p59_xprof_update = -1
    if p59_xprof_directory:
      p59_xprof_update = getattr(self, "_p59_xprof_update", 0)
      self._p59_xprof_update = p59_xprof_update + 1
    p59_xprof_capture = bool(p59_xprof_directory) and p59_xprof_update == 1
    if p59_xprof_directory and p59_xprof_update == 0:
      print(
          "[P59.XPROF] phase=backward_group armed update=1 groups=1",
          flush=True,
      )
    serial_bridge_value = os.environ.get(
        "CANON_P59_DP4_SERIAL_MESH_BRIDGE", ""
    )
    if serial_bridge_value not in ("", "0", "1"):
      raise FunctionalMappingError(
          "CANON_P59_DP4_SERIAL_MESH_BRIDGE must be unset/0/1, "
          f"got {serial_bridge_value!r}"
      )
    serial_mesh_bridge = serial_bridge_value == "1"
    if serial_mesh_bridge and (
        p34
        or workload.name != "gsm8k-p59-dp4-tp1"
        or (contract.dp_size, contract.tp_size) != (4, 1)
    ):
      raise FunctionalMappingError(
          "P59 serial mesh bridge requires the exact DP4xTP1 proxy workload"
      )
    p32_forward_start = time.perf_counter()
    p32_forward_durations = []
    forwards = []
    with gsm8k_xprof.trace_annotation("forward_groups"):
      for index, spec in enumerate(specs):
        with gsm8k_xprof.trace_annotation(
            "forward_group", group_index=index
        ):
          p32_group_start = time.perf_counter()
          forward = self._p32_forward_group(
              segmented, engine_leaves, spec, keep_cache_inputs=False
          )
          forward["logps"].block_until_ready()
          p32_forward_durations.append(time.perf_counter() - p32_group_start)
          forwards.append(forward)
          print(
              f"[P32.DP{contract.dp_size}] forward_group_done "
              f"group={index + 1}/{contract.local_trajectories} "
              f"rows={reverse_groups[index]} "
              f"n_real={spec['host_n_real']}",
              flush=True,
          )
    forwards = tuple(forwards)
    if os.environ.get("CANON_PERF_LOG", "1") != "0" and p32_forward_durations:
      print(
          "[PERF] stage=p32_vag_forward seconds=%.3f groups=%d"
          " mean=%.3f max=%.3f"
          % (
              time.perf_counter() - p32_forward_start,
              len(p32_forward_durations),
              sum(p32_forward_durations) / len(p32_forward_durations),
              max(p32_forward_durations),
          ),
          flush=True,
      )
    grouped_logps = jnp.stack(
        tuple(result["logps"] for result in forwards), axis=0
    ).astype(jnp.float32)
    grouped_entropy = jnp.stack(
        tuple(result["entropy"] for result in forwards), axis=0
    ).astype(jnp.float32)
    per_token_logps = self._ungroup_batch_rows(grouped_logps)
    token_entropy = self._ungroup_batch_rows(grouped_entropy)

    from tunix.rl import algo_core  # pylint: disable=g-import-not-at-top

    def unreduced_loss(logps, entropy):
      return algo_core.grpo_loss_from_precomputed_logps(
          logps, entropy, train_example, algo_config
      ).primary_loss.unreduced_sum

    with gsm8k_xprof.trace_annotation("loss_pullback"):
      unreduced_value, loss_pullback = jax.vjp(
          unreduced_loss, per_token_logps, token_entropy
      )
      dlogps, dentropy = loss_pullback(jnp.ones_like(unreduced_value))
      grouped_dlogps = self._group_batch_rows(dlogps)
      grouped_dentropy = self._group_batch_rows(dentropy)
      loss_output = algo_core.grpo_loss_from_precomputed_logps(
          per_token_logps, token_entropy, train_example, algo_config
      )
      scale = loss_output.primary_loss.compute_scale()
    if p66_tp4_arm:
      reverse = self._p32_reverse_group(
          segmented,
          engine_leaves,
          specs[0],
          grouped_dlogps[0],
          grouped_dentropy[0],
      )
      if not bool(np.asarray(jnp.array_equal(
          reverse["replay_logps"], grouped_logps[0]
      ))):
        raise FunctionalMappingError(
            "P66 TP4 group-0 replay logprobs changed"
        )
      engine_receipt = sft_utils.tree_numeric_receipt(
          reverse["engine_gradients"], ranked=rank_parallel_backward
      )
      print(
          "[P66.TP4.NUMERIC] "
          + json.dumps(
              {
                  "schema": "canon-p66-engine-vjp-v1",
                  "arm": p66_tp4_arm,
                  "stage": "engine_vjp",
                  "group": 0,
                  "groups": contract.local_trajectories,
                  **engine_receipt,
              },
              sort_keys=True,
              separators=(",", ":"),
          ),
          flush=True,
      )
      layerwise_profile = _p66_emit_layerwise_profile(
          segmented, reverse["engine_gradients"], arm=p66_tp4_arm
      )
      if (
          not engine_receipt["all_finite"]
          or not engine_receipt["any_nonzero"]
          or not np.isfinite(engine_receipt["stable_norm"])
          or engine_receipt["stable_norm"] > 1.0e6
      ):
        raise FunctionalMappingError(
            "P66 TP4 diagnostic-fatal engine VJP: "
            f"arm={p66_tp4_arm} receipt={engine_receipt}"
        )
      engine_gradient = reverse["engine_gradients"]
      if rank_parallel_backward:
        engine_gradient = jax.tree.map(
            lambda value: jnp.squeeze(value, axis=0), engine_gradient
        )
      trainer_gradient = self.map_engine_cotangents_to_trainer_state(
          trainer_state, engine_gradient
      )
      trainer_gradient = jax.tree.map(
          lambda value: value.astype(jnp.float32) * scale,
          trainer_gradient,
      )
      return {
          "loss_output": loss_output,
          "loss": loss_output.primary_loss.compute(),
          "per_token_logps": per_token_logps,
          "token_entropy": token_entropy,
          "gradients": trainer_gradient,
          "gradient_microbatches": 0,
          "reports": (),
          "forward_counts": tuple(result["counts"] for result in forwards),
          "dp_reduction_visibility": "P66_GROUP0_NO_REDUCTION",
          "dp_axis": trainer_dp_axis,
          "p66_engine_receipt": engine_receipt,
          "p66_layerwise_profile": layerwise_profile,
          "p66_row_cotangent_summary": reverse[
              "p66_row_cotangent_summary"
          ],
          "p66_vjp_oracle": reverse["p66_vjp_oracle"],
      }
    if numeric_debug:
      _p62_emit_loss_receipt(
          loss_output=loss_output,
          contract=contract,
          mode=numeric_debug_mode,
      )
      _p62_emit_tree_receipt(
          stage="loss_cotangent",
          group=-1,
          group_count=contract.local_trajectories,
          tree={"dentropy": dentropy, "dlogps": dlogps},
          force=True,
          mode=numeric_debug_mode,
      )
      if numeric_debug_mode == "p64":
        _p62_emit_tree_receipt(
            stage="group_input_cotangent",
            group=0,
            group_count=contract.local_trajectories,
            tree={
                "dentropy": grouped_dentropy[0],
                "dlogps": grouped_dlogps[0],
            },
            ranked=True,
            force=True,
            mode=numeric_debug_mode,
        )

    reducer = None

    def reverse_reduce_group(index, spec):
      nonlocal reducer
      if rank_parallel_backward:
        with gsm8k_xprof.trace_annotation(
            "replay_forward", group_index=index
        ):
          replay = self._p32_forward_group(
              segmented, engine_leaves, spec, keep_cache_inputs=True
          )
        with gsm8k_xprof.trace_annotation(
            "model_backward", group_index=index
        ):
          reverse = self._p32_reverse_group(
              segmented,
              engine_leaves,
              spec,
              grouped_dlogps[index],
              grouped_dentropy[index],
              replay=replay,
          )
        if not bool(np.asarray(jnp.array_equal(
            reverse["replay_logps"], grouped_logps[index]
        ))):
          raise FunctionalMappingError(
              f"P59 group {index} parallel replay logprobs changed"
          )
        if numeric_debug:
          _p62_emit_tree_receipt(
              stage="engine_vjp",
              group=index,
              group_count=contract.local_trajectories,
              tree=reverse["engine_gradients"],
              ranked=numeric_debug_mode == "p64",
              mode=numeric_debug_mode,
          )
        with gsm8k_xprof.trace_annotation(
            "report_adjoint", group_index=index
        ):
          adjoint_start = time.perf_counter()
          staged_gradient = self._p59_rank_parallel_report_adjoint(
              trainer_state, reverse["engine_gradients"]
          )
          adjoint_seconds[0] += time.perf_counter() - adjoint_start
        if numeric_debug:
          _p62_emit_tree_receipt(
              stage="trainer_rank_local",
              group=index,
              group_count=contract.local_trajectories,
              tree=staged_gradient,
              ranked=True,
              mode=numeric_debug_mode,
          )
        if reducer is None:
          reducer_factory = getattr(
              self,
              "_p33_gradient_reducer_factory",
              dp_training.FixedDPRankGradientReducer,
          )
          reducer_dp_axis = getattr(self, "_p59_report_dp_axis", None)
          if reducer_dp_axis not in ("data", "dp"):
            raise FunctionalMappingError(
                "P59 report adjoint did not resolve a replicated DP axis"
            )
          if reducer_dp_axis != trainer_dp_axis:
            raise FunctionalMappingError(
                "P59 report and grouped trainer data axes differ"
            )
          reducer = reducer_factory(
              self._p59_reducer_template(trainer_state, staged_gradient),
              dp_size=contract.dp_size,
              dp_axis=reducer_dp_axis,
              require_distinct_fingerprints=False,
          )
          if not callable(getattr(reducer, "finalize_staged", None)):
            raise FunctionalMappingError(
              "P59 gradient reducer cannot consume a parallel rank table"
            )
          print(
              f"[P59.DP{contract.dp_size}] gradient_reducer_ready "
              f"dp_axis={reducer_dp_axis} dp_size={contract.dp_size} "
              "staging=parallel_table",
              flush=True,
          )
        cache_nonzero = sum(
            int(np.asarray(jnp.count_nonzero(value)))
            for value in jax.tree.leaves(
                reverse["initial_cache_cotangents"]
            )
        )
        with gsm8k_xprof.trace_annotation(
            "fixed_dp_reduce", group_index=index
        ):
          one_gradient, reduction_report = reducer.finalize_staged(
              staged_gradient
          )
        if numeric_debug:
          _p62_emit_tree_receipt(
              stage="fixed_dp_reduced",
              group=index,
              group_count=contract.local_trajectories,
              tree=one_gradient,
              mode=numeric_debug_mode,
          )
        leaves = jax.tree.leaves(one_gradient)
        reduction_finite = reduction_report.get(
            "post_reduction_all_finite"
        )
        if reduction_finite is None:
          # Compatibility for test-only reducer doubles. The production fixed
          # reducer always owns this scan before its replica comparison.
          reduction_finite = all(
              bool(np.asarray(jnp.all(jnp.isfinite(value))))
              for value in leaves
          )
        report = {
            "group": index,
            "trajectory_rows": reverse_groups[index],
            "n_real": spec["host_n_real"],
            "rank_counts": (reverse["counts"],),
            "pullback_invocations": 1,
            "gradient_finite": bool(reduction_finite),
            "gradient_nonzero": sum(
                int(np.asarray(jnp.count_nonzero(value))) for value in leaves
            ),
            "initial_cache_cotangent_nonzero": cache_nonzero,
            "dp_reduction": reduction_report,
        }
        seen = set()
        for value in jax.tree.leaves(reverse):
          if isinstance(value, jax.Array) and id(value) not in seen:
            seen.add(id(value))
            if not value.is_deleted():
              value.delete()
        return one_gradient, report

      rank_counts = []
      cache_nonzero = 0
      for rank in range(contract.dp_size):
        rank_dlogps = dp_training.isolate_dp_rank_cotangent(
            grouped_dlogps[index], rank=rank, dp_size=contract.dp_size
        )
        rank_dentropy = dp_training.isolate_dp_rank_cotangent(
            grouped_dentropy[index], rank=rank, dp_size=contract.dp_size
        )
        with gsm8k_xprof.trace_annotation(
            "replay_forward", group_index=index, rank_index=rank
        ):
          replay = self._p32_forward_group(
              segmented, engine_leaves, spec, keep_cache_inputs=True
          )
        with gsm8k_xprof.trace_annotation(
            "model_backward", group_index=index, rank_index=rank
        ):
          reverse = self._p32_reverse_group(
              segmented,
              engine_leaves,
              spec,
              rank_dlogps,
              rank_dentropy,
              replay=replay,
          )
        if not bool(np.asarray(jnp.array_equal(
            reverse["replay_logps"], grouped_logps[index]
        ))):
          raise FunctionalMappingError(
              f"P33 group {index} rank {rank} replay logprobs changed"
          )
        with gsm8k_xprof.trace_annotation(
            "report_adjoint", group_index=index, rank_index=rank
        ):
          adjoint_start = time.perf_counter()
          if batched_report:
            rank_gradient = self._batched_report_adjoint(
                trainer_state, reverse["engine_gradients"]
            )
          else:
            rank_gradient = self.map_engine_cotangents_to_trainer_state(
                trainer_state, reverse["engine_gradients"]
            )
            rank_gradient = jax.tree.map(
                lambda value: value.astype(jnp.float32), rank_gradient
            )
            if report_verify:
              fault = self._p50_rev_verify(
                  "p32 report adjoint",
                  index,
                  rank_gradient,
                  self._batched_report_adjoint(
                      trainer_state, reverse["engine_gradients"]
                  ),
              )
              if fault is not None:
                raise FunctionalMappingError(
                    f"P50 batched-report verify mismatch: {fault}"
                )
          adjoint_seconds[0] += time.perf_counter() - adjoint_start
        if serial_mesh_bridge:
          rank_gradient, bridge_dp_axis = (
              _p59_align_serial_gradient_to_trainer_state(
                  trainer_state,
                  rank_gradient,
                  "P59 DP4 serial report mesh bridge",
              )
          )
          if bridge_dp_axis != trainer_dp_axis:
            raise FunctionalMappingError(
                "P59 serial mesh bridge and grouped trainer data axes differ"
            )
        if reducer is None:
          reducer_factory = getattr(
              self,
              "_p33_gradient_reducer_factory",
              dp_training.FixedDPRankGradientReducer,
          )
          reducer = reducer_factory(
              rank_gradient,
              dp_size=contract.dp_size,
              dp_axis=trainer_dp_axis,
              # Production rewards can legitimately produce identical rank
              # gradients. In particular, RLOO gives every generation an
              # exact zero advantage when all generations for one prompt have
              # the same reward. Cadence, contribution count, fixed reduction
              # order, and post-reduction replica equality remain hard gates.
              require_distinct_fingerprints=False,
          )
          if serial_mesh_bridge:
            print(
                "[P59.DP4] serial_report_mesh_bridge_ready "
                f"dp_axis={trainer_dp_axis} placement=trainer_exact",
                flush=True,
            )
          print(
              f"[{'P34' if p34 else 'P33'}.DP{contract.dp_size}] "
              f"gradient_reducer_ready dp_axis={trainer_dp_axis} "
              f"dp_size={contract.dp_size}",
              flush=True,
          )
        if rank == 0:
          reducer.begin()
        reducer.add(rank, rank_gradient)
        rank_counts.append(reverse["counts"])
        cache_nonzero += sum(
            int(np.asarray(jnp.count_nonzero(value)))
            for value in jax.tree.leaves(
                reverse["initial_cache_cotangents"]
            )
        )
        seen = set()
        for value in jax.tree.leaves((rank_gradient, reverse)):
          if isinstance(value, jax.Array) and id(value) not in seen:
            seen.add(id(value))
            if not value.is_deleted():
              value.delete()
      with gsm8k_xprof.trace_annotation(
          "fixed_dp_reduce", group_index=index
      ):
        one_gradient, reduction_report = reducer.finalize()
      leaves = jax.tree.leaves(one_gradient)
      report = {
          "group": index,
          "trajectory_rows": reverse_groups[index],
          "n_real": spec["host_n_real"],
          "rank_counts": tuple(rank_counts),
          "pullback_invocations": contract.dp_size,
          "gradient_finite": all(
              bool(np.asarray(jnp.all(jnp.isfinite(value))))
              for value in leaves
          ),
          "gradient_nonzero": sum(
              int(np.asarray(jnp.count_nonzero(value))) for value in leaves
          ),
          "initial_cache_cotangent_nonzero": cache_nonzero,
          "dp_reduction": reduction_report,
      }
      return one_gradient, report

    trainer_gradients = None
    reports = []
    adjoint_seconds = [0.0]
    p32_reverse_start = time.perf_counter()
    p32_reverse_durations = []
    if p59_xprof_capture:
      options = jax.profiler.ProfileOptions()
      options.host_tracer_level = 1
      options.python_tracer_level = 0
      jax.profiler.start_trace(
          log_dir=p59_xprof_directory, profiler_options=options
      )
      print(
          "[P59.XPROF] phase=backward_group started update=1 groups=1",
          flush=True,
      )
    reverse_limit = (
        p64_training_capsule.reverse_group_limit(len(specs))
        if numeric_debug_mode == "p64"
        else len(specs)
    )
    reverse_specs = specs[:reverse_limit]
    if len(reverse_specs) != len(specs):
      print(
          "[P64.CAPSULE] backward_scope mode=replay groups=1/32 "
          "selected=group0 optimizer_commits=0",
          flush=True,
      )
    reverse_parent = (
        contextlib.nullcontext()
        if xprof_train_schedule is not None
        else gsm8k_xprof.trace_annotation("reverse_groups")
    )
    with reverse_parent:
      for index, spec in enumerate(reverse_specs):
        train_transaction = (
            xprof_train_schedule.transaction(index)
            if xprof_train_schedule is not None
            else contextlib.nullcontext()
        )
        with contextlib.ExitStack() as transaction_stack:
          transaction_stack.enter_context(train_transaction)
          transaction_stack.enter_context(gsm8k_xprof.trace_annotation(
              "reverse_group", group_index=index
          ))
          p32_group_start = time.perf_counter()
          one_gradient, report = reverse_reduce_group(index, spec)
          p32_reverse_durations.append(time.perf_counter() - p32_group_start)
          if p59_xprof_capture and index == 0:
            jax.block_until_ready(one_gradient)
            jax.profiler.stop_trace()
            print(
                "[P59.XPROF] phase=backward_group stopped update=1 groups=1 "
                "anchor=gradient_ready",
                flush=True,
            )
          if deterministic_repeat:
            repeated_gradient, repeated_report = reverse_reduce_group(
                index, spec
            )
            exact_flags = tuple(
                bool(np.asarray(jnp.array_equal(first, second)))
                for first, second in zip(
                    jax.tree.leaves(one_gradient),
                    jax.tree.leaves(repeated_gradient),
                    strict=True,
                )
            )
            report["deterministic_repeat_exact"] = (
                bool(exact_flags)
                and all(exact_flags)
                and report["dp_reduction"]["rank_local_fingerprints"]
                == repeated_report["dp_reduction"][
                    "rank_local_fingerprints"
                ]
            )
            report["deterministic_repeat_leaf_checks"] = len(exact_flags)
            seen = set()
            for value in jax.tree.leaves(repeated_gradient):
              if isinstance(value, jax.Array) and id(value) not in seen:
                seen.add(id(value))
                if not value.is_deleted():
                  value.delete()
            if not report["deterministic_repeat_exact"]:
              raise FunctionalMappingError(
                  f"P34 group {index} repeated gradient is not array-exact"
              )
          reports.append(report)
          if (
              os.environ.get("CANON_PERF_LOG", "1") != "0"
              and index == len(reverse_specs) - 1
          ):
            print(
                "[PERF] stage=p32_vag_reverse seconds=%.3f groups=%d"
                " mean=%.3f max=%.3f adjoint=%.3f"
                % (
                    time.perf_counter() - p32_reverse_start,
                    len(p32_reverse_durations),
                    sum(p32_reverse_durations) / len(p32_reverse_durations),
                    max(p32_reverse_durations),
                    adjoint_seconds[0],
                ),
                flush=True,
            )
          print(
              f"[{'P34' if p34 else 'P33'}.DP{contract.dp_size}] "
              "reverse_group_done "
              f"group={index + 1}/{contract.local_trajectories} "
              f"rows={reverse_groups[index]} "
              "rank_contributions="
              f"{report['dp_reduction']['rank_contributions']} "
              f"pullback_invocations={report['pullback_invocations']} "
              "unique_rank_fingerprints="
              f"{report['dp_reduction']['rank_local_fingerprint_unique_count']}/"
              f"{contract.dp_size} "
              f"reduction_rounds={report['dp_reduction']['reduction_rounds']} "
              "replicas_exact="
              f"{int(report['dp_reduction']['post_reduction_replicas_exact'])} "
              f"gradient_nonzero={report['gradient_nonzero']} "
              "repeat_exact="
              f"{int(report.get('deterministic_repeat_exact', False))}",
              flush=True,
          )
          with gsm8k_xprof.trace_annotation(
              "gradient_accumulate",
              group_index=index,
              micro_step=index,
              is_last_accumulate=int(index == len(specs) - 1),
          ):
            if gradient_microbatch_sink is None:
              trainer_gradients = (
                  one_gradient
                  if trainer_gradients is None
                  else jax.tree.map(
                      lambda total, value: total + value,
                      trainer_gradients,
                      one_gradient,
                  )
              )
            else:
              # The accumulator averages its streamed groups. Each reduced
              # group is an unreduced sum over one trajectory from every DP
              # rank, so multiplying by the group count makes the final value
              # exactly ``scale * sum(all trajectory gradients)``.
              gradient_microbatch_sink(
                  index,
                  one_gradient,
                  scale
                  * jnp.asarray(contract.local_trajectories, scale.dtype),
              )
          if gradient_microbatch_sink is not None:
            # The sink blocks after donating the persistent accumulator. The
            # reduced per-group gradient is now dead; explicit deletion
            # prevents Python reference lifetime from retaining a second
            # model-sized tree.
            seen = set()
            for value in jax.tree.leaves(one_gradient):
              if isinstance(value, jax.Array) and id(value) not in seen:
                seen.add(id(value))
                if not value.is_deleted():
                  value.delete()
    if gradient_microbatch_sink is None:
      if trainer_gradients is None:
        raise FunctionalMappingError("P32 D3b0 emitted no grouped gradient")
      trainer_gradients = jax.tree.map(
          lambda value: value * scale, trainer_gradients
      )
    return {
        "loss_output": loss_output,
        "loss": loss_output.primary_loss.compute(),
        "per_token_logps": per_token_logps,
        "token_entropy": token_entropy,
        "gradients": trainer_gradients,
        "gradient_microbatches": (
            0 if gradient_microbatch_sink is None else len(reports)
        ),
        "reports": tuple(reports),
        "forward_counts": tuple(result["counts"] for result in forwards),
        "dp_reduction_visibility": "EXPLICIT_FIXED_TREE",
        "dp_axis": trainer_dp_axis,
        "rank_local_gradient_fingerprints": tuple(
            report["dp_reduction"]["rank_local_fingerprints"]
            for report in reports
        ),
        "rank_local_gradient_fingerprint_unique_counts": tuple(
            report["dp_reduction"]["rank_local_fingerprint_unique_count"]
            for report in reports
        ),
        "replica_equality": all(
            report["dp_reduction"]["post_reduction_replicas_exact"]
            for report in reports
        ),
        "dp_reduction_transactions": sum(
            report["dp_reduction"]["reduction_transactions"]
            for report in reports
        ),
        "dp_reduction_rounds_per_transaction": (
            dp_training.fixed_dp_collective_count(contract.dp_size)
        ),
        "dp_rank_pullbacks_per_transaction": contract.dp_size,
        "dp_pullback_invocations_per_transaction": (
            1 if rank_parallel_backward else contract.dp_size
        ),
        "gradient_deterministic_repeat": (
            bool(reports)
            and all(
                report.get("deterministic_repeat_exact") is True
                for report in reports
            )
            if deterministic_repeat
            else None
        ),
    }


  def segmented_grpo_value_and_grad(
      self,
      *,
      trainer_state,
      train_example,
      algo_config,
      pad_id,
      eos_id,
      gradient_microbatch_sink=None,
      gradient_pair_sink=None,
  ):
    """Evaluates and reverses the complete GRPO loss without an outer JIT."""
    del eos_id
    _P28SegmentedEngineForward._reject_outer_transform(  # pylint: disable=protected-access
        trainer_state, train_example
    )
    if getattr(self, "_data_size", 1) != 1:
      raise FunctionalMappingError(
          "the data1 segmented reverse cannot run on a DP mesh; use the "
          "explicit DP segmented transaction"
      )
    if os.environ.get("CANON_P28_SEGMENTED_TRAIN", "") != "1":
      raise FunctionalMappingError(
          "P28 complete loss requires CANON_P28_SEGMENTED_TRAIN=1"
      )
    g5c_only = os.environ.get("CANON_P28_G5C_ONLY", "") == "1"
    g6_update = os.environ.get("CANON_P28_G6_UPDATE", "") == "1"
    if g5c_only == g6_update:
      raise FunctionalMappingError(
          "P28 complete loss requires exactly one of "
          "CANON_P28_G5C_ONLY=1 or CANON_P28_G6_UPDATE=1"
      )
    num_sinks = sum(
        sink is not None
        for sink in (gradient_microbatch_sink, gradient_pair_sink)
    )
    if (g5c_only and num_sinks != 0) or (g6_update and num_sinks != 1):
      raise FunctionalMappingError(
          "P28 G5c must retain one aggregate gradient; G6 must provide "
          "exactly one streaming gradient sink"
      )
    if getattr(train_example, "segment_ids", None) is not None:
      raise FunctionalMappingError(
          "P28 G5c admits unpacked eight-trajectory input only"
      )
    prompts = jnp.asarray(train_example.prompt_ids)
    completions = jnp.asarray(train_example.completion_ids)
    prompt_masks = jnp.asarray(train_example.prompt_mask, dtype=jnp.bool_)
    completion_masks = jnp.asarray(
        train_example.completion_mask, dtype=jnp.bool_
    )
    completion_valid_value = getattr(
        train_example, "completion_valid_mask", None
    )
    completion_valid_masks = (
        completion_masks
        if completion_valid_value is None
        else jnp.asarray(completion_valid_value, dtype=jnp.bool_)
    )
    if prompts.shape != prompt_masks.shape:
      raise FunctionalMappingError("P28 G5c prompt batch/mask mismatch")
    if completions.shape != completion_masks.shape:
      raise FunctionalMappingError("P28 G5c completion batch/mask mismatch")
    if completions.shape != completion_valid_masks.shape:
      raise FunctionalMappingError(
          "P28 G5c completion batch/valid-mask mismatch"
      )
    if bool(np.asarray(jax.device_get(jnp.any(
        completion_masks & ~completion_valid_masks
    )))):
      raise FunctionalMappingError(
          "P28 G5c action mask is not a subset of completion validity"
      )
    if prompts.shape[0] != completions.shape[0]:
      raise FunctionalMappingError("P28 G5c prompt/completion batch mismatch")
    expected_trajectories, expected_widths = _segmented_loss_geometry(
        os.environ
    )
    if int(prompts.shape[0]) != expected_trajectories:
      raise FunctionalMappingError(
          "segmented loss trajectory contract changed: "
          f"expected {expected_trajectories}, got {prompts.shape[0]}"
      )
    if (int(prompts.shape[1]), int(completions.shape[1])) != expected_widths:
      raise FunctionalMappingError(
          "segmented loss prompt/response contract changed: expected "
          f"{expected_widths[0]}/{expected_widths[1]}, got "
          f"{prompts.shape[1]}/{completions.shape[1]}"
      )
    gradient_microbatches = expected_trajectories // 2

    model_config = self._runner.model_config
    mapped = map_trainer_state_to_engine_leaves(
        trainer_state=trainer_state,
        engine_state_contract=self._engine_state_contract,
        key_mappings=self._key_mappings,
        transpose_keys=self._transpose_keys,
        key_mapping_hook_fns=self._hook_fns,
        num_kv_heads=model_config.get_total_num_kv_heads(),
        head_dim=model_config.get_head_size(),
        tp_size=self._tp_size,
    )
    engine_leaves = tuple(mapped.leaves)
    reuse_segmented = (
        os.environ.get("CANON_P30_REUSE_SEGMENTED_ENGINE", "") == "1"
    )
    release_captured_state = (
        os.environ.get("CANON_P30_RELEASE_CAPTURED_STATE", "") == "1"
    )
    if release_captured_state and not reuse_segmented:
      raise FunctionalMappingError(
          "CANON_P30_RELEASE_CAPTURED_STATE requires segmented-engine reuse"
      )
    segmented = getattr(self, "_p30_segmented_engine", None)
    if not reuse_segmented or segmented is None:
      segmented = build_p28_segmented_engine_forward(self._runner)
      if reuse_segmented:
        self._p30_segmented_engine = segmented
        print(
            "[P30.G2] REUSE_SEGMENTED_ENGINE on weights=explicit",
            flush=True,
        )
        if release_captured_state:
          released_bytes = segmented.release_captured_state()
          print(
              "[P30.G2] RELEASE_CAPTURED_STATE on "
              f"bytes={released_bytes} current_weights=explicit",
              flush=True,
          )
    specs = tuple(
        self._p28_sequence_spec(
            prompts[index],
            completions[index],
            prompt_masks[index],
            completion_valid_masks[index],
            algo_config.temperature,
        )
        for index in range(expected_trajectories)
    )
    vag_forward_start = time.perf_counter()
    forwards = tuple(
        self._p28_forward_sequence(
            segmented, engine_leaves, spec, keep_cache_inputs=False
        )
        for spec in specs
    )
    per_token_logps = jnp.stack(
        tuple(result["logps"] for result in forwards), axis=0
    ).astype(jnp.float32)
    token_entropy = jnp.stack(
        tuple(result["entropy"] for result in forwards), axis=0
    ).astype(jnp.float32)
    if os.environ.get("CANON_PERF_LOG", "1") != "0":
      print(
          "[PERF] stage=vag_forward seconds=%.3f trajectories=%d"
          % (time.perf_counter() - vag_forward_start, expected_trajectories),
          flush=True,
      )

    from tunix.rl import algo_core  # pylint: disable=g-import-not-at-top

    def unreduced_loss(logps, entropy):
      return algo_core.grpo_loss_from_precomputed_logps(
          logps, entropy, train_example, algo_config
      ).primary_loss.unreduced_sum

    unreduced_value, loss_pullback = jax.vjp(
        unreduced_loss, per_token_logps, token_entropy
    )
    dlogps, dentropy = loss_pullback(
        jnp.ones_like(unreduced_value)
    )
    loss_output = algo_core.grpo_loss_from_precomputed_logps(
        per_token_logps, token_entropy, train_example, algo_config
    )
    scale = loss_output.primary_loss.compute_scale()
    trainer_gradients = None
    pair_gradient = None
    emitted_microbatches = 0
    reports = []
    vag_reverse_start = time.perf_counter()
    vag_reverse_durations = []
    vag_pre_sum = 0.0
    vag_issue_sum = 0.0
    vag_drain_sum = 0.0
    vag_report_sum = 0.0
    _ISSUE_ANATOMY.update(prep=0.0, call=0.0, n=0)
    # CANON_BATCHED_EVIDENCE=1 keeps every evidence VALUE identical (same
    # per-leaf predicates, same python-side reductions) but collects them
    # through one device_get per trajectory instead of two per leaf.  The
    # per-leaf vectors are fetched raw and reduced on host with int64, so
    # counts match the original python-int sums exactly.
    batched_evidence = (
        os.environ.get("CANON_BATCHED_EVIDENCE", "") == "1"
    )
    report_mode = os.environ.get("CANON_P28_BATCHED_REPORT", "")
    if report_mode not in ("", "0", "1", "verify"):
      raise FunctionalMappingError(
          "CANON_P28_BATCHED_REPORT must be unset/0/1/verify, "
          f"got {report_mode!r}"
      )
    batched_report = report_mode == "1"
    report_verify = report_mode == "verify"

    def _stacked_finite(leaves):
      if not leaves:
        return jnp.ones((0,), jnp.bool_)
      return jnp.stack([jnp.all(jnp.isfinite(value)) for value in leaves])

    def _stacked_nonzero(leaves):
      if not leaves:
        return jnp.zeros((0,), jnp.int32)
      return jnp.stack([jnp.count_nonzero(value) for value in leaves])
    for index, spec in enumerate(specs):
      vag_iter_start = time.perf_counter()
      loss_cotangent_leaves = (dlogps[index], dentropy[index])
      if batched_evidence:
        loss_cotangent_pending = (
            _stacked_finite(loss_cotangent_leaves),
            _stacked_nonzero(loss_cotangent_leaves),
        )
        loss_cotangent_finite = None
        loss_cotangent_nonzero = None
      else:
        loss_cotangent_finite = all(
            bool(np.asarray(jnp.all(jnp.isfinite(value))))
            for value in loss_cotangent_leaves
        )
        loss_cotangent_nonzero = sum(
            int(np.asarray(jnp.count_nonzero(value)))
            for value in loss_cotangent_leaves
        )
      vag_pre_done = time.perf_counter()
      reverse = self._p28_reverse_sequence(
          segmented,
          engine_leaves,
          spec,
          dlogps[index],
          dentropy[index],
      )
      vag_call_done = time.perf_counter()
      if not bool(np.asarray(jnp.array_equal(
          reverse["replay_logps"], per_token_logps[index]
      ))):
        raise FunctionalMappingError(
            f"P28 G5c sequence {index} replay logprobs changed"
        )
      # The replay-identity np.asarray above is the first existing barrier
      # after the reverse dispatch chain: call-return minus it separates
      # host issue time from device drain without adding any new sync.
      vag_drain_done = time.perf_counter()
      vag_pre_sum += vag_pre_done - vag_iter_start
      vag_issue_sum += vag_call_done - vag_pre_done
      vag_drain_sum += vag_drain_done - vag_call_done
      if batched_report:
        one_trainer_gradient = self._batched_report_adjoint(
            trainer_state, reverse["engine_gradients"]
        )
      else:
        one_trainer_gradient = self.map_engine_cotangents_to_trainer_state(
            trainer_state, reverse["engine_gradients"]
        )
        one_trainer_gradient = jax.tree.map(
            lambda value: value.astype(jnp.float32), one_trainer_gradient
        )
        if report_verify:
          fault = self._p50_rev_verify(
              "report adjoint",
              index,
              one_trainer_gradient,
              self._batched_report_adjoint(
                  trainer_state, reverse["engine_gradients"]
              ),
          )
          if fault is not None:
            raise FunctionalMappingError(
                f"P50 batched-report verify mismatch: {fault}"
            )
      trainer_leaves = jax.tree.leaves(one_trainer_gradient)
      report_pending = None
      if batched_evidence and batched_report:
        trainer_finite = None
        trainer_nonzero = None
      elif batched_evidence:
        trainer_pending = (
            _stacked_finite(trainer_leaves),
            _stacked_nonzero(trainer_leaves),
        )
        trainer_finite = None
        trainer_nonzero = None
      else:
        trainer_finite = all(
            bool(np.asarray(jnp.all(jnp.isfinite(value))))
            for value in trainer_leaves
        )
        trainer_nonzero = sum(
            int(np.asarray(jnp.count_nonzero(value)))
            for value in trainer_leaves
        )
      engine_groups = {
          "embed": segmented._embed_full_indices,  # pylint: disable=protected-access
          "norm": segmented._norm_full_indices,  # pylint: disable=protected-access
          "head": segmented._head_full_indices,  # pylint: disable=protected-access
      }
      engine_groups.update({
          f"layer_{layer_index}": indices
          for layer_index, indices in enumerate(
              segmented._local_layer_full_indices  # pylint: disable=protected-access
          )
      })
      group_health = {}
      group_pending = {}
      if batched_evidence and batched_report:
        report_pending = self._batched_report_evidence(
            engine_groups,
            trainer_leaves,
            reverse["engine_gradients"],
            jax.tree.leaves(reverse["initial_cache_cotangents"]),
        )
      else:
        if batched_evidence and report_verify:
          unified = self._batched_report_evidence(
              engine_groups,
              trainer_leaves,
              reverse["engine_gradients"],
              jax.tree.leaves(reverse["initial_cache_cotangents"]),
          )
          eager_trainer = (
              _stacked_finite(trainer_leaves),
              _stacked_nonzero(trainer_leaves),
          )
          eager_groups = {
              label: (
                  _stacked_finite(
                      tuple(reverse["engine_gradients"][i] for i in indices)
                  ),
                  _stacked_nonzero(
                      tuple(reverse["engine_gradients"][i] for i in indices)
                  ),
              )
              for label, indices in engine_groups.items()
          }
          eager_cache_leaves = jax.tree.leaves(
              reverse["initial_cache_cotangents"]
          )
          eager_cache = (
              _stacked_finite(eager_cache_leaves),
              _stacked_nonzero(eager_cache_leaves),
          )
          fault = self._p50_rev_verify(
              "report evidence",
              index,
              {"trainer": eager_trainer, "groups": eager_groups,
               "cache": eager_cache},
              {"trainer": unified["trainer"], "groups": unified["groups"],
               "cache": unified["cache"]},
          )
          if fault is not None:
            raise FunctionalMappingError(
                f"P50 batched-report verify mismatch: {fault}"
            )
        for label, indices in engine_groups.items():
          leaves = tuple(reverse["engine_gradients"][i] for i in indices)
          if batched_evidence:
            group_pending[label] = (
                _stacked_finite(leaves), _stacked_nonzero(leaves)
            )
          else:
            group_health[label] = {
                "finite": all(
                    bool(np.asarray(jnp.all(jnp.isfinite(value))))
                    for value in leaves
                ),
                "nonzero": sum(
                    int(np.asarray(jnp.count_nonzero(value)))
                    for value in leaves
                ),
            }
      cache_leaves = jax.tree.leaves(
          reverse["initial_cache_cotangents"]
      )
      if batched_evidence:
        if report_pending is not None:
          fetched = jax.device_get({
              "loss": loss_cotangent_pending,
              "trainer": report_pending["trainer"],
              "groups": report_pending["groups"],
              "cache": report_pending["cache"],
          })
        else:
          fetched = jax.device_get({
              "loss": loss_cotangent_pending,
              "trainer": trainer_pending,
              "groups": group_pending,
              "cache": (
                  _stacked_finite(cache_leaves),
                  _stacked_nonzero(cache_leaves),
              ),
          })
        loss_cotangent_finite = bool(np.all(fetched["loss"][0]))
        loss_cotangent_nonzero = int(
            np.sum(fetched["loss"][1], dtype=np.int64)
        )
        trainer_finite = bool(np.all(fetched["trainer"][0]))
        trainer_nonzero = int(np.sum(fetched["trainer"][1], dtype=np.int64))
        group_health = {
            label: {
                "finite": bool(np.all(pair[0])),
                "nonzero": int(np.sum(pair[1], dtype=np.int64)),
            }
            for label, pair in fetched["groups"].items()
        }
        cache_finite = bool(np.all(fetched["cache"][0]))
        cache_nonzero = int(np.sum(fetched["cache"][1], dtype=np.int64))
      else:
        cache_finite = all(
            bool(np.asarray(jnp.all(jnp.isfinite(value))))
            for value in cache_leaves
        )
        cache_nonzero = sum(
            int(np.asarray(jnp.count_nonzero(value)))
            for value in cache_leaves
        )
      if gradient_microbatch_sink is None and gradient_pair_sink is None:
        if trainer_gradients is None:
          trainer_gradients = one_trainer_gradient
        elif batched_report:
          trainer_gradients = self._batched_report_add(
              trainer_gradients, one_trainer_gradient
          )
        else:
          jitted_total = (
              self._batched_report_add(
                  trainer_gradients, one_trainer_gradient
              )
              if report_verify
              else None
          )
          trainer_gradients = jax.tree.map(
              lambda total, value: total + value,
              trainer_gradients,
              one_trainer_gradient,
          )
          if jitted_total is not None:
            fault = self._p50_rev_verify(
                "report accumulate", index, trainer_gradients, jitted_total
            )
            if fault is not None:
              raise FunctionalMappingError(
                  f"P50 batched-report verify mismatch: {fault}"
              )
      else:
        if pair_gradient is None:
          pair_gradient = one_trainer_gradient
        else:
          # The stock accumulator averages all micro-step gradients. Each
          # two-trajectory contribution is multiplied by that count so that
          # mean(N * scale * pair_sum) == scale * full-batch sum.
          if gradient_pair_sink is not None:
            gradient_pair_sink(
                emitted_microbatches,
                pair_gradient,
                one_trainer_gradient,
                scale * jnp.asarray(float(gradient_microbatches), scale.dtype),
            )
          else:
            if batched_report:
              pair_gradient = self._batched_report_add(
                  pair_gradient, one_trainer_gradient
              )
            else:
              jitted_pair = (
                  self._batched_report_add(
                      pair_gradient, one_trainer_gradient
                  )
                  if report_verify
                  else None
              )
              pair_gradient = jax.tree.map(
                  lambda total, value: total + value,
                  pair_gradient,
                  one_trainer_gradient,
              )
              if jitted_pair is not None:
                fault = self._p50_rev_verify(
                    "report pair-accumulate", index, pair_gradient, jitted_pair
                )
                if fault is not None:
                  raise FunctionalMappingError(
                      f"P50 batched-report verify mismatch: {fault}"
                  )
            if os.environ.get("CANON_FUSED_TREE_OPS", "") == "1":
              # scale must travel as an argument: a closure would embed the
              # array as a trace constant and force a retrace per value.
              micro_gradient = fused_micro_scale(
                  pair_gradient, scale, gradient_microbatches
              )
            else:
              micro_gradient = jax.tree.map(
                  lambda value: value * scale * jnp.asarray(
                      float(gradient_microbatches), value.dtype
                  ),
                  pair_gradient,
              )
            gradient_microbatch_sink(emitted_microbatches, micro_gradient)
          emitted_microbatches += 1
          pair_gradient = None
      reports.append({
          "trajectory": index,
          "boundary": (
              "final" if index == expected_trajectories - 1 else "pending"
          ),
          "counts": reverse["counts"],
          "loss_cotangent": {
              "finite": loss_cotangent_finite,
              "nonzero": loss_cotangent_nonzero,
          },
          "trainer_gradient": {
              "finite": trainer_finite,
              "nonzero": trainer_nonzero,
              "mapping_adjoint_leaves": len(trainer_leaves),
          },
          "engine_groups": group_health,
          "initial_cache_cotangent": {
              "finite": cache_finite,
              "nonzero": cache_nonzero,
          },
      })
      # Everything after the drain barrier is the per-round report tail
      # (adjoint mapping, evidence prep, accumulation tree ops). Timing it
      # splits the [PERF] report residual into report_ops vs report_other,
      # so the P2.c overlap work aims at a measured component instead of a
      # residual.
      vag_report_sum += time.perf_counter() - vag_drain_done
      vag_reverse_durations.append(time.perf_counter() - vag_iter_start)
    if (
        os.environ.get("CANON_PERF_LOG", "1") != "0"
        and vag_reverse_durations
    ):
      vag_total = time.perf_counter() - vag_reverse_start
      print(
          "[PERF] stage=vag_reverse seconds=%.3f trajectories=%d"
          " mean=%.3f max=%.3f pre=%.3f issue=%.3f drain=%.3f report=%.3f"
          " report_ops=%.3f report_other=%.3f"
          " blk_prep=%.3f blk_call=%.3f blk_n=%d"
          % (
              vag_total,
              len(vag_reverse_durations),
              sum(vag_reverse_durations) / len(vag_reverse_durations),
              max(vag_reverse_durations),
              vag_pre_sum,
              vag_issue_sum,
              vag_drain_sum,
              vag_total - vag_pre_sum - vag_issue_sum - vag_drain_sum,
              vag_report_sum,
              vag_total
              - vag_pre_sum
              - vag_issue_sum
              - vag_drain_sum
              - vag_report_sum,
              _ISSUE_ANATOMY["prep"],
              _ISSUE_ANATOMY["call"],
              _ISSUE_ANATOMY["n"],
          ),
          flush=True,
      )
    if gradient_microbatch_sink is None:
      trainer_gradients = jax.tree.map(
          lambda value: value * scale, trainer_gradients
      )
    elif (
        pair_gradient is not None
        or emitted_microbatches != gradient_microbatches
    ):
      raise FunctionalMappingError(
          "segmented update emitted the wrong number of complete "
          "two-trajectory gradients: "
          f"{emitted_microbatches} != {gradient_microbatches}"
      )
    return {
        "loss_output": loss_output,
        "loss": loss_output.primary_loss.compute(),
        "per_token_logps": per_token_logps,
        "token_entropy": token_entropy,
        "gradients": trainer_gradients,
        "gradient_microbatches": emitted_microbatches,
        "reports": tuple(reports),
        "forward_counts": tuple(result["counts"] for result in forwards),
        "dp_reduction_visibility": "SINGLE_REPLICA_IDENTITY",
        "dp_axis": self._dp_axis,
        "replica_equality": True,
        "dp_reduction_transactions": 0,
        "dp_reduction_rounds_per_transaction": 0,
        "dp_rank_pullbacks_per_transaction": 1,
        "dp_pullback_invocations_per_transaction": 1,
    }

  def run_p28_segmented_forward_gate(self, lengths=(128, 160, 256)):
    """Compares whole-model and host-segmented real Qwen3 forwards.

    This is a forward-only release probe.  It intentionally consumes the
    live engine state rather than trainer-mapped leaves: the existing mapping
    contract already attests those leaves, while G3 isolates only the new JIT
    boundary.  No result from this method is a backward/update claim.
    """
    segmented = build_p28_segmented_engine_forward(self._runner)
    engine_leaves = tuple(self._runner.state_leaves)
    block_tables = jnp.zeros(
        (self._max_num_reqs, self._blocks_per_req), jnp.int32
    )
    block_tables = block_tables.at[0].set(
        jnp.arange(self._blocks_per_req, dtype=jnp.int32)
    )
    block_tables_flat = self._engine_array(block_tables.reshape(-1))
    request_distribution = self._engine_array(
        jnp.asarray((0, 0, 1), jnp.int32)
    )

    def digest(value):
      host = np.asarray(value)
      return hashlib.sha256(host.view(np.uint8).tobytes()).hexdigest()

    def differing_bytes(left, right):
      left = np.asarray(left).view(np.uint8)
      right = np.asarray(right).view(np.uint8)
      if left.shape != right.shape:
        raise FunctionalMappingError(
            f"P28 comparison shape mismatch: {left.shape} != {right.shape}"
        )
      return int(np.count_nonzero(left != right)), int(left.size)

    def run_whole(input_ids, positions, metadata):
      with self._set_forward_context(None, self._runner.vllm_config):
        next_caches, hidden, _, _ = self._runner.model_fn(
            engine_leaves,
            self._fresh_caches(),
            input_ids,
            metadata,
            None,
            positions,
            self._static_kv_indices,
            None,
            None,
            bool(self._runner.is_first_rank),
            bool(self._runner.is_last_rank),
        )
      logits = self._runner.compute_logits_fn(
          engine_leaves, hidden, None
      ).astype(jnp.float32)
      return next_caches, hidden, logits

    def run_segmented(input_ids, metadata):
      with self._set_forward_context(None, self._runner.vllm_config):
        next_caches, hidden = segmented.run(
            engine_leaves,
            self._fresh_caches(),
            input_ids,
            metadata,
        )
      logits = self._runner.compute_logits_fn(
          engine_leaves, hidden, None
      ).astype(jnp.float32)
      return next_caches, hidden, logits

    results = []
    for length in lengths:
      length = int(length)
      if length < 2 or length > self._bucket:
        raise FunctionalMappingError(
            f"P28 probe length must be in [2, {self._bucket}], got {length}"
        )
      rows = jnp.arange(self._bucket, dtype=jnp.int32)
      input_ids = jnp.where(
          rows < length,
          1 + (rows % min(self._vocab_size - 1, 1024)),
          0,
      )
      positions = jnp.where(rows < length, rows, 0)
      query_start = jnp.zeros((self._max_num_reqs + 1,), jnp.int32)
      query_start = query_start.at[1:].set(length)
      seq_lens = jnp.zeros((self._max_num_reqs,), jnp.int32)
      seq_lens = seq_lens.at[0].set(length)
      input_ids = self._engine_array(input_ids)
      positions = self._engine_array(positions)
      metadata = self._metadata_cls(
          input_positions=positions,
          block_tables=block_tables_flat,
          seq_lens=self._engine_array(seq_lens),
          query_start_loc=self._engine_array(query_start),
          request_distribution=request_distribution,
      )
      metadata.padded_num_reqs = self._max_num_reqs

      _, whole_hidden_1, whole_logits_1 = run_whole(
          input_ids, positions, metadata
      )
      _, segmented_hidden_1, segmented_logits_1 = run_segmented(
          input_ids, metadata
      )
      _, whole_hidden_2, whole_logits_2 = run_whole(
          input_ids, positions, metadata
      )
      _, segmented_hidden_2, segmented_logits_2 = run_segmented(
          input_ids, metadata
      )
      valid_rows = slice(0, length - 1)
      comparisons = {
          "hidden_whole_segmented": differing_bytes(
              whole_hidden_1, segmented_hidden_1
          ),
          "hidden_whole_repeat": differing_bytes(
              whole_hidden_1, whole_hidden_2
          ),
          "hidden_segmented_repeat": differing_bytes(
              segmented_hidden_1, segmented_hidden_2
          ),
          "logits_whole_segmented": differing_bytes(
              whole_logits_1[valid_rows], segmented_logits_1[valid_rows]
          ),
          "logits_whole_repeat": differing_bytes(
              whole_logits_1[valid_rows], whole_logits_2[valid_rows]
          ),
          "logits_segmented_repeat": differing_bytes(
              segmented_logits_1[valid_rows],
              segmented_logits_2[valid_rows],
          ),
      }
      hashes = {
          "whole_hidden": digest(whole_hidden_1),
          "segmented_hidden": digest(segmented_hidden_1),
          "whole_logits": digest(whole_logits_1[valid_rows]),
          "segmented_logits": digest(segmented_logits_1[valid_rows]),
      }
      print(
          "[P28.G3] "
          f"length={length} action_rows={length - 1} "
          f"comparisons={comparisons} hashes={hashes}",
          flush=True,
      )
      if any(changed != 0 for changed, _ in comparisons.values()):
        raise FunctionalMappingError(
            f"P28 G3 segmented forward differs at length {length}: "
            f"{comparisons}"
        )
      results.append((length, comparisons, hashes))
    print(
        "[P28.G3] PASS "
        f"contract={dataclasses.asdict(segmented.contract)} "
        f"completed={len(results)}/{len(tuple(lengths))}",
        flush=True,
    )
    return tuple(results)

  def run_p28_block_vjp_gate(
      self, *, layer_index=0, prefix_length=128, chunk_length=32
  ):
    """Runs the preregistered P28.G4 one-real-layer cache-consuming VJP."""
    if os.environ.get("CANON_P28_SEGMENTED_VJP", "") != "1":
      raise FunctionalMappingError(
          "P28 block VJP requires CANON_P28_SEGMENTED_VJP=1"
      )
    segmented = build_p28_segmented_engine_forward(self._runner)
    engine_leaves = tuple(self._runner.state_leaves)
    layer_index = int(layer_index)
    prefix_length = int(prefix_length)
    chunk_length = int(chunk_length)
    extension = os.environ.get("CANON_P28_G4_EXTENSION", "") == "1"
    if extension:
      if layer_index not in (1, 17, 35):
        raise FunctionalMappingError(
            "P28 G4 extension is frozen at layers 1/17/35"
        )
    elif layer_index != 0:
      raise FunctionalMappingError("P28 G4 baseline is frozen at layer=0")
    if (prefix_length, chunk_length) != (128, 32):
      raise FunctionalMappingError(
          "P28 G4 geometry is frozen at prefix=128,chunk=32"
      )
    raw_cap = os.environ.get("CANON_P28_G4_BLOCK_CAP_SECONDS", "").strip()
    block_cap_seconds = 600.0 if not raw_cap else float(raw_cap)
    expected_cap = 21.342724 if extension else 600.0
    if abs(block_cap_seconds - expected_cap) > 1e-6:
      raise FunctionalMappingError(
          "P28 G4 block cap changed: "
          f"{block_cap_seconds} != {expected_cap}"
      )

    def memory_snapshot():
      snapshots = []
      for device in jax.local_devices():
        stats = {}
        try:
          stats = device.memory_stats() or {}
        except Exception:
          pass
        snapshots.append({
            "device": int(device.id),
            "bytes_in_use": stats.get("bytes_in_use"),
            "peak_bytes_in_use": stats.get("peak_bytes_in_use"),
            "bytes_limit": stats.get("bytes_limit"),
        })
      return tuple(snapshots)

    block_tables = jnp.zeros(
        (self._max_num_reqs, self._blocks_per_req), jnp.int32
    ).at[0].set(jnp.arange(self._blocks_per_req, dtype=jnp.int32))
    block_tables_flat = self._engine_array(block_tables.reshape(-1))
    request_distribution = self._engine_array(
        jnp.asarray((0, 0, 1), jnp.int32)
    )

    def make_inputs(start, q_len, kv_len):
      rows = jnp.arange(self._bucket, dtype=jnp.int32)
      ids = jnp.where(rows < q_len, 1 + ((start + rows) % 1024), 0)
      positions = jnp.where(rows < q_len, start + rows, 0)
      query_start = jnp.zeros((self._max_num_reqs + 1,), jnp.int32)
      query_start = query_start.at[1:].set(q_len)
      seq_lens = jnp.zeros((self._max_num_reqs,), jnp.int32)
      seq_lens = seq_lens.at[0].set(kv_len)
      ids = self._engine_array(ids)
      metadata = self._metadata_cls(
          input_positions=self._engine_array(positions),
          block_tables=block_tables_flat,
          seq_lens=self._engine_array(seq_lens),
          query_start_loc=self._engine_array(query_start),
          request_distribution=request_distribution,
      )
      metadata.padded_num_reqs = self._max_num_reqs
      return ids, metadata

    def block_all(value):
      for leaf in jax.tree_util.tree_leaves(value):
        if hasattr(leaf, "block_until_ready"):
          leaf.block_until_ready()
      return value

    def differing(left, right):
      equal = bool(np.asarray(jnp.array_equal(left, right)))
      # G4 needs only a strict bitwise predicate.  Do not host-copy a multi-GB
      # cache merely to count bytes after the hard gate is already red.
      return 0 if equal else 1

    def tree_digest(tree, *, cache=False):
      digest = hashlib.sha256()
      leaves = jax.tree_util.tree_leaves(tree)
      finite = True
      nonzero = 0
      for leaf_index, leaf in enumerate(leaves):
        finite = finite and bool(np.asarray(jnp.all(jnp.isfinite(leaf))))
        nonzero += int(np.asarray(jnp.count_nonzero(leaf)))
        selected = leaf[:1] if cache and leaf.ndim > 0 else leaf
        host = np.asarray(selected)
        digest.update(str((leaf_index, host.shape, host.dtype)).encode())
        digest.update(host.view(np.uint8).tobytes())
      return finite, nonzero, digest.hexdigest()

    prefix_ids, prefix_metadata = make_inputs(0, prefix_length, prefix_length)
    chunk_ids, chunk_metadata = make_inputs(
        prefix_length, chunk_length, prefix_length + chunk_length
    )
    fresh_cache = self._fresh_caches()[layer_index]
    hbm_before = memory_snapshot()
    with self._set_forward_context(None, self._runner.vllm_config):
      prefix_hidden = segmented._embed_fn(engine_leaves, prefix_ids)
      prefix_reference, prefix_isolated = segmented.run_block_forward(
          layer_index,
          engine_leaves,
          fresh_cache,
          prefix_hidden,
          prefix_metadata,
      )
      block_all((prefix_reference, prefix_isolated))
      chunk_hidden = segmented._embed_fn(engine_leaves, chunk_ids)
      start = time.perf_counter()
      first = segmented.run_block_vjp(
          layer_index,
          engine_leaves,
          prefix_isolated[0],
          chunk_hidden,
          chunk_metadata,
      )
      block_all(first)
      first_seconds = time.perf_counter() - start
      start = time.perf_counter()
      second = segmented.run_block_vjp(
          layer_index,
          engine_leaves,
          prefix_isolated[0],
          chunk_hidden,
          chunk_metadata,
      )
      block_all(second)
      repeat_seconds = time.perf_counter() - start
    hbm_after = memory_snapshot()

    primal_diffs = {
        "prefix_cache": differing(prefix_reference[0], prefix_isolated[0]),
        "prefix_hidden": differing(prefix_reference[1], prefix_isolated[1]),
        "chunk_cache": differing(first["reference"][0], first["isolated"][0]),
        "chunk_hidden": differing(first["reference"][1], first["isolated"][1]),
    }
    grad_names = ("parameters", "cache", "hidden")
    summaries = {
        name: tree_digest(grad, cache=(name == "cache"))
        for name, grad in zip(grad_names, first["gradients"])
    }
    repeat_summaries = {
        name: tree_digest(grad, cache=(name == "cache"))
        for name, grad in zip(grad_names, second["gradients"])
    }
    repeat_exact = all(
        bool(np.asarray(jnp.array_equal(a, b)))
        for a, b in zip(
            jax.tree_util.tree_leaves(first["gradients"]),
            jax.tree_util.tree_leaves(second["gradients"]),
        )
    )
    if any(primal_diffs.values()):
      raise FunctionalMappingError(f"P28 G4 primal mismatch: {primal_diffs}")
    if not repeat_exact or summaries != repeat_summaries:
      raise FunctionalMappingError("P28 G4 gradient repeat mismatch")
    if any((not finite) or nonzero == 0 for finite, nonzero, _ in summaries.values()):
      raise FunctionalMappingError(f"P28 G4 dead/nonfinite gradient: {summaries}")
    if first_seconds > block_cap_seconds:
      raise FunctionalMappingError(
          "P28 G4 one-block cap exceeded: "
          f"{first_seconds:.6f}s > {block_cap_seconds:.6f}s"
      )
    result = {
        "contract": dataclasses.asdict(first["contract"]),
        "prefix_length": prefix_length,
        "chunk_length": chunk_length,
        "primal_diffs": primal_diffs,
        "gradient_summaries": summaries,
        "repeat_exact": repeat_exact,
        "first_seconds": first_seconds,
        "repeat_seconds": repeat_seconds,
        "cache_bytes": int(fresh_cache.size * fresh_cache.dtype.itemsize),
        "hidden_bytes": int(chunk_hidden.size * chunk_hidden.dtype.itemsize),
        "hbm_before": hbm_before,
        "hbm_after": hbm_after,
    }
    print(f"[P28.G4] PASS {result}", flush=True)
    return result

  def run_p28_full_chain_gate(self, *, prefix_length=128, chunk_length=32):
    """Runs the preregistered P28.G5b 36-layer staged pullback gate."""
    if os.environ.get("CANON_P28_G5_ONLY", "") != "1":
      raise FunctionalMappingError(
          "P28 full chain requires CANON_P28_G5_ONLY=1"
      )
    if os.environ.get("CANON_P28_SEGMENTED_PULLBACK", "") != "1":
      raise FunctionalMappingError(
          "P28 full chain requires CANON_P28_SEGMENTED_PULLBACK=1"
      )
    prefix_length = int(prefix_length)
    chunk_length = int(chunk_length)
    if (prefix_length, chunk_length) != (128, 32):
      raise FunctionalMappingError(
          "P28 G5b geometry is frozen at prefix=128,chunk=32"
      )
    first_cap = float(os.environ.get("CANON_P28_G5_FIRST_CAP_SECONDS", "0"))
    repeat_cap = float(os.environ.get("CANON_P28_G5_REPEAT_CAP_SECONDS", "0"))
    total_cap = float(os.environ.get("CANON_P28_G5_TOTAL_CAP_SECONDS", "0"))
    if (first_cap, repeat_cap, total_cap) != (600.0, 300.0, 900.0):
      raise FunctionalMappingError(
          "P28 G5b caps changed: "
          f"{(first_cap, repeat_cap, total_cap)} != (600, 300, 900)"
      )

    segmented = build_p28_segmented_engine_forward(self._runner)
    layer_count = len(segmented._local_layer_fns)
    if layer_count != 36:
      raise FunctionalMappingError(
          f"P28 G5b requires 36 real decoder layers, got {layer_count}"
      )
    engine_leaves = tuple(self._runner.state_leaves)
    fresh_caches = tuple(self._fresh_caches())
    if len(fresh_caches) != layer_count:
      raise FunctionalMappingError(
          f"P28 G5b cache count changed: {len(fresh_caches)} != {layer_count}"
      )

    def memory_snapshot():
      snapshots = []
      for device in jax.local_devices():
        stats = {}
        try:
          stats = device.memory_stats() or {}
        except Exception:
          pass
        snapshots.append({
            "device": int(device.id),
            "bytes_in_use": stats.get("bytes_in_use"),
            "peak_bytes_in_use": stats.get("peak_bytes_in_use"),
            "bytes_limit": stats.get("bytes_limit"),
        })
      return tuple(snapshots)

    block_tables = jnp.zeros(
        (self._max_num_reqs, self._blocks_per_req), jnp.int32
    ).at[0].set(jnp.arange(self._blocks_per_req, dtype=jnp.int32))
    block_tables_flat = self._engine_array(block_tables.reshape(-1))
    request_distribution = self._engine_array(
        jnp.asarray((0, 0, 1), jnp.int32)
    )

    def make_inputs(start, q_len, kv_len):
      rows = jnp.arange(self._bucket, dtype=jnp.int32)
      ids = jnp.where(rows < q_len, 1 + ((start + rows) % 1024), 0)
      positions = jnp.where(rows < q_len, start + rows, 0)
      query_start = jnp.zeros((self._max_num_reqs + 1,), jnp.int32)
      query_start = query_start.at[1:].set(q_len)
      seq_lens = jnp.zeros((self._max_num_reqs,), jnp.int32)
      seq_lens = seq_lens.at[0].set(kv_len)
      metadata = self._metadata_cls(
          input_positions=self._engine_array(positions),
          block_tables=block_tables_flat,
          seq_lens=self._engine_array(seq_lens),
          query_start_loc=self._engine_array(query_start),
          request_distribution=request_distribution,
      )
      metadata.padded_num_reqs = self._max_num_reqs
      return self._engine_array(ids), metadata

    def block_all(tree):
      for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "block_until_ready"):
          leaf.block_until_ready()
      return tree

    def tree_summary(tree):
      finite = True
      nonzero = 0
      for leaf in jax.tree_util.tree_leaves(tree):
        finite = finite and bool(np.asarray(jnp.all(jnp.isfinite(leaf))))
        nonzero += int(np.asarray(jnp.count_nonzero(leaf)))
      return finite, nonzero

    def tree_exact(left, right):
      left_leaves = jax.tree_util.tree_leaves(left)
      right_leaves = jax.tree_util.tree_leaves(right)
      if len(left_leaves) != len(right_leaves):
        return False
      return all(
          bool(np.asarray(jnp.array_equal(a, b)))
          for a, b in zip(left_leaves, right_leaves, strict=True)
      )

    prefix_ids, prefix_metadata = make_inputs(0, prefix_length, prefix_length)
    chunk_ids, chunk_metadata = make_inputs(
        prefix_length, chunk_length, prefix_length + chunk_length
    )
    prefix_hidden = segmented._embed_fn(engine_leaves, prefix_ids)
    chunk_hidden = segmented._embed_fn(engine_leaves, chunk_ids)

    def run_chain():
      counts = {
          "prefix_forward": 0,
          "chunk_forward": 0,
          "chunk_pullback": 0,
          "prefix_pullback": 0,
      }
      prefix_tape = []
      prefix_caches = []
      hidden = prefix_hidden
      for index, layer_fn in enumerate(segmented._local_layer_fns):
        prefix_tape.append((fresh_caches[index], hidden))
        cache_out, hidden = layer_fn(
            segmented._local_layer_leaves[index],
            fresh_caches[index],
            hidden,
            prefix_metadata,
        )
        prefix_caches.append(cache_out)
        counts["prefix_forward"] += 1
      prefix_final_hidden = hidden

      chunk_tape = []
      hidden = chunk_hidden
      for index, layer_fn in enumerate(segmented._local_layer_fns):
        chunk_tape.append((prefix_caches[index], hidden))
        _, hidden = layer_fn(
            segmented._local_layer_leaves[index],
            prefix_caches[index],
            hidden,
            chunk_metadata,
        )
        counts["chunk_forward"] += 1

      row = jnp.arange(hidden.shape[0], dtype=jnp.int32)
      feature = jnp.arange(hidden.shape[1], dtype=jnp.int32)
      row_seed = jnp.where(
          row < chunk_length,
          ((row % 17).astype(jnp.float32) + 1.0) / 17.0,
          0.0,
      )
      feature_seed = ((feature % 23).astype(jnp.float32) + 1.0) / 23.0
      dhidden = (
          row_seed[:, None] * feature_seed[None, :]
          / jnp.asarray(chunk_length * hidden.shape[1], jnp.float32)
      ).astype(hidden.dtype)

      chunk_parameter_grads = [None] * layer_count
      prefix_cache_cotangents = [None] * layer_count
      for index in reversed(range(layer_count)):
        cache_in, hidden_in = chunk_tape[index]
        gradients, dcache, dhidden = segmented.run_block_pullback(
            index,
            cache_in,
            hidden_in,
            chunk_metadata,
            jax.tree.map(jnp.zeros_like, cache_in),
            dhidden,
        )
        chunk_parameter_grads[index] = gradients
        prefix_cache_cotangents[index] = dcache
        counts["chunk_pullback"] += 1
      chunk_input_grad = dhidden

      dhidden = jnp.zeros_like(prefix_final_hidden)
      combined_parameter_grads = [None] * layer_count
      for index in reversed(range(layer_count)):
        cache_in, hidden_in = prefix_tape[index]
        gradients, _, dhidden = segmented.run_block_pullback(
            index,
            cache_in,
            hidden_in,
            prefix_metadata,
            prefix_cache_cotangents[index],
            dhidden,
        )
        combined_parameter_grads[index] = jax.tree.map(
            lambda prefix, chunk: prefix + chunk,
            gradients,
            chunk_parameter_grads[index],
        )
        counts["prefix_pullback"] += 1
      prefix_input_grad = dhidden
      return (
          tuple(combined_parameter_grads),
          tuple(prefix_cache_cotangents),
          prefix_input_grad,
          chunk_input_grad,
          counts,
      )

    hbm_before = memory_snapshot()
    method_start = time.perf_counter()
    with self._set_forward_context(None, self._runner.vllm_config):
      start = time.perf_counter()
      first = block_all(run_chain())
      first_seconds = time.perf_counter() - start
      hbm_after_first = memory_snapshot()
      start = time.perf_counter()
      second = block_all(run_chain())
      repeat_seconds = time.perf_counter() - start
    total_seconds = time.perf_counter() - method_start
    hbm_after_repeat = memory_snapshot()

    first_values = first[:4]
    second_values = second[:4]
    counts = first[4]
    parameter_summaries = tuple(
        tree_summary(gradient) for gradient in first[0]
    )
    cache_summaries = tuple(
        tree_summary(cotangent) for cotangent in first[1]
    )
    hidden_summaries = {
        "prefix_input": tree_summary(first[2]),
        "chunk_input": tree_summary(first[3]),
    }
    repeat_exact = tree_exact(first_values, second_values)
    expected_counts = {
        "prefix_forward": 36,
        "chunk_forward": 36,
        "chunk_pullback": 36,
        "prefix_pullback": 36,
    }
    if counts != expected_counts or second[4] != expected_counts:
      raise FunctionalMappingError(
          f"P28 G5b chain counts changed: {counts}, {second[4]}"
      )
    if len(parameter_summaries) != 36 or any(
        (not finite) or nonzero == 0
        for finite, nonzero in parameter_summaries
    ):
      raise FunctionalMappingError(
          f"P28 G5b parameter gradient red: {parameter_summaries}"
      )
    if len(cache_summaries) != 36 or any(
        (not finite) or nonzero == 0 for finite, nonzero in cache_summaries
    ):
      raise FunctionalMappingError(
          f"P28 G5b cache cotangent red: {cache_summaries}"
      )
    if any(
        (not finite) or nonzero == 0
        for finite, nonzero in hidden_summaries.values()
    ):
      raise FunctionalMappingError(
          f"P28 G5b hidden cotangent red: {hidden_summaries}"
      )
    if not repeat_exact:
      raise FunctionalMappingError("P28 G5b repeat is not array-exact")
    if first_seconds > first_cap or repeat_seconds > repeat_cap:
      raise FunctionalMappingError(
          "P28 G5b chain cap exceeded: "
          f"first={first_seconds:.6f}/{first_cap:.6f} "
          f"repeat={repeat_seconds:.6f}/{repeat_cap:.6f}"
      )
    if total_seconds > total_cap:
      raise FunctionalMappingError(
          f"P28 G5b total cap exceeded: {total_seconds:.6f}/{total_cap:.6f}"
      )
    for snapshot_name, snapshot in (
        ("before", hbm_before),
        ("after_first", hbm_after_first),
        ("after_repeat", hbm_after_repeat),
    ):
      if len(snapshot) != 4 or any(
          item.get("peak_bytes_in_use") is None
          or item.get("bytes_limit") is None
          or item["peak_bytes_in_use"] >= item["bytes_limit"]
          for item in snapshot
      ):
        raise FunctionalMappingError(
            f"P28 G5b HBM telemetry red at {snapshot_name}: {snapshot}"
        )
    result = {
        "layers": layer_count,
        "prefix_length": prefix_length,
        "chunk_length": chunk_length,
        "counts": counts,
        "parameter_summaries": parameter_summaries,
        "cache_summaries": cache_summaries,
        "hidden_summaries": hidden_summaries,
        "repeat_exact": repeat_exact,
        "first_seconds": first_seconds,
        "repeat_seconds": repeat_seconds,
        "total_seconds": total_seconds,
        "hbm_before": hbm_before,
        "hbm_after_first": hbm_after_first,
        "hbm_after_repeat": hbm_after_repeat,
    }
    print(f"[P28.G5B] PASS {result}", flush=True)
    return result

  def _sequence_group(
      self,
      engine_leaves,
      prompt,
      completion,
      prompt_valid,
      completion_valid,
      pad_id,
      temperature,
      *,
      return_diagnostics=False,
  ):
    """Runs one independent packed sequence on every engine data rank.

    The group axis is ordered by data rank. Each rank receives one local-M
    sequence, local cache pages, and local request metadata. The global engine
    call retains the admitted M contract and never mixes attention contexts.
    """
    if prompt.ndim != 2 or prompt.shape[0] != self._data_size:
      raise FunctionalMappingError(
          "sequence group must contain one row per engine data rank: "
          f"{prompt.shape} vs data={self._data_size}"
      )
    full = jnp.concatenate((prompt, completion), axis=1)
    valid = jnp.concatenate((prompt_valid, completion_valid), axis=1)
    n_real = jnp.sum(valid, axis=1, dtype=jnp.int32)
    num_chunks = (
        full.shape[1] + self._sequence_bucket - 1
    ) // self._sequence_bucket
    padded_width = num_chunks * self._sequence_bucket

    def pack_row(full_row, valid_row, count):
      order = jnp.nonzero(valid_row, size=padded_width, fill_value=0)[0]
      active = jnp.arange(padded_width, dtype=jnp.int32) < count
      return jnp.where(
          active, full_row[order], jnp.asarray(0, full_row.dtype)
      )

    packed_ids = jax.vmap(pack_row)(full, valid, n_real)
    next_ids = jnp.concatenate(
        (
            packed_ids[:, 1:],
            jnp.zeros((self._data_size, 1), packed_ids.dtype),
        ),
        axis=1,
    )

    prompt_len = jnp.sum(prompt_valid, axis=1, dtype=jnp.int32)
    completion_ordinal = (
        jnp.cumsum(completion_valid, axis=1, dtype=jnp.int32) - 1
    )
    token_positions = prompt_len[:, None] + completion_ordinal
    source_rows = jnp.clip(token_positions - 1, 0, padded_width - 1)

    sampling_metadata = self._sampling_metadata_cls(
        temperature=self._engine_array(
            jnp.full((self._bucket,), temperature, jnp.float32)
        ),
        top_k=self._engine_array(
            jnp.full((self._bucket,), -1, jnp.int32)
        ),
        top_p=self._engine_array(
            jnp.ones((self._bucket,), jnp.float32)
        ),
        do_sampling=True,
        logprobs=True,
    )
    caches = self._fresh_caches()
    chunk_logps = []
    chunk_entropies = []
    if return_diagnostics:
      completion_width = completion.shape[1]
      raw_rows = jnp.zeros(
          (self._data_size, completion_width, self._vocab_size),
          jnp.float32,
      )
      processed_rows = jnp.zeros_like(raw_rows)
      diagnostic_target_ids = jnp.zeros(
          (self._data_size, completion_width), jnp.int32
      )
      raw_targets = jnp.zeros(
          (self._data_size, completion_width), jnp.float32
      )
      processed_targets = jnp.zeros_like(raw_targets)

    print(
        "[PATHTRACE] CANON_ADAPTER_DP_FIXED_M_CHUNKS "
        f"data={self._data_size} static_width={full.shape[1]} "
        f"chunks={num_chunks} global_M={self._bucket} "
        f"local_M={self._sequence_bucket}",
        flush=True,
    )
    for chunk_index in range(num_chunks):
      chunk_start = chunk_index * self._sequence_bucket
      rows = jnp.arange(self._sequence_bucket, dtype=jnp.int32)
      q_len = jnp.clip(
          n_real - chunk_start, 0, self._sequence_bucket
      )
      active = q_len > 0
      kv_len = jnp.where(
          active, jnp.minimum(n_real, chunk_start + self._sequence_bucket), 0
      )
      chunk_ids_group = packed_ids[
          :, chunk_start : chunk_start + self._sequence_bucket
      ]
      chunk_targets_group = next_ids[
          :, chunk_start : chunk_start + self._sequence_bucket
      ]
      positions_group = jnp.where(
          rows[None, :] < q_len[:, None], chunk_start + rows[None, :], 0
      )
      (
          block_tables,
          seq_lens,
          query_start,
          request_distribution,
      ) = _canonical_dp_attention_metadata_arrays(
          data_size=self._data_size,
          max_num_reqs=self._max_num_reqs,
          blocks_per_req=self._blocks_per_req,
          q_len=q_len,
          kv_len=kv_len,
      )
      chunk_ids = self._engine_array(chunk_ids_group.reshape(-1))
      chunk_targets = self._engine_array(chunk_targets_group.reshape(-1))
      positions = self._engine_array(positions_group.reshape(-1))
      metadata = self._metadata_cls(
          input_positions=positions,
          block_tables=self._engine_array(block_tables),
          seq_lens=self._engine_array(seq_lens),
          query_start_loc=self._engine_array(query_start),
          request_distribution=self._engine_array(request_distribution),
      )
      metadata.padded_num_reqs = self._max_num_reqs

      def run_nonempty(active_caches):
        with self._set_forward_context(None, self._runner.vllm_config):
          next_caches, hidden, _, _ = self._runner.model_fn(
              engine_leaves,
              active_caches,
              chunk_ids,
              metadata,
              None,
              positions,
              self._static_kv_indices,
              None,
              None,
              bool(self._runner.is_first_rank),
              bool(self._runner.is_last_rank),
          )
        logits = self._runner.compute_logits_fn(
            engine_leaves, hidden, None
        ).astype(jnp.float32)
        if logits.shape != (self._bucket, self._vocab_size):
          raise FunctionalMappingError(
              "canonical logits shape does not match the admitted fixed-M "
              f"contract: {logits.shape} != "
              f"{(self._bucket, self._vocab_size)}"
          )
        _, processed_logits = self._sample(
            jax.random.PRNGKey(0),
            self._runner.mesh,
            logits,
            sampling_metadata,
        )
        target_logprobs = self._processed_target_logprobs(
            processed_logits, chunk_targets
        ).reshape(self._data_size, self._sequence_bucket)
        normalized = jax.nn.log_softmax(processed_logits, axis=-1)
        probabilities = jnp.exp(normalized)
        entropy_rows = -jnp.sum(
            jnp.where(probabilities > 0, probabilities * normalized, 0.0),
          axis=-1,
        ).reshape(self._data_size, self._sequence_bucket)
        if not return_diagnostics:
          return next_caches, (target_logprobs, entropy_rows)

        local_rows = jnp.clip(
            source_rows - chunk_start, 0, self._sequence_bucket - 1
        )
        belongs = (
            completion_valid
            & (source_rows >= chunk_start)
            & (source_rows < chunk_start + self._sequence_bucket)
        )
        row_mask = belongs[..., None]
        logits_group = logits.reshape(
            self._data_size, self._sequence_bucket, self._vocab_size
        )
        processed_group = processed_logits.reshape(
            self._data_size, self._sequence_bucket, self._vocab_size
        )
        selected_raw = jax.vmap(lambda values, index: values[index])(
            logits_group, local_rows
        )
        selected_processed = jax.vmap(
            lambda values, index: values[index]
        )(processed_group, local_rows)
        selected_target_ids = jax.vmap(
            lambda values, index: values[index]
        )(chunk_targets_group, local_rows)
        selected_raw_targets = jnp.take_along_axis(
            selected_raw, selected_target_ids[..., None], axis=-1
        )[..., 0]
        selected_processed_targets = jnp.take_along_axis(
            selected_processed, selected_target_ids[..., None], axis=-1
        )[..., 0]
        return next_caches, (
            target_logprobs,
            entropy_rows,
            jnp.where(row_mask, selected_raw, 0.0),
            jnp.where(row_mask, selected_processed, 0.0),
            jnp.where(belongs, selected_target_ids, 0),
            jnp.where(belongs, selected_raw_targets, 0.0),
            jnp.where(belongs, selected_processed_targets, 0.0),
        )

      def skip_empty(inactive_caches):
        zero_rows = jnp.zeros(
            (self._data_size, self._sequence_bucket), jnp.float32
        )
        if not return_diagnostics:
          return inactive_caches, (zero_rows, zero_rows)
        zero_action_rows = jnp.zeros(
            (
                self._data_size,
                completion.shape[1],
                self._vocab_size,
            ),
            jnp.float32,
        )
        zero_actions = jnp.zeros(
            (self._data_size, completion.shape[1]), jnp.float32
        )
        return inactive_caches, (
            zero_rows,
            zero_rows,
            zero_action_rows,
            zero_action_rows,
            jnp.zeros(
                (self._data_size, completion.shape[1]), jnp.int32
            ),
            zero_actions,
            zero_actions,
        )

      caches, chunk_output = jax.lax.cond(
          jnp.any(active), run_nonempty, skip_empty, caches
      )
      chunk_logps.append(chunk_output[0])
      chunk_entropies.append(chunk_output[1])
      if return_diagnostics:
        raw_rows = raw_rows + chunk_output[2]
        processed_rows = processed_rows + chunk_output[3]
        diagnostic_target_ids = diagnostic_target_ids + chunk_output[4]
        raw_targets = raw_targets + chunk_output[5]
        processed_targets = processed_targets + chunk_output[6]

    target_logprobs = jnp.concatenate(chunk_logps, axis=1)
    entropy_rows = jnp.concatenate(chunk_entropies, axis=1)
    logps = jnp.take_along_axis(target_logprobs, source_rows, axis=1)
    entropy = jnp.take_along_axis(entropy_rows, source_rows, axis=1)
    zeros = jnp.zeros_like(logps)
    masked_logps = jnp.where(completion_valid, logps, zeros)
    masked_entropy = jnp.where(completion_valid, entropy, zeros)
    if not return_diagnostics:
      return masked_logps, masked_entropy

    row_mask = completion_valid[..., None]
    diagnostics = {
        "target_ids": jnp.where(
            completion_valid, diagnostic_target_ids, 0
        ),
        "raw_rows": jnp.where(row_mask, raw_rows, 0.0),
        "processed_rows": jnp.where(row_mask, processed_rows, 0.0),
        "raw_targets": jnp.where(completion_valid, raw_targets, 0.0),
        "processed_targets": jnp.where(
            completion_valid, processed_targets, 0.0
        ),
        "implied_log_normalizers": jnp.where(
            completion_valid, processed_targets - logps, 0.0
        ),
    }
    return masked_logps, masked_entropy, diagnostics


  def _one_sequence(
      self,
      engine_leaves,
      prompt,
      completion,
      prompt_valid,
      completion_valid,
      pad_id,
      temperature,
      *,
      return_diagnostics=False,
  ):
    """Runs the retained data-size-one adapter contract."""
    if self._data_size != 1:
      raise FunctionalMappingError(
          "one-sequence execution is invalid on a multi-rank data mesh"
      )
    result = self._sequence_group(
        engine_leaves,
        prompt[None, :],
        completion[None, :],
        prompt_valid[None, :],
        completion_valid[None, :],
        pad_id,
        temperature,
        return_diagnostics=return_diagnostics,
    )
    if not return_diagnostics:
      return result[0][0], result[1][0]
    logps, entropy, diagnostics = result
    return (
        logps[0],
        entropy[0],
        jax.tree.map(lambda value: value[0], diagnostics),
    )

  def compute_per_token_logps(
      self,
      *,
      graphdef,
      state,
      prompt_tokens,
      completion_tokens,
      pad_id,
      eos_id,
      images=None,
      stop_gradient=True,
      return_entropy=False,
      segment_ids=None,
      segment_positions=None,
      temperature=1.0,
      chunk_size=0,
      prompt_mask=None,
      completion_mask=None,
  ):
    """Runs the real engine program with trainer weights and fresh caches."""
    del graphdef, eos_id, segment_positions
    if images is not None:
      raise FunctionalMappingError("canonical Qwen3 adapter is text-only")
    if segment_ids is not None:
      raise FunctionalMappingError(
          "canonical Qwen3 adapter does not yet admit sequence packing"
      )
    if chunk_size:
      raise FunctionalMappingError(
          "canonical engine adapter owns its fixed-M chunking; chunk_size must be 0"
      )
    if prompt_tokens.ndim != 2 or completion_tokens.ndim != 2:
      raise FunctionalMappingError("prompt/completion tokens must be rank 2")
    if prompt_tokens.shape[0] != completion_tokens.shape[0]:
      raise FunctionalMappingError("prompt/completion batch sizes differ")
    if prompt_mask is None:
      prompt_mask = prompt_tokens != pad_id
    else:
      prompt_mask = jnp.asarray(prompt_mask, dtype=jnp.bool_)
    if completion_mask is None:
      completion_mask = completion_tokens != pad_id
    else:
      completion_mask = jnp.asarray(completion_mask, dtype=jnp.bool_)
    if prompt_mask.shape != prompt_tokens.shape:
      raise FunctionalMappingError("prompt mask shape differs from prompt tokens")
    if completion_mask.shape != completion_tokens.shape:
      raise FunctionalMappingError(
          "completion mask shape differs from completion tokens"
      )
    if (
        prompt_tokens.shape[1] + completion_tokens.shape[1]
        > self._max_model_len
    ):
      raise FunctionalMappingError(
          "one sequence exceeds the live engine max-model-length contract: "
          f"{prompt_tokens.shape[1]}+{completion_tokens.shape[1]}"
      )

    model_config = self._runner.model_config
    mapped = map_trainer_state_to_engine_leaves(
        trainer_state=state,
        engine_state_contract=self._engine_state_contract,
        key_mappings=self._key_mappings,
        transpose_keys=self._transpose_keys,
        key_mapping_hook_fns=self._hook_fns,
        num_kv_heads=model_config.get_total_num_kv_heads(),
        head_dim=model_config.get_head_size(),
        tp_size=self._tp_size,
    )

    if self._data_size == 1:
      replicated_sharding = jax.sharding.NamedSharding(
          self._runner.mesh, jax.sharding.PartitionSpec(None, None)
      )
      prompt_tokens = _safe_sharding_constraint(
          prompt_tokens, replicated_sharding
      )
      completion_tokens = _safe_sharding_constraint(
          completion_tokens, replicated_sharding
      )
      prompt_mask = _safe_sharding_constraint(prompt_mask, replicated_sharding)
      completion_mask = _safe_sharding_constraint(
          completion_mask, replicated_sharding
      )

      def body(rows):
        prompt, completion, prompt_valid, completion_valid = rows
        return self._one_sequence(
            mapped.leaves,
            prompt,
            completion,
            prompt_valid,
            completion_valid,
            pad_id,
            temperature,
        )

      logps, entropy = jax.lax.map(
          body,
          (prompt_tokens, completion_tokens, prompt_mask, completion_mask),
      )
    else:
      grouped_inputs = jax.tree.map(
          self._group_batch_rows,
          (prompt_tokens, completion_tokens, prompt_mask, completion_mask),
      )

      def grouped_body(rows):
        prompt, completion, prompt_valid, completion_valid = rows
        return self._sequence_group(
            mapped.leaves,
            prompt,
            completion,
            prompt_valid,
            completion_valid,
            pad_id,
            temperature,
        )

      grouped_logps, grouped_entropy = jax.lax.map(
          grouped_body, grouped_inputs
      )
      logps = self._ungroup_batch_rows(grouped_logps)
      entropy = self._ungroup_batch_rows(grouped_entropy)
      output_sharding = jax.sharding.NamedSharding(
          self._runner.mesh, jax.sharding.PartitionSpec("data", None)
      )
      logps = _safe_sharding_constraint(logps, output_sharding)
      entropy = _safe_sharding_constraint(entropy, output_sharding)
    if stop_gradient:
      logps = jax.lax.stop_gradient(logps)
      entropy = jax.lax.stop_gradient(entropy)
    if return_entropy:
      return logps, entropy
    return logps

  def compute_per_token_diagnostics(
      self,
      *,
      graphdef,
      state,
      prompt_tokens,
      completion_tokens,
      pad_id,
      eos_id,
      images=None,
      segment_ids=None,
      segment_positions=None,
      temperature=1.0,
      chunk_size=0,
      prompt_mask=None,
      completion_mask=None,
  ):
    """Diagnostic-only forward that exports already-live action-logit rows."""
    del graphdef, eos_id, segment_positions
    if os.environ.get("CANON_L3_A3_DIAG", "") != "1":
      raise FunctionalMappingError(
          "compute_per_token_diagnostics requires CANON_L3_A3_DIAG=1"
      )
    if images is not None:
      raise FunctionalMappingError("canonical Qwen3 adapter is text-only")
    if segment_ids is not None:
      raise FunctionalMappingError(
          "canonical Qwen3 adapter does not yet admit sequence packing"
      )
    if chunk_size:
      raise FunctionalMappingError(
          "canonical engine adapter owns its fixed-M chunking; chunk_size must be 0"
      )
    if prompt_tokens.ndim != 2 or completion_tokens.ndim != 2:
      raise FunctionalMappingError("prompt/completion tokens must be rank 2")
    if prompt_tokens.shape[0] != completion_tokens.shape[0]:
      raise FunctionalMappingError("prompt/completion batch sizes differ")
    if prompt_mask is None:
      prompt_mask = prompt_tokens != pad_id
    else:
      prompt_mask = jnp.asarray(prompt_mask, dtype=jnp.bool_)
    if completion_mask is None:
      completion_mask = completion_tokens != pad_id
    else:
      completion_mask = jnp.asarray(completion_mask, dtype=jnp.bool_)
    if prompt_mask.shape != prompt_tokens.shape:
      raise FunctionalMappingError("prompt mask shape differs from prompt tokens")
    if completion_mask.shape != completion_tokens.shape:
      raise FunctionalMappingError(
          "completion mask shape differs from completion tokens"
      )
    if (
        prompt_tokens.shape[1] + completion_tokens.shape[1]
        > self._max_model_len
    ):
      raise FunctionalMappingError(
          "one sequence exceeds the live engine max-model-length contract: "
          f"{prompt_tokens.shape[1]}+{completion_tokens.shape[1]}"
      )

    model_config = self._runner.model_config
    mapped = map_trainer_state_to_engine_leaves(
        trainer_state=state,
        engine_state_contract=self._engine_state_contract,
        key_mappings=self._key_mappings,
        transpose_keys=self._transpose_keys,
        key_mapping_hook_fns=self._hook_fns,
        num_kv_heads=model_config.get_total_num_kv_heads(),
        head_dim=model_config.get_head_size(),
        tp_size=self._tp_size,
    )

    if self._data_size == 1:

      def body(rows):
        prompt, completion, prompt_valid, completion_valid = rows
        return self._one_sequence(
            mapped.leaves,
            prompt,
            completion,
            prompt_valid,
            completion_valid,
            pad_id,
            temperature,
            return_diagnostics=True,
        )

      return jax.lax.map(
          body,
          (prompt_tokens, completion_tokens, prompt_mask, completion_mask),
      )

    grouped_inputs = jax.tree.map(
        self._group_batch_rows,
        (prompt_tokens, completion_tokens, prompt_mask, completion_mask),
    )

    def grouped_body(rows):
      prompt, completion, prompt_valid, completion_valid = rows
      return self._sequence_group(
          mapped.leaves,
          prompt,
          completion,
          prompt_valid,
          completion_valid,
          pad_id,
          temperature,
          return_diagnostics=True,
      )

    grouped = jax.lax.map(grouped_body, grouped_inputs)
    return jax.tree.map(self._ungroup_batch_rows, grouped)


def _flat_path(path: Sequence[Any]) -> str:
  return ".".join(str(part) for part in path)


def _mapping_pairs(*, trainer_state, engine_state_contract, key_mappings):
  target_flat = list(engine_state_contract.flat_state())
  source_to_target_contract = generate_utils.build_flat_dict(
      target_flat, dict(key_mappings)
  )
  source_paths = tuple(
      _flat_path(path)
      for path, _ in trainer_state.flat_state()
      if "rng" not in _flat_path(path)
  )
  missing_source = sorted(
      path for path in source_paths if path not in source_to_target_contract
  )
  if missing_source:
    raise FunctionalMappingError(
        "trainer leaves missing canonical engine mappings: "
        f"{missing_source}"
    )
  unrolled = generate_utils._unroll_scanned_layers(  # pylint: disable=protected-access
      trainer_state, source_to_target_contract
  )
  return target_flat, unrolled


def _transform_value(
    value,
    *,
    source_path,
    target_param,
    transpose_keys,
    key_mapping_hook_fns,
    rollout_engine,
    shape_kwargs,
):
  target_value = getattr(target_param, "value", target_param)
  value = generate_utils._apply_transpose(  # pylint: disable=protected-access
      value, source_path, transpose_keys, rollout_engine
  )
  if key_mapping_hook_fns and source_path in key_mapping_hook_fns:
    value = key_mapping_hook_fns[source_path](value)
  value = generate_utils._align_shape(  # pylint: disable=protected-access
      value,
      target_value.shape,
      source_path,
      rollout_engine,
      **shape_kwargs,
  )
  return generate_utils._apply_dtype_cast(  # pylint: disable=protected-access
      value, target_value.dtype, source_path
  )


def inspect_trainer_state_to_engine_contract(
    *,
    trainer_state: Any,
    engine_state_contract: Any,
    key_mappings: Mapping[str, tuple[str, tuple[str | None, ...]]],
    transpose_keys: Mapping[str, tuple[int, ...]] | None = None,
    key_mapping_hook_fns: Mapping[str, Any] | None = None,
    rollout_engine: str = "vllm_jax",
    **shape_kwargs: Any,
) -> MappingManifest:
  """Checks the real mapping inventory without allocating mapped weights."""
  target_flat, unrolled = _mapping_pairs(
      trainer_state=trainer_state,
      engine_state_contract=engine_state_contract,
      key_mappings=key_mappings,
  )
  entries_by_target: dict[str, MappingManifestEntry] = {}
  for (source_path, target_path), (source_value, target_param) in unrolled.items():
    if target_path in entries_by_target:
      raise FunctionalMappingError(
          f"canonical engine target written more than once: {target_path}"
      )
    target_value = getattr(target_param, "value", target_param)
    source_spec = jax.ShapeDtypeStruct(source_value.shape, source_value.dtype)
    mapped_spec = jax.eval_shape(
        lambda value: _transform_value(
            value,
            source_path=source_path,
            target_param=target_param,
            transpose_keys=transpose_keys,
            key_mapping_hook_fns=key_mapping_hook_fns,
            rollout_engine=rollout_engine,
            shape_kwargs=shape_kwargs,
        ),
        source_spec,
    )
    entry = MappingManifestEntry(
        source_path=source_path,
        target_path=target_path,
        source_shape=tuple(source_value.shape),
        source_dtype=str(source_value.dtype),
        target_shape=tuple(target_value.shape),
        target_dtype=str(target_value.dtype),
        mapped_shape=tuple(mapped_spec.shape),
        mapped_dtype=str(mapped_spec.dtype),
    )
    if (
        entry.mapped_shape != entry.target_shape
        or entry.mapped_dtype != entry.target_dtype
    ):
      raise FunctionalMappingError(
          "abstract mapped leaf does not match engine contract: "
          f"{entry}"
      )
    entries_by_target[target_path] = entry

  target_paths = tuple(_flat_path(path) for path, _ in target_flat)
  missing_target = sorted(
      path for path in target_paths if path not in entries_by_target
  )
  if missing_target:
    raise FunctionalMappingError(
        "canonical engine mapping is not target-complete: "
        f"missing={missing_target}"
    )
  return MappingManifest(
      entries=tuple(entries_by_target[path] for path in target_paths),
      target_paths=target_paths,
  )


def inspect_live_engine_contract(
    *, sampler: Any, trainer_state: Any
) -> LiveEngineContract:
  """Fail-closed A1b/A2 inspection without materializing mapped weights."""
  if os.environ.get("CANON_RPA_VJP2", "") != "1":
    raise FunctionalMappingError(
        "canonical engine contract requires CANON_RPA_VJP2=1"
    )
  try:
    runner = sampler._model_runner  # pylint: disable=protected-access
  except (AttributeError, RuntimeError) as exc:
    raise FunctionalMappingError("rollout has no live model runner") from exc

  required = (
      "state",
      "state_leaves",
      "model_fn",
      "compute_logits_fn",
      "mesh",
      "kv_caches",
      "layer_name_to_kvcache_index",
      "is_first_rank",
      "is_last_rank",
  )
  missing = [name for name in required if not hasattr(runner, name)]
  if missing:
    raise FunctionalMappingError(
        f"live tpu_inference runner is missing contract attributes: {missing}"
    )
  if not callable(runner.model_fn) or not callable(runner.compute_logits_fn):
    raise FunctionalMappingError("engine model_fn/compute_logits_fn is not callable")
  if not isinstance(runner.mesh, jax.sharding.Mesh):
    raise FunctionalMappingError("engine runner mesh is not a jax.sharding.Mesh")
  if not runner.kv_caches:
    raise FunctionalMappingError("engine runner exposes no paged kv caches")

  model_config = getattr(runner, "model_config", None)
  if model_config is None:
    raise FunctionalMappingError("engine runner exposes no model_config")
  manifest = inspect_trainer_state_to_engine_contract(
      trainer_state=trainer_state,
      engine_state_contract=runner.state,
      key_mappings=getattr(sampler, "to_hf_key_mappings", None) or {},
      transpose_keys=getattr(sampler, "to_hf_transpose_keys", None),
      key_mapping_hook_fns=getattr(sampler, "to_hf_hook_fns", None),
      num_kv_heads=model_config.get_total_num_kv_heads(),
      head_dim=model_config.get_head_size(),
      tp_size=getattr(sampler, "args", {}).get("tensor_parallel_size", 1),
  )

  runner_leaves = tuple(runner.state_leaves)
  state_leaves = tuple(jax.tree_util.tree_leaves(runner.state))
  if len(runner_leaves) != len(state_leaves):
    raise FunctionalMappingError(
        "runner.state_leaves length disagrees with runner.state: "
        f"{len(runner_leaves)} != {len(state_leaves)}"
    )
  for index, (declared, actual) in enumerate(zip(runner_leaves, state_leaves)):
    if declared.shape != actual.shape or declared.dtype != actual.dtype:
      raise FunctionalMappingError(
          "runner.state_leaves order/contract mismatch at index "
          f"{index}: {(declared.shape, declared.dtype)} != "
          f"{(actual.shape, actual.dtype)}"
      )

  path_digest = hashlib.sha256(
      "\n".join(manifest.target_paths).encode("utf-8")
  ).hexdigest()
  return LiveEngineContract(
      implementation_id=(
          f"{type(runner).__module__}.{type(runner).__qualname__}:qwen3"
      ),
      mapping_entries=len(manifest.entries),
      target_path_sha256=path_digest,
      state_leaves=len(state_leaves),
      mesh_shape=tuple(
          (str(name), int(size)) for name, size in runner.mesh.shape.items()
      ),
      kv_caches=len(runner.kv_caches),
      model_fn=getattr(runner.model_fn, "__name__", type(runner.model_fn).__name__),
      compute_logits_fn=getattr(
          runner.compute_logits_fn,
          "__name__",
          type(runner.compute_logits_fn).__name__,
      ),
  )


def map_trainer_state_to_engine_leaves(
    *,
    trainer_state: Any,
    engine_state_contract: Any,
    key_mappings: Mapping[str, tuple[str, tuple[str | None, ...]]],
    transpose_keys: Mapping[str, tuple[int, ...]] | None = None,
    key_mapping_hook_fns: Mapping[str, Any] | None = None,
    rollout_engine: str = "vllm_jax",
    **shape_kwargs: Any,
) -> FunctionalEngineLeaves:
  """Purely maps trainer parameters into the engine state's leaf contract.

  ``engine_state_contract`` is inspected only for target paths, shapes, dtypes,
  and order.  Its values are never assigned or returned.  All transforms of
  trainer values are JAX operations, so gradients flow back through casts,
  transposes, reshapes, padding, and repetition.

  The function fails closed on every non-RNG trainer leaf without a mapping,
  every engine target leaf not produced exactly once, and duplicate writes.
  That strictness is intentional for the canonical training path; serving's
  best-effort warning semantics are not sufficient evidence here.
  """
  target_flat, unrolled = _mapping_pairs(
      trainer_state=trainer_state,
      engine_state_contract=engine_state_contract,
      key_mappings=key_mappings,
  )
  mapped: dict[str, jax.Array] = {}
  provenance: list[tuple[str, str]] = []
  for (source_path, target_path), (value, target_param) in unrolled.items():
    if target_path in mapped:
      raise FunctionalMappingError(
          f"canonical engine target written more than once: {target_path}"
      )
    value = _transform_value(
        value,
        source_path=source_path,
        target_param=target_param,
        transpose_keys=transpose_keys,
        key_mapping_hook_fns=key_mapping_hook_fns,
        rollout_engine=rollout_engine,
        shape_kwargs=shape_kwargs,
    )
    mapped[target_path] = value
    provenance.append((source_path, target_path))

  target_paths = tuple(_flat_path(path) for path, _ in target_flat)
  missing_target = sorted(path for path in target_paths if path not in mapped)
  unexpected_target = sorted(path for path in mapped if path not in target_paths)
  if missing_target or unexpected_target:
    raise FunctionalMappingError(
        "canonical engine mapping is not target-complete: "
        f"missing={missing_target}, unexpected={unexpected_target}"
    )

  return FunctionalEngineLeaves(
      paths=target_paths,
      leaves=tuple(mapped[path] for path in target_paths),
      source_to_target=tuple(sorted(provenance)),
  )
