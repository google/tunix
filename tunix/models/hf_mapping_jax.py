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
"""JAX execution engine for the declarative 12-Op Weight Conversion Protocol."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import itertools
from typing import Any, Final

import jax
from jax import typing as jax_typing
import jax.numpy as jnp
from tunix.models import hf_mapping

_JAX_DTYPES: Final[Mapping[str, jax_typing.DTypeLike]] = {
    "float32": jnp.float32,
    "bfloat16": jnp.bfloat16,
    "float16": jnp.float16,
    "float8_e4m3fn": jnp.float8_e4m3fn,
    "float8_e5m2": jnp.float8_e5m2,
    "int32": jnp.int32,
    "int8": jnp.int8,
    "uint8": jnp.uint8,
}

_UNARY_OPS: Final[Mapping[str, tuple[Callable[..., jax.Array], str]]] = {
    "reshape": (jnp.reshape, "dst_shape"),
    "transpose": (jnp.transpose, "axes"),
    "scale": (
        lambda t, f: (t.astype(jnp.float32) * float(f)).astype(t.dtype),
        "factor",
    ),
    "cast": (
        lambda t, dt: jax.lax.convert_element_type(t, _JAX_DTYPES[str(dt)]),
        "dst_dtype",
    ),
}

_AXIS_OPS: Final[
    Mapping[str, Callable[[jax.Array, Mapping[str, Any]], jax.Array]]
] = {
    "repeat": lambda t, kw: jnp.repeat(
        t, int(kw["times"]), axis=int(kw["axis"])
    ),
    "unrepeat": lambda t, kw: jax.lax.slice_in_dim(
        t,
        0,
        t.shape[int(kw["axis"])],
        stride=int(kw["times"]),
        axis=int(kw["axis"]),
    ),
    "unpad": lambda t, kw: jax.lax.slice_in_dim(
        t,
        int(kw["sizes"][0]),
        t.shape[int(kw["axis"])] - int(kw["sizes"][1]),
        axis=int(kw["axis"]),
    ),
}

_RUNTIME_SHAPE_CHECKS: Final[
    Mapping[str, Callable[[list[jax.Array], Mapping[str, Any]], bool]]
] = {
    "reshape": (
        lambda ts, kw: [t.shape for t in ts]
        == [tuple(kw["src_shape"])] * len(ts)
    ),
    "concat": (
        lambda ts, kw: bool(ts)
        and [t.shape[int(kw["axis"])] for t in ts]
        == list(kw["sizes"]) * (len(ts) // len(kw["sizes"]))
    ),
    "split": (
        lambda ts, kw: [t.shape[int(kw["axis"])] for t in ts]
        == [sum(kw["sizes"])] * len(ts)
    ),
    "unpad": lambda ts, kw: all(
        t.shape[int(kw["axis"])] > sum(kw["sizes"]) for t in ts
    ),
    "unrepeat": lambda ts, kw: not any(
        t.shape[int(kw["axis"])] % int(kw["times"]) for t in ts
    ),
}


def apply_primitive_op(
    tensors: list[jax.Array], op_name: str, kw: Mapping[str, Any]
) -> list[jax.Array]:
  """Executes one of the 12 self-invertible primitive weight-conversion ops."""
  hf_mapping.validate_op(op_name, kw)
  if op_name in _RUNTIME_SHAPE_CHECKS and not _RUNTIME_SHAPE_CHECKS[op_name](
      tensors, kw
  ):
    raise ValueError(
        f"Runtime shape mismatch for {op_name!r} with {dict(kw)}: "
        f"{[t.shape for t in tensors]}"
    )
  if op_name == "stack":
    return [jnp.stack(tensors, axis=int(kw["axis"]))]
  if op_name == "concat":
    step = len(kw["sizes"])
    return [
        jnp.concatenate(tensors[i : i + step], axis=int(kw["axis"]))
        for i in range(0, len(tensors), step)
    ]
  if op_name in _UNARY_OPS:
    fn, key = _UNARY_OPS[op_name]
    return [fn(t, kw[key]) for t in tensors]
  if op_name in _AXIS_OPS:
    axis_fn = _AXIS_OPS[op_name]
    return [axis_fn(t, kw) for t in tensors]
  if op_name == "unstack":
    axis = int(kw["axis"])
    return list(
        itertools.chain.from_iterable(
            jnp.unstack(t, axis=axis) for t in tensors
        )
    )
  if op_name == "split":
    cuts = tuple(itertools.accumulate(kw["sizes"][:-1]))
    axis = int(kw["axis"])
    return list(
        itertools.chain.from_iterable(
            jnp.split(t, cuts, axis=axis) for t in tensors
        )
    )
  out: list[jax.Array] = []
  for t in tensors:
    pads: list[tuple[int, int]] = [(0, 0)] * t.ndim
    pads[int(kw["axis"])] = (int(kw["sizes"][0]), int(kw["sizes"][1]))
    out.append(jnp.pad(t, pads))
  return out


def apply_rule(
    rule: Mapping[str, Any],
    src_weights: Mapping[str, Any],
    mesh: jax.sharding.Mesh | None = None,
) -> dict[str, jax.Array]:
  """Executes a single declarative weight-conversion rule over `src_weights`."""
  hf_mapping.validate_rule(rule)
  tensors = [jnp.asarray(src_weights[k]) for k in rule["source_keys"]]
  for step in rule["ops"]:
    tensors = apply_primitive_op(tensors, *next(iter(step.items())))
  sharding_spec = rule["sharding"] if "sharding" in rule else None
  if mesh is not None and sharding_spec:
    pspec = jax.sharding.PartitionSpec(
        *(tuple(a) if isinstance(a, list) else a for a in sharding_spec)
    )
    sharding = jax.sharding.NamedSharding(mesh, pspec)
    tensors = [jax.lax.with_sharding_constraint(t, sharding) for t in tensors]
  return dict(zip(rule["target_keys"], tensors, strict=True))


def execute_manifest(
    manifest: hf_mapping.RuleSpec,
    src_weights: Mapping[str, Any],
    mesh: jax.sharding.Mesh | None = None,
) -> dict[str, jax.Array]:
  """Executes a declarative 12-Op weight-conversion manifest."""
  spec = hf_mapping.parse_manifest(manifest)
  dropped = set(spec["dropped_keys"])
  rules = spec["rules"]
  tied_dst = {dst for _, dst in spec["tied_keys"]}
  consumed = (
      set(itertools.chain.from_iterable(r["source_keys"] for r in rules))
      | dropped
      | tied_dst
  )
  out = {k: jnp.asarray(src_weights[k]) for k in src_weights.keys() - consumed}
  for rule in rules:
    if dropped.intersection((*rule["source_keys"], *rule["target_keys"])):
      continue
    out.update(apply_rule(rule, src_weights, mesh=mesh))
  return out | {
      dst: out[src]
      for src, dst in spec["tied_keys"]
      if src in out and dst not in dropped
  }


def to_hf_impl(
    src_spec: hf_mapping.RuleSpec,
    src_weights: Mapping[str, Any],
) -> dict[str, jax.Array]:
  """Pure JAX-jittable kernel converting source weights into HF layout.

  Architectural role: `to_hf_impl` is the inner JAX transformation kernel
  designed to be traced and fused directly inside `jax.jit` or `jax.eval_shape`
  (e.g., Scenario 1 colocated weight sync:
  `jax.jit(lambda w: from_hf_impl(dst_spec, to_hf_impl(src_spec, w), mesh=m))`).
  Unlike the cross-process boundary entrypoint `to_hf`, callers of `to_hf_impl`
  within a JIT-compiled function pass pre-parsed manifest specs or static rule
  mappings alongside traced `jax.Array` pytrees.

  Args:
    src_spec: Declarative `Trainer -> HF` weight-mapping specification authored
      for the source model's layout (independent of the target sampler's
      `dst_spec`; a single-model implementation may also derive this from its
      `HF -> Model` manifest via `hf_mapping.invert_manifest`).
    src_weights: Mapping of source parameter names to `jax.Array`s.

  Returns:
    Mapping of canonical HuggingFace parameter names to transformed
    `jax.Array`s.
  """
  spec = hf_mapping.parse_manifest(src_spec)
  tied_dst = [dst for _, dst in spec["tied_keys"]]
  src_manifest = {
      **spec,
      "tied_keys": [],
      "dropped_keys": [*spec["dropped_keys"], *tied_dst],
  }
  return execute_manifest(src_manifest, src_weights, mesh=None)


def from_hf_impl(
    dst_spec: hf_mapping.RuleSpec,
    hf_weights: Mapping[str, Any],
    mesh: jax.sharding.Mesh | None = None,
) -> dict[str, jax.Array]:
  """Pure JAX-jittable kernel converting HF weights into target layout.

  Architectural role: `from_hf_impl` is the inner JAX transformation and
  GSPMD sharding kernel designed to be traced and fused directly inside
  `jax.jit` or `jax.eval_shape` (e.g., Scenario 1 colocated weight sync:
  `jax.jit(lambda w: from_hf_impl(dst_spec, to_hf_impl(src_spec, w), mesh=m))`).
  Unlike the cross-process boundary entrypoint `from_hf`, `from_hf_impl` is the
  composable XLA kernel invoked inside JIT-compiled weight transfer pipelines.

  Args:
    dst_spec: Declarative `HF -> Target` weight-mapping specification for the
      target model layout.
    hf_weights: Mapping of canonical HuggingFace parameter names to
      `jax.Array`s.
    mesh: Optional target JAX device mesh for `with_sharding_constraint`.

  Returns:
    Mapping of target parameter names to transformed (and optionally sharded)
    `jax.Array`s.
  """
  return execute_manifest(dst_spec, hf_weights, mesh=mesh)


def to_hf(
    src_rule: str,
    src_weights: Mapping[str, Any],
) -> dict[str, jax.Array]:
  """Cross-process API boundary entrypoint converting weights into HF layout.

  Architectural role: `to_hf` is the public cross-process/RPC boundary
  entrypoint (Scenario 2 disaggregated weight sync and Scenario 3 checkpoint
  export). It accepts a raw serialized JSON manifest string (`src_rule: str`)
  across process boundaries, validates and deserializes the JSON payload via
  `WeightMappingSpec.from_json(src_rule)`, and delegates tensor execution to
  the pure JAX kernel `to_hf_impl`. Do not remove `to_hf_impl` or collapse the
  two layers.

  Args:
    src_rule: Raw serialized JSON string describing the source engine's `Trainer
      -> HF` weight-conversion rules (`src_rule_A`, independent of the target
      sampler's `dst_rule_B`).
    src_weights: Mapping of source parameter names to arrays.

  Returns:
    Mapping of canonical HuggingFace parameter names to `jax.Array`s.
  """
  parsed = hf_mapping.WeightMappingSpec.from_json(src_rule)
  return to_hf_impl(parsed, src_weights)


def from_hf(
    dst_rule: str,
    hf_weights: Mapping[str, Any],
    mesh: jax.sharding.Mesh | None = None,
) -> dict[str, jax.Array]:
  """Cross-process API boundary entrypoint converting HF weights to target.

  Architectural role: `from_hf` is the public cross-process/RPC boundary
  entrypoint (Scenario 2 disaggregated weight sync and Scenario 3 checkpoint
  loading). It accepts a raw serialized JSON manifest string (`dst_rule: str`)
  across process boundaries, validates and deserializes the JSON payload via
  `WeightMappingSpec.from_json(dst_rule)`, and delegates tensor execution and
  sharding to the pure JAX kernel `from_hf_impl`. Do not remove `from_hf_impl`
  or collapse the two layers.

  Args:
    dst_rule: Raw serialized JSON string describing the target engine's `HF ->
      Sampler` weight-conversion rules (`dst_rule_B`, independent of the source
      trainer's `src_rule_A`).
    hf_weights: Mapping of canonical HuggingFace parameter names to arrays.
    mesh: Optional target JAX device mesh for sharding constraints.

  Returns:
    Mapping of target parameter names to `jax.Array`s.
  """
  parsed = hf_mapping.WeightMappingSpec.from_json(dst_rule)
  return from_hf_impl(parsed, hf_weights, mesh=mesh)
