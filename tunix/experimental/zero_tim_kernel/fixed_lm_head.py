# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fixed-shape Pallas output head for registered Qwen3 geometries ("P38").

The output head ``[M, K] @ [K, V]`` (V = 151936, vocabulary-sharded over TP)
is evaluated at very different M by the three programs that must agree: the
engine's decode steps (request buckets M in {8, ..., 256}), its prefill /
scoring forward, and the learner (2048 or 4096 rows per call).  P38 routes
every admitted outer shape through ONE fixed Pallas program
``[FIXED_M=256, K] @ [K, V/TP]`` with fixed tiles BM/BN/BK = 128/256/256:

* request buckets M < 256 are zero-padded to 256 rows (rows are independent,
  so padding never changes a real row) and sliced back;
* M == 256 runs directly;
* learner shapes M = c * 256 are reshaped into c chunks and mapped with
  ``lax.map`` over the same 256-row program.  Their custom VJP accumulates the
  weight cotangent chunk by chunk with a loop-carried ``lax.scan``, so the
  reduction order over chunks is fixed (ascending).

Non-registered model/topology/row tuples fail closed instead of retracing.

Under the P59 rank-parallel backward the head is traced inside P59's outer
manual ("data", "model") map: it then runs directly on the local shard, and
the input cotangent is completed over TP in rank order in f32
(``fixed_order_reduce.p59_column_parallel``).

``local_matmul`` is injected by the caller; use
``canonical_local_matmul(contract)`` to build the P22.XK canonical-VJP coat
around the P22.XI padded fixed-tile matmul.
"""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
import os

from absl import logging
import jax
from jax import lax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from tunix.experimental.zero_tim_kernel import canonical_vjp
from tunix.experimental.zero_tim_kernel import fixed_order_reduce
from tunix.experimental.zero_tim_kernel import model_contracts
from tunix.experimental.zero_tim_kernel import padded_matmul

# Enable flag for the fixed-shape LM head.
ENV = "CANON_P38_FIXED_LM_HEAD"
# Pinned request buckets for the max-concurrency-256 target plus
# the registered learner shapes.  Request buckets are padded to FIXED_M;
# learner shapes map to exact FIXED_M chunks.
REQUEST_M = (8, 16, 32, 64, 128, 256)
LEARNER_M = (4096,)
QWEN4B_TP8_LEARNER_M = (2048, 4096)
QWEN8B_TP8_LEARNER_M = (1024, 2048, 4096)
FIXED_M = 256
VOCAB = model_contracts.VOCAB_SIZE
BM = 128
BN = 256
BK = 256
ENDPOINTS = ("untied_lm_head", "tied_embed", "direct_probe")
# CANON_P66_BACKWARD_ARM values of the (diagnostic) P66 DP1xTP4 unit-data
# experiment: the only case in which the P59 head admits a unit data axis.
P66_BACKWARD_ARM_ENV = "CANON_P66_BACKWARD_ARM"
P66_UNIT_DATA_ARMS = (
    "tp4-p59-old",
    "tp4-p59",
    "tp4-gather-off",
    "tp4-vma-oracle",
)


@dataclasses.dataclass(frozen=True)
class Geometry:
  """One model/topology/output-endpoint executable contract."""

  model: str
  hidden: int
  tp_size: int
  endpoint: str
  local_vocab: int
  padded_local_vocab: int


def _geometry(model: str, hidden: int, tp_size: int, endpoint: str) -> Geometry:
  local_vocab = VOCAB // tp_size
  padded_local_vocab = ((local_vocab + BN - 1) // BN) * BN
  return Geometry(
      model=model,
      hidden=hidden,
      tp_size=tp_size,
      endpoint=endpoint,
      local_vocab=local_vocab,
      padded_local_vocab=padded_local_vocab,
  )


GEOMETRIES = {
    (2048, 4): _geometry("qwen3-1p7b", 2048, 4, "tied_embed"),
    (4096, 4): _geometry("qwen3-8b", 4096, 4, "untied_lm_head"),
    (4096, 8): _geometry("qwen3-8b-tp8", 4096, 8, "untied_lm_head"),
    (2560, 4): _geometry("qwen3-4b-tp4", 2560, 4, "tied_embed"),
    (2560, 8): _geometry("qwen3-4b", 2560, 8, "tied_embed"),
    (5120, 8): _geometry("qwen3-32b", 5120, 8, "untied_lm_head"),
}
SUPPORTED_HIDDEN = tuple(sorted({hidden for hidden, _ in GEOMETRIES}))


def _validate_hidden(hidden: int) -> int:
  """Returns `hidden` if it is an admitted size matching the env override."""
  hidden = int(hidden)
  if hidden not in SUPPORTED_HIDDEN:
    raise ValueError(
        "P38 fixed lm_head requires hidden size in "
        f"{SUPPORTED_HIDDEN}, got {hidden}"
    )
  configured = os.environ.get("CANON_QWEN3_HIDDEN_SIZE", "")
  if configured:
    try:
      configured_hidden = int(configured)
    except ValueError as error:
      raise ValueError(
          "P38 fixed lm_head requires an integer "
          f"CANON_QWEN3_HIDDEN_SIZE, got {configured!r}"
      ) from error
    if configured_hidden != hidden:
      raise ValueError(
          "P38 fixed lm_head hidden size disagrees with the profile: "
          f"shape={hidden} env={configured_hidden}"
      )
  return hidden


def resolve_geometry(
    hidden: int, tp_size: int, *, endpoint: str | None = None
) -> Geometry:
  """Returns one registered geometry and rejects model/topology drift."""
  hidden = _validate_hidden(hidden)
  tp_size = int(tp_size)
  try:
    geometry = GEOMETRIES[(hidden, tp_size)]
  except KeyError as error:
    registered = ", ".join(
        f"K{item.hidden}/TP{item.tp_size}/{item.endpoint}"
        for item in GEOMETRIES.values()
    )
    raise ValueError(
        "P38 fixed lm_head model/topology is not registered: "
        f"K{hidden}/TP{tp_size}; registered={registered}"
    ) from error
  configured_tp = os.environ.get("CANON_QWEN3_TP_SIZE", "")
  if configured_tp:
    try:
      parsed_tp = int(configured_tp)
    except ValueError as error:
      raise ValueError(
          "P38 fixed lm_head requires an integer "
          f"CANON_QWEN3_TP_SIZE, got {configured_tp!r}"
      ) from error
    if parsed_tp != geometry.tp_size:
      raise ValueError(
          "P38 fixed lm_head TP size disagrees with the profile: "
          f"mesh={geometry.tp_size} env={parsed_tp}"
      )
  if endpoint is not None and endpoint not in (
      geometry.endpoint,
      "direct_probe",
  ):
    raise ValueError(
        "P38 fixed lm_head endpoint disagrees with the registered model: "
        f"model={geometry.model} expected={geometry.endpoint} got={endpoint}"
    )
  return geometry


def semantic_m_for_geometry(geometry: Geometry) -> tuple[int, ...]:
  """The outer M values admitted for ``geometry`` (request + learner)."""
  learner_m = {
      (2560, 8): QWEN4B_TP8_LEARNER_M,
      (4096, 8): QWEN8B_TP8_LEARNER_M,
  }.get((geometry.hidden, geometry.tp_size), LEARNER_M)
  return REQUEST_M + learner_m


def learner_m_for_geometry(geometry: Geometry) -> tuple[int, ...]:
  """The learner (chunked) M values admitted for ``geometry``."""
  return semantic_m_for_geometry(geometry)[len(REQUEST_M) :]


def validate_global_contract(
    input_shape,
    weight_shape,
    input_dtype,
    weight_dtype,
    *,
    tp_size: int,
) -> int:
  """Validates one caller-global Qwen3 contract and returns semantic M."""
  input_shape = tuple(map(int, input_shape))
  weight_shape = tuple(map(int, weight_shape))
  if len(input_shape) != 2:
    raise ValueError(
        f"P38 fixed lm_head requires rank-2 input, got {input_shape}"
    )
  hidden = _validate_hidden(input_shape[1])
  geometry = resolve_geometry(hidden, tp_size)
  admitted_m = semantic_m_for_geometry(geometry)
  if input_shape[0] not in admitted_m:
    raise ValueError(
        f"P38 fixed lm_head requires semantic M in {admitted_m}, "
        f"got {input_shape}"
    )
  if weight_shape != (hidden, VOCAB):
    raise ValueError(
        "P38 fixed lm_head requires input/weight "
        f"[(M,{hidden}),({hidden},{VOCAB})], got {input_shape}/{weight_shape}"
    )
  if str(input_dtype) != "bfloat16" or str(weight_dtype) != "bfloat16":
    raise ValueError(
        "P38 fixed lm_head requires bf16 input/weight, got "
        f"{input_dtype}/{weight_dtype}"
    )
  return input_shape[0]


def validate_local_contract(
    input_shape,
    weight_shape,
    *,
    tp_size: int,
    admitted_m: tuple[int, ...] | None = None,
) -> int:
  """Validates one shard_map local shard and returns semantic M."""
  input_shape = tuple(map(int, input_shape))
  weight_shape = tuple(map(int, weight_shape))
  if len(input_shape) != 2:
    raise ValueError(f"P38 fixed lm_head local rank invalid: {input_shape}")
  hidden = _validate_hidden(input_shape[1])
  geometry = resolve_geometry(hidden, tp_size)
  admitted_m = (
      semantic_m_for_geometry(geometry)
      if admitted_m is None
      else tuple(map(int, admitted_m))
  )
  if input_shape[0] not in admitted_m:
    raise ValueError(f"P38 fixed lm_head local M invalid: {input_shape}")
  if weight_shape != (hidden, geometry.local_vocab):
    raise ValueError(
        "P38 fixed lm_head local shape mismatch: "
        f"{input_shape}/{weight_shape}, expected (M,{hidden})/"
        f"({hidden},{geometry.local_vocab})"
    )
  return input_shape[0]


def _p59_local_contract(inputs, weight, *, mesh, tp_axis: str):
  """Resolves one already-DP/TP-mapped P59 head call or returns None.

  P59's outer shard_map has already sliced both the DP row dimension and the
  TP vocabulary dimension.  Re-entering the engine's concrete shard_map is
  illegal and would partition the TP-local weight twice.  This admission is
  deliberately structural: ordinary serving with the P59 flag present has no
  two-axis manual outer context and therefore stays on the global path.

  Args:
    inputs: The (DP-local) input rows.
    weight: The TP-local weight slice.
    mesh: The live engine mesh.
    tp_axis: The engine's TP axis name.

  Returns:
    ``(dp_size, global_m, geometry)`` inside the P59 context, else None.

  Raises:
    RuntimeError: If the manual context is present but the engine mesh does
      not provide matching non-unit ``data``/``model`` axes.
    ValueError: If the local shapes/dtypes do not reconstruct one learner M.
  """
  if (
      os.environ.get(fixed_order_reduce.P59_RANK_PARALLEL_BACKWARD_ENV, "")
      != "1"
  ):
    return None
  context = jax.sharding.get_abstract_mesh()
  if tuple(context.axis_names) != ("data", "model"):
    return None
  axis_types = dict(zip(context.axis_names, context.axis_types))
  if (
      axis_types.get("data") is not jax.sharding.AxisType.Manual
      or axis_types.get("model") is not jax.sharding.AxisType.Manual
  ):
    return None
  if (
      tp_axis != "model"
      or tp_axis not in mesh.shape
      or "data" not in mesh.shape
  ):
    raise RuntimeError(
        "P59 local fixed lm_head requires engine data/model axes"
    )
  dp_size = int(context.shape["data"])
  tp_size = int(context.shape["model"])
  p66_unit_data = (
      dp_size == 1
      and tp_size == 4
      and os.environ.get(P66_BACKWARD_ARM_ENV, "") in P66_UNIT_DATA_ARMS
  )
  if (dp_size <= 1 and not p66_unit_data) or tp_size <= 1:
    raise RuntimeError("P59 local fixed lm_head requires non-unit DP and TP")
  if int(mesh.shape["data"]) != dp_size or int(mesh.shape[tp_axis]) != tp_size:
    raise RuntimeError(
        "P59 local fixed lm_head context and engine topology differ"
    )
  input_shape = tuple(map(int, inputs.shape))
  weight_shape = tuple(map(int, weight.shape))
  if len(input_shape) != 2:
    raise ValueError(
        f"P59 local fixed lm_head input rank changed: {input_shape}"
    )
  hidden_size = _validate_hidden(input_shape[1])
  geometry = resolve_geometry(hidden_size, tp_size)
  if weight_shape != (hidden_size, geometry.local_vocab):
    raise ValueError(
        "P59 local fixed lm_head weight shape mismatch: "
        f"{weight_shape} != {(hidden_size, geometry.local_vocab)}"
    )
  if inputs.dtype != jnp.bfloat16 or weight.dtype != jnp.bfloat16:
    raise ValueError(
        "P59 local fixed lm_head requires bf16 input/weight, got "
        f"{inputs.dtype}/{weight.dtype}"
    )
  if p66_unit_data:
    global_m = (256,) if input_shape[0] == 256 else ()
  else:
    global_m = tuple(
        candidate
        for candidate in learner_m_for_geometry(geometry)
        if candidate % dp_size == 0 and candidate // dp_size == input_shape[0]
    )
  if len(global_m) != 1:
    raise ValueError(
        "P59 local fixed lm_head rows do not reconstruct one learner M: "
        f"local_M={input_shape[0]} dp={dp_size} "
        f"learner_M={learner_m_for_geometry(geometry)}"
    )
  return dp_size, global_m[0], geometry


def canonical_local_matmul(
    contract: model_contracts.ModelContract, *, interpret: bool = False
) -> Callable[..., jax.Array]:
  """Builds the canonical ``local_matmul``: P22.XK coat over P22.XI padding.

  The primal is the padded fixed-tile Pallas matmul at the tiles the head
  passes, and the VJP is the canonical one.

  Args:
    contract: The model contract (admits the lm_head's padded local vocab).
    interpret: Run the Pallas kernel in interpret mode (CPU tests).

  Returns:
    ``local_matmul(x, y, *, shape_invariant_numerics=True, **tiles)``.
  """

  def local_matmul(x, y, *, shape_invariant_numerics: bool = True, **tiles):
    def forward(a, b):
      return padded_matmul.matmul(
          a,
          b,
          contract=contract,
          interpret=interpret,
          shape_invariant_numerics=shape_invariant_numerics,
          **tiles,
      )

    return canonical_vjp.matmul(x, y, forward=forward, contract=contract)

  return local_matmul


def fixed_lm_head(
    inputs,
    weight,
    *,
    mesh,
    tp_axis: str,
    local_matmul: Callable[..., jax.Array],
    endpoint: str,
):
  """Runs every registered outer shape through one fixed Pallas shape.

  Args:
    inputs: ``[M, K]`` bf16 hidden states (global, or DP-local under P59).
    weight: ``[K, V]`` bf16 output head (``V/TP`` local slice under P59).
    mesh: The live engine mesh.
    tp_axis: The TP axis name.
    local_matmul: ``local_matmul(a [256, K], w [K, V/TP], *, block_m, block_n,
      block_k, shape_invariant_numerics)``; see ``canonical_local_matmul``.
    endpoint: One of ``ENDPOINTS``.

  Returns:
    ``[M, V]`` bf16 logits (``[M, V/TP]`` local logits under P59).

  Raises:
    RuntimeError: If no live mesh is given, the mesh lacks ``tp_axis``, the
      P59 context and engine topology disagree, or the kernel output shape is
      not the expected one.
    ValueError: If ``endpoint`` or the input/weight shapes are not admitted.
  """
  if endpoint not in ENDPOINTS:
    raise ValueError(
        f"P38 fixed lm_head endpoint must be one of {ENDPOINTS}, got "
        f"{endpoint!r}"
    )
  if mesh is None:
    raise RuntimeError("P38 fixed lm_head requires the live model mesh")
  if tp_axis not in mesh.shape:
    raise RuntimeError(
        f"P38 fixed lm_head mesh lacks axis {tp_axis!r}: {mesh.shape}"
    )
  tp_size = int(mesh.shape[tp_axis])
  p59_local = _p59_local_contract(inputs, weight, mesh=mesh, tp_axis=tp_axis)
  if p59_local is None:
    semantic_m = validate_global_contract(
        inputs.shape,
        weight.shape,
        inputs.dtype,
        weight.dtype,
        tp_size=tp_size,
    )
    p59_dp_size = 0
    p59_global_m = semantic_m
  else:
    p59_dp_size, p59_global_m, _ = p59_local
    semantic_m = int(inputs.shape[0])
  hidden_size = _validate_hidden(inputs.shape[1])
  geometry = resolve_geometry(hidden_size, tp_size, endpoint=endpoint)

  def local(a_local, w_local):
    local_m = validate_local_contract(
        a_local.shape,
        w_local.shape,
        tp_size=tp_size,
        admitted_m=(semantic_m,),
    )
    if a_local.dtype != jnp.bfloat16 or w_local.dtype != jnp.bfloat16:
      raise ValueError(
          "P38 fixed lm_head local dtype mismatch: "
          f"{a_local.dtype}/{w_local.dtype}"
      )

    def run_fixed_with_weight(a_fixed, weight_local):
      return local_matmul(
          a_fixed,
          weight_local,
          block_m=BM,
          block_n=BN,
          block_k=BK,
          shape_invariant_numerics=True,
      )

    def run_fixed(a_fixed):
      return run_fixed_with_weight(a_fixed, w_local)

    def learner_forward(a_learner, weight_local):
      a_chunks = a_learner.reshape((-1, FIXED_M, hidden_size))
      return lax.map(
          lambda a_chunk: run_fixed_with_weight(a_chunk, weight_local),
          a_chunks,
      ).reshape((a_learner.shape[0], geometry.local_vocab))

    @jax.custom_vjp
    def learner_fixed_vjp(a_learner, weight_local):
      return learner_forward(a_learner, weight_local)

    def learner_fwd(a_learner, weight_local):
      output = learner_forward(a_learner, weight_local)
      return output, (a_learner, weight_local)

    def learner_bwd(residual, cotangent):
      learner_m = int(residual[0].shape[0])
      logging.info(
          "[PATHTRACE] CANON_P38_FIXED_LM_HEAD_VJP=1 semantic_M=%d "
          "fixed_M=%d chunks=%d accumulation=lax.scan order=ascending K=%d "
          "TP=%d local_N=%d fixed_N=%d endpoint=%s",
          learner_m,
          FIXED_M,
          learner_m // FIXED_M,
          hidden_size,
          tp_size,
          geometry.local_vocab,
          geometry.padded_local_vocab,
          endpoint,
      )
      a_learner, weight_local = residual
      a_chunks = a_learner.reshape((-1, FIXED_M, hidden_size))
      cotangent_chunks = cotangent.reshape((-1, FIXED_M, geometry.local_vocab))

      def accumulate(weight_cotangent, values):
        a_chunk, output_cotangent = values
        _, pullback = jax.vjp(run_fixed_with_weight, a_chunk, weight_local)
        a_cotangent, chunk_weight_cotangent = pullback(output_cotangent)
        # The loop-carried dependency is the backward contract: chunk q is
        # added only after chunks [0, q) have completed.
        return weight_cotangent + chunk_weight_cotangent, a_cotangent

      weight_cotangent, a_cotangent_chunks = lax.scan(
          accumulate,
          jnp.zeros_like(weight_local),
          (a_chunks, cotangent_chunks),
      )
      return (
          a_cotangent_chunks.reshape(a_learner.shape),
          weight_cotangent,
      )

    learner_fixed_vjp.defvjp(learner_fwd, learner_bwd)

    if local_m < FIXED_M:
      a_fixed = jnp.pad(
          a_local,
          ((0, FIXED_M - local_m), (0, 0)),
          constant_values=0,
      )
      chunks = 1
      out = run_fixed(a_fixed)[:local_m, :]
    elif local_m == FIXED_M:
      chunks = 1
      out = run_fixed(a_local)
    else:
      chunks = local_m // FIXED_M
      out = learner_fixed_vjp(a_local, w_local)
    logging.info(
        "[PATHTRACE] CANON_P38_FIXED_LM_HEAD=1 semantic_M=%d fixed_M=%d K=%d "
        "TP=%d local_N=%d fixed_N=%d BM=%d BN=%d BK=%d chunks=%d "
        "endpoint=%s%s",
        local_m,
        FIXED_M,
        hidden_size,
        tp_size,
        geometry.local_vocab,
        geometry.padded_local_vocab,
        BM,
        BN,
        BK,
        chunks,
        endpoint,
        f" p59_local=1 global_M={p59_global_m} dp={p59_dp_size}"
        if p59_local is not None
        else "",
    )
    if tuple(map(int, out.shape)) != (local_m, geometry.local_vocab):
      raise RuntimeError(
          f"P38 fixed lm_head output shape mismatch: {out.shape}"
      )
    return out

  if p59_local is not None:
    logging.info(
        "[PATHTRACE] CANON_P38_FIXED_LM_HEAD_VJP=1 semantic_M=%d local_M=%d "
        "fixed_M=%d tp_input_reduction=%s K=%d TP=%d local_N=%d endpoint=%s",
        p59_global_m,
        semantic_m,
        FIXED_M,
        "vma_autodiff_psum"
        if os.environ.get(fixed_order_reduce.P66_CHECK_VMA_ENV, "0") == "1"
        else "all_gather_rank_order_f32_barrier",
        hidden_size,
        tp_size,
        geometry.local_vocab,
        endpoint,
    )
    # The replicated input's cotangent is completed over TP in rank order
    # (f32, barriered); the TP-local weight slice keeps its local cotangent.
    output = fixed_order_reduce.p59_column_parallel(
        local,
        inputs,
        weight,
        replicated=(True, False),
        axis_name=tp_axis,
        count=tp_size,
    )
  else:
    mapped = jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(P(None, None), P(None, tp_axis)),
        out_specs=P(None, tp_axis),
        check_vma=False,
    )
    output = mapped(inputs, weight)
  expected_output_shape = (
      (semantic_m, geometry.local_vocab)
      if p59_local is not None
      else (semantic_m, VOCAB)
  )
  if tuple(map(int, output.shape)) != expected_output_shape:
    raise RuntimeError(
        "P38 fixed lm_head output boundary mismatch: "
        f"{output.shape} != {expected_output_shape}"
    )
  return output
