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
"""Fixed rank-order tensor-parallel reductions (Zero-TIM "R1").

Root cause: the bf16 all-reduce that completes a
contract-dim-sharded projection (attention ``o_proj``, MLP ``down_proj``) sums
the per-chip partials in a row-position-dependent order (ring chunking).  bf16
addition is not associative, so the same token at a different padded row gets
different bits -- the decode-vs-prefill seed.  The vocab-sharded embedding
gather ends in the same kind of 4-way all-reduce, and XLA lowers it
differently in a forward-only and a forward+backward program.  Casting the
all-reduce to f32 did NOT fix either.  What fixes both is an explicit sum in
one association order, ``((p0 + p1) + p2) + p3``, identical on every rank, for
every row and in every program:

* ``ring``: ``count - 1`` ``ppermute`` hops bring every partial to every rank
  and ``argsort((j - arange(n)) % n)`` restores rank order;
* ``gather``: one ``all_gather`` (which concatenates in rank order) and the
  same ordered sum -- identical operands and association, so bitwise equal to
  ``ring`` with one collective instead of ``count - 1``;
* ``scatter``: ``all_to_all`` hands row block ``r`` of every rank's partial to
  rank ``r`` stacked by source rank, rank ``r`` adds it in the same order and a
  tiled ``all_gather`` restores the rows -- bitwise equal to ``gather`` with
  ``2/count`` of the bytes.

The association order pins *what* is added; the rounding of every step is
pinned too: by default a bf16 chain adds through ``hard_bf16_add`` (f32 add,
``lax.reduce_precision(8, 7)``), which is the strict-flag bf16 chain bit for
bit under either ``--xla_allow_excess_precision`` setting.  ``plain_add``
restores the plain ``+`` for A/B runs.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import operator
import os

from absl import logging
import jax
from jax import lax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

FIXED_AR_ENV = "CANON_FIXED_AR"
FIXED_AR_EMBED_ENV = "CANON_FIXED_AR_EMBED"
FIXED_AR_GATHER_ENV = "CANON_FIXED_AR_GATHER"
FIXED_AR_SCATTER_ENV = "CANON_FIXED_AR_SCATTER"
P59_RANK_PARALLEL_BACKWARD_ENV = "CANON_P59_RANK_PARALLEL_BACKWARD"
P66_CHECK_VMA_ENV = "CANON_P66_P59_CHECK_VMA"
P67_SCOPE_ENV = "CANON_P67_P66_VMA_P59_ONLY"

RING = "ring"
GATHER = "gather"
SCATTER = "scatter"
MODES = (RING, GATHER, SCATTER)

# Binary addition of two consecutive rank operands; see
# ``fixed_order_tp_sum_ring``.
AddFn = Callable[[jax.Array, jax.Array], jax.Array]

_BF16_EXPONENT_BITS = 8
_BF16_MANTISSA_BITS = 7

# The plain ``+``: rounds after every step only under
# ``--xla_allow_excess_precision=false``.
plain_add: AddFn = operator.add


def _hard_upcast(x: jax.Array) -> jax.Array:
  """``x.astype(f32)`` that re-asserts the bf16 rounding of a bf16 ``x``."""
  x_f32 = x.astype(jnp.float32)
  if x.dtype != jnp.bfloat16:
    return x_f32
  return lax.reduce_precision(
      x_f32,
      exponent_bits=_BF16_EXPONENT_BITS,
      mantissa_bits=_BF16_MANTISSA_BITS,
  )


def hard_bf16_add(acc: jax.Array, operand: jax.Array) -> jax.Array:
  """``acc + operand`` with the bf16 rounding of every step pinned.

  The default ``add`` of the fixed-order sums.  A fixed association order only
  pins *what* is added; a plain bf16 ``+`` chain is rounded after every step
  only under ``--xla_allow_excess_precision=false``.  Under XLA's default flag
  the chain may be carried in f32 and rounded once at its end (or later, inside
  the consumer's fusion), so the bits depend on each program's fusion
  decisions -- the decode-vs-prefill drift attributed in the item-1b
  hard-rounding study.  Upcasting both operands, adding in f32 and rounding the
  sum through ``lax.reduce_precision(8, 7)`` is the same round-to-nearest-even
  as the strict-flag chain, written as arithmetic the algebraic simplifier
  cannot remove, so the result is the strict-flag bits under either flag.
  Operands whose promoted dtype is not bf16 are added as-is (an f32 chain has
  no bf16 rounding point to pin).

  Args:
    acc: The running sum (rank 0's partial at the first step).
    operand: The next rank's partial.

  Returns:
    The rounded sum in ``jnp.result_type(acc, operand)``.
  """
  if jnp.result_type(acc, operand) != jnp.bfloat16:
    return acc + operand
  total = _hard_upcast(acc) + _hard_upcast(operand)
  rounded = lax.reduce_precision(
      total,
      exponent_bits=_BF16_EXPONENT_BITS,
      mantissa_bits=_BF16_MANTISSA_BITS,
  )
  return rounded.astype(jnp.bfloat16)


def _resolve_add(add: AddFn | None) -> AddFn:
  return hard_bf16_add if add is None else add


# The two contract-parallel (row-parallel) projection equations of a Qwen
# decoder layer: MLP down_proj and attention o_proj.
CONTRACT_PARALLEL_EQUATIONS = ("mn,np->mp", "TNH,NHD->TD")

_RECEIPTS: set[str] = set()


def _log_once(key: str, message: str, *args) -> None:
  if key in _RECEIPTS:
    return
  _RECEIPTS.add(key)
  logging.info(message, *args)


def fixed_specs(equation: str, tp_axis: str = "model") -> tuple[P, P]:
  """Returns the (input, weight) partition specs of a contract-parallel einsum.

  The contracted dimension is the TP-sharded one, so every rank holds a partial
  sum of the full output.

  Args:
    equation: One of ``CONTRACT_PARALLEL_EQUATIONS``.
    tp_axis: The tensor-parallel mesh axis name.

  Returns:
    The (input spec, weight spec) pair.
  """
  if equation == "mn,np->mp":
    return P(None, tp_axis), P(tp_axis, None)
  if equation == "TNH,NHD->TD":
    return P(None, tp_axis, None), P(tp_axis, None, None)
  raise ValueError(
      f"not a contract-parallel equation: {equation!r}; expected one of "
      f"{CONTRACT_PARALLEL_EQUATIONS}"
  )


def fixed_ar_enabled_for(prefix: str) -> bool:
  """Returns whether CANON_FIXED_AR=1 is enabled at o_proj or down_proj."""
  return os.environ.get(FIXED_AR_ENV) == "1" and (
      prefix.endswith(".down_proj") or prefix.endswith(".o_proj")
  )


def fixed_ar_mode_from_env() -> str:
  """Returns the reducer selected by CANON_FIXED_AR_GATHER/_SCATTER.

  Returns:
    ``RING`` (default), ``GATHER`` (GATHER=1) or ``SCATTER`` (GATHER=1 and
    SCATTER=1).

  Raises:
    RuntimeError: On a value other than unset/0/1, or SCATTER without GATHER.
  """
  gather_mode = os.environ.get(FIXED_AR_GATHER_ENV, "")
  if gather_mode not in ("", "0", "1"):
    raise RuntimeError(
        f"{FIXED_AR_GATHER_ENV} must be unset/0/1, got {gather_mode!r}"
    )
  scatter_mode = os.environ.get(FIXED_AR_SCATTER_ENV, "")
  if scatter_mode not in ("", "0", "1"):
    raise RuntimeError(
        f"{FIXED_AR_SCATTER_ENV} must be unset/0/1, got {scatter_mode!r}"
    )
  if scatter_mode == "1" and gather_mode != "1":
    raise RuntimeError(
        f"{FIXED_AR_SCATTER_ENV}=1 requires {FIXED_AR_GATHER_ENV}=1"
    )
  if gather_mode == "1":
    return SCATTER if scatter_mode == "1" else GATHER
  return RING


def fixed_order_tp_sum_ring(
    partial, axis_name, count: int, *, add: AddFn | None = None
):
  """P19.F4: the rank-ordered sum of per-rank partials via a ppermute ring.

  Must run inside a manual (``shard_map``) context over ``axis_name``.

  Args:
    partial: This rank's partial sum.
    axis_name: The TP mesh axis.
    count: The TP degree (size of ``axis_name``).
    add: Binary addition ``add(acc, operand) -> acc`` applied once per rank step
      in rank order; ``acc`` is the running sum (rank 0's partial at the first
      step) and ``operand`` the next rank's partial.  ``None`` (the default) is
      ``hard_bf16_add``: bf16 partials round after every step under either
      ``--xla_allow_excess_precision`` setting, other dtypes add as-is.  Pass
      ``plain_add`` for a plain ``+`` chain (strict-flag bits only under the
      strict flag), or any other hook; the association order never changes.

  Returns:
    ``((p_0 + p_1) + p_2) + ...`` in rank order, identical on every rank.
  """
  add = _resolve_add(add)
  p = partial[None]
  parts = [p]
  current = p
  for _ in range(count - 1):
    current = lax.ppermute(
        current, axis_name, [(i, (i + 1) % count) for i in range(count)]
    )
    parts.append(current)
  # After h hops this rank holds the partial of rank (j - h) % count; sorting
  # by that source rank restores rank order.
  index = lax.axis_index(axis_name)
  ordered = jnp.stack(parts)[jnp.argsort((index - jnp.arange(count)) % count)]
  acc = ordered[0]
  for part_index in range(1, count):
    # Fixed order, independent of ring position and of the row chunk.
    acc = add(acc, ordered[part_index])
  return acc[0]


def fixed_order_tp_sum_gather(
    partial, axis_name, count: int, *, add: AddFn | None = None
):
  """P56.4.5a: one all_gather, then the rank-ordered sum on every rank.

  Args:
    partial: This rank's partial sum.
    axis_name: The TP mesh axis.
    count: The TP degree.
    add: See ``fixed_order_tp_sum_ring``.

  Returns:
    The rank-ordered sum, identical on every rank.
  """
  add = _resolve_add(add)
  gathered = lax.all_gather(partial, axis_name, axis=0, tiled=False)
  acc = gathered[0]
  for part_index in range(1, count):
    acc = add(acc, gathered[part_index])
  return acc


def fixed_order_tp_sum_scatter(
    partial, axis_name, count: int, *, add: AddFn | None = None
):
  """P2b: the same rank-ordered sum with 2/count of the bytes.

  Rows are cut into ``count`` blocks.  all_to_all hands block ``r`` of every
  rank's partial to rank ``r``, stacked by source rank exactly as all_gather
  concatenates by axis index, so rank ``r`` adds its block with the identical
  operands in the identical association order (rank 0 + rank 1 + ...) as the
  gather form.  A tiled all_gather then places the completed blocks back in
  row order on every rank: each element is computed once instead of on every
  rank, and the bytes received per rank drop from (count-1) full partials to
  2(count-1)/count of one.  Row counts the TP degree does not divide keep the
  gather form (a static shape decision, so the program identity is
  unchanged).

  Args:
    partial: This rank's partial sum, rows first.
    axis_name: The TP mesh axis.
    count: The TP degree.
    add: See ``fixed_order_tp_sum_ring``.

  Returns:
    The rank-ordered sum, identical on every rank.
  """
  add = _resolve_add(add)
  rows = int(partial.shape[0])
  if count <= 1 or rows % count:
    return fixed_order_tp_sum_gather(partial, axis_name, count, add=add)
  blocks = partial.reshape((count, rows // count) + tuple(partial.shape[1:]))
  received = lax.all_to_all(
      blocks, axis_name, split_axis=0, concat_axis=0, tiled=False
  )
  acc = received[0]
  for part_index in range(1, count):
    acc = add(acc, received[part_index])
  return lax.all_gather(acc, axis_name, axis=0, tiled=True)


def fixed_order_tp_sum(
    partial,
    axis_name,
    count: int,
    *,
    mode: str = RING,
    add: AddFn | None = None,
):
  """Dispatches to the ring/gather/scatter form (all bitwise equal).

  Args:
    partial: This rank's partial sum, rows first.
    axis_name: The TP mesh axis.
    count: The TP degree.
    mode: ``RING``, ``GATHER`` or ``SCATTER``.
    add: See ``fixed_order_tp_sum_ring`` (``None``: ``hard_bf16_add``).

  Returns:
    The rank-ordered sum, identical on every rank.
  """
  if mode == RING:
    return fixed_order_tp_sum_ring(partial, axis_name, count, add=add)
  if mode == GATHER:
    return fixed_order_tp_sum_gather(partial, axis_name, count, add=add)
  if mode == SCATTER:
    return fixed_order_tp_sum_scatter(partial, axis_name, count, add=add)
  raise ValueError(f"unknown fixed-order reduction mode {mode!r}; {MODES}")


def p59_local_tp_context(*, mesh, tp_axis: str) -> bool:
  """Returns whether P59 already maps the live engine over DP and TP.

  P59 (``CANON_P59_RANK_PARALLEL_BACKWARD=1``) runs the trainer's backward
  inside one outer ``shard_map`` that is manual over ("data", "model").  A
  projection traced there is already TP-local and must not re-enter its own
  ``shard_map``.

  Args:
    mesh: The live engine mesh.
    tp_axis: The engine's TP axis name (must be "model" under P59).

  Returns:
    True iff the P59 flag is set and the abstract mesh is the manual
    ("data", "model") map of the same topology as ``mesh``.

  Raises:
    RuntimeError: If the manual context and the engine topology disagree.
  """
  if os.environ.get(P59_RANK_PARALLEL_BACKWARD_ENV, "") != "1":
    return False
  context = jax.sharding.get_abstract_mesh()
  if tuple(context.axis_names) != ("data", "model"):
    return False
  axis_types = dict(zip(context.axis_names, context.axis_types))
  if (
      axis_types.get("data") is not jax.sharding.AxisType.Manual
      or axis_types.get("model") is not jax.sharding.AxisType.Manual
  ):
    return False
  if mesh is None or tp_axis != "model":
    raise RuntimeError(
        "P59 local projection requires the live engine model axis"
    )
  if "data" not in mesh.shape or "model" not in mesh.shape:
    raise RuntimeError("P59 local projection requires engine data/model axes")
  data_size = int(context.shape["data"])
  model_size = int(context.shape["model"])
  # The adapter owns workload admission, including the exact checked-VMA
  # singleton carrier.  This only verifies that P59's live manual map is the
  # same topology as the engine; a size-one data axis is still a valid manual
  # axis and must not depend on a diagnostic P66 selector.
  if (
      data_size < 1
      or model_size <= 1
      or data_size != int(mesh.shape["data"])
      or model_size != int(mesh.shape["model"])
  ):
    raise RuntimeError(
        "P59 local projection context and engine topology differ"
    )
  return True


def p59_fixed_order_tp_sum_f32(value, *, axis_name, count: int):
  """Sums one replicated-input cotangent in explicit TP rank order.

  The per-shard cotangent has the bf16 primal dtype.  Accumulating eight TP
  partials in bf16 introduced multiple avoidable roundings and was measurably
  farther from the fp64 oracle.  Match the canonical matmul accumulator
  contract by adding rank-ordered partials in f32, then cast once at the
  replicated-input boundary.  Barriers on both operands preserve each
  source-level association instead of leaving the f32 chain open to XLA
  reassociation.

  Args:
    value: This rank's partial cotangent.
    axis_name: The TP mesh axis.
    count: The TP degree.

  Returns:
    The completed cotangent (``value`` itself under CANON_P66_P59_CHECK_VMA=1,
    where JAX's transpose of the implicit replicated->varying cast already
    inserted the TP psum; summing again would multiply the gradient by TP).

  Raises:
    RuntimeError: If ``CANON_P66_P59_CHECK_VMA`` is set to anything but 0/1.
  """
  vma_check = os.environ.get(P66_CHECK_VMA_ENV, "0")
  if vma_check not in ("0", "1"):
    raise RuntimeError(f"{P66_CHECK_VMA_ENV} must be exactly 0 or 1")
  if vma_check == "1":
    return value
  gathered = lax.all_gather(
      value.astype(jnp.float32), axis_name, axis=0, tiled=False
  )
  total = gathered[0]
  for rank in range(1, count):
    total = lax.optimization_barrier(total) + lax.optimization_barrier(
        gathered[rank]
    )
  return total.astype(value.dtype)


def p66_replicated_tp_value(value, *, mesh, tp_axis: str):
  """Gives P59 VMA an honest replicated-TP boundary for a fixed-order sum.

  The fixed reducers produce the same completed sum on every TP rank, but
  ordinary elementwise additions leave JAX's manual-axis type as
  ``varying(model)``.  With ``check_vma=True`` that is intentionally not
  interchangeable with the replicated hidden-state contract: treating the
  copies as independent makes reverse mode feed the full cotangent to every
  copy and can multiply gradients once per transformer layer.

  A TP pmean turns that already-identical value into an invariant value and
  gives transpose the matching collective semantics.  It is applied only while
  the projection executes directly inside P59's outer manual data/model map:
  applying it to the ordinary serving map changes its forward collective
  graph and broke the strict decode/prefill identity on TP8.

  Args:
    value: A completed fixed-order sum.
    mesh: The live engine mesh.
    tp_axis: The TP axis name.

  Returns:
    ``pmean(value)`` under CANON_P66_P59_CHECK_VMA=1 in the P59 context, else
    ``value`` unchanged.
  """
  if os.environ.get(P66_CHECK_VMA_ENV, "0") != "1" or not p59_local_tp_context(
      mesh=mesh, tp_axis=tp_axis
  ):
    return value
  return lax.pmean(value, tp_axis)


def contract_parallel_projection(
    inputs,
    weight,
    *,
    equation: str,
    mesh,
    tp_axis: str,
    local_matmul: Callable[..., jax.Array],
    mode: str | None = None,
    add: AddFn | None = None,
):
  """A contract(row)-parallel projection completed by a fixed-order TP sum.

  Every rank computes its partial ``[M, K/TP] @ [K/TP, N]`` with
  ``local_matmul`` (such as the fixed-tile Pallas matmul at the model
  contract's tiles), and the partials are summed in rank order with ``mode``
  (default: from the env).

  Args:
    inputs: The global input, contracted dimension TP-sharded per
      ``fixed_specs(equation)``.
    weight: The global weight, contracted dimension TP-sharded.
    equation: One of ``CONTRACT_PARALLEL_EQUATIONS``.
    mesh: The live engine mesh.
    tp_axis: The TP axis name.
    local_matmul: ``(a2 [M, K_local], w2 [K_local, N]) -> [M, N]`` partial.
    mode: ``RING``/``GATHER``/``SCATTER``; ``None`` reads
      ``fixed_ar_mode_from_env()``.
    add: The per-step addition of the TP sum; see ``fixed_order_tp_sum_ring``
      (``None``: ``hard_bf16_add``).

  Returns:
    The replicated ``[M, N]`` projection output.
  """
  count = int(mesh.shape[tp_axis])
  input_spec, weight_spec = fixed_specs(equation, tp_axis)
  mode = fixed_ar_mode_from_env() if mode is None else mode
  if mode not in MODES:
    raise ValueError(f"unknown fixed-order reduction mode {mode!r}; {MODES}")
  _log_once(
      f"contract_parallel:{equation}:{count}:{mode}",
      "[PATHTRACE] CANON_FIXED_AR=1 %s-ordered-sum (%s, tp=%d)",
      mode,
      equation,
      count,
  )

  def local(a_local, w_local):
    a2 = a_local.reshape(a_local.shape[0], -1)
    w2 = w_local.reshape(a2.shape[1], -1)
    partial = local_matmul(a2, w2)
    acc = fixed_order_tp_sum(partial, tp_axis, count, mode=mode, add=add)
    return p66_replicated_tp_value(acc, mesh=mesh, tp_axis=tp_axis)

  if p59_local_tp_context(mesh=mesh, tp_axis=tp_axis):
    return local(inputs, weight)
  mapped = jax.shard_map(
      local,
      mesh=mesh,
      in_specs=(input_spec, weight_spec),
      out_specs=P(None, None),
      check_vma=False,
  )
  return mapped(inputs, weight)


def _p59_replicate_input(x, *, axis_name, count: int):
  """Attaches P59 fixed-order TP cotangent reduction to a replicated input.

  HBM & Latency Optimization:
  Placing ``@jax.custom_vjp`` as an identity boundary on each TP-replicated
  input operand (saving ``None`` as its residual) rather than wrapping the
  entire ``local(*operands)`` function and calling ``jax.vjp(local, *operands)``
  inside ``bwd`` avoids:
  1. Re-executing the primal of ``local(*operands)`` (i.e., re-launching the
     forward Pallas ``padded_matmul`` / ``pallas_rmsnorm`` / ``fixed_lm_head``
     kernels a second time during the backward pass).
  2. Saving a duplicate outer ``operands`` tuple in HBM on top of the inner
     ``custom_vjp`` residuals.
  Both the forward primal and the backward cotangents remain 100% bitwise
  identical.

  Args:
    x: The TP-replicated input operand.
    axis_name: The TP mesh axis.
    count: The TP degree.

  Returns:
    ``x`` unchanged, with the fixed-order TP cotangent reduction attached.
  """

  @jax.custom_vjp
  def op(val):
    return val

  def fwd(val):
    return val, None

  def bwd(_, cotangent):
    return (
        p59_fixed_order_tp_sum_f32(cotangent, axis_name=axis_name, count=count),
    )

  op.defvjp(fwd, bwd)
  return op(x)


def p59_column_parallel(
    local: Callable[..., jax.Array],
    *args,
    replicated: Sequence[bool],
    axis_name,
    count: int,
):
  """Runs a TP-local column-parallel body with the P59 fixed-order VJP.

  Inside P59's outer manual ("data", "model") map, the TP-replicated inputs
  of a column-parallel projection (hidden states, RMSNorm gamma) receive one
  partial cotangent per TP rank.  Attaching ``_p59_replicate_input`` to each
  replicated input keeps ``local`` verbatim as the primal and completes those
  cotangents with ``p59_fixed_order_tp_sum_f32`` during backward without
  re-running ``local``'s forward Pallas kernels; TP-sharded inputs (the weight
  slice) keep their local cotangent.

  Args:
    local: The TP-local body ``local(*args)``.
    *args: Its operands.
    replicated: Per operand, whether it is TP-replicated.
    axis_name: The TP mesh axis.
    count: The TP degree.

  Returns:
    ``local(*args)`` with the fixed-order VJP attached.
  """
  replicated = tuple(bool(flag) for flag in replicated)
  if len(replicated) != len(args):
    raise ValueError(
        f"replicated has {len(replicated)} flags for {len(args)} operands"
    )
  mapped_args = tuple(
      _p59_replicate_input(arg, axis_name=axis_name, count=count)
      if is_replicated
      else arg
      for arg, is_replicated in zip(args, replicated, strict=True)
  )
  return local(*mapped_args)


def _embed_p67_p59_context() -> bool:
  """P67 scope check (requires a data axis larger than one)."""
  scope = os.environ.get(P67_SCOPE_ENV, "0")
  if scope not in ("0", "1"):
    raise RuntimeError(f"{P67_SCOPE_ENV} must be exactly 0 or 1")
  if scope != "1":
    return True
  context = jax.sharding.get_abstract_mesh()
  axis_types = dict(zip(context.axis_names, context.axis_types))
  return (
      os.environ.get(P59_RANK_PARALLEL_BACKWARD_ENV, "") == "1"
      and tuple(context.axis_names) == ("data", "model")
      and axis_types.get("data") is jax.sharding.AxisType.Manual
      and axis_types.get("model") is jax.sharding.AxisType.Manual
      and int(context.shape["data"]) > 1
      and int(context.shape["model"]) > 1
  )


def fixed_order_embed_local(
    ids_local,
    table_local,
    *,
    axis_name,
    count: int,
    add: AddFn | None = None,
):
  """P19.F4E local body: vocab-sharded lookup completed by the fixed ring.

  Each rank looks up the ids that fall in its vocabulary slice (zeros
  elsewhere); exactly one rank contributes a non-zero row per id, and the
  ring sums the per-rank rows in rank order (every step adds an exact zero to
  a representable value, so the result does not depend on ``add``).

  Args:
    ids_local: ``[T]`` token ids (replicated).
    table_local: ``[V/TP, D]`` this rank's slice of the embedding table.
    axis_name: The TP mesh axis.
    count: The TP degree.
    add: The per-step addition of the ring; see ``fixed_order_tp_sum_ring``.

  Returns:
    The ``[T, D]`` embeddings, identical on every rank.
  """
  j = lax.axis_index(axis_name)
  vloc = table_local.shape[0]
  li = ids_local - j * vloc
  ok = (li >= 0) & (li < vloc)
  part = jnp.where(
      ok[:, None],
      jnp.take(table_local, jnp.clip(li, 0, vloc - 1), axis=0),
      jnp.zeros((), table_local.dtype),
  )
  result = fixed_order_tp_sum_ring(part, axis_name, count, add=add)
  p67_p59_context = _embed_p67_p59_context()
  if os.environ.get(P66_CHECK_VMA_ENV, "0") == "1" and p67_p59_context:
    # The ring has completed the global vocab sum on every TP rank, but
    # elementwise additions remain typed model-varying.  Register the
    # already-identical hidden value as invariant so reverse mode does not
    # treat the TP copies as independent loss outputs.
    result = lax.pmean(result, axis_name)
  return result


def fixed_order_embed_lookup(
    ids, table, *, mesh, tp_axis: str = "model", add: AddFn | None = None
):
  """P19.F4E: the embedding lookup with a fixed-order vocab reduction.

  Args:
    ids: ``[T]`` int token ids.
    table: ``[V, D]`` embedding table, vocab-sharded over ``tp_axis``.
    mesh: The live engine mesh.
    tp_axis: The TP axis name.
    add: The per-step addition of the ring; see ``fixed_order_tp_sum_ring``.

  Returns:
    The replicated ``[T, D]`` embeddings.
  """
  count = int(mesh.shape[tp_axis])
  _log_once(
      f"embed:{count}",
      "[PATHTRACE] CANON_FIXED_AR_EMBED=1 fixed-order embed gather (tp=%d)",
      count,
  )

  def local(ids_local, table_local):
    return fixed_order_embed_local(
        ids_local, table_local, axis_name=tp_axis, count=count, add=add
    )

  mapped = jax.shard_map(
      local,
      mesh=mesh,
      in_specs=(P(None), P(tp_axis, None)),
      out_specs=P(None, None),
      check_vma=False,
  )
  return mapped(ids, table)
