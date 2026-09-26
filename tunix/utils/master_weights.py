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

"""fp32 master weights for optax that keep each parameter leaf's own dtype.

`master_weights(inner)` wraps an optax transformation (MaxText's `optax.adamw`)
so that:

  * every float leaf narrower than fp32 (bf16/fp16/fp8) gets an fp32 master
    copy held in the optimizer state, and the inner transformation (moments,
    weight decay, bias correction) runs on that master;
  * every leaf that is already fp32 -- the norm / gate / A_log / dt_bias /
    conv1d params MaxText keeps in fp32 under `float32_gate_logits=True` -- gets
    NO copy: the parameter is its own master and the inner update passes through
    untouched, so that leaf follows plain fp32 optax bit for bit;
  * leaves an enclosing `optax.multi_transform` / `optax.masked` hands in as
    `optax.MaskedNode` (MaxText's `trainable_parameters_mask` freeze) get no
    master and no moments;
  * after `optax.apply_updates(params, updates)` each leaf has the dtype it came
    in with. Nothing is blanket-cast, so the weights the trainer syncs to the
    rollout keep their dtype and layout.

For a lower-precision leaf p (dtype d) with master m and inner update u:

    new_m  = m + u                              (fp32)
    target = reduce_precision(new_m, d)         (round-to-nearest-even, in fp32)
    emit   = target - p.astype(fp32)            (fp32)

`target` uses `jax.lax.reduce_precision`, not `new_m.astype(d).astype(fp32)`:
under jit, XLA:TPU excess precision (on by default) may drop that convert pair,
which turns the emit into `new_m - p` and leaves p 1 ulp off RNE(new_m) when the
master sits near a bf16 rounding midpoint (measured on v5p; invisible in HLO
and on CPU). reduce_precision cannot be elided. For fp16 it flushes subnormal
targets to zero (bf16 is unaffected), and TPU flushes f32 subnormals anyway.
Check exactness on the host (device_get, then compare bits), never inside jit.

`optax.apply_updates` computes `(p + emit).astype(d)`. Both `target` and `p` are
d-representable, so whenever their fp32 difference is exact the sum is exactly
`target` and the parameter becomes exactly `new_m.astype(d)`. Measured on 2M
pairs (master_weights_test.py::test_exactness_sweep): exact whenever
floor(log2(|target|/|p|)) >= -16, which covers every sign change and every exact
zero. The only inexact corner is 0 < |target| < 2^-16 |p| (a parameter shrinking
by >4 orders of magnitude in one step), where the bf16 parameter is off by at
most 2^-24 |p|; nothing accumulates, because the next step re-targets from the
master rather than from the parameter.

Pure JAX and elementwise, so jit/pjit friendly: master leaves are created with
`p.astype(fp32)` and inherit p's sharding.

`install_in_maxtext_get_optimizer()` is a scoped context manager that makes
MaxText's own `optimizers.get_optimizer` build `master_weights(optax.adamw(...))`
as its base optimizer, so MaxText's `skip_step_on_spikes` and
`apply_trainable_parameters_mask` wrap it exactly as they wrap plain adamw.
"""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import threading
import types
from typing import Any, Callable, Iterator, NamedTuple

import jax
import jax.numpy as jnp
import optax

try:
  from flax import nnx as _nnx  # pylint: disable=g-import-not-at-top

  _VARIABLE_TYPES: tuple[type, ...] = (_nnx.Variable,)
except ImportError:  # pragma: no cover - flax is a tunix dependency
  _VARIABLE_TYPES = ()


class MasterWeightsState(NamedTuple):
  """State of `master_weights`.

  master: a `master_dtype` copy of every lower-precision float parameter leaf,
    and `optax.MaskedNode()` in place of every other leaf (already-fp32 leaves,
    non-float leaves, and leaves an enclosing mask froze).
  inner_state: the wrapped transformation's state, built on the master view.
  """

  master: Any
  inner_state: Any


@dataclasses.dataclass(frozen=True)
class MasterWeightsStats:
  """What `master_weights` found in the parameter tree it was initialized on.

  mastered: leaves that got a master copy (float narrower than the master).
  mastered_dtypes: count of mastered leaves per original dtype name.
  mastered_elements: total elements across the mastered leaves.
  passthrough: leaves with no copy that the inner optimizer updates directly
    (fp32 or wider, or non-float).
  frozen: leaves an enclosing mask replaced by `optax.MaskedNode` (no master,
    no moments).
  master_bytes: bytes the master copies occupy, summed over the global shapes.
  """

  mastered: int
  mastered_dtypes: tuple[tuple[str, int], ...]
  mastered_elements: int
  passthrough: int
  frozen: int
  master_bytes: int


def _is_placeholder(x: Any) -> bool:
  return isinstance(x, optax.MaskedNode)


def needs_master(p: Any, master_dtype: Any = jnp.float32) -> bool:
  """True for float leaves narrower than `master_dtype` (bf16, fp16, fp8 under fp32)."""
  dt = getattr(p, "dtype", None)
  if dt is None or not jnp.issubdtype(dt, jnp.floating):
    return False
  return jnp.finfo(dt).bits < jnp.finfo(jnp.dtype(master_dtype)).bits


def _is_variable(x: Any) -> bool:
  return bool(_VARIABLE_TYPES) and isinstance(x, _VARIABLE_TYPES)


def _make_master(x: Any, master_dtype: Any) -> Any:
  """The master for one leaf, or a placeholder.

  `nnx.Optimizer` hands `init` the params still wrapped in `nnx.Param`, and its
  `to_opt_state` turns every Variable in the returned state into an
  `OptVariable` carrying the param's sharding metadata. A master keeps that
  wrapper (rebuilt the way `jax.tree.map` would), so it is placed like the
  param. A placeholder replaces the whole wrapper with a bare `MaskedNode`: an
  `OptVariable` around a `MaskedNode` has no shape, which `nnx.eval_shape`
  (MaxText's AbstractMaxTextEngine) cannot handle.
  """
  if _is_variable(x):
    leaves, treedef = jax.tree.flatten(x)
    if len(leaves) == 1 and needs_master(leaves[0], master_dtype):
      return jax.tree.unflatten(treedef, [leaves[0].astype(master_dtype)])
    return optax.MaskedNode()
  return x.astype(master_dtype) if needs_master(x, master_dtype) else optax.MaskedNode()


def master_view(params: Any, master: Any) -> Any:
  """The full-precision view of `params`: the master where one exists, else the leaf."""
  return jax.tree.map(lambda p, m: p if _is_placeholder(m) else m, params, master, is_leaf=_is_variable)


def param_stats(params: Any, master_dtype: Any = jnp.float32) -> MasterWeightsStats:
  """Classifies every leaf of `params` the way `master_weights` will treat it."""
  master_dtype = jnp.dtype(master_dtype)
  mastered = passthrough = frozen = elements = 0
  by_dtype: collections.Counter[str] = collections.Counter()
  for leaf in jax.tree.leaves(params, is_leaf=_is_placeholder):
    if _is_placeholder(leaf):
      frozen += 1
    elif needs_master(leaf, master_dtype):
      mastered += 1
      by_dtype[jnp.dtype(leaf.dtype).name] += 1
      size = 1
      for dim in getattr(leaf, "shape", ()):
        size *= int(dim)
      elements += size
    else:
      passthrough += 1
  return MasterWeightsStats(
      mastered=mastered,
      mastered_dtypes=tuple(sorted(by_dtype.items())),
      mastered_elements=elements,
      passthrough=passthrough,
      frozen=frozen,
      master_bytes=elements * master_dtype.itemsize,
  )


def master_weights(
    inner: optax.GradientTransformation,
    master_dtype: Any = jnp.float32,
    on_init: Callable[[MasterWeightsStats], None] | None = None,
) -> optax.GradientTransformationExtraArgs:
  """Runs `inner` on an fp32 master copy while keeping every parameter leaf's dtype.

  Args:
    inner: the optimizer to run on the master weights (e.g. `optax.adamw(...)`).
      Extra keyword arguments passed to `update` (MaxText's skip_step_on_spikes
      `loss` / `grad_norm`) are forwarded to it.
    master_dtype: dtype of the master copy. Leaves at least this wide get no copy.
    on_init: called with the leaf classification every time `init` runs (at
      trace time under jit / eval_shape).

  Returns:
    An `optax.GradientTransformationExtraArgs`. `update` requires `params`.
  """
  master_dtype = jnp.dtype(master_dtype)
  inner = optax.with_extra_args_support(inner)

  def init_fn(params):
    if on_init is not None:
      on_init(param_stats(params, master_dtype))
    master = jax.tree.map(lambda x: _make_master(x, master_dtype), params, is_leaf=_is_variable)
    return MasterWeightsState(master=master, inner_state=inner.init(master_view(params, master)))

  def update_fn(updates, state, params=None, **extra_args):
    if params is None:
      raise ValueError("master_weights needs `params` to know each leaf's dtype.")
    master = state.master
    view = master_view(params, master)
    # Master-backed leaves feed the inner optimizer fp32 gradients; leaves with
    # no master pass through unchanged, so they see exactly what plain `inner`
    # would.
    inner_grads = jax.tree.map(
        lambda g, m: g if _is_placeholder(m) else g.astype(master_dtype), updates, master
    )
    inner_updates, new_inner_state = inner.update(inner_grads, state.inner_state, view, **extra_args)

    new_master = jax.tree.map(
        lambda u, m: m if _is_placeholder(m) else m + u.astype(master_dtype), inner_updates, master
    )

    def emit(p, u, m, nm):
      if _is_placeholder(m):
        return u  # no master: identical to applying `inner` directly
      # Round the new master to p's dtype with reduce_precision, not an astype
      # round trip: under jit, XLA:TPU excess precision (default on) may drop a
      # f32->bf16->f32 convert pair, which leaves p 1 ulp off RNE(master) near
      # rounding midpoints. reduce_precision cannot be elided.
      fi = jnp.finfo(p.dtype)
      target = jax.lax.reduce_precision(nm, exponent_bits=fi.nexp, mantissa_bits=fi.nmant)
      return target - p.astype(master_dtype)

    emitted = jax.tree.map(emit, params, inner_updates, master, new_master)
    return emitted, MasterWeightsState(master=new_master, inner_state=new_inner_state)

  return optax.GradientTransformationExtraArgs(init_fn, update_fn)


def find_master_states(opt_state: Any) -> list[MasterWeightsState]:
  """Every `MasterWeightsState` nested anywhere in `opt_state`."""
  leaves = jax.tree.leaves(opt_state, is_leaf=lambda x: isinstance(x, MasterWeightsState))
  return [x for x in leaves if isinstance(x, MasterWeightsState)]


def materialize_params(params: Any, state: MasterWeightsState) -> Any:
  """`master.astype(p.dtype)` for master-backed leaves, `p` elsewhere (bit-exact)."""
  return jax.tree.map(lambda p, m: p if _is_placeholder(m) else m.astype(p.dtype), params, state.master)


def adam_moment_dtypes(opt_state: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
  """Sorted distinct dtype names of every Adam `mu` and `nu` leaf in `opt_state`."""
  adam_states = [
      x
      for x in jax.tree.leaves(opt_state, is_leaf=lambda x: isinstance(x, optax.ScaleByAdamState))
      if isinstance(x, optax.ScaleByAdamState)
  ]

  def dtypes(trees):
    return tuple(sorted({jnp.dtype(leaf.dtype).name for t in trees for leaf in jax.tree.leaves(t)}))

  return dtypes([s.mu for s in adam_states]), dtypes([s.nu for s in adam_states])


# ---------------------------------------------------------------------------------------------
# Scoped installation into MaxText's get_optimizer.
# ---------------------------------------------------------------------------------------------


@dataclasses.dataclass
class InstallRecord:
  """What happened while `install_in_maxtext_get_optimizer` was active.

  adamw_calls: how many times MaxText's `optimizers.get_optimizer` asked for
    `optax.adamw` (and got it wrapped).
  adamw_kwargs: the keyword arguments of the last such call (mu_dtype etc.).
  init_stats: one entry per `init` of a wrapper built in scope. The AOT engine
    traces `init` more than once; every entry must agree.
  """

  adamw_calls: int = 0
  adamw_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
  init_stats: list[MasterWeightsStats] = dataclasses.field(default_factory=list)


class _OptaxWithMasterAdamw(types.ModuleType):
  """Stands in for `optax` inside `maxtext.optimizers.optimizers` while installed.

  Every attribute resolves to the real optax except `adamw`, which returns
  `master_weights(optax.adamw(...))`.
  """

  def __init__(self, real: types.ModuleType, master_dtype: Any, record: InstallRecord):
    super().__init__(real.__name__, getattr(real, "__doc__", None))
    self._real = real
    self._master_dtype = master_dtype
    self._record = record

  def __getattr__(self, name: str) -> Any:
    # Only reached for names not set on the proxy itself.
    if name.startswith("_") and name in ("_real", "_master_dtype", "_record"):
      raise AttributeError(name)
    return getattr(self._real, name)

  def adamw(self, *args: Any, **kwargs: Any) -> optax.GradientTransformationExtraArgs:
    self._record.adamw_calls += 1
    self._record.adamw_kwargs = dict(kwargs)
    return master_weights(
        self._real.adamw(*args, **kwargs),
        master_dtype=self._master_dtype,
        on_init=self._record.init_stats.append,
    )


_INSTALL_LOCK = threading.Lock()


@contextlib.contextmanager
def install_in_maxtext_get_optimizer(
    optimizers_module: types.ModuleType | None = None,
    master_dtype: Any = jnp.float32,
) -> Iterator[InstallRecord]:
  """Makes MaxText's `get_optimizer` wrap its `optax.adamw` in `master_weights`.

  While active, the module-global `optax` that
  `maxtext.optimizers.optimizers.get_optimizer` resolves at call time is a proxy
  whose `adamw` returns `master_weights(optax.adamw(...))`. Everything else in
  `get_optimizer` runs unmodified, so the result is
  `apply_trainable_parameters_mask(skip_step_on_spikes(master_weights(adamw)))`
  with MaxText's own wrappers in MaxText's own order. The real `optax` is put
  back on exit, even on error. The swap is process-global, so only one scope
  may be active at a time; hold it only around engine construction.

  Args:
    optimizers_module: the module to patch; defaults to
      `maxtext.optimizers.optimizers`.
    master_dtype: dtype of the master copies.

  Yields:
    An `InstallRecord` the caller checks after construction (see
    `verify_installed`).
  """
  if optimizers_module is None:
    from maxtext.optimizers import optimizers as optimizers_module  # pylint: disable=g-import-not-at-top

  real = getattr(optimizers_module, "optax", None)
  if real is None:
    raise RuntimeError(
        f"{optimizers_module.__name__} has no module-level `optax`; cannot install fp32 master"
        " weights into its get_optimizer."
    )
  if not _INSTALL_LOCK.acquire(blocking=False):
    raise RuntimeError("install_in_maxtext_get_optimizer is already active; nesting is not supported.")
  try:
    if isinstance(real, _OptaxWithMasterAdamw):
      raise RuntimeError(f"{optimizers_module.__name__}.optax is already the master-weights proxy.")
    record = InstallRecord()
    optimizers_module.optax = _OptaxWithMasterAdamw(real, master_dtype, record)
    try:
      yield record
    finally:
      optimizers_module.optax = real
  finally:
    _INSTALL_LOCK.release()


def verify_installed(record: InstallRecord, opt_state: Any, master_dtype: Any = jnp.float32) -> MasterWeightsStats:
  """Checks that exactly one wrapper was built and initialized into `opt_state`.

  Args:
    record: the record yielded by `install_in_maxtext_get_optimizer`.
    opt_state: the built optimizer state (pure or nnx-wrapped leaves).
    master_dtype: the expected dtype of every master leaf.

  Returns:
    The leaf classification of the one wrapper.

  Raises:
    RuntimeError: if MaxText did not call `optax.adamw` exactly once, the wrapper
      was never initialized, its init saw different trees, or `opt_state` does not
      hold exactly one `MasterWeightsState` whose masters match that
      classification.
  """
  master_dtype = jnp.dtype(master_dtype)
  if record.adamw_calls != 1:
    raise RuntimeError(
        "fp32 master weights: expected MaxText's get_optimizer to build optax.adamw exactly once"
        f" while installed, saw {record.adamw_calls} call(s). The optimizer was built some other"
        " way (opt_type != adamw, or MaxText's get_optimizer changed), so the master is not in place."
    )
  if not record.init_stats:
    raise RuntimeError("fp32 master weights: the wrapper was built but never initialized.")
  stats = record.init_stats[-1]
  if any(s != stats for s in record.init_stats):
    raise RuntimeError(f"fp32 master weights: init saw different parameter trees: {record.init_stats}")
  states = find_master_states(opt_state)
  if len(states) != 1:
    raise RuntimeError(
        f"fp32 master weights: expected exactly one MasterWeightsState in the optimizer state,"
        f" found {len(states)}."
    )
  master_leaves = jax.tree.leaves(states[0].master)
  if len(master_leaves) != stats.mastered:
    raise RuntimeError(
        f"fp32 master weights: optimizer state holds {len(master_leaves)} master arrays, init"
        f" classified {stats.mastered} leaves as needing one."
    )
  wrong = sorted({jnp.dtype(x.dtype).name for x in master_leaves} - {master_dtype.name})
  if wrong:
    raise RuntimeError(f"fp32 master weights: master leaves have dtype(s) {wrong}, expected {master_dtype.name}.")
  return stats


def summary_line(stats: MasterWeightsStats, mu_dtypes: tuple[str, ...], nu_dtypes: tuple[str, ...]) -> str:
  """The one startup line create_maxtext_engine logs."""
  names = {name for name, _ in stats.mastered_dtypes}
  kind = "bf16" if names <= {"bfloat16"} else "low-precision (" + ", ".join(
      f"{n} {name}" for name, n in stats.mastered_dtypes
  ) + ")"
  return (
      f"fp32 master weights: ON, {stats.mastered} {kind} leaves mastered,"
      f" {stats.passthrough} fp32 leaves passthrough, {stats.frozen} frozen"
      f" (master: {stats.mastered_elements} elements, {stats.master_bytes / 2**30:.2f} GiB global;"
      f" adam mu dtype={'/'.join(mu_dtypes) or 'none'}, nu dtype={'/'.join(nu_dtypes) or 'none'})"
  )
