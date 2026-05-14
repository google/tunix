"""Public transfer APIs for declarative parameter mapping."""

from __future__ import annotations

import abc as py_abc
from collections import abc
from dataclasses import dataclass
import gc
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from absl import logging
from flax import nnx
from flax import traverse_util

from tunix.generate.param_mapping.engine import (
    build_explicit_mapping_spec,
    build_transfer_plan,
    execute_transfer_plan,
)
from tunix.generate.param_mapping.spec import (
  OperationRule,
  flat_key_to_path,
)
from tunix.generate.param_mapping.transfer_support import (
  build_flat_dict,
  delete_target_buffers,
  fuse_moe_weights,
  reshard_in_chunks,
  snapshot_dst_sharding,
  sync_tied_lm_head_if_needed,
  unroll_scanned_layers,
)
from tunix.generate.param_mapping.structural import default_structural_operation_rules


def _flatten_target_values(
    tgt_flat_list: list[tuple[tuple[str, ...], Any]],
) -> Dict[Tuple[str, ...], Any]:
  """Builds a plain flat target dict from a destination flat-state list.

  The explicit-mappings converter needs a mutable, key-addressable view of the
  full destination state so it can overlay only the transferred values while
  preserving untouched leaves. This helper normalizes mixed leaf types by
  reading `.value` when the leaf is an NNX variable-like object and otherwise
  keeping the leaf as-is.
  """
  return {
      key: tgt_params.value if hasattr(tgt_params, 'value') else tgt_params
      for key, tgt_params in tgt_flat_list
  }


@dataclass(frozen=True)
class PreparedTransfer:
  """Flat transfer inputs consumed by the shared planner and executor."""

  src_flat: Mapping[Tuple[str, ...], Any]
  tgt_flat: Mapping[Tuple[str, ...], Any]
  scan_axis: int = 1
  transform_context: Optional[Mapping[str, Any]] = None
  operation_rules: Optional[Tuple[OperationRule, ...]] = None


class BaseConverter(py_abc.ABC):
  """Abstract converter contract for the shared transfer lifecycle."""

  @py_abc.abstractmethod
  def prepare_transfer(
      self,
      src_state: Any,
      dst_state: Any,
      **kwargs,
  ) -> PreparedTransfer:
    """Builds planner-ready flat source and target mappings.

    This is the model-specific preparation hook. Implementations are expected to
    normalize whatever source and destination objects they receive into the
    narrow `PreparedTransfer` contract consumed by the shared planner and
    executor.

    Typical responsibilities include unwrapping framework-specific containers,
    flattening parameter trees, attaching transform context, and selecting the
    ordered `OperationRule`s that should drive planning.
    """

  @py_abc.abstractmethod
  def reshard_transfer(
      self,
      filtered_src_flat: Mapping[Tuple[str, ...], Any],
      filtered_tgt_flat: Mapping[Tuple[str, ...], Any],
      dst_state: Any,
      reshard_fn: Callable[..., Mapping[str, Any]],
      delete_dst_buffers: bool = False,
      reshard_chunk_size: Optional[int] = None,
      **kwargs,
  ) -> Any:
    """Converts executor outputs into destination-compatible sharded values.

    The shared executor produces a flat mapping keyed by destination leaves.
    Converters can use this hook to call the common reshard helper, invoke an
    alternate reshard API, or merge subset updates back into a larger target
    structure before final writeback.
    """

  @py_abc.abstractmethod
  def finalize_transfer(
      self,
      dst_state: Any,
      resharded_weights: Any,
      **kwargs,
  ) -> Any:
    """Writes prepared weights into the destination object and returns it.

    This is the last stage of the lifecycle. Simple structural transfers often
    just call `nnx.update(...)`, while compatibility layers may need to rebuild
    flat state objects or preserve framework-specific tied-weight semantics.
    """


class DirectTransferConverter(BaseConverter):
  """Concrete converter for structural direct transfers."""

  def __init__(
      self,
      scan_axis: int = 1,
      transform_context: Optional[Mapping[str, Any]] = None,
  ):
    self._scan_axis = scan_axis
    self._transform_context = transform_context

  def prepare_transfer(
      self,
      src_state: Any,
      dst_state: Any,
      **kwargs,
  ) -> PreparedTransfer:
    """Runs the stock structural preparation path for direct transfers.

    Direct transfer intentionally stays thin. The converter only chooses the
    scan axis and optional transform context, then delegates the real source and
    destination normalization work to `prepare_structural_transfer(...)`.
    """
    transform_context = kwargs.get('transform_context', self._transform_context)
    return prepare_structural_transfer(
        src_state,
        dst_state,
        scan_axis=kwargs.get('scan_axis', self._scan_axis),
        transform_context=transform_context,
    )

  def reshard_transfer(
      self,
      filtered_src_flat: Mapping[Tuple[str, ...], Any],
      filtered_tgt_flat: Mapping[Tuple[str, ...], Any],
      dst_state: Any,
      reshard_fn: Callable[..., Mapping[str, Any]],
      delete_dst_buffers: bool = False,
      reshard_chunk_size: Optional[int] = None,
      **kwargs,
  ) -> Any:
    """Delegates unchanged to the shared structural reshard helper.

    Structural direct transfer does not need a custom reshaping layer between
    executor output and destination writeback, so it reuses the framework-owned
    helper exactly as-is.
    """
    return reshard_transfer(
        filtered_src_flat=filtered_src_flat,
        filtered_tgt_flat=filtered_tgt_flat,
        dst_state=dst_state,
        reshard_fn=reshard_fn,
        delete_dst_buffers=delete_dst_buffers,
        reshard_chunk_size=reshard_chunk_size,
        **kwargs,
    )

  def finalize_transfer(
      self,
      dst_state: Any,
      resharded_weights: Any,
      **kwargs,
  ) -> Any:
    """Writes resharded weights back with the default `nnx.update(...)` path."""
    del kwargs
    nnx.update(dst_state, resharded_weights)
    return dst_state


class _ExplicitMappingsConverter(BaseConverter):
  """Internal converter that adapts explicit key mappings to the shared flow."""

  def __init__(
      self,
      key_mappings,
      key_mapping_hook_fns=None,
      transpose_keys=None,
      rollout_engine=None,
      extra_transform_context: Optional[Mapping[str, Any]] = None,
  ):
    self._key_mappings = key_mappings
    self._key_mapping_hook_fns = key_mapping_hook_fns
    self._transpose_keys = transpose_keys
    self._rollout_engine = rollout_engine
    self._extra_transform_context = dict(extra_transform_context or {})
    self._tgt_flat_list = None
    self._transferred_target_keys = set()
    self._sharding_dict = None

  def prepare_transfer(
      self,
      src_state: Any,
      dst_state: Any,
      **kwargs,
  ) -> PreparedTransfer:
    """Builds explicit planner inputs from legacy key-mapping metadata.

    The explicit mapping compatibility path still starts from `(src_key ->
    target_key)` tables, optional transpose rules, and optional hook functions.
    This method resolves those legacy inputs into:

    - flat source leaves keyed by synthetic planner keys
    - flat target leaves keyed by normalized target tuples
    - declarative `OperationRule`s that the shared planner can execute

    The destination flat-state list and per-leaf shardings are cached because
    later lifecycle stages need the full target state, not just the subset of
    mapped leaves.
    """
    del kwargs
    self._transferred_target_keys = set()
    self._tgt_flat_list = dst_state.flat_state()
    self._sharding_dict = {
      key: (
        tgt_params.value.sharding
        if hasattr(tgt_params, 'value')
        else tgt_params.sharding
      )
      for key, tgt_params in self._tgt_flat_list
    }
    src_to_tgt_map = build_flat_dict(self._tgt_flat_list, self._key_mappings)
    unscanned_src_to_tgt_flat = unroll_scanned_layers(src_state, src_to_tgt_map)
    mapping_spec, resolved_src_flat, resolved_tgt_flat = build_explicit_mapping_spec(
        unscanned_src_to_tgt_flat,
        key_mapping_hook_fns=self._key_mapping_hook_fns,
        transpose_keys=self._transpose_keys,
        rollout_engine=self._rollout_engine,
    )
    return PreparedTransfer(
        src_flat=resolved_src_flat,
        tgt_flat=resolved_tgt_flat,
        scan_axis=0,
        transform_context={
            'rollout_engine': self._rollout_engine,
            **self._extra_transform_context,
        },
        operation_rules=mapping_spec.operation_rules,
    )

  def reshard_transfer(
      self,
      filtered_src_flat: Mapping[Tuple[str, ...], Any],
      filtered_tgt_flat: Mapping[Tuple[str, ...], Any],
      dst_state: Any,
      reshard_fn: Callable[..., Mapping[str, Any]],
      delete_dst_buffers: bool = False,
      reshard_chunk_size: Optional[int] = None,
      **kwargs,
  ) -> Any:
    """Reshards explicit-mapping updates and overlays them onto the full target.

    The shared reshard helper only knows about the subset of leaves produced by
    the executor. That is correct for normal `nnx.update(...)` flows, but the
    explicit-mappings path reconstructs a full destination flat tree in
    `finalize_transfer(...)`.

    This override therefore reuses the shared reshard implementation for the
    actual reshard work, then merges the resulting updates back into a complete
    target-flat dictionary. That preserves advanced shared behavior such as
    chunked resharding and destination-buffer deletion without losing unmapped
    destination leaves.
    """
    assert self._tgt_flat_list is not None

    tgt_param_by_path = {
        flat_key_to_path(key): key for key, _ in self._tgt_flat_list
    }

    if reshard_fn is None:
      resharded_updates_flat = dict(filtered_src_flat)
    else:
      resharded_updates = reshard_transfer(
          filtered_src_flat=filtered_src_flat,
          filtered_tgt_flat=filtered_tgt_flat,
          dst_state=dst_state,
          reshard_fn=reshard_fn,
          delete_dst_buffers=delete_dst_buffers,
          reshard_chunk_size=reshard_chunk_size,
          **kwargs,
      )
      resharded_updates_flat = traverse_util.flatten_dict(resharded_updates)

    updated_tgt_flat = _flatten_target_values(self._tgt_flat_list)
    self._transferred_target_keys = set()
    for target_key, val in resharded_updates_flat.items():
      actual_target_key = tgt_param_by_path[flat_key_to_path(target_key)]
      updated_tgt_flat[actual_target_key] = val
      self._transferred_target_keys.add(flat_key_to_path(actual_target_key))
    return updated_tgt_flat

  def finalize_transfer(
      self,
      dst_state: Any,
      resharded_weights: Any,
      **kwargs,
  ) -> Any:
    """Writes finalized flat weights back into the explicit destination state.

    Unlike the default converter, this path does not call `nnx.update(...)`
    because explicit mappings may target destination objects whose public update
    surface is a flat-path reconstruction API. The input here is expected to be
    a complete flat target mapping, not only the subset that was transferred.

    After assignment, the method preserves legacy tied-`lm_head` behavior by
    syncing from embedding weights only when `lm_head` was not explicitly
    transferred.
    """
    assert self._tgt_flat_list is not None
    del kwargs

    for tgt_key, tgt_param in self._tgt_flat_list:
      assert tgt_key in resharded_weights, f'Key {tgt_key} not in finalized values'
      if hasattr(tgt_param, 'value'):
        tgt_param.value = resharded_weights[tgt_key]

    sync_tied_lm_head_if_needed(self._tgt_flat_list, self._transferred_target_keys)
    gc.collect()

    return dst_state.from_flat_path(self._tgt_flat_list)


def _safe_has_key(obj: Mapping[str, Any], key: str) -> bool:
  """Returns whether a mapping-like object exposes a given key.

  Some callers pass plain dictionaries while others pass mapping-like objects
  that expose fields through attributes. This helper keeps the wrapper-unwrapping
  logic tolerant to both representations.
  """
  if isinstance(obj, dict):
    return key in obj

  return hasattr(obj, key)


def _unwrap_source_state(src_state: Any) -> Any:
  """Removes the common `base` wrapper used by some source states.

  Direct transfer historically accepted trainer-owned states that wrap the real
  parameter tree under `base`. The planner should operate on the underlying
  parameter tree, not on the wrapper container.
  """
  if isinstance(src_state, abc.Mapping) and _safe_has_key(src_state, 'base'):
    logging.info("Unwrapping 'base' key from source state.")
    return src_state['base']
  return src_state


def _unwrap_target_state(dst_state: Any) -> Any:
  """Removes one or more nested `model` wrappers from a destination state.

  Rollout and serving stacks often nest the actual parameter state under one or
  more `model` containers. Structural transfer should plan against the leaf
  parameter tree, so this helper repeatedly unwraps that common container shape.
  """
  while isinstance(dst_state, abc.Mapping) and _safe_has_key(dst_state, 'model'):
    logging.info("Unwrapping nested 'model' key from target state.")
    dst_state = dst_state['model']
  return dst_state


def _to_pure_spec(node: Any) -> Any:
  """Converts NNX-style state containers into plain Python/JAX tree leaves.

  The planner and structural matcher operate over flattened pytrees. This helper
  recursively strips wrapper objects such as `nnx.Variable`, objects with a
  `.value` field, and state containers with `to_pure_dict()`, producing a tree
  that can be flattened deterministically.
  """
  if hasattr(node, 'to_pure_dict'):
    node = node.to_pure_dict()

  if isinstance(node, abc.Mapping):
    return {k: _to_pure_spec(v) for k, v in node.items()}

  if isinstance(node, nnx.Variable):
    return _to_pure_spec(node[...])
  if hasattr(node, 'value'):
    return node.value

  return node


def prepare_structural_transfer(
    src_state: Any,
    dst_state: Any,
    scan_axis: int = 1,
    transform_context: Optional[Mapping[str, Any]] = None,
) -> PreparedTransfer:
  """Builds the default planner input for structural direct transfer.

  This helper is the shared preparation pipeline behind `transfer_state_directly`
  and any converter that wants the same default semantics. It:

  - unwraps common source and destination containers
  - converts wrapper objects into plain pytree leaves
  - flattens both trees into tuple-key dictionaries
  - opportunistically fuses unscanned MoE `wi_0`/`wi_1` pairs when the target
    expects a fused `wi`

  The output is intentionally narrow so the downstream planner/executor remain
  the single source of truth for rule resolution and materialization.

  This preparation step also attaches the default structural rule set so the
  planner consumes an explicit policy rather than choosing one implicitly.

  Example:
    Input:
      src_state = {'base': {'layers': {'wi_0': ..., 'wi_1': ...}}}
      dst_state = {'model': {'layers': {'wi': ...}}}

    Output:
      PreparedTransfer(
          src_flat={('layers', 'wi'): fused_value},
          tgt_flat={('layers', 'wi'): target_value},
          scan_axis=1,
      )
  """
  src_state = _unwrap_source_state(src_state)
  dst_state = _unwrap_target_state(dst_state)

  full_source_dict = _to_pure_spec(src_state)
  full_target_spec = _to_pure_spec(dst_state)
  src_flat = traverse_util.flatten_dict(full_source_dict)
  tgt_flat = traverse_util.flatten_dict(full_target_spec)
  src_flat = fuse_moe_weights(src_flat, tgt_flat)

  return PreparedTransfer(
      src_flat=src_flat,
      tgt_flat=tgt_flat,
      scan_axis=scan_axis,
      transform_context=transform_context,
      operation_rules=default_structural_operation_rules(),
  )


def reshard_transfer(
    filtered_src_flat: Mapping[Tuple[str, ...], Any],
    filtered_tgt_flat: Mapping[Tuple[str, ...], Any],
    dst_state: Any,
    reshard_fn: Callable[..., Mapping[str, Any]],
    delete_dst_buffers: bool = False,
    reshard_chunk_size: Optional[int] = None,
    **kwargs,
) -> Any:
  """Applies the shared resharding path for executor outputs.

  The executor produces a flat mapping from target keys to materialized source
  values. This helper converts that subset into the shape expected by the
  supplied `reshard_fn`, optionally snapshots destination shardings, optionally
  deletes destination buffers before the reshard call, and optionally processes
  the work in chunks to reduce peak HBM.

  The return value matches the default `finalize_transfer(...)` expectation:
  either a nested update tree or, when chunked, an unflattened tree rebuilt from
  the chunk results.

  Example:
    Input:
      filtered_src_flat = {('decoder', 'weight'): src_value}
      filtered_tgt_flat = {('decoder', 'weight'): tgt_leaf}
      reshard_fn = lambda source, target: source

    Output:
      {'decoder': {'weight': src_value}}
  """
  del dst_state, kwargs

  if reshard_chunk_size is not None:
    resharded_flat = reshard_in_chunks(
        dict(filtered_src_flat),
        dict(filtered_tgt_flat),
        reshard_fn,
        reshard_chunk_size,
        delete_dst_buffers,
    )
    return traverse_util.unflatten_dict(resharded_flat)

  dst_shardings_flat = {
      k: snapshot_dst_sharding(
          tgt_val.value if hasattr(tgt_val, 'value') else tgt_val
      )
      for k, tgt_val in filtered_tgt_flat.items()
  }

  if delete_dst_buffers:
    delete_target_buffers(filtered_tgt_flat, filtered_src_flat)

  return reshard_fn(
      source=traverse_util.unflatten_dict(dict(filtered_src_flat)),
      target=traverse_util.unflatten_dict(dst_shardings_flat),
  )


def transfer_state_with_converter(
    src_state: Any,
    dst_state: Any,
  converter: BaseConverter,
    reshard_fn: Callable[..., Mapping[str, Any]],
    delete_dst_buffers: bool = False,
    reshard_chunk_size: Optional[int] = None,
    **kwargs,
) -> Any:
  """Runs the full transfer lifecycle through a converter and the shared engine.

  The lifecycle is intentionally split into three phases:

  1. `prepare_transfer(...)` turns model-specific inputs into flat planner data.
  2. `reshard_transfer(...)` performs any required sharding conversion on the
     executor output.
  3. `finalize_transfer(...)` applies the prepared values to the destination.

  This keeps model-specific preparation and destination writeback extensible
  without fragmenting the core planner and executor logic.

  Example:
    Input:
      converter = DirectTransferConverter(scan_axis=0)
      reshard_fn = lambda source, target: source

    Output:
      The destination state object returned by `converter.finalize_transfer(...)`
      after planning, materialization, and resharding have completed.
  """
  prepared = converter.prepare_transfer(src_state, dst_state, **kwargs)
  transfer_plan = build_transfer_plan(
      prepared.src_flat,
      prepared.tgt_flat,
      operation_rules=prepared.operation_rules,
  )
  filtered_src_flat, filtered_tgt_flat = execute_transfer_plan(
      transfer_plan,
      prepared.src_flat,
      prepared.tgt_flat,
      prepared.scan_axis,
      transform_context=prepared.transform_context,
  )
  resharded_weights = converter.reshard_transfer(
      filtered_src_flat=filtered_src_flat,
      filtered_tgt_flat=filtered_tgt_flat,
      dst_state=dst_state,
      reshard_fn=reshard_fn,
      delete_dst_buffers=delete_dst_buffers,
      reshard_chunk_size=reshard_chunk_size,
      **kwargs,
  )
  return converter.finalize_transfer(
      dst_state,
      resharded_weights,
      reshard_fn=reshard_fn,
      delete_dst_buffers=delete_dst_buffers,
      reshard_chunk_size=reshard_chunk_size,
      **kwargs,
  )


def transfer_state_with_mappings(
    src_state,
    dst_state,
    key_mappings,
    key_mapping_hook_fns=None,
    transpose_keys=None,
    reshard_fn=None,
    rollout_engine=None,
    **kwargs,
):
  """Transfers state through the explicit-mapping compatibility wrapper.

  This preserves the long-standing Tunix integration surface based on mapping
  tables, optional transpose rules, optional hook functions, and optional
  rollout-engine-specific shape handling. Internally it now lowers those inputs
  into declarative operation rules and then executes them through the shared
  planner and converter lifecycle.
  """
  return transfer_state_with_converter(
      src_state,
      dst_state,
      _ExplicitMappingsConverter(
          key_mappings=key_mappings,
          key_mapping_hook_fns=key_mapping_hook_fns,
          transpose_keys=transpose_keys,
          rollout_engine=rollout_engine,
          extra_transform_context=kwargs,
      ),
      reshard_fn=reshard_fn,
      **kwargs,
  )


def transfer_state_directly(
    src_state: Mapping[str, Any],
    dst_state: Mapping[str, Any],
    reshard_fn: Callable[..., Mapping[str, Any]],
    scan_axis: int = 1,
    delete_dst_buffers: bool = False,
    reshard_chunk_size: Optional[int] = None,
) -> None:
  """Transfers state by structural matching with the default converter.

  This is the public convenience wrapper for the common case where source and
  destination trees are structurally compatible after wrapper normalization. It
  intentionally stays thin and delegates all real work to the shared converter
  pipeline.
  """
  transfer_state_with_converter(
      src_state,
      _unwrap_target_state(dst_state),
      DirectTransferConverter(scan_axis=scan_axis),
      reshard_fn=reshard_fn,
      delete_dst_buffers=delete_dst_buffers,
      reshard_chunk_size=reshard_chunk_size,
  )
