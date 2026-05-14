"""Planner and executor for declarative parameter mapping."""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from absl import logging
import jax.numpy as jnp

from tunix.generate.param_mapping.runtime import (
  _align_shape,
  _align_to_model_shape,
  _apply_dtype_cast,
  _bulk_align_and_unstack,
  _get_n_shards,
  _jit_fuse_and_unstack_moe,
  _resolve_transpose_axes,
  _unstack_scanned_param,
)
from tunix.generate.param_mapping.spec import (
  derive_moe_source_keys,
    MappingError,
    MappingSpec,
    OperationRule,
    PlannedTransfer,
    RuleKey,
    Transform,
    extract_layer_index,
    flat_key_to_path,
    normalize_rule_key,
    remove_layers_n_component,
    replace_layers_n_with_layers,
)
from tunix.generate.param_mapping.structural import find_layer_component_index


def _pattern_match_captures(
    flat_key: Tuple[str, ...], pattern: RuleKey
) -> Optional[Tuple[str, ...]]:
  """Returns wildcard captures when a rule pattern matches a flat key."""
  if isinstance(pattern, tuple):
    return () if normalize_rule_key(pattern) == flat_key else None

  path = flat_key_to_path(flat_key)
  regex_parts = []
  i = 0
  while i < len(pattern):
    if pattern.startswith('**', i):
      regex_parts.append('(.*)')
      i += 2
      continue
    if pattern[i] == '*':
      regex_parts.append('([^.]+)')
      i += 1
      continue
    regex_parts.append(re.escape(pattern[i]))
    i += 1

  matched = re.fullmatch(''.join(regex_parts), path)
  if matched is None:
    return None
  return matched.groups()


def _pattern_matches(flat_key: Tuple[str, ...], pattern: RuleKey) -> bool:
  """Returns whether a flat key matches a declarative rule pattern."""
  return _pattern_match_captures(flat_key, pattern) is not None


def _wildcard_count(pattern: RuleKey) -> int:
  """Counts wildcard capture groups in a string rule pattern."""
  if isinstance(pattern, tuple):
    return 0
  count = 0
  i = 0
  while i < len(pattern):
    if pattern.startswith('**', i):
      count += 1
      i += 2
      continue
    if pattern[i] == '*':
      count += 1
    i += 1
  return count


def _render_pattern(pattern: RuleKey, captures: Tuple[str, ...]) -> Tuple[str, ...]:
  """Renders a wildcard rule pattern back into a concrete flat key tuple."""
  if isinstance(pattern, tuple):
    return normalize_rule_key(pattern)

  rendered_parts = []
  capture_index = 0
  i = 0
  while i < len(pattern):
    if pattern.startswith('**', i):
      rendered_parts.append(captures[capture_index])
      capture_index += 1
      i += 2
      continue
    if pattern[i] == '*':
      rendered_parts.append(captures[capture_index])
      capture_index += 1
      i += 1
      continue
    rendered_parts.append(pattern[i])
    i += 1
  return normalize_rule_key(''.join(rendered_parts))


def _resolve_pattern_source_groups(
    rule: OperationRule,
    src_flat: Mapping[Tuple[str, ...], Any],
    target_captures: Tuple[str, ...],
) -> Optional[Tuple[Tuple[Tuple[str, ...], ...], ...]]:
  """Resolves concrete source-key groups for a pattern-driven rule.

  Pattern rules can either render one concrete source key from the wildcard
  captures already extracted from the target key, or they can match multiple
  source keys directly. The returned tuple preserves grouping because later
  transforms such as module-list merge and concatenate depend on whether values
  came from the same logical source pattern.

  Example:
    Given:
      rule.source_patterns = ('mlp.experts.*.gate_proj', 'mlp.experts.*.up_proj')
      target_captures = ('7',)
      src_flat keys include:
        ('mlp', 'experts', '7', 'gate_proj')
        ('mlp', 'experts', '7', 'up_proj')

    The result is:
      (
          (('mlp', 'experts', '7', 'gate_proj'),),
          (('mlp', 'experts', '7', 'up_proj'),),
      )
  """
  source_groups = []
  sorted_src_keys = tuple(sorted(src_flat))
  for source_pattern in rule.source_patterns:
    if _wildcard_count(source_pattern) == len(target_captures):
      rendered_key = _render_pattern(source_pattern, target_captures)
      if rendered_key not in src_flat:
        return None
      source_groups.append((rendered_key,))
      continue

    matches = tuple(
        key for key in sorted_src_keys if _pattern_matches(key, source_pattern)
    )
    if not matches:
      return None
    source_groups.append(matches)
  return tuple(source_groups)


def _resolve_structural_source_groups(
    resolver: str,
    target_key: Tuple[str, ...],
    src_flat: Mapping[Tuple[str, ...], Any],
) -> Optional[Tuple[Tuple[Tuple[str, ...], ...], ...]]:
  """Resolves structural fallback source groups for one target key.

  Structural rules do not start from explicit source patterns. Instead they use
  naming conventions such as exact key matches, `layers_N -> layers` rewrites,
  implicit layer omission, or paired MoE source derivation. This helper keeps
  those conventions centralized so rule ordering stays declarative.
  """
  layer_component_index = find_layer_component_index(target_key)

  if resolver == 'direct' and target_key in src_flat:
    return ((target_key,),)

  if resolver == 'scanned_layer' and layer_component_index is not None:
    candidate = replace_layers_n_with_layers(target_key)
    if candidate in src_flat:
      return ((candidate,),)

  if resolver == 'implicit_layers' and layer_component_index is not None:
    candidate = remove_layers_n_component(target_key)
    if candidate in src_flat:
      return ((candidate,),)

  if (
      resolver == 'fused_moe'
      and layer_component_index is not None
      and target_key
      and target_key[-1] == 'wi'
  ):
    scanned_prefix = (
        target_key[:layer_component_index]
        + ('layers',)
        + target_key[layer_component_index + 1 : -1]
    )
    wi_0_key, wi_1_key = derive_moe_source_keys(scanned_prefix + ('wi',))
    if wi_0_key in src_flat and wi_1_key in src_flat:
      return ((wi_0_key, wi_1_key),)

  return None


def resolve_operation_target_key(
    target_key: Tuple[str, ...],
    src_flat: Mapping[Tuple[str, ...], Any],
    operation_rules: Tuple[OperationRule, ...],
) -> Optional[PlannedTransfer]:
  """Finds the first ordered rule that can satisfy one target leaf.

  Planning is target-driven: for each destination leaf we scan the ordered rule
  list, check whether the target pattern matches, resolve the candidate source
  groups for that rule, and then capture enough metadata to materialize the
  transfer later. The first successful rule wins.

  Example:
    Input:
      target_key = ('decoder', 'layers_1', 'mlp', 'weight')
      src_flat contains ('decoder', 'layers', 'mlp', 'weight')
      operation_rules = default_structural_operation_rules()

    Output:
      PlannedTransfer(
          rule_name='scanned_layers',
          source_keys=(( 'decoder', 'layers', 'mlp', 'weight'),),
          target_key=('decoder', 'layers_1', 'mlp', 'weight'),
          ...,
      )
  """
  for rule in operation_rules:
    for target_index, target_pattern in enumerate(rule.target_patterns):
      target_captures = _pattern_match_captures(target_key, target_pattern)
      if target_captures is None:
        continue

      for resolver in rule.source_resolvers:
        if resolver == 'pattern':
          source_groups = _resolve_pattern_source_groups(
              rule, src_flat, target_captures
          )
        else:
          source_groups = _resolve_structural_source_groups(
              resolver, target_key, src_flat
          )
        if source_groups is None:
          continue

        return PlannedTransfer(
            rule_name=rule.name,
            source_keys=tuple(key for group in source_groups for key in group),
            target_key=target_key,
            transforms=rule.transforms,
            source_groups=source_groups,
            target_index=target_index,
        )
  return None


def build_transfer_plan(
    src_flat: Mapping[Tuple[str, ...], Any],
    tgt_flat: Mapping[Tuple[str, ...], Any],
    operation_rules: Tuple[OperationRule, ...],
) -> Tuple[PlannedTransfer, ...]:
  """Builds one ordered transfer step for each reachable target leaf.

  The planner never mutates tensors. It only records which rule, source keys,
  grouping information, and transforms should apply to each destination leaf.
  Callers must provide the ordered rule set explicitly so policy selection
  happens before planning.

  Example:
    Input:
      src_flat = {('decoder', 'layers', 'mlp', 'weight'): ...}
      tgt_flat = {('decoder', 'layers_0', 'mlp', 'weight'): ...}

    Output:
      (
          PlannedTransfer(
              rule_name='scanned_layers',
              source_keys=(( 'decoder', 'layers', 'mlp', 'weight'),),
              target_key=('decoder', 'layers_0', 'mlp', 'weight'),
              ...,
          ),
      )
  """
  plan = []
  for tgt_key in tgt_flat:
    step = resolve_operation_target_key(tgt_key, src_flat, operation_rules)
    if step is not None:
      plan.append(step)
  return tuple(plan)


def build_explicit_mapping_spec(
    unscanned_src_to_tgt_flat: Mapping[Tuple[str, str], Tuple[Any, Any]],
    key_mapping_hook_fns: Optional[Mapping[str, Callable[[Any], Any]]] = None,
    transpose_keys: Optional[Dict[str, Tuple[int, ...]]] = None,
    rollout_engine: Optional[str] = None,
) -> Tuple[MappingSpec, Dict[Tuple[str, ...], Any], Dict[Tuple[str, ...], Any]]:
  """Builds planner-ready explicit rules from resolved legacy mapping entries.

  Explicit mapping integrations start from already-resolved source/target leaf
  pairs. This helper synthesizes one operation rule per mapping so the legacy
  surface can execute through the same declarative planner used by structural
  transfer.

  Each synthesized rule bakes in the necessary per-entry transforms such as
  transpose, optional hooks, shape alignment, and dtype casting.

  Example:
    Input:
      unscanned_src_to_tgt_flat = {
          ('encoder.weight', 'decoder.weight'): (src_val, tgt_param),
      }
      transpose_keys = {'encoder.weight': (1, 0)}

    Output:
      (
          MappingSpec(model_type='explicit', operation_rules=(...,)),
          {('__mapped__', 'encoder.weight', 'decoder.weight'): src_val},
          {('decoder', 'weight'): tgt_param.value},
      )
  """
  operation_rules = []
  resolved_src_flat = {}
  resolved_tgt_flat = {}

  for (flat_src_key, flat_tgt_key), (val, tgt_param) in unscanned_src_to_tgt_flat.items():
    target_key = normalize_rule_key(flat_tgt_key)
    source_key = ('__mapped__', flat_src_key, flat_tgt_key)
    transforms = []

    transpose_axes = _resolve_transpose_axes(
        flat_src_key, transpose_keys, rollout_engine
    )
    if transpose_axes is not None:
      transforms.append(Transform.transpose(transpose_axes))

    if key_mapping_hook_fns and flat_src_key in key_mapping_hook_fns:
      transforms.append(Transform.hook(key_mapping_hook_fns[flat_src_key]))

    transforms.append(Transform.align_shape(source_path=flat_src_key))
    transforms.append(Transform.cast_to_target())

    resolved_src_flat[source_key] = val
    resolved_tgt_flat[target_key] = (
        tgt_param.value if hasattr(tgt_param, 'value') else tgt_param
    )
    operation_rules.append(
      OperationRule(
            name='explicit_mapping',
        source_patterns=(source_key,),
        target_patterns=(target_key,),
        source_resolvers=('pattern',),
            transforms=tuple(transforms),
        )
    )

  return (
      MappingSpec(
          model_type='explicit',
          operation_rules=tuple(operation_rules),
      ),
      resolved_src_flat,
      resolved_tgt_flat,
  )


def _collapse_singletons(values):
  """Collapses one-element grouped-value containers after group resolution.

  Group transforms operate on lists because some rule patterns may resolve to
  multiple tensors. Once an operation leaves only one tensor in a group, later
  transforms usually want the tensor itself rather than a length-one list.
  """
  collapsed = []
  for value in values:
    if isinstance(value, list) and len(value) == 1:
      collapsed.append(value[0])
    else:
      collapsed.append(value)
  return collapsed


def _apply_transfer_operation(
    transform: Transform,
    current_values,
    step: PlannedTransfer,
):
  """Executes a multi-source or multi-target group transform.

  These operations run before the standard unary tensor transform pipeline
  because they change how many tensors are flowing through the step. Examples
  include stacking module-list weights, concatenating multiple sources into one
  tensor, or selecting one chunk for a multi-target split rule.
  """
  if transform.kind == 'merge_modulelist':
    axis = transform.args.get('dim', 0)
    return [
        jnp.stack(group_values, axis=axis)
        if len(group_values) > 1
        else group_values[0]
        for group_values in current_values
    ]

  if transform.kind == 'concatenate':
    current_values = _collapse_singletons(current_values)
    if any(isinstance(value, list) for value in current_values):
      raise MappingError(
          f'concatenate requires tensor inputs, got nested groups for {step.rule_name}'
      )
    return [jnp.concatenate(current_values, axis=transform.args['dim'])]

  if transform.kind == 'chunk':
    current_values = _collapse_singletons(current_values)
    if len(current_values) != 1 or isinstance(current_values[0], list):
      raise MappingError(f'chunk requires one tensor source for {step.rule_name}')
    if step.target_index is None:
      raise MappingError(
          f'chunk requires a resolved target index for {step.rule_name}'
      )
    return [
        jnp.split(
            current_values[0],
            transform.args['chunks'],
            axis=transform.args['dim'],
        )[step.target_index]
    ]

  raise MappingError(f'Unsupported group transform: {transform.kind}')


def _transform_copy(**kwargs):
  """Returns the source value unchanged."""
  return kwargs['src_val']


def _transform_cast_to_target(**kwargs):
  """Casts the source value to the target dtype when necessary."""
  return _apply_dtype_cast(
      kwargs['src_val'], kwargs['tgt_val'].dtype, kwargs['source_path']
  )


def _transform_transpose(**kwargs):
  """Applies a declarative transpose transform to the current source value."""
  return jnp.transpose(kwargs['src_val'], kwargs['transform'].args['axes'])


def _transform_hook(**kwargs):
  """Runs a caller-supplied hook over the current source value."""
  return kwargs['transform'].args['fn'](kwargs['src_val'])


def _transform_unstack_scanned(**kwargs):
  """Slices one layer from a scanned parameter, aligning first when required.

  Some scanned weights already match the target per-layer shape once the scan
  axis is removed; others need a bulk align operation before the layer slice is
  extracted. The result is cached per source tensor and target shape to avoid
  repeating expensive align/unstack work across layers.
  """
  key_tuple = kwargs['key_tuple']
  layer_idx = extract_layer_index(key_tuple)
  assert layer_idx is not None
  cache_key = (
      kwargs['cache_source_key'],
      kwargs['tgt_val'].shape,
      kwargs['scan_axis'],
      'aligned',
  )
  if cache_key not in kwargs['value_cache']:
    src_val = kwargs['src_val']
    tgt_val = kwargs['tgt_val']
    scan_axis = kwargs['scan_axis']
    source_path = kwargs['source_path']
    scanned_per_layer_shape = src_val.shape[:scan_axis] + src_val.shape[scan_axis + 1 :]
    if scanned_per_layer_shape == tgt_val.shape:
      kwargs['value_cache'][cache_key] = _unstack_scanned_param(
          src_val, tgt_val, source_path, scan_axis=scan_axis
      )
    else:
      logging.info(
          'Bulk-aligning scanned %s: %s -> per-layer %s',
          source_path,
          src_val.shape,
          tgt_val.shape,
      )
      kwargs['value_cache'][cache_key] = _bulk_align_and_unstack(
          src_val, scan_axis, tgt_val, source_path
      )
  return kwargs['value_cache'][cache_key][layer_idx]


def _transform_align_shape(**kwargs):
  """Aligns a value to the target shape using runtime shape heuristics."""
  align_context = dict(kwargs['transform_context'])
  rollout_engine = align_context.pop('rollout_engine', None)
  align_source_path = kwargs['transform'].args.get(
      'source_path', kwargs['source_path']
  )
  return _align_shape(
      kwargs['src_val'],
      kwargs['tgt_val'].shape,
      align_source_path,
      rollout_engine,
      **align_context,
  )


def _transform_repeat_to_target(**kwargs):
  """Repeats or pads the current value to match the target model layout."""
  return _align_to_model_shape(
      kwargs['src_val'], kwargs['tgt_val'], flat_key_to_path(kwargs['key_tuple'])
  )


def _materialize_fused_moe_transform(
    step: PlannedTransfer,
    src_flat: Mapping[Tuple[str, ...], Any],
    tgt_val: Any,
    scan_axis: int,
    value_cache: Dict[Any, Any],
    key_tuple: Tuple[str, ...],
    path_str: str,
) -> Any:
  """Fuses scanned `wi_0`/`wi_1` MoE weights into one target-layer `wi` tensor.

  This is treated as a step-level transform because it needs coordinated access
  to both source tensors, scan-axis information, target sharding-derived pad
  behavior, and per-layer slicing. The fused-and-unstacked result is cached so
  each target layer reuses one bulk computation.
  """
  layer_idx = extract_layer_index(key_tuple)
  assert layer_idx is not None
  cache_key = (step.source_keys, tgt_val.shape, scan_axis, 'fused')
  if cache_key not in value_cache:
    wi_0_key, wi_1_key = step.source_keys
    wi_0_full = src_flat[wi_0_key]
    wi_1_full = src_flat[wi_1_key]
    transform_kinds = {transform.kind for transform in step.transforms}
    if 'cast_to_target' in transform_kinds:
      wi_0_full = _apply_dtype_cast(
          wi_0_full, tgt_val.dtype, flat_key_to_path(wi_0_key)
      )
      wi_1_full = _apply_dtype_cast(
          wi_1_full, tgt_val.dtype, flat_key_to_path(wi_1_key)
      )
    num_layers = src_flat[wi_0_key].shape[scan_axis]
    wi_0_single_shape = (
        wi_0_full.shape[:scan_axis] + wi_0_full.shape[scan_axis + 1 :]
    )
    mismatched_axes = [
        i
        for i, (src_dim, tgt_dim) in enumerate(zip(wi_0_single_shape, tgt_val.shape))
        if src_dim != tgt_dim
    ]
    tgt_axis = mismatched_axes[-1] if mismatched_axes else len(tgt_val.shape) - 1
    n_shards = _get_n_shards(tgt_val, tgt_axis)
    scan_padded_axis = tgt_axis if tgt_axis < scan_axis else tgt_axis + 1
    value_cache[cache_key] = _jit_fuse_and_unstack_moe(
        wi_0_full,
        wi_1_full,
        scan_axis,
        num_layers,
        n_shards,
        tgt_val.shape,
        scan_padded_axis,
        tgt_axis,
    )
    del wi_0_full, wi_1_full

  sliced_val = value_cache[cache_key][layer_idx]
  return _align_to_model_shape(sliced_val, tgt_val, path_str)


GROUP_TRANSFORM_HANDLERS = {
    'merge_modulelist': _apply_transfer_operation,
    'concatenate': _apply_transfer_operation,
    'chunk': _apply_transfer_operation,
}


VALUE_TRANSFORM_HANDLERS = {
    'copy': _transform_copy,
    'cast_to_target': _transform_cast_to_target,
    'transpose': _transform_transpose,
    'hook': _transform_hook,
    'unstack_scanned': _transform_unstack_scanned,
    'align_shape': _transform_align_shape,
    'repeat_to_target': _transform_repeat_to_target,
}

STEP_TRANSFORM_HANDLERS = {
    'fuse_moe': _materialize_fused_moe_transform,
}


def _apply_value_transform(
    transform: Transform,
    src_val: Any,
    tgt_val: Any,
    source_path: str,
    key_tuple: Tuple[str, ...],
    scan_axis: int,
    value_cache: Dict[Any, Any],
    transform_context: Mapping[str, Any],
    cache_source_key: Tuple[str, ...],
) -> Any:
  """Dispatches one value transform through `VALUE_TRANSFORM_HANDLERS`.

  This helper applies only transforms that consume a single current source
  value and return a single updated value. Group transforms and step transforms
  are handled separately because they operate on grouped inputs or on the full
  step materialization context.
  """
  try:
    handler = VALUE_TRANSFORM_HANDLERS[transform.kind]
  except KeyError as exc:
    raise MappingError(f'Unsupported transfer transform: {transform.kind}') from exc

  return handler(
      transform=transform,
      src_val=src_val,
      tgt_val=tgt_val,
      source_path=source_path,
      key_tuple=key_tuple,
      scan_axis=scan_axis,
      value_cache=value_cache,
      transform_context=transform_context,
      cache_source_key=cache_source_key,
  )


def materialize_transfer_step(
    step: PlannedTransfer,
    src_flat: Mapping[Tuple[str, ...], Any],
    tgt_val: Any,
    scan_axis: int,
    value_cache: Dict[Any, Any],
    transform_context: Optional[Mapping[str, Any]] = None,
) -> Any:
  """Materializes one planned transfer step into its final target value.

  The planner only decides which sources and transforms apply to a target leaf.
  This function performs the actual tensor work by:

  - resolving multi-source groups when needed
  - running group transforms like concatenate or chunk
  - running unary transforms like transpose, hook, and shape alignment
  - using specialized step handlers for transforms such as fused MoE materialization

  The output is a single tensor already aligned to the target leaf semantics.

  Example:
    Input step:
      source_keys = (('attn', 'Wqkv'),)
      target_key = ('self_attn', 'q_proj')
      transforms = (Transform.chunk(dim=0, chunks=3),)

    Output:
      The first chunk of src_flat[('attn', 'Wqkv')] along axis 0, ready to be
      written into ('self_attn', 'q_proj').
  """

  if transform_context is None:
    transform_context = {}
  key_tuple = step.target_key
  path_str = flat_key_to_path(key_tuple)

  for transform in step.transforms:
    if transform.kind in STEP_TRANSFORM_HANDLERS:
      return STEP_TRANSFORM_HANDLERS[transform.kind](
          step=step,
          src_flat=src_flat,
          tgt_val=tgt_val,
          scan_axis=scan_axis,
          value_cache=value_cache,
          key_tuple=key_tuple,
          path_str=path_str,
      )

  current_values = None
  src_val = None
  source_path = None

  if any(transform.kind in GROUP_TRANSFORM_HANDLERS for transform in step.transforms):
    current_values = [
        [src_flat[source_key] for source_key in source_group]
        for source_group in (
            step.source_groups or tuple((key,) for key in step.source_keys)
        )
    ]

  for transform in step.transforms:
    if transform.kind == 'fuse_moe':
      continue
    if transform.kind in GROUP_TRANSFORM_HANDLERS:
      current_values = GROUP_TRANSFORM_HANDLERS[transform.kind](
          transform, current_values, step
      )
      continue

    if src_val is None:
      if current_values is not None:
        if len(current_values) != 1 or isinstance(current_values[0], list):
          raise MappingError(
              f'OperationRule {step.rule_name} did not resolve to one tensor value'
          )
        src_val = current_values[0]
        source_path = flat_key_to_path(step.source_keys[0])
      else:
        if len(step.source_keys) != 1:
          raise MappingError(
              f'Unsupported multi-source transfer without fuse_moe: {step.source_keys}'
          )
        source_key = step.source_keys[0]
        src_val = src_flat[source_key]
        source_path = flat_key_to_path(source_key)

    src_val = _apply_value_transform(
        transform=transform,
        src_val=src_val,
        tgt_val=tgt_val,
        source_path=source_path,
        key_tuple=key_tuple,
        scan_axis=scan_axis,
        value_cache=value_cache,
        transform_context=transform_context,
        cache_source_key=step.source_keys[0],
    )

  if src_val is None and current_values is not None:
    if len(current_values) != 1 or isinstance(current_values[0], list):
      raise MappingError(
          f'OperationRule {step.rule_name} did not resolve to one tensor value'
      )
    src_val = current_values[0]

  if src_val is None:
    raise MappingError(f'Rule {step.rule_name} did not materialize a source value')

  return src_val


def execute_transfer_plan(
    plan: Tuple[PlannedTransfer, ...],
    src_flat: Mapping[Tuple[str, ...], Any],
    tgt_flat: Mapping[Tuple[str, ...], Any],
    scan_axis: int,
    transform_context: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[Tuple[str, ...], Any], Dict[Tuple[str, ...], Any]]:
  """Executes a full transfer plan into filtered flat source and target trees.

  The returned dictionaries contain only the target leaves that were planned.
  The source-side result is keyed by target key because each entry is already
  the final materialized value to apply to that destination leaf.

  Example:
    Input:
      plan = (PlannedTransfer(target_key=('decoder', 'layer0', 'weight'), ...),)

    Output:
      (
          {('decoder', 'layer0', 'weight'): materialized_value},
          {('decoder', 'layer0', 'weight'): original_target_leaf},
      )
  """
  filtered_src_flat = {}
  filtered_tgt_flat = {}
  value_cache = {}

  for step in plan:
    tgt_val = tgt_flat[step.target_key]
    filtered_src_flat[step.target_key] = materialize_transfer_step(
        step, src_flat, tgt_val, scan_axis, value_cache, transform_context
    )
    filtered_tgt_flat[step.target_key] = tgt_val

  return filtered_src_flat, filtered_tgt_flat
