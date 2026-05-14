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
"""Declarative parameter mapping specification types and helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Callable, Mapping, Optional, Tuple


RuleKey = str | Tuple[str, ...]


class MappingError(ValueError):
  """Raised when key mappings are invalid or missing."""

  pass


@dataclass(frozen=True)
class Transform:
  """Declarative transfer operator used by the mapping engine."""

  kind: str
  args: Mapping[str, Any] = field(default_factory=dict)

  @classmethod
  def copy(cls) -> 'Transform':
    """Builds a no-op transform for rules that only redirect keys."""
    return cls('copy')

  @classmethod
  def cast_to_target(cls) -> 'Transform':
    """Builds a transform that casts the source tensor to the target dtype."""
    return cls('cast_to_target')

  @classmethod
  def transpose(cls, axes: Tuple[int, ...]) -> 'Transform':
    """Builds a transform that permutes tensor axes before later steps run."""
    return cls('transpose', {'axes': axes})

  @classmethod
  def hook(cls, fn: Callable[[Any], Any]) -> 'Transform':
    """Builds a transform that delegates tensor rewriting to caller code."""
    return cls('hook', {'fn': fn})

  @classmethod
  def unstack_scanned(cls) -> 'Transform':
    """Builds a transform that slices one layer from a scanned source tensor."""
    return cls('unstack_scanned')

  @classmethod
  def align_shape(cls, source_path: Optional[str] = None) -> 'Transform':
    """Builds a transform that aligns one source tensor to the target shape.

    `source_path` is optional metadata used for clearer diagnostics and for
    runtime heuristics that depend on the originating logical weight name.
    """
    args = {}
    if source_path is not None:
      args['source_path'] = source_path
    return cls('align_shape', args)

  @classmethod
  def repeat_to_target(cls) -> 'Transform':
    """Builds a transform that repeats or pads to a model-specific target layout."""
    return cls('repeat_to_target')

  @classmethod
  def fuse_moe(cls, axis: int = -1) -> 'Transform':
    """Builds a transform that fuses paired MoE expert tensors into one tensor."""
    return cls('fuse_moe', {'axis': axis})

  @classmethod
  def merge_modulelist(cls, dim: int = 0) -> 'Transform':
    """Builds a group transform that stacks a module-list style source set."""
    return cls('merge_modulelist', {'dim': dim})

  @classmethod
  def concatenate(cls, dim: int) -> 'Transform':
    """Builds a group transform that concatenates multiple tensors along one axis."""
    return cls('concatenate', {'dim': dim})

  @classmethod
  def chunk(cls, dim: int, chunks: int) -> 'Transform':
    """Builds a group transform that splits one tensor into indexed target chunks."""
    return cls('chunk', {'dim': dim, 'chunks': chunks})


@dataclass(frozen=True)
class ExecPolicy:
  """Execution-time controls shared by all planned transfer steps."""

  reshard: bool = True
  reshard_chunk_size: Optional[int] = None
  delete_dst_buffers: bool = False
  run_gc_between_chunks: bool = True
  sync_tied_lm_head: bool = True


@dataclass(frozen=True)
class OperationRule:
  """Declarative transfer rule for key resolution and tensor processing."""

  name: str
  source_patterns: Tuple[RuleKey, ...] = ()
  target_patterns: Tuple[RuleKey, ...] = ()
  source_resolvers: Tuple[str, ...] = ('pattern',)
  transforms: Tuple[Transform, ...] = ()
  optional: bool = False


@dataclass(frozen=True)
class MappingSpec:
  """Complete declarative input to the shared transfer planner."""

  model_type: str
  exec_policy: ExecPolicy = field(default_factory=ExecPolicy)
  operation_rules: Tuple[OperationRule, ...] = ()


@dataclass(frozen=True)
class MappingProgram:
  """First-class rule collection for the mapping engine and future tooling."""

  model_type: str
  exec_policy: ExecPolicy = field(default_factory=ExecPolicy)
  operation_rules: Tuple[OperationRule, ...] = ()

  def to_mapping_spec(self) -> MappingSpec:
    """Projects a higher-level rule collection into the planner input model.

    `MappingProgram` exists so callers can assemble reusable rule collections
    without depending on the precise planner-facing container type. Today's
    engine still consumes `MappingSpec`, so this method is the compatibility
    bridge between the two layers.
    """
    return MappingSpec(
        model_type=self.model_type,
        exec_policy=self.exec_policy,
        operation_rules=self.operation_rules,
    )


@dataclass(frozen=True)
class PlannedTransfer:
  """Planner output for one target leaf transfer."""

  rule_name: str
  source_keys: Tuple[Tuple[str, ...], ...]
  target_key: Tuple[str, ...]
  transforms: Tuple[Transform, ...]
  source_groups: Tuple[Tuple[Tuple[str, ...], ...], ...] = ()
  target_index: Optional[int] = None


_LAYER_COMPONENT_RE = re.compile(r'^layers_(\d+)$')


def flat_key_to_path(flat_key: Tuple[str, ...]) -> str:
  """Converts a flat tuple key into the dot-path form used in diagnostics.

  The planner uses tuple keys for deterministic matching. Many logs, rule
  renderers, and compatibility helpers are easier to read in dot-path form, so
  this helper centralizes that conversion.
  """
  return '.'.join(str(part) for part in flat_key)


def extract_layer_index(target_key: Tuple[str, ...]) -> Optional[int]:
  """Extracts the first scanned-layer index embedded in a target key.

  Structural rules use `layers_N` path components to denote a concrete layer in
  an otherwise repeated block. This helper returns that numeric suffix when
  present so unstacking and layer-relative matching can stay centralized.
  """
  for part in target_key:
    matched = _LAYER_COMPONENT_RE.match(str(part))
    if matched:
      return int(matched.group(1))
  return None


def replace_layers_n_with_layers(
    target_key: Tuple[str, ...],
) -> Tuple[str, ...]:
  """Normalizes concrete scanned-layer keys back to their shared source form.

  Structural transfer often needs to map a target like `layers_7.*` onto a
  source tensor stored once as `layers.*`. This helper performs that reversible
  normalization step.
  """
  return tuple(
      'layers' if _LAYER_COMPONENT_RE.match(str(part)) else str(part)
      for part in target_key
  )


def remove_layers_n_component(
    target_key: Tuple[str, ...],
) -> Tuple[str, ...]:
  """Removes one concrete scanned-layer component for implicit-layer matching.

  Some source trees omit the scanned-layer path component entirely and rely on
  the destination shape to determine which slice to select. This helper supports
  that fallback matching mode.
  """
  removed = False
  rewritten = []
  for part in target_key:
    if not removed and _LAYER_COMPONENT_RE.match(str(part)):
      removed = True
      continue
    rewritten.append(str(part))
  return tuple(rewritten)


def derive_moe_source_keys(
    target_key: Tuple[str, ...],
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
  """Derives the two unfused MoE source keys for a fused `wi` target leaf.

  The fused target representation uses one `wi` key while several source trees
  store the same logical tensor as paired `wi_0` and `wi_1` weights. Keeping
  that derivation in one helper avoids reimplementing the naming convention in
  planner and transfer-support code.

  Example:
    Input:
      ('decoder', 'layers_3', 'mlp', 'wi')

    Output:
      (
          ('decoder', 'layers_3', 'mlp', 'wi_0'),
          ('decoder', 'layers_3', 'mlp', 'wi_1'),
      )
  """
  if not target_key or target_key[-1] != 'wi':
    raise MappingError(
        f'Cannot derive MoE source keys for non-wi target: {target_key}'
    )
  prefix = target_key[:-1]
  return prefix + ('wi_0',), prefix + ('wi_1',)


def normalize_rule_key(
    rule_key: str | Tuple[str, ...],
) -> Tuple[str, ...]:
  """Normalizes string or tuple rule keys into planner tuple-key form.

  The planner compares keys in tuple form so both string-based rules and tuple
  literals need a shared normalization path. Numeric path segments are converted
  back to integers where possible so tuple rules preserve the semantics of the
  original flattened pytree keys.

  Example:
    Input:
      'decoder.layers.3.weight'

    Output:
      ('decoder', 'layers', 3, 'weight')
  """

  def _normalize_part(part: Any) -> Any:
    if isinstance(part, str) and part.isdigit():
      return int(part)
    return str(part) if not isinstance(part, int) else part

  if isinstance(rule_key, tuple):
    return tuple(_normalize_part(part) for part in rule_key)
  return tuple(_normalize_part(part) for part in rule_key.split('.'))
