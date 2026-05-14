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
"""Structural fallback rules for parameter mapping."""

from __future__ import annotations

import re
from typing import Optional, Tuple

from tunix.generate.param_mapping.spec import (
    MappingSpec,
    OperationRule,
    Transform,
)


_LAYER_COMPONENT_RE = re.compile(r'^layers_(\d+)$')


def find_layer_component_index(target_key: Tuple[str, ...]) -> Optional[int]:
  """Returns the position of the first concrete scanned-layer component.

  Structural matching rules need to know where a `layers_N` segment sits in the
  target key so they can rewrite only that part of the path when mapping back to
  shared source tensors.
  """
  for i, part in enumerate(target_key):
    if _LAYER_COMPONENT_RE.match(str(part)):
      return i
  return None


def default_structural_operation_rules() -> Tuple[OperationRule, ...]:
  """Returns the ordered fallback rules for structural direct transfer.

  The order is significant:

  - try exact key matches first
  - then try scanned-layer rewrites
  - then try scanned fused-MoE reconstruction

  Keeping these rules explicit makes the direct-transfer path inspectable and
  reusable by tests and by custom converters that want the same default policy.

    Example:
        Input target key:
            ('decoder', 'layers_2', 'mlp', 'weight')

        Typical resolution order:
            1. Try exact match against the same tuple key.
            2. Try scanned-layer rewrite to ('decoder', 'layers', 'mlp', 'weight').
            3. If the target leaf were `wi`, try fused-MoE reconstruction.
  """
  return (
      OperationRule(
          name='direct_match',
          target_patterns=('**',),
          source_resolvers=('direct',),
          transforms=(
              Transform.cast_to_target(),
              Transform.repeat_to_target(),
          ),
      ),
      OperationRule(
          name='scanned_layers',
          target_patterns=('layers_*.**', '**.layers_*.**'),
          source_resolvers=('scanned_layer', 'implicit_layers'),
          transforms=(
              Transform.cast_to_target(),
              Transform.unstack_scanned(),
              Transform.repeat_to_target(),
          ),
      ),
      OperationRule(
          name='scanned_fused_moe',
          target_patterns=(
              'layers_*.wi',
              'layers_*.**.wi',
              '**.layers_*.wi',
              '**.layers_*.**.wi',
          ),
          source_resolvers=('fused_moe',),
          transforms=(
              Transform.cast_to_target(),
              Transform.fuse_moe(axis=-1),
              Transform.unstack_scanned(),
          ),
      ),
  )


def make_structural_mapping_spec() -> MappingSpec:
  """Builds a complete `MappingSpec` wrapper around structural fallback rules."""
  return MappingSpec(
      model_type='structural',
      operation_rules=default_structural_operation_rules(),
  )
