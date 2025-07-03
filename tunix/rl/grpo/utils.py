# Copyright 2025 Google LLC
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

"""Simple utils used by GRPO."""

from flax import nnx
from flax.nnx import statelib
import jax
import jaxtyping


def is_lora_enabled(model: nnx.Module) -> bool:
  for _, value in nnx.iter_graph(model):
    if isinstance(value, nnx.LoRAParam):
      return True
  return False


def to_flat_dict(
    tree: jaxtyping.PyTree | statelib.State,
) -> tuple[dict[tuple[str, ...], jaxtyping.Array], jaxtyping.PyTreeDef]:
  if isinstance(tree, statelib.State):
    tree = nnx.to_pure_dict(tree)
  flattened, tree_def = jax.tree.flatten_with_path(tree)
  return {tuple(k.key for k in keys): v for keys, v in flattened}, tree_def
