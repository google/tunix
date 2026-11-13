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

"""Declarative HF -> vLLM-JAX weight-conversion manifests."""

from __future__ import annotations

from collections.abc import Mapping
import functools
import types
from typing import Any, Protocol, runtime_checkable

from flax import nnx
import jax
import numpy as np
from tunix.dist_rl.weight_sync import weight_sync
from tunix.generate.vllm_hf_manifests import qwen3
from tunix.models import hf_mapping


@runtime_checkable
class HfModelConfig(Protocol):
  """Minimal Hugging Face config contract for manifest dispatch."""

  @property
  def model_type(self) -> str:
    ...


class _FamilyManifestBuilder(Protocol):
  """Callable signature for per-family HF -> vLLM-JAX manifest builders."""

  def __call__(
      self,
      hf_config: Any,
      *,
      tp_size: int = 1,
      head_sharding_size: int | None = None,
      vocab_size: int | None = None,
  ) -> dict[str, Any]:
    ...


_MODEL_MANIFEST_BUILDERS: Mapping[str, _FamilyManifestBuilder] = (
    types.MappingProxyType({
        "qwen3": qwen3.build_from_hf_manifest,
    })
)


def build_from_hf_manifest(
    hf_config: HfModelConfig,
    *,
    tp_size: int = 1,
    head_sharding_size: int | None = None,
    vocab_size: int | None = None,
) -> dict[str, Any]:
  """Builds the declarative `HF -> vLLM-JAX` JSON manifest for `hf_config`."""
  model_type = hf_config.model_type
  if not isinstance(model_type, str) or not model_type:
    raise ValueError(
        f"hf_config.model_type must be a non-empty str, got {model_type!r}."
    )
  try:
    builder = _MODEL_MANIFEST_BUILDERS[model_type]
  except KeyError as exc:
    raise NotImplementedError(
        "Declarative HF-to-vLLM-JAX manifest is not yet implemented for"
        f" model_type={model_type!r}."
    ) from exc
  return builder(
      hf_config,
      tp_size=tp_size,
      head_sharding_size=head_sharding_size,
      vocab_size=vocab_size,
  )


def derive_hf_variables_from_manifest(
    dst_spec: hf_mapping.RuleSpec,
    dst_state: nnx.State | Mapping[Any, Any],
) -> tuple[weight_sync.TensorMetadata, ...]:
  """Derives canonical HF TensorMetadata from a manifest and target state."""
  if isinstance(dst_state, nnx.State):
    flat_by_path = {
        ".".join(str(k) for k in path): var.value
        for path, var in dst_state.flat_state()
        if isinstance(var, nnx.Variable)
    }
  elif isinstance(dst_state, Mapping):
    flat_by_path = {
        (k if isinstance(k, str) else ".".join(str(p) for p in k)): (
            v.value if isinstance(v, nnx.Variable) else v
        )
        for k, v in dst_state.items()
    }
  else:
    raise TypeError(
        f"dst_state must be an nnx.State or Mapping, got {type(dst_state)}"
    )

  parsed = hf_mapping.parse_manifest(dst_spec)
  dropped = set(parsed["dropped_keys"])
  required_dst_keys: set[str] = set()
  for r in parsed["rules"]:
    if not dropped.intersection((*r["source_keys"], *r["target_keys"])):
      required_dst_keys.update(r["target_keys"])
  missing_dst = sorted(required_dst_keys - flat_by_path.keys())
  if missing_dst:
    raise KeyError(
        "Destination state missing target keys required by HF manifest:"
        f" {missing_dst}"
    )

  abstract_dst = {
      k: jax.ShapeDtypeStruct(
          shape=tuple(flat_by_path[k].shape), dtype=flat_by_path[k].dtype
      )
      for k in required_dst_keys
  }
  inv_manifest = hf_mapping.invert_manifest(parsed)
  abstract_hf = jax.eval_shape(
      functools.partial(hf_mapping.to_hf_impl, inv_manifest),
      abstract_dst,
  )
  return tuple(
      weight_sync.TensorMetadata(
          name=hf_key,
          shape=tuple(struct.shape),
          mesh_shape=(1,) * len(struct.shape),
          layout=tuple(range(len(struct.shape) - 1, -1, -1)),
          item_size=int(np.dtype(struct.dtype).itemsize),
      )
      for hf_key, struct in sorted(abstract_hf.items())
  )
