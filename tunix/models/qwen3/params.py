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

"""Utils for loading and converting Qwen3 PT weights."""

import re
from etils import epath
from flax import nnx
import jax
import jax.numpy as jnp
import safetensors.flax as safetensors
import tqdm
from tunix.models.qwen3 import model as model_lib


def _stack_experts(params: dict[str, jax.Array]):
  """Stack experts in the loaded pytorch params."""
  key_fn = lambda x: int(re.match(r"(.*?)experts\.([0-9]+)\..*", x).group(2))  # pytype: disable=attribute-error
  new_params = dict(params).copy()
  for kw in ["gate", "up", "down"]:
    pattern = r"(.*?)experts\.(.*?)\.{}_proj\.(.*)".format(kw)
    keys = [k for k in params.keys() if re.match(pattern, k)]
    prefix_groups = set([re.match(pattern, k).group(1) for k in keys])  # pytype: disable=attribute-error
    for prefix in prefix_groups:
      keys_to_merge = list(
          sorted([k for k in keys if k.startswith(prefix)], key=key_fn)
      )
      for k in keys_to_merge:
        del new_params[k]
      suffix = re.match(pattern, keys_to_merge[0]).group(3)  # pytype: disable=attribute-error
      with jax.default_device(jax.devices("cpu")[0]):
        new_params[f"{prefix}experts.{kw}_proj.{suffix}"] = jnp.stack(
            [params[k] for k in keys_to_merge], 0
        )
  return new_params


def _get_key_and_transform_mapping(cfg: model_lib.ModelConfig):
  # Mapping of torch_keys -> (nnx_keys, (permute_rule, reshape_rule)).
  return {
      r"model\.embed_tokens\.weight": ("embedder.input_embedding", None),
      # attention projection weights
      r"model\.layers\.([0-9]+)\.self_attn\.q_proj\.weight": (
          r"layers.\1.attn.q_proj.w",
          ((1, 0), (cfg.embed_dim, cfg.num_heads, cfg.head_dim)),
      ),
      r"model\.layers\.([0-9]+)\.self_attn\.k_proj\.weight": (
          r"layers.\1.attn.k_proj.w",
          ((1, 0), (cfg.embed_dim, cfg.num_kv_heads, cfg.head_dim)),
      ),
      r"model\.layers\.([0-9]+)\.self_attn\.v_proj\.weight": (
          r"layers.\1.attn.v_proj.w",
          ((1, 0), (cfg.embed_dim, cfg.num_kv_heads, cfg.head_dim)),
      ),
      r"model\.layers\.([0-9]+)\.self_attn\.o_proj\.weight": (
          r"layers.\1.attn.o_proj.w",
          ((1, 0), (cfg.num_heads, cfg.head_dim, cfg.embed_dim)),
      ),
      # mlp
      r"model\.layers\.([0-9]+)\.mlp\.gate_proj\.weight": (
          r"layers.\1.mlp.gate_proj.kernel",
          ((1, 0), None),
      ),
      r"model\.layers\.([0-9]+)\.mlp\.up_proj\.weight": (
          r"layers.\1.mlp.up_proj.kernel",
          ((1, 0), None),
      ),
      r"model\.layers\.([0-9]+)\.mlp\.down_proj\.weight": (
          r"layers.\1.mlp.down_proj.kernel",
          ((1, 0), None),
      ),
      # moe
      r"model\.layers\.([0-9]+)\.mlp\.gate\.weight": (
          r"layers.\1.mlp.router.kernel",
          ((1, 0), None),
      ),
      r"model\.layers\.([0-9]+)\.mlp\.experts\.gate_proj\.weight": (
          r"layers.\1.mlp.gate_proj",
          ((0, 2, 1), None),
      ),
      r"model\.layers\.([0-9]+)\.mlp\.experts\.up_proj\.weight": (
          r"layers.\1.mlp.up_proj",
          ((0, 2, 1), None),
      ),
      r"model\.layers\.([0-9]+)\.mlp\.experts\.down_proj\.weight": (
          r"layers.\1.mlp.down_proj",
          ((0, 2, 1), None),
      ),
      # norms
      r"model\.norm\.weight": ("final_norm.w", None),
      r"model\.layers\.([0-9]+)\.self_attn\.q_norm\.weight": (
          r"layers.\1.attn.q_norm.w",
          None,
      ),
      r"model\.layers\.([0-9]+)\.self_attn\.k_norm\.weight": (
          r"layers.\1.attn.k_norm.w",
          None,
      ),
      # layer norms (pre/post attention)
      r"model\.layers\.([0-9]+)\.input_layernorm\.weight": (
          r"layers.\1.input_layernorm.w",
          None,
      ),
      r"model\.layers\.([0-9]+)\.post_attention_layernorm\.weight": (
          r"layers.\1.post_attention_layernorm.w",
          None,
      ),
      r"lm_head\.weight": ("lm_head.w", ((1, 0), None)),
  }


def _torch_key_to_jax_key(mapping, source_key):
  subs = [
      (re.sub(pat, repl, source_key), reshape)
      for pat, (repl, reshape) in mapping.items()
      if re.match(pat, source_key)
  ]
  if len(subs) != 1:
    raise ValueError(f"Only one key should be found: {subs[0]}")
  else:
    return subs[0]


def _assign_weights(keys, tensor, state_dict, torch_key, transform):
  """Convert weights and assign to nnx state_dict."""
  key = keys[0]
  if len(keys) == 1:
    try:
      if transform is not None:
        permute, reshape = transform
        tensor = tensor.transpose(permute) if permute else tensor
        tensor = tensor.reshape(reshape) if reshape else tensor
    except Exception as e:
      raise RuntimeError(
          f"Failed to transform tensor {torch_key} with shape"
          f" {tensor.shape}: {e}"
      ) from e

    if tensor.shape != state_dict[key].shape:
      raise ValueError(
          f"shape must match for {torch_key}, got {tensor.shape} vs"
          f" {state_dict[key].shape}"
      )
    state_dict[key] = tensor
    return state_dict
  else:
    if key not in state_dict:
      raise ValueError(f"Unfound key {key} in {state_dict}")
    _assign_weights(keys[1:], tensor, state_dict[key], torch_key, transform)
    return state_dict


def _stoi(s):
  try:
    return int(s)
  except ValueError:
    return s


def create_model_from_safe_tensors(
    file_dir: str,
    config: model_lib.ModelConfig,
    mesh: jax.sharding.Mesh | None = None,
) -> model_lib.Qwen3:
  """Load tensors from the safetensors file and create a Qwen3 model."""
  files = list(epath.Path(file_dir).expanduser().glob("*.safetensors"))

  if not files:
    raise ValueError(f"No safetensors found in {file_dir}")

  tensor_dict = {}
  for f in tqdm.tqdm(files):
    tensor_dict |= safetensors.load_file(f)

  if config.num_experts is not None:
    tensor_dict = _stack_experts(tensor_dict)

  qwen3 = nnx.eval_shape(
      lambda: model_lib.Qwen3(config, rngs=nnx.Rngs(params=0))
  )

  graph_def, abs_state = nnx.split(qwen3)
  state_dict = abs_state.to_pure_dict()

  with jax.default_device(jax.devices("cpu")[0]):
    for k, v in tqdm.tqdm(tensor_dict.items()):
      jax_key, transform = _torch_key_to_jax_key(
          _get_key_and_transform_mapping(config), k
      )
      jax_keys = [_stoi(s) for s in jax_key.split(".")]
      _assign_weights(jax_keys, v, state_dict, k, transform)

  if mesh is not None:
    sharding = nnx.get_named_sharding(abs_state, mesh).to_pure_dict()
    state_dict = jax.device_put(state_dict, sharding)
  else:
    state_dict = jax.device_put(state_dict, jax.devices()[0])

  return nnx.merge(graph_def, state_dict)
