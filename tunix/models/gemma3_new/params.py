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

"""Utils for loading and converting Gemma3 PT weights."""

import jax
from tunix.models.gemma3_new import model as model_lib
from tunix.models import safetensors_loader


def _get_key_and_transform_mapping(cfg: model_lib.Gemma3Config):
  return {
      # Embedding
      r"model\.embed_tokens\.weight": ("embedder.input_embedding", None),

      # Attention: HF [out,in] → NNX [in,out]，再 reshape 成 (D, N, H)/(D, K, H)
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

      # MLP
      r"model\.layers\.([0-9]+)\.mlp\.gate_proj\.weight": (
          r"layers.\1.mlp.gate_proj.kernel", ((1, 0), None)
      ),
      r"model\.layers\.([0-9]+)\.mlp\.up_proj\.weight": (
          r"layers.\1.mlp.up_proj.kernel", ((1, 0), None)
      ),
      r"model\.layers\.([0-9]+)\.mlp\.down_proj\.weight": (
          r"layers.\1.mlp.down_proj.kernel", ((1, 0), None)
      ),

      # Norm（关键差异：用 .scale，且层名是 pre_/post_attention_norm）
      r"model\.layers\.([0-9]+)\.input_layernorm\.weight": (
          r"layers.\1.pre_attention_norm.scale", None
      ),
      r"model\.layers\.([0-9]+)\.post_attention_layernorm\.weight": (
          r"layers.\1.post_attention_norm.scale", None
      ),
      r"model\.norm\.weight": ("final_norm.scale", None),

      # 可选：HF 里经常带的 lm_head；本模型权重绑在 embedding 上，可以直接忽略
      r"lm_head\.weight": (r"IGNORED.lm_head.weight", None),

      # 可能出现但本实现没有的 q_norm/k_norm，忽略以免报错
      r"model\.layers\.([0-9]+)\.self_attn\.q_norm\.weight": (
          r"IGNORED.layers.\1.self_attn.q_norm.weight", None
      ),
      r"model\.layers\.([0-9]+)\.self_attn\.k_norm\.weight": (
          r"IGNORED.layers.\1.self_attn.k_norm.weight", None
      ),
  }



def create_model_from_safe_tensors(
    file_dir: str,
    config: model_lib.ModelConfig,
    mesh: jax.sharding.Mesh | None = None,
) -> model_lib.Gemma3:
    """Load tensors from the safetensors file and create a Gemma3 model."""
    return safetensors_loader.load_and_create_model(
        file_dir=file_dir,
        model_class=model_lib.Gemma3,
        config=config,
        key_mapping=_get_key_and_transform_mapping,
        mesh=mesh,
        preprocess_fn=None,
    )