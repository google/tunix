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

"""Loads Gemma 4 parameters from safetensors files."""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
from tunix.models import hf_mapping
from tunix.models import safetensors_loader
from tunix.models.gemma4 import model as model_lib

HfVariant = model_lib.HfVariant
from_hf_impl = hf_mapping.from_hf_impl
to_hf_impl = hf_mapping.to_hf_impl

_FROM_HF_TEXT_BASE_JSON = """[
  {
    "source_pattern": ["${lm}embed_tokens.weight"],
    "target_pattern": ["embedder.input_embedding"],
    "ops": []
  },
  {
    "source_pattern": ["${lm}norm.weight"],
    "target_pattern": ["final_norm.scale"],
    "ops": []
  },
  {
    "source_pattern": [
      "${lm}layers.{i}.input_layernorm.weight"
    ],
    "target_pattern": ["layers.{i}.pre_attention_norm.scale"],
    "ops": []
  },
  {
    "source_pattern": [
      "${lm}layers.{i}.post_attention_layernorm.weight"
    ],
    "target_pattern": ["layers.{i}.post_attention_norm.scale"],
    "ops": []
  },
  {
    "source_pattern": [
      "${lm}layers.{i}.pre_feedforward_layernorm.weight"
    ],
    "target_pattern": ["layers.{i}.pre_ffw_norm.scale"],
    "ops": []
  },
  {
    "source_pattern": [
      "${lm}layers.{i}.post_feedforward_layernorm.weight"
    ],
    "target_pattern": ["layers.{i}.post_ffw_norm.scale"],
    "ops": []
  },
  {
    "source_pattern": [
      "${lm}layers.{i}.self_attn.q_norm.weight"
    ],
    "target_pattern": ["layers.{i}.attn._query_norm.scale"],
    "ops": []
  },
  {
    "source_pattern": [
      "${lm}layers.{kv_i}.self_attn.k_norm.weight"
    ],
    "target_pattern": ["layers.{kv_i}.attn._key_norm.scale"],
    "ops": []
  },
  {
    "source_pattern": [
      "${lm}layers.{i}.mlp.gate_proj.weight"
    ],
    "target_pattern": ["layers.{i}.mlp.gate_proj.kernel"],
    "ops": [{"transpose": {"axes": [1, 0]}}]
  },
  {
    "source_pattern": ["${lm}layers.{i}.mlp.up_proj.weight"],
    "target_pattern": ["layers.{i}.mlp.up_proj.kernel"],
    "ops": [{"transpose": {"axes": [1, 0]}}]
  },
  {
    "source_pattern": [
      "${lm}layers.{i}.mlp.down_proj.weight"
    ],
    "target_pattern": ["layers.{i}.mlp.down_proj.kernel"],
    "ops": [{"transpose": {"axes": [1, 0]}}]
  },
  {
    "source_pattern": ["${lm}layers.{i}.layer_scalar"],
    "target_pattern": ["layers.{i}.skip_scale"],
    "ops": []
  },
  {
    "source_pattern": [
      "${lm}layers.{l}.self_attn.q_proj.weight"
    ],
    "target_pattern": ["layers.{l}.attn.q_einsum.w"],
    "ops": [
      {"reshape": {"src_shape": [$nh, $d], "dst_shape": [$n, $h, $d]}},
      {"transpose": {"axes": [0, 2, 1]}}
    ]
  },
  {
    "source_pattern": [
      "${lm}layers.{l_kv}.self_attn.k_proj.weight",
      "${lm}layers.{l_kv}.self_attn.v_proj.weight"
    ],
    "target_pattern": ["layers.{l_kv}.attn.kv_einsum.w"],
    "ops": [
      {"reshape": {"src_shape": [$kh, $d], "dst_shape": [$k, $h, $d]}},
      {"transpose": {"axes": [0, 2, 1]}},
      {"stack": {"axis": 0}}
    ]
  },
  {
    "source_pattern": [
      "${lm}layers.{l}.self_attn.o_proj.weight"
    ],
    "target_pattern": ["layers.{l}.attn.attn_vec_einsum.w"],
    "ops": [
      {"transpose": {"axes": [1, 0]}},
      {"reshape": {"src_shape": [$nh, $d], "dst_shape": [$n, $h, $d]}}
    ]
  },
  {
    "source_pattern": [
      "${lm}layers.{g}.self_attn.q_proj.weight"
    ],
    "target_pattern": ["layers.{g}.attn.q_einsum.w"],
    "ops": [
      {"reshape": {"src_shape": [$ngh, $d], "dst_shape": [$n, $gh, $d]}},
      {"transpose": {"axes": [0, 2, 1]}}
    ]
  },
  {
    "source_pattern": [
      "${lm}layers.{g_kv}.self_attn.k_proj.weight",
      "${lm}layers.{g_kv}.self_attn.v_proj.weight"
    ],
    "target_pattern": ["layers.{g_kv}.attn.kv_einsum.w"],
    "ops": [
      {"reshape": {"src_shape": [$gkh, $d], "dst_shape": [$gk, $gh, $d]}},
      {"transpose": {"axes": [0, 2, 1]}},
      {"stack": {"axis": 0}}
    ]
  },
  {
    "source_pattern": [
      "${lm}layers.{g_ko}.self_attn.k_proj.weight"
    ],
    "target_pattern": ["layers.{g_ko}.attn.k_einsum.w"],
    "ops": [
      {"reshape": {"src_shape": [$gkh, $d], "dst_shape": [$gk, $gh, $d]}},
      {"transpose": {"axes": [0, 2, 1]}}
    ]
  },
  {
    "source_pattern": [
      "${lm}layers.{g}.self_attn.o_proj.weight"
    ],
    "target_pattern": ["layers.{g}.attn.attn_vec_einsum.w"],
    "ops": [
      {"transpose": {"axes": [1, 0]}},
      {"reshape": {"src_shape": [$ngh, $d], "dst_shape": [$n, $gh, $d]}}
    ]
  }
]"""

_FROM_HF_PLE_JSON = """[
  {
    "source_pattern": [
      "${lm}embed_tokens_per_layer.weight"
    ],
    "target_pattern": ["embedder.per_layer_input_embedding"],
    "ops": []
  },
  {
    "source_pattern": [
      "${lm}per_layer_model_projection.weight"
    ],
    "target_pattern": ["embedder.per_layer_model_projection.w"],
    "ops": [{"transpose": {"axes": [1, 0]}}]
  },
  {
    "source_pattern": [
      "${lm}per_layer_projection_norm.weight"
    ],
    "target_pattern": ["embedder.per_layer_projection_norm.scale"],
    "ops": []
  },
  {
    "source_pattern": [
      "${lm}layers.{i}.per_layer_input_gate.weight"
    ],
    "target_pattern": ["layers.{i}.per_layer_input_gate.w"],
    "ops": [{"transpose": {"axes": [1, 0]}}]
  },
  {
    "source_pattern": [
      "${lm}layers.{i}.per_layer_projection.weight"
    ],
    "target_pattern": ["layers.{i}.per_layer_projection.w"],
    "ops": [{"transpose": {"axes": [1, 0]}}]
  },
  {
    "source_pattern": [
      "${lm}layers.{i}.post_per_layer_input_norm.weight"
    ],
    "target_pattern": ["layers.{i}.post_per_layer_input_norm.scale"],
    "ops": []
  }
]"""

_FROM_HF_MOE_JSON = """[
  {
    "source_pattern": [
      "${lm}layers.{m}.post_feedforward_layernorm_1.weight"
    ],
    "target_pattern": ["layers.{m}.dense_post_ffw_norm.scale"],
    "ops": []
  },
  {
    "source_pattern": [
      "${lm}layers.{m}.pre_feedforward_layernorm_2.weight"
    ],
    "target_pattern": ["layers.{m}.moe_pre_ffw_norm.scale"],
    "ops": []
  },
  {
    "source_pattern": [
      "${lm}layers.{m}.post_feedforward_layernorm_2.weight"
    ],
    "target_pattern": ["layers.{m}.moe_post_ffw_norm.scale"],
    "ops": []
  },
  {
    "source_pattern": ["${lm}layers.{m}.router.proj.weight"],
    "target_pattern": ["layers.{m}.moe.router_logits"],
    "ops": [{"transpose": {"axes": [1, 0]}}]
  },
  {
    "source_pattern": [
      "${lm}layers.{m}.router.per_expert_scale"
    ],
    "target_pattern": ["layers.{m}.moe.per_expert_scale"],
    "ops": []
  },
  {
    "source_pattern": ["${lm}layers.{m}.router.scale"],
    "target_pattern": ["layers.{m}.moe.router_scale"],
    "ops": []
  },
  {
    "source_pattern": ["${lm}layers.{m}.experts.gate_up_proj"],
    "target_pattern": ["layers.{m}.moe.gating_einsum"],
    "ops": [
      {"reshape": {"src_shape": [$ne, $ef2, $d], "dst_shape": [$ne, 2, $ef, $d]}}
    ]
  },
  {
    "source_pattern": ["${lm}layers.{m}.experts.down_proj"],
    "target_pattern": ["layers.{m}.moe.linear"],
    "ops": [{"transpose": {"axes": [0, 2, 1]}}]
  }
]"""

_FROM_HF_VISION_JSON = """[
  {
    "source_pattern": ["${mm}embed_vision.embedding_projection.weight"],
    "target_pattern": ["embedder.mm_input_projection.w"],
    "ops": [{"transpose": {"axes": [1, 0]}}]
  },
  {
    "source_pattern": [
      "${mm}vision_tower.patch_embedder.position_embedding_table"
    ],
    "target_pattern": ["vision_encoder.entry.pos_emb"],
    "ops": [{"transpose": {"axes": [1, 0, 2]}}]
  },
  {
    "source_pattern": ["${mm}vision_tower.patch_embedder.input_proj.weight"],
    "target_pattern": ["vision_encoder.entry.input_projection.w"],
    "ops": [{"transpose": {"axes": [1, 0]}}]
  },
  {
    "source_pattern": [
      "${mm}vision_tower.encoder.layers.{v}.input_layernorm.weight"
    ],
    "target_pattern": [
      "vision_encoder.layers.{v}.pre_attention_norm.scale"
    ],
    "ops": []
  },
  {
    "source_pattern": [
      "${mm}vision_tower.encoder.layers.{v}.post_attention_layernorm.weight"
    ],
    "target_pattern": [
      "vision_encoder.layers.{v}.post_attention_norm.scale"
    ],
    "ops": []
  },
  {
    "source_pattern": [
      "${mm}vision_tower.encoder.layers.{v}.pre_feedforward_layernorm.weight"
    ],
    "target_pattern": ["vision_encoder.layers.{v}.pre_ffw_norm.scale"],
    "ops": []
  },
  {
    "source_pattern": [
      "${mm}vision_tower.encoder.layers.{v}.post_feedforward_layernorm.weight"
    ],
    "target_pattern": ["vision_encoder.layers.{v}.post_ffw_norm.scale"],
    "ops": []
  },
  {
    "source_pattern": [
      "${mm}vision_tower.encoder.layers.{v}.self_attn.q_norm.weight"
    ],
    "target_pattern": ["vision_encoder.layers.{v}.attn.query_norm.scale"],
    "ops": []
  },
  {
    "source_pattern": [
      "${mm}vision_tower.encoder.layers.{v}.self_attn.k_norm.weight"
    ],
    "target_pattern": ["vision_encoder.layers.{v}.attn.key_norm.scale"],
    "ops": []
  },
  {
    "source_pattern": [
      "${mm}vision_tower.encoder.layers.{v}.self_attn.q_proj.linear.weight"
    ],
    "target_pattern": ["vision_encoder.layers.{v}.attn.q_einsum.w"],
    "ops": [
      {"reshape": {"src_shape": [$vnh, $vd], "dst_shape": [$vn, $vh, $vd]}},
      {"transpose": {"axes": [0, 2, 1]}}
    ]
  },
  {
    "source_pattern": [
      "${mm}vision_tower.encoder.layers.{v}.self_attn.k_proj.linear.weight",
      "${mm}vision_tower.encoder.layers.{v}.self_attn.v_proj.linear.weight"
    ],
    "target_pattern": ["vision_encoder.layers.{v}.attn.kv_einsum.w"],
    "ops": [
      {"reshape": {"src_shape": [$vnh, $vd], "dst_shape": [$vn, $vh, $vd]}},
      {"transpose": {"axes": [0, 2, 1]}},
      {"stack": {"axis": 0}}
    ]
  },
  {
    "source_pattern": [
      "${mm}vision_tower.encoder.layers.{v}.self_attn.o_proj.linear.weight"
    ],
    "target_pattern": ["vision_encoder.layers.{v}.attn.attn_vec_einsum.w"],
    "ops": [
      {"reshape": {"src_shape": [$vd, $vnh], "dst_shape": [$vd, $vn, $vh]}},
      {"transpose": {"axes": [1, 2, 0]}}
    ]
  },
  {
    "source_pattern": [
      "${mm}vision_tower.encoder.layers.{v}.mlp.gate_proj.linear.weight",
      "${mm}vision_tower.encoder.layers.{v}.mlp.up_proj.linear.weight"
    ],
    "target_pattern": ["vision_encoder.layers.{v}.mlp.gating_einsum.w"],
    "ops": [{"stack": {"axis": 0}}]
  },
  {
    "source_pattern": [
      "${mm}vision_tower.encoder.layers.{v}.mlp.down_proj.linear.weight"
    ],
    "target_pattern": ["vision_encoder.layers.{v}.mlp.linear.w"],
    "ops": [{"transpose": {"axes": [1, 0]}}]
  }
]"""

_FROM_HF_VISION_CLIP_JSON = """[
  {
    "source_pattern": [
      "${mm}vision_tower.encoder.layers.{v}.self_attn.q_proj.{vc}"
    ],
    "target_pattern": ["vision_encoder.layers.{v}.attn.q_einsum.clip_{vc}"],
    "ops": []
  },
  {
    "source_pattern": [
      "${mm}vision_tower.encoder.layers.{v}.self_attn.k_proj.{vc}",
      "${mm}vision_tower.encoder.layers.{v}.self_attn.v_proj.{vc}"
    ],
    "target_pattern": ["vision_encoder.layers.{v}.attn.kv_einsum.clip_{vc}"],
    "ops": [
      {"stack": {"axis": 0}},
      {"unrepeat": {"axis": 0, "times": 2}},
      {"unstack": {"axis": 0}}
    ]
  },
  {
    "source_pattern": [
      "${mm}vision_tower.encoder.layers.{v}.self_attn.o_proj.{vc}"
    ],
    "target_pattern": [
      "vision_encoder.layers.{v}.attn.attn_vec_einsum.clip_{vc}"
    ],
    "ops": []
  },
  {
    "source_pattern": [
      "${mm}vision_tower.encoder.layers.{v}.mlp.gate_proj.{vc}",
      "${mm}vision_tower.encoder.layers.{v}.mlp.up_proj.{vc}"
    ],
    "target_pattern": ["vision_encoder.layers.{v}.mlp.gating_einsum.clip_{vc}"],
    "ops": [
      {"stack": {"axis": 0}},
      {"unrepeat": {"axis": 0, "times": 2}},
      {"unstack": {"axis": 0}}
    ]
  },
  {
    "source_pattern": [
      "${mm}vision_tower.encoder.layers.{v}.mlp.down_proj.{vc}"
    ],
    "target_pattern": ["vision_encoder.layers.{v}.mlp.linear.clip_{vc}"],
    "ops": []
  }
]"""

_FROM_HF_VISION_STD_JSON = """[
  {
    "source_pattern": ["${mm}vision_tower.std_scale"],
    "target_pattern": ["vision_encoder.standardize.scale"],
    "ops": []
  },
  {
    "source_pattern": ["${mm}vision_tower.std_bias"],
    "target_pattern": ["vision_encoder.standardize.bias"],
    "ops": []
  }
]"""

_FROM_HF_AUDIO_JSON = """[
  {
    "source_pattern": ["${mm}embed_audio.embedding_projection.weight"],
    "target_pattern": ["embedder.audio_input_projection.w"],
    "ops": [{"transpose": {"axes": [1, 0]}}]
  },
  {
    "source_pattern": [
      "${mm}audio_tower.subsample_conv_projection.layer0.conv.weight"
    ],
    "target_pattern": ["audio_encoder.feature.subsampling_0.kernel"],
    "ops": [{"transpose": {"axes": [2, 3, 1, 0]}}]
  },
  {
    "source_pattern": [
      "${mm}audio_tower.subsample_conv_projection.layer0.norm.weight"
    ],
    "target_pattern": ["audio_encoder.feature.norm_0.scale"],
    "ops": []
  },
  {
    "source_pattern": [
      "${mm}audio_tower.subsample_conv_projection.layer1.conv.weight"
    ],
    "target_pattern": ["audio_encoder.feature.subsampling_1.kernel"],
    "ops": [{"transpose": {"axes": [2, 3, 1, 0]}}]
  },
  {
    "source_pattern": [
      "${mm}audio_tower.subsample_conv_projection.layer1.norm.weight"
    ],
    "target_pattern": ["audio_encoder.feature.norm_1.scale"],
    "ops": []
  },
  {
    "source_pattern": [
      "${mm}audio_tower.subsample_conv_projection.input_proj_linear.weight"
    ],
    "target_pattern": ["audio_encoder.feature.input_proj.kernel"],
    "ops": [
      {"reshape": {"src_shape": [$ad, 1024], "dst_shape": [$ad, 32, 32]}},
      {"transpose": {"axes": [1, 2, 0]}}
    ]
  },
  {
    "source_pattern": ["${mm}audio_tower.output_proj.weight"],
    "target_pattern": ["audio_encoder.output_projection.kernel"],
    "ops": [{"transpose": {"axes": [1, 0]}}]
  },
  {
    "source_pattern": ["${mm}audio_tower.output_proj.bias"],
    "target_pattern": ["audio_encoder.output_projection.bias"],
    "ops": []
  },
  {
    "source_pattern": [
      "${mm}audio_tower.layers.{a}.feed_forward1.pre_layer_norm.weight"
    ],
    "target_pattern": [
      "audio_encoder.conformer_layers.{a}.fflayer_start.pre_layer_norm.scale"
    ],
    "ops": []
  },
  {
    "source_pattern": [
      "${mm}audio_tower.layers.{a}.feed_forward1.ffw_layer_1.linear.weight"
    ],
    "target_pattern": [
      "audio_encoder.conformer_layers.{a}.fflayer_start.ffn_layer1.kernel"
    ],
    "ops": [{"transpose": {"axes": [1, 0]}}]
  },
  {
    "source_pattern": [
      "${mm}audio_tower.layers.{a}.feed_forward1.ffw_layer_1.{c}"
    ],
    "target_pattern": [
      "audio_encoder.conformer_layers.{a}.fflayer_start.ffn_layer1.clip_{c}"
    ],
    "ops": []
  },
  {
    "source_pattern": [
      "${mm}audio_tower.layers.{a}.feed_forward1.ffw_layer_2.linear.weight"
    ],
    "target_pattern": [
      "audio_encoder.conformer_layers.{a}.fflayer_start.ffn_layer2.kernel"
    ],
    "ops": [{"transpose": {"axes": [1, 0]}}]
  },
  {
    "source_pattern": [
      "${mm}audio_tower.layers.{a}.feed_forward1.ffw_layer_2.{c}"
    ],
    "target_pattern": [
      "audio_encoder.conformer_layers.{a}.fflayer_start.ffn_layer2.clip_{c}"
    ],
    "ops": []
  },
  {
    "source_pattern": [
      "${mm}audio_tower.layers.{a}.feed_forward1.post_layer_norm.weight"
    ],
    "target_pattern": [
      "audio_encoder.conformer_layers.{a}.fflayer_start.post_layer_norm.scale"
    ],
    "ops": []
  },
  {
    "source_pattern": [
      "${mm}audio_tower.layers.{a}.feed_forward2.pre_layer_norm.weight"
    ],
    "target_pattern": [
      "audio_encoder.conformer_layers.{a}.fflayer_end.pre_layer_norm.scale"
    ],
    "ops": []
  },
  {
    "source_pattern": [
      "${mm}audio_tower.layers.{a}.feed_forward2.ffw_layer_1.linear.weight"
    ],
    "target_pattern": [
      "audio_encoder.conformer_layers.{a}.fflayer_end.ffn_layer1.kernel"
    ],
    "ops": [{"transpose": {"axes": [1, 0]}}]
  },
  {
    "source_pattern": [
      "${mm}audio_tower.layers.{a}.feed_forward2.ffw_layer_1.{c}"
    ],
    "target_pattern": [
      "audio_encoder.conformer_layers.{a}.fflayer_end.ffn_layer1.clip_{c}"
    ],
    "ops": []
  },
  {
    "source_pattern": [
      "${mm}audio_tower.layers.{a}.feed_forward2.ffw_layer_2.linear.weight"
    ],
    "target_pattern": [
      "audio_encoder.conformer_layers.{a}.fflayer_end.ffn_layer2.kernel"
    ],
    "ops": [{"transpose": {"axes": [1, 0]}}]
  },
  {
    "source_pattern": [
      "${mm}audio_tower.layers.{a}.feed_forward2.ffw_layer_2.{c}"
    ],
    "target_pattern": [
      "audio_encoder.conformer_layers.{a}.fflayer_end.ffn_layer2.clip_{c}"
    ],
    "ops": []
  },
  {
    "source_pattern": [
      "${mm}audio_tower.layers.{a}.feed_forward2.post_layer_norm.weight"
    ],
    "target_pattern": [
      "audio_encoder.conformer_layers.{a}.fflayer_end.post_layer_norm.scale"
    ],
    "ops": []
  },
  {
    "source_pattern": ["${mm}audio_tower.layers.{a}.norm_pre_attn.weight"],
    "target_pattern": [
      "audio_encoder.conformer_layers.{a}.trans_atten.pre_norm.scale"
    ],
    "ops": []
  },
  {
    "source_pattern": [
      "${mm}audio_tower.layers.{a}.self_attn.q_proj.linear.weight"
    ],
    "target_pattern": [
      "audio_encoder.conformer_layers.{a}.trans_atten.self_atten.query.kernel"
    ],
    "ops": [{"transpose": {"axes": [1, 0]}}]
  },
  {
    "source_pattern": ["${mm}audio_tower.layers.{a}.self_attn.q_proj.{c}"],
    "target_pattern": [
      "audio_encoder.conformer_layers.{a}.trans_atten.self_atten.query.clip_{c}"
    ],
    "ops": []
  },
  {
    "source_pattern": [
      "${mm}audio_tower.layers.{a}.self_attn.k_proj.linear.weight"
    ],
    "target_pattern": [
      "audio_encoder.conformer_layers.{a}.trans_atten.self_atten.key.kernel"
    ],
    "ops": [{"transpose": {"axes": [1, 0]}}]
  },
  {
    "source_pattern": ["${mm}audio_tower.layers.{a}.self_attn.k_proj.{c}"],
    "target_pattern": [
      "audio_encoder.conformer_layers.{a}.trans_atten.self_atten.key.clip_{c}"
    ],
    "ops": []
  },
  {
    "source_pattern": [
      "${mm}audio_tower.layers.{a}.self_attn.v_proj.linear.weight"
    ],
    "target_pattern": [
      "audio_encoder.conformer_layers.{a}.trans_atten.self_atten.value.kernel"
    ],
    "ops": [{"transpose": {"axes": [1, 0]}}]
  },
  {
    "source_pattern": ["${mm}audio_tower.layers.{a}.self_attn.v_proj.{c}"],
    "target_pattern": [
      "audio_encoder.conformer_layers.{a}.trans_atten.self_atten.value.clip_{c}"
    ],
    "ops": []
  },
  {
    "source_pattern": ["${mm}audio_tower.layers.{a}.self_attn.per_dim_scale"],
    "target_pattern": [
      "audio_encoder.conformer_layers.{a}.trans_atten.self_atten.per_dim_scale"
    ],
    "ops": []
  },
  {
    "source_pattern": [
      "${mm}audio_tower.layers.{a}.self_attn.post.linear.weight"
    ],
    "target_pattern": [
      "audio_encoder.conformer_layers.{a}.trans_atten.post.kernel"
    ],
    "ops": [
      {"reshape": {"src_shape": [$ad, $anah], "dst_shape": [$ad, $an, $ah]}},
      {"transpose": {"axes": [1, 2, 0]}}
    ]
  },
  {
    "source_pattern": ["${mm}audio_tower.layers.{a}.self_attn.post.{c}"],
    "target_pattern": [
      "audio_encoder.conformer_layers.{a}.trans_atten.post.clip_{c}"
    ],
    "ops": []
  },
  {
    "source_pattern": [
      "${mm}audio_tower.layers.{a}.self_attn.relative_k_proj.weight"
    ],
    "target_pattern": [
      "audio_encoder.conformer_layers.{a}.trans_atten.self_atten.relative_position_embedding.pos_proj.kernel"
    ],
    "ops": [
      {"reshape": {"src_shape": [$anah, $ad], "dst_shape": [$an, $ah, $ad]}},
      {"transpose": {"axes": [2, 0, 1]}}
    ]
  },
  {
    "source_pattern": ["${mm}audio_tower.layers.{a}.norm_post_attn.weight"],
    "target_pattern": [
      "audio_encoder.conformer_layers.{a}.trans_atten.post_norm.scale"
    ],
    "ops": []
  },
  {
    "source_pattern": [
      "${mm}audio_tower.layers.{a}.lconv1d.pre_layer_norm.weight"
    ],
    "target_pattern": ["audio_encoder.conformer_layers.{a}.lconv.ln.scale"],
    "ops": []
  },
  {
    "source_pattern": [
      "${mm}audio_tower.layers.{a}.lconv1d.linear_start.linear.weight"
    ],
    "target_pattern": [
      "audio_encoder.conformer_layers.{a}.lconv.linear_start.kernel"
    ],
    "ops": [{"transpose": {"axes": [1, 0]}}]
  },
  {
    "source_pattern": [
      "${mm}audio_tower.layers.{a}.lconv1d.linear_start.{c}"
    ],
    "target_pattern": [
      "audio_encoder.conformer_layers.{a}.lconv.linear_start.clip_{c}"
    ],
    "ops": []
  },
  {
    "source_pattern": [
      "${mm}audio_tower.layers.{a}.lconv1d.depthwise_conv1d.weight"
    ],
    "target_pattern": [
      "audio_encoder.conformer_layers.{a}.lconv.depthwise_conv1d.kernel"
    ],
    "ops": [{"transpose": {"axes": [2, 1, 0]}}]
  },
  {
    "source_pattern": ["${mm}audio_tower.layers.{a}.lconv1d.conv_norm.weight"],
    "target_pattern": [
      "audio_encoder.conformer_layers.{a}.lconv.conv_norm.scale"
    ],
    "ops": []
  },
  {
    "source_pattern": [
      "${mm}audio_tower.layers.{a}.lconv1d.linear_end.linear.weight"
    ],
    "target_pattern": [
      "audio_encoder.conformer_layers.{a}.lconv.linear_end.kernel"
    ],
    "ops": [{"transpose": {"axes": [1, 0]}}]
  },
  {
    "source_pattern": ["${mm}audio_tower.layers.{a}.lconv1d.linear_end.{c}"],
    "target_pattern": [
      "audio_encoder.conformer_layers.{a}.lconv.linear_end.clip_{c}"
    ],
    "ops": []
  },
  {
    "source_pattern": ["${mm}audio_tower.layers.{a}.norm_out.weight"],
    "target_pattern": ["audio_encoder.conformer_layers.{a}.final_ln.scale"],
    "ops": []
  }
]"""

_CLIP_BOUNDS = ("input_min", "input_max", "output_min", "output_max")
_EMPTY_VISION = model_lib.vision.VisionEncoderConfig(num_layers=0)
_EMPTY_AUDIO = model_lib.audio.ConformerConfig(num_layers=0)


def _compute_axes_and_ties(
    cfg: model_lib.ModelConfig,
    v_cfg: model_lib.vision.VisionEncoderConfig,
    a_cfg: model_lib.audio.ConformerConfig,
) -> tuple[dict[str, Any], list[tuple[str, str]]]:
  """Returns template expansion axes and KV-sharing `tied_keys` for `cfg`."""
  pattern = (
      cfg.attention_pattern
      if cfg.attention_pattern
      else model_lib.GEMMA4_ATTENTION_PATTERN
  )
  attn_types = tuple(
      pattern[i % len(pattern)] for i in range(cfg.num_layers)
  )
  kv_origins = model_lib.create_kv_cache_sharing_patterns(
      num_layers=cfg.num_layers,
      frac_shared_layers=cfg.frac_shared_layers,
      share_global=True,
      share_local=True,
      attention_types=attn_types,
  )
  unshared_list = [i for i, orig in enumerate(kv_origins) if i == orig]
  unshared_set = set(unshared_list)
  global_set = [
      i
      for i, t in enumerate(attn_types)
      if t == model_lib.AttentionType.GLOBAL
  ]
  local_set = sorted(set(range(cfg.num_layers)) - set(global_set))
  g_unshared = sorted(set(global_set) & unshared_set)
  g_kv, g_ko = ([], g_unshared) if cfg.k_eq_v_global else (g_unshared, [])
  g_ko_set = set(g_ko)
  tied_keys: list[tuple[str, str]] = []
  for i, orig in enumerate(kv_origins):
    if i == orig:
      continue
    kv_attr = "k_einsum" if orig in g_ko_set else "kv_einsum"
    tied_keys.extend((
        (f"layers.{orig}.attn.{kv_attr}.w", f"layers.{i}.attn.{kv_attr}.w"),
        (
            f"layers.{orig}.attn._key_norm.scale",
            f"layers.{i}.attn._key_norm.scale",
        ),
    ))
  axes = {
      "i": cfg.num_layers,
      "kv_i": unshared_list,
      "l": local_set,
      "l_kv": sorted(set(local_set) & unshared_set),
      "g": global_set,
      "g_kv": g_kv,
      "g_ko": g_ko,
      "m": cfg.num_layers if cfg.enable_moe else 0,
      "v": v_cfg.num_layers,
      "a": a_cfg.num_layers,
      "c": _CLIP_BOUNDS,
      "vc": _CLIP_BOUNDS if v_cfg.use_clipped_linears else (),
  }
  return axes, tied_keys


# Template placeholder values per variant: `lm` prefixes each text decoder key
# and `mm` each vision / audio tower and embedder key.
_HF_AFFIXES: Mapping[HfVariant, Mapping[str, str]] = {
    HfVariant.CAUSAL_LM: {"lm": "model."},
    HfVariant.CONDITIONAL_GENERATION: {
        "lm": "model.language_model.",
        "mm": "model.",
    },
}


def _render_gemma4_templates(
    cfg: model_lib.ModelConfig,
    v_cfg: model_lib.vision.VisionEncoderConfig,
    a_cfg: model_lib.audio.ConformerConfig,
    affixes: Mapping[str, str],
) -> list[hf_mapping.ModelTemplateRule]:
  """Renders parameterized JSON PrimOp templates for `cfg`."""
  d, n, k, h = cfg.embed_dim, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
  gh = h if cfg.global_key_size is None else cfg.global_key_size
  gk = k if cfg.num_global_kv_heads is None else cfg.num_global_kv_heads
  vh = v_cfg.d_model // v_cfg.num_heads
  ah = a_cfg.model_dims // a_cfg.atten_num_heads
  ne = 1 if cfg.num_experts is None else cfg.num_experts
  ef = 1 if cfg.expert_dim is None else cfg.expert_dim
  vision_blocks = (
      (_FROM_HF_VISION_JSON, _FROM_HF_VISION_CLIP_JSON)
      if v_cfg.num_layers > 0
      else ()
  )
  return hf_mapping.render_json_templates(
      _FROM_HF_TEXT_BASE_JSON,
      _FROM_HF_PLE_JSON if cfg.per_layer_input_dim > 0 else "",
      _FROM_HF_MOE_JSON if cfg.enable_moe else "",
      *vision_blocks,
      _FROM_HF_VISION_STD_JSON if v_cfg.standardize_embeddings else "",
      _FROM_HF_AUDIO_JSON if a_cfg.num_layers > 0 else "",
      **affixes,
      d=d,
      n=n,
      k=k,
      h=h,
      nh=n * h,
      kh=k * h,
      gh=gh,
      gk=gk,
      ngh=n * gh,
      gkh=gk * gh,
      ne=ne,
      ef=ef,
      ef2=2 * ef,
      vd=v_cfg.d_model,
      vn=v_cfg.num_heads,
      vh=vh,
      vnh=v_cfg.num_heads * vh,
      ad=a_cfg.model_dims,
      an=a_cfg.atten_num_heads,
      ah=ah,
      anah=a_cfg.atten_num_heads * ah,
  )


def build_from_hf_manifest(
    cfg: model_lib.ModelConfig,
    hf_variant: HfVariant | None = None,
) -> dict[str, Any]:
  """Builds the `HF -> Tunix` manifest for `cfg`."""
  if not isinstance(hf_variant, HfVariant):
    raise ValueError(
        f"Pass `hf_variant` (HfVariant) explicitly; got {hf_variant!r}."
    )
  v_cfg = _EMPTY_VISION if cfg.vision_encoder is None else cfg.vision_encoder
  a_cfg = _EMPTY_AUDIO if cfg.audio_encoder is None else cfg.audio_encoder
  affixes = _HF_AFFIXES[hf_variant]
  if (v_cfg.num_layers > 0 or a_cfg.num_layers > 0) and "mm" not in affixes:
    raise ValueError(
        "Gemma4 multimodal config requires CONDITIONAL_GENERATION hf_variant,"
        " got CAUSAL_LM."
    )
  templates = _render_gemma4_templates(cfg, v_cfg, a_cfg, affixes)
  axes, tied_keys = _compute_axes_and_ties(cfg, v_cfg, a_cfg)
  return hf_mapping.generate_manifest(
      model_type=f"gemma4/{hf_variant.value}",
      template_rules=templates,
      axes=axes,
      tied_keys=tied_keys,
      dropped_keys=["lm_head.weight"],
  )


def build_to_hf_manifest(
    cfg: model_lib.ModelConfig,
    hf_variant: HfVariant | None = None,
) -> dict[str, Any]:
  """Builds the `Tunix -> HF` manifest for `cfg`."""
  return hf_mapping.invert_manifest(build_from_hf_manifest(cfg, hf_variant))


def export_to_hf_state_dict(
    model: model_lib.Gemma4,
    cfg: model_lib.ModelConfig,
    *,
    hf_variant: HfVariant | None = None,
) -> dict[str, jax.Array]:
  """Exports a Tunix Gemma 4 model into an HF state dict."""
  eff_cfg = cfg
  if model.text_only:
    eff_cfg = dataclasses.replace(cfg, vision_encoder=None, audio_encoder=None)
  resolved = model._hf_variant if hf_variant is None else hf_variant  # pylint: disable=protected-access
  manifest = build_to_hf_manifest(eff_cfg, resolved)
  return hf_mapping.to_hf_impl(
      manifest, safetensors_loader.flatten_model_state(model)
  )


def create_model_from_safe_tensors(
    file_dir: str,
    config: model_lib.ModelConfig,
    mesh: jax.sharding.Mesh | None = None,
    dtype: jnp.dtype | None = None,
    mode: str = "auto",
    text_only: bool = True,
    *,
    hf_variant: HfVariant | None = None,
):
  """Loads a safetensors checkpoint into a Gemma 4 model."""
  eff_cfg = config
  if text_only:
    eff_cfg = dataclasses.replace(
        config, vision_encoder=None, audio_encoder=None
    )
  if hf_variant is None:
    local_dir = safetensors_loader.resolve_checkpoint_dir(file_dir)
    resolved = hf_mapping.resolve_hf_variant_from_keys(
        safetensors_loader.read_checkpoint_keys(local_dir), _HF_AFFIXES
    )
  else:
    if not isinstance(hf_variant, HfVariant):
      raise ValueError(
          f"Pass `hf_variant` (HfVariant) explicitly; got {hf_variant!r}."
      )
    local_dir = file_dir
    resolved = hf_variant
  model = safetensors_loader.load_and_create_model(
      file_dir=local_dir,
      model_class=model_lib.Gemma4,
      config=config,
      mesh=mesh,
      dtype=dtype,
      mode=mode,
      model_class_kwargs={"text_only": text_only},
      manifest=build_from_hf_manifest(eff_cfg, resolved),
  )
  model._hf_variant = resolved  # pylint: disable=protected-access
  return model
