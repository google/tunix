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

"""Declarative HF -> vLLM-JAX weight manifest builder for Qwen3."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import types
from typing import Any, Protocol, runtime_checkable

from tunix.models import hf_mapping


@runtime_checkable
class Qwen3HfConfig(Protocol):
  """Minimal Hugging Face Qwen3 config contract."""

  @property
  def vocab_size(self) -> int:
    ...

  @property
  def hidden_size(self) -> int:
    ...

  @property
  def intermediate_size(self) -> int:
    ...

  @property
  def num_hidden_layers(self) -> int:
    ...

  @property
  def num_attention_heads(self) -> int:
    ...

  @property
  def num_key_value_heads(self) -> int | None:
    ...

  @property
  def head_dim(self) -> int | None:
    ...

  @property
  def tie_word_embeddings(self) -> bool:
    ...

  @property
  def attention_bias(self) -> bool:
    ...


_QWEN3_FROM_HF_BASE_JSON = """[
  {
    "id": "embed_tokens",
    "source_pattern": ["model.embed_tokens.weight"],
    "target_pattern": ["model.embed_tokens.weight"],
    "ops": [
      {"pad": {"axis": 0, "sizes": [0, $pad_v]}}
    ]
  },
  {
    "id": "norm",
    "source_pattern": ["model.norm.weight"],
    "target_pattern": ["model.norm.weight"],
    "ops": []
  },
  {
    "id": "input_layernorm",
    "source_pattern": ["model.layers.{i}.input_layernorm.weight"],
    "target_pattern": ["model.layers.{i}.input_layernorm.weight"],
    "ops": []
  },
  {
    "id": "post_attention_layernorm",
    "source_pattern": ["model.layers.{i}.post_attention_layernorm.weight"],
    "target_pattern": ["model.layers.{i}.post_attention_layernorm.weight"],
    "ops": []
  },
  {
    "id": "q_norm",
    "source_pattern": ["model.layers.{i}.self_attn.q_norm.weight"],
    "target_pattern": ["model.layers.{i}.self_attn.q_norm.weight"],
    "ops": [
      {"pad": {"axis": 0, "sizes": [0, $pad_h]}}
    ]
  },
  {
    "id": "k_norm",
    "source_pattern": ["model.layers.{i}.self_attn.k_norm.weight"],
    "target_pattern": ["model.layers.{i}.self_attn.k_norm.weight"],
    "ops": [
      {"pad": {"axis": 0, "sizes": [0, $pad_h]}}
    ]
  },
  {
    "id": "q_proj",
    "source_pattern": ["model.layers.{i}.self_attn.q_proj.weight"],
    "target_pattern": ["model.layers.{i}.self_attn.q_proj.weight"],
    "ops": [
      {"reshape": {"src_shape": [$nh, $d], "dst_shape": [$n, $h, $d]}},
      {"pad": {"axis": 1, "sizes": [0, $pad_h]}},
      {"transpose": {"axes": [2, 0, 1]}},
      {"repeat": {"axis": 1, "times": $rep_q}}
    ]
  },
  {
    "id": "k_proj",
    "source_pattern": ["model.layers.{i}.self_attn.k_proj.weight"],
    "target_pattern": ["model.layers.{i}.self_attn.k_proj.weight"],
    "ops": [
      {"reshape": {"src_shape": [$kh, $d], "dst_shape": [$k, $h, $d]}},
      {"pad": {"axis": 1, "sizes": [0, $pad_h]}},
      {"transpose": {"axes": [2, 0, 1]}},
      {"repeat": {"axis": 1, "times": $rep_kv}}
    ]
  },
  {
    "id": "v_proj",
    "source_pattern": ["model.layers.{i}.self_attn.v_proj.weight"],
    "target_pattern": ["model.layers.{i}.self_attn.v_proj.weight"],
    "ops": [
      {"reshape": {"src_shape": [$kh, $d], "dst_shape": [$k, $h, $d]}},
      {"pad": {"axis": 1, "sizes": [0, $pad_h]}},
      {"transpose": {"axes": [2, 0, 1]}},
      {"repeat": {"axis": 1, "times": $rep_kv}}
    ]
  },
  {
    "id": "o_proj",
    "source_pattern": ["model.layers.{i}.self_attn.o_proj.weight"],
    "target_pattern": ["model.layers.{i}.self_attn.o_proj.weight"],
    "ops": [
      {"reshape": {"src_shape": [$d, $nh], "dst_shape": [$d, $n, $h]}},
      {"pad": {"axis": 2, "sizes": [0, $pad_h]}},
      {"transpose": {"axes": [1, 2, 0]}},
      {"repeat": {"axis": 0, "times": $rep_q}},
      {"scale": {"factor": $inv_rep_q}}
    ]
  },
  {
    "id": "gate_proj",
    "source_pattern": ["model.layers.{i}.mlp.gate_proj.weight"],
    "target_pattern": ["model.layers.{i}.mlp.gate_proj.weight"],
    "ops": [{"transpose": {"axes": [1, 0]}}]
  },
  {
    "id": "up_proj",
    "source_pattern": ["model.layers.{i}.mlp.up_proj.weight"],
    "target_pattern": ["model.layers.{i}.mlp.up_proj.weight"],
    "ops": [{"transpose": {"axes": [1, 0]}}]
  },
  {
    "id": "down_proj",
    "source_pattern": ["model.layers.{i}.mlp.down_proj.weight"],
    "target_pattern": ["model.layers.{i}.mlp.down_proj.weight"],
    "ops": [{"transpose": {"axes": [1, 0]}}]
  }
]"""

_QWEN3_FROM_HF_LM_HEAD_JSON = """[
  {
    "id": "lm_head",
    "source_pattern": ["lm_head.weight"],
    "target_pattern": ["lm_head.weight"],
    "ops": [
      {"pad": {"axis": 0, "sizes": [0, $pad_v]}},
      {"transpose": {"axes": [1, 0]}}
    ]
  }
]"""

_QWEN3_SHARDING_RULES: Mapping[str, tuple[str | None, ...]] = (
    types.MappingProxyType({
        "embed_tokens": ("model", None),
        "norm": (None,),
        "input_layernorm": (None,),
        "post_attention_layernorm": (None,),
        "q_norm": (None,),
        "k_norm": (None,),
        "q_proj": (None, "model", None),
        "k_proj": (None, "model", None),
        "v_proj": (None, "model", None),
        "o_proj": ("model", None, None),
        "gate_proj": (None, "model"),
        "up_proj": (None, "model"),
        "down_proj": ("model", None),
        "lm_head": (None, None),
    })
)


def _validate_positive_int(name: str, value: int) -> int:
  """Validates that `value` is a strictly positive int (not bool)."""
  if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
    raise ValueError(f"{name} must be a positive int, got {value!r}")
  return value


def _head_repeat_factor(heads: int, sharding_size: int) -> int:
  """Computes head replication factor matching `tpu_inference` sharding."""
  if heads >= sharding_size:
    if heads % sharding_size != 0:
      raise ValueError(
          f"Number of heads ({heads}) must be divisible by sharding_size"
          f" ({sharding_size})."
      )
    return 1
  if sharding_size % heads != 0:
    raise ValueError(
        f"sharding_size ({sharding_size}) must be divisible by number of heads"
        f" ({heads})."
    )
  return sharding_size // heads


def _strip_identity_ops(
    ops: Sequence[Mapping[str, Mapping[str, Any]]],
) -> list[dict[str, dict[str, Any]]]:
  """Drops no-op `pad` (`[0, 0]`), `repeat` (`times == 1`), and `scale` (`1.0`)."""
  stripped: list[dict[str, dict[str, Any]]] = []
  for step in ops:
    if "pad" in step and list(step["pad"]["sizes"]) == [0, 0]:
      continue
    if "repeat" in step and int(step["repeat"]["times"]) == 1:
      continue
    if "scale" in step and float(step["scale"]["factor"]) == 1.0:
      continue
    stripped.append({str(k): dict(v) for k, v in step.items()})
  return stripped


def build_from_hf_manifest(
    hf_config: Qwen3HfConfig,
    *,
    tp_size: int = 1,
    head_sharding_size: int | None = None,
    vocab_size: int | None = None,
) -> dict[str, Any]:
  """Builds the declarative `HF -> vLLM-JAX` manifest for a Qwen3 config."""
  if hf_config.attention_bias:
    raise ValueError(
        "tpu_inference Qwen3Attention does not support attention_bias=True."
    )
  _validate_positive_int("tp_size", tp_size)
  sharding_size = tp_size if head_sharding_size is None else head_sharding_size
  _validate_positive_int("head_sharding_size", sharding_size)

  v_size = _validate_positive_int(
      "vocab_size",
      hf_config.vocab_size if vocab_size is None else vocab_size,
  )
  d_model = _validate_positive_int("hidden_size", hf_config.hidden_size)
  _validate_positive_int("intermediate_size", hf_config.intermediate_size)
  num_layers = _validate_positive_int(
      "num_hidden_layers", hf_config.num_hidden_layers
  )
  num_heads = _validate_positive_int(
      "num_attention_heads", hf_config.num_attention_heads
  )
  num_kv_heads = _validate_positive_int(
      "num_key_value_heads",
      num_heads
      if hf_config.num_key_value_heads is None
      else hf_config.num_key_value_heads,
  )
  if hf_config.head_dim is None:
    if d_model % num_heads != 0:
      raise ValueError(
          f"hidden_size ({d_model}) must be divisible by num_attention_heads"
          f" ({num_heads}) when head_dim is None."
      )
    head_dim = d_model // num_heads
  else:
    head_dim = hf_config.head_dim
  _validate_positive_int("head_dim", head_dim)

  v_pad = ((v_size + tp_size - 1) // tp_size) * tp_size
  pad_v = v_pad - v_size
  h_pad = 64 if head_dim == 64 else ((head_dim + 127) // 128) * 128
  pad_h = h_pad - head_dim
  rep_q = _head_repeat_factor(num_heads, sharding_size)
  rep_kv = _head_repeat_factor(num_kv_heads, sharding_size)

  tie_word_embeddings = bool(hf_config.tie_word_embeddings)
  rendered = hf_mapping.render_json_templates(
      _QWEN3_FROM_HF_BASE_JSON,
      "" if tie_word_embeddings else _QWEN3_FROM_HF_LM_HEAD_JSON,
      d=d_model,
      n=num_heads,
      k=num_kv_heads,
      h=head_dim,
      nh=num_heads * head_dim,
      kh=num_kv_heads * head_dim,
      pad_v=pad_v,
      pad_h=pad_h,
      rep_q=rep_q,
      rep_kv=rep_kv,
      inv_rep_q=str(1.0 / rep_q),
  )
  template_rules = [
      {**tmpl, "ops": _strip_identity_ops(tmpl["ops"])} for tmpl in rendered
  ]
  return hf_mapping.generate_manifest(
      model_type="qwen3",
      template_rules=template_rules,
      axes={"i": num_layers},
      sharding_rules=_QWEN3_SHARDING_RULES,
      dropped_keys=["lm_head.weight"] if tie_word_embeddings else [],
  )
