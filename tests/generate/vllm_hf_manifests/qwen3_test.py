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

"""Unit tests for Qwen3 declarative HF -> vLLM-JAX weight manifest builder."""

from __future__ import annotations

import dataclasses
import json

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tunix.generate.vllm_hf_manifests import qwen3
from tunix.models import hf_mapping


@dataclasses.dataclass
class _FakeQwen3Config:
  vocab_size: int
  hidden_size: int
  intermediate_size: int
  num_hidden_layers: int
  num_attention_heads: int
  num_key_value_heads: int | None
  head_dim: int | None
  tie_word_embeddings: bool
  attention_bias: bool


def _make_hf_weights(
    cfg: _FakeQwen3Config,
    *,
    include_lm_head: bool = True,
) -> dict[str, jnp.ndarray]:
  rng = np.random.default_rng(0)
  d_model = cfg.hidden_size
  inter = cfg.intermediate_size
  n_heads = cfg.num_attention_heads
  n_kv = n_heads if cfg.num_key_value_heads is None else cfg.num_key_value_heads
  head_dim = d_model // n_heads if cfg.head_dim is None else cfg.head_dim

  def _arr(shape: tuple[int, ...]) -> jnp.ndarray:
    return jnp.asarray(rng.standard_normal(shape, dtype=np.float32))

  weights: dict[str, jnp.ndarray] = {
      "model.embed_tokens.weight": _arr((cfg.vocab_size, d_model)),
      "model.norm.weight": _arr((d_model,)),
  }
  for layer_idx in range(cfg.num_hidden_layers):
    p = f"model.layers.{layer_idx}"
    weights.update({
        f"{p}.input_layernorm.weight": _arr((d_model,)),
        f"{p}.post_attention_layernorm.weight": _arr((d_model,)),
        f"{p}.self_attn.q_norm.weight": _arr((head_dim,)),
        f"{p}.self_attn.k_norm.weight": _arr((head_dim,)),
        f"{p}.self_attn.q_proj.weight": _arr((n_heads * head_dim, d_model)),
        f"{p}.self_attn.k_proj.weight": _arr((n_kv * head_dim, d_model)),
        f"{p}.self_attn.v_proj.weight": _arr((n_kv * head_dim, d_model)),
        f"{p}.self_attn.o_proj.weight": _arr((d_model, n_heads * head_dim)),
        f"{p}.mlp.gate_proj.weight": _arr((inter, d_model)),
        f"{p}.mlp.up_proj.weight": _arr((inter, d_model)),
        f"{p}.mlp.down_proj.weight": _arr((d_model, inter)),
    })
  if include_lm_head:
    weights["lm_head.weight"] = _arr((cfg.vocab_size, d_model))
  return weights


class Qwen3VllmHfManifestTest(parameterized.TestCase):

  def test_protocol_compliance(self):
    cfg = _FakeQwen3Config(
        vocab_size=1003,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        tie_word_embeddings=True,
        attention_bias=False,
    )
    self.assertIsInstance(cfg, qwen3.Qwen3HfConfig)

  @parameterized.named_parameters(
      dict(
          testcase_name="padded_vocab_padded_head_kv_repeat_tied",
          vocab_size=1003,
          tp_size=4,
          head_sharding_size=None,
          head_dim=32,
          expected_v_pad=1004,
          expected_h_pad=128,
          num_heads=4,
          num_kv_heads=2,
          expected_rep_q=1,
          expected_rep_kv=2,
          tie_word_embeddings=True,
          include_lm_head=True,
      ),
      dict(
          testcase_name="head_dim_64_no_pad_untied",
          vocab_size=1024,
          tp_size=2,
          head_sharding_size=None,
          head_dim=64,
          expected_v_pad=1024,
          expected_h_pad=64,
          num_heads=4,
          num_kv_heads=2,
          expected_rep_q=1,
          expected_rep_kv=1,
          tie_word_embeddings=False,
          include_lm_head=True,
      ),
      dict(
          testcase_name="rep_q_and_rep_kv_both_gt_one",
          vocab_size=1000,
          tp_size=1,
          head_sharding_size=4,
          head_dim=64,
          expected_v_pad=1000,
          expected_h_pad=64,
          num_heads=2,
          num_kv_heads=1,
          expected_rep_q=2,
          expected_rep_kv=4,
          tie_word_embeddings=False,
          include_lm_head=True,
      ),
      dict(
          testcase_name="tied_without_lm_head_in_src",
          vocab_size=1000,
          tp_size=1,
          head_sharding_size=None,
          head_dim=128,
          expected_v_pad=1000,
          expected_h_pad=128,
          num_heads=2,
          num_kv_heads=1,
          expected_rep_q=1,
          expected_rep_kv=1,
          tie_word_embeddings=True,
          include_lm_head=False,
      ),
      dict(
          testcase_name="derived_head_dim_from_hidden_size",
          vocab_size=1024,
          tp_size=2,
          head_sharding_size=None,
          head_dim=None,
          expected_v_pad=1024,
          expected_h_pad=128,
          num_heads=2,
          num_kv_heads=None,
          expected_rep_q=1,
          expected_rep_kv=1,
          tie_word_embeddings=True,
          include_lm_head=True,
      ),
  )
  def test_round_trip_invertibility_and_shapes(
      self,
      vocab_size: int,
      tp_size: int,
      head_sharding_size: int | None,
      head_dim: int | None,
      expected_v_pad: int,
      expected_h_pad: int,
      num_heads: int,
      num_kv_heads: int | None,
      expected_rep_q: int,
      expected_rep_kv: int,
      tie_word_embeddings: bool,
      include_lm_head: bool,
  ):
    d_model = 64
    inter = 96
    eff_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
    cfg = _FakeQwen3Config(
        vocab_size=vocab_size,
        hidden_size=d_model,
        intermediate_size=inter,
        num_hidden_layers=2,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        tie_word_embeddings=tie_word_embeddings,
        attention_bias=False,
    )
    manifest = qwen3.build_from_hf_manifest(
        cfg, tp_size=tp_size, head_sharding_size=head_sharding_size
    )
    manifest_json = json.dumps(manifest)
    self.assertEqual(manifest["model_type"], "qwen3")

    if tie_word_embeddings:
      self.assertEqual(manifest["dropped_keys"], ["lm_head.weight"])
    else:
      self.assertEqual(manifest["dropped_keys"], [])

    hf_weights = _make_hf_weights(cfg, include_lm_head=include_lm_head)
    vllm_weights = hf_mapping.from_hf(manifest_json, hf_weights)

    self.assertEqual(
        vllm_weights["model.embed_tokens.weight"].shape,
        (expected_v_pad, d_model),
    )
    self.assertEqual(
        vllm_weights["model.layers.0.self_attn.q_norm.weight"].shape,
        (expected_h_pad,),
    )
    self.assertEqual(
        vllm_weights["model.layers.0.self_attn.q_proj.weight"].shape,
        (d_model, num_heads * expected_rep_q, expected_h_pad),
    )
    self.assertEqual(
        vllm_weights["model.layers.0.self_attn.k_proj.weight"].shape,
        (d_model, eff_kv_heads * expected_rep_kv, expected_h_pad),
    )
    self.assertEqual(
        vllm_weights["model.layers.0.self_attn.o_proj.weight"].shape,
        (num_heads * expected_rep_q, expected_h_pad, d_model),
    )
    self.assertEqual(
        vllm_weights["model.layers.0.mlp.gate_proj.weight"].shape,
        (d_model, inter),
    )
    self.assertEqual(
        vllm_weights["model.layers.0.mlp.down_proj.weight"].shape,
        (inter, d_model),
    )

    rules_by_target = {r["target_keys"][0]: r for r in manifest["rules"]}
    self.assertEqual(
        rules_by_target["model.embed_tokens.weight"]["sharding"],
        ["model", None],
    )
    self.assertEqual(
        rules_by_target["model.layers.0.self_attn.q_proj.weight"]["sharding"],
        [None, "model", None],
    )
    self.assertEqual(
        rules_by_target["model.layers.0.self_attn.o_proj.weight"]["sharding"],
        ["model", None, None],
    )
    self.assertEqual(
        rules_by_target["model.layers.0.mlp.gate_proj.weight"]["sharding"],
        [None, "model"],
    )
    self.assertEqual(
        rules_by_target["model.layers.0.mlp.down_proj.weight"]["sharding"],
        ["model", None],
    )

    if expected_rep_q > 1:
      self.assertEqual(
          rules_by_target["model.layers.0.self_attn.o_proj.weight"]["ops"][-1],
          {"scale": {"factor": 1.0 / expected_rep_q}},
      )
    else:
      self.assertEqual(
          rules_by_target["model.layers.0.self_attn.o_proj.weight"]["ops"][-1],
          {"transpose": {"axes": [1, 2, 0]}},
      )

    if tie_word_embeddings:
      self.assertNotIn("lm_head.weight", vllm_weights)
    else:
      self.assertEqual(
          vllm_weights["lm_head.weight"].shape, (d_model, expected_v_pad)
      )
      self.assertEqual(
          rules_by_target["lm_head.weight"]["sharding"], [None, None]
      )

    inv_manifest = hf_mapping.invert_manifest(manifest)
    recovered = hf_mapping.to_hf(json.dumps(inv_manifest), vllm_weights)
    expected_keys = set(hf_weights.keys()) - set(manifest["dropped_keys"])
    self.assertEqual(set(recovered.keys()), expected_keys)
    for k in expected_keys:
      np.testing.assert_allclose(recovered[k], hf_weights[k])

  def test_identity_pad_and_repeat_steps_are_stripped(self):
    cfg = _FakeQwen3Config(
        vocab_size=1024,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        tie_word_embeddings=False,
        attention_bias=False,
    )
    manifest = qwen3.build_from_hf_manifest(cfg, tp_size=2)
    rules_by_target = {r["target_keys"][0]: r for r in manifest["rules"]}
    self.assertEmpty(rules_by_target["model.embed_tokens.weight"]["ops"])
    self.assertEmpty(
        rules_by_target["model.layers.0.self_attn.q_norm.weight"]["ops"]
    )
    self.assertEqual(
        rules_by_target["model.layers.0.self_attn.q_proj.weight"]["ops"],
        [
            {"reshape": {"src_shape": [256, 128], "dst_shape": [4, 64, 128]}},
            {"transpose": {"axes": [2, 0, 1]}},
        ],
    )
    self.assertEqual(
        rules_by_target["lm_head.weight"]["ops"],
        [{"transpose": {"axes": [1, 0]}}],
    )

  def test_o_proj_head_replication_scales_and_preserves_attention_output(self):
    d_model = 64
    num_heads = 2
    head_dim = 64
    head_sharding_size = 4
    cfg = _FakeQwen3Config(
        vocab_size=128,
        hidden_size=d_model,
        intermediate_size=96,
        num_hidden_layers=1,
        num_attention_heads=num_heads,
        num_key_value_heads=1,
        head_dim=head_dim,
        tie_word_embeddings=True,
        attention_bias=False,
    )
    manifest = qwen3.build_from_hf_manifest(
        cfg, tp_size=1, head_sharding_size=head_sharding_size
    )
    rules_by_target = {r["target_keys"][0]: r for r in manifest["rules"]}
    self.assertEqual(
        rules_by_target["model.layers.0.self_attn.o_proj.weight"]["ops"][-1],
        {"scale": {"factor": 0.5}},
    )

    hf_weights = _make_hf_weights(cfg, include_lm_head=False)
    vllm_weights = hf_mapping.from_hf(json.dumps(manifest), hf_weights)

    rng = np.random.default_rng(7)
    q = jnp.asarray(
        rng.standard_normal((3, num_heads, head_dim), dtype=np.float32)
    )
    o = (
        hf_weights["model.layers.0.self_attn.o_proj.weight"]
        .reshape(d_model, num_heads, head_dim)
        .transpose(1, 2, 0)
    )
    q_rep = jnp.repeat(q, 2, axis=1)
    o_rep = vllm_weights["model.layers.0.self_attn.o_proj.weight"]
    np.testing.assert_allclose(
        np.asarray(
            jnp.einsum(
                "tnh,nhd->td",
                q_rep,
                o_rep,
                precision=jax.lax.Precision.HIGHEST,
            )
        ),
        np.asarray(
            jnp.einsum(
                "tnh,nhd->td",
                q,
                o,
                precision=jax.lax.Precision.HIGHEST,
            )
        ),
        rtol=1e-5,
        atol=1e-5,
    )

    inv_manifest = hf_mapping.invert_manifest(manifest)
    recovered = hf_mapping.to_hf(json.dumps(inv_manifest), vllm_weights)
    np.testing.assert_allclose(
        recovered["model.layers.0.self_attn.o_proj.weight"],
        hf_weights["model.layers.0.self_attn.o_proj.weight"],
        rtol=1e-5,
        atol=1e-5,
    )

  def test_indivisible_heads_and_invalid_args_raise(self):
    cfg = _FakeQwen3Config(
        vocab_size=1024,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=6,
        num_key_value_heads=2,
        head_dim=None,
        tie_word_embeddings=True,
        attention_bias=False,
    )
    with self.assertRaisesRegex(ValueError, "divisible"):
      qwen3.build_from_hf_manifest(cfg, tp_size=4)
    with self.assertRaisesRegex(ValueError, "divisible"):
      qwen3.build_from_hf_manifest(cfg, tp_size=2)
    with self.assertRaisesRegex(ValueError, "divisible"):
      qwen3.build_from_hf_manifest(
          dataclasses.replace(cfg, head_dim=32), tp_size=1, head_sharding_size=3
      )
    with self.assertRaisesRegex(ValueError, "tp_size"):
      qwen3.build_from_hf_manifest(cfg, tp_size=0)
    with self.assertRaisesRegex(ValueError, "tp_size"):
      qwen3.build_from_hf_manifest(cfg, tp_size=True)
    with self.assertRaisesRegex(ValueError, "head_sharding_size"):
      qwen3.build_from_hf_manifest(cfg, tp_size=2, head_sharding_size=-1)

  def test_attention_bias_raises(self):
    cfg = _FakeQwen3Config(
        vocab_size=1024,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=None,
        tie_word_embeddings=True,
        attention_bias=True,
    )
    with self.assertRaisesRegex(ValueError, "attention_bias=True"):
      qwen3.build_from_hf_manifest(cfg, tp_size=2)

  def test_qwen3_hf_config_protocol_is_covariant_for_non_optional_fields(self):
    @dataclasses.dataclass
    class _ConcreteQwen3Config:
      vocab_size: int = 1024
      hidden_size: int = 128
      intermediate_size: int = 256
      num_hidden_layers: int = 1
      num_attention_heads: int = 2
      num_key_value_heads: int = 2
      head_dim: int = 64
      tie_word_embeddings: bool = True
      attention_bias: bool = False

    concrete_cfg = _ConcreteQwen3Config()
    proto_cfg: qwen3.Qwen3HfConfig = concrete_cfg
    self.assertIsInstance(concrete_cfg, qwen3.Qwen3HfConfig)
    manifest = qwen3.build_from_hf_manifest(proto_cfg, tp_size=2)
    self.assertEqual(manifest["model_type"], "qwen3")


if __name__ == "__main__":
  absltest.main()
