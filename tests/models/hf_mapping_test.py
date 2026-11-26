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

"""Unit, JAX execution, and checkpoint round-trip tests for `hf_mapping`."""

import dataclasses
import enum
import json
import os
from typing import Any, cast
from unittest import mock

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import safetensors.numpy as safe_np
from tunix.models import hf_mapping
from tunix.models import safetensors_loader
from tunix.models.gemma import model as gemma_model
from tunix.models.gemma import params_safetensors as gemma_params_safetensors
from tunix.models.gemma3 import model as gemma3_model
from tunix.models.gemma3 import params_safetensors as gemma3_params_safetensors
from tunix.models.gemma3 import vision as gemma3_vision
from tunix.models.llama3 import model as llama3_model
from tunix.models.llama3 import params as llama3_params
from tunix.models.qwen2 import model as qwen2_model
from tunix.models.qwen2 import params as qwen2_params
from tunix.models.qwen3 import model as qwen3_model
from tunix.models.qwen3 import params as qwen3_params
from tunix.models.qwen3p5 import model as qwen3p5_model
from tunix.models.qwen3p5 import params as qwen3p5_params


class HfMappingTest(absltest.TestCase):
  """Tests PrimOp schema, inversion, templates, JAX engine, and 7-model loading."""

  def test_all_prim_ops_invert_round_trip(self):
    manifest = {
        "schema_version": hf_mapping.SCHEMA_VERSION,
        "model_type": "test_prim_ops",
        "rules": [
            {
                "source_keys": ["k1"],
                "target_keys": ["t1"],
                "ops": [
                    {"transpose": {"axes": [2, 0, 1]}},
                    {
                        "reshape": {
                            "src_shape": [4, 2, 3],
                            "dst_shape": [8, 3],
                        }
                    },
                    {"pad": {"axis": 0, "sizes": [8, 2]}},
                    {"unpad": {"axis": 0, "sizes": [8, 2]}},
                    {"repeat": {"axis": 1, "times": 2}},
                    {"unrepeat": {"axis": 1, "times": 2}},
                    {
                        "cast": {
                            "src_dtype": "float32",
                            "dst_dtype": "bfloat16",
                        }
                    },
                ],
                "sharding": ["tp", None],
            },
            {
                "source_keys": ["a", "b"],
                "target_keys": ["ab_concat"],
                "ops": [{"concat": {"axis": 0, "sizes": [4, 4]}}],
            },
            {
                "source_keys": ["ab_split"],
                "target_keys": ["s0", "s1"],
                "ops": [{"split": {"axis": 0, "sizes": [3, 5]}}],
            },
            {
                "source_keys": ["e0", "e1"],
                "target_keys": ["e_stack"],
                "ops": [{"stack": {"axis": 0}}],
            },
            {
                "source_keys": ["e_unstack"],
                "target_keys": ["u0", "u1"],
                "ops": [{"unstack": {"axis": 0}}],
            },
            {
                "source_keys": ["id_src"],
                "target_keys": ["id_dst"],
                "ops": [{"scale": {"factor": 2.0}}],
            },
        ],
        "tied_keys": [["embed", "lm_head"]],
        "dropped_keys": ["rotary_emb.inv_freq"],
    }
    inv = hf_mapping.invert_manifest(manifest)
    self.assertEqual(inv["tied_keys"], [])
    self.assertIn("lm_head", inv["dropped_keys"])
    self.assertNotIn("sharding", inv["rules"][0])
    round_trip = hf_mapping.invert_manifest(inv)
    expected_rules = [
        {k: v for k, v in r.items() if k != "sharding"}
        for r in manifest["rules"]
    ]
    self.assertEqual(round_trip["rules"], expected_rules)

  def test_spec_dataclass_and_parse_manifest(self):
    spec = hf_mapping.WeightMappingSpec(
        rules=(
            hf_mapping.ParamRule(
                source_keys=("hf.w",),
                target_keys=("tunix.w",),
                ops=({"transpose": {"axes": [1, 0]}},),
                sharding=("tp", None),
            ),
            hf_mapping.ParamRule(
                source_keys=("shared.norm",),
                target_keys=("shared.norm",),
                ops=({"scale": {"factor": 2.0}},),
            ),
        ),
        tied_keys=(("embed", "lm_head"),),
        dropped_keys=("unused",),
        model_type="test_model",
    )
    as_dict = spec.to_dict()
    self.assertEqual(as_dict["model_type"], "test_model")
    self.assertTrue(spec.to_json().startswith('{"dropped_keys":'))
    self.assertEqual(hf_mapping.parse_manifest(spec), as_dict)
    self.assertEqual(hf_mapping.parse_manifest(spec.to_json()), as_dict)
    self.assertEqual(
        hf_mapping.WeightMappingSpec.from_json(spec.to_json()), spec
    )

    inv = hf_mapping.invert_manifest(spec)
    self.assertEqual(inv["rules"][0]["source_keys"], ["tunix.w"])
    self.assertEqual(inv["rules"][0]["target_keys"], ["hf.w"])
    self.assertEqual(inv["rules"][1]["ops"], [{"scale": {"factor": 0.5}}])
    self.assertEqual(inv["tied_keys"], [])
    self.assertIn("lm_head", inv["dropped_keys"])

  def test_generate_manifest_and_template_expansion(self):
    templates = hf_mapping.render_json_templates(
        """[
          {
            "id": "embed_tokens",
            "source_pattern": ["model.embed_tokens.weight"],
            "target_pattern": ["embedder.input_embedding"],
            "ops": [{"cast": {"src_dtype": "bfloat16", "dst_dtype": "float32"}}]
          },
          {
            "id": "q_proj",
            "source_pattern": ["model.layers.{i}.self_attn.q_proj.weight"],
            "target_pattern": ["layers.{i}.attn.q_proj.w"],
            "ops": [
              {"transpose": {"axes": [1, 0]}},
              {"reshape": {"src_shape": [$d, $d], "dst_shape": [$d, $n, $h]}},
              {"repeat": {"axis": 1, "times": 2}}
            ]
          },
          {
            "id": "moe_gate_up",
            "axes": {"i": [1]},
            "source_pattern": [
              "model.layers.{i}.mlp.experts.{e}.gate.weight",
              "model.layers.{i}.mlp.experts.{e}.up.weight"
            ],
            "target_pattern": ["layers.{i}.mlp.gate_up_proj"],
            "ops": [
              {"concat": {"axis": 1, "sizes": [$s, $s]}},
              {"stack": {"axis": 0}}
            ]
          }
        ]""",
        "",
        d=16,
        n=4,
        h=4,
        s=8,
    )
    manifest = hf_mapping.generate_manifest(
        model_type="generic_moe",
        template_rules=templates,
        axes={"i": 2, "e": 2},
        sharding_rules={"q_proj": ["fsdp", "tp", None]},
        tied_keys=[("embedder.input_embedding", "lm_head.w")],
        dropped_keys=["rotary_emb.inv_freq"],
    )
    self.assertEqual(manifest["model_type"], "generic_moe")
    self.assertEqual(
        manifest["tied_keys"], [["embedder.input_embedding", "lm_head.w"]]
    )
    self.assertEqual(manifest["dropped_keys"], ["rotary_emb.inv_freq"])
    self.assertLen(manifest["rules"], 4)
    self.assertEqual(
        manifest["rules"][1]["ops"][1],
        {"reshape": {"src_shape": [16, 16], "dst_shape": [16, 4, 4]}},
    )
    self.assertEqual(manifest["rules"][1]["sharding"], ["fsdp", "tp", None])
    self.assertEqual(
        manifest["rules"][3]["source_keys"],
        [
            "model.layers.1.mlp.experts.0.gate.weight",
            "model.layers.1.mlp.experts.0.up.weight",
            "model.layers.1.mlp.experts.1.gate.weight",
            "model.layers.1.mlp.experts.1.up.weight",
        ],
    )
    inv = hf_mapping.invert_manifest(manifest)
    self.assertEqual(inv["tied_keys"], [])
    self.assertIn("lm_head.w", inv["dropped_keys"])
    self.assertEqual(
        hf_mapping.invert_manifest(inv)["rules"],
        [
            {k: v for k, v in r.items() if k != "sharding"}
            for r in manifest["rules"]
        ],
    )
    src_keys = hf_mapping.active_source_keys(manifest)
    self.assertIn("model.embed_tokens.weight", src_keys)
    self.assertNotIn("rotary_emb.inv_freq", src_keys)
    self.assertEqual(
        hf_mapping.generate_manifest("empty", [])["rules"], []
    )
    self.assertEqual(hf_mapping.active_source_keys({"rules": []}), frozenset())
    dropped_src = hf_mapping.active_source_keys({
        "rules": [
            {"source_keys": ["keep.w"], "target_keys": ["k.w"], "ops": []},
            {"source_keys": ["drop.w"], "target_keys": ["d.w"], "ops": []},
        ],
        "dropped_keys": ["drop.w"],
    })
    self.assertEqual(dropped_src, frozenset({"keep.w"}))

    # Engine-level variant affixes are template placeholders: one template
    # expands to any prefix / suffix layout via explicit template vars, and
    # `lora_mapping_from_manifest` follows the expanded manifest.
    affix_json = """[
      {
        "source_pattern": ["${lm}embed_tokens.weight"],
        "target_pattern": ["embedder.input_embedding"],
        "ops": []
      },
      {
        "source_pattern": ["${lm}layers.{i}.experts.gate_up_proj${moe}"],
        "target_pattern": ["layers.{i}.moe.gating_einsum"],
        "ops": [{"transpose": {"axes": [1, 0]}}]
      },
      {
        "source_pattern": ["${lm}layers.{i}.self_attn.q_proj.weight"],
        "target_pattern": ["layers.{i}.attn.q_einsum.w"],
        "ops": [
          {"reshape": {"src_shape": [8, 4], "dst_shape": [2, 4, 4]}},
          {"transpose": {"axes": [0, 2, 1]}}
        ]
      },
      {
        "source_pattern": ["${lm}layers.{i}.conv.weight"],
        "target_pattern": ["layers.{i}.conv.kernel"],
        "ops": [{"transpose": {"axes": [2, 1, 0]}}]
      },
      {
        "source_pattern": ["${lm}layers.{i}.bias"],
        "target_pattern": ["layers.{i}.bias"],
        "ops": []
      }
    ]"""
    wrapped_m = hf_mapping.generate_manifest(
        model_type="affix_test/wrapped",
        template_rules=hf_mapping.render_json_templates(
            affix_json, lm="model.language_model.", moe=""
        ),
        axes={"i": 1},
        dropped_keys=["lm_head.weight"],
    )
    self.assertEqual(
        wrapped_m["rules"][0]["source_keys"],
        ["model.language_model.embed_tokens.weight"],
    )
    self.assertEqual(
        wrapped_m["rules"][1]["source_keys"],
        ["model.language_model.layers.0.experts.gate_up_proj"],
    )
    rewritten_m = hf_mapping.generate_manifest(
        model_type="affix_test/bare",
        template_rules=hf_mapping.render_json_templates(
            affix_json, lm="", moe=".weight"
        ),
        axes={"i": 1},
        dropped_keys=["lm_head.weight"],
    )
    self.assertEqual(
        rewritten_m["rules"][0]["source_keys"], ["embed_tokens.weight"]
    )
    self.assertEqual(
        rewritten_m["rules"][1]["source_keys"],
        ["layers.0.experts.gate_up_proj.weight"],
    )
    self.assertEqual(rewritten_m["dropped_keys"], ["lm_head.weight"])
    self.assertEqual(rewritten_m["model_type"], "affix_test/bare")
    # A layout that does not define every affix of the template fails fast.
    with self.assertRaises(KeyError):
      hf_mapping.render_json_templates(affix_json, lm="model.")
    lora_map, lora_tr = hf_mapping.lora_mapping_from_manifest(
        hf_mapping.invert_manifest(rewritten_m)
    )
    self.assertEqual(
        lora_map["layers.0.attn.q_einsum"], "layers.0.self_attn.q_proj.weight"
    )
    self.assertEqual(
        lora_tr,
        {"layers.0.self_attn.q_proj.weight": (1, 0)},
    )

  def test_validation_rejects_invalid_ops_and_rules(self):
    for op_name, bad_kw in (
        ("unknown_op", {}),
        ("stack", {"axis": "0"}),
        ("stack", {"axis": True}),
        ("unstack", {"axis": "0"}),
        ("unstack", {"axis": False}),
        ("transpose", {"axes": []}),
        ("transpose", {"axes": [0, 0]}),
        ("reshape", {"src_shape": [], "dst_shape": []}),
        ("reshape", {"src_shape": [2], "dst_shape": []}),
        ("reshape", {"src_shape": [-2, -3], "dst_shape": [6]}),
        ("reshape", {"src_shape": [6], "dst_shape": [-2, -3]}),
        ("reshape", {"src_shape": [2, 3], "dst_shape": [2, 2]}),
        ("concat", {"axis": "0", "sizes": [2, 2]}),
        ("concat", {"axis": True, "sizes": [2, 2]}),
        ("concat", {"axis": 0, "sizes": []}),
        ("concat", {"axis": 0, "sizes": [0, 2]}),
        ("split", {"axis": "0", "sizes": [2, 2]}),
        ("split", {"axis": True, "sizes": [2, 2]}),
        ("split", {"axis": 0, "sizes": []}),
        ("split", {"axis": 0, "sizes": [0, 2]}),
        ("pad", {"axis": "0", "sizes": [1, 1]}),
        ("pad", {"axis": True, "sizes": [1, 1]}),
        ("pad", {"axis": 0, "sizes": [4]}),
        ("pad", {"axis": 0, "sizes": [1, -1]}),
        ("unpad", {"axis": "0", "sizes": [1, 1]}),
        ("unpad", {"axis": True, "sizes": [1, 1]}),
        ("unpad", {"axis": 0, "sizes": [4]}),
        ("unpad", {"axis": 0, "sizes": [1, -1]}),
        ("repeat", {"axis": "0", "times": 2}),
        ("repeat", {"axis": True, "times": 2}),
        ("repeat", {"axis": 0, "times": True}),
        ("repeat", {"axis": 0, "times": 0}),
        ("unrepeat", {"axis": "0", "times": 2}),
        ("unrepeat", {"axis": True, "times": 2}),
        ("unrepeat", {"axis": 0, "times": False}),
        ("unrepeat", {"axis": 0, "times": 0}),
        ("cast", {"src_dtype": "float32", "dst_dtype": ""}),
        ("scale", {"factor": 0.0}),
        ("scale", {"factor": float("inf")}),
        ("scale", {"factor": float("nan")}),
    ):
      with self.assertRaises(ValueError):
        hf_mapping.validate_op(op_name, bad_kw)
      with self.assertRaises(ValueError):
        hf_mapping.invert_op(op_name, bad_kw)

    with self.assertRaises(ValueError):
      hf_mapping.ParamRule(source_keys=(), target_keys=("a",))
    with self.assertRaises(ValueError):
      hf_mapping.ParamRule.from_dict(
          {"source_keys": ["a"], "target_keys": ["b"], "ops": [], "extra": 1}
      )

    for bad_rule in (
        {"source_keys": [], "target_keys": ["a"], "ops": []},
        {"source_keys": "abc", "target_keys": ["a"], "ops": []},
        {
            "source_keys": ["a"],
            "target_keys": ["b"],
            "ops": [],
            "sharding": "tp",
        },
        {"source_keys": ["a"], "target_keys": ["b"], "ops": "not_a_list"},
        {"source_keys": ["a"], "target_keys": ["b"], "ops": [{}]},
        {"source_keys": ["a"], "target_keys": ["b"], "ops": [{"unknown": {}}]},
    ):
      with self.assertRaises(ValueError):
        hf_mapping.validate_rule(bad_rule)

    for bad_manifest in (
        {"schema_version": "0.0", "rules": []},
        "[]",
        "123",
        {"rules": [], "dropped_keys": "abc"},
        {"rules": [], "tied_keys": ["ab"]},
        {"rules": "not_a_list"},
        {"rules": [{"source_keys": [], "target_keys": ["a"], "ops": []}]},
    ):
      with self.assertRaises(ValueError):
        hf_mapping.parse_manifest(bad_manifest)

  def test_all_prim_ops_forward_and_inverse_numeric_round_trip(self):
    src = {
        "x": jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4),
        "p": jnp.ones((2, 3), dtype=jnp.float32),
        "r": jnp.arange(6, dtype=jnp.float32).reshape(2, 3),
        "a": jnp.ones((2, 4), dtype=jnp.float32),
        "b": jnp.full((3, 4), 2.0, dtype=jnp.float32),
        "e0": jnp.arange(6, dtype=jnp.float32).reshape(2, 3),
        "e1": jnp.arange(6, 12, dtype=jnp.float32).reshape(2, 3),
        "shared": jnp.ones((2, 2), dtype=jnp.float32),
        "drop_me": jnp.zeros((2,), dtype=jnp.float32),
    }
    manifest = {
        "schema_version": hf_mapping.SCHEMA_VERSION,
        "model_type": "numeric_prim_ops",
        "rules": [
            {
                "source_keys": ["x"],
                "target_keys": ["y"],
                "ops": [
                    {"transpose": {"axes": [2, 0, 1]}},
                    {
                        "reshape": {
                            "src_shape": [4, 2, 3],
                            "dst_shape": [8, 3],
                        }
                    },
                    {"scale": {"factor": 2.0}},
                    {
                        "cast": {
                            "src_dtype": "float32",
                            "dst_dtype": "bfloat16",
                        }
                    },
                ],
                "sharding": [None, None],
            },
            {
                "source_keys": ["p"],
                "target_keys": ["p_padded"],
                "ops": [{"pad": {"axis": 0, "sizes": [1, 2]}}],
            },
            {
                "source_keys": ["r"],
                "target_keys": ["r_repeated"],
                "ops": [{"repeat": {"axis": 1, "times": 2}}],
            },
            {
                "source_keys": ["a", "b"],
                "target_keys": ["ab"],
                "ops": [{"concat": {"axis": 0, "sizes": [2, 3]}}],
            },
            {
                "source_keys": ["e0", "e1"],
                "target_keys": ["e_stacked"],
                "ops": [{"stack": {"axis": 1}}],
            },
            {
                "source_keys": ["shared"],
                "target_keys": ["shared"],
                "ops": [{"scale": {"factor": 4.0}}],
            },
        ],
        "tied_keys": [
            ["ab", "ab_tied"],
            ["ab", "ab_dropped_tie"],
            ["missing_src", "missing_dst"],
        ],
        "dropped_keys": ["drop_me", "ab_dropped_tie"],
    }
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("tp",))
    manifest_json = hf_mapping.WeightMappingSpec.from_dict(manifest).to_json()
    target = hf_mapping.from_hf(manifest_json, src, mesh=mesh)
    self.assertIn("ab_tied", target)
    self.assertNotIn("ab_dropped_tie", target)
    self.assertNotIn("missing_dst", target)
    self.assertEqual(target["y"].dtype, jnp.bfloat16)
    self.assertEqual(target["p_padded"].shape, (5, 3))
    self.assertEqual(target["r_repeated"].shape, (2, 6))
    self.assertEqual(target["e_stacked"].shape, (2, 2, 3))
    np.testing.assert_allclose(target["shared"], src["shared"] * 4.0)

    expected_hf_keys = set(src.keys()) - {"drop_me"}
    inv_manifest = hf_mapping.invert_manifest(manifest)
    inv_json = hf_mapping.WeightMappingSpec.from_dict(inv_manifest).to_json()
    recovered = hf_mapping.to_hf(inv_json, target)
    self.assertEqual(set(recovered.keys()), expected_hf_keys)
    for k in expected_hf_keys:
      np.testing.assert_allclose(recovered[k], src[k])

    ckpt_loaded = hf_mapping.from_hf_impl(manifest, src)
    ckpt_saved = hf_mapping.to_hf_impl(inv_manifest, ckpt_loaded)
    self.assertEqual(set(ckpt_saved.keys()), expected_hf_keys)
    np.testing.assert_allclose(ckpt_saved["x"], src["x"])

  def test_fused_jax_jit_to_hf_impl_and_from_hf_impl_colocated_transfer(self):
    trainer_src_spec = hf_mapping.WeightMappingSpec.from_dict({
        "schema_version": hf_mapping.SCHEMA_VERSION,
        "model_type": "trainer_model_a",
        "rules": [
            {
                "source_keys": ["trainer.kv_padded_rep"],
                "target_keys": ["hf.k", "hf.v"],
                "ops": [
                    {"unpad": {"axis": 0, "sizes": [1, 0]}},
                    {"unrepeat": {"axis": 1, "times": 2}},
                    {"split": {"axis": 0, "sizes": [2, 2]}},
                ],
            },
            {
                "source_keys": ["trainer.experts_3d"],
                "target_keys": ["hf.e0", "hf.e1"],
                "ops": [{"unstack": {"axis": 0}}],
            },
        ],
        "tied_keys": [["trainer.kv_padded_rep", "trainer.kv_alias"]],
    })
    sampler_dst_spec = hf_mapping.WeightMappingSpec.from_dict({
        "schema_version": hf_mapping.SCHEMA_VERSION,
        "model_type": "sampler_model_b",
        "rules": [
            {
                "source_keys": ["hf.k", "hf.v"],
                "target_keys": ["sampler.kv_einsum"],
                "ops": [
                    {"transpose": {"axes": [1, 0]}},
                    {"stack": {"axis": 0}},
                ],
                "sharding": [["tp"], None, None],
            },
            {
                "source_keys": ["hf.e0", "hf.e1"],
                "target_keys": ["sampler.experts_cat"],
                "ops": [{"concat": {"axis": 0, "sizes": [2, 2]}}],
            },
        ],
        "tied_keys": [["sampler.kv_einsum", "sampler.kv_tied"]],
    })
    k_base = jnp.arange(4, dtype=jnp.float32).reshape(2, 2)
    v_base = jnp.arange(4, 8, dtype=jnp.float32).reshape(2, 2)
    kv_rep = jnp.repeat(jnp.concatenate([k_base, v_base], axis=0), 2, axis=1)
    kv_padded_rep = jnp.pad(kv_rep, ((1, 0), (0, 0)))
    src_trainer = {
        "trainer.kv_padded_rep": kv_padded_rep,
        "trainer.kv_alias": kv_padded_rep,
        "trainer.experts_3d": (
            jnp.arange(12, dtype=jnp.float32).reshape(2, 2, 3)
        ),
    }
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("tp",))
    fused_transfer = jax.jit(
        lambda w: hf_mapping.from_hf_impl(
            sampler_dst_spec,
            hf_mapping.to_hf_impl(trainer_src_spec, w),
            mesh=mesh,
        )
    )
    abstract_out = jax.eval_shape(fused_transfer, src_trainer)
    self.assertEqual(abstract_out["sampler.kv_einsum"].shape, (2, 2, 2))
    self.assertEqual(abstract_out["sampler.kv_tied"].shape, (2, 2, 2))
    self.assertEqual(abstract_out["sampler.experts_cat"].shape, (4, 3))

    expected_kv = jnp.stack([k_base.T, v_base.T], axis=0)
    expected_experts = src_trainer["trainer.experts_3d"].reshape(4, 3)
    transferred = fused_transfer(src_trainer)
    np.testing.assert_allclose(transferred["sampler.kv_einsum"], expected_kv)
    np.testing.assert_allclose(transferred["sampler.kv_tied"], expected_kv)
    np.testing.assert_allclose(
        transferred["sampler.experts_cat"], expected_experts
    )

    hf_via_boundary = hf_mapping.to_hf(trainer_src_spec.to_json(), src_trainer)
    self.assertNotIn("trainer.kv_alias", hf_via_boundary)
    np.testing.assert_allclose(hf_via_boundary["hf.k"], k_base)
    np.testing.assert_allclose(hf_via_boundary["hf.v"], v_base)
    sampler_via_boundary = hf_mapping.from_hf(
        sampler_dst_spec.to_json(), hf_via_boundary, mesh=mesh
    )
    np.testing.assert_allclose(
        sampler_via_boundary["sampler.kv_einsum"], expected_kv
    )
    np.testing.assert_allclose(
        sampler_via_boundary["sampler.experts_cat"], expected_experts
    )

  def test_runtime_shape_validation_and_missing_key_fail_fast(self):
    t23 = jnp.ones((2, 3), dtype=jnp.float32)
    t33 = jnp.ones((3, 3), dtype=jnp.float32)
    cat_out = hf_mapping.apply_primitive_op(
        [t23, t33, t23, t33], "concat", {"axis": 0, "sizes": [2, 3]}
    )
    self.assertLen(cat_out, 2)
    self.assertEqual(cat_out[0].shape, (5, 3))
    split_out = hf_mapping.apply_primitive_op(
        [jnp.arange(6, dtype=jnp.float32)],
        "split",
        {"axis": 0, "sizes": [1, 2, 3]},
    )
    self.assertEqual([t.shape for t in split_out], [(1,), (2,), (3,)])

    for op_name, tensors, kw in (
        ("scale", [], {"factor": 0.0}),
        ("stack", [], {"axis": 0}),
        ("stack", [t23, t33], {"axis": 0}),
        ("unstack", [t23, t23], {"axis": 0}),
        ("reshape", [t23], {"src_shape": [3, 2], "dst_shape": [6]}),
        ("transpose", [t23], {"axes": [0, 1, 2]}),
        ("concat", [], {"axis": 0, "sizes": [2, 3]}),
        ("concat", [t23], {"axis": 0, "sizes": [2, 3]}),
        ("concat", [t23, t23], {"axis": 0, "sizes": [2, 3]}),
        ("split", [t23], {"axis": 0, "sizes": [1, 2]}),
        ("unpad", [t23], {"axis": 0, "sizes": [1, 1]}),
        ("unrepeat", [t33], {"axis": 0, "times": 2}),
        ("unrepeat", [jnp.zeros((5, 2))], {"axis": 0, "times": 2}),
        (
            "cast",
            [jnp.ones((2,), dtype=jnp.float16)],
            {"src_dtype": "float32", "dst_dtype": "bfloat16"},
        ),
    ):
      with self.assertRaises(ValueError):
        hf_mapping.apply_primitive_op(tensors, op_name, kw)
    with self.assertRaisesRegex(ValueError, "Non-identical repeated slices"):
      hf_mapping._apply_primitive_op(
          [np.array([-3.0, -8.0], dtype=np.float32)],
          "unrepeat",
          {"axis": 0, "times": 2},
          use_numpy=True,
      )

    with self.assertRaisesRegex(ValueError, "Duplicate target_keys"):
      hf_mapping.parse_manifest({
          "schema_version": hf_mapping.SCHEMA_VERSION,
          "rules": [{"source_keys": ["k1"], "target_keys": ["t1"], "ops": []}],
          "tied_keys": [["t1", "t1"]],
      })

    np_s = hf_mapping._apply_primitive_op(
        [np.array([3.5], dtype=np.float32)],
        "unstack",
        {"axis": 0},
        use_numpy=True,
    )
    self.assertIsInstance(np_s[0], np.ndarray)
    self.assertEqual(np_s[0].shape, ())
    np_cast = hf_mapping._apply_primitive_op(
        [np.array([1.0, 2.0], dtype=np.float32)],
        "cast",
        {"src_dtype": "float32", "dst_dtype": "bfloat16"},
        use_numpy=True,
    )[0]
    self.assertEqual(np_cast.dtype, np.dtype("bfloat16"))

    for bad_rule in (
        {"source_keys": [], "target_keys": ["a"], "ops": []},
        {"source_keys": ["k1"], "target_keys": ["a", "b"], "ops": []},
        {"source_keys": ["k1"], "target_keys": ["a"], "ops": [], "extra": 1},
    ):
      with self.assertRaises(ValueError):
        hf_mapping.apply_rule(bad_rule, {"k1": t23})

    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("tp",))
    with self.assertRaises(ValueError):
      hf_mapping.apply_rule(
          {
              "source_keys": ["k1"],
              "target_keys": ["a"],
              "ops": [],
              "sharding": ["missing_axis", None],
          },
          {"k1": t23},
          mesh=mesh,
      )
    skipped = hf_mapping.execute_manifest(
        {
            "schema_version": hf_mapping.SCHEMA_VERSION,
            "rules": [{
                "source_keys": ["dropped_src"],
                "target_keys": ["dropped_dst"],
                "ops": [],
            }],
            "dropped_keys": ["dropped_src"],
        },
        {"k1": t23},
    )
    self.assertEqual(set(skipped.keys()), {"k1"})

    multi_rule_manifest = {
        "schema_version": hf_mapping.SCHEMA_VERSION,
        "rules": [
            {"source_keys": ["k1"], "target_keys": ["t1"], "ops": []},
            {"source_keys": ["k2"], "target_keys": ["t2"], "ops": []},
        ],
    }
    with self.assertRaises(KeyError):
      hf_mapping.execute_manifest(multi_rule_manifest, {"k1": t23})

  def test_streaming_executor_and_moe_multi_pattern_expansion(self):
    stream_manifest = {
        "schema_version": hf_mapping.SCHEMA_VERSION,
        "rules": [
            {
                "source_keys": [
                    "model.layers.0.self_attn.k_proj.weight",
                    "model.layers.0.self_attn.v_proj.weight",
                ],
                "target_keys": ["layers.0.attn.kv_einsum.w"],
                "ops": [{"stack": {"axis": 0}}],
            },
            {
                "source_keys": ["dropped_src.weight"],
                "target_keys": ["dropped_dst.weight"],
                "ops": [],
            },
        ],
        "tied_keys": [["embed.w", "lm_head.w"]],
        "dropped_keys": ["dropped_src.weight", "unused.weight"],
    }
    pending_buf: dict[str, jax.Array] = {}
    step_exec = hf_mapping.make_streaming_executor(
        stream_manifest, pending=pending_buf
    )
    k_arr = jnp.ones((2, 4), dtype=jnp.float32)
    v_arr = jnp.full((2, 4), 3.0, dtype=jnp.float32)
    norm_arr = jnp.ones((4,), dtype=jnp.float32)

    out_1 = step_exec({
        "model.layers.0.self_attn.k_proj.weight": k_arr,
        "embed.w": norm_arr,
        "unused.weight": norm_arr,
    })
    self.assertEqual(set(out_1.keys()), {"embed.w", "lm_head.w"})
    self.assertEqual(
        set(pending_buf.keys()), {"model.layers.0.self_attn.k_proj.weight"}
    )

    out_2 = step_exec({"model.layers.0.self_attn.v_proj.weight": v_arr})
    self.assertEqual(set(out_2.keys()), {"layers.0.attn.kv_einsum.w"})
    self.assertEmpty(pending_buf)
    np.testing.assert_allclose(
        out_2["layers.0.attn.kv_einsum.w"], jnp.stack([k_arr, v_arr], axis=0)
    )

    out_3 = step_exec({"final_norm.scale": norm_arr})
    self.assertEqual(set(out_3.keys()), {"final_norm.scale"})

    # Streaming executor must run pure host NumPy (zero `jax.devices` calls)
    # and support shared `source_keys` across shards without premature eviction.
    shared_manifest = {
        "schema_version": hf_mapping.SCHEMA_VERSION,
        "rules": [
            {
                "source_keys": ["shared.k1"],
                "target_keys": ["out.single"],
                "ops": [{"scale": {"factor": 2.0}}],
            },
            {
                "source_keys": ["shared.k1", "shared.k2"],
                "target_keys": ["out.stacked"],
                "ops": [{"stack": {"axis": 0}}],
            },
            {
                "source_keys": ["only_target_dropped.src"],
                "target_keys": ["only_target_dropped.dst"],
                "ops": [],
            },
        ],
        "dropped_keys": ["only_target_dropped.dst"],
    }
    shared_pending: dict[str, np.ndarray] = {}
    orig_devices = jax.devices
    try:
      jax.devices = lambda *a, **kw: (_ for _ in ()).throw(
          RuntimeError("Unknown backend cpu")
      )
      shared_exec = hf_mapping.make_streaming_executor(
          shared_manifest, pending=shared_pending
      )
      s1 = shared_exec({
          "shared.k1": np.array([1.0, 2.0], dtype=np.float32),
          "only_target_dropped.src": np.array([9.0], dtype=np.float32),
      })
      self.assertEqual(set(s1.keys()), {"out.single"})
      self.assertIsInstance(s1["out.single"], np.ndarray)
      np.testing.assert_allclose(s1["out.single"], [2.0, 4.0])
      self.assertEqual(set(shared_pending.keys()), {"shared.k1"})
      s2 = shared_exec({"shared.k2": np.array([3.0, 4.0], dtype=np.float32)})
      self.assertEqual(set(s2.keys()), {"out.stacked"})
      self.assertEmpty(shared_pending)
      np.testing.assert_allclose(
          s2["out.stacked"], [[1.0, 2.0], [3.0, 4.0]]
      )
    finally:
      jax.devices = orig_devices

    moe_manifest = hf_mapping.generate_manifest(
        model_type="moe_multi_pattern",
        template_rules=[{
            "source_pattern": [
                "layers.{i}.experts.{e}.gate.weight",
                "layers.{i}.experts.{e}.up.weight",
            ],
            "target_pattern": ["layers.{i}.mlp.gate_up"],
            "ops": [
                {"concat": {"axis": 1, "sizes": [2, 2]}},
                {"stack": {"axis": 0}},
            ],
        }],
        axes={"i": 1, "e": 2},
    )
    hf_moe = {
        "layers.0.experts.0.gate.weight": jnp.full((2, 2), 1.0),
        "layers.0.experts.0.up.weight": jnp.full((2, 2), 2.0),
        "layers.0.experts.1.gate.weight": jnp.full((2, 2), 3.0),
        "layers.0.experts.1.up.weight": jnp.full((2, 2), 4.0),
    }
    fused = hf_mapping.from_hf_impl(moe_manifest, hf_moe)
    np.testing.assert_allclose(
        fused["layers.0.mlp.gate_up"][0],
        jnp.concatenate(
            [
                hf_moe["layers.0.experts.0.gate.weight"],
                hf_moe["layers.0.experts.0.up.weight"],
            ],
            axis=1,
        ),
    )
    np.testing.assert_allclose(
        fused["layers.0.mlp.gate_up"][1],
        jnp.concatenate(
            [
                hf_moe["layers.0.experts.1.gate.weight"],
                hf_moe["layers.0.experts.1.up.weight"],
            ],
            axis=1,
        ),
    )
    recovered_moe = hf_mapping.to_hf_impl(
        hf_mapping.invert_manifest(moe_manifest), fused
    )
    for k, v in hf_moe.items():
      np.testing.assert_allclose(recovered_moe[k], v, err_msg=k)

  def test_qwen3_dense_and_moe_checkpoint_round_trip(self):
    for num_experts in (None, 2):
      cfg = qwen3_model.ModelConfig(
          num_layers=1,
          vocab_size=32,
          embed_dim=16,
          hidden_dim=32,
          num_heads=2,
          head_dim=8,
          num_kv_heads=1,
          rope_theta=10000,
          norm_eps=1e-6,
          num_experts=num_experts,
          num_experts_per_tok=1 if num_experts else None,
      )
      from_manifest = qwen3_params.build_from_hf_manifest(cfg)
      to_manifest = qwen3_params.build_to_hf_manifest(cfg)
      self.assertEqual(
          to_manifest,
          hf_mapping.invert_manifest(from_manifest),
      )
      model = qwen3_model.Qwen3(cfg, rngs=nnx.Rngs(params=0))
      model.non_param_cache = nnx.Cache(jnp.ones((2,), dtype=jnp.float32))
      hf_state = qwen3_params.export_to_hf_state_dict(model, cfg)
      self.assertNotIn("non_param_cache", hf_state)
      ckpt_dir = self.create_tempdir().full_path
      safe_np.save_file(
          {k: np.ascontiguousarray(np.asarray(v)) for k, v in hf_state.items()},
          os.path.join(ckpt_dir, "model.safetensors"),
      )
      loaded = qwen3_params.create_model_from_safe_tensors(
          ckpt_dir, cfg, dtype=jnp.float32
      )
      orig_flat = safetensors_loader.flatten_model_state(model)
      self.assertNotIn("non_param_cache", orig_flat)
      load_flat = safetensors_loader.flatten_model_state(loaded)
      for key, val in orig_flat.items():
        np.testing.assert_allclose(load_flat[key], val, err_msg=key)

      tokens = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
      positions = jnp.arange(4, dtype=jnp.int32)[None, :]
      mask = jnp.tril(jnp.ones((1, 4, 4), dtype=jnp.bool_))
      logits_orig, _ = model(tokens, positions, None, mask)
      logits_load, _ = loaded(tokens, positions, None, mask)
      np.testing.assert_allclose(logits_load, logits_orig, rtol=1e-5, atol=1e-5)
      re_hf = qwen3_params.export_to_hf_state_dict(loaded, cfg)
      for key, val in hf_state.items():
        np.testing.assert_allclose(re_hf[key], val, err_msg=key)

      round_trip = hf_mapping.from_hf_impl(from_manifest, hf_state)
      for key, val in orig_flat.items():
        np.testing.assert_allclose(round_trip[key], val, err_msg=key)

  def test_gemma3_and_gemma2_checkpoint_round_trip(self):
    tokens = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
    positions = jnp.arange(4, dtype=jnp.int32)[None, :]
    mask = jnp.tril(jnp.ones((1, 4, 4), dtype=jnp.bool_))
    siglip_cfg = gemma3_vision.SigLIPConfig(
        num_mm_tokens_per_image_prepool=4,
        num_mm_tokens_per_image=4,
        image_height=8,
        image_width=8,
        image_channels=3,
        patch_size=(4, 4),
        width=8,
        depth=1,
        mlp_dim=16,
        num_heads=2,
    )
    for num_kv_heads in (1, 2):
      cfg3 = gemma3_model.ModelConfig(
          num_layers=1,
          num_embed=32,
          embed_dim=16,
          hidden_dim=32,
          num_heads=2,
          head_dim=8,
          num_kv_heads=num_kv_heads,
          sliding_window_size=16,
          param_dtype=jnp.float32,
          vision_config=siglip_cfg if num_kv_heads == 1 else None,
      )
      g3_variants = gemma3_params_safetensors.HfVariant
      variant3 = (
          g3_variants.CONDITIONAL_GENERATION_LEGACY
          if num_kv_heads == 1
          else g3_variants.CAUSAL_LM
      )
      from_manifest3 = gemma3_params_safetensors.build_from_hf_manifest(
          cfg3, variant3
      )
      self.assertEqual(from_manifest3["model_type"], f"gemma3/{variant3.value}")
      self.assertEqual(
          gemma3_params_safetensors.build_to_hf_manifest(cfg3, variant3),
          hf_mapping.invert_manifest(from_manifest3),
      )
      model3 = gemma3_model.Gemma3(cfg3, rngs=nnx.Rngs(params=0))
      hf_state = gemma3_params_safetensors.export_to_hf_state_dict(
          model3, cfg3, hf_variant=variant3
      )
      ckpt_dir = self.create_tempdir().full_path
      safe_np.save_file(
          {k: np.ascontiguousarray(np.asarray(v)) for k, v in hf_state.items()},
          os.path.join(ckpt_dir, "model.safetensors"),
      )
      loaded3 = gemma3_params_safetensors.create_model_from_safe_tensors(
          ckpt_dir, cfg3, dtype=jnp.float32, hf_variant=variant3
      )
      orig_flat = safetensors_loader.flatten_model_state(model3)
      load_flat = safetensors_loader.flatten_model_state(loaded3)
      for key, val in orig_flat.items():
        np.testing.assert_allclose(load_flat[key], val, err_msg=key)
      logits3_orig, _ = model3(tokens, positions, None, mask)
      logits3_load, _ = loaded3(tokens, positions, None, mask)
      np.testing.assert_allclose(
          logits3_load, logits3_orig, rtol=1e-5, atol=1e-5
      )
      if num_kv_heads == 1:
        # Published multimodal repos: `language_model.model.` text decoder and
        # top-level towers.
        self.assertIn("language_model.model.embed_tokens.weight", hf_state)
        self.assertIn(
            "vision_tower.vision_model.post_layernorm.weight", hf_state
        )
        self.assertIn(
            "language_model.lm_head.weight", from_manifest3["dropped_keys"]
        )
        # transformers v5 `Gemma3ForConditionalGeneration`: everything under
        # `model.`, tied `lm_head.weight` at top level.
        v5_hf = gemma3_params_safetensors.export_to_hf_state_dict(
            model3, cfg3, hf_variant=g3_variants.CONDITIONAL_GENERATION
        )
        v5_hf["lm_head.weight"] = jnp.ones((32, 16), dtype=jnp.float32)
        self.assertIn("model.language_model.embed_tokens.weight", v5_hf)
        self.assertIn(
            "model.vision_tower.vision_model.post_layernorm.weight", v5_hf
        )
        self.assertIn(
            "model.multi_modal_projector.mm_soft_emb_norm.weight", v5_hf
        )
        v5_dir = self.create_tempdir().full_path
        safe_np.save_file(
            {k: np.ascontiguousarray(np.asarray(v)) for k, v in v5_hf.items()},
            os.path.join(v5_dir, "model.safetensors"),
        )
        loaded3_v5 = gemma3_params_safetensors.create_model_from_safe_tensors(
            v5_dir,
            cfg3,
            dtype=jnp.float32,
            hf_variant=g3_variants.CONDITIONAL_GENERATION,
        )
        logits3_v5, _ = loaded3_v5(tokens, positions, None, mask)
        np.testing.assert_allclose(
            logits3_v5, logits3_orig, rtol=1e-5, atol=1e-5
        )
        # A checkpoint in another layout than the declared variant fails and
        # the error names the declared variant: nothing is probed.
        for wrong in (variant3, g3_variants.CAUSAL_LM):
          with self.assertRaisesRegex(
              ValueError,
              f"declared HF layout 'gemma3/{wrong.value}'; missing",
          ):
            gemma3_params_safetensors.create_model_from_safe_tensors(
                v5_dir,
                dataclasses.replace(cfg3, vision_config=None),
                dtype=jnp.float32,
                hf_variant=wrong,
            )
        # `CAUSAL_LM` has no tower affix: a multimodal config cannot use it.
        with self.assertRaisesRegex(
            ValueError,
            "Gemma3 vision config requires a CONDITIONAL_GENERATION",
        ):
          gemma3_params_safetensors.build_from_hf_manifest(
              cfg3, g3_variants.CAUSAL_LM
          )
      else:
        self.assertIn("model.embed_tokens.weight", hf_state)

      cfg2 = gemma_model.ModelConfig(
          num_layers=1,
          num_embed=32,
          embed_dim=16,
          hidden_dim=32,
          num_heads=2,
          head_dim=8,
          num_kv_heads=num_kv_heads,
          final_logit_softcap=None,
          attention_types=(gemma_model.AttentionType.GLOBAL,),
          use_post_attn_norm=True,
          use_post_ffw_norm=True,
      )
      # Gemma2 has one layout (`GemmaForCausalLM`, `model.`): feed it the text
      # decoder weights of the Gemma3 checkpoint under that prefix.
      gemma2_dir = self.create_tempdir().full_path
      safe_np.save_file(
          {
              k.removeprefix("language_model."): np.ascontiguousarray(
                  np.asarray(v)
              )
              for k, v in hf_state.items()
              if not k.startswith(("vision_tower.", "multi_modal_projector."))
          },
          os.path.join(gemma2_dir, "model.safetensors"),
      )
      loaded2 = gemma_params_safetensors.create_model_from_safe_tensors(
          gemma2_dir, cfg2, dtype=jnp.float32
      )
      flat2 = safetensors_loader.flatten_model_state(loaded2)
      attn_key = (
          "layers.0.attn.qkv_einsum.w"
          if num_kv_heads == 2
          else "layers.0.attn.kv_einsum.w"
      )
      np.testing.assert_allclose(
          flat2["embedder.input_embedding"],
          orig_flat["embedder.input_embedding"],
      )
      np.testing.assert_allclose(flat2[attn_key], orig_flat[attn_key])
      np.testing.assert_allclose(
          flat2["layers.0.attn.attn_vec_einsum.w"],
          orig_flat["layers.0.attn.attn_vec_einsum.w"],
      )
      gemma2_manifest = gemma_params_safetensors.build_from_hf_manifest(cfg2)
      self.assertEqual(
          gemma_params_safetensors.build_to_hf_manifest(cfg2),
          hf_mapping.invert_manifest(gemma2_manifest),
      )
      hf_state2 = gemma_params_safetensors.export_to_hf_state_dict(
          loaded2, cfg2
      )
      rt2 = gemma_params_safetensors.from_hf_impl(gemma2_manifest, hf_state2)
      np.testing.assert_allclose(rt2[attn_key], flat2[attn_key])
      self.assertEqual(
          hf_mapping.invert_manifest(
              hf_mapping.invert_manifest(gemma2_manifest)
          ),
          gemma2_manifest,
      )
    cfg1 = gemma_model.ModelConfig(
        num_layers=1,
        num_embed=32,
        embed_dim=16,
        hidden_dim=32,
        num_heads=2,
        head_dim=8,
        num_kv_heads=1,
        final_logit_softcap=None,
        attention_types=(gemma_model.AttentionType.GLOBAL,),
        use_post_attn_norm=False,
        use_post_ffw_norm=False,
    )
    model1 = gemma_model.Gemma(cfg1, rngs=nnx.Rngs(params=0))
    orig1 = safetensors_loader.flatten_model_state(model1)
    hf1 = gemma_params_safetensors.export_to_hf_state_dict(model1, cfg1)
    self.assertIn("model.layers.0.post_attention_layernorm.weight", hf1)
    self.assertNotIn("model.layers.0.pre_feedforward_layernorm.weight", hf1)
    self.assertNotIn("model.layers.0.post_feedforward_layernorm.weight", hf1)
    np.testing.assert_array_equal(
        hf1["model.layers.0.post_attention_layernorm.weight"],
        orig1["layers.0.pre_ffw_norm.scale"],
    )
    ckpt1 = self.create_tempdir().full_path
    safe_np.save_file(
        {k: np.ascontiguousarray(np.asarray(v)) for k, v in hf1.items()},
        os.path.join(ckpt1, "model.safetensors"),
    )
    loaded1 = gemma_params_safetensors.create_model_from_safe_tensors(
        ckpt1, cfg1, dtype=jnp.float32
    )
    logits1_orig, _ = model1(tokens, positions, None, mask)
    logits1_load, _ = loaded1(tokens, positions, None, mask)
    np.testing.assert_allclose(logits1_load, logits1_orig, rtol=1e-5, atol=1e-5)
    rt1 = gemma_params_safetensors.from_hf_impl(
        gemma_params_safetensors.build_from_hf_manifest(cfg1), hf1
    )
    self.assertEqual(set(rt1.keys()), set(orig1.keys()))
    for k, v in orig1.items():
      np.testing.assert_allclose(rt1[k], v, err_msg=k)
    for bad_fn in (
        gemma_params_safetensors.build_from_hf_manifest,
        gemma_params_safetensors.build_to_hf_manifest,
    ):
      with self.assertRaises(ValueError):
        bad_fn(
            gemma_model.ModelConfig(
                num_layers=1,
                num_embed=32,
                embed_dim=15,
                hidden_dim=32,
                num_heads=3,
                head_dim=5,
                num_kv_heads=1,
                final_logit_softcap=None,
                attention_types=(gemma_model.AttentionType.GLOBAL,),
                use_post_attn_norm=True,
                use_post_ffw_norm=True,
            )
        )
    for bad_fn3 in (
        gemma3_params_safetensors.build_from_hf_manifest,
        gemma3_params_safetensors.build_to_hf_manifest,
    ):
      with self.assertRaisesRegex(ValueError, "head_dim must be even"):
        bad_fn3(
            gemma3_model.ModelConfig(
                num_layers=1,
                num_embed=32,
                embed_dim=15,
                hidden_dim=32,
                num_heads=3,
                head_dim=5,
                num_kv_heads=1,
                sliding_window_size=16,
            ),
            hf_variant=gemma3_params_safetensors.HfVariant.CAUSAL_LM,
        )

  def test_llama3_qwen2_and_qwen3p5_checkpoint_round_trip(self):
    tokens = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
    positions = jnp.arange(4, dtype=jnp.int32)[None, :]
    mask = jnp.tril(jnp.ones((1, 4, 4), dtype=jnp.bool_))
    cfg_llama = llama3_model.ModelConfig(
        num_layers=1,
        vocab_size=32,
        embed_dim=16,
        hidden_dim=32,
        num_heads=2,
        head_dim=8,
        num_kv_heads=1,
        rope_theta=10000,
        norm_eps=1e-5,
        weight_tying=False,
    )
    model_llama = llama3_model.Llama3(cfg_llama, rngs=nnx.Rngs(params=0))
    hf_llama = llama3_params.export_to_hf_state_dict(model_llama, cfg_llama)
    ckpt_llama = self.create_tempdir().full_path
    safe_np.save_file(
        {k: np.ascontiguousarray(np.asarray(v)) for k, v in hf_llama.items()},
        os.path.join(ckpt_llama, "model.safetensors"),
    )
    loaded_llama = llama3_params.create_model_from_safe_tensors(
        ckpt_llama, cfg_llama, dtype=jnp.float32
    )
    orig_llama = safetensors_loader.flatten_model_state(model_llama)
    load_llama = safetensors_loader.flatten_model_state(loaded_llama)
    for k, v in orig_llama.items():
      np.testing.assert_allclose(load_llama[k], v, err_msg=k)
    logits_llama_orig, _ = model_llama(tokens, positions, None, mask)
    logits_llama_load, _ = loaded_llama(tokens, positions, None, mask)
    np.testing.assert_allclose(
        logits_llama_load, logits_llama_orig, rtol=1e-5, atol=1e-5
    )
    llama_manifest = llama3_params.build_from_hf_manifest(cfg_llama)
    llama_to_manifest = llama3_params.build_to_hf_manifest(cfg_llama)
    self.assertEqual(
        llama_to_manifest, hf_mapping.invert_manifest(llama_manifest)
    )
    rt_llama = llama3_params.from_hf_impl(
        llama_manifest,
        llama3_params.to_hf_impl(llama_to_manifest, orig_llama),
    )
    for k, v in orig_llama.items():
      np.testing.assert_allclose(rt_llama[k], v, err_msg=k)
    cfg_llama_tied = llama3_model.ModelConfig(
        num_layers=1,
        vocab_size=32,
        embed_dim=16,
        hidden_dim=32,
        num_heads=2,
        head_dim=8,
        num_kv_heads=1,
        rope_theta=10000,
        norm_eps=1e-5,
        weight_tying=True,
    )
    self.assertEqual(
        llama3_params.build_from_hf_manifest(cfg_llama_tied)["dropped_keys"],
        ["lm_head.weight"],
    )

    cfg_qwen2 = qwen2_model.ModelConfig(
        num_layers=1,
        vocab_size=32,
        embed_dim=16,
        hidden_dim=32,
        num_heads=2,
        head_dim=8,
        num_kv_heads=1,
        rope_theta=10000,
        norm_eps=1e-6,
        use_tied_embedding=False,
    )
    self.assertEqual(
        qwen2_params.build_to_hf_manifest(cfg_qwen2),
        hf_mapping.invert_manifest(
            qwen2_params.build_from_hf_manifest(cfg_qwen2)
        ),
    )
    model_qwen2 = qwen2_model.Qwen2(cfg_qwen2, rngs=nnx.Rngs(params=0))
    hf_qwen2 = qwen2_params.export_to_hf_state_dict(model_qwen2, cfg_qwen2)
    ckpt_qwen2 = self.create_tempdir().full_path
    safe_np.save_file(
        {k: np.ascontiguousarray(np.asarray(v)) for k, v in hf_qwen2.items()},
        os.path.join(ckpt_qwen2, "model.safetensors"),
    )
    loaded_qwen2 = qwen2_params.create_model_from_safe_tensors(
        ckpt_qwen2, cfg_qwen2, dtype=jnp.float32
    )
    orig_qwen2 = safetensors_loader.flatten_model_state(model_qwen2)
    load_qwen2 = safetensors_loader.flatten_model_state(loaded_qwen2)
    for k, v in orig_qwen2.items():
      np.testing.assert_allclose(load_qwen2[k], v, err_msg=k)
    logits_qwen2_orig, _ = model_qwen2(tokens, positions, None, mask)
    logits_qwen2_load, _ = loaded_qwen2(tokens, positions, None, mask)
    np.testing.assert_allclose(
        logits_qwen2_load, logits_qwen2_orig, rtol=1e-5, atol=1e-5
    )
    cfg_qwen2_tied = qwen2_model.ModelConfig(
        num_layers=1,
        vocab_size=32,
        embed_dim=16,
        hidden_dim=32,
        num_heads=2,
        head_dim=8,
        num_kv_heads=1,
        rope_theta=10000,
        norm_eps=1e-6,
        use_tied_embedding=True,
    )
    self.assertEqual(
        qwen2_params.build_from_hf_manifest(cfg_qwen2_tied)["dropped_keys"],
        ["lm_head.weight"],
    )
    model_qwen2_tied = qwen2_model.Qwen2(
        cfg_qwen2_tied, rngs=nnx.Rngs(params=0)
    )
    orig_qwen2_tied = safetensors_loader.flatten_model_state(model_qwen2_tied)
    hf_qwen2_tied = qwen2_params.export_to_hf_state_dict(
        model_qwen2_tied, cfg_qwen2_tied
    )
    rt_qwen2_tied = qwen2_params.from_hf_impl(
        qwen2_params.build_from_hf_manifest(cfg_qwen2_tied),
        {**hf_qwen2_tied, "lm_head.weight": jnp.ones((32, 16))},
    )
    self.assertEqual(set(rt_qwen2_tied.keys()), set(orig_qwen2_tied.keys()))
    for k, v in orig_qwen2_tied.items():
      np.testing.assert_allclose(rt_qwen2_tied[k], v, err_msg=k)

    cfg_q35 = qwen3p5_model.ModelConfig(
        num_layers=2,
        vocab_size=32,
        embed_dim=16,
        hidden_dim=32,
        num_heads=2,
        head_dim=8,
        num_kv_heads=1,
        rope_theta=10000,
        norm_eps=1e-6,
        use_tied_embedding=False,
        full_attention_interval=2,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=2,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
    )
    q35_cg = qwen3p5_params.HfVariant.CONDITIONAL_GENERATION
    self.assertEqual(
        qwen3p5_params.build_to_hf_manifest(cfg_q35, q35_cg),
        hf_mapping.invert_manifest(
            qwen3p5_params.build_from_hf_manifest(cfg_q35, q35_cg)
        ),
    )
    model_q35 = qwen3p5_model.Qwen3p5(cfg_q35, rngs=nnx.Rngs(params=0))
    hf_q35 = qwen3p5_params.export_to_hf_state_dict(
        model_q35, cfg_q35, hf_variant=q35_cg
    )
    self.assertIn("model.language_model.embed_tokens.weight", hf_q35)
    self.assertIn("lm_head.weight", hf_q35)
    ckpt_q35 = self.create_tempdir().full_path
    safe_np.save_file(
        {k: np.ascontiguousarray(np.asarray(v)) for k, v in hf_q35.items()},
        os.path.join(ckpt_q35, "model.safetensors"),
    )
    loaded_q35 = qwen3p5_params.create_model_from_safe_tensors(
        ckpt_q35, cfg_q35, dtype=jnp.float32, hf_variant=q35_cg
    )
    orig_q35 = safetensors_loader.flatten_model_state(model_q35)
    load_q35 = safetensors_loader.flatten_model_state(loaded_q35)
    for k, v in orig_q35.items():
      np.testing.assert_allclose(load_q35[k], v, err_msg=k)
    logits_q35_orig, _ = model_q35(tokens, positions, None, mask)
    logits_q35_load, _ = loaded_q35(tokens, positions, None, mask)
    np.testing.assert_allclose(
        logits_q35_load, logits_q35_orig, rtol=1e-5, atol=1e-5
    )

  def test_six_families_random_weights_legacy_parity_and_round_trip(self):
    """Bit-by-bit parity vs HEAD legacy mappings + round-trip on random weights.

    DEPRECATED (testing-only): The hand-written legacy reference conversion
    dicts (`ref_*`) below are deprecated, retained strictly for bit-by-bit
    parity testing during migration to `hf_mapping`, and will be removed in a
    follow-up cleanup CL.
    """
    # DEPRECATED (testing-only): Legacy hand-written reference dicts below are
    # deprecated, retained strictly for bit-by-bit parity testing during
    # migration to `hf_mapping`, and will be removed in a follow-up cleanup CL.
    rng = np.random.default_rng(42)

    def rand(*shape: int) -> jax.Array:
      return jnp.asarray(rng.standard_normal(shape, dtype=np.float32))

    # 1. Llama3 (asymmetric dims: V=19, D=12, F=17, N=3, K=1, H=4).
    cfg_llama = llama3_model.ModelConfig(
        num_layers=1,
        vocab_size=19,
        embed_dim=12,
        hidden_dim=17,
        num_heads=3,
        head_dim=4,
        num_kv_heads=1,
        rope_theta=10000,
        norm_eps=1e-5,
        weight_tying=False,
    )
    hf_llama = {
        "model.embed_tokens.weight": rand(19, 12),
        "model.norm.weight": rand(12),
        "lm_head.weight": rand(19, 12),
        "model.layers.0.input_layernorm.weight": rand(12),
        "model.layers.0.post_attention_layernorm.weight": rand(12),
        "model.layers.0.self_attn.q_proj.weight": rand(12, 12),
        "model.layers.0.self_attn.k_proj.weight": rand(4, 12),
        "model.layers.0.self_attn.v_proj.weight": rand(4, 12),
        "model.layers.0.self_attn.o_proj.weight": rand(12, 12),
        "model.layers.0.mlp.gate_proj.weight": rand(17, 12),
        "model.layers.0.mlp.up_proj.weight": rand(17, 12),
        "model.layers.0.mlp.down_proj.weight": rand(12, 17),
    }
    ref_llama = {
        "embedder.input_embedding": hf_llama["model.embed_tokens.weight"],
        "final_norm.w": hf_llama["model.norm.weight"],
        "lm_head.w": hf_llama["lm_head.weight"].T,
        "layers.0.input_layernorm.w": hf_llama[
            "model.layers.0.input_layernorm.weight"
        ],
        "layers.0.post_attention_layernorm.w": hf_llama[
            "model.layers.0.post_attention_layernorm.weight"
        ],
        "layers.0.attn.q_proj.w": hf_llama[
            "model.layers.0.self_attn.q_proj.weight"
        ].T.reshape(12, 3, 4),
        "layers.0.attn.k_proj.w": hf_llama[
            "model.layers.0.self_attn.k_proj.weight"
        ].T.reshape(12, 1, 4),
        "layers.0.attn.v_proj.w": hf_llama[
            "model.layers.0.self_attn.v_proj.weight"
        ].T.reshape(12, 1, 4),
        "layers.0.attn.o_proj.w": hf_llama[
            "model.layers.0.self_attn.o_proj.weight"
        ].T.reshape(3, 4, 12),
        "layers.0.mlp.gate_proj.kernel": hf_llama[
            "model.layers.0.mlp.gate_proj.weight"
        ].T,
        "layers.0.mlp.up_proj.kernel": hf_llama[
            "model.layers.0.mlp.up_proj.weight"
        ].T,
        "layers.0.mlp.down_proj.kernel": hf_llama[
            "model.layers.0.mlp.down_proj.weight"
        ].T,
    }
    m_llama = llama3_params.build_from_hf_manifest(cfg_llama)
    got_llama = llama3_params.from_hf_impl(m_llama, hf_llama)
    self.assertEqual(set(got_llama.keys()), set(ref_llama.keys()))
    for k, v in ref_llama.items():
      np.testing.assert_array_equal(got_llama[k], v, err_msg=f"llama3:{k}")
    rt_llama = llama3_params.to_hf_impl(
        llama3_params.build_to_hf_manifest(cfg_llama), got_llama
    )
    for k, v in hf_llama.items():
      np.testing.assert_array_equal(rt_llama[k], v, err_msg=f"llama3_rt:{k}")

    # 2. Qwen2 (adds Q/K/V biases).
    cfg_qwen2 = qwen2_model.ModelConfig(
        num_layers=1,
        vocab_size=19,
        embed_dim=12,
        hidden_dim=17,
        num_heads=3,
        head_dim=4,
        num_kv_heads=1,
        rope_theta=10000,
        norm_eps=1e-6,
        use_tied_embedding=False,
    )
    hf_qwen2 = {
        **hf_llama,
        "model.layers.0.self_attn.q_proj.bias": rand(12),
        "model.layers.0.self_attn.k_proj.bias": rand(4),
        "model.layers.0.self_attn.v_proj.bias": rand(4),
    }
    ref_qwen2 = {
        **ref_llama,
        "layers.0.attn.q_bias": hf_qwen2[
            "model.layers.0.self_attn.q_proj.bias"
        ],
        "layers.0.attn.k_bias": hf_qwen2[
            "model.layers.0.self_attn.k_proj.bias"
        ],
        "layers.0.attn.v_bias": hf_qwen2[
            "model.layers.0.self_attn.v_proj.bias"
        ],
    }
    got_qwen2 = qwen2_params.from_hf_impl(
        qwen2_params.build_from_hf_manifest(cfg_qwen2), hf_qwen2
    )
    self.assertEqual(set(got_qwen2.keys()), set(ref_qwen2.keys()))
    for k, v in ref_qwen2.items():
      np.testing.assert_array_equal(got_qwen2[k], v, err_msg=f"qwen2:{k}")
    rt_qwen2 = qwen2_params.to_hf_impl(
        qwen2_params.build_to_hf_manifest(cfg_qwen2), got_qwen2
    )
    for k, v in hf_qwen2.items():
      np.testing.assert_array_equal(rt_qwen2[k], v, err_msg=f"qwen2_rt:{k}")

    # 3. Qwen3 MoE (3 experts, q_norm/k_norm, router + stacked gate/up/down).
    cfg_q3_moe = qwen3_model.ModelConfig(
        num_layers=1,
        vocab_size=19,
        embed_dim=12,
        hidden_dim=17,
        num_heads=3,
        head_dim=4,
        num_kv_heads=1,
        rope_theta=10000,
        norm_eps=1e-6,
        use_tied_embedding=False,
        num_experts=3,
        num_experts_per_tok=2,
    )
    hf_q3 = {
        k: v
        for k, v in hf_llama.items()
        if ".mlp." not in k
    }
    hf_q3["model.layers.0.self_attn.q_norm.weight"] = rand(4)
    hf_q3["model.layers.0.self_attn.k_norm.weight"] = rand(4)
    hf_q3["model.layers.0.mlp.gate.weight"] = rand(3, 12)
    for e in range(3):
      hf_q3[f"model.layers.0.mlp.experts.{e}.gate_proj.weight"] = rand(17, 12)
      hf_q3[f"model.layers.0.mlp.experts.{e}.up_proj.weight"] = rand(17, 12)
      hf_q3[f"model.layers.0.mlp.experts.{e}.down_proj.weight"] = rand(12, 17)
    ref_q3 = {
        k: v
        for k, v in ref_llama.items()
        if ".mlp." not in k
    }
    ref_q3["layers.0.attn.q_norm.w"] = hf_q3[
        "model.layers.0.self_attn.q_norm.weight"
    ]
    ref_q3["layers.0.attn.k_norm.w"] = hf_q3[
        "model.layers.0.self_attn.k_norm.weight"
    ]
    ref_q3["layers.0.mlp.router.kernel"] = hf_q3[
        "model.layers.0.mlp.gate.weight"
    ].T
    for kw in ("gate_proj", "up_proj", "down_proj"):
      ref_q3[f"layers.0.mlp.{kw}"] = jnp.stack(
          [
              hf_q3[f"model.layers.0.mlp.experts.{e}.{kw}.weight"].T
              for e in range(3)
          ],
          axis=0,
      )
    got_q3 = qwen3_params.from_hf_impl(
        qwen3_params.build_from_hf_manifest(cfg_q3_moe), hf_q3
    )
    self.assertEqual(set(got_q3.keys()), set(ref_q3.keys()))
    for k, v in ref_q3.items():
      np.testing.assert_array_equal(got_q3[k], v, err_msg=f"qwen3_moe:{k}")
    rt_q3 = qwen3_params.to_hf_impl(
        qwen3_params.build_to_hf_manifest(cfg_q3_moe), got_q3
    )
    for k, v in hf_q3.items():
      np.testing.assert_array_equal(rt_q3[k], v, err_msg=f"qwen3_moe_rt:{k}")

    # 4. Qwen3.5 (2 layers: layer 0 GatedDeltaNet + layer 1 Full Attention).
    cfg_q35 = qwen3p5_model.ModelConfig(
        num_layers=2,
        vocab_size=19,
        embed_dim=12,
        hidden_dim=17,
        num_heads=3,
        head_dim=4,
        num_kv_heads=1,
        rope_theta=10000,
        norm_eps=1e-6,
        use_tied_embedding=False,
        attn_output_gate=True,
        full_attention_interval=2,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=6,
        linear_value_head_dim=6,
        linear_conv_kernel_dim=3,
    )
    hf_q35 = {
        "model.embed_tokens.weight": rand(19, 12),
        "model.norm.weight": rand(12),
        "lm_head.weight": rand(19, 12),
    }
    ref_q35 = {
        "embedder.input_embedding": hf_q35["model.embed_tokens.weight"],
        "final_norm.w": hf_q35["model.norm.weight"],
        "lm_head.w": hf_q35["lm_head.weight"].T,
    }
    for i in range(2):
      hf_q35[f"model.layers.{i}.input_layernorm.weight"] = rand(12)
      hf_q35[f"model.layers.{i}.post_attention_layernorm.weight"] = rand(12)
      hf_q35[f"model.layers.{i}.mlp.gate_proj.weight"] = rand(17, 12)
      hf_q35[f"model.layers.{i}.mlp.up_proj.weight"] = rand(17, 12)
      hf_q35[f"model.layers.{i}.mlp.down_proj.weight"] = rand(12, 17)
      ref_q35[f"layers.{i}.input_layernorm.w"] = hf_q35[
          f"model.layers.{i}.input_layernorm.weight"
      ]
      ref_q35[f"layers.{i}.post_attention_layernorm.w"] = hf_q35[
          f"model.layers.{i}.post_attention_layernorm.weight"
      ]
      ref_q35[f"layers.{i}.mlp.gate_proj.kernel"] = hf_q35[
          f"model.layers.{i}.mlp.gate_proj.weight"
      ].T
      ref_q35[f"layers.{i}.mlp.up_proj.kernel"] = hf_q35[
          f"model.layers.{i}.mlp.up_proj.weight"
      ].T
      ref_q35[f"layers.{i}.mlp.down_proj.kernel"] = hf_q35[
          f"model.layers.{i}.mlp.down_proj.weight"
      ].T
    # Layer 0: GatedDeltaNet; Layer 1: Full Attention with output gate (2*H=8).
    hf_q35["model.layers.0.linear_attn.in_proj_qkv.weight"] = rand(36, 12)
    hf_q35["model.layers.0.linear_attn.in_proj_z.weight"] = rand(12, 12)
    hf_q35["model.layers.0.linear_attn.in_proj_b.weight"] = rand(2, 12)
    hf_q35["model.layers.0.linear_attn.in_proj_a.weight"] = rand(2, 12)
    hf_q35["model.layers.0.linear_attn.out_proj.weight"] = rand(12, 12)
    hf_q35["model.layers.0.linear_attn.conv1d.weight"] = rand(36, 1, 3)
    hf_q35["model.layers.0.linear_attn.dt_bias"] = rand(2)
    hf_q35["model.layers.0.linear_attn.A_log"] = rand(2)
    hf_q35["model.layers.0.linear_attn.norm.weight"] = rand(6)
    linear_names = (
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_b",
        "in_proj_a",
        "out_proj",
    )
    for name in linear_names:
      ref_q35[f"layers.0.linear_attn.{name}.kernel"] = hf_q35[
          f"model.layers.0.linear_attn.{name}.weight"
      ].T
    ref_q35["layers.0.linear_attn.conv1d.kernel"] = jnp.transpose(
        hf_q35["model.layers.0.linear_attn.conv1d.weight"], (2, 1, 0)
    )
    ref_q35["layers.0.linear_attn.dt_bias"] = hf_q35[
        "model.layers.0.linear_attn.dt_bias"
    ]
    ref_q35["layers.0.linear_attn.A_log"] = hf_q35[
        "model.layers.0.linear_attn.A_log"
    ]
    ref_q35["layers.0.linear_attn.norm.w"] = hf_q35[
        "model.layers.0.linear_attn.norm.weight"
    ]

    hf_q35["model.layers.1.self_attn.q_proj.weight"] = rand(24, 12)
    hf_q35["model.layers.1.self_attn.k_proj.weight"] = rand(4, 12)
    hf_q35["model.layers.1.self_attn.v_proj.weight"] = rand(4, 12)
    hf_q35["model.layers.1.self_attn.o_proj.weight"] = rand(12, 12)
    hf_q35["model.layers.1.self_attn.q_norm.weight"] = rand(4)
    hf_q35["model.layers.1.self_attn.k_norm.weight"] = rand(4)
    ref_q35["layers.1.attn.q_proj.w"] = hf_q35[
        "model.layers.1.self_attn.q_proj.weight"
    ].T.reshape(12, 3, 8)
    ref_q35["layers.1.attn.k_proj.w"] = hf_q35[
        "model.layers.1.self_attn.k_proj.weight"
    ].T.reshape(12, 1, 4)
    ref_q35["layers.1.attn.v_proj.w"] = hf_q35[
        "model.layers.1.self_attn.v_proj.weight"
    ].T.reshape(12, 1, 4)
    ref_q35["layers.1.attn.o_proj.w"] = hf_q35[
        "model.layers.1.self_attn.o_proj.weight"
    ].T.reshape(3, 4, 12)
    ref_q35["layers.1.attn.q_norm.w"] = hf_q35[
        "model.layers.1.self_attn.q_norm.weight"
    ]
    ref_q35["layers.1.attn.k_norm.w"] = hf_q35[
        "model.layers.1.self_attn.k_norm.weight"
    ]
    got_q35 = qwen3p5_params.from_hf_impl(
        qwen3p5_params.build_from_hf_manifest(
            cfg_q35, qwen3p5_params.HfVariant.CAUSAL_LM
        ),
        hf_q35,
    )
    self.assertEqual(set(got_q35.keys()), set(ref_q35.keys()))
    for k, v in ref_q35.items():
      np.testing.assert_array_equal(got_q35[k], v, err_msg=f"qwen3p5:{k}")
    rt_q35 = qwen3p5_params.to_hf_impl(
        qwen3p5_params.build_to_hf_manifest(
            cfg_q35, qwen3p5_params.HfVariant.CAUSAL_LM
        ),
        got_q35,
    )
    for k, v in hf_q35.items():
      np.testing.assert_array_equal(rt_q35[k], v, err_msg=f"qwen3p5_rt:{k}")

    # 5 & 6. Gemma (MHA & GQA) and Gemma3 (GQA + SigLIP vision).
    for num_kv in (1, 3):
      cfg_g2 = gemma_model.ModelConfig(
          num_layers=1,
          num_embed=19,
          embed_dim=12,
          hidden_dim=17,
          num_heads=3,
          head_dim=4,
          num_kv_heads=num_kv,
          final_logit_softcap=None,
          attention_types=(gemma_model.AttentionType.GLOBAL,),
          use_post_attn_norm=True,
          use_post_ffw_norm=True,
      )
      hf_g = {
          "model.embed_tokens.weight": rand(19, 12),
          "model.norm.weight": rand(12),
          "model.layers.0.input_layernorm.weight": rand(12),
          "model.layers.0.post_attention_layernorm.weight": rand(12),
          "model.layers.0.pre_feedforward_layernorm.weight": rand(12),
          "model.layers.0.post_feedforward_layernorm.weight": rand(12),
          "model.layers.0.self_attn.q_proj.weight": rand(12, 12),
          "model.layers.0.self_attn.k_proj.weight": rand(num_kv * 4, 12),
          "model.layers.0.self_attn.v_proj.weight": rand(num_kv * 4, 12),
          "model.layers.0.self_attn.o_proj.weight": rand(12, 12),
          "model.layers.0.mlp.gate_proj.weight": rand(17, 12),
          "model.layers.0.mlp.up_proj.weight": rand(17, 12),
          "model.layers.0.mlp.down_proj.weight": rand(12, 17),
      }
      # Exact HEAD legacy math: .T.reshape(D, N_or_K, H).transpose(1, 0, 2).
      q_ndh = jnp.transpose(
          hf_g["model.layers.0.self_attn.q_proj.weight"].T.reshape(12, 3, 4),
          (1, 0, 2),
      )
      k_kdh = jnp.transpose(
          hf_g["model.layers.0.self_attn.k_proj.weight"].T.reshape(
              12, num_kv, 4
          ),
          (1, 0, 2),
      )
      v_kdh = jnp.transpose(
          hf_g["model.layers.0.self_attn.v_proj.weight"].T.reshape(
              12, num_kv, 4
          ),
          (1, 0, 2),
      )
      ref_g2 = {
          "embedder.input_embedding": hf_g["model.embed_tokens.weight"],
          "final_norm.scale": hf_g["model.norm.weight"],
          "layers.0.pre_attention_norm.scale": hf_g[
              "model.layers.0.input_layernorm.weight"
          ],
          "layers.0.post_attn_norm.scale": hf_g[
              "model.layers.0.post_attention_layernorm.weight"
          ],
          "layers.0.pre_ffw_norm.scale": hf_g[
              "model.layers.0.pre_feedforward_layernorm.weight"
          ],
          "layers.0.post_ffw_norm.scale": hf_g[
              "model.layers.0.post_feedforward_layernorm.weight"
          ],
          "layers.0.attn.attn_vec_einsum.w": hf_g[
              "model.layers.0.self_attn.o_proj.weight"
          ].T.reshape(3, 4, 12),
          "layers.0.mlp.gate_proj.kernel": hf_g[
              "model.layers.0.mlp.gate_proj.weight"
          ].T,
          "layers.0.mlp.up_proj.kernel": hf_g[
              "model.layers.0.mlp.up_proj.weight"
          ].T,
          "layers.0.mlp.down_proj.kernel": hf_g[
              "model.layers.0.mlp.down_proj.weight"
          ].T,
      }
      if num_kv == 3:
        ref_g2["layers.0.attn.qkv_einsum.w"] = jnp.stack(
            [q_ndh, k_kdh, v_kdh], axis=0
        )
      else:
        ref_g2["layers.0.attn.q_einsum.w"] = q_ndh
        ref_g2["layers.0.attn.kv_einsum.w"] = jnp.stack([k_kdh, v_kdh], axis=0)
      got_g2 = gemma_params_safetensors.from_hf_impl(
          gemma_params_safetensors.build_from_hf_manifest(cfg_g2), hf_g
      )
      self.assertEqual(set(got_g2.keys()), set(ref_g2.keys()))
      for k, v in ref_g2.items():
        np.testing.assert_array_equal(
            got_g2[k], v, err_msg=f"gemma(kv={num_kv}):{k}"
        )
      rt_g2 = gemma_params_safetensors.to_hf_impl(
          gemma_params_safetensors.build_to_hf_manifest(cfg_g2), got_g2
      )
      for k, v in hf_g.items():
        np.testing.assert_array_equal(
            rt_g2[k], v, err_msg=f"gemma_rt(kv={num_kv}):{k}"
        )

      siglip = None
      if num_kv == 1:
        siglip = gemma3_vision.SigLIPConfig(
            num_mm_tokens_per_image_prepool=4,
            num_mm_tokens_per_image=4,
            image_height=8,
            image_width=8,
            image_channels=3,
            patch_size=(4, 4),
            width=6,
            depth=1,
            mlp_dim=10,
            num_heads=2,
        )
      cfg_g3 = gemma3_model.ModelConfig(
          num_layers=1,
          num_embed=19,
          embed_dim=12,
          hidden_dim=17,
          num_heads=3,
          head_dim=4,
          num_kv_heads=num_kv,
          sliding_window_size=16,
          vision_config=siglip,
      )
      hf_g3 = {
          **hf_g,
          "model.layers.0.self_attn.q_norm.weight": rand(4),
          "model.layers.0.self_attn.k_norm.weight": rand(4),
      }
      ref_g3 = {
          k.replace("post_attn_norm", "post_attention_norm"): v
          for k, v in ref_g2.items()
      }
      ref_g3["layers.0.attn._query_norm.scale"] = hf_g3[
          "model.layers.0.self_attn.q_norm.weight"
      ]
      ref_g3["layers.0.attn._key_norm.scale"] = hf_g3[
          "model.layers.0.self_attn.k_norm.weight"
      ]
      g3_variant = gemma3_params_safetensors.HfVariant.CAUSAL_LM
      if siglip is not None:
        # Published multimodal Gemma3 repos keep the text decoder under
        # `language_model.model.` next to the top-level towers.
        g3_variant = (
            gemma3_params_safetensors.HfVariant.CONDITIONAL_GENERATION_LEGACY
        )
        hf_g3 = {f"language_model.{k}": v for k, v in hf_g3.items()}
        vp = "vision_tower.vision_model"
        tb = "vision_encoder.siglip_encoder.transformer.blocks.0"
        hf_g3.update({
            f"{vp}.embeddings.patch_embedding.weight": rand(6, 3, 4, 4),
            f"{vp}.embeddings.patch_embedding.bias": rand(6),
            f"{vp}.embeddings.position_embedding.weight": rand(4, 6),
            f"{vp}.encoder.layers.0.self_attn.q_proj.weight": rand(6, 6),
            f"{vp}.encoder.layers.0.self_attn.k_proj.weight": rand(6, 6),
            f"{vp}.encoder.layers.0.self_attn.v_proj.weight": rand(6, 6),
            f"{vp}.encoder.layers.0.self_attn.out_proj.weight": rand(6, 6),
            f"{vp}.encoder.layers.0.self_attn.q_proj.bias": rand(6),
            f"{vp}.encoder.layers.0.self_attn.k_proj.bias": rand(6),
            f"{vp}.encoder.layers.0.self_attn.v_proj.bias": rand(6),
            f"{vp}.encoder.layers.0.self_attn.out_proj.bias": rand(6),
            f"{vp}.encoder.layers.0.layer_norm1.weight": rand(6),
            f"{vp}.encoder.layers.0.layer_norm1.bias": rand(6),
            f"{vp}.encoder.layers.0.layer_norm2.weight": rand(6),
            f"{vp}.encoder.layers.0.layer_norm2.bias": rand(6),
            f"{vp}.encoder.layers.0.mlp.fc1.weight": rand(10, 6),
            f"{vp}.encoder.layers.0.mlp.fc2.weight": rand(6, 10),
            f"{vp}.encoder.layers.0.mlp.fc1.bias": rand(10),
            f"{vp}.encoder.layers.0.mlp.fc2.bias": rand(6),
            f"{vp}.post_layernorm.weight": rand(6),
            f"{vp}.post_layernorm.bias": rand(6),
            "multi_modal_projector.mm_input_projection_weight": rand(6, 12),
            "multi_modal_projector.mm_soft_emb_norm.weight": rand(6),
        })
        ref_g3.update({
            "vision_encoder.siglip_encoder.embedding.kernel": jnp.transpose(
                hf_g3[f"{vp}.embeddings.patch_embedding.weight"], (2, 3, 1, 0)
            ),
            "vision_encoder.siglip_encoder.embedding.bias": hf_g3[
                f"{vp}.embeddings.patch_embedding.bias"
            ],
            "vision_encoder.siglip_encoder.pos_embedding": hf_g3[
                f"{vp}.embeddings.position_embedding.weight"
            ].reshape(1, 4, 6),
            f"{tb}.attn.query_proj.kernel": hf_g3[
                f"{vp}.encoder.layers.0.self_attn.q_proj.weight"
            ].T,
            f"{tb}.attn.key_proj.kernel": hf_g3[
                f"{vp}.encoder.layers.0.self_attn.k_proj.weight"
            ].T,
            f"{tb}.attn.value_proj.kernel": hf_g3[
                f"{vp}.encoder.layers.0.self_attn.v_proj.weight"
            ].T,
            f"{tb}.attn.out_proj.kernel": hf_g3[
                f"{vp}.encoder.layers.0.self_attn.out_proj.weight"
            ].T,
            f"{tb}.attn.query_proj.bias": hf_g3[
                f"{vp}.encoder.layers.0.self_attn.q_proj.bias"
            ],
            f"{tb}.attn.key_proj.bias": hf_g3[
                f"{vp}.encoder.layers.0.self_attn.k_proj.bias"
            ],
            f"{tb}.attn.value_proj.bias": hf_g3[
                f"{vp}.encoder.layers.0.self_attn.v_proj.bias"
            ],
            f"{tb}.attn.out_proj.bias": hf_g3[
                f"{vp}.encoder.layers.0.self_attn.out_proj.bias"
            ],
            f"{tb}.ln1.scale": hf_g3[
                f"{vp}.encoder.layers.0.layer_norm1.weight"
            ],
            f"{tb}.ln1.bias": hf_g3[f"{vp}.encoder.layers.0.layer_norm1.bias"],
            f"{tb}.ln2.scale": hf_g3[
                f"{vp}.encoder.layers.0.layer_norm2.weight"
            ],
            f"{tb}.ln2.bias": hf_g3[f"{vp}.encoder.layers.0.layer_norm2.bias"],
            f"{tb}.mlp.fc1.kernel": hf_g3[
                f"{vp}.encoder.layers.0.mlp.fc1.weight"
            ].T,
            f"{tb}.mlp.fc2.kernel": hf_g3[
                f"{vp}.encoder.layers.0.mlp.fc2.weight"
            ].T,
            f"{tb}.mlp.fc1.bias": hf_g3[f"{vp}.encoder.layers.0.mlp.fc1.bias"],
            f"{tb}.mlp.fc2.bias": hf_g3[f"{vp}.encoder.layers.0.mlp.fc2.bias"],
            "vision_encoder.siglip_encoder.transformer.encoder_norm.scale": (
                hf_g3[f"{vp}.post_layernorm.weight"]
            ),
            "vision_encoder.siglip_encoder.transformer.encoder_norm.bias": (
                hf_g3[f"{vp}.post_layernorm.bias"]
            ),
            "embedder.mm_input_projection.w": hf_g3[
                "multi_modal_projector.mm_input_projection_weight"
            ],
            "embedder.mm_soft_embedding_norm.scale": hf_g3[
                "multi_modal_projector.mm_soft_emb_norm.weight"
            ],
        })
      got_g3 = gemma3_params_safetensors.from_hf_impl(
          gemma3_params_safetensors.build_from_hf_manifest(
              cfg_g3, g3_variant
          ),
          hf_g3,
      )
      self.assertEqual(set(got_g3.keys()), set(ref_g3.keys()))
      for k, v in ref_g3.items():
        np.testing.assert_array_equal(
            got_g3[k], v, err_msg=f"gemma3(kv={num_kv}):{k}"
        )
      rt_g3 = gemma3_params_safetensors.to_hf_impl(
          gemma3_params_safetensors.build_to_hf_manifest(cfg_g3, g3_variant),
          got_g3,
      )
      for k, v in hf_g3.items():
        np.testing.assert_array_equal(
            rt_g3[k], v, err_msg=f"gemma3_rt(kv={num_kv}):{k}"
        )

  def test_gcs_single_download_for_gemma_vocab_peek_and_qwen3p5_prefixes(self):
    """Verifies `gs://` URIs resolve once across vocab peek & declared HfVariant."""
    calls: list[str] = []
    orig_loader = safetensors_loader.load_file_from_gcs
    self.addCleanup(
        setattr, safetensors_loader, "load_file_from_gcs", orig_loader
    )

    g_cfg = gemma_model.ModelConfig(
        num_layers=1,
        num_embed=11,
        embed_dim=8,
        hidden_dim=16,
        num_heads=2,
        head_dim=4,
        num_kv_heads=1,
        final_logit_softcap=None,
        attention_types=(gemma_model.AttentionType.GLOBAL,),
        use_post_attn_norm=True,
        use_post_ffw_norm=True,
    )
    actual_cfg = dataclasses.replace(g_cfg, num_embed=17)
    g_model = gemma_model.Gemma(actual_cfg, rngs=nnx.Rngs(0))
    g_hf = gemma_params_safetensors.export_to_hf_state_dict(g_model, actual_cfg)
    g_dir = self.create_tempdir().full_path
    safe_np.save_file(
        {k: np.array(v, order="C", copy=True) for k, v in g_hf.items()},
        os.path.join(g_dir, "model.safetensors"),
    )
    safetensors_loader.load_file_from_gcs = (
        lambda uri: calls.append(uri) or g_dir
    )
    g_loaded = gemma_params_safetensors.create_model_from_safe_tensors(
        "gs://bucket/gemma2", g_cfg, dtype=jnp.float32
    )
    self.assertEqual(calls, ["gs://bucket/gemma2"])
    self.assertEqual(g_loaded.embedder.input_embedding.value.shape, (17, 8))
    empty_g_dir = self.create_tempdir().full_path
    with self.assertRaises(FileNotFoundError):
      gemma_params_safetensors.create_model_from_safe_tensors(
          empty_g_dir, g_cfg, dtype=jnp.float32
      )

    calls.clear()
    q_cfg = qwen3p5_model.ModelConfig(
        num_layers=2,
        vocab_size=19,
        embed_dim=12,
        hidden_dim=17,
        num_heads=3,
        head_dim=4,
        num_kv_heads=1,
        rope_theta=10000,
        norm_eps=1e-6,
        use_tied_embedding=False,
        attn_output_gate=True,
        full_attention_interval=2,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=6,
        linear_value_head_dim=6,
        linear_conv_kernel_dim=3,
    )
    q_model = qwen3p5_model.Qwen3p5(q_cfg, rngs=nnx.Rngs(0))
    q_variants = qwen3p5_params.HfVariant
    q_hf = qwen3p5_params.export_to_hf_state_dict(
        q_model, q_cfg, hf_variant=q_variants.CONDITIONAL_GENERATION
    )
    self.assertIn("lm_head.weight", q_hf)
    self.assertIn("model.language_model.norm.weight", q_hf)
    q_dir = self.create_tempdir().full_path
    safe_np.save_file(
        {k: np.array(v, order="C", copy=True) for k, v in q_hf.items()},
        os.path.join(q_dir, "model.safetensors"),
    )
    safetensors_loader.load_file_from_gcs = (
        lambda uri: calls.append(uri) or q_dir
    )
    q_loaded_auto = qwen3p5_params.create_model_from_safe_tensors(
        "gs://bucket/qwen3p5",
        q_cfg,
        dtype=jnp.float32,
    )
    self.assertEqual(calls, ["gs://bucket/qwen3p5"])
    self.assertEqual(
        q_loaded_auto._hf_variant, q_variants.CONDITIONAL_GENERATION
    )
    np.testing.assert_array_equal(
        q_loaded_auto.lm_head.w.value, q_model.lm_head.w.value
    )
    calls.clear()
    q_loaded = qwen3p5_params.create_model_from_safe_tensors(
        "gs://bucket/qwen3p5",
        q_cfg,
        dtype=jnp.float32,
        hf_variant=q_variants.CONDITIONAL_GENERATION,
    )
    self.assertEqual(calls, ["gs://bucket/qwen3p5"])
    np.testing.assert_array_equal(
        q_loaded.lm_head.w.value, q_model.lm_head.w.value
    )

    # The same weights in the `Qwen3_5ForCausalLM` layout load only when that
    # variant is declared; the wrong declaration fails and names itself.
    q_causal_hf = qwen3p5_params.export_to_hf_state_dict(
        q_model, q_cfg, hf_variant=q_variants.CAUSAL_LM
    )
    self.assertIn("model.norm.weight", q_causal_hf)
    self.assertIn("lm_head.weight", q_causal_hf)
    q_causal_dir = self.create_tempdir().full_path
    safe_np.save_file(
        {k: np.array(v, order="C", copy=True) for k, v in q_causal_hf.items()},
        os.path.join(q_causal_dir, "model.safetensors"),
    )
    q_causal_loaded = qwen3p5_params.create_model_from_safe_tensors(
        q_causal_dir, q_cfg, dtype=jnp.float32, hf_variant=q_variants.CAUSAL_LM
    )
    np.testing.assert_array_equal(
        q_causal_loaded.lm_head.w.value, q_model.lm_head.w.value
    )
    with self.assertRaisesRegex(
        ValueError,
        "declared HF layout 'qwen3p5/conditional_generation'; missing",
    ):
      qwen3p5_params.create_model_from_safe_tensors(
          q_causal_dir,
          q_cfg,
          dtype=jnp.float32,
          hf_variant=q_variants.CONDITIONAL_GENERATION,
      )
    q_model.layers[1].attn.q_proj.lora_a = nnx.LoRAParam(
        jnp.ones((12, 2), dtype=jnp.float32)
    )
    q_model.layers[1].attn.q_proj.lora_b = nnx.LoRAParam(
        jnp.ones((2, 3, 8), dtype=jnp.float32)
    )
    with self.assertRaisesRegex(ValueError, "unmerged LoRAParam"):
      safetensors_loader.flatten_model_state(q_model)
    q_merged_dir = self.create_tempdir().full_path
    qwen3p5_params.save_lora_merged_model_as_safetensors(
        local_model_path=q_causal_dir,
        output_dir=q_merged_dir,
        lora_model=q_model,
        rank=2,
        alpha=4.0,
        hf_variant=q_variants.CAUSAL_LM,
    )
    self.assertEqual(
        set(safetensors_loader.read_safetensors_headers(q_merged_dir)),
        set(q_causal_hf),
    )
    q_tied_cfg = dataclasses.replace(q_cfg, use_tied_embedding=True)
    for q_variant in q_variants:
      self.assertEqual(
          qwen3p5_params.build_from_hf_manifest(q_tied_cfg, q_variant)[
              "dropped_keys"
          ],
          ["lm_head.weight"],
      )

  def test_resolve_hf_variant_from_keys_and_read_checkpoint_keys(self):
    class _Layout(enum.Enum):
      CAUSAL = "causal"
      MULTIMODAL = "multimodal"

    affixes = {
        _Layout.CAUSAL: {"lm": "model."},
        _Layout.MULTIMODAL: {"lm": "model.language_model."},
    }
    self.assertEqual(
        hf_mapping.resolve_hf_variant_from_keys(
            ["model.embed_tokens.weight", "lm_head.weight"], affixes
        ),
        _Layout.CAUSAL,
    )
    self.assertEqual(
        hf_mapping.resolve_hf_variant_from_keys(
            ["model.language_model.embed_tokens.weight"], affixes
        ),
        _Layout.MULTIMODAL,
    )
    self.assertEqual(
        hf_mapping.resolve_hf_variant_from_keys(
            ["model.norm.weight"], affixes, anchor_suffix="norm.weight"
        ),
        _Layout.CAUSAL,
    )
    with self.assertRaisesRegex(ValueError, "matched 0"):
      hf_mapping.resolve_hf_variant_from_keys(["unrelated.weight"], affixes)
    with self.assertRaisesRegex(ValueError, "matched 2"):
      hf_mapping.resolve_hf_variant_from_keys(
          [
              "model.embed_tokens.weight",
              "model.language_model.embed_tokens.weight",
          ],
          affixes,
      )
    with self.assertRaisesRegex(ValueError, "Duplicate HF variant anchor key"):
      hf_mapping.resolve_hf_variant_from_keys(
          ["model.embed_tokens.weight"],
          {
              _Layout.CAUSAL: {"lm": "model."},
              _Layout.MULTIMODAL: {"lm": "model."},
          },
      )

    idx_dir = self.create_tempdir().full_path
    with open(
        os.path.join(idx_dir, "model.safetensors.index.json"), "w"
    ) as f:
      json.dump(
          {"weight_map": {"a.weight": "s0.safetensors", "b.weight": "s1"}}, f
      )
    self.assertEqual(
        safetensors_loader.read_checkpoint_keys(idx_dir),
        frozenset({"a.weight", "b.weight"}),
    )

    hdr_dir = self.create_tempdir().full_path
    safe_np.save_file(
        {"c.weight": np.zeros((2, 2), dtype=np.float32)},
        os.path.join(hdr_dir, "model.safetensors"),
    )
    self.assertEqual(
        safetensors_loader.read_checkpoint_keys(hdr_dir),
        frozenset({"c.weight"}),
    )

  def test_hf_variant_memory_override_validation_and_unstack_lora(self):
    g3_v = gemma3_params_safetensors.HfVariant
    q_v = qwen3p5_params.HfVariant
    self.assertIs(g3_v, gemma3_model.HfVariant)
    self.assertIs(q_v, qwen3p5_model.HfVariant)
    self.assertNotIn(
        "hf_variant",
        [f.name for f in dataclasses.fields(gemma3_model.ModelConfig)],
    )
    self.assertNotIn(
        "hf_variant",
        [f.name for f in dataclasses.fields(qwen3p5_model.ModelConfig)],
    )

    for bad_variant in (None, cast(Any, q_v.CAUSAL_LM)):
      with self.assertRaisesRegex(ValueError, "Pass `hf_variant`"):
        gemma3_params_safetensors.build_from_hf_manifest(
            gemma3_model.ModelConfig(
                num_layers=1,
                num_embed=32,
                embed_dim=16,
                hidden_dim=32,
                num_heads=2,
                head_dim=8,
                num_kv_heads=1,
                sliding_window_size=16,
            ),
            bad_variant,
        )
    for bad_q_variant in (None, cast(Any, g3_v.CAUSAL_LM)):
      with self.assertRaisesRegex(ValueError, "Pass `hf_variant`"):
        qwen3p5_params.build_from_hf_manifest(
            qwen3p5_model.ModelConfig(
                num_layers=2,
                vocab_size=19,
                embed_dim=12,
                hidden_dim=17,
                num_heads=3,
                head_dim=4,
                num_kv_heads=1,
                rope_theta=10000,
                norm_eps=1e-6,
                full_attention_interval=2,
            ),
            bad_q_variant,
        )

    g3_cfg = gemma3_model.ModelConfig(
        num_layers=1,
        num_embed=32,
        embed_dim=16,
        hidden_dim=32,
        num_heads=2,
        head_dim=8,
        num_kv_heads=1,
        sliding_window_size=16,
        param_dtype=jnp.float32,
    )
    g3_to_hf = gemma3_params_safetensors.build_to_hf_manifest(
        g3_cfg, g3_v.CAUSAL_LM
    )
    lora_map, lora_tr = hf_mapping.lora_mapping_from_manifest(g3_to_hf)
    self.assertEqual(
        lora_map["layers.0.attn.kv_einsum.0"],
        "model.layers.0.self_attn.k_proj.weight",
    )
    self.assertEqual(
        lora_map["layers.0.attn.kv_einsum.1"],
        "model.layers.0.self_attn.v_proj.weight",
    )
    self.assertEqual(lora_tr["model.layers.0.self_attn.k_proj.weight"], (1, 0))
    split_only_map, _ = hf_mapping.lora_mapping_from_manifest({
        "schema_version": hf_mapping.SCHEMA_VERSION,
        "rules": [{
            "source_keys": ["layers.0.fused.w"],
            "target_keys": ["a.weight", "b.weight"],
            "ops": [{"split": {"axis": 0, "sizes": [4, 4]}}],
        }],
    })
    self.assertEmpty(split_only_map)

    g3_model = gemma3_model.Gemma3(g3_cfg, rngs=nnx.Rngs(params=0))
    self.assertIsNone(g3_model._hf_variant)
    with self.assertRaisesRegex(ValueError, "Pass `hf_variant`"):
      gemma3_params_safetensors.export_to_hf_state_dict(g3_model, g3_cfg)
    g3_hf = gemma3_params_safetensors.export_to_hf_state_dict(
        g3_model, g3_cfg, hf_variant=g3_v.CAUSAL_LM
    )
    self.assertIn("model.embed_tokens.weight", g3_hf)
    g3_dir = self.create_tempdir().full_path
    safe_np.save_file(
        {k: np.ascontiguousarray(np.asarray(v)) for k, v in g3_hf.items()},
        os.path.join(g3_dir, "model.safetensors"),
    )
    g3_loaded = gemma3_params_safetensors.create_model_from_safe_tensors(
        g3_dir, g3_cfg, dtype=jnp.float32
    )
    self.assertEqual(g3_loaded._hf_variant, g3_v.CAUSAL_LM)
    np.testing.assert_array_equal(
        g3_loaded.embedder.input_embedding.value,
        g3_model.embedder.input_embedding.value,
    )
    g3_reexport = gemma3_params_safetensors.export_to_hf_state_dict(
        g3_loaded, g3_cfg
    )
    self.assertIn("model.embed_tokens.weight", g3_reexport)
    g3_legacy_hf = gemma3_params_safetensors.export_to_hf_state_dict(
        g3_loaded, g3_cfg, hf_variant=g3_v.CONDITIONAL_GENERATION_LEGACY
    )
    self.assertIn("language_model.model.embed_tokens.weight", g3_legacy_hf)

    q_cfg = qwen3p5_model.ModelConfig(
        num_layers=2,
        vocab_size=19,
        embed_dim=12,
        hidden_dim=17,
        num_heads=3,
        head_dim=4,
        num_kv_heads=1,
        rope_theta=10000,
        norm_eps=1e-6,
        use_tied_embedding=False,
        attn_output_gate=True,
        full_attention_interval=2,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=6,
        linear_value_head_dim=6,
        linear_conv_kernel_dim=3,
    )
    q_model = qwen3p5_model.Qwen3p5(q_cfg, rngs=nnx.Rngs(0))
    self.assertIsNone(q_model._hf_variant)
    with self.assertRaisesRegex(ValueError, "Pass `hf_variant`"):
      qwen3p5_params.export_to_hf_state_dict(q_model, q_cfg)
    q_hf = qwen3p5_params.export_to_hf_state_dict(
        q_model, q_cfg, hf_variant=q_v.CONDITIONAL_GENERATION
    )
    self.assertIn("model.language_model.embed_tokens.weight", q_hf)
    q_dir = self.create_tempdir().full_path
    safe_np.save_file(
        {k: np.array(v, order="C", copy=True) for k, v in q_hf.items()},
        os.path.join(q_dir, "model.safetensors"),
    )
    q_loaded = qwen3p5_params.create_model_from_safe_tensors(
        q_dir, q_cfg, dtype=jnp.float32
    )
    self.assertEqual(q_loaded._hf_variant, q_v.CONDITIONAL_GENERATION)
    np.testing.assert_array_equal(
        q_loaded.lm_head.w.value, q_model.lm_head.w.value
    )
    q_reexport = qwen3p5_params.export_to_hf_state_dict(q_loaded, q_cfg)
    self.assertIn("model.language_model.embed_tokens.weight", q_reexport)
    q_causal_override = qwen3p5_params.export_to_hf_state_dict(
        q_loaded, q_cfg, hf_variant=q_v.CAUSAL_LM
    )
    self.assertIn("model.embed_tokens.weight", q_causal_override)

    q_model.layers[1].attn.q_proj.lora_a = nnx.LoRAParam(
        jnp.ones((12, 2), dtype=jnp.float32)
    )
    q_model.layers[1].attn.q_proj.lora_b = nnx.LoRAParam(
        jnp.ones((2, 3, 8), dtype=jnp.float32)
    )
    with self.assertRaisesRegex(ValueError, "Pass `hf_variant`"):
      qwen3p5_params.save_lora_merged_model_as_safetensors(
          local_model_path=q_dir,
          output_dir=self.create_tempdir().full_path,
          lora_model=q_model,
          rank=2,
          alpha=4.0,
      )

    q_loaded.layers[1].attn.q_proj.lora_a = nnx.LoRAParam(
        jnp.ones((12, 2), dtype=jnp.float32)
    )
    q_loaded.layers[1].attn.q_proj.lora_b = nnx.LoRAParam(
        jnp.ones((2, 3, 8), dtype=jnp.float32)
    )
    q_lora_dir = self.create_tempdir().full_path
    with (
        mock.patch.object(
            safetensors_loader,
            "read_checkpoint_keys",
            side_effect=AssertionError("must not read checkpoint keys"),
        ),
        mock.patch.object(
            safetensors_loader,
            "read_safetensors_headers",
            side_effect=AssertionError("must not read safetensors headers"),
        ),
    ):
      qwen3p5_params.save_lora_merged_model_as_safetensors(
          local_model_path=q_dir,
          output_dir=q_lora_dir,
          lora_model=q_loaded,
          rank=2,
          alpha=4.0,
      )
    self.assertEqual(
        set(safetensors_loader.read_safetensors_headers(q_lora_dir)),
        set(q_hf),
    )
    q_causal_dir = self.create_tempdir().full_path
    safe_np.save_file(
        {
            k: np.array(v, order="C", copy=True)
            for k, v in q_causal_override.items()
        },
        os.path.join(q_causal_dir, "model.safetensors"),
    )
    q_lora_override_dir = self.create_tempdir().full_path
    qwen3p5_params.save_lora_merged_model_as_safetensors(
        local_model_path=q_causal_dir,
        output_dir=q_lora_override_dir,
        lora_model=q_loaded,
        rank=2,
        alpha=4.0,
        hf_variant=q_v.CAUSAL_LM,
    )
    self.assertEqual(
        set(safetensors_loader.read_safetensors_headers(q_lora_override_dir)),
        set(q_causal_override),
    )


  def test_apply_host_manifest_strict_and_unconsumed_rules(self):
    manifest = {
        "rules": [
            {"source_keys": ["a"], "target_keys": ["hf.a"], "ops": []},
            {"source_keys": ["b"], "target_keys": ["hf.b"], "ops": []},
        ],
        "tied_keys": [],
        "dropped_keys": [],
    }
    tensors = {"a": np.ones((2,), np.float32)}
    with self.assertRaisesRegex(ValueError, r"missing source keys: \['b'\]"):
      hf_mapping.apply_host_manifest(manifest, tensors, drop_tied_targets=True)
    out = hf_mapping.apply_host_manifest(
        manifest, tensors, drop_tied_targets=True, strict=False
    )
    self.assertEqual(list(out), ["hf.a"])
    np.testing.assert_array_equal(out["hf.a"], tensors["a"])

    concat_rules = [{
        "source_keys": ["q", "k"],
        "target_keys": ["qk"],
        "ops": [{"concat": {"axis": 0, "sizes": [1, 1]}}],
    }]
    with self.assertRaisesRegex(ValueError, r"unconsumed source keys: \['q'\]"):
      hf_mapping.apply_host_manifest(
          {"rules": concat_rules, "tied_keys": [], "dropped_keys": []},
          {"q": np.ones(1, np.float32)},
          drop_tied_targets=True,
          strict=False,
      )

  def test_apply_host_manifest_drops_tied_targets(self):
    manifest = hf_mapping.parse_manifest({
        "rules": [
            {"source_keys": ["embed"], "target_keys": ["hf.embed"], "ops": []}
        ],
        "tied_keys": [["hf.embed", "hf.lm_head"]],
    })
    tensors = {"embed": np.arange(3, dtype=np.float32)}
    out = hf_mapping.apply_host_manifest(
        manifest, tensors, drop_tied_targets=True
    )
    self.assertEqual(list(out), ["hf.embed"])
    np.testing.assert_array_equal(out["hf.embed"], tensors["embed"])


if __name__ == "__main__":
  absltest.main()
