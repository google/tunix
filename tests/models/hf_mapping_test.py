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

"""JAX execution and checkpoint round-trip tests for `hf_mapping_jax`."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from tunix.models import hf_mapping
from tunix.models import hf_mapping_jax


class HfMappingJaxTest(absltest.TestCase):
  """Tests all 12 primitive ops, sharding, and fused JIT execution."""

  def test_all_12_ops_forward_and_inverse_numeric_round_trip(self):
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
        "model_type": "numeric_12_ops",
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
                "ops": [{"stack": {"axis": 0}}],
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
    target = hf_mapping_jax.from_hf(manifest_json, src, mesh=mesh)
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
    recovered = hf_mapping_jax.to_hf(inv_json, target)
    self.assertEqual(set(recovered.keys()), expected_hf_keys)
    for k in expected_hf_keys:
      np.testing.assert_allclose(recovered[k], src[k])

    ckpt_loaded = hf_mapping_jax.from_hf_impl(manifest, src)
    ckpt_saved = hf_mapping_jax.to_hf_impl(inv_manifest, ckpt_loaded)
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
                "sharding": ["tp", None, None],
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
        lambda w: hf_mapping_jax.from_hf_impl(
            sampler_dst_spec,
            hf_mapping_jax.to_hf_impl(trainer_src_spec, w),
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

    hf_via_boundary = hf_mapping_jax.to_hf(
        trainer_src_spec.to_json(), src_trainer
    )
    self.assertNotIn("trainer.kv_alias", hf_via_boundary)
    np.testing.assert_allclose(hf_via_boundary["hf.k"], k_base)
    np.testing.assert_allclose(hf_via_boundary["hf.v"], v_base)
    sampler_via_boundary = hf_mapping_jax.from_hf(
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
    cat_out = hf_mapping_jax.apply_primitive_op(
        [t23, t33, t23, t33], "concat", {"axis": 0, "sizes": [2, 3]}
    )
    self.assertLen(cat_out, 2)
    self.assertEqual(cat_out[0].shape, (5, 3))
    split_out = hf_mapping_jax.apply_primitive_op(
        [jnp.arange(6, dtype=jnp.float32)],
        "split",
        {"axis": 0, "sizes": [1, 2, 3]},
    )
    self.assertEqual([t.shape for t in split_out], [(1,), (2,), (3,)])

    for op_name, tensors, kw in (
        ("scale", [], {"factor": 0.0}),
        ("reshape", [t23], {"src_shape": [3, 2], "dst_shape": [6]}),
        ("concat", [t23], {"axis": 0, "sizes": [2, 3]}),
        ("concat", [t23, t23], {"axis": 0, "sizes": [2, 3]}),
        ("split", [t23], {"axis": 0, "sizes": [1, 2]}),
        ("unpad", [t23], {"axis": 0, "sizes": [1, 1]}),
        ("unrepeat", [t33], {"axis": 0, "times": 2}),
    ):
      with self.assertRaises(ValueError):
        hf_mapping_jax.apply_primitive_op(tensors, op_name, kw)

    for bad_rule in (
        {"source_keys": [], "target_keys": ["a"], "ops": []},
        {"source_keys": ["k1"], "target_keys": ["a", "b"], "ops": []},
        {"source_keys": ["k1"], "target_keys": ["a"], "ops": [], "extra": 1},
    ):
      with self.assertRaises(ValueError):
        hf_mapping_jax.apply_rule(bad_rule, {"k1": t23})

    multi_rule_manifest = {
        "schema_version": hf_mapping.SCHEMA_VERSION,
        "rules": [
            {"source_keys": ["k1"], "target_keys": ["t1"], "ops": []},
            {"source_keys": ["k2"], "target_keys": ["t2"], "ops": []},
        ],
    }
    with self.assertRaises(KeyError):
      hf_mapping_jax.execute_manifest(multi_rule_manifest, {"k1": t23})


if __name__ == "__main__":
  absltest.main()
