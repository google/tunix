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

"""Unit tests for Qwen3 safetensors params loading and Subtree all-to-all export."""

import os
import tempfile
from unittest import mock

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tunix.models import safetensors_loader
from tunix.models.qwen3 import model as qwen3_model
from tunix.models.qwen3 import params as qwen3_params


class Qwen3ParamsExportTest(absltest.TestCase):

  def test_dense_and_moe_export_and_reload(self):
    dense_cfg = qwen3_model.ModelConfig(
        num_layers=1,
        vocab_size=64,
        embed_dim=32,
        hidden_dim=64,
        num_heads=2,
        head_dim=16,
        num_kv_heads=2,
        rope_theta=10000,
        norm_eps=1e-6,
        use_tied_embedding=False,
    )
    dense_model = qwen3_model.Qwen3(dense_cfg, rngs=nnx.Rngs(params=0))
    manifest_dense = qwen3_params.build_from_hf_manifest(dense_cfg)
    self.assertTrue(
        any(
            r['source_keys'] == ['lm_head.weight']
            for r in manifest_dense['rules']
        )
    )

    flat_state, specs, unstack = qwen3_params.build_qwen3_export_specs(
        dense_model, output_dtype=jnp.float32
    )
    self.assertEmpty(unstack)
    self.assertLen(specs, 14)
    spec = qwen3_params._make_tensor_spec(
        flat_state, 'final_norm.w', 'model.norm.weight'
    )
    self.assertEqual(spec.hf_key, 'model.norm.weight')
    self.assertEqual(spec.nbytes, 32 * 4)
    self.assertTrue(spec.is_replicated)

    with tempfile.TemporaryDirectory() as tmpdir:
      dense_dir = os.path.join(tmpdir, 'dense')
      metrics = qwen3_params.save_lora_merged_model_as_safetensors(
          None,
          dense_dir,
          dense_model,
          rank=1,
          alpha=1.0,
          num_ranks=2,
          output_dtype=jnp.float32,
          devices=jax.devices(),
      )
      self.assertLen(metrics.shard_paths, 2)
      reloaded_dense = qwen3_params.create_model_from_safe_tensors(
          dense_dir, dense_cfg, dtype=jnp.float32
      )
      np.testing.assert_allclose(
          reloaded_dense.lm_head.w.value,
          dense_model.lm_head.w.value,
          atol=1e-5,
      )

      moe_base = qwen3_model.ModelConfig.qwen3p5_35b_a3b()
      moe_cfg = qwen3_model.ModelConfig(
          num_layers=1,
          vocab_size=128,
          embed_dim=64,
          hidden_dim=32,
          num_heads=4,
          head_dim=16,
          num_kv_heads=2,
          rope_theta=moe_base.rope_theta,
          norm_eps=moe_base.norm_eps,
          num_experts=8,
          num_experts_per_tok=2,
          use_tied_embedding=False,
      )
      moe_model = qwen3_model.Qwen3(moe_cfg, rngs=nnx.Rngs(params=1))
      moe_dir = os.path.join(tmpdir, 'moe')
      moe_metrics = qwen3_params.export_qwen3_as_safetensors(
          moe_model,
          moe_dir,
          num_ranks=2,
          max_bucket_bytes=16384,
          output_dtype=jnp.float32,
          devices=jax.devices(),
          per_wave_shards=False,
      )
      self.assertLen(moe_metrics.shard_paths, 2)
      reloaded_moe = qwen3_params.create_model_from_safe_tensors(
          moe_dir, moe_cfg, dtype=jnp.float32
      )
      manifest = qwen3_params.build_from_hf_manifest(moe_cfg)
      self.assertTrue(
          any(
              r['target_keys'] == ['layers.0.mlp.gate_proj']
              for r in manifest['rules']
          )
      )
      hf_dict = qwen3_params.export_to_hf_state_dict(dense_model, dense_cfg)
      self.assertIn('model.layers.0.self_attn.q_proj.weight', hf_dict)
      tokens = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
      pos = jnp.array([[0, 1, 2, 3]], dtype=jnp.int32)
      mask = jnp.tril(jnp.ones((1, 4, 4), dtype=jnp.bool_))
      logits_orig, _ = moe_model(tokens, pos, None, mask)
      logits_reload, _ = reloaded_moe(tokens, pos, None, mask)
      np.testing.assert_allclose(logits_orig, logits_reload, atol=1e-5)
      moe_wave_dir = os.path.join(tmpdir, 'moe_wave')
      wave_metrics = qwen3_params.export_qwen3_as_safetensors(
          moe_model,
          moe_wave_dir,
          num_ranks=2,
          max_bucket_bytes=16384,
          output_dtype=jnp.float32,
          devices=jax.devices(),
          per_wave_shards=True,
      )
      self.assertGreater(len(wave_metrics.shard_paths), 2)
      self.assertTrue(
          os.path.exists(
              os.path.join(moe_wave_dir, 'model.safetensors.index.json')
          )
      )

  def test_export_manifest_validation_and_no_resolve_checkpoint_dir(self):
    dense_cfg = qwen3_model.ModelConfig(
        num_layers=1,
        vocab_size=64,
        embed_dim=32,
        hidden_dim=64,
        num_heads=2,
        head_dim=16,
        num_kv_heads=2,
        rope_theta=10000,
        norm_eps=1e-6,
        use_tied_embedding=False,
    )
    dense_model = qwen3_model.Qwen3(dense_cfg, rngs=nnx.Rngs(params=0))
    with tempfile.TemporaryDirectory() as tmpdir:
      src_dir = os.path.join(tmpdir, 'src')
      out_dir = os.path.join(tmpdir, 'out')
      os.makedirs(src_dir)
      with open(os.path.join(src_dir, 'config.json'), 'w') as f:
        f.write('{"model_type": "qwen3"}')
      with mock.patch.object(
          safetensors_loader,
          'resolve_checkpoint_dir',
          side_effect=AssertionError('must not download base checkpoint'),
      ):
        qwen3_params.export_qwen3_as_safetensors(
            dense_model,
            out_dir,
            local_model_path=src_dir,
            output_dtype=jnp.float32,
            devices=jax.devices(),
        )
      self.assertTrue(os.path.exists(os.path.join(out_dir, 'config.json')))


if __name__ == '__main__':
  absltest.main()
