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

"""Tests for Gemma 4 safetensors parameter loading helpers."""

import dataclasses
import itertools
import json
import os

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import safetensors.numpy as safe_np
from tunix.models import hf_mapping
from tunix.models import safetensors_loader
from tunix.models.gemma4 import audio as audio_lib
from tunix.models.gemma4 import model as model_lib
from tunix.models.gemma4 import params_safetensors
from tunix.models.gemma4 import vision as vision_lib

# Layout of the published Gemma 4 repos (`Gemma4ForConditionalGeneration`).
_CG = params_safetensors.HfVariant.CONDITIONAL_GENERATION
_CG_MODEL_TYPE = "gemma4/conditional_generation"


class ParamsSafetensorsTest(absltest.TestCase):

  def test_gemma4_12b_key_mapping_uses_local_and_global_shapes(self):
    config = model_lib.ModelConfig.gemma4_12b()
    config.num_layers = 6
    config.embed_dim = 16
    config.num_heads = 2
    config.head_dim = 4
    config.num_kv_heads = 2
    config.num_global_kv_heads = 1
    config.global_key_size = 8

    manifest = params_safetensors.build_from_hf_manifest(config, _CG)
    source_keys = hf_mapping.active_source_keys(manifest)
    self.assertIn(
        "model.language_model.layers.0.self_attn.k_proj.weight", source_keys
    )
    self.assertIn(
        "model.language_model.layers.5.self_attn.k_proj.weight", source_keys
    )

    step_exec = hf_mapping.make_streaming_executor(manifest)
    o0 = jax.random.normal(jax.random.key(1), (16, 8))
    o5 = jax.random.normal(jax.random.key(2), (16, 16))
    out = step_exec({
        "model.language_model.layers.0.self_attn.o_proj.weight": o0,
        "model.language_model.layers.5.self_attn.o_proj.weight": o5,
    })
    # DEPRECATED (testing-only): Legacy hand-written conversion reference math
    # in the streaming tests below is deprecated, retained strictly for
    # bit-by-bit parity testing during migration to `hf_mapping`, and will be
    # removed in a follow-up cleanup CL.
    np.testing.assert_array_equal(
        out["layers.0.attn.attn_vec_einsum.w"], o0.T.reshape(2, 4, 16)
    )
    np.testing.assert_array_equal(
        out["layers.5.attn.attn_vec_einsum.w"], o5.T.reshape(2, 8, 16)
    )

  def test_gemma4_12b_streaming_executor_handles_local_and_global_attention(
      self,
  ):
    config = model_lib.ModelConfig.gemma4_12b()
    config.num_layers = 6
    config.embed_dim = 16
    config.num_heads = 2
    config.head_dim = 4
    config.num_kv_heads = 2
    config.num_global_kv_heads = 1
    config.global_key_size = 8

    q0 = jax.random.normal(jax.random.key(3), (8, 16))
    k0 = jax.random.normal(jax.random.key(4), (8, 16))
    v0 = jax.random.normal(jax.random.key(5), (8, 16))
    q5 = jax.random.normal(jax.random.key(6), (16, 16))
    k5 = jax.random.normal(jax.random.key(7), (8, 16))
    tensors = {
        "model.language_model.layers.0.self_attn.q_proj.weight": q0,
        "model.language_model.layers.0.self_attn.k_proj.weight": k0,
        "model.language_model.layers.0.self_attn.v_proj.weight": v0,
        "model.language_model.layers.5.self_attn.q_proj.weight": q5,
        "model.language_model.layers.5.self_attn.k_proj.weight": k5,
    }

    step_exec = hf_mapping.make_streaming_executor(
        params_safetensors.build_from_hf_manifest(config, _CG)
    )
    out = step_exec(tensors)

    np.testing.assert_array_equal(
        out["layers.0.attn.q_einsum.w"],
        jnp.transpose(q0.T.reshape(16, 2, 4), (1, 0, 2)),
    )
    np.testing.assert_array_equal(
        out["layers.0.attn.kv_einsum.w"],
        jnp.stack(
            [
                jnp.transpose(k0.T.reshape(16, 2, 4), (1, 0, 2)),
                jnp.transpose(v0.T.reshape(16, 2, 4), (1, 0, 2)),
            ],
            axis=0,
        ),
    )
    np.testing.assert_array_equal(
        out["layers.5.attn.q_einsum.w"],
        jnp.transpose(q5.T.reshape(16, 2, 8), (1, 0, 2)),
    )
    np.testing.assert_array_equal(
        out["layers.5.attn.k_einsum.w"],
        jnp.transpose(k5.T.reshape(16, 1, 8), (1, 0, 2)),
    )
    self.assertNotIn("layers.5.attn.kv_einsum.w", out)

  def test_streaming_executor_handles_vision_and_audio_encoders(self):
    config = model_lib.ModelConfig.gemma4_12b()
    config.num_layers = 1
    config.vision_encoder = vision_lib.VisionEncoderConfig(
        num_layers=1, d_model=8, num_heads=2, ffw_hidden=16
    )
    config.audio_encoder = audio_lib.ConformerConfig(
        num_layers=1, model_dims=8, atten_num_heads=2
    )
    vq = jax.random.normal(jax.random.key(10), (8, 8))
    vk = jax.random.normal(jax.random.key(11), (8, 8))
    vv = jax.random.normal(jax.random.key(12), (8, 8))
    vo = jax.random.normal(jax.random.key(13), (8, 8))
    vgate = jax.random.normal(jax.random.key(14), (16, 8))
    vup = jax.random.normal(jax.random.key(15), (16, 8))
    ain = jax.random.normal(jax.random.key(16), (8, 1024))
    apost = jax.random.normal(jax.random.key(17), (8, 8))
    arel = jax.random.normal(jax.random.key(18), (8, 8))
    tensors = {
        "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight": (
            vq
        ),
        "model.vision_tower.encoder.layers.0.self_attn.k_proj.linear.weight": (
            vk
        ),
        "model.vision_tower.encoder.layers.0.self_attn.v_proj.linear.weight": (
            vv
        ),
        "model.vision_tower.encoder.layers.0.self_attn.o_proj.linear.weight": (
            vo
        ),
        "model.vision_tower.encoder.layers.0.mlp.gate_proj.linear.weight": (
            vgate
        ),
        "model.vision_tower.encoder.layers.0.mlp.up_proj.linear.weight": vup,
        "model.audio_tower.subsample_conv_projection.input_proj_linear.weight": (
            ain
        ),
        "model.audio_tower.layers.0.self_attn.post.linear.weight": apost,
        "model.audio_tower.layers.0.self_attn.relative_k_proj.weight": arel,
    }

    step_exec = hf_mapping.make_streaming_executor(
        params_safetensors.build_from_hf_manifest(config, _CG)
    )
    out = step_exec(tensors)

    np.testing.assert_array_equal(
        out["vision_encoder.layers.0.attn.q_einsum.w"],
        jnp.transpose(vq.reshape(2, 4, 8), (0, 2, 1)),
    )
    np.testing.assert_array_equal(
        out["vision_encoder.layers.0.attn.kv_einsum.w"],
        jnp.stack(
            [
                jnp.transpose(vk.reshape(2, 4, 8), (0, 2, 1)),
                jnp.transpose(vv.reshape(2, 4, 8), (0, 2, 1)),
            ],
            axis=0,
        ),
    )
    np.testing.assert_array_equal(
        out["vision_encoder.layers.0.attn.attn_vec_einsum.w"],
        jnp.transpose(vo.reshape(8, 2, 4), (1, 2, 0)),
    )
    np.testing.assert_array_equal(
        out["vision_encoder.layers.0.mlp.gating_einsum.w"],
        jnp.stack([vgate, vup], axis=0),
    )
    np.testing.assert_array_equal(
        out["audio_encoder.feature.input_proj.kernel"],
        jnp.transpose(ain.reshape(8, 32, 32), (1, 2, 0)),
    )
    np.testing.assert_array_equal(
        out["audio_encoder.conformer_layers.0.trans_atten.post.kernel"],
        jnp.transpose(apost.reshape(8, 2, 4), (1, 2, 0)),
    )
    np.testing.assert_array_equal(
        out[
            "audio_encoder.conformer_layers.0.trans_atten.self_atten.relative_position_embedding.pos_proj.kernel"
        ],
        jnp.transpose(arel.reshape(2, 4, 8), (2, 0, 1)),
    )

  def test_build_gemma4_manifest_and_export_roundtrip(self):
    config = model_lib.ModelConfig(
        num_layers=2,
        num_embed=32,
        embed_dim=8,
        hidden_dim=16,
        num_heads=2,
        head_dim=4,
        num_kv_heads=1,
        global_key_size=6,
        num_global_kv_heads=1,
        k_eq_v_global=True,
        sliding_window_size=16,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        attention_pattern=(
            model_lib.AttentionType.LOCAL_SLIDING,
            model_lib.AttentionType.GLOBAL,
        ),
        vision_encoder=None,
        audio_encoder=None,
    )
    manifest = params_safetensors.build_from_hf_manifest(config, _CG)
    to_hf = params_safetensors.build_to_hf_manifest(config, _CG)
    self.assertEqual(manifest["model_type"], _CG_MODEL_TYPE)
    self.assertEqual(to_hf, hf_mapping.invert_manifest(manifest))

    model = model_lib.Gemma4(config, rngs=nnx.Rngs(0), text_only=True)
    hf_state = params_safetensors.export_to_hf_state_dict(
        model, config, hf_variant=_CG
    )
    self.assertEqual(
        hf_state["model.language_model.layers.0.self_attn.q_proj.weight"].shape,
        (8, 8),
    )
    self.assertEqual(
        hf_state["model.language_model.layers.1.self_attn.q_proj.weight"].shape,
        (12, 8),
    )
    self.assertEqual(
        hf_state["model.language_model.layers.1.self_attn.k_proj.weight"].shape,
        (6, 8),
    )
    self.assertNotIn(
        "model.language_model.layers.1.self_attn.v_proj.weight", hf_state
    )
    restored = params_safetensors.from_hf_impl(manifest, hf_state)
    re_exported = params_safetensors.to_hf_impl(to_hf, restored)
    for k, v in hf_state.items():
      np.testing.assert_allclose(re_exported[k], v, rtol=1e-6, atol=1e-6)

    ckpt_dir = self.create_tempdir().full_path
    safe_np.save_file(
        {k: np.ascontiguousarray(np.asarray(v)) for k, v in hf_state.items()},
        os.path.join(ckpt_dir, "model.safetensors"),
    )
    loaded = params_safetensors.create_model_from_safe_tensors(
        ckpt_dir, config, dtype=jnp.float32, text_only=True,
        hf_variant=_CG,
    )
    tokens = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
    positions = jnp.arange(4, dtype=jnp.int32)[None, :]
    mask = jnp.tril(jnp.ones((1, 4, 4), dtype=jnp.bool_))
    logits_orig, _ = model(tokens, positions, None, mask)
    logits_load, _ = loaded(tokens, positions, None, mask)
    np.testing.assert_allclose(logits_load, logits_orig, rtol=1e-5, atol=1e-5)

    full_cfg = model_lib.ModelConfig(
        num_layers=2,
        num_embed=32,
        embed_dim=8,
        hidden_dim=16,
        num_heads=2,
        head_dim=4,
        num_kv_heads=1,
        global_key_size=6,
        num_global_kv_heads=1,
        k_eq_v_global=True,
        sliding_window_size=16,
        per_layer_input_dim=4,
        enable_moe=True,
        num_experts=2,
        num_experts_per_tok=1,
        expert_dim=8,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        attention_pattern=(
            model_lib.AttentionType.LOCAL_SLIDING,
            model_lib.AttentionType.GLOBAL,
        ),
        vision_encoder=vision_lib.VisionEncoderConfig(
            num_layers=1,
            d_model=8,
            num_heads=2,
            ffw_hidden=16,
            use_clipped_linears=True,
            standardize_embeddings=True,
        ),
        audio_encoder=audio_lib.ConformerConfig(
            num_layers=1, model_dims=8, lm_model_dims=8, atten_num_heads=2
        ),
    )
    gemma4_from = params_safetensors.build_from_hf_manifest(full_cfg, _CG)
    gemma4_to = params_safetensors.build_to_hf_manifest(full_cfg, _CG)
    self.assertEqual(gemma4_from["model_type"], _CG_MODEL_TYPE)
    self.assertEqual(gemma4_to, hf_mapping.invert_manifest(gemma4_from))
    no_gk_cfg = dataclasses.replace(
        full_cfg, global_key_size=None, attention_pattern=()
    )
    self.assertEqual(
        params_safetensors.build_from_hf_manifest(no_gk_cfg, _CG)["model_type"],
        _CG_MODEL_TYPE,
    )

    text_only_from_mm = model_lib.Gemma4(
        full_cfg, rngs=nnx.Rngs(0), text_only=True
    )
    hf_text_only = params_safetensors.export_to_hf_state_dict(
        text_only_from_mm, full_cfg,
        hf_variant=_CG,
    )
    self.assertNotIn(
        "model.vision_tower.std_scale",
        hf_text_only,
    )
    mm_ckpt_dir = self.create_tempdir().full_path
    mm_state = {
        k: np.ascontiguousarray(np.asarray(v)) for k, v in hf_text_only.items()
    }
    mm_state[
        "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight"
    ] = np.ones((1, 1), dtype=np.float32)
    safe_np.save_file(
        mm_state,
        os.path.join(mm_ckpt_dir, "model.safetensors"),
    )
    loaded_text_only = params_safetensors.create_model_from_safe_tensors(
        mm_ckpt_dir, full_cfg, dtype=jnp.float32, text_only=True,
        hf_variant=_CG,
    )
    self.assertTrue(loaded_text_only.text_only)

    full_model = model_lib.Gemma4(full_cfg, rngs=nnx.Rngs(0), text_only=False)
    orig_full = safetensors_loader.flatten_model_state(full_model)
    full_hf = params_safetensors.export_to_hf_state_dict(
        full_model, full_cfg, hf_variant=_CG
    )
    for proj in (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ):
      for bound in ("input_min", "input_max", "output_min", "output_max"):
        self.assertIn(
            f"model.vision_tower.encoder.layers.0.{proj}.{bound}", full_hf
        )
    self.assertIn(
        "model.audio_tower.layers.0.self_attn.post.linear.weight", full_hf
    )
    self.assertIn(
        "model.audio_tower.layers.0.self_attn.relative_k_proj.weight", full_hf
    )
    full_restored = params_safetensors.from_hf_impl(
        gemma4_from,
        {
            **full_hf,
            "lm_head.weight": jnp.ones((32, 8), dtype=jnp.float32),
        },
    )
    self.assertEqual(set(full_restored.keys()), set(orig_full.keys()))
    for k, v in orig_full.items():
      np.testing.assert_allclose(
          full_restored[k], v, rtol=1e-6, atol=1e-6, err_msg=k
      )

  def test_random_weights_legacy_parity_explicit_variants_and_kv_sharing(self):
    """Bit-by-bit legacy parity, explicit HF variants, and KV sharing.

    DEPRECATED (testing-only): The inline hand-written legacy conversion math
    below is deprecated, retained strictly for bit-by-bit parity testing during
    migration to `hf_mapping`, and will be removed in a follow-up cleanup CL.
    """
    # Single 1D `jax.random.normal` compile sliced across all distinct shapes
    # avoids per-shape XLA LLVM compilation timeouts on CPU Forge runners.
    normal_pool = np.asarray(
        jax.random.normal(jax.random.key(42), (262144,), dtype=jnp.float32)
    )
    cursor = [0]

    def rand(*shape: int) -> jax.Array:
      size = int(np.prod(shape))
      start = cursor[0]
      cursor[0] = start + size
      return jnp.asarray(normal_pool[start : start + size].reshape(shape))

    def rand_for_model(
        c: model_lib.ModelConfig, *, text_only: bool
    ) -> dict[str, jax.Array]:
      abs_m = nnx.eval_shape(
          lambda: model_lib.Gemma4(c, rngs=nnx.Rngs(0), text_only=text_only)
      )
      leaves, _ = jax.tree_util.tree_flatten_with_path(
          nnx.state(abs_m, nnx.Param).to_pure_dict()
      )
      return {
          safetensors_loader.path_to_key(p): rand(*v.shape) for p, v in leaves
      }

    for k_eq_v_global in (False, True):
      cfg = model_lib.ModelConfig(
          num_layers=2,
          num_embed=23,
          embed_dim=12,
          hidden_dim=17,
          num_heads=3,
          head_dim=4,
          num_kv_heads=2,
          global_key_size=6,
          num_global_kv_heads=1,
          k_eq_v_global=k_eq_v_global,
          sliding_window_size=16,
          per_layer_input_dim=5,
          enable_moe=True,
          num_experts=3,
          num_experts_per_tok=2,
          expert_dim=7,
          dtype=jnp.float32,
          param_dtype=jnp.float32,
          attention_pattern=(
              model_lib.AttentionType.LOCAL_SLIDING,
              model_lib.AttentionType.GLOBAL,
          ),
          vision_encoder=vision_lib.VisionEncoderConfig(
              num_layers=1,
              d_model=10,
              num_heads=2,
              ffw_hidden=14,
              pos_emb_shape_yx=(9, 2),
              use_clipped_linears=True,
              standardize_embeddings=True,
          ),
          audio_encoder=audio_lib.ConformerConfig(
              num_layers=1,
              model_dims=12,
              lm_model_dims=12,
              atten_num_heads=3,
              conv_kernel_size=5,
          ),
      )
      # Randomize every Tunix parameter distinctly, export, then independently
      # compute legacy HEAD _get_key_and_transform_mapping + _make_preprocess_fn
      # on the HF tensors and assert bit-by-bit equality.
      rand_tunix = rand_for_model(cfg, text_only=False)
      to_spec = params_safetensors.build_to_hf_manifest(cfg, _CG)
      hf_dict = params_safetensors.to_hf_impl(to_spec, rand_tunix)

      # DEPRECATED (testing-only): Legacy hand-written conversion reference
      # path is deprecated, retained strictly for bit-by-bit parity testing
      # during migration to `hf_mapping`, and will be removed in a follow-up CL.
      tp = "model.language_model."
      np.testing.assert_array_equal(
          rand_tunix["embedder.per_layer_model_projection.w"],
          hf_dict[f"{tp}per_layer_model_projection.weight"].T,
      )
      for i, (kh, kv_n) in enumerate(((4, 2), (6, 1))):
        q_hf = hf_dict[f"{tp}layers.{i}.self_attn.q_proj.weight"]
        np.testing.assert_array_equal(
            rand_tunix[f"layers.{i}.attn.q_einsum.w"],
            jnp.transpose(q_hf.T.reshape(12, 3, kh), (1, 0, 2)),
        )
        o_hf = hf_dict[f"{tp}layers.{i}.self_attn.o_proj.weight"]
        np.testing.assert_array_equal(
            rand_tunix[f"layers.{i}.attn.attn_vec_einsum.w"],
            o_hf.T.reshape(3, kh, 12),
        )
        k_leg = jnp.transpose(
            hf_dict[f"{tp}layers.{i}.self_attn.k_proj.weight"].T.reshape(
                12, kv_n, kh
            ),
            (1, 0, 2),
        )
        if i == 1 and k_eq_v_global:
          np.testing.assert_array_equal(
              rand_tunix["layers.1.attn.k_einsum.w"], k_leg
          )
        else:
          v_leg = jnp.transpose(
              hf_dict[f"{tp}layers.{i}.self_attn.v_proj.weight"].T.reshape(
                  12, kv_n, kh
              ),
              (1, 0, 2),
          )
          np.testing.assert_array_equal(
              rand_tunix[f"layers.{i}.attn.kv_einsum.w"],
              jnp.stack([k_leg, v_leg], axis=0),
          )
        np.testing.assert_array_equal(
            rand_tunix[f"layers.{i}.moe.gating_einsum"],
            hf_dict[f"{tp}layers.{i}.experts.gate_up_proj"].reshape(
                3, 2, 7, 12
            ),
        )
        np.testing.assert_array_equal(
            rand_tunix[f"layers.{i}.moe.linear"],
            jnp.transpose(
                hf_dict[f"{tp}layers.{i}.experts.down_proj"], (0, 2, 1)
            ),
        )

      np.testing.assert_array_equal(
          rand_tunix["vision_encoder.entry.pos_emb"],
          jnp.transpose(
              hf_dict[
                  "model.vision_tower.patch_embedder.position_embedding_table"
              ],
              (1, 0, 2),
          ),
      )
      np.testing.assert_array_equal(
          rand_tunix["vision_encoder.layers.0.attn.attn_vec_einsum.w"],
          jnp.transpose(
              hf_dict[
                  "model.vision_tower.encoder.layers.0.self_attn.o_proj.linear.weight"
              ].reshape(10, 2, 5),
              (1, 2, 0),
          ),
      )
      np.testing.assert_array_equal(
          rand_tunix["audio_encoder.feature.input_proj.kernel"],
          jnp.transpose(
              hf_dict[
                  "model.audio_tower.subsample_conv_projection.input_proj_linear.weight"
              ].reshape(12, 32, 32),
              (1, 2, 0),
          ),
      )
      np.testing.assert_array_equal(
          rand_tunix[
              "audio_encoder.conformer_layers.0.trans_atten.self_atten.relative_position_embedding.pos_proj.kernel"
          ],
          jnp.transpose(
              hf_dict[
                  "model.audio_tower.layers.0.self_attn.relative_k_proj.weight"
              ].reshape(3, 4, 12),
              (2, 0, 1),
          ),
      )

      # Unclipped vision (`use_clipped_linears=False`) + one explicit variant
      # per HF class, each round-tripped through a checkpoint in its layout:
      unclipped_v = dataclasses.replace(
          cfg.vision_encoder, use_clipped_linears=False
      )
      unclipped_cfg = dataclasses.replace(cfg, vision_encoder=unclipped_v)
      unclipped_m = params_safetensors.build_from_hf_manifest(
          unclipped_cfg, _CG
      )
      unclipped_targets = itertools.chain.from_iterable(
          r["target_keys"] for r in unclipped_m["rules"]
      )
      self.assertFalse(
          any(
              k.startswith("vision_encoder.") and "clip_" in k
              for k in unclipped_targets
          )
      )
      variants = params_safetensors.HfVariant
      fmt_dirs = {}
      for variant, text_p, text_only in (
          (variants.CONDITIONAL_GENERATION, "model.language_model.", False),
          (variants.CAUSAL_LM, "model.", True),
      ):
        mm_prefixes = (
            "vision_encoder.",
            "audio_encoder.",
            "embedder.mm_",
            "embedder.audio_",
        )
        eff_rand = {
            k: v
            for k, v in rand_tunix.items()
            if not text_only or not k.startswith(mm_prefixes)
        }
        eff_c = (
            dataclasses.replace(cfg, vision_encoder=None, audio_encoder=None)
            if text_only
            else cfg
        )
        fmt_hf = params_safetensors.to_hf_impl(
            params_safetensors.build_to_hf_manifest(eff_c, variant), eff_rand
        )
        self.assertIn(f"{text_p}embed_tokens.weight", fmt_hf)
        self.assertIn(f"{text_p}layers.0.experts.gate_up_proj", fmt_hf)
        self.assertIn(f"{text_p}layers.0.experts.down_proj", fmt_hf)
        self.assertEqual(
            "model.vision_tower.std_scale" in fmt_hf, not text_only
        )
        self.assertEqual(
            "model.embed_audio.embedding_projection.weight" in fmt_hf,
            not text_only,
        )
        fmt_dirs[variant] = self.create_tempdir().full_path
        safe_np.save_file(
            {k: np.array(v, order="C", copy=True) for k, v in fmt_hf.items()},
            os.path.join(fmt_dirs[variant], "model.safetensors"),
        )
        loaded_fmt = params_safetensors.create_model_from_safe_tensors(
            fmt_dirs[variant],
            cfg,
            dtype=jnp.float32,
            text_only=text_only,
        )
        self.assertEqual(loaded_fmt._hf_variant, variant)  # pylint: disable=protected-access
        loaded_flat = safetensors_loader.flatten_model_state(loaded_fmt)
        self.assertEqual(set(loaded_flat.keys()), set(eff_rand.keys()))
        for k, v in eff_rand.items():
          np.testing.assert_array_equal(loaded_flat[k], v, err_msg=k)
        # Export uses the remembered `_hf_variant` without explicit override.
        re_exp = params_safetensors.export_to_hf_state_dict(loaded_fmt, cfg)
        self.assertIn(f"{text_p}embed_tokens.weight", re_exp)
        # Also verify `model.safetensors.index.json` anchor resolution path.
        idx_dir = self.create_tempdir().full_path
        safe_np.save_file(
            {k: np.array(v, order="C", copy=True) for k, v in fmt_hf.items()},
            os.path.join(idx_dir, "model-00001-of-00001.safetensors"),
        )
        with open(
            os.path.join(idx_dir, "model.safetensors.index.json"), "w"
        ) as f:
          json.dump(
              {
                  "weight_map": {
                      k: "model-00001-of-00001.safetensors" for k in fmt_hf
                  }
              },
              f,
          )
        loaded_idx = params_safetensors.create_model_from_safe_tensors(
            idx_dir,
            cfg,
            dtype=jnp.float32,
            text_only=text_only,
        )
        self.assertEqual(loaded_idx._hf_variant, variant)  # pylint: disable=protected-access
      # A checkpoint in another layout than the declared variant fails and the
      # error names the declared variant: checkpoint keys are never probed.
      for declared, actual in (
          (variants.CONDITIONAL_GENERATION, variants.CAUSAL_LM),
          (variants.CAUSAL_LM, variants.CONDITIONAL_GENERATION),
      ):
        with self.assertRaisesRegex(
            ValueError,
            f"declared HF layout 'gemma4/{declared.value}'; missing",
        ):
          params_safetensors.create_model_from_safe_tensors(
              fmt_dirs[actual],
              cfg,
              dtype=jnp.float32,
              text_only=True,
              hf_variant=declared,
          )
      # `CAUSAL_LM` has no tower affix: a multimodal config cannot use it.
      with self.assertRaisesRegex(
          ValueError,
          "Gemma4 multimodal config requires CONDITIONAL_GENERATION",
      ):
        params_safetensors.build_from_hf_manifest(cfg, variants.CAUSAL_LM)
      audio_only_cfg = dataclasses.replace(cfg, vision_encoder=None)
      with self.assertRaisesRegex(
          ValueError,
          "Gemma4 multimodal config requires CONDITIONAL_GENERATION",
      ):
        params_safetensors.build_from_hf_manifest(
            audio_only_cfg, variants.CAUSAL_LM
        )
      with self.assertRaisesRegex(ValueError, "Pass `hf_variant`"):
        params_safetensors.build_from_hf_manifest(cfg)
      fresh_model = model_lib.Gemma4(cfg, rngs=nnx.Rngs(0), text_only=True)
      self.assertIsNone(fresh_model._hf_variant)  # pylint: disable=protected-access
      with self.assertRaisesRegex(ValueError, "Pass `hf_variant`"):
        params_safetensors.export_to_hf_state_dict(fresh_model, cfg)
      with self.assertRaisesRegex(ValueError, "Pass `hf_variant`"):
        params_safetensors.create_model_from_safe_tensors(
            fmt_dirs[variants.CAUSAL_LM],
            cfg,
            dtype=jnp.float32,
            text_only=True,
            hf_variant="causal_lm",  # type: ignore[arg-type]
        )

      # KV-shared layers (4 layers: 0,1 unshared; 2->0 local, 3->1 global) +
      # double-wide MLP on shared layers (`override_kv_shared_ffw_hidden=26`).
      kv_cfg = dataclasses.replace(
          cfg,
          num_layers=4,
          num_heads=4,
          head_dim=8,
          frac_shared_layers=0.5,
          override_kv_shared_ffw_hidden=26,
          vision_encoder=None,
          audio_encoder=None,
      )
      kv_rand = rand_for_model(kv_cfg, text_only=True)
      kv_hf = params_safetensors.to_hf_impl(
          params_safetensors.build_to_hf_manifest(kv_cfg, _CG), kv_rand
      )
      for shared_l in (2, 3):
        self.assertNotIn(
            f"{tp}layers.{shared_l}.self_attn.k_proj.weight", kv_hf
        )
        self.assertNotIn(
            f"{tp}layers.{shared_l}.self_attn.v_proj.weight", kv_hf
        )
        self.assertNotIn(
            f"{tp}layers.{shared_l}.self_attn.k_norm.weight", kv_hf
        )
      kv_dir = self.create_tempdir().full_path
      safe_np.save_file(
          {k: np.array(v, order="C", copy=True) for k, v in kv_hf.items()},
          os.path.join(kv_dir, "model.safetensors"),
      )
      kv_loaded = params_safetensors.create_model_from_safe_tensors(
          kv_dir, kv_cfg, dtype=jnp.float32, text_only=True,
          hf_variant=_CG,
      )
      kv_loaded_flat = safetensors_loader.flatten_model_state(kv_loaded)
      np.testing.assert_array_equal(
          kv_loaded_flat["layers.2.attn.kv_einsum.w"],
          kv_loaded_flat["layers.0.attn.kv_einsum.w"],
      )
      np.testing.assert_array_equal(
          kv_loaded_flat["layers.2.attn._key_norm.scale"],
          kv_loaded_flat["layers.0.attn._key_norm.scale"],
      )
      kv_3_attr = "k_einsum" if k_eq_v_global else "kv_einsum"
      np.testing.assert_array_equal(
          kv_loaded_flat[f"layers.3.attn.{kv_3_attr}.w"],
          kv_loaded_flat[f"layers.1.attn.{kv_3_attr}.w"],
      )
      np.testing.assert_array_equal(
          kv_loaded_flat["layers.3.attn._key_norm.scale"],
          kv_loaded_flat["layers.1.attn._key_norm.scale"],
      )
      kv_re_exported = params_safetensors.export_to_hf_state_dict(
          kv_loaded, kv_cfg,
          hf_variant=_CG,
      )
      self.assertEqual(set(kv_re_exported.keys()), set(kv_hf.keys()))
      for k, v in kv_hf.items():
        np.testing.assert_array_equal(kv_re_exported[k], v, err_msg=k)
      if k_eq_v_global:
        tokens = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
        pos = jnp.arange(4, dtype=jnp.int32)[None, :]
        mask = jnp.tril(jnp.ones((1, 4, 4), dtype=jnp.bool_))
        logits, _ = kv_loaded(tokens, pos, None, mask)
        self.assertEqual(logits.shape, (1, 4, 23))


if __name__ == "__main__":
  absltest.main()
