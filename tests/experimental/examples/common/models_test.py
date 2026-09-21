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

"""Tests for the shared example model builders."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from jax import numpy as jnp
from tunix.experimental.examples.common import models
from tunix.models.gemma import model as gemma_model_lib
from tunix.models.qwen3 import model as qwen3_model_lib


class GemmaConfigTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("gemma2_dashed", "gemma-2-2b", 26),
      ("gemma2_underscored", "Gemma2_2B", 26),
      ("gemma1", "gemma-2b", 18),
  )
  def test_selects_expected_config(self, model_name, expected_num_layers):
    config = models._gemma_config(model_name)
    self.assertEqual(config.num_layers, expected_num_layers)

  def test_unsupported_model_name_raises(self):
    with self.assertRaisesRegex(ValueError, "Unsupported gemma model_name"):
      models._gemma_config("gemma-7b")


class Qwen3ConfigTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("0p6b_dotted", "Qwen3-0.6B", 1024),
      ("0p6b_sanitized", "qwen3_0p6b", 1024),
      ("1p7b_dotted", "Qwen3-1.7B", 2048),
      ("1p7b_sanitized", "qwen3_1p7b", 2048),
      ("8b", "Qwen3-8B", 4096),
      ("32b", "Qwen3-32B", 5120),
  )
  def test_selects_expected_config(self, model_name, expected_embed_dim):
    config = models._qwen3_config(model_name)
    self.assertEqual(config.embed_dim, expected_embed_dim)

  def test_unsupported_model_name_raises(self):
    with self.assertRaisesRegex(ValueError, "Unsupported qwen3 model_name"):
      models._qwen3_config("qwen3-3b")

  def test_applies_default_overrides(self):
    config = models._qwen3_config("Qwen3-0.6B")

    self.assertEqual(
        config.shd_config,
        qwen3_model_lib.ShardingConfig.get_default_sharding(),
    )
    self.assertEqual(config.dtype, jnp.bfloat16)
    self.assertEqual(config.param_dtype, jnp.float32)
    self.assertEqual(config.remat_config, qwen3_model_lib.RematConfig.NONE)
    self.assertFalse(config.use_flash_attention)
    self.assertEqual(
        config.flash_attention_block_size,
        qwen3_model_lib.ModelConfig.qwen3_0p6b().flash_attention_block_size,
    )

  @parameterized.named_parameters(
      ("none", "none", qwen3_model_lib.RematConfig.NONE),
      ("block", "block", qwen3_model_lib.RematConfig.BLOCK),
      ("decoder", "decoder", qwen3_model_lib.RematConfig.DECODER),
      ("mixed_case", "BlOcK", qwen3_model_lib.RematConfig.BLOCK),
  )
  def test_remat_config_is_parsed_case_insensitively(
      self, remat_config, expected
  ):
    config = models._qwen3_config("Qwen3-0.6B", remat_config=remat_config)
    self.assertEqual(config.remat_config, expected)

  def test_unknown_remat_config_raises(self):
    with self.assertRaises(KeyError):
      models._qwen3_config("Qwen3-0.6B", remat_config="full")

  def test_applies_flash_attention_overrides(self):
    config = models._qwen3_config(
        "Qwen3-0.6B",
        use_flash_attention=True,
        flash_attention_block_size=512,
    )

    self.assertTrue(config.use_flash_attention)
    self.assertEqual(config.flash_attention_block_size, 512)

  def test_unset_block_size_keeps_model_config_default(self):
    config = models._qwen3_config(
        "Qwen3-0.6B",
        use_flash_attention=True,
        flash_attention_block_size=None,
    )

    self.assertTrue(config.use_flash_attention)
    self.assertEqual(
        config.flash_attention_block_size,
        qwen3_model_lib.ModelConfig.qwen3_0p6b().flash_attention_block_size,
    )


class CreateModelTest(parameterized.TestCase):

  def test_creates_gemma_model(self):
    mesh = mock.sentinel.mesh
    with mock.patch.object(
        models.gemma_params_lib,
        "create_model_from_safe_tensors",
        autospec=True,
        return_value=mock.sentinel.gemma_model,
    ) as create_gemma:
      model = models.create_model("gemma-2-2b", "/tmp/gemma", mesh)

    self.assertIs(model, mock.sentinel.gemma_model)
    create_gemma.assert_called_once()
    args, kwargs = create_gemma.call_args
    self.assertEqual(args[0], "/tmp/gemma")
    self.assertIsInstance(args[1], gemma_model_lib.ModelConfig)
    self.assertIs(kwargs["mesh"], mesh)

  def test_creates_qwen3_model_with_overrides(self):
    mesh = mock.sentinel.mesh
    with mock.patch.object(
        models.qwen3_params_lib,
        "create_model_from_safe_tensors",
        autospec=True,
        return_value=mock.sentinel.qwen3_model,
    ) as create_qwen3:
      model = models.create_model(
          "Qwen3-1.7B",
          "/tmp/qwen3",
          mesh,
          parameter_dtype=jnp.float32,
          remat_config="decoder",
          use_flash_attention=True,
          flash_attention_block_size=256,
      )

    self.assertIs(model, mock.sentinel.qwen3_model)
    create_qwen3.assert_called_once()
    args, kwargs = create_qwen3.call_args
    self.assertEqual(args[0], "/tmp/qwen3")
    config = args[1]
    self.assertEqual(
        config.remat_config, qwen3_model_lib.RematConfig.DECODER
    )
    self.assertTrue(config.use_flash_attention)
    self.assertEqual(config.flash_attention_block_size, 256)
    self.assertIs(args[2], mesh)
    self.assertEqual(kwargs["dtype"], jnp.float32)

  def test_unsupported_model_name_raises(self):
    with self.assertRaisesRegex(ValueError, "Unsupported demo model_name"):
      models.create_model("llama-3-8b", "/tmp/llama", mock.sentinel.mesh)


if __name__ == "__main__":
  absltest.main()
