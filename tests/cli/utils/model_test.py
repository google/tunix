# Copyright 2025 Google LLC
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

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
from tunix.cli.utils import model
from tunix.models import naming


def _get_all_models_test_parameters():
  return (
      dict(testcase_name="gemma-2b", model_name="gemma-2b"),
      dict(testcase_name="gemma-2b-it", model_name="gemma-2b-it"),
      dict(testcase_name="gemma-7b", model_name="gemma-7b"),
      dict(testcase_name="gemma-7b-it", model_name="gemma-7b-it"),
      dict(testcase_name="gemma1.1-2b-it", model_name="gemma1.1-2b-it"),
      dict(testcase_name="gemma1.1-7b-it", model_name="gemma1.1-7b-it"),
      dict(testcase_name="gemma2-2b", model_name="gemma2-2b"),
      dict(testcase_name="gemma2-2b-it", model_name="gemma2-2b-it"),
      dict(testcase_name="gemma2-9b", model_name="gemma2-9b"),
      dict(testcase_name="gemma2-9b-it", model_name="gemma2-9b-it"),
      dict(testcase_name="gemma3-270m", model_name="gemma3-270m"),
      dict(testcase_name="gemma3-270m-it", model_name="gemma3-270m-it"),
      dict(testcase_name="gemma3-1b", model_name="gemma3-1b"),
      dict(testcase_name="gemma3-1b-it", model_name="gemma3-1b-it"),
      dict(testcase_name="gemma3-4b", model_name="gemma3-4b"),
      dict(testcase_name="gemma3-4b-it", model_name="gemma3-4b-it"),
      dict(testcase_name="gemma3-12b", model_name="gemma3-12b"),
      dict(testcase_name="gemma3-12b-it", model_name="gemma3-12b-it"),
      dict(testcase_name="gemma3-27b", model_name="gemma3-27b"),
      dict(testcase_name="gemma3-27b-it", model_name="gemma3-27b-it"),
      dict(testcase_name="gemma-3-270m", model_name="gemma-3-270m"),
      dict(testcase_name="gemma-3-270m-it", model_name="gemma-3-270m-it"),
      dict(testcase_name="gemma-3-1b", model_name="gemma-3-1b"),
      dict(testcase_name="gemma-3-1b-it", model_name="gemma-3-1b-it"),
      dict(testcase_name="gemma-3-4b", model_name="gemma-3-4b"),
      dict(testcase_name="gemma-3-4b-it", model_name="gemma-3-4b-it"),
      dict(testcase_name="gemma-3-12b", model_name="gemma-3-12b"),
      dict(testcase_name="gemma-3-12b-it", model_name="gemma-3-12b-it"),
      dict(testcase_name="gemma-3-27b", model_name="gemma-3-27b"),
      dict(testcase_name="gemma-3-27b-it", model_name="gemma-3-27b-it"),
      dict(testcase_name="llama3-70b", model_name="llama3-70b"),
      dict(testcase_name="llama3-405b", model_name="llama3-405b"),
      dict(testcase_name="llama3.1-8b", model_name="llama3.1-8b"),
      dict(testcase_name="llama3.2-1b", model_name="llama3.2-1b"),
      dict(testcase_name="llama3.2-3b", model_name="llama3.2-3b"),
      dict(testcase_name="qwen2.5-0.5b", model_name="qwen2.5-0.5b"),
      dict(testcase_name="qwen2.5-1.5b", model_name="qwen2.5-1.5b"),
      dict(testcase_name="qwen2.5-3b", model_name="qwen2.5-3b"),
      dict(testcase_name="qwen2.5-7b", model_name="qwen2.5-7b"),
      dict(testcase_name="qwen2.5-math-1.5b", model_name="qwen2.5-math-1.5b"),
      dict(
          testcase_name="deepseek-r1-distill-qwen-1.5b",
          model_name="deepseek-r1-distill-qwen-1.5b",
      ),
      dict(testcase_name="qwen3-0.6b", model_name="qwen3-0.6b"),
      dict(testcase_name="qwen3-1.7b", model_name="qwen3-1.7b"),
      dict(testcase_name="qwen3-8b", model_name="qwen3-8b"),
      dict(testcase_name="qwen3-14b", model_name="qwen3-14b"),
      dict(testcase_name="qwen3-30b", model_name="qwen3-30b"),
  )


class ModelTest(parameterized.TestCase):

  @parameterized.named_parameters(*_get_all_models_test_parameters())
  def test_obtain_model_params_valid(self, model_name: str):
    model.obtain_model_params(model_name)

  @parameterized.named_parameters(*_get_all_models_test_parameters())
  def test_get_model_module(self, model_name: str):
    params_module = model.get_model_module(model_name, model.ModelModule.PARAMS)
    if naming.get_model_config_category(model_name) not in ["gemma", "gemma3"]:
      # TODO(b/444572467)
      getattr(params_module, "create_model_from_safe_tensors")

    model_lib_module = model.get_model_module(
        model_name, model.ModelModule.MODEL
    )
    getattr(model_lib_module, "ModelConfig")

  def test_get_model_module_invalid(self):
    with self.assertRaises(ValueError):
      model.get_model_module("invalid-model", model.ModelModule.PARAMS)

  @parameterized.named_parameters(*_get_all_models_test_parameters())
  def test_create_model_dynamically(self, model_name: str):
    if naming.get_model_config_category(model_name) in ["gemma", "gemma3"]:
      self.skipTest(
          "Gemma models do not support create_model_from_safe_tensors"
      )
    mock_create_fn = mock.Mock()
    mock_params_module = mock.Mock()
    mock_params_module.create_model_from_safe_tensors = mock_create_fn
    mock_params_module.__name__ = "mock_params_module"
    with mock.patch.object(
        model, "get_model_module", return_value=mock_params_module
    ):
      mesh = jax.sharding.Mesh(jax.devices(), ("devices",))
      model.create_model_dynamically(
          model_name, "file_dir", "model_config", mesh
      )
      mock_create_fn.assert_called_once_with(
          file_dir="file_dir", config="model_config", mesh=mesh
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="no_path",
          tokenizer_path=None,
          expected_path="path1",
      ),
      dict(
          testcase_name="with_path",
          tokenizer_path="path2",
          expected_path="path2",
      ),
  )
  @mock.patch("tunix.generate.tokenizer_adapter.Tokenizer")
  def test_create_tokenizer(
      self, mock_tokenizer, tokenizer_path, expected_path
  ):
    tokenizer_config = {
        "toknenizer_path": "path1",
        "tokenizer_type": "type1",
        "add_bos": True,
        "add_eos": False,
    }
    model.create_tokenizer(tokenizer_config, tokenizer_path=tokenizer_path)
    mock_tokenizer.assert_called_once_with(
        "type1", expected_path, True, False, mock.ANY
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="no_quant",
          lora_config={
              "module_path": "path",
              "rank": 1,
              "alpha": 1.0,
          },
      ),
      dict(
          testcase_name="quant",
          lora_config={
              "module_path": "path",
              "rank": 1,
              "alpha": 1.0,
              "tile_size": 1,
              "weight_qtype": "int8",
          },
      ),
  )
  @mock.patch("qwix.LoraProvider")
  @mock.patch("qwix.apply_lora_to_model")
  @mock.patch("tunix.rl.reshard.reshard_model_to_mesh")
  def test_apply_lora_to_model(
      self, mock_reshard, mock_apply_lora, mock_lora_provider, lora_config
  ):
    base_model = mock.Mock()
    base_model.get_model_input.return_value = {}
    mesh = mock.Mock()
    model.apply_lora_to_model(base_model, mesh, lora_config)
    mock_lora_provider.assert_called_once_with(**lora_config)
    mock_apply_lora.assert_called_once()
    mock_reshard.assert_called_once()

  @parameterized.named_parameters(
      dict(
          testcase_name="gemma",
          model_name="gemma-2b",
          expected_version="2b",
      ),
      dict(
          testcase_name="gemma2",
          model_name="gemma2-2b-it",
          expected_version="2-2b-it",
      ),
  )
  @mock.patch.object(model, "get_model_module")
  def test_create_gemma_model_from_params(
      self,
      mock_get_model_module,
      model_name,
      expected_version,
  ):
    mock_params_lib = mock.Mock()
    mock_model_lib = mock.Mock()
    mock_get_model_module.side_effect = [mock_params_lib, mock_model_lib]

    model._create_gemma_model_from_params("path", model_name)

    mock_params_lib.load_and_format_params.assert_called_once_with("path")
    mock_model_lib.Gemma.from_params.assert_called_once_with(
        mock_params_lib.load_and_format_params.return_value,
        version=expected_version,
    )


if __name__ == "__main__":
  absltest.main()
