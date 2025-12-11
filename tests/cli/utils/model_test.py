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
from tunix.cli.utils import model


class ModelTest(parameterized.TestCase):

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
  @mock.patch("tunix.generate.tokenizer_adapter.Tokenizer", autospec=True)
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
  @mock.patch("qwix.LoraProvider", autospec=True)
  @mock.patch("qwix.apply_lora_to_model", autospec=True)
  @mock.patch("tunix.rl.reshard.reshard_model_to_mesh", autospec=True)
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


if __name__ == "__main__":
  absltest.main()
