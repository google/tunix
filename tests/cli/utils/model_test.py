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

import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import qwix
from tunix.cli.utils import model
from tunix.generate import tokenizer_adapter
from tunix.models import automodel
from tunix.rl import reshard
from tunix.tests import test_common


class ModelTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='no_path',
          tokenizer_path=None,
          expected_path='path1',
      ),
      dict(
          testcase_name='with_path',
          tokenizer_path='path2',
          expected_path='path2',
      ),
  )
  @mock.patch.object(tokenizer_adapter, 'Tokenizer', autospec=True)
  def test_create_tokenizer(
      self, mock_tokenizer, tokenizer_path, expected_path
  ):
    tokenizer_config = {
        'tokenizer_path': 'path1',
        'tokenizer_type': 'type1',
        'add_bos': True,
        'add_eos': False,
    }
    model.create_tokenizer(tokenizer_config, tokenizer_path=tokenizer_path)
    mock_tokenizer.assert_called_once_with(
        "type1", expected_path, True, False, mock.ANY
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='no_quant',
          lora_config={
              'module_path': 'path',
              'rank': 1,
              'alpha': 1.0,
          },
      ),
      dict(
          testcase_name='quant',
          lora_config={
              'module_path': 'path',
              'rank': 1,
              'alpha': 1.0,
              'tile_size': 1,
              'weight_qtype': 'int8',
          },
      ),
  )
  @mock.patch.object(qwix, 'LoraProvider', autospec=True)
  @mock.patch.object(qwix, 'apply_lora_to_model', autospec=True)
  @mock.patch.object(reshard, 'reshard_model_to_mesh', autospec=True)
  def test_apply_lora_to_model(
      self, mock_reshard, mock_apply_lora, mock_lora_provider, lora_config
  ):
    base_model = mock.create_autospec(
        test_common.ToyTransformer, instance=True, spec_set=True
    )
    base_model.get_model_input.return_value = {}
    mesh = mock.create_autospec(jax.sharding.Mesh, instance=True, spec_set=True)
    model.apply_lora_to_model(base_model, mesh, lora_config)
    mock_lora_provider.assert_called_once_with(**lora_config)
    mock_apply_lora.assert_called_once()
    mock_reshard.assert_called_once()

  @parameterized.named_parameters(
      dict(
          testcase_name='no_lora_default_tokenizer',
          model_config={
              'model_name': 'other_model',
              'model_source': 'gcs',
              'model_id': 'model1',
              'model_display': False,
          },
          tokenizer_config={'tokenizer_path': model._DEFAULT_TOKENIZER_PATH},
          expected_tokenizer_path=model._DEFAULT_TOKENIZER_PATH,
          apply_lora=False,
          model_display=False,
      ),
      dict(
          testcase_name='no_lora_gemma3_gcs_tokenizer',
          model_config={
              'model_name': 'gemma3-x',
              'model_source': 'gcs',
              'model_id': 'model1',
              'model_display': False,
          },
          tokenizer_config={'tokenizer_path': model._DEFAULT_TOKENIZER_PATH},
          expected_tokenizer_path=(
              'gs://gemma-data/tokenizers/tokenizer_gemma3.model'
          ),
          apply_lora=False,
          model_display=False,
      ),
      dict(
          testcase_name='no_lora_gemma_kaggle_tokenizer',
          model_config={
              'model_name': 'gemma-x',
              'model_source': 'kaggle',
              'model_id': 'model1',
              'model_display': False,
          },
          tokenizer_config={'tokenizer_path': model._DEFAULT_TOKENIZER_PATH},
          expected_tokenizer_path=os.path.join(
              'mock_model_path', 'tokenizer.model'
          ),
          apply_lora=False,
          model_display=False,
      ),
      dict(
          testcase_name='with_lora',
          model_config={
              'model_name': 'other_model',
              'model_source': 'huggingface',
              'model_id': 'model1',
              'model_display': False,
              'lora_config': {'rank': 1},
          },
          tokenizer_config={'tokenizer_path': model._DEFAULT_TOKENIZER_PATH},
          expected_tokenizer_path=model._DEFAULT_TOKENIZER_PATH,
          apply_lora=True,
          model_display=False,
      ),
      dict(
          testcase_name='model_display',
          model_config={
              'model_name': 'other_model',
              'model_source': 'gcs',
              'model_id': 'model1',
              'model_display': True,
          },
          tokenizer_config={'tokenizer_path': model._DEFAULT_TOKENIZER_PATH},
          expected_tokenizer_path=model._DEFAULT_TOKENIZER_PATH,
          apply_lora=False,
          model_display=True,
      ),
  )
  @mock.patch.object(model, 'apply_lora_to_model', autospec=True)
  @mock.patch.object(automodel, 'AutoModel', autospec=True)
  @mock.patch.object(nnx, 'display', autospec=True)
  def test_create_model(
      self,
      mock_nnx_display,
      mock_automodel,
      mock_apply_lora,
      model_config,
      tokenizer_config,
      expected_tokenizer_path,
      apply_lora,
      model_display,
  ):
    mesh = mock.create_autospec(jax.sharding.Mesh, instance=True, spec_set=True)
    mock_model = mock.create_autospec(nnx.Module, instance=True, spec_set=True)
    mock_automodel.from_pretrained.return_value = (
        mock_model,
        'mock_model_path',
    )
    mock_lora_model = mock.create_autospec(
        nnx.Module, instance=True, spec_set=True
    )
    mock_apply_lora.return_value = mock_lora_model

    returned_model, tokenizer_path = model.create_model(
        model_config, tokenizer_config, mesh
    )

    self.assertEqual(tokenizer_path, expected_tokenizer_path)
    mock_automodel.from_pretrained.assert_called_once()
    if apply_lora:
      mock_apply_lora.assert_called_once_with(
          mock_model, mesh, model_config['lora_config']
      )
      self.assertEqual(returned_model, mock_lora_model)
    else:
      mock_apply_lora.assert_not_called()
      self.assertEqual(returned_model, mock_model)
    if model_display:
      mock_nnx_display.assert_called_once_with(returned_model)
    else:
      mock_nnx_display.assert_not_called()


if __name__ == "__main__":
  absltest.main()
