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

import dataclasses
import inspect
from absl.testing import absltest
from absl.testing import parameterized
from tunix.models import naming
from tunix.models.gemma import model as gemma_model
from tunix.models.gemma3 import model as gemma3_model
from tunix.models.llama3 import model as llama3_model
from tunix.models.qwen2 import model as qwen2_model
from tunix.models.qwen3 import model as qwen3_model


@dataclasses.dataclass(frozen=True)
class ModelTestInfo:
  family: str
  version: str
  id: str
  category: str


_MODEL_INFO_MAP = {
    'gemma-2b': ModelTestInfo(
        family='gemma',
        version='2b',
        id='gemma_2b',
        category='gemma',
    ),
    'gemma-2b-it': ModelTestInfo(
        family='gemma',
        version='2b_it',
        id='gemma_2b_it',
        category='gemma',
    ),
    'gemma1.1-2b-it': ModelTestInfo(
        family='gemma1p1',
        version='2b_it',
        id='gemma1p1_2b_it',
        category='gemma',
    ),
    'gemma-7b': ModelTestInfo(
        family='gemma',
        version='7b',
        id='gemma_7b',
        category='gemma',
    ),
    'gemma-7b-it': ModelTestInfo(
        family='gemma',
        version='7b_it',
        id='gemma_7b_it',
        category='gemma',
    ),
    'gemma1.1-7b-it': ModelTestInfo(
        family='gemma1p1',
        version='7b_it',
        id='gemma1p1_7b_it',
        category='gemma',
    ),
    'gemma2-2b': ModelTestInfo(
        family='gemma2',
        version='2b',
        id='gemma2_2b',
        category='gemma',
    ),
    'gemma2-2b-it': ModelTestInfo(
        family='gemma2',
        version='2b_it',
        id='gemma2_2b_it',
        category='gemma',
    ),
    'gemma2-9b': ModelTestInfo(
        family='gemma2',
        version='9b',
        id='gemma2_9b',
        category='gemma',
    ),
    'gemma2-9b-it': ModelTestInfo(
        family='gemma2',
        version='9b_it',
        id='gemma2_9b_it',
        category='gemma',
    ),
    'gemma-3-270m': ModelTestInfo(
        family='gemma3',
        version='270m',
        id='gemma3_270m',
        category='gemma3',
    ),
    'gemma-3-270m-it': ModelTestInfo(
        family='gemma3',
        version='270m_it',
        id='gemma3_270m_it',
        category='gemma3',
    ),
    'gemma-3-1b': ModelTestInfo(
        family='gemma3',
        version='1b',
        id='gemma3_1b',
        category='gemma3',
    ),
    'gemma-3-1b-it': ModelTestInfo(
        family='gemma3',
        version='1b_it',
        id='gemma3_1b_it',
        category='gemma3',
    ),
    'gemma-3-4b': ModelTestInfo(
        family='gemma3',
        version='4b',
        id='gemma3_4b',
        category='gemma3',
    ),
    'gemma-3-4b-it': ModelTestInfo(
        family='gemma3',
        version='4b_it',
        id='gemma3_4b_it',
        category='gemma3',
    ),
    'gemma-3-12b': ModelTestInfo(
        family='gemma3',
        version='12b',
        id='gemma3_12b',
        category='gemma3',
    ),
    'gemma-3-12b-it': ModelTestInfo(
        family='gemma3',
        version='12b_it',
        id='gemma3_12b_it',
        category='gemma3',
    ),
    'gemma-3-27b': ModelTestInfo(
        family='gemma3',
        version='27b',
        id='gemma3_27b',
        category='gemma3',
    ),
    'gemma-3-27b-it': ModelTestInfo(
        family='gemma3',
        version='27b_it',
        id='gemma3_27b_it',
        category='gemma3',
    ),
    'llama3-70b': ModelTestInfo(
        family='llama3',
        version='70b',
        id='llama3_70b',
        category='llama3',
    ),
    'llama3-405b': ModelTestInfo(
        family='llama3',
        version='405b',
        id='llama3_405b',
        category='llama3',
    ),
    'llama3.1-8b': ModelTestInfo(
        family='llama3p1',
        version='8b',
        id='llama3p1_8b',
        category='llama3',
    ),
    'llama3.2-1b': ModelTestInfo(
        family='llama3p2',
        version='1b',
        id='llama3p2_1b',
        category='llama3',
    ),
    'llama3.2-3b': ModelTestInfo(
        family='llama3p2',
        version='3b',
        id='llama3p2_3b',
        category='llama3',
    ),
    'qwen2.5-0.5b': ModelTestInfo(
        family='qwen2p5',
        version='0p5b',
        id='qwen2p5_0p5b',
        category='qwen2',
    ),
    'qwen2.5-1.5b': ModelTestInfo(
        family='qwen2p5',
        version='1p5b',
        id='qwen2p5_1p5b',
        category='qwen2',
    ),
    'qwen2.5-3b': ModelTestInfo(
        family='qwen2p5',
        version='3b',
        id='qwen2p5_3b',
        category='qwen2',
    ),
    'qwen2.5-7b': ModelTestInfo(
        family='qwen2p5',
        version='7b',
        id='qwen2p5_7b',
        category='qwen2',
    ),
    'qwen2.5-math-1.5b': ModelTestInfo(
        family='qwen2p5',
        version='math_1p5b',
        id='qwen2p5_math_1p5b',
        category='qwen2',
    ),
    'deepseek-r1-distill-qwen-1.5b': ModelTestInfo(
        family='deepseek_r1_distill_qwen',
        version='1p5b',
        id='deepseek_r1_distill_qwen_1p5b',
        category='qwen2',
    ),
    'qwen3-0.6b': ModelTestInfo(
        family='qwen3',
        version='0p6b',
        id='qwen3_0p6b',
        category='qwen3',
    ),
    'qwen3-1.7b': ModelTestInfo(
        family='qwen3',
        version='1p7b',
        id='qwen3_1p7b',
        category='qwen3',
    ),
    'qwen3-8b': ModelTestInfo(
        family='qwen3',
        version='8b',
        id='qwen3_8b',
        category='qwen3',
    ),
    'qwen3-14b': ModelTestInfo(
        family='qwen3',
        version='14b',
        id='qwen3_14b',
        category='qwen3',
    ),
    'qwen3-30b': ModelTestInfo(
        family='qwen3',
        version='30b',
        id='qwen3_30b',
        category='qwen3',
    ),
}
_ALL_MODEL_MODULES = [
    gemma_model,
    gemma3_model,
    llama3_model,
    qwen2_model,
    qwen3_model,
]


def _validate_full_model_coverage():
  config_ids = []
  all_model_ids = {k.id for k in _MODEL_INFO_MAP.values()}
  # Check that all model configs in ModelConfig class are in _MODEL_INFO_MAP.
  for model_module in _ALL_MODEL_MODULES:
    if hasattr(model_module, 'ModelConfig'):
      for name, member in inspect.getmembers(model_module.ModelConfig):
        if (
            name.startswith('_')
            or name == 'get_default_sharding'
            or not inspect.ismethod(member)
            or member.__self__ is not model_module.ModelConfig
        ):
          continue
        if name not in all_model_ids:
          raise ValueError(
              f'Model id {name} not found in _MODEL_INFO_MAP. Make sure the'
              ' model is added to the map for full test coverage.'
          )
        config_ids.append(name)

  # Check each item in _MODEL_INFO_MAP maps to a valid config id.This is to
  # prevent deprecated models from lingering in the map.
  for model_info in _MODEL_INFO_MAP.values():
    if model_info.id not in config_ids:
      raise ValueError(
          f'Model name {model_info.id} not found in config_ids {config_ids}.'
          ' Seems to be an oboslete/deprecated model. Remove from'
          ' _MODEL_INFO_MAP.'
      )


def _get_test_cases_for_get_model_config_id():
  test_cases = []
  _validate_full_model_coverage()
  for name, model_info in _MODEL_INFO_MAP.items():
    test_cases.append({
        'testcase_name': model_info.id,
        'model_name': name,
        'expected_config_id': model_info.id,
    })
  return test_cases


def _get_test_cases_for_get_model_family_and_version():
  test_cases = []
  _validate_full_model_coverage()
  for name, model_info in _MODEL_INFO_MAP.items():
    test_cases.append({
        'testcase_name': model_info.id,
        'model_name': name,
        'expected_family': model_info.family,
        'expected_version': model_info.version,
    })
  test_cases.append({
      'testcase_name': 'family_only',
      'model_name': 'gemma1.1',
      'expected_family': 'gemma1p1',
      'expected_version': '',
  })
  return test_cases


def _get_test_cases_for_get_model_config_category():
  test_cases_dict = {}
  _validate_full_model_coverage()
  for name, model_info in _MODEL_INFO_MAP.items():
    if model_info.family not in test_cases_dict:
      test_cases_dict[model_info.family] = {
          'testcase_name': model_info.family,
          'model_name': name,
          'expected_category': model_info.category,
      }
  return list(test_cases_dict.values())


class TestNaming(parameterized.TestCase):

  def test_get_model_name_from_model_id(self):
    self.assertEqual(
        naming.get_model_name_from_model_id('meta-llama/Llama-3.1-8B'),
        'llama-3.1-8b',
    )
    self.assertEqual(
        naming.get_model_name_from_model_id('Qwen/Qwen2.5-0.5B'), 'qwen2.5-0.5b'
    )
    with self.assertRaises(ValueError):
      naming.get_model_name_from_model_id('Llama-3.1-8B')

  @parameterized.named_parameters(
      _get_test_cases_for_get_model_family_and_version()
  )
  def test_get_model_family_and_version(
      self, model_name, expected_family, expected_version
  ):
    self.assertEqual(
        naming.get_model_family_and_version(model_name),
        (expected_family, expected_version),
    )

  @parameterized.named_parameters(_get_test_cases_for_get_model_config_id())
  def test_get_model_config_id(self, model_name, expected_config_id):
    self.assertEqual(naming.get_model_config_id(model_name), expected_config_id)

  @parameterized.named_parameters(
      _get_test_cases_for_get_model_config_category()
  )
  def test_get_model_config_category(self, model_name, expected_category):
    self.assertEqual(
        naming.get_model_config_category(model_name), expected_category
    )

if __name__ == '__main__':
  absltest.main()
