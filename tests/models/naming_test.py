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
  id: str
  name: str
  family: str
  version: str
  config_id: str
  category: str


_TEST_MODEL_INFOS = (
    ModelTestInfo(
        id='google/gemma-2b',
        name='gemma-2b',
        family='gemma',
        version='2b',
        config_id='gemma_2b',
        category='gemma',
    ),
    ModelTestInfo(
        id='google/gemma-2b-it',
        name='gemma-2b-it',
        family='gemma',
        version='2b_it',
        config_id='gemma_2b_it',
        category='gemma',
    ),
    ModelTestInfo(
        id='google/gemma-1.1-2b-it',
        name='gemma-1.1-2b-it',
        family='gemma1p1',
        version='2b_it',
        config_id='gemma1p1_2b_it',
        category='gemma',
    ),
    ModelTestInfo(
        id='google/gemma-7b',
        name='gemma-7b',
        family='gemma',
        version='7b',
        config_id='gemma_7b',
        category='gemma',
    ),
    ModelTestInfo(
        id='google/gemma-7b-it',
        name='gemma-7b-it',
        family='gemma',
        version='7b_it',
        config_id='gemma_7b_it',
        category='gemma',
    ),
    ModelTestInfo(
        id='google/gemma-1.1-7b-it',
        name='gemma-1.1-7b-it',
        family='gemma1p1',
        version='7b_it',
        config_id='gemma1p1_7b_it',
        category='gemma',
    ),
    ModelTestInfo(
        id='google/gemma-2-2b',
        name='gemma-2-2b',
        family='gemma2',
        version='2b',
        config_id='gemma2_2b',
        category='gemma',
    ),
    ModelTestInfo(
        id='google/gemma-2-2b-it',
        name='gemma-2-2b-it',
        family='gemma2',
        version='2b_it',
        config_id='gemma2_2b_it',
        category='gemma',
    ),
    ModelTestInfo(
        id='google/gemma-2-9b',
        name='gemma-2-9b',
        family='gemma2',
        version='9b',
        config_id='gemma2_9b',
        category='gemma',
    ),
    ModelTestInfo(
        id='google/gemma-2-9b-it',
        name='gemma-2-9b-it',
        family='gemma2',
        version='9b_it',
        config_id='gemma2_9b_it',
        category='gemma',
    ),
    ModelTestInfo(
        id='google/gemma-3-270m',
        name='gemma-3-270m',
        family='gemma3',
        version='270m',
        config_id='gemma3_270m',
        category='gemma3',
    ),
    ModelTestInfo(
        id='google/gemma-3-270m-it',
        name='gemma-3-270m-it',
        family='gemma3',
        version='270m_it',
        config_id='gemma3_270m_it',
        category='gemma3',
    ),
    ModelTestInfo(
        id='google/gemma-3-1b',  #  Only superstring of the id exists on HF.
        name='gemma-3-1b',
        family='gemma3',
        version='1b',
        config_id='gemma3_1b',
        category='gemma3',
    ),
    ModelTestInfo(
        id='google/gemma-3-1b-it',
        name='gemma-3-1b-it',
        family='gemma3',
        version='1b_it',
        config_id='gemma3_1b_it',
        category='gemma3',
    ),
    ModelTestInfo(
        id='google/gemma-3-4b',  #  Only superstring of the id exists on HF.
        name='gemma-3-4b',
        family='gemma3',
        version='4b',
        config_id='gemma3_4b',
        category='gemma3',
    ),
    ModelTestInfo(
        id='google/gemma-3-4b-it',
        name='gemma-3-4b-it',
        family='gemma3',
        version='4b_it',
        config_id='gemma3_4b_it',
        category='gemma3',
    ),
    ModelTestInfo(
        id='google/gemma-3-12b',  #  Only superstring of the id exists on HF.
        name='gemma-3-12b',
        family='gemma3',
        version='12b',
        config_id='gemma3_12b',
        category='gemma3',
    ),
    ModelTestInfo(
        id='google/gemma-3-12b-it',
        name='gemma-3-12b-it',
        family='gemma3',
        version='12b_it',
        config_id='gemma3_12b_it',
        category='gemma3',
    ),
    ModelTestInfo(
        id='google/gemma-3-27b',  #  Only superstring of the id exists on HF.
        name='gemma-3-27b',
        family='gemma3',
        version='27b',
        config_id='gemma3_27b',
        category='gemma3',
    ),
    ModelTestInfo(
        id='google/gemma-3-27b-it',
        name='gemma-3-27b-it',
        family='gemma3',
        version='27b_it',
        config_id='gemma3_27b_it',
        category='gemma3',
    ),
    ModelTestInfo(
        id='meta-llama/Llama-3-70B',
        name='llama-3-70b',
        family='llama3',
        version='70b',
        config_id='llama3_70b',
        category='llama3',
    ),
    ModelTestInfo(
        id='meta-llama/Llama-3.1-405B',
        name='llama-3.1-405b',
        family='llama3p1',
        version='405b',
        config_id='llama3p1_405b',
        category='llama3',
    ),
    ModelTestInfo(
        id='meta-llama/Llama-3.1-8B',
        name='llama-3.1-8b',
        family='llama3p1',
        version='8b',
        config_id='llama3p1_8b',
        category='llama3',
    ),
    ModelTestInfo(
        id='meta-llama/Llama-3.2-1B',
        name='llama-3.2-1b',
        family='llama3p2',
        version='1b',
        config_id='llama3p2_1b',
        category='llama3',
    ),
    ModelTestInfo(
        id='meta-llama/Llama-3.2-3B',
        name='llama-3.2-3b',
        family='llama3p2',
        version='3b',
        config_id='llama3p2_3b',
        category='llama3',
    ),
    ModelTestInfo(
        id='Qwen/Qwen2.5-0.5B',
        name='qwen2.5-0.5b',
        family='qwen2p5',
        version='0p5b',
        config_id='qwen2p5_0p5b',
        category='qwen2',
    ),
    ModelTestInfo(
        id='Qwen/Qwen2.5-1.5B',
        name='qwen2.5-1.5b',
        family='qwen2p5',
        version='1p5b',
        config_id='qwen2p5_1p5b',
        category='qwen2',
    ),
    ModelTestInfo(
        id='Qwen/Qwen2.5-3B',
        name='qwen2.5-3b',
        family='qwen2p5',
        version='3b',
        config_id='qwen2p5_3b',
        category='qwen2',
    ),
    ModelTestInfo(
        id='Qwen/Qwen2.5-7B',
        name='qwen2.5-7b',
        family='qwen2p5',
        version='7b',
        config_id='qwen2p5_7b',
        category='qwen2',
    ),
    ModelTestInfo(
        id='Qwen/Qwen2.5-Math-1.5B',
        name='qwen2.5-math-1.5b',
        family='qwen2p5',
        version='math_1p5b',
        config_id='qwen2p5_math_1p5b',
        category='qwen2',
    ),
    ModelTestInfo(
        id='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
        name='deepseek-r1-distill-qwen-1.5b',
        family='deepseek_r1_distill_qwen',
        version='1p5b',
        config_id='deepseek_r1_distill_qwen_1p5b',
        category='qwen2',
    ),
    ModelTestInfo(
        id='Qwen/Qwen3-0.6B',
        name='qwen3-0.6b',
        family='qwen3',
        version='0p6b',
        config_id='qwen3_0p6b',
        category='qwen3',
    ),
    ModelTestInfo(
        id='Qwen/Qwen3-1.7B',
        name='qwen3-1.7b',
        family='qwen3',
        version='1p7b',
        config_id='qwen3_1p7b',
        category='qwen3',
    ),
    ModelTestInfo(
        id='Qwen/Qwen3-4B',
        name='qwen3-4b',
        family='qwen3',
        version='4b',
        config_id='qwen3_4b',
        category='qwen3',
    ),
    ModelTestInfo(
        id='Qwen/Qwen3-4B-Instruct-2507',
        name='qwen3-4b-instruct-2507',
        family='qwen3',
        version='4b_instruct_2507',
        config_id='qwen3_4b_instruct_2507',
        category='qwen3',
    ),
    ModelTestInfo(
        id='Qwen/Qwen3-4B-Thinking-2507',
        name='qwen3-4b-thinking-2507',
        family='qwen3',
        version='4b_thinking_2507',
        config_id='qwen3_4b_thinking_2507',
        category='qwen3',
    ),
    ModelTestInfo(
        id='Qwen/Qwen3-8B',
        name='qwen3-8b',
        family='qwen3',
        version='8b',
        config_id='qwen3_8b',
        category='qwen3',
    ),
    ModelTestInfo(
        id='Qwen/Qwen3-14B',
        name='qwen3-14b',
        family='qwen3',
        version='14b',
        config_id='qwen3_14b',
        category='qwen3',
    ),
    ModelTestInfo(
        id='Qwen/Qwen3-30B-A3B',
        name='qwen3-30b-a3b',
        family='qwen3',
        version='30b_a3b',
        config_id='qwen3_30b_a3b',
        category='qwen3',
    ),
)

_ALL_MODEL_MODULES = [
    gemma_model,
    gemma3_model,
    llama3_model,
    qwen2_model,
    qwen3_model,
]


def _validate_full_model_coverage():
  config_ids = []
  all_model_config_ids = {k.config_id for k in _TEST_MODEL_INFOS}
  # Check that all model configs in ModelConfig class are in _MODEL_INFOS.
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
        if name not in all_model_config_ids:
          raise ValueError(
              f'Model id {name} not found in _MODEL_INFOS. Make sure the'
              ' model is added to the map for full test coverage.'
          )
        config_ids.append(name)

  # Check each item in _MODEL_INFOS maps to a valid config id.This is to
  # prevent deprecated models from lingering in the map.
  for model_info in _TEST_MODEL_INFOS:
    if model_info.config_id not in config_ids:
      raise ValueError(
          f'Model name {model_info.config_id} not found in config_ids'
          f' {config_ids}. Seems to be an oboslete/deprecated model. Remove'
          ' from _MODEL_INFOS.'
      )


def _get_test_cases_for_get_model_config_id():
  test_cases = []
  _validate_full_model_coverage()
  for model_info in _TEST_MODEL_INFOS:
    test_cases.append({
        'testcase_name': model_info.config_id,
        'model_name': model_info.name,
        'expected_config_id': model_info.config_id,
    })
  return test_cases


def _get_test_cases_for_get_model_family_and_version():
  test_cases = []
  _validate_full_model_coverage()
  for model_info in _TEST_MODEL_INFOS:
    test_cases.append({
        'testcase_name': model_info.config_id,
        'model_name': model_info.name,
        'expected_family': model_info.family,
        'expected_version': model_info.version,
    })
  return test_cases


def _get_test_cases_for_get_model_config_category():
  test_cases_dict = {}
  _validate_full_model_coverage()
  for model_info in _TEST_MODEL_INFOS:
    if model_info.family not in test_cases_dict:
      test_cases_dict[model_info.family] = {
          'testcase_name': model_info.family,
          'model_name': model_info.name,
          'expected_category': model_info.category,
      }
  return list(test_cases_dict.values())


def _get_test_cases_for_get_model_name_from_model_id():
  test_cases = []
  _validate_full_model_coverage()
  for model_info in _TEST_MODEL_INFOS:
    test_cases.append({
        'testcase_name': model_info.config_id,
        'model_id': model_info.id,
        'expected_name': model_info.name,
    })
  return test_cases


class TestNaming(parameterized.TestCase):

  @parameterized.named_parameters(
      _get_test_cases_for_get_model_name_from_model_id()
  )
  def test_get_model_name_from_model_id(self, model_id, expected_name):
    self.assertEqual(
        naming.get_model_name_from_model_id(model_id),
        expected_name,
    )

  def test_get_model_name_from_model_id_invalid_fails(self):
    with self.assertRaisesRegex(ValueError, 'Invalid model ID format'):
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

  def test_get_model_family_and_version_invalid_fails(self):
    with self.assertRaisesRegex(
        ValueError, 'Could not determine model family for: foobar.'
    ):
      naming.get_model_family_and_version('foobar')

  def test_get_model_family_and_version_invalid_version_fails(self):
    with self.assertRaisesRegex(ValueError, 'Invalid model version format'):
      naming.get_model_family_and_version('gemma-@b')

  def test_split(self):
    self.assertEqual(naming.split('gemma-7b'), ('gemma-', '7b'))
    self.assertEqual(naming.split('gemma-1.1-7b'), ('gemma-1.1-', '7b'))

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
