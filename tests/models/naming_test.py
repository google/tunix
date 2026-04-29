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

from absl.testing import absltest
from absl.testing import parameterized
import requests
import tenacity
from tunix.models import naming
from tunix.models import registry
from tunix.utils import env_utils


def _get_test_cases_for_get_model_config_id() -> list[dict[str, str]]:
  test_cases = []
  for model_info in registry.MODEL_CATALOG:
    test_cases.append({
        'testcase_name': model_info.model_config_id,
        'model_name': model_info.model_name,
        'expected_config_id': model_info.model_config_id,
    })
  return test_cases


def _get_test_cases_for_get_model_family_and_version() -> list[dict[str, str]]:
  test_cases = []
  for model_info in registry.MODEL_CATALOG:
    test_cases.append({
        'testcase_name': model_info.model_config_id,
        'model_name': model_info.model_name,
        'expected_family': model_info.model_family,
        'expected_version': model_info.model_version,
    })
  return test_cases


def _get_test_cases_for_get_model_config_category() -> list[dict[str, str]]:
  test_cases_dict = {}
  for model_info in registry.MODEL_CATALOG:
    if model_info.model_family not in test_cases_dict:
      test_cases_dict[model_info.model_family] = {
          'testcase_name': model_info.model_family,
          'model_name': model_info.model_name,
          'expected_category': model_info.model_config_category,
      }
  return list(test_cases_dict.values())


def _get_test_cases_for_get_model_name_from_model_id() -> list[dict[str, str]]:
  test_cases = []
  for model_info in registry.MODEL_CATALOG:
    test_cases.append({
        'testcase_name': model_info.model_config_id,
        'model_id': model_info.model_id,
        'expected_name': model_info.model_name,
    })
  return test_cases


def _get_test_cases_for_model_id_exists() -> list[dict[str, str]]:
  return [
      {
          'testcase_name': model_info.model_config_id,
          'model_id': model_info.model_id,
      }
      for model_info in registry.MODEL_CATALOG
  ]


def _get_test_cases_for_auto_population_with_HF_model_id() -> (
    list[dict[str, str]]
):
  test_cases = []
  for model_info in registry.MODEL_CATALOG:
    test_cases.append({
        'testcase_name': model_info.model_config_id,
        'model_id': model_info.model_id,
        'expected_name': model_info.model_name,
        'expected_family': model_info.model_family,
        'expected_version': model_info.model_version,
        'expected_category': model_info.model_config_category,
        'expected_config_id': model_info.model_config_id,
    })
  return test_cases


def _get_test_cases_for_auto_population_with_config_id() -> (
    list[dict[str, str]]
):
  test_cases = []
  for model_info in registry.MODEL_CATALOG:
    test_cases.append({
        'testcase_name': model_info.model_config_id,
        'model_id': model_info.model_config_id,
        'expected_family': model_info.model_family,
        'expected_version': model_info.model_version,
        'expected_category': model_info.model_config_category,
        'expected_config_id': model_info.model_config_id,
    })
  return test_cases


class TestNaming(parameterized.TestCase):

  @parameterized.named_parameters(
      _get_test_cases_for_get_model_name_from_model_id()
  )
  def test_get_model_name_from_model_id(
      self, model_id: str, expected_name: str
  ):
    self.assertEqual(
        naming.get_model_name_from_model_id(model_id),
        expected_name,
    )

  def test_get_model_name_from_model_id_no_slash_succeeds(self):
    self.assertEqual(
        naming.get_model_name_from_model_id('Llama-3.1-8B'), 'llama-3.1-8b'
    )

  def test_get_model_name_from_model_id_config_id(self):
    self.assertEqual(
        naming.get_model_name_from_model_id('llama3p1_8b'), 'llama3p1_8b'
    )

  def test_get_model_name_from_model_id_nested_path(self):
    self.assertEqual(
        naming.get_model_name_from_model_id('google/gemma-2/flax/gemma2-2b-it'),
        'gemma2-2b-it',
    )

  def test_get_model_name_from_model_id_empty_model_name(self):
    with self.assertRaisesRegex(
        ValueError, 'Invalid model ID format: .*. Model name cannot be empty.'
    ):
      naming.get_model_name_from_model_id('google/')

  @tenacity.retry(
      stop=tenacity.stop_after_attempt(3),
      wait=tenacity.wait_exponential(multiplier=1, min=2, max=10),
      retry=(
          tenacity.retry_if_exception_type((
              requests.exceptions.ConnectionError,
              requests.exceptions.Timeout,
          ))
          | tenacity.retry_if_result(
              lambda response: response.status_code >= 500
          )
      ),
      reraise=True,
  )
  def _head_huggingface_model(self, model_id: str) -> requests.Response:
    return requests.head(
        f'https://huggingface.co/{model_id}',
        allow_redirects=True,
        timeout=10,
    )

  @parameterized.named_parameters(_get_test_cases_for_model_id_exists())
  def test_model_id_exists_on_huggingface(self, model_id: str):
    if env_utils.is_internal_env():
      self.skipTest('Skipping Hugging Face check in internal environment')

    response = self._head_huggingface_model(model_id)
    self.assertEqual(
        response.status_code,
        200,
        f'Model {model_id!r} not found on Hugging Face (status code:'
        f' {response.status_code}). Please ensure that the model config added'
        ' matches exaclty to a valid model id on Hugging Face.',
    )

  @parameterized.named_parameters(
      _get_test_cases_for_get_model_family_and_version()
  )
  def test_get_model_family_and_version(
      self, model_name: str, expected_family: str, expected_version: str
  ):
    self.assertEqual(
        naming.get_model_family_and_version(model_name),
        (expected_family, expected_version),
    )

  def test_get_model_family_and_version_invalid_fails(self):
    with self.assertRaisesRegex(
        ValueError, 'Could not determine model family for: foo-bar.'
    ):
      naming.get_model_family_and_version('foo-bar')

  def test_get_model_family_and_version_invalid_format_fails(self):
    with self.assertRaisesRegex(
        ValueError,
        'Invalid model ID format: .* Expected a Huggingface model ID or a'
        ' ConfigId.',
    ):
      naming.get_model_family_and_version('foobar')

  def test_get_model_family_and_version_invalid_version_fails(self):
    with self.assertRaisesRegex(ValueError, 'Invalid model version format'):
      naming.get_model_family_and_version('gemma-@b')

  def test_split(self):
    self.assertEqual(naming.split('gemma-7b'), ('gemma-', '7b'))
    self.assertEqual(naming.split('gemma-1.1-7b'), ('gemma-1.1-', '7b'))
    self.assertEqual(naming.split('gemma_7b'), ('gemma_', '7b'))
    self.assertEqual(naming.split('gemma1p1_7b'), ('gemma1p1_', '7b'))

  @parameterized.named_parameters(_get_test_cases_for_get_model_config_id())
  def test_get_model_config_id(self, model_name: str, expected_config_id: str):
    self.assertEqual(naming.get_model_config_id(model_name), expected_config_id)

  @parameterized.named_parameters(
      _get_test_cases_for_get_model_config_category()
  )
  def test_get_model_config_category(
      self, model_name: str, expected_category: str
  ):
    self.assertEqual(
        naming.get_model_config_category(model_name), expected_category
    )

  @parameterized.named_parameters(
      _get_test_cases_for_auto_population_with_HF_model_id()
  )
  def test_model_naming_auto_population_with_HF_model_id(
      self,
      *,
      model_id: str,
      expected_name: str,
      expected_family: str,
      expected_version: str,
      expected_category: str,
      expected_config_id: str,
  ):
    with self.subTest(name='Test Model naming creation with HFModelId'):
      naming_info = naming.ModelNaming(model_id=naming.HFModelId(model_id))
      self.assertEqual(naming_info.model_id, model_id)
      self.assertEqual(naming_info.model_name, expected_name)
      self.assertEqual(naming_info.model_family, expected_family)
      self.assertEqual(naming_info.model_version, expected_version)
      self.assertEqual(naming_info.model_config_category, expected_category)
      self.assertEqual(naming_info.model_config_id, expected_config_id)

    with self.subTest(name='Test Model id type detection'):
      self.assertTrue(naming._is_hf_model_id_type(naming_info.model_id))
      self.assertFalse(naming._is_config_id_type(naming_info.model_id))
      self.assertTrue(naming._is_hf_model_id_type(naming_info.model_name))
      self.assertFalse(naming._is_config_id_type(naming_info.model_name))

  @parameterized.named_parameters(
      _get_test_cases_for_auto_population_with_config_id()
  )
  def test_model_naming_auto_population_with_config_id_model_id(
      self,
      *,
      model_id: str,
      expected_family: str,
      expected_version: str,
      expected_category: str,
      expected_config_id: str,
  ):
    with self.subTest(name='Test Model naming creation with ConfigId'):
      naming_info = naming.ModelNaming(model_id=naming.ConfigId(model_id))
      self.assertEqual(naming_info.model_id, model_id)
      self.assertEqual(naming_info.model_name, model_id)
      self.assertEqual(naming_info.model_family, expected_family)
      self.assertEqual(naming_info.model_version, expected_version)
      self.assertEqual(naming_info.model_config_category, expected_category)
      self.assertEqual(naming_info.model_config_id, expected_config_id)

    with self.subTest(name='Test Model id type detection'):
      self.assertFalse(naming._is_hf_model_id_type(naming_info.model_id))
      self.assertTrue(naming._is_config_id_type(naming_info.model_id))
      self.assertFalse(naming._is_hf_model_id_type(naming_info.model_name))
      self.assertTrue(naming._is_config_id_type(naming_info.model_name))

  def test_model_naming_no_model_id(self):
    model_name = 'gemma-2b'
    naming_info = naming.ModelNaming(model_name=model_name)
    self.assertIsNone(naming_info.model_id)
    self.assertEqual(naming_info.model_name, 'gemma-2b')
    self.assertEqual(naming_info.model_family, 'gemma')
    self.assertEqual(naming_info.model_version, '2b')
    self.assertEqual(naming_info.model_config_category, 'gemma')
    self.assertEqual(naming_info.model_config_id, 'gemma_2b')

  def test_model_naming_missing_args(self):
    with self.assertRaisesRegex(
        ValueError, 'Either model_name or model_id must be provided'
    ):
      naming.ModelNaming()

  def test_model_naming_invalid_model_name(self):
    with self.assertRaisesRegex(
        ValueError, 'Could not determine model family for: invalid-model'
    ):
      naming.ModelNaming(model_name='invalid-model')

  def test_model_naming_mismatch(self):
    with self.assertRaisesRegex(
        ValueError,
        'model_name set in ModelNaming and one inferred from model_id do not'
        ' match',
    ):
      naming.ModelNaming(
          model_name='gemma-7b', model_id=naming.HFModelId('google/gemma-2b')
      )

  def test_model_naming_family_mismatch(self):
    with self.assertRaisesRegex(
        ValueError,
        'model_family mismatch:',
    ):
      naming.ModelNaming(model_name='gemma-2b', model_family='llama3')

  def test_model_naming_version_mismatch(self):
    with self.assertRaisesRegex(
        ValueError,
        'model_version mismatch:',
    ):
      naming.ModelNaming(model_name='gemma-2b', model_version='7b')

  def test_model_naming_category_mismatch(self):
    with self.assertRaisesRegex(
        ValueError,
        'model_config_category mismatch:',
    ):
      naming.ModelNaming(model_name='gemma-2b', model_config_category='llama3')

  def test_model_naming_config_id_mismatch(self):
    with self.assertRaisesRegex(
        ValueError,
        'model_config_id mismatch:',
    ):
      naming.ModelNaming(model_name='gemma-2b', model_config_id='gemma_7b')


if __name__ == '__main__':
  absltest.main()
