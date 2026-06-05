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

import inspect
from absl.testing import absltest
from tunix.models import registry
from tunix.models.gemma import model as gemma_model
from tunix.models.gemma3 import model as gemma3_model
from tunix.models.gemma4 import model as gemma4_model
from tunix.models.llama3 import model as llama3_model
from tunix.models.qwen2 import model as qwen2_model
from tunix.models.qwen3 import model as qwen3_model

_ALL_MODEL_MODULES = [
    gemma_model,
    gemma3_model,
    gemma4_model,
    llama3_model,
    qwen2_model,
    qwen3_model,
]


def _get_all_model_config_ids_from_modules() -> set[str]:
  """Extracts all valid model config IDs from all ModelConfig classes."""
  config_ids = set()
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
        config_ids.add(name)
  return config_ids


class RegistryTest(absltest.TestCase):

  def test_all_model_configs_are_in_catalog(self):
    """Check that all model configs in ModelConfig classes are in MODEL_CATALOG."""
    catalog_config_ids = {k.model_config_id for k in registry.MODEL_CATALOG}
    module_config_ids = _get_all_model_config_ids_from_modules()
    for config_id in module_config_ids:
      self.assertIn(
          config_id,
          catalog_config_ids,
          f'Model id {config_id} not found in MODEL_CATALOG. Make sure the'
          ' model is added to the map for full test coverage.',
      )

  def test_no_deprecated_models_in_catalog(self):
    """Check each item in MODEL_CATALOG maps to a valid config id."""
    module_config_ids = _get_all_model_config_ids_from_modules()
    for model_info in registry.MODEL_CATALOG:
      self.assertIn(
          model_info.model_config_id,
          module_config_ids,
          f'Model name {model_info.model_config_id} not found in module'
          ' config_ids. Seems to be an obsolete/deprecated model. Remove from'
          ' MODEL_CATALOG.',
      )


if __name__ == '__main__':
  absltest.main()
