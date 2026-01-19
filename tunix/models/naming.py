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


"""Model naming utilities.

This module provides utility functions to parse and handle model names and
convert them to internal model naming structures.
"""


import dataclasses
import immutabledict


@dataclasses.dataclass(frozen=True)
class ModelNaming:
  """Model naming information.

  Attributes:
    model_id: The full model name identifier (case sensitive), as it appears on
      Huggingface, or Kaggle, including the parent directory. E.g.,
      "meta-llama/Llama-3.1-8B" or "google/gemma-2/flax/gemma2-2b-it".
    model_name: The unique full name identifier of the model. This should be the
      full name and should match exactly with the model name used in Hugging
      Face. e.g., "gemma-2b","llama-3.1-8b" or Kaggle, e.g., "gemma2-2b-it". The
      model name is all lowercase and typically formatted as
      <model-family>-<model-version>.
    model_family: The standardized model family, e.g., "gemma", "gemma2", or
      "qwen2p5". The model is standerdardized by removing unnecessary '-', e.g.,
      "gemma-2" --> "gemma2", replacing '-' with '_',  and replacing '.' with
      'p',  e.g., "qwen2.5" would be standardized to "qwen2p5".
    model_version: The standardized version of this model family. This would be
      the second portion of the model name. It includes additional information
      such as size, whether it is instruction tuned ("it'), etc. The model
      version is standardized by lowercasing, replacing '-' with '_', and
      replacing '.' with 'p'. e.g., "2b-it" would be standardized to "2b_it".
    model_config_category: The model config category is the python class name of
      the ModelConfig class. e.g., both gemma and gemma2 models have the
      category "gemma" with the ModelConfig class being defined under
      gemma/model.py.
    model_config_id: The standardized config id composed of the model family and
      model version, used in the ModelConfig class. e.g., "gemma_2b_it" or
      "qwen2p5_0p5b".
  """

  model_id: str | None = None
  model_name: str | None = None
  model_family: str = dataclasses.field(init=False)
  model_version: str = dataclasses.field(init=False)
  model_config_category: str = dataclasses.field(init=False)
  model_config_id: str = dataclasses.field(init=False)

  def __post_init__(self):
    if self.model_id:
      # We infer model_name from model_id.
      model_name = get_model_name_from_model_id(self.model_id)
      if self.model_name and self.model_name != model_name:
        raise ValueError(
            'model_name set in ModelNaming and one inferred from model_id do'
            f' not match. model_name: {self.model_name} and model_id:'
            f' {self.model_id}, model_name inferred from model_id:'
            f' {model_name}.'
        )
    else:
      # If no model_id is provided, we use the model_name.
      model_name = self.model_name
    if not model_name:
      raise ValueError('Either model_name or model_id must be provided.')
    object.__setattr__(self, 'model_name', model_name)

    family, version = get_model_family_and_version(model_name)
    object.__setattr__(self, 'model_family', family)
    object.__setattr__(self, 'model_version', version)

    object.__setattr__(
        self, 'model_config_category', get_model_config_category(model_name)
    )
    object.__setattr__(self, 'model_config_id', get_model_config_id(model_name))


@dataclasses.dataclass(frozen=True)
class _ModelFamilyInfo:
  """Configuration for handling model family mappings."""

  family: str  # standardized model family, as used in id in ModelConfig
  config_category: str  # category in the path to the ModelConfig class


# Mapping of all model families from the hugging face model id to the internal
# model_family and config_category. Key is the prefix of the hugging face model
# id and value is the internal model family and config_category.
_MODEL_FAMILY_INFO_MAPPING = immutabledict.immutabledict({
    'gemma-': _ModelFamilyInfo(family='gemma', config_category='gemma'),
    'gemma1.1-': _ModelFamilyInfo(family='gemma1p1', config_category='gemma'),
    'gemma-1.1-': _ModelFamilyInfo(family='gemma1p1', config_category='gemma'),
    'gemma2-': _ModelFamilyInfo(family='gemma2', config_category='gemma'),
    'gemma-2-': _ModelFamilyInfo(family='gemma2', config_category='gemma'),
    'gemma3-': _ModelFamilyInfo(family='gemma3', config_category='gemma3'),
    'gemma-3-': _ModelFamilyInfo(family='gemma3', config_category='gemma3'),
    'llama3-': _ModelFamilyInfo(family='llama3', config_category='llama3'),
    'llama-3-': _ModelFamilyInfo(family='llama3', config_category='llama3'),
    'llama3.1-': _ModelFamilyInfo(family='llama3p1', config_category='llama3'),
    'llama-3.1-': _ModelFamilyInfo(family='llama3p1', config_category='llama3'),
    'llama3.2-': _ModelFamilyInfo(family='llama3p2', config_category='llama3'),
    'llama-3.2-': _ModelFamilyInfo(family='llama3p2', config_category='llama3'),
    'qwen2.5-': _ModelFamilyInfo(family='qwen2p5', config_category='qwen2'),
    'qwen3-': _ModelFamilyInfo(family='qwen3', config_category='qwen3'),
    'deepseek-r1-distill-qwen-': _ModelFamilyInfo(
        family='deepseek_r1_distill_qwen', config_category='qwen2'
    ),
})


def split(model_name: str) -> tuple[str, str]:
  """Splits model name into model family and model version.

  Find the longest matching prefix of the model name in the
  _MODEL_FAMILY_INFO_MAPPING. Returns the remaining string as the model version,
  stripping leading hyphens.

  Args:
    model_name: The model name, e.g., llama3.1-8b.

  Returns:
    A tuple containing the un-standardized model_family and model_version.
  """
  model_name = model_name.lower()
  matched_family = ''
  for family in _MODEL_FAMILY_INFO_MAPPING:
    if model_name.startswith(family) and len(family) > len(matched_family):
      matched_family = family
  if matched_family:
    return matched_family, model_name[len(matched_family) :].lstrip('-')
  else:
    raise ValueError(
        f'Could not determine model family for: {model_name}. Not one of the'
        ' known families:'
        f' {list(_MODEL_FAMILY_INFO_MAPPING.keys())}'
    )


def _standardize_model_version(raw_model_version: str) -> str:
  """Standardizes model version name.

  Operations include:
  - Lowercase
  - Replace hyphens with underscores
  - Replace dots with underscores
  - Validate the model version starts with an alphanumeric character.

  Args:
    raw_model_version: The raw model version string.

  Returns:
    The standardized model version name.
  """
  if not raw_model_version:
    return ''
  model_version = raw_model_version.lower().replace('-', '_').replace('.', 'p')

  # Validate the model version starts with an alphanumeric character.
  if len(model_version) > 1 and not model_version[0].isalnum():
    raise ValueError(
        'Invalid model version format. Expected alphanumeric starting'
        f' character, found: {model_version}'
    )
  return model_version


def get_model_family_and_version(model_name: str) -> tuple[str, str]:
  """Splits model name into internal, standardized model family and model version."""
  raw_model_family, raw_model_version = split(model_name)
  model_family = _MODEL_FAMILY_INFO_MAPPING[raw_model_family].family
  model_version = _standardize_model_version(raw_model_version)
  return model_family, model_version


def get_model_config_category(model_name: str) -> str:
  """Returns the model config category from the model family."""
  raw_model_family, _ = split(model_name)
  return _MODEL_FAMILY_INFO_MAPPING[raw_model_family].config_category


def get_model_config_id(model_name: str) -> str:
  """Returns the model config ID from the model name."""
  model_family, model_version = get_model_family_and_version(model_name)
  config_id = f'{model_family}_{model_version}'
  config_id = config_id.replace('.', 'p').replace('-', '_')
  return config_id


def get_model_name_from_model_id(model_id: str) -> str:
  """Extracts model name from model ID by taking the last part of path.

  Args:
    model_id: The full model name identifier, as it appears on huggingface,
      including the parent directory. E.g., meta-llama/Llama-3.1-8B.

  Returns:
    The model_name string.
  """
  if '/' in model_id:
    model_name = model_id.split('/')[-1].lower()
    if not model_name:
      raise ValueError(
          f'Invalid model ID format: {model_id!r}. Model name cannot be empty.'
      )
    if model_name.startswith('meta-llama-'):
      return model_name.replace('meta-llama-', 'llama-', 1)
    return model_name
  else:
    raise ValueError(
        f'Invalid model ID format: {model_id!r}. Model ID should be in the'
        ' format of <parent-dir>/<model-name>'
    )
