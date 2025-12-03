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

model_id:
The full model name identifier, as it appears on huggingface, including the
parent directory. E.g., meta-llama/Llama-3.1-8B or Qwen/Qwen2.5-0.5B . This
model ID is case sensitive and should match exactly with HF. This is the path
that would be used to download the model from HF.

model_name:
The unique full name identifier of the model. This should be the full name and
should match exactlywith the model name used in Hugging Face. e.g.,
"gemma-2b","llama3.1-8b", "qwen2.5-0.5b". The model name is all lowercase and
typically formatted as <model-family>-<model-version>

model_family:
The overall model family, e.g., "gemma", "gemma2", or "qwen2.5". Typically the
first part of the model name. Internally, we use the standardized version of the
model family, e.g., "qwen2.5" would be standardized to "qwen2_5".

model_version:
The specific version of this model family. This would be the second portion of
the model name. It usually includes the size information, but also can have
other information, e.g., `it` representing instruction tuned. Internally, the
model version is standardized by lowercasing, replacing hyphens and dots with
underscores, and removing any leading underscore. e.g., "2b-it" would become
"2b_it".

model_config_category:
The model config category is the python class name of the ModelConfig class.
e.g.,
both gemma and gemma2 models  have the category "gemma" with the ModelConfig
class being defined under gemma/model.py."

model_config_id:
The model config ID is the standardized version of the model family and model
version. It is used as the ID of the ModelConfig class. e.g., "gemma_2b_it" or
"qwen2p5_0p5b".
"""
# TODO(b/451662153): add README on naming conventions and update naming
# descriptions in //third_party/py/tunix/cli/base_config.yaml.

import dataclasses

@dataclasses.dataclass(frozen=True)
class _ModelFamilyInfo:
  """Configuration for handling model family mappings."""

  family: str  # standardized model family, as used in id in ModelConfig
  config_category: str  # category in the path to the ModelConfig class


# Mapping of all model families from the hugging face model id to the internal
# model_family and config_category. Key is the prefix of the hugging face model
# id and value is the internal model family and config_category.
_MODEL_FAMILY_INFO_MAPPING = {
    'gemma': _ModelFamilyInfo(family='gemma', config_category='gemma'),
    'gemma1.1': _ModelFamilyInfo(family='gemma1p1', config_category='gemma'),
    'gemma2': _ModelFamilyInfo(family='gemma2', config_category='gemma'),
    # Support both gemma3 and gemma-3 as model prefixes.
    'gemma3': _ModelFamilyInfo(family='gemma3', config_category='gemma3'),
    'gemma-3': _ModelFamilyInfo(family='gemma3', config_category='gemma3'),
    'llama3': _ModelFamilyInfo(family='llama3', config_category='llama3'),
    'llama3.1': _ModelFamilyInfo(family='llama3p1', config_category='llama3'),
    'llama3.2': _ModelFamilyInfo(family='llama3p2', config_category='llama3'),
    'qwen2.5': _ModelFamilyInfo(family='qwen2p5', config_category='qwen2'),
    'qwen3': _ModelFamilyInfo(family='qwen3', config_category='qwen3'),
    'deepseek-r1-distill-qwen': _ModelFamilyInfo(
        family='deepseek_r1_distill_qwen', config_category='qwen2'
    ),
}


def split(model_name: str) -> tuple[str, str]:
  """Splits model name into model family and model version.

  Find the longest matching prefix of the model name in the
  _MODEL_FAMILY_INFO_MAPPING. Returns the remaining string as the model version,
  stripping leading hyphens.

  Args:
    model_name: The model name, e.g., llama3.1-8b.

  Returns:
    A tuple containing the un-standardized model_family andmodel_version.
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
    return model_id.split('/')[-1].lower()
  else:
    raise ValueError(
        f'Invalid model ID format: {model_id}. Model ID should be in the format'
        ' of <parent-dir>/<model-name>'
    )
