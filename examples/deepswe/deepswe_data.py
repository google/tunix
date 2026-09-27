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

"""Canonical Qwen3-4B clean tasks for the DeepSWE agentic GRPO recipe.

Returns a raw grain.MapDataset ready for post_init_dataset().
The module-level ``batch_fn`` is picked up automatically by
AgenticGrpoPipeline as the custom_batch_fn for post_init_dataset.
"""

import os

import grain
import numpy as np


def create_dataset(
    dataset_name: str = "R2E-Gym/R2E-Gym-Subset",
    dataset_revision: str = "2e8108ff942f24fcb5686badfaf7f9a8808566d5",
    dataset_split: str = "train",
    dataset_path: str = "",
    gold_whitelist: str = "",
    cache_dir: str | None = None,
    shuffle: bool = True,
    seed: int = 42,
) -> grain.MapDataset:
  """Load the pinned R2E-Gym subset joined to the Q4 clean selector.

  Args:
    dataset_name: HuggingFace dataset identifier.
    dataset_revision: Pinned source revision.
    dataset_split: Which split to load.
    dataset_path: Optional local Hugging Face dataset.
    gold_whitelist: Optional clean selector; defaults to the canonical 1,012.
    cache_dir: Local directory for dataset caching. Defaults to
      <cwd>/dataset_cache.
    shuffle: Whether to shuffle the dataset.
    seed: Random seed for shuffling.

  Returns:
    Grain dataset of full task records in the distributed recipe's order.
  """
  return grain.MapDataset.source(load_clean_dataset(
      dataset_name=dataset_name,
      dataset_revision=dataset_revision,
      dataset_split=dataset_split,
      dataset_path=dataset_path,
      gold_whitelist=gold_whitelist,
      cache_dir=cache_dir,
      shuffle=shuffle,
      seed=seed,
  ))


def load_clean_dataset(
    dataset_name: str = "R2E-Gym/R2E-Gym-Subset",
    dataset_revision: str = "2e8108ff942f24fcb5686badfaf7f9a8808566d5",
    dataset_split: str = "train",
    dataset_path: str = "",
    gold_whitelist: str = "",
    cache_dir: str | None = None,
    shuffle: bool = True,
    seed: int = 42,
):
  """Use the distributed recipe's verified selector, join, and task order."""
  from tunix.experimental.examples.deepswe_dist import deepswe

  if cache_dir is None:
    cache_dir = os.path.join(os.getcwd(), "dataset_cache")
  os.makedirs(cache_dir, exist_ok=True)
  return deepswe.load_deepswe_dataset(
      dataset_name=dataset_name,
      dataset_revision=dataset_revision,
      dataset_split=dataset_split,
      dataset_path=dataset_path,
      gold_whitelist=gold_whitelist or deepswe.DEFAULT_GOLD_WHITELIST,
      cache_dir=cache_dir,
      shuffle=shuffle,
      seed=seed,
  )


# R2E-Gym has heterogeneous field types that grain's default batching can't
# handle; this function is picked up automatically by AgenticGrpoPipeline.
_STR_KEYS = {
    "repo_name",
    "docker_image",
    "commit_hash",
    "parsed_commit_content",
    "execution_result_content",
}
_DICT_KEYS = {
    "modified_files",
    "relevant_files",
    "modified_entity_summaries",
}
_ARRAY_KEYS = {
    "num_non_test_files",
    "num_non_test_func_methods",
    "num_non_test_lines",
    "prompt",
    "problem_statement",
    "expected_output_json",
}


def batch_fn(elements: list[dict]) -> dict:
  """Batch a list of R2E-Gym examples into a dict of lists / arrays."""
  batched: dict = {}
  for key in elements[0].keys():
    if key in _STR_KEYS or key in _DICT_KEYS:
      batched[key] = [item[key] for item in elements]
    elif key in _ARRAY_KEYS:
      batched[key] = np.array([item[key] for item in elements])
    else:
      batched[key] = [item[key] for item in elements]
  return batched
