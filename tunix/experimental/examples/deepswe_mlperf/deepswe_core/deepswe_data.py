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

"""Dataset loading for the DeepSWE agentic GRPO recipe.

Returns a raw grain.MapDataset ready for post_init_dataset().
The module-level ``batch_fn`` is picked up automatically by
AgenticGrpoPipeline as the custom_batch_fn for post_init_dataset.
"""

import json
import os

import grain
import numpy as np


def create_dataset(
    dataset_name: str = "R2E-Gym/R2E-Gym-V1",
    dataset_split: str = "train",
    cache_dir: str | None = None,
    shuffle: bool = True,
    seed: int = 42,
) -> grain.MapDataset:
  """Load the R2E-Gym dataset and return a raw grain.MapDataset.

  Args:
    dataset_name: HuggingFace dataset identifier.
    dataset_split: Which split to load.
    cache_dir: Local directory for dataset caching. Defaults to
      <cwd>/dataset_cache.
    shuffle: Whether to shuffle the dataset.
    seed: Random seed for shuffling.

  Returns:
    grain.MapDataset where each element is a raw R2E-Gym example dict with
    list fields JSON-serialised to strings.
  """
  from datasets import load_dataset  # pylint: disable=g-import-not-at-top

  if cache_dir is None:
    cache_dir = os.path.join(os.getcwd(), "dataset_cache")
  os.makedirs(cache_dir, exist_ok=True)

  dataset = load_dataset(
      dataset_name,
      split=dataset_split,
      cache_dir=cache_dir,
      trust_remote_code=True,
  )

  def _transform(entry):
    for k, v in entry.items():
      if isinstance(v, list):
        entry[k] = json.dumps(v)
    return entry

  dataset = dataset.map(_transform, keep_in_memory=True)

  if shuffle:
    dataset = dataset.shuffle(seed)

  return grain.MapDataset.source(dataset)


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
