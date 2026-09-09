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

"""GSM8K Dataset Loader for Tunix Agentic RL.

Provides dataset pipeline utilities to load, clean, and format GSM8K problems
for consumption by GSM8KEnv and the Tunix GRPO learner.

Supports:
  1. Hugging Face Datasets ('openai/gsm8k').
  2. TensorFlow Datasets ('gsm8k').
  3. Local Parquet / JSONL.
  4. In-memory demo fallback for smoke-testing and debugging.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Iterator

from examples.math_gsm8k.env import (
    DEFAULT_PROMPT_TEMPLATE,
    extract_hash_answer,
    normalize_answer,
)

# Canonical 4-problem smoke-test fallback
SMOKE_TEST_PROBLEMS = (
    (
        "Natalia sold clips to 48 friends in April, and then she sold half as "
        "many clips in May. How many clips did Natalia sell altogether in "
        "April and May?",
        "72",
    ),
    (
        "Weng earns $12 an hour for babysitting. Yesterday, she babysat for 3 "
        "hours. How much did she earn?",
        "36",
    ),
    (
        "A robe takes 2 bolts of blue fiber and half that much white fiber. "
        "How many bolts of fiber does it take?",
        "3",
    ),
    (
        "Betty is saving money for a wallet which costs $100. She has $15 "
        "saved. How much more does she need?",
        "85",
    ),
)


def format_gsm8k_example(
    question: str,
    answer: str,
    *,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    task_id: str | None = None,
) -> dict[str, Any]:
  """Formats a question-answer pair into an environment task dictionary."""
  clean_q = str(question).strip()
  gold = normalize_answer(extract_hash_answer(str(answer)))
  prompt = prompt_template.format(question=clean_q)
  return {
      "question": clean_q,
      "answer": str(answer),
      "gold_answer": gold,
      "prompts": prompt,
      "prompt": prompt,
      "task_id": task_id or clean_q[:32],
  }


def create_smoke_test_dataset(
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
) -> list[dict[str, Any]]:
  """Returns an in-memory list of canonical smoke test GSM8K task dictionaries."""
  return [
      format_gsm8k_example(
          q,
          f"#### {a}",
          prompt_template=prompt_template,
          task_id=f"demo_{i}",
      )
      for i, (q, a) in enumerate(SMOKE_TEST_PROBLEMS)
  ]


def load_gsm8k_huggingface(
    split: str = "train",
    *,
    shuffle_seed: int | None = 42,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
) -> Any:
  """Loads GSM8K from Hugging Face Datasets ('openai/gsm8k')."""
  import datasets as hf_datasets  # pylint: disable=g-import-not-at-top

  ds = hf_datasets.load_dataset("openai/gsm8k", "main", split=split)
  if shuffle_seed is not None:
    ds = ds.shuffle(seed=shuffle_seed)

  try:
    import grain  # pylint: disable=g-import-not-at-top

    grain_ds = grain.MapDataset.source(ds)
    return grain_ds.map(
        lambda x: format_gsm8k_example(
            x["question"], x["answer"], prompt_template=prompt_template
        )
    )
  except ImportError:
    # Generator fallback
    return (
        format_gsm8k_example(
            x["question"], x["answer"], prompt_template=prompt_template
        )
        for x in ds
    )


def create_dataset(
    split: str = "train",
    *,
    data_source: str = "huggingface",
    seed: int = 42,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    **kwargs,
) -> Any:
  """Creates a GSM8K dataset for RL training or evaluation.

  Args:
    split: Dataset split ('train' or 'test').
    data_source: Source selector ('huggingface', 'smoke_test', 'tfds').
    seed: Random seed for shuffling.
    prompt_template: Prompt template to wrap the question with.
    **kwargs: Additional parameters passed to dataset loaders.

  Returns:
    A Grain MapDataset or iterable of formatted task dictionaries.
  """
  if data_source == "smoke_test" or data_source == "demo":
    return create_smoke_test_dataset(prompt_template=prompt_template)
  elif data_source == "huggingface":
    return load_gsm8k_huggingface(
        split=split,
        shuffle_seed=seed,
        prompt_template=prompt_template,
    )
  elif data_source == "tfds":
    from tunix.examples.data import math_dataset  # pylint: disable=g-import-not-at-top
    return math_dataset.create_dataset(
        data_source="tfds",
        dataset="gsm8k",
        split=split,
        **kwargs,
    )
  else:
    raise ValueError(f"Unsupported GSM8K data_source: {data_source!r}")
