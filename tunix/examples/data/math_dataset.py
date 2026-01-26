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

import logging
import os

import grain
import tensorflow_datasets as tfds
# For OSS usage
import tensorflow_datasets.text.gsm8k

reasoning_start = "<reasoning>"
reasoning_end = "</reasoning>"
solution_start = "<answer>"
solution_end = "</answer>"


SYSTEM_PROMPT = f"""You are given a problem. Think about the problem and \
provide your reasoning. Place it between {reasoning_start} and \
{reasoning_end}. Then, provide the final answer (i.e., just one numerical \
value) between {solution_start} and {solution_end}."""

TEMPLATE = """<start_of_turn>user
{system_prompt}

{question}<end_of_turn>
<start_of_turn>model"""


def extract_hash_answer(text: str) -> str | None:
  if "####" not in text:
    return None
  return text.split("####")[1].strip().replace(",", "")


# TODO(noghabi): Move these common dataset functions to a separate module.
def apply_template(
    dataset,
    tokenizer,
    tokenize=False,
    add_generation_prompt=True,
):
  """Applies chat template with tokenizer to dataset."""

  _apply_template = None
  if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
    logging.info("Applying chat template with tokenizer to dataset")
    _apply_template = lambda question: tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        tokenize=tokenize,
        add_generation_prompt=add_generation_prompt,
    )
  else:
    logging.info("Applying custom template to dataset")
    _apply_template = lambda question: TEMPLATE.format(
        system_prompt=SYSTEM_PROMPT, question=question
    )

  def _process_element(x):
    item = dict(x)
    for key, value in item.items():
      if isinstance(value, bytes):
        item[key] = value.decode("utf-8")

    return {
        "prompts": _apply_template(item["question"]),
        "question": item["question"],
        "answer": extract_hash_answer(item["answer"]),
    }

  return dataset.map(_process_element)


def get_tfds_dataset(
    dataset_name: str,
    data_dir: str | None,
    download: bool,
    split: str,
    shuffle_seed: int = 42,
) -> grain.MapDataset:
  """Get dataset from tfds.

  Args:
    dataset_name: The name of the dataset in tfds.
    data_dir: The directory to store the downloaded dataset.
    download: the download flag when using TFDS datasets.
    split: The dataset split to use (e.g., "train", "validation").
    shuffle_seed: The seed to use for shuffling the tfds dataset.

  Returns:
    A grain.MapDataset containing the processed dataset.
  """
  if data_dir and not os.path.exists(data_dir):
    os.makedirs(data_dir)

  data = tfds.data_source(
      dataset_name,
      split=split,
      data_dir=data_dir,
      builder_kwargs={"file_format": tfds.core.FileFormat.ARRAY_RECORD},
      download=download,
  )

  dataset = grain.MapDataset.source(data).shuffle(seed=shuffle_seed)
  return dataset


def create_dataset(
    data_source: str,
    dataset: str,
    tokenizer=None,
    tfds_download: bool = True,
    split: str = "train",
):
  """Creates a dataset based on the given name.

  Args:
    data_source: The source of dataset. The currently supported options are
      'tfds' (load from tensorflow_datasets) and 'local' (load local from
      parquet file).
    dataset: The name of the dataset to create. For 'tfds' data_source, the
      supported options are ['gsm8k']. For 'local' data_source, this is the path
      to a parquet file or directory.
    tokenizer: The tokenizer to use for processing prompts. If no tokenizer is
      provided, the fixed template is used.
    tfds_download: the download flag when using TFDS datasets. If false, the
      data_dir used will be set to `None` and chosen by default by tfds.
    split: The dataset split to use (e.g., "train", "test").

  Returns:
    A batched grain.MapDataset or grain.experimental.ParquetIterDataset.

  Raises:
    ValueError: If the dataset is not supported.
  """
  # parquet dataset
  if data_source == "local" and dataset.endswith(".parquet"):
    ds = grain.experimental.ParquetIterDataset(dataset)
  # tfds dataset
  elif data_source == "tfds" and dataset in ["gsm8k"]:
    data_dir = os.path.join("./data", split) if tfds_download else None
    ds = get_tfds_dataset(
        dataset_name=dataset,
        data_dir=data_dir,
        download=tfds_download,
        split=split,
    )
  else:
    raise ValueError(
        f"Unsupported combination of dataset='{dataset}' and"
        f" data_source='{data_source}'"
    )

  # Apply template
  ds = apply_template(ds, tokenizer, tokenize=False)

  return ds
