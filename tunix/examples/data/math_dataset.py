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
  return text.split("####")[1].strip()


def get_dataset_from_tfds(
    data_dir, split="train", tfds_download: bool = True
) -> grain.MapDataset:
  """Get dataset from tfds.

  Args:
    data_dir: The directory to store the downloaded dataset.
    split: The dataset split to use (e.g., "train", "validation").
    tfds_download: the download flag for tfds.

  Returns:
    A grain.MapDataset containing the processed dataset.
  """
  if data_dir and not os.path.exists(data_dir):
    os.makedirs(data_dir)

  data = tfds.data_source(
      "gsm8k",
      split=split,
      data_dir=data_dir,
      builder_kwargs={"file_format": tfds.core.FileFormat.ARRAY_RECORD},
      download=tfds_download,
  )

  dataset = (
      grain.MapDataset.source(data)
      .shuffle(seed=42)
      .map(
          lambda x: {
              # passed to model forward pass
              "prompts": TEMPLATE.format(
                  system_prompt=SYSTEM_PROMPT,
                  question=x["question"].decode("utf-8"),
              ),
              # passed to reward functions
              "question": x["question"].decode("utf-8"),
              # passed to reward functions
              "answer": extract_hash_answer(x["answer"].decode("utf-8")),
          }
      )
  )
  return dataset


def get_dataset_from_parquet(parquet_path, tokenizer):
  """Get dataset from a parquet file.

  Args:
    parquet_path: The path to the parquet file.
    tokenizer: The tokenizer to use for processing prompts.

  Returns:
    A grain.MapDataset.
  """
  dataset = grain.experimental.ParquetIterDataset(parquet_path)

  def process_element(x):
    return {
        "prompts": tokenizer.apply_chat_template(
            x["prompt"], tokenize=False, add_generation_prompt=True
        ),
        **{k: v for k, v in x.items() if k != "prompt"},
    }

  dataset = dataset.map(process_element)
  return dataset


def create_dataset(
    dataset_name: str,
    batch_size: int,
    num_batches: int,
    tfds_download: bool,
):
  """Creates a dataset based on the given name.

  Args:
    dataset_name: The name of the dataset to create (e.g., "gsm8k").
    batch_size: The desired batch size.
    num_batches: The number of batches to include in the dataset.
    tfds_download: the download flag when using TFDS datasets. If false, the
      data_dir used will be set to `None` and chosen by default by tfds.

  Returns:
    A batched grain.MapDataset.

  Raises:
    ValueError: If the dataset_name is not supported.
  """
  if dataset_name == "gsm8k":
    data_dir = "./train/data" if tfds_download else None
    return get_dataset_from_tfds(data_dir, "train", tfds_download).batch(
        batch_size
    )[:num_batches]
  else:
    raise ValueError(f"Unsupported dataset: {dataset_name}")
