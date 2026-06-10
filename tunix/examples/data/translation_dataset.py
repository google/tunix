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

"""Data loading and preprocessing."""

from collections.abc import Iterable
from typing import Any

import datasets
from grain import python as grain
import numpy as np
import tensorflow_datasets as tfds
from tunix.cli.utils import data as data_lib
from tunix.generate import tokenizer_adapter as tokenizer_lib
from tunix.sft.peft_trainer import TrainingInput  # pylint: disable=g-importing-member


INPUT_TEMPLATE = {
    "prefix": "Translate this into French:\n",
    "suffix": "\n",
}

INPUT_TEMPLATE_IT = {
    "prefix": "<start_of_turn>user\nTranslate this into French:\n",
    "suffix": "\n<end_of_turn>\n<start_of_turn>model\n",
}


def create_datasets(
    dataset_name: str,
    global_batch_size: int,
    max_target_length: int,
    num_train_epochs: int | None,
    tokenizer: tokenizer_lib.Tokenizer,
    instruct_tuned: bool = False,
    tfds_download: bool = True,
    input_template: dict[str, str] | None = None,
) -> tuple[Iterable[TrainingInput], Iterable[TrainingInput]]:
  """Creates train and eval data iterator.

  Args:
    dataset_name: The name of the dataset to use.
    global_batch_size: The global batch size to use for both train and eval.
    max_target_length: The maximum length of the target sequence.
    num_train_epochs: The number of epochs to use for training. If None, the
      dataset will be repeated indefinitely.
    tokenizer: The tokenizer to use for tokenizing the dataset.
    instruct_tuned: Whether the dataset should be instruct tuned.
    tfds_download: the download flag when using TFDS datasets.
    input_template: The input template to use for the dataset.

  Returns:
    A tuple of train and eval data iterators.

  Raises:
    ValueError: If ``global_batch_size`` is not divisible by the number of JAX
      processes. Under multi-controller JAX each process must contribute an
      equal-sized per-process shard of the global batch, so the global batch
      size must split evenly across processes.
  """
  # Fail fast on an unshardable global batch before any expensive data loading.
  data_lib.validate_global_batch_divisible(global_batch_size)

  if dataset_name == "mtnt/en-fr":
    import tensorflow_datasets.translate.mtnt

    train_ds, eval_ds = tfds.data_source(
        dataset_name, split=("train", "valid"), download=tfds_download
    )
  elif dataset_name == "Helsinki-NLP/opus-100":  # Hugging Face dataloader
    train_ds, eval_ds = datasets.load_dataset(
        dataset_name, data_dir="en-fr", split=("train", "validation")
    )
  else:
    raise ValueError(f"Unsupported dataset: {dataset_name}")

  input_template = INPUT_TEMPLATE_IT if instruct_tuned else INPUT_TEMPLATE

  train_loader = _build_data_loader(
      data_source=train_ds,
      batch_size=global_batch_size,
      num_epochs=num_train_epochs,
      max_seq_len=max_target_length,
      tokenizer=tokenizer,
      input_template=input_template,
  )
  eval_loader = _build_data_loader(
      data_source=eval_ds,
      batch_size=global_batch_size,
      num_epochs=1,
      max_seq_len=max_target_length,
      tokenizer=tokenizer,
      input_template=input_template,
  )
  return train_loader, eval_loader


def _build_data_loader(
    *,
    data_source: grain.RandomAccessDataSource,
    batch_size: int,
    num_epochs: int | None,
    max_seq_len: int,
    tokenizer: tokenizer_lib.Tokenizer,
    input_template: dict[str, str],
) -> grain.IterDataset:
  """Builds a data iterator for the given data source.

  ``batch_size`` is the GLOBAL batch size. The pipeline is a ``grain.MapDataset``
  chain (``map(tokenize) -> map(build_input) -> filter(overlength)``) that is
  sharded across JAX processes by ``shard_by_process``: under multi-controller
  JAX each host slices a disjoint strided shard of the records and batches only
  its local ``batch_size // jax.process_count()`` examples.
  ``tunix.sft.sharding_utils.shard_input`` then assembles those per-process local
  batches into the global batch via ``jax.make_array_from_process_local_data``
  (which expects each process to supply exactly its local sub-batch).

  On a single host ``jax.process_count() == 1``, ``shard_by_process`` is the
  identity (whole dataset, original order) and the per-process batch equals the
  global batch, so the produced records, their order, and the batching are
  byte-for-byte identical to the previous ``grain.NoSharding()`` configuration.
  ``MapDataset.batch`` cannot follow a ``filter`` (which may drop elements), so
  the filtered ``MapDataset`` is converted to an ``IterDataset`` before batching,
  exactly as the RL ``post_init_dataset`` path does.
  """
  local_source, per_process_batch_size = data_lib.shard_by_process(
      grain.MapDataset.source(data_source), batch_size
  )
  # ``repeat(None)`` is an infinite repeat and ``repeat(N)`` is N epochs, which
  # matches the previous ``IndexSampler(num_epochs=...)`` semantics exactly.
  dataset = (
      local_source.map(_Tokenize(tokenizer, input_template))
      .map(_BuildTrainInput(max_seq_len, tokenizer.pad_id()))
      .filter(_FilterOverlength(max_seq_len))
      .repeat(num_epochs)
  )
  return dataset.to_iter_dataset().batch(
      per_process_batch_size, drop_remainder=True
  )


class _Tokenize(grain.MapTransform):
  """Tokenize the input."""

  def __init__(
      self, tokenizer: tokenizer_lib.Tokenizer, input_template: dict[str, str]
  ):
    self._tokenizer = tokenizer
    self._input_template = input_template

  def map(self, element: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Tokenize the input."""
    if "src" in element.keys():  ## MTNT dataset
      src_tokens = self._tokenizer.tokenize(
          element["src"].decode(),
          prefix=self._input_template["prefix"],
          suffix=self._input_template["suffix"],
          add_eos=False,
      )
      dst_tokens = self._tokenizer.tokenize(
          element["dst"].decode(), add_eos=True
      )
    else:  ## OPUS-100 dataset
      src_tokens = self._tokenizer.tokenize(
          element["translation"]["en"],
          prefix=self._input_template["prefix"],
          suffix=self._input_template["suffix"],
          add_eos=False,
      )
      dst_tokens = self._tokenizer.tokenize(
          element["translation"]["fr"], add_eos=True
      )
    return src_tokens, dst_tokens


class _BuildTrainInput(grain.MapTransform):
  """Build a TrainingInput from a tuple of source and destination tokens."""

  def __init__(self, max_seq_len: int, pad_value: int | bool):
    self._max_seq_len = max_seq_len
    self._pad_value = pad_value

  def map(self, tokens: tuple[np.ndarray, np.ndarray]) -> TrainingInput:
    src_tokens, dst_tokens = tokens

    # The input sequence fed to the model is simply the concatenation of the
    # source and the destination.
    tokens = np.concat([src_tokens, dst_tokens], axis=0)

    # To prevent the model from updating based on the source (input)
    # tokens, add a target mask to each input.
    q_mask = np.zeros_like(src_tokens, dtype=np.bool)
    a_mask = np.ones_like(dst_tokens, dtype=np.bool)
    mask = np.concat([q_mask, a_mask], axis=0)

    # If the input tokens sequence is smaller than the target sequence size,
    # then pad it with pad tokens.
    tokens = self._pad_up_to_max_len(tokens, self._pad_value)

    # Don't want to perform the backward pass on the pad tokens.
    mask = self._pad_up_to_max_len(mask, 0)

    return TrainingInput(input_tokens=tokens, input_mask=mask)

  def _pad_up_to_max_len(
      self, input_tensor: np.ndarray, pad_value: int
  ) -> np.ndarray:
    """Pad the given tensor up to sequence length of a batch."""
    seq_len = input_tensor.shape[0]
    to_pad = np.maximum(self._max_seq_len - seq_len, 0)
    return np.pad(
        input_tensor,
        [[0, to_pad]],
        mode="constant",
        constant_values=pad_value,
    )


class _FilterOverlength(grain.FilterTransform):
  """Filter out overlength examples."""

  def __init__(self, max_seq_len: int):
    self._max_seq_len = max_seq_len

  def filter(self, element: TrainingInput) -> bool:
    return element.input_tokens.shape[0] <= self._max_seq_len
