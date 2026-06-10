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

"""Tests for the VLM SFT data pipeline (per-process MapDataset sharding).

These tests exercise ``vlm_training.load_dataset`` on CPU with fake image/text
processors, pinning the same multi-controller contract as the translation
loader: on a single host the new ``grain.MapDataset`` pipeline is byte-identical
to the previous ``grain.DataLoader``, and across two processes each host reads a
disjoint strided shard batched at the local ``global_batch_size //
process_count`` size.

The example script lives at ``examples/sft/vlm_training.py`` (outside the
``tunix`` package), so it is loaded by path.
"""

import importlib.util
import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from grain import python as grain
import numpy as np
from tunix.cli.utils import data as data_lib


def _load_vlm_module():
  here = os.path.dirname(os.path.abspath(__file__))
  path = os.path.normpath(
      os.path.join(here, "../../../examples/sft/vlm_training.py")
  )
  spec = importlib.util.spec_from_file_location("vlm_training", path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


vlm_training = _load_vlm_module()


class _FakeImageProcessor:
  """Maps each record to a fixed-shape image array tagged with its record id."""

  def __call__(self, image):
    # ``image`` is the record id; emit a deterministic image so equality checks
    # across the two loader implementations are meaningful.
    return [np.full((2, 2, 3), image, dtype=np.int32)]


class _FakeTokenizer:

  def pad_id(self) -> int:
    return 0

  def tokenize(self, text, add_eos=False):
    # The destination text carries the record id; the source prompt is constant.
    if add_eos:
      record_id = int(text)
      return np.full((2,), record_id + 1, dtype=np.int32)
    return np.full((2,), 99, dtype=np.int32)


class _ListDataSource:
  """A grain RandomAccessDataSource over LaTeX-OCR-style records."""

  def __init__(self, num_records: int):
    # ``image`` encodes the record id (an int the fake processor passes through);
    # ``text`` is the stringified id used by the fake tokenizer.
    self._records = [{"image": i, "text": str(i)} for i in range(num_records)]

  def __len__(self) -> int:
    return len(self._records)

  def __getitem__(self, idx: int):
    return self._records[idx]


def _build_loader(num_records, batch_size, num_epochs=1, max_seq_len=8):
  return vlm_training.load_dataset(
      data_source=_ListDataSource(num_records),
      image_processor=_FakeImageProcessor(),
      tokenizer=_FakeTokenizer(),
      batch_size=batch_size,
      num_epochs=num_epochs,
      max_seq_len=max_seq_len,
  )


def _legacy_loader(num_records, batch_size, num_epochs=1, max_seq_len=8):
  # Faithful reconstruction of the previous DataLoader implementation.
  return grain.DataLoader(
      data_source=_ListDataSource(num_records),
      sampler=grain.IndexSampler(
          num_records=num_records,
          num_epochs=num_epochs,
          shard_options=grain.NoSharding(),
      ),
      operations=[
          vlm_training._Preprocess(_FakeImageProcessor(), _FakeTokenizer()),
          vlm_training._BuildTrainInput(max_seq_len, _FakeTokenizer().pad_id()),
          vlm_training._FilterOverlength(max_seq_len),
          grain.Batch(batch_size=batch_size, drop_remainder=True),
      ],
      # worker_count=0 reads in-process: the emitted records/order/batching are
      # identical to the production worker_count=8 config, but it avoids spawning
      # multiprocess workers that touch unparsed absl flags under pytest.
      worker_count=0,
  )


def _record_ids_in_batch(batch) -> set[int]:
  # The destination half's first token encodes (record_id + 1); the destination
  # is the second pair of tokens (source half is the constant 99).
  return {int(row[2]) - 1 for row in batch.input_tokens}


class VlmLoadDatasetTest(parameterized.TestCase):

  @parameterized.parameters((8, 4, 1), (10, 2, 1), (6, 3, 2))
  def test_mapdataset_matches_legacy_dataloader_single_host(
      self, num_records, batch_size, num_epochs
  ):
    with mock.patch.object(
        data_lib.jax, "process_count", return_value=1
    ), mock.patch.object(data_lib.jax, "process_index", return_value=0):
      new_batches = list(
          _build_loader(
              num_records=num_records,
              batch_size=batch_size,
              num_epochs=num_epochs,
          )
      )
    legacy_batches = list(
        _legacy_loader(
            num_records=num_records,
            batch_size=batch_size,
            num_epochs=num_epochs,
        )
    )

    self.assertEqual(len(new_batches), len(legacy_batches))
    self.assertNotEmpty(new_batches)
    for new_b, legacy_b in zip(new_batches, legacy_batches):
      np.testing.assert_array_equal(new_b.input_tokens, legacy_b.input_tokens)
      np.testing.assert_array_equal(new_b.input_mask, legacy_b.input_mask)
      np.testing.assert_array_equal(new_b.images, legacy_b.images)

  def test_two_processes_split_records_and_use_local_batch(self):
    global_batch_size = 4
    num_records = 8

    per_process_seen = {}
    per_process_shapes = {}
    for proc_index in (0, 1):
      with mock.patch.object(
          data_lib.jax, "process_count", return_value=2
      ), mock.patch.object(
          data_lib.jax, "process_index", return_value=proc_index
      ):
        batches = list(
            _build_loader(num_records=num_records, batch_size=global_batch_size)
        )
      seen = set()
      shapes = []
      for batch in batches:
        shapes.append(batch.input_tokens.shape[0])
        seen |= _record_ids_in_batch(batch)
      per_process_seen[proc_index] = seen
      per_process_shapes[proc_index] = shapes

    # Each process batches at the LOCAL size (global // process_count == 2).
    for proc_index in (0, 1):
      self.assertNotEmpty(per_process_shapes[proc_index])
      for shape in per_process_shapes[proc_index]:
        self.assertEqual(shape, global_batch_size // 2)

    # Disjoint shards that together cover every record.
    self.assertEqual(per_process_seen[0] & per_process_seen[1], set())
    self.assertEqual(
        per_process_seen[0] | per_process_seen[1], set(range(num_records))
    )

  def test_rejects_indivisible_global_batch(self):
    with mock.patch.object(
        data_lib.jax, "process_count", return_value=3
    ), mock.patch.object(data_lib.jax, "process_index", return_value=0):
      with self.assertRaisesRegex(ValueError, "divisible"):
        _build_loader(num_records=8, batch_size=4)


if __name__ == "__main__":
  absltest.main()
