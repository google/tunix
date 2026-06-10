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

"""Tests for per-process sharding in tunix.examples.data.translation_dataset.

These tests pin down the multi-controller data-loading contract: each JAX
process must read a disjoint shard of the records and batch only its local
``global_batch_size // process_count`` slice, so that
``sharding_utils.shard_input`` can reassemble the global batch via
``jax.make_array_from_process_local_data``. The loader is a ``grain.MapDataset``
chain sharded by ``cli.utils.data.shard_by_process`` (a strided slice), so the
process-count/index are mocked on that module.
"""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from grain import python as grain
import numpy as np
from tunix.cli.utils import data as data_lib
from tunix.examples.data import translation_dataset


class _FakeTokenizer:
  """Minimal tokenizer that maps each example to a fixed-length token vector.

  The token value encodes the source record id so a test can verify exactly
  which records a process consumed.
  """

  def __init__(self, seq_len: int = 4):
    self._seq_len = seq_len

  def pad_id(self) -> int:
    return 0

  def tokenize(self, text, prefix="", suffix="", add_eos=False):
    # ``text`` is the per-record marker like "r3"; encode its integer id into
    # every token so downstream assertions can recover which record this was.
    record_id = int(text.lstrip("r")) + 1  # +1 so ids are never 0 (the pad id)
    if add_eos:
      # Destination half: two tokens.
      return np.full((2,), record_id, dtype=np.int32)
    # Source half (prefix/suffix carry no extra tokens in this fake): two tokens.
    return np.full((2,), record_id, dtype=np.int32)


class _ListDataSource:
  """A trivial grain RandomAccessDataSource over a list of MTNT-style records."""

  def __init__(self, num_records: int):
    self._records = [
        {"src": f"r{i}".encode(), "dst": f"r{i}".encode()}
        for i in range(num_records)
    ]

  def __len__(self) -> int:
    return len(self._records)

  def __getitem__(self, idx: int):
    return self._records[idx]


def _build_loader(num_records, batch_size, num_epochs=1, max_seq_len=8):
  return translation_dataset._build_data_loader(
      data_source=_ListDataSource(num_records),
      batch_size=batch_size,
      num_epochs=num_epochs,
      max_seq_len=max_seq_len,
      tokenizer=_FakeTokenizer(),
      input_template=translation_dataset.INPUT_TEMPLATE,
  )


def _record_ids_in_batch(batch) -> set[int]:
  """Recovers the original record ids contained in one batch."""
  # Each row's first token encodes (record_id + 1); subtract to recover the id.
  return {int(row[0]) - 1 for row in batch.input_tokens}


class BuildDataLoaderShardingTest(parameterized.TestCase):

  def test_single_process_uses_full_global_batch(self):
    # process_count == 1 -> per-process batch == global batch, identical to the
    # old grain.NoSharding() behavior.
    with mock.patch.object(
        data_lib.jax, "process_count", return_value=1
    ), mock.patch.object(data_lib.jax, "process_index", return_value=0):
      loader = _build_loader(num_records=8, batch_size=4)
      batches = list(loader)

    self.assertNotEmpty(batches)
    for batch in batches:
      # Global == local on one process.
      self.assertEqual(batch.input_tokens.shape[0], 4)
    # All 8 records are consumed across the two batches.
    seen = set()
    for batch in batches:
      seen |= _record_ids_in_batch(batch)
    self.assertEqual(seen, set(range(8)))

  def test_two_processes_split_records_and_use_local_batch(self):
    # Simulate the two hosts in-process by toggling process_index. Each process
    # must (a) batch at global // 2, and (b) read a disjoint half of records.
    global_batch_size = 4
    num_records = 8

    per_process_seen = {}
    per_process_batch_shapes = {}
    for proc_index in (0, 1):
      with mock.patch.object(
          data_lib.jax, "process_count", return_value=2
      ), mock.patch.object(
          data_lib.jax, "process_index", return_value=proc_index
      ):
        loader = _build_loader(
            num_records=num_records, batch_size=global_batch_size
        )
        batches = list(loader)
      seen = set()
      shapes = []
      for batch in batches:
        shapes.append(batch.input_tokens.shape[0])
        seen |= _record_ids_in_batch(batch)
      per_process_seen[proc_index] = seen
      per_process_batch_shapes[proc_index] = shapes

    # Each process batches at the LOCAL size (global // process_count == 2).
    for proc_index in (0, 1):
      self.assertNotEmpty(per_process_batch_shapes[proc_index])
      for shape in per_process_batch_shapes[proc_index]:
        self.assertEqual(shape, global_batch_size // 2)

    # The two processes read DISJOINT shards...
    self.assertEqual(
        per_process_seen[0] & per_process_seen[1],
        set(),
        "Processes must not read overlapping records.",
    )
    # ...and together cover every record (8 records, 4 per process shard).
    self.assertEqual(
        per_process_seen[0] | per_process_seen[1], set(range(num_records))
    )
    self.assertLen(per_process_seen[0], 4)
    self.assertLen(per_process_seen[1], 4)

  def test_build_loader_rejects_indivisible_batch(self):
    with mock.patch.object(
        data_lib.jax, "process_count", return_value=3
    ), mock.patch.object(data_lib.jax, "process_index", return_value=0):
      with self.assertRaisesRegex(ValueError, "divisible"):
        _build_loader(num_records=8, batch_size=4)


class SingleHostByteIdenticalTest(parameterized.TestCase):
  """Proves the MapDataset loader matches the legacy DataLoader on one host.

  Acceptance criterion: on a single host (process_count == 1) the new
  ``grain.MapDataset`` pipeline must produce byte-identical batches (same
  records, same order, same batching) as the original ``grain.DataLoader``
  (``IndexSampler`` + ``grain.Batch``) implementation it replaced.
  """

  def _legacy_loader(
      self, num_records, batch_size, num_epochs=1, max_seq_len=8
  ):
    # Faithful reconstruction of the previous DataLoader implementation.
    return grain.DataLoader(
        data_source=_ListDataSource(num_records),
        sampler=grain.IndexSampler(
            num_records=num_records,
            num_epochs=num_epochs,
            shard_options=grain.NoSharding(),
        ),
        operations=[
            translation_dataset._Tokenize(
                _FakeTokenizer(), translation_dataset.INPUT_TEMPLATE
            ),
            translation_dataset._BuildTrainInput(
                max_seq_len, _FakeTokenizer().pad_id()
            ),
            translation_dataset._FilterOverlength(max_seq_len),
            grain.Batch(batch_size=batch_size, drop_remainder=True),
        ],
    )

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
        self._legacy_loader(
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


class CreateDatasetsValidationTest(parameterized.TestCase):

  def test_create_datasets_rejects_indivisible_global_batch(self):
    with mock.patch.object(data_lib.jax, "process_count", return_value=4):
      with self.assertRaisesRegex(ValueError, "divisible"):
        translation_dataset.create_datasets(
            dataset_name="mtnt/en-fr",
            global_batch_size=6,
            max_target_length=8,
            num_train_epochs=1,
            tokenizer=_FakeTokenizer(),
        )


if __name__ == "__main__":
  absltest.main()
