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

"""Tests for tunix.cli.utils.data.post_init_dataset."""

from __future__ import annotations

from absl.testing import absltest
from tunix.cli.utils import data as data_lib


class _FakeTokenizer:

  def tokenize(self, text: str):
    # Simple tokenization: one token per whitespace-separated chunk
    return text.split()


class _BaseDataset:
  """Minimal dataset to mimic grain interfaces used in post_init_dataset."""

  def __init__(self, records):
    self._records = list(records)

  def __len__(self):
    return len(self._records)

  def __getitem__(self, idx):
    if isinstance(idx, slice):
      return _BaseDataset(self._records[idx])
    return self._records[idx]

  def filter(self, fn):
    return _BaseDataset([x for x in self._records if fn(x)])

  def repeat(self, n):
    return _RepeatDataset(self, n)

  def to_iter_dataset(self):
    return _IterDataset(self._records)

  def map(self, fn):  # Not used in tests, but kept for fidelity.
    return _BaseDataset([fn(x) for x in self._records])


class _RepeatDataset:

  def __init__(self, base: _BaseDataset, n: int):
    self._base = base
    self._n = n

  def __len__(self):
    return len(self._base) * self._n

  def to_iter_dataset(self):
    return _IterDataset(self._base._records * self._n)


class _IterDataset:

  def __init__(self, records):
    self._records = list(records)

  def batch(self, batch_size: int):
    return _BatchedDataset(self._records, batch_size)


class _BatchedDataset:

  def __init__(self, records, batch_size: int):
    self._records = records
    self._batch_size = batch_size

  def __iter__(self):
    for i in range(0, len(self._records), self._batch_size):
      yield self._records[i : i + self._batch_size]


class PostInitDatasetTest(absltest.TestCase):

  def test_filters_by_prompt_length(self):
    tokenizer = _FakeTokenizer()
    dataset = _BaseDataset([
        {"prompts": "short", "answer": 1},
        {"prompts": "this is too long", "answer": 2},
    ])

    first, second = data_lib.post_init_dataset(
        dataset,
        tokenizer=tokenizer,
        batch_size=2,
        num_batches=None,
        max_prompt_length=2,  # only the first record should remain
    )

    batches = list(first)
    self.assertIsNone(second)
    self.assertLen(batches, 1)
    self.assertEqual(batches[0], [{"prompts": "short", "answer": 1}])

  def test_limits_num_batches(self):
    tokenizer = _FakeTokenizer()
    dataset = _BaseDataset(
        [{"prompts": f"p{i}", "answer": i} for i in range(10)]
    )

    first, _ = data_lib.post_init_dataset(
        dataset,
        tokenizer=tokenizer,
        batch_size=3,
        num_batches=2,  # keep at most 2 batches * 3 = 6 examples
        max_prompt_length=None,
    )

    batches = list(first)
    self.assertLen(batches, 2)
    self.assertEqual([len(b) for b in batches], [3, 3])
    self.assertEqual(batches[0][0]["prompts"], "p0")
    self.assertEqual(batches[-1][-1]["prompts"], "p5")

  def test_fraction_split_and_repeat(self):
    tokenizer = _FakeTokenizer()
    dataset = _BaseDataset(
        [{"prompts": f"p{i}", "answer": i} for i in range(8)]
    )

    first, second = data_lib.post_init_dataset(
        dataset,
        tokenizer=tokenizer,
        batch_size=2,
        num_batches=None,
        max_prompt_length=None,
        fraction=0.5,
        num_epochs=1,
    )

    first_batches = list(first)
    second_batches = list(second)

    self.assertLen(first_batches, 2)  # 4 items / batch_size 2
    self.assertLen(second_batches, 2)  # remaining 4 items / batch_size 2
    self.assertEqual(first_batches[0][0]["prompts"], "p0")
    self.assertEqual(second_batches[-1][-1]["prompts"], "p7")

  def test_num_epochs_repeats_dataset(self):
    tokenizer = _FakeTokenizer()
    dataset = _BaseDataset(
        [{"prompts": "p0", "answer": 0}, {"prompts": "p1", "answer": 1}]
    )

    first, second = data_lib.post_init_dataset(
        dataset,
        tokenizer=tokenizer,
        batch_size=1,
        num_batches=None,
        max_prompt_length=None,
        num_epochs=3,
    )

    self.assertIsNone(second)
    batches = list(first)
    # 2 items * 3 epochs = 6 batches of size 1
    self.assertLen(batches, 6)
    self.assertEqual(
        [b[0]["prompts"] for b in batches], ["p0", "p1", "p0", "p1", "p0", "p1"]
    )


if __name__ == "__main__":
  absltest.main()
