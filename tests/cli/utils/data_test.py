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

import os
import tempfile
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from tunix.cli.utils import data as data_lib


class _FakeTokenizer:

  def encode(self, text: str):
    # Simple tokenization: one token per whitespace-separated chunk
    return text.split()


class _FakeChatTemplateTokenizer(_FakeTokenizer):

  def apply_chat_template(
      self, messages, tokenize=False, add_generation_prompt=True
  ):
    del tokenize
    del add_generation_prompt
    return " | ".join(message["content"] for message in messages)


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

  def batch(self, batch_size: int, *, batch_fn=None):
    if batch_fn:
      # In this mock, we don't fully implement custom batch_fn,
      # but we allow it to be passed.
      pass
    return _BatchedDataset(self._records, batch_size)


class _BatchedDataset:

  def __init__(self, records, batch_size: int):
    self._records = records
    self._batch_size = batch_size

  def __iter__(self):
    for i in range(0, len(self._records), self._batch_size):
      yield self._records[i : i + self._batch_size]


class PostInitDatasetTest(absltest.TestCase):

  def test_get_dataset_from_module_passes_kwargs_and_templates_prompt(self):
    module_source = """
class FakeDataset:
  def __init__(self, records):
    self._records = list(records)

  def __len__(self):
    return len(self._records)

  def __getitem__(self, idx):
    return self._records[idx]

  def map(self, fn):
    return FakeDataset([fn(record) for record in self._records])


def create_dataset(train_data_path, eval_data_path):
  return FakeDataset([
      {
          "prompt": [
              {"role": "user", "content": train_data_path},
              {"role": "assistant", "content": eval_data_path},
          ],
          "meta": "kept",
      }
  ])
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
      f.write(module_source)
      module_path = f.name

    self.addCleanup(lambda: os.unlink(module_path))

    dataset = data_lib.get_dataset_from_module(
        module_path,
        tokenizer=_FakeChatTemplateTokenizer(),
        apply_chat_template_to_dataset=True,
        train_data_path="train.json",
        eval_data_path="eval.parquet",
    )

    self.assertEqual(
        dataset[0],
        {"prompts": "train.json | eval.parquet", "meta": "kept"},
    )

  def test_get_dataset_from_module_keeps_existing_prompts(self):
    module_source = """
class FakeDataset:
  def __init__(self, records):
    self._records = list(records)

  def __len__(self):
    return len(self._records)

  def __getitem__(self, idx):
    return self._records[idx]

  def map(self, fn):
    return FakeDataset([fn(record) for record in self._records])


def create_dataset():
  return FakeDataset([
      {"prompts": "already formatted", "value": 1}
  ])
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
      f.write(module_source)
      module_path = f.name

    self.addCleanup(lambda: os.unlink(module_path))

    dataset = data_lib.get_dataset_from_module(
        module_path,
        tokenizer=_FakeChatTemplateTokenizer(),
        apply_chat_template_to_dataset=False,
    )

    self.assertEqual(dataset[0], {"prompts": "already formatted", "value": 1})

  def test_filters_by_prompt_length(self):
    tokenizer = _FakeTokenizer()
    dataset = _BaseDataset([
        {"prompts": "short", "answer": 1},
        {"prompts": "this is too long", "answer": 2},
    ])

    first, second = data_lib.post_init_dataset(
        dataset,
        tokenizer=tokenizer,  # pytype: disable=wrong-arg-types
        batch_size=2,
        num_batches=None,
        max_prompt_length=2,  # only the first record should remain
    )

    batches = list(first)
    self.assertIsNone(second)
    self.assertLen(batches, 1)
    self.assertEqual(batches[0], [{"prompts": "short", "answer": 1}])

  def test_raises_when_prompt_length_filter_removes_all_examples(self):
    tokenizer = _FakeTokenizer()
    dataset = _BaseDataset([
        {"prompts": "this is too long", "answer": 1},
        {"prompts": "also too long", "answer": 2},
    ])

    with self.assertRaisesRegex(
        ValueError, "empty after post_init_dataset filtering"
    ):
      data_lib.post_init_dataset(
          dataset,
          tokenizer=tokenizer,  # pytype: disable=wrong-arg-types
          batch_size=2,
          num_batches=None,
          max_prompt_length=2,
      )

  def test_raises_when_fraction_makes_training_split_empty(self):
    tokenizer = _FakeTokenizer()
    dataset = _BaseDataset([
        {"prompts": "short", "answer": 1},
    ])

    with self.assertRaisesRegex(
        ValueError, "empty after post_init_dataset split"
    ):
      data_lib.post_init_dataset(
          dataset,
          tokenizer=tokenizer,  # pytype: disable=wrong-arg-types
          batch_size=1,
          num_batches=None,
          max_prompt_length=None,
          fraction=0.5,
      )

  def test_limits_num_batches(self):
    tokenizer = _FakeTokenizer()
    dataset = _BaseDataset(
        [{"prompts": f"p{i}", "answer": i} for i in range(10)]
    )

    first, _ = data_lib.post_init_dataset(
        dataset,
        tokenizer=tokenizer,  # pytype: disable=wrong-arg-types
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
        tokenizer=tokenizer,  # pytype: disable=wrong-arg-types
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

  def test_normalizes_prompt_key_to_prompts(self):
    tokenizer = _FakeTokenizer()
    dataset = _BaseDataset([
        {"question": "short prompt", "answer": 1},
        {"question": "another prompt", "answer": 2},
    ])

    first, second = data_lib.post_init_dataset(
        dataset,
        tokenizer=tokenizer,  # pytype: disable=wrong-arg-types
        batch_size=2,
        num_batches=None,
        max_prompt_length=None,
        prompt_key="question",
    )

    self.assertIsNone(second)
    batches = list(first)
    self.assertEqual(
        batches[0],
        [
            {
                "question": "short prompt",
                "answer": 1,
                "prompts": "short prompt",
            },
            {
                "question": "another prompt",
                "answer": 2,
                "prompts": "another prompt",
            },
        ],
    )

  def test_normalizes_bytes_prompt_key_to_sanitized_prompts(self):
    tokenizer = _FakeTokenizer()
    dataset = _BaseDataset([
        {"question": b"<start_of_turn>short prompt", "answer": 1},
        {"question": memoryview(b"another prompt"), "answer": 2},
    ])

    first, second = data_lib.post_init_dataset(
        dataset,
        tokenizer=tokenizer,  # pytype: disable=wrong-arg-types
        batch_size=2,
        num_batches=None,
        max_prompt_length=None,
        prompt_key="question",
    )

    self.assertIsNone(second)
    batches = list(first)
    self.assertEqual(
        batches[0],
        [
            {
                "question": b"<start_of_turn>short prompt",
                "answer": 1,
                "prompts": "short prompt",
            },
            {
                "question": memoryview(b"another prompt"),
                "answer": 2,
                "prompts": "another prompt",
            },
        ],
    )

  def test_num_epochs_repeats_dataset(self):
    tokenizer = _FakeTokenizer()
    dataset = _BaseDataset(
        [{"prompts": "p0", "answer": 0}, {"prompts": "p1", "answer": 1}]
    )

    first, second = data_lib.post_init_dataset(
        dataset,
        tokenizer=tokenizer,  # pytype: disable=wrong-arg-types
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


class PostInitDatasetShardingTest(parameterized.TestCase):
  """post_init_dataset shards the RL MapDataset per process (Plan C).

  Uses a real ``grain.MapDataset`` so the strided slice, repeat, and
  ``to_iter_dataset().batch()`` chain are exercised end to end. Mocks
  ``jax.process_count``/``process_index`` to simulate the two hosts in-process.
  """

  def _records(self, n):
    import grain

    return grain.MapDataset.source(
        [{"prompts": f"p{i}", "answer": i} for i in range(n)]
    )

  def test_single_process_yields_full_global_batch_in_order(self):
    # process_count == 1 -> identity sharding, byte-for-byte the legacy path.
    with mock.patch.object(
        data_lib.jax, "process_count", return_value=1
    ), mock.patch.object(data_lib.jax, "process_index", return_value=0):
      first, second = data_lib.post_init_dataset(
          self._records(8),
          tokenizer=_FakeTokenizer(),  # pytype: disable=wrong-arg-types
          batch_size=4,
          num_batches=None,
          max_prompt_length=None,
      )
      batches = list(first)
    self.assertIsNone(second)
    self.assertLen(batches, 2)
    for batch in batches:
      self.assertLen(batch["answer"], 4)  # global batch size
    answers = [a for batch in batches for a in batch["answer"]]
    self.assertEqual(answers, list(range(8)))  # original order

  def test_two_processes_get_disjoint_shards_at_local_batch_size(self):
    global_batch_size = 4
    num_records = 8

    per_process_answers = {}
    per_process_batch_sizes = {}
    for proc_index in (0, 1):
      with mock.patch.object(
          data_lib.jax, "process_count", return_value=2
      ), mock.patch.object(
          data_lib.jax, "process_index", return_value=proc_index
      ):
        first, _ = data_lib.post_init_dataset(
            self._records(num_records),
            tokenizer=_FakeTokenizer(),  # pytype: disable=wrong-arg-types
            batch_size=global_batch_size,
            num_batches=None,
            max_prompt_length=None,
        )
        batches = list(first)
      sizes = [len(b["answer"]) for b in batches]
      answers = {a for b in batches for a in b["answer"]}
      per_process_batch_sizes[proc_index] = sizes
      per_process_answers[proc_index] = answers

    # Each process batches at the LOCAL size (global // process_count == 2).
    for proc_index in (0, 1):
      self.assertNotEmpty(per_process_batch_sizes[proc_index])
      for size in per_process_batch_sizes[proc_index]:
        self.assertEqual(size, global_batch_size // 2)

    # Disjoint strided shards covering every record: proc 0 -> evens, 1 -> odds.
    self.assertEqual(per_process_answers[0], set(range(0, num_records, 2)))
    self.assertEqual(per_process_answers[1], set(range(1, num_records, 2)))
    self.assertEqual(per_process_answers[0] & per_process_answers[1], set())

  def test_odd_record_count_yields_equal_batch_counts_across_processes(self):
    # Regression for B1: with a record count that is not a multiple of
    # process_count, both processes must still emit the SAME number of batches,
    # otherwise the cross-host make_array_from_process_local_data collective
    # hangs. 7 records / 2 processes -> truncate to 6 -> 3 records each ->
    # local batch size 1 -> 3 batches each; record 6 is dropped from the tail.
    global_batch_size = 2
    num_records = 7

    per_process_answers = {}
    per_process_num_batches = {}
    for proc_index in (0, 1):
      with mock.patch.object(
          data_lib.jax, "process_count", return_value=2
      ), mock.patch.object(
          data_lib.jax, "process_index", return_value=proc_index
      ):
        first, _ = data_lib.post_init_dataset(
            self._records(num_records),
            tokenizer=_FakeTokenizer(),  # pytype: disable=wrong-arg-types
            batch_size=global_batch_size,
            num_batches=None,
            max_prompt_length=None,
        )
        batches = list(first)
      per_process_num_batches[proc_index] = len(batches)
      per_process_answers[proc_index] = {
          a for b in batches for a in b["answer"]
      }

    # Equal batch counts across hosts is the property that prevents the hang.
    self.assertEqual(per_process_num_batches[0], per_process_num_batches[1])
    self.assertEqual(per_process_num_batches[0], 3)  # floor(7/2) / local_bs 1

    # Disjoint strided shards; the tail record (6) is dropped, the rest covered.
    self.assertEqual(per_process_answers[0], {0, 2, 4})
    self.assertEqual(per_process_answers[1], {1, 3, 5})
    self.assertEqual(per_process_answers[0] & per_process_answers[1], set())
    self.assertNotIn(6, per_process_answers[0] | per_process_answers[1])

  def test_rejects_indivisible_global_batch(self):
    with mock.patch.object(
        data_lib.jax, "process_count", return_value=3
    ), mock.patch.object(data_lib.jax, "process_index", return_value=0):
      with self.assertRaisesRegex(ValueError, "divisible"):
        data_lib.post_init_dataset(
            self._records(9),
            tokenizer=_FakeTokenizer(),  # pytype: disable=wrong-arg-types
            batch_size=4,
            num_batches=None,
            max_prompt_length=None,
        )


class ShardByProcessTest(parameterized.TestCase):

  def test_single_process_is_identity(self):
    # process_count == 1 -> whole dataset in original order, global batch size.
    dataset = _BaseDataset([{"v": i} for i in range(8)])
    with mock.patch.object(
        data_lib.jax, "process_count", return_value=1
    ), mock.patch.object(data_lib.jax, "process_index", return_value=0):
      local, local_bs = data_lib.shard_by_process(dataset, global_batch_size=4)
    self.assertIs(local, dataset)
    self.assertEqual(local_bs, 4)
    self.assertEqual([r["v"] for r in local], list(range(8)))

  @parameterized.parameters((0,), (1,))
  def test_two_processes_get_disjoint_strided_shards(self, proc_index):
    dataset = _BaseDataset([{"v": i} for i in range(8)])
    with mock.patch.object(
        data_lib.jax, "process_count", return_value=2
    ), mock.patch.object(
        data_lib.jax, "process_index", return_value=proc_index
    ):
      local, local_bs = data_lib.shard_by_process(dataset, global_batch_size=4)
    # Local batch is global // process_count.
    self.assertEqual(local_bs, 2)
    # Each process reads a disjoint strided shard: proc 0 -> evens, 1 -> odds.
    self.assertEqual([r["v"] for r in local], list(range(proc_index, 8, 2)))

  @parameterized.parameters((0,), (1,))
  def test_odd_length_yields_equal_shards_dropping_tail(self, proc_index):
    # Regression for B1: an odd record count must not produce unequal shards
    # (which would hang the cross-host collective). 7 records / 2 processes
    # truncates to 6 -> proc 0 {0,2,4}, proc 1 {1,3,5}; record 6 is tail-dropped.
    dataset = _BaseDataset([{"v": i} for i in range(7)])
    with mock.patch.object(
        data_lib.jax, "process_count", return_value=2
    ), mock.patch.object(
        data_lib.jax, "process_index", return_value=proc_index
    ):
      local, local_bs = data_lib.shard_by_process(dataset, global_batch_size=4)
    self.assertEqual(local_bs, 2)
    self.assertEqual([r["v"] for r in local], list(range(proc_index, 6, 2)))

  def test_odd_length_shards_have_equal_length_across_processes(self):
    # The two processes' shards must be the SAME length so they emit the same
    # number of batches (the property whose absence hangs multi-host training).
    dataset = _BaseDataset([{"v": i} for i in range(7)])
    lengths = []
    for proc_index in (0, 1):
      with mock.patch.object(
          data_lib.jax, "process_count", return_value=2
      ), mock.patch.object(
          data_lib.jax, "process_index", return_value=proc_index
      ):
        local, _ = data_lib.shard_by_process(dataset, global_batch_size=4)
      lengths.append(len(list(local)))
    self.assertEqual(lengths[0], lengths[1])
    self.assertEqual(lengths[0], 3)  # floor(7 / 2)

  def test_two_process_shards_are_disjoint_and_cover_all(self):
    dataset = _BaseDataset([{"v": i} for i in range(8)])
    shards = {}
    for proc_index in (0, 1):
      with mock.patch.object(
          data_lib.jax, "process_count", return_value=2
      ), mock.patch.object(
          data_lib.jax, "process_index", return_value=proc_index
      ):
        local, _ = data_lib.shard_by_process(dataset, global_batch_size=4)
        shards[proc_index] = {r["v"] for r in local}
    self.assertEqual(shards[0] & shards[1], set())
    self.assertEqual(shards[0] | shards[1], set(range(8)))

  def test_rejects_indivisible_global_batch(self):
    dataset = _BaseDataset([{"v": i} for i in range(8)])
    with mock.patch.object(
        data_lib.jax, "process_count", return_value=3
    ), mock.patch.object(data_lib.jax, "process_index", return_value=0):
      with self.assertRaisesRegex(ValueError, "divisible"):
        data_lib.shard_by_process(dataset, global_batch_size=4)


if __name__ == "__main__":
  absltest.main()
