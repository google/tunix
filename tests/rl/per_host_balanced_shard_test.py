# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Length-balanced per-host prompt shards (TUNIX_PER_HOST_BALANCED_SHARD=1).

With the flag on, every process of the per-host learner rolls out the prompts
that `balanced_shard_indices` assigns to it. Each process computes the
partition on its own, so it must be exact and deterministic: a regression
would silently drop or duplicate prompts across processes while every shape
still matches.
"""

# pylint: disable=protected-access

import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
from tunix.rl import algorithm_config as algo_config_lib
from tunix.rl import rl_learner

_FLAG = "TUNIX_PER_HOST_BALANCED_SHARD"

# (name, per-row lengths, process count).
_PARTITIONS = (
    ("two_processes", [5, 1, 9, 3, 7, 2, 8, 4], 2),
    (
        "four_processes",
        [3, 14, 15, 9, 2, 6, 5, 35, 8, 9, 7, 9, 32, 38, 4, 6],
        4,
    ),
    ("one_process", [4, 2, 6], 1),
    ("one_row_per_process", [1, 2, 3, 4], 4),
    ("all_tied", [3] * 12, 3),
    ("empty_prompts", [0, 0, 5, 0, 1, 0], 2),
    ("many_rows", np.random.default_rng(0).integers(1, 1000, 96).tolist(), 8),
)


class _Config(algo_config_lib.AlgorithmConfig):
  pass


class _Learner(rl_learner.RLLearner[_Config]):
  """Implements the abstract methods; the tests only shard batches."""

  def _generate_and_compute_advantage(self, training_input, mode):
    raise NotImplementedError

  def _compute_trajectory_ids(self, example, steps):
    raise NotImplementedError

  def _num_iterations(self):
    return 1

  def _num_generations(self):
    return 1


class BalancedShardIndicesTest(parameterized.TestCase):

  @parameterized.named_parameters(*_PARTITIONS)
  def test_exact_equal_count_partition(self, lengths, process_count):
    shards = rl_learner.balanced_shard_indices(lengths, process_count)
    self.assertLen(shards, process_count)
    rows = [row for shard in shards for row in shard]
    # Disjoint, and together they hold every row exactly once.
    self.assertLen(set(rows), len(rows))
    self.assertCountEqual(rows, range(len(lengths)))
    for shard in shards:
      self.assertLen(shard, len(lengths) // process_count)
      self.assertEqual(shard, sorted(shard))

  @parameterized.named_parameters(*_PARTITIONS)
  def test_length_totals_differ_by_at_most_the_length_spread(
      self, lengths, process_count
  ):
    shards = rl_learner.balanced_shard_indices(lengths, process_count)
    totals = [sum(lengths[i] for i in shard) for shard in shards]
    self.assertLessEqual(max(totals) - min(totals), max(lengths) - min(lengths))

  @parameterized.named_parameters(*_PARTITIONS)
  def test_every_copy_of_the_lengths_gives_the_same_partition(
      self, lengths, process_count
  ):
    # Every process computes the partition on its own, from its own copy of
    # the global batch.
    want = rl_learner.balanced_shard_indices(lengths, process_count)
    for copy in (list(lengths), tuple(lengths), np.asarray(lengths)):
      self.assertEqual(
          rl_learner.balanced_shard_indices(copy, process_count), want
      )

  def test_tied_lengths_are_dealt_in_row_order(self):
    self.assertEqual(
        rl_learner.balanced_shard_indices([7] * 6, 2), [[0, 3, 4], [1, 2, 5]]
    )
    # Rows 3 and 4 tie, and so do rows 0 and 2: the lower index goes first.
    self.assertEqual(
        rl_learner.balanced_shard_indices([2, 9, 2, 5, 5, 1], 3),
        [[1, 5], [2, 3], [0, 4]],
    )

  def test_snake_deal_reverses_every_round(self):
    lengths = list(range(1, 13))  # Longest first: rows 11, 10, ..., 0.
    shards = rl_learner.balanced_shard_indices(lengths, 4)
    # Round 0 deals rows 11 10 9 8 to processes 0 1 2 3, round 1 deals rows
    # 7 6 5 4 to processes 3 2 1 0, round 2 deals rows 3 2 1 0 to 0 1 2 3.
    self.assertEqual(shards, [[3, 4, 11], [2, 5, 10], [1, 6, 9], [0, 7, 8]])
    totals = [sum(lengths[i] for i in shard) for shard in shards]
    self.assertEqual(totals, [21, 20, 19, 18])
    # The stride shard [p::4] of the same rows spans 15 to 24.
    stride = [sum(lengths[p::4]) for p in range(4)]
    self.assertEqual(stride, [15, 18, 21, 24])


class ShardBatchForProcessTest(parameterized.TestCase):

  def _shards(self, batch, process_count, flag=None):
    """Every process's `_shard_batch_for_process` with `_FLAG` set to `flag`."""
    learner = _Learner.__new__(_Learner)
    shards = []
    with mock.patch.dict(os.environ):
      os.environ.pop(_FLAG, None)
      if flag is not None:
        os.environ[_FLAG] = flag
      for process_index in range(process_count):
        with (
            mock.patch.object(jax, "process_count", return_value=process_count),
            mock.patch.object(jax, "process_index", return_value=process_index),
        ):
          shards.append(learner._shard_batch_for_process(batch))
    return shards

  def _batch(self, prompts):
    rows = len(prompts)
    return {
        "prompts": prompts,
        "answer": [f"answer {i}" for i in range(rows)],
        "row": np.arange(rows),
        "features": np.arange(2 * rows).reshape(rows, 2),
        # Not batch-aligned: passed through to every process.
        "options": {"temperature": 1.0},
    }

  @parameterized.named_parameters(
      ("text", ["a" * k for k in (5, 1, 9, 3, 7, 2, 8, 4)], 2),
      ("text_four_processes", [str(i) * (i % 5 + 1) for i in range(16)], 4),
      ("token_ids", [list(range(k)) for k in (3, 6, 1, 4, 2, 5)], 3),
  )
  def test_flag_on_shards_by_the_balanced_partition(
      self, prompts, process_count
  ):
    batch = self._batch(prompts)
    want = rl_learner.balanced_shard_indices(
        [len(prompt) for prompt in prompts], process_count
    )
    shards = self._shards(batch, process_count, flag="1")
    for shard, rows in zip(shards, want):
      self.assertEqual(shard["prompts"], [prompts[i] for i in rows])
      self.assertEqual(shard["answer"], [batch["answer"][i] for i in rows])
      np.testing.assert_array_equal(shard["row"], rows)
      np.testing.assert_array_equal(shard["features"], batch["features"][rows])
      self.assertIs(shard["options"], batch["options"])
    # Every prompt on exactly one process, equally many on each.
    rows = np.concatenate([shard["row"] for shard in shards]).tolist()
    self.assertCountEqual(rows, range(len(prompts)))
    self.assertLen({len(shard["prompts"]) for shard in shards}, 1)

  @parameterized.named_parameters(("unset", None), ("zero", "0"))
  def test_flag_off_keeps_the_stride(self, flag):
    prompts = ["a" * k for k in (5, 1, 9, 3, 7, 2, 8, 4)]
    shards = self._shards(self._batch(prompts), 2, flag)
    for process_index, shard in enumerate(shards):
      self.assertEqual(shard["prompts"], prompts[process_index::2])
      np.testing.assert_array_equal(
          shard["row"], np.arange(8)[process_index::2]
      )

  def test_tied_prompt_lengths_give_the_same_shards_every_time(self):
    prompts = [f"prompt {i}" for i in range(8)]  # All of the same length.
    first = self._shards(self._batch(prompts), 2, flag="1")
    again = self._shards(self._batch(list(prompts)), 2, flag="1")
    self.assertEqual(
        [shard["prompts"] for shard in first],
        [shard["prompts"] for shard in again],
    )
    # Tied rows are dealt in row order.
    self.assertEqual(first[0]["prompts"], [prompts[i] for i in (0, 3, 4, 7)])
    self.assertEqual(first[1]["prompts"], [prompts[i] for i in (1, 2, 5, 6)])

  def test_prompts_without_len_are_balanced_as_length_zero(self):
    prompts = [None, "aaaa", None, "bb"]
    shards = self._shards(self._batch(prompts), 2, flag="1")
    want = rl_learner.balanced_shard_indices([0, 4, 0, 2], 2)
    self.assertEqual([shard["row"].tolist() for shard in shards], want)
    self.assertEqual(shards[0]["prompts"], [prompts[i] for i in want[0]])

  def test_uneven_batch_still_raises(self):
    with self.assertRaisesRegex(ValueError, "not divisible"):
      self._shards(self._batch(["a", "bb", "ccc"]), 2, flag="1")


if __name__ == "__main__":
  absltest.main()
