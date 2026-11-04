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

"""Tests for submission-order mini-batch emission."""

# The learner tests drive its private producer hooks directly.
# pylint: disable=protected-access

import asyncio
import queue
import random
import types
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from tunix.rl import rl_cluster as rl_engine_lib
from tunix.rl import utils as rl_utils
from tunix.rl.agentic import agentic_rl_learner
from tunix.rl.agentic import dynamic_sampling
from tunix.rl.agentic import submission_order
from tunix.rl.rollout import base_rollout


class SubmissionOrderEmitterTest(parameterized.TestCase):

  def test_in_order_completion_streams_through(self):
    em = submission_order.SubmissionOrderEmitter(window_size=2)
    self.assertEqual(em.complete(0, "a"), ["a"])
    self.assertEqual(em.complete(1, "b"), ["b"])
    self.assertEqual(em.complete(2, "c"), ["c"])
    self.assertEqual(em.held, 0)

  def test_members_of_open_window_release_in_any_order(self):
    em = submission_order.SubmissionOrderEmitter(window_size=4)
    # Group 3 is in the first window (ids 0..3) and releases at once.
    self.assertEqual(em.complete(3, "d"), ["d"])
    self.assertEqual(em.complete(1, "b"), ["b"])
    # Group 4 belongs to the second window: held until 0 and 2 are done.
    self.assertEqual(em.complete(4, "e"), [])
    self.assertEqual(em.held, 1)
    self.assertEqual(em.complete(0, "a"), ["a"])
    self.assertEqual(em.complete(2, "c"), ["c", "e"])
    self.assertEqual(em.held, 0)

  def test_fast_long_tail_does_not_jump_to_first_minibatch(self):
    # A batch of 8 in 2 mini-batches of 4; the slow groups are 0 and 1.
    em = submission_order.SubmissionOrderEmitter(window_size=4)
    out = []
    for g in (2, 3, 4, 5, 6, 7):
      out += em.complete(g, g)
    # Only 2 and 3 are first-window members; 4..7 wait for 0 and 1.
    self.assertEqual(out, [2, 3])
    self.assertEqual(em.held, 4)
    out += em.complete(1, 1)
    out += em.complete(0, 0)
    self.assertEqual(out, [2, 3, 1, 0, 4, 5, 6, 7])

  def test_dropped_group_gives_its_place_to_the_next_id(self):
    em = submission_order.SubmissionOrderEmitter(window_size=2)
    self.assertEqual(em.complete(2, "c"), [])  # Second window for now.
    self.assertEqual(em.complete(0, "a"), ["a"])
    # Dropping 1 moves 2 into the first window.
    self.assertEqual(em.drop(1), ["c"])
    self.assertEqual(em.held, 0)
    # The window slid: 3 and 4 form the next one, 5 waits.
    self.assertEqual(em.complete(5, "f"), [])
    self.assertEqual(em.complete(4, "e"), ["e"])
    self.assertEqual(em.drop(3), ["f"])

  def test_drop_of_a_completed_group_raises(self):
    em = submission_order.SubmissionOrderEmitter(window_size=2)
    self.assertEqual(em.complete(2, "c"), [])
    with self.assertRaisesRegex(ValueError, "already completed or dropped"):
      em.drop(2)

  def test_first_group_id_offset_and_double_resolution(self):
    em = submission_order.SubmissionOrderEmitter(
        window_size=2, first_group_id=10
    )
    self.assertEqual(em.complete(11, "b"), ["b"])
    self.assertEqual(em.complete(12, "c"), [])
    self.assertEqual(em.complete(10, "a"), ["a", "c"])
    with self.assertRaisesRegex(ValueError, "already completed or dropped"):
      em.complete(10, "again")
    with self.assertRaisesRegex(ValueError, "already completed or dropped"):
      em.drop(11)
    em.complete(14, "e")
    with self.assertRaisesRegex(ValueError, "already completed or dropped"):
      em.complete(14, "e")
    em.drop(16)
    with self.assertRaisesRegex(ValueError, "already completed or dropped"):
      em.drop(16)

  def test_rejects_non_positive_window(self):
    with self.assertRaisesRegex(ValueError, "window_size"):
      submission_order.SubmissionOrderEmitter(window_size=0)

  def test_flush_releases_held_in_id_order(self):
    em = submission_order.SubmissionOrderEmitter(window_size=2)
    em.complete(3, "d")
    em.complete(2, "c")
    self.assertEqual(em.flush(), ["c", "d"])
    self.assertEqual(em.held, 0)

  @parameterized.parameters(1, 2, 4, 8)
  def test_random_completion_order_yields_index_blocks(self, window):
    rng = random.Random(window)
    for _ in range(50):
      ids = list(range(window * 3 + 5))
      drops = {g for g in ids if rng.random() < 0.3}
      em = submission_order.SubmissionOrderEmitter(window_size=window)
      order = ids[:]
      rng.shuffle(order)
      out = []
      for g in order:
        out += em.drop(g) if g in drops else em.complete(g, g)
      out += em.flush()
      kept = [g for g in ids if g not in drops]
      self.assertCountEqual(out, kept)
      # Every full window is exactly the next block of kept ids.
      for m in range(len(kept) // window):
        self.assertCountEqual(
            out[m * window : (m + 1) * window],
            kept[m * window : (m + 1) * window],
        )


class _Trajectory:
  """Trajectory item stand-in that only carries its group id."""

  def __init__(self, group_id):
    self.group_id = group_id


class _Learner(agentic_rl_learner.AgenticRLLearner):
  """Scores a group as its own id; drops the ids in `drop_ids`."""

  _SUPPORTS_GROUP_FILTERS = True
  drop_ids = frozenset()
  fail_ids = frozenset()

  def _process_results(self, trajectories, mode=None, expected_step=None):
    del mode, expected_step
    gid = trajectories[0].group_id
    if gid in self.fail_ids:
      raise RuntimeError(f"scoring group {gid} failed")
    if gid in self.drop_ids:
      raise dynamic_sampling.GroupDroppedError("zero_variance")
    return [types.SimpleNamespace(gid=gid, advantages=np.array([1.0, -1.0]))]


def _make_learner(global_steps=0, max_steps=None, **algo_kwargs):
  """Returns a `_Learner` on a mock engine at `global_steps`."""
  rl_engine = mock.MagicMock()
  rl_engine.cluster_config = mock.Mock()
  rl_engine.cluster_config.rollout_engine = "generic"
  rl_engine.cluster_config.role_to_mesh = {
      rl_engine_lib.Role.ACTOR: mock.Mock(),
      rl_engine_lib.Role.ROLLOUT: mock.Mock(),
  }
  rl_engine.cluster_config.rollout_config = base_rollout.RolloutConfig(
      max_tokens_to_generate=10, return_logprobs=True
  )
  rl_engine.cluster_config.training_config.max_steps = max_steps
  rl_engine.cluster_config.training_config.rollout_micro_batch_size = None
  rl_engine.cluster_config.training_config.compute_logps_micro_batch_size = None
  rl_engine.actor_trainer = mock.Mock()
  rl_engine.actor_trainer.restored_global_step.return_value = global_steps
  rl_engine.actor_trainer.iter_steps = 0
  rl_engine.tokenizer = mock.Mock()
  with mock.patch.object(rl_utils, "is_sharing_weights", return_value=False):
    return _Learner(
        rl_engine=rl_engine,
        reward_fns=mock.Mock(),
        algo_config=agentic_rl_learner.AgenticRLConfig(
            max_response_length=10, **algo_kwargs
        ),
    )


class LearnerSubmissionOrderTest(absltest.TestCase):
  """Drives `_producer` with groups finishing out of submission order."""

  def _run_producer(
      self, learner, completion_order, in_consumer=False, error=None
  ):
    """Runs `_producer` on groups that complete in `completion_order`.

    Args:
      learner: The learner.
      completion_order: Group ids in completion order.
      in_consumer: Whether groups are scored in the consumer.
      error: The exception type that `_producer` is expected to raise.

    Returns:
      The group ids of the queued items, in queue order.
    """
    learner._full_batch_size = 8
    learner._process_in_consumer = in_consumer
    if learner.algo_config.group_filters:
      learner._dynamic_sampler = mock.Mock(expected_step=0)

    async def fake_stream(**kwargs):
      del kwargs
      for gid in completion_order:
        yield [_Trajectory(gid), _Trajectory(gid)]

    learner._orchestrator_producer = fake_stream
    train_data_queue = queue.Queue()
    producer = learner._producer(
        orchestrator=mock.Mock(),
        prompt_feed=mock.Mock(),
        train_data_queue=train_data_queue,
    )
    if error is None:
      asyncio.run(producer)
    else:
      with self.assertRaises(error):
        asyncio.run(producer)
    out = []
    while (item := train_data_queue.get_nowait()) is not None:
      out.append(item[0].group_id if in_consumer else item.gid)
    return out

  def test_completion_order_without_emitter(self):
    learner = _make_learner(minibatch_order="completion")
    self.assertIsNone(learner._start_submission_order(8, 4))
    out = self._run_producer(learner, [2, 7, 5, 0, 1, 3, 6, 4])
    self.assertEqual(out, [2, 7, 5, 0, 1, 3, 6, 4])

  def test_completion_order_is_the_default(self):
    self.assertIsNone(_make_learner()._start_submission_order(8, 4))

  def test_no_emitter_with_one_optimizer_step_per_batch(self):
    learner = _make_learner(minibatch_order="submission")
    self.assertIsNone(learner._start_submission_order(8, 8))

  def test_minibatch_zero_streams_to_queue_before_later_minibatches_finish(
      self,
  ):
    learner = _make_learner(minibatch_order="submission")
    learner._full_batch_size = 6
    learner._submission_emitter = learner._start_submission_order(6, 2)
    train_data_queue = queue.Queue()
    emitted_per_yield = []

    async def fake_stream(**kwargs):
      del kwargs
      for gid in (1, 0, 4, 3, 2, 5):
        yield [_Trajectory(gid), _Trajectory(gid)]
        ids = []
        while not train_data_queue.empty():
          ids.append(train_data_queue.get_nowait().gid)
        emitted_per_yield.append(ids)

    learner._orchestrator_producer = fake_stream
    asyncio.run(
        learner._producer(
            orchestrator=mock.Mock(),
            prompt_feed=mock.Mock(),
            train_data_queue=train_data_queue,
        )
    )
    # Mini-batch 0 ({0, 1}) reaches `train_data_queue` immediately upon each
    # member's completion, before any group in mini-batch 1 ({2, 3}) or 2
    # ({4, 5}) starts or finishes; group 4 waits until group 2 closes batch 1.
    self.assertEqual(emitted_per_yield, [[1], [0], [], [3], [2, 4], [5]])
    self.assertIsNone(train_data_queue.get_nowait())

  def test_submission_order_minibatches_with_drops(self):
    learner = _make_learner(
        group_filters=["zero_variance"], minibatch_order="submission"
    )
    learner._submission_emitter = learner._start_submission_order(8, 4)
    learner.drop_ids = frozenset({1, 6})
    # Long groups 0 and 4 finish last; 8 and 9 replace 1 and 6.
    out = self._run_producer(learner, [2, 7, 5, 1, 3, 6, 9, 8, 0, 4])
    # Mini-batch 1 is the first four kept ids, whatever finished first.
    self.assertCountEqual(out[:4], [0, 2, 3, 4])
    self.assertCountEqual(out[4:], [5, 7, 8, 9])
    # Members of an open window are not held back by slower members.
    self.assertEqual(out[:2], [2, 3])
    self.assertEqual(learner._dynamic_sampler.on_group_dropped.call_count, 2)

  def test_held_groups_are_released_when_the_stream_ends(self):
    learner = _make_learner(minibatch_order="submission")
    learner._submission_emitter = learner._start_submission_order(8, 2)
    # Group 0 never completes: 2 and 3 are held, then flushed at the end.
    self.assertEqual(self._run_producer(learner, [1, 3, 2]), [1, 2, 3])

  def test_held_groups_are_not_released_when_the_producer_fails(self):
    learner = _make_learner(minibatch_order="submission")
    learner._submission_emitter = learner._start_submission_order(8, 2)
    learner.fail_ids = frozenset({4})
    out = self._run_producer(learner, [1, 3, 2, 4], error=RuntimeError)
    # 3 stays held behind the missing group 0.
    self.assertEqual(out, [1])

  def test_raw_groups_scored_in_the_consumer_follow_submission_order(self):
    learner = _make_learner(minibatch_order="submission")
    learner._submission_emitter = learner._start_submission_order(8, 2)
    out = self._run_producer(learner, [1, 3, 2, 0], in_consumer=True)
    self.assertEqual(out, [1, 0, 2, 3])

  def test_resume_numbers_groups_from_the_restored_step(self):
    learner = _make_learner(global_steps=3, minibatch_order="submission")
    emitter = learner._start_submission_order(8, 4)
    self.assertEqual(emitter.complete(24, "a"), ["a"])
    self.assertEqual(emitter.complete(28, "e"), [])

  def test_rejects_unknown_minibatch_order(self):
    with self.assertRaisesRegex(ValueError, "minibatch_order"):
      _make_learner(minibatch_order="random")

  def test_post_max_steps_skip_waits_for_held_submission_order_groups(self):
    learner = _make_learner(
        max_steps=1,
        group_filters=["zero_variance"],
        minibatch_order="submission",
    )
    learner._full_batch_size = 2
    learner._submission_emitter = learner._start_submission_order(2, 1)
    kept = 0

    def on_group_kept():
      nonlocal kept
      kept += 1
      sampler.expected_step = kept // 2

    sampler = mock.Mock(expected_step=0, on_group_kept=on_group_kept)
    learner._dynamic_sampler = sampler

    async def fake_stream(**kwargs):
      del kwargs
      # Lookahead groups 2 and 3 finish before step 0's groups 0 and 1; group 4
      # arrives after 0 and 1 have released the held groups.
      for gid in (2, 3, 0, 1, 4):
        yield [_Trajectory(gid), _Trajectory(gid)]

    learner._orchestrator_producer = fake_stream
    learner.fail_ids = frozenset({4})  # Must be skipped without scoring.
    train_data_queue = queue.Queue()
    asyncio.run(
        learner._producer(
            orchestrator=mock.Mock(),
            prompt_feed=sampler,
            train_data_queue=train_data_queue,
        )
    )
    out = []
    while (item := train_data_queue.get_nowait()) is not None:
      out.append(item.gid)
    self.assertEqual(out[:2], [0, 1])


if __name__ == "__main__":
  absltest.main()
