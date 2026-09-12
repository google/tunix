"""Orchestrator contract tests; no model or TPU invocation."""

import asyncio
from pathlib import Path
import tempfile
import threading
import types
import unittest

import numpy as np

from examples.frozenlake.gemma4_tim import admission, recipe


class FakeLearner:
  def __init__(self, loop, mutation=None):
    self.loop, self.mutation = loop, mutation
    self.rl_cluster = types.SimpleNamespace(actor_trainer=types.SimpleNamespace(train_steps=0), global_steps=0)

  def _create_micro_batch_iterator(self, dataset, size):
    assert list(dataset) == ["first-batch"] and size == 1
    return iter(range(64))

  def _num_generations(self):
    return 8

  def _build_orchestrator(self):
    return object()

  async def _orchestrator_producer(self, orchestrator, prompts, num_generations):
    assert len(prompts) == 64 and num_generations == 8
    for group_id in range(64):
      group = [types.SimpleNamespace(group_id=group_id, pair_index=i, traj={
          "conversation_tokens": np.array([3, 0, 4], np.int32),
          "conversation_masks": np.array([1, 0, 1], np.int32),
          "old_logprobs": np.array([-1., 0., -2.], np.float32),
          "prompt_tokens": np.array([0, 2], np.int32), "prompt_length": 1,
          "policy_version": 0, "trajectory_reward": float(i % 2),
      }) for i in range(8)]
      if self.mutation and group_id == 0:
        self.mutation(group, self)
      yield group


class AdmissionTest(unittest.TestCase):
  def setUp(self):
    self.loop = asyncio.new_event_loop()
    self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
    self.thread.start()

  def tearDown(self):
    self.loop.call_soon_threadsafe(self.loop.stop)
    self.thread.join(timeout=5)
    self.assertFalse(self.thread.is_alive())
    self.loop.close()

  def test_full_512_rollouts_are_persisted_without_optimizer(self):
    with tempfile.TemporaryDirectory() as temp:
      output = Path(temp)
      result = admission.stock_admission(FakeLearner(self.loop), ["first-batch"], output)
      self.assertEqual(result["trajectories"], 512)
      self.assertEqual(result["optimizer_updates"], 0)
      self.assertEqual(len(list(output.glob("admission-group-*.npz"))), 64)
      with np.load(output / "admission-group-000.npz", allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["generation_0_actions"], [True, False, True])
        self.assertEqual(int(arrays["generation_0_prompt_length"]), 1)

  def test_coverage_duplicate_stale_mask_and_update_negatives(self):
    mutations = (
        lambda group, learner: group.pop(),
        lambda group, learner: setattr(group[0], "pair_index", 1),
        lambda group, learner: setattr(group[0], "group_id", 10),
        lambda group, learner: group[0].traj.update(policy_version=1),
        lambda group, learner: group[0].traj.update(conversation_masks=np.array([2, 0, 0])),
        lambda group, learner: group[0].traj.update(trajectory_reward=float("nan")),
        lambda group, learner: setattr(learner.rl_cluster, "global_steps", 1),
    )
    for mutation in mutations:
      with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
        with self.assertRaises(recipe.RecipeError):
          admission.stock_admission(FakeLearner(self.loop, mutation), ["first-batch"], Path(temp))
        self.assertFalse((Path(temp) / "stock-admission.json").exists())


if __name__ == "__main__":
  unittest.main()
