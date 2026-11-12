# Copyright 2025 Google LLC
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

import dataclasses
import gc
import inspect
import multiprocessing
import time
from typing import Any, List
from unittest import mock
import warnings

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import numpy.testing as npt
from tunix.rl import algorithm_config as algo_config_lib
from tunix.rl import reward_manager


# --- Test Reward Functions ---
def len_reward(
    prompts: List[str], completions: List[str], **kwargs: Any
) -> List[float]:
  del prompts, kwargs  # Unused
  res = [float(len(c)) for c in completions]
  return res


len_reward.__name__ = "len_reward"


def prompt_len_reward(
    prompts: List[str],
    completions: List[str],
    custom_param: float = 1.0,
    **kwargs: Any,
) -> List[float]:
  del completions, kwargs  # Unused
  res = [custom_param * len(p) for p in prompts]
  return res


prompt_len_reward.__name__ = "prompt_len_reward"


def nan_reward(
    prompts: List[str], completions: List[str], **kwargs: Any
) -> List[float]:
  del completions, kwargs  # Unused
  return [np.nan] * len(prompts)


nan_reward.__name__ = "nan_reward"


def _step_value(log_metrics: dict[str, Any], name: str) -> float:
  """Reduces a `(values, agg_fn)` log metric the way the step loggers do."""
  values, agg_fn = log_metrics[name]
  return float(agg_fn(np.asarray(values)))


@dataclasses.dataclass(slots=True, kw_only=True)
class TestAlgoConfig(algo_config_lib.AlgorithmConfig):
  """Test Algorithm Config."""

  reward_mode: str = "reward_fn"
  custom_param: float = 2.0


# --- Test Class ---
class SequenceRewardManagerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.test_algo_config = TestAlgoConfig()
    self.prompts = ["p1", "p22"]
    self.completions = ["c1_long", "c2"]

  def test_initialization(self):
    manager = reward_manager.SequenceRewardManager(
        reward_fns=len_reward,
        algo_config=self.test_algo_config,
    )
    self.assertEqual(manager.reward_fns, [len_reward])
    self.assertEqual(manager.algo_config, self.test_algo_config)

  def test_single_reward_fn(self):
    manager = reward_manager.SequenceRewardManager(
        reward_fns=[len_reward],
        algo_config=self.test_algo_config,
    )
    rewards_info = manager(
        self.prompts,
        self.completions,
    )

    expected_rewards = np.array([float(len("c1_long")), float(len("c2"))])
    np.testing.assert_array_equal(rewards_info["rewards"], expected_rewards)
    self.assertLen(rewards_info["log_metrics"], 9)

  def test_multiple_reward_fns(self):
    manager = reward_manager.SequenceRewardManager(
        reward_fns=[len_reward, prompt_len_reward],
        algo_config=self.test_algo_config,
    )
    rewards_info = manager(
        self.prompts,
        self.completions,
    )

    # custom_param is 2.0 from test_algo_config
    r1 = np.array(len_reward(self.prompts, self.completions))
    r2 = np.array(
        prompt_len_reward(self.prompts, self.completions, custom_param=2.0)
    )
    expected_rewards = r1 + r2
    np.testing.assert_array_almost_equal(
        rewards_info["rewards"], expected_rewards
    )
    test_metrics = rewards_info["log_metrics"]
    for metric_name, v in test_metrics.items():
      if metric_name.startswith("rewards/"):
        self.assertLen(v[0], 2)
    npt.assert_allclose(
        test_metrics["rewards/sum"][0],
        expected_rewards,
        err_msg="rewards/sum mismatch",
    )
    npt.assert_allclose(
        test_metrics["rewards/len_reward"][0],
        r1,
        err_msg="rewards/len_reward mismatch",
    )
    npt.assert_allclose(
        test_metrics["rewards/prompt_len_reward"][0],
        r2,
        err_msg="rewards/prompt_len_reward mismatch",
    )
    # `rewards/{mean,std,min,max,sum}` summarize total rewards over the step.
    self.assertAlmostEqual(
        _step_value(test_metrics, "rewards/mean"), np.mean(expected_rewards)
    )
    self.assertAlmostEqual(
        _step_value(test_metrics, "rewards/std"), np.std(expected_rewards)
    )
    self.assertAlmostEqual(
        _step_value(test_metrics, "rewards/min"), np.min(expected_rewards)
    )
    self.assertAlmostEqual(
        _step_value(test_metrics, "rewards/max"), np.max(expected_rewards)
    )
    self.assertAlmostEqual(
        _step_value(test_metrics, "rewards/sum"), np.sum(expected_rewards)
    )
    self.assertEqual(_step_value(test_metrics, "rewards/nonfinite_frac"), 0.0)

  def test_algo_config_param_passing(self):
    # Mock the reward function to spy on its call arguments
    mock_fn = mock.Mock(wraps=prompt_len_reward)
    mock_fn.__name__ = prompt_len_reward.__name__
    # Restore the signature for introspection
    mock_fn.__signature__ = inspect.signature(prompt_len_reward)

    manager = reward_manager.SequenceRewardManager(
        reward_fns=[mock_fn],
        algo_config=self.test_algo_config,
    )
    manager(
        self.prompts,
        self.completions,
    )

    mock_fn.assert_called_once()
    _, kwargs = mock_fn.call_args
    self.assertEqual(kwargs["custom_param"], 2.0)
    self.assertNotIn(
        "another_param", kwargs
    )  # Not in prompt_len_reward signature

  def test_nan_handling(self):
    manager = reward_manager.SequenceRewardManager(
        reward_fns=[len_reward, nan_reward],
        algo_config=self.test_algo_config,
    )
    rewards_info = manager(
        self.prompts,
        self.completions,
    )
    # np.nansum should treat nan as 0 for summation
    expected_rewards = np.array([float(len(c)) for c in self.completions])
    np.testing.assert_array_almost_equal(
        rewards_info["rewards"], expected_rewards
    )
    # Check logged metrics for NaN
    test_metrics = rewards_info["log_metrics"]
    self.assertTrue(np.isnan(test_metrics["rewards/nan_reward"][0]).all())
    np.testing.assert_allclose(
        test_metrics["rewards/sum"][0],
        expected_rewards,
        err_msg="rewards/sum mismatch",
    )

  def test_partial_nan_reward_fn_does_not_poison_step_metrics(self):
    def partial_nan(prompts: List[str], completions: List[str]) -> List[float]:
      del prompts, completions
      return [1.0, np.nan]

    manager = reward_manager.SequenceRewardManager(
        reward_fns=[len_reward, partial_nan],
        algo_config=self.test_algo_config,
    )
    log_metrics = manager(self.prompts, self.completions)["log_metrics"]
    # A NaN reward-function score counts as 0 in the total, so both
    # trajectories stay finite: totals are [7 + 1, 2].
    self.assertEqual(_step_value(log_metrics, "rewards/min"), 2.0)
    self.assertEqual(_step_value(log_metrics, "rewards/max"), 8.0)
    self.assertEqual(_step_value(log_metrics, "rewards/nonfinite_frac"), 0.0)
    # The per-function series skips the NaN score.
    self.assertEqual(_step_value(log_metrics, "rewards/partial_nan"), 1.0)
    manager.close()

  @parameterized.named_parameters(
      dict(
          testcase_name="reward_fn_returns_none",
          reward_fns=[lambda prompts, completions, **kw: None],
          expected_regex="Failed to obtain result.*Result is None",
          error_type=RuntimeError,
      ),
      dict(
          testcase_name="reward_fn_bad_length",
          reward_fns=[
              lambda prompts, completions, **kw: [1.0] * (len(prompts) + 1)
          ],
          expected_regex="Length mismatch",
          error_type=RuntimeError,
      ),
  )
  def test_errors(
      self, expected_regex, error_type, kwargs=None, reward_fns=None
  ):
    if reward_fns is None:
      reward_fns = [len_reward]
    for i, fn in enumerate(reward_fns):
      if not hasattr(fn, "__name__"):
        fn.__name__ = f"test_fn_{i}"

    manager = reward_manager.SequenceRewardManager(
        reward_fns=reward_fns,
        algo_config=self.test_algo_config,
    )
    with self.assertRaisesRegex(error_type, expected_regex):
      manager(
          self.prompts,
          self.completions,
          **(kwargs or {}),
      )

  @parameterized.parameters(([],), (None,))
  def test_no_reward_fns_raises_error(self, empty_fns):
    with self.assertRaisesRegex(ValueError, "reward_fns cannot be empty"):
      reward_manager.SequenceRewardManager(
          reward_fns=empty_fns,
          algo_config=self.test_algo_config,
      )

  @parameterized.parameters(([None],), ([len_reward, None],), ("not_a_fn",))
  def test_non_callable_reward_fns_raises_type_error(self, invalid_fns):
    with self.assertRaisesRegex(
        TypeError, "All items in reward_fns must be callable"
    ):
      reward_manager.SequenceRewardManager(
          reward_fns=invalid_fns,
          algo_config=self.test_algo_config,
      )

  def test_reward_fn_without_var_keyword_ignores_extra_kwargs(self):
    def strict_two_arg_reward(prompts: List[str], completions: List[str]):
      return [float(len(p) + len(c)) for p, c in zip(prompts, completions)]

    manager = reward_manager.SequenceRewardManager(
        reward_fns=[strict_two_arg_reward],
        algo_config=self.test_algo_config,
    )
    rewards_info = manager(
        self.prompts,
        self.completions,
        mode="train",
        answer=["42", "42"],
        trajectory_rewards=[1.0, 2.0],
    )
    npt.assert_allclose(rewards_info["rewards"], [2.0 + 7.0, 3.0 + 2.0])
    manager.close()

  def test_dataset_env_rewards_kwarg_forwarded_to_reward_fn(self):
    def uses_env_rewards_col(
        prompts: List[str], completions: List[str], env_rewards: List[float]
    ) -> List[float]:
      del prompts, completions
      return [x * 0.5 for x in env_rewards]

    manager = reward_manager.SequenceRewardManager(
        reward_fns=[uses_env_rewards_col],
        algo_config=self.test_algo_config,
    )
    res = manager(self.prompts, self.completions, env_rewards=[100.0, 200.0])
    npt.assert_allclose(res["rewards"], [50.0, 100.0])
    self.assertNotIn("rewards/env", res["log_metrics"])
    manager.close()

  def test_compute_rewards_positional_and_kwargs_compatibility(self):
    manager = reward_manager.SequenceRewardManager(
        reward_fns=[answer_reward],
        algo_config=self.test_algo_config,
    )
    res = manager._compute_rewards(  # pylint: disable=protected-access
        self.prompts, self.completions, answer=["3.0", "5.0"]
    )
    npt.assert_allclose(res["rewards"], [3.0, 5.0])
    manager.close()

  @parameterized.parameters("sum", "mean", "min", "max")
  def test_reward_fn_named_after_summary_metric_raises(self, name):
    def fn(prompts: List[str], completions: List[str]) -> List[float]:
      del prompts, completions
      return [1.0, 2.0]

    fn.__name__ = name
    with self.assertRaisesRegex(ValueError, "reserved"):
      reward_manager.SequenceRewardManager(
          reward_fns=[len_reward, fn],
          algo_config=self.test_algo_config,
      )


class AgenticSequenceRewardManagerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.test_algo_config = TestAlgoConfig()
    self.prompts = ["p1", "p22"]
    self.completions = ["c1_long", "c2"]

  def test_log_metrics_non_interference_with_reward_fns(self):
    manager = reward_manager.AgenticSequenceRewardManager(
        reward_fns=[len_reward],
        algo_config=self.test_algo_config,
    )
    traj_rewards = [10.0, 20.0]
    rewards_info = manager(
        self.prompts, self.completions, trajectory_rewards=traj_rewards
    )
    log_metrics = rewards_info["log_metrics"]
    self.assertFalse(
        any(k.startswith("trajectory_rewards/") for k in log_metrics)
    )
    npt.assert_allclose(rewards_info["rewards"], [17.0, 22.0])
    npt.assert_allclose(log_metrics["rewards/sum"][0], [17.0, 22.0])
    self.assertEqual(_step_value(log_metrics, "rewards/sum"), 39.0)
    self.assertEqual(_step_value(log_metrics, "rewards/mean"), 19.5)
    self.assertEqual(_step_value(log_metrics, "rewards/min"), 17.0)
    self.assertEqual(_step_value(log_metrics, "rewards/max"), 22.0)
    self.assertEqual(_step_value(log_metrics, "rewards/env"), 15.0)
    self.assertEqual(_step_value(log_metrics, "rewards/len_reward"), 4.5)
    manager.close()

  def test_log_metrics_non_interference_no_reward_fns(self):
    manager = reward_manager.AgenticSequenceRewardManager(
        reward_fns=None,
        algo_config=self.test_algo_config,
    )
    traj_rewards = [2.0, 6.0]
    rewards_info = manager(
        self.prompts, self.completions, trajectory_rewards=traj_rewards
    )
    log_metrics = rewards_info["log_metrics"]
    self.assertFalse(
        any(k.startswith("trajectory_rewards/") for k in log_metrics)
    )
    self.assertNotIn("rewards/env", log_metrics)
    npt.assert_allclose(rewards_info["rewards"], [2.0, 6.0])
    self.assertEqual(_step_value(log_metrics, "rewards/sum"), 8.0)
    self.assertEqual(_step_value(log_metrics, "rewards/mean"), 4.0)
    self.assertEqual(_step_value(log_metrics, "rewards/std"), 2.0)
    self.assertEqual(_step_value(log_metrics, "rewards/min"), 2.0)
    self.assertEqual(_step_value(log_metrics, "rewards/max"), 6.0)
    manager.close()

  def test_empty_batch_raises(self):
    manager = reward_manager.AgenticSequenceRewardManager(
        reward_fns=None,
        algo_config=self.test_algo_config,
    )
    with self.assertRaisesRegex(ValueError, "empty batch"):
      manager([], [], trajectory_rewards=[])
    with self.assertRaisesRegex(ValueError, "empty batch"):
      manager(np.array([]), [], trajectory_rewards=[])
    manager.close()

  def test_trajectory_rewards_shape_mismatch_raises(self):
    manager = reward_manager.AgenticSequenceRewardManager(
        reward_fns=None,
        algo_config=self.test_algo_config,
    )
    with self.assertRaisesRegex(ValueError, "trajectory_rewards"):
      manager(self.prompts, self.completions, trajectory_rewards=[1.0])
    manager.close()

  def test_numpy_ndarray_prompts(self):
    manager = reward_manager.AgenticSequenceRewardManager(
        reward_fns=None,
        algo_config=self.test_algo_config,
    )
    res = manager(
        np.array(self.prompts), self.completions, trajectory_rewards=[2.0, 6.0]
    )
    npt.assert_allclose(res["rewards"], [2.0, 6.0])
    manager.close()

  def test_nan_trajectory_rewards_propagate(self):
    pure_manager = reward_manager.AgenticSequenceRewardManager(
        reward_fns=None,
        algo_config=self.test_algo_config,
    )
    pure_res = pure_manager(
        self.prompts, self.completions, trajectory_rewards=[np.nan, 1.0]
    )
    self.assertTrue(np.isnan(pure_res["rewards"][0]))
    npt.assert_allclose(pure_res["rewards"][1:], [1.0])
    pure_manager.close()

    hybrid_manager = reward_manager.AgenticSequenceRewardManager(
        reward_fns=[len_reward],
        algo_config=self.test_algo_config,
    )
    hybrid_res = hybrid_manager(
        self.prompts, self.completions, trajectory_rewards=[np.nan, 1.0]
    )
    self.assertTrue(np.isnan(hybrid_res["rewards"][0]))
    npt.assert_allclose(hybrid_res["rewards"][1:], [3.0])
    hybrid_manager.close()

  def test_nan_env_reward_trajectory_excluded_from_step_metrics(self):
    manager = reward_manager.AgenticSequenceRewardManager(
        reward_fns=[len_reward],
        algo_config=self.test_algo_config,
    )
    log_metrics = manager(
        self.prompts, self.completions, trajectory_rewards=[np.nan, 1.0]
    )["log_metrics"]
    # Only the second trajectory (total 1 + 2) is finite; the NaN one is
    # excluded rather than turning the step's metrics into NaN.
    for name in ("mean", "min", "max", "sum"):
      self.assertEqual(_step_value(log_metrics, f"rewards/{name}"), 3.0)
    self.assertEqual(_step_value(log_metrics, "rewards/std"), 0.0)
    self.assertEqual(_step_value(log_metrics, "rewards/nonfinite_frac"), 0.5)
    self.assertEqual(_step_value(log_metrics, "rewards/env"), 1.0)
    self.assertEqual(_step_value(log_metrics, "rewards/len_reward"), 2.0)
    manager.close()

  def test_all_nonfinite_env_rewards_report_nan_summary(self):
    manager = reward_manager.AgenticSequenceRewardManager(
        reward_fns=None,
        algo_config=self.test_algo_config,
    )
    log_metrics = manager(
        self.prompts, self.completions, trajectory_rewards=[np.nan, np.inf]
    )["log_metrics"]
    self.assertEqual(_step_value(log_metrics, "rewards/nonfinite_frac"), 1.0)
    with warnings.catch_warnings():
      warnings.simplefilter("ignore", RuntimeWarning)
      self.assertTrue(np.isnan(_step_value(log_metrics, "rewards/mean")))
      self.assertTrue(np.isnan(_step_value(log_metrics, "rewards/max")))
    manager.close()

  def test_reward_fn_named_sum_raises(self):
    def sum(prompts: List[str], completions: List[str]) -> List[float]:  # pylint: disable=redefined-builtin
      del prompts, completions
      return [7.0, 2.0]

    with self.assertRaisesRegex(ValueError, "reserved"):
      reward_manager.AgenticSequenceRewardManager(
          reward_fns=[sum],
          algo_config=self.test_algo_config,
      )

  def test_dataset_env_rewards_kwarg_forwarded_to_reward_fn(self):
    def uses_env_rewards_col(
        prompts: List[str], completions: List[str], env_rewards: List[float]
    ) -> List[float]:
      del prompts, completions
      return [x * 0.5 for x in env_rewards]

    manager = reward_manager.AgenticSequenceRewardManager(
        reward_fns=[uses_env_rewards_col],
        algo_config=self.test_algo_config,
    )
    res = manager(
        self.prompts,
        self.completions,
        trajectory_rewards=[1.0, 2.0],
        env_rewards=[100.0, 200.0],
    )
    npt.assert_allclose(res["rewards"], [51.0, 102.0])
    npt.assert_allclose(res["log_metrics"]["rewards/env"][0], [1.0, 2.0])
    manager.close()

  def test_reserved_env_reward_fn_name_raises(self):
    def env(prompts: List[str], completions: List[str]) -> List[float]:
      del completions
      return [1.0] * len(prompts)

    with self.assertRaisesRegex(ValueError, "reserved"):
      reward_manager.AgenticSequenceRewardManager(
          reward_fns=[env],
          algo_config=self.test_algo_config,
      )

  def test_empty_list_reward_fns_env_mode(self):
    manager = reward_manager.AgenticSequenceRewardManager(
        reward_fns=[],
        algo_config=self.test_algo_config,
    )
    self.assertEqual(manager.reward_fns, [])
    traj_rewards = [1.0, 2.0]
    rewards_info = manager(
        self.prompts, self.completions, trajectory_rewards=traj_rewards
    )
    npt.assert_allclose(rewards_info["rewards"], [1.0, 2.0])
    self.assertNotIn("rewards/env", rewards_info["log_metrics"])
    manager.close()

  def test_int_and_ndarray_trajectory_rewards_not_mutated_with_reward_fns(self):
    manager = reward_manager.AgenticSequenceRewardManager(
        reward_fns=[len_reward],
        algo_config=self.test_algo_config,
    )
    int_rewards = [1, 2]
    res_int = manager(
        self.prompts, self.completions, trajectory_rewards=int_rewards
    )
    npt.assert_allclose(res_int["rewards"], [1.0 + 7.0, 2.0 + 2.0])

    arr_rewards = np.array([10.0, 20.0])
    res_arr = manager(
        self.prompts, self.completions, trajectory_rewards=arr_rewards
    )
    npt.assert_allclose(res_arr["rewards"], [17.0, 22.0])
    npt.assert_allclose(arr_rewards, [10.0, 20.0])
    manager.close()


def answer_reward(
    prompts: List[str],
    completions: List[str],
    answer: List[str],
    **kwargs: Any,
) -> List[float]:
  del completions, kwargs  # Unused
  assert len(answer) == len(prompts)
  return [float(a) for a in answer]


answer_reward.__name__ = "answer_reward"


def _noop():
  pass


def child_spawning_reward(
    prompts: List[str], completions: List[str], **kwargs: Any
) -> List[float]:
  """Spawns a subprocess of its own, which a daemonic pool worker forbids."""
  del completions, kwargs  # Unused
  ctx = multiprocessing.get_context("forkserver")
  p = ctx.Process(target=_noop)
  p.start()
  p.join()
  return [7.0] * len(prompts)


child_spawning_reward.__name__ = "child_spawning_reward"


def slow_reward(
    prompts: List[str], completions: List[str], **kwargs: Any
) -> List[float]:
  """Takes longer than the (tiny) worker timeout used in the timeout test."""
  del completions, kwargs  # Unused
  time.sleep(0.5)
  return [3.0] * len(prompts)


slow_reward.__name__ = "slow_reward"


@absltest.skipIf(
    reward_manager._fork_context() is None,  # pylint: disable=protected-access
    "parallel reward evaluation requires the forkserver start method",
)
class ParallelSequenceRewardManagerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.serial_config = TestAlgoConfig()
    self.parallel_config = TestAlgoConfig(reward_num_workers=4)
    self.prompts = [f"p{i}" for i in range(10)]
    self.completions = [f"c{i}" * (i + 1) for i in range(10)]
    self.answers = [str(float(i)) for i in range(10)]
    self._managers = []

  def tearDown(self):
    for m in self._managers:
      m.close()
    super().tearDown()

  def _make(self, reward_fns, config):
    manager = reward_manager.SequenceRewardManager(
        reward_fns=reward_fns, algo_config=config
    )
    self._managers.append(manager)
    return manager

  def test_parallel_matches_serial(self):
    fns = [len_reward, prompt_len_reward, answer_reward, nan_reward]
    serial = self._make(fns, self.serial_config)(
        self.prompts, self.completions, answer=self.answers
    )
    parallel = self._make(fns, self.parallel_config)(
        self.prompts, self.completions, answer=self.answers
    )
    np.testing.assert_array_equal(serial["rewards"], parallel["rewards"])
    for name in (
        "rewards/len_reward",
        "rewards/prompt_len_reward",
        "rewards/answer_reward",
    ):
      np.testing.assert_array_equal(
          serial["log_metrics"][name][0],
          parallel["log_metrics"][name][0],
          err_msg=f"{name} mismatch",
      )

  def test_per_example_kwargs_sliced_in_order(self):
    manager = self._make([answer_reward], self.parallel_config)
    rewards_info = manager(self.prompts, self.completions, answer=self.answers)
    np.testing.assert_array_equal(
        rewards_info["rewards"], np.array([float(i) for i in range(10)])
    )

  def test_unpicklable_fn_evaluated_in_parent(self):
    unpicklable = lambda prompts, completions, **kw: [2.0] * len(prompts)
    unpicklable.__name__ = "unpicklable_fn"
    manager = self._make([unpicklable], self.parallel_config)
    rewards_info = manager(self.prompts, self.completions)
    np.testing.assert_array_equal(rewards_info["rewards"], np.full(10, 2.0))
    self.assertIn("unpicklable_fn", manager._parent_only_fns)

  def test_subprocess_spawning_fn_evaluated_in_parent(self):
    manager = self._make([child_spawning_reward], self.parallel_config)
    rewards_info = manager(self.prompts, self.completions)
    np.testing.assert_array_equal(rewards_info["rewards"], np.full(10, 7.0))
    self.assertIn("child_spawning_reward", manager._parent_only_fns)
    # Subsequent calls still succeed (fn now runs in the parent).
    rewards_info = manager(self.prompts, self.completions)
    np.testing.assert_array_equal(rewards_info["rewards"], np.full(10, 7.0))

  def test_minus_one_uses_one_worker_per_cpu(self):
    # Pin the CPU count: the point is the resolution rule, and a real
    # many-core test machine must not spawn one worker per core here.
    with mock.patch.object(reward_manager.os, "cpu_count", return_value=3):
      manager = self._make([len_reward], TestAlgoConfig(reward_num_workers=-1))
      rewards_info = manager(self.prompts, self.completions)
      self.assertEqual(manager._pool._processes, 3)
    expected = np.array([float(len(c)) for c in self.completions])
    np.testing.assert_array_equal(rewards_info["rewards"], expected)

  def test_empty_batch_raises(self):
    manager = self._make([len_reward], self.parallel_config)
    with self.assertRaisesRegex(ValueError, "empty batch"):
      manager([], [])
    self.assertIsNone(manager._pool)

  def test_zero_workers_creates_no_pool(self):
    manager = self._make([len_reward], self.serial_config)
    manager(self.prompts, self.completions)
    self.assertIsNone(manager._pool)

  def test_default_is_serial(self):
    self.assertEqual(algo_config_lib.AlgorithmConfig().reward_num_workers, 0)

  def test_worker_timeout_is_configurable(self):
    config = TestAlgoConfig(
        reward_num_workers=2, reward_worker_timeout_seconds=0.01
    )
    manager = self._make([slow_reward], config)
    rewards_info = manager(self.prompts, self.completions)
    # The chunk timed out in the worker, so the fn was evaluated in the
    # parent and is parent-only from now on; the result is still correct.
    np.testing.assert_array_equal(rewards_info["rewards"], np.full(10, 3.0))
    self.assertIn("slow_reward", manager._parent_only_fns)

  def test_dropped_manager_terminates_its_workers(self):
    manager = reward_manager.SequenceRewardManager(
        reward_fns=[len_reward], algo_config=self.parallel_config
    )
    manager(self.prompts, self.completions)
    workers = list(manager._pool._pool)  # pylint: disable=protected-access
    self.assertTrue(all(w.is_alive() for w in workers))
    del manager
    gc.collect()
    for w in workers:
      w.join(timeout=5)
    self.assertFalse(any(w.is_alive() for w in workers))

  def test_invalid_timeout_rejected(self):
    with self.assertRaisesRegex(ValueError, "reward_worker_timeout_seconds"):
      TestAlgoConfig(reward_worker_timeout_seconds=0)

  def test_reward_components_returned_for_sequence_and_env_managers(self):
    seq_mgr = reward_manager.SequenceRewardManager(
        reward_fns=[len_reward], algo_config=TestAlgoConfig()
    )
    out = seq_mgr(["p0", "p1"], ["ab", "abcd"])
    self.assertIn("reward_components", out)
    np.testing.assert_array_equal(
        out["reward_components"]["len_reward"], np.array([2.0, 4.0])
    )

    env_mgr = reward_manager.AgenticSequenceRewardManager(
        reward_fns=[len_reward], algo_config=TestAlgoConfig(reward_mode="env")
    )
    env_out = env_mgr(
        ["p0", "p1"], ["ab", "abcd"], trajectory_rewards=[1.0, 0.0]
    )
    np.testing.assert_array_equal(
        env_out["reward_components"]["env"], np.array([1.0, 0.0])
    )
    np.testing.assert_array_equal(
        env_out["reward_components"]["len_reward"], np.array([2.0, 4.0])
    )
    np.testing.assert_array_equal(env_out["rewards"], np.array([3.0, 4.0]))

  def test_un_deepcopyable_group_filter_and_duplicate_lambda_reward_names(self):
    import threading

    class LockFilter:

      def __init__(self):
        self.lock = threading.Lock()

      def __call__(self, rewards, valid_mask=None):
        del rewards, valid_mask
        with self.lock:
          return False

    cfg = algo_config_lib.GRPOConfig(
        num_generations=2, group_filters=[LockFilter()]
    )
    mgr = reward_manager.SequenceRewardManager(
        reward_fns=[
            lambda prompts, completions, **_: [1.0, 0.0],
            lambda prompts, completions, **_: [-0.1, -0.1],
        ],
        algo_config=cfg,
    )
    out = mgr(["p0", "p1"], ["a", "bb"])
    # Duplicate `<lambda>` names must not overwrite index 0's primary task reward!
    self.assertEqual(
        list(out["reward_components"].keys()), ["<lambda>", "<lambda>_1"]
    )
    np.testing.assert_array_equal(
        out["reward_components"]["<lambda>"], np.array([1.0, 0.0])
    )


if __name__ == "__main__":
  absltest.main()
