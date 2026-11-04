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

"""Tests for the partial-rollout weight sync of AgenticRLLearner."""

# The weight sync and the per-request sampler kwargs are private hooks of the
# learner; these tests drive them directly.
# pylint: disable=protected-access

import threading
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from tunix.rl import rl_cluster as rl_engine_lib
from tunix.rl import utils as rl_utils
from tunix.rl.agentic import agentic_rl_learner
from tunix.rl.rollout import base_rollout


class _Learner(agentic_rl_learner.AgenticRLLearner):
  """Minimal concrete learner."""

  def _process_results(self, trajectories, mode=None, expected_step=None):
    del trajectories, mode, expected_step
    return []


class _PausableRollout:
  """Rollout stand-in that records the order of pause / resume calls."""

  def __init__(self, events, *, supported=True, pause_error=None):
    self._events = events
    self._supported = supported
    self._pause_error = pause_error

  def model(self):
    return mock.Mock()

  def supports_partial_rollout_sync(self):
    return self._supported

  def pause_for_weight_sync(self):
    self._events.append("pause")
    if self._pause_error is not None:
      raise self._pause_error

  def resume_after_weight_sync(self):
    self._events.append("resume")


class _MinimalRollout(base_rollout.BaseRollout):
  """BaseRollout with no-op abstract methods, to test its defaults."""

  def __init__(self):  # pylint: disable=super-init-not-called
    pass

  def generate(self, prompts, rollout_config, **kwargs):
    raise NotImplementedError

  def get_per_token_logps(self, prompt_tokens, completion_tokens):
    raise NotImplementedError

  def update_params(self, params, filter_types=None):
    raise NotImplementedError

  def pad_id(self):
    return 0

  def eos_id(self):
    return 1

  def model(self):
    return None


def _make_engine(rollout, *, partial_rollout, rollout_config_as_dict=False):
  """Returns a mock RL engine whose rollout config sets `partial_rollout`."""
  rl_engine = mock.MagicMock()  # perf_v2.span is used as a context manager.
  rl_engine.cluster_config = mock.Mock()
  rl_engine.cluster_config.rollout_engine = "generic"
  rl_engine.cluster_config.role_to_mesh = {
      rl_engine_lib.Role.ACTOR: mock.Mock(),
      rl_engine_lib.Role.ROLLOUT: mock.Mock(),
  }
  rollout_config = base_rollout.RolloutConfig(
      max_tokens_to_generate=10,
      return_logprobs=True,
      partial_rollout=partial_rollout,
  )
  rl_engine.cluster_config.rollout_config = (
      {rl_engine_lib.Mode.TRAIN: rollout_config}
      if rollout_config_as_dict
      else rollout_config
  )
  rl_engine.actor_trainer = mock.Mock()
  rl_engine.actor_trainer.restored_global_step.return_value = 3
  rl_engine.actor_trainer.iter_steps = 0
  rl_engine.rollout = rollout
  rl_engine.tokenizer = mock.Mock()
  return rl_engine


def _make_learner(rl_engine, *, sharing_weights=False):
  """Returns a learner on `rl_engine`, with or without shared weights."""
  with mock.patch.object(
      rl_utils, "is_sharing_weights", return_value=sharing_weights
  ):
    return _Learner(
        rl_engine=rl_engine,
        reward_fns=mock.Mock(),
        algo_config=agentic_rl_learner.AgenticRLConfig(max_response_length=10),
    )


def _salt(rl_engine):
  """Returns the cache salt of the last generate call."""
  return rl_engine.generate.call_args.kwargs["sampling_kwargs"]["cache_salt"]


class PartialRolloutSyncTest(parameterized.TestCase):

  @parameterized.parameters(False, True)
  def test_flag_follows_train_rollout_config(self, as_dict):
    def _learner(partial_rollout):
      return _make_learner(
          _make_engine(
              _PausableRollout([]),
              partial_rollout=partial_rollout,
              rollout_config_as_dict=as_dict,
          )
      )

    self.assertTrue(_learner(True)._partial_rollout_sync)
    self.assertFalse(_learner(False)._partial_rollout_sync)

  def test_flag_off_when_weights_are_shared(self):
    learner = _make_learner(
        _make_engine(_PausableRollout([]), partial_rollout=True),
        sharing_weights=True,
    )
    self.assertFalse(learner._partial_rollout_sync)

  def test_rollout_without_support_raises(self):
    rollout = _PausableRollout([], supported=False)
    with self.assertRaisesRegex(ValueError, "cannot pause in-flight"):
      _make_learner(_make_engine(rollout, partial_rollout=True))

  def test_base_rollout_does_not_support_partial_rollout_sync(self):
    rollout = _MinimalRollout()
    self.assertFalse(rollout.supports_partial_rollout_sync())
    with self.assertRaises(NotImplementedError):
      rollout.pause_for_weight_sync()
    with self.assertRaises(NotImplementedError):
      rollout.resume_after_weight_sync()

  def test_pause_sync_resume_order_and_version_bump(self):
    events = []
    rl_engine = _make_engine(_PausableRollout(events), partial_rollout=True)
    rl_engine.sync_weights.side_effect = lambda: events.append("sync")
    learner = _make_learner(rl_engine)
    version = learner.policy_version
    rl_engine.global_steps = 5

    learner._sync_weights()

    self.assertEqual(events, ["pause", "sync", "resume"])
    self.assertEqual(learner.policy_version, version + 1)
    metrics = rl_engine.buffer_metrics_async.call_args
    self.assertIn("perf/partial_rollout_pause_s", metrics.args[0])
    self.assertEqual(metrics.kwargs["step"], 5)

  def test_resumes_when_sync_fails(self):
    events = []
    rl_engine = _make_engine(_PausableRollout(events), partial_rollout=True)
    rl_engine.sync_weights.side_effect = RuntimeError("reshard failed")
    learner = _make_learner(rl_engine)
    version = learner.policy_version

    with self.assertRaisesRegex(RuntimeError, "reshard failed"):
      learner._sync_weights()

    self.assertEqual(events, ["pause", "resume"])
    # Requests queued during the failed sync already carry the new version.
    self.assertEqual(learner.policy_version, version + 1)

  def test_resumes_when_pause_fails(self):
    events = []
    rollout = _PausableRollout(events, pause_error=TimeoutError("pause"))
    rl_engine = _make_engine(rollout, partial_rollout=True)
    learner = _make_learner(rl_engine)
    version = learner.policy_version

    with self.assertRaises(TimeoutError):
      learner._sync_weights()

    self.assertEqual(events, ["pause", "resume"])
    self.assertEqual(learner.policy_version, version)
    rl_engine.sync_weights.assert_not_called()

  def test_requests_submitted_during_the_sync_carry_the_new_salt(self):
    rl_engine = _make_engine(_PausableRollout([]), partial_rollout=True)
    learner = _make_learner(rl_engine)
    version = learner.policy_version
    salts = []

    def _sync():
      learner._model_call(None, prompt_token_ids=[1])
      salts.append(_salt(rl_engine))

    rl_engine.sync_weights.side_effect = _sync
    learner._sync_weights()
    self.assertEqual(salts, [f"policy_v{version + 1}"])

  def test_multi_turn_episode_keeps_first_policy_version_and_salt(self):
    rl_engine = _make_engine(_PausableRollout([]), partial_rollout=True)
    learner = _make_learner(rl_engine)
    env = mock.Mock()
    env.task = {}
    env.extra_kwargs = {}
    learner.policy_version = 3
    learner._model_call(None, env=env, prompt_token_ids=[1])
    learner.policy_version = 4
    learner._model_call(None, env=env, prompt_token_ids=[1, 2])
    self.assertEqual(env.task["policy_version"], 3)
    self.assertEqual(_salt(rl_engine), "policy_v3")

  def test_does_not_wait_for_in_flight_rollouts(self):
    # An episode in flight holds the rollout side of RolloutSyncLock. The
    # draining sync would block on it; the partial-rollout sync must not.
    rl_engine = _make_engine(_PausableRollout([]), partial_rollout=True)
    learner = _make_learner(rl_engine)
    learner._rollout_sync_lock.acquire_rollout()
    try:
      done = threading.Event()
      worker = threading.Thread(
          target=lambda: (learner._sync_weights(), done.set())
      )
      worker.start()
      self.assertTrue(done.wait(timeout=10))
      worker.join()
    finally:
      learner._rollout_sync_lock.release_rollout()
    rl_engine.sync_weights.assert_called_once()

  def test_draining_sync_holds_the_lock_and_bumps_after_the_sync(self):
    rl_engine = _make_engine(_PausableRollout([]), partial_rollout=False)
    learner = _make_learner(rl_engine)
    version = learner.policy_version
    observed = []

    def _sync():
      observed.append((
          learner.policy_version,
          learner._rollout_sync_lock.try_acquire_rollout(),
      ))

    rl_engine.sync_weights.side_effect = _sync
    learner._sync_weights()
    self.assertEqual(observed, [(version, False)])
    self.assertEqual(learner.policy_version, version + 1)
    self.assertTrue(learner._rollout_sync_lock.try_acquire_rollout())


class CacheSaltTest(absltest.TestCase):

  def test_requests_are_salted_with_policy_version(self):
    rl_engine = _make_engine(_PausableRollout([]), partial_rollout=True)
    learner = _make_learner(rl_engine)
    learner.policy_version = 7
    learner._model_call(None, prompt_token_ids=[1, 2, 3])
    self.assertEqual(
        rl_engine.generate.call_args.kwargs["sampling_kwargs"],
        {"cache_salt": "policy_v7"},
    )

  def test_no_salt_without_partial_rollout(self):
    rl_engine = _make_engine(_PausableRollout([]), partial_rollout=False)
    learner = _make_learner(rl_engine)
    learner._model_call(None, prompt_token_ids=[1, 2, 3])
    self.assertIsNone(rl_engine.generate.call_args.kwargs["sampling_kwargs"])


class RolloutConfigKvPolicyTest(absltest.TestCase):

  def test_default_is_keep(self):
    self.assertEqual(
        base_rollout.RolloutConfig().partial_rollout_kv_policy, "keep"
    )

  def test_rejects_unknown_policy(self):
    with self.assertRaisesRegex(ValueError, "partial_rollout_kv_policy"):
      base_rollout.RolloutConfig(partial_rollout_kv_policy="drop")


if __name__ == "__main__":
  absltest.main()
