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

"""Tests for agentic_rl_learner."""

import asyncio
import concurrent.futures
import inspect
import queue
import threading
from types import SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl import utils as rl_utils
from tunix.rl import alignment
from tunix.rl.agentic import agentic_rl_learner
from tunix.rl.rollout import base_rollout


class DummyLearner(agentic_rl_learner.AgenticRLLearner):
  def _process_results(self, **kwargs):
    return []


class AgenticRLLearnerTest(parameterized.TestCase):

  def test_p57_evaluate_only_covers_dataset_without_train_update(self):
    learner = object.__new__(DummyLearner)
    learner.algo_config = agentic_rl_learner.AgenticRLConfig(num_generations=8)
    learner._rewards_window_lock = threading.Lock()
    learner._eval_rewards_window = []
    learner._full_batch_size = 0
    learner._build_orchestrator = mock.Mock(return_value=object())

    async def producer(orchestrator, prompt_iterator, num_generations):
      del orchestrator
      self.assertEqual(num_generations, 8)
      for _ in prompt_iterator:
        yield [object() for _ in range(8)]

    prompt_index = 0

    def process(batch, mode):
      nonlocal prompt_index
      self.assertLen(batch, 8)
      self.assertEqual(mode, rl_cluster_lib.Mode.EVAL)
      with learner._rewards_window_lock:
        learner._eval_rewards_window.extend([float(prompt_index)] * 8)
      prompt_index += 1
      return []

    learner._orchestrator_producer = producer
    learner._batch_to_train_example = mock.Mock(side_effect=process)
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    learner.loop = loop
    try:
      result = learner.evaluate_only(
          [{"prompts": np.asarray(["p0", "p1"])}], policy_step=20
      )
    finally:
      loop.call_soon_threadsafe(loop.stop)
      thread.join(timeout=5)
      loop.close()

    self.assertEqual(result["policy_step"], 20)
    self.assertEqual(result["prompts"], 2)
    self.assertEqual(result["generations"], 8)
    self.assertEqual(result["batches"], 2)
    self.assertEqual(result["n"], 16)
    self.assertEqual(result["reward"], 0.5)
    self.assertEqual(result["solve"], 0.5)
    self.assertEqual(learner._full_batch_size, 2)
    self.assertEqual(learner._eval_rewards_window, [])
    self.assertEqual(learner._batch_to_train_example.call_count, 2)

  def test_p57_rollout_only_evaluate_skips_trainer_recompute(self):
    learner = object.__new__(DummyLearner)
    learner.algo_config = agentic_rl_learner.AgenticRLConfig(num_generations=2)
    learner._full_batch_size = 0
    learner._build_orchestrator = mock.Mock(return_value=object())
    learner._batch_to_train_example = mock.Mock(
        side_effect=AssertionError("trainer recompute must remain unreachable")
    )
    learner.rl_cluster = SimpleNamespace(
        global_steps=0, actor_trainer=SimpleNamespace(train_steps=0)
    )

    async def producer(orchestrator, prompt_iterator, num_generations):
      del orchestrator
      for group_id, _ in enumerate(prompt_iterator):
        yield [
            SimpleNamespace(
                group_id=group_id,
                pair_index=pair_index,
                traj={
                    "prompt_tokens": np.arange(4),
                    "prompt_length": 4,
                    "conversation_tokens": np.arange(6),
                    "conversation_masks": np.asarray([1, 1, 0, 1, 0, 0]),
                    "conversation_text": [
                        {"role": "assistant", "content": "Down"},
                        {"role": "user", "content": "grid"},
                    ],
                    "status": "SUCCEEDED" if pair_index else "MAX_STEPS_REACHED",
                    "trajectory_reward": float(pair_index),
                    "invalid_action_count": 0,
                    "ineffective_action_count": 1,
                    "policy_version": 0,
                    "original_input": {
                        # Deliberately wrong: production trajectories retain
                        # presentation text here, not the Grain source row.
                        "p57_index": -1,
                        "size": -1,
                        "shortest_path": -1,
                        "map_sha256": "",
                    },
                },
            )
            for pair_index in range(num_generations)
        ]

    learner._orchestrator_producer = producer
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    learner.loop = loop
    try:
      result = learner.rollout_only_evaluate(
          [{
              "prompts": np.asarray(["p0", "p1"]),
              "p57_index": np.asarray([10, 11]),
              "size": np.asarray([5, 6]),
              "shortest_path": np.asarray([4, 5]),
              "map_sha256": np.asarray(["a" * 64, "b" * 64]),
          }],
          policy_step=0,
      )
    finally:
      loop.call_soon_threadsafe(loop.stop)
      thread.join(timeout=5)
      loop.close()

    self.assertEqual(result["prompts"], 2)
    self.assertEqual(result["trajectories"], 4)
    self.assertEqual(result["train_steps_before"], 0)
    self.assertEqual(result["train_steps_after"], 0)
    self.assertEqual(result["records"][0]["context_tokens"], 10)
    self.assertEqual(result["records"][0]["assistant_tokens"], 3)
    self.assertEqual(result["records"][0]["ineffective_actions"], 1)
    self.assertEqual(result["records"][0]["p57_index"], 10)
    self.assertEqual(result["records"][1]["map_sha256"], "a" * 64)
    self.assertEqual(result["records"][2]["p57_index"], 11)
    self.assertEqual(result["records"][3]["map_sha256"], "b" * 64)
    learner._batch_to_train_example.assert_not_called()

  def test_p61_validation_follows_p33_workload_initialization(self):
    source = inspect.getsource(
        agentic_rl_learner.AgenticRLLearner._run_p28_g6_update  # pylint: disable=protected-access
    )
    self.assertLess(
        source.index("workload = dp_workloads.active_workload()"),
        source.index("p61_capture_dir = os.environ.get("),
    )

  def test_p60_xprof_compute_mode_is_forwarded(self):
    options = agentic_rl_learner._canon_xprof_profile_options(  # pylint: disable=protected-access
        host_tracer=1,
        python_tracer=0,
        tpu_trace_mode="TRACE_COMPUTE",
    )
    self.assertEqual(options.host_tracer_level, 1)
    self.assertEqual(options.python_tracer_level, 0)
    self.assertEqual(
        options.advanced_configuration, {"tpu_trace_mode": "TRACE_COMPUTE"}
    )

  def test_p60_xprof_rejects_unknown_tpu_trace_mode(self):
    with self.assertRaisesRegex(ValueError, "CANON_XPROF_TPU_TRACE_MODE"):
      agentic_rl_learner._canon_xprof_profile_options(  # pylint: disable=protected-access
          host_tracer=1,
          python_tracer=0,
          tpu_trace_mode="TRACE_EVERYTHING",
      )

  def test_rollout_batch_watchdog_fails_waiting_for_first_group(self):
    class StalledOrchestrator:

      async def run_producers_from_stream(self, **kwargs):
        del kwargs
        await asyncio.Event().wait()

      async def yield_batches(self, batch_size):
        del batch_size
        await asyncio.Event().wait()
        if False:
          yield []

    async def run():
      learner = object.__new__(DummyLearner)
      learner.loop = asyncio.get_running_loop()
      learner.rl_cluster = mock.Mock(global_steps=0)
      learner._full_batch_size = 1
      learner._background_tasks = set()
      learner.algo_config = agentic_rl_learner.AgenticRLConfig(
          num_generations=2,
          rollout_batch_timeout=0.01,
      )
      stream = learner._orchestrator_producer(
          StalledOrchestrator(), iter(()), 2, "Token"
      )
      with self.assertRaisesRegex(
          TimeoutError, "completed_prompt_groups=0/1"
      ):
        await anext(stream)
      await stream.aclose()

    asyncio.run(run())

  def test_p38_diagnostic_consumer_covers_all_prompt_groups(self):
    self.assertEqual(
        agentic_rl_learner._p38_diagnostic_consumer_contract(
            enabled=True,
            full_batch_size=32,
            mini_batch_size=4,
            train_micro_batch_size=4,
            num_generations=8,
            process_in_consumer=True,
        ),
        (32, True, 8),
    )

  def test_p38_diagnostic_consumer_is_noop_when_disabled(self):
    self.assertEqual(
        agentic_rl_learner._p38_diagnostic_consumer_contract(
            enabled=False,
            full_batch_size=17,
            mini_batch_size=5,
            train_micro_batch_size=3,
            num_generations=2,
            process_in_consumer=False,
        ),
        (3, False, 0),
    )

  def test_p38_diagnostic_consumer_admits_onehost_rehearsal(self):
    self.assertEqual(
        agentic_rl_learner._p38_diagnostic_consumer_contract(
            enabled=True,
            full_batch_size=2,
            mini_batch_size=2,
            train_micro_batch_size=4,
            num_generations=2,
            process_in_consumer=True,
            onehost_rehearsal=True,
        ),
        (2, True, 1),
    )

  def test_p38_diagnostic_consumer_admits_p58_vma_diagnostic(self):
    self.assertEqual(
        agentic_rl_learner._p38_diagnostic_consumer_contract(
            enabled=True,
            full_batch_size=8,
            mini_batch_size=8,
            train_micro_batch_size=8,
            num_generations=16,
            process_in_consumer=True,
            p58_vma_diagnostic=True,
        ),
        (8, True, 1),
    )

  def test_p38_diagnostic_consumer_admits_p58_seam_localization(self):
    self.assertEqual(
        agentic_rl_learner._p38_diagnostic_consumer_contract(
            enabled=True,
            full_batch_size=8,
            mini_batch_size=8,
            train_micro_batch_size=8,
            num_generations=16,
            process_in_consumer=True,
            p58_seam_localization=True,
        ),
        (8, True, 1),
    )

  def test_p38_diagnostic_consumer_admits_p58_q4_continue_kv(self):
    self.assertEqual(
        agentic_rl_learner._p38_diagnostic_consumer_contract(
            enabled=True,
            full_batch_size=1,
            mini_batch_size=1,
            train_micro_batch_size=1,
            num_generations=2,
            process_in_consumer=False,
            p58_q4_tp4_continue_kv=True,
        ),
        (1, True, 1),
    )

  def test_p38_diagnostic_consumer_keeps_legacy_producer_rejection(self):
    with self.assertRaisesRegex(ValueError, "raw trajectories"):
      agentic_rl_learner._p38_diagnostic_consumer_contract(
          enabled=True,
          full_batch_size=32,
          mini_batch_size=4,
          train_micro_batch_size=4,
          num_generations=8,
          process_in_consumer=False,
      )

  def test_p38_diagnostic_consumer_rejects_subset_geometry(self):
    with self.assertRaisesRegex(ValueError, "coverage geometry changed"):
      agentic_rl_learner._p38_diagnostic_consumer_contract(
          enabled=True,
          full_batch_size=32,
          mini_batch_size=5,
          train_micro_batch_size=5,
          num_generations=8,
          process_in_consumer=True,
      )

  def test_p38_diagnostic_consumer_rejects_partial_tail(self):
    learner = object.__new__(DummyLearner)
    data = queue.Queue()
    for value in range(5):
      data.put(value)
    data.put(None)
    batches = learner._data_consumer_batch_generator(
        data, 32, require_full_batch=True
    )
    with self.assertRaisesRegex(RuntimeError, "refusing subset alignment"):
      next(batches)

  def test_p58_partial_consumer_propagates_producer_timeout(self):
    learner = object.__new__(DummyLearner)
    data = queue.Queue()
    for value in range(2):
      data.put(value)
    data.put(None)
    producer_future = concurrent.futures.Future()
    producer_future.set_exception(
        TimeoutError("rollout batch exceeded hard timeout: 32/128")
    )
    batches = learner._data_consumer_batch_generator(
        data,
        8,
        require_full_batch=True,
        producer_future=producer_future,
        contract_name="P58",
    )
    with self.assertRaisesRegex(TimeoutError, "32/128"):
      next(batches)

  def test_p58_full_batch_group_contract_rejects_missing_generation(self):
    groups = []
    for group_id in range(8):
      width = 15 if group_id == 7 else 16
      groups.append([
          SimpleNamespace(group_id=group_id, pair_index=pair_index)
          for pair_index in range(width)
      ])
    with self.assertRaisesRegex(
        alignment.AlignmentGateError, "generation count changed"
    ):
      agentic_rl_learner._validate_p58_full_batch_groups(groups)

  def test_normal_consumer_keeps_legacy_partial_tail_behavior(self):
    learner = object.__new__(DummyLearner)
    data = queue.Queue()
    for value in range(5):
      data.put(value)
    data.put(None)
    batches = learner._data_consumer_batch_generator(data, 32)
    self.assertEqual(next(batches), list(range(5)))

  def test_model_call_wraps_one_conversation_as_a_prompt_batch(self):
    learner = object.__new__(DummyLearner)
    learner.chat_parser = None
    learner.rl_cluster = mock.Mock()
    learner.rl_cluster.generate.return_value = mock.sentinel.rollout
    conversation = [{"role": "user", "content": "hello"}]

    result = learner._model_call(conversation)

    self.assertIs(result, mock.sentinel.rollout)
    self.assertEqual(
        learner.rl_cluster.generate.call_args.kwargs["prompts"],
        [conversation],
    )

  def test_model_call_routes_signed_deepswe_pre_tokenized_prompt_exactly(self):
    learner = object.__new__(DummyLearner)
    learner.chat_parser = mock.Mock()
    learner.rl_cluster = mock.Mock()
    learner.rl_cluster.generate.return_value = mock.sentinel.rollout
    prompt_ids = np.asarray([151644, 28, 1725], dtype=np.int32)

    with mock.patch.object(
        agentic_rl_learner.deepswe_debug,
        "deepswe_exact_token_continuity",
        return_value=True,
    ):
      result = learner._model_call(
          [{"role": "user", "content": "ignored"}],
          prompt_token_ids=prompt_ids,
      )

    self.assertIs(result, mock.sentinel.rollout)
    learner.chat_parser.parse.assert_not_called()
    self.assertEqual(
        learner.rl_cluster.generate.call_args.kwargs["prompts"],
        [[151644, 28, 1725]],
    )
    self.assertFalse(
        learner.rl_cluster.generate.call_args.kwargs["apply_chat_template"]
    )

  def test_model_call_rejects_unsigned_pre_tokenized_prompt(self):
    learner = object.__new__(DummyLearner)
    learner.chat_parser = mock.Mock()
    learner.rl_cluster = mock.Mock()
    with mock.patch.object(
        agentic_rl_learner.deepswe_debug,
        "deepswe_exact_token_continuity",
        return_value=False,
    ), mock.patch.object(
        agentic_rl_learner.token_continuity,
        "m15_token_continuity_mode",
        return_value=None,
    ):
      with self.assertRaisesRegex(ValueError, "signed DeepSWE or M15"):
        learner._model_call(
            [{"role": "user", "content": "ignored"}],
            prompt_token_ids=[1, 2, 3],
        )
    learner.rl_cluster.generate.assert_not_called()

  def test_model_call_routes_signed_m15_pre_tokenized_prompt_exactly(self):
    learner = object.__new__(DummyLearner)
    learner.chat_parser = mock.Mock()
    learner.rl_cluster = mock.Mock()
    learner.rl_cluster.generate.return_value = mock.sentinel.rollout
    prompt_ids = np.asarray([151644, 28, 1725], dtype=np.int32)

    with mock.patch.object(
        agentic_rl_learner.deepswe_debug,
        "deepswe_exact_token_continuity",
        return_value=False,
    ), mock.patch.object(
        agentic_rl_learner.token_continuity,
        "m15_token_continuity_mode",
        return_value="exact",
    ):
      result = learner._model_call(
          [{"role": "user", "content": "ignored"}],
          prompt_token_ids=prompt_ids,
      )

    self.assertIs(result, mock.sentinel.rollout)
    learner.chat_parser.parse.assert_not_called()
    self.assertEqual(
        learner.rl_cluster.generate.call_args.kwargs["prompts"],
        [[151644, 28, 1725]],
    )
    self.assertFalse(
        learner.rl_cluster.generate.call_args.kwargs["apply_chat_template"]
    )

  def test_model_call_rejects_simultaneous_deepswe_and_m15_admission(self):
    learner = object.__new__(DummyLearner)
    learner.chat_parser = mock.Mock()
    learner.rl_cluster = mock.Mock()

    with mock.patch.object(
        agentic_rl_learner.deepswe_debug,
        "deepswe_exact_token_continuity",
        return_value=True,
    ), mock.patch.object(
        agentic_rl_learner.token_continuity,
        "m15_token_continuity_mode",
        return_value="exact",
    ):
      with self.assertRaisesRegex(ValueError, "mutually exclusive"):
        learner._model_call(
            [{"role": "user", "content": "ignored"}],
            prompt_token_ids=[1, 2, 3],
        )
    learner.rl_cluster.generate.assert_not_called()

  def test_frozenlake_evaluation_metrics_are_finite_and_complete(self):
    metrics = agentic_rl_learner._frozenlake_evaluation_metrics(
        [0.0, 0.2, 1.0], wall_seconds=2.5, policy_step=10
    )
    self.assertEqual(metrics["n"], 3)
    self.assertAlmostEqual(metrics["solve"], 2.0 / 3.0)
    self.assertEqual(metrics["wall_seconds"], 2.5)
    self.assertEqual(metrics["policy_step"], 10)

  def test_frozenlake_evaluation_metrics_reject_nonfinite_rewards(self):
    with self.assertRaisesRegex(ValueError, "nonempty and finite"):
      agentic_rl_learner._frozenlake_evaluation_metrics(
          [0.0, float("nan")], wall_seconds=1.0, policy_step=0
      )

  def test_p31_segmented_eval_uses_preupdate_step_exactly_once(self):
    self.assertEqual(
        agentic_rl_learner._eval_schedule_step(
            segmented_update=True,
            pre_update_train_step=0,
            current_train_step=1,
        ),
        0,
    )
    self.assertTrue(
        agentic_rl_learner._should_run_eval(
            prompt_count=100,
            schedule_step=0,
            eval_every_n_steps=25,
            last_eval_train_step=-1,
        )
    )
    self.assertFalse(
        agentic_rl_learner._should_run_eval(
            prompt_count=100,
            schedule_step=0,
            eval_every_n_steps=25,
            last_eval_train_step=0,
        )
    )
    self.assertEqual(
        agentic_rl_learner._eval_schedule_step(
            segmented_update=False,
            pre_update_train_step=0,
            current_train_step=1,
        ),
        1,
    )
    self.assertFalse(
        agentic_rl_learner._should_run_eval(
            prompt_count=100,
            schedule_step=1,
            eval_every_n_steps=25,
            last_eval_train_step=-1,
        )
    )

  def test_nonpositive_eval_cadence_disables_evaluation(self):
    for cadence in (0, -1):
      with self.subTest(cadence=cadence):
        self.assertFalse(
            agentic_rl_learner._should_run_eval(
                prompt_count=100,
                schedule_step=0,
                eval_every_n_steps=cadence,
                last_eval_train_step=-1,
            )
        )

  def test_validate_rollout_config_mismatch_max_tokens(self):
    rl_cluster = mock.Mock()
    rl_cluster.cluster_config = mock.Mock()
    rl_cluster.cluster_config.rollout_engine = "generic"
    rollout_config = base_rollout.RolloutConfig(
        max_prompt_length=32,
        max_tokens_to_generate=10,
        return_logprobs=True,
    )
    rl_cluster.cluster_config.rollout_config = rollout_config

    algo_config = agentic_rl_learner.AgenticRLConfig(
        max_response_length=20,  # Mismatch: 10 != 20
        use_rollout_logps=True,
    )

    with self.assertRaisesRegex(
        ValueError, r"max_tokens_to_generate \(10\) must match AgenticRLConfig max_response_length \(20\)"
    ):
      DummyLearner(
          rl_cluster=rl_cluster,
          reward_fns=mock.Mock(),
          algo_config=algo_config,
      )

  def test_validate_rollout_config_missing_logprobs(self):
    rl_cluster = mock.Mock()
    rl_cluster.cluster_config = mock.Mock()
    rl_cluster.cluster_config.rollout_engine = "generic"
    rollout_config = base_rollout.RolloutConfig(
        max_prompt_length=32,
        max_tokens_to_generate=10,
        return_logprobs=False,  # Should be True
    )
    rl_cluster.cluster_config.rollout_config = rollout_config

    algo_config = agentic_rl_learner.AgenticRLConfig(
        max_response_length=10,
        use_rollout_logps=True,
    )

    with self.assertRaisesRegex(
        ValueError, r"must have return_logprobs=True"
    ):
      DummyLearner(
          rl_cluster=rl_cluster,
          reward_fns=mock.Mock(),
          algo_config=algo_config,
      )

  def test_validate_rollout_config_dict_mode(self):
    rl_cluster = mock.Mock()
    rl_cluster.cluster_config = mock.Mock()
    rl_cluster.cluster_config.rollout_engine = "generic"
    rollout_config_train = base_rollout.RolloutConfig(
        max_prompt_length=32,
        max_tokens_to_generate=10,
        return_logprobs=True,
    )
    rollout_config_eval = base_rollout.RolloutConfig(
        max_prompt_length=32,
        max_tokens_to_generate=10,
        return_logprobs=False,  # Mismatch in eval mode
    )
    rl_cluster.cluster_config.rollout_config = {
        "train": rollout_config_train,
        "eval": rollout_config_eval,
    }

    algo_config = agentic_rl_learner.AgenticRLConfig(
        max_response_length=10,
        use_rollout_logps=True,
    )

    with self.assertRaisesRegex(
        ValueError, r"RolloutConfig \(eval\) must have return_logprobs=True"
    ):
      DummyLearner(
          rl_cluster=rl_cluster,
          reward_fns=mock.Mock(),
          algo_config=algo_config,
      )

  def test_validate_rollout_config_vllm_missing_server_mode(self):
    rl_cluster = mock.Mock()
    rl_cluster.cluster_config = mock.Mock()
    rl_cluster.cluster_config.rollout_engine = "vllm"
    rollout_config = base_rollout.RolloutConfig(
        max_prompt_length=32,
        max_tokens_to_generate=10,
        return_logprobs=True,
        rollout_vllm_server_mode=False,  # Should be True for vLLM
    )
    rl_cluster.cluster_config.rollout_config = rollout_config

    algo_config = agentic_rl_learner.AgenticRLConfig(
        max_response_length=10,
        use_rollout_logps=True,
    )

    with self.assertRaisesRegex(
        ValueError,
        r"must have rollout_vllm_server_mode set to True for AgenticRLLearner"
        r" if using vLLM engine",
    ):
      DummyLearner(
          rl_cluster=rl_cluster,
          reward_fns=mock.Mock(),
          algo_config=algo_config,
      )

  def test_train_batch_size_mismatch_raises_error(self):
    with mock.patch.object(
        rl_utils, "is_sharing_weights", return_value=False
    ):
      rl_cluster = mock.Mock()
      rl_cluster.cluster_config = mock.Mock()
      rl_cluster.cluster_config.role_to_mesh = {
          rl_cluster_lib.Role.ACTOR: mock.Mock(),
          rl_cluster_lib.Role.ROLLOUT: mock.Mock(),
      }
      training_config = mock.Mock()
      training_config.compute_logps_micro_batch_size = 2
      training_config.train_micro_batch_size = 1
      training_config.mini_batch_size = None
      rl_cluster.cluster_config.training_config = training_config
      rl_cluster.cluster_config.rollout_config = base_rollout.RolloutConfig(
          max_tokens_to_generate=10, return_logprobs=True
      )
      rl_cluster.cluster_config.rollout_engine = 'generic'
      rl_cluster.actor_trainer = mock.Mock()
      rl_cluster.actor_trainer.restored_global_step.return_value = 0
      rl_cluster.actor_trainer.iter_steps = 0
      rl_cluster.rollout = mock.Mock()
      rl_cluster.tokenizer = mock.Mock()
      algo_config = agentic_rl_learner.AgenticRLConfig(max_response_length=10)
      learner = DummyLearner(
          rl_cluster=rl_cluster,
          reward_fns=mock.Mock(),
          algo_config=algo_config,
      )
      train_dataset = [{'prompt': ['p1']}]
      with self.assertRaisesRegex(
          ValueError,
          r'compute_logps_micro_batch_size \(2\) must be equal to'
          r' train_micro_batch_size \(1\)',
      ):
        learner.train(train_dataset)


if __name__ == "__main__":
  absltest.main()
