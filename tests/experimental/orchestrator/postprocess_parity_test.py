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

"""Numeric parity between the orchestrated postprocess and the agentic learner.

The orchestrated GRPO path re-expresses the agentic learner's postprocess on
the orchestrator primitives. Nothing else in the tree checks that the two
produce the same numbers -- the end-to-end tests only assert that training
happens -- so a divergence in advantages, padding, masks, or log-probabilities
would look exactly like a healthy run.

This feeds identical worker responses through both paths over the same cluster
and compares the resulting train examples field by field. Metrics are
deliberately out of scope here: the two paths emit different metric sets by
construction, and the reference's emission has known defects.
"""

import os
from typing import Any, Mapping

from absl.testing import absltest
from absl.testing import parameterized
import chex
from flax import nnx
import jax.numpy as jnp
import numpy as np
import optax
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import algorithm_adapter
from tunix.experimental.orchestrator import orchestrator_rl_cluster
from tunix.experimental.orchestrator import rl_orchestrator
from tunix.experimental.orchestrator import rollout_response_adapter
from tunix.generate import tokenizer_adapter
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.agentic import agentic_grpo_learner
from tunix.rl.rollout import base_rollout
from tunix.tests import test_common

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

MAX_RESPONSE_LENGTH = 8
TRAIN_MAX_PROMPT_LENGTH = 16
# Deliberately different, so a path that ignores the mode pads differently.
EVAL_MAX_PROMPT_LENGTH = 12


def _reward_fn(prompts, completions, **kwargs):
  """Rewards depend on the dataset row, exercising reward-kwarg forwarding."""
  del completions
  answers = kwargs.get("answer")
  if answers is None:
    answers = [""] * len(prompts)
  return [1.0 + 0.5 * len(str(answer)) for answer in answers]


class _MockChatParser:

  def parse(self, messages, add_generation_prompt=False, is_first_msg=False):
    del is_first_msg, add_generation_prompt
    return " ".join(message["content"] for message in messages)

  @property
  def assistant_token(self):
    return "Assistant: "

  def update_assistant_end_tokens(self, tokens):
    return tokens, 0


def _request(index: int, row: Mapping[str, Any]) -> datatypes.RolloutRequest:
  return datatypes.RolloutRequest(
      request_id=f"req-{index}",
      prompt=row,
      prompt_id="p0",
      group_id="g0",
  )


def _response(index: int) -> datatypes.RolloutResponse:
  """A deterministic worker answer with distinct tokens, masks, and logps."""
  completion = np.array([2 + index, 3 + index, 4 + index], dtype=np.int32)
  return datatypes.RolloutResponse(
      request_id=f"req-{index}",
      status="SUCCEEDED",
      prompt_tokens=np.array([5, 6, 7], dtype=np.int32),
      segments=[
          datatypes.TokenSegment(
              source="assistant",
              tokens=completion,
              loss_mask=np.ones_like(completion),
              logps=np.array(
                  [-0.1 * (index + 1), -0.2, -0.3], dtype=np.float32
              ),
          )
      ],
      env_reward=float(index),
      policy_version=3,
  )


class PostprocessParityTest(parameterized.TestCase):
  """The orchestrated postprocess must equal the reference, field for field."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    try:
      chex.set_n_cpu_devices(2)
    except RuntimeError:
      # JAX is already initialized by another test in this process.
      pass

  def setUp(self):
    super().setUp()
    self.vocab = test_common.MockVocab()
    self.tokenizer = tokenizer_adapter.TokenizerAdapter(self.vocab)
    self.cluster = self._build_cluster()
    self.actor_logps_calls = []
    self.ref_logps_calls = []
    self._stub_scoring(self.cluster)

  def _build_cluster(self) -> rl_cluster_lib.RLCluster:
    """A real cluster with a per-mode rollout config, shared by both paths."""
    model = test_common.ToyTransformer(
        config=test_common.ModelConfig(vocab_size=self.vocab.GetPieceSize()),
        rngs=nnx.Rngs(0),
    )
    ref_model = test_common.ToyTransformer(
        config=test_common.ModelConfig(vocab_size=self.vocab.GetPieceSize()),
        rngs=nnx.Rngs(0),
    )
    from jax.interpreters import pxla  # pylint: disable=g-import-not-at-top

    mesh = pxla.thread_resources.env.physical_mesh
    cluster_config = rl_cluster_lib.ClusterConfig(
        role_to_mesh={
            rl_cluster_lib.Role.ACTOR: mesh,
            rl_cluster_lib.Role.REFERENCE: mesh,
            rl_cluster_lib.Role.ROLLOUT: mesh,
        },
        rollout_engine="vanilla",
        offload_to_cpu=False,
        training_config=rl_cluster_lib.RLTrainingConfig(
            actor_optimizer=optax.sgd(1e-2),
            eval_every_n_steps=100,
            max_steps=2,
            mini_batch_size=1,
            train_micro_batch_size=1,
            rollout_micro_batch_size=1,
            compute_logps_micro_batch_size=1,
        ),
        rollout_config={
            rl_cluster_lib.Mode.TRAIN: base_rollout.RolloutConfig(
                max_prompt_length=TRAIN_MAX_PROMPT_LENGTH,
                max_tokens_to_generate=MAX_RESPONSE_LENGTH,
                return_logprobs=True,
                kv_cache_size=256,
                temperature=0.5,
            ),
            rl_cluster_lib.Mode.EVAL: base_rollout.RolloutConfig(
                max_prompt_length=EVAL_MAX_PROMPT_LENGTH,
                max_tokens_to_generate=MAX_RESPONSE_LENGTH,
                return_logprobs=True,
                kv_cache_size=256,
                temperature=0.5,
            ),
        },
    )
    return rl_cluster_lib.RLCluster(
        actor=model,
        reference=ref_model,
        tokenizer=self.tokenizer,
        cluster_config=cluster_config,
    )

  def _stub_scoring(self, cluster: rl_cluster_lib.RLCluster) -> None:
    """Deterministic scoring, recording how each path chunks its request.

    The actor recompute cannot run here anyway: it needs the log-prob anchor
    that a weight sync installs.
    """

    def actor_logps(
        prompt_tokens, completion_tokens, pad_id, eos_id, micro_batch_size=None
    ):
      del prompt_tokens, pad_id, eos_id
      self.actor_logps_calls.append(micro_batch_size)
      return jnp.asarray(completion_tokens, dtype=jnp.float32) * -0.02

    def ref_logps(
        prompt_tokens, completion_tokens, pad_id, eos_id, micro_batch_size=None
    ):
      del prompt_tokens, pad_id, eos_id
      self.ref_logps_calls.append(micro_batch_size)
      return jnp.asarray(completion_tokens, dtype=jnp.float32) * 0.01

    cluster.get_actor_per_token_logps = actor_logps
    cluster.get_ref_per_token_logps = ref_logps

  def _config(self, **overrides) -> agentic_grpo_learner.GRPOConfig:
    kwargs: dict[str, Any] = {
        "num_generations": 2,
        "num_iterations": 1,
        "beta": 0.0,
        "max_response_length": MAX_RESPONSE_LENGTH,
    }
    kwargs.update(overrides)
    return agentic_grpo_learner.GRPOConfig(**kwargs)

  def _items(self):
    """A complete two-sample group, as the wire adapter would deliver it."""
    row = {
        "prompts": "what is 2+2?",
        # An extra dataset column, so reward-kwarg forwarding is exercised.
        "answer": "four",
    }
    requests = [_request(i, row) for i in range(2)]
    responses = [_response(i) for i in range(2)]
    return rollout_response_adapter.to_trajectory_items(
        responses, requests, tokenizer=self.tokenizer
    )

  def _reference_learner(self, algo_config) -> agentic_grpo_learner.GRPOLearner:
    return agentic_grpo_learner.GRPOLearner(
        rl_cluster=self.cluster,
        reward_fns=_reward_fn,
        algo_config=algo_config,
        chat_parser=_MockChatParser(),
    )

  def _run_both(self, algo_config, mode):
    """Runs both postprocess implementations over the same items and cluster."""
    reference = self._reference_learner(algo_config)
    expected = reference._process_results(  # pylint: disable=protected-access
        self._items(), mode=mode, expected_step=0
    )

    adapter = algorithm_adapter.GRPOAdapter(algo_config)
    orchestrator = rl_orchestrator.RLOrchestrator(
        orchestrator_rl_cluster.OrchestratorRLCluster(self.cluster), adapter
    )
    actual = adapter.postprocess_group(
        orchestrator,
        self._items(),
        # The same reward path, so any difference is postprocess, not rewards.
        compute_rewards=reference._compute_rewards,  # pylint: disable=protected-access
        mode=mode,
        expected_step=0,
    )
    return expected, actual

  def _assert_examples_equal(self, expected, actual):
    self.assertLen(actual, len(expected))
    for want, got in zip(expected, actual):
      for field in (
          "prompt_ids",
          "prompt_mask",
          "completion_ids",
          "completion_mask",
          "advantages",
          "ref_per_token_logps",
          "old_per_token_logps",
          "policy_version",
          "sampler_is_weights",
      ):
        want_value = getattr(want, field)
        got_value = getattr(got, field)
        if want_value is None or got_value is None:
          self.assertIs(
              got_value if want_value is None else want_value,
              None,
              msg=f"{field}: one path produced None and the other did not",
          )
          continue
        np.testing.assert_allclose(
            np.asarray(got_value, dtype=np.float64),
            np.asarray(want_value, dtype=np.float64),
            rtol=1e-6,
            atol=1e-6,
            err_msg=f"{field} differs between the two postprocess paths",
        )

  @parameterized.named_parameters(
      ("on_policy_no_kl", 0.0, True, rl_cluster_lib.Mode.TRAIN),
      ("on_policy_with_kl", 0.04, True, rl_cluster_lib.Mode.TRAIN),
      ("recomputed_logps_no_kl", 0.0, False, rl_cluster_lib.Mode.TRAIN),
      ("recomputed_logps_with_kl", 0.04, False, rl_cluster_lib.Mode.TRAIN),
      ("eval_on_policy_no_kl", 0.0, True, rl_cluster_lib.Mode.EVAL),
      ("eval_on_policy_with_kl", 0.04, True, rl_cluster_lib.Mode.EVAL),
      ("eval_recomputed_logps", 0.0, False, rl_cluster_lib.Mode.EVAL),
      ("eval_recomputed_logps_with_kl", 0.04, False, rl_cluster_lib.Mode.EVAL),
  )
  def test_postprocess_matches_reference_train_examples(
      self, beta, use_rollout_logps, mode
  ):
    algo_config = self._config(beta=beta, use_rollout_logps=use_rollout_logps)
    expected, actual = self._run_both(algo_config, mode)
    self._assert_examples_equal(expected, actual)

  def test_eval_mode_pads_prompts_to_the_eval_length(self):
    """Guards the parity above: the two modes must not be interchangeable."""
    algo_config = self._config()
    train_expected, _ = self._run_both(algo_config, rl_cluster_lib.Mode.TRAIN)
    eval_expected, _ = self._run_both(algo_config, rl_cluster_lib.Mode.EVAL)

    self.assertEqual(
        train_expected[0].prompt_ids.shape[1], TRAIN_MAX_PROMPT_LENGTH
    )
    self.assertEqual(
        eval_expected[0].prompt_ids.shape[1], EVAL_MAX_PROMPT_LENGTH
    )

  def test_both_paths_score_with_the_same_chunk_size(self):
    """A different chunk size compiles a different shape for the same math."""
    algo_config = self._config(use_rollout_logps=False, beta=0.04)
    self._run_both(algo_config, rl_cluster_lib.Mode.TRAIN)

    # Each path scored the actor once and the reference once.
    self.assertLen(self.actor_logps_calls, 2)
    self.assertLen(self.ref_logps_calls, 2)
    self.assertEqual(self.actor_logps_calls[0], self.actor_logps_calls[1])
    self.assertEqual(self.ref_logps_calls[0], self.ref_logps_calls[1])

  def _buffered_keys(self, buffered):
    keys = set()
    for entry in buffered:
      keys.update(entry)
    return keys

  def test_emits_the_same_diagnostics_as_the_reference(self):
    """Metric coverage, not values: the two paths must report the same things."""
    algo_config = self._config()
    reference = self._reference_learner(algo_config)
    adapter = algorithm_adapter.GRPOAdapter(algo_config)
    orchestrator = rl_orchestrator.RLOrchestrator(
        orchestrator_rl_cluster.OrchestratorRLCluster(self.cluster), adapter
    )

    reference_buffered = []
    orchestrated_buffered = []

    def _capture(sink):
      def _buffer(metrics, **kwargs):
        del kwargs
        sink.append(dict(metrics))

      return _buffer

    self.cluster.buffer_metrics_async = _capture(reference_buffered)
    reference._process_results(  # pylint: disable=protected-access
        self._items(), mode=rl_cluster_lib.Mode.TRAIN, expected_step=0
    )

    self.cluster.buffer_metrics_async = _capture(orchestrated_buffered)
    adapter.postprocess_group(
        orchestrator,
        self._items(),
        compute_rewards=reference._compute_rewards,  # pylint: disable=protected-access
        mode=rl_cluster_lib.Mode.TRAIN,
        expected_step=0,
    )

    reference_keys = self._buffered_keys(reference_buffered)
    orchestrated_keys = self._buffered_keys(orchestrated_buffered)
    # The fixtures carry no timing dictionaries, which is exactly the shape
    # that makes the reference emit nothing at all; the orchestrated path
    # still reports its batch diagnostics.
    self.assertEmpty(reference_keys - orchestrated_keys)
    self.assertContainsSubset(
        {
            "generation/prompts/mean_length",
            "generation/completions/mean_length",
            "generation/completions/mean_raw_length",
            "generation/completions/clip_ratio",
            "rewards/advantage/mean",
            "rewards/advantage/std",
        },
        orchestrated_keys,
    )

  def test_user_metric_functions_are_invoked(self):
    algo_config = self._config()
    reference = self._reference_learner(algo_config)
    adapter = algorithm_adapter.GRPOAdapter(algo_config)
    orchestrator = rl_orchestrator.RLOrchestrator(
        orchestrator_rl_cluster.OrchestratorRLCluster(self.cluster), adapter
    )
    seen = []

    def _metric_fn(prompts, completions, advantages, rewards, **kwargs):
      del completions, advantages, rewards, kwargs
      seen.append(len(prompts))
      return {"user/metric": 1.0}

    buffered = []
    self.cluster.buffer_metrics_async = lambda metrics, **kwargs: buffered.append(
        dict(metrics)
    )
    adapter.postprocess_group(
        orchestrator,
        self._items(),
        compute_rewards=reference._compute_rewards,  # pylint: disable=protected-access
        mode=rl_cluster_lib.Mode.TRAIN,
        expected_step=0,
        metric_fns=[_metric_fn],
    )

    self.assertLen(seen, 1)
    self.assertIn("user/metric", self._buffered_keys(buffered))

  def test_trajectories_are_handed_to_the_logger(self):
    algo_config = self._config()
    reference = self._reference_learner(algo_config)
    adapter = algorithm_adapter.GRPOAdapter(algo_config)
    orchestrator = rl_orchestrator.RLOrchestrator(
        orchestrator_rl_cluster.OrchestratorRLCluster(self.cluster), adapter
    )
    logged = []

    adapter.postprocess_group(
        orchestrator,
        self._items(),
        compute_rewards=reference._compute_rewards,  # pylint: disable=protected-access
        mode=rl_cluster_lib.Mode.TRAIN,
        expected_step=0,
        trajectory_logger=type(
            "_Logger", (), {"log_item_async": lambda _self, traj: logged.append(traj)}
        )(),
    )

    self.assertLen(logged, 2)

  def _items_missing_logps(self):
    """A group whose second trajectory came back without log-probabilities."""
    items = self._items()
    items[1].traj["old_logprobs"] = None
    return items

  def test_orchestrated_path_rejects_a_trajectory_without_logps(self):
    """Zero-substitution would anchor the ratio at probability 1, silently."""
    algo_config = self._config()
    reference = self._reference_learner(algo_config)
    adapter = algorithm_adapter.GRPOAdapter(algo_config)
    orchestrator = rl_orchestrator.RLOrchestrator(
        orchestrator_rl_cluster.OrchestratorRLCluster(self.cluster), adapter
    )

    with self.assertRaises(algorithm_adapter.MissingRolloutLogpsError):
      adapter.postprocess_group(
          orchestrator,
          self._items_missing_logps(),
          compute_rewards=reference._compute_rewards,  # pylint: disable=protected-access
          mode=rl_cluster_lib.Mode.TRAIN,
          expected_step=0,
      )

  def test_reference_substitutes_zeros_unless_asked_to_be_strict(self):
    """The legacy behavior is unchanged by default, and refusable on request."""
    permissive = self._reference_learner(self._config())
    examples = permissive._process_results(  # pylint: disable=protected-access
        self._items_missing_logps(),
        mode=rl_cluster_lib.Mode.TRAIN,
        expected_step=0,
    )
    np.testing.assert_array_equal(
        np.asarray(examples[0].old_per_token_logps[1]),
        np.zeros(MAX_RESPONSE_LENGTH, dtype=np.float32),
    )

    strict = self._reference_learner(
        self._config(strict_rollout_logps=True)
    )
    with self.assertRaises(RuntimeError):
      strict._process_results(  # pylint: disable=protected-access
          self._items_missing_logps(),
          mode=rl_cluster_lib.Mode.TRAIN,
          expected_step=0,
      )

  @parameterized.named_parameters(
      ("no_kl", 0.0),
      ("with_kl", 0.04),
  )
  def test_sampler_importance_sampling_matches_the_reference(self, beta):
    """The correction weights and the ratio anchor must both match."""
    algo_config = self._config(sampler_is="token", beta=beta)
    expected, actual = self._run_both(algo_config, rl_cluster_lib.Mode.TRAIN)

    self.assertIsNotNone(expected[0].sampler_is_weights)
    self._assert_examples_equal(expected, actual)

  def test_sampler_importance_sampling_anchors_on_the_trainer_recompute(self):
    """Guards the cell above: with the correction on, old logps change source."""
    plain, _ = self._run_both(self._config(), rl_cluster_lib.Mode.TRAIN)
    corrected, _ = self._run_both(
        self._config(sampler_is="token"), rl_cluster_lib.Mode.TRAIN
    )

    self.assertIsNone(plain[0].sampler_is_weights)
    self.assertFalse(
        np.allclose(
            np.asarray(plain[0].old_per_token_logps),
            np.asarray(corrected[0].old_per_token_logps),
        )
    )

  @parameterized.named_parameters(
      ("on_policy", True),
      ("recomputed_logps", False),
  )
  def test_multiple_iterations_matches_the_reference(self, use_rollout_logps):
    """Legal on both paths once real old log-probabilities are guaranteed."""
    algo_config = self._config(
        num_iterations=2, use_rollout_logps=use_rollout_logps
    )
    expected, actual = self._run_both(algo_config, rl_cluster_lib.Mode.TRAIN)
    self._assert_examples_equal(expected, actual)

  def test_single_turn_assembly_pads_like_the_group_postprocess(self):
    """The two assembly entry points must lay tokens out identically.

    Single-turn assembly and group postprocess build train examples from
    different inputs but must agree on padding, or a loop that switches
    between them silently changes the trained layout.
    """
    algo_config = self._config()
    expected, _ = self._run_both(algo_config, rl_cluster_lib.Mode.TRAIN)
    from_postprocess = expected[0]

    adapter = algorithm_adapter.GRPOAdapter(algo_config)
    items = self._items()
    from_assembly = adapter.assemble_train_example(
        [item.traj["prompt_tokens"] for item in items],
        [item.traj["conversation_tokens"] for item in items],
        from_postprocess.advantages,
        max_prompt_length=TRAIN_MAX_PROMPT_LENGTH,
        max_response_length=MAX_RESPONSE_LENGTH,
        pad_id=self.cluster.rollout.pad_id(),
    )

    for field in ("prompt_ids", "prompt_mask", "completion_ids",
                  "completion_mask"):
      np.testing.assert_array_equal(
          np.asarray(getattr(from_assembly, field)),
          np.asarray(getattr(from_postprocess, field)),
          err_msg=f"{field} differs between the two assembly paths",
      )


if __name__ == "__main__":
  absltest.main()
