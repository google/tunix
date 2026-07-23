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

"""Tests for prepared diffusion rollouts in the standard GRPO family."""

# pylint: disable=missing-class-docstring,missing-function-docstring,protected-access

import contextlib
import types
from unittest import mock

from absl.testing import absltest
from flax import nnx
import jax
from jax.interpreters import pxla
import jax.numpy as jnp
import numpy as np
import optax
from tunix.diffusion import types as diffusion_types
from tunix.perf import trace as trace_lib
from tunix.perf.experimental import tracer as perf_tracer_v2
from tunix.rl import algo_core
from tunix.rl import common
from tunix.rl import diffusion
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl import rl_learner
from tunix.rl import trainer as rl_trainer
from tunix.rl.agentic import agentic_grpo_learner
from tunix.rl.grpo import dapo_learner
from tunix.rl.grpo import grpo_learner
from tunix.rl.ppo import ppo_learner
from tunix.rl.rollout import base_rollout


class _LogitModel(nnx.Module):

  def __init__(self, logits):
    self.logits = nnx.Param(jnp.asarray(logits, dtype=jnp.float32))


def _score_fn(model, model_inputs):
  return model.logits[model_inputs["example_ids"]]


def _batch(target_ids, loss_weights=None):
  target_ids = jnp.asarray(target_ids, dtype=jnp.int32)
  if loss_weights is None:
    loss_weights = jnp.ones_like(target_ids, dtype=jnp.float32)
  return diffusion_types.DiffusionTokenBatch.create(
      model_inputs={
          "example_ids": jnp.arange(target_ids.shape[0], dtype=jnp.int32),
          "conditioning": (
              jnp.arange(target_ids.shape[0] * 2, dtype=jnp.float32).reshape(
                  target_ids.shape[0], 2
              )
          ),
      },
      target_ids=target_ids,
      loss_weights=jnp.asarray(loss_weights, dtype=jnp.float32),
  )


def _config(*, beta=0.0, num_iterations=1):
  config = grpo_learner.GRPOConfig(
      num_generations=2,
      num_iterations=num_iterations,
      beta=beta,
  )
  config.temperature = 1.0
  return config


def _example(batch, *, old_logps=None, ref_logps=None):
  batch_size = batch.target_ids.shape[0]
  return grpo_learner.TrainExample(
      prompt_ids=jnp.zeros((batch_size, 1), dtype=jnp.int32),
      prompt_mask=jnp.ones((batch_size, 1), dtype=jnp.bool_),
      completion_ids=batch.target_ids,
      completion_mask=batch.loss_weights,
      advantages=jnp.array([1.0, -0.25], dtype=jnp.float32)[:batch_size],
      ref_per_token_logps=ref_logps,
      old_per_token_logps=old_logps,
      diffusion_batch=batch,
  )


class _FakeRollout:

  def pad_id(self):
    return 0

  def eos_id(self):
    return 3


class _SamePadEosRollout(_FakeRollout):

  def eos_id(self):
    return 0


class _RoutingCluster:

  def __init__(self, rollout_output, ref_scores, anchor_scores):
    self._rollout_output = rollout_output
    self._ref_scores = ref_scores
    self._anchor_scores = anchor_scores
    self.ref_calls = []
    self.anchor_calls = []
    self.events = []
    self.placed_examples = []
    self.rollout = _FakeRollout()
    self.global_steps = 0
    self.perf = trace_lib.NoopTracer()
    self.perf_v2 = perf_tracer_v2.NoopTracer()
    empty_devices = types.SimpleNamespace(devices=[])
    self.r2m = {
        rl_cluster_lib.Role.REFERENCE: empty_devices,
        rl_cluster_lib.Role.ACTOR: empty_devices,
    }
    self.cluster_config = types.SimpleNamespace(
        rollout_config=base_rollout.RolloutConfig(
            max_tokens_to_generate=2,
            temperature=1.0,
        )
    )

  def generate(self, **unused_kwargs):
    self.events.append("generate")
    return self._rollout_output

  def snapshot_anchor_policy(self):
    self.events.append("snapshot")

  def place_pytree_on_role(self, pytree, role):
    self.placed_examples.append((pytree, role))
    return pytree

  def get_ref_diffusion_per_token_logps(self, **kwargs):
    self.ref_calls.append(kwargs)
    return self._ref_scores

  def get_anchor_diffusion_per_token_logps(self, **kwargs):
    self.anchor_calls.append(kwargs)
    return self._anchor_scores

  def get_ref_per_token_logps(self, **unused_kwargs):
    raise AssertionError("diffusion reference scoring used the AR path")

  def get_old_per_token_logps(self, **unused_kwargs):
    raise AssertionError(
        "diffusion old-policy scoring used the rollout AR path"
    )

  def buffer_metrics(self, unused_metrics, mode):
    del unused_metrics, mode


class DiffusionGRPOLossTest(absltest.TestCase):

  def test_live_scores_have_gradient_and_ratio_is_one_before_update(self):
    model = _LogitModel([
        [[2.0, -1.0, 0.0], [-1.0, 0.0, 2.0]],
        [[0.0, 1.0, -2.0], [2.0, -1.0, 0.0]],
    ])
    batch = _batch([[0, 2], [1, 0]])
    old_logps = diffusion.compute_diffusion_per_token_logps(
        model, batch, _score_fn, stop_gradient=True
    )
    example = _example(batch, old_logps=old_logps)
    config = _config()

    output = algo_core.grpo_loss_fn(
        model,
        example,
        config,
        pad_id=0,
        eos_id=3,
        diffusion_logits_fn=_score_fn,
    )
    grads = nnx.grad(
        lambda policy: algo_core.grpo_loss_fn(
            policy,
            example,
            config,
            pad_id=0,
            eos_id=3,
            diffusion_logits_fn=_score_fn,
        ).primary_loss.compute()
    )(model)

    self.assertAlmostEqual(float(output.aux_metrics["is_ratio/mean"]), 1.0)
    self.assertAlmostEqual(float(output.aux_metrics["ppo_kl"].compute()), 0.0)
    self.assertGreater(float(jnp.linalg.norm(grads.logits[...])), 0.0)

  def test_reference_kl_uses_prepared_scores(self):
    model = _LogitModel([
        [[2.0, 0.0], [0.0, 2.0]],
        [[1.0, -1.0], [-1.0, 1.0]],
    ])
    reference = _LogitModel([
        [[0.0, 2.0], [2.0, 0.0]],
        [[-1.0, 1.0], [1.0, -1.0]],
    ])
    batch = _batch([[0, 1], [0, 1]])
    old_logps = diffusion.compute_diffusion_per_token_logps(
        model, batch, _score_fn
    )
    ref_logps = diffusion.compute_diffusion_per_token_logps(
        reference, batch, _score_fn
    )

    output = algo_core.grpo_loss_fn(
        model,
        _example(batch, old_logps=old_logps, ref_logps=ref_logps),
        _config(beta=0.1),
        pad_id=0,
        eos_id=3,
        diffusion_logits_fn=_score_fn,
    )

    self.assertGreater(float(output.aux_metrics["kl"].compute()), 0.0)

  def test_ar_default_still_calls_standard_scorer(self):
    model = _LogitModel(jnp.zeros((2, 2, 3)))
    example = grpo_learner.TrainExample(
        prompt_ids=jnp.zeros((2, 1), dtype=jnp.int32),
        prompt_mask=jnp.ones((2, 1), dtype=jnp.bool_),
        completion_ids=jnp.array([[1, 2], [2, 1]], dtype=jnp.int32),
        completion_mask=jnp.ones((2, 2), dtype=jnp.float32),
        advantages=jnp.array([1.0, -1.0]),
        ref_per_token_logps=None,
        old_per_token_logps=None,
    )
    with mock.patch.object(
        common,
        "compute_per_token_logps",
        return_value=(jnp.zeros((2, 2)), jnp.zeros((2, 2))),
    ) as ar_scorer:
      algo_core.grpo_loss_fn(model, example, _config(), pad_id=0, eos_id=3)

    ar_scorer.assert_called_once()

  def test_rejects_diffusion_batch_without_scorer(self):
    batch = _batch([[0, 1], [1, 0]])
    with self.assertRaisesRegex(ValueError, "refusing to score"):
      algo_core.grpo_loss_fn(
          _LogitModel(jnp.zeros((2, 2, 2))),
          _example(batch),
          _config(),
          pad_id=0,
          eos_id=3,
      )


class DiffusionGRPORoutingTest(absltest.TestCase):

  def test_generation_context_hook_runs_once_before_rollout_chunks(self):
    events = []

    def generate(prompts, unused_config):
      events.append(("generate", tuple(prompts)))
      return base_rollout.RolloutOutput(
          text=["x"] * len(prompts),
          logits=None,
          tokens=[np.array([1])] * len(prompts),
          left_padded_prompt_tokens=np.ones((len(prompts), 1), dtype=np.int32),
          logprobs=None,
      )

    rollout = types.SimpleNamespace(
        model=object,
        generate=generate,
        set_generation_context=lambda **kwargs: events.append(
            ("context", kwargs)
        ),
    )
    cluster = object.__new__(rl_cluster_lib.RLCluster)
    cluster._rollout = rollout
    cluster.tokenizer = None
    cluster.global_steps = 7
    cluster.cluster_config = types.SimpleNamespace(
        offload_to_cpu=False,
        rollout_config=base_rollout.RolloutConfig(),
    )
    cluster._get_mesh_and_logical_axis_rules_cm = (  # pylint: disable=protected-access
        lambda unused_role: contextlib.nullcontext(
            (types.SimpleNamespace(devices=[]), None)
        )
    )
    cluster._maybe_load_model_from_cpu = (  # pylint: disable=protected-access
        lambda unused_model, unused_role: None
    )
    cluster._maybe_offload_model_to_cpu = (  # pylint: disable=protected-access
        lambda unused_model, unused_role: None
    )
    cluster._perf = trace_lib.NoopTracer()
    cluster._perf_v2 = perf_tracer_v2.NoopTracer()

    cluster.generate(
        ["p0", "p1", "p2"],
        mode=rl_cluster_lib.Mode.EVAL,
        micro_batch_size=2,
    )

    self.assertEqual(
        events,
        [
            (
                "context",
                {"global_step": 7, "mode": rl_cluster_lib.Mode.EVAL},
            ),
            ("generate", ("p0", "p1")),
            ("generate", ("p2",)),
        ],
    )

  def test_learner_routes_reference_and_old_policy_through_prepared_scorer(
      self,
  ):
    batch = _batch([[1, 2], [2, 1]])
    rollout_output = base_rollout.RolloutOutput(
        text=["a", "b"],
        logits=None,
        tokens=[np.array([1, 2]), np.array([2, 1])],
        left_padded_prompt_tokens=np.array([[4], [5]]),
        logprobs=[np.array([-9.0, -9.0]), np.array([-8.0, -8.0])],
        diffusion_batch=batch,
    )
    ref_scores = jnp.full((2, 2), -0.4)
    anchor_scores = jnp.full((2, 2), -0.5)
    cluster = _RoutingCluster(rollout_output, ref_scores, anchor_scores)
    learner = object.__new__(grpo_learner.GRPOLearner)
    learner.rl_cluster = cluster
    learner.algo_config = _config(beta=0.1, num_iterations=2)
    learner.diffusion_logits_fn = _score_fn
    learner.should_sync_weights = False
    learner._rollout_micro_batch_size = 1
    learner._compute_logps_micro_batch_size = 1
    learner.metric_fns = []
    learner._compute_rewards = lambda **unused_kwargs: np.array([1.0, 2.0])

    result = learner._generate_and_compute_advantage({"prompts": ["p0", "p1"]})

    self.assertLen(cluster.ref_calls, 1)
    self.assertLen(cluster.anchor_calls, 1)
    self.assertEqual(cluster.events[:2], ["snapshot", "generate"])
    self.assertIs(result.diffusion_batch, batch)
    self.assertLen(cluster.placed_examples, 1)
    self.assertIs(cluster.placed_examples[0][1], rl_cluster_lib.Role.ACTOR)
    np.testing.assert_array_equal(result.ref_per_token_logps, ref_scores)
    np.testing.assert_array_equal(result.old_per_token_logps, anchor_scores)

  def test_prepared_targets_and_weights_must_match_completion(self):
    with self.assertRaisesRegex(ValueError, "target_ids must exactly match"):
      grpo_learner._validate_diffusion_rollout_batch(  # pylint: disable=protected-access
          _batch([[1, 0], [2, 0]], [[1, 0], [1, 0]]),
          [np.array([2]), np.array([2])],
      )
    with self.assertRaisesRegex(ValueError, "active loss_weights"):
      grpo_learner._validate_diffusion_rollout_batch(  # pylint: disable=protected-access
          _batch([[1, 0], [2, 0]], [[0, 1], [1, 0]]),
          [np.array([1]), np.array([2])],
      )

  def test_inactive_targets_can_retain_full_trace_sentinels(self):
    batch = _batch([[1, -1], [2, 99]], [[1, 0], [1, 0]])

    result = grpo_learner._validate_diffusion_rollout_batch(  # pylint: disable=protected-access
        batch,
        [np.array([1]), np.array([2])],
    )

    self.assertIs(result, batch)

  def test_sampler_parity_requires_full_trace_logprobs(self):
    with self.assertRaisesRegex(ValueError, "full prepared trace"):
      grpo_learner._stack_rollout_logps(  # pylint: disable=protected-access
          [np.array([-0.1, -0.2])], batch_size=1, trace_length=3
      )

  def test_missing_prepared_batch_fails_instead_of_falling_back(self):
    rollout_output = base_rollout.RolloutOutput(
        text=["a", "b"],
        logits=None,
        tokens=[np.array([1]), np.array([2])],
        left_padded_prompt_tokens=np.array([[4], [5]]),
        logprobs=None,
    )
    cluster = _RoutingCluster(
        rollout_output, jnp.zeros((2, 2)), jnp.zeros((2, 2))
    )
    learner = object.__new__(grpo_learner.GRPOLearner)
    learner.rl_cluster = cluster
    learner.algo_config = _config()
    learner.diffusion_logits_fn = _score_fn
    learner.should_sync_weights = True
    learner._rollout_micro_batch_size = 1
    learner._compute_logps_micro_batch_size = 1
    learner.metric_fns = []

    with self.assertRaisesRegex(ValueError, "requires every rollout"):
      learner._generate_and_compute_advantage({"prompts": ["p0", "p1"]})

  def test_full_trace_survives_eos_when_pad_and_eos_ids_match(self):
    batch = _batch([[1, 0, 7], [2, 0, 8]])
    rollout_output = base_rollout.RolloutOutput(
        text=["a", "b"],
        logits=None,
        tokens=[np.array([1, 0]), np.array([2, 0])],
        left_padded_prompt_tokens=np.array([[4], [5]]),
        logprobs=None,
        diffusion_batch=batch,
    )
    cluster = _RoutingCluster(
        rollout_output, jnp.zeros((2, 3)), jnp.zeros((2, 3))
    )
    cluster.rollout = _SamePadEosRollout()
    learner = object.__new__(grpo_learner.GRPOLearner)
    learner.rl_cluster = cluster
    learner.algo_config = _config()
    learner.diffusion_logits_fn = _score_fn
    learner.should_sync_weights = True
    learner._rollout_micro_batch_size = 1
    learner._compute_logps_micro_batch_size = 1
    learner.metric_fns = []
    learner._compute_rewards = lambda **unused_kwargs: np.array([1.0, 2.0])

    result = learner._generate_and_compute_advantage({"prompts": ["p0", "p1"]})

    np.testing.assert_array_equal(result.completion_ids, [[1, 0, 7], [2, 0, 8]])
    np.testing.assert_array_equal(result.completion_mask, np.ones((2, 3)))

  def test_ppo_rejects_prepared_diffusion_rollout(self):
    rollout_output = base_rollout.RolloutOutput(
        text=["a"],
        logits=None,
        tokens=[np.array([1])],
        left_padded_prompt_tokens=np.array([[4]]),
        logprobs=None,
        diffusion_batch=_batch([[1]]),
    )
    learner = object.__new__(ppo_learner.PPOLearner)
    learner.rl_cluster = types.SimpleNamespace(
        cluster_config=types.SimpleNamespace(
            rollout_config=base_rollout.RolloutConfig()
        ),
        rollout=_FakeRollout(),
        generate=lambda **unused_kwargs: rollout_output,
    )
    learner._rollout_micro_batch_size = 1

    with self.assertRaisesRegex(ValueError, "not supported by PPO"):
      learner._generate_and_compute_advantage({"prompts": ["p0"]})

  def test_agentic_rl_rejects_prepared_diffusion_rollout(self):
    rollout_output = base_rollout.RolloutOutput(
        text=["a"],
        logits=None,
        tokens=[np.array([1])],
        left_padded_prompt_tokens=np.array([[4]]),
        logprobs=None,
        diffusion_batch=_batch([[1]]),
    )
    learner = object.__new__(agentic_grpo_learner.GRPOLearner)
    learner.rl_cluster = types.SimpleNamespace(
        generate=lambda **unused_kwargs: rollout_output
    )
    learner.chat_parser = None
    learner.policy_version = 0
    learner._full_batch_size = 0

    with self.assertRaisesRegex(ValueError, "not supported by agentic RL"):
      learner._model_call([])


class DiffusionPolicySnapshotTest(absltest.TestCase):

  def _bare_cluster(self):
    cluster = object.__new__(rl_cluster_lib.RLCluster)
    cluster.cluster_config = types.SimpleNamespace(
        offload_to_cpu=False,
        training_config=types.SimpleNamespace(data_sharding_axis=()),
    )
    cluster.r2m = {
        rl_cluster_lib.Role.ACTOR: pxla.thread_resources.env.physical_mesh,
        rl_cluster_lib.Role.REFERENCE: pxla.thread_resources.env.physical_mesh,
    }
    cluster._get_mesh_and_logical_axis_rules_cm = (  # pylint: disable=protected-access
        lambda unused_role: contextlib.nullcontext()
    )
    cluster._maybe_load_model_from_cpu = (  # pylint: disable=protected-access
        lambda unused_model, unused_role: None
    )
    cluster._maybe_offload_model_to_cpu = (  # pylint: disable=protected-access
        lambda unused_model, unused_role: None
    )
    cluster.get_rollout_config = lambda mode: base_rollout.RolloutConfig(
        temperature=1.0
    )
    return cluster

  def test_reference_scorer_microbatches_the_whole_prepared_pytree(self):
    model = _LogitModel([
        [[2.0, 0.0]],
        [[0.0, 2.0]],
        [[1.0, -1.0]],
    ])
    batch = _batch([[0], [1], [0]])
    seen_shapes = []

    def score_fn(policy, model_inputs):
      seen_shapes.append(
          tuple(leaf.shape[0] for leaf in jax.tree.leaves(model_inputs))
      )
      return _score_fn(policy, model_inputs)

    cluster = self._bare_cluster()
    cluster._inference_worker = types.SimpleNamespace(  # pylint: disable=protected-access
        get_model=lambda unused_name: model
    )
    scores = cluster.get_ref_diffusion_per_token_logps(
        batch, score_fn, micro_batch_size=2
    )

    self.assertEqual(seen_shapes, [(2, 2), (1, 1)])
    expected = diffusion.compute_diffusion_per_token_logps(
        model, batch, _score_fn
    )
    np.testing.assert_allclose(scores, expected)

  def test_anchor_scorer_uses_start_of_step_state_after_live_model_changes(
      self,
  ):
    initial_logits = jnp.array([
        [[2.0, 0.0]],
        [[0.0, 2.0]],
        [[1.0, -1.0]],
    ])
    live_model = _LogitModel(initial_logits)
    _, initial_state = nnx.split(live_model)
    anchor_state = jax.tree.map(jnp.copy, initial_state)
    live_model.logits[...] = -initial_logits
    batch = _batch([[0], [1], [0]])

    cluster = self._bare_cluster()
    cluster._actor_trainer = types.SimpleNamespace(  # pylint: disable=protected-access
        model=live_model
    )
    cluster._anchor_policy_state = anchor_state  # pylint: disable=protected-access
    cluster._default_memory_kind = jax.devices()[0].default_memory().kind  # pylint: disable=protected-access
    cluster._is_state_on_device = lambda unused_state: True  # pylint: disable=protected-access
    scores = cluster.get_anchor_diffusion_per_token_logps(
        batch, _score_fn, micro_batch_size=2
    )

    expected = diffusion.compute_diffusion_per_token_logps(
        _LogitModel(initial_logits), batch, _score_fn
    )
    live_scores = diffusion.compute_diffusion_per_token_logps(
        live_model, batch, _score_fn
    )
    np.testing.assert_allclose(scores, expected)
    self.assertFalse(np.allclose(scores, live_scores))

  def test_dapo_forwards_diffusion_scorer(self):
    scorer = mock.Mock()
    with mock.patch.object(grpo_learner.GRPOLearner, "__init__") as base_init:
      dapo_learner.DAPOLearner(
          rl_cluster=mock.Mock(),
          algo_config=dapo_learner.DAPOConfig(overlong_buffer=None),
          reward_fns=lambda **unused_kwargs: [0.0],
          diffusion_logits_fn=scorer,
      )

    self.assertIs(base_init.call_args.kwargs["diffusion_logits_fn"], scorer)

  def test_complete_train_example_is_placed_on_actor_mesh(self):
    if jax.device_count() < 2:
      self.skipTest("requires two devices to construct disjoint role meshes")
    actor_mesh = jax.sharding.Mesh(np.asarray([jax.devices()[0]]), ("fsdp",))
    other_mesh = jax.sharding.Mesh(np.asarray([jax.devices()[1]]), ("fsdp",))
    other_sharding = jax.sharding.NamedSharding(
        other_mesh, jax.sharding.PartitionSpec()
    )
    batch = jax.tree.map(
        lambda value: jax.device_put(value, other_sharding),
        _batch([[0, 1], [1, 0]]),
    )
    ref_logps = jax.device_put(jnp.full((2, 2), -0.5), other_sharding)
    example = _example(batch, ref_logps=ref_logps)

    cluster = object.__new__(rl_cluster_lib.RLCluster)
    cluster.r2m = {rl_cluster_lib.Role.ACTOR: actor_mesh}
    cluster.cluster_config = types.SimpleNamespace(
        training_config=types.SimpleNamespace(data_sharding_axis=("fsdp",))
    )
    cluster._get_mesh_and_logical_axis_rules_cm = (  # pylint: disable=protected-access
        lambda role: cluster.r2m[role]
    )
    placed = cluster.place_pytree_on_role(example, rl_cluster_lib.Role.ACTOR)

    for leaf in jax.tree.leaves(placed):
      self.assertIsInstance(leaf, jax.Array)
      self.assertEqual(leaf.sharding.mesh, actor_mesh)

  def test_disjoint_rollout_is_initially_synced_without_advancing_step(self):
    class MinimalLearner(rl_learner.RLLearner):

      def _generate_and_compute_advantage(self, training_input, mode):
        del training_input, mode

      def _compute_trajectory_ids(self, example, steps):
        del example, steps
        return []

      def _num_iterations(self):
        return 1

      def _num_generations(self):
        return 1

    actor_trainer = types.SimpleNamespace(
        model=_LogitModel(jnp.zeros((1, 1, 2))),
        restored_global_step=lambda: 7,
        iter_steps=0,
        is_managed_externally=False,
    )
    cluster = types.SimpleNamespace(
        actor_trainer=actor_trainer,
        rollout=types.SimpleNamespace(
            model=lambda: _LogitModel(jnp.ones((1, 1, 2)))
        ),
        cluster_config=types.SimpleNamespace(
            training_config=types.SimpleNamespace(
                rollout_micro_batch_size=None,
                compute_logps_micro_batch_size=None,
            ),
            role_to_mesh={
                rl_cluster_lib.Role.ACTOR: "actor",
                rl_cluster_lib.Role.ROLLOUT: "rollout",
            },
        ),
        global_steps=0,
        sync_weights=mock.Mock(),
    )
    with mock.patch(
        "tunix.rl.rl_learner.rl_utils.is_sharing_weights", return_value=False
    ), mock.patch("tunix.rl.rl_learner.sft_utils.show_hbm_usage"):
      learner = MinimalLearner(
          rl_cluster=cluster,
          algo_config=_config(),
          reward_fns=lambda **unused_kwargs: [0.0],
      )

    cluster.sync_weights.assert_called_once_with(increment_global_steps=False)
    self.assertEqual(cluster.global_steps, 7)
    learner.executor.shutdown()


class RLCheckpointMetadataTest(absltest.TestCase):

  def test_empty_metadata_default_preserves_existing_behavior(self):
    config = rl_cluster_lib.RLTrainingConfig(
        actor_optimizer=optax.sgd(1e-3),
        eval_every_n_steps=1,
    )
    self.assertEmpty(config.checkpoint_metadata)

    trainer = object.__new__(rl_trainer.Trainer)
    trainer._train_steps = 4
    trainer._restored_custom_metadata = {"global_step": 4}
    trainer._validate_restored_checkpoint_metadata({})  # pylint: disable=protected-access

  def test_restored_metadata_requires_every_matching_value(self):
    trainer = object.__new__(rl_trainer.Trainer)
    trainer._train_steps = 4
    trainer._restored_custom_metadata = {
        "global_step": 4,
        "objective": "diffusion-grpo",
        "stop_ids": [1, 2],
    }
    trainer._validate_restored_checkpoint_metadata(  # pylint: disable=protected-access
        {"objective": "diffusion-grpo", "stop_ids": (1, 2)}
    )

    with self.assertRaisesRegex(ValueError, "mismatch"):
      trainer._validate_restored_checkpoint_metadata(  # pylint: disable=protected-access
          {"objective": "autoregressive"}
      )
    with self.assertRaisesRegex(ValueError, "missing required"):
      trainer._validate_restored_checkpoint_metadata(  # pylint: disable=protected-access
          {"mask_id": 123}
      )

  def test_metadata_config_rejects_reserved_and_non_json_values(self):
    with self.assertRaisesRegex(ValueError, "reserved"):
      rl_cluster_lib.RLTrainingConfig(
          actor_optimizer=optax.sgd(1e-3),
          eval_every_n_steps=1,
          checkpoint_metadata={"role": "actor"},
      )
    with self.assertRaisesRegex(TypeError, "JSON scalar or tuple"):
      rl_cluster_lib.RLTrainingConfig(
          actor_optimizer=optax.sgd(1e-3),
          eval_every_n_steps=1,
          checkpoint_metadata={"bad": [1, 2]},
      )


if __name__ == "__main__":
  absltest.main()
