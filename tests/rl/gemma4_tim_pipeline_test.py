"""Pinned-image CPU construction, actual CLI builders and GRPO assembly.

Tiny learner fixture; engine/real-weight/TPU parity are not claimed.
"""

import contextlib
import copy
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from examples.frozenlake.gemma4_tim import artifacts, observer, recipe
from examples.frozenlake.gemma4_tim.pipeline import GemmaPipeline
from tunix.cli.config import HyperParameters
from tunix.cli.grpo_main import GrpoPipeline
from tunix.generate import tokenizer_adapter
from tunix.rl import rl_cluster as cluster_lib
from tunix.rl.agentic.agentic_grpo_learner import GRPOConfig, GRPOLearner
from tunix.rl.rollout import base_rollout
from tunix.tests import test_common


class PipelineTest(unittest.TestCase):

  def test_real_cli_default_consumer(self):
    # Only credential checks are mocked. No loader, engine or device runs.
    with mock.patch.object(HyperParameters, "_validate_tokenizer"), mock.patch.object(
        HyperParameters, "_validate_model_source"):
      original = GrpoPipeline([
          "grpo_main", str(recipe.REPO / "tunix/cli/base_agentic_config.yaml"),
          "override_config_file=" + str(recipe.REPO / "examples/frozenlake/configs/gemma4_e2b.yaml"),
      ])
    for arm in recipe.ARMS:
      with tempfile.TemporaryDirectory() as temp:
        pipeline = GemmaPipeline(recipe.resolve(arm), snapshot=Path("/model"),
                                 data=Path("/data"), output=Path(temp))
        original.compute_params(list(range(10000)))
        pipeline.compute_params(list(range(10000)))
        a = copy.deepcopy(original.config["rl_training_config"])
        b = copy.deepcopy(pipeline.config["rl_training_config"])
        a.pop("metrics_logging_options")
        b.pop("metrics_logging_options")
        b.pop("checkpoint_root_directory")
        self.assertEqual(a, b)
        self.assertEqual(b["actor_optimizer_config"]["warmup_steps"], 0.5)
        algo = pipeline._create_agentic_grpo_config()
        self.assertEqual(algo.sampler_is, "token" if arm == "tis" else None)
        self.assertTrue(algo.use_rollout_logps)
        training = pipeline.create_rl_training_config()
        self.assertEqual(training.mini_batch_size, 64)
        self.assertEqual(training.train_micro_batch_size, 2)
        self.assertIsNone(training.compute_logps_micro_batch_size)
        self.assertEqual(training.gradient_accumulation_steps, 32)

  def test_actual_four_device_role_mesh_and_rollout(self):
    self.assertEqual(jax.device_count(), 4, "run with four forced CPU devices")
    pipeline = GemmaPipeline(recipe.resolve("tis"), snapshot=Path("/model"),
                             data=Path("/data"), output=Path("/output"))
    # CPU devices have no TPU coords. Substitute only the physical box
    # allocator; role ownership and real four-device Mesh creation still run.
    with mock.patch("tunix.utils.mesh.allocate_named_mesh_device_slices",
                    return_value={"actor_model_config": list(jax.devices())}):
      roles = pipeline.create_role_to_mesh()
    actor = roles[cluster_lib.Role.ACTOR]
    self.assertEqual(actor.devices.shape, (1, 4))
    self.assertTrue(all(mesh is actor for mesh in roles.values()))
    rollout = pipeline.create_rollout_config(role_to_mesh=roles)
    self.assertEqual(rollout.rollout_vllm_model_version, "/model")
    self.assertEqual(rollout.rollout_vllm_max_num_seqs, 32)
    self.assertEqual(rollout.rollout_vllm_max_num_batched_tokens, 8192)
    self.assertEqual(rollout.temperature, 0.7)

  def test_observer_byte_negative_and_nonfinite(self):
    a = np.array([[-1.0, 0.0]], dtype=np.float32)
    b = a.copy()
    mask = np.ones(a.shape, dtype=bool)
    self.assertEqual(observer.compare(a, b, mask)["differing_bytes"], 0)
    b[0, 0] = -1.125
    self.assertGreater(observer.compare(a, b, mask)["differing_bytes"], 0)
    b[:] = a
    b[0, 1] = -0.0
    self.assertEqual(observer.compare(a, b, mask)["differing_elements"], 1)
    b[0, 0] = np.nan
    with self.assertRaises(recipe.RecipeError):
      observer.compare(a, b, mask)

  def test_real_learner_native_tis_selection_and_observer_neutrality(self):
    # Execute the existing GRPO _process_results, not a reimplementation of
    # the TIS formula. Rollout and trainer values deliberately disagree.
    vocab = test_common.MockVocab(mapping_text_to_id={
        "<pad>": 0, "<s>": 1, "</s>": 2, "hello": 3, "answer": 4,
    })
    tokenizer = tokenizer_adapter.TokenizerAdapter(vocab)
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]).reshape((1, 1)), ("fsdp", "tp"))
    for arm in ("native", "tis"):
      with self.subTest(arm=arm), tempfile.TemporaryDirectory() as temp:
        model = test_common.ToyTransformer(config=test_common.ModelConfig(vocab_size=5), rngs=nnx.Rngs(0))
        reference = test_common.ToyTransformer(config=test_common.ModelConfig(vocab_size=5), rngs=nnx.Rngs(0))
        cluster = cluster_lib.RLCluster(
            actor=model, reference=reference, tokenizer=tokenizer,
            cluster_config=cluster_lib.ClusterConfig(
                role_to_mesh={role: mesh for role in (cluster_lib.Role.ACTOR, cluster_lib.Role.REFERENCE, cluster_lib.Role.ROLLOUT)},
                rollout_engine="vanilla", offload_to_cpu=False,
                training_config=cluster_lib.RLTrainingConfig(actor_optimizer=optax.sgd(1e-3), max_steps=1, eval_every_n_steps=0),
                rollout_config=base_rollout.RolloutConfig(max_prompt_length=4, max_tokens_to_generate=4, return_logprobs=True),
            ),
        )
        algo = GRPOConfig(beta=0.0, num_generations=2, max_response_length=4,
                          sampler_is="token" if arm == "tis" else None,
                          use_rollout_logps=True)
        learner = GRPOLearner(cluster, algo_config=algo, chat_parser=mock.Mock())
        trajectories = [types.SimpleNamespace(group_id=0, pair_index=i, traj={
            "conversation_text": [{"role": "assistant", "content": "answer"}],
            "conversation_tokens": np.array([1, 0, 2]),
            "conversation_masks": np.array([1, 0, 1]),
            "old_logprobs": np.array([-2.0, 0.0, -2.0], dtype=np.float32),
            "policy_version": 0, "trajectory_reward": float(i),
            "prompt_tokens": np.array([0, 3]), "prompt_length": 1,
            "original_input": {"prompts": "hello"}, "group_id": 0,
        }) for i in range(2)]
        capture = observer.BatchObserver(Path(temp), recipe.resolve(arm))
        try:
          with mock.patch.object(cluster, "get_actor_per_token_logps", return_value=jnp.full((2, 4), -1.0)):
            unobserved = learner._process_results(trajectories, expected_step=0)
            learner._processed_batch_observer = capture
            observed = learner._process_results(trajectories, expected_step=0)
          for a, b in zip(jax.tree.leaves(unobserved), jax.tree.leaves(observed)):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
          batch = observed[0]
          self.assertEqual(capture.records, 1)
          if arm == "native":
            np.testing.assert_array_equal(np.asarray(batch.old_per_token_logps)[0], [-2, 0, -2, 0])
            self.assertIsNone(batch.sampler_is_weights)
          else:
            np.testing.assert_array_equal(batch.old_per_token_logps, -np.ones((2, 4)))
            np.testing.assert_array_equal(np.asarray(batch.sampler_is_weights)[0], [2, 0, 2, 0])
          with mock.patch.object(cluster, "get_actor_per_token_logps", return_value=jnp.full((2, 4), -1.0)):
            with self.assertRaisesRegex(recipe.RecipeError, "duplicate trajectory across"):
              learner._process_results(trajectories, expected_step=0)
          learner._processed_batch_observer = mock.Mock(side_effect=recipe.RecipeError("negative observer"))
          with mock.patch.object(cluster, "get_actor_per_token_logps", return_value=jnp.full((2, 4), -1.0)):
            with self.assertRaisesRegex(recipe.RecipeError, "negative observer"):
              learner._process_results(trajectories, expected_step=0)
        finally:
          if learner._trajectory_logger is not None:
            learner._trajectory_logger.stop()
          learner.loop.call_soon_threadsafe(learner.loop.stop)
          cluster.close()


if __name__ == "__main__":
  unittest.main()
