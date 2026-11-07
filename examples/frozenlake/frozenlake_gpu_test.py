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

"""End-to-end GPU test for FrozenLake environment, agent, and sub-1B Qwen3 GRPO step."""

import os
import tempfile
from unittest import mock

from absl.testing import absltest
from flax import nnx
import jax
from jax import numpy as jnp
import numpy as np
import optax
from tunix.models.qwen3 import model as qwen3_model
from tunix.oss import utils as oss_utils
from examples.frozenlake import agent as frozenlake_agent
from examples.frozenlake import env as frozenlake_env
from tunix.rl import algo_core
from tunix.rl import common as rl_common
from tunix.rl.agentic import agentic_grpo_learner
from tunix.sft import utils as sft_utils
from tunix.utils import topology


class FrozenLakeGpuEndToEndTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "gpu":
      self.skipTest("Requires GPU backend")

  def test_gpu_topology_and_hbm_reporting(self):
    self.assertTrue(topology.is_gpu_family("NVIDIA H100 80GB HBM3"))
    self.assertTrue(topology.is_gpu_family("NVIDIA A100-SXM4-40GB"))
    self.assertTrue(topology.is_gpu_family("NVIDIA L4"))
    self.assertEqual(
        topology.supported_topology_shapes_for_chip_count(
            "NVIDIA H100 80GB HBM3", 8
        ),
        [],
    )
    sft_utils.show_hbm_usage("FrozenLake GPU HBM test")

  def test_hf_pipeline_marker_and_recipe_remote_data_guard(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      src_file = os.path.join(tmpdir, "src_model.safetensors")
      with open(src_file, "wb") as f:
        f.write(b"weights")
      dst_dir = os.path.join(tmpdir, "download")
      with (
          mock.patch.object(
              oss_utils.hf,
              "list_repo_files",
              return_value=["model.safetensors", "original/ignored.bin"],
          ),
          mock.patch.object(
              oss_utils.hf, "hf_hub_download", return_value=src_file
          ),
      ):
        oss_utils.hf_pipeline("Qwen/Qwen3-0.6B", dst_dir)
      marker = os.path.join(dst_dir, ".hf_complete")
      self.assertTrue(os.path.exists(marker))
      with open(marker) as f:
        self.assertEqual(f.read(), "Qwen/Qwen3-0.6B")

    recipe_path = os.path.join(
        os.path.dirname(__file__), "train_frozenlake_qwen3.py"
    )
    with open(recipe_path) as f:
      recipe_src = f.read()
    self.assertIn(
        'is_local_path = "://" not in train_ds_path and "://" not in'
        " test_ds_path",
        recipe_src,
    )
    self.assertIn(
        'os.path.join(MODEL_DOWNLOAD_DIR, ".hf_complete")', recipe_src
    )

  def test_frozenlake_agent_env_and_sub1b_qwen3_grpo_step_on_gpu(self):
    # 1. Step through a solvable 4x4 FrozenLake episode via FrozenLakeAgent.
    env = frozenlake_env.FrozenLakeEnv(
        entry={"size": 4, "p": 0.8, "seed": 42},
        max_steps=5,
        is_slippery=False,
    )
    agent = frozenlake_agent.FrozenLakeAgent()
    obs, _ = env.reset()
    agent.update_from_env(obs, reward=0.0, done=False, info={})
    self.assertGreaterEqual(len(agent.chat_completions), 2)

    action_step = agent.update_from_model("Let us move right.\n```Right```")
    step_out = env.step(action_step.action)
    next_obs, reward, done = step_out[0], step_out[1], step_out[2]
    self.assertIsInstance(next_obs, str)
    self.assertIsInstance(float(reward), float)
    self.assertIsInstance(bool(done), bool)

    # 2. Run a full GRPO policy gradient update step on the GPU mesh using
    # Tunix's canonical algo_core.grpo_loss_fn and Tokamax Flash Attention.
    mesh = jax.sharding.Mesh(
        np.array(jax.devices()[:1]).reshape(1, 1), ("fsdp", "tp")
    )
    cfg = qwen3_model.ModelConfig(
        num_layers=4,
        vocab_size=2048,
        embed_dim=512,
        hidden_dim=1536,
        num_heads=8,
        head_dim=64,
        num_kv_heads=4,
        rope_theta=1_000_000,
        norm_eps=1e-6,
        use_flash_attention=True,
        dtype=jnp.dtype(jnp.bfloat16),
    )
    with mesh:
      actor = qwen3_model.Qwen3(cfg, rngs=nnx.Rngs(params=0))
      ref = qwen3_model.Qwen3(cfg, rngs=nnx.Rngs(params=0))
      optimizer = nnx.Optimizer(
          actor, optax.adamw(learning_rate=1e-4), wrt=nnx.Param
      )

      group_size = 4
      prompt_len = 64
      completion_len = 64
      prompt_ids = jax.random.randint(
          jax.random.PRNGKey(1), (group_size, prompt_len), minval=1, maxval=2000
      )
      completion_ids = jax.random.randint(
          jax.random.PRNGKey(2),
          (group_size, completion_len),
          minval=1,
          maxval=2000,
      )
      prompt_mask = jnp.ones_like(prompt_ids, dtype=jnp.bool_)
      completion_mask = jnp.ones_like(completion_ids, dtype=jnp.bool_)
      rewards = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
      advantages = jnp.asarray(
          algo_core.compute_advantages(rewards, num_generations=group_size)
      )

      ref_graphdef, ref_state = nnx.split(ref)
      ref_token_logps = rl_common.compute_per_token_logps(
          ref_graphdef,
          ref_state,
          prompt_tokens=prompt_ids,
          completion_tokens=completion_ids,
          pad_id=0,
          eos_id=2,
          stop_gradient=True,
          return_entropy=False,
      )
      assert isinstance(ref_token_logps, jax.Array)
      train_example = agentic_grpo_learner.TrainExample(
          prompt_ids=prompt_ids,
          prompt_mask=prompt_mask,
          completion_ids=completion_ids,
          completion_mask=completion_mask,
          ref_per_token_logps=ref_token_logps,
          advantages=advantages,
          old_per_token_logps=ref_token_logps,
      )
      grpo_config = agentic_grpo_learner.GRPOConfig(
          num_generations=group_size,
          beta=0.001,
          epsilon=0.2,
          loss_algo="grpo",
      )
      grpo_config.temperature = 1.0  # pyrefly: ignore[missing-attribute]

      def loss_fn(model: qwen3_model.Qwen3) -> jax.Array:
        loss_out = algo_core.grpo_loss_fn(
            model,
            train_example=train_example,
            algo_config=grpo_config,
            pad_id=0,
            eos_id=2,
        )
        return loss_out.primary_loss.compute()

      loss_before, grads = nnx.value_and_grad(loss_fn)(actor)
      optimizer.update(actor, grads)
      loss_after = loss_fn(actor)

      self.assertTrue(bool(jnp.isfinite(loss_before)))
      self.assertTrue(bool(jnp.isfinite(loss_after)))
      self.assertLess(float(loss_after), float(loss_before))


if __name__ == "__main__":
  absltest.main()
