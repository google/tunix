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

import asyncio
import dataclasses
import gc
from io import StringIO
import os
import sys
import tempfile
import types
import unittest
from unittest import mock
from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import transformers
from tunix.generate import mappings
from tunix.generate import tokenizer_adapter as tok_adapter
from tunix.generate import vllm_sampler
from tunix.models.qwen2 import model as qwen2_model
from tunix.models.qwen2 import params as qwen2_params
from tunix.models.qwen3 import mapping_vllm_jax
from tunix.models.qwen3 import model as qwen3_model
from tunix.models.qwen3 import params as qwen3_params
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.agentic import agentic_grpo_learner
from tunix.rl.agentic.agents import agent_types
from tunix.rl.agentic.agents.base_agent import ConversationAgentBase
from tunix.rl.agentic.environments.base_environment import BaseTaskEnv
from tunix.rl.agentic.environments.base_environment import EnvStepResult
from tunix.rl.agentic.trajectory import trajectory_collect_engine
from tunix.rl.rollout import base_rollout
from tunix.tests import test_common as tc


def _clear_test_runtime_state() -> None:
  jax.clear_caches()
  gc.collect()


class VllmSamplerQwenTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls) -> None:
    super().setUpClass()
    cls.repo_id = "Qwen/Qwen3-1.7B"
    temp_dir = tempfile.gettempdir()
    cls.model_path = os.path.join(temp_dir, "models", cls.repo_id)

    tc.download_from_huggingface(repo_id=cls.repo_id, model_path=cls.model_path)

    mesh_shape = (1, len(jax.devices()))
    axis_names = ("fsdp", "tp")
    cls.mesh = jax.make_mesh(
        mesh_shape,
        axis_names,
        devices=jax.devices(),
        axis_types=(jax.sharding.AxisType.Auto,) * len(axis_names),
    )

  def test_qwen3_base_mapping_no_errors(self):
    """Tests that vLLM accepts Tunix's Qwen3 LoRA mappings without logging missing keys."""

    # 1. Create Tunix Actor Model
    config = qwen3_model.ModelConfig.qwen3_1p7b()
    config.num_layers = 1
    base_model = qwen3_params.create_model_from_safe_tensors(
        self.model_path, config, self.mesh
    )

    # 2. Configure Sampler with Explicit Mappings
    tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_path)

    # CRITICAL: Bypass the qwix-wrapped actor_model and inject the dictionary directly
    mapping_config = mappings.MappingConfig(**mapping_vllm_jax.VLLM_JAX_MAPPING)

    vllm_config = vllm_sampler.VllmConfig(
        mesh=self.mesh,
        hbm_utilization=0.3,
        init_with_random_weights=True,
        tpu_backend_type="jax",
        mapping_config=mapping_config,
        server_mode=False,
        engine_kwargs={
            "model": self.model_path,
            "max_model_len": 128,
        },
    )

    sampler = vllm_sampler.VllmSampler(
        tokenizer=tokenizer,
        config=vllm_config,
    )

    # 3. Capture sys.stderr to physically trap the absl C++ logs
    stderr_capture = StringIO()
    original_stderr = sys.stderr
    sys.stderr = stderr_capture

    try:
      # Mock the RPC calls to delete and reinitialize kv cache
      with mock.patch.object(sampler.llm, "reset_prefix_cache"), \
           mock.patch.object(sampler.llm, "collective_rpc"):
        # 4. Trigger param update to force mapping of the base model weights
        sampler.load_checkpoint(nnx.state(base_model))

        # 5. Check the mocked logger to see if it was called with mapping errors
    finally:
      # Always restore stderr so we don't break console output for other tests
      sys.stderr = original_stderr
      if hasattr(sampler, "stop"):
        sampler.stop()
      del sampler
      del base_model
      _clear_test_runtime_state()

    # 5. Parse the captured stderr string
    logs = stderr_capture.getvalue()

    # 6. Strictly Assert
    self.assertNotIn(
        "No mapping for source key",
        logs,
        f"Missing LoRA mappings found in utils.py! Captured Logs:\n{logs}",
    )

class _SingleTurnQuestionEnv(BaseTaskEnv):

  def __init__(
      self,
      *,
      prompt_text: str,
      group_id: str,
      policy_version: int = 1,
  ):
    super().__init__(
        task={"policy_version": policy_version, "prompts": prompt_text},
        max_steps=1,
        group_id=group_id,
    )
    self._prompt_text = prompt_text

  def _initial_observation(self):
    return self._prompt_text

  def _step_impl(self, action) -> EnvStepResult:
    del action
    return EnvStepResult(
        observation="done",
        reward=1.0,
        done=True,
        info={},
    )


class _SingleTurnAgent(ConversationAgentBase):

  def __init__(self):
    super().__init__(system_prompt="")

  def update_from_model(self, response: str, **kwargs) -> agent_types.Action:
    del kwargs
    action = agent_types.Action(action=response)
    step = agent_types.Step(model_response=response, action=action)
    self._trajectory.steps.append(step)
    self._messages.append({"role": "assistant", "content": response})
    return action


class _TestChatParser:

  def parse(self, messages, add_generation_prompt=False, is_first_msg=False):
    del is_first_msg
    rendered = []
    for message in messages:
      rendered.append(f"{message['role']}: {message['content']}")
    if add_generation_prompt:
      rendered.append(self.assistant_token)
    return "\n".join(rendered)

  @property
  def assistant_token(self):
    return "assistant:"

  def update_assistant_end_tokens(self, tokens):
    return np.asarray(tokens, dtype=np.int32), 0


class VllmAgenticTokenFlowTpuTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls) -> None:
    super().setUpClass()
    if not any(device.platform == "tpu" for device in jax.devices()):
      raise unittest.SkipTest("TPU-only integration test.")

    cls.repo_id = "Qwen/Qwen2.5-0.5B-Instruct"
    temp_dir = tempfile.gettempdir()
    cls.model_path = os.path.join(temp_dir, "models", cls.repo_id)
    tc.download_from_huggingface(repo_id=cls.repo_id, model_path=cls.model_path)

    mesh_shape = (len(jax.devices()), 1)
    axis_names = ("fsdp", "tp")
    cls.mesh = jax.make_mesh(
        mesh_shape,
        axis_names,
        devices=jax.devices(),
        axis_types=(jax.sharding.AxisType.Auto,) * len(axis_names),
    )
    cls.tokenizer = tok_adapter.TokenizerAdapter(
      transformers.AutoTokenizer.from_pretrained(cls.model_path)
    )

  def _build_cluster(self) -> rl_cluster_lib.RLCluster:
    actor = qwen2_params.create_model_from_safe_tensors(
        self.model_path,
        qwen2_model.ModelConfig.qwen2p5_0p5b(),
        self.mesh,
    )
    cluster_config = rl_cluster_lib.ClusterConfig(
        role_to_mesh={
            rl_cluster_lib.Role.ACTOR: self.mesh,
            rl_cluster_lib.Role.REFERENCE: self.mesh,
            rl_cluster_lib.Role.ROLLOUT: self.mesh,
        },
        rollout_engine="vllm",
        offload_to_cpu=False,
        training_config=rl_cluster_lib.RLTrainingConfig(
            actor_optimizer=optax.sgd(1e-3),
            eval_every_n_steps=10,
        ),
        rollout_config=base_rollout.RolloutConfig(
            max_prompt_length=128,
            max_tokens_to_generate=32,
            kv_cache_size=256,
            temperature=0.0,
            top_k=1,
            return_logprobs=True,
            tensor_parallel_size=1,
            rollout_vllm_server_mode=True,
            rollout_vllm_model_version=self.model_path,
            rollout_vllm_hbm_utilization=0.4,
            rollout_vllm_init_with_random_weights=True,
            rollout_vllm_tpu_backend_type="jax",
            rollout_vllm_async_scheduling=False,
        ),
    )
    cluster = rl_cluster_lib.RLCluster(
        actor=actor,
        reference=None,
        tokenizer=self.tokenizer,
        cluster_config=cluster_config,
    )
    self.addCleanup(cluster.close)
    self.addCleanup(cluster.rollout._sampler.stop)
    self.addCleanup(_clear_test_runtime_state)
    return cluster

  def test_vllm_generation_tokens_show_up_in_train_example(self):
    rl_cluster = self._build_cluster()
    learner = agentic_grpo_learner.GRPOLearner(
        rl_cluster=rl_cluster,
        reward_fns=lambda prompts, completions, **_: [
            float(index) for index in range(len(completions))
        ],
        algo_config=agentic_grpo_learner.GRPOConfig(
            beta=0.0,
            force_compute_kl=False,
            num_generations=2,
            num_iterations=1,
            max_response_length=32,
            use_rollout_logps=True,
        ),
        chat_parser=_TestChatParser(),
    )

    rollout_config = rl_cluster.get_rollout_config(rl_cluster_lib.Mode.TRAIN)
    prompt_text = "What is the capital of France? Answer with one word."
    recorded_outputs = []

    def collect_token_trajectory(group_id: str):
      env = _SingleTurnQuestionEnv(prompt_text=prompt_text, group_id=group_id)
      agent = _SingleTurnAgent()

      def model_call(chat_completions, runtime_env, max_generation_steps=None):
        del runtime_env
        rendered_prompt = self.tokenizer.apply_chat_template(
            chat_completions,
            tokenize=False,
            add_generation_prompt=True,
        )
        active_rollout_config = rollout_config
        if max_generation_steps is not None:
          active_rollout_config = dataclasses.replace(
              rollout_config,
              max_tokens_to_generate=max_generation_steps,
          )
        output = rl_cluster.rollout.generate(
            prompts=[rendered_prompt],
            rollout_config=active_rollout_config,
        )
        recorded_outputs.append(output)
        return output

      token_data = asyncio.run(
          trajectory_collect_engine.TrajectoryCollectEngine(
              agent=agent,
              env=env,
              model_call=model_call,
              tokenizer=self.tokenizer,
              chat_parser=_TestChatParser(),
              max_response_length=256,
          ).collect(mode="Token")
      )

      output = recorded_outputs[-1]
      self.assertTrue(len(output.tokens[0]) > 0)
      self.assertTrue(output.text[0].strip())
      self.assertEqual(
          output.text[0],
          self.tokenizer.decode(output.tokens[0].tolist()),
      )
      np.testing.assert_array_equal(
          token_data["conversation_tokens"], output.tokens[0]
      )
      np.testing.assert_allclose(token_data["old_logprobs"], output.logprobs[0])
      return types.SimpleNamespace(traj=token_data), output

    traj_item_1, output_1 = collect_token_trajectory(group_id="real-vllm-group")
    traj_item_2, output_2 = collect_token_trajectory(group_id="real-vllm-group")

    with mock.patch.object(
        rl_cluster,
        "get_actor_per_token_logps",
        return_value=jnp.zeros((2, 32), dtype=jnp.float32),
        autospec=True,
    ):
      results = learner._process_results(
          [traj_item_1, traj_item_2],
          expected_step=1,
      )

    self.assertLen(results, 1)
    train_example = results[0]
    self.assertEqual(train_example.completion_ids.shape[0], 2)

    expected_outputs = [output_1, output_2]
    for row_index, output in enumerate(expected_outputs):
      generated_tokens = np.asarray(output.tokens[0], dtype=np.int32)
      generated_logprobs = np.asarray(output.logprobs[0], dtype=np.float32)
      generated_count = len(generated_tokens)

      np.testing.assert_array_equal(
          np.asarray(train_example.prompt_ids[row_index]),
          np.asarray(output.left_padded_prompt_tokens[0], dtype=np.int32),
      )
      np.testing.assert_array_equal(
          np.asarray(train_example.completion_ids[row_index])[:generated_count],
          generated_tokens,
      )
      np.testing.assert_array_equal(
          np.asarray(train_example.completion_mask[row_index])[:generated_count],
          np.ones(generated_count, dtype=np.int32),
      )
      np.testing.assert_allclose(
          np.asarray(train_example.old_per_token_logps[row_index])[:generated_count],
          generated_logprobs,
      )


if __name__ == "__main__":
  absltest.main()
