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

import functools
import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import chex
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
from jax import numpy as jnp
import numpy as np
import optax
from transformers import tokenization_utils_base
from tunix.generate import mappings
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl import utils
from tunix.rl.rollout import base_rollout
from tunix.tests import test_common as tc


PreTrainedTokenizerBase = tokenization_utils_base.PreTrainedTokenizerBase
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

Mesh = jax.sharding.Mesh


class RlClusterTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()

    cls.num_cpus = int(os.environ.get('DEVICE_COUNTS', 4))
    chex.set_n_cpu_devices(cls.num_cpus)
    print(f'Setting up test with {cls.num_cpus} CPU devices before JAX init')
    cls.device_count = jax.device_count()

  def test_model_loading_with_resharding(self):
    split_index = self.device_count // 2

    actor_mesh = Mesh(
        np.array(jax.devices()[:split_index]).reshape(split_index, 1),
        ('fsdp', 'tp'),
    )
    rollout_mesh = Mesh(
        np.array(jax.devices()[split_index:]).reshape(1, split_index),
        ('fsdp', 'tp'),
    )
    cluster_config = rl_cluster_lib.ClusterConfig(
        role_to_mesh={
            rl_cluster_lib.Role.ACTOR: actor_mesh,
            rl_cluster_lib.Role.REFERENCE: actor_mesh,
            rl_cluster_lib.Role.ROLLOUT: rollout_mesh,
        },
        rollout_engine='vanilla',
        offload_to_cpu=False,
        training_config=rl_cluster_lib.RLTrainingConfig(
            actor_optimizer=optax.sgd(1e-3),
            eval_every_n_steps=1,
            max_steps=10,
            gradient_accumulation_steps=None,
        ),
        rollout_config=base_rollout.RolloutConfig(
            max_tokens_to_generate=10,
            max_prompt_length=256,
            kv_cache_size=1024,
            data_type=jnp.bfloat16,
        ),
    )

    vocab = tc.MockVocab()
    model = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()), rngs=nnx.Rngs(0)
    )
    ref_model = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()), rngs=nnx.Rngs(0)
    )

    original_actor_mesh = utils.get_pytree_mesh_info(nnx.state(model))
    self.assertIsNone(original_actor_mesh)

    rl_cluster = rl_cluster_lib.RLCluster(
        actor=model,
        reference=ref_model,
        tokenizer=vocab,
        cluster_config=cluster_config,
    )
    trainer_actor_mesh = utils.get_pytree_mesh_info(
        nnx.state(rl_cluster.actor_trainer.model)
    )
    self.assertEqual(trainer_actor_mesh, actor_mesh)

    rollout_actor_mesh = utils.get_pytree_mesh_info(
        nnx.state(rl_cluster.rollout.model())
    )
    rollout_actor_data_type = jax.tree.leaves(
        nnx.state(rl_cluster.rollout.model())
    )[0].dtype
    self.assertEqual(rollout_actor_mesh, rollout_mesh)
    self.assertEqual(rollout_actor_data_type, jnp.bfloat16)

    actor_data_type = jax.tree.leaves(
        nnx.state(rl_cluster.actor_trainer.model)
    )[0].dtype
    self.assertEqual(actor_data_type, jnp.float32)

    ref_model_mesh = utils.get_pytree_mesh_info(
        nnx.state(rl_cluster.inference_worker._models['reference'])
    )
    self.assertEqual(ref_model_mesh, actor_mesh)

  @parameterized.named_parameters(
      dict(
          testcase_name='2d_mesh',
          reshape_dims=(-1, 1),
          mesh_axes=('fsdp', 'tp'),
      ),
      dict(
          testcase_name='3d_mesh',
          reshape_dims=(1, -1, 1),
          mesh_axes=('data', 'fsdp', 'tp'),
      ),
  )
  def test_init_with_perf_config(self, reshape_dims, mesh_axes):
    mesh = Mesh(np.array(jax.devices()).reshape(*reshape_dims), mesh_axes)
    cluster_config = rl_cluster_lib.ClusterConfig(
        role_to_mesh={
            rl_cluster_lib.Role.ACTOR: mesh,
            rl_cluster_lib.Role.REFERENCE: mesh,
            rl_cluster_lib.Role.ROLLOUT: mesh,
        },
        rollout_engine='vanilla',
        offload_to_cpu=False,
        training_config=rl_cluster_lib.RLTrainingConfig(
            actor_optimizer=optax.sgd(1e-3),
            eval_every_n_steps=1,
            max_steps=10,
            gradient_accumulation_steps=None,
        ),
        rollout_config=base_rollout.RolloutConfig(
            max_tokens_to_generate=10,
            max_prompt_length=256,
            kv_cache_size=1024,
            data_type=jnp.bfloat16,
        ),
    )
    vocab = tc.MockVocab()
    model = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()), rngs=nnx.Rngs(0)
    )
    perf_config = rl_cluster_lib.perf_metrics.PerfMetricsConfig()
    rl_cluster = rl_cluster_lib.RLCluster(
        actor=model,
        tokenizer=vocab,
        cluster_config=cluster_config,
        perf_config=perf_config,
    )
    self.assertIsInstance(rl_cluster.perf, rl_cluster_lib.perf_trace.PerfTracer)

  def test_batch_size_config(self):
    cfg = rl_cluster_lib.RLTrainingConfig(
        actor_optimizer=optax.sgd(1e-3),
        critic_optimizer=None,
        mini_batch_size=8,
        train_micro_batch_size=4,
        eval_every_n_steps=1,
    )
    self.assertEqual(cfg.gradient_accumulation_steps, 2)

    cfg = rl_cluster_lib.RLTrainingConfig(
        actor_optimizer=optax.sgd(1e-3),
        eval_every_n_steps=1,
    )
    self.assertEqual(cfg.gradient_accumulation_steps, None)

    for mini_batch_size, train_micro_batch_size in zip(
        [8, -8, None], [3, 4, 4]
    ):
      with self.assertRaises(ValueError):
        rl_cluster_lib.RLTrainingConfig(
            actor_optimizer=optax.sgd(1e-3),
            critic_optimizer=None,
            mini_batch_size=mini_batch_size,
            train_micro_batch_size=train_micro_batch_size,
            eval_every_n_steps=1,
        )

  def test_generate_with_chat_template(self):  # pylint: disable=g-doc-args
    mesh = Mesh(
        np.array(jax.devices()).reshape(self.device_count, 1), ('fsdp', 'tp')
    )
    cluster_config = rl_cluster_lib.ClusterConfig(
        role_to_mesh={
            rl_cluster_lib.Role.ACTOR: mesh,
            rl_cluster_lib.Role.ROLLOUT: mesh,
        },
        rollout_engine='vanilla',
        offload_to_cpu=False,
        training_config=rl_cluster_lib.RLTrainingConfig(
            actor_optimizer=optax.sgd(1e-3),
            critic_optimizer=None,
            eval_every_n_steps=1,
            max_steps=10,
            mini_batch_size=1,
            rollout_micro_batch_size=1,
        ),
        rollout_config=base_rollout.RolloutConfig(
            max_tokens_to_generate=10,
            max_prompt_length=256,
            kv_cache_size=1024,
        ),
    )

    mock_tokenizer = mock.MagicMock(spec=PreTrainedTokenizerBase)
    mock_tokenizer.apply_chat_template.return_value = 'formatted prompt'
    mock_tokenizer.bos_id = 0
    mock_tokenizer.eos_id = 1
    mock_tokenizer.pad_id = 0

    vocab = tc.MockVocab()
    model = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()), rngs=nnx.Rngs(0)
    )

    rl_cluster = rl_cluster_lib.RLCluster(
        actor=model,
        tokenizer=mock_tokenizer,
        cluster_config=cluster_config,
    )

    expected_text = 'generated text'
    rl_cluster.rollout.generate = mock.MagicMock(
        return_value=base_rollout.RolloutOutput(
            text=[expected_text],
            logits=np.zeros((1, 1, 1)),
            tokens=np.zeros((1, 1)),
            left_padded_prompt_tokens=np.zeros((1, 1)),
            logprobs=None,
        )
    )

    messages = [[{'role': 'user', 'content': 'Hello'}]]
    result = rl_cluster.generate(
        prompts=messages,
        apply_chat_template=True,
        mode=rl_cluster_lib.Mode.EVAL,
    )

    self.assertEqual(result.text[0], expected_text)
    mock_tokenizer.apply_chat_template.assert_called_once_with(
        messages[0],
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    rl_cluster.rollout.generate.assert_called_once()
    called_prompts = rl_cluster.rollout.generate.call_args[0][0]
    self.assertEqual(called_prompts, ['formatted prompt'])

  def test_user_defined_rollout_engine_class(self):
    class CustomRolloutEngine(base_rollout.BaseRollout):

      def __init__(self, my_arg: int = 0, **kwargs):
        self.my_arg = my_arg
        self.config = kwargs['rollout_config']

      def generate(
          self,
          prompts: list[str],
          rollout_config: base_rollout.RolloutConfig,
          **kwargs,
      ) -> base_rollout.RolloutOutput:
        return base_rollout.RolloutOutput(
            text=['generated text'],
            logits=np.zeros((1, 1, 1)),
            tokens=np.zeros((1, 1)),
            left_padded_prompt_tokens=np.zeros((1, 1)),
            logprobs=None,
        )

      def eos_id(self) -> int:
        return 1

      def pad_id(self) -> int:
        return 0

      def get_per_token_logps(
          self,
          prompt_tokens: jax.Array,
          completion_tokens: jax.Array,
          completion_mask: jax.Array | None = None,
      ) -> jax.Array:
        return jax.nn.log_softmax(prompt_tokens)

      def model(self) -> nnx.Module:
        pass

      def update_params(self, params, filter_types):
        pass

    split_index = self.device_count // 2

    actor_mesh = Mesh(
        np.array(jax.devices()[:split_index]).reshape(split_index, 1),
        ('fsdp', 'tp'),
    )
    rollout_mesh = Mesh(
        np.array(jax.devices()[split_index:]).reshape(1, split_index),
        ('fsdp', 'tp'),
    )

    def create_cluster_config(rollout_engine):
      return rl_cluster_lib.ClusterConfig(
          role_to_mesh={
              rl_cluster_lib.Role.ACTOR: actor_mesh,
              rl_cluster_lib.Role.REFERENCE: actor_mesh,
              rl_cluster_lib.Role.ROLLOUT: rollout_mesh,
          },
          rollout_engine=rollout_engine,
          offload_to_cpu=False,
          training_config=rl_cluster_lib.RLTrainingConfig(
              actor_optimizer=optax.sgd(1e-3),
              eval_every_n_steps=1,
              max_steps=10,
              gradient_accumulation_steps=None,
          ),
          rollout_config=base_rollout.RolloutConfig(
              max_tokens_to_generate=10,
              max_prompt_length=256,
              kv_cache_size=1024,
              data_type=jnp.bfloat16,
              rollout_mapping_config=mappings.MappingConfig.build(
                  mapping_obj={
                      'to_hf_mappings': None,
                      'lora_to_hf_mappings': None,
                      'to_hf_hook_fns': None,
                      'to_hf_transpose_keys': None,
                  },
                  model=None,
                  backend=None,
              ),
          ),
      )

    vocab = tc.MockVocab()
    model = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()), rngs=nnx.Rngs(0)
    )
    ref_model = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()), rngs=nnx.Rngs(0)
    )

    original_actor_mesh = utils.get_pytree_mesh_info(nnx.state(model))
    self.assertIsNone(original_actor_mesh)

    # 1. partial type
    MyCustomizedRolloutEngine = functools.partial(CustomRolloutEngine, my_arg=1)  # pylint: disable=invalid-name
    cluster_config = create_cluster_config(MyCustomizedRolloutEngine)

    rl_cluster = rl_cluster_lib.RLCluster(
        actor=model,
        reference=ref_model,
        tokenizer=vocab,
        cluster_config=cluster_config,
    )
    self.assertIsInstance(rl_cluster.rollout, CustomRolloutEngine)
    self.assertEqual(rl_cluster.rollout.my_arg, 1)
    self.assertEqual(rl_cluster.rollout.config, cluster_config.rollout_config)

    # 2. class type
    cluster_config = create_cluster_config(CustomRolloutEngine)
    rl_cluster = rl_cluster_lib.RLCluster(
        actor=model,
        reference=ref_model,
        tokenizer=vocab,
        cluster_config=cluster_config,
    )
    self.assertIsInstance(rl_cluster.rollout, CustomRolloutEngine)
    self.assertEqual(rl_cluster.rollout.my_arg, 0)
    self.assertEqual(rl_cluster.rollout.config, cluster_config.rollout_config)

  @parameterized.named_parameters(
      dict(
          testcase_name='no_rule',
          role_to_logical_axis_rules=None,
          expected_logical_axis_rules=(),
      ),
      dict(
          testcase_name='missing_role',
          role_to_logical_axis_rules={
              rl_cluster_lib.Role.ACTOR: ['fsdp'],
          },
          expected_logical_axis_rules=(),
      ),
      dict(
          testcase_name='with_rule',
          role_to_logical_axis_rules={
              rl_cluster_lib.Role.REFERENCE: ['fsdp'],
          },
          expected_logical_axis_rules=['fsdp'],
      ),
  )
  def test_logical_axis_rules_cm(
      self, role_to_logical_axis_rules, expected_logical_axis_rules
  ):
    mesh = Mesh(np.array(jax.devices()).reshape(1, -1), ('fsdp', 'tp'))
    cluster_config = rl_cluster_lib.ClusterConfig(
        role_to_mesh={
            rl_cluster_lib.Role.ACTOR: mesh,
            rl_cluster_lib.Role.REFERENCE: mesh,
            rl_cluster_lib.Role.ROLLOUT: mesh,
        },
        role_to_logical_axis_rule=role_to_logical_axis_rules,
        rollout_engine='vanilla',
        offload_to_cpu=False,
        training_config=rl_cluster_lib.RLTrainingConfig(
            actor_optimizer=optax.sgd(1e-3),
            eval_every_n_steps=1,
            max_steps=10,
            gradient_accumulation_steps=None,
        ),
        rollout_config=base_rollout.RolloutConfig(
            max_tokens_to_generate=10,
            max_prompt_length=256,
            kv_cache_size=1024,
            data_type=jnp.bfloat16,
        ),
    )
    vocab = tc.MockVocab()
    model = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()), rngs=nnx.Rngs(0)
    )
    ref_model = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()), rngs=nnx.Rngs(0)
    )
    rl_cluster = rl_cluster_lib.RLCluster(
        actor=model,
        reference=ref_model,
        tokenizer=vocab,
        cluster_config=cluster_config,
    )

    invoked = False

    def mock_fn(*args, **kwargs):  # pylint: disable=unused-argument
      nonlocal invoked
      invoked = True
      self.assertEqual(
          nn_partitioning.get_axis_rules(), expected_logical_axis_rules
      )
      return jnp.zeros((1, 1))

    self.assertEqual(nn_partitioning.get_axis_rules(), ())

    old_fn = rl_cluster.inference_worker.get_ref_per_token_logps
    try:
      rl_cluster.inference_worker.get_ref_per_token_logps = mock_fn
      rl_cluster.get_ref_per_token_logps(
          prompt_tokens=jnp.zeros((1, 1)),
          completion_tokens=jnp.zeros((1, 1)),
          pad_id=0,
          eos_id=1,
          micro_batch_size=1,
      )
    finally:
      rl_cluster.inference_worker.get_ref_per_token_logps = old_fn

    self.assertTrue(invoked)
    self.assertEqual(nn_partitioning.get_axis_rules(), ())


if __name__ == '__main__':
  absltest.main()
