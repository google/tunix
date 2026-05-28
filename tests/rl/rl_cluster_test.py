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
import os
import unittest
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
from tunix.rl.rollout import mock_rollout
from tunix.tests import test_common as tc

# Some tests relying on SGLang and vLLM cannot run in run_prod environment.
is_run_prod = os.environ.get('GITHUB_JOB') == 'run_prod'

PreTrainedTokenizerBase = tokenization_utils_base.PreTrainedTokenizerBase
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

Mesh = jax.sharding.Mesh


def _dummy_export_fn(*args, **kwargs) -> None:
  """Provides a dummy export function for testing."""
  del args, kwargs


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
          testcase_name='2d_mesh_perf_v1_only',
          reshape_dims=(-1, 1),
          mesh_axes=('fsdp', 'tp'),
          export_fns_by_version={'v1': _dummy_export_fn},
          expected_perf_type=rl_cluster_lib.perf_trace.PerfTracer,
          expected_perf_v2_type=rl_cluster_lib.perf_tracer_v2.NoopTracer,
      ),
      dict(
          testcase_name='3d_mesh_perf_v1_only',
          reshape_dims=(1, -1, 1),
          mesh_axes=('data', 'fsdp', 'tp'),
          export_fns_by_version={'v1': _dummy_export_fn},
          expected_perf_type=rl_cluster_lib.perf_trace.PerfTracer,
          expected_perf_v2_type=rl_cluster_lib.perf_tracer_v2.NoopTracer,
      ),
      dict(
          testcase_name='2d_mesh_perf_v2_only',
          reshape_dims=(-1, 1),
          mesh_axes=('fsdp', 'tp'),
          export_fns_by_version={'v2': _dummy_export_fn},
          expected_perf_type=rl_cluster_lib.perf_trace.NoopTracer,
          expected_perf_v2_type=rl_cluster_lib.perf_tracer_v2.PerfTracer,
      ),
      dict(
          testcase_name='2d_mesh_both_v1_and_v2',
          reshape_dims=(-1, 1),
          mesh_axes=('fsdp', 'tp'),
          export_fns_by_version={
              'v1': _dummy_export_fn,
              'v2': _dummy_export_fn,
          },
          expected_perf_type=rl_cluster_lib.perf_trace.PerfTracer,
          expected_perf_v2_type=rl_cluster_lib.perf_tracer_v2.PerfTracer,
      ),
      dict(
          testcase_name='2d_mesh_no_perf',
          reshape_dims=(-1, 1),
          mesh_axes=('fsdp', 'tp'),
          export_fns_by_version={},
          expected_perf_type=rl_cluster_lib.perf_trace.NoopTracer,
          expected_perf_v2_type=rl_cluster_lib.perf_tracer_v2.NoopTracer,
      ),
  )
  def test_init_with_perf_config(
      self,
      *,
      reshape_dims,
      mesh_axes,
      export_fns_by_version,
      expected_perf_type,
      expected_perf_v2_type,
  ):
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
    perf_config = rl_cluster_lib.perf_metrics.PerfMetricsConfig(
        custom_export_fn=export_fns_by_version.get('v1'),
        custom_export_fn_v2=export_fns_by_version.get('v2'),
    )
    rl_cluster = rl_cluster_lib.RLCluster(
        actor=model,
        tokenizer=vocab,
        cluster_config=cluster_config,
        perf_config=perf_config,
    )
    self.assertIsInstance(rl_cluster.perf, expected_perf_type)
    self.assertIsInstance(rl_cluster.perf_v2, expected_perf_v2_type)

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

  def _create_test_rl_cluster(
      self,
      rollout_engine: str,
      rollout_config: base_rollout.RolloutConfig,
  ) -> rl_cluster_lib.RLCluster:
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
        rollout_engine=rollout_engine,
        offload_to_cpu=False,
        training_config=rl_cluster_lib.RLTrainingConfig(
            actor_optimizer=optax.sgd(1e-3),
            eval_every_n_steps=1,
            max_steps=10,
            gradient_accumulation_steps=None,
        ),
        rollout_config=rollout_config,
    )
    vocab = tc.MockVocab()
    model = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()), rngs=nnx.Rngs(0)
    )
    return rl_cluster_lib.RLCluster(
        actor=model, tokenizer=vocab, cluster_config=cluster_config
    )

  def test_init_cluster_invalid_engine_string(self):
    with self.assertRaisesRegex(
        ValueError, '`cluster_config.rollout_engine` should be one of'
    ):
      self._create_test_rl_cluster(
          'invalid_engine', base_rollout.RolloutConfig()
      )

  @parameterized.parameters('vanilla', 'vllm', 'sglang_jax')
  def test_init_rollout_engine_missing_config_raises_error(self, engine):
    with self.assertRaisesRegex(
        ValueError, '`cluster_config.rollout_config` cannot be None.'
    ):
      self._create_test_rl_cluster(engine, None)

  @parameterized.parameters('vanilla', 'vllm', 'sglang_jax')
  def test_init_rollout_engine_empty_dict_config_raises_error(self, engine):
    with self.assertRaisesRegex(
        ValueError,
        'Rollout config is a dict but missing a train config.',
    ):
      self._create_test_rl_cluster(engine, {})

  @parameterized.named_parameters(
      dict(
          testcase_name='single_config',
          rollout_config=base_rollout.RolloutConfig(
              max_tokens_to_generate=10,
              kv_cache_size=1024,
              data_type=jnp.bfloat16,
          ),
          expected_cache_size=1024,
      ),
      dict(
          testcase_name='dict_config',
          rollout_config={
              rl_cluster_lib.Mode.TRAIN: base_rollout.RolloutConfig(
                  max_tokens_to_generate=10,
                  kv_cache_size=1024,
                  data_type=jnp.bfloat16,
              ),
              rl_cluster_lib.Mode.EVAL: base_rollout.RolloutConfig(
                  max_tokens_to_generate=10,
                  kv_cache_size=2048,
                  data_type=jnp.bfloat16,
              ),
          },
          expected_cache_size=2048,
      ),
  )
  @mock.patch.object(
      rl_cluster_lib.vanilla_rollout, 'VanillaRollout', autospec=True
  )
  def test_init_vanilla_rollout_engine(
      self, mock_vanilla_cls, rollout_config, expected_cache_size
  ):
    rl_cluster = self._create_test_rl_cluster('vanilla', rollout_config)

    mock_vanilla_cls.assert_called_once()
    self.assertEqual(rl_cluster.rollout, mock_vanilla_cls.return_value)
    called_kwargs = mock_vanilla_cls.call_args.kwargs
    self.assertIsInstance(
        called_kwargs['cache_config_or_size'], base_rollout.CacheConfig
    )
    self.assertEqual(
        called_kwargs['cache_config_or_size'].cache_size, expected_cache_size
    )

  def test_init_vanilla_rollout_engine_missing_model_config(self):
    split_index = self.device_count // 2
    actor_mesh = Mesh(
        np.array(jax.devices()[:split_index]).reshape(split_index, 1),
        ('fsdp', 'tp'),
    )
    cluster_config = rl_cluster_lib.ClusterConfig(
        role_to_mesh={
            rl_cluster_lib.Role.ACTOR: actor_mesh,
            rl_cluster_lib.Role.REFERENCE: actor_mesh,
            rl_cluster_lib.Role.ROLLOUT: actor_mesh,
        },
        rollout_engine='vanilla',
        offload_to_cpu=False,
        training_config=rl_cluster_lib.RLTrainingConfig(
            actor_optimizer=optax.sgd(1e-3),
            eval_every_n_steps=1,
        ),
        rollout_config=base_rollout.RolloutConfig(),
    )

    # A dummy model without config
    class DummyModel(nnx.Module):

      def __init__(self):
        self.w = nnx.Param(jnp.zeros((1,)))

    with self.assertRaisesRegex(
        ValueError, '`self.rollout_actor` must have a config attribute.'
    ):
      rl_cluster_lib.RLCluster(
          actor=DummyModel(),
          tokenizer=tc.MockVocab(),
          cluster_config=cluster_config,
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='single_config',
          rollout_config=base_rollout.RolloutConfig(
              max_tokens_to_generate=10, kv_cache_size=1024
          ),
          expected_train_config=base_rollout.RolloutConfig(
              max_tokens_to_generate=10, kv_cache_size=1024
          ),
      ),
      dict(
          testcase_name='dict_config',
          rollout_config={
              rl_cluster_lib.Mode.TRAIN: base_rollout.RolloutConfig(
                  max_tokens_to_generate=10, kv_cache_size=1024
              ),
              rl_cluster_lib.Mode.EVAL: base_rollout.RolloutConfig(
                  max_tokens_to_generate=20, kv_cache_size=2048
              ),
          },
          expected_train_config=base_rollout.RolloutConfig(
              max_tokens_to_generate=10, kv_cache_size=1024
          ),
      ),
  )
  def test_init_mock_rollout_engine(
      self,
      rollout_config,
      expected_train_config,
  ):
    with mock.patch.object(
        mock_rollout.MockRollout, '__init__', autospec=True, return_value=None
    ) as mock_init:
      rl_cluster = self._create_test_rl_cluster(
          mock_rollout.MockRollout, rollout_config
      )

      mock_init.assert_called_once()
      self.assertIsInstance(rl_cluster.rollout, mock_rollout.MockRollout)
      called_kwargs = mock_init.call_args.kwargs
      self.assertEqual(called_kwargs['rollout_config'], expected_train_config)

  @parameterized.named_parameters(
      dict(
          testcase_name='single_config',
          rollout_config=base_rollout.RolloutConfig(
              max_tokens_to_generate=10,
              kv_cache_size=1024,
              rollout_vllm_model_version='dummy_version',
          ),
          expected_train_config=base_rollout.RolloutConfig(
              max_tokens_to_generate=10,
              kv_cache_size=1024,
              rollout_vllm_model_version='dummy_version',
          ),
          expected_cache_size=1024,
      ),
      dict(
          testcase_name='dict_config',
          rollout_config={
              rl_cluster_lib.Mode.TRAIN: base_rollout.RolloutConfig(
                  max_tokens_to_generate=10,
                  kv_cache_size=1024,
                  rollout_vllm_model_version='dummy_version',
              ),
              rl_cluster_lib.Mode.EVAL: base_rollout.RolloutConfig(
                  max_tokens_to_generate=20,
                  kv_cache_size=2048,
                  rollout_vllm_model_version='dummy_version',
              ),
          },
          expected_train_config=base_rollout.RolloutConfig(
              max_tokens_to_generate=10,
              kv_cache_size=1024,
              rollout_vllm_model_version='dummy_version',
          ),
          expected_cache_size=2048,
      ),
  )
  @unittest.skipIf(is_run_prod, 'Skipping in run_prod')
  def test_init_vllm_rollout_engine(
      self,
      rollout_config,
      expected_train_config,
      expected_cache_size,
  ):
    # Internal placeholder for vllm rollout worker stub, don't change this line.
    from tunix.rl.rollout import vllm_rollout

    with mock.patch.object(
        vllm_rollout, 'VllmRollout', autospec=True
    ) as mock_vllm_cls:
      rl_cluster = self._create_test_rl_cluster('vllm', rollout_config)

      mock_vllm_cls.assert_called_once()
      self.assertEqual(rl_cluster.rollout, mock_vllm_cls.return_value)
      called_kwargs = mock_vllm_cls.call_args.kwargs
      self.assertEqual(called_kwargs['rollout_config'], expected_train_config)
      self.assertEqual(
          called_kwargs['cache_config_or_size'], expected_cache_size
      )
      self.assertIn('mesh', called_kwargs)

  @unittest.skipIf(is_run_prod, 'Skipping in run_prod')
  def test_init_vllm_rollout_engine_missing_version_raises(self):
    rollout_config = base_rollout.RolloutConfig(
        rollout_vllm_model_version=None,
    )
    with self.assertRaisesRegex(
        ValueError, 'Rollout vllm model version or path is missing!'
    ):
      self._create_test_rl_cluster('vllm', rollout_config)

  @parameterized.named_parameters(
      dict(
          testcase_name='single_config',
          rollout_config=base_rollout.RolloutConfig(
              max_tokens_to_generate=10, kv_cache_size=1024
          ),
          expected_train_config=base_rollout.RolloutConfig(
              max_tokens_to_generate=10, kv_cache_size=1024
          ),
      ),
      dict(
          testcase_name='dict_config',
          rollout_config={
              rl_cluster_lib.Mode.TRAIN: base_rollout.RolloutConfig(
                  max_tokens_to_generate=10, kv_cache_size=1024
              ),
              rl_cluster_lib.Mode.EVAL: base_rollout.RolloutConfig(
                  max_tokens_to_generate=20, kv_cache_size=2048
              ),
          },
          expected_train_config=base_rollout.RolloutConfig(
              max_tokens_to_generate=10, kv_cache_size=1024
          ),
      ),
  )
  @unittest.skipIf(is_run_prod, 'Skipping in run_prod')
  def test_init_sglang_jax_rollout_engine(
      self, rollout_config, expected_train_config
  ):
    # Internal placeholder for sglang_jax rollout worker stub, don't change this line.
    from tunix.rl.rollout import sglang_jax_rollout

    with mock.patch.object(
        sglang_jax_rollout, 'SglangJaxRollout', autospec=True
    ) as mock_sglang_cls:
      rl_cluster = self._create_test_rl_cluster('sglang_jax', rollout_config)

      mock_sglang_cls.assert_called_once()
      self.assertEqual(rl_cluster.rollout, mock_sglang_cls.return_value)
      called_kwargs = mock_sglang_cls.call_args.kwargs
      self.assertEqual(called_kwargs['rollout_config'], expected_train_config)
      self.assertIn('mesh', called_kwargs)

  @unittest.skipIf(is_run_prod, 'Skipping in run_prod')
  @mock.patch.object(rl_cluster_lib.sft_utils, 'is_lora_enabled', autospec=True)
  def test_init_sglang_jax_rollout_engine_lora_error(self, mock_is_lora):
    mock_is_lora.return_value = True
    rollout_config = base_rollout.RolloutConfig(
        rollout_sglang_jax_enable_static_lora=False
    )

    with self.assertRaisesRegex(
        ValueError, 'Rollout sglang jax lora config is missing'
    ):
      self._create_test_rl_cluster('sglang_jax', rollout_config)

  def test_init_cluster_unsupported_engine_type(self):
    class InvalidEngine:
      pass

    with self.assertRaisesRegex(
        NotImplementedError, 'Rollout engine .* not supported'
    ):
      self._create_test_rl_cluster(InvalidEngine, base_rollout.RolloutConfig())

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

      @property
      def mesh(self):
        return Mesh(
            np.array(jax.devices()[:1]).reshape(1, 1),
            ('fsdp', 'tp'),
        )

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
    self.assertEqual(
        rl_cluster.r2m[rl_cluster_lib.Role.ROLLOUT], rl_cluster.rollout.mesh
    )

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
