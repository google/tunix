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

"""Integration tests for file-backed fallback weight sync."""

import asyncio
import os
import tempfile
from unittest import mock

from absl.testing import absltest
import flax
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from tunix.experimental.rollout import inprocess_vllm_sampler_adapter
from tunix.experimental.rollout import sampler as base_sampler_lib
from tunix.experimental.train import peft_trainer_v2
from tunix.experimental.weight_sync import safetensors_checkpoint
from tunix.experimental.weight_sync import weight_sync
from tunix.generate import base_sampler
from tunix.generate import utils as gen_utils
from tunix.rl import reshard
from tunix.tests import test_common

TrainingConfig = peft_trainer_v2.TrainingConfig
PeftTrainer = peft_trainer_v2.PeftTrainer


class MockVllmSampler(base_sampler.BaseSampler):
	"""Mock vLLM sampler exposing VllmToyTransformer state and file loading."""

	def __init__(self, config: test_common.ModelConfig, *, rngs: nnx.Rngs):
		self._transformer = test_common.VllmToyTransformer(config, rngs=rngs)
		self._transformer_state = nnx.state(self._transformer)
		self.mesh = None

	@property
	def transformer(self) -> nnx.Module:
		return self._transformer

	@property
	def transformer_state(self):
		return self._transformer_state

	@transformer_state.setter
	def transformer_state(self, state):
		self._transformer_state = state

	def update_params(self, params, filter_types=None):
		del filter_types
		self._transformer_state = gen_utils.transfer_state_with_mappings(
				src_state=params,
				dst_state=self._transformer_state,
				key_mappings=test_common.TOY_TRANSFORMER_TO_HF_MAPPINGS,
				reshard_fn=reshard.reshard_pytree,
				rollout_engine="vllm_jax",
		)

	def load_checkpoint(self, path_or_weights):
		if isinstance(path_or_weights, str):
			restored = safetensors_checkpoint.load_state_from_safetensors(
					path_or_weights,
					self._transformer_state,
			)
			self._transformer_state = restored
			return
		self.update_params(path_or_weights)

	def __call__(self, *args, **kwargs):
		return base_sampler.SamplerOutput(
				text=["mock completion"],
				logits=None,
				tokens=np.array([1, 2, 3], dtype=np.int32),
				padded_prompt_tokens=np.array([[1]], dtype=np.int32),
				logprobs=None,
		)

	def tokenize(self, input_string: str) -> np.ndarray:
		del input_string
		return np.array([1, 2], dtype=np.int32)

	def get_target_state(self):
		return jax.tree.map(
				lambda x: nnx.Param(jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype)),
				self._transformer_state,
				is_leaf=lambda x: isinstance(x, nnx.Variable),
		)


def _to_flat_dict(state):
	if isinstance(state, nnx.State):
		state = nnx.to_pure_dict(state)
	if isinstance(state, dict):
		flat = flax.traverse_util.flatten_dict(state)
		return {".".join(str(p) for p in path): value for path, value in flat.items()}
	return dict(state)


class FallbackSafetensorsWeightSyncTest(absltest.TestCase):

	def setUp(self):
		super().setUp()
		self.model_config = test_common.ModelConfig(num_layers=2, vocab_size=128)
		self.toy_trainer_model = test_common.ToyTransformer(
				self.model_config, rngs=nnx.Rngs(0)
		)
		self.mock_vllm_sampler = MockVllmSampler(
				self.model_config, rngs=nnx.Rngs(1)
		)

	def test_file_backed_fallback_sync_with_vllm_adapter(self):
		rollout_config = mock.Mock(
				sampler_type="inprocess_vllm",
				weight_sync_mode=weight_sync.WeightSyncMode.FALLBACK,
		)
		adapter = inprocess_vllm_sampler_adapter.InprocessVllmSamplerAdapter(
				server_id="vllm_sampler_fallback_0",
				config=rollout_config,
		)
		adapter.vllm_sampler = self.mock_vllm_sampler

		trainer = PeftTrainer(
				model=self.toy_trainer_model,
				optimizer=optax.sgd(1e-3),
				training_config=TrainingConfig(
						eval_every_n_steps=2,
						max_steps=10,
						checkpoint_root_directory=tempfile.mkdtemp(),
				),
				sampler_type="inprocess_vllm",
		)
		trainer.set_target_state(adapter.get_target_state())

		new_embedding = (
				jnp.ones_like(self.toy_trainer_model.emb.embedding.value) * 17.0
		)
		self.toy_trainer_model.emb.embedding.value = new_embedding

		with mock.patch.dict(os.environ, {"WEIGHT_SYNC_MODE": "fallback"}):
			prepared = trainer.prepare_weight_sync(
					sync_request=base_sampler_lib.WeightSyncRequest(policy_version=7)
			)

		self.assertLen(prepared, 1)
		self.assertTrue(
				prepared[0].artifact_path.endswith("policy_7/model.safetensors")
		)

		result = asyncio.run(
				adapter.weight_sync(
						base_sampler_lib.WeightSyncRequest(
								policy_version=7,
								source_metadata=tuple(prepared),
						)
				)
		)
		self.assertTrue(result)

		tgt_flat = _to_flat_dict(self.mock_vllm_sampler.transformer_state)
		np.testing.assert_allclose(
			np.array(tgt_flat["model.embed_tokens.embedding"]),
				np.array(new_embedding),
		)


if __name__ == "__main__":
	absltest.main()
