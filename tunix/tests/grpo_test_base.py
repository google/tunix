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

"""Shared test helpers for GRPO-family learner tests.

Centralizes the toy-model RLCluster setup, dummy datasets, and reward
functions used by GRPOLearner, RLOOLearner, DrGRPOLearner, etc. Tests should
import from here rather than duplicating these helpers.
"""

from typing import Any, Dict, Optional

from flax import nnx
from grain import python as grain
import jax
from jax.interpreters import pxla
import jax.numpy as jnp
import optax
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.rollout import base_rollout
from tunix.tests import test_common as tc

# Token strings defined in MockVocab in test_common.py.
DUMMY_DATA = [
    'input string',
    'hello world',
    'My name',
    'hello there',
]


def reward_1(completions, **kargs):  # pylint: disable=unused-argument
  """Reward function returning a per-completion rank."""
  return jnp.arange(len(completions))


def reward_2(prompts, answer, **kargs):  # pylint: disable=unused-argument
  """Reward function that depends on the optional ``answer`` field."""
  return jnp.arange(len(answer))


class MySource(grain.RandomAccessDataSource):
  """A simple grain data source backed by an in-memory list."""

  def __init__(self, data=None, repeat=1):
    if data is None:
      data = DUMMY_DATA
    self._data = data * repeat

  def __getitem__(self, idx):
    return self._data[idx]

  def __len__(self):
    return len(self._data)


def dummy_dataset(source: Optional[grain.RandomAccessDataSource] = None,
                  batch_size: int = 1):
  """Builds a batched grain dataset yielding ``{'prompts', 'answer'}`` dicts."""
  if source is None:
    source = MySource()
  return (
      grain.MapDataset.source(source)
      .batch(batch_size)
      .map(lambda x: {'prompts': x, 'answer': x})
  )


def setup(kwargs: Optional[Dict[str, Any]] = None):
  """Builds a toy RLCluster with matching actor/reference models.

  Returns:
    A tuple ``(rl_cluster, model, original_variables)`` where
    ``original_variables`` is a snapshot of the actor parameters before any
    training, useful for asserting that training updates the parameters.
  """
  if kwargs is None:
    kwargs = {}
  vocab = tc.MockVocab()
  model = tc.ToyTransformer(
      config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()),
      rngs=nnx.Rngs(0),
  )
  original_variables = jax.tree.map(jnp.copy, nnx.state(model, nnx.Param))
  ref_model = tc.ToyTransformer(
      config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()), rngs=nnx.Rngs(0)
  )

  mesh = pxla.thread_resources.env.physical_mesh
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
          eval_every_n_steps=kwargs.get('eval_every_n_steps', 2),
          max_steps=10,
          gradient_accumulation_steps=kwargs.get(
              'gradient_accumulation_steps', None
          ),
          max_seq_token_per_tpu=kwargs.get('max_seq_token_per_tpu', None),
      ),
      rollout_config=base_rollout.RolloutConfig(
          max_tokens_to_generate=10,
          max_prompt_length=256,
          kv_cache_size=1024,
      ),
  )
  rl_cluster = rl_cluster_lib.RLCluster(
      actor=model,
      reference=ref_model,
      tokenizer=vocab,
      cluster_config=cluster_config,
  )
  return rl_cluster, model, original_variables
