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

from absl.testing import absltest
import chex
from flax import nnx
import jax
import numpy as np
import optax
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl import utils
from tunix.rl.rollout import base_rollout
from tunix.tests import test_common as tc

Mesh = jax.sharding.Mesh


class RlClusterTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.num_cpus = 4
    chex.set_n_cpu_devices(self.num_cpus)
    assert len(jax.devices()) == self.num_cpus

  def test_model_loading_with_resharding(self):
    actor_mesh = Mesh(
        np.array(jax.devices()[: self.num_cpus // 2]).reshape(2, 1),
        ('fsdp', 'tp'),
    )
    rollout_mesh = Mesh(
        np.array(jax.devices()[self.num_cpus // 2 :]).reshape(1, 2),
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
            gradient_accumulation_steps=1,
        ),
        rollout_config=base_rollout.RolloutConfig(
            max_tokens_to_generate=10,
            max_prompt_length=256,
            kv_cache_size=1024,
        ),
    )

    vocab = tc.MockVocab()
    model = tc.ToyTransformer(rngs=nnx.Rngs(0), vocab_size=vocab.GetPieceSize())
    ref_model = tc.ToyTransformer(
        rngs=nnx.Rngs(0), vocab_size=vocab.GetPieceSize()
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
    self.assertEqual(rollout_actor_mesh, rollout_mesh)

    ref_model_mesh = utils.get_pytree_mesh_info(
        nnx.state(rl_cluster.inference_worker._models['reference'])
    )
    self.assertEqual(ref_model_mesh, actor_mesh)


if __name__ == '__main__':
  absltest.main()
