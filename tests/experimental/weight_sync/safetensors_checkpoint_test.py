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

"""Tests for file-backed safetensors weight sync artifacts."""

import tempfile

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from tunix.experimental.weight_sync import safetensors_checkpoint
from tunix.tests import test_common


class SafetensorsCheckpointTest(absltest.TestCase):

  def test_round_trips_matching_state(self):
    model = test_common.ToyTransformer(
        config=test_common.ModelConfig(num_layers=1, num_kv_heads=1, head_dim=2),
        rngs=nnx.Rngs(0),
    )
    state = nnx.to_pure_dict(nnx.state(model))

    updated = jax.tree_util.tree_map(
        lambda x: x + jnp.ones_like(x),
        state,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
      path = safetensors_checkpoint.save_state_to_safetensors(updated, tmpdir)
      restored = safetensors_checkpoint.load_state_from_safetensors(path, state)

    jax.tree.map(np.testing.assert_array_equal, updated, restored)


if __name__ == "__main__":
  absltest.main()
