# Copyright 2026 The Tunix Authors.
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
from flax import nnx
import jax.numpy as jnp
import numpy as np
from tunix.experimental.generate import engine as engine_lib
from tunix.experimental.generate import kv_cache_manager as kv_cache_manager_lib
from tunix.experimental.generate import model_runner as model_runner_lib
from tunix.experimental.generate import scheduler as scheduler_lib
from tunix.experimental.generate import testing_utils


class LLMEngineTest(absltest.TestCase):

  def test_non_positive_max_model_len_raises(self):
    with self.assertRaisesRegex(ValueError, 'max_model_len'):
      testing_utils.make_engine(max_model_len=0)

  def test_max_pages_per_seq_over_num_device_pages_raises(self):
    # max_model_len=24 + num_scheduler_steps=1 spans 7 pages of 4 slots.
    with self.assertRaisesRegex(ValueError, 'needs up to 7 pages'):
      testing_utils.make_engine(
          max_model_len=24, max_device_size_gib=6 * testing_utils.GIB_PER_PAGE
      )

  def test_mismatched_num_scheduler_steps_raises(self):
    with self.assertRaisesRegex(ValueError, 'num_scheduler_steps'):
      engine_lib.LLMEngine(
          testing_utils.PagedSumTransformer(),
          cache_config=kv_cache_manager_lib.CacheConfig(
              max_device_size_gib=64 * testing_utils.GIB_PER_PAGE,
              page_size=testing_utils.PAGE_SIZE,
              dtype=jnp.float32,
          ),
          scheduler_config=scheduler_lib.SchedulerConfig(
              max_num_batched_tokens=32,
              max_num_seqs=4,
              max_chunked_prefill_length=32,
              num_scheduler_steps=2,
          ),
          model_runner_config=model_runner_lib.ModelRunnerConfig(
              max_top_k=4, mesh=testing_utils.mesh(), num_scheduler_steps=1
          ),
          max_model_len=24,
      )

  def test_exposes_the_model(self):
    transformer = testing_utils.PagedSumTransformer(offset=3.0)
    engine = testing_utils.make_engine(transformer)

    graphdef, state = engine.model_def_and_state()

    self.assertEqual(graphdef, nnx.graphdef(transformer))
    np.testing.assert_array_equal(
        nnx.state(nnx.merge(graphdef, state))['offset'].value, 3.0
    )

  def test_truncates_chunked_prefill_length_to_sliding_window(self):
    transformer = testing_utils.PagedSumTransformer(
        geometries={
            'layer_0': kv_cache_manager_lib.CacheGeometry(
                num_kv_heads=1, head_dim=1, window_size=12
            )
        }
    )
    engine = testing_utils.make_engine(
        transformer, chunked_prefill_length=32
    )

    self.assertEqual(engine._scheduler_config.max_chunked_prefill_length, 8)
    self.assertEqual(engine._scheduler._config.max_chunked_prefill_length, 8)


if __name__ == '__main__':
  absltest.main()
