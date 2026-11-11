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

"""Unit tests for v2 generation engine configuration dataclasses."""

import math
import os

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tunix.experimental.generate import configs

# Sharding tests need up to 4 devices.
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"


def _create_mesh(
    shape: tuple[int, ...] = (2, 2),
    axis_names: tuple[str, ...] = ("dp", "tp"),
) -> jax.sharding.Mesh:
  devices = jax.devices()[: math.prod(shape)]
  return jax.sharding.Mesh(np.array(devices).reshape(shape), axis_names)


class CacheConfigTest(parameterized.TestCase):

  def test_valid_config(self):
    cfg = configs.CacheConfig(
        max_device_size_gib=1.0,
        max_host_size_gib=0.5,
        page_size=16,
        dtype=jnp.float32,
    )
    self.assertEqual(cfg.page_size, 16)
    self.assertEqual(cfg.dtype, jnp.float32)

  def test_invalid_page_size_raises(self):
    with self.assertRaisesRegex(ValueError, r"page_size must be positive"):
      configs.CacheConfig(
          page_size=0,
          max_device_size_gib=1.0,
          dtype=jnp.float32,
      )

  def test_invalid_negative_size_raises(self):
    with self.assertRaisesRegex(
        ValueError, r"max_device_size_gib must be positive"
    ):
      configs.CacheConfig(
          page_size=16,
          max_device_size_gib=-1.0,
          dtype=jnp.float32,
      )
    with self.assertRaisesRegex(
        ValueError, r"max_host_size_gib cannot be negative"
    ):
      configs.CacheConfig(
          page_size=16,
          max_device_size_gib=1.0,
          max_host_size_gib=-1.0,
          dtype=jnp.float32,
      )

  def test_zero_device_size_raises(self):
    with self.assertRaisesRegex(
        ValueError, r"max_device_size_gib must be positive"
    ):
      configs.CacheConfig(
          page_size=16,
          max_device_size_gib=0.0,
          dtype=jnp.float32,
      )


class SchedulerConfigTest(parameterized.TestCase):

  def test_valid_config(self):
    config = configs.SchedulerConfig(
        max_num_batched_tokens=128,
        max_num_seqs=8,
        max_chunked_prefill_length=16,
        num_scheduler_steps=1,
    )
    self.assertEqual(config.max_num_batched_tokens, 128)
    self.assertEqual(config.max_num_seqs, 8)
    self.assertEqual(config.max_chunked_prefill_length, 16)
    self.assertEqual(config.num_scheduler_steps, 1)

  @parameterized.named_parameters(
      dict(
          testcase_name="zero_max_num_batched_tokens",
          field="max_num_batched_tokens",
          value=0,
      ),
      dict(
          testcase_name="negative_max_num_batched_tokens",
          field="max_num_batched_tokens",
          value=-1,
      ),
      dict(testcase_name="zero_max_num_seqs", field="max_num_seqs", value=0),
      dict(
          testcase_name="negative_max_num_seqs",
          field="max_num_seqs",
          value=-1,
      ),
      dict(
          testcase_name="zero_max_chunked_prefill_length",
          field="max_chunked_prefill_length",
          value=0,
      ),
      dict(
          testcase_name="negative_max_chunked_prefill_length",
          field="max_chunked_prefill_length",
          value=-2,
      ),
      dict(
          testcase_name="zero_num_scheduler_steps",
          field="num_scheduler_steps",
          value=0,
      ),
      dict(
          testcase_name="negative_num_scheduler_steps",
          field="num_scheduler_steps",
          value=-1,
      ),
  )
  def test_non_positive_field_raises(self, field, value):
    kwargs = dict(
        max_num_batched_tokens=128,
        max_num_seqs=8,
        max_chunked_prefill_length=16,
        num_scheduler_steps=1,
    )
    kwargs[field] = value
    with self.assertRaisesRegex(ValueError, rf"{field} must be positive"):
      configs.SchedulerConfig(**kwargs)

  def test_max_chunked_prefill_length_exceeds_max_batch_tokens_raises(self):
    with self.assertRaisesRegex(
        ValueError, "must be less than or equal to max_num_batched_tokens"
    ):
      configs.SchedulerConfig(
          max_num_batched_tokens=16,
          max_num_seqs=8,
          max_chunked_prefill_length=32,
      )


class ModelRunnerConfigTest(absltest.TestCase):

  def test_non_positive_max_top_k_sets_no_limit(self):
    for max_top_k in (0, -1):
      config = configs.ModelRunnerConfig(
          max_top_k=max_top_k, mesh=_create_mesh()
      )
      self.assertEqual(config.max_top_k, max_top_k)

  def test_non_positive_num_scheduler_steps_raises(self):
    with self.assertRaisesRegex(ValueError, "num_scheduler_steps"):
      configs.ModelRunnerConfig(
          max_top_k=1, mesh=_create_mesh(), num_scheduler_steps=0
      )


if __name__ == "__main__":
  absltest.main()
