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

"""Unit tests for `tunix.cli.optimizer_jax`."""

from typing import Any, cast
from unittest import mock

from absl.testing import absltest
import jax.numpy as jnp
import optax
from tunix.cli import optimizer_jax


class OptimizerJaxTest(absltest.TestCase):

  def test_get_schedule_fn_valid_and_invalid(self):
    fn = optimizer_jax.get_schedule_fn("constant_schedule", "cfg.yaml")
    self.assertIs(fn, optax.schedules.constant_schedule)
    self.assertEqual(float(fn(value=0.25)(0)), 0.25)

    with self.assertRaisesRegex(
        AttributeError, "'not_a_schedule' is not a valid function"
    ):
      optimizer_jax.get_schedule_fn("not_a_schedule", "cfg.yaml")

  def test_create_learning_rate_schedule_and_scalar(self):
    sched = optimizer_jax.create_learning_rate(
        {
            "schedule_type": "warmup_cosine_decay_schedule",
            "init_value": 0.0,
            "peak_value": 1e-3,
            "warmup_steps": 10,
            "decay_steps": 100,
            "end_value": 0.0,
        },
        "cfg.yaml",
    )
    self.assertTrue(callable(sched))
    self.assertEqual(float(sched(0)), 0.0)

    self.assertEqual(
        optimizer_jax.create_learning_rate({"learning_rate": 0.05}, "cfg.yaml"),
        0.05,
    )
    self.assertIsNone(optimizer_jax.create_learning_rate({}, "cfg.yaml"))

    with self.assertRaisesRegex(TypeError, "learning_rate must be a scalar"):
      optimizer_jax.create_learning_rate(
          {"learning_rate": "bad_lr"}, "cfg.yaml"
      )

  def test_create_optimizer_and_chaining(self):
    opt = optimizer_jax.create_optimizer(
        {
            "opt_type": "sgd",
            "learning_rate": 0.5,
            "opt_chain_type": "clip_by_global_norm",
            "chain_kwargs": {"max_norm": 1.0},
        },
        "cfg.yaml",
    )
    params = {"w": jnp.zeros((2,))}
    grads = {"w": jnp.array([3.0, 4.0])}
    updates, _ = opt.update(grads, opt.init(params), params)
    self.assertAlmostEqual(float(optax.global_norm(updates)), 0.5, places=4)

  def test_create_optimizer_validation_errors(self):
    with self.assertRaisesRegex(
        ValueError, "optimizer_config must be a dictionary"
    ):
      optimizer_jax.create_optimizer(cast(Any, ["not_a_dict"]), "cfg.yaml")

    with self.assertRaisesRegex(ValueError, "Optimizer name is required"):
      optimizer_jax.create_optimizer({}, "cfg.yaml")

    with self.assertRaisesRegex(
        ValueError, "Optimizer type 'unknown_opt' not supported"
    ):
      optimizer_jax.create_optimizer({"opt_type": "unknown_opt"}, "cfg.yaml")

    with self.assertRaisesRegex(
        ValueError, "Missing required argument 'learning_rate'"
    ):
      optimizer_jax.create_optimizer({"opt_type": "sgd"}, "cfg.yaml")

    with self.assertRaisesRegex(
        ValueError, "Missing required argument 'max_norm'"
    ):
      optimizer_jax.create_optimizer(
          {"opt_type": "clip_by_global_norm"}, "cfg.yaml"
      )

    def _bad_clip(max_norm: float):
      raise TypeError(f"bad max_norm {max_norm}")

    with mock.patch.object(optax, "clip_by_global_norm", _bad_clip):
      with self.assertRaisesRegex(
          TypeError, "Error calling clip_by_global_norm"
      ):
        optimizer_jax.create_optimizer(
            {"opt_type": "clip_by_global_norm", "max_norm": 1.0},
            "cfg.yaml",
        )


if __name__ == "__main__":
  absltest.main()
