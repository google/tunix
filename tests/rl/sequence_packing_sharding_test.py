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

"""Sharding correctness tests for sequence packing under FSDP/SP/TP.

Verifies that the packing-aware loss path (`algo_core.grpo_loss_fn` with
`segment_ids`) runs correctly on four canonical mesh shapes:

| Mesh                              | Note                |
|-----------------------------------|---------------------|
| `(fsdp=2,)`                       | FSDP-only baseline  |
| `(fsdp=2, sp=2)`                  | Sequence parallel    |
| `(fsdp=2, tp=2)`                  | Tensor parallel      |
| `(fsdp=2, sp=2, tp=2)`            | Full mesh            |

8 fake CPU devices (forced via `XLA_FLAGS`). Each test builds a small toy
model on the mesh, packs a tiny batch, runs `grpo_loss_fn` forward + reverse,
and asserts the loss and grads are finite.

A separate `ApplySpShardingHelperTest` directly exercises
`tunix.rl.utils.apply_sp_sharding` to verify the no-op / constrain branches.
"""

import os
# Force 8 fake CPU devices BEFORE any JAX import (the env var is read by XLA
# at backend init time).
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
from jax.interpreters import pxla
import jax.numpy as jnp
import jax.sharding as shd
import numpy as np

# Importing `algo_core` triggers loss-fn registration; we call its
# `grpo_loss_fn` directly below.
from tunix.rl import algo_core
from tunix.rl import common
from tunix.rl import function_registry as fr
from tunix.rl import utils as rl_utils
from tunix.tests import test_common as tc


_VOCAB_SIZE = 16


def _create_sharded_model(rngs, mesh):
  """Place a small `ToyTransformer` on `mesh`. Returns the resharded model.

  Borrowed from `tests/distillation/distillation_trainer_test.py` —
  initializes the model under a `with_sharding_constraint(state, pspecs)`
  so all parameter arrays carry the partitioning declared on the layer.
  """

  @nnx.jit
  def _make():
    model = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=_VOCAB_SIZE, num_layers=1),
        rngs=rngs,
    )
    state = nnx.state(model)
    pspecs = nnx.get_partition_spec(state)
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(model, sharded_state)
    return model

  with mesh:
    return _make()


def _make_packed_example(batch_size: int = 1):
  """Build a tiny packed `TrainExample` for sharding-correctness tests.

  Two segments per row, prompt prefix mask=0. Shapes are kept small (T=8)
  because we're verifying sharding compatibility, not full convergence —
  fp32 finiteness is sufficient.
  """
  rng = np.random.default_rng(0)
  t = 8
  tokens = rng.integers(low=1, high=_VOCAB_SIZE, size=(batch_size, t))
  # mask 0 at positions 0 and 4 (start of each segment); 1 elsewhere.
  mask = np.tile([0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0], (batch_size, 1))
  seg_ids = np.tile([1, 1, 1, 1, 2, 2, 2, 2], (batch_size, 1)).astype(np.int32)
  positions = np.tile([0, 1, 2, 3, 0, 1, 2, 3], (batch_size, 1)).astype(np.int32)
  advantages = rng.normal(size=(batch_size, t)).astype(np.float32)
  ref_logps = rng.normal(size=(batch_size, t)).astype(np.float32)

  return common.TrainExample(
      prompt_ids=jnp.zeros((batch_size, 0), dtype=jnp.int32),
      prompt_mask=jnp.zeros((batch_size, 0), dtype=jnp.int32),
      completion_ids=jnp.asarray(tokens, dtype=jnp.int32),
      completion_mask=jnp.asarray(mask, dtype=jnp.float32),
      advantages=jnp.asarray(advantages),
      ref_per_token_logps=jnp.asarray(ref_logps),
      old_per_token_logps=None,
      segment_ids=jnp.asarray(seg_ids),
      segment_positions=jnp.asarray(positions),
      num_segments=3,
  )


class _StubAlgoConfig:
  """Minimal stand-in for `GRPOConfig` used inside `grpo_loss_fn`."""

  loss_agg_mode = "token-mean"
  loss_algo = "grpo"
  beta = 0.05
  epsilon = 0.2
  epsilon_high = 0.2
  kl_loss_mode = "low_var_kl"
  policy_loss_fn = "grpo"
  temperature = 1.0


def _loss_value(model, train_example):
  """Run `grpo_loss_fn` forward and return `primary_loss.compute()`."""
  loss_fn = fr.default_registry.get("policy_loss_fn", "grpo")
  out = loss_fn(model, train_example, _StubAlgoConfig(), pad_id=0, eos_id=0)
  return out.primary_loss.compute()


def _build_mesh(shape: dict[str, int]) -> shd.Mesh:
  """Build a Mesh from a name->axis-size dict over the 8 fake CPU devices."""
  axis_names = tuple(shape.keys())
  axis_sizes = tuple(shape.values())
  total = 1
  for s in axis_sizes:
    total *= s
  assert total <= jax.device_count(), (
      f"Mesh shape {shape} requires {total} devices but only "
      f"{jax.device_count()} are available."
  )
  device_grid = np.asarray(jax.devices()[:total]).reshape(axis_sizes)
  return shd.Mesh(device_grid, axis_names=axis_names)


class ApplySpShardingHelperTest(absltest.TestCase):
  """Direct unit test for the SP-sharding helper.

  Covers all three branches:
  - No active mesh (single-device CPU) -> no-op.
  - Active mesh without `sp` axis -> no-op.
  - Active mesh with `sp` axis -> applies `with_sharding_constraint`.
  """

  def test_no_active_mesh_is_noop(self):
    x = jnp.ones((8,), dtype=jnp.float32)
    out = rl_utils.apply_sp_sharding(x, shd.PartitionSpec("sp"))
    # The helper just returns the same array; we cannot assert object identity
    # with JAX arrays generally, so use array_equal on the values.
    np.testing.assert_array_equal(np.asarray(out), np.asarray(x))

  def test_no_sp_axis_is_noop(self):
    mesh = _build_mesh(dict(fsdp=2, tp=2))
    x = jnp.ones((4, 8), dtype=jnp.float32)
    with mesh:
      out = rl_utils.apply_sp_sharding(
          x, shd.PartitionSpec("fsdp", "sp")
      )
    np.testing.assert_array_equal(np.asarray(out), np.asarray(x))

  def test_sp_axis_applies_constraint(self):
    mesh = _build_mesh(dict(fsdp=2, sp=4))
    x = jnp.ones((2, 8), dtype=jnp.float32)
    with mesh:
      # Wrap in a jit so XLA actually realizes the constraint; outside jit
      # `with_sharding_constraint` returns a `Sharded` array we can inspect.
      @jax.jit
      def f(y):
        return rl_utils.apply_sp_sharding(
            y, shd.PartitionSpec("fsdp", "sp")
        )

      out = f(x)
    self.assertIsInstance(out.sharding, shd.NamedSharding)
    self.assertEqual(out.sharding.spec, shd.PartitionSpec("fsdp", "sp"))


class SequencePackingShardingTest(parameterized.TestCase):
  """End-to-end packed-loss-runs-on-mesh checks across FSDP/SP/TP.

  For each shape we run forward + reverse through `grpo_loss_fn` under the
  mesh context and assert the loss + grads are finite. We do not compare
  against an "unsharded baseline" — JAX's sharding propagation makes the
  arithmetic equivalent up to reduction-order tolerance, but the
  bit-identity isn't the property of interest here. What is the property of
  interest: the packed loss path doesn't deadlock, doesn't OOM in
  partitioning, and produces finite gradients.
  """

  @parameterized.named_parameters(
      dict(
          testcase_name="fsdp_only",
          mesh_shape=dict(fsdp=2),
          mesh_id="(fsdp=2,)",
      ),
      dict(
          testcase_name="fsdp_and_sp",
          mesh_shape=dict(fsdp=2, sp=2),
          mesh_id="(fsdp=2, sp=2)",
      ),
      dict(
          testcase_name="fsdp_and_tp",
          mesh_shape=dict(fsdp=2, tp=2),
          mesh_id="(fsdp=2, tp=2)",
      ),
      dict(
          testcase_name="full_mesh_fsdp_sp_tp",
          mesh_shape=dict(fsdp=2, sp=2, tp=2),
          mesh_id="(fsdp=2, sp=2, tp=2)",
      ),
  )
  def test_packed_grpo_loss_on_mesh(self, mesh_shape, mesh_id):
    mesh = _build_mesh(mesh_shape)
    rngs = nnx.Rngs(0)

    with mesh:
      model = _create_sharded_model(rngs, mesh)

    # batch_size = fsdp axis size so the batch dimension is shardable.
    train_example = _make_packed_example(batch_size=mesh_shape["fsdp"])

    @nnx.jit
    def loss_and_grad(m, ex):
      def fn(m_):
        return _loss_value(m_, ex)
      return nnx.value_and_grad(fn)(m)

    with mesh:
      loss_val, grads = loss_and_grad(model, train_example)

    self.assertTrue(
        jnp.isfinite(loss_val),
        msg=f"loss not finite on mesh {mesh_id}: got {loss_val}",
    )
    # Verify every leaf of `grads` is finite. Use `jax.tree.map` for a
    # PyTree-walk that respects nested NNX states.
    def _check_finite(path, leaf):
      if isinstance(leaf, jax.Array):
        self.assertTrue(
            jnp.all(jnp.isfinite(leaf)),
            msg=f"non-finite grad at {path} on mesh {mesh_id}",
        )

    jax.tree.map_with_path(_check_finite, grads)


if __name__ == "__main__":
  absltest.main()
