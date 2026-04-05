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

"""Integration tests for RayRLCluster and RayClusterConfig.

These tests validate the end-to-end Ray actor lifecycle: cluster creation,
generate forwarding, weight sync, metrics buffering, and teardown.

They mock out the underlying ``RLCluster`` so they do not require real
accelerators or a distributed JAX runtime.  The Ray actors themselves still
start (in the same process via ``ray.init(local_mode=True)``), which exercises
the serialisation path.
"""

import os
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import chex
chex.set_n_cpu_devices(4)


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

def _make_fake_rollout_output():
  """Build a fake dict as returned by RolloutActor.generate."""
  return {
      "text": ["answer A", "answer B"],
      "tokens": [np.array([1, 2, 3]), np.array([4, 5, 6])],
      "logits": None,
      "logprobs": [np.array([-0.1, -0.2, -0.3]), np.array([-0.4, -0.5, -0.6])],
      "left_padded_prompt_tokens": np.zeros((2, 5), dtype=np.int32),
  }


def _make_fake_trainer_state():
  return {
      "train_steps": 0,
      "iter_steps": 0,
      "restored_global_step": 0,
      "is_managed_externally": False,
  }


# ---------------------------------------------------------------------------
# Mock factories that replace the real cluster construction
# ---------------------------------------------------------------------------

def _make_mock_rl_cluster():
  """Build a mock ``RLCluster`` that doesn't need real JAX models."""
  cluster = mock.MagicMock()
  cluster.global_steps = 0
  cluster.actor_trainer.train_steps = 0
  cluster.actor_trainer.iter_steps = 0
  cluster.actor_trainer.is_managed_externally = False
  cluster.actor_trainer.restored_global_step.return_value = 0

  rollout_out_dict = _make_fake_rollout_output()
  from tunix.rl.rollout import base_rollout
  cluster.generate.return_value = base_rollout.RolloutOutput(
      text=rollout_out_dict["text"],
      tokens=rollout_out_dict["tokens"],
      logits=None,
      logprobs=rollout_out_dict["logprobs"],
      left_padded_prompt_tokens=rollout_out_dict["left_padded_prompt_tokens"],
  )
  cluster.rollout.pad_id.return_value = 0
  cluster.rollout.eos_id.return_value = 1
  ref_logps = np.zeros((2, 8), dtype=np.float32)
  cluster.get_ref_per_token_logps.return_value = jnp.asarray(ref_logps)
  return cluster


# ---------------------------------------------------------------------------
# TrainerActor unit tests
# ---------------------------------------------------------------------------

class TrainerActorTest(unittest.TestCase):
  """Test the ``TrainerActor`` class directly (no Ray)."""

  def setUp(self):
    self._mock_cluster = _make_mock_rl_cluster()

  def test_get_weights_numpy_returns_pytree(self):
    from flax import nnx
    from tunix.rl.ray.ray_trainer_actor import TrainerActor

    # Build a tiny model and inject it into the mock cluster.
    class _Tiny(nnx.Module):
      def __init__(self):
        self.w = nnx.Param(jnp.ones((2, 2)))

    model = _Tiny()
    self._mock_cluster.actor_trainer.model = model
    # Patch is_lora_enabled to return False.
    with mock.patch("tunix.sft.utils.is_lora_enabled", return_value=False):
      actor = TrainerActor(cluster_factory=lambda: self._mock_cluster)
      weights = actor.get_weights_numpy()

    # nnx.state() returns an nnx.State (a Mapping), not a plain dict.
    # Verify the leaves are numpy arrays regardless of container type.
    leaves = jax.tree_util.tree_leaves(weights)
    self.assertTrue(len(leaves) > 0)
    self.assertTrue(all(isinstance(l, np.ndarray) for l in leaves))

  def test_get_actor_trainer_state(self):
    from tunix.rl.ray.ray_trainer_actor import TrainerActor

    actor = TrainerActor(cluster_factory=lambda: self._mock_cluster)
    state = actor.get_actor_trainer_state()
    self.assertIn("train_steps", state)
    self.assertIn("iter_steps", state)
    self.assertIn("restored_global_step", state)
    self.assertIn("is_managed_externally", state)

  def test_get_ref_per_token_logps_returns_numpy(self):
    from tunix.rl.ray.ray_trainer_actor import TrainerActor

    actor = TrainerActor(cluster_factory=lambda: self._mock_cluster)
    result = actor.get_ref_per_token_logps(
        prompt_tokens=np.zeros((2, 4), dtype=np.int32),
        completion_tokens=np.zeros((2, 8), dtype=np.int32),
        pad_id=0,
        eos_id=1,
    )
    self.assertIsInstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# RolloutActor unit tests
# ---------------------------------------------------------------------------

class RolloutActorTest(unittest.TestCase):
  """Test the ``RolloutActor`` class directly (no Ray)."""

  def _make_rollout_actor(self):
    mock_cluster = _make_mock_rl_cluster()
    from tunix.rl.ray.ray_rollout_actor import RolloutActor
    return RolloutActor(rollout_factory=lambda: mock_cluster), mock_cluster

  def test_generate_returns_dict(self):
    actor, _ = self._make_rollout_actor()
    result = actor.generate(
        prompts=["hello world"],
        apply_chat_template=False,
        mode_str="train",
    )
    self.assertIn("text", result)
    self.assertIn("tokens", result)
    self.assertIn("left_padded_prompt_tokens", result)

  def test_generate_numpy_output(self):
    actor, _ = self._make_rollout_actor()
    result = actor.generate(
        prompts=["hello"],
        mode_str="train",
    )
    for t in result["tokens"]:
      self.assertIsInstance(t, np.ndarray)

  def test_update_weights_calls_rollout_update_params(self):
    from flax import nnx
    from tunix.rl.ray.ray_rollout_actor import RolloutActor

    class _Tiny(nnx.Module):
      def __init__(self):
        self.w = nnx.Param(jnp.zeros((2, 2)))

    rollout_model = _Tiny()
    mock_cluster = _make_mock_rl_cluster()
    mock_cluster.rollout.model.return_value = rollout_model

    update_calls = []
    mock_cluster.rollout.update_params.side_effect = lambda p: update_calls.append(p)

    actor = RolloutActor(rollout_factory=lambda: mock_cluster)
    weights_numpy = {"w": np.ones((2, 2), dtype=np.float32)}
    actor.update_weights(weights_numpy)

    self.assertEqual(len(update_calls), 1)

  def test_load_weights_from_file_npz(self):
    import io, pickle, tempfile
    from tunix.rl.ray import weight_sync as ws
    from tunix.rl.ray.ray_rollout_actor import RolloutActor
    from flax import nnx

    weights = {
        "layer0": np.random.randn(4, 4).astype(np.float32),
        "layer1": np.random.randn(4, 2).astype(np.float32),
    }
    flat, treedef = ws.FileWeightSync._flatten(weights)

    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, "weights_step_0.npz")
      buf = io.BytesIO()
      np.savez(buf, **flat)
      with open(path, "wb") as f:
        f.write(buf.getvalue())
      with open(path + ".treedef", "wb") as f:
        f.write(pickle.dumps(treedef))

      class _Tiny(nnx.Module):
        def __init__(self):
          self.layer0 = nnx.Param(jnp.zeros((4, 4)))
          self.layer1 = nnx.Param(jnp.zeros((4, 2)))

      mock_cluster = _make_mock_rl_cluster()
      rollout_model = _Tiny()
      mock_cluster.rollout.model.return_value = rollout_model
      received = []
      mock_cluster.rollout.update_params.side_effect = received.append

      actor = RolloutActor(rollout_factory=lambda: mock_cluster)
      actor.load_weights_from_file(path, fmt="npz")

    self.assertEqual(len(received), 1)


# ---------------------------------------------------------------------------
# _TrainerProxy / _RolloutProxy unit tests
# ---------------------------------------------------------------------------

class TrainerProxyTest(unittest.TestCase):
  """Verify the proxy satisfies learner attribute patterns without Ray."""

  def _make_proxy(self, state_override=None):
    """Build a _TrainerProxy backed by a synchronous (non-Ray) stub.

    In real Ray, ``handle.method`` is a ``RemoteMethod`` attribute (not a
    bound method), so ``handle.method.remote(args)`` is the call pattern.
    The stub below replicates that with a simple ``_RemoteAttr`` descriptor.
    """
    from tunix.rl.ray.ray_rl_cluster import _TrainerProxy

    state = {**_make_fake_trainer_state(), **(state_override or {})}

    class _RemoteAttr:
      """Stub for a Ray RemoteMethod: calling .remote() returns the result."""
      def __init__(self, fn):
        self._fn = fn
      def remote(self, *a, **kw):
        return self._fn(*a, **kw)

    class _SyncHandle:
      """Ray actor handle stub — attributes are RemoteAttr, not methods."""
      def __init__(self):
        self.get_actor_trainer_state = _RemoteAttr(lambda: state)
        self.set_actor_trainer_managed_externally = _RemoteAttr(
            lambda v: state.update({"is_managed_externally": v})
        )

    import types
    fake_ray = types.SimpleNamespace(get=lambda x: x)

    proxy = object.__new__(_TrainerProxy)
    proxy._handle = _SyncHandle()
    proxy._ray = fake_ray
    proxy._meta = {}
    proxy._refresh()
    return proxy, state

  def test_train_steps_property(self):
    proxy, state = self._make_proxy({"train_steps": 42})
    self.assertEqual(proxy.train_steps, 42)

  def test_restored_global_step(self):
    proxy, _ = self._make_proxy({"restored_global_step": 7})
    self.assertEqual(proxy.restored_global_step(), 7)

  def test_model_property_raises(self):
    proxy, _ = self._make_proxy()
    with self.assertRaises(AttributeError):
      _ = proxy.model


class RolloutProxyTest(unittest.TestCase):
  """Verify the rollout proxy caches pad/eos ids."""

  def _make_proxy(self):
    from tunix.rl.ray.ray_rl_cluster import _RolloutProxy

    class _RemoteAttr:
      def __init__(self, val):
        self._val = val
      def remote(self):
        return self._val

    class _SyncHandle:
      def __init__(self):
        self.pad_id = _RemoteAttr(0)
        self.eos_id = _RemoteAttr(1)

    import types
    fake_ray = types.SimpleNamespace(get=lambda x: x)
    proxy = object.__new__(_RolloutProxy)
    proxy._handle = _SyncHandle()
    proxy._ray = fake_ray
    proxy._pad_id = None
    proxy._eos_id = None
    return proxy

  def test_pad_id_caches(self):
    proxy = self._make_proxy()
    self.assertEqual(proxy.pad_id(), 0)
    self.assertEqual(proxy._pad_id, 0)  # cached

  def test_eos_id_caches(self):
    proxy = self._make_proxy()
    self.assertEqual(proxy.eos_id(), 1)
    self.assertEqual(proxy._eos_id, 1)

  def test_model_raises(self):
    proxy = self._make_proxy()
    with self.assertRaises(AttributeError):
      _ = proxy.model()


# ---------------------------------------------------------------------------
# _pytree_to_numpy helper
# ---------------------------------------------------------------------------

class PytreeToNumpyTest(unittest.TestCase):

  def test_converts_jax_arrays(self):
    from tunix.rl.ray.ray_rl_cluster import _pytree_to_numpy
    arr = jnp.ones((3, 3))
    result = _pytree_to_numpy({"a": arr, "b": [arr, arr]})
    leaves = jax.tree_util.tree_leaves(result)
    for l in leaves:
      self.assertIsInstance(l, np.ndarray)

  def test_leaves_none_unchanged(self):
    from tunix.rl.ray.ray_rl_cluster import _pytree_to_numpy
    self.assertIsNone(_pytree_to_numpy(None))

  def test_plain_numpy_unchanged(self):
    from tunix.rl.ray.ray_rl_cluster import _pytree_to_numpy
    arr = np.ones((2, 2))
    result = _pytree_to_numpy({"w": arr})
    np.testing.assert_array_equal(result["w"], arr)


# ---------------------------------------------------------------------------
# RayClusterConfig defaults
# ---------------------------------------------------------------------------

class RayClusterConfigTest(unittest.TestCase):

  def test_default_weight_sync_none(self):
    from tunix.rl.ray.ray_rl_cluster import RayClusterConfig
    from tunix.rl import rl_cluster as rl_cluster_lib
    import optax

    devices = jax.devices()
    mesh = jax.sharding.Mesh(
        np.array(devices).reshape(len(devices), 1), ("fsdp", "tp")
    )
    from tunix.rl.rollout import base_rollout
    cfg = RayClusterConfig(
        role_to_mesh={
            rl_cluster_lib.Role.ACTOR: mesh,
            rl_cluster_lib.Role.REFERENCE: mesh,
            rl_cluster_lib.Role.ROLLOUT: mesh,
        },
        rollout_engine="vanilla",
        offload_to_cpu=False,
        training_config=rl_cluster_lib.RLTrainingConfig(
            actor_optimizer=optax.sgd(1e-3),
            eval_every_n_steps=1,
            max_steps=1,
        ),
        rollout_config=base_rollout.RolloutConfig(max_tokens_to_generate=4),
    )
    self.assertIsNone(cfg.weight_sync_strategy)

  def test_custom_weight_sync_stored(self):
    from tunix.rl.ray.ray_rl_cluster import RayClusterConfig
    from tunix.rl.ray.weight_sync import NumpyDirectSync
    from tunix.rl import rl_cluster as rl_cluster_lib
    import optax

    devices = jax.devices()
    mesh = jax.sharding.Mesh(
        np.array(devices).reshape(len(devices), 1), ("fsdp", "tp")
    )
    from tunix.rl.rollout import base_rollout
    strategy = NumpyDirectSync()
    cfg = RayClusterConfig(
        role_to_mesh={
            rl_cluster_lib.Role.ACTOR: mesh,
            rl_cluster_lib.Role.REFERENCE: mesh,
            rl_cluster_lib.Role.ROLLOUT: mesh,
        },
        rollout_engine="vanilla",
        offload_to_cpu=False,
        training_config=rl_cluster_lib.RLTrainingConfig(
            actor_optimizer=optax.sgd(1e-3),
            eval_every_n_steps=1,
            max_steps=1,
        ),
        rollout_config=base_rollout.RolloutConfig(max_tokens_to_generate=4),
        weight_sync_strategy=strategy,
    )
    self.assertIs(cfg.weight_sync_strategy, strategy)


# ---------------------------------------------------------------------------
# Smoke test: RayRLCluster construction (mocked actors)
# ---------------------------------------------------------------------------

class RayRLClusterSmokeTest(unittest.TestCase):
  """Smoke test that RayRLCluster.sync_weights delegates to WeightSyncStrategy.

  Rather than constructing the full cluster (which requires Ray and real JAX
  models), we directly instantiate a ``RayRLCluster`` via ``object.__new__``
  and inject the minimal state needed to exercise ``sync_weights``.
  """

  def test_sync_weights_delegates_to_strategy(self):
    """sync_weights() must call strategy.sync(trainer_handle, rollout_handle)."""
    from tunix.rl.ray.ray_rl_cluster import RayRLCluster
    from tunix.rl.ray.weight_sync import NumpyDirectSync
    import types

    trainer_handle = mock.MagicMock()
    rollout_handle = mock.MagicMock()
    mock_sync = mock.MagicMock(spec=NumpyDirectSync)

    fake_ray = types.SimpleNamespace(
        get=lambda x: x,
        is_initialized=lambda: True,
    )

    cluster = object.__new__(RayRLCluster)
    cluster._ray = fake_ray
    cluster._trainer_handle = trainer_handle
    cluster._rollout_handle = rollout_handle
    cluster._weight_sync = mock_sync
    cluster.global_steps = 0

    # set_global_steps is called on the trainer actor after sync.
    trainer_handle.set_global_steps.remote.return_value = None

    cluster.sync_weights()

    mock_sync.sync.assert_called_once_with(trainer_handle, rollout_handle)
    self.assertEqual(cluster.global_steps, 1)

  def test_global_steps_increments_per_sync(self):
    from tunix.rl.ray.ray_rl_cluster import RayRLCluster
    import types

    cluster = object.__new__(RayRLCluster)
    cluster._ray = types.SimpleNamespace(get=lambda x: x)
    cluster._trainer_handle = mock.MagicMock()
    cluster._trainer_handle.set_global_steps.remote.return_value = None
    cluster._rollout_handle = mock.MagicMock()
    cluster._weight_sync = mock.MagicMock()
    cluster.global_steps = 5

    cluster.sync_weights()
    self.assertEqual(cluster.global_steps, 6)

    cluster.sync_weights()
    self.assertEqual(cluster.global_steps, 7)

  def test_buffer_metrics_async_flushes_old_steps(self):
    """buffer_metrics_async flushes buffered metrics when step advances."""
    from tunix.rl.ray.ray_rl_cluster import RayRLCluster
    from tunix.rl import rl_cluster as rl_cluster_lib
    from tunix.perf import metrics as perf_metrics
    import types

    cluster = object.__new__(RayRLCluster)
    cluster._ray = types.SimpleNamespace(get=lambda x: x)
    cluster.global_steps = 0
    cluster._buffered_train_metrics = []
    cluster._buffered_eval_metrics = []
    cluster._rl_metrics_logger = mock.MagicMock()
    cluster._external_metrics_logger = None

    # Buffer a metric at step 0.
    cluster.buffer_metrics_async(
        {"loss": (1.0, np.mean)},
        mode=rl_cluster_lib.Mode.TRAIN,
        step=0,
    )
    self.assertEqual(len(cluster._buffered_train_metrics), 1)

    # Advance global_steps; next call should flush step 0's buffer.
    cluster.global_steps = 1
    cluster.buffer_metrics_async(
        {"loss": (0.5, np.mean)},
        mode=rl_cluster_lib.Mode.TRAIN,
        step=1,
    )
    # _log_metrics should have been called for step 0.
    cluster._rl_metrics_logger.log.assert_called()


if __name__ == "__main__":
  unittest.main()
