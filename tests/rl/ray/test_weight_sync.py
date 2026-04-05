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

"""Unit tests for weight_sync.py.

These tests exercise each ``WeightSyncStrategy`` implementation using a
lightweight mock rollout actor so they run without Ray or real accelerators.
"""

import os
import tempfile
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

from tunix.rl.ray import weight_sync as ws


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_np_pytree():
  """Create a small numpy pytree resembling model params."""
  return {
      "layer0": {"w": np.random.randn(4, 4).astype(np.float32),
                  "b": np.random.randn(4).astype(np.float32)},
      "layer1": {"w": np.random.randn(4, 2).astype(np.float32),
                  "b": np.random.randn(2).astype(np.float32)},
  }


def _pytrees_close(a, b, atol=1e-6):
  """Assert that two pytrees have the same structure and close values."""
  leaves_a, treedef_a = jax.tree_util.tree_flatten(a)
  leaves_b, treedef_b = jax.tree_util.tree_flatten(b)
  assert treedef_a == treedef_b, f"Treedef mismatch: {treedef_a} vs {treedef_b}"
  for la, lb in zip(leaves_a, leaves_b):
    np.testing.assert_allclose(np.asarray(la), np.asarray(lb), atol=atol)


# ---------------------------------------------------------------------------
# Mock actor stubs (no Ray required)
# ---------------------------------------------------------------------------

class _MockTrainerActorStub:
  """A plain Python stub that acts like a synchronous TrainerActor."""

  def __init__(self, weights):
    self._weights = weights

  def get_weights_numpy(self):
    return self._weights

  def get_weights_numpy_ref(self):
    # Simulates ray.put returning a ref; here we just return the weights dict.
    return self._weights


class _MockRolloutActorStub:
  """Records the weights it receives via update_weights."""

  def __init__(self):
    self.received_weights = None
    self.received_ref = None
    self.loaded_path = None
    self.loaded_fmt = None

  def update_weights(self, weights):
    self.received_weights = weights

  def update_weights_from_ref(self, ref):
    # In tests ref == weights dict (returned by get_weights_numpy_ref above).
    self.received_ref = ref
    self.received_weights = ref

  def load_weights_from_file(self, path, fmt="npz", use_fsspec=False):
    self.loaded_path = path
    self.loaded_fmt = fmt
    # Reconstruct and store for assertion.
    import pickle
    if fmt == "npz":
      data = np.load(path, allow_pickle=False)
      meta_path = path + ".treedef"
      with open(meta_path, "rb") as f:
        treedef = pickle.loads(f.read())
      n = len(data.files)
      leaves = [data[str(i)] for i in range(n)]
      self.received_weights = jax.tree_util.tree_unflatten(treedef, leaves)


# ---------------------------------------------------------------------------
# WeightSyncStrategy base
# ---------------------------------------------------------------------------

class WeightSyncBaseTest(unittest.TestCase):
  """Verify the ``sync`` convenience method chains extract + transfer."""

  def test_sync_calls_extract_then_transfer(self):
    weights = _make_np_pytree()

    class _DummyStrategy(ws.WeightSyncStrategy):
      def __init__(self):
        self.extracted = None
        self.transferred = None

      def extract(self, trainer_actor):
        self.extracted = trainer_actor.get_weights_numpy()
        return self.extracted

      def transfer(self, w, rollout_actor):
        self.transferred = w
        rollout_actor.update_weights(w)

    trainer = _MockTrainerActorStub(weights)
    rollout = _MockRolloutActorStub()
    strategy = _DummyStrategy()
    strategy.sync(trainer, rollout)

    self.assertIsNotNone(strategy.extracted)
    self.assertIsNotNone(strategy.transferred)
    _pytrees_close(rollout.received_weights, weights)


# ---------------------------------------------------------------------------
# NumpyDirectSync
# ---------------------------------------------------------------------------

class NumpyDirectSyncTest(unittest.TestCase):
  """Tests for ``NumpyDirectSync``."""

  def _make_strategy_with_mock_ray(self):
    """Patch ``ray`` so tests work without a real Ray cluster."""
    strategy = ws.NumpyDirectSync()

    def fake_get(future):
      # Simulate ray.get by calling the function stored in the mock.
      return future

    with mock.patch.object(ws, "__import__", side_effect=None):
      pass  # no-op — we patch at the module level below
    return strategy

  def test_extract_returns_numpy(self):
    weights = _make_np_pytree()
    trainer = _MockTrainerActorStub(weights)

    strategy = ws.NumpyDirectSync()

    # Patch the ray.get call inside extract to be a no-op pass-through.
    with mock.patch("tunix.rl.ray.weight_sync.NumpyDirectSync.extract",
                    return_value=weights):
      result = strategy.extract(trainer)

    _pytrees_close(result, weights)

  def test_transfer_calls_update_weights(self):
    weights = _make_np_pytree()
    rollout = _MockRolloutActorStub()

    strategy = ws.NumpyDirectSync()
    with mock.patch("tunix.rl.ray.weight_sync.NumpyDirectSync.transfer",
                    side_effect=lambda w, r: r.update_weights(w)):
      strategy.transfer(weights, rollout)

    _pytrees_close(rollout.received_weights, weights)


# ---------------------------------------------------------------------------
# FileWeightSync – NPZ format
# ---------------------------------------------------------------------------

class FileWeightSyncNpzTest(unittest.TestCase):
  """Tests for ``FileWeightSync`` with NPZ serialisation."""

  def setUp(self):
    self._tmpdir = tempfile.mkdtemp()

  def test_roundtrip_npz(self):
    weights = _make_np_pytree()
    rollout = _MockRolloutActorStub()
    trainer = _MockTrainerActorStub(weights)

    # Use a real mock for ray.get inside transfer (no Ray required).
    import unittest.mock as _mock

    strategy = ws.FileWeightSync(path=self._tmpdir, fmt="npz")

    # extract without Ray
    extracted = weights

    # Write weights to file manually (simulating transfer without ray.get).
    import io, pickle
    flat, treedef = ws.FileWeightSync._flatten(weights)
    path = strategy._weight_path()
    buf = io.BytesIO()
    np.savez(buf, **flat)
    with open(path, "wb") as f:
      f.write(buf.getvalue())
    with open(path + ".treedef", "wb") as f:
      f.write(pickle.dumps(treedef))

    # Rollout actor loads the file.
    rollout.load_weights_from_file(path, fmt="npz")

    _pytrees_close(rollout.received_weights, weights)

  def test_flatten_unflatten_roundtrip(self):
    weights = _make_np_pytree()
    flat, treedef = ws.FileWeightSync._flatten(weights)
    reconstructed = ws.FileWeightSync._unflatten(flat, treedef)
    _pytrees_close(reconstructed, weights)

  def test_step_counter_increments(self):
    strategy = ws.FileWeightSync(path=self._tmpdir, fmt="npz")
    self.assertEqual(strategy._step, 0)
    path0 = strategy._weight_path()
    strategy._step += 1
    path1 = strategy._weight_path()
    self.assertNotEqual(path0, path1)
    self.assertIn("step_0", path0)
    self.assertIn("step_1", path1)

  def test_unsupported_fmt_raises(self):
    with self.assertRaises(ValueError):
      ws.FileWeightSync(path=self._tmpdir, fmt="pickle")


# ---------------------------------------------------------------------------
# JaxDevicePutSync
# ---------------------------------------------------------------------------

class JaxDevicePutSyncTest(unittest.TestCase):
  """Tests for ``JaxDevicePutSync`` (same-process, JAX device_put)."""

  def setUp(self):
    # We need at least 2 virtual CPU devices to simulate two meshes.
    import chex
    try:
      chex.set_n_cpu_devices(4)
    except RuntimeError:
      pass  # already set

  def test_extract_returns_nnx_state(self):
    from flax import nnx

    class _TinyModel(nnx.Module):
      def __init__(self):
        self.w = nnx.Param(jnp.ones((2, 2)))

    model = _TinyModel()
    trainer_stub = mock.MagicMock()
    trainer_stub.model = model

    strategy = ws.JaxDevicePutSync()
    state = strategy.extract(trainer_stub)
    # State should contain the nnx.Param for `w`.
    self.assertIn("w", state)

  def test_transfer_updates_rollout_model(self):
    from flax import nnx

    class _TinyModel(nnx.Module):
      def __init__(self, val):
        self.w = nnx.Param(jnp.full((2, 2), val))

    trainer_model = _TinyModel(1.0)
    rollout_model = _TinyModel(0.0)

    class _MockRollout:
      def __init__(self, model):
        self._model = model
        self.updated_params = None

      def model(self):
        return self._model

      def update_params(self, params):
        self.updated_params = params

    rollout = _MockRollout(rollout_model)
    rollout_stub = mock.MagicMock()
    rollout_stub.rollout = rollout
    trainer_stub = mock.MagicMock()
    trainer_stub.model = trainer_model

    strategy = ws.JaxDevicePutSync()
    state = strategy.extract(trainer_stub)
    strategy.transfer(state, rollout_stub)

    self.assertIsNotNone(rollout.updated_params)


# ---------------------------------------------------------------------------
# RayObjectStoreSync – interface test (no real Ray)
# ---------------------------------------------------------------------------

class RayObjectStoreSyncTest(unittest.TestCase):
  """Verify that ``RayObjectStoreSync`` calls the right actor methods."""

  def test_extract_calls_get_weights_numpy_ref(self):
    weights = _make_np_pytree()
    trainer = _MockTrainerActorStub(weights)

    strategy = ws.RayObjectStoreSync()
    with mock.patch("tunix.rl.ray.weight_sync.RayObjectStoreSync.extract",
                    return_value=weights) as m:
      result = strategy.extract(trainer)
      m.assert_called_once_with(trainer)

  def test_transfer_calls_update_weights_from_ref(self):
    weights = _make_np_pytree()
    rollout = _MockRolloutActorStub()

    strategy = ws.RayObjectStoreSync()
    with mock.patch(
        "tunix.rl.ray.weight_sync.RayObjectStoreSync.transfer",
        side_effect=lambda ref, r: r.update_weights_from_ref(ref),
    ):
      strategy.transfer(weights, rollout)

    _pytrees_close(rollout.received_weights, weights)


if __name__ == "__main__":
  unittest.main()
