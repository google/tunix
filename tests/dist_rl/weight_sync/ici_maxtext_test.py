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

"""Unit tests for MaxText trainer integration with ICI weight sync."""

from __future__ import annotations

import os
import sys
import types
from unittest import mock

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from tunix.dist_rl.launch.common import resolve_config
from tunix.dist_rl.launch.nodes import node_utils
from tunix.dist_rl.launch.nodes import rollout as rollout_node
from tunix.dist_rl.launch.nodes import trainer as trainer_node
from tunix.dist_rl.weight_sync import ici_handler
from tunix.dist_rl.weight_sync import ici_weight_sync_delegate
from tunix.utils import maxtext_utils


class _FakeThrottler:

  def __init__(self):
    self.wait_calls = 0

  def wait_for_all(self):
    self.wait_calls += 1


class _FakeConverter:
  """Simulates MaxTextToMaxTextConverter: unscans layers and casts to bf16."""

  def __init__(self):
    self.calls = []

  def convert(self, params_state, *, target_state=None):
    self.calls.append((params_state, target_state))
    scanned = params_state["decoder"]["layers"]["kernel"]
    return {
        "token_embedder": {
            "embedding": params_state["token_embedder"]["embedding"].astype(
                jnp.bfloat16
            )
        },
        "decoder": {
            f"layers_{i}": {"kernel": scanned[i].astype(jnp.bfloat16)}
            for i in range(scanned.shape[0])
        },
    }


class _FakeMaxTextEngine:

  def __init__(self, *, use_weight_converter: bool = True, scan_layers: bool = True):
    self.train_step = 0
    self.model = "fake-model"
    self._throttler = _FakeThrottler()
    self._target_state = {"sentinel": True}
    self._use_weight_converter = use_weight_converter
    self._weight_converter = _FakeConverter() if use_weight_converter else None
    self._config = types.SimpleNamespace(
        rollout_weight_dtype="bfloat16",
        scan_layers=scan_layers,
        num_decoder_layers=2,
        param_scan_axis=0,
        inhomogeneous_layer_cycle_interval=1,
    )
    self.closed = False
    self._params = {
        "token_embedder": {
            "embedding": jnp.ones((4, 4), dtype=jnp.float32),
        },
        "decoder": {
            "layers": {
                "kernel": jnp.arange(16, dtype=jnp.float32).reshape(2, 2, 4),
            }
        },
    }

  def _get_trainable_params_state(self):
    return self._params

  def update(self):
    self.train_step += 1
    return self.train_step

  def close(self):
    self.closed = True


class IciMaxTextTrainerAdapterTest(absltest.TestCase):

  def test_prepare_converts_binds_reuses_and_releases(self):
    engine = _FakeMaxTextEngine(use_weight_converter=True)
    adapter = ici_weight_sync_delegate.IciMaxTextTrainerAdapter(
        engine, job_name="trainer"
    )
    # Delegates non-ICI attributes transparently.
    self.assertEqual(adapter.model, "fake-model")

    meta_list = adapter.prepare_weight_sync()
    self.assertLen(meta_list, 1)
    meta = meta_list[0]
    self.assertEqual(meta.transport_mode, ici_handler.TRANSPORT_MODE)
    self.assertEqual(meta.unit.job_name, "trainer")
    self.assertEqual(engine._throttler.wait_calls, 1)
    self.assertLen(engine._weight_converter.calls, 1)
    self.assertEqual(engine._weight_converter.calls[0][1], {"sentinel": True})

    by_name = {v.name: v for v in meta.variables}
    self.assertCountEqual(
        by_name.keys(),
        [
            "decoder.layers.0.kernel",
            "decoder.layers.1.kernel",
            "token_embedder.embedding",
        ],
    )
    for var in by_name.values():
      self.assertEqual(var.dtype, "bfloat16")
    self.assertEqual(by_name["decoder.layers.0.kernel"].shape, (2, 4))

    # Same train_step reuses staged metadata without re-converting.
    meta_reused = adapter.prepare_weight_sync()
    self.assertEqual(meta_reused, meta_list)
    self.assertLen(engine._weight_converter.calls, 1)

    # ici_check with plan=None reports this process and world.
    report = adapter.ici_check({"plan": None})
    self.assertEqual(report["process"], jax.process_index())
    self.assertIn(jax.process_index(), report["world"])

    # Advancing step re-converts on the next prepare.
    self.assertEqual(adapter.update(), 1)
    adapter.prepare_weight_sync()
    self.assertLen(engine._weight_converter.calls, 2)

    # Releasing drops bound references so the next prepare re-converts.
    self.assertTrue(adapter.release_weight_sync())
    self.assertFalse(adapter._ici_sync.bound)
    adapter.prepare_weight_sync()
    self.assertLen(engine._weight_converter.calls, 3)

    adapter.close()
    self.assertTrue(engine.closed)
    self.assertIsNone(adapter._ici_sync)

  def test_fallback_without_weight_converter_casts_and_unscans(self):
    engine = _FakeMaxTextEngine(use_weight_converter=False, scan_layers=True)
    fake_unscan = types.ModuleType("raiden_unscan")
    unscan_calls = []

    def _unscan_layers(params, *, num_layers, scan_axis, cycle_interval):
      unscan_calls.append((num_layers, scan_axis, cycle_interval))
      scanned = params["decoder"]["layers"]["kernel"]
      return {
          "token_embedder": params["token_embedder"],
          "decoder": {
              f"layers_{i}": {"kernel": scanned[i]} for i in range(num_layers)
          },
      }

    fake_unscan.unscan_layers = _unscan_layers
    fake_pkg = types.ModuleType("maxtext.integration.tunix.weight_mapping")
    fake_pkg.raiden_unscan = fake_unscan
    modules = {
        "maxtext": types.ModuleType("maxtext"),
        "maxtext.integration": types.ModuleType("maxtext.integration"),
        "maxtext.integration.tunix": types.ModuleType(
            "maxtext.integration.tunix"
        ),
        "maxtext.integration.tunix.weight_mapping": fake_pkg,
        "maxtext.integration.tunix.weight_mapping.raiden_unscan": fake_unscan,
    }
    with mock.patch.dict(sys.modules, modules):
      adapter = ici_weight_sync_delegate.IciMaxTextTrainerAdapter(engine)
      meta_list = adapter.prepare_weight_sync()

    self.assertEqual(unscan_calls, [(2, 0, 1)])
    by_name = {v.name: v for v in meta_list[0].variables}
    self.assertCountEqual(
        by_name.keys(),
        [
            "decoder.layers.0.kernel",
            "decoder.layers.1.kernel",
            "token_embedder.embedding",
        ],
    )
    self.assertEqual(by_name["decoder.layers.0.kernel"].dtype, "bfloat16")

  def test_ici_check_before_prepare_reports_process(self):
    engine = _FakeMaxTextEngine()
    adapter = ici_weight_sync_delegate.IciMaxTextTrainerAdapter(engine)
    report = adapter.ici_check({"plan": None})
    self.assertEqual(report["process"], jax.process_index())


class MaxTextIciConfigAndWiringTest(absltest.TestCase):

  def _maxtext_ici_hp(self, *extra: str):
    return resolve_config.initialize([
        "model_config.model_source=maxtext",
        "model_config.kwargs.maxtext_model_name=qwen3-0.6b",
        "actor_model_config.model_path=gs://bucket/ckpt",
        "rollout_engine=inprocess_vllm",
        "weight_sync_config.mode=ici",
        *extra,
    ])

  def test_resolve_config_allows_maxtext_with_ici_inprocess_vllm(self):
    hp = self._maxtext_ici_hp("weight_sync_config.ici_drop_old_weights=true")
    self.assertEqual(
        hp.config_mapping("weight_sync_config")["mode"], "ici"
    )
    self.assertEqual(
        hp.config_mapping("actor_model_config")["model_source"], "maxtext"
    )

  def test_resolve_config_still_rejects_maxtext_ici_with_out_of_process_vllm(
      self,
  ):
    with self.assertRaisesRegex(ValueError, "inprocess_vllm"):
      self._maxtext_ici_hp("rollout_engine=vllm")

  def test_create_maxtext_mesh_forwards_devices_subset(self):
    fake_m_utils = mock.Mock()
    dev0, dev1 = mock.Mock(name="d0"), mock.Mock(name="d1")
    fake_m_utils.create_device_mesh.return_value = np.array([[dev0, dev1]])
    cfg = types.SimpleNamespace(mesh_axes=("fsdp", "tp"))

    with (
        mock.patch.object(
            maxtext_utils,
            "maxtext_modules",
            return_value=(None, None, fake_m_utils),
        ),
        mock.patch("jax.sharding.Mesh") as mock_mesh,
    ):
      maxtext_utils.create_maxtext_mesh(cfg)
      fake_m_utils.create_device_mesh.assert_called_once_with(cfg)
      fake_m_utils.create_device_mesh.reset_mock()

      maxtext_utils.create_maxtext_mesh(cfg, devices=[dev0, dev1])
      fake_m_utils.create_device_mesh.assert_called_once_with(
          cfg, devices=[dev0, dev1]
      )
      self.assertEqual(mock_mesh.call_count, 2)

  @mock.patch.object(
      node_utils,
      "load_hf_tokenizer",
      return_value=mock.Mock(pad_token_id=7, eos_token_id=9),
  )
  @mock.patch.object(maxtext_utils, "create_maxtext_mesh")
  @mock.patch.object(maxtext_utils, "build_maxtext_config", autospec=True)
  def test_maxtext_trainer_factory_wraps_engine_and_scopes_devices_for_ici(
      self, mock_build_cfg, mock_create_mesh, _
  ):
    dev_p0 = [mock.Mock(id=0, process_index=0), mock.Mock(id=1, process_index=0)]
    dev_p1 = [mock.Mock(id=2, process_index=1), mock.Mock(id=3, process_index=1)]
    dev_p2 = [mock.Mock(id=4, process_index=2), mock.Mock(id=5, process_index=2)]
    all_devs = dev_p0 + dev_p1 + dev_p2

    hp = self._maxtext_ici_hp(
        "actor_model_config.mesh.shape=(4,1)",
        "rollout_model_config.mesh.shape=(1,2)",
        "rl_training_config.train_micro_batch_size=4",
    )
    with (
        mock.patch.object(jax, "process_count", return_value=3),
        mock.patch.object(jax, "process_index", return_value=0),
        mock.patch.object(jax, "devices", return_value=all_devs),
        mock.patch.dict(
            os.environ, {"ICI_TRAINER_PROCESS_INDICES": "0,1"}, clear=False
        ),
    ):
      factory, _ = trainer_node._create_maxtext_trainer_factory(hp, "trainer-0")

    self.assertEqual(mock_build_cfg.call_args.kwargs["num_devices"], 4)
    mock_create_mesh.assert_called_once_with(
        mock_build_cfg.return_value, devices=dev_p0 + dev_p1
    )

    fake_engine = _FakeMaxTextEngine()
    with mock.patch.object(
        maxtext_utils, "create_maxtext_engine", return_value=fake_engine
    ):
      wrapped = factory()
    self.assertIsInstance(
        wrapped, ici_weight_sync_delegate.IciMaxTextTrainerAdapter
    )
    self.assertIs(wrapped._engine, fake_engine)

  def test_rollout_main_initializes_jax_runtime_for_inprocess_vllm_under_ici(
      self,
  ):
    hp = self._maxtext_ici_hp(
        "cluster_config.registry_module=tunix.dist_rl.examples.math_gsm8k_dist.gsm8k"
    )
    fake_ctx = mock.Mock()
    fake_ctx.ipc.discovery = mock.Mock()

    async def _noop_serve(*args, **kwargs):
      del args, kwargs

    with (
        mock.patch.dict(os.environ, {}, clear=False),
        mock.patch.object(resolve_config, "initialize", return_value=hp),
        mock.patch.object(node_utils, "apply_library_env"),
        mock.patch.object(node_utils, "setup_otel_metrics"),
        mock.patch.object(node_utils, "ensure_repo_root_on_path"),
        mock.patch.object(
            node_utils, "log_resolved_config", return_value="sha"
        ),
        mock.patch.object(node_utils, "load_hf_tokenizer", return_value=mock.Mock()),
        mock.patch.object(node_utils, "serve_worker", side_effect=_noop_serve),
        mock.patch.object(rollout_node.importlib, "import_module"),
        mock.patch.object(rollout_node.gcs_cache, "restore_jax_cache"),
        mock.patch.object(
            rollout_node, "_create_rollout_worker", return_value=mock.Mock(trajectory_store=None)
        ),
    ):
      rollout_node.main([], context=fake_ctx)

    fake_ctx.initialize_runtime.assert_called_once_with(
        hp.cluster_targets.rollout
    )


if __name__ == "__main__":
  absltest.main()
