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

import types
from typing import Any
from unittest import mock

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from tunix.dist_rl.launch.common import resolve_config
from tunix.dist_rl.launch.nodes import node_utils
from tunix.dist_rl.launch.nodes import trainer as trainer_node
from tunix.dist_rl.weight_sync import ici_handler
from tunix.dist_rl.weight_sync import ici_maxtext_adapter
from tunix.utils import maxtext_utils


class _FakeThrottler:

  def __init__(self):
    self.wait_calls = 0

  def wait_for_all(self) -> None:
    self.wait_calls += 1


class _FakeConverter:
  """Simulates MaxTextToMaxTextConverter: unscans layers and casts to bf16."""

  def __init__(self):
    self.calls: list[tuple[Any, Any]] = []

  def convert(
      self, params_state: Any, *, target_state: Any = None
  ) -> dict[str, Any]:
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


class _FakeMaxTextEngineBase:
  """Engine surface shared by current and legacy MaxText engines."""

  def __init__(
      self,
      *,
      use_weight_converter: bool = True,
      scan_layers: bool = True,
  ):
    self.train_step = 0
    self.model = "fake-model"
    self._throttler = _FakeThrottler()
    self._target_state = {"sentinel": True}
    self._mesh = None
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
    self.fwd_only_fn: Any = None
    self.fwd_only_calls: list[Any] = []
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

  def _get_trainable_params_state(self) -> dict[str, Any]:
    return self._params

  def update(self) -> int:
    self.train_step += 1
    return self.train_step

  def restore_checkpoint(self, *, step: int | None = None) -> dict[str, Any]:
    return {"step": step or 0}

  def close(self) -> None:
    self.closed = True


class _FakeMaxTextEngine(_FakeMaxTextEngineBase):
  """MaxText engine with `with_fwd_only_fn` and `fwd_only(payload)`."""

  def with_fwd_only_fn(self, fn: Any) -> _FakeMaxTextEngine:
    self.fwd_only_fn = fn
    return self

  def fwd_only(self, payload: Any) -> Any:
    self.fwd_only_calls.append(payload)
    return {"logits": payload}


class _LegacyFakeMaxTextEngine(_FakeMaxTextEngineBase):
  """MaxText engine predating `with_fwd_only_fn`; takes `fn` on each call."""

  def fwd_only(self, fn: Any, *args: Any) -> Any:
    self.fwd_only_calls.append((fn, args))
    return fn(self.model, *args)


class IciMaxTextTrainerAdapterTest(absltest.TestCase):

  def test_prepare_converts_binds_reuses_restores_and_releases(self):
    engine = _FakeMaxTextEngine(use_weight_converter=True)
    adapter = ici_maxtext_adapter.IciMaxTextTrainerAdapter(
        engine, job_name="trainer"
    )
    self.assertEqual(adapter.model, "fake-model")

    meta_list = adapter.prepare_weight_sync()
    self.assertLen(meta_list, 1)
    meta = meta_list[0]
    self.assertEqual(meta.transport_mode, ici_handler.TRANSPORT_MODE)
    self.assertEqual(meta.unit.job_name, "trainer")
    self.assertEqual(engine._throttler.wait_calls, 1)
    assert engine._weight_converter is not None
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

    # Restoring a checkpoint invalidates the staged cache even at same step.
    self.assertEqual(adapter.restore_checkpoint(step=0), {"step": 0})
    adapter.prepare_weight_sync()
    self.assertLen(engine._weight_converter.calls, 2)

    # ici_check with plan=None reports this process and world.
    report = adapter.ici_check({"plan": None})
    self.assertEqual(report["process"], jax.process_index())
    self.assertIn(jax.process_index(), report["world"])

    # Advancing step re-converts on the next prepare.
    self.assertEqual(adapter.update(), 1)
    adapter.prepare_weight_sync()
    self.assertLen(engine._weight_converter.calls, 3)

    # Releasing drops bound references so the next prepare re-converts.
    self.assertTrue(adapter.release_weight_sync())
    assert adapter._ici_sync is not None
    self.assertFalse(adapter._ici_sync.bound)
    adapter.prepare_weight_sync()
    self.assertLen(engine._weight_converter.calls, 4)

    adapter.close()
    self.assertTrue(engine.closed)
    self.assertIsNone(adapter._ici_sync)

  def test_use_converter_without_converter_instance_fails_fast(self):
    engine = _FakeMaxTextEngine(use_weight_converter=True)
    engine._weight_converter = None
    adapter = ici_maxtext_adapter.IciMaxTextTrainerAdapter(
        engine, job_name="trainer"
    )
    with self.assertRaisesRegex(RuntimeError, "_weight_converter is None"):
      adapter.prepare_weight_sync()

  def test_fallback_without_weight_converter_casts_and_unscans(self):
    engine = _FakeMaxTextEngine(use_weight_converter=False, scan_layers=True)
    unscan_calls: list[tuple[int, int, int]] = []

    def _fake_unscan(params, *, num_layers, scan_axis, cycle_interval):
      unscan_calls.append((num_layers, scan_axis, cycle_interval))
      scanned = params["decoder"]["layers"]["kernel"]
      return {
          "token_embedder": params["token_embedder"],
          "decoder": {
              f"layers_{i}": {"kernel": scanned[i]} for i in range(num_layers)
          },
      }

    with mock.patch.object(
        ici_maxtext_adapter,
        "_unscan_maxtext_layers",
        side_effect=_fake_unscan,
        autospec=True,
    ):
      adapter = ici_maxtext_adapter.IciMaxTextTrainerAdapter(
          engine, job_name="trainer"
      )
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

  def test_with_fwd_only_fn_and_fwd_only_delegation(self):
    engine = _FakeMaxTextEngine()
    adapter = ici_maxtext_adapter.IciMaxTextTrainerAdapter(
        engine, job_name="trainer"
    )
    fn = lambda m, x: x
    self.assertIs(adapter.with_fwd_only_fn(fn), adapter)
    self.assertIs(engine.fwd_only_fn, fn)
    self.assertEqual(adapter.fwd_only({"x": 1}), {"logits": {"x": 1}})

  def test_legacy_engine_with_fwd_only_fn_skips_registration(self):
    engine = _LegacyFakeMaxTextEngine()
    adapter = ici_maxtext_adapter.IciMaxTextTrainerAdapter(
        engine, job_name="trainer"
    )

    with self.assertLogs(level="WARNING") as logs:
      result = adapter.with_fwd_only_fn(lambda model, x: x)

    self.assertIs(result, adapter)
    self.assertIn("predates with_fwd_only_fn", logs.output[0])
    self.assertIsNone(engine.fwd_only_fn)

  def test_legacy_engine_fwd_only_raises_without_calling_engine(self):
    engine = _LegacyFakeMaxTextEngine()
    adapter = ici_maxtext_adapter.IciMaxTextTrainerAdapter(
        engine, job_name="trainer"
    )
    adapter.with_fwd_only_fn(lambda model, x: x)

    with self.assertRaisesRegex(NotImplementedError, "with_fwd_only_fn"):
      adapter.fwd_only({"x": 1})
    self.assertEmpty(engine.fwd_only_calls)

  def test_engine_without_any_fwd_only_api_fails_at_construction(self):
    with self.assertRaisesRegex(TypeError, "callable fwd_only"):
      ici_maxtext_adapter.IciMaxTextTrainerAdapter(
          _FakeMaxTextEngineBase(), job_name="trainer"
      )

  def test_ici_check_before_prepare_reports_process(self):
    engine = _FakeMaxTextEngine()
    adapter = ici_maxtext_adapter.IciMaxTextTrainerAdapter(
        engine, job_name="trainer"
    )
    report = adapter.ici_check({"plan": None})
    self.assertEqual(report["process"], jax.process_index())

  def test_empty_active_processes_in_multiprocess_world_fails_fast(self):
    engine = _FakeMaxTextEngine()
    with mock.patch.object(jax, "process_count", return_value=2):
      with self.assertRaisesRegex(RuntimeError, "yielded no JAX process"):
        ici_maxtext_adapter.IciMaxTextTrainerAdapter(
            engine, job_name="trainer"
        )

  def test_multihost_prepare_batch_places_array_leaves(self):
    engine: Any = _FakeMaxTextEngine()
    engine._mesh = types.SimpleNamespace(
        devices=np.array([
            types.SimpleNamespace(process_index=0),
            types.SimpleNamespace(process_index=1),
        ])
    )
    placed: list[tuple[tuple[int, ...], Any]] = []

    def _leaf_data_sharding(leaf: Any, spec: Any) -> jax.sharding.Sharding:
      placed.append((leaf.shape, spec))
      return jax.sharding.SingleDeviceSharding(jax.devices()[0])

    engine._prepare_batch = lambda payload: payload
    engine._input_data_spec = lambda: "data-spec"
    engine._leaf_data_sharding = _leaf_data_sharding
    with mock.patch.object(jax, "process_count", return_value=2):
      ici_maxtext_adapter.IciMaxTextTrainerAdapter(
          engine, job_name="trainer"
      )

    batch = engine._prepare_batch(
        {"x": np.zeros((2, 2), dtype=np.float32), "n": 3}
    )

    self.assertEqual(batch["n"], 3)
    self.assertEqual(placed, [((2, 2), "data-spec")])

class MaxTextIciConfigAndWiringTest(absltest.TestCase):

  def _maxtext_ici_hp(self, *extra: str) -> resolve_config.DistHyperParameters:
    return resolve_config.initialize([
        "model_config.model_source=maxtext",
        "model_config.kwargs.maxtext_model_name=qwen3-0.6b",
        "actor_model_config.model_path=gs://bucket/ckpt",
        "rollout_engine=inprocess_vllm",
        "weight_sync_config.mode=ici",
        "cluster_config.fault_tolerance.enabled=false",
        *extra,
    ])

  def test_resolve_config_allows_maxtext_with_ici_inprocess_vllm(self):
    hp = self._maxtext_ici_hp()
    self.assertEqual(hp.config_mapping("weight_sync_config")["mode"], "ici")
    self.assertEqual(
        hp.config_mapping("actor_model_config")["model_source"], "maxtext"
    )

  def test_resolve_config_rejects_maxtext_ici_with_out_of_process_vllm(self):
    with self.assertRaisesRegex(ValueError, "inprocess_vllm"):
      self._maxtext_ici_hp("rollout_engine=vllm")

  def test_maxtext_trainer_factory_wraps_engine_and_scopes_local_devices(self):
    dev_p0 = [
        types.SimpleNamespace(id=0, process_index=0),
        types.SimpleNamespace(id=1, process_index=0),
    ]
    dev_p1 = [
        types.SimpleNamespace(id=2, process_index=1),
        types.SimpleNamespace(id=3, process_index=1),
    ]
    all_devs = dev_p0 + dev_p1

    hp = self._maxtext_ici_hp(
        "actor_model_config.mesh.shape=(2,1)",
        "rollout_model_config.mesh.shape=(1,2)",
        "rl_training_config.train_micro_batch_size=2",
    )
    mock_build_cfg = self.enter_context(
        mock.patch.object(maxtext_utils, "build_maxtext_config", autospec=True)
    )
    mock_create_mesh = self.enter_context(
        mock.patch.object(maxtext_utils, "create_maxtext_mesh", autospec=True)
    )
    self.enter_context(
        mock.patch.object(
            node_utils,
            "load_hf_tokenizer",
            return_value=types.SimpleNamespace(pad_token_id=7, eos_token_id=9),
            autospec=True,
        )
    )
    with (
        mock.patch.object(jax, "process_count", return_value=2),
        mock.patch.object(jax, "process_index", return_value=0),
        mock.patch.object(jax, "devices", return_value=all_devs),
    ):
      factory, _ = trainer_node._create_maxtext_trainer_factory(hp, "trainer-0")

    self.assertEqual(mock_build_cfg.call_args.kwargs["num_devices"], 2)
    mock_create_mesh.assert_called_once_with(
        mock_build_cfg.return_value, devices=dev_p0
    )

    fake_engine = _FakeMaxTextEngine()
    with mock.patch.object(
        maxtext_utils,
        "create_maxtext_engine",
        return_value=fake_engine,
        autospec=True,
    ):
      wrapped = factory()
    self.assertIsInstance(
        wrapped, ici_maxtext_adapter.IciMaxTextTrainerAdapter
    )
    assert isinstance(wrapped, ici_maxtext_adapter.IciMaxTextTrainerAdapter)
    self.assertIs(wrapped._engine, fake_engine)


if __name__ == "__main__":
  absltest.main()
