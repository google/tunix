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

"""Unit tests for the helpers shared by the distributed RL node mains."""

import asyncio
from collections.abc import Awaitable, Callable
import pathlib
import pickle
import signal
import sys
import tempfile
from unittest import mock

from absl.testing import absltest
import jax
import transformers
from tunix.dist_rl.launch.common import resolve_config
from tunix.dist_rl.launch.nodes import node_utils
from tunix.dist_rl.worker import remote_execution


class RequireDiscoveryContextTest(absltest.TestCase):

  def test_returns_context_with_discovery(self):
    context = mock.MagicMock()
    self.assertIs(node_utils.require_discovery_context(context), context)

  def test_raises_without_context_or_discovery(self):
    for context in (None, mock.MagicMock(ipc=None)):
      with self.subTest(context=context):
        with self.assertRaisesRegex(RuntimeError, "Require discovery API"):
          node_utils.require_discovery_context(context)


class EnsureRepoRootOnPathTest(absltest.TestCase):

  def test_inserts_repo_root_once(self):
    repo_root = str(resolve_config.REPO_ROOT)
    saved_path = list(sys.path)
    try:
      sys.path[:] = [p for p in sys.path if p != repo_root]
      node_utils.ensure_repo_root_on_path()
      self.assertEqual(sys.path[0], repo_root)
      node_utils.ensure_repo_root_on_path()
      self.assertEqual(sys.path.count(repo_root), 1)
    finally:
      sys.path[:] = saved_path


class PadAndEosIdsTest(absltest.TestCase):

  def test_returns_pad_and_eos_with_fallbacks(self):
    tok = mock.Mock(pad_token_id=2, eos_token_id=9)
    self.assertEqual(node_utils.pad_and_eos_ids(tok), (2, 9))

    tok_no_eos = mock.Mock(pad_token_id=5, eos_token_id=None)
    self.assertEqual(node_utils.pad_and_eos_ids(tok_no_eos), (5, 5))

    tok_none = mock.Mock(pad_token_id=None, eos_token_id=None)
    self.assertEqual(node_utils.pad_and_eos_ids(tok_none), (0, 0))


class TrajectoryStoreConfigTest(absltest.TestCase):

  def test_disabled_returns_none(self):
    hp = resolve_config.initialize([])
    self.assertIsNone(node_utils.trajectory_store_config(hp))

  def test_enabled_builds_config(self):
    hp = resolve_config.initialize([
        "trajectory_store_config.enable=true",
        "trajectory_store_config.dir=/tmp/store",
        "cluster_config.run_id=run_1",
    ])
    self.assertEqual(
        node_utils.trajectory_store_config(hp),
        {"backend": "file", "root_dir": "/tmp/store", "run_id": "run_1"},
    )

  def test_enabled_without_dir_omits_root_dir(self):
    hp = resolve_config.initialize([
        "trajectory_store_config.enable=true",
        "trajectory_store_config.dir=",
        "cluster_config.run_id=run_1",
    ])
    self.assertEqual(
        node_utils.trajectory_store_config(hp),
        {"backend": "file", "run_id": "run_1"},
    )


class WorkerIdAndPortTest(absltest.TestCase):

  def test_local_target_uses_fixed_local_worker_id(self):
    # The k8s ids are always resolved, but only used on target=k8s.
    hp = resolve_config.initialize(["cluster_config.k8s.trainer_id=ignored"])
    self.assertEqual(
        node_utils.worker_id(hp, "trainer", "trainer-0"), "trainer-0"
    )

  def test_k8s_target_uses_role_id(self):
    hp = resolve_config.initialize([
        "cluster_config.target=k8s",
        "cluster_config.k8s.trainer_id=trainer-1",
        "cluster_config.k8s.rollout_id=roll-1",
        "cluster_config.k8s.inference_id=inf-1",
    ])
    self.assertEqual(
        node_utils.worker_id(hp, "trainer", "trainer-0"), "trainer-1"
    )
    self.assertEqual(node_utils.worker_id(hp, "rollout", "r-0"), "roll-1")
    self.assertEqual(node_utils.worker_id(hp, "inference", "i-0"), "inf-1")

  def test_service_port_reads_role_port(self):
    hp = resolve_config.initialize(["cluster_config.ports.rollout=20051"])
    self.assertEqual(node_utils.service_port(hp, "trainer"), 20000)
    self.assertEqual(node_utils.service_port(hp, "rollout"), 20051)
    self.assertEqual(node_utils.service_port(hp, "inference"), 20002)


class OptionalBoolTest(absltest.TestCase):

  def test_absent_or_null_key_is_none(self):
    self.assertIsNone(node_utils.optional_bool({}, "flag"))
    self.assertIsNone(node_utils.optional_bool({"flag": None}, "flag"))

  def test_set_key_is_bool(self):
    self.assertIs(node_utils.optional_bool({"flag": True}, "flag"), True)
    self.assertIs(node_utils.optional_bool({"flag": False}, "flag"), False)
    self.assertIs(node_utils.optional_bool({"flag": 1}, "flag"), True)


class ServeWorkerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.context = mock.MagicMock()
    self.server = mock.MagicMock(
        spec=remote_execution.GrpcRemoteExecutionServer
    )
    self.server.start_serving_async = mock.AsyncMock()
    self.server.stop_serving = mock.AsyncMock()
    self.enter_context(
        mock.patch.object(
            remote_execution,
            "GrpcRemoteExecutionServer",
            return_value=self.server,
        )
    )

  def _serve(
      self,
      *,
      drain_fn: Callable[[], Awaitable[None] | None] | None = None,
      signal_to_send: signal.Signals | None = signal.SIGTERM,
  ) -> None:
    """Runs `serve_worker`, firing `signal_to_send`'s handler once attached.

    With `signal_to_send=None`, `add_signal_handler` raises
    `NotImplementedError` (as on platforms without signal support) and the
    serving task is cancelled instead.

    Args:
      drain_fn: Passed through to `serve_worker`.
      signal_to_send: Signal whose registered handler is invoked, or None.
    """

    async def _run() -> None:
      loop = asyncio.get_running_loop()

      def _add_signal_handler(
          sig: signal.Signals, callback: Callable[[], None]
      ):
        if signal_to_send is None:
          raise NotImplementedError
        if sig == signal_to_send:
          loop.call_soon(callback)

      with mock.patch.object(
          loop, "add_signal_handler", side_effect=_add_signal_handler
      ):
        task = asyncio.ensure_future(
            node_utils.serve_worker(
                mock.sentinel.worker,
                process_context=self.context,
                role="rollout",
                worker_id_str="rollout-0",
                port=20001,
                extra_metadata={"trajectory_store_config": {"backend": "file"}},
                drain_fn=drain_fn,
            )
        )
        if signal_to_send is None:
          await asyncio.sleep(0)
          task.cancel()
        await task

    asyncio.run(_run())

  def test_registers_drains_and_stops_on_sigterm(self):
    drained: list[str] = []

    async def _drain() -> None:
      drained.append("drained")

    self._serve(drain_fn=_drain)

    self.server.start_serving_async.assert_called_once_with(20001)
    self.context.ipc.discovery.register.assert_called_once()
    registered = pickle.loads(
        self.context.ipc.discovery.register.call_args.kwargs["metadata"]
    )
    self.assertEqual(
        registered,
        {
            "service_type": "rollout",
            "service_port": 20001,
            "worker_id": "rollout-0",
            "trajectory_store_config": {"backend": "file"},
        },
    )
    self.assertEqual(drained, ["drained"])
    self.server.stop_serving.assert_called_once()

  def test_sigint_and_sync_drain_fn(self):
    drain = mock.Mock(return_value=None)
    self._serve(drain_fn=drain, signal_to_send=signal.SIGINT)
    drain.assert_called_once_with()
    self.server.stop_serving.assert_called_once()

  def test_stops_without_drain_fn_when_cancelled_without_signal_support(self):
    self._serve(signal_to_send=None)
    self.server.stop_serving.assert_called_once()

  def test_stops_server_when_drain_fails(self):
    drain = mock.Mock(side_effect=RuntimeError("drain failed"))
    self._serve(drain_fn=drain)
    drain.assert_called_once_with()
    self.server.stop_serving.assert_called_once()

  def test_stops_server_and_raises_when_registration_fails(self):
    self.context.ipc.discovery.register.side_effect = RuntimeError(
        "discovery down"
    )
    drain = mock.Mock(return_value=None)
    with self.assertRaisesRegex(RuntimeError, "discovery down"):
      self._serve(drain_fn=drain)
    drain.assert_called_once_with()
    self.server.stop_serving.assert_called_once()


class LoadHfTokenizerTest(absltest.TestCase):

  def test_loads_from_tokenizer_path_and_defaults_pad_to_eos(self):
    hp = resolve_config.initialize(
        ["tokenizer_config.tokenizer_path=/tmp/models/qwen"]
    )
    tokenizer = mock.Mock(pad_token_id=None, eos_token="<eos>", pad_token=None)
    with mock.patch.object(
        transformers.AutoTokenizer, "from_pretrained", return_value=tokenizer
    ) as from_pretrained:
      self.assertIs(node_utils.load_hf_tokenizer(hp), tokenizer)
    from_pretrained.assert_called_once_with(
        "/tmp/models/qwen", trust_remote_code=True
    )
    self.assertEqual(tokenizer.pad_token, "<eos>")

  def test_keeps_existing_pad_token(self):
    hp = resolve_config.initialize(
        ["tokenizer_config.tokenizer_path=/tmp/models/qwen"]
    )
    tokenizer = mock.Mock(pad_token_id=7, eos_token="<eos>", pad_token="<pad>")
    with mock.patch.object(
        transformers.AutoTokenizer, "from_pretrained", return_value=tokenizer
    ):
      node_utils.load_hf_tokenizer(hp)
    self.assertEqual(tokenizer.pad_token, "<pad>")

  def test_raises_when_nothing_loaded(self):
    hp = resolve_config.initialize(
        ["tokenizer_config.tokenizer_path=/tmp/models/qwen"]
    )
    with mock.patch.object(
        transformers.AutoTokenizer, "from_pretrained", return_value=None
    ):
      with self.assertRaisesRegex(ValueError, "No tokenizer could be loaded"):
        node_utils.load_hf_tokenizer(hp)


class CreateMeshTest(absltest.TestCase):

  def test_validates_device_count(self):
    hp = resolve_config.initialize(["actor_model_config.mesh.shape=(2,2)"])
    with mock.patch.object(jax, "device_count", return_value=2):
      with self.assertRaisesRegex(
          ValueError, "Trainer mesh dimensions must multiply"
      ):
        node_utils.create_mesh(hp, "actor_model_config", "Trainer")

  def test_rejects_non_2d_dp_or_expert_axes(self):
    hp = resolve_config.initialize([
        "actor_model_config.mesh.shape=(2,1,2)",
        "actor_model_config.mesh.axis_names=('dp','fsdp','tp')",
    ])
    with self.assertRaisesRegex(
        ValueError, "only supports 2D \\('fsdp', 'tp'\\) sharding"
    ):
      node_utils.create_mesh(hp, "actor_model_config", "Trainer")

  def test_builds_fsdp_tp_mesh(self):
    # Uses the real device count, which differs across CPU test environments.
    num_devices = jax.device_count()
    hp = resolve_config.initialize(
        [f"actor_model_config.mesh.shape=({num_devices},1)"]
    )
    mesh = node_utils.create_mesh(hp, "actor_model_config", "Trainer")
    self.assertEqual(mesh.axis_names, ("fsdp", "tp"))
    self.assertEqual(mesh.devices.shape, (num_devices, 1))


class EnsureLocalSafetensorsTest(absltest.TestCase):

  def test_has_direct_safetensors(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      p = pathlib.Path(tmp_dir)
      self.assertFalse(node_utils.has_direct_safetensors(p))
      (p / "model.safetensors").touch()
      self.assertTrue(node_utils.has_direct_safetensors(p))

  def test_returns_existing_dir_without_download(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      (pathlib.Path(tmp_dir) / "model.safetensors").touch()
      with mock.patch.object(node_utils, "_download_hf_model") as download:
        self.assertEqual(
            node_utils.ensure_local_safetensors(
                tmp_dir, "Qwen/Qwen3-1.7B", model_key="actor_model_config"
            ),
            tmp_dir,
        )
      download.assert_not_called()

  def test_downloads_when_missing(self):
    def _fake_download(model_id: str, model_dir: str) -> None:
      del model_id
      (pathlib.Path(model_dir) / "model.safetensors").touch()

    with tempfile.TemporaryDirectory() as tmp_dir:
      model_dir = str(pathlib.Path(tmp_dir) / "models" / "qwen")
      with mock.patch.object(
          node_utils, "_download_hf_model", side_effect=_fake_download
      ) as download:
        self.assertEqual(
            node_utils.ensure_local_safetensors(
                model_dir, "Qwen/Qwen3-1.7B", model_key="actor_model_config"
            ),
            model_dir,
        )
      download.assert_called_once_with("Qwen/Qwen3-1.7B", model_dir)

  def test_raises_when_download_has_no_direct_shards(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      with mock.patch.object(node_utils, "_download_hf_model"):
        with self.assertRaisesRegex(ValueError, "no '\\*.safetensors'"):
          node_utils.ensure_local_safetensors(
              tmp_dir, "Qwen/Qwen3-1.7B", model_key="actor_model_config"
          )

  def test_raises_for_empty_or_file(self):
    with self.assertRaisesRegex(
        ValueError, "actor_model_config.model_download_path is required"
    ):
      node_utils.ensure_local_safetensors(
          "", "Qwen/Qwen3-1.7B", model_key="actor_model_config"
      )

    with tempfile.NamedTemporaryFile() as tmp_file:
      with self.assertRaisesRegex(
          ValueError, "must point to a local directory"
      ):
        node_utils.ensure_local_safetensors(
            tmp_file.name, "Qwen/Qwen3-1.7B", model_key="actor_model_config"
        )


if __name__ == "__main__":
  absltest.main()
