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
from collections.abc import Awaitable, Callable, Mapping, Sequence
import json
import os
import pathlib
import pickle
import signal
import sys
import tempfile
from typing import Literal
import threading
from unittest import mock

from absl.testing import absltest
import jax
import transformers
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.common import otel_setup
from tunix.dist_rl.launch.common import resolve_config
from tunix.dist_rl.launch.nodes import node_utils
from tunix.dist_rl.worker import abstract_worker
from tunix.dist_rl.worker import remote_execution

# A run whose processes another launcher starts; it needs an absolute artifact
# root.
_EXTERNAL = (
    "cluster_config.target=external",
    "cluster_config.artifact_root=/artifacts/run",
)


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


class LogResolvedConfigTest(absltest.TestCase):

  def test_logs_one_json_record_and_returns_fingerprint(self):
    hp = resolve_config.initialize(["cluster_config.run_id=run_1"])
    env = {
        "TPU_VISIBLE_CHIPS": "0,1",
        "VLLM_TPU_RPA_VERSION": "2",
        "DISABLE_MOSAIC_ATTN": "1",
        "TUNIX_BULK_TCP_LZ4": "0",
        node_utils.ROLLOUT_REPLICA_ENV: "1",
        node_utils.WORKER_PORT_ENV: "41234",
        "HF_TOKEN": "secret",
        "VLLM_API_KEY": "secret",
        "VLLM_api_key": "secret",
        "VERIFY_WEIGHTS_EXTRA": "1",
    }
    with mock.patch.dict(os.environ, env):
      with self.assertLogs(level="INFO") as logs:
        fingerprint = node_utils.log_resolved_config(hp, "trainer")

    self.assertEqual(fingerprint, resolve_config.config_fingerprint(hp))
    marker = node_utils.RESOLVED_CONFIG_LOG_MARKER + " "
    (line,) = [m for m in logs.output if marker in m]
    record = json.loads(line.split(marker, 1)[1])
    self.assertEqual(record["role"], "trainer")
    self.assertEqual(record["config_sha256"], fingerprint)
    self.assertEqual(
        record["config"],
        json.loads(json.dumps(resolve_config.plain_config(hp))),
    )
    # Variables under a logged prefix, and the exact toggles, are recorded.
    self.assertEqual(record["env"]["TPU_VISIBLE_CHIPS"], "0,1")
    self.assertEqual(record["env"]["VLLM_TPU_RPA_VERSION"], "2")
    self.assertEqual(record["env"]["DISABLE_MOSAIC_ATTN"], "1")
    self.assertEqual(record["env"]["TUNIX_BULK_TCP_LZ4"], "0")
    self.assertEqual(record["env"][node_utils.ROLLOUT_REPLICA_ENV], "1")
    self.assertEqual(record["env"][node_utils.WORKER_PORT_ENV], "41234")
    # Credentials are never recorded, even under a logged prefix and in
    # mixed case.
    self.assertNotIn("HF_TOKEN", record["env"])
    self.assertNotIn("VLLM_API_KEY", record["env"])
    self.assertNotIn("VLLM_api_key", record["env"])
    # Exact toggle names are not prefixes.
    self.assertNotIn("VERIFY_WEIGHTS_EXTRA", record["env"])


# The variables `apply_library_env` sets.
_LIBRARY_ENV_VARS = (
    "VERIFY_WEIGHTS",
    "TPU_RAIDEN_DATA_NICS",
    "SKIP_JAX_PRECOMPILE",
)


def _apply_library_env(
    argv: Sequence[str],
    role: Literal["trainer", "rollout"],
    env: Mapping[str, str] | None = None,
) -> dict[str, str]:
  """Runs `apply_library_env` and returns the library variables it leaves.

  Args:
    argv: Config overrides.
    role: Node role.
    env: Library variables already in the environment; the others are unset.

  Returns:
    The `_LIBRARY_ENV_VARS` entries present in `os.environ` after the call.
  """
  start = {k: v for k, v in os.environ.items() if k not in _LIBRARY_ENV_VARS}
  start.update(env or {})
  with mock.patch.dict(os.environ, start, clear=True):
    node_utils.apply_library_env(resolve_config.initialize(list(argv)), role)
    return {
        name: os.environ[name]
        for name in _LIBRARY_ENV_VARS
        if name in os.environ
    }


class ApplyLibraryEnvTest(absltest.TestCase):

  def test_defaults(self):
    self.assertEqual(
        _apply_library_env([], "trainer"),
        {"VERIFY_WEIGHTS": "false", "TPU_RAIDEN_DATA_NICS": "eth0"},
    )
    self.assertEqual(
        _apply_library_env([], "rollout"),
        {
            "VERIFY_WEIGHTS": "false",
            "TPU_RAIDEN_DATA_NICS": "eth0",
            "SKIP_JAX_PRECOMPILE": "1",
        },
    )
    # Empty lets Raiden pick its own NICs: `TPU_RAIDEN_DATA_NICS` stays unset.
    self.assertEqual(
        _apply_library_env(
            ["weight_sync_config.raiden_data_nics=''"], "trainer"
        ),
        {"VERIFY_WEIGHTS": "false"},
    )

  def test_sets_the_configured_values(self):
    argv = [
        "weight_sync_config.verify_weights=true",
        "weight_sync_config.raiden_data_nics=eth0,eth1",
        "vllm_config.skip_jax_precompile=false",
    ]
    self.assertEqual(
        _apply_library_env(argv, "trainer"),
        {
            "VERIFY_WEIGHTS": "true",
            "TPU_RAIDEN_DATA_NICS": "eth0,eth1",
        },
    )
    self.assertEqual(
        _apply_library_env(argv, "rollout"),
        {
            "VERIFY_WEIGHTS": "true",
            "TPU_RAIDEN_DATA_NICS": "eth0,eth1",
            "SKIP_JAX_PRECOMPILE": "0",
        },
    )

  def test_accepts_the_same_value_in_the_environment(self):
    env = {
        "VERIFY_WEIGHTS": "false",
        "TPU_RAIDEN_DATA_NICS": "eth0",
        "SKIP_JAX_PRECOMPILE": "1",
    }
    self.assertEqual(_apply_library_env([], "rollout", env), env)

  def test_rejects_another_value_in_the_environment(self):
    with self.assertRaisesRegex(
        ValueError,
        "SKIP_JAX_PRECOMPILE='true' is set in the environment, but"
        " vllm_config.skip_jax_precompile sets it to '1'",
    ):
      _apply_library_env([], "rollout", {"SKIP_JAX_PRECOMPILE": "true"})
    with self.assertRaisesRegex(
        ValueError,
        "VERIFY_WEIGHTS='true' .* weight_sync_config.verify_weights sets it to"
        " 'false'.*set weight_sync_config.verify_weights instead",
    ):
      _apply_library_env([], "trainer", {"VERIFY_WEIGHTS": "true"})

  def test_rejects_a_variable_the_config_leaves_unset(self):
    with self.assertRaisesRegex(
        ValueError,
        "weight_sync_config.raiden_data_nics leaves it unset.*set"
        " weight_sync_config.raiden_data_nics instead",
    ):
      _apply_library_env(
          ["weight_sync_config.raiden_data_nics=''"],
          "trainer",
          {"TPU_RAIDEN_DATA_NICS": "eth0"},
      )

  def test_sets_nothing_when_a_variable_conflicts(self):
    env = {k: v for k, v in os.environ.items() if k not in _LIBRARY_ENV_VARS}
    env["SKIP_JAX_PRECOMPILE"] = "0"
    hp = resolve_config.initialize([])
    with mock.patch.dict(os.environ, env, clear=True):
      with self.assertRaisesRegex(ValueError, "SKIP_JAX_PRECOMPILE"):
        node_utils.apply_library_env(hp, "rollout")
      self.assertNotIn("VERIFY_WEIGHTS", os.environ)


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

  def test_enabled_file_store_without_dir_uses_artifact_root(self):
    hp = resolve_config.initialize([
        *_EXTERNAL,
        "trajectory_store_config.enable=true",
        "trajectory_store_config.dir=",
        "cluster_config.run_id=run_1",
    ])
    self.assertEqual(
        node_utils.trajectory_store_config(hp),
        {
            "backend": "file",
            "root_dir": "/artifacts/run/trajectory_store",
            "run_id": "run_1",
        },
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
        "agentic_grpo_config.beta=0.0",
        "cluster_config.k8s.trainer_id=trainer-1",
        "cluster_config.k8s.rollout_id=roll-1",
        "cluster_config.k8s.inference_id=inf-1",
    ])
    self.assertEqual(
        node_utils.worker_id(hp, "trainer", "trainer-0"), "trainer-1"
    )
    self.assertEqual(node_utils.worker_id(hp, "rollout", "r-0"), "roll-1")
    self.assertEqual(node_utils.worker_id(hp, "inference", "i-0"), "inf-1")

  def test_k8s_rollout_replica_appends_index_from_env(self):
    hp = resolve_config.initialize([
        "cluster_config.target=k8s",
        "cluster_config.k8s.trainer_id=trainer-1",
        "cluster_config.k8s.rollout_id=roll-1",
        "cluster_config.rollout_replicas=2",
        # `k8s.sh` deploys no reference worker.
        "agentic_grpo_config.beta=0.0",
    ])
    replica_env = node_utils.ROLLOUT_REPLICA_ENV
    with mock.patch.dict(os.environ, {replica_env: "1"}):
      self.assertEqual(node_utils.worker_id(hp, "rollout", "r-0"), "roll-1-1")
      # Only rollout replicas carry an index.
      self.assertEqual(
          node_utils.worker_id(hp, "trainer", "trainer-0"), "trainer-1"
      )
    with mock.patch.dict(os.environ, {replica_env: "2"}):
      with self.assertRaisesRegex(ValueError, "out of range"):
        node_utils.worker_id(hp, "rollout", "r-0")
    env = {k: v for k, v in os.environ.items() if k != replica_env}
    with mock.patch.dict(os.environ, env, clear=True):
      with self.assertRaisesRegex(ValueError, "must be set"):
        node_utils.worker_id(hp, "rollout", "r-0")

  def test_external_target_uses_fixed_local_worker_ids(self):
    hp = resolve_config.initialize(list(_EXTERNAL))
    replica_env = node_utils.ROLLOUT_REPLICA_ENV
    env = {k: v for k, v in os.environ.items() if k != replica_env}
    with mock.patch.dict(os.environ, env, clear=True):
      for role, local_worker_id in (
          ("trainer", "trainer-0"),
          ("rollout", "rollout-0"),
          ("inference", "reference-inference-0"),
      ):
        self.assertEqual(
            node_utils.worker_id(hp, role, local_worker_id), local_worker_id
        )
    # Trainer and inference ignore `DIST_RL_ROLLOUT_REPLICA`; a single rollout
    # accepts index "0" (still returning `local_worker_id`) and rejects an
    # out-of-range or malformed index.
    with mock.patch.dict(os.environ, {replica_env: "5"}):
      self.assertEqual(
          node_utils.worker_id(hp, "trainer", "trainer-0"), "trainer-0"
      )
      self.assertEqual(
          node_utils.worker_id(hp, "inference", "reference-inference-0"),
          "reference-inference-0",
      )
      with self.assertRaisesRegex(
          ValueError, "out of range for cluster_config.rollout_replicas=1"
      ):
        node_utils.worker_id(hp, "rollout", "rollout-0")
    with mock.patch.dict(os.environ, {replica_env: "0"}):
      self.assertEqual(
          node_utils.worker_id(hp, "rollout", "rollout-0"), "rollout-0"
      )

  def test_external_rollout_replica_takes_index_from_env(self):
    hp = resolve_config.initialize(
        [*_EXTERNAL, "cluster_config.rollout_replicas=3"]
    )
    replica_env = node_utils.ROLLOUT_REPLICA_ENV
    with mock.patch.dict(os.environ, {replica_env: "2"}):
      self.assertEqual(
          node_utils.worker_id(hp, "rollout", "rollout-0"), "rollout-2"
      )
      # Only rollout replicas carry an index.
      self.assertEqual(
          node_utils.worker_id(hp, "trainer", "trainer-0"), "trainer-0"
      )
    with mock.patch.dict(os.environ, {replica_env: "3"}):
      with self.assertRaisesRegex(
          ValueError, "out of range for cluster_config.rollout_replicas=3"
      ):
        node_utils.worker_id(hp, "rollout", "rollout-0")
    for bad_replica in ("", "x", "+1", "-1", " 0"):
      with self.subTest(bad_replica=bad_replica):
        with mock.patch.dict(os.environ, {replica_env: bad_replica}):
          with self.assertRaisesRegex(
              ValueError, f"{replica_env} must be a non-negative integer"
          ):
            node_utils.worker_id(hp, "rollout", "rollout-0")
    env = {k: v for k, v in os.environ.items() if k != replica_env}
    with mock.patch.dict(os.environ, env, clear=True):
      with self.assertRaisesRegex(ValueError, f"{replica_env} must be set"):
        node_utils.worker_id(hp, "rollout", "rollout-0")

  def test_service_port_reads_role_port(self):
    port_env = node_utils.WORKER_PORT_ENV
    env = {k: v for k, v in os.environ.items() if k != port_env}
    for target in ("local", "k8s"):
      hp = resolve_config.initialize([
          f"cluster_config.target={target}",
          "cluster_config.ports.rollout=20051",
          "agentic_grpo_config.beta=0.0",
      ])
      with self.subTest(target=target):
        with mock.patch.dict(os.environ, env, clear=True):
          self.assertEqual(node_utils.service_port(hp, "trainer"), 20000)
          self.assertEqual(node_utils.service_port(hp, "rollout"), 20051)
          self.assertEqual(node_utils.service_port(hp, "inference"), 20002)
        # The port variable is only for the external target.
        with mock.patch.dict(os.environ, {port_env: "41234"}):
          with self.assertRaisesRegex(
              ValueError,
              f"{port_env} is only for cluster_config.target=external",
          ):
            node_utils.service_port(hp, "trainer")

  def test_external_service_port_reads_worker_port_env(self):
    hp = resolve_config.initialize(list(_EXTERNAL))
    port_env = node_utils.WORKER_PORT_ENV
    with mock.patch.dict(os.environ, {port_env: "41234"}):
      for role in ("trainer", "rollout", "inference"):
        self.assertEqual(node_utils.service_port(hp, role), 41234)
    env = {k: v for k, v in os.environ.items() if k != port_env}
    with mock.patch.dict(os.environ, env, clear=True):
      with self.assertRaisesRegex(ValueError, f"{port_env} must be set"):
        node_utils.service_port(hp, "trainer")
    for port in ("", "port", "-1", "0", "65536", " 41234"):
      with self.subTest(port=port):
        with mock.patch.dict(os.environ, {port_env: port}):
          with self.assertRaisesRegex(
              ValueError, f"{port_env} must be a port number"
          ):
            node_utils.service_port(hp, "rollout")


class SetupOtelMetricsTest(absltest.TestCase):

  def test_names_the_service_and_tags_role_run_and_worker(self):
    hp = resolve_config.initialize(["cluster_config.run_id=run_1"])
    with mock.patch.object(
        otel_setup, "setup_metrics", return_value=True
    ) as setup_metrics:
      self.assertTrue(
          node_utils.setup_otel_metrics(hp, "rollout", "rollout-0")
      )
      node_utils.setup_otel_metrics(hp, "orchestrator", worker_id_str=None)
    self.assertEqual(
        setup_metrics.call_args_list,
        [
            mock.call(
                default_service_name="tunix-rollout",
                resource_attributes={
                    "tunix.role": "rollout",
                    "tunix.run_id": "run_1",
                    "tunix.worker_id": "rollout-0",
                },
            ),
            # The orchestrator registers no worker id.
            mock.call(
                default_service_name="tunix-orchestrator",
                resource_attributes={
                    "tunix.role": "orchestrator",
                    "tunix.run_id": "run_1",
                },
            ),
        ],
    )


class OptionalBoolTest(absltest.TestCase):

  def test_absent_or_null_key_is_none(self):
    self.assertIsNone(node_utils.optional_bool({}, "flag"))
    self.assertIsNone(node_utils.optional_bool({"flag": None}, "flag"))

  def test_set_key_is_bool(self):
    self.assertIs(node_utils.optional_bool({"flag": True}, "flag"), True)
    self.assertIs(node_utils.optional_bool({"flag": False}, "flag"), False)
    self.assertIs(node_utils.optional_bool({"flag": 1}, "flag"), True)


class _StubWorker(abstract_worker.Worker):
  """Minimal concrete `Worker` for exercising `serve_worker` lifecycle hooks."""

  def __init__(self, *, stop_error: Exception | None = None) -> None:
    self._stop_error = stop_error
    self.on_stop_callbacks: list[Callable[[bool], None] | None] = []

  def set_on_stop(self, callback: Callable[[bool], None] | None) -> None:
    self.on_stop_callbacks.append(callback)
    super().set_on_stop(callback)

  def initialize(self) -> datatypes.Response:
    return datatypes.Response()

  def compile(self, dummy_data: object) -> datatypes.Response:
    del dummy_data
    return datatypes.Response()

  def start(self) -> datatypes.Response:
    return datatypes.Response()

  def stop(self, *, abort: bool = False) -> datatypes.Response:
    try:
      self.state = datatypes.WorkerState.STOPPED
      if self._stop_error is not None:
        abort = True
        raise self._stop_error
      return datatypes.Response(
          metadata={"stopped": True, "abort": bool(abort)}
      )
    finally:
      self._notify_stopped(abort)

  def info(self) -> datatypes.WorkerInfo:
    return datatypes.WorkerInfo(
        worker_id="rollout-0", roles=frozenset({"rollout"})
    )

  def heartbeat(self) -> datatypes.HealthReport:
    return datatypes.HealthReport(state=self.state)


class ServeWorkerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.context = mock.MagicMock()
    self.worker = _StubWorker()
    self.server = mock.MagicMock(
        spec=remote_execution.GrpcRemoteExecutionServer
    )
    self.server.start_serving_async = mock.AsyncMock()
    self.server.stop_serving = mock.AsyncMock()
    self.mock_server_cls = self.enter_context(
        mock.patch.object(
            remote_execution,
            "GrpcRemoteExecutionServer",
            return_value=self.server,
        )
    )

  def _serve(
      self,
      *,
      worker: _StubWorker | None = None,
      drain_fn: Callable[[], Awaitable[None] | None] | None = None,
      close_fn: Callable[[], None] | None = None,
      signal_to_send: signal.Signals | None = signal.SIGTERM,
  ) -> None:
    """Runs `serve_worker`, firing `signal_to_send`'s handler once attached.

    With `signal_to_send=None`, `add_signal_handler` raises
    `NotImplementedError` (as on platforms without signal support) and the
    serving task is cancelled instead.

    Args:
      worker: Worker service passed to `serve_worker`; defaults to
        `self.worker`.
      drain_fn: Passed through to `serve_worker`.
      close_fn: Passed through to `serve_worker`.
      signal_to_send: Signal whose registered handler is invoked, or None.
    """
    target_worker = self.worker if worker is None else worker

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
                target_worker,
                process_context=self.context,
                role="rollout",
                worker_id_str="rollout-0",
                port=20001,
                config_sha256="config-sha",
                extra_metadata={"trajectory_store_config": {"backend": "file"}},
                drain_fn=drain_fn,
                close_fn=close_fn,
            )
        )
        if signal_to_send is None:
          await asyncio.sleep(0)
          task.cancel()
        await task

    asyncio.run(_run())

  def _serve_until_remote_stop(self, *, abort: bool, calls: list[str]) -> None:
    """Runs `serve_worker` and invokes `worker.stop(abort=...)` off-thread."""
    registered = asyncio.Event()
    self.context.ipc.discovery.register.side_effect = (
        lambda **_: registered.set()
    )

    def _check_on_stop_registered(_: int) -> None:
      self.assertLen(self.worker.on_stop_callbacks, 1)
      self.assertIsNotNone(self.worker.on_stop_callbacks[0])

    self.server.start_serving_async.side_effect = _check_on_stop_registered
    self.server.stop_serving.side_effect = lambda **_: calls.append(
        "stop_serving"
    )

    async def _run() -> None:
      loop = asyncio.get_running_loop()
      with mock.patch.object(loop, "add_signal_handler"):
        task = asyncio.create_task(
            node_utils.serve_worker(
                self.worker,
                process_context=self.context,
                role="rollout",
                worker_id_str="rollout-0",
                port=20001,
                config_sha256="config-sha",
                drain_fn=lambda: calls.append("drain"),
                close_fn=lambda: calls.append("close"),
            )
        )
        await registered.wait()
        stop_thread = threading.Thread(
            target=lambda: self.worker.stop(abort=abort)
        )
        stop_thread.start()
        try:
          await task
        finally:
          stop_thread.join(timeout=5.0)

    asyncio.run(_run())

  def test_registers_drains_and_stops_on_sigterm(self):
    drained: list[str] = []

    async def _drain() -> None:
      drained.append("drained")

    self._serve(drain_fn=_drain)

    self.mock_server_cls.assert_called_once_with(self.worker)
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
            node_utils.CONFIG_SHA256_METADATA_KEY: "config-sha",
            "trajectory_store_config": {"backend": "file"},
        },
    )
    self.assertEqual(drained, ["drained"])
    self.server.stop_serving.assert_called_once()
    self.assertIsNone(self.worker.on_stop_callbacks[-1])

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
    close = mock.Mock(return_value=None)
    with self.assertRaisesRegex(RuntimeError, "discovery down"):
      self._serve(drain_fn=drain, close_fn=close)
    drain.assert_called_once_with()
    self.server.stop_serving.assert_called_once()
    close.assert_called_once_with()
    self.assertIsNone(self.worker.on_stop_callbacks[-1])

  def test_drains_then_stops_serving_then_closes(self):
    calls: list[str] = []

    def _drain() -> None:
      calls.append("drain")

    def _stop_serving(grace: float) -> None:
      del grace  # Unused.
      calls.append("stop_serving")

    def _close() -> None:
      calls.append("close")

    self.server.stop_serving.side_effect = _stop_serving
    self._serve(drain_fn=_drain, close_fn=_close)
    self.assertEqual(calls, ["drain", "stop_serving", "close"])

  def test_closes_and_raises_when_stop_serving_fails(self):
    self.server.stop_serving.side_effect = RuntimeError("stop failed")
    close = mock.Mock(return_value=None)
    with self.assertRaisesRegex(RuntimeError, "stop failed"):
      self._serve(close_fn=close)
    close.assert_called_once_with()
    self.assertIsNone(self.worker.on_stop_callbacks[-1])

  def test_close_failure_raises_after_stop_serving(self):
    close = mock.Mock(side_effect=RuntimeError("close failed"))
    with self.assertRaisesRegex(RuntimeError, "close failed"):
      self._serve(close_fn=close, signal_to_send=signal.SIGINT)
    self.server.stop_serving.assert_called_once()
    close.assert_called_once_with()
    self.assertIsNone(self.worker.on_stop_callbacks[-1])

  def test_clean_remote_stop_wakes_drains_closes_and_returns_cleanly(self):
    calls: list[str] = []
    self._serve_until_remote_stop(abort=False, calls=calls)
    self.assertEqual(calls, ["drain", "stop_serving", "close"])
    self.assertEqual(self.worker.state, datatypes.WorkerState.STOPPED)
    self.assertLen(self.worker.on_stop_callbacks, 2)
    self.assertIsNone(self.worker.on_stop_callbacks[-1])

  def test_abort_remote_stop_drains_closes_and_exits_nonzero(self):
    calls: list[str] = []
    with self.assertRaises(SystemExit) as cm:
      self._serve_until_remote_stop(abort=True, calls=calls)
    self.assertEqual(cm.exception.code, 1)
    self.assertEqual(calls, ["drain", "stop_serving", "close"])
    self.assertEqual(self.worker.state, datatypes.WorkerState.STOPPED)
    self.assertLen(self.worker.on_stop_callbacks, 2)
    self.assertIsNone(self.worker.on_stop_callbacks[-1])

  def test_sigterm_drain_stop_failure_escalates_to_nonzero_exit(self):
    failing_worker = _StubWorker(stop_error=RuntimeError("cleanup failed"))
    calls: list[str] = []
    self.server.stop_serving.side_effect = lambda **_: calls.append(
        "stop_serving"
    )

    def _drain() -> None:
      calls.append("drain")
      failing_worker.stop()

    with self.assertRaises(SystemExit) as cm:
      self._serve(
          worker=failing_worker,
          drain_fn=_drain,
          close_fn=lambda: calls.append("close"),
      )
    self.assertEqual(cm.exception.code, 1)
    self.assertEqual(calls, ["drain", "stop_serving", "close"])
    self.assertIsNone(failing_worker.on_stop_callbacks[-1])

  def test_stops_serving_with_a_grace_period_for_the_stop_rpc(self):
    self._serve()
    self.server.stop_serving.assert_awaited_once_with(
        grace=node_utils._STOP_SERVING_GRACE_S  # pylint: disable=protected-access
    )

  def test_removes_its_signal_handlers_before_draining(self):
    calls: list[str] = []

    async def _run() -> None:
      loop = asyncio.get_running_loop()
      with (
          mock.patch.object(
              loop,
              "add_signal_handler",
              side_effect=lambda sig, callback: (
                  loop.call_soon(callback) if sig == signal.SIGTERM else None
              ),
          ),
          mock.patch.object(
              loop,
              "remove_signal_handler",
              side_effect=lambda sig: calls.append(f"remove:{sig.name}"),
          ),
      ):
        await node_utils.serve_worker(
            self.worker,
            process_context=self.context,
            role="trainer",
            worker_id_str="trainer-0",
            port=20000,
            config_sha256="config-sha",
            drain_fn=lambda: calls.append("drain"),
        )

    asyncio.run(_run())
    self.assertEqual(calls, ["remove:SIGINT", "remove:SIGTERM", "drain"])

  def test_close_failure_after_a_failed_stop_exits_nonzero(self):
    # The trainer node stops its worker in `close_fn`. A failed stop escalates
    # to an abort and re-raises; the process still exits 1.
    failing_worker = _StubWorker(stop_error=RuntimeError("cleanup failed"))

    def _close() -> None:
      failing_worker.stop()

    with self.assertRaises(SystemExit) as cm:
      self._serve(worker=failing_worker, close_fn=_close)
    self.assertEqual(cm.exception.code, 1)
    self.assertIsNone(failing_worker.on_stop_callbacks[-1])

  def test_unregisters_on_stop_when_start_serving_fails(self):
    def _fail_after_check(_: int) -> None:
      self.assertLen(self.worker.on_stop_callbacks, 1)
      self.assertIsNotNone(self.worker.on_stop_callbacks[0])
      raise OSError("port in use")

    self.server.start_serving_async.side_effect = _fail_after_check
    drain = mock.Mock()
    close = mock.Mock()
    with self.assertRaisesRegex(OSError, "port in use"):
      self._serve(drain_fn=drain, close_fn=close)
    self.assertLen(self.worker.on_stop_callbacks, 2)
    self.assertIsNone(self.worker.on_stop_callbacks[-1])
    self.context.ipc.discovery.register.assert_not_called()
    drain.assert_not_called()
    self.server.stop_serving.assert_not_called()
    close.assert_not_called()


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
