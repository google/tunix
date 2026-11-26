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

"""Unit tests for `ici_slice_worker` torus-rank dispatch and validation."""

from __future__ import annotations

import os
import threading
import types
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
from orbax.checkpoint import multihost as ocp_multihost
from tunix.common import datatypes as common_datatypes
from tunix.dist_rl.distributed.runtime import context as context_lib
from tunix.dist_rl.launch.nodes import ici_slice_worker
from tunix.dist_rl.launch.nodes import node_utils
from tunix.utils import subworld_checkpoint


_BASE_ICI_ARGV = [
    "cluster_config.target=k8s",
    "cluster_config.k8s.tunix_image=img:latest",
    "cluster_config.trainer_replicas=2",
    "cluster_config.rollout_replicas=2",
    "weight_sync_config.mode=ici",
    "cluster_config.fault_tolerance.enabled=false",
    "rollout_engine=inprocess_vllm",
]


def _fake_context() -> Any:
  ctx = mock.create_autospec(context_lib.ProcessContext, instance=True)
  ctx.ipc = types.SimpleNamespace(discovery=object())
  return ctx


class IciSliceWorkerTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="low_rank_trainer",
          rank=1,
          stale_env=node_utils.ROLLOUT_REPLICA_ENV,
          expected_role="trainer",
          expected_env=node_utils.TRAINER_REPLICA_ENV,
          expected_replica="1",
      ),
      dict(
          testcase_name="high_rank_rollout",
          rank=3,
          stale_env=node_utils.TRAINER_REPLICA_ENV,
          expected_role="rollout",
          expected_env=node_utils.ROLLOUT_REPLICA_ENV,
          expected_replica="1",
      ),
  )
  def test_dispatches_rank_to_role(
      self,
      rank: int,
      stale_env: str,
      expected_role: str,
      expected_env: str,
      expected_replica: str,
  ):
    ctx = _fake_context()
    call_order: list[str] = []
    ctx.prepare_env.side_effect = lambda *a, **kw: call_order.append(
        "prepare_env"
    )
    ctx.initialize_runtime.side_effect = lambda *a, **kw: call_order.append(
        "initialize_runtime"
    )
    jax_init = mock.MagicMock(
        side_effect=lambda **kw: call_order.append("jax_initialize")
    )
    jax_shutdown = mock.MagicMock(
        side_effect=lambda: call_order.append("jax_shutdown")
    )
    exit_barrier = mock.MagicMock(
        side_effect=lambda *a, **kw: call_order.append("exit_barrier")
    )
    fakes = {
        role: types.SimpleNamespace(
            main=mock.MagicMock(
                side_effect=lambda *a, r=role: call_order.append(f"{r}_main")
            )
        )
        for role in ("trainer", "rollout")
    }

    def _fake_import(name: str) -> Any:
      suffix = name.rsplit(".", 1)[-1]
      if suffix in fakes:
        return fakes[suffix]
      raise AssertionError(f"Unexpected import: {name}")

    with (
        mock.patch.dict(os.environ, {stale_env: "9"}, clear=False),
        mock.patch.object(jax.distributed, "initialize", jax_init),
        mock.patch.object(jax.distributed, "shutdown", jax_shutdown),
        mock.patch.object(
            ocp_multihost, "sync_global_processes", exit_barrier
        ),
        mock.patch.object(
            subworld_checkpoint,
            "enable_distributed_process_ids",
            side_effect=lambda: call_order.append("enable_distributed_ids"),
        ),
        mock.patch.object(jax, "process_index", return_value=rank),
        mock.patch.object(jax, "process_count", return_value=4),
        mock.patch.object(
            jax, "local_devices", return_value=[types.SimpleNamespace(id=rank)]
        ),
        mock.patch.object(
            ici_slice_worker.importlib,
            "import_module",
            side_effect=_fake_import,
        ),
    ):
      ici_slice_worker.dispatch_slice_worker(_BASE_ICI_ARGV, context=ctx)
      self.assertEqual(os.environ[expected_env], expected_replica)
      self.assertNotIn(stale_env, os.environ)

    self.assertEqual(
        call_order,
        [
            "prepare_env",
            "jax_initialize",
            "enable_distributed_ids",
            "initialize_runtime",
            f"{expected_role}_main",
            "exit_barrier",
            "jax_shutdown",
        ],
    )
    # The default 1200 s checkpoint deadline plus the 300 s slack.
    jax_init.assert_called_once_with(shutdown_timeout_seconds=1500)
    exit_barrier.assert_called_once_with(
        mock.ANY, timeout=1500, processes={0, 1, 2, 3}
    )
    self.assertEqual(
        ctx.prepare_env.call_args.kwargs["weight_sync_mode"],
        common_datatypes.WeightSyncMode.ICI,
    )
    other_role = "rollout" if expected_role == "trainer" else "trainer"
    fakes[expected_role].main.assert_called_once_with(_BASE_ICI_ARGV, ctx)
    fakes[other_role].main.assert_not_called()

  def test_main_exits_after_dispatch(self):
    ctx = _fake_context()
    with (
        mock.patch.object(
            ici_slice_worker, "dispatch_slice_worker"
        ) as dispatch,
        mock.patch.object(ici_slice_worker.logging, "shutdown"),
        mock.patch.object(ici_slice_worker.os, "_exit") as os_exit,
    ):
      os_exit.side_effect = lambda code: dispatch.assert_called_once_with(
          _BASE_ICI_ARGV, context=ctx
      )
      ici_slice_worker.main(_BASE_ICI_ARGV, ctx)
    os_exit.assert_called_once_with(0)

  def test_main_skips_exit_when_dispatch_fails(self):
    with (
        mock.patch.object(
            ici_slice_worker,
            "dispatch_slice_worker",
            side_effect=ValueError("boom"),
        ),
        mock.patch.object(ici_slice_worker.os, "_exit") as os_exit,
    ):
      with self.assertRaisesRegex(ValueError, "boom"):
        ici_slice_worker.main(_BASE_ICI_ARGV, _fake_context())
    os_exit.assert_not_called()

  def test_main_rejects_non_main_thread(self):
    errors: list[BaseException] = []

    def _run() -> None:
      try:
        ici_slice_worker.main(_BASE_ICI_ARGV, _fake_context())
      except RuntimeError as e:
        errors.append(e)

    with (
        mock.patch.object(
            ici_slice_worker, "dispatch_slice_worker"
        ) as dispatch,
        mock.patch.object(ici_slice_worker.os, "_exit") as os_exit,
    ):
      thread = threading.Thread(target=_run)
      thread.start()
      thread.join()
    self.assertLen(errors, 1)
    self.assertIn("only --process", str(errors[0]))
    dispatch.assert_not_called()
    os_exit.assert_not_called()

  def test_rejects_world_size_mismatch(self):
    ctx = _fake_context()
    with (
        mock.patch.object(jax.distributed, "initialize"),
        mock.patch.object(
            subworld_checkpoint, "enable_distributed_process_ids"
        ),
        mock.patch.object(jax, "process_index", return_value=0),
        mock.patch.object(jax, "process_count", return_value=3),
    ):
      with self.assertRaisesRegex(ValueError, "world size"):
        ici_slice_worker.dispatch_slice_worker(_BASE_ICI_ARGV, context=ctx)

  def test_rejects_non_positive_checkpoint_timeout_before_jax_init(self):
    ctx = _fake_context()
    argv = [
        *_BASE_ICI_ARGV,
        "rl_training_config.checkpointing_options.timeout_secs=0",
    ]
    with mock.patch.object(jax.distributed, "initialize") as jax_init:
      with self.assertRaisesRegex(ValueError, "timeout_secs"):
        ici_slice_worker.dispatch_slice_worker(argv, context=ctx)
    jax_init.assert_not_called()

  def test_rejects_non_ici_weight_sync_mode(self):
    ctx = _fake_context()
    argv = [
        "cluster_config.target=k8s",
        "cluster_config.k8s.tunix_image=img:latest",
        "weight_sync_config.mode=raiden",
    ]
    with self.assertRaisesRegex(ValueError, "weight_sync_config.mode='ici'"):
      ici_slice_worker.dispatch_slice_worker(argv, context=ctx)


if __name__ == "__main__":
  absltest.main()
