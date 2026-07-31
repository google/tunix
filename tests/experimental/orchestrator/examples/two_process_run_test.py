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

"""Training driven from a process that holds no models.

Every distributed test before this one ran several servers inside one process
sharing a single cluster, which exercises the call path but not what makes
distribution hard. Here the trainer and the sampler are separate operating
system processes with their own memory: the orchestrator cannot reach their
state, weights have to be carried between them, and the loss has to be
described rather than handed over.

What is asserted is what that buys, not merely that it ran: the orchestrator
allocates no arrays, the trainer's parameters move in its own process, the
sampler's output changes only after a weight version is transported to it, and
the trainer's loss decreases across steps.
"""

import os
import subprocess
import sys
import tempfile
import time
from typing import Any

from absl.testing import absltest
import numpy as np
import portpicker
import pytest
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import rpc_workers
from tunix.experimental.orchestrator import trainer_handle as trainer_handle_lib
from tunix.experimental.orchestrator import weight_sync_coordinator
from tunix.experimental.orchestrator import weight_transport
from tunix.experimental.orchestrator.examples import worker_process
from tunix.rl.agentic import agentic_grpo_learner

_STARTUP_TIMEOUT_S = 120.0


def _rollout_request(request_id: str) -> datatypes.RolloutRequest:
  return datatypes.RolloutRequest(
      request_id=request_id, prompt="prompt", prompt_id="p0", group_id="g0"
  )

_VOCAB_SIZE = 8


class _WorkerProcess:
  """A worker running in its own process, served over localhost gRPC."""

  def __init__(self, role: str, transport_dir: str):
    self.role = role
    self.port = portpicker.pick_unused_port()
    self._process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tunix.experimental.orchestrator.examples.worker_process",
            f"--role={role}",
            f"--port={self.port}",
            f"--transport_dir={transport_dir}",
            f"--vocab_size={_VOCAB_SIZE}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
    )

  def wait_until_serving(self) -> None:
    """Blocks until the worker announces its port, or the process dies."""
    deadline = time.monotonic() + _STARTUP_TIMEOUT_S
    while time.monotonic() < deadline:
      line = self._process.stdout.readline()
      if not line:
        raise RuntimeError(
            f"{self.role} process exited before serving:\n{self.output()}"
        )
      if worker_process.READY_MARKER in line:
        return
    raise TimeoutError(f"{self.role} process never announced readiness.")

  def output(self) -> str:
    try:
      return self._process.communicate(timeout=5)[0] or ""
    except Exception:  # pylint: disable=broad-exception-caught
      return "(no output captured)"

  def terminate(self) -> None:
    self._process.terminate()
    try:
      self._process.wait(timeout=10)
    except subprocess.TimeoutExpired:
      self._process.kill()


@pytest.mark.multiprocess
class TwoProcessRunTest(absltest.TestCase):
  """The trainer and the sampler are elsewhere; the orchestrator coordinates."""

  def setUp(self):
    super().setUp()
    self.transport_dir = tempfile.mkdtemp()
    self.transport = weight_transport.FileWeightTransport(self.transport_dir)
    self.processes = []

  def tearDown(self):
    for process in self.processes:
      process.terminate()
    super().tearDown()

  def _start(self, role: str) -> Any:
    process = _WorkerProcess(role, self.transport_dir)
    self.processes.append(process)
    process.wait_until_serving()
    return process

  def _handle(self, cls, process):
    return cls.from_address(
        f"grpc://localhost:{process.port}", worker_id=process.role
    )

  def _payload(self, token: int, advantage: float):
    ids = np.array([[token]], dtype=np.int32)
    return agentic_grpo_learner.TrainExample(
        prompt_ids=np.zeros((1, 1), dtype=np.int32),
        prompt_mask=np.zeros((1, 1), dtype=np.int32),
        completion_ids=ids,
        completion_mask=np.ones_like(ids),
        advantages=np.array([advantage], dtype=np.float32),
        ref_per_token_logps=None,
        old_per_token_logps=None,
    )

  def test_trainer_and_sampler_run_in_their_own_processes(self):
    trainer_process = self._start("trainer")
    rollout_process = self._start("rollout")

    trainer = self._handle(rpc_workers.RemoteTrainerWorker, trainer_process)
    rollout = self._handle(
        rpc_workers.RemoteHostedRolloutWorker, rollout_process
    )
    trainer.initialize()
    rollout.initialize()

    # What the sampler produces before any weights reach it.
    before = rollout.generate(_rollout_request("before"))
    self.assertEqual(before.status, "SUCCEEDED")

    # Train in the trainer's process. Token 5 is rewarded, so its score rises
    # and it becomes the sampler's preferred token once the weights arrive.
    losses = []
    for _ in range(4):
      trainer.train([self._payload(token=5, advantage=1.0)], None, False)
      metrics = trainer.drain_metrics()
      if "loss" in metrics:
        losses.append(metrics["loss"])

    self.assertLen(losses, 4)
    self.assertLess(losses[-1], losses[0])

    # Carry one version across, and require the sampler to acknowledge it.
    coordinator = weight_sync_coordinator.WeightSyncCoordinator(
        trainer=trainer, replicas=[rollout]
    )
    outcome = coordinator.sync(version=1)

    self.assertTrue(outcome.all_synced, outcome.quarantined)
    self.assertEqual(outcome.synced, ["rollout"])
    self.assertEqual(rollout.heartbeat().policy_version, 1)

    # The transported weights changed what the sampler emits.
    after = rollout.generate(_rollout_request("after"))
    self.assertEqual(int(after.segments[0].tokens[0]), 5)
    self.assertNotEqual(
        int(before.segments[0].tokens[0]),
        int(after.segments[0].tokens[0]),
    )
    self.assertEqual(after.policy_version, 1)

  def test_the_orchestrator_process_holds_no_arrays(self):
    """The models are in the worker processes, not this one."""
    import jax  # pylint: disable=g-import-not-at-top

    trainer_process = self._start("trainer")
    before = len(jax.live_arrays())

    trainer = self._handle(rpc_workers.RemoteTrainerWorker, trainer_process)
    trainer.initialize()
    trainer.train([self._payload(token=3, advantage=1.0)], None, False)

    # Training happened, and nothing was allocated here to make it happen.
    self.assertEqual(len(jax.live_arrays()), before)

  def test_a_staged_version_is_readable_by_the_other_process(self):
    trainer_process = self._start("trainer")
    trainer = self._handle(rpc_workers.RemoteTrainerWorker, trainer_process)
    trainer.initialize()
    trainer.train([self._payload(token=2, advantage=1.0)], None, False)

    coordinates = trainer.prepare_weight_sync(
        datatypes.WeightSyncRequest(policy_version=7)
    )

    self.assertIsNotNone(coordinates)
    self.assertEqual(coordinates.version, 7)
    fetched = self.transport.fetch(coordinates)
    # Trained in another process, read back here.
    self.assertGreater(float(np.max(fetched["w"])), 0.0)


if __name__ == "__main__":
  absltest.main()
