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

"""Validation demo and test for local and remote worker transport.

This module demonstrates and tests unified execution transport parity across
local and remote workers using `tunix.experimental.worker.remote_execution`:
  1. Local Worker: Co-located in the orchestrator process via in-process
     transport (`transport.local(...)`).
  2. Remote Worker: Running in a separate OS subprocess via gRPC transport
     (`transport.remote(...)`), whose address is dynamically resolved using
     Tunix
     peer discovery (`context.ipc.discovery`).

Both transport modes are validated using a uniform orchestrator test loop
(`validate_worker_transport`), ensuring identical method invocation behavior
regardless of the underlying transport layer.
"""

import argparse
import asyncio
import os
import pickle
import subprocess
import sys
import threading

from absl.testing import absltest
import portpicker
from tunix.experimental.distributed.examples.worker_transport import transport
from tunix.experimental.distributed.examples.worker_transport.worker import Worker
from tunix.experimental.distributed.runtime.contexts.local_context import LocalProcessContext
from tunix.experimental.worker import remote_execution


def start_remote_worker_process(
    service_port: int, discovery_port: int, discovery_addrs: str
) -> subprocess.Popen:
  """Dedicated helper method to spawn the remote worker server in a separate process."""
  remote_worker_bin = os.path.join(
      os.path.dirname(sys.argv[0]), "remote_worker_server"
  )
  if os.path.exists(remote_worker_bin):
    cmd = [remote_worker_bin]
  else:
    python_bin = sys.executable or "python"
    cmd = [
        python_bin,
        "-m",
        "tunix.experimental.distributed.examples.worker_transport.remote_worker_server",
    ]
  return subprocess.Popen(
      cmd
      + [
          f"--port={service_port}",
          "--discovery_id=remote_worker_server",
          f"--discovery_port={discovery_port}",
          f"--discovery_addrs={discovery_addrs}",
      ],
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      text=True,
  )


async def validate_worker_transport(
    handle: remote_execution.ActorHandle, expected_ack: str
) -> None:
  """Connects Orchestrator to a worker via unified ActorHandle interface to validate transport."""
  res = None
  for _ in range(50):
    try:
      res = await handle.asubmit("ping", msg="hello")
      break
    except Exception:  # pylint: disable=broad-exception-caught
      await asyncio.sleep(0.1)
  if res is None:
    res = await handle.asubmit("ping", msg="hello")

  assert res == expected_ack, f"Expected {expected_ack}, got {res}"


class WorkerTransportTest(absltest.TestCase):

  def test_local_worker(self):
    local_handle = transport.local(Worker, name="local")
    asyncio.run(
        validate_worker_transport(
            local_handle, expected_ack="[local] ack: hello"
        )
    )

  def test_remote_worker(self):
    discovery_port = portpicker.pick_unused_port()
    service_port = portpicker.pick_unused_port()

    args = argparse.Namespace(
        discovery_id="orchestrator",
        discovery_port=discovery_port,
        discovery_addrs=f"orchestrator:{discovery_port}",
    )
    with LocalProcessContext(args) as ctx:
      discovered_addr = None
      discovery_event = threading.Event()

      def on_register(hostname: str, _: int, metadata: bytes) -> None:
        nonlocal discovered_addr
        md = pickle.loads(metadata)
        discovered_addr = f"grpc://{hostname}:{md['service_port']}"
        discovery_event.set()

      ctx.ipc.discovery.on_register(on_register)

      proc = start_remote_worker_process(
          service_port=service_port,
          discovery_port=portpicker.pick_unused_port(),
          discovery_addrs=f"orchestrator:{discovery_port}",
      )
      try:
        if not discovery_event.wait(timeout=5.0):
          if proc.poll() is not None:
            out, err = proc.communicate()
            raise RuntimeError(f"Remote worker process exited: {err}")
          raise RuntimeError(
              "Failed to resolve worker via discovery (timed out)"
          )

        assert (
            discovered_addr is not None
        ), "Failed to resolve worker via discovery"
        remote_handle = transport.remote(Worker, address=discovered_addr)
        asyncio.run(
            validate_worker_transport(
                remote_handle, expected_ack="[remote] ack: hello"
            )
        )
      finally:
        if proc.poll() is None:
          proc.terminate()
          try:
            proc.wait(timeout=2.0)
          except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


if __name__ == "__main__":
  absltest.main()
