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

"""Mock Trainer worker process runner shared by distributed RL examples.

Runs on CPU with zero physical TPU or JAX dependencies. Satisfies the
TrainerWorker RPC protocol expected by ClusterOrchestrator and
DistributedRLEngine by subclassing MockWorker.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import pickle
import signal
import sys
from typing import Any

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if REPO_ROOT not in sys.path:
  sys.path.insert(0, REPO_ROOT)

# pylint: disable=g-import-not-at-top
from tunix.experimental.common import datatypes
from tunix.experimental.worker import mock_worker
from tunix.experimental.worker import remote_execution
# pylint: enable=g-import-not-at-top


class MockTrainerWorker(mock_worker.MockWorker):
  """Mock TrainerWorker that satisfies the orchestrator RPC protocol on CPU."""

  def __init__(self, worker_id: str = "mock-trainer-0"):
    super().__init__(
        worker_id=worker_id,
        roles={"trainer", "weight_sync"},
        resources={"trainer": "MockTrainerWorker"},
    )
    self.step_count = 0
    self.policy_version = 0

  def with_loss_fn(
      self, loss_fn: Any, has_aux: bool = False
  ) -> datatypes.Response:
    del loss_fn, has_aux
    return datatypes.Response()

  def with_gen_model_input_fn(
      self, gen_model_input_fn: Any
  ) -> datatypes.Response:
    del gen_model_input_fn
    return datatypes.Response()

  def set_target_state(self, target_state: Any) -> datatypes.Response:
    del target_state
    return datatypes.Response()

  def fwd_bwd(
      self, request: datatypes.TrainRequest, **kwargs: Any
  ) -> datatypes.Response:
    del kwargs
    req_id = getattr(request, "request_id", "")
    logging.info("Mock trainer fwd_bwd called for request %s (no-op).", req_id)
    return datatypes.Response(
        request_id=req_id,
        metadata={"queued": True, "worker_id": self._info.worker_id},
    )

  def update(self, **kwargs: Any) -> int:
    del kwargs
    self.step_count += 1
    self.policy_version += 1
    logging.info(
        "Mock trainer update step %d -> policy_version %d.",
        self.step_count,
        self.policy_version,
    )
    return self.step_count

  def eval_step(
      self, request: datatypes.TrainRequest, **kwargs: Any
  ) -> datatypes.Response:
    del kwargs
    req_id = getattr(request, "request_id", "")
    return datatypes.Response(request_id=req_id, metadata={"evaluated": True})

  def run_eval(self, eval_ds: Any, **kwargs: Any) -> datatypes.Response:
    del eval_ds, kwargs
    return datatypes.Response(metadata={"evaluated": True, "eval_batches": 0})

  def get_metrics(self) -> dict[str, Any]:
    return {
        "loss": 0.0,
        "step": self.step_count,
        "policy_version": self.policy_version,
    }

  def prepare_weight_sync(
      self, sync_request: Any = None, **kwargs: Any
  ) -> datatypes.WeightSyncMetadata:
    del sync_request, kwargs
    return datatypes.WeightSyncMetadata(
        new_policy_version=self.policy_version,
        source_endpoints=[],
    )

  def release_weight_sync(
      self, sync_request: Any = None, **kwargs: Any
  ) -> None:
    del sync_request, kwargs

  def save_checkpoint(
      self, metadata: Any = None, **kwargs: Any
  ) -> datatypes.Response:
    del metadata, kwargs
    return datatypes.Response(metadata={"checkpoint_saved": True})

  def restore_checkpoint(self, **kwargs: Any) -> dict[str, Any]:
    del kwargs
    return {}

  def close(self) -> None:
    self.stop()


def _parse_args(argv: list[str]) -> argparse.Namespace:
  """Parses command line arguments."""
  parser = argparse.ArgumentParser(description="Mock Trainer Worker")
  parser.add_argument("--port", type=int, default=20002)
  parser.add_argument("--worker_id", type=str, default="mock-trainer-0")
  args, _ = parser.parse_known_args(argv)
  return args


def main(argv: list[str], context: Any = None) -> None:
  """Runs the MockTrainerWorker node with discovery registration."""
  if not (context and context.ipc and context.ipc.discovery):
    raise RuntimeError(
        "Require discovery API, but process context doesn't support."
    )

  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s - [MockTrainerNode] %(message)s",
      force=True,
  )
  args = _parse_args(argv)
  logging.info(
      "Starting Mock Trainer Node: worker_id=%s port=%d",
      args.worker_id,
      args.port,
  )

  worker_service = MockTrainerWorker(worker_id=args.worker_id)

  async def grpc_server_main() -> None:
    server = remote_execution.GrpcRemoteExecutionServer(worker_service)
    await server.start_serving_async(args.port)
    logging.info("Mock trainer server listening on port %d.", args.port)

    context.ipc.discovery.register(
        metadata=pickle.dumps({
            "service_type": "trainer",
            "service_port": args.port,
            "worker_id": args.worker_id,
        })
    )
    logging.info("Mock trainer registered with discovery.")

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
      try:
        loop.add_signal_handler(sig, stop_event.set)
      except NotImplementedError:
        pass

    try:
      await stop_event.wait()
    except asyncio.CancelledError:
      pass
    finally:
      logging.info("Stopping mock trainer...")
      worker_service.stop()
      await server.stop_serving()

  asyncio.run(grpc_server_main())


if __name__ == "__main__":
  main(sys.argv[1:])
