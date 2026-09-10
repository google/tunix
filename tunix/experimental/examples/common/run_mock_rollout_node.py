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

"""Mock Rollout worker process runner shared by distributed RL examples.

Runs on CPU without TPUs or vLLM. Returns wire-compatible RolloutResponses
with synthetic completions and rewards.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import pickle
import signal
import sys
from typing import Any, Sequence

import numpy as np

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if REPO_ROOT not in sys.path:
  sys.path.insert(0, REPO_ROOT)

# pylint: disable=g-import-not-at-top
from tunix.experimental.common import datatypes
from tunix.experimental.worker import mock_worker
from tunix.experimental.worker import remote_execution
from tunix.rl.rollout import base_rollout
# pylint: enable=g-import-not-at-top

DEFAULT_FIXED_RESPONSE = (
    "echo 'Mock patch applied successfully' > /dev/null\n"
    "FINAL_ANSWER: Completed mock SWE problem resolution."
)


class MockRolloutWorker(mock_worker.MockWorker):
  """Mock RolloutWorker returning fixed responses on CPU."""

  def __init__(
      self,
      worker_id: str = "mock-rollout-0",
      fixed_response: str = DEFAULT_FIXED_RESPONSE,
      reward: float = 1.0,
  ):
    super().__init__(
        worker_id=worker_id,
        roles={"rollout"},
        resources={"sampler": "MockRolloutWorker"},
    )
    self.fixed_response = fixed_response
    self.reward = reward
    self.policy_version = 0
    self.total_rollouts = 0

  def pause(self) -> datatypes.Response:
    return datatypes.Response()

  def resume(self) -> datatypes.Response:
    return datatypes.Response()

  def get_metrics(self, **kwargs: Any) -> dict[str, Any]:
    del kwargs
    return {
        "rollout_count": self.total_rollouts,
        "policy_version": self.policy_version,
    }

  def pre_weight_sync(self, metadata: Any = None, **kwargs: Any) -> int:
    del kwargs
    return getattr(metadata, "new_policy_version", self.policy_version)

  def weight_sync(self, metadata: Any = None, **kwargs: Any) -> int:
    del kwargs
    if metadata and hasattr(metadata, "new_policy_version"):
      self.policy_version = metadata.new_policy_version
    logging.info(
        "Mock rollout updated policy_version to %d.", self.policy_version
    )
    return self.policy_version

  def get_target_state(self) -> Any:
    return None

  def _process_single_request(self, req: Any) -> datatypes.RolloutResponse:
    self.total_rollouts += 1
    if isinstance(req, str):
      prompt_id = f"prompt_{self.total_rollouts}"
      group_index = 0
      prompt_text = req
      metadata = {}
      req_id = f"req_{self.total_rollouts}"
    else:
      prompt_id = getattr(req, "prompt_id", f"prompt_{self.total_rollouts}")
      group_index = getattr(req, "group_index", 0)
      prompt_text = getattr(req, "prompt", str(req))
      metadata = dict(getattr(req, "metadata", {}) or {})
      req_id = getattr(req, "request_id", f"req_{self.total_rollouts}")

    completion_text = self.fixed_response

    prompt_tokens = np.array(
        [ord(c) % 1000 for c in str(prompt_text)[:128]] or [101],
        dtype=np.int32,
    )
    completion_tokens = np.array(
        [ord(c) % 1000 for c in completion_text[:512]] or [102],
        dtype=np.int32,
    )
    loss_mask = np.ones_like(completion_tokens, dtype=np.float32)
    logps = np.zeros_like(completion_tokens, dtype=np.float32)

    metadata.setdefault("text", completion_text)

    payload = datatypes.TrajectoryItem(
        prompt_id=prompt_id,
        group_index=group_index,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        action_mask=loss_mask,
        segments=[
            datatypes.TokenSegment(
                source="assistant",
                tokens=completion_tokens,
                loss_mask=loss_mask,
                logps=logps,
            )
        ],
        traj={"reward": self.reward, "status": "SUCCEEDED"},
        policy_version=self.policy_version,
        metadata=metadata,
    )

    return datatypes.RolloutResponse(
        request_id=req_id,
        status="COMPLETED",
        payload=payload,
        metadata=metadata,
    )

  async def generate(
      self,
      requests: Any = None,
      prompts: Any = None,
      **kwargs: Any,
  ) -> list[datatypes.RolloutResponse]:
    del kwargs
    if requests is None:
      requests = prompts
    if requests is None:
      return []

    req_list = (
        list(requests) if isinstance(requests, (list, tuple)) else [requests]
    )
    responses = [self._process_single_request(req) for req in req_list]

    logging.info(
        "Mock rollout generated %d response(s).",
        len(responses),
    )
    return list(responses)

  async def sample_prompts(
      self,
      prompts: str | Sequence[str],
      **kwargs: Any,
  ) -> base_rollout.RolloutOutput:
    del kwargs
    prompt_list = [prompts] if isinstance(prompts, str) else list(prompts)
    tokens = [
        np.array([ord(c) % 1000 for c in self.fixed_response], dtype=np.int32)
        for _ in prompt_list
    ]
    return base_rollout.RolloutOutput(
        text=[self.fixed_response for _ in prompt_list],
        logits=None,
        tokens=tokens,
        left_padded_prompt_tokens=np.zeros(
            (len(prompt_list), 1), dtype=np.int32
        ),
        logprobs=[np.zeros_like(t, dtype=np.float32) for t in tokens],
    )

  def poll_responses(self, timeout_s: float = 0.0) -> list[Any]:
    del timeout_s
    return []


def _parse_args(argv: list[str]) -> argparse.Namespace:
  """Parses command line arguments."""
  parser = argparse.ArgumentParser(description="Mock Rollout Worker")
  parser.add_argument("--port", type=int, default=20001)
  parser.add_argument("--worker_id", type=str, default="mock-rollout-0")
  parser.add_argument(
      "--fixed_response",
      type=str,
      default=os.getenv("FIXED_RESPONSE", DEFAULT_FIXED_RESPONSE),
  )
  parser.add_argument("--reward", type=float, default=1.0)
  args, _ = parser.parse_known_args(argv)
  return args


def main(argv: list[str], context: Any = None) -> None:
  """Runs the MockRolloutWorker node with discovery registration."""
  if not (context and context.ipc and context.ipc.discovery):
    raise RuntimeError(
        "Require discovery API, but process context doesn't support."
    )

  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s - [MockRolloutNode] %(message)s",
      force=True,
  )
  args = _parse_args(argv)
  logging.info(
      "Starting Mock Rollout Node: worker_id=%s port=%d",
      args.worker_id,
      args.port,
  )

  worker_service = MockRolloutWorker(
      worker_id=args.worker_id,
      fixed_response=args.fixed_response,
      reward=args.reward,
  )

  async def grpc_server_main() -> None:
    server = remote_execution.GrpcRemoteExecutionServer(worker_service)
    await server.start_serving_async(args.port)
    logging.info("Mock rollout server listening on port %d.", args.port)

    context.ipc.discovery.register(
        metadata=pickle.dumps({
            "service_type": "rollout",
            "service_port": args.port,
            "worker_id": args.worker_id,
        })
    )
    logging.info("Mock rollout registered with discovery.")

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
      logging.info("Stopping mock rollout...")
      worker_service.stop()
      await server.stop_serving()

  asyncio.run(grpc_server_main())


if __name__ == "__main__":
  main(sys.argv[1:])
