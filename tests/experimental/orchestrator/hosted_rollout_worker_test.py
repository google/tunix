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

"""Contract tests for the per-trajectory rollout worker, including over gRPC."""

import asyncio
import contextlib
from typing import Any, Sequence

from absl.testing import absltest
import numpy as np
import portpicker
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import hosted_rollout_worker
from tunix.experimental.orchestrator import rollout_pool
from tunix.experimental.worker import remote_execution as remote_lib


class _Output:
  """The shape the cluster's batched generation returns."""

  def __init__(self, prompts: Sequence[str]):
    self.text = [f"completion for {p}" for p in prompts]
    self.tokens = [np.array([7, 8, 9], dtype=np.int32) for _ in prompts]
    self.logprobs = [
        np.array([-0.1, -0.2, -0.3], dtype=np.float32) for _ in prompts
    ]
    self.left_padded_prompt_tokens = np.array(
        [[0, 1, 2] for _ in prompts], dtype=np.int32
    )
    self.logits = None


class _Engine:
  """A batched generation engine, as the cluster exposes one."""

  def __init__(self, fail_on: str = ""):
    self.prompts_seen = []
    self._fail_on = fail_on

  def generate(self, prompts, *args, **kwargs):
    del args, kwargs
    self.prompts_seen.extend(prompts)
    if self._fail_on and self._fail_on in prompts[0]:
      raise RuntimeError("sampler exploded")
    return _Output(prompts)


def _request(index: int) -> datatypes.RolloutRequest:
  return datatypes.RolloutRequest(
      request_id=f"req-{index}",
      prompt={"prompts": f"prompt {index}"},
      prompt_id=f"p{index}",
      group_id="g0",
  )


@contextlib.asynccontextmanager
async def _served(worker: Any):
  """Serves the worker over localhost gRPC."""
  port = portpicker.pick_unused_port()
  server = remote_lib.GrpcRemoteExecutionServer(worker)
  await server.start_serving_async(port=port)
  handle = remote_lib.GrpcRemoteActorHandle(
      target_address=f"grpc://localhost:{port}", rpc_timeout_s=30.0
  )
  try:
    yield handle
  finally:
    await handle.close()
    await server.stop_serving()


class HostedRolloutWorkerTest(absltest.TestCase):

  def test_single_request_returns_one_stamped_response(self):
    worker = hosted_rollout_worker.HostedRolloutWorker(
        _Engine(), policy_version=4
    )

    response = asyncio.run(worker.generate(_request(0)))

    self.assertIsInstance(response, datatypes.RolloutResponse)
    self.assertEqual(response.request_id, "req-0")
    self.assertEqual(response.status, "SUCCEEDED")
    self.assertEqual(response.policy_version, 4)
    np.testing.assert_array_equal(
        response.segments[0].tokens, np.array([7, 8, 9], dtype=np.int32)
    )
    self.assertEqual(
        response.metadata["completion_text"], "completion for prompt 0"
    )

  def test_batch_returns_one_response_per_request_in_order(self):
    worker = hosted_rollout_worker.HostedRolloutWorker(_Engine())
    streamed = []

    responses = asyncio.run(
        worker.generate(
            [_request(i) for i in range(3)], on_complete=streamed.append
        )
    )

    self.assertEqual(
        [r.request_id for r in responses], ["req-0", "req-1", "req-2"]
    )
    self.assertLen(streamed, 3)

  def test_a_generation_failure_is_reported_in_band(self):
    worker = hosted_rollout_worker.HostedRolloutWorker(
        _Engine(fail_on="prompt 1")
    )

    responses = asyncio.run(worker.generate([_request(0), _request(1)]))

    self.assertEqual(responses[0].status, "SUCCEEDED")
    self.assertEqual(responses[1].status, "FAILED")
    self.assertIn("sampler exploded", responses[1].error.message)
    # The caller can still account for every request it made.
    self.assertEqual(responses[1].request_id, "req-1")

  def test_weight_sync_advances_the_stamped_version(self):
    worker = hosted_rollout_worker.HostedRolloutWorker(_Engine())

    worker.prepare_weight_sync(None)
    version = worker.sync_weights(
        datatypes.WeightSyncRequest(policy_version=9)
    )

    self.assertEqual(version, 9)
    response = asyncio.run(worker.generate(_request(0)))
    self.assertEqual(response.policy_version, 9)
    self.assertEqual(worker.heartbeat().policy_version, 9)

  def test_sync_installs_weights_before_adopting_the_version(self):
    installed = []
    worker = hosted_rollout_worker.HostedRolloutWorker(
        _Engine(), install_weights_fn=installed.append
    )

    version = worker.sync_weights(
        datatypes.WeightSyncRequest(
            policy_version=3, source_metadata="weights-at-3"
        )
    )

    self.assertEqual(version, 3)
    self.assertLen(installed, 1)
    self.assertEqual(installed[0].source_metadata, "weights-at-3")

  def test_a_failed_install_does_not_claim_the_new_version(self):
    """Claiming it would let a round record this worker as synced."""

    def _explode(metadata):
      del metadata
      raise RuntimeError("transfer failed")

    worker = hosted_rollout_worker.HostedRolloutWorker(
        _Engine(), policy_version=1, install_weights_fn=_explode
    )

    with self.assertRaises(RuntimeError):
      worker.sync_weights(datatypes.WeightSyncRequest(policy_version=2))

    self.assertEqual(worker.policy_version, 1)

  def test_an_engine_that_can_update_weights_is_used_automatically(self):
    class _UpdatableEngine(_Engine):

      def __init__(self):
        super().__init__()
        self.updates = []

      def update_weights(self, metadata):
        self.updates.append(metadata.policy_version)

    engine = _UpdatableEngine()
    worker = hosted_rollout_worker.HostedRolloutWorker(engine)

    worker.sync_weights(datatypes.WeightSyncRequest(policy_version=6))

    self.assertEqual(engine.updates, [6])

  def test_reports_its_role_and_lifecycle(self):
    worker = hosted_rollout_worker.HostedRolloutWorker(_Engine())

    self.assertIn("rollout", worker.info().roles)
    worker.initialize()
    worker.start()
    self.assertEqual(
        worker.heartbeat().state, datatypes.WorkerState.READY
    )
    worker.stop()
    self.assertEqual(
        worker.heartbeat().state, datatypes.WorkerState.STOPPED
    )


class HostedRolloutWorkerOverGrpcTest(absltest.TestCase):
  """The per-trajectory contract has to survive the wire."""

  def test_single_batch_and_failure_over_grpc(self):
    async def _run():
      worker = hosted_rollout_worker.HostedRolloutWorker(
          _Engine(fail_on="prompt 2"), policy_version=2
      )
      async with _served(worker) as handle:
        single = await handle.asubmit("generate", _request(0))
        self.assertEqual(single.request_id, "req-0")
        self.assertEqual(single.policy_version, 2)

        batch = await handle.asubmit(
            "generate", [_request(1), _request(2)]
        )
        self.assertEqual([r.status for r in batch], ["SUCCEEDED", "FAILED"])
        self.assertIsNotNone(batch[1].error)

    asyncio.run(_run())

  def test_a_pool_drives_two_hosted_workers_over_grpc(self):
    """The end the pool actually consumes: many workers, one contract."""

    async def _run():
      engines = [_Engine(), _Engine()]
      workers = [
          hosted_rollout_worker.HostedRolloutWorker(
              engine, worker_id=f"rollout-{i}"
          )
          for i, engine in enumerate(engines)
      ]
      async with contextlib.AsyncExitStack() as stack:
        handles = [
            await stack.enter_async_context(_served(worker))
            for worker in workers
        ]
        pool = rollout_pool.PooledRolloutWorker(handles, max_concurrency=1)

        responses = await asyncio.wait_for(
            pool.generate([_request(i) for i in range(6)]), timeout=60.0
        )

        self.assertLen(responses, 6)
        self.assertTrue(all(r.status == "SUCCEEDED" for r in responses))
        self.assertEqual(
            [r.request_id for r in responses],
            [f"req-{i}" for i in range(6)],
        )
        # Both workers took a share.
        served = [len(engine.prompts_seen) for engine in engines]
        self.assertTrue(all(count > 0 for count in served), served)
        self.assertEqual(sum(served), 6)

    asyncio.run(_run())


if __name__ == "__main__":
  absltest.main()
