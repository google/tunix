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

"""Tests for RolloutWorker weight sync phases."""

import unittest

from absl.testing import absltest
from tunix.experimental.common import datatypes
from tunix.experimental.worker import rollout_worker as rollout_worker_lib

WorkerState = datatypes.WorkerState


class _FakeManager:

  def __init__(self):
    self.calls = []
    self.admission_open = True

  async def pre_weight_sync(self, sync_request=None, **kwargs):
    self.calls.append("pre")
    return "ok"

  async def weight_sync(self, sync_request=None, **kwargs):
    self.calls.append("sync")
    return 1

  async def post_weight_sync(self, sync_request=None, **kwargs):
    self.calls.append("post")
    return "ok"

  async def get_weight_sync_metadata(self, **kwargs):
    self.calls.append("metadata")
    return [{"unit": "u0"}]

  def resume_all(self):
    self.calls.append("resume")

  def reopen_admission(self):
    self.admission_open = True
    return True


class _Request:

  def __init__(self, req_id, uuid):
    self.extra_config = {"req_id": req_id, "uuid": uuid}


class WeightSyncPhasesTest(unittest.IsolatedAsyncioTestCase):

  def _worker(self):
    worker = rollout_worker_lib.RolloutWorker(
        worker_id="w0", tokenizer="mock", chat_parser="mock"
    )
    worker.manager = _FakeManager()
    worker._state = WorkerState.READY
    return worker

  async def test_pre_leaves_worker_syncing(self):
    worker = self._worker()
    await worker.pre_weight_sync(_Request("r1", 1))
    self.assertEqual(worker.state, WorkerState.SYNCING)

  async def test_post_restores_ready(self):
    worker = self._worker()
    await worker.pre_weight_sync(_Request("r1", 1))
    await worker.weight_sync(_Request("r1", 1))
    await worker.post_weight_sync(_Request("r1", 1))
    self.assertEqual(worker.state, WorkerState.READY)

  async def test_status_reports_round(self):
    worker = self._worker()
    await worker.pre_weight_sync(_Request("r1", 1))
    status = await worker.get_weight_sync_status()
    self.assertEqual(status["req_id"], "r1")
    self.assertEqual(status["uuid"], 1)
    self.assertEqual(status["phase"], "prepared")

  async def test_abort_resumes_serving(self):
    worker = self._worker()
    await worker.pre_weight_sync(_Request("r1", 1))
    await worker.abort_weight_sync(_Request("r1", 1))
    self.assertEqual(worker.state, WorkerState.READY)
    status = await worker.get_weight_sync_status()
    self.assertEqual(status["phase"], "aborted")

  async def test_metadata_delegates_to_manager(self):
    worker = self._worker()
    result = await worker.get_weight_sync_metadata()
    self.assertEqual(result, [{"unit": "u0"}])

  async def test_sync_falls_back_to_metadata_kwarg(self):
    worker = self._worker()
    await worker.weight_sync(metadata=_Request("r2", 2))
    status = await worker.get_weight_sync_status()
    self.assertEqual(status["req_id"], "r2")
    self.assertEqual(status["phase"], "h2d_done")

  async def test_full_round_call_order(self):
    worker = self._worker()
    req = _Request("r1", 1)
    await worker.bind_weight_sync()
    await worker.get_weight_sync_metadata()
    await worker.pre_weight_sync(req)
    await worker.weight_sync(req)
    await worker.post_weight_sync(req)
    self.assertEqual(worker.manager.calls, ["metadata", "pre", "sync", "post"])


if __name__ == "__main__":
  absltest.main()
