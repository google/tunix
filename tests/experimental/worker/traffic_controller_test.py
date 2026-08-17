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

"""Tests for traffic_controller."""

import asyncio
import unittest

from absl.testing import absltest
from tunix.experimental.common import datatypes
from tunix.experimental.worker import traffic_controller

WorkerState = datatypes.WorkerState


class TrafficControllerTest(unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    super().setUp()
    self.controller = traffic_controller.TrafficController()

  def test_initial_state(self):
    self.assertEqual(self.controller.state, WorkerState.READY)
    self.assertTrue(self.controller.is_admission_open())
    self.assertEqual(len(self.controller.get_active_tasks()), 0)

  async def test_try_admit_success(self):
    async def dummy_task():
      await asyncio.sleep(0.01)

    task = asyncio.create_task(dummy_task())
    self.assertTrue(self.controller.try_admit(task))
    self.assertIn(task, self.controller.get_active_tasks())

    await task

    # Task should be removed automatically
    self.assertEqual(len(self.controller.get_active_tasks()), 0)

  async def test_try_admit_rejected_when_syncing(self):
    self.controller.transition_to_syncing()
    self.assertEqual(self.controller.state, WorkerState.SYNCING)
    self.assertFalse(self.controller.is_admission_open())

    async def dummy_task():
      pass

    task = asyncio.create_task(dummy_task())
    self.assertFalse(self.controller.try_admit(task))
    await asyncio.sleep(0)
    self.assertTrue(task.cancelled())
    self.assertEqual(len(self.controller.get_active_tasks()), 0)

  async def test_reopen(self):
    self.controller.transition_to_syncing()
    self.assertTrue(self.controller.reopen())
    self.assertEqual(self.controller.state, WorkerState.READY)
    self.assertTrue(self.controller.is_admission_open())

  async def test_stop_and_cancel_all(self):
    async def dummy_task():
      await asyncio.sleep(1.0)

    task1 = asyncio.create_task(dummy_task())
    task2 = asyncio.create_task(dummy_task())

    self.controller.try_admit(task1)
    self.controller.try_admit(task2)

    cancelled_tasks = self.controller.stop_and_cancel_all()
    self.assertCountEqual(cancelled_tasks, [task1, task2])
    await asyncio.sleep(0)
    self.assertTrue(task1.cancelled())
    self.assertTrue(task2.cancelled())

    self.assertEqual(self.controller.state, WorkerState.STOPPED)
    self.assertFalse(self.controller.is_admission_open())

    # Cannot reopen after stop
    self.assertFalse(self.controller.reopen())
    self.assertEqual(self.controller.state, WorkerState.STOPPED)


  async def test_track_and_drain(self):
    done = asyncio.Event()

    async def dummy_task():
      await done.wait()

    task = asyncio.create_task(dummy_task())
    self.controller.track(task)
    self.assertIn(task, self.controller.get_active_tasks())

    drain_future = asyncio.create_task(self.controller.drain(timeout_s=5.0))
    await asyncio.sleep(0.01)
    self.assertFalse(drain_future.done())

    done.set()
    await task
    await drain_future
    self.assertEqual(len(self.controller.get_active_tasks()), 0)

  def test_admission_closed_error_inheritance(self):
    err = traffic_controller.AdmissionClosedError("test message")
    self.assertIsInstance(err, RuntimeError)


if __name__ == "__main__":
  absltest.main()
