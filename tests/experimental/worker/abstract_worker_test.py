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

"""Tests for worker lifecycle transitions across all worker implementations."""

import asyncio
import contextlib
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from tunix.experimental.common import datatypes
from tunix.experimental.common import test_utils as mocks
from tunix.experimental.worker import inference_worker
from tunix.experimental.worker import rollout_worker
from tunix.experimental.worker import trainer_worker

WorkerState = datatypes.WorkerState


class DummyScoringCore:

  def get_ref_per_token_logps(self, *args, **kwargs):
    pass

  def get_rewards(self, *args, **kwargs):
    pass


class DummyTrainer:

  def compile(self, *args, **kwargs):
    pass

  def close(self):
    pass


class AbstractWorkerTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="inference_worker",
          worker_cls=inference_worker.InferenceWorker,
          module_path=(
              "tunix.experimental.worker.inference_worker.datatypes.Response"
          ),
          kwargs=dict(
              core=DummyScoringCore(),
              worker_id="w1",
              pad_id=0,
              eos_id=1,
          ),
      ),
      dict(
          testcase_name="rollout_worker",
          worker_cls=rollout_worker.RolloutWorker,
          module_path=(
              "tunix.experimental.worker.rollout_worker.datatypes.Response"
          ),
          kwargs=dict(
              worker_id="w2",
              sampler=mocks.MockBaseSamplerImpl(sampler_name="mock_sampler"),
              tokenizer=mocks.MockTokenizer(),
              chat_parser=mocks.MockChatParser(),
          ),
      ),
      dict(
          testcase_name="trainer_worker",
          worker_cls=trainer_worker.TrainerWorker,
          module_path=(
              "tunix.experimental.worker.trainer_worker.datatypes.Response"
          ),
          kwargs=dict(trainer_factory=DummyTrainer, worker_id="w3"),
      ),
  )
  def test_lifecycle_state_transitions(self, worker_cls, module_path, kwargs):
    worker = worker_cls(**kwargs)
    self.assertEqual(worker.state, WorkerState.PENDING)

    with mock.patch(module_path) as mock_response:
      # Hook into the Response() creation inside the try-block to check state
      def verify_initializing(*args, **kwargs):
        del args, kwargs
        self.assertEqual(worker.state, WorkerState.INITIALIZING)
        return mock.DEFAULT

      mock_response.side_effect = verify_initializing
      worker.initialize()
      self.assertEqual(worker.state, WorkerState.READY)

      def verify_compiling(*args, **kwargs):
        del args, kwargs
        self.assertEqual(worker.state, WorkerState.COMPILING)
        return mock.DEFAULT

      mock_response.side_effect = verify_compiling
      worker.compile(None)
      self.assertEqual(worker.state, WorkerState.READY)

    worker.stop()
    self.assertEqual(worker.state, WorkerState.STOPPED)

  def test_execution_context_wraps_public_methods_and_is_reentrant(self):
    call_log = []

    class CustomContext:

      def __enter__(self):
        call_log.append("enter")
        return self

      def __exit__(self, *args):
        call_log.append("exit")

    ctx = CustomContext()
    worker = trainer_worker.TrainerWorker(
        trainer_factory=DummyTrainer,
        worker_id="w_ctx",
        execution_context=ctx,
    )
    # Trainer instantiation in __init__ enters and exits once.
    self.assertEqual(call_log, ["enter", "exit"])
    call_log.clear()

    # Calling compile() while PENDING internally invokes self.initialize().
    # Because execution_context() is re-entrant per thread, CustomContext
    # should only be entered and exited once for the outer compile() call.
    worker.compile(None)
    self.assertEqual(call_log, ["enter", "exit"])

  def test_execution_context_zero_arg_callable_factory_and_reentrancy(self):
    factory_calls = []
    enter_exit_log = []

    class SingleUseContext:

      def __enter__(self):
        enter_exit_log.append("enter")
        return "ctx_val"

      def __exit__(self, *args):
        enter_exit_log.append("exit")

    def make_context():
      factory_calls.append(1)
      return SingleUseContext()

    worker = trainer_worker.TrainerWorker(
        trainer_factory=DummyTrainer,
        worker_id="w_factory",
        execution_context=make_context,
    )
    self.assertLen(factory_calls, 1)
    self.assertEqual(enter_exit_log, ["enter", "exit"])
    factory_calls.clear()
    enter_exit_log.clear()

    # Verify re-entrant entry only invokes factory once.
    with worker.execution_context():
      with worker.execution_context():
        pass
    self.assertLen(factory_calls, 1)
    self.assertEqual(enter_exit_log, ["enter", "exit"])

  def test_execution_context_invalid_type_raises_type_error(self):
    with self.assertRaisesRegex(TypeError, "execution_context must be None"):
      trainer_worker.TrainerWorker(
          trainer_factory=DummyTrainer,
          worker_id="w_bad",
          execution_context=12345,
      )

  def test_execution_context_asyncio_task_isolation(self):
    events = []

    class AsyncTrackingContext:

      def __enter__(self):
        events.append("enter")
        return self

      def __exit__(self, *args):
        events.append("exit")

    class DummyAsyncWorker(trainer_worker.TrainerWorker):

      async def step(self, name: str, delay: float):
        events.append(f"{name}_start")
        await asyncio.sleep(delay)
        events.append(f"{name}_end")

    worker = DummyAsyncWorker(
        trainer_factory=DummyTrainer,
        worker_id="w_async",
        execution_context=AsyncTrackingContext,
    )
    # Clear the enter/exit from __init__ trainer_factory
    events.clear()

    async def run_concurrent():
      await asyncio.gather(
          worker.step("task1", 0.02),
          worker.step("task2", 0.01),
      )

    asyncio.run(run_concurrent())
    # Each task must enter and exit its own context instance despite overlapping
    # on the same event loop thread.
    self.assertEqual(events.count("enter"), 2)
    self.assertEqual(events.count("exit"), 2)

  def test_execution_context_sequence_of_multiple_contexts(self):
    log = []

    class NamedContext:

      def __init__(self, name: str):
        self.name = name

      def __enter__(self):
        log.append(f"enter_{self.name}")
        return f"val_{self.name}"

      def __exit__(self, *args):
        log.append(f"exit_{self.name}")

    expected_lifo_log = [
        "enter_Mesh",
        "enter_TransferGuard",
        "exit_TransferGuard",
        "exit_Mesh",
    ]
    # Pass a sequence mixing a reusable context manager and a zero-arg callable
    worker = trainer_worker.TrainerWorker(
        trainer_factory=DummyTrainer,
        worker_id="w_multi_seq",
        execution_context=[
            NamedContext("Mesh"),
            lambda: NamedContext("TransferGuard"),
        ],
    )
    # __init__ enters both in order and exits in LIFO order
    self.assertEqual(log, expected_lifo_log)
    log.clear()

    # Verify re-entrant entry only enters/exits the sequence once.
    with worker.execution_context():
      with worker.execution_context():
        pass

    self.assertEqual(log, expected_lifo_log)
    log.clear()

    # Calling compile() (which internally calls initialize()) enters/exits once
    worker.compile(None)
    self.assertEqual(log, expected_lifo_log)

  def test_execution_context_composite_contextmanager_callable(self):
    log = []

    @contextlib.contextmanager
    def composite_context():
      log.append("enter_1")
      log.append("enter_2")
      try:
        yield ("c1", "c2")
      finally:
        log.append("exit_2")
        log.append("exit_1")

    worker = trainer_worker.TrainerWorker(
        trainer_factory=DummyTrainer,
        worker_id="w_composite",
        execution_context=composite_context,
    )
    self.assertEqual(log, ["enter_1", "enter_2", "exit_2", "exit_1"])
    log.clear()

    worker.compile(None)
    self.assertEqual(log, ["enter_1", "enter_2", "exit_2", "exit_1"])


if __name__ == "__main__":
  absltest.main()
