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

import threading
import time

from absl.testing import absltest
from tunix.dist_rl.worker import device_idle_tracker


class _FakeClock:
  """Thread-safe monotonic fake clock for deterministic tracker tests."""

  def __init__(self, initial: float = 0.0) -> None:
    self._cond = threading.Condition()
    self._now = initial
    self._observer_reads = 0

  def __call__(self) -> float:
    with self._cond:
      if threading.current_thread().name == "device_idle_tracker":
        self._observer_reads += 1
        self._cond.notify_all()
      return self._now

  def observer_reads(self) -> int:
    with self._cond:
      return self._observer_reads

  def wait_for_observer_reads(self, target: int) -> None:
    with self._cond:
      ok = self._cond.wait_for(
          lambda: self._observer_reads >= target, timeout=5.0
      )
      if not ok:
        raise TimeoutError(
            f"Timed out waiting for observer clock reads >= {target}"
        )

  def set(self, value: float) -> None:
    with self._cond:
      self._now = value

  def advance_to(self, value: float) -> None:
    with self._cond:
      if value > self._now:
        self._now = value


class _ControlledToken:
  """Deterministic device computation synchronized via `threading.Event`s."""

  def __init__(
      self,
      completion_time: float,
      clock: _FakeClock,
      *,
      block: bool = False,
      error: Exception | None = None,
  ) -> None:
    self._completion_time = completion_time
    self._clock = clock
    self._block = block
    self._error = error
    self.entered_wait = threading.Event()
    self.release = threading.Event()
    self.completed = threading.Event()

  def block_until_ready(self) -> "_ControlledToken":
    self.entered_wait.set()
    if self._block:
      self.release.wait()
    self._clock.advance_to(self._completion_time)
    self.completed.set()
    if self._error is not None:
      raise self._error
    return self


class DeviceIdleTrackerTest(absltest.TestCase):

  def setUp(self) -> None:
    super().setUp()
    self.clock = _FakeClock(initial=0.5)
    self.tracker = device_idle_tracker.DeviceIdleTracker(clock=self.clock)
    self.addCleanup(lambda: self.tracker.close(wait=False))

  def test_async_overlap_reports_zero_intra_step_starvation(self) -> None:
    self.clock.set(0.8)
    self.tracker.reset()

    self.clock.set(1.0)
    t_enter_0 = self.tracker.now()
    self.clock.set(1.002)
    tok_0 = _ControlledToken(1.500, self.clock, block=True)
    self.addCleanup(tok_0.release.set)
    self.tracker.record(t_enter_0, tok_0.block_until_ready)
    self.assertTrue(tok_0.entered_wait.wait(timeout=5.0))

    # mb_1 arrives at t=1.120 while device is still executing mb_0 until 1.500.
    self.clock.set(1.120)
    t_enter_1 = self.tracker.now()
    self.clock.set(1.123)
    tok_1 = _ControlledToken(1.800, self.clock)
    self.tracker.record(t_enter_1, tok_1.block_until_ready)

    tok_0.release.set()
    waits = self.tracker.flush()

    self.assertLen(waits, 2)
    self.assertAlmostEqual(waits[0], 0.2)
    self.assertAlmostEqual(waits[1], 0.0)

  def test_true_intra_step_starvation_accumulates_gap(self) -> None:
    self.clock.set(1.0)
    self.tracker.reset()

    reads_before = self.clock.observer_reads()
    self.clock.set(1.100)
    t_enter_0 = self.tracker.now()
    self.clock.set(1.102)
    self.tracker.record(
        t_enter_0,
        _ControlledToken(1.500, self.clock).block_until_ready,
    )
    self.clock.wait_for_observer_reads(reads_before + 1)

    tok_1_done = _ControlledToken(2.100, self.clock)
    self.clock.set(1.850)
    t_enter_1 = self.tracker.now()
    self.clock.set(1.853)
    self.tracker.record(t_enter_1, tok_1_done.block_until_ready)

    waits = self.tracker.flush()
    self.assertLen(waits, 2)
    self.assertAlmostEqual(waits[0], 0.100)
    self.assertAlmostEqual(waits[1], 0.350)

  def test_single_microbatch_and_per_update_reset(self) -> None:
    self.clock.set(2.0)
    self.tracker.reset()

    self.clock.set(2.25)
    t_enter_0 = self.tracker.now()
    self.clock.set(2.26)
    self.tracker.record(
        t_enter_0,
        _ControlledToken(2.60, self.clock).block_until_ready,
    )

    step1 = self.tracker.flush()
    self.assertLen(step1, 1)
    self.assertAlmostEqual(step1[0], 0.25)

    # Empty flush right after step 1 returns ().
    self.assertEqual(self.tracker.flush(), ())

    # Step 2 starts from boundary at 3.0 and does not include step 1's waits.
    self.clock.set(3.0)
    self.tracker.mark_busy_until_now()
    self.clock.set(3.05)
    t_enter_s2 = self.tracker.now()
    self.clock.set(3.06)
    self.tracker.record(
        t_enter_s2,
        _ControlledToken(3.40, self.clock).block_until_ready,
    )

    step2 = self.tracker.flush()
    self.assertLen(step2, 1)
    self.assertAlmostEqual(step2[0], 0.05)

  def test_reset_queued_behind_unresolved_token_clears_idle_times(
      self,
  ) -> None:
    self.clock.set(1.0)
    self.tracker.reset()

    self.clock.set(1.3)
    t_enter_0 = self.tracker.now()
    tok_0 = _ControlledToken(1.8, self.clock, block=True)
    self.addCleanup(tok_0.release.set)
    self.tracker.record(t_enter_0, tok_0.block_until_ready)
    self.assertTrue(tok_0.entered_wait.wait(timeout=5.0))

    reset_done = threading.Event()

    def _run_reset() -> None:
      self.clock.advance_to(2.0)
      self.tracker.reset()
      reset_done.set()

    reset_thread = threading.Thread(target=_run_reset)
    reset_thread.start()
    tok_0.release.set()
    reset_thread.join(timeout=5.0)
    self.assertTrue(reset_done.is_set())
    self.assertEqual(self.tracker.flush(), ())

  def test_observer_error_latches_short_circuits_and_reraises_on_flush(
      self,
  ) -> None:
    self.clock.set(1.0)
    self.tracker.reset()

    t_enter_0 = self.tracker.now()
    bad_tok = _ControlledToken(
        1.2, self.clock, error=RuntimeError("device poison")
    )
    self.tracker.record(t_enter_0, bad_tok.block_until_ready)

    # Subsequent microbatch in the broken step is short-circuited without
    # hanging even if its token would block.
    never_tok = _ControlledToken(9.9, self.clock, block=True)
    self.addCleanup(never_tok.release.set)
    self.tracker.record(self.tracker.now(), never_tok.block_until_ready)

    with self.assertRaisesWithLiteralMatch(RuntimeError, "device poison"):
      self.tracker.flush()
    self.assertFalse(never_tok.entered_wait.is_set())

    # Tracker recovers cleanly after reset().
    self.clock.set(3.0)
    self.tracker.reset()
    self.clock.set(3.1)
    t_enter_ok = self.tracker.now()
    self.tracker.record(
        t_enter_ok,
        _ControlledToken(3.5, self.clock).block_until_ready,
    )
    waits = self.tracker.flush()
    self.assertLen(waits, 1)
    self.assertAlmostEqual(waits[0], 0.1)

  def test_close_with_wait_joins_observer_thread(self) -> None:
    tok = _ControlledToken(1.5, self.clock)
    self.tracker.record(self.tracker.now(), tok.block_until_ready)
    self.tracker.close(wait=True)
    self.assertTrue(tok.completed.is_set())
    self.assertIsNotNone(self.tracker._thread)
    self.assertFalse(self.tracker._thread.is_alive())

  def test_close_without_wait_returns_immediately_and_fails_calls(self) -> None:
    wedged_tok = _ControlledToken(5.0, self.clock, block=True)
    self.addCleanup(wedged_tok.release.set)
    self.tracker.record(self.tracker.now(), wedged_tok.block_until_ready)
    self.assertTrue(wedged_tok.entered_wait.wait(timeout=5.0))

    flush_errors = []

    def _flush_behind_wedged_token() -> None:
      try:
        self.tracker.flush()
      except RuntimeError as exc:
        flush_errors.append(exc)

    flush_thread = threading.Thread(target=_flush_behind_wedged_token)
    flush_thread.start()
    # `close` only fails calls that are already queued.
    while self.tracker._tasks.empty():
      time.sleep(0.001)
    start = time.monotonic()
    self.tracker.close(wait=False)
    self.assertLess(time.monotonic() - start, 1.0)
    flush_thread.join(timeout=5.0)

    self.assertFalse(flush_thread.is_alive())
    self.assertLen(flush_errors, 1)
    self.assertIn("closed before", str(flush_errors[0]))
    with self.assertRaisesWithLiteralMatch(
        RuntimeError, "DeviceIdleTracker is closed."
    ):
      self.tracker.flush()
    with self.assertRaisesWithLiteralMatch(
        RuntimeError, "DeviceIdleTracker is closed."
    ):
      self.tracker.reset()


if __name__ == "__main__":
  absltest.main()
