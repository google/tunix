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

"""Failure paths, memory bound and hole rule of the pipelined step boundary.

These reuse the harness of `rl_program_pipelined_boundary_test` (the trainer
double behind a real `TrainerWorker`, and the frozen base loop) and cover what
its equivalence checks cannot see:

* How far the take-ahead packs (orchestrator memory). The trainer cannot tell
  a take-ahead that holds one micro-batch from one that holds a whole step,
  so a memory runaway passes every equivalence check. Checked in the
  harness's default packing and in a streaming regime (one row per
  micro-batch) where, as in production, a step's micro-batches come out as
  its groups are fed, so the take-ahead stops after one and restarts after
  the send-ahead.
* An optimizer update that fails on the device, against a trainer that
  confirms its updates (maxtext#5571), where a planned stage makes the fused
  last call return at dispatch and the stage meets the failure.
* A stage that fails. On the way to the trainer (transport), or in the
  reference policy's log-probs just before it, the micro-batch is sent whole
  instead and the run goes on; on the trainer, the run ends with that error.
* The `serving >= next` hole rule, fuzzed at the queue: invisible to a
  consumer that never reads ahead (flags off), visible to one that does.
* A stage left on a trainer by an orchestrator that died without `stop()`.

Each check comes with a control that fails on a deliberately wrong input.
"""

import asyncio
import collections
import contextlib
import logging
import random
import threading
import types
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import rl_program_pipelined_boundary_test as boundary_test
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import rl_program
from tunix.experimental.queue_manager import trajectory_queue_manager


def _worker_of(run: boundary_test._Run):
  """The TrainerWorker behind a harness run's trainer handle."""
  handle = run.program.engine._trainer_workers[datatypes.Role.ACTOR]
  return handle._server.bound_instance


@contextlib.contextmanager
def _warnings():
  """The messages logged at WARNING or above in the block (none is fine)."""
  messages: list[str] = []
  handler = logging.Handler(level=logging.WARNING)
  handler.emit = lambda record: messages.append(record.getMessage())
  root = logging.getLogger()
  root.addHandler(handler)
  try:
    yield messages
  finally:
    root.removeHandler(handler)


# ---------------------------------------------------------------------------
# The take-ahead packs at most one feed, for the next step only.
# ---------------------------------------------------------------------------


class _StreamingRows:
  """One 64-token row per micro-batch, so micro-batches stream out as groups
  are fed: production's regime (`pack_size` = fsdp x expert = 64 rows of
  65,536 tokens against 16-trajectory groups). With the harness's default two
  rows, a step's micro-batches mostly come out together at its last feed, so
  the take-ahead's stop-after-one-micro-batch and its restart after the
  send-ahead are rarely exercised."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.assembler._inner.batch_size = 1


class _PrepackProbe(rl_program.StandardRLProgram):
  """Records every group taken, ahead (`current_step=None`) or by its step."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    # (running step, state tag, micro-batches ready, staged) at each take-ahead.
    self.ahead: list[tuple[int, int, int, bool]] = []
    # State tag -> steps that took groups from it themselves.
    self.own: dict[int, set[int]] = collections.defaultdict(set)
    # Micro-batches ready right after each take-ahead (one feed's output).
    self.ready_after: list[int] = []
    # Take-aheads restarted after the step's first micro-batch was sent ahead.
    self.restarts = 0
    self._num_tags = 0

  async def _prepack(self, st):
    if st.num_microbatches:
      self.restarts += 1
    await super()._prepack(st)

  def _tag(self, st) -> int:
    tag = getattr(st, "_probe_tag", None)
    if tag is None:
      self._num_tags += 1
      tag = st._probe_tag = self._num_tags
    return tag

  async def _take_group(self, st, *, current_step):
    tag = self._tag(st)
    if current_step is None:
      self.ahead.append((self._step, tag, len(st.ready), st.staged is not None))
    else:
      self.own[tag].add(current_step)
    took = await super()._take_group(st, current_step=current_step)
    if current_step is None:
      self.ready_after.append(len(st.ready))
    return took


class _RunawayPrepack(_PrepackProbe):
  """WRONG: takes the whole next step ahead instead of one micro-batch.

  It feeds the same groups in the same order, so the trainer gets the same
  micro-batches; but the orchestrator holds every packed micro-batch of the
  next step through this step's tail and weight sync (each one dense routing
  data for the 397B), which is what the bound below forbids.
  """

  async def _prepack(self, st):
    while st.groups_consumed < self.full_batch_size:
      if not await self._take_group(st, current_step=None):
        return


class _StreamingBase(_StreamingRows, boundary_test._BaseLoopProgram):
  pass


class _StreamingProbe(_StreamingRows, _PrepackProbe):
  pass


class _StreamingRunaway(_StreamingRows, _RunawayPrepack):
  pass


_STREAMING_REFERENCES: dict[tuple[str, type], boundary_test._Run] = {}


def _streaming_reference(
    name: str, trainer_cls=boundary_test._Trainer
) -> boundary_test._Run:
  key = (name, trainer_cls)
  if key not in _STREAMING_REFERENCES:
    _STREAMING_REFERENCES[key] = boundary_test._run(
        boundary_test._SCENARIOS[name],
        pipelined=False,
        program_cls=_StreamingBase,
        trainer_cls=trainer_cls,
    )
  return _STREAMING_REFERENCES[key]


def _prepack_violations(program: _PrepackProbe) -> list[str]:
  """Breaches of the take-ahead's memory bound, as messages."""
  out = []
  for step, tag, ready, staged in program.ahead:
    if ready or staged:
      out.append(
          f"state {tag} taken ahead during step {step} while {ready}"
          f" micro-batch(es) were packed (staged={staged})"
      )
  running: dict[int, set[int]] = collections.defaultdict(set)
  for step, tag, _, _ in program.ahead:
    running[tag].add(step)
  for tag, steps in running.items():
    owners = program.own.get(tag, set())
    if len(owners) > 1:
      out.append(f"state {tag} was trained by several steps {sorted(owners)}")
    if owners:
      (owner,) = owners
      # Taken ahead while the step before it ran, or at its own start (the
      # restarted take-ahead it awaits); never two steps ahead.
      early = {s for s in steps if s not in (owner - 1, owner)}
      if early:
        out.append(
            f"state {tag} of step {owner} was taken ahead during steps"
            f" {sorted(early)}"
        )
  return out


_READY_AFTER: dict[str, list[int]] = collections.defaultdict(list)
_RESTARTS: collections.Counter[str] = collections.Counter()

_BOTH_MODES = [
    (f"{name}_{mode}", name, mode)
    for name in boundary_test._SCENARIOS
    for mode in ("unstaged", "staged")
]


class PrepackBoundTest(parameterized.TestCase):
  """The take-ahead holds at most one feed's micro-batches, one step ahead."""

  @parameterized.named_parameters(*_BOTH_MODES)
  def test_default_rows_bound_and_measure(self, name, mode):
    run = boundary_test._run(
        boundary_test._SCENARIOS[name],
        program_cls=_PrepackProbe,
        **boundary_test._MODES[mode],
    )
    self.assertEqual(_prepack_violations(run.program), [])
    if name in boundary_test._SENT_AHEAD:
      self.assertNotEmpty(run.program.ahead, "nothing was taken ahead")
    _READY_AFTER["default"].extend(run.program.ready_after)
    _RESTARTS["default"] += run.program.restarts
    # One feed's output in this harness: at most two micro-batches.
    self.assertLessEqual(max(run.program.ready_after, default=0), 2)

  @parameterized.named_parameters(*_BOTH_MODES)
  def test_streaming_take_ahead_is_bounded_and_trains_the_same(
      self, name, mode
  ):
    run = boundary_test._run(
        boundary_test._SCENARIOS[name],
        program_cls=_StreamingProbe,
        **boundary_test._MODES[mode],
    )
    self.assertEqual(_prepack_violations(run.program), [])
    boundary_test._assert_equivalent(self, _streaming_reference(name), run)
    boundary_test._assert_weights_follow_updates(self, run)
    if name in boundary_test._SENT_AHEAD:
      self.assertNotEmpty(run.program.ahead, "nothing was taken ahead")
    _READY_AFTER["streaming"].extend(run.program.ready_after)
    _RESTARTS["streaming"] += run.program.restarts

  @parameterized.named_parameters(
      (name, name) for name in boundary_test._SCENARIOS
  )
  def test_streaming_staged_against_a_token_trainer(self, name):
    run = boundary_test._run(
        boundary_test._SCENARIOS[name],
        program_cls=_StreamingProbe,
        pipelined=True,
        staged=True,
        trainer_cls=boundary_test._TokenTrainer,
    )
    self.assertEqual(_prepack_violations(run.program), [])
    boundary_test._assert_equivalent_but_metrics_fetch(
        self, _streaming_reference(name, boundary_test._TokenTrainer), run
    )
    boundary_test._staged_boundaries_confirm_the_update(self, run)
    _RESTARTS["streaming_token_staged"] += run.program.restarts

  def test_the_streaming_regime_runs_the_restart_path(self):
    # Non-vacuous: the take-ahead stops after one micro-batch with groups
    # left, and restarts once that micro-batch is sent ahead.
    restarts = 0
    for name in ("prompt_batch", "arrival", "reference_kl"):
      for mode in ("unstaged", "staged"):
        run = boundary_test._run(
            boundary_test._SCENARIOS[name],
            program_cls=_StreamingProbe,
            **boundary_test._MODES[mode],
        )
        restarts += run.program.restarts
    self.assertGreater(restarts, 0)

  def test_the_bound_catches_a_runaway_that_the_equivalence_check_misses(self):
    for name in ("prompt_batch", "arrival"):
      ref = _streaming_reference(name)
      run = boundary_test._run(
          boundary_test._SCENARIOS[name],
          pipelined=True,
          program_cls=_StreamingRunaway,
      )
      # The trainer cannot tell: same calls, inputs, weights and outputs.
      self.assertTrue(boundary_test._check_equivalent(ref, run), name)
      # The bound can.
      self.assertNotEmpty(_prepack_violations(run.program), name)


# ---------------------------------------------------------------------------
# An update that fails on the device, with wait_for_update=False.
# ---------------------------------------------------------------------------


class _BadToken:
  """`last_update_token` of an update that failed on the device."""

  def __init__(self, trainer, version):
    self._trainer = trainer
    self._version = version

  def block_until_ready(self):
    self._trainer.log.append(("token_wait_failed", self._version))
    raise RuntimeError(f"XLA: update {self._version} failed on device")


class _FailsSecondUpdate(boundary_test._TokenTrainer):
  """A token trainer whose second update fails on the device.

  Reading anything the failed update produced fails too: `get_metrics`
  returns its gradient norm (maxtext#5571's token), and a save or a weight
  sync reads the weights it wrote (as MaxText is assumed to do). Each such
  attempt is logged, so a path that saves after the failure shows up.
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.update_failed = False

  def update(self, **kwargs):
    version = super().update(**kwargs)
    if version == 2:
      self.update_failed = True
      self.last_update_token = _BadToken(self, version)
    return version

  def _read_after_failure(self, what):
    if self.update_failed:
      self.log.append(("after_failed_update", what))
      raise RuntimeError(f"XLA: update 2 failed on device (read by {what})")

  def get_metrics(self):
    self._read_after_failure("get_metrics")
    return super().get_metrics()

  def save_checkpoint(self, metadata, **kwargs):
    self._read_after_failure("save_checkpoint")
    return super().save_checkpoint(metadata, **kwargs)

  def prepare_weight_sync(self, sync_request=None, **kwargs):
    self._read_after_failure("prepare_weight_sync")
    return super().prepare_weight_sync(sync_request, **kwargs)


class _FailsSecondUpdateAtTheDevice(_FailsSecondUpdate):
  """The same, where the stage's device wait meets the failure first.

  MaxText's `wait_for_device` is its throttler's `block_until_ready()`,
  which waits on every queued output, the update's gradient norm included
  (AI-Hypercomputer/maxtext#5718, the MaxText half of this change). There
  the stage's device wait, not its token wait, is what raises.
  """

  def wait_for_device(self):
    if self.update_failed:
      self.log.append(("device_wait_failed", 2))
      raise RuntimeError(
          "XLA: update 2 failed on device (read by wait_for_device)"
      )
    return super().wait_for_device()


_FAILED_UPDATE_TRAINERS = {
    "token_wait": _FailsSecondUpdate,
    "device_wait": _FailsSecondUpdateAtTheDevice,
}

# Per (mode, trainer): whether a get_metrics met the failed update.
_GET_METRICS_AFTER_FAILURE: dict[str, bool] = {}

_FAILED_UPDATE_MODES = {
    "base": dict(pipelined=False, program_cls=boundary_test._BaseLoopProgram),
    "off": dict(pipelined=False),
    "unstaged": dict(pipelined=True),
    "staged": dict(pipelined=True, staged=True),
}

_FAILED_UPDATE_CASES = [
    (f"{mode}_{trainer}", mode, trainer)
    for mode in _FAILED_UPDATE_MODES
    for trainer in _FAILED_UPDATE_TRAINERS
]


def _run_failed_update(
    mode: str, trainer: str
) -> tuple[boundary_test._Run, list[BaseException]]:
  errors: list[BaseException] = []
  run = boundary_test._run(
      boundary_test._SCENARIOS["prompt_batch"],
      trainer_cls=_FAILED_UPDATE_TRAINERS[trainer],
      errors=errors,
      **_FAILED_UPDATE_MODES[mode],
  )
  return run, errors


def _stage_met_the_failure(run: boundary_test._Run) -> bool:
  """Whether the boundary after the failed update 2 was staged."""
  update_2 = max(
      i for i, e in enumerate(run.log) if e[0] == "update" and e[1] == 2
  )
  return "stage" in [e[0] for e in run.log[update_2:]]


class FailedUpdateTest(parameterized.TestCase):
  """A failed update is never saved or stamped, and is reported as the
  optimizer step, in every mode."""

  @parameterized.named_parameters(*_FAILED_UPDATE_CASES)
  def test_a_failed_update_is_never_checkpointed_or_stamped(
      self, mode, trainer
  ):
    run, errors = _run_failed_update(mode, trainer)
    self.assertLen(errors, 1, run.log)
    self.assertIn("XLA: update 2 failed on device", str(errors[0]))
    # Step 0 (optimizer step 1) saved; no save or weight sync was attempted
    # after the failed update (a get_metrics meeting it would be the barrier
    # doing its job, so it is not counted here).
    self.assertEqual([c["step"] for c in run.checkpoints], [1], run.log)
    for what in ("save_checkpoint", "prepare_weight_sync"):
      self.assertNotIn(("after_failed_update", what), run.log, run.log)
    _GET_METRICS_AFTER_FAILURE[f"{mode}_{trainer}"] = (
        "after_failed_update",
        "get_metrics",
    ) in run.log
    self.assertEqual(
        run.program.last_step_timestamp_ms, run.checkpoints[0]["timestamp_ms"]
    )
    self.assertEqual(_worker_of(run).state, datatypes.WorkerState.ERROR)
    self.assertIsNone(_worker_of(run)._staged_fwd_bwd)
    if mode == "staged":
      if _stage_met_the_failure(run):
        # The stage, not the fused call, met the failure: the call returned
        # at the update's dispatch and the stage waited for it.
        failed_wait = (
            "device_wait_failed"
            if trainer == "device_wait"
            else "token_wait_failed"
        )
        self.assertIn((failed_wait, 2), run.log)
      else:
        # Planned but nothing packed in time (a loaded machine): get_metrics
        # is the barrier again, and it is what meets the failure.
        self.assertIn(("after_failed_update", "get_metrics"), run.log)

  @parameterized.named_parameters(*_FAILED_UPDATE_CASES)
  def test_the_failure_is_attributed_to_the_optimizer_step(self, mode, trainer):
    # Off staging, google/tunix#2719's fused call says what failed, in the
    # error and in the worker's health report. A staged boundary must too:
    # there the stage meets the failure, and ends the run with it, rather
    # than falling back to get_metrics, whose error would not say it.
    run, errors = _run_failed_update(mode, trainer)
    if mode == "staged":
      self.assertTrue(
          _stage_met_the_failure(run),
          "precondition: the boundary after update 2 was staged (timing; the"
          " pipelined-boundary suite expects every prompt_batch boundary"
          " staged)",
      )
    head = str(errors[0]).splitlines()[0]
    self.assertIn("optimizer update() failed", head)
    last_error = _worker_of(run).heartbeat().last_error
    self.assertIn("optimizer update() failed", last_error)
    if mode == "staged":
      self.assertIn("stage_fwd_bwd", head)
      self.assertIn("XLA: update 2 failed on device", head)


# ---------------------------------------------------------------------------
# A stage that fails (checkpoint every step: val_start_step is None).
# ---------------------------------------------------------------------------


def _first_fingerprint_of_step_1() -> str:
  """Content hash of step 1's first micro-batch, from the base loop's run."""
  ref = boundary_test._reference("prompt_batch")
  return next(e[1] for e in ref.log if e[0] == "fwd_bwd" and e[2] == 1)


_TRANSPORT_FAILURE_MODES = {
    "base": dict(pipelined=False, program_cls=boundary_test._BaseLoopProgram),
    "unstaged": dict(pipelined=True),
    "staged": dict(pipelined=True, staged=True),
}


def _run_with_transport_failure(
    mode: str,
    program_cls: type[rl_program.StandardRLProgram] | None = None,
) -> tuple[boundary_test._Run, list[BaseException], dict[str, object]]:
  """Runs prompt_batch with the request carrying step 1's first micro-batch
  failing once in transport, before the trainer worker runs it."""
  target = _first_fingerprint_of_step_1()
  real_execute = boundary_test._RemoteHandle._execute
  fired: dict[str, object] = {"n": 0, "method": None}

  async def _execute(self, request):
    if request.method_name in ("fwd_bwd", "stage_fwd_bwd") and not fired["n"]:
      payload = getattr(request.kwargs.get("request"), "payload", None)
      if payload is not None and boundary_test._fingerprint(payload) == target:
        fired["n"] = 1
        fired["method"] = request.method_name
        raise ConnectionError("transport: stream reset before the call ran")
    return await real_execute(self, request)

  errors: list[BaseException] = []
  kwargs = dict(_TRANSPORT_FAILURE_MODES[mode])
  if program_cls is not None:
    kwargs["program_cls"] = program_cls
  with mock.patch.object(boundary_test._RemoteHandle, "_execute", _execute):
    run = boundary_test._run(
        boundary_test._SCENARIOS["prompt_batch"], errors=errors, **kwargs
    )
  return run, errors, fired


def _assert_fell_back_and_trained_the_same(
    test: absltest.TestCase, name: str, run: boundary_test._Run
) -> None:
  """The boundary went on unstaged, and the trainer cannot tell the run from
  the base loop's: the micro-batch went whole in step 1's first train call."""
  boundary_test._assert_equivalent(test, boundary_test._reference(name), run)
  boundary_test._assert_weights_follow_updates(test, run)
  test.assertNotIn("stage", [e[0] for e in run.log])
  # Sent ahead at every boundary, staged at none: off for the rest of the run.
  test.assertEqual(boundary_test._boundaries(run), [(True, False)] * 2, run.log)
  test.assertTrue(run.program._staging_latched_off)
  worker = _worker_of(run)
  test.assertEqual(worker.state, datatypes.WorkerState.READY)
  test.assertIsNone(worker._staged_fwd_bwd)


class FailedStageTransportTest(parameterized.TestCase):
  """What one failed send of step 1's first micro-batch costs."""

  @parameterized.parameters("base", "unstaged")
  def test_the_base_loop_and_the_unstaged_send_keep_step_0(self, mode):
    # Sent in a train call, the micro-batch fails after step 0's barrier,
    # checkpoint and bookkeeping.
    run, errors, fired = _run_with_transport_failure(mode)
    self.assertEqual(fired["n"], 1)
    self.assertEqual(fired["method"], "fwd_bwd")
    self.assertLen(errors, 1)
    self.assertIsInstance(errors[0], ConnectionError)
    self.assertEqual([c["step"] for c in run.checkpoints], [1])
    self.assertIn(("on_step_end", 0), run.log)
    self.assertEqual(run.program.step, 1)

  def test_a_stage_that_fails_on_the_way_falls_back_to_the_unstaged_send(
      self,
  ):
    with _warnings() as warnings:
      run, errors, fired = _run_with_transport_failure("staged")
    self.assertEqual(
        fired["method"],
        "stage_fwd_bwd",
        "precondition: boundary 0 was staged (timing; the pipelined-boundary"
        " suite expects every prompt_batch boundary staged)",
    )
    # The stage was optional: the run keeps step 0's barrier, checkpoint and
    # bookkeeping, sends the micro-batch whole, and (the failure being
    # one-shot) finishes.
    self.assertEqual(errors, [])
    _assert_fell_back_and_trained_the_same(self, "prompt_batch", run)
    self.assertTrue(
        any(
            "[pipeline] Staging failed" in m and "ConnectionError" in m
            for m in warnings
        ),
        warnings,
    )

  def test_without_the_fallback_the_failed_stage_costs_step_0(self):
    # Negative control: the same failure, treated as fatal (the code before
    # the fallback), loses step 0's checkpoint and bookkeeping, and the
    # checks above see it.
    with mock.patch.object(
        rl_program, "_raised_by_worker_method", lambda exc: True
    ):
      run, errors, fired = _run_with_transport_failure("staged")
    self.assertEqual(fired["method"], "stage_fwd_bwd")
    self.assertLen(errors, 1)
    self.assertEqual(run.checkpoints, [])
    self.assertNotIn(("on_step_end", 0), run.log)
    probe = absltest.TestCase()
    with self.assertRaises(AssertionError):
      _assert_fell_back_and_trained_the_same(probe, "prompt_batch", run)


class _MarksItsStage(rl_program.StandardRLProgram):
  """Marks the stage's own reference call, so a test can fail exactly it."""

  staging = threading.Event()

  async def _maybe_stage_next_microbatch(self, st, step_result):
    self.staging.set()
    try:
      return await super()._maybe_stage_next_microbatch(st, step_result)
    finally:
      self.staging.clear()


class FailedReferenceWhileStagingTest(absltest.TestCase):
  """The reference policy's log-probs, fetched for the stage, fail once.

  That call is not the trainer's: whether it failed on the way or in the
  reference worker itself, the trainer is untouched, so the micro-batch goes
  whole in step 1's first train call (which fetches the log-probs again), as
  without staging.
  """

  def setUp(self):
    super().setUp()
    _MarksItsStage.staging.clear()
    self.fired = 0

  def _one_shot(self) -> bool:
    if _MarksItsStage.staging.is_set() and not self.fired:
      self.fired += 1
      return True
    return False

  def _run(self) -> tuple[boundary_test._Run, list[BaseException]]:
    errors: list[BaseException] = []
    with _warnings() as warnings:
      run = boundary_test._run(
          boundary_test._SCENARIOS["reference_kl"],
          pipelined=True,
          staged=True,
          program_cls=_MarksItsStage,
          errors=errors,
      )
    self.assertEqual(
        self.fired,
        1,
        "precondition: boundary 0 was staged (timing; the pipelined-boundary"
        " suite expects every reference_kl boundary staged)",
    )
    self.warnings = warnings
    return run, errors

  def test_a_transport_failure_falls_back(self):
    real_execute = boundary_test._RemoteHandle._execute
    outer = self

    async def _execute(self, request):
      if (
          request.method_name == "per_token_logps"
          and isinstance(self._server.bound_instance, boundary_test._Reference)
          and outer._one_shot()
      ):
        raise ConnectionError("transport: reference stream reset")
      return await real_execute(self, request)

    with mock.patch.object(boundary_test._RemoteHandle, "_execute", _execute):
      run, errors = self._run()
    self.assertEqual(errors, [])
    _assert_fell_back_and_trained_the_same(self, "reference_kl", run)
    self.assertTrue(
        any("ConnectionError" in m for m in self.warnings), self.warnings
    )

  def test_a_failure_in_the_reference_worker_falls_back(self):
    # A remote error, like a trainer-side one, but from the reference worker.
    real_logps = boundary_test._Reference.per_token_logps
    outer = self

    def _per_token_logps(self, items, **kwargs):
      if outer._one_shot():
        raise RuntimeError("reference model: device lost")
      return real_logps(self, items, **kwargs)

    with mock.patch.object(
        boundary_test._Reference, "per_token_logps", _per_token_logps
    ):
      run, errors = self._run()
    self.assertEqual(errors, [])
    _assert_fell_back_and_trained_the_same(self, "reference_kl", run)
    self.assertTrue(
        any(
            "RemoteExecutionError" in m and "reference model: device lost" in m
            for m in self.warnings
        ),
        self.warnings,
    )


# ---------------------------------------------------------------------------
# The hole rule, fuzzed at the queue.
# ---------------------------------------------------------------------------


def _advance_over_holes_eq(self):
  """The hole rule before the pipelined boundary: serving == next only."""
  while (
      self._serving_batch_idx == self._next_batch_idx
      and self._next_batch_idx in self._sealed
      and self._admitted.get(self._next_batch_idx, 0) == 0
  ):
    hole = self._next_batch_idx
    self._drop_batch_state(hole)
    self._advance_to(hole + 1)


class _EqRuleQueue(trajectory_queue_manager.BatchOrderedQueueManager):
  """The queue with the hole rule as it was before (serving == next)."""

  _advance_over_holes = _advance_over_holes_eq


def _queue_scenario(seed: int):
  """Batches (some short, some holes) and a producer schedule."""
  rng = random.Random(seed)
  full = rng.choice([2, 3])
  num_batches = rng.randint(3, 7)
  per_batch = []
  for b in range(num_batches):
    n = full if rng.random() < 0.6 else rng.randint(0, full - 1)
    kinds = []
    for _ in range(n):
      r = rng.random()
      kinds.append(
          "valid" if r < 0.65 else "filtered" if r < 0.85 else "dropped"
      )
    per_batch.append(
        [("expect", b, n)] + [(k, b, j) for j, k in enumerate(kinds)]
    )
  window = rng.choice([1, 2, 3])
  events, active, nxt = [], [], 0
  while nxt < num_batches or active:
    while len(active) < window and nxt < num_batches:
      active.append(nxt)
      nxt += 1
    b = rng.choice(active)
    events.append(per_batch[b].pop(0))
    if not per_batch[b]:
      active.remove(b)
  producer_yields = [rng.randint(0, 3) for _ in events]
  consumer_yields = [rng.randint(0, 3) for _ in range(4 * len(events) + 8)]
  return full, events, producer_yields, consumer_yields


def _drive_queue(seed: int, queue_cls, *, read_ahead: bool):
  """Runs one producer/consumer schedule; returns (trace, steps trained)."""
  full, events, producer_yields, consumer_yields = _queue_scenario(seed)
  trace: list[tuple] = []
  q = queue_cls(
      full_batch_size=full,
      num_generations=1,
      filter_fn=lambda g: ([], list(g)) if g[0].drop else (list(g), []),
      on_cursor_advance=lambda idx: trace.append(("cursor", idx)),
  )
  cy = iter(consumer_yields)

  def record(*event):
    trace.append(event + (q.next_batch_idx, q.serving_batch_idx))

  def ids(result):
    return None if result is None else (result[0], result[1][0].prompt_id)

  async def produce():
    for (kind, b, x), y in zip(events, producer_yields):
      for _ in range(y):
        await asyncio.sleep(0)
      if kind == "expect":
        q.expect(b, x)
      elif kind == "dropped":
        q.dropped(b, 1)
      else:
        await q.put(
            types.SimpleNamespace(
                prompt_id=f"b{b}g{x}",
                metadata={"batch_idx": b},
                is_valid=True,
                policy_version=0,
                drop=kind == "filtered",
            )
        )
    await q.close()

  async def consume():
    steps = []
    ahead = None
    while True:
      first = None
      if ahead is not None:
        first = await ahead
        ahead = None
        record("ahead", ids(first))
      if first is None:
        first = await q.get_ordered_group(None)
        record("get", None, ids(first))
        if first is None:
          return steps
      b, group = first
      taken = [group[0].prompt_id]
      while len(taken) < full:
        r = await q.get_ordered_group(b)
        record("get", b, ids(r))
        if r is None:
          break
        taken.append(r[1][0].prompt_id)
      if read_ahead:
        # The pipelined boundary: read the next batch by index before this
        # one commits.
        ahead = asyncio.ensure_future(q.get_ordered_group(b + 1))
      for _ in range(next(cy, 0)):
        await asyncio.sleep(0)
      q.commit_batch(b)
      record("commit", b)
      steps.append((b, tuple(taken)))

  async def main():
    producer = asyncio.ensure_future(produce())
    steps = await consume()
    await producer
    return steps

  steps = asyncio.run(asyncio.wait_for(main(), timeout=10))
  return trace, steps


_SEEDS = range(300)


class HoleRuleIdentityFuzzTest(absltest.TestCase):

  def test_the_new_rule_is_invisible_without_reading_ahead(self):
    differ = [
        s
        for s in _SEEDS
        if _drive_queue(s, _EqRuleQueue, read_ahead=False)
        != _drive_queue(
            s,
            trajectory_queue_manager.BatchOrderedQueueManager,
            read_ahead=False,
        )
    ]
    self.assertEqual(differ, [])

  def test_the_fuzz_sees_the_rule_when_reading_ahead(self):
    # Negative control: the same fuzz does tell the two rules apart once the
    # consumer reads ahead, which is what the rule was changed for.
    differ = [
        s
        for s in _SEEDS
        if _drive_queue(s, _EqRuleQueue, read_ahead=True)
        != _drive_queue(
            s,
            trajectory_queue_manager.BatchOrderedQueueManager,
            read_ahead=True,
        )
    ]
    self.assertNotEmpty(differ)

  def test_reading_ahead_trains_the_same_batches_and_groups(self):
    queue_cls = trajectory_queue_manager.BatchOrderedQueueManager
    for s in _SEEDS:
      _, base_steps = _drive_queue(s, queue_cls, read_ahead=False)
      _, ahead_steps = _drive_queue(s, queue_cls, read_ahead=True)
      self.assertEqual(base_steps, ahead_steps, f"seed {s}")


# ---------------------------------------------------------------------------
# A stage left on a trainer by an orchestrator that died.
# ---------------------------------------------------------------------------


class _RestorableTrainer(boundary_test._Trainer):
  """The trainer double, plus the bring-up calls a successor makes."""

  def compile(self, dummy_data=None):
    del dummy_data

  def restore_checkpoint(self, **kwargs):
    del kwargs
    return {}


class StaleStageAfterRestartTest(absltest.TestCase):
  """An orchestrator that dies between a stage and the train call that runs
  it (OOM kill, preemption: no `stop()`) leaves the micro-batch held. Its
  successor brings the surviving trainer worker up again (`initialize`,
  `compile`) and restores the checkpoint, which must drop it."""

  def _crashed_worker(self, held: bool):
    worker = boundary_test._Worker(
        lambda: _RestorableTrainer([], boundary_test._Costs())
    )
    worker.initialize()
    if held:
      worker.stage_fwd_bwd(
          datatypes.TrainRequest(
              request_id="dead-orchestrator",
              payload=boundary_test._payload(1, logps=False),
              metadata={},
          )
      )
    return worker

  def _stage_new(self, worker):
    return worker.stage_fwd_bwd(
        datatypes.TrainRequest(
            request_id="successor",
            payload=boundary_test._payload(2, logps=False),
            metadata={},
        )
    )

  def test_a_held_stage_survives_a_bare_initialize(self):
    # Control: the stage really is held, and initialize() alone (a no-op on
    # a ready worker) does not drop it, so the successor's first stage would
    # be refused if nothing else did.
    worker = self._crashed_worker(held=True)
    worker.initialize()
    with self.assertRaisesRegex(
        RuntimeError, "already holds staged micro-batch 'dead-orchestrator'"
    ):
      self._stage_new(worker)

  def test_a_restored_worker_without_a_held_stage_stages(self):
    # Control: the successor's bring-up sequence is fine on its own.
    worker = self._crashed_worker(held=False)
    worker.initialize()
    worker.compile(None)
    worker.restore_checkpoint()
    self.assertTrue(self._stage_new(worker).metadata["staged"])

  def test_a_successor_can_stage_after_a_crash(self):
    worker = self._crashed_worker(held=True)
    with _warnings() as warnings:
      worker.initialize()
      worker.compile(None)
      worker.restore_checkpoint()
    self.assertIsNone(worker._staged_fwd_bwd)
    self.assertTrue(any("dead-orchestrator" in m for m in warnings), warnings)
    self.assertTrue(self._stage_new(worker).metadata["staged"])


if __name__ == "__main__":
  absltest.main()
