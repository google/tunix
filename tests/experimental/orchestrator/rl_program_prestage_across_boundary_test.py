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

"""Staging the next step's second micro-batch across the step boundary.

With the pipelined boundary, step k+1's first micro-batch is staged on the
trainer's own slot before step k's update (`stage_next_microbatch`). With the
intra-step queue as well (`stage_train_batches`, `max_staged_batches >= 2`),
the second micro-batch follows through that queue while step k's update,
metrics, checkpoint and weight sync run (`_prestage_across_boundary`), so step
k+1 starts with it already on the trainer instead of staging it once its first
train call is pending. The trainer then gets the same micro-batches in the same
order, on the same weights, as the base loop: these tests reuse the harness of
`rl_program_pipelined_boundary_test` and add a trainer double with
maxtext#5751's keyed queue (`stage_batch` / `fwd_bwd(None, stage_key=...)`).
"""

import dataclasses
import os
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import rl_program_pipelined_boundary_test as boundary_test
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import rl_program
from tunix.experimental.worker import trainer_worker


_INTEGRATED = dict(
    stage_before_update=True,
    stage_train_batches=True,
    max_staged_batches=2,
)


class _QueueTrainer(boundary_test._Trainer):
  """The harness trainer with a keyed staging queue (maxtext#5751's).

  `stage_batch` holds the payload under its key; `fwd_bwd(None,
  stage_key=...)` runs it. Logged as a `stage` event (which the equivalence
  checks ignore) with a fourth element naming the queue and the key, so the
  ordering checks below can find it.
  """

  def __init__(self, *args: Any, **kwargs: Any):
    super().__init__(*args, **kwargs)
    self._staged: dict[str, Any] = {}

  def stage_batch(self, payload: Any, key: Any = None) -> bool:
    self.log.append((
        "stage",
        boundary_test._fingerprint(payload),
        self.weights_version,
        "queue",
        key,
    ))
    self._staged[key] = payload
    return True

  def reset_staged(self) -> int:
    dropped = len(self._staged)
    self._staged.clear()
    return dropped

  def fwd_bwd(self, payload: Any, **kwargs: Any) -> None:
    key = kwargs.pop("stage_key", None)
    if payload is None:
      payload = self._staged.pop(key)
    elif key is not None:
      self._staged.pop(key, None)
    super().fwd_bwd(payload, **kwargs)


class _DecliningQueueTrainer(_QueueTrainer):
  """A trainer whose queue declines every stage (an uncompiled engine)."""

  def stage_batch(self, payload: Any, key: Any = None) -> bool:
    del payload, key
    return False


def _three_microbatches(sc: boundary_test._Scenario) -> boundary_test._Scenario:
  """`sc` with eight groups per step, which the harness packs into three
  micro-batches, so the second one does not close its step."""
  return dataclasses.replace(
      sc, full_batch_size=8, mini_batch_size=8, num_prompts=24, max_steps=3
  )


_SCENARIOS = {
    "prompt_batch": _three_microbatches(
        boundary_test._SCENARIOS["prompt_batch"]
    ),
    "arrival": _three_microbatches(boundary_test._SCENARIOS["arrival"]),
    "reference_kl": _three_microbatches(
        boundary_test._SCENARIOS["reference_kl"]
    ),
    "zero_costs": _three_microbatches(boundary_test._SCENARIOS["zero_costs"]),
    "groups_arrive_late": _three_microbatches(
        boundary_test._SCENARIOS["groups_arrive_late"]
    ),
    # Two micro-batches per step: the second closes its step, so it is never
    # staged across the boundary.
    "second_closes_the_step": boundary_test._SCENARIOS["prompt_batch"],
    # Steps 2-3 write scored checkpoints: those boundaries are not staged at
    # all, so nothing follows across them either.
    "scored_checkpoints": _three_microbatches(
        dataclasses.replace(
            boundary_test._SCENARIOS["checkpoints_from_step_3"], max_steps=4
        )
    ),
}
_SCENARIOS["scored_checkpoints"] = dataclasses.replace(
    _SCENARIOS["scored_checkpoints"], num_prompts=32
)


def _base(
    sc: boundary_test._Scenario,
    trainer_cls: type[boundary_test._Trainer] = _QueueTrainer,
) -> boundary_test._Run:
  return boundary_test._run(
      sc,
      pipelined=False,
      program_cls=boundary_test._BaseLoopProgram,
      trainer_cls=trainer_cls,
  )


def _integrated(
    sc: boundary_test._Scenario,
    trainer_cls: type[boundary_test._Trainer] = _QueueTrainer,
    program_cls: type[
        rl_program.StandardRLProgram
    ] = rl_program.StandardRLProgram,
    **program_kwargs: Any,
) -> boundary_test._Run:
  kwargs = dict(_INTEGRATED)
  kwargs.update(program_kwargs)
  return boundary_test._run(
      sc,
      pipelined=True,
      staged=True,
      program_cls=program_cls,
      trainer_cls=trainer_cls,
      program_kwargs=kwargs,
  )


def _steps(log: list[Any]) -> list[dict[str, Any]]:
  """Per step: the indices of its fwd_bwd events, and of its update."""
  steps: list[dict[str, Any]] = [{"fwd_bwd": [], "update": None}]
  for idx, event in enumerate(log):
    if event[0] == "fwd_bwd":
      steps[-1]["fwd_bwd"].append(idx)
    elif event[0] == "update":
      steps[-1]["update"] = idx
      steps.append({"fwd_bwd": [], "update": None})
  if not steps[-1]["fwd_bwd"]:
    steps.pop()
  return steps


def _queue_stages(log: list[Any]) -> dict[str, int]:
  """Fingerprint -> index of its stage through the queue."""
  return {
      e[1]: idx
      for idx, e in enumerate(log)
      if e[0] == "stage" and len(e) > 3 and e[3] == "queue"
  }


def _prestaged_boundaries(run: boundary_test._Run) -> list[bool]:
  """Per boundary k -> k+1: step k+1's second micro-batch was staged through
  the queue after step k's update and before step k+1's first train call."""
  log = run.log
  steps = _steps(log)
  stages = _queue_stages(log)
  out = []
  for k in range(len(steps) - 1):
    nxt = steps[k + 1]
    if len(nxt["fwd_bwd"]) < 2:
      out.append(False)
      continue
    second_fp = log[nxt["fwd_bwd"][1]][1]
    idx = stages.get(second_fp)
    out.append(
        idx is not None and steps[k]["fwd_bwd"][-1] < idx < nxt["fwd_bwd"][0]
    )
  return out


def _sync_follows_prestage(run: boundary_test._Run) -> bool:
  """Every stage across a boundary went out before that step's weight sync
  staged (so the sync was not held back for it)."""
  log = run.log
  steps = _steps(log)
  stages = _queue_stages(log)
  for k in range(len(steps) - 1):
    nxt = steps[k + 1]
    if len(nxt["fwd_bwd"]) < 2:
      continue
    idx = stages.get(log[nxt["fwd_bwd"][1]][1])
    if idx is None or idx < steps[k]["fwd_bwd"][-1]:
      continue
    sync = next(
        (
            i
            for i, e in enumerate(log)
            if e[0] == "prepare_weight_sync" and i > steps[k]["update"]
        ),
        None,
    )
    if sync is not None and sync < idx:
      return False
  return True


def _without_stage_metrics(records):
  """The intra-step queue logs `stage/fallbacks` each step; the base loop,
  which has no queue, does not. Everything else must match."""
  return [r for r in records if r[0] != "stage"]


def _assert_equivalent(
    test: absltest.TestCase, ref: boundary_test._Run, new: boundary_test._Run
) -> None:
  """`boundary_test._assert_equivalent`, but for the queue's own metric."""
  test.assertEqual(
      boundary_test._trainer_calls(ref.log),
      boundary_test._trainer_calls(new.log),
  )
  test.assertEqual(
      [(c["step"], c["checkpoint_path"]) for c in ref.checkpoints],
      [(c["step"], c["checkpoint_path"]) for c in new.checkpoints],
  )
  test.assertEqual(
      _without_stage_metrics(boundary_test._data_metrics(ref.metrics)),
      _without_stage_metrics(boundary_test._data_metrics(new.metrics)),
  )
  a, b = ref.last_step_result, new.last_step_result
  test.assertEqual(
      (a.step, a.policy_version, a.num_rollouts, a.num_microbatches),
      (b.step, b.policy_version, b.num_rollouts, b.num_microbatches),
  )
  test.assertEqual((a.reward_mean, a.reward_std), (b.reward_mean, b.reward_std))
  test.assertEqual(
      (a.advantage_mean, a.advantage_std), (b.advantage_mean, b.advantage_std)
  )


class PrestageEquivalenceTest(parameterized.TestCase):
  """The trainer cannot tell the integrated boundary from the base loop."""

  @parameterized.named_parameters((name, name) for name in _SCENARIOS)
  def test_matches_the_base_loop(self, name):
    sc = _SCENARIOS[name]
    ref = _base(sc)
    new = _integrated(sc)
    _assert_equivalent(self, ref, new)
    boundary_test._assert_weights_follow_updates(self, new)
    self.assertEmpty(new.trainer._staged, "staged micro-batches left unused")
    self.assertEqual(new.program._stage_fallbacks, 0)
    self.assertIsNone(new.program._pending_train)
    self.assertIsNone(new.program._prestage_task)


class PrestageOrderingTest(absltest.TestCase):
  """Where the second micro-batch's stage sits."""

  def test_second_microbatch_is_staged_across_every_boundary(self):
    # The harness's `_SENT_AHEAD` scenarios: the next step's first micro-batch
    # is packed by the time the last call goes out, so every boundary is
    # staged (with no device costs it is not, and nothing follows it either).
    for name in ("prompt_batch", "arrival", "reference_kl"):
      with self.subTest(name):
        run = _integrated(_SCENARIOS[name])
        self.assertEqual(
            boundary_test._microbatches_per_step(run), [3, 3, 3], run.log
        )
        self.assertEqual(_prestaged_boundaries(run), [True, True], run.log)
        self.assertTrue(_sync_follows_prestage(run), run.log)

  def test_the_stage_is_keyed_to_the_step_it_belongs_to(self):
    run = _integrated(_SCENARIOS["prompt_batch"])
    log = run.log
    steps = _steps(log)
    for k in range(len(steps) - 1):
      second_fp = log[steps[k + 1]["fwd_bwd"][1]][1]
      key = next(
          e[4]
          for e in log
          if e[0] == "stage" and len(e) > 3 and e[1] == second_fp
      )
      self.assertStartsWith(key, f"step_{k + 1}_stage_")

  def test_a_second_microbatch_that_closes_its_step_is_not_staged_ahead(self):
    run = _integrated(_SCENARIOS["second_closes_the_step"])
    self.assertEqual(boundary_test._microbatches_per_step(run), [2, 2, 2])
    self.assertEqual(_prestaged_boundaries(run), [False, False], run.log)
    _assert_equivalent(self, _base(_SCENARIOS["second_closes_the_step"]), run)

  def test_scored_boundaries_are_left_alone(self):
    sc = _SCENARIOS["scored_checkpoints"]
    run = _integrated(sc)
    steps = boundary_test._microbatches_per_step(run)
    self.assertGreaterEqual(len(steps), 3, run.log)
    prestaged = _prestaged_boundaries(run)
    for k, flag in enumerate(prestaged):
      with self.subTest(boundary=k):
        if boundary_test._writes_scored_checkpoint(sc, k):
          self.assertFalse(flag, run.log)
        else:
          self.assertTrue(flag, run.log)
    _assert_equivalent(self, _base(sc), run)

  def test_off_without_the_intra_step_queue(self):
    sc = _SCENARIOS["prompt_batch"]
    run = _integrated(sc, stage_train_batches=False)
    self.assertEqual(_queue_stages(run.log), {})
    self.assertEqual(_prestaged_boundaries(run), [False, False])
    _assert_equivalent(self, _base(sc), run)

  def test_off_with_a_single_slot_queue(self):
    sc = _SCENARIOS["prompt_batch"]
    run = _integrated(sc, max_staged_batches=1)
    self.assertEqual(_prestaged_boundaries(run), [False, False], run.log)
    _assert_equivalent(self, _base(sc), run)


class PrestageFallbackTest(absltest.TestCase):

  def test_a_declined_stage_goes_whole(self):
    sc = _SCENARIOS["prompt_batch"]
    ref = _base(sc, _DecliningQueueTrainer)
    run = _integrated(sc, _DecliningQueueTrainer)
    _assert_equivalent(self, ref, run)
    self.assertEqual(_queue_stages(run.log), {})
    fallbacks = sum(
        r[2] for r in run.metrics if r[0] == "stage" and r[1] == "fallbacks"
    )
    self.assertGreater(fallbacks, 0, run.metrics)
    self.assertEmpty(run.trainer._staged)

  def test_the_step_adopts_a_stage_still_in_flight(self):
    # A trainer whose queue stage takes longer than the boundary: the step
    # starts with the stage pending, adopts it, and waits for it before the
    # second micro-batch's train call. Same trainer calls either way.
    class _SlowQueueTrainer(_QueueTrainer):

      def stage_batch(self, payload: Any, key: Any = None) -> bool:
        import time  # pylint: disable=g-import-not-at-top

        staged = super().stage_batch(payload, key=key)  # logged at the call
        time.sleep(0.08)
        return staged

    sc = _SCENARIOS["prompt_batch"]
    ref = _base(sc, _SlowQueueTrainer)
    run = _integrated(sc, _SlowQueueTrainer)
    _assert_equivalent(self, ref, run)
    self.assertEqual(_prestaged_boundaries(run), [True, True], run.log)


class ExecutorMethodsTest(absltest.TestCase):

  def test_deferred_update_runs_off_the_loop_thread_with_staging(self):
    with mock.patch.dict(
        os.environ, {trainer_worker.STAGE_BATCH_ENV: "stage"}, clear=False
    ):
      worker = trainer_worker.TrainerWorker(
          lambda: boundary_test._Trainer([], boundary_test._Costs())
      )
      self.assertIn("apply_deferred_update", worker.remote_executor_methods)
    with mock.patch.dict(os.environ, {}, clear=False):
      os.environ.pop(trainer_worker.STAGE_BATCH_ENV, None)
      worker = trainer_worker.TrainerWorker(
          lambda: boundary_test._Trainer([], boundary_test._Costs())
      )
      self.assertNotIn("apply_deferred_update", worker.remote_executor_methods)


class _HeldProbe(rl_program.StandardRLProgram):
  """Records, at every take-ahead, the packed micro-batches the orchestrator
  still holds with their payload: the boundary's memory bound
  (`rl_program_pipelined_boundary_robustness_test.PrepackBoundTest`) is that
  this never exceeds one feed's output while the previous step finishes."""

  def __init__(self, *args: Any, **kwargs: Any):
    super().__init__(*args, **kwargs)
    self.held_at_take_ahead: list[int] = []

  async def _take_group(self, st, *, current_step, **kwargs):
    if current_step is None:
      held = sum(mb.payload is not None for mb in st.ready)
      if st.staged is not None:
        held += st.staged[0].payload is not None
      held += sum(mb.payload is not None for mb, _, _ in st.prestaged)
      self.held_at_take_ahead.append(held)
    return await super()._take_group(st, current_step=current_step, **kwargs)


class PrestageMemoryBoundTest(absltest.TestCase):

  def test_nothing_unshipped_is_held_while_the_second_microbatch_is_taken(
      self,
  ):
    # The first micro-batch's payload is dropped once it is staged, so taking
    # the second one ahead holds at most that one feed's output, as the plain
    # take-ahead does for the first.
    run = _integrated(_SCENARIOS["prompt_batch"], program_cls=_HeldProbe)
    self.assertNotEmpty(run.program.held_at_take_ahead)
    self.assertEqual(max(run.program.held_at_take_ahead), 0, run.log)
    self.assertEqual(_prestaged_boundaries(run), [True, True], run.log)


class PrestageWantedTest(absltest.TestCase):

  def test_needs_every_flag(self):
    for flags, wanted in (
        (dict(_INTEGRATED), True),
        (dict(_INTEGRATED, max_staged_batches=1), False),
        (dict(_INTEGRATED, stage_train_batches=False), False),
    ):
      with self.subTest(flags=flags):
        program = boundary_test._new_program(
            _SCENARIOS["prompt_batch"],
            pipelined=True,
            staged=True,
            program_cls=rl_program.StandardRLProgram,
            log=[],
            checkpoints=[],
            program_kwargs=flags,
        )
        self.assertEqual(program._prestage_wanted(), wanted)
    for pipelined, staged in ((False, False), (True, False)):
      with self.subTest(pipelined=pipelined, staged=staged):
        program = boundary_test._new_program(
            _SCENARIOS["prompt_batch"],
            pipelined=pipelined,
            staged=staged,
            program_cls=rl_program.StandardRLProgram,
            log=[],
            checkpoints=[],
            program_kwargs=dict(_INTEGRATED),
        )
        self.assertFalse(program._prestage_wanted())


if __name__ == "__main__":
  absltest.main()
