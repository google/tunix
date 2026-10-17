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

"""Staging the next step's first micro-batch before the step's update.

`stage_next_microbatch` sends step k+1's first micro-batch to the trainer
while the device still computes step k's last one. By default
(`stage_before_update`), when the trainer can defer a step's update, the stage
goes between step k's last train call, sent with its update deferred, and that
update. The hook it replaces, right after the last call, cannot hide anything
where the trainer's `update()` waits until the last micro-batch has finished
before it dispatches: MaxText with `retry_when_tokens_dropped` (dropless
replay, maxtext#5570, and the final dropless micro-batch of maxtext#5673)
reads each micro-batch's overflow flag on the host, the previous one's in
`fwd_bwd` and the last one's in `update()`. The fused last call then returns
only once the device has nothing left but the update, and the transfer finds
it idle. Where an idle device's wait reads ~0, as with the trainer doubles
here, the latch then switches staging off at the first boundary.

These tests reuse the harness of `rl_program_pipelined_boundary_test` (the
trainer double behind a real `TrainerWorker`, and the frozen base loop),
which tests the hook after the last call, and add trainer doubles with that
host blocking (`_ReplayTrainer`) and with real replays: a micro-batch that
overflows is rerun, dropless, when its flag is read, and the rerun's
gradients replace its own (`_LastOverflowsTrainer`, `_FirstOverflowsTrainer`).

Each check comes with a control that fails on a deliberately wrong input:
the hook after the last call under the replay double (it latches), a latch
that never trips, a rerun that reads the staged micro-batch, and programs or
workers that run the update before the stage.

Numbers in "units" are seconds of the real system scaled by `U`.
"""

import asyncio
import collections
import contextlib
import dataclasses
import logging
import time
import types
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import rl_program_pipelined_boundary_test as boundary_test
from tunix.experimental.common import datatypes
from tunix.experimental.examples.deepswe_dist import run_deepswe_dist
from tunix.experimental.orchestrator import rl_program


_SBU = {"stage_before_update": True}
LATCH_S = rl_program.StandardRLProgram._STAGING_HIDDEN_WAIT_S

# One production second, in test seconds: larger than the harness's 0.02, so a
# loaded machine does not turn a hidden stage into an idle device.
U = 0.05


def _run(sc: boundary_test._Scenario, **kwargs: Any) -> boundary_test._Run:
  """A harness run with the default hook: stage before the update."""
  program_kwargs = dict(_SBU)
  program_kwargs.update(kwargs.pop("program_kwargs", None) or {})
  return boundary_test._run(sc, program_kwargs=program_kwargs, **kwargs)


def _base(
    sc: boundary_test._Scenario, trainer_cls: type[boundary_test._Trainer]
) -> boundary_test._Run:
  """The frozen base loop's run of `sc` against `trainer_cls`."""
  return boundary_test._run(
      sc,
      pipelined=False,
      program_cls=boundary_test._BaseLoopProgram,
      trainer_cls=trainer_cls,
  )


def _worker_of(run: boundary_test._Run):
  handle = run.program.engine._trainer_workers[datatypes.Role.ACTOR]
  return handle._server.bound_instance


@contextlib.contextmanager
def _warnings():
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
# Trainer doubles with MaxText's dropless replay
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _Check:
  """A micro-batch whose overflow flag is not read yet (maxtext
  `_PendingOverflowCheck`)."""

  fingerprint: str
  ends_at: float
  index: int  # into the step's accumulated gradients
  overflows: bool
  weights_version: int


class _ReplayTrainer(boundary_test._Trainer):
  """The harness trainer with `retry_when_tokens_dropped`'s host blocking.

  Like maxtext 853d655's engine: `fwd_bwd(i)` dispatches micro-batch i, then
  reads micro-batch i-1's overflow flag on the host (`:2598` ->
  `_resolve_overflow_check` `:2694`), so it returns once i-1 has finished;
  `update()` reads the last micro-batch's flag before it dispatches the
  update (`:2813-2814`). A micro-batch that overflowed is rerun when its flag
  is read, after whatever was dispatched since, and the rerun's gradients and
  loss replace its own (`:2700-2712`; the capped run accumulated zeros). A
  micro-batch's loss is recorded when its flag is read. Here none overflows;
  subclasses pick which do (`_overflows`).
  """

  def __init__(self, *args: Any, **kwargs: Any):
    super().__init__(*args, **kwargs)
    self._pending: _Check | None = None
    self._micro = 0
    self.reruns: list[str] = []
    # (where the host was held, seconds), per flag read.
    self.blocked: list[tuple[str, float]] = []

  def _overflows(self, fingerprint: str, micro: int, final: bool) -> bool:
    del fingerprint, micro, final
    return False

  def _rerun_input(self, check: _Check) -> str:
    """What the rerun computes on: the overflowed micro-batch's own inputs."""
    return check.fingerprint

  def fwd_bwd(self, payload: Any, **kwargs: Any) -> None:
    fingerprint = boundary_test._fingerprint(payload)
    self._dispatch("fwd_bwd", fingerprint, self.costs.device_s)
    self.log.append(("fwd_bwd", fingerprint, self.weights_version))
    self._accumulated.append(fingerprint)
    check = _Check(
        fingerprint=fingerprint,
        ends_at=self._device_free_at,
        index=len(self._accumulated) - 1,
        overflows=self._overflows(
            fingerprint, self._micro, bool(kwargs.get("apply_optimizer"))
        ),
        weights_version=self.weights_version,
    )
    self._micro += 1
    previous, self._pending = self._pending, check
    if previous is not None:
      self._resolve(previous, "fwd_bwd")

  def _resolve(self, check: _Check, where: str) -> None:
    start = time.monotonic()
    self._wait_until(check.ends_at)
    self.blocked.append((where, time.monotonic() - start))
    if check.overflows:
      source = self._rerun_input(check)
      self._dispatch("rerun", source, self.costs.device_s)
      self.log.append(("rerun", source, check.weights_version))
      self.reruns.append(source)
      self._accumulated[check.index] = f"{source}/dropless"
      self._step_buffer().append(
          boundary_test._loss(f"{source}/dropless", check.weights_version)
      )
    else:
      self._step_buffer().append(
          boundary_test._loss(check.fingerprint, check.weights_version)
      )

  def update(self, **kwargs: Any) -> int:
    if self._pending is not None:
      check, self._pending = self._pending, None
      self._resolve(check, "update")
    self._micro = 0
    return super().update(**kwargs)


class _ReplayTokenTrainer(_ReplayTrainer, boundary_test._TokenTrainer):
  """`_ReplayTrainer` that also confirms its updates (maxtext#5571's
  `last_update_token`): the fused-call + deferred-fetch regime."""


class _LastOverflowsTrainer(_ReplayTrainer):
  """Every step's last micro-batch overflows: its rerun runs in `update()`,
  after the stage that went before the update."""

  def _overflows(self, fingerprint, micro, final):
    del fingerprint, micro
    return final


class _LastOverflowsTokenTrainer(
    _LastOverflowsTrainer, boundary_test._TokenTrainer
):
  pass


class _FirstOverflowsTrainer(_ReplayTrainer):
  """Every step's first micro-batch overflows: the staged one, rerun from the
  input it was staged as."""

  def _overflows(self, fingerprint, micro, final):
    del fingerprint, final
    return micro == 0


class _EveryOverflowsTrainer(_ReplayTrainer):

  def _overflows(self, fingerprint, micro, final):
    del fingerprint, micro, final
    return True


class _RerunReadsTheStagedTrainer(_LastOverflowsTrainer):
  """WRONG: a rerun computes on the micro-batch staged since, not its own.

  The hazard staging before the update adds: a replay of step k's last
  micro-batch runs while step k+1's first is already on the trainer.
  """

  def __init__(self, *args: Any, **kwargs: Any):
    super().__init__(*args, **kwargs)
    self._held: str | None = None

  def prepare_fwd_bwd_input(self, payload: Any) -> Any:
    self._held = boundary_test._fingerprint(payload)
    return super().prepare_fwd_bwd_input(payload)

  def fwd_bwd(self, payload: Any, **kwargs: Any) -> None:
    if self._held == boundary_test._fingerprint(payload):
      self._held = None
    super().fwd_bwd(payload, **kwargs)

  def _rerun_input(self, check):
    return self._held if self._held is not None else check.fingerprint


# ---------------------------------------------------------------------------
# Programs: wrong on purpose, or observing
# ---------------------------------------------------------------------------


class _AsBuilt(rl_program.StandardRLProgram):
  """The hook after the last call (`--no-stage_before_update`)."""

  def __init__(self, *args: Any, **kwargs: Any):
    kwargs["stage_before_update"] = False
    super().__init__(*args, **kwargs)


class _NeverLatches(rl_program.StandardRLProgram):

  def _staging_hidden(self, device_wait_s):
    return True


class _UpdatesBeforeStaging(rl_program.StandardRLProgram):
  """WRONG: runs the deferred update first, then stages."""

  async def _stage_next_before_update(self, st, step_result):
    assert self.engine is not None
    update = await self.engine.apply_deferred_update(role=datatypes.Role.ACTOR)
    staged = await self._send_stage(st) if self._stage_ok(st) else None
    if staged is not None and not self._staging_hidden(
        update.get("device_wait_s")
    ):
      self._staging_latched_off = True
    return {**step_result, **update}

  def _stage_ok(self, st):
    return (
        not self._staging_latched_off
        and self._intake_done()
        and st.staged is None
        and st.ready
        and not st.ready[0].is_final_batch
    )


class _RecordingWaits(rl_program.StandardRLProgram):
  """Records what the latch reads at each staged boundary."""

  def __init__(self, *args: Any, **kwargs: Any):
    super().__init__(*args, **kwargs)
    self.seen: list[float | None] = []

  def _staging_hidden(self, device_wait_s):
    self.seen.append(device_wait_s)
    return super()._staging_hidden(device_wait_s)


# ---------------------------------------------------------------------------
# Readouts
# ---------------------------------------------------------------------------

# Trainer-side events that move device state (or end a step) between a stage
# and the update it must precede.
_STEP_EVENTS = (
    "fwd_bwd",
    "update",
    "get_metrics",
    "save",
    "prepare_weight_sync",
)


def _stages(run: boundary_test._Run) -> int:
  return sum(1 for e in run.log if e[0] == "stage")


def _stage_positions(run: boundary_test._Run) -> list[str]:
  """Per stage: 'before_update' if the next step event after it is an update
  (a rerun may come in between), else what came next."""
  out = []
  log = run.log
  for i, event in enumerate(log):
    if event[0] != "stage":
      continue
    nxt = next((e[0] for e in log[i + 1 :] if e[0] in _STEP_EVENTS), None)
    out.append("before_update" if nxt == "update" else str(nxt))
  return out


def _staged_microbatches_trained_once(test, run: boundary_test._Run) -> None:
  """Each staged micro-batch is trained exactly once, after the update that
  followed its stage, and never sent again."""
  log = run.log
  for i, event in enumerate(log):
    if event[0] != "stage":
      continue
    fingerprint = event[1]
    trained = [
        j
        for j, e in enumerate(log)
        if e[0] == "fwd_bwd" and e[1] == fingerprint
    ]
    test.assertLen(trained, 1, (fingerprint, log))
    updates = [j for j in range(i, trained[0]) if log[j][0] == "update"]
    test.assertLen(updates, 1, (fingerprint, log))


def _boundary_gaps(
    trainer_cls: type[boundary_test._Trainer],
    *,
    pipelined: bool = True,
    staged: bool = True,
    program_cls: type[
        rl_program.StandardRLProgram
    ] = rl_program.StandardRLProgram,
    before_update: bool = True,
) -> tuple[list[float], boundary_test._Run]:
  """The harness's one-boundary timing scenario at `U`, gaps in units."""
  sc = boundary_test._Scenario(
      order="prompt_batch",
      num_prompts=12,
      full_batch_size=6,
      mini_batch_size=6,
      max_steps=2,
      costs=boundary_test._Costs(
          device_s=boundary_test.T_DEV * U,
          update_s=boundary_test.T_UPD * U,
          stage_s=boundary_test.T_STAGE * U,
          host_s=boundary_test.T_HOST * U,
          metrics_s=boundary_test.T_METRICS * U,
      ),
      dest_s=boundary_test.T_DEST * U,
      round_s=0.0,
      pack_s=boundary_test.T_PACK * U,
      tail_s=boundary_test.T_TAIL * U,
  )
  with mock.patch.object(rl_program, "_release_host_memory", lambda: None):
    run = boundary_test._run(
        sc,
        pipelined=pipelined,
        staged=staged,
        trainer_cls=trainer_cls,
        program_cls=program_cls,
        program_kwargs={"stage_before_update": before_update},
    )
  gaps = []
  device = run.trainer.device
  for i, (kind, _, _, end) in enumerate(device):
    if kind == "update" and i + 1 < len(device):
      gaps.append(round((device[i + 1][2] - end) / U, 2))
  return gaps, run


def _costs(*, device, update, stage, host, metrics) -> boundary_test._Costs:
  return boundary_test._Costs(
      device_s=device * U,
      update_s=update * U,
      stage_s=stage * U,
      host_s=host * U,
      metrics_s=metrics * U,
  )


# The submission run's trainer costs (the harness's `_submission_run_costs`),
# in production seconds.
SUBMISSION_RUN = dict(
    device=15.11, update=1.0, stage=4.45, host=9.53, metrics=0.5
)
# Another 397B trainer run's costs, from its TRAINER_STEP_TIMING lines:
# D 13.396 s, update 1.59 s, prepare_weight_sync 4.60 s, gRPC leg 11.18 s,
# get_metrics and coordinator ~0.9 s.
OTHER_RUN = dict(device=13.4, update=1.6, stage=4.6, host=11.2, metrics=0.9)


def _latch_scenario(
    costs: dict[str, float], **kwargs: Any
) -> boundary_test._Scenario:
  """12 groups per step (several train calls per step, as in the submission
  run), at U."""
  base = dict(dest_s=1.64 * U, pack_s=2.1 * U, tail_s=2.7 * U)
  base.update(kwargs)
  return boundary_test._latch_scenario(_costs(**costs), **base)


def _waits_units(run: boundary_test._Run) -> list[float]:
  return [round(w / U, 2) for w in run.trainer.device_waits]


def _quiet(fn, *args, **kwargs):
  with mock.patch.object(rl_program, "_release_host_memory", lambda: None):
    return fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# The replay doubles behave as MaxText does (two-sided)
# ---------------------------------------------------------------------------


class ReplayDoubleTest(absltest.TestCase):
  """The doubles block where MaxText blocks, and replay what they say."""

  def test_update_waits_for_the_last_micro_batch_only_under_replay(self):
    sc = _latch_scenario(SUBMISSION_RUN)
    replay = _quiet(_base, sc, _ReplayTrainer)
    updates = [
        s / U for where, s in replay.trainer.blocked if where == "update"
    ]
    msg = f"update held (units): {[round(s, 2) for s in updates]}"
    self.assertLen(updates, sc.max_steps, msg)
    # The update waited for most of a micro-batch of device time, every step.
    for held in updates:
      self.assertGreater(held, 0.5 * SUBMISSION_RUN["device"], msg)
    # Negative control: the harness trainer has no such wait.
    plain = _quiet(_base, sc, boundary_test._Trainer)
    self.assertFalse(hasattr(plain.trainer, "blocked"))

  def test_reruns_change_what_is_trained(self):
    sc = boundary_test._SCENARIOS["prompt_batch"]
    plain = _base(sc, _ReplayTrainer)
    last = _base(sc, _LastOverflowsTrainer)
    first = _base(sc, _FirstOverflowsTrainer)
    self.assertEmpty(plain.trainer.reruns)
    steps = len(boundary_test._microbatches_per_step(last))
    self.assertLen(last.trainer.reruns, steps)
    self.assertLen(first.trainer.reruns, steps)
    # A rerun replaces the capped run's gradients: a different update, and a
    # different logged loss, from the same micro-batches.
    self.assertFalse(boundary_test._check_equivalent(plain, last))
    self.assertFalse(boundary_test._check_equivalent(plain, first))
    updates = lambda run: [e for e in run.log if e[0] == "update"]
    self.assertNotEqual(updates(plain), updates(last))


# ---------------------------------------------------------------------------
# The trainer cannot tell staging before the update from the base loop
# ---------------------------------------------------------------------------

_TRAINERS = {
    "plain": boundary_test._Trainer,
    "token": boundary_test._TokenTrainer,
    "replay": _ReplayTrainer,
    "replay_token": _ReplayTokenTrainer,
    "last_overflows": _LastOverflowsTrainer,
    "last_overflows_token": _LastOverflowsTokenTrainer,
    "first_overflows": _FirstOverflowsTrainer,
}

_REFERENCE_RUNS: dict[tuple[str, str], boundary_test._Run] = {}


def _reference(name: str, trainer: str) -> boundary_test._Run:
  key = (name, trainer)
  if key not in _REFERENCE_RUNS:
    _REFERENCE_RUNS[key] = _base(
        boundary_test._SCENARIOS[name], _TRAINERS[trainer]
    )
  return _REFERENCE_RUNS[key]


class StageBeforeUpdateEquivalenceTest(parameterized.TestCase):
  """Every scenario of the harness, against every double, in the staged mode
  with the default hook, passes the STRICT check: the base loop's trainer
  calls exactly (reruns included), its checkpoints and metric values. Unlike
  the hook after the last call, a stage that was planned but found nothing
  packed leaves the update confirmed (the deferred update call waits for the
  token), so no relaxation is needed. In the `_SENT_AHEAD` scenarios every
  boundary not after a scored checkpoint is staged, and every stage sits
  before its step's update."""

  @parameterized.named_parameters(
      (f"{name}_{trainer}", name, trainer)
      for name in boundary_test._SCENARIOS
      for trainer in _TRAINERS
  )
  def test_trainer_cannot_tell_it_from_the_base_loop(self, name, trainer):
    sc = boundary_test._SCENARIOS[name]
    ref = _reference(name, trainer)
    new = _run(sc, pipelined=True, staged=True, trainer_cls=_TRAINERS[trainer])
    boundary_test._assert_equivalent(self, ref, new)
    boundary_test._assert_weights_follow_updates(self, new)
    boundary_test._assert_metrics_fetched_at_the_boundary(self, new)
    self.assertTrue(
        all(e[2] == 1 for e in new.log if e[0] == "get_metrics"), new.log
    )
    self.assertIsNone(new.program._pending_train)
    self.assertIsNone(_worker_of(new)._deferred_update)
    self.assertEqual(
        _stage_positions(new), ["before_update"] * _stages(new), new.log
    )
    _staged_microbatches_trained_once(self, new)
    if name in boundary_test._SENT_AHEAD:
      boundaries = len(boundary_test._microbatches_per_step(new)) - 1
      expected = sum(
          not boundary_test._writes_scored_checkpoint(sc, k)
          for k in range(boundaries)
      )
      self.assertEqual(_stages(new), expected, new.log)
    if trainer.startswith(("last_overflows", "first_overflows")):
      self.assertEqual(new.trainer.reruns, ref.trainer.reruns)
      self.assertNotEmpty(new.trainer.reruns)


class StageBeforeUpdateChecksFailOnWrongProgramsTest(absltest.TestCase):
  """Negative controls for the checks above."""

  def test_a_rerun_that_reads_the_staged_micro_batch_is_caught(self):
    # The wrong double only differs from `_LastOverflowsTrainer` while a stage
    # is held, i.e. when staging before the update: the base loop and the
    # hook after the last call pass with it, this mode does not.
    sc = boundary_test._SCENARIOS["prompt_batch"]
    ref = _base(sc, _LastOverflowsTrainer)
    bad = _run(
        sc, pipelined=True, staged=True, trainer_cls=_RerunReadsTheStagedTrainer
    )
    self.assertGreater(_stages(bad), 0, bad.log)
    self.assertFalse(boundary_test._check_equivalent(ref, bad))
    # Controls: with nothing held at a rerun the wrong double is harmless.
    self.assertTrue(
        boundary_test._check_equivalent(
            ref, _base(sc, _RerunReadsTheStagedTrainer)
        )
    )
    after = boundary_test._run(
        sc,
        pipelined=True,
        staged=True,
        trainer_cls=_RerunReadsTheStagedTrainer,
        program_cls=_AsBuilt,
    )
    self.assertTrue(boundary_test._check_equivalent(ref, after))
    # And the right double passes in this mode.
    good = _run(
        sc, pipelined=True, staged=True, trainer_cls=_LastOverflowsTrainer
    )
    self.assertTrue(boundary_test._check_equivalent(ref, good))

  def test_the_position_check_catches_an_update_before_the_stage(self):
    sc = boundary_test._SCENARIOS["prompt_batch"]
    bad = _run(
        sc, pipelined=True, staged=True, program_cls=_UpdatesBeforeStaging
    )
    self.assertGreater(_stages(bad), 0)
    self.assertNotIn("before_update", _stage_positions(bad), bad.log)
    good = _run(sc, pipelined=True, staged=True)
    self.assertEqual(_stage_positions(good), ["before_update"] * _stages(good))

  def test_the_harness_wrong_programs_are_still_caught(self):
    sc = boundary_test._SCENARIOS["prompt_batch"]
    # Packing the next step from the wrong batch: trained differently.
    wide = dataclasses.replace(sc, num_prompts=20)
    wrong_batch = _run(
        wide,
        pipelined=True,
        staged=True,
        program_cls=boundary_test._PrepacksWrongBatch,
        trainer_cls=_ReplayTrainer,
    )
    self.assertFalse(
        boundary_test._check_equivalent(
            _base(wide, _ReplayTrainer), wrong_batch
        )
    )
    # Sending step k+1's first micro-batch right after step k's last call:
    # with the update deferred, the trainer refuses to train it on top of the
    # open step, or to stage it twice, so the run fails loudly.
    errors: list[BaseException] = []
    sends = _run(
        sc,
        pipelined=True,
        staged=True,
        program_cls=boundary_test._SendsBeforeBoundary,
        trainer_cls=_ReplayTrainer,
        errors=errors,
    )
    self.assertLen(errors, 1, sends.log)
    self.assertRegex(str(errors[0]), "is deferred|already holds staged")


# ---------------------------------------------------------------------------
# A replay happens: the staged micro-batch stays as it is
# ---------------------------------------------------------------------------


class DroplessReplayTest(absltest.TestCase):
  """When step k's last micro-batch overflows, its rerun runs inside the
  deferred update, after step k+1's first micro-batch was staged.
  The staged micro-batch needs no invalidation (its inputs are mapped from
  its own payload alone; the rerun reads the inputs it kept): it is trained
  once, unchanged, and the run is the base loop's to the trainer, reruns
  included. The stage still hides: the wait measured before the update is
  that micro-batch's remaining device time (D - T, the rerun comes after it).
  When the staged micro-batch itself overflows, it is rerun from the input it
  was staged as, with the same result.
  """

  def _check_replayed_run(self, costs, label, trainer_cls, base_cls=None):
    sc = _latch_scenario(costs)
    ref = _quiet(_base, sc, base_cls or trainer_cls)
    run = _quiet(_run, sc, pipelined=True, staged=True, trainer_cls=trainer_cls)
    boundaries = len(boundary_test._microbatches_per_step(run)) - 1
    waits = _waits_units(run)
    msg = (
        f"{label}: stages={_stages(run)}/{boundaries} waits(units)={waits}"
        f" reruns={len(run.trainer.reruns)}"
        f" latched={run.program._staging_latched_off}"
    )
    self.assertGreaterEqual(
        min(boundary_test._microbatches_per_step(run)), 3, msg
    )
    self.assertEqual(_stages(run), boundaries, msg)
    self.assertFalse(run.program._staging_latched_off, msg)
    self.assertEqual(run.trainer.reruns, ref.trainer.reruns, msg)
    boundary_test._assert_equivalent(self, ref, run)
    _staged_microbatches_trained_once(self, run)
    self.assertEqual(_stage_positions(run), ["before_update"] * _stages(run))
    return run, waits

  def test_last_micro_batch_replayed_submission_run(self):
    run, waits = self._check_replayed_run(
        SUBMISSION_RUN, "last-submission-run", _LastOverflowsTrainer
    )
    self.assertLen(
        run.trainer.reruns, len(boundary_test._microbatches_per_step(run))
    )
    for wait in waits:
      self.assertAlmostEqual(
          wait, SUBMISSION_RUN["device"] - SUBMISSION_RUN["host"], delta=2.0
      )

  def test_last_micro_batch_replayed_other_run(self):
    _, waits = self._check_replayed_run(
        OTHER_RUN, "last-other-run", _LastOverflowsTrainer
    )
    for wait in waits:
      self.assertAlmostEqual(
          wait, OTHER_RUN["device"] - OTHER_RUN["host"], delta=1.5
      )

  def test_last_micro_batch_replayed_with_a_token_trainer(self):
    self._check_replayed_run(
        SUBMISSION_RUN, "last-submission-run-token", _LastOverflowsTokenTrainer
    )

  def test_staged_micro_batch_replayed(self):
    run, _ = self._check_replayed_run(
        SUBMISSION_RUN, "first-submission-run", _FirstOverflowsTrainer
    )
    staged = {e[1] for e in run.log if e[0] == "stage"}
    self.assertTrue(staged & set(run.trainer.reruns), run.log)

  def test_every_micro_batch_replayed(self):
    self._check_replayed_run(
        SUBMISSION_RUN, "every-submission-run", _EveryOverflowsTrainer
    )

  def test_the_worker_still_holds_the_stage_after_a_replayed_update(self):
    # Direct readout, no harness timing: after the deferred update ran a
    # rerun, the worker holds the same staged micro-batch, and the next step's
    # first call names it instead of carrying it.
    sc = boundary_test._SCENARIOS["prompt_batch"]
    held_after_update = []
    real = boundary_test._Worker.apply_deferred_update

    def _recording(worker):
      resp = real(worker)
      staged = worker._staged_fwd_bwd
      held_after_update.append(
          None if staged is None else boundary_test._fingerprint(staged[1])
      )
      return resp

    with mock.patch.object(
        boundary_test._Worker, "apply_deferred_update", _recording
    ):
      run = _run(
          sc, pipelined=True, staged=True, trainer_cls=_LastOverflowsTrainer
      )
    stages = [e[1] for e in run.log if e[0] == "stage"]
    self.assertNotEmpty(stages)
    self.assertEqual([h for h in held_after_update if h is not None], stages)
    self.assertGreaterEqual(len(run.trainer.reruns), len(stages))


# ---------------------------------------------------------------------------
# No replay: the stage hides the boundary; under replay it still does
# ---------------------------------------------------------------------------


class StageBeforeUpdateHidesTest(absltest.TestCase):
  """Under the replay double the hook after the last call stages once and
  latches (positive control: its wait < 50 ms), while staging before the
  update stages at every boundary with a wait ~ D - T (5.6 units at the
  submission run's costs, 2.2 at the other run's), and its boundary gap is
  the harness's staged gap, 6.6 units (the hook after the last call: 15.1
  under replay = T_HOST - T_UPD + T_METRICS + T_DEST + T_STAGE). Without
  replay it also stages at every boundary and the gap is 6.6 (no regression
  where update() returns at dispatch).
  """

  def _check_trainer_bound(self, costs, label, trainer_cls, *, expect_wait):
    sc = _latch_scenario(costs)
    run = _quiet(
        _run,
        sc,
        pipelined=True,
        staged=True,
        trainer_cls=trainer_cls,
        program_cls=_RecordingWaits,
    )
    boundaries = len(boundary_test._microbatches_per_step(run)) - 1
    waits = _waits_units(run)
    msg = f"{label}: stages={_stages(run)}/{boundaries} waits(units)={waits}"
    self.assertGreaterEqual(
        min(boundary_test._microbatches_per_step(run)), 3, msg
    )
    self.assertEqual(_stages(run), boundaries, msg)
    self.assertFalse(run.program._staging_latched_off, msg)
    # What the latch read is what the trainer measured, one per stage.
    self.assertEqual(run.program.seen, run.trainer.device_waits, msg)
    self.assertLen(waits, boundaries, msg)
    for wait in waits:
      self.assertGreaterEqual(wait * U, LATCH_S, msg)
      if expect_wait is not None:
        self.assertAlmostEqual(wait, expect_wait, delta=2.0, msg=msg)
    boundary_test._assert_equivalent(self, _quiet(_base, sc, trainer_cls), run)
    return run

  def test_replay_submission_run(self):
    self._check_trainer_bound(
        SUBMISSION_RUN,
        "replay-submission-run",
        _ReplayTrainer,
        expect_wait=SUBMISSION_RUN["device"] - SUBMISSION_RUN["host"],
    )

  def test_replay_submission_run_token(self):
    self._check_trainer_bound(
        SUBMISSION_RUN,
        "replay-submission-run-token",
        _ReplayTokenTrainer,
        expect_wait=None,
    )

  def test_replay_other_run(self):
    self._check_trainer_bound(
        OTHER_RUN,
        "replay-other-run",
        _ReplayTrainer,
        expect_wait=OTHER_RUN["device"] - OTHER_RUN["host"],
    )

  def test_no_replay_submission_run(self):
    self._check_trainer_bound(
        SUBMISSION_RUN,
        "plain-submission-run",
        boundary_test._Trainer,
        expect_wait=None,
    )

  def test_the_hook_after_the_last_call_latches_under_replay(self):
    # Positive control: same scenario, the old hook.
    sc = _latch_scenario(SUBMISSION_RUN)
    run = _quiet(
        boundary_test._run,
        sc,
        pipelined=True,
        staged=True,
        trainer_cls=_ReplayTrainer,
        program_cls=_AsBuilt,
    )
    msg = (
        f"as-built under replay: stages={_stages(run)}"
        f" waits={_waits_units(run)}"
    )
    self.assertEqual(_stages(run), 1, msg)
    self.assertTrue(run.program._staging_latched_off, msg)
    self.assertLess(run.trainer.device_waits[0], LATCH_S, msg)
    # Its one stage sat after the update; it still trains the same thing.
    self.assertEqual(_stage_positions(run), ["get_metrics"], run.log)
    boundary_test._assert_equivalent(
        self, _quiet(_base, sc, _ReplayTrainer), run
    )

  def test_boundary_gaps(self):
    replay_new, replay_run = _boundary_gaps(_ReplayTrainer)
    plain_new, _ = _boundary_gaps(boundary_test._Trainer)
    replay_old, _ = _boundary_gaps(_ReplayTrainer, before_update=False)
    replay_pipelined, _ = _boundary_gaps(_ReplayTrainer, staged=False)
    replay_today, _ = _boundary_gaps(
        _ReplayTrainer, pipelined=False, staged=False
    )
    last_new, _ = _boundary_gaps(_LastOverflowsTrainer)
    pred_old = (
        boundary_test.T_HOST
        - boundary_test.T_UPD
        + boundary_test.T_METRICS
        + boundary_test.T_DEST
        + boundary_test.T_STAGE
    )  # 15.1
    msg = (
        f"today={replay_today} pipelined={replay_pipelined}"
        f" as_built={replay_old} (pred {pred_old}) before_update={replay_new}"
        f" plain={plain_new} last_overflows={last_new} (pred"
        f" {boundary_test.PRED_STAGED}) waits={_waits_units(replay_run)}"
    )
    tol = boundary_test.TOLERANCE
    for gaps in (replay_new, plain_new, last_new):
      self.assertLen(gaps, 1, msg)
      self.assertAlmostEqual(
          gaps[0], boundary_test.PRED_STAGED, delta=tol, msg=msg
      )
    self.assertAlmostEqual(
        replay_today[0], boundary_test.PRED_TODAY, delta=tol, msg=msg
    )
    self.assertAlmostEqual(
        replay_pipelined[0], boundary_test.PRED_PIPELINED, delta=tol, msg=msg
    )
    self.assertAlmostEqual(replay_old[0], pred_old, delta=tol, msg=msg)
    # The difference that matters, with the harness latency cancelled.
    self.assertGreater(replay_old[0] - replay_new[0], 6.0, msg)

  def test_feed_bound_latches_after_one_stage(self):
    # The moved latch, two-sided: a transfer (0.3 s) longer than a
    # micro-batch (0.05 s) finds the device idle at the deferred update.
    sc = boundary_test._latch_scenario(
        boundary_test._FEED_BOUND,
        num_prompts=60,
        max_steps=5,
        dest_s=0.0,
        pack_s=0.0,
        tail_s=0.0,
    )
    run = _run(sc, pipelined=True, staged=True, trainer_cls=_ReplayTrainer)
    msg = (
        f"feed-bound waits (s) {run.trainer.device_waits} stages={_stages(run)}"
    )
    self.assertEqual(_stages(run), 1, msg)
    self.assertTrue(run.program._staging_latched_off, msg)
    self.assertLess(run.trainer.device_waits[0], LATCH_S, msg)
    control = _run(
        sc,
        pipelined=True,
        staged=True,
        trainer_cls=_ReplayTrainer,
        program_cls=_NeverLatches,
    )
    boundaries = len(boundary_test._microbatches_per_step(control)) - 1
    self.assertEqual(_stages(control), boundaries, control.trainer.device_waits)
    for r in (run, control):
      boundary_test._assert_equivalent(self, _base(sc, _ReplayTrainer), r)

  def test_the_wait_is_taken_before_the_update(self):
    # Negative control for the order in `apply_deferred_update`: a worker
    # that dispatches the update first measures the update's own device time
    # (the replay double's update() blocks until the last micro-batch is
    # done), so even a feed-bound trainer looks hidden and never latches.
    class _UpdatesFirstWorker(boundary_test._Worker):

      def apply_deferred_update(self):
        self._deferred_update = None
        train_step = self.update()
        wait = self._trainer.wait_for_device()
        return self._response(
            updated=True, train_step=train_step, device_wait_s=wait
        )

    sc = boundary_test._latch_scenario(
        dataclasses.replace(boundary_test._FEED_BOUND, update_s=0.06),
        num_prompts=60,
        max_steps=5,
        dest_s=0.0,
        pack_s=0.0,
        tail_s=0.0,
    )
    bad = _run(
        sc,
        pipelined=True,
        staged=True,
        trainer_cls=_ReplayTrainer,
        worker_cls=_UpdatesFirstWorker,
    )
    good = _run(sc, pipelined=True, staged=True, trainer_cls=_ReplayTrainer)
    msg = f"bad stages={_stages(bad)} good stages={_stages(good)}"
    self.assertTrue(good.program._staging_latched_off, msg)
    self.assertFalse(bad.program._staging_latched_off, msg)
    self.assertGreater(_stages(bad), _stages(good), msg)


# ---------------------------------------------------------------------------
# Which hook runs
# ---------------------------------------------------------------------------


class _OlderWorker(boundary_test._Worker):
  """A trainer image with staging but without deferred updates."""

  apply_deferred_update = None

  def info(self) -> datatypes.WorkerInfo:
    info = super().info()
    resources = {
        k: v for k, v in info.resources.items() if k != "stage_before_update"
    }
    return dataclasses.replace(info, resources=resources)

  def fwd_bwd(self, request, apply_optimizer=False, **kwargs):
    if "defer_update" in kwargs:
      raise TypeError("this worker has no defer_update")
    return super().fwd_bwd(request, apply_optimizer, **kwargs)


class HookSelectionTest(absltest.TestCase):

  def test_default_is_before_the_update(self):
    self.assertTrue(run_deepswe_dist._parse_args([]).stage_before_update)
    self.assertFalse(
        run_deepswe_dist._parse_args(
            ["--no-stage_before_update"]
        ).stage_before_update
    )
    program = boundary_test._new_program(
        boundary_test._SCENARIOS["prompt_batch"],
        pipelined=True,
        staged=True,
        program_cls=rl_program.StandardRLProgram,
        log=[],
        checkpoints=[],
        program_kwargs={"stage_before_update": True},
    )
    self.assertTrue(program.stage_before_update)

  def test_library_default_is_before_the_update(self):
    class _LibraryDefault(rl_program.StandardRLProgram):

      def __init__(self, *args, stage_before_update=None, **kwargs):
        del stage_before_update
        super().__init__(*args, **kwargs)

    sc = boundary_test._SCENARIOS["prompt_batch"]
    run = boundary_test._run(
        sc, pipelined=True, staged=True, program_cls=_LibraryDefault
    )
    self.assertTrue(run.program.stage_before_update)
    self.assertGreater(_stages(run), 0)
    self.assertEqual(_stage_positions(run), ["before_update"] * _stages(run))

  def test_flag_off_stages_after_the_last_call(self):
    sc = boundary_test._SCENARIOS["prompt_batch"]
    run = _run(
        sc,
        pipelined=True,
        staged=True,
        program_kwargs={"stage_before_update": False},
    )
    self.assertGreater(_stages(run), 0)
    self.assertNotIn("before_update", _stage_positions(run))
    self.assertIsNone(run.program._stage_before_update_supported)

  def test_a_worker_without_deferred_updates_gets_the_old_hook(self):
    sc = boundary_test._SCENARIOS["prompt_batch"]
    with _warnings() as warnings:
      run = _run(sc, pipelined=True, staged=True, worker_cls=_OlderWorker)
    self.assertGreater(_stages(run), 0)
    self.assertNotIn("before_update", _stage_positions(run))
    self.assertFalse(run.program._stage_before_update_supported)
    self.assertTrue(
        any("cannot defer a step's update" in w for w in warnings), warnings
    )
    boundary_test._assert_equivalent(
        self, _reference("prompt_batch", "plain"), run
    )

  def test_without_the_check_such_a_worker_fails_the_run(self):
    # Negative control: deferring without asking the trainer first.
    class _SkipsTheCheck(rl_program.StandardRLProgram):

      async def _trainer_can_stage_before_update(self):
        return True

    errors: list[BaseException] = []
    run = _run(
        boundary_test._SCENARIOS["prompt_batch"],
        pipelined=True,
        staged=True,
        worker_cls=_OlderWorker,
        program_cls=_SkipsTheCheck,
        errors=errors,
    )
    self.assertLen(errors, 1)
    self.assertIn("defer_update", str(errors[0]))
    self.assertNotIn(("on_step_end", 0), run.log)

  def test_the_trainer_is_asked_once(self):
    calls = []
    real_info = boundary_test._Worker.info

    def _counting_info(self):
      calls.append(1)
      return real_info(self)

    with mock.patch.object(boundary_test._Worker, "info", _counting_info):
      run = _run(boundary_test._SUPPORT_SCENARIO, pipelined=True, staged=True)
    # One answer covers staging and the deferred update.
    self.assertLen(calls, 1)
    self.assertEqual(_stages(run), 3)
    self.assertTrue(run.program._staging_supported)
    self.assertTrue(run.program._stage_before_update_supported)

  def test_a_worker_that_runs_the_update_anyway_is_taken_as_fused(self):
    # A worker that does not know `defer_update` and runs the update in the
    # call (an older TrainerWorker forwards unknown keywords to the trainer,
    # which ignores them) leaves `update_deferred` unset: the engine handles
    # the reply as a fused call (one update, never two) and the program
    # stages after it, as with a worker that says it cannot defer.
    class _IgnoresDefer(boundary_test._Worker):

      def fwd_bwd(self, request, apply_optimizer=False, **kwargs):
        kwargs.pop("defer_update", None)
        return super().fwd_bwd(request, apply_optimizer, **kwargs)

    class _SkipsTheCheck(rl_program.StandardRLProgram):

      async def _trainer_can_stage_before_update(self):
        return True

    sc = boundary_test._SCENARIOS["prompt_batch"]
    run = _run(
        sc,
        pipelined=True,
        staged=True,
        worker_cls=_IgnoresDefer,
        program_cls=_SkipsTheCheck,
    )
    self.assertGreater(_stages(run), 0)
    self.assertNotIn("before_update", _stage_positions(run))
    boundary_test._assert_equivalent(
        self, _reference("prompt_batch", "plain"), run
    )
    updates = [e for e in run.log if e[0] == "update"]
    self.assertLen(updates, len(boundary_test._microbatches_per_step(run)))

  def test_flags_off_never_take_the_new_paths(self):
    def _boom(*args, **kwargs):
      del args, kwargs
      raise AssertionError("a stage-before-update path ran")

    sc = dataclasses.replace(
        boundary_test._SCENARIOS["prompt_batch"], num_prompts=16, max_steps=4
    )
    patches = dict(
        _trainer_can_stage_before_update=_boom,
        _stage_next_before_update=_boom,
    )
    with mock.patch.multiple(rl_program.StandardRLProgram, **patches):
      for kwargs in (
          dict(pipelined=False, staged=False),
          dict(pipelined=True, staged=False),
      ):
        run = _run(sc, **kwargs)
        boundary_test._assert_equivalent(
            self, _base(sc, boundary_test._Trainer), run
        )
        self.assertNotIn("stage", [e[0] for e in run.log])
    # Two-sided: staged, the same patches fire.
    errors: list[BaseException] = []
    with mock.patch.multiple(rl_program.StandardRLProgram, **patches):
      _run(sc, pipelined=True, staged=True, errors=errors)
    self.assertLen(errors, 1)
    self.assertIn("stage-before-update path ran", str(errors[0]))


# ---------------------------------------------------------------------------
# Checkpoint stamps
# ---------------------------------------------------------------------------


class CheckpointStampTest(absltest.TestCase):
  """The MLPerf rule: a checkpoint is stamped after its last weight update."""

  def _assert_stamped_after_update(self, run: boundary_test._Run) -> None:
    self.assertNotEmpty(run.checkpoints)
    for ckpt in run.checkpoints:
      landed = run.trainer.update_landed_ms[ckpt["step"] - 1]
      self.assertGreaterEqual(ckpt["timestamp_ms"], int(landed))

  # The harness's busy costs with a longer update. Against a trainer whose
  # fwd_bwd returns at dispatch, the deferred last call returns right after
  # that micro-batch's dispatch (a fused call would also have dispatched the
  # update first), so whether the next step's first micro-batch is packed by
  # then is timing; under the replay doubles it returns once the micro-batch
  # before it finished, and it always is.
  _COSTS = boundary_test._Costs(device_s=0.15, update_s=0.05, host_s=0.02)

  def test_stamp_follows_the_landed_update(self):
    costs = self._COSTS
    sc = dataclasses.replace(
        boundary_test._SCENARIOS["prompt_batch"], costs=costs
    )
    for trainer_cls in (
        boundary_test._Trainer,
        boundary_test._TokenTrainer,
        _ReplayTrainer,
        _LastOverflowsTokenTrainer,
    ):
      run = _run(sc, pipelined=True, staged=True, trainer_cls=trainer_cls)
      if issubclass(trainer_cls, _ReplayTrainer):
        self.assertGreater(_stages(run), 0)
      self._assert_stamped_after_update(run)

  def test_stamp_check_catches_a_stamp_taken_when_the_last_call_returns(self):
    # With the update deferred the last call returns before the update is
    # even dispatched; a stamp there is early against every trainer.
    costs = self._COSTS
    sc = dataclasses.replace(
        boundary_test._SCENARIOS["prompt_batch"], costs=costs
    )
    bad = _run(
        sc,
        pipelined=True,
        staged=True,
        trainer_cls=_ReplayTokenTrainer,
        program_cls=boundary_test._StampsBeforeUpdateLands,
    )
    self.assertGreater(_stages(bad), 0)
    with self.assertRaises(AssertionError):
      self._assert_stamped_after_update(bad)
    # Control: the same trainer, the program's own stamp.
    good = _run(
        sc, pipelined=True, staged=True, trainer_cls=_ReplayTokenTrainer
    )
    self.assertGreater(_stages(good), 0)
    self._assert_stamped_after_update(good)

  def test_scored_steps_are_not_staged(self):
    costs = boundary_test._Costs(device_s=0.05, update_s=0.01, host_s=0.3)
    sc = dataclasses.replace(
        boundary_test._SCENARIOS["checkpoints_from_step_3"], costs=costs
    )
    run = _run(sc, pipelined=True, staged=True, program_cls=_NeverLatches)
    log = run.log
    # Boundaries 0 and 1 stage; 2 and 3 follow steps that write scored
    # checkpoints (steps 3 and 4).
    stage_steps = []
    for i, event in enumerate(log):
      if event[0] == "stage":
        stage_steps.append(sum(1 for e in log[:i] if e[0] == "update"))
    self.assertEqual(stage_steps, [0, 1], log)
    self._assert_stamped_after_update(run)


# ---------------------------------------------------------------------------
# Failures on the new path
# ---------------------------------------------------------------------------


class FailureTest(absltest.TestCase):

  def _scenario(self):
    return dataclasses.replace(
        boundary_test._SCENARIOS["prompt_batch"], num_prompts=16, max_steps=4
    )

  def test_a_stage_lost_on_the_way_falls_back_and_the_update_still_runs(self):
    sc = self._scenario()
    real_execute = boundary_test._RemoteHandle._execute
    fired = {"n": 0}

    async def _execute(handle, request):
      if request.method_name == "stage_fwd_bwd" and not fired["n"]:
        fired["n"] = 1
        raise ConnectionError("transport: stream reset before the stage ran")
      return await real_execute(handle, request)

    with _warnings() as warnings:
      with mock.patch.object(boundary_test._RemoteHandle, "_execute", _execute):
        run = _run(sc, pipelined=True, staged=True, trainer_cls=_ReplayTrainer)
    self.assertEqual(fired["n"], 1, "precondition: boundary 0 was staged")
    self.assertEqual(run.program.step, sc.max_steps)
    self.assertTrue(run.program._staging_latched_off)
    self.assertEqual(_stages(run), 0, run.log)
    self.assertTrue(
        any("Staging failed" in w and "ConnectionError" in w for w in warnings),
        warnings,
    )
    boundary_test._assert_equivalent(self, _base(sc, _ReplayTrainer), run)
    worker = _worker_of(run)
    self.assertIsNone(worker._deferred_update)
    self.assertEqual(worker.state, datatypes.WorkerState.READY)

  def test_without_the_fallback_a_lost_stage_ends_the_run(self):
    # Negative control: the same failure treated as fatal.
    sc = self._scenario()
    real_execute = boundary_test._RemoteHandle._execute
    fired = {"n": 0}

    async def _execute(handle, request):
      if request.method_name == "stage_fwd_bwd" and not fired["n"]:
        fired["n"] = 1
        raise ConnectionError("transport: stream reset before the stage ran")
      return await real_execute(handle, request)

    errors: list[BaseException] = []
    with mock.patch.object(
        rl_program, "_raised_by_worker_method", lambda exc: True
    ):
      with mock.patch.object(boundary_test._RemoteHandle, "_execute", _execute):
        run = _run(sc, pipelined=True, staged=True, errors=errors)
    self.assertLen(errors, 1)
    self.assertNotIn(("on_step_end", 0), run.log)

  def test_a_failed_reference_call_falls_back(self):
    class _MarksItsStage(rl_program.StandardRLProgram):

      staging = False

      async def _send_stage(self, st):
        type(self).staging = True
        try:
          return await super()._send_stage(st)
        finally:
          type(self).staging = False

    sc = boundary_test._SCENARIOS["reference_kl"]
    real = boundary_test._Reference.per_token_logps
    fired = {"n": 0}

    def _flaky(reference, items, **kwargs):
      if _MarksItsStage.staging and not fired["n"]:
        fired["n"] = 1
        raise RuntimeError("reference model: device lost")
      return real(reference, items, **kwargs)

    errors: list[BaseException] = []
    with mock.patch.object(boundary_test._Reference, "per_token_logps", _flaky):
      run = _run(
          sc,
          pipelined=True,
          staged=True,
          program_cls=_MarksItsStage,
          errors=errors,
      )
    self.assertEqual(fired["n"], 1, "precondition: boundary 0 was staged")
    self.assertEqual(errors, [])
    self.assertTrue(run.program._staging_latched_off)
    boundary_test._assert_equivalent(
        self, _reference("reference_kl", "plain"), run
    )

  def test_a_stage_that_fails_on_the_trainer_is_fatal(self):
    sc = self._scenario()
    real = boundary_test._Worker.stage_fwd_bwd

    def _raises(worker, request, **kwargs):
      del request, kwargs
      raise ConnectionError("raised by the worker's stage_fwd_bwd")

    errors: list[BaseException] = []
    with mock.patch.object(boundary_test._Worker, "stage_fwd_bwd", _raises):
      run = _run(sc, pipelined=True, staged=True, errors=errors)
    del real
    self.assertLen(errors, 1)
    self.assertIn("raised by the worker's stage_fwd_bwd", str(errors[0]))
    self.assertNotIn(("on_step_end", 0), run.log)

  def test_a_failed_deferred_update_ends_the_run(self):
    sc = self._scenario()

    class _FailingUpdateTrainer(boundary_test._Trainer):

      def update(self, **kwargs):
        raise RuntimeError("update lost on the device")

    errors: list[BaseException] = []
    run = _run(
        sc,
        pipelined=True,
        staged=True,
        trainer_cls=_FailingUpdateTrainer,
        errors=errors,
    )
    self.assertLen(errors, 1)
    self.assertIn("deferred optimizer update failed", str(errors[0]))
    self.assertIn("update lost on the device", str(errors[0]))
    self.assertNotIn(("on_step_end", 0), run.log)
    worker = _worker_of(run)
    self.assertEqual(worker.state, datatypes.WorkerState.ERROR)
    self.assertIn(
        "deferred optimizer update failed", worker.heartbeat().last_error
    )

  def test_an_update_that_is_never_applied_stops_the_next_step(self):
    # Negative control for the worker's guard: a program that forgets the
    # deferred update cannot train the next step on top of it.
    class _ForgetsTheUpdate(rl_program.StandardRLProgram):

      async def _stage_next_before_update(self, st, step_result):
        if st.ready and self._intake_done():
          await self._send_stage(st)
        return {**step_result, "updated": True, "train_step": 1}

    errors: list[BaseException] = []
    run = _run(
        self._scenario(),
        pipelined=True,
        staged=True,
        program_cls=_ForgetsTheUpdate,
        errors=errors,
    )
    self.assertLen(errors, 1)
    self.assertIn("is deferred", str(errors[0]))
    self.assertEqual(len([e for e in run.log if e[0] == "update"]), 0)


# ---------------------------------------------------------------------------
# Randomized scenarios
# ---------------------------------------------------------------------------


class RandomizedEquivalenceTest(parameterized.TestCase):
  """All 30 draws of the harness pass the strict check in the staged mode with
  the default hook, against the plain, token and replay doubles; and the
  biased trainer-bound draws stage in at least one draw."""

  @parameterized.parameters(*range(boundary_test._NUM_DRAWS))
  def test_matches_the_base_loop(self, seed):
    sc = boundary_test._draw(seed)
    for trainer in ("plain", "token", "last_overflows_token"):
      trainer_cls = _TRAINERS[trainer]
      ref = _base(sc, trainer_cls)
      new = _run(sc, pipelined=True, staged=True, trainer_cls=trainer_cls)
      with self.subTest(trainer=trainer, scenario=repr(sc)):
        boundary_test._assert_equivalent(self, ref, new)
        if any(e[0] == "fwd_bwd" for e in new.log):
          boundary_test._assert_weights_follow_updates(self, new)
        self.assertIsNone(new.program._pending_train)
        self.assertEqual(
            _stage_positions(new), ["before_update"] * _stages(new)
        )

  def test_trainer_bound_draws_stage(self):
    staged_draws = 0
    for seed in range(boundary_test._NUM_DRAWS):
      sc = boundary_test._draw_trainer_bound(seed)
      for trainer in ("plain", "replay_token"):
        trainer_cls = _TRAINERS[trainer]
        ref = _base(sc, trainer_cls)
        new = _run(
            sc,
            pipelined=True,
            staged=True,
            trainer_cls=trainer_cls,
            program_cls=_NeverLatches,
        )
        with self.subTest(seed=seed, trainer=trainer):
          boundary_test._assert_equivalent(self, ref, new)
        staged_draws += _stages(new) > 0
    self.assertGreater(staged_draws, 0)


class GrpcMicrobatchPipelineTest(absltest.TestCase):
  """End-to-end pipelined step boundary and intra-step microbatch overlap over
  real gRPC (`GrpcRemoteActorHandle` + `GrpcRemoteExecutionServer`)."""

  def test_zero_inter_microbatch_device_gaps_over_real_grpc(self):
    sc = boundary_test._Scenario(
        order="prompt_batch",
        num_prompts=54,
        full_batch_size=27,
        mini_batch_size=27,
        max_steps=2,
        costs=boundary_test._Costs(
            device_s=0.16,
            update_s=0.02,
            stage_s=0.04,
            host_s=0.0,
            metrics_s=0.01,
        ),
        dest_s=0.02,
        round_s=0.01,
        pack_s=0.005,
        tail_s=0.08,
    )
    ref = _quiet(_base, sc, _ReplayTokenTrainer)

    log: list[Any] = []
    checkpoints: list[dict[str, Any]] = []
    trainer = _ReplayTokenTrainer(log, sc.costs)
    worker = boundary_test._Worker(lambda: trainer)
    worker.initialize()
    ref_handle = boundary_test._RemoteHandle(boundary_test._Reference())

    async def _main():
      import portpicker  # pylint: disable=g-import-not-at-top
      port = portpicker.pick_unused_port()
      server = boundary_test.remote_execution.GrpcRemoteExecutionServer(worker)
      await server.start_serving_async(port=port)
      handle = boundary_test.remote_execution.GrpcRemoteActorHandle(
          f"grpc://localhost:{port}"
      )
      coordinator = boundary_test._Coordinator(handle, sc.dest_s, sc.round_s)
      engine = boundary_test.distributed_rl_engine.DistributedRLEngine(
          rollout_workers=[],
          trainer_workers={datatypes.Role.ACTOR: handle},
          inference_workers={datatypes.Role.REFERENCE: ref_handle},
          weight_sync_coordinator=coordinator,
      )
      program = boundary_test._new_program(
          sc,
          pipelined=True,
          staged=True,
          program_cls=_NeverLatches,
          log=log,
          checkpoints=checkpoints,
          program_kwargs=dict(_SBU),
      )
      program.engine = engine
      feeder = asyncio.create_task(boundary_test._feed_queue(program, sc))
      try:
        await asyncio.wait_for(program.train_stage(), timeout=60)
      finally:
        feeder.cancel()
        await handle.close()
        await server.stop_serving()
      return program

    try:
      program = _quiet(asyncio.run, _main())
    finally:
      ref_handle.close()

    run = boundary_test._Run(
        log=log,
        metrics=program.metrics_logger.records,  # pyrefly: ignore
        checkpoints=checkpoints,
        last_step_result=program.last_step_result,
        trainer=trainer,
        program=program,
    )
    boundary_test._assert_equivalent(self, ref, run)
    self.assertEqual(_stages(run), 1)
    # In Step 1 (after the first update), every microbatch (including mb1 -> mb2
    # across Step 0's 0.10s tail_s bookkeeping) is dispatched before the
    # previous microbatch finishes on the device, yielding 0.0s idle gap.
    first_upd_idx = next(
        i for i, (kind, *_) in enumerate(trainer.device) if kind == "update"
    )
    step1_mbs = [
        span
        for span in trainer.device[first_upd_idx + 1 :]
        if span[0] == "fwd_bwd"
    ]
    self.assertGreaterEqual(len(step1_mbs), 9)
    inter_mb_gaps = [
        round(step1_mbs[i][2] - step1_mbs[i - 1][3], 4)
        for i in range(1, len(step1_mbs))
    ]
    self.assertEqual(inter_mb_gaps, [0.0] * len(inter_mb_gaps), inter_mb_gaps)


if __name__ == "__main__":
  absltest.main()
