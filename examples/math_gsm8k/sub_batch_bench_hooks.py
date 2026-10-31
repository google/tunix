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

"""Preemption and evidence hooks for the sub-batch checkpointing benchmark.

`qwen3_grpo_sub_batch_benchmark.py` runs `qwen3_grpo_demo.py` under these
hooks. They wrap the learner's collaborators from the outside, so Tunix itself
is unchanged:

  - `rl_engine.update_actor`: a `train` event per call. Once training passes a
    trainer iteration listed in `TUNIX_SB_PREEMPT_AT`, the process waits for
    its checkpoints and snapshots to be durable, then hard-exits. With
    `chaos_prob`, a call may instead hard-exit at random, without waiting.
  - `SubBatchCheckpointManager`: a `snapshot` event per background save, with
    its time on the manager's writer thread, and a `resume` event with the
    restored groups.
  - The learner's rollout producer: a `rollout` event per training group.

Events are JSON lines appended to `TUNIX_SB_EVENT_LOG`. Progress metrics go
to wandb, timed from `TUNIX_BENCH_START_TIME` so attempts share one clock.
"""

from __future__ import annotations

import contextlib
import json
import os
import random
import threading
import time
from typing import Any

from absl import logging
from tunix.rl import sub_batch_checkpoint
from tunix.rl.agentic import agentic_rl_learner

# Exit code of a simulated preemption.
KILL_EXIT_CODE = 42


def install(
    learner: agentic_rl_learner.AgenticRLLearner, *, chaos_prob: float = 0.0
) -> None:
  """Installs the benchmark hooks around `learner` before `train()`."""
  _Hooks(learner, chaos_prob).install()


class _Hooks:
  """The benchmark hooks of one learner."""

  def __init__(
      self, learner: agentic_rl_learner.AgenticRLLearner, chaos_prob: float
  ):
    self._learner = learner
    self._trainer = learner.rl_engine.actor_trainer
    self._chaos_prob = chaos_prob
    self._event_log = os.environ.get("TUNIX_SB_EVENT_LOG")
    self._event_lock = threading.Lock()
    self._preempt_at = {
        int(v)
        for v in os.environ.get("TUNIX_SB_PREEMPT_AT", "").split(",")
        if v.strip()
    }
    self._start_time = float(
        os.environ.get("TUNIX_BENCH_START_TIME") or time.time()
    )
    self._manager: sub_batch_checkpoint.SubBatchCheckpointManager | None = (
        None
    )
    # Updates restored from a checkpoint were logged by earlier attempts.
    self._last_logged_update = self._trainer.train_steps

  def install(self) -> None:
    self._event(
        "start",
        train_steps=self._trainer.train_steps,
        iter_steps=self._trainer.iter_steps,
        global_step=self._learner.rl_engine.global_steps,
    )
    self._wrap_update_actor()
    self._wrap_manager()
    self._wrap_rollouts()

  def _wrap_update_actor(self) -> None:
    engine = self._learner.rl_engine
    update_actor = engine.update_actor

    def wrapped(chunks, *args, **kwargs):
      if self._chaos_prob and random.random() < self._chaos_prob:
        logging.warning(
            "Sub-batch bench: random preemption at trainer iteration %d.",
            self._trainer.iter_steps,
        )
        os._exit(KILL_EXIT_CODE)  # pylint: disable=protected-access
      iter_before = self._trainer.iter_steps
      train_before = self._trainer.train_steps
      update_actor(chunks, *args, **kwargs)
      self._event(
          "train",
          iter_before=iter_before,
          iter_steps=self._trainer.iter_steps,
          train_before=train_before,
          train_steps=self._trainer.train_steps,
          global_step=engine.global_steps,
          loss=self._loss(),
      )
      self._log_progress()
      self._maybe_preempt(iter_before)

    setattr(engine, "update_actor", wrapped)

  def _wrap_manager(self) -> None:
    manager_cls = sub_batch_checkpoint.SubBatchCheckpointManager
    record = manager_cls.record_rollout_group
    # Private, but it is the save itself: it runs on the writer thread, so
    # timing it measures the save without blocking training.
    save_records = manager_cls._save_records  # pylint: disable=protected-access
    restore = manager_cls.restore

    def wrapped_record(manager, global_step: int, group) -> None:
      # A preemption waits for this manager's saves.
      self._manager = manager
      record(manager, global_step, group)

    def wrapped_save_records(manager) -> None:
      # A save finds no records when an earlier one already took them.
      saving = bool(manager._records)  # pylint: disable=protected-access
      start = time.perf_counter()
      save_records(manager)
      if saving:
        self._event(
            "snapshot",
            # Training never waits for a snapshot.
            wait=False,
            iter_steps=self._trainer.iter_steps,
            global_step=self._learner.rl_engine.global_steps,
            save_s=round(time.perf_counter() - start, 4),
        )

    def wrapped_restore(manager, global_step: int):
      self._manager = manager
      state = restore(manager, global_step)
      self._event(
          "resume",
          global_step=global_step,
          groups=(
              []
              if state is None
              else [group[0].prompt_id for group in state.active_groups]
          ),
      )
      return state

    setattr(manager_cls, "record_rollout_group", wrapped_record)
    setattr(manager_cls, "_save_records", wrapped_save_records)
    setattr(manager_cls, "restore", wrapped_restore)

  def _wrap_rollouts(self) -> None:
    produce = self._learner._orchestrator_producer  # pylint: disable=protected-access

    async def wrapped(orchestrator, prompt_iterator, **kwargs):
      # Evaluation passes its prompts as a list.
      training = not isinstance(prompt_iterator, list)
      async with contextlib.aclosing(
          produce(orchestrator, prompt_iterator, **kwargs)
      ) as groups:
        async for group in groups:
          if training:
            self._event(
                "rollout",
                group_id=group[0].prompt_id,
                global_step=self._learner.rl_engine.global_steps,
            )
          yield group

    setattr(self._learner, "_orchestrator_producer", wrapped)

  def _maybe_preempt(self, iter_before: int) -> None:
    iter_now = self._trainer.iter_steps
    fired = sorted(t for t in self._preempt_at if iter_before < t <= iter_now)
    if not fired:
      return
    # Every recovery point is durable at the kill, so attempts resume from
    # the last one deterministically.
    self._trainer.checkpoint_manager.wait()
    if self._manager is not None:
      self._manager.wait()
    for target in fired:
      self._event(
          "preempt",
          iter_steps=target,
          killed_after_iter=iter_now,
          global_step=self._learner.rl_engine.global_steps,
      )
    logging.info(
        "Sub-batch bench: preemption at trainer iteration %d (targets %s).",
        iter_now,
        fired,
    )
    self._finish_wandb()
    os._exit(KILL_EXIT_CODE)  # pylint: disable=protected-access

  def _event(self, name: str, **fields: Any) -> None:
    """Appends one event to the event log; write failures are only logged."""
    if not self._event_log:
      return
    fields.update(event=name, ts=time.time(), pid=os.getpid())
    try:
      with self._event_lock, open(self._event_log, "a") as f:
        f.write(json.dumps(fields) + "\n")
    except OSError:
      logging.exception("Sub-batch bench: event log write failed.")

  def _loss(self) -> float | None:
    """The trainer's loss over the update window trained so far.

    The trainer writes its metrics one update late, so a kill can lose the last
    one. Its buffers have it as soon as `update_actor` returns: right after an
    update, the completed window moved to the previous buffer.
    """
    trainer = self._trainer
    # pylint: disable=protected-access
    buffered = (
        trainer._buffered_train_metrics or trainer._prev_buffered_train_metrics
    )
    # pylint: enable=protected-access
    if buffered is None or not buffered.losses:
      return None
    return float(buffered.loss)

  def _log_progress(self) -> None:
    """Logs progress against the shared wall clock, if wandb is running."""
    try:
      # Not available internally: only the OSS benchmark harness runs this.
      import wandb  # pyrefly: ignore[missing-import]  # pytype: disable=import-error  # pylint: disable=g-import-not-at-top
    except ImportError:
      return
    if wandb.run is None:
      return
    payload: dict[str, float] = {
        "perf/sub_batch_cumulative_time": time.time() - self._start_time,
        "perf/sub_batch_progress_micro_steps": self._trainer.iter_steps,
    }
    loss = self._loss()
    if loss is not None:
      payload["perf/sub_batch_loss"] = loss
      update = self._trainer.train_steps
      if update > self._last_logged_update:
        # One point per update, logged by the attempt that performed it.
        payload["perf/sub_batch_apply_loss"] = loss
        payload["perf/sub_batch_apply_step"] = update
        self._last_logged_update = update
    try:
      wandb.log(payload)
    except Exception:  # pylint: disable=broad-except
      # Metrics are observability; they must not stop the run under test.
      logging.exception("Sub-batch bench: wandb progress log failed.")

  def _finish_wandb(self) -> None:
    """Flushes wandb before a deterministic kill; a real one has no grace."""
    try:
      # Not available internally: only the OSS benchmark harness runs this.
      import wandb  # pyrefly: ignore[missing-import]  # pytype: disable=import-error  # pylint: disable=g-import-not-at-top
    except ImportError:
      return
    if wandb.run is None:
      return
    try:
      wandb.run.finish(exit_code=KILL_EXIT_CODE)
    except Exception:  # pylint: disable=broad-except
      # The kill proceeds either way.
      logging.exception("Sub-batch bench: wandb flush before kill failed.")
