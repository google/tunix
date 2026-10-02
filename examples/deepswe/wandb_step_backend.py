"""W&B scalar logging with an explicit optimizer-step axis.

Metrax logs one scalar at a time. Its default backend commits each scalar at
W&B's global step, so subsequent scalars from the same optimizer step are
dropped. Keep W&B's append-only history independent of the training step.

Resume: a restarted job logs into the same W&B run (fixed run id with
resume="allow") and re-trains the steps between its last checkpoint and the
crash. W&B history is append-only (run rewind is a private preview), so on
resume we read the existing history and skip any scalar whose metric already
has a point at the same or a later training step. Every curve then keeps
exactly one point per training step.
"""

import collections
import time

from absl import logging
import numpy as np
from tunix.sft.metrics_logger import WandbBackend


class TrainingStepWandbBackend(WandbBackend):
  """Plot every scalar against the learner's optimizer step."""

  def __init__(self, project: str, name: str | None = None, **kwargs):
    super().__init__(project=project, name=name, **kwargs)
    # metric name -> largest training_step already in the run's history.
    self._logged_max_step: dict[str, int] = {}
    self._skipped = collections.Counter()
    if self._is_active:
      self.wandb.define_metric("training_step")
      self.wandb.define_metric("*", step_metric="training_step")
      if getattr(self.wandb.run, "resumed", False):
        self._load_logged_steps()

  def _load_logged_steps(self):
    """Reads the max training_step per metric from the resumed run."""
    run = self.wandb.run
    path = f"{run.entity}/{run.project}/{run.id}"
    for attempt in range(1, 4):
      start = time.time()
      try:
        api = self.wandb.Api(timeout=120)
        max_step: dict[str, int] = {}
        rows = 0
        for row in api.run(path).scan_history(page_size=10000):
          rows += 1
          step = row.get("training_step")
          if step is None:
            continue
          step = int(step)
          for key, value in row.items():
            if key.startswith("_") or key == "training_step" or value is None:
              continue
            if step > max_step.get(key, -1):
              max_step[key] = step
        self._logged_max_step = max_step
        logging.info(
            "[wandb] resumed run %s: %d history rows, %d metrics, latest"
            " training_step %d (read in %.1fs); re-logged points at or before"
            " a metric's latest step will be skipped.",
            path,
            rows,
            len(max_step),
            max(max_step.values(), default=-1),
            time.time() - start,
        )
        return
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.warning(
            "[wandb] reading history of %s failed (attempt %d/3): %s",
            path,
            attempt,
            e,
        )
        time.sleep(10 * attempt)
    logging.error(
        "[wandb] could not read history of %s; points re-logged after this"
        " resume may be duplicated.",
        path,
    )

  def log_scalar(self, event: str, value, **kwargs):
    if self.wandb is None or not self._is_active:
      return
    value_array = np.asarray(value)
    if value_array.size != 1:
      raise ValueError(f"Expected scalar W&B metric {event}, got {value_array.shape}")
    step = int(kwargs.get("step") or 0)
    key = event.lstrip("/")
    # Step 0 is also used by step-less events (e.g. compile times); keep them.
    if step > 0 and step <= self._logged_max_step.get(key, -1):
      self._skipped[key] += 1
      total = sum(self._skipped.values())
      if total == 1 or total % 100 == 0:
        logging.info(
            "[wandb] skipped %d re-logged points after resume (latest: %s at"
            " training_step %d)",
            total,
            key,
            step,
        )
      return
    self.wandb.log({
        "training_step": step,
        key: float(value_array.item()),
    })
