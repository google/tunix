"""W&B scalar logging with an explicit optimizer-step axis.

Metrax logs one scalar at a time. Its default backend commits each scalar at
W&B's global step, so subsequent scalars from the same optimizer step are
dropped. Keep W&B's append-only history independent of the training step.
"""

import numpy as np
from tunix.sft.metrics_logger import WandbBackend


class TrainingStepWandbBackend(WandbBackend):
  """Plot every scalar against the learner's optimizer step."""

  def __init__(self, project: str, name: str | None = None, **kwargs):
    super().__init__(project=project, name=name, **kwargs)
    if self._is_active:
      self.wandb.define_metric("training_step")
      self.wandb.define_metric("*", step_metric="training_step")

  def log_scalar(self, event: str, value, **kwargs):
    if self.wandb is None or not self._is_active:
      return
    value_array = np.asarray(value)
    if value_array.size != 1:
      raise ValueError(f"Expected scalar W&B metric {event}, got {value_array.shape}")
    self.wandb.log({
        "training_step": int(kwargs.get("step") or 0),
        event.lstrip("/"): float(value_array.item()),
    })
