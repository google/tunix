# Copyright 2025 Google LLC
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

"""RL trainer."""

from collections.abc import Callable
from typing import Any

from flax import nnx
import optax
from tunix.sft import peft_trainer
from typing_extensions import override


class Trainer(peft_trainer.PeftTrainer):
  """Handles additional RL metrics logging and display."""

  def __init__(
      self,
      model: nnx.Module,
      optimizer: optax.GradientTransformation,
      training_config: peft_trainer.TrainingConfig,
  ):
    super().__init__(
        model,
        optimizer,
        training_config,
    )
    self.rl_metrics_to_log = {}  # Metric name -> key in aux.
    self.tqdm_metrics_to_display = []

  def with_rl_metrics_to_log(self, rl_metrics_to_log: dict[str, str]) -> None:
    self.rl_metrics_to_log = rl_metrics_to_log

  def with_tqdm_metrics_to_display(
      self, tqdm_metrics_to_display: list[str | Callable[[], str]]
  ) -> None:
    self.tqdm_metrics_to_display = tqdm_metrics_to_display

  @override
  def _post_process_train_step(self, aux: Any) -> None:
    for metric_name, metric_key in self.rl_metrics_to_log.items():
      self.metrics_logger.log(
          metric_name, aux[metric_key], self._mode, self._train_steps
      )

  @override
  def _post_process_eval_step(self, aux: Any) -> None:
    for metric_name, metric_key in self.rl_metrics_to_log.items():
      self.metrics_logger.log(
          metric_name, aux[metric_key], self._mode, self._eval_steps
      )

  def _get_additional_tqdm_metrics(self) -> list[str]:
    metrics = set()
    for key_or_fn in self.tqdm_metrics_to_display:
      if isinstance(key_or_fn, str):
        metrics.add(key_or_fn)
      elif val := key_or_fn():
        metrics.add(val)
    return list(metrics)

  @property
  def _tqdm_train_metrics(self) -> list[str]:
    metrics = super()._tqdm_train_metrics
    metrics.extend(self._get_additional_tqdm_metrics())
    return metrics

  @property
  def _tqdm_eval_metrics(self) -> list[str]:
    metrics = super()._tqdm_eval_metrics
    metrics.extend(self._get_additional_tqdm_metrics())
    return metrics
