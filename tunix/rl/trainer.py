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

"""RL Trainer."""

from typing import Any, Callable, Optional

from flax import nnx
from jax.typing import ArrayLike  # pylint: disable=g-importing-member
import optax
from tunix.sft import peft_trainer
from typing_extensions import override
from tunix.sft.metrics_logger import MetricsLogger  # pylint: disable=unused-import


class Trainer(peft_trainer.PeftTrainer):
  """Handles additional RL metrics logging and display."""

  def __init__(
      self,
      model: nnx.Module,
      optimizer: optax.GradientTransformation,
      training_config: peft_trainer.TrainingConfig,
      custom_checkpoint_metadata_fn: Callable[[], dict[str, Any]],
      metrics_logger: Optional[MetricsLogger] = None,
  ):
    super().__init__(
        model,
        optimizer,
        training_config,
        metrics_logger,
    )
    self.rl_metrics_to_log = {}  # Metric name -> key in aux.
    self.tqdm_metrics_to_display = []
    self.custom_checkpoint_metadata_fn = custom_checkpoint_metadata_fn

  def with_rl_metrics_to_log(
      self,
      rl_metrics_to_log: dict[str, Callable[[ArrayLike], ArrayLike]],
  ) -> None:
    self.rl_metrics_to_log = rl_metrics_to_log

  def with_tqdm_metrics_to_display(
      self, tqdm_metrics_to_display: list[str | Callable[[], str]]
  ) -> None:
    self.tqdm_metrics_to_display = tqdm_metrics_to_display

  @override
  def custom_checkpoint_metadata(self) -> dict[str, Any]:
    return self.custom_checkpoint_metadata_fn()

  def restored_global_step(self) -> int:
    return self._restored_custom_metadata.get("global_step", 0)

  @override
  def _post_process_train_step(self, aux: Any) -> None:
    assert self._buffered_train_metrics is not None
    for metric_name, op in self.rl_metrics_to_log.items():
      if metric_name not in self._buffered_train_metrics.additional_metrics:
        self._buffered_train_metrics.additional_metrics[metric_name] = (
            [aux[metric_name]],
            op,
        )
      else:
        self._buffered_train_metrics.additional_metrics[metric_name][0].append(
            aux[metric_name]
        )

  @override
  def _post_process_eval_step(self, aux: Any) -> None:
    assert self._buffered_eval_metrics is not None
    for metric_name, op in self.rl_metrics_to_log.items():
      if metric_name not in self._buffered_eval_metrics.additional_metrics:
        self._buffered_eval_metrics.additional_metrics[metric_name] = (
            [aux[metric_name]],
            op,
        )
      else:
        self._buffered_eval_metrics.additional_metrics[metric_name][0].append(
            aux[metric_name]
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
