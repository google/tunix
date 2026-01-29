# Copyright 2025 Google LLC
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

"""Distillation trainer."""
import dataclasses
from typing import Any, Callable

import flax
from flax import nnx
from jax.typing import ArrayLike  # pylint: disable=g-importing-member
import optax
from tunix.distillation import strategies
from tunix.sft import metrics_logger
from tunix.sft import peft_trainer
from typing_extensions import override


@dataclasses.dataclass(slots=True, kw_only=True)
class TrainingConfig(peft_trainer.TrainingConfig):
  """Distillation training config."""


@flax.struct.dataclass(frozen=True)
class TrainingInput(peft_trainer.TrainingInput):
  """Distillation training input."""

  teacher_output: Any = None


class DistillationTrainer(peft_trainer.PeftTrainer):
  """Distillation trainer."""

  def __init__(
      self,
      student_model: nnx.Module,
      teacher_model: nnx.Module,
      strategy: strategies.BaseStrategy,
      optimizer: optax.GradientTransformation,
      training_config: TrainingConfig,
  ):
    """Initializes the DistillationTrainer.

    Args:
        student_model: The student model to train.
        teacher_model: The teacher model to use for distillation.
        strategy: The distillation strategy to use.
        optimizer: The optimizer to use for training.
        training_config: The training config.
    """
    student_model, teacher_model = strategy.pre_process_models(
        student_model, teacher_model
    )
    super().__init__(student_model, optimizer, training_config)
    self.strategy = strategy
    self.teacher_model = teacher_model
    self.loss_fn = self.get_train_loss
    self.eval_loss_fn = self.get_eval_loss

    # Since distillation strategies return (loss, metrics),
    # we explicitly enable auxiliary metric handling.
    self._has_aux = True

    self.gen_model_input_fn = lambda x: {
        "inputs": {"input_tokens": x.input_tokens, "input_mask": x.input_mask},
        "teacher_output": (
            x.teacher_output if hasattr(x, "teacher_output") else None
        ),
    }

  @override
  def with_gen_model_input_fn(
      self, gen_model_input_fn: Callable[[Any], dict[str, ArrayLike]]
  ) -> "DistillationTrainer":
    self.gen_model_input_fn = lambda x: {
        "inputs": gen_model_input_fn(x),
        "teacher_output": (
            x.teacher_output if hasattr(x, "teacher_output") else None
        ),
    }
    return self

  @override
  def with_loss_fn(
      self,
      loss_fn: Callable[..., ArrayLike | tuple[ArrayLike, Any]],
      has_aux: bool = False,
  ) -> "DistillationTrainer":
    raise NotImplementedError(
        "with_loss_fn is not supported for distillation. Use the strategy to"
        " define the loss."
    )

  @override
  def _prepare_inputs(self, input_data: TrainingInput) -> TrainingInput:
    inputs = self.gen_model_input_fn(input_data)["inputs"]
    if self._mode == metrics_logger.Mode.EVAL:
      teacher_output = None
    else:
      teacher_output = self.strategy.get_teacher_outputs(
          self.teacher_model, inputs
      )

    return TrainingInput(
        input_tokens=input_data.input_tokens,
        input_mask=input_data.input_mask,
        teacher_output=teacher_output,
    )

  def _standardize_loss_output(
      self, output: Any
  ) -> tuple[ArrayLike, dict[str, Any]]:
    """Adapts strategy output to always be (loss, metrics).

    This helper function ensures that the return value from a strategy's
    loss computation is consistent, regardless of whether the strategy
    returns just a scalar loss or a tuple of (loss, metrics).

    Args:
        output: The raw output from a strategy's `compute_loss` or
          `compute_eval_loss` method. Can be a single scalar (loss) or a tuple.

    Returns:
        A tuple containing:
          - loss: The scalar loss value (ArrayLike).
          - metrics: A dictionary of auxiliary metrics. If the input was just
            a scalar, this will be an empty dictionary.
    """
    if isinstance(output, tuple):
      return output
    # If strategy returned just a scalar loss, wrap it with empty metrics
    return output, {}

  def get_train_loss(
      self,
      model: nnx.Module,
      teacher_output: Any,
      inputs: dict[str, ArrayLike],
  ) -> tuple[ArrayLike, dict[str, Any]]:
    output = self.strategy.get_train_loss(model, teacher_output, inputs)
    return self._standardize_loss_output(output)

  def get_eval_loss(
      self,
      model: nnx.Module,
      teacher_output: Any,
      inputs: dict[str, ArrayLike],
  ) -> tuple[ArrayLike, dict[str, Any]]:
    del teacher_output  # Not computed in eval.
    output = self.strategy.get_eval_loss(model, inputs)
    return self._standardize_loss_output(output)

  def close(self):
    super().close()
    self.model, self.teacher_model = self.strategy.post_process_models(
        self.model, self.teacher_model
    )
