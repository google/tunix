"""Experimental metric logger."""

import collections
import dataclasses
import datetime
import enum
import functools
import logging

import jax
import numpy as np
from tensorboardX import writer

try:
  # pylint: disable=g-import-not-at-top
  # pytype: disable=import-error
  import wandb
except ImportError:
  wandb = None
  logging.info("Could not import `wandb`. Logging to W&B not possible.")

_DEFAULT_STEP = 0


@dataclasses.dataclass
class MetricsLoggerOptions:
  """Metrics Logger options."""

  log_dir: str
  flush_every_n_steps: int = 100


class Mode(str, enum.Enum):
  TRAIN = "train"
  EVAL = "eval"

  def __str__(self):
    return self.value


def _get_step(kwargs: dict[str, str | int]) -> int:
  """Returns the step from the kwargs, or 0 if not provided."""
  step = kwargs.get("step")
  return _DEFAULT_STEP if step is None else int(step)


def _preprocess_event_name(event_name: str) -> str:
  """Preprocesses the event name before logging."""
  return event_name.lstrip("/")  # Remove leading slashes


def log_to_tensorboard(
    summary_writer: writer.SummaryWriter,
    flush_every_n_steps: int,
    event: str,
    scalar_value: float | np.ndarray,
    **kwargs: str | int,
):
  """Creates a TensorBoard event listener for jax.monitoring.

  Requires partial application of the first two arguments.

  Args:
    summary_writer: TensorBoard summary writer.
    flush_every_n_steps: Flush the summary writer every n steps.
    event: The name of the event.
    scalar_value: The value of the event.
    **kwargs: Additional keyword arguments, including 'step'.

  Raises:
    ValueError: If 'step' is not provided in `kwargs`.
  """
  current_step = _get_step(kwargs)
  event = _preprocess_event_name(event)
  summary_writer.add_scalar(event, scalar_value, current_step)
  if current_step % flush_every_n_steps == 0:
    summary_writer.flush()


def log_to_wandb(
    event: str,
    scalar_value: float | np.ndarray,
    **kwargs: str | int,
):
  """Creates a W&B event listener for jax.monitoring.

  Args:
    event: The name of the event.
    scalar_value: The value of the event.
    **kwargs: Additional keyword arguments, including 'step'.

  Raises:
    ValueError: If 'step' is not provided in `kwargs`.
  """
  current_step = _get_step(kwargs)

  if wandb is not None:
    event = _preprocess_event_name(event)
    wandb.log({event: scalar_value}, step=current_step)


def register_jax_monitoring(metrics_logger_options: MetricsLoggerOptions):
  """Registers jax.monitoring event listeners to JAX.

  Args:
    metrics_logger_options: Options for configuring the metrics logger.

  Returns:
    A list containing registered metric writers. Currently only returns a
    single TensorBoard Summary Writer instance.
  """
  # Only register the writer on the main process to avoid race conditions.
  if jax.process_index() != 0:
    return []
  # Register TensorBoard backend.
  tensorboard_summary_writer = writer.SummaryWriter(
      logdir=metrics_logger_options.log_dir
  )
  jax.monitoring.register_scalar_listener(
      functools.partial(
          log_to_tensorboard,
          tensorboard_summary_writer,
          metrics_logger_options.flush_every_n_steps,
      )
  )
  # Register Weights & Biases backend.
  if wandb is not None:
    wandb_run_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    wandb.init(project="tunix", name=wandb_run_name, anonymous="allow")
    if wandb.run:
      logging.info("W&B run URL: %s", wandb.run.url)
    jax.monitoring.register_scalar_listener(log_to_wandb)
  return [tensorboard_summary_writer]


def _calculate_geometric_mean(x: np.ndarray) -> np.ndarray:
  """Calculates geometric mean of a batch of values."""
  return np.exp(np.mean(np.log(x)))


class MetricsLogger:
  """Simple Metrics logger."""

  def __init__(
      self,
      metrics_logger_options: MetricsLoggerOptions | None = None,
      metric_prefix: str = "",
  ):
    self._metrics = {
        Mode.TRAIN: collections.defaultdict(list),
        Mode.EVAL: collections.defaultdict(list),
    }

    self._summary_writers = None
    if metrics_logger_options:
      self._summary_writers = register_jax_monitoring(metrics_logger_options)

    self.metric_prefix = metric_prefix

  def log(
      self,
      metric_name: str,
      scalar_value: float | np.ndarray,
      mode: Mode | str,
      step: int,
  ):
    """Logs the scalar metric value for the given metric name and mode."""
    self._metrics[mode][metric_name].append(scalar_value)
    jax.monitoring.record_scalar(
        f"{self.metric_prefix}{mode}/{metric_name}", scalar_value, step=step
    )

  def metric_exists(self, metric_name: str, mode: Mode | str) -> bool:
    """Checks if the metric exists for the given metric name and mode."""
    return metric_name in self._metrics[mode]

  def get_metric(self, metric_name: str, mode: Mode | str):
    """Returns the mean metric value for the given metric name and mode."""
    if metric_name not in self._metrics[mode]:
      raise ValueError(
          f"Metric {metric_name} not found for mode {mode}. Available metrics"
          f" for mode {mode}: {self._metrics[mode].keys()}"
      )
    if metric_name == "perplexity":
      return _calculate_geometric_mean(
          np.stack(self._metrics[mode][metric_name])
      )
    return np.mean(np.stack(self._metrics[mode][metric_name]))

  def get_metric_history(self, metric_name: str, mode: Mode | str):
    """Returns the all past metric values for the given metric name and mode."""
    if metric_name not in self._metrics[mode]:
      raise ValueError(
          f"Metric {metric_name} not found for mode {mode}. Available metrics"
          f" for mode {mode}: {self._metrics[mode].keys()}"
      )
    return np.stack(self._metrics[mode][metric_name])

  def close(self):
    """Closes the metrics logger."""
    try:
      # Either unregister our listeners with Jax private API, or just clear all
      # the listeners. No perfect solution here.
      jax.monitoring.clear_event_listeners()
    except Exception:  # pylint: disable=broad-exception-caught
      # We didn't register the scalar listener, so this is expected.
      pass

    if self._summary_writers:
      # TODO(b/413717077): Solution for destructing lister in jax.monitoring.
      for summary_writer in self._summary_writers:
        summary_writer.close()
    if wandb is not None:
      wandb.finish()
