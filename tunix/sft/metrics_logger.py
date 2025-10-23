"""Metric logger with a unified, protocol-based backend system."""

import collections
import dataclasses
import enum

import jax
from metrax import logging as metrax_logging
import numpy as np

LoggingBackend = metrax_logging.LoggingBackend
TensorBoardBackend = metrax_logging.TensorBoardBackend
WandbBackend = metrax_logging.WandbBackend


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


def _calculate_geometric_mean(x: np.ndarray) -> np.ndarray:
  """Calculates geometric mean of a batch of values."""
  return np.exp(np.mean(np.log(x)))


class MetricsLogger:
  """Simple Metrics logger."""

  def __init__(
      self,
      metrics_logger_options: MetricsLoggerOptions | None = None,
      metric_prefix: str = "",
      additional_backends: list[LoggingBackend] | None = None,
  ):
    self._metrics = {
        Mode.TRAIN: collections.defaultdict(list),
        Mode.EVAL: collections.defaultdict(list),
    }
    self.metric_prefix = metric_prefix

    all_backends = additional_backends or []
    if metrics_logger_options:
      # Use Tensorboard and W&B backends by default.
      all_backends.append(
          TensorBoardBackend(
              log_dir=metrics_logger_options.log_dir,
              flush_every_n_steps=metrics_logger_options.flush_every_n_steps,
          )
      )
      all_backends.append(WandbBackend(project="tunix"))
    self._backends = all_backends
    if jax.process_index() == 0:
      for backend in self._backends:
        jax.monitoring.register_scalar_listener(backend.log_scalar)

  def log(
      self,
      metric_name: str,
      scalar_value: float | np.ndarray,
      mode: Mode | str,
      step: int,
  ):
    """Logs the scalar metric value to local history and via jax.monitoring."""
    self._metrics[mode][metric_name].append(scalar_value)
    jax.monitoring.record_scalar(
        f"{self.metric_prefix}{mode}/{metric_name}", scalar_value, step=step
    )

  def metric_exists(self, metric_name: str, mode: Mode | str) -> bool:
    """Checks if the metric exists for the given metric name and mode."""
    return metric_name in self._metrics[mode]

  def get_metric(self, metric_name: str, mode: Mode | str):
    """Returns the mean metric value for the given metric name and mode."""
    if not self.metric_exists(metric_name, mode):
      raise ValueError(f"Metric '{metric_name}' not found for mode '{mode}'.")
    values = np.stack(self._metrics[mode][metric_name])
    if metric_name == "perplexity":
      return _calculate_geometric_mean(values)
    return np.mean(values)

  def get_metric_history(self, metric_name: str, mode: Mode | str):
    """Returns all past metric values for the given metric name and mode."""
    if not self.metric_exists(metric_name, mode):
      raise ValueError(
          f"Metric '{metric_name}' not found for mode '{mode}'. Available"
          f" metrics for mode '{mode}': {list(self._metrics[mode].keys())}"
      )
    return np.stack(self._metrics[mode][metric_name])

  def close(self):
    """Closes all registered logging backends."""
    for backend in self._backends:
      backend.close()
