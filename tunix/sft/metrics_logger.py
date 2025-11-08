"""Metric logger with a unified, protocol-based backend system."""

import collections
import dataclasses
import enum
import logging
from typing import Callable

import jax
from metrax import logging as metrax_logging
import numpy as np

LoggingBackend = metrax_logging.LoggingBackend
TensorboardBackend = metrax_logging.TensorboardBackend
WandbBackend = metrax_logging.WandbBackend

# User backends MUST be factories (callables) to keep Options pure and copyable.
BackendFactory = Callable[[], LoggingBackend]


@dataclasses.dataclass
class MetricsLoggerOptions:
  """Metrics Logger options."""

  log_dir: str
  flush_every_n_steps: int = 100
  backend_factories: list[BackendFactory] | None = None

  def create_backends(self) -> list[LoggingBackend]:
    """Factory method to create a fresh set of live backends."""
    # Only create live backends on the main process.
    if jax.process_index() != 0:
      return []

    # Case 1: Override. Use user-provided factories.
    if self.backend_factories:
      return [factory() for factory in self.backend_factories]

    # Case 2: Defaults.
    active_backends = [
        TensorboardBackend(
            log_dir=self.log_dir,
            flush_every_n_steps=self.flush_every_n_steps,
        )
    ]
    try:
      active_backends.append(WandbBackend(project="tunix"))
    except ImportError:
      logging.info("WandbBackend skipped: 'wandb' library not installed.")
    return active_backends


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
  ):
    self._metrics = {
        Mode.TRAIN: collections.defaultdict(list),
        Mode.EVAL: collections.defaultdict(list),
    }
    self.metric_prefix = metric_prefix
    self._backends = (
        metrics_logger_options.create_backends()
        if metrics_logger_options
        else []
    )
    if metrics_logger_options and jax.process_index() == 0:
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
    try:
      jax.monitoring.clear_event_listeners()
    except Exception:  # pylint: disable=broad-exception-caught
      # We didn't register the scalar listener, so this is expected.
      pass
