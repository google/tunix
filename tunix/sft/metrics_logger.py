"""Metric logger with a unified, protocol-based backend system."""

import collections
import dataclasses
import enum
from typing import Any, Callable

from absl import logging
import jax
from metrax import logging as metrax_logging
import numpy as np
from tunix.utils import env_utils

LoggingBackend = metrax_logging.LoggingBackend
TensorboardBackend = metrax_logging.TensorboardBackend
WandbBackend = metrax_logging.WandbBackend
CluBackend = getattr(metrax_logging, "CluBackend", None)

# User backends MUST be factories (callables) to keep Options pure and copyable.
BackendFactory = Callable[[], LoggingBackend]


@dataclasses.dataclass
class MetricsLoggerOptions:
  """Metrics Logger options."""

  log_dir: str
  project_name: str = "tunix"
  run_name: str = ""
  flush_every_n_steps: int = 100
  # Keyword arguments for backend initialization. The key is the backend name
  # (e.g., 'wandb', 'clu', 'tensorboard' or 'custom_backend' which uses custom
  # LoggingBackend factories) and the value is a dictionary of
  # keyword arguments to be passed to the backend's constructor.
  # For example:
  # backend_kwargs={
  #   'wandb': {
  #     'resume': 'must',
  #     'id': '12345',
  #     'project': 'my-project',
  #     'name': 'my-run',
  #   },
  #   'tensorboard': {
  #      'log_dir': '/path/to/log',
  #      'flush_every_n_steps': 100,
  #   }
  # }
  backend_kwargs: dict[str, dict[str, Any] | list[BackendFactory]] = (
      dataclasses.field(default_factory=dict)
  )

  def create_backends(self) -> list[LoggingBackend]:
    """Factory method to create a fresh set of live backends."""
    # Only create live backends on the main process.
    if jax.process_index() != 0:
      return []

    # Case 1: Override. Use user-provided factories.
    if (
        "custom_backend" in self.backend_kwargs
        and self.backend_kwargs["custom_backend"]
    ):
      return [factory() for factory in self.backend_kwargs["custom_backend"]]

    # Case 2: Defaults.
    active_backends = []
    kwargs_dict = self.backend_kwargs or {}

    if env_utils.is_internal_env():
      if CluBackend is None:
        raise ImportError(
            "Internal environment detected, but CluBackend not available."
        )
      clu_kwargs = kwargs_dict.get("clu", {})
      active_backends.append(CluBackend(log_dir=self.log_dir, **clu_kwargs))
    else:
      tb_kwargs = kwargs_dict.get("tensorboard", {})
      active_backends.append(
          TensorboardBackend(
              log_dir=self.log_dir,
              flush_every_n_steps=self.flush_every_n_steps,
              **tb_kwargs,
          )
      )
      try:
        wandb_kwargs = kwargs_dict.get("wandb", {})
        active_backends.append(
            WandbBackend(
                project=self.project_name,
                name=self.run_name,
                **wandb_kwargs,
            )
        )
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
  """Simple Metrics logger.

  Log metrics to multiple backends. If no backends are specified, it will log to
  the default backends.
  """

  def __init__(
      self,
      metrics_logger_options: MetricsLoggerOptions | None = None,
  ):
    self._metrics = {}
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
      metrics_prefix: str,
      metric_name: str,
      scalar_value: float | np.ndarray,
      mode: Mode | str,
      step: int,
  ):
    """Logs the scalar metric value to local history and via jax.monitoring."""
    prefix_metrics = self._metrics.setdefault(metrics_prefix, {})
    mode_metrics = prefix_metrics.setdefault(
        mode, collections.defaultdict(list)
    )
    mode_metrics[metric_name].append(scalar_value)

    jax.monitoring.record_scalar(
        f"{metrics_prefix}/{mode}/{metric_name}", scalar_value, step=step
    )

  def metric_exists(
      self, metrics_prefix, metric_name: str, mode: Mode | str
  ) -> bool:
    """Checks if the metric exists for the given metric name and mode."""
    if metrics_prefix not in self._metrics:
      return False
    if mode not in self._metrics[metrics_prefix]:
      return False
    return metric_name in self._metrics[metrics_prefix][mode]

  def get_metric(self, metrics_prefix, metric_name: str, mode: Mode | str):
    """Returns the mean metric value for the given metric name and mode."""
    if not self.metric_exists(metrics_prefix, metric_name, mode):
      raise ValueError(
          f"Metric '{metrics_prefix}/{mode}/{metric_name}' not found."
      )
    values = np.stack(self._metrics[metrics_prefix][mode][metric_name])
    if metric_name == "perplexity":
      return _calculate_geometric_mean(values)
    return np.mean(values)

  def get_metric_history(
      self, metrics_prefix, metric_name: str, mode: Mode | str
  ):
    """Returns all past metric values for the given metric name and mode."""
    if not self.metric_exists(metrics_prefix, metric_name, mode):
      raise ValueError(
          f" Metric '{metrics_prefix}/{mode}/{metric_name}' not found."
          f" Available metrics for mode '{mode}':"
          f" {list(self._metrics[metrics_prefix][mode].keys())}"
      )
    return np.stack(self._metrics[metrics_prefix][mode][metric_name])

  def close(self):
    """Closes all registered logging backends."""
    for backend in self._backends:
      backend.close()
    try:
      jax.monitoring.clear_event_listeners()
    except Exception:  # pylint: disable=broad-exception-caught
      # We didn't register the scalar listener, so this is expected.
      pass
