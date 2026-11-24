"""Metric logger with a unified, protocol-based backend system."""

import collections
from collections.abc import Callable
import dataclasses
import enum
from typing import Any

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
      return [factory() for factory in self.backend_kwargs["custom_backend"]]  # pyrefly: ignore[not-callable]

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
              **tb_kwargs,  # pyrefly: ignore[bad-unpacking]
          )
      )
      try:
        wandb_kwargs = kwargs_dict.get("wandb", {})
        active_backends.append(
            WandbBackend(
                project=self.project_name,
                name=self.run_name,
                **wandb_kwargs,  # pyrefly: ignore[bad-unpacking]
            )
        )
      except (ImportError, Exception) as e:
        logging.info("WandbBackend skipped: %s", e)
    return active_backends


class Mode(str, enum.Enum):
  TRAIN = "train"
  EVAL = "eval"

  def __str__(self):
    return self.value


def _calculate_geometric_mean(x: np.ndarray) -> np.ndarray:
  """Calculates geometric mean of a batch of values."""
  return np.exp(np.mean(np.log(x)))


# Mirrors `tunix.sft.utils._WEIGHTED_METRIC_FIELDS` / `WeightedMetricLike`
# without pulling `flax`/`nnx` dependencies into `metrics_logger`.
_WEIGHTED_METRIC_FIELDS = ("unreduced_sum", "denominator", "eps", "min_denom")

# Metric namespaces that already carry their own top-level prefix and should be
# logged directly under that namespace rather than nested under `actor/` or
# `metrics_prefix`.
TOP_LEVEL_METRIC_PREFIXES = (
    "sampler_trainer/",
    "sampler_is/",
    "sampler_rs/",
    "score_centering/",
)


def default_scalar_reducer(name: str) -> Callable[[Any], Any]:
  """Returns the canonical cross-microbatch reducer for a scalar metric name.

  Args:
    name: Metric key (with or without namespace prefix, e.g.
      ``"sampler_trainer/logp_diff_max"`` or ``"is_ratio/min"``).

  Returns:
    ``np.max`` for ``_max``/``/max`` keys, ``np.min`` for ``_min``/``/min``
    keys, ``np.sum`` for ``_count``/``/count`` keys, and ``np.mean`` otherwise.
  """
  if name.endswith(("_max", "/max")):
    return np.max
  if name.endswith(("_min", "/min")):
    return np.min
  if name.endswith(("_count", "/count")):
    return np.sum
  return np.mean


def _reduce_weighted_metric(val: Any) -> float | None:
  """Reduces a single or microbatch-stacked `WeightedMetricLike` to a float."""
  reduce_fn = getattr(type(val), "reduce", None)
  try:
    if callable(reduce_fn):
      return float(reduce_fn([val]))
    num = float(np.sum(np.asarray(val.unreduced_sum, dtype=np.float64)))
    denom = float(np.sum(np.asarray(val.denominator, dtype=np.float64)))
    if val.eps is not None:
      denom += float(val.eps)
    if val.min_denom is not None:
      denom = max(denom, float(val.min_denom))
    return float(num / denom) if denom else 0.0
  except Exception as exc:  # pylint: disable=broad-exception-caught
    logging.warning(
        "Failed to reduce weighted metric %s: %s", type(val).__name__, exc
    )
    return None


def extract_scalar(
    val: Any,
    aggregation_fn: Callable[[Any], Any] | None = None,
) -> float | None:
  """Safely extracts a scalar float from arrays, tensors, or WeightedMetrics.

  Args:
    val: Scalar, NumPy/JAX array, ``WeightedMetricLike`` (including
      microbatch-stacked ``WeightedMetric`` or ``PearsonMetric``), or metric
      object exposing ``compute()``.
    aggregation_fn: Optional reduction applied to non-weighted scalar/array
      values (e.g. ``np.max``, ``np.min``, ``np.sum``). When ``None``,
      multi-element arrays default to ``np.mean``.

  Returns:
    The reduced Python ``float``, or ``None`` if ``val`` is ``None``, empty, or
    cannot be computed/converted.
  """
  if val is None:
    return None
  if all(hasattr(val, field) for field in _WEIGHTED_METRIC_FIELDS):
    return _reduce_weighted_metric(val)
  if hasattr(val, "compute") and callable(val.compute):
    try:
      val = val.compute()
    except Exception:  # pylint: disable=broad-exception-caught
      return None
  try:
    arr = np.asarray(val)
    if arr.size == 0:
      return None
    if aggregation_fn is not None:
      arr = np.asarray(aggregation_fn(arr))
      if arr.size == 0:
        return None
    return float(arr.item()) if arr.size == 1 else float(np.mean(arr))
  except Exception:  # pylint: disable=broad-exception-caught
    return None


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
        f"{metrics_prefix}/{mode}/{metric_name}", scalar_value, step=step  # pyrefly: ignore[bad-argument-type]
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
