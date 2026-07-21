"""Metric logger with a unified, protocol-based backend system.

In addition to the Metrax logging backends, ``MetricsLogger`` supports an
opt-in OpenTelemetry double-write path: when
``MetricsLoggerOptions.enable_opentelemetry`` is set, every scalar logged
through ``log`` is emitted both through the existing ``jax.monitoring``
backends and as an OpenTelemetry gauge measurement. The existing backends
remain the default and are unaffected by the flag.

Tunix does not configure or shut down OpenTelemetry providers. Applications
configure the global meter provider (readers, exporters) or inject a provider
when constructing a ``MetricsLogger``.
"""

import collections
import dataclasses
import enum
import importlib.metadata
import re
import threading
from typing import Any, Callable

from absl import logging
import jax
from metrax import logging as metrax_logging
import numpy as np
from tunix.utils import env_utils

try:
  from opentelemetry import metrics as otel_metrics  # pylint: disable=g-import-not-at-top
except ImportError:
  otel_metrics = None

LoggingBackend = metrax_logging.LoggingBackend
TensorboardBackend = metrax_logging.TensorboardBackend
WandbBackend = metrax_logging.WandbBackend
CluBackend = getattr(metrax_logging, "CluBackend", None)

# User backends MUST be factories (callables) to keep Options pure and copyable.
BackendFactory = Callable[[], LoggingBackend]

_OTEL_INSTRUMENTATION_NAME = "tunix"
_OTEL_METRIC_NAME_PATTERN = re.compile(r"[^a-zA-Z0-9_.-]+")

try:
  _OTEL_INSTRUMENTATION_VERSION = importlib.metadata.version("google-tunix")
except importlib.metadata.PackageNotFoundError:
  _OTEL_INSTRUMENTATION_VERSION = None


@dataclasses.dataclass(frozen=True, slots=True)
class _OtelMetricSpec:
  name: str
  unit: str = "1"
  description: str = ""


_OTEL_KNOWN_METRICS = {
    "loss": _OtelMetricSpec(
        name="tunix.training.loss",
        description="Loss reported by a Tunix training or evaluation step.",
    ),
    "perplexity": _OtelMetricSpec(
        name="tunix.training.perplexity",
        description=(
            "Perplexity reported by a Tunix training or evaluation step."
        ),
    ),
    "learning_rate": _OtelMetricSpec(
        name="tunix.training.learning_rate",
        description="Learning rate used by the Tunix optimizer.",
    ),
    "grad_norm": _OtelMetricSpec(
        name="tunix.training.gradient.norm",
        description="Gradient norm reported by a Tunix training step.",
    ),
}

_OTEL_STEP_METRIC = _OtelMetricSpec(
    name="tunix.training.step",
    unit="{step}",
    description="Current logical Tunix training or evaluation step.",
)


def _normalize_otel_metric_name(metric_name: str) -> str:
  """Builds a valid, namespaced OpenTelemetry instrument name."""
  normalized = _OTEL_METRIC_NAME_PATTERN.sub(".", metric_name).strip(".")
  if not normalized:
    normalized = "unnamed"
  return f"tunix.{normalized.lower()}"


def _otel_metric_spec(metric_name: str) -> _OtelMetricSpec:
  """Returns the stable specification for a Tunix metric."""
  return _OTEL_KNOWN_METRICS.get(
      metric_name,
      _OtelMetricSpec(
          name=_normalize_otel_metric_name(metric_name),
          description=f"Tunix metric derived from {metric_name!r}.",
      ),
  )


def _as_otel_number(value: Any) -> int | float | None:
  """Converts a scalar value to an OpenTelemetry-compatible number."""
  try:
    array = np.asarray(value)
  except Exception:  # pylint: disable=broad-exception-caught
    return None
  if array.size != 1:
    return None
  scalar = array.reshape(()).item()
  if isinstance(scalar, (bool, np.bool_)):
    return None
  if isinstance(scalar, (int, float, np.integer, np.floating)):
    return scalar.item() if isinstance(scalar, np.generic) else scalar
  return None


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
  # Experimental: additionally emit every logged scalar as an OpenTelemetry
  # gauge measurement (double-write). The Metrax backends above keep working
  # unchanged; this flag only adds a second emission path. Requires the
  # OpenTelemetry API ("pip install google-tunix[otel]"). Exporters and
  # provider lifecycle are owned by the application, not by Tunix.
  enable_opentelemetry: bool = False

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
  the default backends. When ``MetricsLoggerOptions.enable_opentelemetry`` is
  set, every scalar is additionally emitted as an OpenTelemetry gauge
  measurement (double-write) without changing the backend behavior.
  """

  def __init__(
      self,
      metrics_logger_options: MetricsLoggerOptions | None = None,
      *,
      otel_meter_provider: Any = None,
  ):
    """Initializes the metrics logger.

    Args:
      metrics_logger_options: Logger configuration. ``None`` disables backend
        creation and OpenTelemetry emission; local metric history still works.
      otel_meter_provider: Optional OpenTelemetry ``MeterProvider`` used when
        ``enable_opentelemetry`` is set. Defaults to the global provider. This
        keyword-only argument exists for tests and embedding applications.
    """
    self._metrics = {}
    self._backends = (
        metrics_logger_options.create_backends()
        if metrics_logger_options
        else []
    )
    if metrics_logger_options and jax.process_index() == 0:
      for backend in self._backends:
        jax.monitoring.register_scalar_listener(backend.log_scalar)

    self._otel_meter = None
    self._otel_instruments: dict[str, Any] = {}
    self._otel_instrument_lock = threading.Lock()
    self._otel_last_steps: dict[tuple[str, str], int] = {}
    if metrics_logger_options and metrics_logger_options.enable_opentelemetry:
      if otel_metrics is None:
        raise ImportError(
            "MetricsLoggerOptions.enable_opentelemetry is set but the"
            " OpenTelemetry API is not installed. Install it with: pip install"
            " 'google-tunix[otel]'."
        )
      provider = otel_meter_provider or otel_metrics.get_meter_provider()
      self._otel_meter = provider.get_meter(
          _OTEL_INSTRUMENTATION_NAME,
          _OTEL_INSTRUMENTATION_VERSION,
      )

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

    if self._otel_meter is not None:
      # The OpenTelemetry double-write is best effort: emission failures must
      # not interrupt training or the primary logging path.
      try:
        self._emit_otel_metric(
            metrics_prefix, metric_name, scalar_value, mode, step
        )
      except Exception:  # pylint: disable=broad-exception-caught
        logging.exception(
            "Failed to emit OpenTelemetry metric %s/%s/%s.",
            metrics_prefix,
            mode,
            metric_name,
        )

  def _otel_gauge(self, spec: _OtelMetricSpec):
    """Returns a cached synchronous gauge for a metric specification."""
    with self._otel_instrument_lock:
      gauge = self._otel_instruments.get(spec.name)
      if gauge is None:
        gauge = self._otel_meter.create_gauge(
            spec.name,
            unit=spec.unit,
            description=spec.description,
        )
        self._otel_instruments[spec.name] = gauge
      return gauge

  def _emit_otel_metric(
      self,
      metrics_prefix: str,
      metric_name: str,
      scalar_value: Any,
      mode: Mode | str,
      step: int,
  ) -> None:
    """Emits one OpenTelemetry metric value and its associated logical step."""
    # Match the backend policy: only the main process emits telemetry.
    if jax.process_index() != 0:
      return

    value = _as_otel_number(scalar_value)
    if value is None:
      logging.warning(
          "Skipping non-scalar OpenTelemetry metric %s/%s/%s with shape %s.",
          metrics_prefix,
          mode,
          metric_name,
          np.shape(scalar_value),
      )
      return

    attributes = {"tunix.training.mode": str(mode)}
    if metrics_prefix:
      attributes["tunix.metrics.prefix"] = metrics_prefix
    self._otel_gauge(_otel_metric_spec(metric_name)).set(value, attributes)

    # The logical step is a separate gauge rather than a metric attribute: an
    # attribute value per step would create unbounded time-series cardinality.
    step_key = (metrics_prefix, str(mode))
    with self._otel_instrument_lock:
      if self._otel_last_steps.get(step_key) == step:
        return
      self._otel_last_steps[step_key] = step
    self._otel_gauge(_OTEL_STEP_METRIC).set(step, attributes)

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
    """Closes all registered logging backends.

    OpenTelemetry providers are application-owned process-wide resources; the
    application that configured them owns flushing and shutdown, so ``close``
    leaves them running.
    """
    for backend in self._backends:
      backend.close()
    try:
      jax.monitoring.clear_event_listeners()
    except Exception:  # pylint: disable=broad-exception-caught
      # We didn't register the scalar listener, so this is expected.
      pass
