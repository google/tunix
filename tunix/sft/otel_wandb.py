"""OpenTelemetry metric exporter for Weights & Biases runs.

Weights & Biases natively ingests OpenTelemetry traces (through the Weave OTLP
endpoint) but has no OTLP endpoint for run metrics. This module provides an
OpenTelemetry SDK ``MetricExporter`` that forwards Tunix gauge measurements to
``wandb.log`` so the OpenTelemetry double-write path of
``tunix.sft.metrics_logger.MetricsLogger`` can feed a W&B run.

Requires the ``opentelemetry-sdk`` (``pip install 'google-tunix[otel]'``) and
``wandb`` packages. Example:

.. code-block:: python

  import wandb
  from opentelemetry.sdk.metrics import MeterProvider
  from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
  from tunix.sft import metrics_logger
  from tunix.sft import otel_wandb

  run = wandb.init(project="my-project")
  reader = PeriodicExportingMetricReader(
      otel_wandb.WandbMetricsExporter(run), export_interval_millis=10_000
  )
  provider = MeterProvider(metric_readers=[reader])

  options = metrics_logger.MetricsLoggerOptions(
      log_dir="/tmp/logs", enable_opentelemetry=True
  )
  logger = metrics_logger.MetricsLogger(
      options, otel_meter_provider=provider
  )
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

from absl import logging
from opentelemetry.sdk.metrics import export as otel_export

_STEP_METRIC_NAME = "tunix.training.step"
_PREFIX_ATTRIBUTE = "tunix.metrics.prefix"
_MODE_ATTRIBUTE = "tunix.training.mode"


def _wandb_key(prefix: str, mode: str, metric_name: str) -> str:
  """Builds a W&B chart key mirroring the legacy `{prefix}/{mode}/{name}`."""
  return "/".join(part for part in (prefix, mode, metric_name) if part)


def _wandb_payloads(
    metrics_data: otel_export.MetricsData,
) -> Iterator[tuple[int | None, dict[str, Any]]]:
  """Groups gauge data points into per-step W&B log payloads.

  Data points are grouped by their Tunix prefix and mode attributes so each
  group can be logged against the logical training step carried by the
  ``tunix.training.step`` gauge of the same group. Groups are yielded in
  ascending step order because W&B requires non-decreasing steps.

  Args:
    metrics_data: One OpenTelemetry metric export batch.

  Yields:
    ``(step, values)`` pairs, where ``step`` may be ``None`` when the batch
    carries no step gauge for the group.
  """
  groups: dict[tuple[str, str], dict[str, Any]] = {}
  for resource_metrics in metrics_data.resource_metrics:
    for scope_metrics in resource_metrics.scope_metrics:
      for metric in scope_metrics.metrics:
        for point in getattr(metric.data, "data_points", ()):
          attributes: Mapping[str, Any] = point.attributes or {}
          group_key = (
              str(attributes.get(_PREFIX_ATTRIBUTE, "")),
              str(attributes.get(_MODE_ATTRIBUTE, "")),
          )
          group = groups.setdefault(group_key, {"step": None, "values": {}})
          if metric.name == _STEP_METRIC_NAME:
            group["step"] = int(point.value)
          else:
            key = _wandb_key(group_key[0], group_key[1], metric.name)
            group["values"][key] = point.value
  ordered = sorted(
      (group for group in groups.values() if group["values"]),
      key=lambda group: (group["step"] is None, group["step"]),
  )
  for group in ordered:
    yield group["step"], group["values"]


class WandbMetricsExporter(otel_export.MetricExporter):
  """Exports OpenTelemetry gauge data points to a Weights & Biases run.

  Chart keys mirror the legacy Metrax backend layout,
  ``{prefix}/{mode}/{instrument name}`` (for example
  ``actor/train/tunix.training.loss``), so double-written runs are easy to
  compare side by side. The ``tunix.training.step`` gauge provides the W&B
  step for the other measurements in its group.
  """

  def __init__(self, run: Any = None, **kwargs):
    """Initializes the exporter.

    Args:
      run: The W&B run to log to (the object returned by ``wandb.init``).
        When ``None``, the active global ``wandb.run`` is resolved at export
        time.
      **kwargs: Forwarded to ``MetricExporter`` (for example
        ``preferred_temporality``).
    """
    super().__init__(**kwargs)
    self._run = run

  def _resolve_run(self) -> Any:
    if self._run is not None:
      return self._run
    import wandb  # pylint: disable=g-import-not-at-top

    if wandb.run is None:
      raise RuntimeError(
          "WandbMetricsExporter has no W&B run to log to. Pass a run to the"
          " exporter or call wandb.init() before exporting."
      )
    return wandb.run

  def export(
      self,
      metrics_data: otel_export.MetricsData,
      timeout_millis: float = 10_000,
      **kwargs,
  ) -> otel_export.MetricExportResult:
    """Logs one metric export batch to the W&B run."""
    del timeout_millis, kwargs
    try:
      run = self._resolve_run()
      for step, values in _wandb_payloads(metrics_data):
        run.log(values, step=step)
    except Exception:  # pylint: disable=broad-exception-caught
      logging.exception("Failed to export OpenTelemetry metrics to W&B.")
      return otel_export.MetricExportResult.FAILURE
    return otel_export.MetricExportResult.SUCCESS

  def force_flush(self, timeout_millis: float = 10_000) -> bool:
    """Nothing to flush; ``export`` logs synchronously."""
    del timeout_millis
    return True

  def shutdown(self, timeout_millis: float = 30_000, **kwargs) -> None:
    """Leaves the W&B run open; the application owns ``run.finish()``."""
    del timeout_millis, kwargs
