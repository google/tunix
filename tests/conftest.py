"""Shared test configuration.

The test environment installs ``wandb`` for the OpenTelemetry W&B exporter
end-to-end test. Installing it has a side effect on the legacy path: the
default Metrax ``WandbBackend`` becomes constructible, registers a
process-global ``jax.monitoring`` listener that receives JAX's own internal
compile metrics, and shares the module-global ``wandb.run`` across logger
instances — one logger's ``close()`` (``wandb.finish()``) then breaks any
other still-registered backend in the process.

To keep the legacy default-backend behavior identical to an environment
without ``wandb`` (which is how CI has always run), the default
``WandbBackend`` is replaced with a stub that raises ``ImportError``, which
``MetricsLoggerOptions.create_backends`` already handles by skipping the
backend. Tests that exercise wandb do so explicitly: the OpenTelemetry
exporter tests pass an offline ``wandb.init`` run object directly and never
touch the Metrax backend.
"""

import os

import pytest

os.environ.setdefault("WANDB_MODE", "disabled")


class _WandbBackendUnavailable:

  def __init__(self, *args, **kwargs):
    raise ImportError(
        "The default Metrax WandbBackend is disabled in tests; construct a"
        " wandb run explicitly instead."
    )


@pytest.fixture(autouse=True)
def _disable_default_wandb_backend(monkeypatch):
  monkeypatch.setattr(
      "tunix.sft.metrics_logger.WandbBackend", _WandbBackendUnavailable
  )
  yield
