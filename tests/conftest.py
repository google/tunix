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

This module also parses absl flags so that ``absltest`` helpers work under
bare ``pytest`` (see below).
"""

import os
import sys

from absl import flags
from absl.testing import absltest  # pylint: disable=unused-import
import pytest

os.environ.setdefault("WANDB_MODE", "disabled")

# ``absltest.TestCase.create_tempdir()`` reads the ``--test_tmpdir`` flag, but
# under ``pytest`` nothing ever calls ``absltest.main()`` / ``app.run()``, so
# absl flags stay unparsed and any access raises ``UnparsedFlagAccessError``.
# Parse them once here (argv[:1] means "defaults only", ``known_only`` ignores
# pytest's own arguments) so tests can use ``create_tempdir()`` directly.
if not flags.FLAGS.is_parsed():
  flags.FLAGS(sys.argv[:1], known_only=True)


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
