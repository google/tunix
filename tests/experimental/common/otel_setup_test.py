# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests verifying OpenTelemetry MeterProvider initialization and configuration in otel_setup."""

import atexit
import os
import sys
from unittest import mock

from absl.testing import absltest
from tunix.experimental.common import otel_setup


class OtelSetupTest(absltest.TestCase):

  def test_disabled_without_endpoint(self):
    with mock.patch.dict(os.environ):
      os.environ.pop("OTEL_EXPORTER_OTLP_ENDPOINT", None)
      self.assertFalse(otel_setup.setup_metrics(default_service_name="x"))

  def test_disabled_with_warning_when_exporter_missing(self):
    # The endpoint is set but the OTLP exporter is not importable: export
    # stays off, the default backends keep working, and the operator is told.
    env = {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://127.0.0.1:1"}
    missing = {"opentelemetry.exporter.otlp.proto.grpc.metric_exporter": None}
    with mock.patch.dict(os.environ, env), mock.patch.dict(
        sys.modules, missing
    ):
      with self.assertLogs(level="WARNING") as logs:
        self.assertFalse(otel_setup.setup_metrics(default_service_name="x"))
    self.assertTrue(any("google-tunix[otel]" in line for line in logs.output))

  def test_installs_sdk_provider_with_service_name(self):
    try:
      from opentelemetry.exporter.otlp.proto.grpc import metric_exporter  # pylint: disable=g-import-not-at-top,unused-import
    except ImportError:
      self.skipTest("OTLP gRPC exporter not installed")
    from opentelemetry import metrics as otel_metrics  # pylint: disable=g-import-not-at-top
    from opentelemetry.sdk.metrics import MeterProvider  # pylint: disable=g-import-not-at-top

    env = {
        "OTEL_EXPORTER_OTLP_ENDPOINT": "http://127.0.0.1:1",  # nothing listens
        "OTEL_METRIC_EXPORT_INTERVAL": "60000",
        "OTEL_EXPORTER_OTLP_TIMEOUT": "1",
        "OTEL_SERVICE_NAME": "unit-test-orch",
    }
    with mock.patch.dict(os.environ, env):
      self.assertTrue(otel_setup.setup_metrics(default_service_name="default"))

    provider = otel_metrics.get_meter_provider()
    self.assertIsInstance(provider, MeterProvider)
    resource = provider._sdk_config.resource  # pylint: disable=protected-access
    self.assertEqual(resource.attributes["service.name"], "unit-test-orch")
    # Shut down here so the exit-time drain does not stall on the dead endpoint.
    atexit.unregister(provider.shutdown)
    provider.shutdown(timeout_millis=1500)


if __name__ == "__main__":
  absltest.main()
