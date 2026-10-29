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
from opentelemetry import metrics as otel_metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.util import _once
from tunix.dist_rl.common import otel_setup


def _reset_global_meter_provider() -> None:
  """Resets the process-global OpenTelemetry MeterProvider between tests."""
  # pylint: disable=protected-access
  otel_metrics._internal._METER_PROVIDER = None
  otel_metrics._internal._METER_PROVIDER_SET_ONCE = _once.Once()
  # pylint: enable=protected-access


class OtelSetupTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    _reset_global_meter_provider()
    self.addCleanup(_reset_global_meter_provider)

  def test_disabled_without_endpoint(self):
    with mock.patch.dict(os.environ):
      os.environ.pop("OTEL_EXPORTER_OTLP_ENDPOINT", None)
      self.assertFalse(otel_setup.setup_metrics(default_service_name="x"))

  def test_disabled_with_warning_when_exporter_missing(self):
    # The endpoint is set but the OTLP exporter is not importable: export
    # stays off, the default backends keep working, and the operator is told.
    env = {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://127.0.0.1:1"}
    missing = {"opentelemetry.exporter.otlp.proto.grpc.metric_exporter": None}
    with (
        mock.patch.dict(os.environ, env),
        mock.patch.dict(sys.modules, missing),
    ):
      with self.assertLogs(level="WARNING") as logs:
        self.assertFalse(otel_setup.setup_metrics(default_service_name="x"))
    self.assertTrue(any("google-tunix[otel]" in line for line in logs.output))

  def test_installs_sdk_provider_with_service_name_and_resource_attributes(
      self,
  ):
    env = {
        "OTEL_EXPORTER_OTLP_ENDPOINT": "http://127.0.0.1:1",  # nothing listens
        "OTEL_METRIC_EXPORT_INTERVAL": "60000",
        "OTEL_EXPORTER_OTLP_TIMEOUT": "1",
        "OTEL_SERVICE_NAME": "unit-test-orch",
    }
    with mock.patch.dict(os.environ, env):
      self.assertTrue(
          otel_setup.setup_metrics(
              default_service_name="default",
              resource_attributes={
                  "tunix.worker_id": "rollout_0",
                  "tunix.role": "rollout",
                  "tunix.run_id": "run_42",
                  "tunix.empty": "",
                  "service.name": "ignored-override",
              },
          )
      )

    provider = otel_metrics.get_meter_provider()
    self.assertIsInstance(provider, MeterProvider)
    # pylint: disable=protected-access
    resource = provider._sdk_config.resource
    self.assertEqual(resource.attributes["service.name"], "unit-test-orch")
    self.assertEqual(resource.attributes["tunix.worker_id"], "rollout_0")
    self.assertEqual(resource.attributes["tunix.role"], "rollout")
    self.assertEqual(resource.attributes["tunix.run_id"], "run_42")
    self.assertNotIn("tunix.empty", resource.attributes)
    reader = provider._metric_readers[0]
    self.assertEqual(reader._export_interval_millis, 60_000)
    # pylint: enable=protected-access
    # Shut down here so the exit-time drain does not stall on the dead endpoint.
    atexit.unregister(provider.shutdown)
    provider.shutdown(timeout_millis=1500)

  def test_uses_default_service_name_and_interval_when_env_blank(self):
    env = {
        "OTEL_EXPORTER_OTLP_ENDPOINT": "http://127.0.0.1:1",
        "OTEL_METRIC_EXPORT_INTERVAL": "  ",
        "OTEL_EXPORTER_OTLP_TIMEOUT": "1",
        "OTEL_SERVICE_NAME": "",
    }
    with mock.patch.dict(os.environ, env):
      self.assertTrue(
          otel_setup.setup_metrics(default_service_name="tunix-trainer")
      )

    provider = otel_metrics.get_meter_provider()
    self.assertIsInstance(provider, MeterProvider)
    # pylint: disable=protected-access
    resource = provider._sdk_config.resource
    self.assertEqual(resource.attributes["service.name"], "tunix-trainer")
    reader = provider._metric_readers[0]
    self.assertEqual(reader._export_interval_millis, 10_000)
    # pylint: enable=protected-access
    atexit.unregister(provider.shutdown)
    provider.shutdown(timeout_millis=1500)

  def test_rejects_invalid_or_non_positive_export_interval(self):
    for bad_interval in ("abc", "0", "-100"):
      with self.subTest(interval=bad_interval):
        env = {
            "OTEL_EXPORTER_OTLP_ENDPOINT": "http://127.0.0.1:1",
            "OTEL_METRIC_EXPORT_INTERVAL": bad_interval,
        }
        with mock.patch.dict(os.environ, env):
          with self.assertRaisesRegex(
              ValueError,
              "OTEL_METRIC_EXPORT_INTERVAL must be a positive integer",
          ):
            otel_setup.setup_metrics(default_service_name="tunix-trainer")

  def test_idempotent_when_sdk_meter_provider_already_installed(self):
    existing_provider = MeterProvider()
    env = {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://127.0.0.1:1"}
    with (
        mock.patch.dict(os.environ, env),
        mock.patch.object(
            otel_metrics, "get_meter_provider", return_value=existing_provider
        ),
        mock.patch.object(otel_metrics, "set_meter_provider") as mock_set,
    ):
      self.assertTrue(
          otel_setup.setup_metrics(default_service_name="tunix-orchestrator")
      )
    mock_set.assert_not_called()
    atexit.unregister(existing_provider.shutdown)
    provider = existing_provider
    provider.shutdown()

  def test_logs_non_blocking_warning_when_collector_unreachable(self):
    env = {
        "OTEL_EXPORTER_OTLP_ENDPOINT": "http://127.0.0.1:1",
        "OTEL_METRIC_EXPORT_INTERVAL": "60000",
        "OTEL_EXPORTER_OTLP_TIMEOUT": "1",
    }
    with mock.patch.dict(os.environ, env):
      with self.assertLogs(level="WARNING") as logs:
        self.assertTrue(
            otel_setup.setup_metrics(default_service_name="tunix-rollout")
        )
    self.assertTrue(
        any(
            "OpenTelemetry collector at http://127.0.0.1:1 (127.0.0.1:1) is"
            " unreachable from service.name=tunix-rollout"
            in line
            for line in logs.output
        )
    )
    provider = otel_metrics.get_meter_provider()
    self.assertIsInstance(provider, MeterProvider)
    atexit.unregister(provider.shutdown)
    provider.shutdown(timeout_millis=1500)


if __name__ == "__main__":
  absltest.main()
