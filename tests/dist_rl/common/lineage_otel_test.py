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

"""Unit tests for tunix.dist_rl.common.lineage_otel."""

from typing import Any

from absl.testing import absltest
import numpy as np
from opentelemetry.exporter.otlp.proto.common import _log_encoder
from opentelemetry.sdk import _logs as sdk_logs
from opentelemetry.sdk._logs import export as sdk_logs_export
from tunix.dist_rl.common import lineage
from tunix.dist_rl.common import lineage_otel


class LineageOtelTest(absltest.TestCase):

  def test_emits_sanitized_attributes_drops_invalid_scalars_and_keeps_identity(
      self,
  ):
    exporter = sdk_logs_export.InMemoryLogRecordExporter()
    provider = sdk_logs.LoggerProvider()
    provider.add_log_record_processor(
        sdk_logs_export.SimpleLogRecordProcessor(exporter)
    )
    self.addCleanup(provider.shutdown)
    prev_sink = lineage_otel.install(provider)
    self.addCleanup(lineage.set_event_sink, prev_sink)

    ctx = lineage.LineageContext(
        tracking_id="traj_nan",
        task_name="deepswe",
        parent_tracking_ids=["p0"],
    )
    ev = ctx.add_event(
        "orchestrator.critique",
        "scored",
        {
            "reward": float("nan"),
            "score_inf": float("inf"),
            "huge_int": 1 << 63,
            "underflow_int": -(1 << 63) - 1,
            "valid_score": np.float32(1.5),
            "valid_seq": [np.int64(1), np.int64(2)],
            "bool_seq": [np.bool_(True), False],
            "np_flag": np.bool_(True),
            "empty_seq": [],
            "mixed_seq": [1.0, float("nan")],
            "hetero_int_str": [1, "a"],
            "hetero_bool_int": [True, 1],
            "hetero_np_bool_int": [np.bool_(True), 1],
            "hetero_int_float": [1, 1.5],
            "unsupported_obj": {"nested": 1},
            "flag": True,
            "component": "attempted_override",
            "tracking_id": "attempted_override",
        },
    )
    lineage.LineageContext.merge(
        batch_id="batch_0",
        contexts=[ctx],
        component="orchestrator.assembler",
        operation="pack",
        attributes={"num_items": 1},
    )

    records = [item.log_record for item in exporter.get_finished_logs()]
    self.assertLen(records, 2)
    self.assertEqual(
        records[0].event_name, "tunix.lineage.orchestrator.critique.scored"
    )
    self.assertEqual(
        records[0].attributes["event.name"],
        "tunix.lineage.orchestrator.critique.scored",
    )
    self.assertEqual(records[0].body, {"timestamp_s": ev.timestamp_s})
    self.assertAlmostEqual(records[0].attributes["tunix.valid_score"], 1.5)
    self.assertEqual(records[0].attributes["tunix.valid_seq"], (1, 2))
    self.assertEqual(records[0].attributes["tunix.bool_seq"], (True, False))
    self.assertIs(records[0].attributes["tunix.np_flag"], True)
    self.assertEqual(records[0].attributes["tunix.empty_seq"], ())
    self.assertIs(records[0].attributes["tunix.flag"], True)
    self.assertEqual(
        records[0].attributes["tunix.component"], "orchestrator.critique"
    )
    self.assertEqual(records[0].attributes["tunix.tracking_id"], "traj_nan")
    self.assertNotIn("tunix.reward", records[0].attributes)
    self.assertNotIn("tunix.score_inf", records[0].attributes)
    self.assertNotIn("tunix.huge_int", records[0].attributes)
    self.assertNotIn("tunix.underflow_int", records[0].attributes)
    self.assertNotIn("tunix.mixed_seq", records[0].attributes)
    self.assertNotIn("tunix.hetero_int_str", records[0].attributes)
    self.assertNotIn("tunix.hetero_bool_int", records[0].attributes)
    self.assertNotIn("tunix.hetero_np_bool_int", records[0].attributes)
    self.assertNotIn("tunix.hetero_int_float", records[0].attributes)
    self.assertNotIn("tunix.unsupported_obj", records[0].attributes)
    self.assertEqual(
        records[1].attributes["tunix.parent_tracking_ids"], ("traj_nan",)
    )
    self.assertEqual(records[1].attributes["tunix.num_items"], 1)

  def test_guards_invalid_timestamps_and_encodes_otlp_proto(self):
    exporter = sdk_logs_export.InMemoryLogRecordExporter()
    provider = sdk_logs.LoggerProvider()
    provider.add_log_record_processor(
        sdk_logs_export.SimpleLogRecordProcessor(exporter)
    )
    self.addCleanup(provider.shutdown)
    logger = provider.get_logger(lineage_otel.LINEAGE_LOGGER_NAME)
    ctx = lineage.LineageContext(tracking_id="traj_ts", task_name="gsm8k")

    bad_timestamps: tuple[Any, ...] = (
        0.0,
        -0.0,
        -5.0,
        float("nan"),
        float("inf"),
        float("-inf"),
        1e30,
        1.79e308,
        1 << 2000,
        False,
    )
    for raw_ts in bad_timestamps:
      event = lineage.LineageEvent(
          component="worker.rollout",
          operation="step",
          timestamp_s=raw_ts,
      )
      # pylint: disable-next=protected-access
      lineage_otel._emit(logger, ctx, event)

    finished = exporter.get_finished_logs()
    self.assertLen(finished, 10)
    self.assertEqual(finished[0].log_record.timestamp, 0)
    self.assertEqual(finished[0].log_record.body, {"timestamp_s": 0.0})
    self.assertEqual(finished[1].log_record.timestamp, 0)
    for item in finished[2:]:
      self.assertGreater(item.log_record.timestamp, 0)
      self.assertLessEqual(item.log_record.timestamp, (1 << 64) - 1)
      self.assertGreater(item.log_record.body["timestamp_s"], 0.0)

    # Verify full OTLP protobuf fixed64 encoding succeeds across the batch.
    encoded = _log_encoder.encode_logs(finished)
    self.assertNotEmpty(encoded.resource_logs)


if __name__ == "__main__":
  absltest.main()
