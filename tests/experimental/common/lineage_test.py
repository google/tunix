# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for LineageContext and LineageEvent."""

import time
from absl.testing import absltest
import cloudpickle
from tunix.experimental.common import lineage


class LineageTest(absltest.TestCase):

  def test_lineage_event_creation(self):
    before = time.time()
    event = lineage.LineageEvent(
        component="engine.dispatch",
        operation="rollout",
        attributes={"policy_version": 1, "group_index": 0},
    )
    after = time.time()

    self.assertEqual(event.component, "engine.dispatch")
    self.assertEqual(event.operation, "rollout")
    self.assertGreaterEqual(event.timestamp_s, before)
    self.assertLessEqual(event.timestamp_s, after)
    self.assertEqual(
        event.attributes, {"policy_version": 1, "group_index": 0}
    )

  def test_lineage_context_add_event(self):
    ctx = lineage.LineageContext(
        tracking_id="traj_prompt_1_0",
        parent_tracking_ids=["prompt_1"],
    )
    self.assertEqual(ctx.tracking_id, "traj_prompt_1_0")
    self.assertEqual(ctx.parent_tracking_ids, ["prompt_1"])
    self.assertEmpty(ctx.events)

    event = ctx.add_event(
        component="worker.rollout",
        operation="generate",
        attributes={"worker_id": "w0", "latency_ms": 42.5},
    )

    self.assertLen(ctx.events, 1)
    self.assertIs(ctx.events[0], event)
    self.assertEqual(event.component, "worker.rollout")
    self.assertEqual(event.operation, "generate")
    self.assertEqual(event.attributes["worker_id"], "w0")
    self.assertEqual(event.attributes["latency_ms"], 42.5)

  def test_lineage_context_merge(self):
    ctx1 = lineage.LineageContext(
        tracking_id="traj_p1_0",
        parent_tracking_ids=["p1"],
    )
    ctx1.add_event("engine.dispatch", "rollout")

    ctx2 = lineage.LineageContext(
        tracking_id="traj_p2_0",
        parent_tracking_ids=["p2"],
    )
    ctx2.add_event("engine.dispatch", "rollout")

    merged = lineage.LineageContext.merge(
        batch_id="batch_0",
        contexts=[ctx1, ctx2, None],
        component="orchestrator.assembler",
        operation="pack",
        attributes={"bin_size": 2, "packed_len": 4096},
    )

    self.assertEqual(merged.tracking_id, "batch_0")
    self.assertEqual(merged.parent_tracking_ids, ["traj_p1_0", "traj_p2_0"])
    self.assertLen(merged.events, 1)
    self.assertEqual(merged.events[0].component, "orchestrator.assembler")
    self.assertEqual(merged.events[0].operation, "pack")
    self.assertEqual(
        merged.events[0].attributes, {"bin_size": 2, "packed_len": 4096}
    )

  def test_lineage_context_merge_deduplication(self):
    ctx1 = lineage.LineageContext(
        tracking_id="traj_p1_0",
        parent_tracking_ids=["p1"],
    )
    ctx2 = lineage.LineageContext(
        tracking_id="traj_p1_0",
        parent_tracking_ids=["p1"],
    )
    ctx3 = lineage.LineageContext(
        tracking_id="traj_p2_0",
        parent_tracking_ids=["p2"],
    )

    merged = lineage.LineageContext.merge(
        batch_id="batch_0",
        contexts=[ctx1, ctx2, ctx3, None, ctx1],
        component="orchestrator.assembler",
        operation="pack",
    )

    self.assertEqual(merged.tracking_id, "batch_0")
    self.assertEqual(merged.parent_tracking_ids, ["traj_p1_0", "traj_p2_0"])
    self.assertLen(merged.events, 1)

  def test_lineage_context_cloudpickle_round_trip(self):
    ctx = lineage.LineageContext(
        tracking_id="batch_42",
        parent_tracking_ids=["t1", "t2", "t3"],
    )
    ctx.add_event("worker.trainer", "fwd_bwd", {"loss": 0.35})

    restored = cloudpickle.loads(cloudpickle.dumps(ctx))

    self.assertEqual(restored.tracking_id, "batch_42")
    self.assertEqual(restored.parent_tracking_ids, ["t1", "t2", "t3"])
    self.assertLen(restored.events, 1)
    self.assertEqual(restored.events[0].component, "worker.trainer")
    self.assertEqual(restored.events[0].operation, "fwd_bwd")
    self.assertEqual(restored.events[0].attributes["loss"], 0.35)


if __name__ == "__main__":
  absltest.main()
