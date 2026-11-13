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

import os
import time

from absl.testing import absltest
import cloudpickle
from tunix.dist_rl.common import lineage


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

  def test_lineage_context_merge_preserves_homogeneous_task_name(self):
    ctx_gsm_1 = lineage.LineageContext(
        tracking_id="traj_p1_0",
        task_name="gsm8k",
        parent_tracking_ids=["p1"],
    )
    ctx_gsm_2 = lineage.LineageContext(
        tracking_id="traj_p2_0",
        task_name="gsm8k",
        parent_tracking_ids=["p2"],
    )

    homogeneous = lineage.LineageContext.merge(
        batch_id="batch_homogeneous",
        contexts=[ctx_gsm_1, ctx_gsm_2, None],
        component="orchestrator.assembler",
        operation="pack",
    )
    self.assertEqual(homogeneous.task_name, "gsm8k")

  def test_lineage_context_merge_clears_heterogeneous_or_partially_unset_task_name(
      self,
  ):
    ctx_gsm = lineage.LineageContext(
        tracking_id="traj_p1_0",
        task_name="gsm8k",
        parent_tracking_ids=["p1"],
    )
    ctx_unset = lineage.LineageContext(
        tracking_id="traj_p2_0",
        parent_tracking_ids=["p2"],
    )
    ctx_swe = lineage.LineageContext(
        tracking_id="traj_p3_0",
        task_name="deepswe",
        parent_tracking_ids=["p3"],
    )

    mixed = lineage.LineageContext.merge(
        batch_id="batch_mixed",
        contexts=[ctx_gsm, ctx_swe],
        component="orchestrator.assembler",
        operation="pack",
    )
    self.assertEqual(mixed.task_name, "")

    partially_unset = lineage.LineageContext.merge(
        batch_id="batch_partially_unset",
        contexts=[ctx_gsm, ctx_unset],
        component="orchestrator.assembler",
        operation="pack",
    )
    self.assertEqual(partially_unset.task_name, "")

  def test_lineage_context_cloudpickle_round_trip(self):
    ctx = lineage.LineageContext(
        tracking_id="batch_42",
        task_name="gsm8k",
        parent_tracking_ids=["t1", "t2", "t3"],
    )
    ctx.add_event("worker.trainer", "fwd_bwd", {"loss": 0.35})

    restored = cloudpickle.loads(cloudpickle.dumps(ctx))

    self.assertEqual(restored.tracking_id, "batch_42")
    self.assertEqual(restored.task_name, "gsm8k")
    self.assertEqual(restored.parent_tracking_ids, ["t1", "t2", "t3"])
    self.assertLen(restored.events, 1)
    self.assertEqual(restored.events[0].component, "worker.trainer")
    self.assertEqual(restored.events[0].operation, "fwd_bwd")
    self.assertEqual(restored.events[0].attributes["loss"], 0.35)

  def test_event_sink_notified_on_add_event_and_merge_without_pickling_sink(
      self,
  ):
    recorded: list[tuple[str, str, str]] = []

    def _sink(ctx: lineage.LineageContext, ev: lineage.LineageEvent) -> None:
      recorded.append((ctx.tracking_id, ev.component, ev.operation))

    prev = lineage.set_event_sink(_sink)
    self.addCleanup(lineage.set_event_sink, prev)
    ctx = lineage.LineageContext(
        tracking_id="traj_p0_g0",
        task_name="deepswe",
        parent_tracking_ids=["p0"],
    )
    ctx.add_event("engine.dispatch", "rollout", {"policy_version": 2})
    merged = lineage.LineageContext.merge(
        batch_id="batch_0",
        contexts=[ctx],
        component="orchestrator.assembler",
        operation="pack",
    )
    payload_bytes = cloudpickle.dumps(merged)
    self.assertNotIn(b"_sink", payload_bytes)
    self.assertEqual(
        recorded,
        [
            ("traj_p0_g0", "engine.dispatch", "rollout"),
            ("batch_0", "orchestrator.assembler", "pack"),
        ],
    )
    self.assertIs(lineage.set_event_sink(None), _sink)
    ctx.add_event("worker.rollout", "generate")
    self.assertLen(recorded, 2)

  @absltest.skipUnless(hasattr(os, "fork"), "Requires os.fork")
  def test_forked_child_clears_inherited_event_sink(self):
    calls: list[str] = []
    prev = lineage.set_event_sink(lambda ctx, ev: calls.append(ctx.tracking_id))
    self.addCleanup(lineage.set_event_sink, prev)
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
      try:
        calls.clear()
        lineage.LineageContext(tracking_id="child_traj").add_event("w", "gen")
        os.write(write_fd, str(len(calls)).encode())
      finally:
        os._exit(0)
    os.close(write_fd)
    _, status = os.waitpid(pid, 0)
    child_calls = os.read(read_fd, 64)
    os.close(read_fd)
    self.assertEqual(os.waitstatus_to_exitcode(status), 0)
    self.assertEqual(child_calls, b"0")
    lineage.LineageContext(tracking_id="parent_traj").add_event("w", "gen")
    self.assertEqual(calls, ["parent_traj"])

  def test_event_sink_exception_logs_warning_without_aborting_add_event(self):
    def _broken_sink(ctx, ev):
      del ctx, ev
      raise RuntimeError("telemetry boom")

    prev = lineage.set_event_sink(_broken_sink)
    self.addCleanup(lineage.set_event_sink, prev)
    ctx = lineage.LineageContext(tracking_id="traj_safe")
    with self.assertLogs(level="WARNING") as logs:
      ev1 = ctx.add_event("worker.rollout", "generate", {"turn": 1})
      ev2 = ctx.add_event("worker.rollout", "generate", {"turn": 2})
    self.assertEqual(ctx.events, [ev1, ev2])
    self.assertLen(logs.output, 1)
    self.assertIn("Lineage event sink failed", logs.output[0])

  def test_extract_lineage_validates_type_and_handles_absent(self):
    self.assertIsNone(lineage.extract_lineage(None))
    self.assertIsNone(lineage.extract_lineage({}))
    self.assertIsNone(lineage.extract_lineage({"lineage": None}))
    ctx = lineage.LineageContext(tracking_id="traj_0")
    self.assertIs(lineage.extract_lineage({"lineage": ctx}), ctx)
    with self.assertRaisesRegex(TypeError, "Expected LineageContext"):
      lineage.extract_lineage({"lineage": "not_a_context"})


if __name__ == "__main__":
  absltest.main()
