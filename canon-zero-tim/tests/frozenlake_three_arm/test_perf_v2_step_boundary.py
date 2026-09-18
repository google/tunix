#!/usr/bin/env python3

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]
LEARNER = ROOT / "tunix/rl/agentic/agentic_rl_learner.py"


class PerfV2StepBoundaryContractTest(unittest.TestCase):

  @staticmethod
  def _assert_export_before_queue(window: str) -> None:
    export_pos = window.index("self.rl_cluster.perf_v2.export()")
    queue_tokens = (
        "self._put_prompts_to_queue(prompt_queue",
        "prompt_queue.put(None)",
    )
    queue_positions = [
        pos
        for token in queue_tokens
        for pos in [window.find(token)]
        if pos >= 0
    ]
    if not queue_positions:
      raise AssertionError("step-finalization window has no producer queue")
    if not all(export_pos < pos for pos in queue_positions):
      raise AssertionError("Perf v2 export must precede every next-step queue")

  def test_full_train_exports_completed_step_before_queueing_next_rollout(self):
    learner = LEARNER.read_text(encoding="utf-8")
    window = learner.split("if p58_all_filtered_no_commit:", 1)[1]
    window = window.split("update_steps_since_last_sync = 0", 1)[0]
    self._assert_export_before_queue(window)

    export_pos = window.index("self.rl_cluster.perf_v2.export()")
    queue_pos = window.index("self._put_prompts_to_queue(prompt_queue")
    host_gc_pos = window.index("if p45_host_memory_enabled:")
    self.assertLess(export_pos, queue_pos)
    self.assertLess(queue_pos, host_gc_pos)

  def test_reversed_order_negative_is_detected(self):
    with self.assertRaisesRegex(
        AssertionError, "export must precede"
    ):
      self._assert_export_before_queue(
          "self._put_prompts_to_queue(prompt_queue, batch)\n"
          "self.rl_cluster.perf_v2.export()\n"
      )


if __name__ == "__main__":
  unittest.main()
