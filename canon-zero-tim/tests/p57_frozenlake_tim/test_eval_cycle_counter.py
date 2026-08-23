"""Counter-lifecycle tests for P57 in-process evaluation receipts."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]
LEARNER = ROOT / "tunix/rl/agentic/agentic_rl_learner.py"
LEARNER_TREE = ast.parse(
    LEARNER.read_text(encoding="utf-8"), filename=str(LEARNER)
)


def _load_helper():
  definitions = [
      node
      for node in LEARNER_TREE.body
      if isinstance(node, ast.FunctionDef)
      and node.name == "_p57_eval_cycle_enclosing_step"
  ]
  if len(definitions) != 1:
    raise AssertionError(
        "expected exactly one _p57_eval_cycle_enclosing_step definition"
    )
  namespace = {}
  exec(
      compile(
          ast.Module(body=definitions, type_ignores=[]),
          filename=str(LEARNER),
          mode="exec",
      ),
      namespace,
  )
  return namespace["_p57_eval_cycle_enclosing_step"]


eval_cycle_enclosing_step = _load_helper()


class EvalCycleCounterTest(unittest.TestCase):

  def test_train_receipt_wires_committed_and_deferred_counters(self):
    calls = [
        node
        for node in ast.walk(LEARNER_TREE)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_p57_eval_cycle_enclosing_step"
    ]
    self.assertEqual(len(calls), 1)
    keywords = {
        keyword.arg: ast.unparse(keyword.value) for keyword in calls[0].keywords
    }
    self.assertEqual(
        keywords,
        {
            "policy_step": "p57_eval_policy_step_this_cycle",
            "actor_train_steps": (
                "self.rl_cluster.actor_trainer.train_steps"
            ),
            "cluster_global_steps": "self.rl_cluster.global_steps",
        },
    )

  def test_post_update_receipt_uses_committed_actor_step(self):
    # Standard update_actor and P28/G6 both reach the receipt before
    # sync_weights advances the cluster counter.
    for update_path in ("standard", "p28_g6"):
      with self.subTest(update_path=update_path):
        self.assertEqual(
            eval_cycle_enclosing_step(
                policy_step=50,
                actor_train_steps=51,
                cluster_global_steps=50,
            ),
            51,
        )

  def test_uncommitted_actor_step_is_rejected(self):
    with self.assertRaisesRegex(RuntimeError, "committed_train_step=50"):
      eval_cycle_enclosing_step(
          policy_step=50,
          actor_train_steps=50,
          cluster_global_steps=50,
      )

  def test_early_cluster_advance_is_rejected(self):
    with self.assertRaisesRegex(RuntimeError, "deferred_global_step=51"):
      eval_cycle_enclosing_step(
          policy_step=50,
          actor_train_steps=51,
          cluster_global_steps=51,
      )


if __name__ == "__main__":
  unittest.main()
