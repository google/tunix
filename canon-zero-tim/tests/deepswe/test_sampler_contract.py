"""Workload-specific sampler policy tests for canonical DeepSWE."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "tunix/rl/agentic/agentic_grpo_learner.py"
tree = ast.parse(SOURCE.read_text(), filename=str(SOURCE))
matches = [
    node
    for node in tree.body
    if isinstance(node, ast.FunctionDef)
    and node.name == "_canonical_alignment_sampler_is_valid"
]
if len(matches) != 1:
  raise RuntimeError("cannot isolate the canonical sampler contract")
namespace = {}
exec(compile(ast.Module(body=matches, type_ignores=[]), str(SOURCE), "exec"), namespace)
_sampler_is_valid = namespace["_canonical_alignment_sampler_is_valid"]


class DeepSWESamplerContractTest(unittest.TestCase):

  def test_signed_deepswe_allows_direct_rollout_logprobs_without_tis(self):
    self.assertTrue(
        _sampler_is_valid(
            None,
            "",
            p34_deepswe=True,
            p34_disable_sampler_is=True,
            p34_disable_tis=True,
        )
    )

  def test_deepswe_rejects_partial_sampler_policy_attestation(self):
    self.assertFalse(
        _sampler_is_valid(
            None,
            "",
            p34_deepswe=True,
            p34_disable_sampler_is=True,
            p34_disable_tis=False,
        )
    )

  def test_unregistered_workload_still_rejects_missing_sampler_correction(self):
    self.assertFalse(_sampler_is_valid(None, "unregistered"))


if __name__ == "__main__":
  unittest.main()
