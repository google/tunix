"""Static launch-boundary checks for the real DeepSWE program."""

from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import types
import unittest


ROOT = Path(__file__).resolve().parents[3]


class DeepSWEScriptContractTest(unittest.TestCase):

  def test_canonical_entrypoint_initializes_pathways_before_program_import(self):
    text = (ROOT / "examples/deepswe/canonical_entrypoint.py").read_text()
    self.assertLess(text.index("pathwaysutils.initialize()"), text.index("runpy.run_module"))
    self.assertNotIn("import jax", text)

  def test_canonical_entrypoint_file_launch_bootstraps_repository_root(self):
    entrypoint = ROOT / "examples/deepswe/canonical_entrypoint.py"
    probe = textwrap.dedent(
        """
        import pathlib
        import runpy
        import sys

        entrypoint = pathlib.Path(sys.argv[1]).resolve()
        repository_root = str(entrypoint.parents[2])
        namespace = runpy.run_path(str(entrypoint), run_name="p34_entrypoint_probe")
        calls = []
        namespace["runpy"].run_module = (
            lambda name, run_name: calls.append((name, run_name))
        )
        namespace["main"]()
        assert sys.path[0] == repository_root, sys.path
        assert calls == [
            ("examples.deepswe.train_deepswe_nb", "__main__")
        ], calls
        """
    )
    env = os.environ.copy()
    env.update({"CANON_P34_DEEPSWE": "1", "JAX_PLATFORMS": "cpu"})
    env.pop("CANON_PATHWAYS_INITIALIZED", None)
    completed = subprocess.run(
        [sys.executable, "-I", "-c", probe, str(entrypoint)],
        cwd="/tmp",
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    self.assertIn(
        "[P34.PATHWAYS] initialized_once=1 before_jax=1",
        completed.stdout,
    )

  def test_strict_mode_registers_scheduler_flags(self):
    text = (ROOT / "examples/deepswe/train_deepswe_nb.py").read_text()
    self.assertIn('"--rollout_vllm_max_num_seqs"', text)
    self.assertIn('"--max_num_batched_tokens"', text)
    self.assertIn("args = parser.parse_args()", text)
    self.assertIn('"enable_prefix_caching": not P34_DEEPSWE', text)
    self.assertIn("scheduler_per_dp={p34.max_num_seqs_per_dp}/", text)

  def test_optimizer_cli_cannot_parse_false_as_true(self):
    text = (ROOT / "examples/deepswe/train_deepswe_nb.py").read_text()
    start = text.index('"--optimizer_offload"')
    end = text.index("# Checkpointing", start)
    block = text[start:end]
    self.assertIn("argparse.BooleanOptionalAction", block)
    self.assertIn('"--optimizer-offload"', block)
    self.assertNotIn("type=bool", block)

  def test_full_startup_cannot_read_the_onehost_replay_selector_unbound(self):
    source = ROOT / "examples/deepswe/train_deepswe_nb.py"
    tree = ast.parse(source.read_text())

    def assigned_name(node, name):
      return (
          isinstance(node, ast.Assign)
          and any(
              isinstance(target, ast.Name) and target.id == name
              for target in node.targets
          )
      )

    onehost_block = next(
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "ONEHOST_SMOKE"
        and node.lineno > 800
    )
    replay_default = next(
        node
        for node in tree.body
        if assigned_name(node, "P58_Q4_TP4_TRAJECTORY_REPLAY")
    )
    replay_geometry = next(
        node
        for node in tree.body
        if assigned_name(node, "P58_REPLAY_UPDATE_GEOMETRY")
    )
    self.assertLess(replay_default.lineno, onehost_block.lineno)
    self.assertIs(ast.literal_eval(replay_default.value), False)
    self.assertIsInstance(replay_geometry.value, ast.IfExp)
    self.assertIsInstance(replay_geometry.value.test, ast.BoolOp)
    self.assertIsInstance(replay_geometry.value.test.op, ast.And)

    onehost_names = {
        node.id
        for node in ast.walk(onehost_block)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Store)
        and node.id.isupper()
    }
    names_bound_before_onehost = {
        node.id
        for statement in tree.body
        if getattr(statement, "lineno", 0) < onehost_block.lineno
        for node in ast.walk(statement)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Store)
        and node.id.isupper()
    }
    names_loaded_after_onehost = {
        node.id
        for statement in tree.body
        if getattr(statement, "lineno", 0) > onehost_block.end_lineno
        for node in ast.walk(statement)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id.isupper()
    }
    self.assertEqual(
        (onehost_names & names_loaded_after_onehost)
        - names_bound_before_onehost,
        set(),
    )

    calls = []
    namespace = {
        "ONEHOST_SMOKE": False,
        "deepswe_debug": types.SimpleNamespace(
            q4_tp4_trajectory_replay_update_geometry=(
                lambda env: calls.append(dict(env)) or (4, 2)
            )
        ),
        "os": types.SimpleNamespace(environ={}),
    }
    probe = ast.fix_missing_locations(
        ast.Module(body=[replay_default, replay_geometry], type_ignores=[])
    )
    exec(compile(probe, str(source), "exec"), namespace)
    self.assertIsNone(namespace["P58_REPLAY_UPDATE_GEOMETRY"])
    self.assertEqual(calls, [])

    namespace["ONEHOST_SMOKE"] = True
    namespace["P58_Q4_TP4_TRAJECTORY_REPLAY"] = True
    replay_probe = ast.fix_missing_locations(
        ast.Module(body=[replay_geometry], type_ignores=[])
    )
    exec(compile(replay_probe, str(source), "exec"), namespace)
    self.assertEqual(namespace["P58_REPLAY_UPDATE_GEOMETRY"], (4, 2))
    self.assertEqual(calls, [{}])

  def test_full_capture_runs_before_rollout_only_or_backward(self):
    learner = (
        ROOT / "tunix/rl/agentic/agentic_grpo_learner.py"
    ).read_text()
    consumer = (
        ROOT / "tunix/rl/agentic/agentic_rl_learner.py"
    ).read_text()
    self.assertIn('"p34-production"', learner)
    self.assertIn("deepswe_debug.persist_batch(", learner)
    self.assertLess(
        learner.index("deepswe_debug.persist_batch("),
        learner.index("policy_versions = np.array("),
    )
    self.assertLess(
        consumer.index("deepswe_debug.rollout_only()"),
        consumer.index("self._run_p28_g6_update("),
    )

  def test_p34_uses_dp_axes_and_replicated_parameters(self):
    text = (ROOT / "examples/deepswe/train_deepswe_nb.py").read_text()
    self.assertIn(
        'rollout_dims = [("dp", p34.dp_size), ("tp", p34.tp_size)]',
        text,
    )
    self.assertIn(
        'train_dims = [("dp", p34.dp_size), ("tp", p34.tp_size)]',
        text,
    )
    self.assertIn("deepswe_contract.active_workload(os.environ)", text)
    self.assertIn("configure_replicated_parameter_sharding", text)
    self.assertIn("P34 forbids FSDP", text)
    self.assertIn(
        "training_data_sharding_axis = (train_axis_names[0],)", text
    )
    self.assertIn("[DEEPSWE.DATA_SHARDING] PASS", text)
    self.assertIn(
        "data_sharding_axis=training_data_sharding_axis", text
    )

  def test_backward_no_commit_requires_full_array_repeat(self):
    adapter = (ROOT / "tunix/rl/canonical_qwen3_adapter.py").read_text()
    learner = (ROOT / "tunix/rl/agentic/agentic_rl_learner.py").read_text()
    self.assertIn("deterministic_repeat_exact", adapter)
    self.assertIn("jnp.array_equal(first, second)", adapter)
    self.assertIn(
        '"deterministic_repeat": (p34_workload and p33_no_commit)', learner
    )
    self.assertIn("**segmented_kwargs", learner)

  def test_cross_role_weights_are_checked_before_rescore(self):
    learner = (
        ROOT / "tunix/rl/agentic/agentic_grpo_learner.py"
    ).read_text()
    weight_gate = learner.index("persist_weight_attestation(")
    rescore = learner.index(
        "rescore_source = self.rl_cluster.rollout.get_prefill_rescore_logps"
    )
    self.assertLess(weight_gate, rescore)
    self.assertIn("attest_actor_anchor_matches_engine()", learner)

  def test_weight_evidence_is_fail_closed_and_classified(self):
    runner = (ROOT / "canon-zero-tim/cluster/steps/90_run.sh").read_text()
    self.assertIn("report_keys+=(CANON_P34_WEIGHT_REPORT)", runner)
    self.assertIn("[P34.WEIGHT_ARTIFACT_JSON]", runner)
    self.assertIn('--weight-report "$CANON_P34_WEIGHT_REPORT"', runner)


if __name__ == "__main__":
  unittest.main()
