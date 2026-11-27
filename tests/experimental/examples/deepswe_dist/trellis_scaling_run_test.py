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

"""Unit tests for the standalone trellis_scaling_run.sh recipe."""

from __future__ import annotations

import os
import pathlib
import shlex
import subprocess
import sys
import types
from typing import Any

from absl.testing import absltest
import yaml

from tunix.dist_rl.examples.common import run_rollout_node
from tunix.dist_rl.examples.common import run_trainer_node

try:
  import r2egym  # pylint: disable=unused-import
except ImportError:
  _r2e_mod = types.ModuleType("r2egym")
  _r2e_agenthub = types.ModuleType("r2egym.agenthub")
  _r2e_action = types.ModuleType("r2egym.agenthub.action")
  _r2e_action.Action = object  # type: ignore[attr-defined]
  sys.modules.setdefault("r2egym", _r2e_mod)
  sys.modules.setdefault("r2egym.agenthub", _r2e_agenthub)
  sys.modules.setdefault("r2egym.agenthub.action", _r2e_action)

from tunix.experimental.examples.deepswe_dist import run_deepswe_dist  # pylint: disable=g-import-not-at-top

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
_RECIPE = (
    _REPO_ROOT / "tunix/experimental/examples/recipes/trellis_scaling_run.sh"
)


def _render_jobsets(**env_overrides: str) -> dict[str, dict[str, Any]]:
  env = dict(os.environ)
  env.update({
      "PYTHON": sys.executable,
      "DRY_RUN": "true",
      "JOB_PREFIX": "test-1023u",
      "ROLLOUT_REPLICAS": "2",
      "DISABLE_JAX_CACHE": "1",
  })
  env.update(env_overrides)
  proc = subprocess.run(
      ["bash", str(_RECIPE), "start", "--dry-run"],
      env=env,
      capture_output=True,
      text=True,
      check=True,
  )
  yaml_lines = [
      ln
      for ln in proc.stdout.splitlines()
      if not ln.startswith("kubectl delete ")
  ]
  docs = [
      doc
      for doc in yaml.safe_load_all("\n".join(yaml_lines))
      if isinstance(doc, dict)
  ]
  return {doc["metadata"]["name"]: doc for doc in docs}


def _startup_cmd(jobset_doc: dict[str, Any]) -> str:
  rjobs = jobset_doc["spec"]["replicatedJobs"]
  for rjob in rjobs:
    containers = rjob["template"]["spec"]["template"]["spec"]["containers"]
    for c in containers:
      for arg in c.get("args", []) + c.get("command", []):
        if "tunix.dist_rl.distributed.runtime.main" in str(arg):
          return str(arg)
  raise AssertionError(
      f"No runtime.main startup command found in {jobset_doc['metadata']['name']}"
  )


_RUNTIME_WRAPPER_FLAGS = (
    "--process_executor=",
    "--process_main=",
    "--discovery_id=",
    "--discovery_port=",
    "--discovery_addrs=",
)


def _extract_env_and_process_argv(
    startup_cmd: str,
) -> tuple[dict[str, str], list[str]]:
  tokens = shlex.split(startup_cmd)
  try:
    rt_idx = tokens.index("tunix.dist_rl.distributed.runtime.main")
  except ValueError as exc:
    raise AssertionError(
        f"No runtime.main invocation found in: {startup_cmd}"
    ) from exc

  shell_seps = {"(", ")", "&", "&&", ";", "||", "|"}
  start_idx = 0
  for idx in range(rt_idx - 1, -1, -1):
    if tokens[idx] in shell_seps or tokens[idx] in ("then", "else", "do"):
      start_idx = idx + 1
      break
  end_idx = len(tokens)
  for idx in range(rt_idx + 1, len(tokens)):
    if tokens[idx] in shell_seps or tokens[idx] in ("fi", "done"):
      end_idx = idx
      break
  cmd_tokens = tokens[start_idx:end_idx]

  env_vars: dict[str, str] = {}
  py_idx = None
  for idx, tok in enumerate(cmd_tokens):
    if tok in ("python", "python3") or tok.endswith("/python"):
      py_idx = idx
      break
    if "=" in tok and not tok.startswith("-"):
      k, v = tok.split("=", 1)
      env_vars[k] = v

  if py_idx is None:
    raise AssertionError(f"No python invocation found in: {startup_cmd}")
  if cmd_tokens[py_idx + 1 : py_idx + 3] != [
      "-m",
      "tunix.dist_rl.distributed.runtime.main",
  ]:
    raise AssertionError(
        f"Unexpected python entrypoint: {cmd_tokens[py_idx:py_idx + 4]}"
    )

  process_argv = [
      tok
      for tok in cmd_tokens[py_idx + 3 :]
      if not tok.startswith(_RUNTIME_WRAPPER_FLAGS)
  ]
  return env_vars, process_argv


class TrellisScalingRunRecipeTest(absltest.TestCase):

  def test_renders_valid_orchestrator_trainer_and_rollout_jobsets(self):
    jobsets = _render_jobsets()
    self.assertIn("test-1023u-orch", jobsets)
    self.assertIn("test-1023u-train", jobsets)
    self.assertIn("test-1023u-roll-0", jobsets)
    self.assertIn("test-1023u-roll-1", jobsets)

    orch_cmd = _startup_cmd(jobsets["test-1023u-orch"])
    _, orch_argv = _extract_env_and_process_argv(orch_cmd)
    orch_args = run_deepswe_dist._parse_args(orch_argv)
    self.assertEqual(orch_args.batch_size, 64)
    self.assertEqual(orch_args.num_generations, 16)
    self.assertEqual(orch_args.max_turns, 50)
    self.assertEqual(orch_args.max_staleness, 4)
    self.assertEqual(orch_args.trajectory_group_order, "trajectory_completion")
    self.assertEqual(orch_args.sampler_is, "token")
    self.assertAlmostEqual(orch_args.sampler_is_threshold, 5.0)
    self.assertEqual(orch_args.sampler_rs, "geometric")
    self.assertAlmostEqual(orch_args.sampler_rs_min, 0.98)
    self.assertAlmostEqual(orch_args.sampler_rs_max, 1.02)
    self.assertAlmostEqual(orch_args.seq_logprob_error_threshold, 3.5)
    self.assertTrue(orch_args.use_rollout_logps)
    self.assertEqual(orch_args.routed_experts_num_experts, 256)

    train_cmd = _startup_cmd(jobsets["test-1023u-train"])
    train_env, train_argv = _extract_env_and_process_argv(train_cmd)
    train_args = run_trainer_node._parse_args(train_argv)
    self.assertAlmostEqual(train_args.optimizer_learning_rate, 2e-6)
    self.assertEqual(train_args.model_dtype, "bfloat16")
    self.assertEqual(train_args.mesh_fsdp, 32)
    self.assertEqual(train_args.mesh_expert, 2)
    self.assertEqual(train_args.maxtext_mesh_context, 4)
    self.assertIn("'max_norm': 0.0625", train_cmd)

    mt_flags = set(train_env["MAXTEXT_EXTRA_FLAGS"].split())
    self.assertIn("learning_rate_final_fraction=1.0", mt_flags)
    self.assertIn("async_checkpointing=true", mt_flags)
    self.assertIn("skip_step_on_nan=true", mt_flags)
    self.assertIn(
        "trainable_parameters_mask=^(?!.*routed_experts/gate/kernel).*",
        mt_flags,
    )

    roll_cmd = _startup_cmd(jobsets["test-1023u-roll-0"])
    _, roll_argv = _extract_env_and_process_argv(roll_cmd)
    roll_args = run_rollout_node._parse_args(roll_argv)
    self.assertAlmostEqual(roll_args.vllm_hbm_utilization, 0.9)
    self.assertEqual(roll_args.vllm_max_num_seqs, 64)
    self.assertEqual(
        roll_args.maxtext_attention, "vllm_batched_rpa_long_ctx"
    )
    self.assertTrue(roll_args.prefuse_moe_weights)
    vllm_cfg = run_rollout_node._parse_vllm_config_json(
        roll_args.vllm_config_json,
        has_maxtext_model=bool(roll_args.maxtext_model_name),
    )
    self.assertEqual(
        vllm_cfg["additional_config"]["sharding"]["sharding_strategy"][
            "expert_parallelism"
        ],
        8,
    )
    self.assertTrue(
        vllm_cfg["additional_config"]["maxtext_config"]["fp8_moe"]
    )

  def test_orchestrator_extra_env_appends_to_base_raiden_vars(self):
    jobsets = _render_jobsets(ORCHESTRATOR_EXTRA_ENV="CUSTOM_ORCH_VAR=42")
    orch_cmd = _startup_cmd(jobsets["test-1023u-orch"])
    self.assertRegex(orch_cmd, r"\bWEIGHT_SYNC_BIND_TIMEOUT_S=900\b")
    self.assertRegex(orch_cmd, r"\bRAIDEN_PARALLELISM=16\b")
    self.assertRegex(orch_cmd, r"\bCUSTOM_ORCH_VAR=42\b")


if __name__ == "__main__":
  absltest.main()
