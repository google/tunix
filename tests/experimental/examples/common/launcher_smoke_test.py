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

"""Hermetic dry-run and CLI flag validation tests for distributed launchers."""

import os
import pathlib
import shlex
import subprocess
import sys
import tempfile
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
from tunix.experimental.examples.common import run_inference_node
from tunix.experimental.examples.common import run_rollout_node
from tunix.experimental.examples.common import run_trainer_node
from tunix.experimental.examples.deepswe_dist import run_deepswe_dist
from tunix.experimental.examples.math_gsm8k_dist import run_gsm8k_dist_grpo


REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
EXAMPLES_DIR = REPO_ROOT / "tunix" / "experimental" / "examples"

_RUNTIME_MAIN_PREFIX_FLAGS = (
    "--discovery_addrs=",
    "--discovery_id=",
    "--discovery_port=",
    "--process_main=",
    "--process_executor=",
)


def _extract_commands(output: str) -> dict[str, list[str]]:
  """Extracts commands printed via `print_command "<Label> command"`."""
  commands: dict[str, list[str]] = {}
  lines = output.splitlines()
  for idx, line in enumerate(lines):
    stripped = line.strip()
    if stripped in (
        "Trainer command:",
        "Rollout command:",
        "Inference command:",
        "Orchestrator command:",
    ):
      if idx + 1 < len(lines):
        label = stripped.removesuffix(" command:")
        commands[label] = shlex.split(lines[idx + 1].strip())
  return commands


def _strip_runtime_wrapper(cmd: list[str]) -> list[str]:
  """Strips `python -m tunix.experimental.distributed.runtime.main` wrapper."""
  assert len(cmd) >= 3, f"Unexpected command format: {cmd}"
  assert cmd[1] == "-m"
  assert cmd[2] == "tunix.experimental.distributed.runtime.main"
  return [
      arg
      for arg in cmd[3:]
      if not any(arg.startswith(prefix) for prefix in _RUNTIME_MAIN_PREFIX_FLAGS)
  ]


def _run_dry_run(
    script_relpath: str, extra_env: dict[str, str] | None = None
) -> dict[str, list[str]]:
  """Runs a launcher script in DRY_RUN=true mode and returns parsed commands."""
  script_path = EXAMPLES_DIR / script_relpath
  with tempfile.TemporaryDirectory() as tmpdir:
    env = os.environ.copy()
    env.update({
        "DRY_RUN": "true",
        "LOG_ROOT": tmpdir,
        "ARTIFACT_ROOT": os.path.join(tmpdir, "artifacts"),
    })
    if extra_env:
      env.update(extra_env)
    proc = subprocess.run(
        ["bash", str(script_path)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
      raise AssertionError(
          f"{script_relpath} failed with exit code {proc.returncode}:\n"
          f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
      )
    return _extract_commands(proc.stdout)


class LauncherSmokeTest(parameterized.TestCase):

  def test_all_example_shell_scripts_have_valid_bash_syntax(self):
    shell_scripts = sorted(EXAMPLES_DIR.rglob("*.sh"))
    self.assertNotEmpty(shell_scripts)
    for script in shell_scripts:
      proc = subprocess.run(
          ["bash", "-n", str(script)],
          cwd=str(REPO_ROOT),
          capture_output=True,
          text=True,
          check=False,
      )
      self.assertEqual(
          proc.returncode,
          0,
          f"bash -n failed for {script}:\n{proc.stderr}",
      )

  def test_ci_smoke_gsm8k_qwen3_0p6b_tunix_flags(self):
    commands = _run_dry_run("recipes/ci_smoke_gsm8k_qwen3_0p6b.sh")
    self.assertContainsSubset(
        {"Trainer", "Rollout", "Orchestrator"}, commands.keys()
    )

    trainer_args = run_trainer_node._parse_args(
        _strip_runtime_wrapper(commands["Trainer"])
    )
    self.assertEqual(trainer_args.trainer_backend, "tunix")
    self.assertEqual(trainer_args.model_name, "Qwen3-0.6B")
    self.assertEqual(trainer_args.mesh_tp, 4)
    self.assertEqual(trainer_args.sampler_type, "inprocess_vllm")

    rollout_args = run_rollout_node._parse_args(
        _strip_runtime_wrapper(commands["Rollout"])
    )
    self.assertEqual(rollout_args.sampler, "inprocess_vllm")
    self.assertEqual(rollout_args.mesh_tp, 4)
    self.assertEqual(rollout_args.weight_sync_mode.value, "raiden")
    self.assertEqual(rollout_args.maxtext_model_name, "")

    orch_args = run_gsm8k_dist_grpo._parse_args(
        _strip_runtime_wrapper(commands["Orchestrator"])
    )
    self.assertEqual(orch_args.max_steps, 2)
    self.assertEqual(orch_args.batch_size, 2)
    self.assertEqual(orch_args.num_generations, 2)
    self.assertEqual(orch_args.weight_sync_mode.value, "raiden")

  def test_ci_smoke_gsm8k_qwen3_0p6b_maxtext_flags(self):
    commands = _run_dry_run(
        "recipes/ci_smoke_gsm8k_qwen3_0p6b_maxtext.sh",
        extra_env={"MAX_SEQ_TOKEN_PER_TPU": "2048"},
    )
    self.assertContainsSubset(
        {"Trainer", "Rollout", "Orchestrator"}, commands.keys()
    )

    trainer_argv = _strip_runtime_wrapper(commands["Trainer"])
    max_seq_flags = [
        arg for arg in trainer_argv if arg.startswith("--max_seq_token_per_tpu")
    ]
    self.assertLen(max_seq_flags, 1)

    trainer_args = run_trainer_node._parse_args(trainer_argv)
    self.assertEqual(trainer_args.trainer_backend, "maxtext")
    self.assertEqual(trainer_args.maxtext_model_name, "qwen3-0.6b")
    self.assertEqual(trainer_args.rollout_mesh_tp, 4)
    self.assertEqual(trainer_args.max_seq_token_per_tpu, 2048)
    self.assertTrue(
        os.path.isabs(trainer_args.maxtext_output_directory),
        f"Expected absolute maxtext_output_directory, got {trainer_args.maxtext_output_directory}",
    )

    rollout_args = run_rollout_node._parse_args(
        _strip_runtime_wrapper(commands["Rollout"])
    )
    self.assertEqual(rollout_args.sampler, "vllm")
    self.assertEqual(rollout_args.maxtext_model_name, "qwen3-0.6b")

    orch_args = run_gsm8k_dist_grpo._parse_args(
        _strip_runtime_wrapper(commands["Orchestrator"])
    )
    self.assertEqual(orch_args.max_steps, 2)
    self.assertEqual(orch_args.max_seq_token_per_tpu, 2048)

  def test_gsm8k_launcher_with_inference_node_flags(self):
    commands = _run_dry_run(
        "math_gsm8k_dist/launcher.sh",
        extra_env={
            "BETA": "0.04",
            "RUN_INFERENCE_NODE": "1",
            "INFERENCE_TPU_CHIPS": "4,5",
        },
    )
    self.assertContainsSubset(
        {"Trainer", "Rollout", "Inference", "Orchestrator"}, commands.keys()
    )
    run_trainer_node._parse_args(_strip_runtime_wrapper(commands["Trainer"]))
    run_rollout_node._parse_args(_strip_runtime_wrapper(commands["Rollout"]))
    inf_args = run_inference_node._parse_args(
        _strip_runtime_wrapper(commands["Inference"])
    )
    self.assertEqual(inf_args.port, 20002)
    orch_args = run_gsm8k_dist_grpo._parse_args(
        _strip_runtime_wrapper(commands["Orchestrator"])
    )
    self.assertEqual(orch_args.inference_addr, "localhost:20002")

  @parameterized.named_parameters(
      ("tunix", "tunix"),
      ("maxtext", "maxtext"),
  )
  def test_deepswe_launcher_flags(self, trainer_backend: str):
    commands = _run_dry_run(
        "deepswe_dist/launcher.sh",
        extra_env={
            "TRAINER_BACKEND": trainer_backend,
            "MAX_SEQ_TOKEN_PER_TPU": "4096",
        },
    )
    self.assertContainsSubset(
        {"Trainer", "Rollout", "Orchestrator"}, commands.keys()
    )
    trainer_argv = _strip_runtime_wrapper(commands["Trainer"])
    max_seq_flags = [
        arg for arg in trainer_argv if arg.startswith("--max_seq_token_per_tpu")
    ]
    self.assertLen(max_seq_flags, 1)

    trainer_args = run_trainer_node._parse_args(trainer_argv)
    self.assertEqual(trainer_args.trainer_backend, trainer_backend)
    self.assertEqual(trainer_args.max_seq_token_per_tpu, 4096)

    rollout_args = run_rollout_node._parse_args(
        _strip_runtime_wrapper(commands["Rollout"])
    )
    self.assertEqual(rollout_args.env_name, "deepswe_env")
    self.assertEqual(rollout_args.agent_name, "deepswe_agent")

    orch_args = run_deepswe_dist._parse_args(
        _strip_runtime_wrapper(commands["Orchestrator"])
    )
    self.assertEqual(orch_args.max_seq_token_per_tpu, 4096)

  def test_frozenlake_launcher_flags(self):
    commands = _run_dry_run("frozenlake_dist/launcher.sh")
    self.assertContainsSubset(
        {"Trainer", "Rollout", "Orchestrator"}, commands.keys()
    )
    trainer_args = run_trainer_node._parse_args(
        _strip_runtime_wrapper(commands["Trainer"])
    )
    self.assertEqual(trainer_args.remat_policy, "decoder")

    rollout_args = run_rollout_node._parse_args(
        _strip_runtime_wrapper(commands["Rollout"])
    )
    self.assertEqual(rollout_args.env_name, "frozenlake_env")
    self.assertEqual(rollout_args.agent_name, "frozenlake_agent")
    self.assertFalse(rollout_args.enable_thinking)

    with mock.patch.dict(
        sys.modules,
        {
            "tunix.experimental.examples.frozenlake_dist.frozenlake": (
                mock.MagicMock()
            ),
        },
    ):
      from tunix.experimental.examples.frozenlake_dist import (  # pylint: disable=g-import-not-at-top
          run_frozenlake_dist,
      )

      orch_args = run_frozenlake_dist._parse_args(
          _strip_runtime_wrapper(commands["Orchestrator"])
      )
    self.assertEqual(orch_args.batch_size, 64)
    self.assertEqual(orch_args.num_generations, 8)


if __name__ == "__main__":
  absltest.main()
