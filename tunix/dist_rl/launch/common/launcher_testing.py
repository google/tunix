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

"""Test helpers that dry-run launcher scripts and capture their config.

`launcher_env` returns the hermetic environment the launch scripts run with in
tests (also used by `launch_test.py`).

`dry_run` executes a launcher script (an example launcher under
`dist_rl/examples/` such as `math_gsm8k_dist/k8s_launcher.sh`, a recipe, or
`launch.sh` itself) with `--dry-run` (local: prints the node and orchestrator
commands; k8s: renders the manifests). A recording wrapper around
`resolve_config` captures the exact `override_config_file=` +
`section.key=value` arguments the script composes, so tests can resolve them
in-process, in the same environment, and check key values.
`LauncherTestCase.assert_launcher` checks the contract every example launcher
shares.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import os
import pathlib
import shlex
import stat
import subprocess
import sys
from typing import Literal
from unittest import mock

from absl.testing import parameterized
from tunix.dist_rl.launch.common import resolve_config
import yaml

ORCHESTRATOR_MAIN = "tunix.dist_rl.launch.nodes.orchestrator.main"
RUN_ID = "run_launcher_test"
# Stand-in for the Hugging Face token: the resolver requires one for hub
# tokenizer paths, and `k8s.sh` writes it into the rendered commands, so a real
# token must not reach the dry-run output.
FAKE_HF_TOKEN = "test_token"

# Located through `resolve_config` (as in `launch_test.py`), so the paths hold
# wherever the OSS export puts the test code.
_COMMON_DIR = pathlib.Path(resolve_config.__file__).parent
_DIST_RL_DIR = _COMMON_DIR.parents[1]
# Example launchers and recipes live under `dist_rl/examples/`.
EXAMPLES_DIR = _DIST_RL_DIR / "examples"
_RESOLVE_CONFIG_BIN = _COMMON_DIR / "resolve_config"
_YAML_GEN_BIN = _DIST_RL_DIR / "distributed" / "deployment" / "yaml_generator"
# `tunix.cli.config` reads `T_<KEY>` environment variables as config overrides.
_CONFIG_OVERRIDE_ENV_PREFIX = "T_"


def launcher_env() -> dict[str, str]:
  """Returns the hermetic environment to run the launch scripts with in tests.

  Starts from `os.environ` without the `T_*` config overrides, and sets the
  current interpreter, the built `resolve_config` and `yaml_generator` binaries
  when present (they are not in the OSS export), a fixed `USER` (the k8s
  resource names include it), and `FAKE_HF_TOKEN`.

  Returns:
    The environment.
  """
  env = {
      key: value
      for key, value in os.environ.items()
      if not key.startswith(_CONFIG_OVERRIDE_ENV_PREFIX)
  }
  env["PYTHON_BIN"] = sys.executable or "python3"
  if _RESOLVE_CONFIG_BIN.is_file():
    env["RESOLVE_CONFIG_BIN"] = str(_RESOLVE_CONFIG_BIN)
  if _YAML_GEN_BIN.is_file():
    env["YAML_GEN_BIN"] = str(_YAML_GEN_BIN)
  env["USER"] = "testuser"
  env["HF_TOKEN"] = FAKE_HF_TOKEN
  return env


@dataclasses.dataclass(frozen=True, kw_only=True)
class DryRun:
  """Result of a launcher dry run.

  Attributes:
    returncode: Exit status of the launcher script.
    stdout: Rendered commands (local) or manifests (k8s).
    stderr: Launcher and resolver diagnostics.
    config_args: Arguments the launcher passed to `resolve_config` (the same
      arguments every node and the orchestrator receive), or None if the script
      exited before running it (e.g. `--help`, or a failure in the script
      itself).
  """

  returncode: int
  stdout: str
  stderr: str
  config_args: tuple[str, ...] | None

  def recorded_config_args(self) -> tuple[str, ...]:
    """Returns `config_args`.

    Raises:
      ValueError: If the script exited before running `resolve_config`.
    """
    if self.config_args is None:
      raise ValueError(
          f"The launcher exited with status {self.returncode} before running"
          f" resolve_config. stderr:\n{self.stderr}"
      )
    return self.config_args

  def resolve(self) -> resolve_config.DistHyperParameters:
    """Resolves `config_args` in-process, in the `launcher_env()` environment.

    Returns:
      The resolved configuration, as the launcher's nodes would see it.

    Raises:
      ValueError: If the script exited before running `resolve_config`.
    """
    config_args = self.recorded_config_args()
    with mock.patch.dict(os.environ, launcher_env(), clear=True):
      return resolve_config.initialize(list(config_args))


def dry_run(
    script: pathlib.Path, args: Sequence[str], *, tmp_dir: pathlib.Path
) -> DryRun:
  """Runs `<script> --dry-run cluster_config.run_id=RUN_ID args`.

  Args:
    script: Launcher script to run.
    args: Extra arguments appended after the run id (the script's `"$@"`).
    tmp_dir: Scratch directory for the recording resolver wrapper.

  Returns:
    The dry-run result. `launch.sh` runs `resolve_config` before anything else,
    so `config_args` is recorded even when the launcher fails afterwards; it is
    None only if the script exits before that.
  """
  record_file = tmp_dir / "resolve_config_args"
  if _RESOLVE_CONFIG_BIN.is_file():
    real_resolver = [str(_RESOLVE_CONFIG_BIN)]
  else:
    # OSS: no built binary; run the resolver module directly.
    real_resolver = [
        sys.executable or "python3",
        str(_COMMON_DIR / "resolve_config.py"),
    ]
  recorder = tmp_dir / "resolve_config_recorder.sh"
  recorder.write_text(
      "#!/bin/bash\n"
      f"printf '%s\\n' \"$@\" > {shlex.quote(str(record_file))}\n"
      f'exec {shlex.join(real_resolver)} "$@"\n',
      encoding="utf-8",
  )
  recorder.chmod(recorder.stat().st_mode | stat.S_IXUSR)

  env = launcher_env()
  env["RESOLVE_CONFIG_BIN"] = str(recorder)
  res = subprocess.run(
      [
          "bash",
          str(script),
          "--dry-run",
          f"cluster_config.run_id={RUN_ID}",
          *args,
      ],
      env=env,
      capture_output=True,
      text=True,
      check=False,
  )
  return DryRun(
      returncode=res.returncode,
      stdout=res.stdout,
      stderr=res.stderr,
      config_args=(
          tuple(record_file.read_text(encoding="utf-8").splitlines())
          if record_file.is_file()
          else None
      ),
  )


@dataclasses.dataclass(frozen=True, kw_only=True)
class Expected:
  """A launcher script and the values its configuration must resolve to.

  Attributes:
    name: Test case name.
    script: Launcher script path relative to `EXAMPLES_DIR`.
    example: Example package (`examples/<example>`) the script launches.
    module: Example module (`examples/<example>/<module>.py`), named by both
      `data_module` and `cluster_config.registry_module`; the script layers its
      config `examples/<example>/<module>.yaml`.
    target: Expected `cluster_config.target`.
    batch_size: Expected `batch_size`.
    mini_batch_size: Expected `rl_training_config.mini_batch_size`.
    num_generations: Expected `agentic_grpo_config.num_generations`.
    max_steps: Expected `rl_training_config.max_steps`.
    actor_mesh: Expected `actor_model_config.mesh.shape`.
    rollout_mesh: Expected `rollout_model_config.mesh.shape`.
    model_id: Expected `actor_model_config.model_id` and
      `tokenizer_config.tokenizer_path`.
    extra_args: Arguments appended as the script's `"$@"`.
  """

  name: str
  script: str
  example: str
  module: str
  target: Literal["local", "k8s"]
  batch_size: int
  mini_batch_size: int
  num_generations: int
  max_steps: int
  actor_mesh: tuple[int, ...]
  rollout_mesh: tuple[int, ...]
  model_id: str
  extra_args: tuple[str, ...] = ()


class LauncherTestCase(parameterized.TestCase):
  """Base class for launcher dry-run tests."""

  def run_launcher(self, script: str, *args: str) -> DryRun:
    """Dry-runs `EXAMPLES_DIR / script` without checking its exit status."""
    return dry_run(
        EXAMPLES_DIR / script,
        args,
        tmp_dir=pathlib.Path(self.create_tempdir().full_path),
    )

  def dry_run(self, script: str, *args: str) -> DryRun:
    """Dry-runs `EXAMPLES_DIR / script` and asserts that it succeeds."""
    run = self.run_launcher(script, *args)
    self.assertEqual(run.returncode, 0, msg=run.stderr)
    return run

  def assert_launcher(
      self, expected: Expected
  ) -> tuple[DryRun, resolve_config.DistHyperParameters]:
    """Dry-runs `expected.script` and checks the shared launcher contract.

    The example config `<example>/<module>.yaml` must be the only YAML
    layered on the shared base, user arguments (`"$@"`) must come last, and
    the resolved entry points, example module, batch geometry, meshes, and
    model (whose hub id is also the tokenizer path) must match `expected`.

    Args:
      expected: The launcher and its expected values.

    Returns:
      The dry run and its resolved configuration.

    Raises:
      ValueError: If `expected.target` is not "local" or "k8s".
    """
    run = self.dry_run(expected.script, *expected.extra_args)
    config_args = run.recorded_config_args()
    overrides = [
        arg for arg in config_args if arg.startswith("override_config_file=")
    ]
    self.assertLen(overrides, 1)
    self.assertEqual(config_args[0], overrides[0])
    config_file = f"{expected.example}/{expected.module}.yaml"
    self.assertTrue(overrides[0].endswith(f"/{config_file}"), msg=overrides)
    self.assertEqual(
        config_args[-len(expected.extra_args) - 1 :],
        (f"cluster_config.run_id={RUN_ID}", *expected.extra_args),
    )

    hp = run.resolve()
    config = hp.config
    cluster = config["cluster_config"]
    self.assertEqual(cluster["target"], expected.target)
    self.assertEqual(cluster["orchestrator_main"], ORCHESTRATOR_MAIN)
    module = f"tunix.dist_rl.examples.{expected.example}.{expected.module}"
    self.assertEqual(config["data_module"], module)
    self.assertEqual(cluster["registry_module"], module)
    self.assertEqual(config["batch_size"], expected.batch_size)
    rl_train = config["rl_training_config"]
    self.assertEqual(rl_train["mini_batch_size"], expected.mini_batch_size)
    self.assertEqual(rl_train["max_steps"], expected.max_steps)
    self.assertEqual(
        config["agentic_grpo_config"]["num_generations"],
        expected.num_generations,
    )
    self.assertEqual(
        hp.parse_mesh_config("actor_model_config")[0], expected.actor_mesh
    )
    self.assertEqual(
        hp.parse_mesh_config("rollout_model_config")[0], expected.rollout_mesh
    )
    self.assertEqual(
        config["actor_model_config"]["model_id"], expected.model_id
    )
    self.assertEqual(
        config["tokenizer_config"]["tokenizer_path"], expected.model_id
    )
    self.assertIsNone(rl_train["gradient_accumulation_steps"])
    # Every example overrides the shared artifact root.
    self.assertNotEqual(pathlib.Path(cluster["artifact_root"]).name, "dist_rl")

    self.assertIn(f"--process_main={ORCHESTRATOR_MAIN}", run.stdout)
    self.assertIn(f"cluster_config.run_id={RUN_ID}", run.stdout)
    if expected.target == "local":
      self.assertIn("Trainer command:", run.stdout)
      self.assertIn("Rollout command:", run.stdout)
      self.assertIn("Orchestrator command:", run.stdout)
    elif expected.target == "k8s":
      self.assertIn("kubectl delete jobset testuser-orch", run.stdout)
      # The container commands resolve the repo-relative example config.
      self.assertIn(
          f"override_config_file=tunix/dist_rl/examples/{config_file}",
          run.stdout,
      )
      yaml_sections = run.stdout.split("\n---\n")[1:]
      self.assertNotEmpty(yaml_sections)
      for section in yaml_sections:
        self.assertNotEmpty(list(yaml.safe_load_all(section)))
    else:
      raise ValueError(
          f"Unsupported launcher target {expected.target!r}; expected 'local'"
          " or 'k8s'."
      )
    return run, hp
