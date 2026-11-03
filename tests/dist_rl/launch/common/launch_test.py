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

from __future__ import annotations

import os
import pathlib
import re
import subprocess

from absl.testing import absltest
from tunix.dist_rl.launch.common import launcher_testing
from tunix.dist_rl.launch.common import resolve_config
import yaml

# The drivers live next to `resolve_config.py`. Locate them through the module
# (not this test file), since the OSS export moves tests under `tests/`.
LAUNCH_DIR = pathlib.Path(resolve_config.__file__).parent
LAUNCH_SH = LAUNCH_DIR / "launch.sh"


# Variables the drivers read from the caller's environment (not from
# `render_bash_exports`).
_ENV_PASSTHROUGH = frozenset({
    "BASH_SOURCE",
    "ENTER_KUBE_CONTEXT",
    "HF_TOKEN",
    "LOG_ROOT",
    "ORCHESTRATOR_ID",
    "PYTHON",
    "PYTHONPATH",
    "PYTHON_BIN",
    "RESOLVE_CONFIG_BIN",
    "SECONDS",
    "TRAINER_IFRT_PROXY_LARGE_TRANSFER_THRESHOLD_BYTES",
    "WANDB_API_KEY",
    "YAML_DIR",
    "YAML_GEN",
    "YAML_GEN_BIN",
})
_VAR_REF = re.compile(r"\$\{?([A-Z][A-Z0-9_]*)")
_VAR_DEF = re.compile(
    r"^\s*(?:local\s+|export\s+)?([A-Z][A-Z0-9_]*)(?:\[[^]]*\])?\+?=|"
    r"\bfor\s+([A-Z][A-Z0-9_]*)\s+in\b",
    re.MULTILINE,
)


def _script_vars(text: str) -> tuple[set[str], set[str]]:
  """Returns (referenced, defined) upper-case variable names of a script."""
  code = "\n".join(
      line for line in text.splitlines() if not line.lstrip().startswith("#")
  )
  referenced = set(_VAR_REF.findall(code))
  # Arithmetic contexts reference variables without `$`.
  for expr in re.findall(r"\(\((.*?)\)\)", code):
    referenced |= set(re.findall(r"\b([A-Z][A-Z0-9_]*)\b", expr))
  defined = {a or b for a, b in _VAR_DEF.findall(code)}
  return referenced, defined


class LaunchScriptsTest(absltest.TestCase):

  def _run_launch(
      self,
      args: list[str],
      extra_env: dict[str, str] | None = None,
  ) -> subprocess.CompletedProcess[str]:
    env = launcher_testing.launcher_env()
    if extra_env:
      env.update(extra_env)
    return subprocess.run(
        ["bash", str(LAUNCH_SH), *args],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

  def test_local_dry_run_renders_worker_and_orchestrator_commands(self):
    res = self._run_launch(
        [
            "--dry-run",
            "cluster_config.target=local",
            "cluster_config.run_id=run_unit_test",
            "agentic_grpo_config.beta=0.0",
            "rl_training_config.max_steps=3",
        ],
    )
    self.assertEqual(res.returncode, 0, msg=res.stderr)
    self.assertIn("Trainer command:", res.stdout)
    self.assertIn("Rollout command:", res.stdout)
    self.assertNotIn("Inference command:", res.stdout)
    self.assertIn("Orchestrator command:", res.stdout)
    self.assertIn("cluster_config.run_id=run_unit_test", res.stdout)
    self.assertIn("rl_training_config.max_steps=3", res.stdout)

  def test_local_requires_inference_chips_when_inference_node_enabled(self):
    res = self._run_launch(
        [
            "--dry-run",
            "cluster_config.target=local",
            "agentic_grpo_config.beta=0.04",
        ],
    )
    self.assertNotEqual(res.returncode, 0)
    self.assertIn("cluster_config.local.inference_tpu_chips", res.stderr)

  def test_local_dry_run_includes_inference_node_when_chips_set(self):
    res = self._run_launch(
        [
            "--dry-run",
            "cluster_config.target=local",
            "agentic_grpo_config.beta=0.04",
            "cluster_config.local.inference_tpu_chips=4,5",
        ],
    )
    self.assertEqual(res.returncode, 0, msg=res.stderr)
    self.assertIn("Inference command:", res.stdout)
    # The reference worker registers through discovery; every node, including
    # the orchestrator, gets the same config arguments.
    self.assertNotIn("inference_addr", res.stdout)

  def test_local_dry_run_flag_does_not_launch(self):
    res = self._run_launch([
        "--dry-run",
        "cluster_config.target=local",
        "agentic_grpo_config.beta=0.0",
    ])
    self.assertEqual(res.returncode, 0, msg=res.stderr)
    self.assertNotIn("Launching trainer node", res.stdout)

  def test_local_target_rejects_non_start_command(self):
    res = self._run_launch([
        "stop",
        "--dry-run",
        "cluster_config.target=local",
        "agentic_grpo_config.beta=0.0",
    ])
    self.assertNotEqual(res.returncode, 0)
    self.assertIn("command 'stop' is not supported", res.stderr)
    self.assertNotIn("Trainer command:", res.stdout)

  def test_resolver_error_fails_launch_with_resolver_message(self):
    res = self._run_launch([
        "--dry-run",
        "cluster_config.target=local",
        "cluster_config.not_a_real_key=1",
    ])
    self.assertNotEqual(res.returncode, 0)
    self.assertIn("not_a_real_key", res.stderr)
    self.assertNotIn("unbound variable", res.stderr)
    self.assertNotIn("Trainer command:", res.stdout)

  def test_legacy_launcher_env_vars_fail_fast(self):
    args = [
        "--dry-run",
        "cluster_config.target=local",
        "agentic_grpo_config.beta=0.0",
    ]
    res = self._run_launch(
        args, extra_env={"MODEL_NAME": "Qwen3-8B", "MAX_STEPS": "2"}
    )
    self.assertNotEqual(res.returncode, 0)
    self.assertIn(
        "old example launchers are set: MAX_STEPS MODEL_NAME", res.stderr
    )
    self.assertIn("rl_training_config.max_steps=2", res.stderr)
    self.assertNotIn("Trainer command:", res.stdout)
    # An empty variable is not a setting.
    res = self._run_launch(args, extra_env={"MODEL_NAME": ""})
    self.assertEqual(res.returncode, 0, msg=res.stderr)
    self.assertIn("Trainer command:", res.stdout)

  def test_legacy_launcher_env_vars_exclude_driver_inputs(self):
    text = (LAUNCH_DIR / "legacy_env.sh").read_text()
    body = text.split("LEGACY_LAUNCHER_ENV_VARS=(", 1)[1].split(")", 1)[0]
    legacy = re.findall(r"^\s*([A-Z][A-Z0-9_]*)\s*$", body, re.MULTILINE)
    self.assertNotEmpty(legacy)
    self.assertEqual(legacy, sorted(set(legacy)))
    # Variables the drivers read from the caller's environment stay usable.
    self.assertEmpty(set(legacy) & _ENV_PASSTHROUGH)
    # `local.sh` takes the MaxText model name from the config now
    # (`ACTOR_MAXTEXT_MODEL_NAME`), so the old variable is rejected.
    self.assertIn("MAXTEXT_MODEL_NAME", legacy)
    # Extra MaxText flags come from `dist_training_config.maxtext_extra_flags`.
    self.assertIn("MAXTEXT_EXTRA_FLAGS", legacy)

  def test_maxtext_env_overrides_fail_fast(self):
    args = [
        "--dry-run",
        "cluster_config.target=local",
        "agentic_grpo_config.beta=0.0",
    ]
    res = self._run_launch(
        args, extra_env={"MAXTEXT_EXTRA_FLAGS": "remat_policy=full"}
    )
    self.assertNotEqual(res.returncode, 0)
    self.assertIn(
        "old example launchers are set: MAXTEXT_EXTRA_FLAGS", res.stderr
    )
    res = self._run_launch(args, extra_env={"M_REMAT_POLICY": "full"})
    self.assertNotEqual(res.returncode, 0)
    self.assertIn(
        "MaxText config overrides are set in the environment: M_REMAT_POLICY",
        res.stderr,
    )
    self.assertIn("dist_training_config.maxtext_extra_flags", res.stderr)
    self.assertNotIn("Trainer command:", res.stdout)
    # MaxText reads an empty variable too: it would set `remat_policy` to "".
    res = self._run_launch(args, extra_env={"M_REMAT_POLICY": ""})
    self.assertNotEqual(res.returncode, 0)
    self.assertIn(
        "MaxText config overrides are set in the environment: M_REMAT_POLICY",
        res.stderr,
    )

  def test_k8s_trainer_gets_maxtext_extra_flags_from_the_config(self):
    args = [
        "--dry-run",
        "cluster_config.target=k8s",
        "cluster_config.k8s.tunix_image=test-image",
        "agentic_grpo_config.beta=0.0",
        "model_config.model_source=maxtext",
        "model_config.model_path=/ckpt/qwen3-1.7b/0/items",
        "model_config.kwargs.maxtext_model_name=qwen3-1.7b",
    ]
    flags = (
        "dist_training_config.maxtext_extra_flags="
        "remat_policy=custom opt_type=sgd"
    )
    res = self._run_launch([*args, flags])
    self.assertEqual(res.returncode, 0, msg=res.stderr)
    self.assertRegex(
        res.stdout, r"MAXTEXT_EXTRA_FLAGS=remat_policy=custom\\+ opt_type=sgd"
    )
    res = self._run_launch(args)
    self.assertEqual(res.returncode, 0, msg=res.stderr)
    self.assertNotIn("MAXTEXT_EXTRA_FLAGS", res.stdout)

  def test_drivers_refuse_direct_invocation(self):
    for script in ("local.sh", "k8s.sh"):
      with self.subTest(script=script):
        env = {
            k: v for k, v in os.environ.items() if k != "CLUSTER_TARGET"
        }
        res = subprocess.run(
            ["bash", str(LAUNCH_DIR / script), "start"],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("must be invoked via launch.sh", res.stderr)

  def test_k8s_dry_run_renders_manifests_and_normalizes_override_path(self):
    recipe_path = LAUNCH_DIR / "base_dist_config.yaml"
    res = self._run_launch([
        "start",
        "--dry-run",
        f"override_config_file={recipe_path}",
        "cluster_config.target=k8s",
        "cluster_config.k8s.tunix_image=test-image",
        "cluster_config.run_id=run_k8s_test",
        "weight_sync_config.mode=raiden",
        "cluster_config.k8s.rollout.replicas=2",
        "agentic_grpo_config.beta=0.0",
    ])
    self.assertEqual(res.returncode, 0, msg=res.stderr)
    self.assertIn("kubectl delete jobset testuser-orch -n default", res.stdout)
    self.assertIn("kubectl delete jobset testuser-train -n default", res.stdout)
    self.assertIn(
        "kubectl delete leaderworkerset testuser-roll-0 -n default", res.stdout
    )
    self.assertIn(
        "kubectl delete leaderworkerset testuser-roll-1 -n default", res.stdout
    )
    self.assertIn("RAIDEN_USE_FFI=1", res.stdout)
    self.assertIn("RAIDEN_USE_FFI=0", res.stdout)
    self.assertIn(
        "override_config_file="
        "tunix/dist_rl/launch/common/base_dist_config.yaml",
        res.stdout,
    )
    # Every pod gets the same config arguments, with the launcher's resource
    # ids; a rollout replica gets its index from the environment instead.
    self.assertIn("cluster_config.k8s.trainer_id=testuser-train", res.stdout)
    self.assertRegex(
        res.stdout, r"cluster_config\.k8s\.rollout_id=testuser-roll(?![\w-])"
    )
    self.assertNotIn("cluster_config.k8s.rollout_id=testuser-roll-", res.stdout)
    self.assertIn("DIST_RL_ROLLOUT_REPLICA=0", res.stdout)
    self.assertIn("DIST_RL_ROLLOUT_REPLICA=1", res.stdout)

    # Verify that the rendered YAML documents after the kubectl delete lines
    # parse as valid Kubernetes YAML manifests.
    yaml_sections = res.stdout.split("\n---\n")[1:]
    self.assertLen(yaml_sections, 4)
    for section in yaml_sections:
      docs = list(yaml.safe_load_all(section))
      self.assertNotEmpty(docs)

  def test_k8s_requires_tunix_image(self):
    res = self._run_launch([
        "--dry-run",
        "cluster_config.target=k8s",
        "agentic_grpo_config.beta=0.0",
    ])
    self.assertNotEqual(res.returncode, 0)
    self.assertIn("cluster_config.k8s.tunix_image is empty", res.stderr)

  def test_k8s_rejects_reference_worker(self):
    res = self._run_launch([
        "--dry-run",
        "cluster_config.target=k8s",
        "cluster_config.k8s.tunix_image=test-image",
        "agentic_grpo_config.beta=0.04",
    ])
    self.assertNotEqual(res.returncode, 0)
    self.assertIn("does not deploy a reference inference worker", res.stderr)
    self.assertNotIn("apiVersion:", res.stdout)

  def test_k8s_stop_command_only_deletes_resources(self):
    res = self._run_launch([
        "stop",
        "--dry-run",
        "cluster_config.target=k8s",
        "cluster_config.k8s.tunix_image=test-image",
        "agentic_grpo_config.beta=0.0",
    ])
    self.assertEqual(res.returncode, 0, msg=res.stderr)
    self.assertIn("kubectl delete jobset testuser-orch -n default", res.stdout)
    self.assertNotIn("apiVersion:", res.stdout)


  def test_drivers_only_read_exported_or_known_variables(self):
    exported = set(
        re.findall(
            r"^export ([A-Z0-9_]+)=",
            resolve_config.render_bash_exports(resolve_config.initialize([])),
            flags=re.MULTILINE,
        )
    )
    referenced: set[str] = set()
    defined: set[str] = set()
    for script in ("common.sh", "launch.sh", "local.sh", "k8s.sh"):
      refs, defs = _script_vars((LAUNCH_DIR / script).read_text())
      referenced |= refs
      defined |= defs
    missing = referenced - exported - defined - _ENV_PASSTHROUGH
    self.assertEmpty(
        missing,
        msg=(
            "Variables read by the bash drivers but neither exported by"
            " resolve_config.render_bash_exports nor set by the scripts."
        ),
    )
    unused = exported - referenced
    self.assertEmpty(
        unused, msg="Exports not read by any bash driver."
    )


if __name__ == "__main__":
  absltest.main()
