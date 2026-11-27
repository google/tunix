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

import ast
import json
import os
import pathlib
import re
import shlex
import subprocess
import sys
from typing import Any

from absl.testing import absltest
from tunix.dist_rl.distributed.deployment import yaml_generator
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
    # Only checked: `k8s.sh` warns that the rollout pods don't read it.
    "BUCKET",
    "ENTER_KUBE_CONTEXT",
    "HF_TOKEN",
    "LOG_ROOT",
    "ORCHESTRATOR_ID",
    "PYTHON",
    "PYTHONPATH",
    "PYTHON_BIN",
    "RESOLVE_CONFIG_BIN",
    "SECONDS",
    "WANDB_API_KEY",
    "YAML_DIR",
    "YAML_GEN",
    "YAML_GEN_BIN",
})
# Environment variables read by `yaml_generator.py` that intentionally pass
# through from the caller's environment (not configured via `base_dist_config`).
_YAML_GEN_ENV_PASSTHROUGH = frozenset({
    "USER",
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


def _yaml_generator_env_vars() -> set[str]:
  """Extracts environment variable names read by `yaml_generator.py`."""
  tree = ast.parse(pathlib.Path(yaml_generator.__file__).read_text())
  env_vars = set()
  for node in ast.walk(tree):
    if isinstance(node, ast.Call):
      if (
          isinstance(node.func, ast.Attribute)
          and isinstance(node.func.value, ast.Attribute)
          and node.func.value.attr == "environ"
          and node.func.attr == "get"
          and node.args
          and isinstance(node.args[0], ast.Constant)
          and isinstance(node.args[0].value, str)
      ):
        env_vars.add(node.args[0].value)
      elif (
          isinstance(node.func, ast.Attribute)
          and node.func.attr == "parse_env_bool"
          and node.args
          and isinstance(node.args[0], ast.Constant)
          and isinstance(node.args[0].value, str)
      ):
        env_vars.add(node.args[0].value)
    elif isinstance(node, ast.Subscript):
      if (
          isinstance(node.value, ast.Attribute)
          and node.value.attr == "environ"
          and isinstance(node.slice, ast.Constant)
          and isinstance(node.slice.value, str)
      ):
        env_vars.add(node.slice.value)
  return env_vars


def _dry_run_jobsets(stdout: str) -> dict[str, Any]:
  """Returns the JobSet manifests of a `start --dry-run`, keyed by name."""
  jobsets = {}
  for section in stdout.split("\n---\n")[1:]:
    for doc in yaml.safe_load_all(section):
      if doc and doc["kind"] == "JobSet":
        jobsets[doc["metadata"]["name"]] = doc
  return jobsets


def _pod_specs(jobset: Any) -> list[Any]:
  """Returns the pod spec of every replicated job of a JobSet manifest."""
  return [
      job["template"]["spec"]["template"]["spec"]
      for job in jobset["spec"]["replicatedJobs"]
  ]


class LaunchScriptsTest(absltest.TestCase):

  def _run_launch(
      self,
      args: list[str],
      extra_env: dict[str, str] | None = None,
      *,
      task: bool = True,
  ) -> subprocess.CompletedProcess[str]:
    """Runs `launch.sh`, by default with `launcher_testing.TASK_ARGS`."""
    env = launcher_testing.launcher_env()
    if extra_env:
      env.update(extra_env)
    task_args = launcher_testing.TASK_ARGS if task else ()
    return subprocess.run(
        ["bash", str(LAUNCH_SH), *task_args, *args],
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
    self.assertIn("cluster_config.local.inference_devices", res.stderr)

  def test_local_dry_run_includes_inference_node_when_chips_set(self):
    res = self._run_launch(
        [
            "--dry-run",
            "cluster_config.target=local",
            "agentic_grpo_config.beta=0.04",
            "cluster_config.local.inference_devices=4,5",
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
    # Rogue K8s environment variables fail fast before rendering.
    res = self._run_launch(args, extra_env={"KUEUE_QUEUE_NAME": "rogue-q"})
    self.assertNotEqual(res.returncode, 0)
    self.assertIn(
        "old example launchers are set: KUEUE_QUEUE_NAME", res.stderr
    )
    res = self._run_launch(args, extra_env={"K8S_NAMESPACE": "rogue-ns"})
    self.assertNotEqual(res.returncode, 0)
    self.assertIn(
        "old example launchers are set: K8S_NAMESPACE", res.stderr
    )
    res = self._run_launch(args, extra_env={"ROLLOUT_FAULT_TOLERANCE": "false"})
    self.assertNotEqual(res.returncode, 0)
    self.assertIn(
        "old example launchers are set: ROLLOUT_FAULT_TOLERANCE", res.stderr
    )
    for legacy_ft_var in (
        "MIN_ROLLOUT_WORKERS",
        "MIN_ROLLOUT_REPLICAS",
        "MAX_TASK_RETRIES",
        "POLL_RETRY_BUDGET_S",
        "MIN_ACTIVE_ROLLOUT_RATIO",
        "MIN_ACTIVE_ROLLOUT_GRACE_S",
    ):
      with self.subTest(var=legacy_ft_var):
        res = self._run_launch(args, extra_env={legacy_ft_var: "1"})
        self.assertNotEqual(res.returncode, 0)
        self.assertIn(
            f"old example launchers are set: {legacy_ft_var}", res.stderr
        )
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
    # The trainer takes these from `dist_training_config`.
    self.assertIn("MAXTEXT_EXTRA_FLAGS", legacy)
    # Environment variables read by `yaml_generator.py` that do not pass through
    # must fail fast if set in the caller's environment.
    yaml_gen_envs = _yaml_generator_env_vars()
    self.assertNotEmpty(yaml_gen_envs)
    self.assertEmpty(set(legacy) & _YAML_GEN_ENV_PASSTHROUGH)
    self.assertContainsSubset(yaml_gen_envs - _YAML_GEN_ENV_PASSTHROUGH, legacy)
    # `yaml_generator.py` reads these when `k8s.sh` leaves the matching flag
    # unset, so a caller's value would override `cluster_config.k8s`.
    self.assertContainsSubset(
        {"FAIL_FAST", "GANG_ID", "STARTUP_RETRIES", "TEARDOWN_GANG_ON_EXIT"},
        legacy,
    )
    self.assertIn("CKPT_D2H_CONCURRENT_GB", legacy)
    # The old launchers' settings the config replaced: the resolver's own
    # `WEIGHT_SYNC_MODE` export would overwrite a caller's value unseen.
    self.assertContainsSubset(
        {
            "INFERENCE_DEVICES",
            "INFERENCE_TPU_CHIPS",
            "KUEUE_QUEUE",
            "ROLLOUT_DEVICES",
            "ROLLOUT_TPU_CHIPS",
            "TRAINER_DEVICES",
            "TRAINER_EXPERT",
            "TRAINER_TPU_CHIPS",
            "WEIGHT_SYNC_MODE",
        },
        legacy,
    )
    # The Pathways, Raiden and reservation settings are config keys now
    # (`cluster_config.pathways`, `weight_sync_config.raiden`,
    # `cluster_config.k8s.tpu_reservation`).
    self.assertContainsSubset(
        {
            "COLOCATED_PYTHON_SIDECAR_IMAGE",
            "COLOCATED_PYTHON_SIDECAR_MEMORY",
            "COLOCATED_PYTHON_SIDECAR_SHM",
            "ENABLE_MULTI_NUMA",
            "PATHWAYS_PROXY_EXTRA_ARGS",
            "PATHWAYS_WORKER_EXTRA_ENV",
            "RAIDEN_BROADCAST_K",
            "TPU_RESERVATION",
            "TRAINER_IFRT_PROXY_LARGE_TRANSFER_THRESHOLD_BYTES",
        },
        legacy,
    )
    # `launch.sh` would ignore these, silently dropping the old launchers'
    # rejection sampling or logging the trajectories they turned off.
    self.assertContainsSubset(
        {
            "SAMPLER_RS",
            "SAMPLER_RS_MAX",
            "SAMPLER_RS_MIN",
            "SEQ_LOGPROB_ERROR_THRESHOLD",
            "TRAJECTORY_LOGGING",
        },
        legacy,
    )
    # Settings of the old mlperf recipe that only its launcher and the old
    # example mains read: `launch.sh` would silently ignore them.
    self.assertContainsSubset(
        {
            "FLOAT32_GATE_LOGITS",
            "FLOAT32_LOGITS",
            "INIT_TIMEOUT_S",
            "MAX_CONCURRENCY",
            "MAX_VERSION_SPAN",
            "MAX_WARMPOOL_REPLICAS",
            "ORCHESTRATOR_EXTRA_ENV",
            "OVERLONG_FILTER",
            "REMAT_POLICY",
            "ROLLOUT_EXTRA_ENV",
            "ROLLOUT_MESH_EXPERT",
            "RPC_TIMEOUT_S",
            "TRAINER_BASE_NUM_KV_HEADS",
            "TRAINER_LIBTPU_INIT_ARGS",
            "TRAINER_MAXTEXT_ATTENTION",
            "TRAINER_MESH_CONTEXT",
            "VLLM_ASYNC_SCHEDULING",
            "VLLM_CONFIG_JSON",
            "VLLM_DATA_PARALLEL_SIZE",
            "WARMUP_STEPS_FRACTION",
        },
        legacy,
    )
    # The sandbox placement: the sandbox code reads these, but `k8s.sh` and
    # `local.sh` set them from `kubernetes_config`, so the pods would drop a
    # caller's value, and `local.sh` would overwrite it whenever the config
    # sets them.
    self.assertContainsSubset(
        {
            "NAMESPACE",
            "NODE_SELECTOR_KEY",
            "NODE_SELECTOR_VAL",
            "SANDBOX_NAMESPACE",
            "SANDBOX_NODE_SELECTOR_KEY",
            "SANDBOX_NODE_SELECTOR_VAL",
        },
        legacy,
    )
    # The recipe's other variables are left alone: the processes read them (the
    # experimental DeepSWE environment, Raiden, vLLM, W&B), or the v2 recipe
    # and `yaml_generator.py` do. `local.sh`'s processes inherit them from the
    # caller's environment; the GKE pods don't, so there they only arrive
    # through the config (e.g. `cluster_config.roles.trainer.env`).
    self.assertEmpty(
        set(legacy)
        & {
            "IMAGE_REWRITE_PREFIX",
            "JOB_PREFIX",
            "OPENHANDS_MULTI_TOOL_CALLS",
            "OPENHANDS_SERVER_IMAGE",
            "RAIDEN_BROADCAST_HOST_RATIO",
            "RAIDEN_BROADCAST_PIPELINE_STAGES",
            "TUNIX_IMAGE",
            "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS",
            "WANDB_ENTITY",
            "WANDB_PROJECT",
        }
    )
    # Libraries read it; the nodes set it from the config and reject another
    # value in the environment (`node_utils.apply_library_env`).
    self.assertNotIn("VERIFY_WEIGHTS", legacy)

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
    self.assertIn("dist_training_config.maxtext_extra_config.", res.stderr)
    self.assertNotIn("Trainer command:", res.stdout)
    # MaxText reads an empty variable too: it would set `remat_policy` to "".
    res = self._run_launch(args, extra_env={"M_REMAT_POLICY": ""})
    self.assertNotEqual(res.returncode, 0)
    self.assertIn(
        "MaxText config overrides are set in the environment: M_REMAT_POLICY",
        res.stderr,
    )

  def test_k8s_pods_get_library_settings_as_config_arguments_only(self):
    res = self._run_launch([
        "--dry-run",
        "cluster_config.target=k8s",
        "cluster_config.k8s.tunix_image=test-image",
        "agentic_grpo_config.beta=0.0",
        "model_config.model_source=maxtext",
        "model_config.model_path=/ckpt/qwen3-1.7b/0/items",
        "model_config.kwargs.maxtext_model_name=qwen3-1.7b",
        "weight_sync_config.mode=raiden",
        "dist_training_config.maxtext_extra_config.remat_policy=custom",
        "dist_training_config.maxtext_extra_config.opt_type=sgd",
        "weight_sync_config.verify_weights=true",
    ])
    self.assertEqual(res.returncode, 0, msg=res.stderr)
    # The pods get the settings with the config arguments ...
    for arg in (
        "dist_training_config.maxtext_extra_config.remat_policy=custom",
        "dist_training_config.maxtext_extra_config.opt_type=sgd",
    ):
      self.assertIn(arg, res.stdout)
    # ... and the nodes set what the libraries read from the environment
    # (`node_utils.apply_library_env`).
    for name in (
        "MAXTEXT_EXTRA_FLAGS",
        "CKPT_D2H_CONCURRENT_GB",
        "VERIFY_WEIGHTS",
        "SKIP_JAX_PRECOMPILE",
    ):
      with self.subTest(name=name):
        self.assertNotIn(f"{name}=", res.stdout)

  def test_k8s_rejects_library_env_vars_from_the_caller(self):
    args = [
        "start",
        "--dry-run",
        "cluster_config.target=k8s",
        "cluster_config.k8s.tunix_image=test-image",
        "agentic_grpo_config.beta=0.0",
    ]
    # The nodes set these from the config; the pods don't inherit the
    # launcher's environment, so the launch fails instead of dropping them.
    res = self._run_launch(
        args,
        extra_env={"SKIP_JAX_PRECOMPILE": "0", "VERIFY_WEIGHTS": "true"},
    )
    self.assertNotEqual(res.returncode, 0)
    self.assertIn(
        "SKIP_JAX_PRECOMPILE VERIFY_WEIGHTS set in the environment", res.stderr
    )
    self.assertIn("vllm_config.skip_jax_precompile", res.stderr)

  def test_external_target_is_not_launched_here(self):
    # The resolver accepts the target, with an absolute artifact root.
    res = self._run_launch([
        "--dry-run",
        "cluster_config.target=external",
        "cluster_config.artifact_root=/artifacts/external_run",
    ])
    self.assertEqual(res.returncode, 1, msg=res.stderr)
    self.assertIn(
        "cluster_config.target=external runs are started by another launcher",
        res.stderr,
    )
    self.assertEqual(res.stdout, "")

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
        "cluster_config.rollout_replicas=2",
        "agentic_grpo_config.beta=0.0",
    ])
    self.assertEqual(res.returncode, 0, msg=res.stderr)
    self.assertIn("kubectl delete jobset testuser-orch -n default", res.stdout)
    self.assertIn("kubectl delete jobset testuser-train -n default", res.stdout)
    self.assertIn(
        "kubectl delete leaderworkerset testuser-roll-0 testuser-roll-1 -n"
        " default",
        res.stdout,
    )
    self.assertIn("RAIDEN_USE_FFI=1", res.stdout)
    self.assertIn("RAIDEN_USE_FFI=0", res.stdout)
    self.assertIn("ENABLE_PATHWAYS_PERSISTENCE=0", res.stdout)
    self.assertNotIn("ENABLE_PATHWAYS_PERSISTENCE=false", res.stdout)
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

    res_persistence = self._run_launch([
        "trainer",
        "--dry-run",
        "cluster_config.target=k8s",
        "cluster_config.k8s.tunix_image=test-image",
        "cluster_config.pathways.trainer.persistence=true",
        "agentic_grpo_config.beta=0.0",
    ])
    self.assertEqual(res_persistence.returncode, 0, msg=res_persistence.stderr)
    self.assertIn("ENABLE_PATHWAYS_PERSISTENCE=1", res_persistence.stdout)
    self.assertNotIn("ENABLE_PATHWAYS_PERSISTENCE=true", res_persistence.stdout)

  def test_k8s_rollout_pathways_persistence(self):
    args = [
        "rollout",
        "--dry-run",
        "cluster_config.target=k8s",
        "cluster_config.k8s.tunix_image=test-image",
        "agentic_grpo_config.beta=0.0",
        "cluster_config.k8s.rollout.tpu_slice=tpu7x:2x2x1",
    ]
    pathways = ["cluster_config.k8s.rollout.jobset_yaml=jobset.pathways.yaml"]
    persistence = ["cluster_config.pathways.rollout.persistence=true"]
    for overrides, expected in (
        (pathways, False),
        (pathways + persistence, True),
        # Persistence is a Pathways setting; other templates ignore it.
        (persistence, False),
    ):
      with self.subTest(overrides=overrides):
        res = self._run_launch(args + overrides)
        self.assertEqual(res.returncode, 0, msg=res.stderr)
        self.assertEqual(
            "ENABLE_PATHWAYS_PERSISTENCE=1" in res.stdout, expected
        )
        self.assertNotIn("ENABLE_PATHWAYS_PERSISTENCE=0", res.stdout)

  def test_k8s_requires_tunix_image(self):
    res = self._run_launch([
        "--dry-run",
        "cluster_config.target=k8s",
        "agentic_grpo_config.beta=0.0",
    ])
    self.assertNotEqual(res.returncode, 0)
    self.assertIn("cluster_config.k8s.tunix_image is empty", res.stderr)

  def test_k8s_rejects_reference_worker(self):
    # A KL penalty enables the reference worker, which k8s.sh does not deploy.
    res = self._run_launch([
        "start",
        "--dry-run",
        "cluster_config.target=k8s",
        "cluster_config.k8s.tunix_image=test-image",
        "agentic_grpo_config.beta=0.04",
    ])
    self.assertNotEqual(res.returncode, 0)
    self.assertIn("does not deploy a reference inference worker", res.stderr)
    self.assertNotIn("apiVersion:", res.stdout)

  def test_k8s_stop_command_only_deletes_resources(self):
    # Deleting the jobs needs no node image and no task, and a reference
    # worker (which k8s.sh does not deploy) is no reason to fail.
    res = self._run_launch(
        [
            "stop",
            "--dry-run",
            "cluster_config.target=k8s",
            "agentic_grpo_config.beta=0.04",
        ],
        task=False,
    )
    self.assertEqual(res.returncode, 0, msg=res.stderr)
    self.assertIn("kubectl delete jobset testuser-orch -n default", res.stdout)
    self.assertNotIn("apiVersion:", res.stdout)

  def test_commands_that_start_nodes_require_a_task(self):
    # `resolve_config_test` covers which keys the message names. The trainer
    # node does not check the task itself, so `launch.sh` is its only guard.
    k8s = ["cluster_config.target=k8s", "cluster_config.k8s.tunix_image=img"]
    for args in (
        ["start", "cluster_config.target=local"],
        ["start", *k8s],
        ["orchestrator", *k8s],
        ["trainer", *k8s],
        ["rollout", *k8s],
    ):
      with self.subTest(args=args[:2]):
        res = self._run_launch(["--dry-run", *args], task=False)
        self.assertNotEqual(res.returncode, 0)
        self.assertIn(
            "Error: The config names no task: data_module, env_class_path,"
            " agent_class_path, registry_module must be set.",
            res.stderr,
        )
        self.assertNotIn("command:", res.stdout)
        self.assertNotIn("apiVersion:", res.stdout)

  def test_unsupported_target_or_command_is_reported_before_the_task(self):
    for args, error in (
        (
            ["trainer", "cluster_config.target=local"],
            (
                "command 'trainer' is not supported for"
                " cluster_config.target=local"
            ),
        ),
        (
            # `target: external` requires an absolute artifact root.
            [
                "start",
                "cluster_config.target=external",
                "cluster_config.artifact_root=/tmp/artifacts",
            ],
            "target=external runs are started by another launcher",
        ),
    ):
      with self.subTest(args=args[:2]):
        res = self._run_launch(["--dry-run", *args], task=False)
        self.assertNotEqual(res.returncode, 0)
        self.assertIn(error, res.stderr)
        self.assertNotIn("names no task", res.stderr)

  def test_k8s_rollout_batches_manifests_and_deletes(self):
    tempdir = self.create_tempdir().full_path
    kubectl_log = os.path.join(tempdir, "kubectl_calls.log")
    applied_manifest = os.path.join(tempdir, "applied.yaml")
    kubectl_bin = os.path.join(tempdir, "kubectl")
    with open(kubectl_bin, "w") as f:
      f.write(
          "#!/bin/bash\n"
          f'echo "$*" >> "{kubectl_log}"\n'
          'if [[ "$*" == *"apply -f -"* ]]; then\n'
          f'  cat > "{applied_manifest}"\n'
          "fi\n"
      )
    os.chmod(kubectl_bin, 0o755)

    res = self._run_launch(
        [
            "rollout",
            "cluster_config.target=k8s",
            "cluster_config.k8s.tunix_image=test-image",
            "cluster_config.rollout_replicas=4",
            "agentic_grpo_config.beta=0.0",
        ],
        extra_env={
            "PATH": f"{tempdir}:{os.environ.get('PATH', '')}",
            "ENTER_KUBE_CONTEXT": "/dev/null",
        },
    )
    self.assertEqual(res.returncode, 0, msg=res.stderr)

    with open(kubectl_log) as f:
      calls = [line.strip() for line in f if line.strip()]
    self.assertEqual(
        calls,
        [
            (
                "delete leaderworkerset testuser-roll-0 testuser-roll-1"
                " testuser-roll-2 testuser-roll-3 -n default --ignore-not-found"
                " --wait=true"
            ),
            "apply -f -",
        ],
    )

    with open(applied_manifest) as f:
      docs = list(yaml.safe_load_all(f.read()))
    self.assertLen(docs, 4)
    self.assertEqual(
        [doc["metadata"]["name"] for doc in docs],
        [f"testuser-roll-{i}" for i in range(4)],
    )
    for i, doc in enumerate(docs):
      self.assertIn(f"DIST_RL_ROLLOUT_REPLICA={i}", yaml.safe_dump(doc))

  def test_k8s_rollout_compilation_cache_comes_from_the_config(self):
    args = [
        "start",
        "--dry-run",
        "cluster_config.target=k8s",
        "cluster_config.k8s.tunix_image=test-image",
        "agentic_grpo_config.beta=0.0",
    ]
    # The rollout node reads these, but from the launcher's environment they
    # would reach no pod: the launch fails instead.
    res = self._run_launch(
        args,
        extra_env={
            "JAX_CACHE_BUCKET": "stray",
            "SAVE_JAX_CACHE": "false",
            "VLLM_DISABLE_COMPILE_CACHE": "1",
        },
    )
    self.assertNotEqual(res.returncode, 0)
    self.assertIn(
        "JAX_CACHE_BUCKET SAVE_JAX_CACHE VLLM_DISABLE_COMPILE_CACHE set in the"
        " environment",
        res.stderr,
    )
    self.assertIn("cluster_config.k8s.jax_cache", res.stderr)
    self.assertNotIn("apiVersion:", res.stdout)
    # A generic `BUCKET` only gets a warning, and reaches no pod either.
    res = self._run_launch(args, extra_env={"BUCKET": "stray"})
    self.assertEqual(res.returncode, 0, msg=res.stderr)
    self.assertIn("don't read BUCKET", res.stderr)
    self.assertNotIn("stray", res.stdout)
    self.assertNotIn("JAX_CACHE_GCS_DIR=", res.stdout)

    res = self._run_launch(
        [
            *args,
            "cluster_config.k8s.jax_cache.bucket=my-bucket",
            "cluster_config.k8s.jax_cache.local_dir=/cache",
        ],
        extra_env={"BUCKET": "stray"},
    )
    self.assertEqual(res.returncode, 0, msg=res.stderr)
    self.assertNotIn("stray", res.stdout)
    uri = (
        r"gs://my-bucket/jax_cache/v5e/qwen3-1\.7b/"
        r"rollout_4x4_ep1_tp2_[0-9a-f]{10}"
    )
    # The rollout node restores and saves the cache there, ...
    self.assertRegex(
        res.stdout,
        rf"LOCAL_JAX_CACHE_DIR=/cache ROLLOUT_JAX_CACHE_GCS_DIR={uri}"
        r" SAVE_JAX_CACHE=true ",
    )
    # ... and the multi-host rollout's Ray workers download it first.
    self.assertRegex(res.stdout, rf"gcs_cache download /cache {uri} ")
    # The pods get no other cache location, and no slice for the cache key.
    self.assertNotRegex(
        res.stdout, r"JAX_CACHE_BUCKET=|(?<!ROLLOUT_)JAX_CACHE_GCS_DIR="
    )
    self.assertNotIn("ROLLOUT_TPU_SLICE=", res.stdout)
    self.assertNotIn("VLLM_DISABLE_COMPILE_CACHE", res.stdout)

    # `disable` turns off both `gcs_cache` and vLLM's local XLA compile cache on
    # the head and Ray workers.
    res = self._run_launch([
        *args,
        "cluster_config.k8s.jax_cache.bucket=my-bucket",
        "cluster_config.k8s.jax_cache.disable=true",
    ])
    self.assertEqual(res.returncode, 0, msg=res.stderr)
    self.assertIn(
        "DISABLE_JAX_CACHE=1 VLLM_DISABLE_COMPILE_CACHE=1 ", res.stdout
    )
    self.assertRegex(
        res.stdout,
        r"export VLLM_DISABLE_COMPILE_CACHE=1\n\s*"
        r'echo "Starting Ray Head\.\.\."',
    )
    self.assertRegex(
        res.stdout,
        r"export VLLM_DISABLE_COMPILE_CACHE=1\n\s*ray start --address=",
    )
    self.assertNotIn("JAX_COMPILATION_CACHE_DIR", res.stdout)
    self.assertNotIn("gcs_cache download", res.stdout)

  def test_k8s_passes_otel_settings_to_every_pod(self):
    base_args = [
        "start",
        "--dry-run",
        "cluster_config.target=k8s",
        "cluster_config.k8s.tunix_image=test-image",
        "agentic_grpo_config.beta=0.0",
    ]
    res = self._run_launch([
        *base_args,
        "otel_config.enable=true",
        "otel_config.export_interval_ms=5000",
        "otel_config.dashboard_id=dash-1",
        "otel_config.dashboard_project=proj-1",
    ])
    self.assertEqual(res.returncode, 0, msg=res.stderr)
    mains = {
        "tunix.dist_rl.launch.nodes.orchestrator.main",
        "tunix.dist_rl.launch.nodes.trainer.main",
        "tunix.dist_rl.launch.nodes.rollout.main",
    }
    # Every pod gets the pinned collector id and the `otel_config.*` arguments,
    # with no `OTEL_*` environment variable prefix.
    pinned = set(
        re.findall(
            r"--process_main=(\S+) [^\n]*"
            r"cluster_config\.k8s\.otel_id=testuser-otel [^\n]*"
            r"otel_config\.enable=true otel_config\.export_interval_ms=5000 "
            r"otel_config\.dashboard_id=dash-1 "
            r"otel_config\.dashboard_project=proj-1",
            res.stdout,
        )
    )
    self.assertEqual(pinned, mains)
    self.assertNotIn("OTEL_", res.stdout)

    # Legacy `OTEL_*` environment variables fail fast with a migration hint.
    legacy = self._run_launch(
        base_args,
        extra_env={"OTEL_EXPORTER_OTLP_ENDPOINT": "http://testuser-otel:4317"},
    )
    self.assertNotEqual(legacy.returncode, 0)
    self.assertIn("OTEL_EXPORTER_OTLP_ENDPOINT", legacy.stderr)
    self.assertIn("otel_config.enable=true", legacy.stderr)

  def test_k8s_passes_restart_and_scheduling_settings_to_the_jobsets(self):
    args = [
        "start",
        "--dry-run",
        "cluster_config.target=k8s",
        "cluster_config.k8s.tunix_image=test-image",
        "cluster_config.run_id=run_k8s_gang",
        "agentic_grpo_config.beta=0.0",
        # The default `leaderworkerset.*.yaml` rollout has no fail-fast mode.
        "cluster_config.k8s.rollout.jobset_yaml=jobset.mcjax.ray.yaml",
    ]
    res = self._run_launch([
        *args,
        "cluster_config.k8s.fail_fast=true",
        "cluster_config.k8s.startup_retries=5",
        "cluster_config.k8s.teardown_gang_on_exit=true",
        "cluster_config.k8s.preemptible=true",
        "cluster_config.k8s.priority_class=rl-high",
        "cluster_config.k8s.service_account=rl-sa",
        "cluster_config.k8s.termination_grace_seconds=120",
        "cluster_config.k8s.kueue_queue_name=rl-queue",
        "cluster_config.k8s.cpu_nodepool=cpu-np",
        "cluster_config.roles.orchestrator.memory.request=64Gi",
        "cluster_config.k8s.head_excluded_nodepools=[cpu-np,sandbox-pool]",
    ])
    self.assertEqual(res.returncode, 0, msg=res.stderr)
    jobsets = _dry_run_jobsets(res.stdout)
    self.assertCountEqual(
        jobsets, ["testuser-orch", "testuser-train", "testuser-roll"]
    )
    for name, jobset in jobsets.items():
      with self.subTest(jobset=name):
        labels = jobset["metadata"]["labels"]
        self.assertEqual(labels["kueue.x-k8s.io/queue-name"], "rl-queue")
        self.assertEqual(labels["scheduling.x-k8s.io/gang-id"], "run_k8s_gang")
        self.assertEqual(labels["scheduling.x-k8s.io/preemptible"], "true")
        for pod in _pod_specs(jobset):
          self.assertEqual(pod["priorityClassName"], "rl-high")
          self.assertEqual(pod["serviceAccountName"], "rl-sa")
          self.assertEqual(pod["terminationGracePeriodSeconds"], 120)
    # The workers fail fast: an exit-0 guard, no in-place pod restarts, and up
    # to `startup_retries` whole-JobSet recreations.
    for name in ("testuser-train", "testuser-roll"):
      with self.subTest(jobset=name):
        spec = jobsets[name]["spec"]
        self.assertEqual(spec["failurePolicy"]["maxRestarts"], 5)
        self.assertIn("forcing exit code 1", json.dumps(spec))
    rollout_job = jobsets["testuser-roll"]["spec"]["replicatedJobs"][0]
    self.assertEqual(rollout_job["template"]["spec"]["backoffLimit"], 0)
    (rollout_pod,) = _pod_specs(jobsets["testuser-roll"])
    self.assertEqual(rollout_pod["restartPolicy"], "Never")
    # The orchestrator runs in the CPU node pool and, when it exits, deletes
    # the other JobSets of its run.
    (orch_pod,) = _pod_specs(jobsets["testuser-orch"])
    self.assertEqual(
        orch_pod["nodeSelector"], {"cloud.google.com/gke-nodepool": "cpu-np"}
    )
    orch_container = orch_pod["containers"][0]
    self.assertEqual(
        orch_container["resources"]["requests"]["memory"], "64Gi"
    )
    self.assertIn(
        '-l "scheduling.x-k8s.io/gang-id=run_k8s_gang"'
        ' --field-selector="metadata.name!=testuser-orch"',
        orch_container["command"][2],
    )
    # The Pathways trainer head pod, on a pw-node host, keeps out of the listed
    # node pools.
    train_head_pod = _pod_specs(jobsets["testuser-train"])[0]
    self.assertEqual(
        train_head_pod["affinity"]["nodeAffinity"][
            "requiredDuringSchedulingIgnoredDuringExecution"
        ]["nodeSelectorTerms"],
        [{
            "matchExpressions": [{
                "key": "cloud.google.com/gke-nodepool",
                "operator": "NotIn",
                "values": ["cpu-np", "sandbox-pool"],
            }]
        }],
    )

    # Unset, the settings keep the template defaults.
    res = self._run_launch(args)
    self.assertEqual(res.returncode, 0, msg=res.stderr)
    jobsets = _dry_run_jobsets(res.stdout)
    for name, jobset in jobsets.items():
      with self.subTest(jobset=name):
        self.assertNotIn("labels", jobset["metadata"])
        self.assertNotIn("forcing exit code 1", json.dumps(jobset))
    for name in ("testuser-train", "testuser-roll"):
      with self.subTest(jobset=name):
        spec = jobsets[name]["spec"]
        self.assertEqual(spec["failurePolicy"]["maxRestarts"], 3)
    (rollout_pod,) = _pod_specs(jobsets["testuser-roll"])
    self.assertEqual(rollout_pod["restartPolicy"], "OnFailure")
    (orch_pod,) = _pod_specs(jobsets["testuser-orch"])
    self.assertNotIn("kubectl delete", orch_pod["containers"][0]["command"][2])
    train_head_pod = _pod_specs(jobsets["testuser-train"])[0]
    self.assertEqual(list(train_head_pod["affinity"]), ["podAffinity"])

  def test_k8s_pathways_rollout_gets_cpu_machine_and_memory_settings(self):
    args = [
        "start",
        "--dry-run",
        "cluster_config.target=k8s",
        "cluster_config.k8s.tunix_image=test-image",
        "agentic_grpo_config.beta=0.0",
        "cluster_config.k8s.rollout.jobset_yaml=jobset.pathways.yaml",
        "cluster_config.k8s.rollout.tpu_slice=tpu7x:2x2x1",
        "cluster_config.k8s.use_dynamic_slicing=true",
        "cluster_config.k8s.cpu_machine=n2-highmem-32",
        "cluster_config.pathways.proxy_memory.limit=190G",
        "cluster_config.pathways.proxy_memory.request=24G",
        "cluster_config.pathways.rm_memory.request=8G",
        "cluster_config.roles.rollout.memory.request=56G",
        "cluster_config.roles.rollout.memory.limit=80G",
        "cluster_config.pathways.server_memory.request=120G",
        # Settings that used to be environment variables of the launcher.
        "cluster_config.pathways.worker_env.MY_WORKER_VAR=a b",
        "cluster_config.pathways.proxy_extra_args=--xla_a=1 --xla_b=2",
        (
            "cluster_config.pathways.trainer.ifrt_proxy_large_transfer_threshold"
            "_bytes=1048576"
        ),
        "cluster_config.k8s.tpu_reservation=my-res",
        "weight_sync_config.raiden.data_nics=eth0,eth1",
        "weight_sync_config.raiden.multi_numa=true",
        "weight_sync_config.raiden.broadcast_k=128",
    ]
    res = self._run_launch(args)
    self.assertEqual(res.returncode, 0, msg=res.stderr)
    jobsets = _dry_run_jobsets(res.stdout)
    rollout_jobset = jobsets["testuser-roll"]
    rjobs = {
        rj["name"]: rj["template"]["spec"]["template"]["spec"]
        for rj in rollout_jobset["spec"]["replicatedJobs"]
    }

    # Head pod (proc) uses node.kubernetes.io/instance-type selector.
    proc_spec = rjobs["proc"]
    self.assertEqual(
        proc_spec["nodeSelector"],
        {"node.kubernetes.io/instance-type": "n2-highmem-32"},
    )

    # Init containers (pathways-rm, pathways-proxy) and main user container.
    init_containers = {c["name"]: c for c in proc_spec["initContainers"]}
    self.assertEqual(
        init_containers["pathways-rm"]["resources"]["requests"]["memory"], "8G"
    )
    self.assertEqual(
        init_containers["pathways-proxy"]["resources"]["requests"]["memory"],
        "24G",
    )
    self.assertEqual(
        init_containers["pathways-proxy"]["resources"]["limits"]["memory"],
        "190G",
    )

    containers = {c["name"]: c for c in proc_spec["containers"]}
    self.assertEqual(
        containers["main"]["resources"]["requests"]["memory"], "56G"
    )
    self.assertEqual(
        containers["main"]["resources"]["limits"]["memory"], "80G"
    )
    # The trainer keeps its own request and limit
    # (`cluster_config.roles.trainer.memory`).
    trainer_proc = next(
        rj["template"]["spec"]["template"]["spec"]
        for rj in jobsets["testuser-train"]["spec"]["replicatedJobs"]
        if rj["name"] == "proc"
    )
    trainer_main = next(
        c for c in trainer_proc["containers"] if c["name"] == "main"
    )
    self.assertEqual(trainer_main["resources"]["requests"]["memory"], "48G")
    self.assertEqual(trainer_main["resources"]["limits"]["memory"], "70G")
    # Only the trainer gets the IFRT transfer threshold; the rollout keeps the
    # generator's default.
    threshold = "IFRT_PROXY_GRPC_LARGE_TRANSFER_OPTIMIZATION_THRESHOLD_BYTES"
    self.assertIn(
        {"name": threshold, "value": "1048576"}, trainer_main["env"]
    )
    self.assertIn(
        {"name": threshold, "value": str(2**63 - 1)}, containers["main"]["env"]
    )
    for flag in ("--xla_a=1", "--xla_b=2"):
      self.assertIn(flag, init_containers["pathways-proxy"]["args"])

    # Worker pod (pw-node) container (pathways-worker).
    pw_spec = rjobs["pw-node"]
    pw_containers = {c["name"]: c for c in pw_spec["containers"]}
    self.assertEqual(
        pw_containers["pathways-worker"]["resources"]["requests"]["memory"],
        "120G",
    )
    worker_env = pw_containers["pathways-worker"]["env"]
    for name, value in (
        ("MY_WORKER_VAR", "a b"),
        ("TPU_RAIDEN_DATA_NICS", "eth0,eth1"),
        ("ENABLE_MULTI_NUMA", "1"),
        ("RAIDEN_BROADCAST_K", "128"),
    ):
      self.assertIn({"name": name, "value": value}, worker_env)
    self.assertEqual(
        pw_spec["nodeSelector"]["cloud.google.com/reservation-name"], "my-res"
    )

    # With `head_nodepool` set alongside `use_dynamic_slicing=true`, both the
    # trainer and rollout Pathways head (`proc`) pods pin to that node pool.
    res = self._run_launch([*args, "cluster_config.k8s.head_nodepool=head-np"])
    self.assertEqual(res.returncode, 0, msg=res.stderr)
    jobsets = _dry_run_jobsets(res.stdout)
    for name in ("testuser-train", "testuser-roll"):
      with self.subTest(jobset=name):
        rjobs = {
            rj["name"]: rj["template"]["spec"]["template"]["spec"]
            for rj in jobsets[name]["spec"]["replicatedJobs"]
        }
        self.assertEqual(
            rjobs["proc"]["nodeSelector"],
            {"cloud.google.com/gke-nodepool": "head-np"},
        )
        self.assertEqual(
            rjobs["proc"]["tolerations"],
            [{
                "key": "cloud.google.com/gke-nodepool",
                "operator": "Equal",
                "value": "head-np",
                "effect": "NoSchedule",
            }],
        )

  def test_k8s_checkpoint_gcsfuse_mounts_on_the_trainer_alone(self):
    ckpt_gcsfuse = "cluster_config.pathways.trainer.checkpoint_gcsfuse"
    res = self._run_launch([
        "start",
        "--dry-run",
        "cluster_config.target=k8s",
        "cluster_config.k8s.tunix_image=test-image",
        "agentic_grpo_config.beta=0.0",
        "cluster_config.k8s.rollout.jobset_yaml=jobset.pathways.yaml",
        "cluster_config.k8s.rollout.tpu_slice=tpu7x:2x2x1",
        f"{ckpt_gcsfuse}.bucket=gs://my-rapid-bucket/",
        f"{ckpt_gcsfuse}.mount_path=/mnt/ckpt",
        f"{ckpt_gcsfuse}.mount_options=implicit-dirs,only-dir=run1",
    ])
    self.assertEqual(res.returncode, 0, msg=res.stderr)
    jobsets = _dry_run_jobsets(res.stdout)

    volume_mounts = {}
    for job in jobsets["testuser-train"]["spec"]["replicatedJobs"]:
      pod = job["template"]["spec"]["template"]
      with self.subTest(job=job["name"]):
        self.assertEqual(
            pod["metadata"]["annotations"]["gke-gcsfuse/volumes"], "true"
        )
        self.assertIn(
            {
                "name": "ckpt-gcsfuse",
                "csi": {
                    "driver": "gcsfuse.csi.storage.gke.io",
                    "volumeAttributes": {
                        "bucketName": "my-rapid-bucket",
                        "mountOptions": "implicit-dirs,only-dir=run1",
                        "hostNetworkPodKSA": "true",
                    },
                },
            },
            pod["spec"]["volumes"],
        )
      for container in pod["spec"]["containers"]:
        volume_mounts[container["name"]] = container["volumeMounts"]
    for name in ("main", "pathways-worker"):
      with self.subTest(container=name):
        self.assertIn(
            {"mountPath": "/mnt/ckpt", "name": "ckpt-gcsfuse"},
            volume_mounts[name],
        )

    # The rollout renders the same Pathways template, but mounts no bucket.
    for job in jobsets["testuser-roll"]["spec"]["replicatedJobs"]:
      pod = job["template"]["spec"]["template"]
      with self.subTest(rollout_job=job["name"]):
        self.assertNotIn("gke-gcsfuse/volumes", pod["metadata"]["annotations"])
        self.assertNotIn(
            "ckpt-gcsfuse", [v["name"] for v in pod["spec"]["volumes"]]
        )

  def test_otel_commands_deploy_and_delete_the_collector(self):
    # The collector needs no `cluster_config.k8s.tunix_image` and no task, and
    # a reference worker (which k8s.sh does not deploy) is no reason to fail.
    k8s = ["cluster_config.target=k8s", "agentic_grpo_config.beta=0.04"]
    res = self._run_launch(
        ["otel", "--dry-run", *k8s, "cluster_config.ports.otel=4318"],
        task=False,
    )
    self.assertEqual(res.returncode, 0, msg=res.stderr)
    self.assertIn(
        "kubectl delete deployment testuser-otel -n default", res.stdout
    )
    self.assertIn(
        "kubectl create configmap testuser-otel-config -n default"
        " --from-file=otel-collector.yaml=",
        res.stdout,
    )
    manifests = {}
    for section in res.stdout.split("\n---\n")[1:]:
      for doc in yaml.safe_load_all(section):
        if doc:
          manifests[doc["kind"]] = doc
    self.assertEqual(set(manifests), {"Deployment", "Service"})
    service = manifests["Service"]
    self.assertEqual(service["metadata"]["name"], "testuser-otel")
    self.assertEqual(service["spec"]["ports"][0]["port"], 4318)

    res = self._run_launch(["stop-otel", "--dry-run", *k8s], task=False)
    self.assertEqual(res.returncode, 0, msg=res.stderr)
    self.assertIn("kubectl delete service testuser-otel -n default", res.stdout)
    self.assertNotIn("apiVersion:", res.stdout)

    # Single-host runs use a collector started separately.
    res = self._run_launch([
        "otel",
        "--dry-run",
        "cluster_config.target=local",
        "agentic_grpo_config.beta=0.0",
    ])
    self.assertNotEqual(res.returncode, 0)
    self.assertIn("command 'otel' is not supported", res.stderr)

    # The previous k8s launcher's collector settings are config keys now.
    res = self._run_launch(
        ["otel", "--dry-run", *k8s], extra_env={"OTEL_PORT": "4318"}
    )
    self.assertNotEqual(res.returncode, 0)
    self.assertIn("OTEL_PORT", res.stderr)

  def test_role_env_and_sandbox_settings_render_into_processes(self):
    role_args = [
        "agentic_grpo_config.beta=0.0",
        "cluster_config.roles.orchestrator.env.ORCH_FLAG=x",
        "cluster_config.roles.trainer.env.TRAIN_FLAG='a b'",
        "cluster_config.roles.rollout.env.ROLL_FLAG='0'",
    ]
    local_res = self._run_launch(
        ["--dry-run", "cluster_config.target=local", *role_args]
    )
    self.assertEqual(local_res.returncode, 0, msg=local_res.stderr)
    for expected in (
        "Trainer environment: TRAIN_FLAG='a b'",
        "Rollout environment: ROLL_FLAG=0",
        "Orchestrator environment: ORCH_FLAG=x",
    ):
      self.assertIn(expected, local_res.stdout)
    inference_res = self._run_launch([
        "--dry-run",
        "cluster_config.target=local",
        "agentic_grpo_config.beta=0.04",
        "cluster_config.local.inference_devices=4,5",
        "cluster_config.roles.inference.env.INF_FLAG=y",
    ])
    self.assertEqual(inference_res.returncode, 0, msg=inference_res.stderr)
    self.assertIn("Inference environment: INF_FLAG=y", inference_res.stdout)

    k8s_res = self._run_launch([
        "start",
        "--dry-run",
        "cluster_config.target=k8s",
        "cluster_config.k8s.tunix_image=test-image",
        "cluster_config.k8s.orchestrator_id=custom-orch",
        "kubernetes_config.namespace=swe-ns",
        "kubernetes_config.node_selector_key=cloud.google.com/gke-nodepool",
        "kubernetes_config.node_selector_val=swe-pool",
        "cluster_config.k8s.rollout.jobset_yaml=jobset.mcjax.ray.yaml",
        *role_args,
    ])
    self.assertEqual(k8s_res.returncode, 0, msg=k8s_res.stderr)
    # Multi-host Ray rollouts export the role env on the head and worker hosts
    # before starting Ray.
    self.assertEqual(k8s_res.stdout.count(" && export ROLL_FLAG=0"), 2)
    for expected in ("ORCH_FLAG=x ", "TRAIN_FLAG='a b' ", "ROLL_FLAG=0 "):
      self.assertIn(f"{expected}python -m", k8s_res.stdout)
    self.assertEqual(
        set(
            re.findall(
                r"ORCHESTRATOR_ID=custom-orch NAMESPACE=swe-ns "
                r"NODE_SELECTOR_KEY=cloud\.google\.com/gke-nodepool "
                r"NODE_SELECTOR_VAL=swe-pool .*--process_main=(\S+)",
                k8s_res.stdout,
            )
        ),
        {
            "tunix.dist_rl.launch.nodes.orchestrator.main",
            "tunix.dist_rl.launch.nodes.rollout.main",
        },
    )

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

  def test_local_live_download_model_weights_knob(self):
    with self.assertRaisesRegex(
        TypeError, "cluster_config.local.download_model_weights must be a bool"
    ):
      resolve_config.initialize(
          ["cluster_config.local.download_model_weights=maybe"]
      )

    tempdir = pathlib.Path(self.create_tempdir().full_path)
    real_py = shlex.quote(sys.executable or "python3")
    stub_py = tempdir / "stub_python.sh"
    stub_py.write_text(
        "#!/bin/bash\n"
        "set -e\n"
        'if [[ "$1" == */resolve_config.py ]]; then\n'
        f'  exec {real_py} "$@"\n'
        "fi\n"
        'if [[ "$1" == "-" ]]; then\n'
        "  script=$(cat)\n"
        '  if [[ "$script" == *"hf_pipeline"* ]]; then\n'
        "    exit 99\n"
        "  fi\n"
        "  exit 0\n"
        "fi\n"
        'if [[ "$*" == *"--discovery_addrs="* ]]; then\n'
        "  exec sleep 30\n"
        "fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    stub_py.chmod(0o755)
    artifact_root = tempdir / "artifacts"
    base_args = [
        "cluster_config.target=local",
        f"cluster_config.artifact_root={artifact_root}",
        "model_config.model_id=local/non_hf_tokenizer_dir",
        "agentic_grpo_config.beta=0.0",
        "cluster_config.local.worker_shutdown_grace_secs=0",
        "cluster_config.local.trainer_shutdown_grace_secs=0",
    ]
    # Default download_model_weights=true runs pre-download even when custom
    # wrapper mains (e.g. benchmarks/rl_efficiency) are configured.
    wrapped_default = self._run_launch(
        [
            *base_args,
            "cluster_config.roles.trainer.main=custom.wrapper.trainer_main",
        ],
        extra_env={"PYTHON_BIN": str(stub_py)},
    )
    self.assertEqual(
        wrapped_default.returncode,
        99,
        msg=wrapped_default.stdout + wrapped_default.stderr,
    )

    skipped = self._run_launch(
        [
            *base_args,
            "cluster_config.local.download_model_weights=false",
            "cluster_config.roles.trainer.main=custom.mock.trainer_main",
            "cluster_config.roles.rollout.main=custom.mock.rollout_main",
            "weight_sync_config.mode=noop",
        ],
        extra_env={"PYTHON_BIN": str(stub_py)},
    )
    self.assertEqual(skipped.returncode, 0, msg=skipped.stdout + skipped.stderr)
    self.assertIn(
        "Distributed RL pipeline finished successfully.", skipped.stdout
    )

  def test_local_auto_detect_devices_gpu(self):
    tempdir = pathlib.Path(self.create_tempdir().full_path)
    dev_dir = tempdir / "dev"
    dev_dir.mkdir()
    for name in (
        "nvidia0",
        "nvidia1",
        "nvidia2",
        "nvidia3",
        "nvidia0_cap",
        "nvidiactl",
    ):
      (dev_dir / name).write_text("", encoding="utf-8")

    # Test `/dev/nvidia[0-9]*` node detection when `nvidia-smi` is unavailable.
    common_sh = LAUNCH_DIR / "common.sh"
    res_dev = subprocess.run(
        [
            "bash",
            "-c",
            f'PATH=/nonexistent; source "{common_sh}" &&'
            f' detect_gpu_count "{dev_dir}"',
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    self.assertEqual(res_dev.stdout.strip(), "4")

    # When `nvidia-smi` exits non-zero with an error message on stdout,
    # `detect_gpu_count` ignores its stdout and falls back to `/dev`.
    fail_bin_dir = tempdir / "fail_bin"
    fail_bin_dir.mkdir()
    fail_smi = fail_bin_dir / "nvidia-smi"
    fail_smi.write_text(
        "#!/bin/bash\n"
        "printf 'NVIDIA-SMI has failed because it could not communicate\\n'\n"
        "exit 9\n",
        encoding="utf-8",
    )
    fail_smi.chmod(0o755)
    empty_dev = tempdir / "empty_dev"
    empty_dev.mkdir()
    res_fail_smi = subprocess.run(
        [
            "bash",
            "-c",
            f'PATH="{fail_bin_dir}:/usr/bin:/bin"; source "{common_sh}" &&'
            f' detect_gpu_count "{empty_dev}"',
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    self.assertEqual(res_fail_smi.stdout.strip(), "0")

    # Test `nvidia-smi` detection in `local.sh` across disaggregated and
    # colocated GPU runs.
    bin_dir = tempdir / "bin"
    bin_dir.mkdir()
    fake_smi = bin_dir / "nvidia-smi"
    fake_smi.write_text(
        "#!/bin/bash\n"
        "printf 'GPU 0\\nGPU 1\\nGPU 2\\nGPU 3\\n'\n",
        encoding="utf-8",
    )
    fake_smi.chmod(0o755)
    path_with_smi = f"{bin_dir}:{os.environ.get('PATH', '')}"

    res_disagg = self._run_launch(
        [
            "--dry-run",
            "cluster_config.target=local",
            "execution_targets.target=jax:gpu",
            "agentic_grpo_config.beta=0.0",
            "cluster_config.local.auto_detect_devices=true",
        ],
        extra_env={"PATH": path_with_smi},
    )
    self.assertEqual(res_disagg.returncode, 0, msg=res_disagg.stderr)
    self.assertIn(
        "Discovered GPU devices: Trainer=[0,1], Rollout=[2,3], Inference=[]",
        res_disagg.stdout,
    )

    res_coloc = self._run_launch(
        [
            "--dry-run",
            "cluster_config.target=local",
            "execution_targets.target=jax:gpu",
            "execution_targets.placement=colocated",
            "agentic_grpo_config.beta=0.04",
            "cluster_config.local.auto_detect_devices=true",
        ],
        extra_env={"PATH": path_with_smi},
    )
    self.assertEqual(res_coloc.returncode, 0, msg=res_coloc.stderr)
    self.assertIn(
        "Discovered GPU devices: Trainer=[0,1,2,3], Rollout=[0,1,2,3],"
        " Inference=[0,1,2,3]",
        res_coloc.stdout,
    )

  def test_configure_local_worker_env(self):
    local_sh = (LAUNCH_DIR / "local.sh").read_text()
    fn_match = re.search(
        r"(?ms)^configure_local_worker_env\(\) \{.*?^\}", local_sh
    )
    self.assertIsNotNone(fn_match)
    fn_src = fn_match.group(0)
    common_sh = LAUNCH_DIR / "common.sh"
    tracked_keys = (
        "CUDA_VISIBLE_DEVICES",
        "TPU_VISIBLE_DEVICES",
        "TPU_VISIBLE_CHIPS",
        "TPU_CHIPS_PER_HOST_BOUNDS",
        "TPU_HOST_BOUNDS",
        "LIBTPU_INIT_ARGS",
        "JAX_PLATFORMS",
        "JAX_BACKEND_TARGET",
        "TRAINER_PATHWAYS_LOCAL_INIT",
        "ALLOW_MULTIPLE_LIBTPU_LOAD",
    )

    def run_worker_env(
        framework: str, accelerator: str, devices: str, use_pathways: str
    ) -> dict[str, str | None]:
      script = (
          f'source "{common_sh}"\n'
          "export LOCAL_TPU_CHIPS_PER_HOST_BOUNDS=1,2,1\n"
          "export LOCAL_TPU_HOST_BOUNDS=1,1,1\n"
          "export LOCAL_JAX_BACKEND_TARGET=grpc://127.0.0.1:29000\n"
          "export LOCAL_ALLOW_MULTIPLE_LIBTPU_LOAD=true\n"
          "export CUDA_VISIBLE_DEVICES=stale_gpu\n"
          "export TPU_VISIBLE_DEVICES=stale_tpu\n"
          "export TPU_VISIBLE_CHIPS=stale_tpu\n"
          "export TPU_CHIPS_PER_HOST_BOUNDS=stale_bounds\n"
          "export TPU_HOST_BOUNDS=stale_hosts\n"
          "export LIBTPU_INIT_ARGS=stale_args\n"
          "export JAX_PLATFORMS=stale_plat\n"
          f"{fn_src}\n"
          f'configure_local_worker_env "{framework}" "{accelerator}"'
          f' "{devices}" "{use_pathways}"\n'
          "env -0\n"
      )
      res = subprocess.run(
          ["bash", "-c", script], capture_output=True, text=True, check=True
      )
      env_map = dict(
          item.split("=", 1) for item in res.stdout.split("\0") if "=" in item
      )
      return {k: env_map.get(k) for k in tracked_keys}

    gpu_env = run_worker_env("jax", "gpu", "0,1", "false")
    self.assertEqual(gpu_env["CUDA_VISIBLE_DEVICES"], "0,1")
    self.assertEqual(gpu_env["JAX_PLATFORMS"], "cuda,cpu")
    for tpu_key in (
        "TPU_VISIBLE_DEVICES",
        "TPU_VISIBLE_CHIPS",
        "TPU_CHIPS_PER_HOST_BOUNDS",
        "TPU_HOST_BOUNDS",
        "LIBTPU_INIT_ARGS",
    ):
      self.assertIsNone(gpu_env[tpu_key], msg=tpu_key)

    gpu_empty_env = run_worker_env("jax", "gpu", "", "false")
    self.assertIsNone(gpu_empty_env["CUDA_VISIBLE_DEVICES"])
    self.assertEqual(gpu_empty_env["JAX_PLATFORMS"], "cuda,cpu")

    jax_tpu_env = run_worker_env("jax", "tpu", "2,3", "false")
    self.assertIsNone(jax_tpu_env["CUDA_VISIBLE_DEVICES"])
    self.assertEqual(jax_tpu_env["JAX_PLATFORMS"], "tpu,cpu")
    self.assertEqual(jax_tpu_env["TPU_VISIBLE_DEVICES"], "2,3")
    self.assertEqual(jax_tpu_env["TPU_VISIBLE_CHIPS"], "2,3")
    self.assertEqual(jax_tpu_env["TPU_CHIPS_PER_HOST_BOUNDS"], "1,2,1")
    self.assertEqual(jax_tpu_env["TPU_HOST_BOUNDS"], "1,1,1")
    self.assertEqual(
        jax_tpu_env["LIBTPU_INIT_ARGS"],
        "--deepsea_chips_per_host_bounds=1,2,1 --deepsea_host_bounds=1,1,1",
    )
    self.assertEqual(jax_tpu_env["ALLOW_MULTIPLE_LIBTPU_LOAD"], "1")

    torch_tpu_env = run_worker_env("torch", "tpu", "0,1", "false")
    self.assertIsNone(torch_tpu_env["CUDA_VISIBLE_DEVICES"])
    self.assertIsNone(torch_tpu_env["JAX_PLATFORMS"])
    self.assertEqual(torch_tpu_env["TPU_VISIBLE_DEVICES"], "0,1")
    self.assertEqual(torch_tpu_env["TPU_VISIBLE_CHIPS"], "0,1")

    pathways_tpu_env = run_worker_env("jax", "tpu", "0,1", "true")
    self.assertIsNone(pathways_tpu_env["CUDA_VISIBLE_DEVICES"])
    self.assertEqual(pathways_tpu_env["JAX_PLATFORMS"], "proxy,cpu")
    self.assertEqual(
        pathways_tpu_env["JAX_BACKEND_TARGET"], "grpc://127.0.0.1:29000"
    )
    self.assertEqual(pathways_tpu_env["TRAINER_PATHWAYS_LOCAL_INIT"], "1")
    for tpu_key in (
        "TPU_VISIBLE_DEVICES",
        "TPU_VISIBLE_CHIPS",
        "LIBTPU_INIT_ARGS",
    ):
      self.assertIsNone(pathways_tpu_env[tpu_key], msg=tpu_key)

if __name__ == "__main__":
  absltest.main()
