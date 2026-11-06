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
import subprocess
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
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_TIMEOUT",
    "OTEL_METRIC_EXPORT_INTERVAL",
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
# Environment variables read by `yaml_generator.py` that intentionally pass
# through from the caller's environment (not configured via `base_dist_config`).
_YAML_GEN_ENV_PASSTHROUGH = frozenset({
    "COLOCATED_PYTHON_SIDECAR_IMAGE",
    "COLOCATED_PYTHON_SIDECAR_MEMORY",
    "COLOCATED_PYTHON_SIDECAR_SHM",
    "ENABLE_MULTI_NUMA",
    "PATHWAYS_PROXY_EXTRA_ARGS",
    "PATHWAYS_WORKER_EXTRA_ENV",
    "RAIDEN_BROADCAST_K",
    "TPU_RAIDEN_DATA_NICS",
    "TPU_RESERVATION",
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
          isinstance(node.func, ast.Name)
          and node.func.id == "_env_bool"
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
        "weight_sync_config.mode=raiden",
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
        "kubectl delete leaderworkerset testuser-roll-0 -n default", res.stdout
    )
    self.assertIn(
        "kubectl delete leaderworkerset testuser-roll-1 -n default", res.stdout
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
        "cluster_config.k8s.enable_pathways_persistence=true",
        "agentic_grpo_config.beta=0.0",
    ])
    self.assertEqual(res_persistence.returncode, 0, msg=res_persistence.stderr)
    self.assertIn("ENABLE_PATHWAYS_PERSISTENCE=1", res_persistence.stdout)
    self.assertNotIn("ENABLE_PATHWAYS_PERSISTENCE=true", res_persistence.stdout)

  def test_k8s_requires_tunix_image(self):
    res = self._run_launch([
        "--dry-run",
        "cluster_config.target=k8s",
        "agentic_grpo_config.beta=0.0",
    ])
    self.assertNotEqual(res.returncode, 0)
    self.assertIn("cluster_config.k8s.tunix_image is empty", res.stderr)

  def test_k8s_rejects_reference_worker(self):
    # The default beta (0.04) enables the reference worker, which k8s.sh does
    # not deploy.
    res = self._run_launch([
        "start",
        "--dry-run",
        "cluster_config.target=k8s",
        "cluster_config.k8s.tunix_image=test-image",
    ])
    self.assertNotEqual(res.returncode, 0)
    self.assertIn("does not deploy a reference inference worker", res.stderr)
    self.assertNotIn("apiVersion:", res.stdout)

  def test_k8s_stop_command_only_deletes_resources(self):
    # Deleting the jobs needs no node image, and the default beta's reference
    # worker (which k8s.sh does not deploy) is no reason to fail.
    res = self._run_launch(["stop", "--dry-run", "cluster_config.target=k8s"])
    self.assertEqual(res.returncode, 0, msg=res.stderr)
    self.assertIn("kubectl delete jobset testuser-orch -n default", res.stdout)
    self.assertNotIn("apiVersion:", res.stdout)

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
    args = [
        "start",
        "--dry-run",
        "cluster_config.target=k8s",
        "cluster_config.k8s.tunix_image=test-image",
        "agentic_grpo_config.beta=0.0",
    ]
    res = self._run_launch(
        args,
        extra_env={
            "OTEL_EXPORTER_OTLP_ENDPOINT": "http://testuser-otel:4317",
            "OTEL_METRIC_EXPORT_INTERVAL": "5000",
            "OTEL_EXPORTER_OTLP_TIMEOUT": "10000",
        },
    )
    self.assertEqual(res.returncode, 0, msg=res.stderr)
    # Each startup command is one line of its rendered manifest.
    exporting = set(
        re.findall(
            r"OTEL_EXPORTER_OTLP_ENDPOINT=http://testuser-otel:4317 "
            r"OTEL_METRIC_EXPORT_INTERVAL=5000 "
            r"OTEL_EXPORTER_OTLP_TIMEOUT=10000 .*--process_main=(\S+)",
            res.stdout,
        )
    )
    mains = {
        "tunix.dist_rl.launch.nodes.orchestrator.main",
        "tunix.dist_rl.launch.nodes.trainer.main",
        "tunix.dist_rl.launch.nodes.rollout.main",
    }
    self.assertEqual(exporting, mains)
    # Every pod resolves the collector name the launcher deployed, not one
    # from the container's `$USER`.
    pinned = set(
        re.findall(
            r"--process_main=(\S+) [^\n]*"
            r"cluster_config\.k8s\.otel_id=testuser-otel",
            res.stdout,
        )
    )
    self.assertEqual(pinned, mains)
    # Unset in the caller's environment, the pods get neither.
    res = self._run_launch(args)
    self.assertEqual(res.returncode, 0, msg=res.stderr)
    self.assertNotIn("OTEL_", res.stdout)

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
        "cluster_config.k8s.cpu_memory=64Gi",
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
        "cluster_config.k8s.pathways_proxy_memory_limit=190G",
        "cluster_config.k8s.pathways_proxy_memory=24G",
        "cluster_config.k8s.pathways_rm_memory=8G",
        "cluster_config.k8s.user_container_memory=56G",
        "cluster_config.k8s.user_container_memory_limit=80G",
        "cluster_config.k8s.pathways_worker_memory=120G",
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

    # Worker pod (pw-node) container (pathways-worker).
    pw_spec = rjobs["pw-node"]
    pw_containers = {c["name"]: c for c in pw_spec["containers"]}
    self.assertEqual(
        pw_containers["pathways-worker"]["resources"]["requests"]["memory"],
        "120G",
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

  def test_otel_commands_deploy_and_delete_the_collector(self):
    # The collector needs no `cluster_config.k8s.tunix_image`, and the default
    # beta's reference worker (which k8s.sh does not deploy) is no reason to
    # fail.
    k8s = ["cluster_config.target=k8s"]
    res = self._run_launch(
        ["otel", "--dry-run", *k8s, "cluster_config.ports.otel=4318"]
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

    res = self._run_launch(["stop-otel", "--dry-run", *k8s])
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
